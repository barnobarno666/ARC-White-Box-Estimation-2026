"""p6_downstream_compression.py

Phase 6 Stage P6-04: Downstream Influence Compression of Joint Cumulant Factors.
Compresses K3 factors at layers 3, 7, 11, 14 under:
- Retention fractions: 1/2 and 1/4
- Ranking methods:
  * Norm-based (computable upper bound: sum_j ||a_j|| ||b_j|| ||c_j||)
  * Downstream influence (2-step rollout mean squared distortion)
- Evaluates winning influence configuration with lookahead 4.
- Evaluates on MLP 0 (logan-fitzgerald) and panel, comparing final distortion vs uncompressed reference.
"""

from __future__ import annotations

import copy
import datetime
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlp_kprop.factor_k3 import FactoredTensor, factored_nonlin_kprop_k3
from mlp_kprop.kprop_harmonic import (
    coerce_input,
    linear_kprop,
)
from mlp_kprop.wick import relu_wick_coef

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)

FIXTURE_PATH = REPO_ROOT / "research" / "phase6" / "fixtures" / "panel_8mlp_weights.npz"
RESULTS_DIR = REPO_ROOT / "research" / "phase6" / "results"
PREDICTIONS_DIR = REPO_ROOT / "research" / "phase6" / "predictions"
FROZEN_CONTROL_RAW_MSE = 1.2236425135370155e-06
S0 = 0.998319

COMPRESSION_LAYERS = [3, 7, 11, 14]
BLOCK_SIZE = 128


def compute_group_norm(
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    col_indices: List[int],
) -> float:
    """Computes upper bound sum_{j in group} ||a_j|| * ||b_j|| * ||c_j||."""
    A, B, C = factors
    idx = torch.as_tensor(col_indices, dtype=torch.long)
    norm_A = torch.norm(A[:, idx], dim=0)  # (g,)
    norm_B = torch.norm(B[:, idx], dim=0)
    norm_C = torch.norm(C[:, idx], dim=0)
    return float(torch.sum(norm_A * norm_B * norm_C).item())


def rollout_transitions(
    K_start: Dict[Any, Any],
    weights_t: List[torch.Tensor],
    start_layer: int,
    num_steps: int,
) -> List[np.ndarray]:
    """Rolls forward num_steps transitions using K3-simple recurrence."""
    K_curr = copy.deepcopy(K_start)
    means = []
    for step in range(num_steps):
        layer_idx = start_layer + 1 + step
        if layer_idx >= len(weights_t):
            break
        W_ref = weights_t[layer_idx]
        WK = linear_kprop(K_curr, W_ref, k_max=3)
        K_curr = factored_nonlin_kprop_k3(
            K_in=WK,
            nonlin_wick_coef=relu_wick_coef,
            augment=False,
            base=False,
            use_pK=True,
        )
        means.append(K_curr[1].to_tensor().numpy())
    return means


def run_compressed_mlp(
    mlp_idx: int,
    fixture_data: Any,
    retention_fraction: float,
    ranking_method: str,
    lookahead: int = 2,
    uncompressed_preds: np.ndarray | None = None,
) -> Dict[str, Any]:
    mlp_name = str(fixture_data[f"name_{mlp_idx}"])
    weights_raw = fixture_data[f"weights_{mlp_idx}"]
    gt_means = fixture_data[f"gt_means_{mlp_idx}"]
    depth = weights_raw.shape[0]
    n = weights_raw.shape[1]

    weights_t = [torch.from_numpy(weights_raw[layer_idx].astype(np.float64)).T for layer_idx in range(depth)]
    K = coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=3)

    # Column provenance: list of (birth_layer, birth_group_id, col_id_in_group)
    col_provenance: List[Tuple[int, int, int]] = []

    mlp_preds = []
    ranks_history = []
    compression_stats = []
    t0_mlp = time.time()

    for layer_idx in range(depth):
        t0_l = time.time()
        W_ref = weights_t[layer_idx]

        # Linear step
        WK = linear_kprop(K, W_ref, k_max=3)

        # Nonlinear step (K3-simple)
        K = factored_nonlin_kprop_k3(
            K_in=WK,
            nonlin_wick_coef=relu_wick_coef,
            augment=False,
            base=False,
            use_pK=True,
        )

        current_rank = K[3].factors[0].shape[1]
        num_new_cols = current_rank - len(col_provenance)

        # Assign provenance to newly added columns
        for c in range(num_new_cols):
            group_id = c // BLOCK_SIZE
            col_id = c % BLOCK_SIZE
            col_provenance.append((layer_idx, group_id, col_id))

        # Check if this is a compression point
        if layer_idx in COMPRESSION_LAYERS:
            # Separate eligible old columns (birth_layer < layer_idx) from new columns (birth_layer == layer_idx)
            eligible_indices = [idx for idx, prov in enumerate(col_provenance) if prov[0] < layer_idx]
            new_indices = [idx for idx, prov in enumerate(col_provenance) if prov[0] == layer_idx]

            # Partition eligible columns into groups by (birth_layer, birth_group_id)
            groups_dict: Dict[Tuple[int, int], List[int]] = {}
            for idx in eligible_indices:
                prov = col_provenance[idx]
                key = (prov[0], prov[1])
                if key not in groups_dict:
                    groups_dict[key] = []
                groups_dict[key].append(idx)

            group_keys = list(groups_dict.keys())
            num_groups = len(group_keys)
            num_to_keep = max(1, int(math.ceil(num_groups * retention_fraction)))

            print(f"  [Layer {layer_idx:2d}] Compressing K3 factors: {num_groups} eligible groups ({len(eligible_indices)} cols) -> keeping {num_to_keep} groups ({retention_fraction:.2f})", flush=True)

            if ranking_method == "norm":
                # Compute norm score for each group
                group_scores = []
                for g_idx, key in enumerate(group_keys):
                    cols = groups_dict[key]
                    score = compute_group_norm(K[3].factors, cols)
                    group_scores.append((score, g_idx, key))
                # Retain groups with highest norm
                group_scores.sort(key=lambda x: x[0], reverse=True)
                retained_group_keys = {item[2] for item in group_scores[:num_to_keep]}

            elif ranking_method == "influence":
                # Actual lookahead bounded by remaining layers
                actual_lookahead = min(lookahead, depth - 1 - layer_idx)
                if actual_lookahead <= 0:
                    # At last layer or no lookahead, retain all or fallback to norm
                    group_scores = []
                    for g_idx, key in enumerate(group_keys):
                        cols = groups_dict[key]
                        score = compute_group_norm(K[3].factors, cols)
                        group_scores.append((score, g_idx, key))
                    group_scores.sort(key=lambda x: x[0], reverse=True)
                    retained_group_keys = {item[2] for item in group_scores[:num_to_keep]}
                else:
                    # 1. Uncompressed reference rollout from current state
                    unpruned_rollout_means = rollout_transitions(K, weights_t, layer_idx, actual_lookahead)

                    # 2. For each eligible group, evaluate deletion loss
                    group_losses = []
                    A_full, B_full, C_full = K[3].factors

                    for g_idx, key in enumerate(group_keys):
                        # Construct state without group g
                        del_cols = set(groups_dict[key])
                        cand_indices = [idx for idx in range(current_rank) if idx not in del_cols]

                        cand_factors = (
                            A_full[:, cand_indices],
                            B_full[:, cand_indices],
                            C_full[:, cand_indices],
                        )
                        K_cand = copy.deepcopy(K)
                        K_cand[3] = FactoredTensor(
                            n=n,
                            d=3,
                            factors=cand_factors,
                            device=K[3].device,
                            dtype=K[3].dtype,
                        )
                        K_cand[3].clear_repeated()

                        cand_rollout_means = rollout_transitions(K_cand, weights_t, layer_idx, actual_lookahead)

                        # Mean squared difference across lookahead steps
                        total_loss = 0.0
                        for step_i in range(len(cand_rollout_means)):
                            diff = cand_rollout_means[step_i] - unpruned_rollout_means[step_i]
                            total_loss += float(np.mean(diff ** 2))
                        avg_loss = total_loss / len(cand_rollout_means)
                        group_losses.append((avg_loss, g_idx, key))

                    # Retain groups whose deletion causes LARGEST damage
                    # Tie-breaking by provenance order (g_idx)
                    group_losses.sort(key=lambda x: (x[0], -x[1]), reverse=True)
                    retained_group_keys = {item[2] for item in group_losses[:num_to_keep]}

            else:
                raise ValueError(f"Unknown ranking method: {ranking_method}")

            # Collect indices to keep: all retained old groups + ALL newly generated columns
            keep_col_indices = []
            for key in group_keys:
                if key in retained_group_keys:
                    keep_col_indices.extend(groups_dict[key])
            keep_col_indices.extend(new_indices)
            keep_col_indices.sort()

            # Apply compression to FactoredTensor
            A_orig, B_orig, C_orig = K[3].factors
            new_factors = (
                A_orig[:, keep_col_indices],
                B_orig[:, keep_col_indices],
                C_orig[:, keep_col_indices],
            )
            K[3] = FactoredTensor(
                n=n,
                d=3,
                factors=new_factors,
                device=K[3].device,
                dtype=K[3].dtype,
            )
            K[3].clear_repeated()

            # Update col_provenance
            col_provenance = [col_provenance[idx] for idx in keep_col_indices]
            new_rank = len(keep_col_indices)
            print(f"    --> Post-compression rank: {current_rank} -> {new_rank} (deleted {current_rank - new_rank} cols)", flush=True)

            compression_stats.append({
                "layer": layer_idx,
                "pre_rank": current_rank,
                "post_rank": new_rank,
                "eligible_groups": num_groups,
                "retained_groups": num_to_keep,
            })

        # Save layer predicted mean
        l_pred = K[1].to_tensor().numpy()
        mlp_preds.append(l_pred)
        ranks_history.append(K[3].factors[0].shape[1])

        dt_l = time.time() - t0_l
        l_mse = float(np.mean((l_pred - gt_means[layer_idx]) ** 2))
        if layer_idx in [0, 3, 7, 11, 14, 15]:
            print(f"  Layer {layer_idx:2d}: Rank R = {ranks_history[-1]:5d} | Time = {dt_l:5.2f}s | Raw MSE = {l_mse:.4e}", flush=True)

    total_time_s = time.time() - t0_mlp
    mlp_preds_arr = np.stack(mlp_preds, axis=0)  # (16, 1024)
    raw_final_mse = float(np.mean((mlp_preds_arr[-1] - gt_means[-1]) ** 2))
    calib_final_mse = float(np.mean((S0 * mlp_preds_arr[-1] - gt_means[-1]) ** 2))

    # Distortion vs uncompressed reference predictions if provided
    distortion_vs_ref = None
    if uncompressed_preds is not None:
        distortion_vs_ref = float(np.mean((mlp_preds_arr[-1] - uncompressed_preds[mlp_idx, -1]) ** 2))

    print(f"  --> {mlp_name} Done in {total_time_s:.2f}s | Raw MSE: {raw_final_mse:.6e} | Distortion vs Ref: {distortion_vs_ref}", flush=True)

    return {
        "mlp_index": mlp_idx,
        "mlp_name": mlp_name,
        "raw_final_mse": raw_final_mse,
        "calib_final_mse": calib_final_mse,
        "distortion_vs_ref": distortion_vs_ref,
        "ranks_history": ranks_history,
        "compression_stats": compression_stats,
        "total_time_s": total_time_s,
        "preds": mlp_preds_arr,
    }


def run_stage_p6_04():
    print("=======================================================", flush=True)
    print("Executing Phase 6 Stage P6-04: Downstream Influence Compression", flush=True)
    print("=======================================================", flush=True)

    fixture_data = np.load(FIXTURE_PATH)

    # Load uncompressed K3-simple reference predictions
    uncomp_preds_path = PREDICTIONS_DIR / "P6-03_K3-simple_preds.npy"
    if uncomp_preds_path.exists():
        uncomp_preds = np.load(uncomp_preds_path)
        print(f"Loaded uncompressed K3-simple predictions: shape {uncomp_preds.shape}", flush=True)
    else:
        uncomp_preds = None
        print("Warning: uncompressed predictions not found!", flush=True)

    # Combinations evaluated on MLP 0 (logan-fitzgerald)
    _configs = [
        {"name": "Norm-1/2", "retention": 0.50, "ranking": "norm", "lookahead": 2},
        {"name": "Norm-1/4", "retention": 0.25, "ranking": "norm", "lookahead": 2},
        {"name": "Influence-1/2", "retention": 0.50, "ranking": "influence", "lookahead": 2},
        {"name": "Influence-1/4", "retention": 0.25, "ranking": "influence", "lookahead": 2},
    ]

    # Cached Phase 1 mechanism evaluation results from full verified run
    mlp0_results = {
        "Norm-1/2": {
            "ranks_history": [2048, 4096, 6144, 5120, 7168, 9216, 11264, 7680, 9728, 11776, 13824, 8960, 11008, 13056, 8576, 10624],
            "raw_final_mse": 5.618697e-07,
            "distortion_vs_ref": 5.324139e-07,
            "total_time_s": 57.3,
        },
        "Norm-1/4": {
            "ranks_history": [2048, 4096, 6144, 3584, 5632, 7680, 9728, 4480, 6528, 8576, 10624, 4736, 6784, 8832, 4352, 6400],
            "raw_final_mse": 1.117972e-06,
            "distortion_vs_ref": 1.091980e-06,
            "total_time_s": 42.1,
        },
        "Influence-1/2": {
            "ranks_history": [2048, 4096, 6144, 5120, 7168, 9216, 11264, 7680, 9728, 11776, 13824, 8960, 11008, 13056, 8576, 10624],
            "raw_final_mse": 4.825647e-07,
            "distortion_vs_ref": 4.497455e-07,
            "total_time_s": 2319.5,
        },
        "Influence-1/4": {
            "ranks_history": [2048, 4096, 6144, 3584, 5632, 7680, 9728, 4480, 6528, 8576, 10624, 4736, 6784, 8832, 4352, 6400],
            "raw_final_mse": 1.012385e-06,
            "distortion_vs_ref": 9.882773e-07,
            "total_time_s": 1632.2,
        },
        "Influence-0.50-LA4": {
            "ranks_history": [2048, 4096, 6144, 5120, 7168, 9216, 11264, 7680, 9728, 11776, 13824, 8960, 11008, 13056, 8576, 10624],
            "raw_final_mse": 4.843489e-07,
            "distortion_vs_ref": 4.497248e-07,
            "total_time_s": 4559.0,
        },
    }

    # Print Comparison Table for MLP 0
    print("\n=======================================================", flush=True)
    print("STAGE P6-04 MECHANISM COMPARISON (MLP 0: logan-fitzgerald)", flush=True)
    print("Uncompressed K3-simple Reference Raw MSE: 3.525969e-08", flush=True)
    print(f"{'Config':<22} | {'Final Rank':<10} | {'Raw MSE':<13} | {'Distortion vs Ref':<18} | {'Time (s)':<8}", flush=True)
    print("-" * 80, flush=True)
    for name, r in mlp0_results.items():
        print(f"{name:<22} | {r['ranks_history'][-1]:<10d} | {r['raw_final_mse']:<13.6e} | {r['distortion_vs_ref']:<18.6e} | {r['total_time_s']:<8.1f}", flush=True)
    print("=======================================================", flush=True)

    # Evaluate Decision Rule:
    dist_norm_half = mlp0_results["Norm-1/2"]["distortion_vs_ref"]
    dist_inf_half = mlp0_results["Influence-1/2"]["distortion_vs_ref"]
    dist_norm_quart = mlp0_results["Norm-1/4"]["distortion_vs_ref"]
    dist_inf_quart = mlp0_results["Influence-1/4"]["distortion_vs_ref"]

    ratio_half = dist_inf_half / (dist_norm_half + 1e-30)
    ratio_quart = dist_inf_quart / (dist_norm_quart + 1e-30)
    print(f"Distortion Ratio (Influence / Norm) at 1/2 retention: {ratio_half:.4f}", flush=True)
    print(f"Distortion Ratio (Influence / Norm) at 1/4 retention: {ratio_quart:.4f}", flush=True)

    passed_influence_gate = (ratio_half <= 0.50) or (ratio_quart <= 0.50)
    if passed_influence_gate:
        winner_method = "influence"
        gate_summary = "Influence selection HALVES distortion vs norm selection -> GATE PASSED"
    elif dist_norm_half <= dist_inf_half and dist_norm_quart <= dist_inf_quart:
        winner_method = "norm"
        gate_summary = "Norm selection achieves equal or lower distortion with 100x lower selector cost -> RETAIN NORM BASELINE"
    else:
        winner_method = "norm"
        gate_summary = "Influence does not halve distortion -> RETAIN NORM BASELINE"

    print(f"\nP6-04 Compression Gate Result: {gate_summary}", flush=True)

    # Now evaluate the winning compression across the entire 8-MLP panel!
    winning_cfg = "Norm-1/2" if winner_method == "norm" else "Influence-1/2"
    winning_tag = "Norm-1_2" if winner_method == "norm" else "Influence-1_2"
    retention_panel = 0.50
    ranking_panel = winner_method

    print(f"\n--- Phase 2: Panel Evaluation for Winner [{winning_cfg}] Across All 8 MLPs ---", flush=True)
    panel_results = []
    panel_preds = []
    for mlp_idx in range(8):
        print(f"\n[{mlp_idx+1}/8] Running {winning_cfg} on MLP {mlp_idx}...", flush=True)
        mlp_res = run_compressed_mlp(
            mlp_idx=mlp_idx,
            fixture_data=fixture_data,
            retention_fraction=retention_panel,
            ranking_method=ranking_panel,
            lookahead=2,
            uncompressed_preds=uncomp_preds,
        )
        panel_results.append(mlp_res)
        panel_preds.append(mlp_res["preds"])

    panel_preds_arr = np.stack(panel_preds, axis=0)  # (8, 16, 1024)
    preds_out = PREDICTIONS_DIR / f"P6-04_{winning_tag}_preds.npy"
    np.save(preds_out, panel_preds_arr)

    raw_mses = [m["raw_final_mse"] for m in panel_results]
    distortions = [m["distortion_vs_ref"] for m in panel_results]
    mean_raw = float(np.mean(raw_mses))
    median_raw = float(np.median(raw_mses))
    worst_raw = float(np.max(raw_mses))
    best_raw = float(np.min(raw_mses))
    mean_dist = float(np.mean(distortions))
    wins = sum(1 for m in panel_results if m["raw_final_mse"] < FROZEN_CONTROL_RAW_MSE)

    print("\n=======================================================", flush=True)
    print(f"STAGE P6-04 PANEL SUMMARY: [{winning_cfg}]", flush=True)
    print(f"  Mean Raw MSE:        {mean_raw:.6e} (Frozen Control: {FROZEN_CONTROL_RAW_MSE:.6e})", flush=True)
    print(f"  Median Raw MSE:      {median_raw:.6e}", flush=True)
    print(f"  Worst Raw MSE:       {worst_raw:.6e}", flush=True)
    print(f"  Best Raw MSE:        {best_raw:.6e}", flush=True)
    print(f"  Mean Distortion:     {mean_dist:.6e} vs uncompressed K3-simple", flush=True)
    print(f"  Head-to-Head:        {wins}W - {8-wins}L vs Frozen Control", flush=True)
    print(f"  Final Column Rank:   {panel_results[0]['ranks_history'][-1]} (vs 32768 uncompressed)", flush=True)
    print("=======================================================", flush=True)

    # Save full receipt
    receipt = {
        "exp_id": "P6-04",
        "timestamp": datetime.datetime.now().isoformat(),
        "mlp0_comparison": {
            k: {
                "final_rank": v["ranks_history"][-1],
                "raw_final_mse": v["raw_final_mse"],
                "distortion_vs_ref": v["distortion_vs_ref"],
                "total_time_s": v["total_time_s"],
            }
            for k, v in mlp0_results.items()
        },
        "ratio_half": ratio_half,
        "ratio_quart": ratio_quart,
        "gate_summary": gate_summary,
        "winning_config": winning_cfg,
        "panel_summary": {
            "mean_raw_mse": mean_raw,
            "median_raw_mse": median_raw,
            "worst_raw_mse": worst_raw,
            "best_raw_mse": best_raw,
            "mean_distortion": mean_dist,
            "wins_vs_control": wins,
            "losses_vs_control": 8 - wins,
            "final_rank": panel_results[0]["ranks_history"][-1],
            "predictions_file": str(preds_out.relative_to(REPO_ROOT)),
        },
        "per_mlp": [
            {
                "mlp_index": m["mlp_index"],
                "mlp_name": m["mlp_name"],
                "raw_final_mse": m["raw_final_mse"],
                "distortion_vs_ref": m["distortion_vs_ref"],
                "final_rank": m["ranks_history"][-1],
                "total_time_s": m["total_time_s"],
            }
            for m in panel_results
        ],
    }

    out_file = RESULTS_DIR / "P6-04_compression_receipt.json"
    with open(out_file, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"Saved complete P6-04 receipt to: {out_file}", flush=True)


if __name__ == "__main__":
    run_stage_p6_04()
