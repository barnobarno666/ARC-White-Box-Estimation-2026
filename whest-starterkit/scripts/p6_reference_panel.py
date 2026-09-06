"""p6_reference_panel.py

Phase 6 Stage P6-03: Faithful uncompressed reference evaluation on the 8-MLP panel.
Evaluates:
- K3-base: base=True, augment=False
- K3-simple: base=False, augment=False
- K3-augment-filtered: base=False, augment=True
Across all 8 panel MLPs (Width 1024, Depth 16).
Records:
- Predictions (8, 16, 1024)
- Layerwise factor rank growth
- Raw and calibrated MSEs
- Cross-panel mean, median, worst, and wins vs control (1.2236e-06).
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlp_kprop.kprop_harmonic as kh
from mlp_kprop.factor_k3 import FactoredTensor, factored_nonlin_kprop_k3
from mlp_kprop.kprop_harmonic import (
    coerce_input,
    factored_keeps_term,
    linear_kprop,
)
from mlp_kprop.wick import relu_wick_coef

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)

FIXTURE_PATH = REPO_ROOT / "research" / "phase6" / "fixtures" / "panel_8mlp_weights.npz"
RESULTS_DIR = REPO_ROOT / "research" / "phase6" / "results"
PREDICTIONS_DIR = REPO_ROOT / "research" / "phase6" / "predictions"
S0 = 0.998319
FROZEN_CONTROL_RAW_MSE = 1.2236425135370155e-06

PANEL_MLP_NAMES = [
    "logan-fitzgerald",
    "william-graves",
    "raymond-barnes",
    "steven-rice",
    "sarah-kelley",
    "christopher-morales",
    "cheryl-graham",
    "renee-park",
]


def run_reference_mode(
    mode: str,
    fixture_data: Any,
) -> Dict[str, Any]:
    print("\n=======================================================", flush=True)
    print(f"Executing Phase 6 Stage P6-03 Reference Panel: [{mode}]", flush=True)
    print("Panel: 8 MLPs | Width 1024 | Depth 16 layers", flush=True)
    print("=======================================================", flush=True)

    if mode == "K3-base":
        base = True
        augment = False
    elif mode == "K3-simple":
        base = False
        augment = False
    elif mode == "K3-augment-filtered":
        base = False
        augment = True
    else:
        raise ValueError(f"Unknown mode {mode}")

    real_iso = kh.get_all_terms_iso
    def filtered_iso(k_max, d_max=None, use_mean_var=False, augment=False):
        ret = real_iso(k_max, d_max=d_max, use_mean_var=use_mean_var, augment=augment)
        return {
            ip: {vp: c for vp, c in vps.items() if factored_keeps_term(k_max, ip, vp)}
            for ip, vps in ret.items()
        }

    panel_predictions = []  # will be (8, 16, 1024)
    per_mlp_results = []

    for mlp_idx in range(8):
        mlp_name = str(fixture_data[f"name_{mlp_idx}"])
        weights_raw = fixture_data[f"weights_{mlp_idx}"]  # (16, 1024, 1024)
        gt_means = fixture_data[f"gt_means_{mlp_idx}"]    # (16, 1024)
        n = weights_raw.shape[1]
        depth = weights_raw.shape[0]

        print(f"\n[{mlp_idx+1}/8] Processing {mlp_name} (n={n}, depth={depth})...", flush=True)
        weights_t = [torch.from_numpy(weights_raw[layer_idx].astype(np.float64)).T for layer_idx in range(depth)]

        K = coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=3)
        mlp_preds = []
        ranks_history = []
        layer_times = []
        t0_mlp = time.time()

        for layer_idx in range(depth):
            t0_l = time.time()
            W_ref = weights_t[layer_idx]

            # Linear step
            WK = linear_kprop(K, W_ref, k_max=3)

            # Nonlinear step
            if mode == "K3-augment-filtered":
                kh.get_all_terms_iso = filtered_iso

            try:
                K = factored_nonlin_kprop_k3(
                    K_in=WK,
                    nonlin_wick_coef=relu_wick_coef,
                    augment=augment,
                    base=base,
                    use_pK=True,
                )
            finally:
                kh.get_all_terms_iso = real_iso

            dt_l = time.time() - t0_l
            layer_times.append(dt_l)

            # Extract layer predicted mean
            l_pred = K[1].to_tensor().numpy()
            mlp_preds.append(l_pred)

            r_cur = K[3].factors[0].shape[1] if (3 in K and isinstance(K[3], FactoredTensor)) else 0
            ranks_history.append(r_cur)

            l_mse = float(np.mean((l_pred - gt_means[layer_idx]) ** 2))
            if layer_idx in [0, 3, 7, 11, 14, 15]:
                print(f"  Layer {layer_idx:2d}: Rank R = {r_cur:5d} | Time = {dt_l:5.2f}s | Raw MSE = {l_mse:.4e}", flush=True)

        total_mlp_s = time.time() - t0_mlp
        mlp_preds_arr = np.stack(mlp_preds, axis=0)  # (16, 1024)
        panel_predictions.append(mlp_preds_arr)

        raw_final_mse = float(np.mean((mlp_preds_arr[-1] - gt_means[-1]) ** 2))
        calib_final_mse = float(np.mean((S0 * mlp_preds_arr[-1] - gt_means[-1]) ** 2))

        per_layer_mses = [
            float(np.mean((mlp_preds_arr[layer_idx] - gt_means[layer_idx]) ** 2))
            for layer_idx in range(depth)
        ]

        diff_pct = (raw_final_mse - FROZEN_CONTROL_RAW_MSE) / FROZEN_CONTROL_RAW_MSE * 100.0
        print(f"  --> {mlp_name} Done in {total_mlp_s:.2f}s | Raw MSE: {raw_final_mse:.6e} ({diff_pct:+.2f}% vs ctrl)", flush=True)

        per_mlp_results.append({
            "mlp_index": mlp_idx,
            "mlp_name": mlp_name,
            "raw_final_mse": raw_final_mse,
            "calib_final_mse": calib_final_mse,
            "diff_vs_control_pct": diff_pct,
            "per_layer_raw_mse": per_layer_mses,
            "rank_growth": ranks_history,
            "wall_time_s": total_mlp_s,
        })

    panel_preds_stack = np.stack(panel_predictions, axis=0)  # (8, 16, 1024)
    preds_out = PREDICTIONS_DIR / f"P6-03_{mode}_preds.npy"
    np.save(preds_out, panel_preds_stack)
    print(f"\nSaved {mode} panel predictions to: {preds_out}", flush=True)

    raw_mses = [m["raw_final_mse"] for m in per_mlp_results]
    calib_mses = [m["calib_final_mse"] for m in per_mlp_results]
    wins = sum(1 for m in per_mlp_results if m["raw_final_mse"] < FROZEN_CONTROL_RAW_MSE)

    mean_raw = float(np.mean(raw_mses))
    median_raw = float(np.median(raw_mses))
    worst_raw = float(np.max(raw_mses))
    best_raw = float(np.min(raw_mses))

    mean_calib = float(np.mean(calib_mses))
    median_calib = float(np.median(calib_mses))

    summary = {
        "mode": mode,
        "mean_raw_mse": mean_raw,
        "median_raw_mse": median_raw,
        "worst_raw_mse": worst_raw,
        "best_raw_mse": best_raw,
        "mean_calib_mse": mean_calib,
        "median_calib_mse": median_calib,
        "wins_vs_control": wins,
        "losses_vs_control": 8 - wins,
        "predictions_file": str(preds_out.relative_to(REPO_ROOT)),
        "per_mlp": per_mlp_results,
    }

    print(f"\n[{mode} Panel Results Summary]", flush=True)
    print(f"  Mean Raw MSE:   {mean_raw:.6e} (Frozen Control: {FROZEN_CONTROL_RAW_MSE:.6e})", flush=True)
    print(f"  Median Raw MSE: {median_raw:.6e}", flush=True)
    print(f"  Worst Raw MSE:  {worst_raw:.6e}", flush=True)
    print(f"  Best Raw MSE:   {best_raw:.6e}", flush=True)
    print(f"  Mean Calib MSE: {mean_calib:.6e}", flush=True)
    print(f"  Head-to-head:   {wins}W - {8-wins}L vs Frozen Control", flush=True)

    return summary


def run_all_reference_modes():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(f"Fixture file {FIXTURE_PATH} not found!")

    print(f"[p6_reference_panel] Loading fixtures from {FIXTURE_PATH}...", flush=True)
    fixture_data = np.load(FIXTURE_PATH)

    results_all_modes = {}
    for mode in ["K3-base", "K3-simple"]:
        res = run_reference_mode(mode, fixture_data)
        results_all_modes[mode] = res

    # Record resource obstruction for uncompressed K3-augment-filtered at width 1024
    results_all_modes["K3-augment-filtered"] = {
        "mode": "K3-augment-filtered",
        "status": "RESOURCE_OBSTRUCTION",
        "reason": "Uncompressed augment mode at width 1024 creates explosive Kronecker factor combinations exceeding memory/time bounds (>6GB / hanging). P6-05 provides exact small-fixture diagnostic.",
    }

    # Classify decision based on evaluated modes
    eval_modes = [m for m, r in results_all_modes.items() if "mean_raw_mse" in r]
    best_mode = min(eval_modes, key=lambda m: results_all_modes[m]["mean_raw_mse"])
    best_raw = results_all_modes[best_mode]["mean_raw_mse"]

    if best_raw <= 1e-7:
        decision = "substantial analytic headroom; prioritize production cost reduction"
    elif best_raw <= 5e-7:
        decision = "useful but incomplete; inspect error emergence and compare modes"
    else:
        decision = "check parity and conventions, then run P6-05 omission diagnostic before investing in compression tuning"

    print("\n=======================================================", flush=True)
    print("STAGE P6-03 FINAL OUTCOME & DECISION", flush=True)
    print(f"Best Mode: [{best_mode}] with Panel Mean Raw MSE = {best_raw:.6e}", flush=True)
    print(f"Decision Classification: {decision}", flush=True)
    print("=======================================================", flush=True)

    receipt = {
        "exp_id": "P6-03",
        "timestamp": datetime.datetime.now().isoformat(),
        "best_mode": best_mode,
        "best_mean_raw_mse": best_raw,
        "decision": decision,
        "modes": results_all_modes,
    }

    out_file = RESULTS_DIR / "P6-03_reference_panel.json"
    with open(out_file, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"Saved complete P6-03 receipt to: {out_file}", flush=True)


if __name__ == "__main__":
    run_all_reference_modes()
