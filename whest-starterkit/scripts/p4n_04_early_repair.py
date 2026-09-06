"""P4N-04: Early-Layer Sample Repair Using Exact Layer-0 Moments.

Implements Experiment P4N-04 from PHASE5_NEXT_EXPERIMENTS.md (Section 10):
1. Evaluates across all 8 Phase 2 MLPs (using whestbench load_dataset, split='mini').
2. Derives exact first post-ReLU mean m0[j] = ||W0[:, j]|| / sqrt(2*pi) and exact
   first-layer covariance C0 from the zero-mean Cho-Saul arc-cosine kernel.
3. Implements the three repair operators on post-layer 0 activations H0:
   a. Positive mean matching: H0 * [(1-t) + t*m0 / max(sample_mean(H0), 1e-12)], ratio capped in [0.8, 1.2].
   b. Additive mean matching: H0 + t*(m0 - sample_mean(H0)).
   c. Mean + covariance transport: center H0; map empirical covariance to exact C0 using
      symmetric square roots on supported modes (ranks 32, 128); then add m0.
   Tests strengths t in {0.25, 0.5, 1.0}.
4. Continues actual network forward pass through remaining layers 1..15.
5. Tests sample counts N in {2048, 8192, 16384} to expose bias vs variance trends.
6. Measures raw final MSE, paired difference vs unrepaired path, and scored error
   with re-estimated grouped leave-one-MLP-out (LOO) blend.
7. Checks promotion gates (15% adjusted gain or >=30% reduction in sampling residual).
8. Saves complete diagnostic results to research/phase4_next/diagnostics/p4n_04_early_repair.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.special

# Ensure immediate line buffering so logs are visible in real time
sys.stdout.reconfigure(line_buffering=True)

from whestbench.domain import MLP
from whestbench.dataset import load_dataset, resolve_seed_context

DATASET_PATH = r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini"
DIAGNOSTICS_DIR = Path("research/phase4_next/diagnostics")
CONTROL_RECEIPT_PATH = Path("scripts/champion_8mlp.json")

# Standard control parameters
_S_PRIOR = 0.998319
_CONTROL_SCORE = 1.223643e-07
_BUDGET_FLOPS = float(2**41)


def numpy_json_default(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))


def compute_analytic_covariance_stack(mlp: MLP) -> np.ndarray:
    """Computes exact dual-kernel blended Hermite analytic mean stack (16, 1024)."""
    width = mlp.width
    mu = np.zeros(width, dtype=np.float32)
    cov = np.eye(width, dtype=np.float32)
    rows = []

    c_centered = np.float32(0.80 * 0.20 * 0.07957747154594767)
    c_thresh = np.float32(0.20 * 0.5)

    for weight in mlp.weights:
        w = np.asarray(weight, dtype=np.float32)
        mu_pre = w.T @ mu
        cov_pre = w.T @ cov @ w
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sigma_pre = np.sqrt(var_pre)
        a = mu_pre / sigma_pre
        phi = norm_pdf(a).astype(np.float32)
        cdf = norm_cdf(a).astype(np.float32)

        mu = (mu_pre * cdf + sigma_pre * phi).astype(np.float32)
        second = ((mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi).astype(np.float32)
        var_post = np.maximum(second - mu * mu, 0.0)
        gain = np.where(sigma_pre > 1e-12, cdf, 0.0).astype(np.float32)

        cov_linear = np.outer(gain, gain) * cov_pre
        cov_pre_sq = cov_pre * cov_pre
        inv_sigma = np.where(sigma_pre > 1e-12, 1.0 / sigma_pre, 0.0)
        u_thresh = np.where(sigma_pre > 1e-12, phi / sigma_pre, 0.0)

        kernel_quad = c_centered * np.outer(inv_sigma, inv_sigma) + c_thresh * np.outer(u_thresh, u_thresh)
        cov_quad = kernel_quad * cov_pre_sq

        cov = cov_linear + cov_quad
        np.fill_diagonal(cov, var_post)
        cov = 0.5 * (cov + cov.T)
        rows.append(mu)

    return np.stack(rows, axis=0)


def compute_exact_layer0_moments(W0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes exact post-ReLU mean m0 and Cho-Saul covariance C0 for Layer 0."""
    # Pre-activation covariance Sigma_0 = W0.T @ W0
    cov_pre_0 = W0.T @ W0
    var_0 = np.maximum(np.diag(cov_pre_0), 1e-12)
    sigma_0 = np.sqrt(var_0)

    # Exact mean: m0[j] = norm(W0[:, j]) / sqrt(2*pi)
    m0 = (sigma_0 / np.sqrt(2.0 * np.pi)).astype(np.float32)

    # Exact zero-mean Cho-Saul arc-cosine kernel:
    inv_sigma_0 = 1.0 / sigma_0
    rho_0 = np.clip((inv_sigma_0[:, None] * cov_pre_0) * inv_sigma_0[None, :], -0.999999, 0.999999)
    term1 = np.sqrt(1.0 - rho_0 * rho_0)
    term2 = rho_0 * (np.pi / 2.0 + np.arcsin(rho_0))
    e_joint = (term1 + term2) / (2.0 * np.pi)

    # C0[j, k] = E[H0[j] H0[k]] - m0[j] m0[k]
    C0 = (sigma_0[:, None] * e_joint) * sigma_0[None, :] - m0[:, None] * m0[None, :]
    np.fill_diagonal(C0, 0.5 * var_0 - m0 * m0)
    C0 = 0.5 * (C0 + C0.T)

    return m0, C0.astype(np.float32)


def repair_positive_mean_matching(
    H0: np.ndarray,
    m0: np.ndarray,
    t: float,
    eps: float = 1e-12,
    cap_low: float = 0.8,
    cap_high: float = 1.2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Operator A: Positive mean matching H0 * [(1-t) + t*m0 / max(sample_mean(H0), eps)], capped in [0.8, 1.2]."""
    sample_mean_H0 = np.mean(H0, axis=0)
    raw_ratio = m0 / np.maximum(sample_mean_H0, eps)
    hit_low = int(np.sum(raw_ratio < cap_low))
    hit_high = int(np.sum(raw_ratio > cap_high))

    ratio_capped = np.clip(raw_ratio, cap_low, cap_high)
    multiplier = (1.0 - t) + t * ratio_capped
    H0_repaired = H0 * multiplier

    meta = {
        "hit_low_count": hit_low,
        "hit_high_count": hit_high,
        "mean_ratio": float(np.mean(raw_ratio)),
        "max_ratio": float(np.max(raw_ratio)),
        "min_ratio": float(np.min(raw_ratio)),
    }
    return H0_repaired, meta


def repair_additive_mean_matching(
    H0: np.ndarray,
    m0: np.ndarray,
    t: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Operator B: Additive mean matching H0 + t*(m0 - sample_mean(H0))."""
    sample_mean_H0 = np.mean(H0, axis=0)
    shift = t * (m0 - sample_mean_H0)
    H0_repaired = H0 + shift

    neg_elements = int(np.sum(H0_repaired < 0.0))
    neg_pct = float(neg_elements / H0_repaired.size) * 100.0
    min_val = float(np.min(H0_repaired))

    meta = {
        "neg_elements": neg_elements,
        "neg_pct": neg_pct,
        "min_val": min_val,
        "shift_norm": float(np.linalg.norm(shift)),
    }
    return H0_repaired, meta


def repair_covariance_transport(
    H0: np.ndarray,
    m0: np.ndarray,
    C0: np.ndarray,
    rank: int,
    t: float,
    evecs_C0: np.ndarray,
    evals_C0: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Operator C: Mean + covariance transport on supported modes of C0.
    
    1. Center H0.
    2. Map empirical covariance to exact C0 using symmetric square roots on top r modes of C0.
    3. Interpolate between identity and transport map using t.
    4. Add m0 (with mean shift interpolated by t).
    """
    N, d = H0.shape
    sample_mean_H0 = np.mean(H0, axis=0)
    H0_cent = H0 - sample_mean_H0

    # Top r eigenvectors and eigenvalues of C0
    U_r = evecs_C0[:, :rank]  # (d, r)
    Lambda_r = evals_C0[:rank]  # (r,)

    # Project centered activations onto subspace: Y_r = H0_cent @ U_r (N, r)
    Y_r = H0_cent @ U_r

    # Empirical covariance in subspace: (r, r)
    Sigma_r = (Y_r.T @ Y_r) / N
    Sigma_r = 0.5 * (Sigma_r + Sigma_r.T)

    # Eigendecomposition of Sigma_r
    s_vals, s_vecs = np.linalg.eigh(Sigma_r)
    s_vals = np.maximum(s_vals, 1e-12)

    Sigma_r_inv_sqrt = s_vecs @ np.diag(1.0 / np.sqrt(s_vals)) @ s_vecs.T
    Sigma_r_sqrt = s_vecs @ np.diag(np.sqrt(s_vals)) @ s_vecs.T

    # Symmetric Bures-Wasserstein transport map:
    # M_r = Sigma_r^{-1/2} (Sigma_r^{1/2} Lambda_r Sigma_r^{1/2})^{1/2} Sigma_r^{-1/2}
    mid = Sigma_r_sqrt @ np.diag(Lambda_r) @ Sigma_r_sqrt
    mid = 0.5 * (mid + mid.T)
    m_vals, m_vecs = np.linalg.eigh(mid)
    m_vals = np.maximum(m_vals, 1e-12)
    mid_sqrt = m_vecs @ np.diag(np.sqrt(m_vals)) @ m_vecs.T
    M_r = Sigma_r_inv_sqrt @ mid_sqrt @ Sigma_r_inv_sqrt

    # Interpolate between identity and M_r using t:
    M_t = (1.0 - t) * np.eye(rank, dtype=np.float32) + t * M_r.astype(np.float32)

    # Transport in subspace and reconstruct
    Y_r_t = Y_r @ M_t
    H0_cent_t = H0_cent + (Y_r_t - Y_r) @ U_r.T

    # Add mean: interpolate mean shift by t
    mean_target = sample_mean_H0 + t * (m0 - sample_mean_H0)
    H0_repaired = H0_cent_t + mean_target

    meta = {
        "rank": rank,
        "t": t,
        "mean_eigval_target": float(np.mean(Lambda_r)),
        "mean_eigval_sample": float(np.mean(s_vals)),
    }
    return H0_repaired, meta


def forward_suffix(H0: np.ndarray, weights: List[np.ndarray]) -> np.ndarray:
    """Continues forward pass from H0 through layers 1..15."""
    curr = H0
    for w in weights[1:]:
        curr = np.maximum(curr @ w, 0.0)
    return np.mean(curr, axis=0)


def evaluate_loo_blend(
    y_calib_list: List[np.ndarray],
    y_sample_list: List[np.ndarray],
    y_true_list: List[np.ndarray],
) -> Tuple[float, List[float], float]:
    """Evaluates grouped Leave-One-MLP-Out (LOO) blending:
    y_blend = (1 - alpha) * y_calib + alpha * y_sample.
    Returns: (outer_loo_mse, per_mlp_loo_mse, refit_alpha)
    """
    n_mlps = len(y_calib_list)
    loo_preds = []
    loo_errors = []

    # 1. Outer LOO loop
    for holdout_idx in range(n_mlps):
        # Fit alpha on remaining 7 MLPs
        num = 0.0
        denom = 0.0
        for i in range(n_mlps):
            if i == holdout_idx:
                continue
            e_calib = y_calib_list[i] - y_true_list[i]
            d = y_sample_list[i] - y_calib_list[i]
            # loss = ||e_calib + alpha * d||^2 -> alpha = -(e_calib . d) / ||d||^2
            num -= np.dot(e_calib, d)
            denom += np.dot(d, d)

        alpha_fit = float(np.clip(num / max(denom, 1e-12), 0.0, 1.0))

        # Predict on holdout MLP
        pred_holdout = (1.0 - alpha_fit) * y_calib_list[holdout_idx] + alpha_fit * y_sample_list[holdout_idx]
        mse_holdout = float(np.mean((pred_holdout - y_true_list[holdout_idx]) ** 2))
        loo_preds.append(pred_holdout)
        loo_errors.append(mse_holdout)

    mean_loo_mse = float(np.mean(loo_errors))

    # 2. Refit on all 8 MLPs
    all_num = 0.0
    all_denom = 0.0
    for i in range(n_mlps):
        e_calib = y_calib_list[i] - y_true_list[i]
        d = y_sample_list[i] - y_calib_list[i]
        all_num -= np.dot(e_calib, d)
        all_denom += np.dot(d, d)
    refit_alpha = float(np.clip(all_num / max(all_denom, 1e-12), 0.0, 1.0))

    return mean_loo_mse, loo_errors, refit_alpha


def estimate_compute_utilization(N: int, op_type: str, rank: int = 0) -> float:
    """Estimates compute utilization for an estimator that runs analytic branch + repaired MC."""
    d = 1024
    analytic_flops = 1.05e11  # Measured standard blended Hermite
    l0_flops = 2.0 * N * d * d

    repair_flops = 0.0
    if op_type == "pos_mean" or op_type == "add_mean":
        repair_flops = 4.0 * N * d
    elif op_type == "cov_transport":
        repair_flops = 4.0 * N * d * rank + 2.0 * N * (rank**2) + 20.0 * (rank**3)

    suffix_flops = 15.0 * 2.0 * N * d * d
    total_flops = analytic_flops + l0_flops + repair_flops + suffix_flops
    return float(total_flops / _BUDGET_FLOPS)


def run_experiment_p4n_04():
    print("=" * 80)
    print("P4N-04: EARLY-LAYER SAMPLE REPAIR USING EXACT LAYER-0 MOMENTS")
    print("=" * 80)

    ds = load_dataset(DATASET_PATH, split="mini")
    protocol_version, salt = resolve_seed_context(ds)
    print(f"Dataset: {DATASET_PATH} (protocol={protocol_version}, salt={salt})")

    STRENGTHS = [0.25, 0.5, 1.0]
    RANKS = [32, 128]
    SAMPLE_COUNTS = [2048, 8192, 16384]

    mlp_data = []
    print("\n[Phase 1] Precomputing exact layer 0 moments and analytic predictions...")
    # Restrict to the 8 Phase 2 panel MLPs
    ds_8 = [ds[i] for i in range(8)]
    for idx, row in enumerate(ds_8):
        mlp_name = row.get("mlp_name", f"mlp_{idx}")
        mlp = MLP.from_row(row, seed_protocol_version=protocol_version, seed_salt=salt)
        weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
        y_true = np.asarray(row["all_layer_means"][-1], dtype=np.float32)

        m0, C0 = compute_exact_layer0_moments(weights[0])

        evals_C0, evecs_C0 = np.linalg.eigh(C0)
        idx_sort = np.argsort(evals_C0)[::-1]
        evals_C0 = evals_C0[idx_sort]
        evecs_C0 = evecs_C0[:, idx_sort]

        c_stack = compute_analytic_covariance_stack(mlp)
        c_calib = (_S_PRIOR * c_stack[-1]).astype(np.float32)
        c_mse = float(np.mean((c_calib - y_true) ** 2))

        mlp_data.append({
            "idx": idx,
            "mlp_name": mlp_name,
            "mlp": mlp,
            "weights": weights,
            "y_true": y_true,
            "m0": m0,
            "C0": C0,
            "evals_C0": evals_C0,
            "evecs_C0": evecs_C0,
            "c_calib": c_calib,
            "c_mse": c_mse,
        })
        print(f"  MLP {idx+1}/8: {mlp_name} | Analytic MSE: {c_mse:.6e} | C0 trace: {np.sum(evals_C0):.2f}")

    # -------------------------------------------------------------
    # Step 1: Pre-generate samples and baseline at N=2048
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: SCREENING OPERATORS AND STRENGTHS AT N=2048")
    print("=" * 80)

    N_SCREEN = 2048
    # Precompute H0 and y_unrep once per MLP
    precomputed_samples_2k = []
    print("Precomputing N=2048 activations and baseline unrepaired outputs...")
    for item in mlp_data:
        mlp = item["mlp"]
        weights = item["weights"]
        rng = np.random.default_rng(mlp.seed + 30003)
        half = N_SCREEN // 2
        x_half = rng.standard_normal((half, mlp.width), dtype=np.float32)
        X = np.concatenate([x_half, -x_half], axis=0)
        H0 = np.maximum(X @ weights[0], 0.0)
        y_unrep = forward_suffix(H0, weights)
        precomputed_samples_2k.append({
            "H0": H0,
            "y_unrep": y_unrep,
        })

    operators_to_screen = []
    operators_to_screen.append(("unrepaired", 0.0, 0))
    for t in STRENGTHS:
        operators_to_screen.append(("pos_mean", t, 0))
    for t in STRENGTHS:
        operators_to_screen.append(("add_mean", t, 0))
    for rank in RANKS:
        for t in STRENGTHS:
            operators_to_screen.append(("cov_transport", t, rank))

    screen_results = {}

    for op_type, t, rank in operators_to_screen:
        cfg_name = f"{op_type}_t{t}" if rank == 0 else f"{op_type}_r{rank}_t{t}"
        print(f"\n--- Evaluating Config: {cfg_name} (N={N_SCREEN}) ---")

        raw_mse_list = []
        paired_diff_norm_list = []
        y_sample_list = []
        op_meta_list = []

        t_start = time.time()
        for i, item in enumerate(mlp_data):
            weights = item["weights"]
            y_true = item["y_true"]
            m0 = item["m0"]
            C0 = item["C0"]
            evals_C0 = item["evals_C0"]
            evecs_C0 = item["evecs_C0"]

            H0 = precomputed_samples_2k[i]["H0"]
            y_unrep = precomputed_samples_2k[i]["y_unrep"]

            meta = {}
            if op_type == "unrepaired":
                y_sample = y_unrep
                paired_diff = 0.0
            else:
                if op_type == "pos_mean":
                    H0_rep, meta = repair_positive_mean_matching(H0, m0, t=t)
                elif op_type == "add_mean":
                    H0_rep, meta = repair_additive_mean_matching(H0, m0, t=t)
                elif op_type == "cov_transport":
                    H0_rep, meta = repair_covariance_transport(H0, m0, C0, rank=rank, t=t, evecs_C0=evecs_C0, evals_C0=evals_C0)

                y_sample = forward_suffix(H0_rep, weights)
                paired_diff = float(np.mean((y_sample - y_unrep) ** 2))

            mse = float(np.mean((y_sample - y_true) ** 2))
            raw_mse_list.append(mse)
            paired_diff_norm_list.append(paired_diff)
            y_sample_list.append(y_sample)
            op_meta_list.append(meta)

        elapsed = time.time() - t_start
        mean_raw_mse = float(np.mean(raw_mse_list))

        y_calib_list = [item["c_calib"] for item in mlp_data]
        y_true_list = [item["y_true"] for item in mlp_data]
        loo_mse, per_mlp_loo, refit_alpha = evaluate_loo_blend(y_calib_list, y_sample_list, y_true_list)

        util = estimate_compute_utilization(N_SCREEN, op_type, rank=rank)
        multiplier = max(0.10, util)
        scored_error = loo_mse * multiplier

        screen_results[cfg_name] = {
            "op_type": op_type,
            "t": t,
            "rank": rank,
            "N": N_SCREEN,
            "raw_mse_mean": mean_raw_mse,
            "raw_mse_per_mlp": raw_mse_list,
            "paired_diff_norm_mean": float(np.mean(paired_diff_norm_list)),
            "loo_mse": loo_mse,
            "refit_alpha": refit_alpha,
            "compute_utilization": util,
            "multiplier": multiplier,
            "scored_error": scored_error,
            "elapsed_s": elapsed,
            "op_meta": op_meta_list[0] if op_meta_list else {},
        }

        print(f"  Raw MSE:       {mean_raw_mse:.6e}")
        print(f"  LOO Blend MSE: {loo_mse:.6e} (refit alpha: {refit_alpha:.4f})")
        print(f"  Scored Error:  {scored_error:.6e} (util: {util*100:.2f}%, mult: {multiplier:.4f})")
        print(f"  Paired Diff:   {float(np.mean(paired_diff_norm_list)):.6e}")

    # Summary table
    print("\n" + "=" * 90)
    print("SCREENING RESULTS TABLE (N=2048)")
    print("=" * 90)
    print(f"{'Configuration':<24} | {'Raw MSE':<12} | {'LOO Blend':<12} | {'Alpha':<6} | {'Scored Error':<12} | {'vs Ctrl (%)':<10}")
    print("-" * 90)

    for name, res in screen_results.items():
        diff_ctrl = ((res["scored_error"] - _CONTROL_SCORE) / _CONTROL_SCORE) * 100.0
        print(f"{name:<24} | {res['raw_mse_mean']:<12.6e} | {res['loo_mse']:<12.6e} | {res['refit_alpha']:<6.3f} | {res['scored_error']:<12.6e} | {diff_ctrl:+9.2f}%")

    # Select best 2 repair configurations
    repair_candidates = [k for k in screen_results.keys() if k != "unrepaired_t0.0"]
    sorted_by_loo = sorted(repair_candidates, key=lambda k: screen_results[k]["loo_mse"])
    sorted_by_raw = sorted(repair_candidates, key=lambda k: screen_results[k]["raw_mse_mean"])

    best_repair_1 = sorted_by_loo[0]
    best_repair_2 = sorted_by_loo[1] if len(sorted_by_loo) > 1 else sorted_by_raw[0]
    if best_repair_2 == best_repair_1 and len(sorted_by_raw) > 1:
        best_repair_2 = [k for k in sorted_by_raw if k != best_repair_1][0]

    print(f"\nBest 2 repair configurations selected for Sample Count Ladder:")
    print(f"  1. {best_repair_1} (LOO MSE: {screen_results[best_repair_1]['loo_mse']:.6e}, Raw MSE: {screen_results[best_repair_1]['raw_mse_mean']:.6e})")
    print(f"  2. {best_repair_2} (LOO MSE: {screen_results[best_repair_2]['loo_mse']:.6e}, Raw MSE: {screen_results[best_repair_2]['raw_mse_mean']:.6e})")

    # -------------------------------------------------------------
    # Step 2: Sample Count Ladder N in {2048, 8192, 16384}
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: SAMPLE COUNT LADDER TO EXPOSE BIAS VS VARIANCE TRENDS")
    print("=" * 80)

    configs_to_ladder = [
        ("unrepaired", screen_results["unrepaired_t0.0"]["op_type"], 0.0, 0),
        (best_repair_1, screen_results[best_repair_1]["op_type"], screen_results[best_repair_1]["t"], screen_results[best_repair_1]["rank"]),
        (best_repair_2, screen_results[best_repair_2]["op_type"], screen_results[best_repair_2]["t"], screen_results[best_repair_2]["rank"]),
    ]

    ladder_results = {}
    for cfg_label, _, _, _ in configs_to_ladder:
        ladder_results[cfg_label] = {}

    for N in SAMPLE_COUNTS:
        print(f"\n=======================================================")
        print(f"Evaluating Sample Count N = {N} across selected configs")
        print(f"=======================================================")

        # Precompute activations and unrepaired outputs for count N
        cache_N = []
        for item in mlp_data:
            mlp = item["mlp"]
            weights = item["weights"]
            rng = np.random.default_rng(mlp.seed + 40004 + N)
            half = N // 2
            x_half = rng.standard_normal((half, mlp.width), dtype=np.float32)
            X = np.concatenate([x_half, -x_half], axis=0)
            H0 = np.maximum(X @ weights[0], 0.0)
            y_unrep = forward_suffix(H0, weights)
            cache_N.append({
                "H0": H0,
                "y_unrep": y_unrep,
            })

        for cfg_label, op_type, t, rank in configs_to_ladder:
            print(f"Running {cfg_label} at N={N}...")
            raw_mse_list = []
            paired_diff_list = []
            y_sample_list = []
            corr_with_unrep_err = []

            t_start = time.time()
            for i, item in enumerate(mlp_data):
                weights = item["weights"]
                y_true = item["y_true"]
                m0 = item["m0"]
                C0 = item["C0"]
                evals_C0 = item["evals_C0"]
                evecs_C0 = item["evecs_C0"]

                H0 = cache_N[i]["H0"]
                y_unrep = cache_N[i]["y_unrep"]

                if op_type == "unrepaired":
                    y_sample = y_unrep
                    diff = np.zeros_like(y_sample)
                    diff_norm = 0.0
                    corr = 0.0
                else:
                    if op_type == "pos_mean":
                        H0_rep, _ = repair_positive_mean_matching(H0, m0, t=t)
                    elif op_type == "add_mean":
                        H0_rep, _ = repair_additive_mean_matching(H0, m0, t=t)
                    elif op_type == "cov_transport":
                        H0_rep, _ = repair_covariance_transport(H0, m0, C0, rank=rank, t=t, evecs_C0=evecs_C0, evals_C0=evals_C0)

                    y_sample = forward_suffix(H0_rep, weights)
                    diff = y_sample - y_unrep
                    diff_norm = float(np.mean(diff ** 2))
                    err_unrep = y_unrep - y_true
                    denom = float(np.linalg.norm(diff) * np.linalg.norm(err_unrep))
                    corr = float(np.dot(diff, err_unrep) / max(denom, 1e-12))

                mse = float(np.mean((y_sample - y_true) ** 2))
                raw_mse_list.append(mse)
                paired_diff_list.append(diff_norm)
                y_sample_list.append(y_sample)
                corr_with_unrep_err.append(corr)

            elapsed = time.time() - t_start
            mean_raw_mse = float(np.mean(raw_mse_list))

            y_calib_list = [item["c_calib"] for item in mlp_data]
            y_true_list = [item["y_true"] for item in mlp_data]
            loo_mse, per_mlp_loo, refit_alpha = evaluate_loo_blend(y_calib_list, y_sample_list, y_true_list)

            util = estimate_compute_utilization(N, op_type, rank=rank)
            multiplier = max(0.10, util)
            scored_error = loo_mse * multiplier

            ladder_results[cfg_label][str(N)] = {
                "N": N,
                "raw_mse_mean": mean_raw_mse,
                "raw_mse_per_mlp": raw_mse_list,
                "paired_diff_mean": float(np.mean(paired_diff_list)),
                "corr_diff_unrep_err": float(np.mean(corr_with_unrep_err)),
                "loo_mse": loo_mse,
                "refit_alpha": refit_alpha,
                "compute_utilization": util,
                "multiplier": multiplier,
                "scored_error": scored_error,
                "elapsed_s": elapsed,
            }

            print(f"  -> Raw MSE: {mean_raw_mse:.6e} | LOO MSE: {loo_mse:.6e} (alpha={refit_alpha:.3f}) | Scored: {scored_error:.6e} | Paired diff: {float(np.mean(paired_diff_list)):.6e}")

    # -------------------------------------------------------------
    # Step 3: Bias vs Variance Analysis
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("BIAS VS VARIANCE ANALYSIS ACROSS SAMPLE COUNTS")
    print("=" * 90)
    print(f"{'Config':<24} | {'N=2048 Raw':<12} | {'N=8192 Raw':<12} | {'N=16384 Raw':<12} | {'Scaling (2k->16k)':<18} | {'Regime'}")
    print("-" * 90)

    for cfg_label in configs_to_ladder:
        name = cfg_label[0]
        mse_2k = ladder_results[name]["2048"]["raw_mse_mean"]
        mse_8k = ladder_results[name]["8192"]["raw_mse_mean"]
        mse_16k = ladder_results[name]["16384"]["raw_mse_mean"]

        scaling_factor = mse_2k / max(mse_16k, 1e-12)
        regime = "Variance-dominated" if scaling_factor >= 5.0 else ("Mixed" if scaling_factor >= 2.5 else "Bias-dominated")
        print(f"{name:<24} | {mse_2k:<12.6e} | {mse_8k:<12.6e} | {mse_16k:<12.6e} | {scaling_factor:<6.2f}x (ideal: 8x) | {regime}")

    # -------------------------------------------------------------
    # Step 4: Gate Checks & Final Synthesis
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GATE CHECKS (PHASE5_NEXT_EXPERIMENTS.md SECTION 10)")
    print("=" * 80)

    best_overall_scored = min(
        [ladder_results[cfg[0]][str(N)]["scored_error"] for cfg in configs_to_ladder for N in SAMPLE_COUNTS]
    )
    gate1_threshold = float(0.85 * _CONTROL_SCORE)
    gate1_pass = bool(best_overall_scored < gate1_threshold)

    print(f"Gate 1: 15% adjusted improvement vs control ({_CONTROL_SCORE:.6e})")
    print(f"  Target:            < {gate1_threshold:.6e}")
    print(f"  Best Scored Error:   {best_overall_scored:.6e}")
    print(f"  Result:              {'PASS' if gate1_pass else 'REJECT'}")

    unrep_raw_2k = ladder_results["unrepaired"]["2048"]["raw_mse_mean"]
    best_rep_raw_2k = min([ladder_results[cfg[0]]["2048"]["raw_mse_mean"] for cfg in configs_to_ladder if cfg[0] != "unrepaired"])
    sampling_residual_reduction = float(((unrep_raw_2k - best_rep_raw_2k) / unrep_raw_2k) * 100.0)
    gate2_pass = bool(sampling_residual_reduction >= 30.0)

    print(f"\nGate 2: >= 30% reduction in sampling residual at matched cost")
    print(f"  Unrepaired Raw MSE (N=2048):  {unrep_raw_2k:.6e}")
    print(f"  Best Repaired Raw MSE (N=2048):{best_rep_raw_2k:.6e}")
    print(f"  Residual Reduction:           {sampling_residual_reduction:+.2f}%")
    print(f"  Result:                       {'PASS' if gate2_pass else 'REJECT'}")

    # Save complete diagnostic artifact
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAGNOSTICS_DIR / "p4n_04_early_repair.json"

    diagnostic_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_id": "P4N-04",
        "description": "Early-layer sample repair using exact first-layer moments (m0 and Cho-Saul C0)",
        "control_score": float(_CONTROL_SCORE),
        "screening_N2048": screen_results,
        "sample_count_ladder": ladder_results,
        "gates": {
            "gate1_15pct_adjusted_gain": {
                "target": float(gate1_threshold),
                "achieved": float(best_overall_scored),
                "pass": bool(gate1_pass),
            },
            "gate2_30pct_residual_reduction": {
                "unrepaired_raw_mse_2048": float(unrep_raw_2k),
                "best_repaired_raw_mse_2048": float(best_rep_raw_2k),
                "reduction_pct": float(sampling_residual_reduction),
                "pass": bool(gate2_pass),
            },
        },
        "selected_best_repairs": [best_repair_1, best_repair_2],
    }

    with open(out_path, "w") as f:
        json.dump(diagnostic_payload, f, indent=2, default=numpy_json_default)

    print(f"\nComplete diagnostic results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_p4n_04()
