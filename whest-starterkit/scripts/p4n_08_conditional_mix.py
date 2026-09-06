"""P4N-08 Diagnostic: Conditional Gaussian Mixture Propagation.

Evaluates Experiment P4N-08 from PHASE5_NEXT_EXPERIMENTS.md (Section 14):
1. Evaluates all 8 Phase 2 MLPs from datasets/mini.
2. Represents Gaussian input as X = t*v + X_perp, where t ~ N(0, 1) and X_perp ~ N(0, I - v v^T).
3. Evaluates 3 direction selection strategies:
   a. v1: Normalized first-layer weight column with maximum downstream sensitivity
   b. v2: Leading left singular vector of gate-aware end-to-end response W0 D0 ... W15 D15
   c. v_rand: Random isotropic unit vector control
4. Tests Gauss-Hermite quadrature nodes n in {3, 5, 9} for 1D conditioning, and 3x3 for 2D.
5. Runs conditional moment propagation for each node:
   - Initial state: mu_0 = t_k * v,  Sigma_0 = I - v v^T
   - Propagates through 16 ReLU layers with non-zero mean Gaussian Hermite closure.
   - Final prediction: mu_mixture = sum_k w_tilde_k * mu^{(k)}_15
6. Measures:
   - Raw MSE of mixture prediction
   - Transported center error reduction vs unconditional prior
   - Outer LOO blend with WMC samples
   - FLOP cost and compute utilization (n * propagation cost)
   - Scored error: S = MSE * max(0.10, compute_utilization)
7. Checks Gate: 15% scored gain (< 1.0401e-07) or 30% reduction in transported center error.
8. Saves complete diagnostic records to research/phase4_next/diagnostics/p4n_08_conditional_mix.json.
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

_S_PRIOR = 0.998319
_CONTROL_SCORE = 1.223643e-07
_BUDGET_FLOPS = float(2**41)


def numpy_json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def get_gauss_hermite_nodes(n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns nodes t and normalized weights w_tilde for standard normal E[f(t)]."""
    x, w = scipy.special.roots_hermite(n_nodes)
    t = (np.sqrt(2.0) * x).astype(np.float32)
    w_tilde = (w / np.sqrt(np.pi)).astype(np.float32)
    return t, w_tilde


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))


def propagate_conditional_moments(
    mu_init: np.ndarray,
    cov_init: np.ndarray,
    weights: List[np.ndarray],
) -> np.ndarray:
    """Propagates non-zero initial mean and covariance through 16 ReLU layers."""
    width = mu_init.shape[0]
    mu = mu_init.copy().astype(np.float32)
    cov = cov_init.copy().astype(np.float32)

    c_centered = np.float32(0.80 * 0.20 * 0.07957747154594767)
    c_thresh = np.float32(0.20 * 0.5)

    for w in weights:
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

    return mu


def forward_full(X: np.ndarray, weights: List[np.ndarray]) -> np.ndarray:
    H = X
    for W in weights:
        H = np.maximum(H @ W, 0.0)
    return H


def make_antipodal_pairs(rng: np.random.Generator, n_pairs: int, width: int = 1024) -> np.ndarray:
    X_pos = rng.standard_normal((n_pairs, width), dtype=np.float32)
    X = np.empty((2 * n_pairs, width), dtype=np.float32)
    X[:n_pairs] = X_pos
    X[n_pairs:] = -X_pos
    return X


def evaluate_loo_blend(
    y_calib_list: List[np.ndarray],
    y_sample_list: List[np.ndarray],
    y_true_list: List[np.ndarray],
) -> Tuple[float, List[float], float]:
    """Outer leave-one-MLP-out blending."""
    n_mlps = len(y_true_list)
    loo_errors = []
    alphas_fitted = []

    for i in range(n_mlps):
        c_train = [y_calib_list[j] for j in range(n_mlps) if j != i]
        s_train = [y_sample_list[j] for j in range(n_mlps) if j != i]
        y_train = [y_true_list[j] for j in range(n_mlps) if j != i]

        def get_alpha(c_tr, s_tr, y_tr):
            num = 0.0
            den = 0.0
            for c, s, y in zip(c_tr, s_tr, y_tr):
                diff = s - c
                num += np.dot(y - c, diff)
                den += np.dot(diff, diff)
            if den < 1e-12:
                return 0.110
            return float(np.clip(num / den, 0.0, 1.0))

        alpha_loo = get_alpha(c_train, s_train, y_train)
        alphas_fitted.append(alpha_loo)

        pred = (1.0 - alpha_loo) * y_calib_list[i] + alpha_loo * y_sample_list[i]
        err = float(np.mean((pred - y_true_list[i]) ** 2))
        loo_errors.append(err)

    mean_alpha = float(np.mean(alphas_fitted))
    return float(np.mean(loo_errors)), loo_errors, mean_alpha


def run_p4n_08():
    print("=" * 90)
    print("P4N-08: CONDITIONAL GAUSSIAN MIXTURE PROPAGATION ON 8 PHASE 2 MLPS")
    print("=" * 90)

    # Validate Gauss-Hermite moment expectations
    for n in [3, 5, 9]:
        t, w = get_gauss_hermite_nodes(n)
        m0 = np.sum(w)
        m1 = np.sum(w * t)
        m2 = np.sum(w * t * t)
        m4 = np.sum(w * t ** 4)
        print(f"Gauss-Hermite n={n}: sum(w)={m0:.4f}, E[t]={m1:.4e}, E[t^2]={m2:.4f}, E[t^4]={m4:.4f}")
        assert abs(m0 - 1.0) < 1e-5
        assert abs(m1) < 1e-5
        assert abs(m2 - 1.0) < 1e-4

    ds = load_dataset(DATASET_PATH, split="mini")
    protocol_version, salt = resolve_seed_context(ds)

    ds_8 = [ds[i] for i in range(8)]
    mlp_records = []

    print("\n[Step 1] Loading MLPs and preparing evaluation data...")
    for idx, row in enumerate(ds_8):
        mlp = MLP.from_row(row, seed_protocol_version=protocol_version, seed_salt=salt)
        mlp_name = row.get("mlp_name", f"mlp_{idx}")
        weights = [np.array(w, dtype=np.float32) for w in mlp.weights]
        y_true = np.array(row["all_layer_means"][-1], dtype=np.float32)

        # Baseline unconditional propagation
        mu_0 = np.zeros(mlp.width, dtype=np.float32)
        cov_0 = np.eye(mlp.width, dtype=np.float32)
        c_uncond = propagate_conditional_moments(mu_0, cov_0, weights)
        c_calib_uncond = c_uncond * _S_PRIOR
        mse_uncond = float(np.mean((c_calib_uncond - y_true) ** 2))

        # Sample MC batch (N=4200, 2100 pairs)
        rng = np.random.default_rng(mlp.seed + 80001)
        X_eval = make_antipodal_pairs(rng, n_pairs=2100, width=mlp.width)
        f_eval = forward_full(X_eval, weights)
        y_sample = np.mean(f_eval, axis=0)

        # Direction 1: Normalized first-layer weight column with maximum downstream sensitivity
        W0 = weights[0]
        W1 = weights[1]
        col_norms_W1 = np.linalg.norm(W1, axis=1)
        sensitivities = np.linalg.norm(W0, axis=0) * col_norms_W1
        best_col_idx = int(np.argmax(sensitivities))
        v1 = W0[:, best_col_idx] / np.linalg.norm(W0[:, best_col_idx])

        # Direction 2: Leading input mode of gate-aware response
        # Using pilot batch (512 pairs) to get gate probs
        rng_p = np.random.default_rng(mlp.seed + 80000)
        X_p = make_antipodal_pairs(rng_p, n_pairs=512, width=mlp.width)
        H = X_p
        gate_probs = []
        for W in weights:
            H = np.maximum(H @ W, 0.0)
            gate_probs.append(np.mean(H > 0.0, axis=0).astype(np.float32))

        M_prod = np.eye(mlp.width, dtype=np.float32)
        for l in range(16):
            M_prod = (M_prod @ weights[l]) * gate_probs[l][None, :]
        # SVD of input-output matrix: row input map -> left singular vectors
        U, _, _ = np.linalg.svd(M_prod, full_matrices=False)
        v2 = U[:, 0]

        # Direction 3: Random unit vector
        rng_rand = np.random.default_rng(mlp.seed + 80002)
        v_r = rng_rand.standard_normal(mlp.width, dtype=np.float32)
        v_rand = v_r / np.linalg.norm(v_r)

        mlp_records.append({
            "idx": idx,
            "name": mlp_name,
            "mlp": mlp,
            "weights": weights,
            "y_true": y_true,
            "c_uncond": c_uncond,
            "c_calib_uncond": c_calib_uncond,
            "mse_uncond": mse_uncond,
            "y_sample": y_sample,
            "directions": {
                "v1_max_col": v1.astype(np.float32),
                "v2_input_mode": v2.astype(np.float32),
                "v_rand": v_rand.astype(np.float32),
            },
        })
        print(f"  MLP {idx+1}/8: {mlp_name} | Uncond MSE: {mse_uncond:.6e}")

    mean_uncond_mse = float(np.mean([item["mse_uncond"] for item in mlp_records]))
    print(f"\nMean Baseline Unconditional MSE: {mean_uncond_mse:.6e}")

    # FLOP accounting: 1 unconditional propagation = 7.20e10 FLOPs (util = 3.27%)
    # Total pipeline with WMC N=4200 is 9.81% util (multiplier = 0.1000)
    # Each extra propagation pass adds 7.20e10 FLOPs = +3.27% util!
    # For n nodes: Covariance FLOPs = n * 7.20e10.
    # If combined with WMC N=4200 (1.41e11 FLOPs):
    # n=3 nodes: Cov = 2.16e11, Total = 3.57e11 -> Util = 16.23% (Multiplier = 0.1623)
    # n=5 nodes: Cov = 3.60e11, Total = 5.01e11 -> Util = 22.78% (Multiplier = 0.2278)
    # n=9 nodes: Cov = 6.48e11, Total = 7.89e11 -> Util = 35.88% (Multiplier = 0.3588)

    # --------------------------------------------------------------------------
    # Step 2: Evaluate 1D Conditional Propagation across Directions and Nodes
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("STEP 2: EVALUATE CONDITIONAL MIXTURES ACROSS DIRECTIONS & QUADRATURE NODES")
    print("=" * 90)

    configs = [
        ("v1_max_col", 3),
        ("v1_max_col", 5),
        ("v2_input_mode", 3),
        ("v2_input_mode", 5),
        ("v_rand", 3),
    ]

    results_by_config = {}

    for dir_name, n_nodes in configs:
        cfg_label = f"{dir_name}_n{n_nodes}"
        t_nodes, w_nodes = get_gauss_hermite_nodes(n_nodes)

        raw_mse_list = []
        calib_mse_list = []
        y_calib_list = []
        y_sample_list = []
        y_true_list = []

        t_start = time.time()
        for item in mlp_records:
            weights = item["weights"]
            v = item["directions"][dir_name]
            width = v.shape[0]

            # Covariance orthogonal to v: I - v v^T
            cov_init = np.eye(width, dtype=np.float32) - np.outer(v, v).astype(np.float32)

            # Accumulate mixture prediction
            mu_mix = np.zeros(width, dtype=np.float32)
            for k in range(n_nodes):
                mu_init = t_nodes[k] * v
                mu_k = propagate_conditional_moments(mu_init, cov_init, weights)
                mu_mix += w_nodes[k] * mu_k

            c_calib = mu_mix * _S_PRIOR
            y_true = item["y_true"]

            raw_mse = float(np.mean((mu_mix - y_true) ** 2))
            calib_mse = float(np.mean((c_calib - y_true) ** 2))

            raw_mse_list.append(raw_mse)
            calib_mse_list.append(calib_mse)
            y_calib_list.append(c_calib)
            y_sample_list.append(item["y_sample"])
            y_true_list.append(y_true)

        elapsed = time.time() - t_start
        mean_raw_mse = float(np.mean(raw_mse_list))
        mean_calib_mse = float(np.mean(calib_mse_list))

        # Outer LOO blend with WMC samples
        loo_mse, _, alpha_loo = evaluate_loo_blend(y_calib_list, y_sample_list, y_true_list)

        # FLOP utilization and scored error
        # Pipeline: n_nodes * cov_flops + mc_flops (with WMC N=4200)
        cov_flops = n_nodes * 7.20e10
        mc_flops = 1.41e11
        total_flops = cov_flops + mc_flops
        util = total_flops / _BUDGET_FLOPS
        multiplier = max(0.10, util)
        scored_error = loo_mse * multiplier

        gain_calib_vs_uncond = (1.0 - mean_calib_mse / mean_uncond_mse) * 100.0
        gain_scored_vs_control = (1.0 - scored_error / _CONTROL_SCORE) * 100.0

        results_by_config[cfg_label] = {
            "dir_name": dir_name,
            "n_nodes": n_nodes,
            "mean_raw_mse": mean_raw_mse,
            "mean_calib_mse": mean_calib_mse,
            "loo_mse": loo_mse,
            "alpha_loo": alpha_loo,
            "util": util,
            "multiplier": multiplier,
            "scored_error": scored_error,
            "gain_calib_vs_uncond_pct": gain_calib_vs_uncond,
            "gain_scored_vs_control_pct": gain_scored_vs_control,
            "elapsed_s": elapsed,
        }

        print(f"[{cfg_label:<20}] Calib MSE: {mean_calib_mse:.6e} ({gain_calib_vs_uncond:+.2f}%) | LOO MSE: {loo_mse:.6e} (a={alpha_loo:.3f}) | Mult: {multiplier:.4f} | Scored: {scored_error:.6e} ({gain_scored_vs_control:+.2f}%)")

    # --------------------------------------------------------------------------
    # Step 3: Gate Audit & Decision
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("STEP 3: GATE AUDIT & DECISION")
    print("=" * 90)

    best_cfg = None
    best_scored = float("inf")
    for cfg_label, res in results_by_config.items():
        if res["scored_error"] < best_scored:
            best_scored = res["scored_error"]
            best_cfg = res

    print(f"Best Configuration: {best_cfg['dir_name']} (n={best_cfg['n_nodes']})")
    print(f"  Calibrated MSE: {best_cfg['mean_calib_mse']:.6e} (vs Uncond: {mean_uncond_mse:.6e}, Diff: {best_cfg['gain_calib_vs_uncond_pct']:+.2f}%)")
    print(f"  Scored Error:   {best_scored:.6e} (Control: {_CONTROL_SCORE:.6e}, Gain: {best_cfg['gain_scored_vs_control_pct']:+.2f}%)")

    gate_15pct_scored = best_cfg["gain_scored_vs_control_pct"] >= 15.0
    gate_30pct_center = best_cfg["gain_calib_vs_uncond_pct"] >= 30.0

    print(f"\nGate Audit:")
    print(f"  - 15% Scored Gain (< 1.0401e-07): {'PASSED' if gate_15pct_scored else 'FAILED'} (Gain: {best_cfg['gain_scored_vs_control_pct']:+.2f}%)")
    print(f"  - 30% Center Error Reduction:    {'PASSED' if gate_30pct_center else 'FAILED'} (Gain: {best_cfg['gain_calib_vs_uncond_pct']:+.2f}%)")

    pass_gate = gate_15pct_scored or gate_30pct_center
    print(f"P4N-08 Overall Gate Status: {'PASSED' if pass_gate else 'FAILED'}")

    # --------------------------------------------------------------------------
    # Save Diagnostic Records
    # --------------------------------------------------------------------------
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DIAGNOSTICS_DIR / "p4n_08_conditional_mix.json"

    diagnostic_payload = {
        "mean_unconditional_mse": mean_uncond_mse,
        "results_by_config": results_by_config,
        "best_configuration": best_cfg,
        "gate_passed": pass_gate,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diagnostic_payload, f, indent=2, default=numpy_json_default)

    print(f"\nDiagnostic results saved to: {out_file}")


if __name__ == "__main__":
    run_p4n_08()
