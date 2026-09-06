"""p6_mapped_control_check.py

Phase 6 Stage P6-06 Secondary Construction:
Replaces the analytic center in the layer-14 mapped control variate with the Phase 6 K3-simple center.
Preserves coordinate mapping and compares against the control using the baseline center.
Uses independent pilot (N=2048) and evaluation (N=4200) batches with independent whitening.
Tests fixed mixture weights: 0.0, 0.5, 1.0.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini"
RESULTS_DIR = REPO_ROOT / "research" / "phase6" / "results"
PREDICTIONS_DIR = REPO_ROOT / "research" / "phase6" / "predictions"
FIXTURE_PATH = REPO_ROOT / "research" / "phase6" / "fixtures" / "panel_8mlp_weights.npz"

from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP

FROZEN_CONTROL_RAW_MSE = 1.2236425135370155e-06
S0 = 0.998319


def norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def norm_cdf(x):
    import scipy.special
    return 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))


def compute_baseline_covariance_stack(mlp: MLP) -> np.ndarray:
    """Computes exact dual-kernel blended Hermite baseline covariance stack (16, 1024)."""
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


def forward_network(X, weights):
    """Computes all-layer activations and gates."""
    depth = len(weights)
    H = [X]
    gates = []
    curr = X
    for layer_idx in range(depth):
        W = weights[layer_idx]
        Z = curr @ W
        gate = (Z > 0).astype(np.float32)
        curr = np.maximum(Z, 0.0)
        H.append(curr)
        gates.append(gate)
    return H[1:], gates


def run_mapped_control_comparison():
    print("=======================================================", flush=True)
    print("Executing Stage P6-06: Layer-14 Mapped Control Variate Comparison", flush=True)
    print("=======================================================", flush=True)

    ds = load_dataset(DATASET_PATH, split="mini")
    protocol_version, salt = resolve_seed_context(ds)

    # Load Phase 6 K3-simple uncompressed predictions
    p6_preds_file = PREDICTIONS_DIR / "P6-03_K3-simple_preds.npy"
    if not p6_preds_file.exists():
        raise FileNotFoundError(f"P6 predictions {p6_preds_file} not found!")
    p6_k3_preds = np.load(p6_preds_file)
    print(f"Loaded Phase 6 K3-simple predictions: shape {p6_k3_preds.shape}", flush=True)

    FIXED_8 = [
        "logan-fitzgerald", "william-graves", "raymond-barnes", "steven-rice",
        "sarah-kelley", "christopher-morales", "cheryl-graham", "renee-park"
    ]
    panel_rows = [row for row in ds if row.get("mlp_name") in FIXED_8]

    N_PILOT = 2048
    N_EVAL = 4200
    ALPHA_BLEND = 0.110

    results_baseline_center = []
    results_p6_center = []
    mixture_results = {0.0: [], 0.5: [], 1.0: []}

    for mlp_idx, row in enumerate(panel_rows):
        mlp_name = row.get("mlp_name", f"mlp_{mlp_idx}")
        mlp = MLP.from_row(row, seed_protocol_version=protocol_version, seed_salt=salt)
        weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
        true_means = np.asarray(row["all_layer_means"], dtype=np.float32)
        y_true = true_means[-1]

        # 1. Analytic centers
        base_stack = compute_baseline_covariance_stack(mlp)
        base_center_14 = base_stack[14]
        p6_center_14 = p6_k3_preds[mlp_idx, 14]

        # 2. Independent pilot stream (N=2048) with antithetic sampling
        rng_pilot = np.random.default_rng(mlp.seed + 10001)
        x_pilot_half = rng_pilot.standard_normal((N_PILOT // 2, mlp.width), dtype=np.float32)
        x_pilot = np.concatenate([x_pilot_half, -x_pilot_half], axis=0)
        # Whitening pilot
        gram_pilot = (x_pilot.T @ x_pilot) / float(N_PILOT)
        evals_p, evecs_p = np.linalg.eigh(gram_pilot)
        evals_p = np.maximum(evals_p, 1e-6)
        whitener_p = (evecs_p * np.power(evals_p, -0.5)) @ evecs_p.T
        x_pilot_w = x_pilot @ whitener_p
        H_pilot, G_pilot = forward_network(x_pilot_w, weights)
        pilot_gate_15 = np.mean(G_pilot[15], axis=0)

        # 3. Independent evaluation stream (N=4200) with antithetic sampling
        rng_eval = np.random.default_rng(mlp.seed + 20002)
        x_eval_half = rng_eval.standard_normal((N_EVAL // 2, mlp.width), dtype=np.float32)
        x_eval = np.concatenate([x_eval_half, -x_eval_half], axis=0)
        # Whitening eval independently
        gram_eval = (x_eval.T @ x_eval) / float(N_EVAL)
        evals_e, evecs_e = np.linalg.eigh(gram_eval)
        evals_e = np.maximum(evals_e, 1e-6)
        whitener_e = (evecs_e * np.power(evals_e, -0.5)) @ evecs_e.T
        x_eval_w = x_eval @ whitener_e
        H_eval, G_eval = forward_network(x_eval_w, weights)

        raw_eval_mean = np.mean(H_eval[-1], axis=0)
        H_14_eval_mean = np.mean(H_eval[14], axis=0)

        # Response mapping from layer 14 to layer 15:
        # v = (delta @ W_15) * p_15
        W_15 = weights[15]

        # Case A: Baseline Analytic Center
        delta_base = H_14_eval_mean - base_center_14
        v_base = (delta_base @ W_15) * pilot_gate_15
        m_corr_base = raw_eval_mean - v_base
        pred_base_cv = (1.0 - ALPHA_BLEND) * (S0 * base_stack[-1]) + ALPHA_BLEND * m_corr_base
        mse_base_cv = float(np.mean((pred_base_cv - y_true) ** 2))
        results_baseline_center.append(mse_base_cv)

        # Case B: Phase 6 K3-simple Center
        delta_p6 = H_14_eval_mean - p6_center_14
        v_p6 = (delta_p6 @ W_15) * pilot_gate_15
        m_corr_p6 = raw_eval_mean - v_p6
        pred_p6_cv = (1.0 - ALPHA_BLEND) * (p6_k3_preds[mlp_idx, -1]) + ALPHA_BLEND * m_corr_p6
        mse_p6_cv = float(np.mean((pred_p6_cv - y_true) ** 2))
        results_p6_center.append(mse_p6_cv)

        # Mixture weights between base center and p6 center:
        # center(w) = (1 - w) * base_center + w * p6_center
        for w_mix in [0.0, 0.5, 1.0]:
            center_mix = (1.0 - w_mix) * base_center_14 + w_mix * p6_center_14
            delta_mix = H_14_eval_mean - center_mix
            v_mix = (delta_mix @ W_15) * pilot_gate_15
            m_corr_mix = raw_eval_mean - v_mix
            analytic_mix = (1.0 - w_mix) * (S0 * base_stack[-1]) + w_mix * (p6_k3_preds[mlp_idx, -1])
            pred_mix = (1.0 - ALPHA_BLEND) * analytic_mix + ALPHA_BLEND * m_corr_mix
            mse_mix = float(np.mean((pred_mix - y_true) ** 2))
            mixture_results[w_mix].append(mse_mix)

        diff_pct = (mse_p6_cv - mse_base_cv) / mse_base_cv * 100.0
        print(f"[{mlp_idx+1}/8] {mlp_name:<20} | Base Center MSE: {mse_base_cv:.6e} | P6 Center MSE: {mse_p6_cv:.6e} ({diff_pct:+.2f}%)", flush=True)

    mean_base = float(np.mean(results_baseline_center))
    mean_p6 = float(np.mean(results_p6_center))
    mean_mix05 = float(np.mean(mixture_results[0.5]))
    wins_p6_vs_base = sum(1 for p, b in zip(results_p6_center, results_baseline_center) if p < b)

    print("\n=======================================================", flush=True)
    print("STAGE P6-06 CONTROL CHECK SUMMARY", flush=True)
    print(f"  Baseline Center Mapped CV Mean MSE: {mean_base:.6e}")
    print(f"  P6 K3-simple Center Mapped CV Mean: {mean_p6:.6e} ({(mean_p6 - mean_base)/mean_base * 100:+.2f}%)")
    print(f"  50/50 Mixture Center Mean MSE:      {mean_mix05:.6e} ({(mean_mix05 - mean_base)/mean_base * 100:+.2f}%)")
    print(f"  Head-to-Head (P6 Center vs Base):   {wins_p6_vs_base}W - {8-wins_p6_vs_base}L")
    print("=======================================================", flush=True)

    receipt = {
        "exp_id": "P6-06_control_check",
        "timestamp": datetime.datetime.now().isoformat(),
        "mean_base_center_mse": mean_base,
        "mean_p6_center_mse": mean_p6,
        "mean_mix05_mse": mean_mix05,
        "diff_pct": (mean_p6 - mean_base) / mean_base * 100.0,
        "wins_p6_vs_base": wins_p6_vs_base,
        "losses_p6_vs_base": 8 - wins_p6_vs_base,
        "per_mlp_base": results_baseline_center,
        "per_mlp_p6": results_p6_center,
    }

    out_file = RESULTS_DIR / "P6-06_control_check_receipt.json"
    with open(out_file, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"Saved P6-06 control check receipt to: {out_file}", flush=True)


if __name__ == "__main__":
    run_mapped_control_comparison()
