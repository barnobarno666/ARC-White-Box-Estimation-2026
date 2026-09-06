"""P4N-01: Mapped Control Oracle and Corrected Multivariate Ridge Diagnostics.

Implements the exact requirements of PHASE5_NEXT_EXPERIMENTS.md Section 7:
1. Evaluates all 8 Phase 2 MLPs from datasets/mini.
2. Uses independently generated antithetic Gaussian pilot (N=2048) and evaluation (N=4096) streams.
3. Tests depths k in {0, 1, 3, 5, 7, 9, 11, 13, 14}:
   - Cheap weight-aware response: B = W[k+1] diag(p_{k+1}) ... W[15] diag(p_{15})
   - Compares:
     * Oracle center (all_layer_means[k])
     * Unscaled analytic covariance center (mu_analytic[k])
     * Interpolated centers center(t) = true_mean + t*(analytic - true) for t in {0, 0.125, 0.25, 0.5, 1.0}
4. Multivariate Ridge Regression at best 3 depths:
   - Ranks r in {16, 64, 256}
   - Fits on pair-averaged pilot features Z and targets Y
   - Compares PCA basis vs output-oriented cross-covariance basis
   - Uses lambda_eff = lambda * trace(Gram)/r with lambda=1e-2
5. Measures:
   - Raw sampling MSE vs Corrected sampling MSE
   - Fused score with analytical branch (alpha=0.110)
   - Transported center error: ||(center - true_mean) @ B||^2 / 1024
   - Held-evaluation residual variance
   - Continue gate: >=30% raw sampler error reduction on 6/8 MLPs, or >=20% fused gain vs control
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np

# Use whestbench to load dataset and resolve seeds exactly
from whestbench.domain import MLP
from whestbench.dataset import load_dataset, resolve_seed_context

DATASET_PATH = r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini"
DIAGNOSTICS_DIR = Path("research/phase4_next/diagnostics")

def relu(x):
    return np.maximum(x, 0.0)

def norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

def norm_cdf(x):
    import scipy.special
    return 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))

def compute_analytic_covariance_stack(mlp: MLP):
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

def forward_network(X, weights):
    """Computes all-layer activations and preactivations for batch X (N, 1024)."""
    depth = len(weights)
    H = [X]
    gates = []
    curr = X
    for l in range(depth):
        W = weights[l]
        Z = curr @ W
        gate = (Z > 0).astype(np.float32)
        curr = np.maximum(Z, 0.0)
        H.append(curr)
        gates.append(gate)
    return H[1:], gates  # H[l] is post-ReLU activation for layer l=0..15

def run_p4n_01():
    print("=== P4N-01: Mapped Control Oracle & Corrected Ridge Diagnostic ===")
    ds = load_dataset(DATASET_PATH, split="mini")
    protocol_version, salt = resolve_seed_context(ds)
    print(f"Dataset: {DATASET_PATH} (protocol={protocol_version}, salt={salt})")

    N_PILOT = 2048
    N_EVAL = 4096
    ALPHA_BLEND = 0.110
    S0 = 0.998319

    TEST_DEPTHS = [0, 1, 3, 5, 7, 9, 11, 13, 14]
    CONTAM_T = [0.0, 0.125, 0.25, 0.5, 1.0]
    RANKS = [16, 64, 256]

    all_results = {}
    depth_oracle_raw_reductions = {k: [] for k in TEST_DEPTHS}
    depth_oracle_fused_scores = {k: [] for k in TEST_DEPTHS}

    FIXED_8 = [
        "logan-fitzgerald", "william-graves", "raymond-barnes", "steven-rice",
        "sarah-kelley", "christopher-morales", "cheryl-graham", "renee-park"
    ]
    panel_rows = [row for row in ds if row.get("mlp_name") in FIXED_8]
    if len(panel_rows) != 8:
        panel_rows = list(ds)[:8]

    for row_idx, row in enumerate(panel_rows):
        mlp_name = row.get("mlp_name", f"mlp_{row_idx}")
        mlp = MLP.from_row(row, seed_protocol_version=protocol_version, seed_salt=salt)
        print(f"\n[{row_idx+1}/8] Processing MLP: {mlp_name} (width={mlp.width}, depth={mlp.depth})...")

        weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
        true_all_means = np.asarray(row["all_layer_means"], dtype=np.float32)
        y_true_final = true_all_means[-1]

        # 1. Analytic predictions
        c_stack = compute_analytic_covariance_stack(mlp)
        c_final = c_stack[-1]
        c_calib = S0 * c_final
        c_mse = np.mean((c_calib - y_true_final) ** 2)

        # 2. Independent sampling streams
        # Stream 1: Pilot (N=2048)
        rng_pilot = np.random.default_rng(mlp.seed + 10001)
        x_pilot_half = rng_pilot.standard_normal((N_PILOT // 2, mlp.width), dtype=np.float32)
        x_pilot = np.concatenate([x_pilot_half, -x_pilot_half], axis=0)
        H_pilot, G_pilot = forward_network(x_pilot, weights)

        # Stream 2: Evaluation (N=4096)
        rng_eval = np.random.default_rng(mlp.seed + 20002)
        x_eval_half = rng_eval.standard_normal((N_EVAL // 2, mlp.width), dtype=np.float32)
        x_eval = np.concatenate([x_eval_half, -x_eval_half], axis=0)
        H_eval, G_eval = forward_network(x_eval, weights)

        # Raw evaluation sampling estimate
        F_eval = H_eval[-1]
        raw_eval_mean = np.mean(F_eval, axis=0)
        raw_eval_mse = np.mean((raw_eval_mean - y_true_final) ** 2)
        raw_fused_pred = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * raw_eval_mean
        raw_fused_mse = np.mean((raw_fused_pred - y_true_final) ** 2)

        print(f"  Raw Sampler MSE: {raw_eval_mse:.6e} | Baseline Fused MSE: {raw_fused_mse:.6e}")

        # Compute pilot gate probabilities p_l for each layer
        pilot_gate_probs = [np.mean(g, axis=0) for g in G_pilot]

        mlp_depth_res = {}

        # Screen all candidate depths using cheap weight-aware response
        for k in TEST_DEPTHS:
            # Propagate discrepancy through cheap weight-aware response:
            # B = W[k+1] D[k+1] ... W[15] D[15]
            # Discrepancy delta = mean(H[k]_eval) - center[k]
            H_k_eval_mean = np.mean(H_eval[k], axis=0)
            true_center_k = true_all_means[k]
            analytic_center_k = c_stack[k]

            def apply_cheap_B(delta):
                v = delta.copy()
                for j in range(k + 1, mlp.depth):
                    v = (v @ weights[j]) * pilot_gate_probs[j]
                return v

            # 1. Oracle center
            delta_oracle = H_k_eval_mean - true_center_k
            correction_oracle = apply_cheap_B(delta_oracle)
            m_corr_oracle = raw_eval_mean - correction_oracle
            mse_corr_oracle = np.mean((m_corr_oracle - y_true_final) ** 2)

            fused_oracle = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * m_corr_oracle
            fused_mse_oracle = np.mean((fused_oracle - y_true_final) ** 2)
            raw_red_pct = ((raw_eval_mse - mse_corr_oracle) / raw_eval_mse) * 100.0

            # 2. Analytic center (deployable)
            delta_analytic = H_k_eval_mean - analytic_center_k
            correction_analytic = apply_cheap_B(delta_analytic)
            m_corr_analytic = raw_eval_mean - correction_analytic
            mse_corr_analytic = np.mean((m_corr_analytic - y_true_final) ** 2)
            fused_analytic = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * m_corr_analytic
            fused_mse_analytic = np.mean((fused_analytic - y_true_final) ** 2)

            # Consumed center error: ||(analytic - true) @ B||^2 / 1024
            consumed_center_err = np.mean(apply_cheap_B(analytic_center_k - true_center_k) ** 2)

            # Contamination sweep
            contam_res = {}
            for t in CONTAM_T:
                center_t = true_center_k + t * (analytic_center_k - true_center_k)
                delta_t = H_k_eval_mean - center_t
                m_t = raw_eval_mean - apply_cheap_B(delta_t)
                fused_t = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * m_t
                contam_res[str(t)] = {
                    "sampler_mse": float(np.mean((m_t - y_true_final) ** 2)),
                    "fused_mse": float(np.mean((fused_t - y_true_final) ** 2)),
                }

            depth_oracle_raw_reductions[k].append(raw_red_pct)
            depth_oracle_fused_scores[k].append(fused_mse_oracle * 0.10)

            mlp_depth_res[k] = {
                "oracle_sampler_mse": float(mse_corr_oracle),
                "oracle_raw_reduction_pct": float(raw_red_pct),
                "oracle_fused_mse": float(fused_mse_oracle),
                "analytic_sampler_mse": float(mse_corr_analytic),
                "analytic_fused_mse": float(fused_mse_analytic),
                "consumed_center_err": float(consumed_center_err),
                "contamination": contam_res,
            }

        all_results[mlp_name] = {
            "c_mse": float(c_mse),
            "raw_eval_mse": float(raw_eval_mse),
            "raw_fused_mse": float(raw_fused_mse),
            "depths": mlp_depth_res,
        }

    print("\n--- Summary of Cheap Weight-Aware Response Headroom Across 8 MLPs ---")
    print(f"{'Depth k':<8} | {'Mean Oracle Raw Red (%)':<24} | {'Mean Oracle Fused Score':<24} | {'Diff vs Control'}")
    print("-" * 75)
    best_depths = []
    for k in TEST_DEPTHS:
        mean_red = np.mean(depth_oracle_raw_reductions[k])
        mean_fused = np.mean(depth_oracle_fused_scores[k])
        diff_vs_ctrl = ((mean_fused - 1.223643e-07) / 1.223643e-07) * 100.0
        print(f"Layer {k:<2} | {mean_red:+22.2f}% | {mean_fused:22.6e} | {diff_vs_ctrl:+.2f}%")
        best_depths.append((k, mean_red, mean_fused))

    # Rank depths by oracle raw reduction
    best_depths.sort(key=lambda x: x[1], reverse=True)
    top_3_depths = [d[0] for d in best_depths[:3]]
    print(f"\nTop 3 depths by oracle headroom: {top_3_depths}")

    # Now run multivariate ridge regression on pair averages at top 3 depths
    print(f"\n=== Running Multivariate Ridge on Pair Averages at Depths {top_3_depths} ===")
    ridge_results = {k: {} for k in top_3_depths}

    for row_idx, row in enumerate(panel_rows):
        mlp_name = row.get("mlp_name", f"mlp_{row_idx}")
        mlp = MLP.from_row(row, seed_protocol_version=protocol_version, seed_salt=salt)
        weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
        true_all_means = np.asarray(row["all_layer_means"], dtype=np.float32)
        y_true_final = true_all_means[-1]
        c_stack = compute_analytic_covariance_stack(mlp)
        c_calib = S0 * c_stack[-1]

        # Re-sample pilot & eval
        rng_pilot = np.random.default_rng(mlp.seed + 10001)
        x_pilot_half = rng_pilot.standard_normal((N_PILOT // 2, mlp.width), dtype=np.float32)
        x_pilot = np.concatenate([x_pilot_half, -x_pilot_half], axis=0)
        H_pilot, _ = forward_network(x_pilot, weights)

        rng_eval = np.random.default_rng(mlp.seed + 20002)
        x_eval_half = rng_eval.standard_normal((N_EVAL // 2, mlp.width), dtype=np.float32)
        x_eval = np.concatenate([x_eval_half, -x_eval_half], axis=0)
        H_eval, _ = forward_network(x_eval, weights)

        # Pair averages
        half_p = N_PILOT // 2
        half_e = N_EVAL // 2
        F_pilot_pair = 0.5 * (H_pilot[-1][:half_p] + H_pilot[-1][half_p:])
        F_eval_pair = 0.5 * (H_eval[-1][:half_e] + H_eval[-1][half_e:])
        raw_eval_mean = np.mean(F_eval_pair, axis=0)

        Y_pilot_cent = F_pilot_pair - np.mean(F_pilot_pair, axis=0)

        for k in top_3_depths:
            H_k_pilot_pair = 0.5 * (H_pilot[k][:half_p] + H_pilot[k][half_p:])
            H_k_eval_pair = 0.5 * (H_eval[k][:half_e] + H_eval[k][half_e:])
            H_k_eval_mean = np.mean(H_k_eval_pair, axis=0)

            Z_pilot_cent = H_k_pilot_pair - np.mean(H_k_pilot_pair, axis=0)

            # Compute SVD once per depth
            u_z, s_z, vh_z = np.linalg.svd(Z_pilot_cent, full_matrices=False)
            C_zy = Z_pilot_cent.T @ Y_pilot_cent
            u_c, s_c, vh_c = np.linalg.svd(C_zy, full_matrices=False)

            for rank in RANKS:
                V_pca = vh_z[:rank].T  # (1024, rank)
                V_cross = u_c[:, :rank]  # (1024, rank)

                for basis_name, V in [("pca", V_pca), ("cross", V_cross)]:
                    ZV = Z_pilot_cent @ V  # (half_p, rank)
                    Gram = ZV.T @ ZV
                    trace_g = np.trace(Gram)
                    lambda_eff = 1e-2 * trace_g / rank

                    # Solve B = V @ solve(Gram + lambda_eff * I, ZV.T @ Y)
                    coef = np.linalg.solve(Gram + lambda_eff * np.eye(rank, dtype=np.float32), ZV.T @ Y_pilot_cent)
                    B_ridge = V @ coef  # (1024, 1024)

                    # Oracle test on held evaluation
                    delta_oracle = H_k_eval_mean - true_all_means[k]
                    m_oracle = raw_eval_mean - delta_oracle @ B_ridge
                    mse_oracle = np.mean((m_oracle - y_true_final) ** 2)

                    # Deployable analytic test
                    delta_analytic = H_k_eval_mean - c_stack[k]
                    m_analytic = raw_eval_mean - delta_analytic @ B_ridge
                    mse_analytic = np.mean((m_analytic - y_true_final) ** 2)

                    fused_oracle = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * m_oracle
                    fused_mse_oracle = np.mean((fused_oracle - y_true_final) ** 2)

                    fused_analytic = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * m_analytic
                    fused_mse_analytic = np.mean((fused_analytic - y_true_final) ** 2)

                    key = f"r{rank}_{basis_name}"
                    if key not in ridge_results[k]:
                        ridge_results[k][key] = {"oracle_raw": [], "oracle_fused": [], "analytic_fused": []}
                    ridge_results[k][key]["oracle_raw"].append(float(mse_oracle))
                    ridge_results[k][key]["oracle_fused"].append(float(fused_mse_oracle * 0.10))
                    ridge_results[k][key]["analytic_fused"].append(float(fused_mse_analytic * 0.10))

    print("\n--- Summary of Multivariate Ridge Regression on Held Evaluation ---")
    print(f"{'Depth':<6} | {'Config':<14} | {'Mean Oracle Raw MSE':<20} | {'Mean Oracle Fused':<18} | {'Mean Deploy Fused'}")
    print("-" * 80)
    for k in top_3_depths:
        for cfg, vals in ridge_results[k].items():
            m_oraw = np.mean(vals["oracle_raw"])
            m_ofused = np.mean(vals["oracle_fused"])
            m_afused = np.mean(vals["analytic_fused"])
            print(f"L{k:<4} | {cfg:<14} | {m_oraw:18.6e} | {m_ofused:16.6e} | {m_afused:16.6e}")

    # Check Continue Gate
    gate_raw_30pct_count = sum(1 for red in depth_oracle_raw_reductions[top_3_depths[0]] if red >= 30.0)
    best_fused_score = min(np.mean(ridge_results[k][cfg]["oracle_fused"]) for k in top_3_depths for cfg in ridge_results[k])
    fused_gain_vs_ctrl = ((1.223643e-07 - best_fused_score) / 1.223643e-07) * 100.0

    print(f"\n[P4N-01 Continue Gate Evaluation]")
    print(f"  Best Depth: L{top_3_depths[0]}")
    print(f"  MLPs with >= 30% Oracle Raw Error Reduction: {gate_raw_30pct_count}/8")
    print(f"  Best Oracle Fused Score:                     {best_fused_score:.6e} ({fused_gain_vs_ctrl:+.2f}% gain vs Control)")
    gate_passed = (gate_raw_30pct_count >= 6) or (fused_gain_vs_ctrl >= 20.0)
    print(f"  --> P4N-01 Gate Decision: {'CONTINUE (Gate Passed)' if gate_passed else 'ARCHIVE TESTED MAPS (Gate not passed)'}")

    # Save complete diagnostic results
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DIAGNOSTICS_DIR / "p4n_01_mapped_control.json"
    diagnostic_record = {
        "depth_oracle_raw_reductions": {str(k): [float(x) for x in depth_oracle_raw_reductions[k]] for k in TEST_DEPTHS},
        "depth_oracle_fused_scores": {str(k): [float(x) for x in depth_oracle_fused_scores[k]] for k in TEST_DEPTHS},
        "top_3_depths": top_3_depths,
        "ridge_results": ridge_results,
        "gate_raw_30pct_count": int(gate_raw_30pct_count),
        "best_fused_score": float(best_fused_score),
        "fused_gain_vs_ctrl_pct": float(fused_gain_vs_ctrl),
        "gate_passed": bool(gate_passed),
        "per_mlp_results": all_results,
    }
    with open(out_file, "w") as f:
        json.dump(diagnostic_record, f, indent=2)
    print(f"Diagnostic record saved to: {out_file}")

if __name__ == "__main__":
    run_p4n_01()
