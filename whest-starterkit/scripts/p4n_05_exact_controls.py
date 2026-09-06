"""P4N-05: Controls with an Exact, Lawful Mean.

Implements the exact requirements of PHASE5_NEXT_EXPERIMENTS.md Section 11:
1. Route around the approximate-center bottleneck using controls with EXACT Gaussian expectations.
2. Part A: First-layer nonlinear feature control
   - q(X) = relu(X @ W0)
   - Exact expectation: m0[j] = ||W0[:, j]|| / sqrt(2*pi)
   - Evaluated under paired-even features with antithetic Gaussian carrier.
   - Independent pilot (N=2048, 4096) to fit ridge response B:
     mu_hat = mean(F_eval) - (mean(q_eval) - m0) @ B
   - Tests ranks r in {64, 256, 512, 1024}.
3. Part B: Shifted even ridge-feature surrogate
   - q_{v, t}(x) = 0.5 * [relu(v^T x - t) + relu(-v^T x - t)]
   - Exact expectation: E[q_{v, t}] = phi(t) - t*Phi(-t)
   - Directions: normalized first-weight columns v_j = W0[:, j] / ||W0[:, j]||
   - Threshold sets: {0, 1}, {0, 0.5, 1.5} with 64 and 128 directions.
   - Fits surrogate g(x) = b + q(x) @ B on independent pilot.
   - Evaluates:
     mu_hat = b + E[q] @ B + mean(F_eval - g(X_eval))
4. Measures:
   - Residual variance reduction: Var(F - g) / Var(F)
   - Center error: exactly 0.0 by construction!
   - Standalone MSE & fused MSE with analytic branch (alpha=0.110).
   - Gate: residual variance reduction must pay for feature evaluation and fit (projected scored gain >= 20%).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
import scipy.special

from whestbench.domain import MLP
from whestbench.dataset import load_dataset, resolve_seed_context

DATASET_PATH = r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini"
DIAGNOSTICS_DIR = Path("research/phase4_next/diagnostics")

def norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

def norm_cdf(x):
    return 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))

def forward_network(X, weights):
    curr = X
    for W in weights:
        curr = np.maximum(curr @ W, 0.0)
    return curr

def run_p4n_05():
    print("=== P4N-05: Exact Lawful Mean Controls Diagnostic ===")
    ds = load_dataset(DATASET_PATH, split="mini")
    protocol_version, salt = resolve_seed_context(ds)

    N_PILOT = 2048
    N_EVAL = 4096
    ALPHA_BLEND = 0.110
    S0 = 0.998319

    RANKS_A = [64, 256, 512, 1024]
    THRESH_SETS = {
        "t0_1_d64": {"thresh": [0.0, 1.0], "n_dir": 64},
        "t0_1_d128": {"thresh": [0.0, 1.0], "n_dir": 128},
        "t0_05_15_d128": {"thresh": [0.0, 0.5, 1.5], "n_dir": 128},
    }

    results_A = {r: {"raw_mse": [], "fused_mse": [], "var_red": []} for r in RANKS_A}
    results_B = {cfg: {"raw_mse": [], "fused_mse": [], "var_red": []} for cfg in THRESH_SETS}
    baseline_raw_mse_list = []
    baseline_fused_mse_list = []

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
        print(f"\n[{row_idx+1}/8] Processing MLP: {mlp_name}...")

        weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
        true_all_means = np.asarray(row["all_layer_means"], dtype=np.float32)
        y_true_final = true_all_means[-1]

        # Exact m0 for Layer 0
        W0 = weights[0]  # (1024, 1024)
        W0_col_norms = np.linalg.norm(W0, axis=0)  # (1024,)
        m0_exact = (W0_col_norms / np.sqrt(2.0 * np.pi)).astype(np.float32)

        # Baseline analytic prediction
        from estimator import _blended_hermite_covariance
        # Using numpy reproduction of analytic covariance
        from p4n_01_mapped_control import compute_analytic_covariance_stack
        c_stack = compute_analytic_covariance_stack(mlp)
        c_calib = S0 * c_stack[-1]

        # Independent Pilot stream (N=2048)
        rng_p = np.random.default_rng(mlp.seed + 50001)
        xp_half = rng_p.standard_normal((N_PILOT // 2, mlp.width), dtype=np.float32)
        xp = np.concatenate([xp_half, -xp_half], axis=0)
        F_pilot = forward_network(xp, weights)

        # Independent Evaluation stream (N=4096)
        rng_e = np.random.default_rng(mlp.seed + 60002)
        xe_half = rng_e.standard_normal((N_EVAL // 2, mlp.width), dtype=np.float32)
        xe = np.concatenate([xe_half, -xe_half], axis=0)
        F_eval = forward_network(xe, weights)

        # Pair averages for evaluation
        half_p = N_PILOT // 2
        half_e = N_EVAL // 2
        Fp_pairs = 0.5 * (F_pilot[:half_p] + F_pilot[half_p:])
        Fe_pairs = 0.5 * (F_eval[:half_e] + F_eval[half_e:])

        raw_eval_mean = np.mean(Fe_pairs, axis=0)
        raw_eval_mse = np.mean((raw_eval_mean - y_true_final) ** 2)
        raw_fused = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * raw_eval_mean
        raw_fused_mse = np.mean((raw_fused - y_true_final) ** 2)

        baseline_raw_mse_list.append(raw_eval_mse)
        baseline_fused_mse_list.append(raw_fused_mse)

        var_raw = np.mean(np.var(Fe_pairs, axis=0))

        # -------------------------------------------------------------
        # Part A: First-layer nonlinear feature control q(X) = relu(X @ W0)
        # -------------------------------------------------------------
        Qp = np.maximum(xp @ W0, 0.0)
        Qe = np.maximum(xe @ W0, 0.0)
        Qp_pairs = 0.5 * (Qp[:half_p] + Qp[half_p:])
        Qe_pairs = 0.5 * (Qe[:half_e] + Qe[half_e:])

        Qp_cent = Qp_pairs - np.mean(Qp_pairs, axis=0)
        Yp_cent = Fp_pairs - np.mean(Fp_pairs, axis=0)

        for rank in RANKS_A:
            if rank < mlp.width:
                u_q, s_q, vh_q = np.linalg.svd(Qp_cent, full_matrices=False)
                V = vh_q[:rank].T
                QV = Qp_cent @ V
                Gram = QV.T @ QV
                lambda_eff = 1e-2 * np.trace(Gram) / rank
                coef = np.linalg.solve(Gram + lambda_eff * np.eye(rank, dtype=np.float32), QV.T @ Yp_cent)
                B = V @ coef
            else:
                # Full rank
                Gram = Qp_cent.T @ Qp_cent
                lambda_eff = 1e-2 * np.trace(Gram) / rank
                B = np.linalg.solve(Gram + lambda_eff * np.eye(rank, dtype=np.float32), Qp_cent.T @ Yp_cent)

            # Evaluate control on held evaluation batch
            # Expected mean of Qe is m0_exact!
            q_discrepancy = np.mean(Qe_pairs, axis=0) - m0_exact
            correction = q_discrepancy @ B
            m_hat = raw_eval_mean - correction

            mse_A = np.mean((m_hat - y_true_final) ** 2)
            fused_A = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * m_hat
            fused_mse_A = np.mean((fused_A - y_true_final) ** 2)

            # Residual variance on held pairs
            Fe_corrected_pairs = Fe_pairs - (Qe_pairs - m0_exact) @ B
            var_A = np.mean(np.var(Fe_corrected_pairs, axis=0))
            var_red = ((var_raw - var_A) / var_raw) * 100.0

            results_A[rank]["raw_mse"].append(mse_A)
            results_A[rank]["fused_mse"].append(fused_mse_A * 0.10)
            results_A[rank]["var_red"].append(var_red)

        # -------------------------------------------------------------
        # Part B: Shifted even ridge-feature surrogate
        # -------------------------------------------------------------
        # Normalized directions v_j from W0 columns
        V_dir_all = W0 / np.maximum(W0_col_norms[None, :], 1e-12)

        for cfg_name, cfg in THRESH_SETS.items():
            thresholds = cfg["thresh"]
            n_dir = cfg["n_dir"]
            V_sub = V_dir_all[:, :n_dir]  # (1024, n_dir)

            # Build feature matrices for pilot and eval
            def build_even_features(X_batch):
                # S = X @ V_sub (N, n_dir)
                S = X_batch @ V_sub
                feats = []
                means = []
                for t in thresholds:
                    f_pos = np.maximum(S - t, 0.0)
                    f_neg = np.maximum(-S - t, 0.0)
                    q = 0.5 * (f_pos + f_neg)
                    feats.append(q)
                    # E[q] = phi(t) - t*Phi(-t)
                    eq = norm_pdf(t) - t * norm_cdf(-t)
                    means.append(np.full(n_dir, eq, dtype=np.float32))
                return np.concatenate(feats, axis=1), np.concatenate(means, axis=0)

            Qp_feat, Eq_feat = build_even_features(xp)
            Qe_feat, _ = build_even_features(xe)

            Qp_f_pairs = 0.5 * (Qp_feat[:half_p] + Qp_feat[half_p:])
            Qe_f_pairs = 0.5 * (Qe_feat[:half_e] + Qe_feat[half_e:])

            # Fit ridge response
            Qp_fc = Qp_f_pairs - np.mean(Qp_f_pairs, axis=0)
            dim_feat = Qp_fc.shape[1]
            Gram = Qp_fc.T @ Qp_fc
            lambda_eff = 1e-2 * np.trace(Gram) / dim_feat
            B_even = np.linalg.solve(Gram + lambda_eff * np.eye(dim_feat, dtype=np.float32), Qp_fc.T @ Yp_cent)
            b_offset = np.mean(Fp_pairs, axis=0) - np.mean(Qp_f_pairs, axis=0) @ B_even

            # Deployable surrogate estimate:
            # mu_hat = b_offset + Eq_feat @ B_even + mean(Fe_pairs - (b_offset + Qe_f_pairs @ B_even))
            # Notice this algebraically equals: raw_eval_mean - (mean(Qe_f_pairs) - Eq_feat) @ B_even !
            q_even_discrepancy = np.mean(Qe_f_pairs, axis=0) - Eq_feat
            m_hat_B = raw_eval_mean - q_even_discrepancy @ B_even

            mse_B = np.mean((m_hat_B - y_true_final) ** 2)
            fused_B = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * m_hat_B
            fused_mse_B = np.mean((fused_B - y_true_final) ** 2)

            Fe_corr_even = Fe_pairs - (Qe_f_pairs - Eq_feat) @ B_even
            var_B = np.mean(np.var(Fe_corr_even, axis=0))
            var_red_B = ((var_raw - var_B) / var_raw) * 100.0

            results_B[cfg_name]["raw_mse"].append(mse_B)
            results_B[cfg_name]["fused_mse"].append(fused_mse_B * 0.10)
            results_B[cfg_name]["var_red"].append(var_red_B)

    print("\n--- Summary: Part A (Layer 0 Nonlinear Feature Control) ---")
    print(f"{'Rank':<8} | {'Mean Raw MSE':<18} | {'Mean Fused Score':<18} | {'Var Reduction (%)'}")
    print("-" * 65)
    for r in RANKS_A:
        m_raw = np.mean(results_A[r]["raw_mse"])
        m_fused = np.mean(results_A[r]["fused_mse"])
        m_vred = np.mean(results_A[r]["var_red"])
        print(f"Rank {r:<4} | {m_raw:16.6e} | {m_fused:16.6e} | {m_vred:+15.2f}%")

    print("\n--- Summary: Part B (Shifted Even Ridge-Feature Surrogate) ---")
    print(f"{'Config':<18} | {'Mean Raw MSE':<18} | {'Mean Fused Score':<18} | {'Var Reduction (%)'}")
    print("-" * 75)
    for cfg_name in THRESH_SETS:
        m_raw = np.mean(results_B[cfg_name]["raw_mse"])
        m_fused = np.mean(results_B[cfg_name]["fused_mse"])
        m_vred = np.mean(results_B[cfg_name]["var_red"])
        print(f"{cfg_name:<18} | {m_raw:16.6e} | {m_fused:16.6e} | {m_vred:+15.2f}%")

    base_raw = np.mean(baseline_raw_mse_list)
    base_fused = np.mean(baseline_fused_mse_list) * 0.10
    print(f"\nBaseline Raw Sampler MSE: {base_raw:.6e} | Baseline Fused Score: {base_fused:.6e}")

    # Gate verification:
    best_fused_A = min(np.mean(results_A[r]["fused_mse"]) for r in RANKS_A)
    best_fused_B = min(np.mean(results_B[cfg]["fused_mse"]) for cfg in THRESH_SETS)
    best_fused_overall = min(best_fused_A, best_fused_B)
    diff_vs_control = ((best_fused_overall - 1.223643e-07) / 1.223643e-07) * 100.0

    print(f"\n[P4N-05 Gate Check]")
    print(f"  Best Part A Fused Score: {best_fused_A:.6e}")
    print(f"  Best Part B Fused Score: {best_fused_B:.6e}")
    print(f"  Best Overall Score:      {best_fused_overall:.6e} ({diff_vs_control:+.2f}% vs Control)")
    gate_passed = (diff_vs_control <= -20.0)
    print(f"  --> P4N-05 Gate Decision: {'CONTINUE (Gate Passed >=20%)' if gate_passed else 'ARCHIVE (Gate not reached)'}")

    # Save diagnostic record
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DIAGNOSTICS_DIR / "p4n_05_exact_controls.json"
    rec = {
        "results_A": {str(r): {k: [float(x) for x in v] for k, v in res.items()} for r, res in results_A.items()},
        "results_B": {cfg: {k: [float(x) for x in v] for k, v in res.items()} for cfg, res in results_B.items()},
        "baseline_raw_mse": float(base_raw),
        "baseline_fused_score": float(base_fused),
        "best_fused_overall": float(best_fused_overall),
        "diff_vs_control_pct": float(diff_vs_control),
        "gate_passed": bool(gate_passed),
    }
    with open(out_file, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Diagnostic record saved to: {out_file}")

if __name__ == "__main__":
    run_p4n_05()
