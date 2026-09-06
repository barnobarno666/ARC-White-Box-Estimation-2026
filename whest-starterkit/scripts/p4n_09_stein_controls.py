"""P4N-09: Nontrivial Even Stein Controls.

Implements the requirements of PHASE5_NEXT_EXPERIMENTS.md Section 15:
1. Stein identity:
   For X ~ N(0, I), unit v, s = v^T X:
   q(X) = s * psi(s) - psi'(s) has exact expectation zero under Gaussian measure.
   With psi(s) = relu(s - t) and psi'(s) = 1_{s > t}:
   q_t(X) = s * relu(s - t) - 1_{s > t}.
2. Antipodal pair averaging:
   q_bar_t(X) = 0.5 * [q_t(X) + q_t(-X)].
   For |s| > t: q_bar_t = 0.5 * [s^2 - t*|s| - 1].
   For |s| <= t: q_bar_t = 0.
   At t=0: q_bar_0 = 0.5 * [s^2 - 1] (quadratic negative control, annihilated by covariance matching).
3. Direction banks:
   - Bank 1: Normalized columns of W0 (v_j = W0[:, j] / ||W0[:, j]||).
   - Bank 2: Top left singular vectors of end-to-end response matrix W0 D0 ... W15 D15.
4. Threshold sets:
   - Negative control: {0.0}
   - Set 1: {0.5, 1.0}
   - Set 2: {1.0, 2.0}
   - Set 3: {0.5, 1.0, 2.0}
   Directions: 32 and 64 directions.
5. Pilot & Evaluation Protocol:
   - Independent pilot (N=2048) fits low-rank ridge response B.
   - Independent evaluation (N=4096) applies control subtracting exact mean 0.0:
     mu_hat = mean(F_eval) - mean(q_bar_eval) @ B.
   - Tests both Ordinary Gaussian sampling and Whitened Gaussian sampling.
6. Gate:
   - At least 20% residual variance reduction on 8 MLPs, or >= 10% scored gain (< 1.1013e-07).
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
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def forward_network(X: np.ndarray, weights: list[np.ndarray]) -> np.ndarray:
    curr = X
    for W in weights:
        curr = np.maximum(curr @ W, 0.0)
    return curr


def compute_stein_feature(s: np.ndarray, t: float) -> np.ndarray:
    """Computes pair-averaged Stein feature q_bar_t(s) = 0.5 * [q_t(s) + q_t(-s)].
    
    s has shape (N, D).
    """
    abs_s = np.abs(s)
    mask = abs_s > t
    res = np.zeros_like(s)
    res[mask] = 0.5 * (s[mask] ** 2 - t * abs_s[mask] - 1.0)
    return res


def verify_stein_math():
    """Verifies that E[q_bar_t(s)] == 0.0 to high numerical accuracy."""
    from scipy.integrate import quad

    def integrand(s, t):
        abs_s = abs(s)
        if abs_s <= t:
            val = 0.0
        else:
            val = 0.5 * (s ** 2 - t * abs_s - 1.0)
        phi = np.exp(-0.5 * s * s) / np.sqrt(2.0 * np.pi)
        return val * phi

    print("Verifying Stein mathematical identity via scipy quad...")
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        if t == 0.0:
            val, _ = quad(integrand, -10.0, 10.0, args=(t,), points=[0.0])
        else:
            val, _ = quad(integrand, -10.0, 10.0, args=(t,), points=[-t, t])
        assert abs(val) < 1e-8, f"Failed Stein expectation at t={t}: {val}"
    print("Stein expectation verified to < 1e-8 precision across all test thresholds!")


def get_gate_aware_response_modes(mlp: MLP, weights: list[np.ndarray], n_modes: int = 64) -> np.ndarray:
    """Compute left singular vectors of gate-aware response matrix."""
    # Approximate gate probabilities p_l = 0.5 for ReLU
    M = weights[0] * 0.5
    for l in range(1, len(weights)):
        M = M @ (weights[l] * 0.5)
    # SVD of M: M = U S V^T. Left singular vectors are columns of U
    u, s, vh = np.linalg.svd(M, full_matrices=False)
    return u[:, :n_modes]


def run_p4n_09():
    print("=== P4N-09: Nontrivial Even Stein Controls Diagnostic ===")
    verify_stein_math()

    ds = load_dataset(DATASET_PATH, split="mini")
    protocol_version, salt = resolve_seed_context(ds)

    N_PILOT = 2048
    N_EVAL = 4096
    ALPHA_BLEND = 0.110
    S0 = 0.998319

    FIXED_8 = [
        "logan-fitzgerald", "william-graves", "raymond-barnes", "steven-rice",
        "sarah-kelley", "christopher-morales", "cheryl-graham", "renee-park"
    ]
    panel_rows = [row for row in ds if row.get("mlp_name") in FIXED_8]
    if len(panel_rows) != 8:
        panel_rows = list(ds)[:8]

    # Configurations to test:
    # 2 direction banks: "w0_cols", "response_modes"
    # Directions count: 32, 64
    # Threshold sets:
    # "t0" (neg ctrl): [0.0]
    # "t_05_1": [0.5, 1.0]
    # "t_1_2": [1.0, 2.0]
    # "t_05_1_2": [0.5, 1.0, 2.0]
    configs = []
    for bank in ["w0_cols", "response_modes"]:
        for n_dir in [32, 64]:
            for t_name, t_vals in [
                ("t0_negctrl", [0.0]),
                ("t_05_1", [0.5, 1.0]),
                ("t_1_2", [1.0, 2.0]),
                ("t_05_1_2", [0.5, 1.0, 2.0]),
            ]:
                configs.append({
                    "name": f"{bank}_d{n_dir}_{t_name}",
                    "bank": bank,
                    "n_dir": n_dir,
                    "thresholds": t_vals
                })

    # Summary metrics accumulator across 8 MLPs
    results_summary = {
        cfg["name"]: {
            "var_red_pct": [],
            "raw_mse": [],
            "fused_mse": [],
            "wins_vs_ctrl": 0
        }
        for cfg in configs
    }

    baseline_ctrl_scores = []
    baseline_raw_mse_list = []

    print(f"Total configurations per MLP: {len(configs)}")

    from p4n_01_mapped_control import compute_analytic_covariance_stack

    for row_idx, row in enumerate(panel_rows):
        mlp_name = row.get("mlp_name", f"mlp_{row_idx}")
        mlp = MLP.from_row(row, seed_protocol_version=protocol_version, seed_salt=salt)
        print(f"\n[{row_idx+1}/8] MLP: {mlp_name}...")

        weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
        true_all_means = np.asarray(row["all_layer_means"], dtype=np.float32)
        y_true_final = true_all_means[-1]

        # Baseline analytic prediction
        c_stack = compute_analytic_covariance_stack(mlp)
        c_calib = S0 * c_stack[-1]

        # Draw independent Pilot stream (N=2048) with antipodal pairing
        rng_p = np.random.default_rng(mlp.seed + 50001)
        xp_half = rng_p.standard_normal((N_PILOT // 2, mlp.width), dtype=np.float32)
        xp = np.concatenate([xp_half, -xp_half], axis=0)
        F_pilot = forward_network(xp, weights)

        # Draw independent Evaluation stream (N=4096) with antipodal pairing
        rng_e = np.random.default_rng(mlp.seed + 60002)
        xe_half = rng_e.standard_normal((N_EVAL // 2, mlp.width), dtype=np.float32)
        xe = np.concatenate([xe_half, -xe_half], axis=0)
        F_eval = forward_network(xe, weights)

        half_p = N_PILOT // 2
        half_e = N_EVAL // 2
        Fp_pairs = 0.5 * (F_pilot[:half_p] + F_pilot[half_p:])
        Fe_pairs = 0.5 * (F_eval[:half_e] + F_eval[half_e:])

        raw_eval_mean = np.mean(Fe_pairs, axis=0)
        raw_eval_mse = np.mean((raw_eval_mean - y_true_final) ** 2)
        raw_fused = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * raw_eval_mean
        raw_fused_mse = np.mean((raw_fused - y_true_final) ** 2)

        # Frozen control reference score
        ctrl_score = raw_fused_mse * 0.10
        baseline_ctrl_scores.append(ctrl_score)
        baseline_raw_mse_list.append(raw_eval_mse)

        var_raw = np.mean(np.var(Fe_pairs, axis=0))

        # Direction banks
        W0 = weights[0]
        W0_col_norms = np.linalg.norm(W0, axis=0)
        V_w0 = W0 / np.maximum(W0_col_norms[None, :], 1e-12)
        V_modes = get_gate_aware_response_modes(mlp, weights, n_modes=64)

        # Evaluate each configuration
        for cfg in configs:
            cfg_name = cfg["name"]
            bank = cfg["bank"]
            n_dir = cfg["n_dir"]
            thresholds = cfg["thresholds"]

            if bank == "w0_cols":
                V_sub = V_w0[:, :n_dir]
            else:
                V_sub = V_modes[:, :n_dir]

            # Compute inner projections s = X @ V for pilot and eval
            # Under antipodal pairing, x_pair = xp_half
            Sp = xp_half @ V_sub  # (half_p, n_dir)
            Se = xe_half @ V_sub  # (half_e, n_dir)

            # Build feature matrices
            Qp_list = []
            Qe_list = []
            for t in thresholds:
                Qp_list.append(compute_stein_feature(Sp, t))
                Qe_list.append(compute_stein_feature(Se, t))

            Qp = np.concatenate(Qp_list, axis=-1)  # (half_p, n_dir * len(thresholds))
            Qe = np.concatenate(Qe_list, axis=-1)  # (half_e, n_dir * len(thresholds))

            # Feature variance check
            feat_var = np.mean(np.var(Qp, axis=0))
            if feat_var < 1e-12:
                # Degenerate
                results_summary[cfg_name]["var_red_pct"].append(0.0)
                results_summary[cfg_name]["raw_mse"].append(raw_eval_mse)
                results_summary[cfg_name]["fused_mse"].append(ctrl_score)
                continue

            # Fit low-rank ridge response B on pilot
            D_feat = Qp.shape[1]
            rank = min(D_feat, 64)

            # SVD of Qp
            u_q, s_q, vh_q = np.linalg.svd(Qp, full_matrices=False)
            V_basis = vh_q[:rank].T
            QV = Qp @ V_basis
            Gram = QV.T @ QV
            lambda_eff = 1e-2 * np.trace(Gram) / rank
            rhs = QV.T @ (Fp_pairs - np.mean(Fp_pairs, axis=0))
            coef = np.linalg.solve(Gram + lambda_eff * np.eye(rank, dtype=np.float32), rhs)
            B = V_basis @ coef

            # Evaluation: subtract exact mean (0.0)!
            # mu_hat = raw_eval_mean - mean(Qe) @ B
            q_eval_mean = np.mean(Qe, axis=0)  # exact mean is 0.0
            correction = q_eval_mean @ B
            m_hat = raw_eval_mean - correction

            mse_cand = np.mean((m_hat - y_true_final) ** 2)
            fused_cand = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * m_hat
            fused_mse_cand = np.mean((fused_cand - y_true_final) ** 2)
            score_cand = fused_mse_cand * 0.10

            # Residual variance on held pairs
            Fe_corrected = Fe_pairs - Qe @ B
            var_corrected = np.mean(np.var(Fe_corrected, axis=0))
            var_red = ((var_raw - var_corrected) / var_raw) * 100.0

            results_summary[cfg_name]["var_red_pct"].append(var_red)
            results_summary[cfg_name]["raw_mse"].append(mse_cand)
            results_summary[cfg_name]["fused_mse"].append(score_cand)
            if score_cand < ctrl_score:
                results_summary[cfg_name]["wins_vs_ctrl"] += 1

    # Print summary table
    print("\n" + "=" * 95)
    print(f"{'Configuration':<28} | {'Var Red %':<12} | {'Raw MSE':<12} | {'Fused Score':<14} | {'Diff vs Ctrl':<14} | {'Wins':<6}")
    print("=" * 95)

    mean_ctrl_score = np.mean(baseline_ctrl_scores)
    best_cfg = None
    best_score = float("inf")

    summary_export = {}

    for cfg_name, stats in results_summary.items():
        mean_var_red = float(np.mean(stats["var_red_pct"]))
        mean_raw_mse = float(np.mean(stats["raw_mse"]))
        mean_fused = float(np.mean(stats["fused_mse"]))
        diff_pct = ((mean_fused - mean_ctrl_score) / mean_ctrl_score) * 100.0
        wins = stats["wins_vs_ctrl"]

        if mean_fused < best_score:
            best_score = mean_fused
            best_cfg = cfg_name

        print(f"{cfg_name:<28} | {mean_var_red:>10.2f}% | {mean_raw_mse:>11.4e} | {mean_fused:>12.6e} | {diff_pct:>+12.2f}% | {wins}/8")

        summary_export[cfg_name] = {
            "mean_var_red_pct": mean_var_red,
            "mean_raw_mse": mean_raw_mse,
            "mean_fused_score": mean_fused,
            "diff_vs_ctrl_pct": diff_pct,
            "wins_vs_ctrl": wins
        }

    print("=" * 95)
    print(f"Baseline Control Adjusted Score: {mean_ctrl_score:.6e}")
    print(f"Best Configuration: {best_cfg} -> Score: {best_score:.6e} ({((best_score - mean_ctrl_score)/mean_ctrl_score)*100:+.2f}%)")

    # Gate check: at least 20% cost-adjusted residual variance reduction or >= 10% scored gain (< 1.1013e-07)
    best_stats = summary_export[best_cfg]
    gate_passed = (best_stats["mean_var_red_pct"] >= 20.0) or (best_stats["mean_fused_score"] < 1.101278e-07)
    print(f"\nGATE STATUS: {'PASSED' if gate_passed else 'FAILED (Reject/Archive)'}")
    print(f"Var Red Gate (>=20%): {best_stats['mean_var_red_pct']:.2f}%")
    print(f"Score Gate (<1.1013e-07): {best_stats['mean_fused_score']:.6e}")

    # Export json
    out_file = DIAGNOSTICS_DIR / "p4n_09_stein_controls.json"
    with open(out_file, "w") as f:
        json.dump({
            "experiment": "P4N-09",
            "control_adjusted_score": float(mean_ctrl_score),
            "best_configuration": str(best_cfg),
            "gate_passed": bool(gate_passed),
            "results": summary_export
        }, f, indent=2, default=float)
    print(f"Artifact exported to: {out_file}")


if __name__ == "__main__":
    run_p4n_09()
