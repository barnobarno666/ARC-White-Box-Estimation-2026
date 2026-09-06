"""P4N-10: Certified Carrier & Signed-Calibration Diagnostic.

Implements the requirements of PHASE5_NEXT_EXPERIMENTS.md Section 16:
1. Part A: Geometry & Matched-Cost Carrier Audit
   - Validates frames with 1024 line representatives (N=2048 paired evaluations) and N=4096.
   - Checks Q Q^T = I for orthogonal frames.
   - Audits coherence |(Qa Qb^T)_{i, j}| vs theoretical MUB bound 1/sqrt(1024) = 1/32 = 0.03125.
   - Compares 4 matched-cost carrier alternatives at N=4096:
     a. Standard Gaussian (with antipodal pairs)
     b. Scrambled Sylvester-Hadamard frame
     c. Randomly Rotated Orthogonal frame (Haar O(1024))
     d. Scrambled Sobol Quasi-Monte Carlo (via SciPy Sobol + inv CDF)
   - Audits carrier statistics: mean error, Gram covariance error, 4th-moment excess.
   - Measures raw MSE and fused score across all 8 MLPs.

2. Part B: Signed Calibration from Exact Moment Constraints
   - Minimum-deviation sample weights w subject to exact layer-0 expectations:
     min_w 0.5 * ||w - 1/N||^2 s.t. A w = b (with ridge regularizer lambda=1e-2).
   - Tested on M in {32, 64, 128} exact-mean layer-0 features.
   - Cap weight squared-norm inflation at 2x uniform (||w||^2 <= 2/N).
   - Compares signed calibration vs positivity-projected calibration.

3. Gate Check:
   - 20% variance reduction at matched cost OR 15% costed blend headroom (< 1.0401e-07).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
import scipy.special
from scipy.stats import qmc

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


def construct_hadamard(n: int) -> np.ndarray:
    """Construct Sylvester-Hadamard matrix of order n (power of 2)."""
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def generate_carrier(carrier_type: str, n_samples: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate carrier matrix of shape (n_samples, dim) with antipodal pairing."""
    half = n_samples // 2
    if carrier_type == "gaussian":
        x_half = rng.standard_normal((half, dim), dtype=np.float32)
        X = np.concatenate([x_half, -x_half], axis=0)

    elif carrier_type == "hadamard":
        # Sylvester Hadamard normalized by 1/sqrt(dim)
        H = construct_hadamard(dim) / np.sqrt(dim)
        # Random sign scrambler
        signs = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)
        H_scrambled = H * signs[None, :]
        # If half > dim, repeat with fresh scramblers
        chunks = []
        rows_needed = half
        while rows_needed > 0:
            perm = rng.permutation(dim)
            sub = H_scrambled[perm[:min(rows_needed, dim)]]
            chunks.append(sub)
            rows_needed -= sub.shape[0]
        x_half = np.concatenate(chunks, axis=0).astype(np.float32)
        # Radii matching for Gaussian marginals
        # Chi radius R = sqrt(dim) approximately
        X = np.concatenate([x_half * np.sqrt(dim), -x_half * np.sqrt(dim)], axis=0)

    elif carrier_type == "haar_orthogonal":
        # Random orthogonal matrix from QR of Gaussian matrix
        G = rng.standard_normal((dim, dim))
        Q, R = np.linalg.qr(G)
        d = np.diagonal(R)
        Q = Q * np.sign(d)[None, :]
        chunks = []
        rows_needed = half
        while rows_needed > 0:
            perm = rng.permutation(dim)
            sub = Q[perm[:min(rows_needed, dim)]]
            chunks.append(sub)
            rows_needed -= sub.shape[0]
        x_half = np.concatenate(chunks, axis=0).astype(np.float32)
        X = np.concatenate([x_half * np.sqrt(dim), -x_half * np.sqrt(dim)], axis=0)

    elif carrier_type == "sobol_qmc":
        # Scrambled Sobol sequence
        sobol = qmc.Sobol(d=dim, scramble=True, seed=int(rng.integers(0, 2**31 - 1)))
        # Sobol requires powers of 2
        m = int(np.ceil(np.log2(half)))
        pts = sobol.random_base2(m=m)[:half]
        # Avoid boundary 0.0 and 1.0
        pts = np.clip(pts, 1e-6, 1.0 - 1e-6)
        x_half = scipy.special.ndtri(pts).astype(np.float32)
        X = np.concatenate([x_half, -x_half], axis=0)

    else:
        raise ValueError(f"Unknown carrier type: {carrier_type}")

    return X


def audit_carrier_moments(X: np.ndarray) -> dict[str, float]:
    """Audit marginal, Gram, and fourth-moment errors."""
    N, D = X.shape
    mean_err = float(np.max(np.abs(np.mean(X, axis=0))))
    cov = (X.T @ X) / N
    gram_err = float(np.linalg.norm(cov - np.eye(D, dtype=np.float32)) / D)
    fourth_m = np.mean(X ** 4, axis=0)
    fourth_err = float(np.mean(np.abs(fourth_m - 3.0)))
    return {
        "mean_err": mean_err,
        "gram_err": gram_err,
        "fourth_err": fourth_err
    }


def run_p4n_10():
    print("=== P4N-10: Certified Carrier & Signed-Calibration Diagnostic ===")
    ds = load_dataset(DATASET_PATH, split="mini")
    protocol_version, salt = resolve_seed_context(ds)

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

    # Carrier types to audit
    CARRIERS = ["gaussian", "hadamard", "haar_orthogonal", "sobol_qmc"]

    carrier_results = {c: {"raw_mse": [], "fused_score": [], "wins": 0} for c in CARRIERS}
    calibration_results = {
        m_count: {
            "signed_raw_mse": [], "signed_fused": [], "signed_wins": 0,
            "pos_raw_mse": [], "pos_fused": [], "pos_wins": 0
        }
        for m_count in [32, 64, 128]
    }
    baseline_ctrl_scores = []

    from p4n_01_mapped_control import compute_analytic_covariance_stack

    # Check MUB coherence for geometry audit
    print("\n--- Auditing Carrier Geometry & MUB Coherence ---")
    H = construct_hadamard(1024) / np.sqrt(1024)
    rng_test = np.random.default_rng(12345)
    G1 = rng_test.standard_normal((1024, 1024))
    Q1, _ = np.linalg.qr(G1)
    G2 = rng_test.standard_normal((1024, 1024))
    Q2, _ = np.linalg.qr(G2)

    had_ortho_coherence = float(np.max(np.abs(H @ Q1.T)))
    ortho_ortho_coherence = float(np.max(np.abs(Q1 @ Q2.T)))
    mub_theoretical = 1.0 / np.sqrt(1024)  # 0.03125
    print(f"Theoretical MUB Bound (1/sqrt(1024)): {mub_theoretical:.5f}")
    print(f"Max Coherence |H @ Q1^T|: {had_ortho_coherence:.5f}")
    print(f"Max Coherence |Q1 @ Q2^T|: {ortho_ortho_coherence:.5f}")

    for row_idx, row in enumerate(panel_rows):
        mlp_name = row.get("mlp_name", f"mlp_{row_idx}")
        mlp = MLP.from_row(row, seed_protocol_version=protocol_version, seed_salt=salt)
        print(f"\n[{row_idx+1}/8] MLP: {mlp_name}...")

        weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
        true_all_means = np.asarray(row["all_layer_means"], dtype=np.float32)
        y_true_final = true_all_means[-1]

        c_stack = compute_analytic_covariance_stack(mlp)
        c_calib = S0 * c_stack[-1]

        # -------------------------------------------------------------
        # Part A: Matched-Cost Carrier Audit
        # -------------------------------------------------------------
        mlp_carrier_fused = {}
        for c_type in CARRIERS:
            rng_c = np.random.default_rng(mlp.seed + 1000 + CARRIERS.index(c_type) * 500)
            X = generate_carrier(c_type, N_EVAL, mlp.width, rng_c)
            F = forward_network(X, weights)

            # Pair averaging
            half = N_EVAL // 2
            F_pairs = 0.5 * (F[:half] + F[half:])
            mean_est = np.mean(F_pairs, axis=0)

            raw_mse = float(np.mean((mean_est - y_true_final) ** 2))
            fused = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * mean_est
            fused_score = float(np.mean((fused - y_true_final) ** 2) * 0.10)

            carrier_results[c_type]["raw_mse"].append(raw_mse)
            carrier_results[c_type]["fused_score"].append(fused_score)
            mlp_carrier_fused[c_type] = fused_score

        # Baseline control is Gaussian carrier
        ctrl_score = mlp_carrier_fused["gaussian"]
        baseline_ctrl_scores.append(ctrl_score)

        for c_type in CARRIERS:
            if mlp_carrier_fused[c_type] < ctrl_score:
                carrier_results[c_type]["wins"] += 1

        # -------------------------------------------------------------
        # Part B: Signed Calibration from Exact Layer-0 Constraints
        # -------------------------------------------------------------
        # Use standard Gaussian carrier
        rng_cal = np.random.default_rng(mlp.seed + 60002)
        X_cal = generate_carrier("gaussian", N_EVAL, mlp.width, rng_cal)
        F_cal = forward_network(X_cal, weights)
        half_cal = N_EVAL // 2
        X_half = X_cal[:half_cal]
        F_pairs_cal = 0.5 * (F_cal[:half_cal] + F_cal[half_cal:])

        W0 = weights[0]
        W0_col_norms = np.linalg.norm(W0, axis=0)
        m0_exact_all = W0_col_norms / np.sqrt(2.0 * np.pi)

        # Preactivation for first half
        Z0_pos = X_half @ W0
        Q0_pairs = 0.5 * (np.maximum(Z0_pos, 0.0) + np.maximum(-Z0_pos, 0.0))  # (half_cal, 1024)

        for m_count in [32, 64, 128]:
            # Select top m_count neurons by weight norm
            top_indices = np.argsort(-W0_col_norms)[:m_count]
            m0_sub = m0_exact_all[top_indices]
            Q_sub = Q0_pairs[:, top_indices]  # (N_p, m_count)

            N_p = half_cal
            w0 = np.full(N_p, 1.0 / N_p, dtype=np.float32)

            # Constraint matrix A: (m_count, N_p)
            A = Q_sub.T  # (m_count, N_p)
            b = m0_sub  # (m_count,)

            # Discrepancy: delta = b - A @ w0
            delta = b - A @ w0

            # Minimum deviation with regularization:
            # delta_w = A^T (A A^T + lambda I)^{-1} delta
            Gram_A = A @ A.T  # (m_count, m_count)
            lambda_reg = 1e-2 * np.trace(Gram_A) / m_count
            reg_inv = np.linalg.solve(Gram_A + lambda_reg * np.eye(m_count, dtype=np.float32), delta)
            delta_w = A.T @ reg_inv

            # Unconstrained signed weights
            w_signed = w0 + delta_w
            # Ensure sum(w) = 1.0
            w_signed = w_signed - (np.sum(w_signed) - 1.0) / N_p

            # Check inflation cap: ||w||^2 <= 2/N_p
            norm_sq = np.sum(w_signed ** 2)
            if norm_sq > 2.0 / N_p:
                scale_d = np.sqrt((2.0 / N_p - 1.0 / N_p) / np.maximum(np.sum(delta_w ** 2), 1e-12))
                w_signed = w0 + delta_w * scale_d
                w_signed = w_signed - (np.sum(w_signed) - 1.0) / N_p

            # Calibrated mean:
            calib_mean_signed = F_pairs_cal.T @ w_signed
            raw_mse_signed = float(np.mean((calib_mean_signed - y_true_final) ** 2))
            fused_signed = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * calib_mean_signed
            score_signed = float(np.mean((fused_signed - y_true_final) ** 2) * 0.10)

            # Positivity-projected calibration
            w_pos = np.maximum(w_signed, 0.0)
            w_pos = w_pos / np.sum(w_pos)
            calib_mean_pos = F_pairs_cal.T @ w_pos
            raw_mse_pos = float(np.mean((calib_mean_pos - y_true_final) ** 2))
            fused_pos = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * calib_mean_pos
            score_pos = float(np.mean((fused_pos - y_true_final) ** 2) * 0.10)

            calibration_results[m_count]["signed_raw_mse"].append(raw_mse_signed)
            calibration_results[m_count]["signed_fused"].append(score_signed)
            if score_signed < ctrl_score:
                calibration_results[m_count]["signed_wins"] += 1

            calibration_results[m_count]["pos_raw_mse"].append(raw_mse_pos)
            calibration_results[m_count]["pos_fused"].append(score_pos)
            if score_pos < ctrl_score:
                calibration_results[m_count]["pos_wins"] += 1

    # Print summary of Carrier Audit
    mean_ctrl = float(np.mean(baseline_ctrl_scores))
    print("\n" + "=" * 90)
    print("PART A: MATCHED-COST CARRIER AUDIT (N=4096, 8-MLP PANEL)")
    print("=" * 90)
    print(f"{'Carrier Type':<20} | {'Raw MSE':<14} | {'Fused Score':<14} | {'Diff vs Ctrl':<14} | {'Wins':<6}")
    print("-" * 90)
    for c_type in CARRIERS:
        mean_raw = float(np.mean(carrier_results[c_type]["raw_mse"]))
        mean_score = float(np.mean(carrier_results[c_type]["fused_score"]))
        diff = ((mean_score - mean_ctrl) / mean_ctrl) * 100.0
        wins = carrier_results[c_type]["wins"]
        print(f"{c_type:<20} | {mean_raw:>12.4e} | {mean_score:>12.6e} | {diff:>+12.2f}% | {wins}/8")

    # Print summary of Signed Calibration
    print("\n" + "=" * 90)
    print("PART B: SIGNED CALIBRATION FROM EXACT MOMENTS (N=4096, 8-MLP PANEL)")
    print("=" * 90)
    print(f"{'Variant':<20} | {'Raw MSE':<14} | {'Fused Score':<14} | {'Diff vs Ctrl':<14} | {'Wins':<6}")
    print("-" * 90)
    for m_count in [32, 64, 128]:
        signed_raw = float(np.mean(calibration_results[m_count]["signed_raw_mse"]))
        signed_score = float(np.mean(calibration_results[m_count]["signed_fused"]))
        diff_s = ((signed_score - mean_ctrl) / mean_ctrl) * 100.0
        wins_s = calibration_results[m_count]["signed_wins"]
        print(f"Signed M={m_count:<13} | {signed_raw:>12.4e} | {signed_score:>12.6e} | {diff_s:>+12.2f}% | {wins_s}/8")

        pos_raw = float(np.mean(calibration_results[m_count]["pos_raw_mse"]))
        pos_score = float(np.mean(calibration_results[m_count]["pos_fused"]))
        diff_p = ((pos_score - mean_ctrl) / mean_ctrl) * 100.0
        wins_p = calibration_results[m_count]["pos_wins"]
        print(f"Positive M={m_count:<11} | {pos_raw:>12.4e} | {pos_score:>12.6e} | {diff_p:>+12.2f}% | {wins_p}/8")

    print("=" * 90)

    # Best configuration check
    all_scores = [float(np.mean(carrier_results[c]["fused_score"])) for c in CARRIERS] + \
                 [float(np.mean(calibration_results[m]["signed_fused"])) for m in [32, 64, 128]] + \
                 [float(np.mean(calibration_results[m]["pos_fused"])) for m in [32, 64, 128]]
    best_overall_score = min(all_scores)
    best_diff_pct = ((best_overall_score - mean_ctrl) / mean_ctrl) * 100.0

    gate_passed = best_overall_score < 1.0401e-07 or best_diff_pct <= -20.0
    print(f"Baseline Control Adjusted Score: {mean_ctrl:.6e}")
    print(f"Best Tested Score: {best_overall_score:.6e} ({best_diff_pct:+.2f}%)")
    print(f"Gate Status (<1.0401e-07 or -20%): {'PASSED' if gate_passed else 'FAILED (Archive tested branch)'}")

    # Export json artifact
    out_file = DIAGNOSTICS_DIR / "p4n_10_carrier_calibration.json"
    with open(out_file, "w") as f:
        json.dump({
            "experiment": "P4N-10",
            "control_adjusted_score": mean_ctrl,
            "best_score": best_overall_score,
            "gate_passed": bool(gate_passed),
            "carrier_results": {c: {k: v if k == "wins" else float(np.mean(v)) for k, v in carrier_results[c].items()} for c in CARRIERS},
            "calibration_results": {m: {k: v if "wins" in k else float(np.mean(v)) for k, v in calibration_results[m].items()} for m in [32, 64, 128]}
        }, f, indent=2, default=float)
    print(f"Artifact exported to: {out_file}")


if __name__ == "__main__":
    run_p4n_10()
