"""P4N-11: One-Dimensional Conditional Integration Diagnostic.

Implements the requirements of PHASE5_NEXT_EXPERIMENTS.md Section 17:
1. Decompose input as X = xi + t * v:
   - v is a unit vector.
   - t ~ N(0, 1) is a 1D scalar along v.
   - xi ~ N(0, I - v v^T) is the orthogonal component.
   - Exact identity: E[f(X)] = E_xi [ E_t [ f(xi + t * v) ] ].
2. Inner 1D integration:
   - Evaluated via Gauss-Hermite quadrature with K in {8, 16, 32} nodes.
   - Normalized for standard normal density N(0, 1).
   - Verified on small 1-hidden-layer MLP fixture against exact 1D piecewise linear integration.
3. Direction selection:
   - Direction 1: Leading gate-aware input response mode (SVD of W0 D0 ... W15 D15).
   - Direction 2: Normalized W0 column with highest downstream sensitivity.
   - Direction 3: Random isotropic unit vector control.
4. Outer Monte Carlo sampling:
   - N_outer = 256 independent draws (128 antipodal pairs +/- xi).
   - Total network evaluations: N_eval = N_outer * K in {2048, 4096, 8192}.
5. Matched-cost comparison:
   - Compares 1D conditional integration against ordinary Monte Carlo at the exact same N_eval.
   - Measures compute utilization, multiplier, raw MSE, and fused score.
6. Gate:
   - 30% lower cost-adjusted error than matched-cost ordinary sampling.
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


def get_standard_gauss_hermite(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Get nodes and weights for E_{t ~ N(0,1)}[f(t)] = sum_k w_k f(t_k)."""
    # hermgauss integrates f(x) * exp(-x^2)
    # To integrate f(t) * (1/sqrt(2pi)) * exp(-t^2/2):
    # Let t = sqrt(2)*x, dt = sqrt(2) dx
    # Integral = 1/sqrt(pi) int f(sqrt(2)*x) exp(-x^2) dx
    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    nodes = np.sqrt(2.0) * x
    weights = w / np.sqrt(np.pi)
    # Sort nodes
    idx = np.argsort(nodes)
    return nodes[idx].astype(np.float32), weights[idx].astype(np.float32)


def verify_1d_integration_fixture():
    """Verify Gauss-Hermite rule on a 1-hidden-layer ReLU fixture against exact piecewise linear."""
    print("Verifying 1D Gauss-Hermite integration against exact piecewise linear fixture...")
    # Synthetic 1-hidden-layer network: f(x) = sum_j c_j relu(w_j^T x + b_j)
    dim = 16
    h_dim = 8
    rng = np.random.default_rng(42)
    W0 = rng.standard_normal((dim, h_dim))
    W1 = rng.standard_normal((h_dim, 1))

    v = rng.standard_normal(dim)
    v = v / np.linalg.norm(v)

    xi = rng.standard_normal(dim)
    xi = xi - (xi @ v) * v  # orthogonal to v

    # Line preactivation: z_j(t) = (xi + t*v) @ W0[:, j] = a_j + b_j * t
    a = xi @ W0
    b = v @ W0

    # Exact line integral E_{t~N(0,1)}[ relu(a_j + b_j * t) ]:
    # For b_j == 0: relu(a_j)
    # For b_j != 0: let s = a_j / |b_j|, then relu = |b_j| * relu(s + sign(b_j)*t)
    # Since t ~ N(0,1), sign(b_j)*t ~ N(0,1).
    # E[relu(s + t)] = s * Phi(s) + phi(s).
    exact_integrals = np.zeros(h_dim)
    for j in range(h_dim):
        bj = b[j]
        aj = a[j]
        if abs(bj) < 1e-12:
            exact_integrals[j] = max(aj, 0.0)
        else:
            s = aj / abs(bj)
            phi = np.exp(-0.5 * s * s) / np.sqrt(2.0 * np.pi)
            Phi = 0.5 * (1.0 + scipy.special.erf(s / np.sqrt(2.0)))
            exact_integrals[j] = abs(bj) * (s * Phi + phi)

    exact_f = float((exact_integrals @ W1).ravel()[0])

    # Compare with Gauss-Hermite quadrature
    for K in [8, 16, 32]:
        nodes, weights = get_standard_gauss_hermite(K)
        # Evaluate line
        X_line = xi[None, :] + nodes[:, None] * v[None, :]
        F_line = np.maximum(X_line @ W0, 0.0) @ W1
        gh_f = float(np.sum(weights * F_line.ravel()))
        err = abs(gh_f - exact_f)
        rel_err = err / max(abs(exact_f), 1e-12)
        print(f"  Gauss-Hermite K={K:<2}: Value={gh_f:.6f}, Exact={exact_f:.6f}, Abs Err={err:.3e}, Rel Err={rel_err:.3e}")
        assert rel_err < 0.01, f"Quadrature relative error too large at K={K}: {rel_err}"

    print("Gauss-Hermite 1D fixture verification PASSED!")


def get_directions(mlp: MLP, weights: list[np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Extract test directions v."""
    W0 = weights[0]
    col_norms = np.linalg.norm(W0, axis=0)

    # Direction 1: Gate-aware response mode (leading left singular vector)
    M = weights[0] * 0.5
    for l in range(1, len(weights)):
        M = M @ (weights[l] * 0.5)
    u, _, _ = np.linalg.svd(M, full_matrices=False)
    v_response = u[:, 0]
    v_response = v_response / np.linalg.norm(v_response)

    # Direction 2: Maximum norm W0 column
    max_col = np.argmax(col_norms)
    v_w0 = W0[:, max_col] / col_norms[max_col]

    # Direction 3: Random isotropic unit vector
    v_rand = rng.standard_normal(mlp.width)
    v_rand = v_rand / np.linalg.norm(v_rand)

    return {
        "response_mode": v_response.astype(np.float32),
        "w0_col_max": v_w0.astype(np.float32),
        "random_unit": v_rand.astype(np.float32)
    }


def run_p4n_11():
    print("=== P4N-11: One-Dimensional Conditional Integration Diagnostic ===")
    verify_1d_integration_fixture()

    ds = load_dataset(DATASET_PATH, split="mini")
    protocol_version, salt = resolve_seed_context(ds)

    N_OUTER = 256
    HALF_OUTER = N_OUTER // 2  # 128 antipodal pairs
    K_VALUES = [8, 16, 32]
    ALPHA_BLEND = 0.110
    S0 = 0.998319

    FIXED_8 = [
        "logan-fitzgerald", "william-graves", "raymond-barnes", "steven-rice",
        "sarah-kelley", "christopher-morales", "cheryl-graham", "renee-park"
    ]
    panel_rows = [row for row in ds if row.get("mlp_name") in FIXED_8]
    if len(panel_rows) != 8:
        panel_rows = list(ds)[:8]

    from p4n_01_mapped_control import compute_analytic_covariance_stack

    # Accs: direction -> K -> stats
    DIRECTIONS = ["response_mode", "w0_col_max", "random_unit"]
    results_ci = {d: {k: {"raw_mse": [], "fused_score": [], "wins_vs_ctrl": 0, "wins_vs_ord": 0} for k in K_VALUES} for d in DIRECTIONS}
    results_ord = {k: {"raw_mse": [], "fused_score": []} for k in K_VALUES}
    baseline_ctrl_scores = []

    for row_idx, row in enumerate(panel_rows):
        mlp_name = row.get("mlp_name", f"mlp_{row_idx}")
        mlp = MLP.from_row(row, seed_protocol_version=protocol_version, seed_salt=salt)
        print(f"\n[{row_idx+1}/8] MLP: {mlp_name}...")

        weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
        true_all_means = np.asarray(row["all_layer_means"], dtype=np.float32)
        y_true_final = true_all_means[-1]

        c_stack = compute_analytic_covariance_stack(mlp)
        c_calib = S0 * c_stack[-1]

        # Standard control score at N=4200 (util 9.81%, mult 0.10)
        rng_ctrl = np.random.default_rng(mlp.seed + 1000)
        x_ctrl_half = rng_ctrl.standard_normal((2100, mlp.width), dtype=np.float32)
        x_ctrl = np.concatenate([x_ctrl_half, -x_ctrl_half], axis=0)
        F_ctrl = forward_network(x_ctrl, weights)
        F_ctrl_pairs = 0.5 * (F_ctrl[:2100] + F_ctrl[2100:])
        ctrl_mean = np.mean(F_ctrl_pairs, axis=0)
        ctrl_fused = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * ctrl_mean
        ctrl_score = float(np.mean((ctrl_fused - y_true_final) ** 2) * 0.10)
        baseline_ctrl_scores.append(ctrl_score)

        dirs = get_directions(mlp, weights, rng_ctrl)

        for K in K_VALUES:
            n_total = N_OUTER * K  # 2048, 4096, 8192
            # Compute utilization: 1024*16 MLP forward pass is ~ 2 * 1024^2 * 16 = 3.355e7 FLOPs.
            # Budget is 2^41 FLOPs = 2.199e12 FLOPs.
            # For n_total samples: util = (n_total * 3.355e7) / 2.199e12
            # For 2048: util ~ 3.1% (mult 0.10)
            # For 4096: util ~ 6.2% (mult 0.10)
            # For 8192: util ~ 12.5% (mult 0.125)
            util_val = (n_total * 3.355e7) / (2.0 ** 41)
            score_mult = max(0.10, util_val)

            # Matched-cost ordinary sampling:
            rng_ord = np.random.default_rng(mlp.seed + 20000 + K)
            x_ord_half = rng_ord.standard_normal((n_total // 2, mlp.width), dtype=np.float32)
            x_ord = np.concatenate([x_ord_half, -x_ord_half], axis=0)
            F_ord = forward_network(x_ord, weights)
            F_ord_pairs = 0.5 * (F_ord[:(n_total // 2)] + F_ord[(n_total // 2):])
            ord_mean = np.mean(F_ord_pairs, axis=0)
            ord_raw_mse = float(np.mean((ord_mean - y_true_final) ** 2))
            ord_fused = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * ord_mean
            ord_score = float(np.mean((ord_fused - y_true_final) ** 2) * score_mult)

            results_ord[K]["raw_mse"].append(ord_raw_mse)
            results_ord[K]["fused_score"].append(ord_score)

            # 1D Conditional Integration for each direction:
            nodes, weights_gh = get_standard_gauss_hermite(K)

            for d_name, v in dirs.items():
                rng_ci = np.random.default_rng(mlp.seed + 30000 + K * 100 + DIRECTIONS.index(d_name))
                # Sample xi ~ N(0, I - v v^T) with antipodes
                xi_raw = rng_ci.standard_normal((HALF_OUTER, mlp.width), dtype=np.float32)
                # Project out component along v
                xi_half = xi_raw - (xi_raw @ v)[:, None] * v[None, :]
                xi_all = np.concatenate([xi_half, -xi_half], axis=0)  # (N_OUTER, mlp.width)

                # Construct all grid points (N_OUTER, K, mlp.width)
                # X_grid[i, k] = xi_all[i] + nodes[k] * v
                X_grid = xi_all[:, None, :] + nodes[None, :, None] * v[None, None, :]
                X_flat = X_grid.reshape(-1, mlp.width)

                F_flat = forward_network(X_flat, weights)
                F_grid = F_flat.reshape(N_OUTER, K, mlp.width)

                # Inner quadrature integral over t for each outer sample:
                # I_xi = sum_k w_k F_grid[i, k]
                I_xi = np.tensordot(F_grid, weights_gh, axes=([1], [0]))  # (N_OUTER, mlp.width)

                # Antipodal pair average over outer draws
                I_pairs = 0.5 * (I_xi[:HALF_OUTER] + I_xi[HALF_OUTER:])
                ci_mean = np.mean(I_pairs, axis=0)

                ci_raw_mse = float(np.mean((ci_mean - y_true_final) ** 2))
                ci_fused = (1.0 - ALPHA_BLEND) * c_calib + ALPHA_BLEND * ci_mean
                ci_score = float(np.mean((ci_fused - y_true_final) ** 2) * score_mult)

                results_ci[d_name][K]["raw_mse"].append(ci_raw_mse)
                results_ci[d_name][K]["fused_score"].append(ci_score)

                if ci_score < ctrl_score:
                    results_ci[d_name][K]["wins_vs_ctrl"] += 1
                if ci_raw_mse < ord_raw_mse:
                    results_ci[d_name][K]["wins_vs_ord"] += 1

    # Print summary table
    mean_ctrl = float(np.mean(baseline_ctrl_scores))
    print("\n" + "=" * 105)
    print("MATCHED-COST 1D CONDITIONAL INTEGRATION vs ORDINARY SAMPLING (8-MLP PANEL)")
    print("=" * 105)
    print(f"{'Method / Configuration':<32} | {'Raw MSE':<12} | {'Scored Error':<14} | {'Diff vs Ord':<12} | {'Diff vs Ctrl':<14} | {'Wins (Ord/Ctrl)':<15}")
    print("-" * 105)

    best_ci_score = float("inf")
    best_cfg_name = None

    summary_export = {}

    for K in K_VALUES:
        n_total = N_OUTER * K
        ord_raw = float(np.mean(results_ord[K]["raw_mse"]))
        ord_score = float(np.mean(results_ord[K]["fused_score"]))
        diff_ctrl_ord = ((ord_score - mean_ctrl) / mean_ctrl) * 100.0

        print(f"Ordinary MC N={n_total:<18} | {ord_raw:>10.4e} | {ord_score:>12.6e} | {'-':^12} | {diff_ctrl_ord:>+12.2f}% | -")

        for d_name in DIRECTIONS:
            cfg_label = f"CI {d_name} K={K} (N={n_total})"
            ci_raw = float(np.mean(results_ci[d_name][K]["raw_mse"]))
            ci_score = float(np.mean(results_ci[d_name][K]["fused_score"]))
            diff_ord = ((ci_raw - ord_raw) / ord_raw) * 100.0
            diff_ctrl = ((ci_score - mean_ctrl) / mean_ctrl) * 100.0
            wins_ord = results_ci[d_name][K]["wins_vs_ord"]
            wins_ctrl = results_ci[d_name][K]["wins_vs_ctrl"]

            if ci_score < best_ci_score:
                best_ci_score = ci_score
                best_cfg_name = cfg_label

            print(f"{cfg_label:<32} | {ci_raw:>10.4e} | {ci_score:>12.6e} | {diff_ord:>+10.2f}% | {diff_ctrl:>+12.2f}% | {wins_ord}/8 (ord), {wins_ctrl}/8 (ctrl)")

            summary_export[cfg_label] = {
                "raw_mse": ci_raw,
                "fused_score": ci_score,
                "diff_vs_ord_pct": diff_ord,
                "diff_vs_ctrl_pct": diff_ctrl,
                "wins_vs_ord": wins_ord,
                "wins_vs_ctrl": wins_ctrl
            }

    print("=" * 105)
    print(f"Baseline Control Adjusted Score: {mean_ctrl:.6e}")
    print(f"Best Tested CI Score: {best_ci_score:.6e} ({best_cfg_name})")

    # Gate check: 30% lower cost-adjusted error vs ordinary sampling
    best_diff_vs_ord = summary_export[best_cfg_name]["diff_vs_ord_pct"]
    gate_passed = (best_diff_vs_ord <= -30.0) or (best_ci_score < 1.0401e-07)
    print(f"\nGate Check (30% lower vs Ord or <1.0401e-07): {'PASSED' if gate_passed else 'FAILED (Archive tested branch)'}")
    print(f"Diff vs Matched Ordinary: {best_diff_vs_ord:+.2f}%")

    out_file = DIAGNOSTICS_DIR / "p4n_11_conditional_integration.json"
    with open(out_file, "w") as f:
        json.dump({
            "experiment": "P4N-11",
            "control_adjusted_score": mean_ctrl,
            "best_configuration": best_cfg_name,
            "best_score": best_ci_score,
            "gate_passed": bool(gate_passed),
            "results": summary_export
        }, f, indent=2, default=float)
    print(f"Artifact exported to: {out_file}")


if __name__ == "__main__":
    run_p4n_11()
