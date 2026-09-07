"""p8_gate_residuals.py

Phase 8 Stage P8-G0/G1: Exact Gate-Residual Decomposition and Variance-Cost Diagnostic.
Reference: phase8plan.md Section 13.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
P8_DIR = REPO_ROOT / "research" / "phase8"
DIAG_DIR = P8_DIR / "diagnostics"


def run_gate_residual_decomposition(
    weights: List[np.ndarray],
    pilot_mu_pre: List[np.ndarray],
    X: np.ndarray,
) -> Dict[str, Any]:
    """Given weights, pilot preactivation means, and input samples X (shape N x n):
    Computes:
    - Gates D_l = diag(1[mu_pre_l >= 0])
    - Suffix products Q_l = A_15 ... A_(l+1)
    - Residuals r_l(X) = ReLU(z_l(X)) - D_l z_l(X)
    - Output contributions q_l(X) = Q_l r_l(X)
    - Exact l=0 source expectation: E[r_0] = row_norm(W_0^T) / sqrt(2*pi)
    - Verification of exact identity: h_15(X) = A_15...A_0 X + sum_l Q_l r_l(X)
    """
    depth = len(weights)
    N, n = X.shape

    # 1. Forward pass saving activations and computing gates
    # D_l = diag(1[mu_pre_l >= 0])
    # A_l = D_l @ W_l^T (shape n x n)
    A_list = []
    D_list = []
    z_list = []
    h_list = []

    curr_h = X.copy()
    for l in range(depth):
        W_ref = np.asarray(weights[l], dtype=np.float64).T  # (n, n)
        z_l = curr_h @ W_ref.T  # (N, n)
        z_list.append(z_l)

        mu_pre_l = pilot_mu_pre[l]
        d_l = (mu_pre_l >= 0.0).astype(np.float64)  # (n,)
        D_list.append(d_l)

        A_l = d_l[:, None] * W_ref  # (n, n)
        A_list.append(A_l)

        curr_h = np.maximum(0.0, z_l)  # ReLU
        h_list.append(curr_h)

    h_15 = h_list[-1]

    # 2. Backward suffix matrices:
    # Q_15 = I
    # Q_l = A_15 ... A_(l+1)
    Q_list = [None] * depth
    Q_list[depth - 1] = np.eye(n, dtype=np.float64)
    curr_Q = np.eye(n, dtype=np.float64)
    for l in range(depth - 2, -1, -1):
        curr_Q = curr_Q @ A_list[l + 1]
        Q_list[l] = curr_Q.copy()

    # 3. Residuals and transported contributions:
    # r_l(X) = ReLU(z_l) - z_l @ D_l
    # q_l(X) = r_l(X) @ Q_l^T
    r_list = []
    q_list = []
    for l in range(depth):
        d_l = D_list[l]
        r_l = np.maximum(0.0, z_list[l]) - z_list[l] * d_l[None, :]  # (N, n)
        r_list.append(r_l)
        q_l = r_l @ Q_list[l].T  # (N, n)
        q_list.append(q_l)

    # 4. Linear term: A_15 ... A_0 X
    linear_matrix = Q_list[0] @ A_list[0]
    linear_term = X @ linear_matrix.T  # (N, n)

    # 5. Check samplewise identity:
    # h_15 == linear_term + sum_l q_l
    reconstructed_h15 = linear_term + sum(q_list)
    max_decomp_diff = float(np.max(np.abs(h_15 - reconstructed_h15)))
    rel_decomp_err = float(max_decomp_diff / max(np.max(np.abs(h_15)), 1e-12))

    # 6. Exact l=0 residual expectation:
    # E[r_0] = row_norm(W_0^T) / sqrt(2*pi)
    W0_ref = np.asarray(weights[0], dtype=np.float64).T
    row_norms_W0 = np.linalg.norm(W0_ref, axis=1)  # (n,)
    exact_E_r0 = row_norms_W0 / np.sqrt(2.0 * np.pi)
    exact_E_q0 = exact_E_r0 @ Q_list[0].T  # (n,)

    return {
        "max_decomp_diff": max_decomp_diff,
        "rel_decomp_err": rel_decomp_err,
        "passes_samplewise_identity": bool(rel_decomp_err < 1e-10),
        "exact_E_r0": exact_E_r0,
        "exact_E_q0": exact_E_q0,
        "q_list": q_list,
        "h_15": h_15,
        "linear_term": linear_term,
    }


def verify_g0_identities(n: int = 5, depth: int = 4, n_samples: int = 50_000, seed: int = 8101) -> Dict[str, Any]:
    """Mandatory Section 13.1 G0 checks."""
    rng = np.random.default_rng(seed)
    weights = [rng.standard_normal((n, n)) * np.sqrt(2.0 / n) for _ in range(depth)]
    # Arbitrary preactivation means for pilot
    pilot_mu_pre = [rng.standard_normal(n) for _ in range(depth)]
    pilot_mu_pre[0] = np.zeros(n)  # Layer 0 preactivation mean is strictly 0 for Gaussian input

    X = rng.standard_normal((n_samples, n))

    res = run_gate_residual_decomposition(weights, pilot_mu_pre, X)

    # 1. Samplewise identity pass
    passes_identity = res["passes_samplewise_identity"]

    # 2. Linear term mean should be 0 in expectation (within MC SE)
    lin_mean = np.mean(res["linear_term"], axis=0)
    lin_se = np.std(res["linear_term"], axis=0) / np.sqrt(n_samples)
    lin_passes = bool(np.all(np.abs(lin_mean) < 3.5 * lin_se))

    # 3. Exact E[r_0] check vs sample mean
    r0_sample_mean = np.mean(res["q_list"][0], axis=0)  # transported
    exact_q0 = res["exact_E_q0"]
    q0_se = np.std(res["q_list"][0], axis=0) / np.sqrt(n_samples)
    q0_diff = np.max(np.abs(r0_sample_mean - exact_q0))
    q0_passes = bool(np.all(np.abs(r0_sample_mean - exact_q0) < 3.5 * q0_se))

    return {
        "samplewise_rel_err": res["rel_decomp_err"],
        "samplewise_passes": passes_identity,
        "linear_term_zero_mean_passes": lin_passes,
        "exact_r0_passes": q0_passes,
        "exact_r0_max_diff": float(q0_diff),
    }


def run_g1_diagnostic(
    dataset_path: str,
    n_pilot_samples: int = 2048,
    n_eval_samples: int = 4096,
    seed: int = 8101,
) -> Dict[str, Any]:
    """Stage P8-G1: Measure per-source variances, prefix costs, and evaluate
    independent and grouped optimal allocations under 0.15B and 0.25B budgets.
    """
    from whestbench.dataset import load_dataset, resolve_seed_context
    from whestbench.domain import MLP
    from scripts.p8_direct_sources import run_pilot_k2k4

    B = 2 ** 41
    ds = load_dataset(dataset_path, split="mini")
    salt_ver, seed_salt = resolve_seed_context(ds)

    mlp_diagnostics = []

    for idx in range(8):
        row = ds[idx]
        mlp = MLP.from_row(row, seed_protocol_version=salt_ver, seed_salt=seed_salt)
        depth = mlp.depth
        n = mlp.width
        weights = [np.asarray(w, dtype=np.float64) for w in mlp.weights]

        # Analytical pilot for gates
        pilot = run_pilot_k2k4(weights)
        pilot_mu_pre = [l["mu_pre"] for l in pilot["layers"]]

        rng = np.random.default_rng(seed + idx)
        X_pilot = rng.standard_normal((n_pilot_samples, n))
        X_eval = rng.standard_normal((n_eval_samples, n))

        decomp_pilot = run_gate_residual_decomposition(weights, pilot_mu_pre, X_pilot)
        decomp_eval = run_gate_residual_decomposition(weights, pilot_mu_pre, X_eval)

        # Output contributions: q_l for l=1..15
        # Exact l=0 is known analytically, variance = 0!
        variances = []
        for l in range(1, depth):
            q_l = decomp_pilot["q_list"][l]  # (N_pilot, n)
            v_l = float(np.mean(np.var(q_l, axis=0)))
            variances.append(v_l)

        # Prefix cost: cost to compute prefix up to layer l, residual, and transport by Q_l
        # Prefix forward to l: l * (2 * n^2)
        # Transport Q_l: 2 * n^2
        # Suffix matrix construction: fixed overhead ~ 15 * 2 * n^3
        # Prefix cost per sample at layer l: c_l = (2 * l + 2) * n^2
        costs = [(2 * l + 2) * (n ** 2) for l in range(1, depth)]

        # Group variances for four shared-prefix groups:
        # G1: {1, 2, 3}, G2: {4, 5, 6, 7}, G3: {8, 9, 10, 11}, G4: {12, 13, 14, 15}
        group_indices = [
            list(range(0, 3)),   # layers 1, 2, 3
            list(range(3, 7)),   # layers 4, 5, 6, 7
            list(range(7, 11)),  # layers 8, 9, 10, 11
            list(range(11, 15)), # layers 12, 13, 14, 15
        ]
        group_variances = []
        group_costs = []
        group_end_layers = [3, 7, 11, 15]
        for g_idx, (g_cols, end_l) in enumerate(zip(group_indices, group_end_layers)):
            # Summed contribution of group
            q_grp = sum(decomp_pilot["q_list"][l + 1] for l in g_cols)
            v_grp = float(np.mean(np.var(q_grp, axis=0)))
            group_variances.append(v_grp)
            # Group cost per sample: prefix to end_l plus group additions
            group_costs.append(2 * end_l * (n ** 2))

        # Total ordinary Monte Carlo variance of full output
        full_eval_var = float(np.mean(np.var(decomp_eval["h_15"], axis=0)))

        mlp_diagnostics.append({
            "mlp_name": mlp.name,
            "variances": variances,
            "costs": costs,
            "group_variances": group_variances,
            "group_costs": group_costs,
            "full_output_var": full_eval_var,
        })

    # Summary across MLPs
    mean_source_vars = np.mean([d["variances"] for d in mlp_diagnostics], axis=0).tolist()
    mean_group_vars = np.mean([d["group_variances"] for d in mlp_diagnostics], axis=0).tolist()
    mean_full_var = float(np.mean([d["full_output_var"] for d in mlp_diagnostics]))

    diag_result = {
        "mean_source_variances": mean_source_vars,
        "source_costs": mlp_diagnostics[0]["costs"],
        "mean_group_variances": mean_group_vars,
        "group_costs": mlp_diagnostics[0]["group_costs"],
        "mean_full_output_variance": mean_full_var,
        "per_mlp": mlp_diagnostics,
    }

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DIAG_DIR / "p8_g1_diagnostics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diag_result, f, indent=2)

    return diag_result


if __name__ == "__main__":
    print("Verifying G0 identities...")
    g0_res = verify_g0_identities(n=5, depth=4)
    print("G0 Identity Results:")
    for k, v in g0_res.items():
        print(f"  {k}: {v}")
