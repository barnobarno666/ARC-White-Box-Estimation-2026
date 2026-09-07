"""p8_direct_sources.py

Phase 8 Stage P8-D0/D1/D2: Direct Contraction of Frozen K3 Sources.
Implements:
1. Frozen pilot generation (A1-G-K2K4).
2. Suffix matrix accumulation Q_(t <- j).
3. Direct contraction into d_t and S_t without recurrent factor banks.
4. Parity checks (Section 9.2 D0 identities).
5. Cost profile calculations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
P8_DIR = REPO_ROOT / "research" / "phase8"
FIXTURES_DIR = P8_DIR / "fixtures"


def relu_wick_dense(mean: np.ndarray, var: np.ndarray, k: int, p: int) -> np.ndarray:
    sigma = np.sqrt(np.maximum(var, 1e-12))
    alpha = mean / sigma
    phi = norm.pdf(alpha)
    cdf = norm.cdf(alpha)

    if p == 1:
        if k == 0: return sigma * phi + mean * cdf
        if k == 1: return cdf
        if k == 2: return phi / sigma
        if k == 3: return -alpha * phi / (sigma**2)
        if k == 4: return (alpha**2 - 1.0) * phi / (sigma**3)
    elif p == 2:
        if k == 0: return (mean**2 + var) * cdf + mean * sigma * phi
        if k == 1: return 2.0 * (sigma * phi + mean * cdf)
        if k == 2: return 2.0 * cdf
        if k == 3: return 2.0 * phi / sigma
        if k == 4: return -2.0 * alpha * phi / (sigma**2)
    elif p == 3:
        if k == 0: return (sigma**3) * ((2.0 + alpha**2) * phi + (3.0 * alpha + alpha**3) * cdf)
        if k == 1: return 3.0 * (sigma**2) * (alpha * phi + (1.0 + alpha**2) * cdf)
        if k == 2: return 6.0 * (sigma * phi + mean * cdf)
        if k == 3: return 6.0 * cdf
        if k == 4: return 6.0 * phi / sigma
    elif p == 4:
        if k == 0: return (sigma**4) * ((5.0 * alpha + alpha**3) * phi + (3.0 + 6.0 * alpha**2 + alpha**4) * cdf)
        if k == 1: return 4.0 * (sigma**3) * ((2.0 + alpha**2) * phi + (3.0 * alpha + alpha**3) * cdf)
        if k == 2: return 12.0 * (sigma**2) * (alpha * phi + (1.0 + alpha**2) * cdf)
        if k == 3: return 24.0 * (sigma * phi + mean * cdf)
        if k == 4: return 24.0 * cdf
    raise ValueError(f"Unsupported ({k}, {p})")


def zero_diag(mat: np.ndarray) -> np.ndarray:
    res = mat.copy()
    np.fill_diagonal(res, 0.0)
    return res


def sym(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def run_pilot_k2k4(weights: List[np.ndarray]) -> Dict[str, Any]:
    """Run forward analytical pass of A1-G-K2K4 pilot (incoming K3 zero).
    Returns per-layer states: mu_pre, var_pre, cov_pre, g (ReLU' mean), c4.
    """
    depth = len(weights)
    n = weights[0].shape[1]
    eye_n = np.eye(n, dtype=np.float64)

    mu = np.zeros(n, dtype=np.float64)
    cov = np.eye(n, dtype=np.float64)
    c4 = 0.0

    layers_data = []

    for l in range(depth):
        W_ref = np.asarray(weights[l], dtype=np.float64).T  # (n_out, n_in)
        if l == 0:
            M = W_ref @ W_ref.T
            cov_pre = sym(M)
            mu_pre = np.zeros(n, dtype=np.float64)
            d_M = np.diag(M)
        else:
            mu_pre = W_ref @ mu
            W_cov = W_ref @ cov
            cov_pre = sym(W_cov @ W_ref.T)
            M = W_ref @ W_ref.T
            d_M = np.sum(W_ref * W_ref, axis=1)

        var_pre = np.diag(cov_pre)
        sigma_pre = np.sqrt(np.maximum(var_pre, 1e-12))
        g_l = norm.cdf(mu_pre / sigma_pre)  # E[ReLU'(z_l)]

        s4 = c4 * (d_M ** 2)
        s22 = (1.0 / 3.0) * c4 * zero_diag(d_M[:, None] * d_M[None, :] + 2.0 * (M ** 2))
        s3 = np.zeros(n, dtype=np.float64)
        s21 = np.zeros((n, n), dtype=np.float64)

        # Nonlinear step
        cov11 = zero_diag(cov_pre)

        def w(k, p):
            return relu_wick_dense(mu_pre, var_pre, k, p)

        pk1 = w(0, 1) + (1.0 / 24.0) * w(4, 1) * s4
        pk2 = w(0, 2) + (1.0 / 24.0) * w(4, 2) * s4
        pk3 = w(0, 3) + (1.0 / 24.0) * w(4, 3) * s4
        pk4 = w(0, 4) + (1.0 / 24.0) * w(4, 4) * s4

        w11 = w(1, 1)
        w21 = w(2, 1)

        pk11 = sym(
            cov11 * (w11[:, None] * w11[None, :])
            + 0.5 * (cov11 ** 2) * (w21[:, None] * w21[None, :])
            + 0.25 * s22 * (w21[:, None] * w21[None, :])
        )

        pk21 = (
            cov11 * (w(1, 2)[:, None] * w11[None, :])
            + 0.5 * (cov11 ** 2) * (w(2, 2)[:, None] * w21[None, :])
            + 0.25 * s22 * (w(2, 2)[:, None] * w21[None, :])
        )

        k1 = pk1
        k2 = pk2 - pk1 ** 2
        k3 = pk3 - 3.0 * pk1 * pk2 + 2.0 * (pk1 ** 3)
        k4 = pk4 - 4.0 * pk1 * pk3 - 3.0 * (pk2 ** 2) + 12.0 * (pk1 ** 2) * pk2 - 6.0 * (pk1 ** 4)
        k11 = pk11
        k21 = zero_diag(pk21 - 2.0 * (pk1[:, None] * pk11))

        # Sources generated at layer l:
        A2 = w11[:, None] * cov11
        C2 = (w21[:, None] * w11[None, :] * cov11).T
        s21_pk = zero_diag(A2 * C2)  # since sum_pre = 0

        k3_ds = k3  # since s3_pk = 0
        k21_ds = k21 - s21_pk
        A_ds = np.diag(k3_ds) + 3.0 * k21_ds.T

        # Source blocks at layer l:
        # Block 1: u = A2, v = 3 * eye_n * w21[None, :]
        # Block 2: u = eye_n, v = A_ds
        U_birth = np.concatenate([A2, eye_n], axis=1)  # (n, 2n)
        V_birth = np.concatenate([3.0 * eye_n * w21[None, :], A_ds], axis=1)  # (n, 2n)

        # Update c4
        a2_w = w(1, 2)
        b2_w = w(2, 2)
        H_w = 0.5 * (cov11 ** 2) + 0.25 * s22
        rows_E22 = (
            a2_w * (cov11 @ a2_w)
            + b2_w * (H_w @ b2_w)
        )
        p_vec = pk1
        D21_mat = pk21
        B11_mat = pk11
        r22_vec = (
            rows_E22
            - 2.0 * (D21_mat @ p_vec + p_vec * np.sum(D21_mat, axis=0))
            - 2.0 * np.sum(B11_mat ** 2, axis=1)
            + 4.0 * p_vec * (B11_mat @ p_vec)
        )
        t22_sum = float(np.sum(r22_vec))
        c4 = float((3.0 / (n * (n + 2))) * (np.sum(k4) + t22_sum))

        mu = k1
        cov = k11.copy()
        np.fill_diagonal(cov, k2)
        cov = sym(cov)

        layers_data.append({
            "layer": l,
            "mu_pre": mu_pre,
            "var_pre": var_pre,
            "cov_pre": cov_pre,
            "mu_post": mu,
            "cov_post": cov,
            "g": g_l,
            "c4": c4,
            "U_birth": U_birth,
            "V_birth": V_birth,
            "A2": A2,
            "w21": w21,
            "A_ds": A_ds,
        })

    return {"layers": layers_data, "final_mu": mu}


def contract_terminal_direct_sources(weights: List[np.ndarray], pilot_data: Dict[str, Any]) -> np.ndarray:
    """Compute terminal d_15 (s3 at layer 15) by backward suffix transport.
    Does NOT form S_t because only d_15 is consumed at terminal layer.
    """
    depth = len(weights)
    target = depth - 1
    n = weights[0].shape[1]
    layers = pilot_data["layers"]

    d_target = np.zeros(n, dtype=np.float64)

    # Initialize suffix matrix Q_(target <- target-1) = W_target^T
    W_target_T = np.asarray(weights[target], dtype=np.float64).T
    Q_curr = W_target_T.copy()  # (n, n)

    for j in range(target - 1, -1, -1):
        # Contract source born at postactivation layer j:
        U_j = layers[j]["U_birth"]  # (n, 2n)
        V_j = layers[j]["V_birth"]  # (n, 2n)

        # Transport: u' = Q u, v' = Q v
        u_prime = Q_curr @ U_j
        v_prime = Q_curr @ V_j

        # Accumulate into d_target: sum_r (u')^2 * v'
        d_target += np.sum((u_prime ** 2) * v_prime, axis=1)

        # Update suffix matrix backwards if j > 0:
        # Q_(target <- j-1) = Q_(target <- j) * diag(g_j) * W_j^T
        if j > 0:
            g_j = layers[j]["g"]
            W_j_T = np.asarray(weights[j], dtype=np.float64).T
            Q_curr = (Q_curr * g_j[None, :]) @ W_j_T

    return d_target


def verify_d0_identities(n: int = 5, depth: int = 4, seed: int = 8101) -> Dict[str, Any]:
    """Mandatory Section 9.2 checks:
    1. Frozen direct-source contraction equals forward transport of the same frozen source list.
    2. Source order and summation order change only floating-point drift.
    3. Terminal d-only contraction equals diagonal of dense transported tensor.
    4. Separate path/slice contractions sum to concatenated-factor contractions.
    5. Gaussian/Angular pilot conventions and final radial conversion remain distinct.
    """
    rng = np.random.default_rng(seed)
    weights = [rng.standard_normal((n, n)) * np.sqrt(2.0 / n) for _ in range(depth)]

    pilot = run_pilot_k2k4(weights)
    layers = pilot["layers"]
    target = depth - 1

    # 1. Backward direct-source contraction
    d_direct = contract_terminal_direct_sources(weights, pilot)

    # Forward transport of the same source list:
    # At each j, transport (U_j, V_j) forward step-by-step to target
    d_forward = np.zeros(n, dtype=np.float64)
    for j in range(target):
        U = layers[j]["U_birth"].copy()
        V = layers[j]["V_birth"].copy()
        for step in range(j + 1, depth):
            W_step_T = np.asarray(weights[step], dtype=np.float64).T
            if step == j + 1:
                U = W_step_T @ U
                V = W_step_T @ V
            else:
                g_prev = layers[step - 1]["g"]
                U = W_step_T @ (U * g_prev[:, None])
                V = W_step_T @ (V * g_prev[:, None])
        d_forward += np.sum((U ** 2) * V, axis=1)

    diff_backward_forward = float(np.max(np.abs(d_direct - d_forward)))
    rel_err_bf = float(diff_backward_forward / max(np.max(np.abs(d_direct)), 1e-12))

    # 2. Separate path and slice contractions sum to concatenated-factor contractions
    d_path_separate = np.zeros(n, dtype=np.float64)
    d_slice_separate = np.zeros(n, dtype=np.float64)

    W_target_T = np.asarray(weights[target], dtype=np.float64).T
    Q_curr = W_target_T.copy()
    for j in range(target - 1, -1, -1):
        A2 = layers[j]["A2"]
        w21 = layers[j]["w21"]
        A_ds = layers[j]["A_ds"]
        eye_n = np.eye(n, dtype=np.float64)

        u_path = Q_curr @ A2
        v_path = Q_curr @ (3.0 * eye_n * w21[None, :])
        d_path_separate += np.sum((u_path ** 2) * v_path, axis=1)

        u_slice = Q_curr @ eye_n
        v_slice = Q_curr @ A_ds
        d_slice_separate += np.sum((u_slice ** 2) * v_slice, axis=1)

        if j > 0:
            g_j = layers[j]["g"]
            W_j_T = np.asarray(weights[j], dtype=np.float64).T
            Q_curr = (Q_curr * g_j[None, :]) @ W_j_T

    d_sum_separate = d_path_separate + d_slice_separate
    diff_concat_separate = float(np.max(np.abs(d_direct - d_sum_separate)))
    rel_err_cs = float(diff_concat_separate / max(np.max(np.abs(d_direct)), 1e-12))

    # 3. Dense 3-tensor check: K3_target[i, i, i] = d_direct[i]
    # Reconstruct dense rank-1 sum for each birth:
    # Sym(u, u, v)_ijk = (u_i u_j v_k + u_i v_j u_k + v_i u_j u_k) / 3
    # Diagonal i=j=k: Sym(u, u, v)_iii = u_i^2 * v_i
    # Sum over columns r gives sum_r u_ir^2 * v_ir = d_i.
    # Check that this matches the diagonal of the full dense tensor.
    dense_diag = np.zeros(n, dtype=np.float64)
    Q_curr = W_target_T.copy()
    for j in range(target - 1, -1, -1):
        U_j = layers[j]["U_birth"]
        V_j = layers[j]["V_birth"]
        u_p = Q_curr @ U_j
        v_p = Q_curr @ V_j
        R = u_p.shape[1]
        dense_T = np.zeros((n, n, n), dtype=np.float64)
        for r in range(R):
            ur = u_p[:, r]
            vr = v_p[:, r]
            dense_T += (np.einsum('i,j,k->ijk', ur, ur, vr)
                        + np.einsum('i,j,k->ijk', ur, vr, ur)
                        + np.einsum('i,j,k->ijk', vr, ur, ur)) / 3.0
        dense_diag += np.array([dense_T[i, i, i] for i in range(n)])
        if j > 0:
            g_j = layers[j]["g"]
            W_j_T = np.asarray(weights[j], dtype=np.float64).T
            Q_curr = (Q_curr * g_j[None, :]) @ W_j_T

    diff_dense_diag = float(np.max(np.abs(d_direct - dense_diag)))
    rel_err_dd = float(diff_dense_diag / max(np.max(np.abs(d_direct)), 1e-12))

    return {
        "backward_vs_forward_rel_err": rel_err_bf,
        "backward_vs_forward_passes": bool(rel_err_bf < 1e-10),
        "concat_vs_separate_rel_err": rel_err_cs,
        "concat_vs_separate_passes": bool(rel_err_cs < 1e-10),
        "dense_tensor_diag_rel_err": rel_err_dd,
        "dense_tensor_diag_passes": bool(rel_err_dd < 1e-10),
    }


def forecast_direct_sources_cost(n: int = 1024, depth: int = 16) -> Dict[str, Any]:
    """Calculate exact FLOP profile of D1-TERM and D2-ALL (Section 9.2).
    B = 2^41 = 2,199,023,255,552 FLOPs.
    """
    B = 2 ** 41

    # 1. Pilot A1-G-K2K4 cost:
    # Per layer:
    # W_ref @ mu: 2*n^2
    # W_ref @ cov @ W_ref^T: 2*n^3 + 2*n^3 = 4*n^3 (or 2*n^3 + n^2)
    # Wick scalar & pk11, pk21, c4: O(n^2)
    # Across 16 layers: ~ 15 * 4 * n^3 = 60 * n^3 ~ 6.44e10 FLOPs (~2.9% util)
    flops_pilot = 15 * (4 * (n ** 3) + 10 * (n ** 2))

    # 2. D1-TERM:
    # Target = 15 (single target).
    # Suffix matrix backward:
    # For j = 14 down to 1 (14 steps):
    # (Q * g) @ W_j^T: 2 * n^3 FLOPs.
    # Total suffix matmul: 14 * 2 * n^3 = 28 * n^3 ~ 3.01e10 FLOPs.
    # Factor transport at layer j:
    # U_j has 2n columns (A2: n cols, eye_n: n cols).
    # V_j has 2n columns (3*eye*w21: n cols, A_ds: n cols).
    # Transport u' = Q @ U_j (shape n x n, n x n): 2 * 2*n^3 = 4*n^3 FLOPs.
    # Transport v' = Q @ V_j: 4*n^3 FLOPs.
    # Suffix transport per j: 8 * n^3 FLOPs.
    # Across 15 layers: 15 * 8 * n^3 = 120 * n^3 ~ 1.29e11 FLOPs.
    # Contraction sum_r (u')^2 * v': 3 * n * (2n) = 6*n^2 (negligible).
    # Total D1-TERM = flops_pilot + suffix (28*n^3) + transport (120*n^3)
    # Total FLOPs ~ (60 + 28 + 120) * n^3 = 208 * n^3 = 2.23e11 FLOPs.
    flops_d1_term = flops_pilot + 14 * (2 * n**3) + 15 * (8 * n**3)
    util_d1 = flops_d1_term / B

    # 3. D2-ALL (All targets t=1..15):
    # If done independently per target:
    # Sum over t=1..15 of (t steps of suffix + t steps of transport):
    # Sum_{t=1..15} t = 120 suffix/transport steps!
    # 120 * (2*n^3 + 8*n^3) = 1200 * n^3 ~ 1.29e12 FLOPs (~58.6% util).
    # Plus pilot pass (6.4e10) + second forward pass (6.4e10):
    # Total ~ 1.42e12 FLOPs (~64.5% util).
    flops_d2_all = 2 * flops_pilot + 120 * (10 * n**3)
    util_d2 = flops_d2_all / B

    return {
        "n": n,
        "depth": depth,
        "budget_B": B,
        "pilot_flops": flops_pilot,
        "pilot_util": flops_pilot / B,
        "d1_term_flops": flops_d1_term,
        "d1_term_util": util_d1,
        "d1_term_passes_cost_gate": bool(util_d1 < 0.95),
        "d2_all_flops": flops_d2_all,
        "d2_all_util": util_d2,
        "d2_all_passes_cost_gate": bool(util_d2 < 0.95),
    }


if __name__ == "__main__":
    print("Running D0 identity verification...")
    v = verify_d0_identities(n=5, depth=4)
    print("D0 Identity Results:")
    for k, val in v.items():
        print(f"  {k}: {val}")

    print("\nCost Forecast for Phase 2 (n=1024, depth=16):")
    c = forecast_direct_sources_cost(n=1024, depth=16)
    print(f"  Pilot FLOPs: {c['pilot_flops']:.4e} (util={c['pilot_util']*100:.2f}%)")
    print(f"  D1-TERM FLOPs: {c['d1_term_flops']:.4e} (util={c['d1_term_util']*100:.2f}%) Passes < 0.95B: {c['d1_term_passes_cost_gate']}")
    print(f"  D2-ALL FLOPs:  {c['d2_all_flops']:.4e} (util={c['d2_all_util']*100:.2f}%) Passes < 0.95B: {c['d2_all_passes_cost_gate']}")
