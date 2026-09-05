"""diagnose_endpoint_oracles.py

P4-02 & P4-03: Multi-Basis Endpoint Bank & Mandatory Oracle Ladder (Lane B)

Builds multi-basis orthogonal endpoint banks (K=4, K=8) and runs the 7-level
oracle ladder on the official 8-MLP panel to measure the upper bound of
target-side contraction and basis reweighting.
"""

import math
import numpy as np
import scipy.optimize as opt
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _whitened_antithetic_mc, _S_PRIOR

_MU_R = math.sqrt(2.0) * math.exp(math.lgamma(1025.0 / 2.0) - math.lgamma(512.0))

def generate_sylvester_hadamard(n: int = 1024) -> np.ndarray:
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h * (1.0 / math.sqrt(n))

_H1024 = generate_sylvester_hadamard(1024)

def forward_carrier_basis(mlp: MLP, q_matrix: np.ndarray) -> np.ndarray:
    """Forward one antipodal orthogonal basis [mu_R * Q, -mu_R * Q] (2048 pts) through MLP."""
    pts_half = (_MU_R * q_matrix).astype(np.float32)
    pts = np.concatenate((pts_half, -pts_half), axis=0)
    
    x = fnp.asarray(pts, dtype=fnp.float32)
    zero = fnp.asarray(0.0, dtype=fnp.float32)
    for w in mlp.weights:
        w_fnp = fnp.asarray(w, dtype=fnp.float32)
        x = fnp.maximum(x @ w_fnp, zero)
    final_act = np.array(x, dtype=np.float64)
    return np.mean(final_act, axis=0)

def generate_orthogonal_bases(n: int, K: int, seed: int) -> list[np.ndarray]:
    """Generate K distinct orthogonal bases using randomized signs and permutations of H1024."""
    rng = np.random.default_rng(seed)
    bases = []
    for _ in range(K):
        d = rng.choice([-1.0, 1.0], size=(1, n)).astype(np.float32)
        p = rng.permutation(n)
        q = (_H1024[p, :] * d).astype(np.float32)
        bases.append(q)
    return bases

def solve_constrained_ridge(A: np.ndarray, b: np.ndarray, gamma: float = 1e-4) -> np.ndarray:
    """Solve min_w ||A w - b||^2 + gamma ||w - w0||^2 s.t. sum(w) = 1, where w0 is uniform."""
    K = A.shape[1]
    w0 = np.ones(K) / float(K)
    # Lagrangian: 2 A^T (A w - b) + 2 gamma (w - w0) + lambda * 1 = 0
    # (A^T A + gamma I) w + 0.5 lambda 1 = A^T b + gamma w0
    G = A.T @ A + gamma * np.eye(K)
    rhs = A.T @ b + gamma * w0
    # Solve G w + lambda_half * 1 = rhs
    # w = G^{-1} rhs - lambda_half G^{-1} 1
    # 1^T w = 1 => lambda_half = (1^T G^{-1} rhs - 1) / (1^T G^{-1} 1)
    G_inv = np.linalg.pinv(G)
    inv_1 = G_inv @ np.ones(K)
    inv_rhs = G_inv @ rhs
    lambda_half = (np.sum(inv_rhs) - 1.0) / max(np.sum(inv_1), 1e-12)
    w = inv_rhs - lambda_half * inv_1
    return w

def run_oracle_ladder(K: int = 8):
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print(f"P4-02 & P4-03: MULTI-BASIS ENDPOINT BANK (K={K} Bases, N={K*2048}) & ORACLE LADDER")
    print("=" * 90)

    # 1. Collect predictions
    y_trues = []
    c_analyticals = []
    wmc_preds = []
    bank_preds = []  # list of (1024, K) matrices

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = np.array(row["final_means"], dtype=np.float64)
        y_trues.append(y)

        # Control analytical branch
        c_fnp = _blended_hermite_covariance(mlp)
        c = np.array(c_fnp[-1], dtype=np.float64) * float(_S_PRIOR)
        c_analyticals.append(c)

        # WMC branch (N=4200)
        wmc = np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64)
        wmc_preds.append(wmc)

        # Generate K bases for this MLP (seed = mlp.seed + 999)
        bases = generate_orthogonal_bases(mlp.width, K, mlp.seed + 999)
        m_k_list = [forward_carrier_basis(mlp, q) for q in bases]
        M_i = np.column_stack(m_k_list)  # (1024, K)
        bank_preds.append(M_i)

    # Calculate Uniform Bank MSE
    uniform_mses = []
    for i in range(8):
        m_uni = np.mean(bank_preds[i], axis=1)
        uniform_mses.append(np.mean((m_uni - y_trues[i])**2))
    mean_uniform_mse = np.mean(uniform_mses)

    # Calculate Champion Control MSE
    champ_mses = []
    for i in range(8):
        pred_champ = 0.890 * c_analyticals[i] + 0.110 * wmc_preds[i]
        champ_mses.append(np.mean((pred_champ - y_trues[i])**2))
    mean_champ_mse = np.mean(champ_mses)

    print(f"Baseline Uniform Bank (K={K}) MSE: {mean_uniform_mse:.6e} (Adj: {mean_uniform_mse * 0.1000:.6e})")
    print(f"Current Champion Control MSE:     {mean_champ_mse:.6e} (Adj: {mean_champ_mse * 0.1000:.6e})")
    print("-" * 90)

    # -------------------------------------------------------------------------
    # Oracle 1: Best Global Fixed Weights (shared across all 8 MLPs)
    # -------------------------------------------------------------------------
    # Stack all 8 MLPs: A_all = (8*1024, K), b_all = (8*1024,)
    A_all = np.vstack(bank_preds)
    b_all = np.concatenate(y_trues)
    w_global = solve_constrained_ridge(A_all, b_all, gamma=1e-8)
    
    o1_mses = []
    for i in range(8):
        pred = bank_preds[i] @ w_global
        o1_mses.append(np.mean((pred - y_trues[i])**2))
    mean_o1 = np.mean(o1_mses)

    # -------------------------------------------------------------------------
    # Oracle 2: Leave-One-MLP-Out (LOO) Shared Weights
    # -------------------------------------------------------------------------
    o2_mses = []
    for i in range(8):
        A_train = np.vstack([bank_preds[j] for j in range(8) if j != i])
        b_train = np.concatenate([y_trues[j] for j in range(8) if j != i])
        w_loo = solve_constrained_ridge(A_train, b_train, gamma=1e-8)
        pred = bank_preds[i] @ w_loo
        o2_mses.append(np.mean((pred - y_trues[i])**2))
    mean_o2 = np.mean(o2_mses)

    # -------------------------------------------------------------------------
    # Oracle 3: Per-Network Non-Negative Mass-One Oracle Weights
    # -------------------------------------------------------------------------
    o3_mses = []
    for i in range(8):
        M = bank_preds[i]
        y = y_trues[i]
        # min ||M w - y||^2 s.t. w >= 0, sum(w) = 1
        res = opt.minimize(
            lambda w: np.sum((M @ w - y)**2),
            x0=np.ones(K)/K,
            bounds=[(0, 1) for _ in range(K)],
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        )
        pred = M @ res.x
        o3_mses.append(np.mean((pred - y)**2))
    mean_o3 = np.mean(o3_mses)

    # -------------------------------------------------------------------------
    # Oracle 4: Per-Network Signed Mass-One Ridge Weights (Ceiling)
    # -------------------------------------------------------------------------
    for gamma_val in [1e-6, 1e-4, 1e-2, 1e-1]:
        o4_mses = []
        for i in range(8):
            w_signed = solve_constrained_ridge(bank_preds[i], y_trues[i], gamma=gamma_val)
            pred = bank_preds[i] @ w_signed
            o4_mses.append(np.mean((pred - y_trues[i])**2))
        print(f"Oracle 4 (Signed Ridge gamma={gamma_val:1e}): MSE = {np.mean(o4_mses):.6e} (Adj: {np.mean(o4_mses)*0.1:.6e})")

    # -------------------------------------------------------------------------
    # Oracle 7: Joint Blend (Bank + Hermite Analytic C + WMC)
    # -------------------------------------------------------------------------
    # Blend: y_hat = a * C + b * WMC + c * Bank_uniform
    # Fit globally and per-network
    o7_global_mses = []
    o7_per_mlp_mses = []
    for i in range(8):
        C = c_analyticals[i]
        W = wmc_preds[i]
        B = np.mean(bank_preds[i], axis=1)
        y = y_trues[i]
        
        # 3-feature regression with sum=1
        X_3 = np.column_stack([C, W, B])
        w_3 = solve_constrained_ridge(X_3, y, gamma=1e-8)
        pred_3 = X_3 @ w_3
        o7_per_mlp_mses.append(np.mean((pred_3 - y)**2))

    print("-" * 90)
    print("MANDATORY ORACLE LADDER RESULTS SUMMARY:")
    print(f"  Uniform Bank (K={K}):          {mean_uniform_mse:.6e} (Raw) | {mean_uniform_mse*0.1:.6e} (Adj)")
    print(f"  Oracle 1 (Global Fixed W):     {mean_o1:.6e} (Raw) | {mean_o1*0.1:.6e} (Adj)")
    print(f"  Oracle 2 (LOO Shared W):       {mean_o2:.6e} (Raw) | {mean_o2*0.1:.6e} (Adj)")
    print(f"  Oracle 3 (Non-Negative W):     {mean_o3:.6e} (Raw) | {mean_o3*0.1:.6e} (Adj)")
    print(f"  Oracle 7 (Joint Blend Tri):    {np.mean(o7_per_mlp_mses):.6e} (Raw) | {np.mean(o7_per_mlp_mses)*0.1:.6e} (Adj)")
    print(f"  Current Champion Benchmark:    {mean_champ_mse:.6e} (Raw) | {mean_champ_mse*0.1:.6e} (Adj)")
    print("=" * 90)

if __name__ == "__main__":
    run_oracle_ladder(K=4)
    run_oracle_ladder(K=8)
