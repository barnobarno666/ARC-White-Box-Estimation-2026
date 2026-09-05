"""diagnose_cubic_quartic_hermite.py

Lane A1: High-Order Hermite Series (Cubic rho^3 & Quartic rho^4):
Price's theorem gives the exact Taylor-Hermite series expansion of bivariate ReLU:
E[ReLU(u) ReLU(v)] = sum_{k=0}^infinity C_k rho^k
k=1: Linear term: Phi(a_i) Phi(a_j) rho
k=2: Quadratic term: 1/2 phi(a_i) phi(a_j) rho^2
k=3: Cubic term: 1/6 a_i a_j phi(a_i) phi(a_j) rho^3
k=4: Quartic term: 1/24 (a_i^2 - 1)(a_j^2 - 1) phi(a_i) phi(a_j) rho^4

Tests whether adding the cubic and quartic terms reduces the analytical covariance MSE.
"""

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _whitened_antithetic_mc, _S_PRIOR

_C_CENTERED = fnp.asarray(0.80 * 0.20 * 0.07957747154594767, dtype=fnp.float32)
_C_THRESH = fnp.asarray(0.20 * 0.5, dtype=fnp.float32)
_ZERO = fnp.asarray(0.0, dtype=fnp.float32)
_ONE_SIXTH = fnp.asarray(1.0 / 6.0, dtype=fnp.float32)
_ONE_24TH = fnp.asarray(1.0 / 24.0, dtype=fnp.float32)

def compute_higher_order_hermite(mlp: MLP, eta3: float = 0.0, eta4: float = 0.0) -> np.ndarray:
    width = mlp.width
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
    eta3_fnp = fnp.asarray(eta3, dtype=fnp.float32)
    eta4_fnp = fnp.asarray(eta4, dtype=fnp.float32)

    for weight in mlp.weights:
        w = fnp.asarray(weight, dtype=fnp.float32)
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), fnp.asarray(1e-12, dtype=fnp.float32))
        sigma_pre = fnp.sqrt(var_pre)
        a = mu_pre / sigma_pre
        phi = flops.stats.norm.pdf(a).astype(fnp.float32)
        cdf = flops.stats.norm.cdf(a).astype(fnp.float32)

        mu = mu_pre * cdf + sigma_pre * phi
        second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
        var_post = fnp.maximum(second - mu * mu, _ZERO)
        gain = fnp.where(sigma_pre > 1e-12, cdf, _ZERO)

        # 1. Base linear
        cov_linear = fnp.multiply(fnp.outer(gain, gain), cov_pre)

        # 2. Dual-kernel blended quadratic
        cov_pre_sq = cov_pre * cov_pre
        inv_sigma = fnp.where(sigma_pre > 1e-12, 1.0 / sigma_pre, _ZERO)
        u_thresh = fnp.where(sigma_pre > 1e-12, phi / sigma_pre, _ZERO)
        kernel_quad = _C_CENTERED * fnp.outer(inv_sigma, inv_sigma) + _C_THRESH * fnp.outer(u_thresh, u_thresh)
        cov_quad = fnp.multiply(kernel_quad, cov_pre_sq)

        cov_total = cov_linear + cov_quad

        # 3. Cubic term
        if eta3 != 0.0:
            u_cube = fnp.where(sigma_pre > 1e-12, (a * phi) / (sigma_pre * sigma_pre), _ZERO)
            cov_cube = _ONE_SIXTH * fnp.multiply(fnp.outer(u_cube, u_cube), cov_pre_sq * cov_pre)
            cov_total = cov_total + eta3_fnp * cov_cube

        # 4. Quartic term
        if eta4 != 0.0:
            u_quart = fnp.where(sigma_pre > 1e-12, ((a * a - 1.0) * phi) / (sigma_pre * sigma_pre * sigma_pre), _ZERO)
            cov_quart = _ONE_24TH * fnp.multiply(fnp.outer(u_quart, u_quart), cov_pre_sq * cov_pre_sq)
            cov_total = cov_total + eta4_fnp * cov_quart

        fnp.fill_diagonal(cov_total, var_post)
        cov = flops.as_symmetric(cov_total, symmetry=(0, 1))

    return np.array(mu, dtype=np.float64)

def test_higher_order_hermite():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("HIGHER-ORDER HERMITE SERIES (CUBIC & QUARTIC) DIAGNOSTIC")
    print("=" * 90)

    y_trues = []
    wmc_preds = []
    mlps = []
    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        mlps.append(mlp)
        y_trues.append(np.array(row["final_means"], dtype=np.float64))
        wmc_preds.append(np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64))

    # Benchmark: eta3=0, eta4=0
    c_base = [compute_higher_order_hermite(mlps[i], 0.0, 0.0) for i in range(8)]
    base_cal = [c_base[i] * float(_S_PRIOR) for i in range(8)]
    base_fused = [0.890 * base_cal[i] + 0.110 * wmc_preds[i] for i in range(8)]
    base_mse = np.mean([np.mean((base_fused[i] - y_trues[i])**2) for i in range(8)])
    print(f"Current Control Fused MSE (eta3=0, eta4=0): {base_mse:.6e} (Adj: {base_mse*0.1:.6e})")
    print("-" * 90)

    configs = [
        (0.05, 0.0),
        (0.10, 0.0),
        (-0.05, 0.0),
        (-0.10, 0.0),
        (0.0, 0.05),
        (0.0, 0.10),
        (0.0, -0.05),
        (0.05, 0.05),
        (-0.05, -0.05),
    ]

    for eta3, eta4 in configs:
        c_preds = [compute_higher_order_hermite(mlps[i], eta3, eta4) for i in range(8)]
        # Optimal scale for this config
        s_opt = np.mean([float(np.sum(y_trues[i] * c_preds[i]) / np.sum(c_preds[i] * c_preds[i])) for i in range(8)])
        
        cov_mses = [np.mean((s_opt * c_preds[i] - y_trues[i])**2) for i in range(8)]
        fused_mses = [np.mean((0.890 * (s_opt * c_preds[i]) + 0.110 * wmc_preds[i] - y_trues[i])**2) for i in range(8)]
        mean_fused = np.mean(fused_mses)
        diff = ((mean_fused - base_mse) / base_mse) * 100

        print(f"eta3={eta3:+.2f}, eta4={eta4:+.2f} (s_opt={s_opt:.6f}): Cov MSE={np.mean(cov_mses):.6e} | Fused={mean_fused:.6e} (Diff: {diff:+.2f}%)")

if __name__ == "__main__":
    test_higher_order_hermite()
