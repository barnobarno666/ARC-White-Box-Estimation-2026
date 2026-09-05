"""diagnose_recursive_calibration.py

Tests Recursive / Multi-Layer Scale Calibration:
Compare:
1. Terminal calibration only (Current Champion):
   mu_15_cal = s_prior * mu_15
2. Recursive Layer-by-Layer Calibration:
   At each layer l, scale mu and cov by (s_l / s_{l-1}) or a fixed deflation factor per layer.
3. Test a grid of per-layer shrinkage factors rho:
   mu_l <- rho * mu_l, cov_l <- (rho^2) * cov_l
4. Measure final layer MSE before and after blending with WMC.
"""

import math
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _whitened_antithetic_mc, _S_PRIOR

_C_CENTERED = fnp.asarray(0.80 * 0.20 * 0.07957747154594767, dtype=fnp.float32)
_C_THRESH = fnp.asarray(0.20 * 0.5, dtype=fnp.float32)
_ZERO = fnp.asarray(0.0, dtype=fnp.float32)

def propagate_recursive_cov(mlp: MLP, rho_per_layer: float) -> np.ndarray:
    width = mlp.width
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
    rho_fnp = fnp.asarray(rho_per_layer, dtype=fnp.float32)
    rho_sq_fnp = fnp.asarray(rho_per_layer * rho_per_layer, dtype=fnp.float32)
    rows = []

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

        cov_linear = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        cov_pre_sq = cov_pre * cov_pre
        inv_sigma = fnp.where(sigma_pre > 1e-12, 1.0 / sigma_pre, _ZERO)
        u_thresh = fnp.where(sigma_pre > 1e-12, phi / sigma_pre, _ZERO)

        kernel_quad = _C_CENTERED * fnp.outer(inv_sigma, inv_sigma) + _C_THRESH * fnp.outer(u_thresh, u_thresh)
        cov_quad = fnp.multiply(kernel_quad, cov_pre_sq)

        cov = cov_linear + cov_quad
        fnp.fill_diagonal(cov, var_post)

        # Apply recursive scale calibration
        if rho_per_layer != 1.0:
            mu = mu * rho_fnp
            cov = cov * rho_sq_fnp

        cov = flops.as_symmetric(cov, symmetry=(0, 1))
        rows.append(mu)

    return np.array(rows[-1], dtype=np.float64)

def run_recursive_experiment():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("RECURSIVE LAYER-BY-LAYER SCALE CALIBRATION EXPERIMENT")
    print("=" * 90)

    # Pre-extract ground truth and WMC
    y_trues = []
    wmc_preds = []
    mlps = []
    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        mlps.append(mlp)
        y_trues.append(np.array(row["final_means"], dtype=np.float64))
        wmc_preds.append(np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64))

    # Control: Terminal scale s_0 = 0.998319
    ctrl_cov_raw = [propagate_recursive_cov(mlps[i], 1.0) for i in range(8)]
    ctrl_cov_cal = [ctrl_cov_raw[i] * float(_S_PRIOR) for i in range(8)]
    ctrl_champ = [0.890 * ctrl_cov_cal[i] + 0.110 * wmc_preds[i] for i in range(8)]
    ctrl_mse = np.mean([np.mean((ctrl_champ[i] - y_trues[i])**2) for i in range(8)])
    print(f"Current Control Fused MSE (Terminal s_0=0.998319): {ctrl_mse:.6e} (Adj: {ctrl_mse*0.1:.6e})")
    print("-" * 90)

    # Sweep rho_per_layer around (0.998319)^(1/16) ≈ 0.9998948
    rho_base = 0.998319 ** (1.0 / 16.0)
    print(f"Theoretical uniform per-layer rho = (0.998319)^(1/16) = {rho_base:.7f}")

    rhos = [
        1.0,
        0.99995,
        0.99992,
        0.99990,
        rho_base,
        0.99988,
        0.99985,
        0.99980,
    ]

    for rho in rhos:
        c_preds = [propagate_recursive_cov(mlps[i], rho) for i in range(8)]
        
        # Test 1: Raw covariance MSE
        raw_cov_mses = [np.mean((c_preds[i] - y_trues[i])**2) for i in range(8)]
        
        # Test 2: Optimal terminal scale on top of recursive rho
        sy_list = [float(np.sum(y_trues[i] * c_preds[i]) / np.sum(c_preds[i] * c_preds[i])) for i in range(8)]
        s_opt = np.mean(sy_list)
        scaled_cov_mses = [np.mean((s_opt * c_preds[i] - y_trues[i])**2) for i in range(8)]

        # Test 3: Fused with WMC at alpha=0.110
        fused_mses = [np.mean((0.890 * (s_opt * c_preds[i]) + 0.110 * wmc_preds[i] - y_trues[i])**2) for i in range(8)]
        mean_fused = np.mean(fused_mses)
        diff = ((mean_fused - ctrl_mse) / ctrl_mse) * 100

        print(f"rho={rho:.7f}: Cov Raw={np.mean(raw_cov_mses):.6e} | Cov Scaled={np.mean(scaled_cov_mses):.6e} | Fused={mean_fused:.6e} (Diff: {diff:+.2f}%)")

if __name__ == "__main__":
    run_recursive_experiment()
