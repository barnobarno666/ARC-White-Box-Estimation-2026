"""diagnose_regime_calibration.py

Regime-Dependent Calibration & Blending (On vs Kink vs Dead):
Tests whether separating neurons into regimes based on a_i = mu_i / sigma_i:
1. Dead (a_i < -2.5): Set to 0.0 or low-threshold damping
2. On   (a_i > +2.5): Optimal scale s_on and blend weight alpha_on
3. Kink (-2.5 <= a_i <= +2.5): Optimal scale s_kink and blend weight alpha_kink

Measures final MSE across the 8 MLPs!
"""

import numpy as np
import scipy.optimize as opt
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _whitened_antithetic_mc, _S_PRIOR

def diagnose_regimes():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("REGIME-DEPENDENT CALIBRATION & BLENDING (ON vs KINK vs DEAD)")
    print("=" * 90)

    y_trues = []
    c_finals = []
    wmc_finals = []
    a15_list = []

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = np.array(row["final_means"], dtype=np.float64)
        y_trues.append(y)

        c = np.array(_blended_hermite_covariance(mlp)[-1], dtype=np.float64)
        c_finals.append(c)

        wmc = np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64)
        wmc_finals.append(wmc)

        # Estimate a15 from covariance propagation
        # Approximate sigma_15 from uncalibrated c
        # (c_i is approximately mu_i * cdf + sigma * phi)
        # Using empirical ratio or forward pass
        # Let's get exact a15 from covariance
        w15 = np.array(mlp.weights[-1], dtype=np.float64)
        # c[-1] has the layer 15 mean
        # Let's use c as a proxy or compute exact a15
        width = mlp.width
        # Run cheap forward of cov up to 14
        mu = fnp.zeros(width, dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
        zero = fnp.asarray(0.0, dtype=fnp.float32)
        for l in range(15):
            w = fnp.asarray(mlp.weights[l], dtype=fnp.float32)
            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
            var_pre = fnp.maximum(fnp.diag(cov_pre), fnp.asarray(1e-12, dtype=fnp.float32))
            sigma_pre = fnp.sqrt(var_pre)
            a = mu_pre / sigma_pre
            phi = flops.stats.norm.pdf(a).astype(fnp.float32)
            cdf = flops.stats.norm.cdf(a).astype(fnp.float32)
            mu = mu_pre * cdf + sigma_pre * phi
            second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
            var_post = fnp.maximum(second - mu * mu, zero)
            gain = fnp.where(sigma_pre > 1e-12, cdf, zero)
            cov_linear = fnp.multiply(fnp.outer(gain, gain), cov_pre)
            inv_sigma = fnp.where(sigma_pre > 1e-12, 1.0 / sigma_pre, zero)
            u_thresh = fnp.where(sigma_pre > 1e-12, phi / sigma_pre, zero)
            kernel_quad = fnp.asarray(0.80 * 0.20 * 0.07957747, dtype=fnp.float32) * fnp.outer(inv_sigma, inv_sigma) + fnp.asarray(0.10, dtype=fnp.float32) * fnp.outer(u_thresh, u_thresh)
            cov = cov_linear + fnp.multiply(kernel_quad, cov_pre * cov_pre)
            fnp.fill_diagonal(cov, var_post)
            cov = flops.as_symmetric(cov, symmetry=(0, 1))

        w15_fnp = fnp.asarray(mlp.weights[15], dtype=fnp.float32)
        mu_pre15 = np.array(w15_fnp.T @ mu, dtype=np.float64)
        cov_pre15 = np.array(fnp.einsum("ij,ia,jb->ab", cov, w15_fnp, w15_fnp), dtype=np.float64)
        sigma_pre15 = np.sqrt(np.maximum(np.diag(cov_pre15), 1e-12))
        a15 = mu_pre15 / sigma_pre15
        a15_list.append(a15)

    # 1. Baseline Benchmark MSE (Global s_0 = 0.998319, alpha = 0.110)
    base_mses = []
    for i in range(8):
        pred_base = 0.890 * (float(_S_PRIOR) * c_finals[i]) + 0.110 * wmc_finals[i]
        base_mses.append(np.mean((pred_base - y_trues[i])**2))
    print(f"Current Control Benchmark MSE: {np.mean(base_mses):.6e} (Adj: {np.mean(base_mses)*0.1:.6e})")
    print("-" * 90)

    # 2. Analyze Optimal Scale and Alpha by Regime
    # Regimes: Dead (a < -2.5), Kink (-2.5 <= a <= 2.5), On (a > 2.5)
    for thresh in [2.0, 2.5, 3.0]:
        print(f"\n--- THRESHOLD a_thresh = {thresh:.1f} ---")
        s_on_list, s_kink_list = [], []
        for i in range(8):
            a = a15_list[i]
            y = y_trues[i]
            c = c_finals[i]
            on = a > thresh
            kink = (a >= -thresh) & (a <= thresh)
            
            s_on = float(np.sum(y[on] * c[on]) / np.sum(c[on] * c[on]))
            s_kink = float(np.sum(y[kink] * c[kink]) / np.sum(c[kink] * c[kink]))
            s_on_list.append(s_on)
            s_kink_list.append(s_kink)

        print(f"  Mean s_on:   {np.mean(s_on_list):.6f} +/- {np.std(s_on_list):.5f}")
        print(f"  Mean s_kink: {np.mean(s_kink_list):.6f} +/- {np.std(s_kink_list):.5f}")
        print(f"  Global s_0:  {float(_S_PRIOR):.6f}")

        # Test Regime-Calibrated Predictions across MLPs:
        s_on_mean = np.mean(s_on_list)
        s_kink_mean = np.mean(s_kink_list)

        # Sweep alpha_on and alpha_kink
        for alpha_on in [0.05, 0.110, 0.15]:
            for alpha_kink in [0.110, 0.15, 0.20]:
                test_mses = []
                for i in range(8):
                    a = a15_list[i]
                    y = y_trues[i]
                    c = c_finals[i]
                    w = wmc_finals[i]
                    
                    on = a > thresh
                    dead = a < -thresh
                    kink = ~on & ~dead

                    pred = np.zeros_like(y)
                    # On regime:
                    pred[on] = (1.0 - alpha_on) * (s_on_mean * c[on]) + alpha_on * w[on]
                    # Kink regime:
                    pred[kink] = (1.0 - alpha_kink) * (s_kink_mean * c[kink]) + alpha_kink * w[kink]
                    # Dead regime: c[dead] or zero or blended:
                    pred[dead] = 0.890 * (s_kink_mean * c[dead]) + 0.110 * w[dead]

                    test_mses.append(np.mean((pred - y)**2))

                mean_test = np.mean(test_mses)
                diff = ((mean_test - np.mean(base_mses)) / np.mean(base_mses)) * 100
                if diff < -0.1:
                    print(f"  * alpha_on={alpha_on:.3f}, alpha_kink={alpha_kink:.3f}: MSE = {mean_test:.6e} (Diff: {diff:+.2f}%)")

if __name__ == "__main__":
    diagnose_regimes()
