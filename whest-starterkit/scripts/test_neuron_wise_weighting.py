"""test_neuron_wise_weighting.py

Tests Neuron-Wise Precision Weighting (Heteroscedastic Gauss-Markov Filter):
Instead of a single scalar alpha = 0.110 across all 1024 neurons:
alpha_i = tau_i^2 / (tau_i^2 + sigma_i^2 / N)

Where:
- sigma_i^2 is the marginal variance of neuron i from covariance propagation
- tau_i^2 is the model error variance of the covariance prediction
- We test:
  1. tau_i^2 = tau_0^2 (constant model error)
  2. tau_i^2 = tau_0^2 * sigma_i^2 (proportional model error)
  3. tau_i^2 = tau_0^2 * (1 + beta * a_i^2) (curvature-dependent model error)
"""

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _whitened_antithetic_mc, _S_PRIOR

def test_neuron_weighting():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("NEURON-WISE HETEROSCEDASTIC PRECISION WEIGHTING EXPERIMENT")
    print("=" * 90)

    y_trues = []
    c_finals = []
    wmc_finals = []
    var_finals = []
    a15_list = []

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = np.array(row["final_means"], dtype=np.float64)
        y_trues.append(y)

        # Get c and marginal variances
        width = mlp.width
        mu = fnp.zeros(width, dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
        zero = fnp.asarray(0.0, dtype=fnp.float32)

        for l in range(16):
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

            if l == 15:
                var_finals.append(np.array(var_post, dtype=np.float64))
                a15_list.append(np.array(a, dtype=np.float64))

        c = np.array(mu, dtype=np.float64) * float(_S_PRIOR)
        c_finals.append(c)

        wmc = np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64)
        wmc_finals.append(wmc)

    # Baseline scalar blend: alpha = 0.110
    base_mses = []
    for i in range(8):
        pred_base = 0.890 * c_finals[i] + 0.110 * wmc_finals[i]
        base_mses.append(np.mean((pred_base - y_trues[i])**2))
    print(f"Current Control Benchmark (Scalar alpha=0.110): MSE = {np.mean(base_mses):.6e} (Adj: {np.mean(base_mses)*0.1:.6e})")
    print("-" * 90)

    # Model 1: Heteroscedastic Kalman gain
    # MC sample variance: V_mc = sigma_i^2 / N
    # Covariance error variance: V_cov = tau_0
    # alpha_i = V_cov / (V_cov + sigma_i^2 / N) = 1 / (1 + (sigma_i^2 / N) / tau_0)
    # Let theta = tau_0 * N
    # alpha_i = theta / (theta + sigma_i^2)
    print("Testing Model 1: alpha_i = theta / (theta + var_i)")
    for theta in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        mses = []
        for i in range(8):
            var_i = var_finals[i]
            alpha_vec = theta / (theta + var_i)
            # Clip alpha to reasonable range [0.01, 0.50]
            alpha_vec = np.clip(alpha_vec, 0.01, 0.50)
            pred = (1.0 - alpha_vec) * c_finals[i] + alpha_vec * wmc_finals[i]
            mses.append(np.mean((pred - y_trues[i])**2))
        mean_m = np.mean(mses)
        diff = ((mean_m - np.mean(base_mses)) / np.mean(base_mses)) * 100
        print(f"  theta={theta:.2f}: MSE = {mean_m:.6e} (Diff: {diff:+.2f}%)")

    # Model 2: Variance-normalized + Curvature weight
    print("\nTesting Model 2: alpha_i = alpha_0 * (var_i / mean(var_i))^gamma")
    for alpha_0 in [0.09, 0.11, 0.13]:
        for gamma in [-0.5, -0.25, 0.25, 0.5, 1.0]:
            mses = []
            for i in range(8):
                var_i = var_finals[i]
                v_norm = var_i / np.mean(var_i)
                alpha_vec = alpha_0 * (v_norm ** gamma)
                alpha_vec = np.clip(alpha_vec, 0.01, 0.50)
                pred = (1.0 - alpha_vec) * c_finals[i] + alpha_vec * wmc_finals[i]
                mses.append(np.mean((pred - y_trues[i])**2))
            mean_m = np.mean(mses)
            diff = ((mean_m - np.mean(base_mses)) / np.mean(base_mses)) * 100
            if diff < 0.0:
                print(f"  alpha_0={alpha_0:.3f}, gamma={gamma:+.2f}: MSE = {mean_m:.6e} (Diff: {diff:+.2f}%)")

if __name__ == "__main__":
    test_neuron_weighting()
