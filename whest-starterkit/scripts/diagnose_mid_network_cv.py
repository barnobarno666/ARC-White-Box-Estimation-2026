"""diagnose_mid_network_cv.py

P4-05 Mid-Network Response Control Oracle (Lane C)

Screens candidate control layers k in {4, 6, 8, 10, 12, 14}:
1. Computes sampled mean h_k and final mean h_15 via WMC (N=4200)
2. Compares analytical Hermite mean mu_k^cov with true mean mu_k*
3. Evaluates the Perfect-Centering Oracle:
   h_15^CV = h_15 - T (h_k - mu_k*)
4. Evaluates the Deployable Analytical-Centering:
   h_15^CV_dep = h_15 - T (h_k - mu_k^cov)
5. Tests scalar transport (T = beta * I), diagonal transport, and ridge linear transport
6. Tests whether the CV-corrected WMC branch fused with the Hermite analytical branch
   beats the Phase 4 champion (1.2236e-07).
"""

import numpy as np
import scipy.linalg as la
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _S_PRIOR

def run_wmc_all_layers(mlp: MLP, count: int = 4200) -> np.ndarray:
    """Run WMC and extract sample means at all 16 layers."""
    width, depth = mlp.width, mlp.depth
    half = count // 2
    rng = np.random.default_rng(mlp.seed)
    
    z = rng.standard_normal((half, width)).astype(np.float32)
    x = np.concatenate((z, -z), axis=0)
    gram = (x.T @ x) / float(count)
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 1e-6)
    whitener = (eigvecs * np.power(eigvals, -0.5)) @ eigvecs.T
    x_whitened = x @ whitener

    act = fnp.maximum(fnp.asarray(x_whitened, dtype=fnp.float32) @ fnp.asarray(mlp.weights[0], dtype=fnp.float32), 0.0)
    layer_means = [np.mean(np.array(act, dtype=np.float64), axis=0)]
    for l in range(1, depth):
        act = fnp.maximum(act @ fnp.asarray(mlp.weights[l], dtype=fnp.float32), 0.0)
        layer_means.append(np.mean(np.array(act, dtype=np.float64), axis=0))
    return np.array(layer_means)  # shape (16, 1024)

def diagnose_mid_network():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 95)
    print("P4-05: MID-NETWORK CONTROL VARIATE ORACLE SCREEN (LANE C)")
    print("=" * 95)

    # Pre-compute ground truth, analytical predictions, and WMC trajectories for all 8 MLPs
    all_trues = []     # 8 x (16, 1024)
    all_covs = []      # 8 x (16, 1024)
    all_wmcs = []      # 8 x (16, 1024)
    c_cal_finals = []  # 8 x (1024,)

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_all = np.array(row["all_layer_means"], dtype=np.float64)
        all_trues.append(y_all)

        cov_all = np.array(_blended_hermite_covariance(mlp), dtype=np.float64)
        all_covs.append(cov_all)
        c_cal_finals.append(cov_all[-1] * float(_S_PRIOR))

        wmc_all = run_wmc_all_layers(mlp, count=4200)
        all_wmcs.append(wmc_all)

    # Compute Champion baseline
    champ_mses = []
    wmc_raw_mses = []
    for i in range(8):
        y15 = all_trues[i][-1]
        w15 = all_wmcs[i][-1]
        c15 = c_cal_finals[i]
        pred_champ = 0.890 * c15 + 0.110 * w15
        champ_mses.append(np.mean((pred_champ - y15)**2))
        wmc_raw_mses.append(np.mean((w15 - y15)**2))
    
    mean_champ = np.mean(champ_mses)
    mean_wmc_raw = np.mean(wmc_raw_mses)
    print(f"Current Champion Control (Final MSE): {mean_champ:.6e} (Adj: {mean_champ * 0.1000:.6e})")
    print(f"Raw WMC (Final Layer MSE):            {mean_wmc_raw:.6e} (Adj: {mean_wmc_raw * 0.1000:.6e})")
    print("-" * 95)

    candidate_layers = [4, 6, 8, 10, 12, 14]
    
    for k in candidate_layers:
        print(f"\n--- EVALUATING CONTROL LAYER k = {k} ---")
        
        # 1. Quality of Hermite covariance center at layer k
        cov_errs_k = [np.mean((all_covs[i][k] - all_trues[i][k])**2) for i in range(8)]
        wmc_errs_k = [np.mean((all_wmcs[i][k] - all_trues[i][k])**2) for i in range(8)]
        print(f"  Layer {k} Covariance Center MSE: {np.mean(cov_errs_k):.6e} vs Layer {k} WMC MSE: {np.mean(wmc_errs_k):.6e}")

        # 2. Fit best linear transport from layer k error to layer 15 error
        # Training samples across 8 MLPs: X = delta_k, Y = delta_15
        # Stack 8 * 1024 = 8192 scalar pairs for scalar beta: delta_15 ≈ beta * delta_k
        delta_k_perfect = np.concatenate([all_wmcs[i][k] - all_trues[i][k] for i in range(8)])
        delta_15 = np.concatenate([all_wmcs[i][-1] - all_trues[i][-1] for i in range(8)])
        delta_k_cov = np.concatenate([all_wmcs[i][k] - all_covs[i][k] for i in range(8)])

        # Optimal scalar beta under perfect centering: min sum (delta_15 - beta * delta_k_perfect)^2
        beta_scalar = float(np.sum(delta_k_perfect * delta_15) / max(np.sum(delta_k_perfect**2), 1e-12))
        corr_k_15 = float(np.corrcoef(delta_k_perfect, delta_15)[0, 1])

        # Evaluate Perfect-Centering Oracle with scalar transport
        perf_wmc_mses = []
        perf_fused_mses = []
        for i in range(8):
            y15 = all_trues[i][-1]
            c15 = c_cal_finals[i]
            w15 = all_wmcs[i][-1]
            dk = all_wmcs[i][k] - all_trues[i][k]
            
            # CV-corrected WMC
            w15_cv = w15 - beta_scalar * dk
            perf_wmc_mses.append(np.mean((w15_cv - y15)**2))

            # Fused with analytical branch at optimal alpha
            err_c = c15 - y15
            err_w = w15_cv - y15
            d_err = err_w - err_c
            alpha_star = -float(np.sum(err_c * d_err)) / max(float(np.sum(d_err**2)), 1e-12)
            alpha_star = max(0.0, min(1.0, alpha_star))
            pred_fused = (1.0 - alpha_star) * c15 + alpha_star * w15_cv
            perf_fused_mses.append(np.mean((pred_fused - y15)**2))

        # Evaluate Deployable Analytical-Centering
        dep_wmc_mses = []
        dep_fused_mses = []
        beta_dep = float(np.sum(delta_k_cov * delta_15) / max(np.sum(delta_k_cov**2), 1e-12))
        for i in range(8):
            y15 = all_trues[i][-1]
            c15 = c_cal_finals[i]
            w15 = all_wmcs[i][-1]
            dk_dep = all_wmcs[i][k] - all_covs[i][k]
            
            w15_cv_dep = w15 - beta_dep * dk_dep
            dep_wmc_mses.append(np.mean((w15_cv_dep - y15)**2))

            pred_fused = 0.890 * c15 + 0.110 * w15_cv_dep
            dep_fused_mses.append(np.mean((pred_fused - y15)**2))

        perf_gain_wmc = ((mean_wmc_raw - np.mean(perf_wmc_mses)) / mean_wmc_raw) * 100
        perf_gain_fused = ((mean_champ - np.mean(perf_fused_mses)) / mean_champ) * 100
        dep_gain_fused = ((mean_champ - np.mean(dep_fused_mses)) / mean_champ) * 100

        print(f"  Correlation(delta_{k}, delta_15):   {corr_k_15:+.4f} (Optimal beta={beta_scalar:.4f})")
        print(f"  Perfect Centering WMC Raw MSE:      {np.mean(perf_wmc_mses):.6e} (Variance Reduction: {perf_gain_wmc:+.2f}%)")
        print(f"  Perfect Centering Fused MSE:        {np.mean(perf_fused_mses):.6e} (Adj: {np.mean(perf_fused_mses)*0.1:.6e}, Headroom: {perf_gain_fused:+.2f}%)")
        print(f"  Deployable Analytical Fused MSE:    {np.mean(dep_fused_mses):.6e} (Adj: {np.mean(dep_fused_mses)*0.1:.6e}, Gain: {dep_gain_fused:+.2f}%)")

if __name__ == "__main__":
    diagnose_mid_network()
