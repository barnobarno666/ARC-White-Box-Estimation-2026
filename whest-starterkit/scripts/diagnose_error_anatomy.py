import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _whitened_antithetic_mc, _S_PRIOR

def analyze_error_anatomy():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("ANATOMY OF PREDICTION ERROR ACROSS 8 MLPS")
    print("=" * 90)

    s0 = float(_S_PRIOR)
    n_mlps = len(ds)
    
    c0_errors = []
    m_errors = []
    cross_covs = []
    champ_errors = []

    c0_mses = []
    m_mses = []
    champ_mses = []
    optimal_fused_mses = []

    for i in range(n_mlps):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_true = np.array(row["all_layer_means"][-1], dtype=np.float64)

        c = _blended_hermite_covariance(mlp)
        c15 = np.array(c[-1], dtype=np.float64)
        c0 = s0 * c15

        m = _whitened_antithetic_mc(mlp)
        m15 = np.array(m[-1], dtype=np.float64)

        pred_champ = 0.89 * c0 + 0.11 * m15

        e_c = c0 - y_true
        e_m = m15 - y_true
        e_champ = pred_champ - y_true

        mse_c = float(np.mean(e_c**2))
        mse_m = float(np.mean(e_m**2))
        mse_champ = float(np.mean(e_champ**2))

        # Optimal alpha for this MLP:
        # e(alpha) = (1 - alpha)*e_c + alpha*e_m = e_c + alpha*(e_m - e_c)
        # alpha* = - <e_c, e_m - e_c> / ||e_m - e_c||^2
        d = e_m - e_c
        alpha_opt = -float(np.dot(e_c, d)) / float(np.dot(d, d))
        pred_opt = (1.0 - alpha_opt) * c0 + alpha_opt * m15
        mse_opt = float(np.mean((pred_opt - y_true)**2))

        cov_cm = float(np.mean(e_c * e_m))
        corr_cm = cov_cm / np.sqrt(mse_c * mse_m)

        c0_mses.append(mse_c)
        m_mses.append(mse_m)
        champ_mses.append(mse_champ)
        optimal_fused_mses.append(mse_opt)

        print(f"MLP {i} ({row['mlp_name']}):")
        print(f"  MSE c0 (Analytic):  {mse_c:.6e} (Adj: {mse_c * 0.1:.6e})")
        print(f"  MSE m  (WMC 4200):  {mse_m:.6e} (Adj: {mse_m * 0.1:.6e})")
        print(f"  Error Correlation:  {corr_cm:+.4f} (Cov: {cov_cm:+.6e})")
        print(f"  Champ (alpha=0.11): {mse_champ:.6e} (Adj: {mse_champ * 0.1:.6e})")
        print(f"  Oracle Alpha*:      {alpha_opt:.4f} -> Best Possible Linear Blend: {mse_opt:.6e} (Adj: {mse_opt * 0.1:.6e})")

    print("-" * 90)
    print(f"MEAN MSE c0 (Analytic):  {np.mean(c0_mses):.6e} (Adj: {np.mean(c0_mses)*0.1:.6e})")
    print(f"MEAN MSE m  (WMC 4200):  {np.mean(m_mses):.6e} (Adj: {np.mean(m_mses)*0.1:.6e})")
    print(f"MEAN Champ (alpha=0.11): {np.mean(champ_mses):.6e} (Adj: {np.mean(champ_mses)*0.1:.6e})")
    print(f"MEAN Oracle Linear Blend:{np.mean(optimal_fused_mses):.6e} (Adj: {np.mean(optimal_fused_mses)*0.1:.6e})")

if __name__ == "__main__":
    analyze_error_anatomy()
