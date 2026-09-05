import math
import numpy as np
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _hermite_gain_covariance, _S_PRIOR
from candidates.estimator_lane_a1_thresh_quad_eta1 import _threshold_aware_hermite_covariance

def test_sampling_carriers():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("LANE C DIAGNOSTIC: SAMPLING CARRIERS & VARIANCE REDUCTION (4200 samples)")
    print("=" * 90)

    d = 1024
    count = 4200
    half = count // 2

    # Exact E[R] for chi_1024 using math.lgamma
    log_er = 0.5 * math.log(2.0) + math.lgamma((d + 1) / 2.0) - math.lgamma(d / 2.0)
    exact_E_R = math.exp(log_er)
    print(f"Exact E[R] for d=1024: {exact_E_R:.8f} (vs sqrt(1024) = 32.0)")

    mse_std_wmc_list = []
    mse_spherical_wmc_list = []
    mse_fused_std_list = []
    mse_fused_spherical_list = []

    s0 = float(_S_PRIOR)

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_true = np.array(row["final_means"], dtype=np.float64)

        rng = np.random.default_rng(mlp.seed)
        x_half = rng.standard_normal((half, d)).astype(np.float32)
        x = np.concatenate((x_half, -x_half), axis=0)

        # 1. Standard Whitened Antithetic MC
        gram = (x.T @ x) / float(count)
        eigvals, eigvecs = np.linalg.eigh(gram)
        eigvals = np.maximum(eigvals, 1e-6)
        whitener = (eigvecs * (eigvals ** -0.5)) @ eigvecs.T

        w0 = np.array(mlp.weights[0], dtype=np.float32)
        x_whitened = x @ whitener

        # Forward pass for standard WMC
        act = np.maximum(x_whitened @ w0, 0.0)
        for layer in range(1, mlp.depth):
            w = np.array(mlp.weights[layer], dtype=np.float32)
            act = np.maximum(act @ w, 0.0)
        m_std = np.mean(act, axis=0)

        # 2. Spherical normalization: normalize each row to exact_E_R
        norms = np.linalg.norm(x_whitened, axis=1, keepdims=True)
        x_spherical = (x_whitened / norms) * exact_E_R

        act_sph = np.maximum(x_spherical @ w0, 0.0)
        for layer in range(1, mlp.depth):
            w = np.array(mlp.weights[layer], dtype=np.float32)
            act_sph = np.maximum(act_sph @ w, 0.0)
        m_spherical = np.mean(act_sph, axis=0)

        # Covariance prediction
        c = np.array(_threshold_aware_hermite_covariance(mlp)[-1], dtype=np.float64)
        c_calib = s0 * c

        # Errors
        mse_std = float(np.mean((m_std - y_true) ** 2))
        mse_sph = float(np.mean((m_spherical - y_true) ** 2))

        fused_std = 0.89 * c_calib + 0.11 * m_std
        fused_sph = 0.89 * c_calib + 0.11 * m_spherical

        mse_fused_std = float(np.mean((fused_std - y_true) ** 2))
        mse_fused_sph = float(np.mean((fused_sph - y_true) ** 2))

        mse_std_wmc_list.append(mse_std)
        mse_spherical_wmc_list.append(mse_sph)
        mse_fused_std_list.append(mse_fused_std)
        mse_fused_spherical_list.append(mse_fused_sph)

        diff_pct = ((mse_fused_sph - mse_fused_std) / mse_fused_std) * 100.0
        print(f"{row['mlp_name']:<22} | WMC: {mse_std:.4e} -> Sph: {mse_sph:.4e} | Fused: {mse_fused_std:.6e} -> {mse_fused_sph:.6e} ({diff_pct:+.2f}%)")

    mean_std = np.mean(mse_std_wmc_list)
    mean_sph = np.mean(mse_spherical_wmc_list)
    mean_fused_std = np.mean(mse_fused_std_list)
    mean_fused_sph = np.mean(mse_fused_spherical_list)

    print("-" * 90)
    print(f"MEAN RAW WMC MSE:       {mean_std:.4e}")
    print(f"MEAN SPHERICAL WMC MSE: {mean_sph:.4e} (Change: {((mean_sph - mean_std)/mean_std)*100:+.2f}%)")
    print(f"MEAN FUSED STD MSE:     {mean_fused_std:.6e} (Adj: {mean_fused_std*0.1:.6e})")
    print(f"MEAN FUSED SPH MSE:     {mean_fused_sph:.6e} (Adj: {mean_fused_sph*0.1:.6e}) (Change: {((mean_fused_sph - mean_fused_std)/mean_fused_std)*100:+.2f}%)")

if __name__ == "__main__":
    test_sampling_carriers()
