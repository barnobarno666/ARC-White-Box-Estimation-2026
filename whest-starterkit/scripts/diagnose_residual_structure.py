"""diagnose_residual_structure.py

Lane D & Residual Spectral Analysis:
Examines the ground-truth residual e = c_cal - y* across all 8 MLPs:
1. Spectral decomposition: Does e lie in the span of W_15, W_14 @ W_15, or Sigma_15?
2. Skewness / Kurtosis diagnostic: What is the non-Gaussianity of pre-activations at layer 15?
3. Localized final-layer skewness correction:
   Under Gram-Charlier / Edgeworth expansion, the mean of ReLU(z) when z has skewness gamma_1 is:
   E[ReLU(z)] = mu * Phi(a) + sigma * phi(a) + (gamma_1 / 6) * sigma * (a^2 - 1) * phi(a) + ...
   Can this skewness correction capture the remaining bias?
"""

import numpy as np
import scipy.stats as stats
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _S_PRIOR, _whitened_antithetic_mc

def analyze_residual():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("RESIDUAL SPECTRAL & NON-GAUSSIAN SKEWNESS DIAGNOSTIC")
    print("=" * 90)

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = np.array(row["final_means"], dtype=np.float64)
        c_all = _blended_hermite_covariance(mlp)
        c = np.array(c_all[-1], dtype=np.float64) * float(_S_PRIOR)
        wmc = np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64)

        pred_champ = 0.890 * c + 0.110 * wmc
        err_champ = pred_champ - y
        err_c = c - y
        err_wmc = wmc - y

        mse_champ = np.mean(err_champ**2)
        mse_c = np.mean(err_c**2)
        mse_wmc = np.mean(err_wmc**2)

        # Spectral projection of err_c onto singular vectors of W_15
        W15 = np.array(mlp.weights[-1], dtype=np.float64)
        U, S, Vt = np.linalg.svd(W15)

        # How much variance of err_c is explained by top k singular vectors of W15?
        projs = []
        for k in [4, 16, 64, 256, 512, 1024]:
            Uk = U[:, :k]  # (1024, k)
            # Projection of err_c onto Uk: Uk @ Uk.T @ err_c
            err_proj = Uk @ (Uk.T @ err_c)
            var_explained = np.sum(err_proj**2) / np.sum(err_c**2)
            projs.append((k, var_explained))

        print(f"MLP {i+1} ({row['mlp_name']}):")
        print(f"  MSE Champ: {mse_champ:.6e} | MSE C: {mse_c:.6e} | MSE WMC: {mse_wmc:.6e}")
        print(f"  Energy in top singular vectors of W_15: " + ", ".join([f"top-{k}: {v*100:.1f}%" for k, v in projs]))

        # Check empirical skewness from Monte Carlo samples
        # Sample N=4200 preactivations at layer 15
        rng = np.random.default_rng(mlp.seed)
        half = 2100
        z = rng.standard_normal((half, mlp.width)).astype(np.float32)
        x = np.concatenate((z, -z), axis=0)
        # Forward up to layer 14
        zero = fnp.asarray(0.0, dtype=fnp.float32)
        act = fnp.asarray(x, dtype=fnp.float32)
        for l in range(mlp.depth - 1):
            act = fnp.maximum(act @ fnp.asarray(mlp.weights[l], dtype=fnp.float32), zero)
        # Preactivation at layer 15
        pre_15 = np.array(act @ fnp.asarray(mlp.weights[-1], dtype=fnp.float32), dtype=np.float64)
        skew_15 = stats.skew(pre_15, axis=0)
        kurt_15 = stats.kurtosis(pre_15, axis=0)  # excess kurtosis

        print(f"  Layer 15 Preactivations: Mean Skewness={np.mean(skew_15):+.4f} (std={np.std(skew_15):.4f}), Mean Excess Kurtosis={np.mean(kurt_15):+.4f}")

if __name__ == "__main__":
    analyze_residual()
