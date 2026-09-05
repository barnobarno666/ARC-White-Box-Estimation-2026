"""diagnose_dead_on_kink.py

Examines the dead/on/kink structure of layer 15 neurons across the 8 MLPs:
1. Computes preactivation standardized values a_i = mu_i / sigma_i
2. Classifies neurons:
   - Dead: a_i < -3.0 (fire probability < 0.0013)
   - On:   a_i > +3.0 (fire probability > 0.9987)
   - Kink: -3.0 <= a_i <= +3.0
3. Measures how much of the MSE in c_cal and wmc comes from Kink vs Dead vs On neurons!
"""

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _whitened_antithetic_mc, _S_PRIOR

def diagnose_dead_on_kink():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("DEAD / ON / KINK NEURON ANALYSIS AT LAYER 15")
    print("=" * 90)

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = np.array(row["final_means"], dtype=np.float64)

        # Get preactivation stats from covariance propagation
        # Forward through layer 14 to get mu_14 and cov_14
        width = mlp.width
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

        # Layer 15 preactivation:
        w15 = fnp.asarray(mlp.weights[15], dtype=fnp.float32)
        mu_pre15 = np.array(w15.T @ mu, dtype=np.float64)
        cov_pre15 = np.array(fnp.einsum("ij,ia,jb->ab", cov, w15, w15), dtype=np.float64)
        sigma_pre15 = np.sqrt(np.maximum(np.diag(cov_pre15), 1e-12))
        a15 = mu_pre15 / sigma_pre15

        c = np.array(_blended_hermite_covariance(mlp)[-1], dtype=np.float64) * float(_S_PRIOR)
        wmc = np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64)
        pred_champ = 0.890 * c + 0.110 * wmc

        err_sq = (pred_champ - y)**2
        mse_total = np.mean(err_sq)

        dead_mask = a15 < -2.5
        on_mask = a15 > +2.5
        kink_mask = ~dead_mask & ~on_mask

        n_dead = np.sum(dead_mask)
        n_on = np.sum(on_mask)
        n_kink = np.sum(kink_mask)

        mse_dead = np.mean(err_sq[dead_mask]) if n_dead > 0 else 0.0
        mse_on = np.mean(err_sq[on_mask]) if n_on > 0 else 0.0
        mse_kink = np.mean(err_sq[kink_mask]) if n_kink > 0 else 0.0

        # Also check ground truth values for dead and on
        y_dead_mean = np.mean(y[dead_mask]) if n_dead > 0 else 0.0
        y_on_mean = np.mean(y[on_mask]) if n_on > 0 else 0.0

        print(f"MLP {i+1} ({row['mlp_name']}): Total MSE = {mse_total:.6e}")
        print(f"  Dead (|a|<-2.5): Count={n_dead:3d} | MSE={mse_dead:.6e} | Mean GT={y_dead_mean:.6f}")
        print(f"  On   (|a|>+2.5): Count={n_on:3d} | MSE={mse_on:.6e} | Mean GT={y_on_mean:.6f}")
        print(f"  Kink (rest)    : Count={n_kink:3d} | MSE={mse_kink:.6e}")

if __name__ == "__main__":
    diagnose_dead_on_kink()
