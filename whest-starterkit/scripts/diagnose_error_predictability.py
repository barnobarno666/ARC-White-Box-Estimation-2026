import numpy as np
import scipy.stats as stats
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _S_PRIOR

def diagnose_error_predictability():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 95)
    print("PREDICTABILITY OF ANALYTICAL COVARIANCE RESIDUAL ERROR e_c = c0 - y*")
    print("=" * 95)

    s0 = float(_S_PRIOR)
    
    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_true = np.array(row["all_layer_means"][-1], dtype=np.float64)  # (1024,)

        # Run covariance propagation and extract internal state at layer 15
        width = mlp.width
        mu = fnp.zeros(width, dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
        _ZERO = fnp.asarray(0.0, dtype=fnp.float32)
        _C_CENTERED = fnp.asarray(0.80 * 0.20 * 0.07957747154594767, dtype=fnp.float32)
        _C_THRESH = fnp.asarray(0.20 * 0.5, dtype=fnp.float32)

        mu_pre_last = None
        var_pre_last = None
        cdf_last = None
        phi_last = None
        gain_last = None

        for l, weight in enumerate(mlp.weights):
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
            cov = flops.as_symmetric(cov, symmetry=(0, 1))

            if l == mlp.depth - 1:
                mu_pre_last = np.array(mu_pre, dtype=np.float64)
                var_pre_last = np.array(var_pre, dtype=np.float64)
                cdf_last = np.array(cdf, dtype=np.float64)
                phi_last = np.array(phi, dtype=np.float64)
                var_post_last = np.array(var_post, dtype=np.float64)

        c_last = np.array(mu, dtype=np.float64)
        c0 = s0 * c_last
        e_c = c0 - y_true
        mse_c0 = np.mean(e_c**2)

        # Compute column norms of weight matrix W_15
        w_last = np.array(mlp.weights[-1], dtype=np.float64)
        w_col_norms = np.linalg.norm(w_last, axis=0)  # (1024,)

        # Correlations between e_c and features
        features = {
            "c0": c0,
            "mu_pre": mu_pre_last,
            "sigma_pre": np.sqrt(var_pre_last),
            "cdf (activation prob)": cdf_last,
            "phi": phi_last,
            "sigma_post": np.sqrt(var_post_last),
            "||w_col||": w_col_norms,
            "a (mu_pre/sigma_pre)": mu_pre_last / np.sqrt(var_pre_last),
            "c0^2": c0**2,
        }

        print(f"\nMLP {i} ({row['mlp_name']}) - MSE c0: {mse_c0:.6e}:")
        for fname, fval in features.items():
            r = np.corrcoef(e_c, fval)[0, 1]
            print(f"  Corr(e_c, {fname:<22}): {r:+.4f}")

if __name__ == "__main__":
    diagnose_error_predictability()
