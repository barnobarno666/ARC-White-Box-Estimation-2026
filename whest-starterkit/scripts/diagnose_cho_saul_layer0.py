import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _S_PRIOR

_PI = fnp.asarray(np.pi, dtype=fnp.float32)
_HALF_PI = fnp.asarray(np.pi / 2.0, dtype=fnp.float32)
_INV_2PI = fnp.asarray(1.0 / (2.0 * np.pi), dtype=fnp.float32)

def exact_cho_saul_cov_0(mlp: MLP) -> fnp.ndarray:
    """Computes exact covariance at layer 0 using Cho-Saul arc-cosine kernel, then propagates."""
    width = mlp.width
    # Layer 0 weights
    w0 = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
    # At layer 0: z0 ~ N(0, Sigma_0) where Sigma_0 = w0.T @ w0
    cov_pre_0 = w0.T @ w0
    var_0 = fnp.maximum(fnp.diag(cov_pre_0), fnp.asarray(1e-12, dtype=fnp.float32))
    sigma_0 = fnp.sqrt(var_0)
    
    # Exact mean at layer 0: E[ReLU(z_0)] = sigma_0 / sqrt(2*pi)
    mu_0 = sigma_0 / fnp.sqrt(2.0 * _PI)
    
    # Exact covariance at layer 0:
    inv_sigma_0 = 1.0 / sigma_0
    rho_0 = fnp.clip((inv_sigma_0[:, None] * cov_pre_0) * inv_sigma_0[None, :], -0.999999, 0.999999)
    # Cho-Saul degree 1 kernel for standard normals:
    # E[ReLU(u) ReLU(v)] = (sqrt(1 - rho^2) + rho * (pi/2 + arcsin(rho))) / (2*pi)
    term1 = fnp.sqrt(1.0 - rho_0 * rho_0)
    term2 = rho_0 * (_HALF_PI + fnp.arcsin(rho_0))
    e_joint = (term1 + term2) * _INV_2PI
    cov_0 = (sigma_0[:, None] * e_joint) * sigma_0[None, :] - mu_0[:, None] * mu_0[None, :]
    fnp.fill_diagonal(cov_0, 0.5 * var_0 - mu_0 * mu_0)
    cov_0 = flops.as_symmetric(cov_0, symmetry=(0, 1))

    # Now propagate from layer 1 onwards using blended Hermite
    cov = cov_0
    mu = mu_0
    rows = [mu_0]

    _ZERO = fnp.asarray(0.0, dtype=fnp.float32)
    _C_CENTERED = fnp.asarray(0.80 * 0.20 * 0.07957747154594767, dtype=fnp.float32)
    _C_THRESH = fnp.asarray(0.20 * 0.5, dtype=fnp.float32)

    for l in range(1, mlp.depth):
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
        rows.append(mu)

    return fnp.stack(rows, axis=0)

def test_cho_saul():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("TESTING EXACT CHO-SAUL LAYER 0 COVARIANCE")
    print("=" * 90)

    s0 = float(_S_PRIOR)
    mses_old = []
    mses_new = []

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_true = np.array(row["all_layer_means"][-1], dtype=np.float64)

        c_old = np.array(_blended_hermite_covariance(mlp)[-1], dtype=np.float64) * s0
        c_new = np.array(exact_cho_saul_cov_0(mlp)[-1], dtype=np.float64) * s0

        mse_old = float(np.mean((c_old - y_true)**2))
        mse_new = float(np.mean((c_new - y_true)**2))

        mses_old.append(mse_old)
        mses_new.append(mse_new)

        diff = ((mse_new - mse_old) / mse_old) * 100
        print(f"MLP {i} ({row['mlp_name']}): Old={mse_old:.6e}, New={mse_new:.6e} ({diff:+.4f}%)")

    print("-" * 90)
    print(f"Mean Old MSE: {np.mean(mses_old):.6e}")
    print(f"Mean New MSE: {np.mean(mses_new):.6e}")
    total_diff = ((np.mean(mses_new) - np.mean(mses_old)) / np.mean(mses_old)) * 100
    print(f"Overall Difference: {total_diff:+.4f}%")

if __name__ == "__main__":
    test_cho_saul()
