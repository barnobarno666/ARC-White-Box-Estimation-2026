import flopscope as flops
import flopscope.numpy as fnp
from local_engine import build_mlp, monte_carlo_layer_means

def test_variance_reduction():
    width = 1024
    depth = 16
    seed = 42
    mlp = build_mlp(width=width, depth=depth, seed=seed)

    print("Computing high-sample reference (N=50,000)...")
    # compute reference with N=50,000 to measure raw MSE
    truth = monte_carlo_layer_means(mlp, n_samples=50_000, seed=123)

    # 1. Standard Gain-Covariance Propagation
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
    cov_means = []
    gains = []
    _ZERO = fnp.asarray(0.0, dtype=fnp.float32)

    for weight in mlp.weights:
        w = fnp.asarray(weight, dtype=fnp.float32)
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), fnp.asarray(1e-12, dtype=fnp.float32))
        sigma_pre = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        phi = flops.stats.norm.pdf(alpha).astype(fnp.float32)
        cdf = flops.stats.norm.cdf(alpha).astype(fnp.float32)

        mu = mu_pre * cdf + sigma_pre * phi
        second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
        var_post = fnp.maximum(second - mu * mu, _ZERO)
        gain = fnp.where(sigma_pre > fnp.asarray(1e-12, dtype=fnp.float32), cdf, _ZERO)
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)
        cov = flops.as_symmetric(cov, symmetry=(0, 1))
        cov_means.append(mu)
        gains.append(gain)

    cov_pred = fnp.stack(cov_means, axis=0)
    cov_mse = fnp.mean((cov_pred[-1] - truth[-1]) ** 2)
    print(f"Covariance Propagation Final Layer MSE: {float(cov_mse):.6e}")

    # 2. Whitened Antithetic MC (N=4,038)
    count = 4_038
    half = count // 2
    scale = fnp.asarray(1.0 / count, dtype=fnp.float32)
    rng = fnp.random.default_rng(mlp.seed)
    x_half = fnp.asarray(rng.standard_normal((half, width)), dtype=fnp.float32)
    x = fnp.concatenate((x_half, -x_half), axis=0)
    gram = (x.T @ x) / fnp.asarray(float(count), dtype=fnp.float32)
    eigenvalues, eigenvectors = fnp.linalg.eigh(gram)
    eigenvalues = fnp.maximum(eigenvalues, fnp.asarray(1e-6, dtype=fnp.float32))
    whitener = (eigenvectors * fnp.power(eigenvalues, -0.5)) @ eigenvectors.T
    first_weight = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
    activations = fnp.maximum(x @ (whitener @ first_weight), _ZERO)

    mc_means = [fnp.sum(activations, axis=0) * scale]
    for layer in range(1, depth):
        weight = fnp.asarray(mlp.weights[layer], dtype=fnp.float32)
        activations = fnp.maximum(activations @ weight, _ZERO)
        mc_means.append(fnp.sum(activations, axis=0) * scale)
    mc_pred = fnp.stack(mc_means, axis=0)
    mc_mse = fnp.mean((mc_pred[-1] - truth[-1]) ** 2)
    print(f"Whitened Antithetic MC Final Layer MSE: {float(mc_mse):.6e}")

    # Baseline 75/25 blend
    blend_pred = 0.75 * cov_pred + 0.25 * mc_pred
    blend_mse = fnp.mean((blend_pred[-1] - truth[-1]) ** 2)
    print(f"Baseline Blend (0.75/0.25) Final Layer MSE: {float(blend_mse):.6e}")

    # 3. Test Mid-Network Control Variate anchored at layer k
    for k in [4, 6, 8, 10, 12]:
        diff = mc_pred[k] - cov_pred[k]
        # Propagate diff through subsequent linear gates
        v = diff
        for l in range(k + 1, depth):
            w = fnp.asarray(mlp.weights[l], dtype=fnp.float32)
            g = gains[l]
            v = (v @ w) * g
        
        for beta in [0.5, 0.7, 0.9, 1.0]:
            cv_pred = mc_pred[-1] - beta * v
            cv_mse = fnp.mean((cv_pred - truth[-1]) ** 2)
            # Also blend with covariance
            blend_cv = 0.75 * cov_pred[-1] + 0.25 * cv_pred
            blend_cv_mse = fnp.mean((blend_cv - truth[-1]) ** 2)
            print(f"k={k}, beta={beta:.1f} -> MC_CV MSE: {float(cv_mse):.6e}, Blend_CV MSE: {float(blend_cv_mse):.6e}")

if __name__ == "__main__":
    test_variance_reduction()
