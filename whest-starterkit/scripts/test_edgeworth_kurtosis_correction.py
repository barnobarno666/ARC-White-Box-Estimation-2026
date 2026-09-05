import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _S_PRIOR

def edgeworth_covariance(mlp: MLP, eta_kurt: float = 1.0) -> fnp.ndarray:
    """Gain-covariance propagation with Edgeworth kurtosis correction."""
    width = mlp.width
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
    rows = []

    _ZERO = fnp.asarray(0.0, dtype=fnp.float32)
    _C_CENTERED = fnp.asarray(0.80 * 0.20 * 0.07957747154594767, dtype=fnp.float32)
    _C_THRESH = fnp.asarray(0.20 * 0.5, dtype=fnp.float32)

    for l, weight in enumerate(mlp.weights):
        w = fnp.asarray(weight, dtype=fnp.float32)
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), fnp.asarray(1e-12, dtype=fnp.float32))
        sigma_pre = fnp.sqrt(var_pre)
        a = mu_pre / sigma_pre
        phi = flops.stats.norm.pdf(a).astype(fnp.float32)
        cdf = flops.stats.norm.cdf(a).astype(fnp.float32)

        # Base Gaussian expectation
        mu_gauss = mu_pre * cdf + sigma_pre * phi

        # Edgeworth excess kurtosis correction:
        # gamma_2 accumulates linearly with layer index: gamma_2(l) = 0.0042 * l
        gamma_2 = float(eta_kurt * 0.0042 * l)
        c_kurt = fnp.asarray(gamma_2 / 24.0, dtype=fnp.float32)
        kurt_correction = sigma_pre * c_kurt * (a * a - 1.0) * phi

        mu = mu_gauss + kurt_correction

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

def test_edgeworth():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("TESTING EDGEWORTH KURTOSIS CORRECTION ON 8-MLP PANEL")
    print("=" * 90)

    s0 = float(_S_PRIOR)
    y_true_list = []
    c_base_list = []

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = np.array(row["all_layer_means"][-1], dtype=np.float64)
        y_true_list.append(y)
        c = np.array(_blended_hermite_covariance(mlp)[-1], dtype=np.float64) * s0
        c_base_list.append(c)

    base_mses = [float(np.mean((c_base_list[i] - y_true_list[i])**2)) for i in range(8)]
    mean_base = np.mean(base_mses)
    print(f"Base Blended Hermite Cov MSE: {mean_base:.6e}")

    for eta in [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]:
        test_mses = []
        for i in range(8):
            row = ds[i]
            mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
            c_edge = np.array(edgeworth_covariance(mlp, eta_kurt=eta)[-1], dtype=np.float64) * s0
            test_mses.append(float(np.mean((c_edge - y_true_list[i])**2)))
        
        mean_test = np.mean(test_mses)
        diff = ((mean_test - mean_base) / mean_base) * 100
        wins = sum(1 for a, b in zip(test_mses, base_mses) if a < b)
        print(f"eta={eta:4.1f}: MSE = {mean_test:.6e} ({diff:+.3f}%, Wins: {wins}/8)")

if __name__ == "__main__":
    test_edgeworth()
