import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance

def test_analytic_vs_mc_stats():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    row = ds[0]
    mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
    
    # Run 50,000 MC samples to get near-exact empirical ground truth for layer 1 and layer 2
    count = 50000
    rng = np.random.default_rng(mlp.seed)
    x = rng.standard_normal((count, mlp.width)).astype(np.float32)
    
    # Layer 0
    w0 = np.array(mlp.weights[0], dtype=np.float32)
    z0 = x @ w0
    h0 = np.maximum(z0, 0.0)
    
    # Layer 1
    w1 = np.array(mlp.weights[1], dtype=np.float32)
    z1 = h0 @ w1
    h1 = np.maximum(z1, 0.0)

    # True statistics at layer 1
    mu_z1_true = np.mean(z1, axis=0)
    var_z1_true = np.var(z1, axis=0)
    mu_h1_true = np.mean(h1, axis=0)
    var_h1_true = np.var(h1, axis=0)

    # Analytic statistics at layer 1
    c = _blended_hermite_covariance(mlp)
    
    print(f"Layer 0 True mean h0: {np.mean(h0):.6f}, Analytic: {np.mean(c[0]):.6f}")
    print(f"Layer 1 True mean z1: {np.mean(mu_z1_true):.6f}")
    print(f"Layer 1 True var z1:  {np.mean(var_z1_true):.6f}")
    print(f"Layer 1 True mean h1: {np.mean(mu_h1_true):.6f}, Analytic: {np.mean(c[1]):.6f}")
    print(f"Layer 1 Diff mean h1: {np.mean(c[1]) - np.mean(mu_h1_true):.6e}")

    # Check non-Gaussianity of z1
    # Skewness and kurtosis of z1
    from scipy.stats import skew, kurtosis
    sk = skew(z1, axis=0)
    kt = kurtosis(z1, axis=0)
    print(f"Layer 1 z1 skewness: mean = {np.mean(sk):.6f}, max = {np.max(np.abs(sk)):.6f}")
    print(f"Layer 1 z1 excess kurtosis: mean = {np.mean(kt):.6f}, max = {np.max(np.abs(kt)):.6f}")

    # Check correlation matrix of h0
    cov_h0_true = np.cov(h0.T)
    diag_h0 = np.diag(cov_h0_true)
    corr_h0_true = cov_h0_true / np.sqrt(np.outer(diag_h0, diag_h0))
    np.fill_diagonal(corr_h0_true, 0)
    print(f"Layer 0 h0 off-diagonal correlations: mean = {np.mean(corr_h0_true):.6e}, rms = {np.sqrt(np.mean(corr_h0_true**2)):.6f}, max = {np.max(np.abs(corr_h0_true)):.6f}")

if __name__ == "__main__":
    test_analytic_vs_mc_stats()
