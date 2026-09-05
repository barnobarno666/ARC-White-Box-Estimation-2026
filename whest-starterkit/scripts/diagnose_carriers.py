"""diagnose_carriers.py

P4-01 Carrier Tournament:
Compare pure directional / sampling carriers on the official 8-MLP panel:
1. Pure WMC (Whitened Antithetic MC, N=4096)
2. Exact-Radius Antithetic MC (ER-AMC, N=4096)
3. Exact-Radius Whitened Antithetic MC (ER-WMC, N=4096)
4. Randomized Shifted Lattice with Baker Transform (QMC, N=4096)
5. Randomized Hadamard Frame (N=4096)
6. Multi-Basis Orthogonal Frame Bank (Identity + 3 Hadamard frames, N=4096)

Computes for each carrier:
- Raw MSE on Layer 15 against ground truth
- Correlation with ground truth and residual variance
- Headroom when blended with the analytical Hermite branch
"""

import math
import numpy as np
import scipy.stats as stats
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _S_PRIOR

# Exact expected radius of Chi(1024)
_MU_R = math.sqrt(2.0) * math.exp(math.lgamma(1025.0 / 2.0) - math.lgamma(512.0))
# Approximately 31.99218705353147

def generate_sylvester_hadamard(n: int = 1024) -> np.ndarray:
    """Generate normalized n x n Sylvester Hadamard matrix (H @ H.T = I)."""
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h * (1.0 / math.sqrt(n))

_H1024 = generate_sylvester_hadamard(1024)

def forward_carrier(mlp: MLP, x_input: np.ndarray) -> np.ndarray:
    """Forward carrier points x_input through mlp.weights and compute final-layer mean."""
    # Convert to fnp
    x = fnp.asarray(x_input, dtype=fnp.float32)
    zero = fnp.asarray(0.0, dtype=fnp.float32)
    for w in mlp.weights:
        w_fnp = fnp.asarray(w, dtype=fnp.float32)
        x = fnp.maximum(x @ w_fnp, zero)
    # Return numpy mean on final layer
    final_act = np.array(x, dtype=np.float64)
    return np.mean(final_act, axis=0)

def carrier_wmc(mlp: MLP, count: int = 4096) -> np.ndarray:
    """Standard Whitened Antithetic Monte Carlo."""
    half = count // 2
    rng = np.random.default_rng(mlp.seed)
    z = rng.standard_normal((half, mlp.width)).astype(np.float32)
    x = np.concatenate((z, -z), axis=0)
    # Gram whitening
    gram = (x.T @ x) / float(count)
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 1e-6)
    whitener = (eigvecs * np.power(eigvals, -0.5)) @ eigvecs.T
    x_whitened = x @ whitener
    return forward_carrier(mlp, x_whitened)

def carrier_er_amc(mlp: MLP, count: int = 4096) -> np.ndarray:
    """Exact-Radius Antithetic Monte Carlo (zero radial variance)."""
    half = count // 2
    rng = np.random.default_rng(mlp.seed)
    z = rng.standard_normal((half, mlp.width)).astype(np.float32)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    u = z / norms  # On unit sphere S^{1023}
    x_half = (_MU_R * u).astype(np.float32)
    x = np.concatenate((x_half, -x_half), axis=0)
    return forward_carrier(mlp, x)

def carrier_er_wmc(mlp: MLP, count: int = 4096) -> np.ndarray:
    """Exact-Radius + Layer-1 Whitening."""
    half = count // 2
    rng = np.random.default_rng(mlp.seed)
    z = rng.standard_normal((half, mlp.width)).astype(np.float32)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    u = z / norms
    x_half = (_MU_R * u).astype(np.float32)
    x = np.concatenate((x_half, -x_half), axis=0)
    gram = (x.T @ x) / float(count)
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 1e-6)
    whitener = (eigvecs * np.power(eigvals, -0.5)) @ eigvecs.T
    x_whitened = x @ whitener
    # Renormalize rows to exact radius after whitening
    norms_w = np.linalg.norm(x_whitened, axis=1, keepdims=True)
    x_final = (_MU_R * (x_whitened / np.maximum(norms_w, 1e-12))).astype(np.float32)
    return forward_carrier(mlp, x_final)

def carrier_qmc_lattice(mlp: MLP, count: int = 4096) -> np.ndarray:
    """Randomized rank-1 lattice with baker transform and exact radius."""
    rng = np.random.default_rng(mlp.seed)
    # Generator vector for rank-1 lattice in 1024 dimensions
    # Using Korobov generator: g_j = a^{j-1} mod N
    a = 104729  # Large prime
    j = np.arange(mlp.width)
    # Use powers modulo count
    g = np.array([pow(a, int(k), count) for k in j], dtype=np.float64) / float(count)
    
    # Uniform points with random shift
    i = np.arange(count // 2)[:, None]
    shift = rng.uniform(0.0, 1.0, size=(1, mlp.width))
    pts = np.mod(i * g + shift, 1.0)
    
    # Baker's tent transform: fold [0,1] to ensure symmetry
    tent = 1.0 - np.abs(2.0 * pts - 1.0)
    # Map to standard normal quantiles, clip away from 0 and 1
    tent = np.clip(tent, 1e-6, 1.0 - 1e-6)
    z = stats.norm.ppf(tent).astype(np.float32)
    
    # Project to exact sphere
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    u = z / np.maximum(norms, 1e-12)
    x_half = (_MU_R * u).astype(np.float32)
    x = np.concatenate((x_half, -x_half), axis=0)
    return forward_carrier(mlp, x)

def carrier_hadamard(mlp: MLP, count: int = 4096) -> np.ndarray:
    """Randomized Hadamard frames on S^1023 (4 blocks of 1024 antipodal lines = 4096)."""
    rng = np.random.default_rng(mlp.seed)
    n_blocks = count // 2048  # Each block gives 1024 lines = 2048 points (+ and -)
    blocks = []
    for _ in range(n_blocks):
        # Random sign flip + random column permutation
        d = rng.choice([-1.0, 1.0], size=(1, mlp.width)).astype(np.float32)
        p = rng.permutation(mlp.width)
        h_scrambled = (_H1024[p, :] * d).astype(np.float32)
        pts_half = (_MU_R * h_scrambled).astype(np.float32)
        blocks.append(pts_half)
        blocks.append(-pts_half)
    x = np.concatenate(blocks, axis=0)
    return forward_carrier(mlp, x)

def carrier_multi_basis(mlp: MLP, count: int = 4096) -> np.ndarray:
    """Multi-Basis Bank: Identity coordinate basis + 3 scrambled Hadamard bases."""
    rng = np.random.default_rng(mlp.seed)
    # 4 bases of 1024 points each (512 antipodal pairs per basis = 2048 points total, or 2 bases of 1024 pairs)
    # Let's allocate 1024 antipodal lines across 2 bases (2048 pts each = 4096 total):
    # Basis 1: Randomly rotated orthogonal matrix (from QR or random sign Hadamard)
    d1 = rng.choice([-1.0, 1.0], size=(1, mlp.width)).astype(np.float32)
    p1 = rng.permutation(mlp.width)
    b1 = (_H1024[p1, :] * d1).astype(np.float32)
    
    # Basis 2: Disjointly scrambled Hadamard
    d2 = rng.choice([-1.0, 1.0], size=(1, mlp.width)).astype(np.float32)
    p2 = rng.permutation(mlp.width)
    b2 = (_H1024[p2, :] * d2).astype(np.float32)
    
    pts1 = (_MU_R * b1).astype(np.float32)
    pts2 = (_MU_R * b2).astype(np.float32)
    x = np.concatenate((pts1, -pts1, pts2, -pts2), axis=0)
    return forward_carrier(mlp, x)

def run_carrier_tournament():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("P4-01: CARRIER TOURNAMENT SCREEN (8 MLPs, N=4096)")
    print("=" * 90)

    # Pre-extract ground truth and analytical Hermite branch
    mlps = []
    y_trues = []
    c_analyticals = []

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        mlps.append(mlp)
        y_trues.append(np.array(row["final_means"], dtype=np.float64))
        # Analytical calibrated branch
        c_fnp = _blended_hermite_covariance(mlp)
        c_cal = np.array(c_fnp[-1], dtype=np.float64) * float(_S_PRIOR)
        c_analyticals.append(c_cal)

    carriers = [
        ("1. Pure WMC (N=4096)", carrier_wmc),
        ("2. Exact-Radius AMC (N=4096)", carrier_er_amc),
        ("3. Exact-Radius WMC (N=4096)", carrier_er_wmc),
        ("4. QMC Lattice (N=4096)", carrier_qmc_lattice),
        ("5. Scrambled Hadamard (N=4096)", carrier_hadamard),
        ("6. Multi-Basis Bank (N=4096)", carrier_multi_basis),
    ]

    for name, carrier_fn in carriers:
        raw_mses = []
        oracle_blend_mses = []
        fixed_blend_mses = []
        res_corrs = []

        for i in range(8):
            mlp = mlps[i]
            y = y_trues[i]
            c = c_analyticals[i]
            m = carrier_fn(mlp)

            # Raw sampler MSE
            err_m = m - y
            mse_m = float(np.mean(err_m * err_m))
            raw_mses.append(mse_m)

            # Error correlation between analytical branch and sampler
            err_c = c - y
            dot = float(np.sum(err_c * err_m))
            norm_c = float(np.linalg.norm(err_c))
            norm_m = float(np.linalg.norm(err_m))
            corr = dot / max(norm_c * norm_m, 1e-12)
            res_corrs.append(corr)

            # Optimal per-MLP oracle convex blend: min_alpha ||(1-alpha)*c + alpha*m - y||^2
            # d_err = m - c
            # alpha* = - <err_c, d_err> / ||d_err||^2
            d_err = m - c
            d_norm_sq = float(np.sum(d_err * d_err))
            alpha_star = -float(np.sum(err_c * d_err)) / max(d_norm_sq, 1e-12)
            alpha_star = max(0.0, min(1.0, alpha_star))
            pred_oracle = (1.0 - alpha_star) * c + alpha_star * m
            oracle_blend_mses.append(float(np.mean((pred_oracle - y)**2)))

            # Fixed blend at alpha=0.110 (as in current champion)
            pred_fixed = 0.890 * c + 0.110 * m
            fixed_blend_mses.append(float(np.mean((pred_fixed - y)**2)))

        mean_raw = np.mean(raw_mses)
        mean_oracle = np.mean(oracle_blend_mses)
        mean_fixed = np.mean(fixed_blend_mses)
        adj_fixed = mean_fixed * 0.1000
        mean_corr = np.mean(res_corrs)

        print(f"\nCarrier: {name}")
        print(f"  Raw Sampler MSE:    {mean_raw:.6e}")
        print(f"  Residual Corr to C: {mean_corr:+.4f}")
        print(f"  Fixed Blend MSE:    {mean_fixed:.6e} -> Adj Score: {adj_fixed:.6e}")
        print(f"  Oracle Blend MSE:   {mean_oracle:.6e} -> Oracle Adj Score: {mean_oracle * 0.1000:.6e}")

if __name__ == "__main__":
    run_carrier_tournament()
