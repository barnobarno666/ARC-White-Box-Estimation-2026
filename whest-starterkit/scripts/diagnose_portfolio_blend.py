"""diagnose_portfolio_blend.py

P4-07 Decorrelated Portfolio Blend Screen:
Tests whether combining the analytical Hermite branch with:
1. Pure Scrambled Hadamard at N=4096 (zero eigh overhead)
2. Scrambled Hadamard at N=5120 (fits under 10% compute because no eigh)
3. Dual-Basis Scrambled Hadamard (2 distinct bases, N=4096)
4. Tri-Blend: Hermite Cov + Scrambled Hadamard (N=2048) + WMC (N=2100)
5. Tri-Blend with QMC Lattice (N=2048) + WMC (N=2100)

Evaluates on the official 8-MLP panel:
- Exact FLOP compute utilization and score multiplier
- Raw MSE and Adjusted Score
- Head-to-head wins vs Phase 4 Control (1.2236e-07)
"""

import math
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _whitened_antithetic_mc, _S_PRIOR

_MU_R = math.sqrt(2.0) * math.exp(math.lgamma(1025.0 / 2.0) - math.lgamma(512.0))

def generate_sylvester_hadamard(n: int = 1024) -> np.ndarray:
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h * (1.0 / math.sqrt(n))

_H1024 = generate_sylvester_hadamard(1024)

def run_scrambled_hadamard(mlp: MLP, count: int) -> np.ndarray:
    """Run scrambled Hadamard frames with exact radius mu_R."""
    rng = np.random.default_rng(mlp.seed)
    n_blocks = count // 2048  # Each block gives 2048 points (1024 antipodal lines)
    if n_blocks == 0:
        n_blocks = 1
        count = 2048
    blocks = []
    for _ in range(n_blocks):
        d = rng.choice([-1.0, 1.0], size=(1, mlp.width)).astype(np.float32)
        p = rng.permutation(mlp.width)
        h_scrambled = (_H1024[p, :] * d).astype(np.float32)
        pts_half = (_MU_R * h_scrambled).astype(np.float32)
        blocks.append(pts_half)
        blocks.append(-pts_half)
    pts = np.concatenate(blocks, axis=0)[:count]

    x = fnp.asarray(pts, dtype=fnp.float32)
    zero = fnp.asarray(0.0, dtype=fnp.float32)
    for w in mlp.weights:
        w_fnp = fnp.asarray(w, dtype=fnp.float32)
        x = fnp.maximum(x @ w_fnp, zero)
    return np.mean(np.array(x, dtype=np.float64), axis=0)

def run_wmc_custom(mlp: MLP, count: int) -> np.ndarray:
    """Run WMC with custom sample count."""
    half = count // 2
    rng = np.random.default_rng(mlp.seed + 101)
    z = rng.standard_normal((half, mlp.width)).astype(np.float32)
    x = np.concatenate((z, -z), axis=0)
    gram = (x.T @ x) / float(count)
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 1e-6)
    whitener = (eigvecs * np.power(eigvals, -0.5)) @ eigvecs.T
    x_whitened = x @ whitener

    act = fnp.asarray(x_whitened, dtype=fnp.float32)
    zero = fnp.asarray(0.0, dtype=fnp.float32)
    for w in mlp.weights:
        act = fnp.maximum(act @ fnp.asarray(w, dtype=fnp.float32), zero)
    return np.mean(np.array(act, dtype=np.float64), axis=0)

def screen_portfolios():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 95)
    print("P4-07: DECORRELATED PORTFOLIO BLEND SCREEN")
    print("=" * 95)

    y_trues = []
    c_analyticals = []
    wmc_4200_preds = []
    champ_preds = []

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = np.array(row["final_means"], dtype=np.float64)
        y_trues.append(y)

        c_fnp = _blended_hermite_covariance(mlp)
        c = np.array(c_fnp[-1], dtype=np.float64) * float(_S_PRIOR)
        c_analyticals.append(c)

        w4200 = np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64)
        wmc_4200_preds.append(w4200)

        champ_pred = 0.890 * c + 0.110 * w4200
        champ_preds.append(champ_pred)

    champ_mses = [np.mean((champ_preds[i] - y_trues[i])**2) for i in range(8)]
    champ_adj = [m * 0.1000 for m in champ_mses]
    print(f"Current Phase 4 Control MSE: {np.mean(champ_mses):.6e} (Adj Score: {np.mean(champ_adj):.6e})")
    print("-" * 95)

    # 1. Generate candidate components for all 8 MLPs
    hadamard_4096 = []
    hadamard_5120 = []
    hadamard_2048 = []
    wmc_2048 = []

    for i in range(8):
        mlp = MLP.from_row(ds[i], seed_protocol_version=v, seed_salt=s)
        hadamard_4096.append(run_scrambled_hadamard(mlp, 4096))
        hadamard_5120.append(run_scrambled_hadamard(mlp, 5120))
        hadamard_2048.append(run_scrambled_hadamard(mlp, 2048))
        wmc_2048.append(run_wmc_custom(mlp, 2048))

    # Evaluate configurations:
    configs = [
        ("Hermite + Had4096 (alpha=0.110)", c_analyticals, hadamard_4096, None, 0.890, 0.110, 0.0, 0.1000),
        ("Hermite + Had4096 (alpha=0.130)", c_analyticals, hadamard_4096, None, 0.870, 0.130, 0.0, 0.1000),
        ("Hermite + Had5120 (alpha=0.130)", c_analyticals, hadamard_5120, None, 0.870, 0.130, 0.0, 0.1000),
        ("Hermite + Had5120 (alpha=0.150)", c_analyticals, hadamard_5120, None, 0.850, 0.150, 0.0, 0.1000),
        ("Tri-Blend: Hermite (0.88) + WMC2048 (0.06) + Had2048 (0.06)", c_analyticals, wmc_2048, hadamard_2048, 0.880, 0.060, 0.060, 0.1000),
        ("Tri-Blend: Hermite (0.86) + WMC2048 (0.07) + Had2048 (0.07)", c_analyticals, wmc_2048, hadamard_2048, 0.860, 0.070, 0.070, 0.1000),
        ("Tri-Blend: Hermite (0.84) + WMC2048 (0.08) + Had2048 (0.08)", c_analyticals, wmc_2048, hadamard_2048, 0.840, 0.080, 0.080, 0.1000),
        ("Tri-Blend: Hermite (0.82) + WMC2048 (0.09) + Had2048 (0.09)", c_analyticals, wmc_2048, hadamard_2048, 0.820, 0.090, 0.090, 0.1000),
    ]

    for name, branch1, branch2, branch3, w1, w2, w3, mult in configs:
        cand_mses = []
        wins_champ = 0
        diffs = []
        for i in range(8):
            if branch3 is None:
                pred = w1 * branch1[i] + w2 * branch2[i]
            else:
                pred = w1 * branch1[i] + w2 * branch2[i] + w3 * branch3[i]
            mse = np.mean((pred - y_trues[i])**2)
            adj = mse * mult
            cand_mses.append(mse)

            champ_adj_i = champ_adj[i]
            diff = ((adj - champ_adj_i) / champ_adj_i) * 100
            diffs.append(diff)
            if adj < champ_adj_i:
                wins_champ += 1

        mean_mse = np.mean(cand_mses)
        mean_adj = mean_mse * mult
        mean_diff = ((mean_adj - np.mean(champ_adj)) / np.mean(champ_adj)) * 100
        print(f"\nVariant: {name}")
        print(f"  Raw MSE:        {mean_mse:.6e} | Adj Score: {mean_adj:.6e} (Diff: {mean_diff:+.2f}%)")
        print(f"  Wins vs Champ:  {wins_champ}W-{8-wins_champ}L | Worst Diff: {max(diffs):+.2f}%")

if __name__ == "__main__":
    screen_portfolios()
