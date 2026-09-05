"""P4.6-03b Lane E last stand: SAMPLED-skewness final-layer correction.

Untested cumulant (fixed-kurtosis died in 03a, both signs). Uses the MC
branch's OWN whitened samples (zero extra sampling FLOPs, lawful, no target
contact): per-neuron sample skewness g1 of final preactivations Z_15, then
  mu_15 = mu_Gauss_15 - eta * sigma_pre * g1 * a * phi(a) / 6
(derived Edgeworth skew term; verified by direct Hermite integration.)
eta in {0.5, 1.0}. Same aggressive gate: beat control on BOTH mlps, mean>=3%.
"""
import numpy as np
from scipy.stats import norm
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP

S0 = 0.998319

def analytic_final(mlp, lam=0.20):
    width = mlp.width
    mu = np.zeros(width)
    cov = np.eye(width)
    C_C = (1 - lam) * 0.20 * 0.07957747154594767
    C_T = lam * 0.5
    for w_ in mlp.weights:
        w = np.array(w_, dtype=np.float64)
        mu_pre = w.T @ mu
        cov_pre = w.T @ (cov @ w)
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sig = np.sqrt(var_pre)
        a = mu_pre / sig
        phi = norm.pdf(a)
        cdf = norm.cdf(a)
        mu = mu_pre * cdf + sig * phi
        second = (mu_pre ** 2 + var_pre) * cdf + mu_pre * sig * phi
        var_post = np.maximum(second - mu * mu, 0.0)
        cov = (cdf[:, None] * cdf[None, :]) * cov_pre
        inv = 1.0 / sig
        u = phi / sig
        K = C_C * np.outer(inv, inv) + C_T * np.outer(u, u)
        cov = cov + K * (cov_pre * cov_pre)
        np.fill_diagonal(cov, var_post)
    return mu, mu_pre, sig, a, phi

def whitened_final(mlp, n=4200):
    d = mlp.width
    half = n // 2
    rng = np.random.default_rng(mlp.seed)
    xh = rng.standard_normal((half, d))
    x = np.concatenate([xh, -xh], axis=0)
    gram = (x.T @ x) / float(n)
    ev, U = np.linalg.eigh(gram)
    ev = np.maximum(ev, 1e-6)
    W = (U * (ev ** -0.5)) @ U.T
    act = np.maximum(x @ (W @ np.array(mlp.weights[0], dtype=np.float64)), 0.0)
    for l in range(1, mlp.depth - 1):
        act = np.maximum(act @ np.array(mlp.weights[l], dtype=np.float64), 0.0)
    Z = act @ np.array(mlp.weights[-1], dtype=np.float64)
    return Z

def main():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)
    print("=== P4.6-03b sampled-skewness correction (2 MLPs) ===")
    for i in [0, 1]:
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = np.array(row["final_means"], dtype=np.float64)
        mu_g, mu_pre, sig, a, phi = analytic_final(mlp)
        c0 = S0 * mu_g
        m0 = float(np.mean((c0 - y) ** 2))
        Z = whitened_final(mlp)
        zc = Z - Z.mean(axis=0)
        m2 = (zc ** 2).mean(axis=0) + 1e-12
        g1 = (zc ** 3).mean(axis=0) / (m2 ** 1.5)
        line = f" {row['mlp_name']}: control={m0:.4e} | skew_mean={g1.mean():+.4f} skew_std={g1.std():.4f}"
        for eta in [0.5, 1.0]:
            mu_c = S0 * (mu_g - eta * sig * g1 * a * phi / 6.0)
            mg = float(np.mean((mu_c - y) ** 2))
            line += f" | eta={eta}: {mg:.4e} ({(mg-m0)/m0*100:+.2f}%)"
        print(line)

if __name__ == "__main__":
    main()
