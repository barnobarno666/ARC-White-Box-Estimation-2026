"""P4.6-05 Lane F smoke: shrunk MEAN-restart at k=12 (0.9 analytic + 0.1 sampled).

B2 killed pure-truth restart (+60-104%: off-manifold). This tests the untested
middle: 90% self-consistent analytic state + 10% unbiased drift correction,
covariance untouched (mean-only = cheapest, PSD-safe). Numpy, 2 MLPs.
BASE = lam05 + final skew eta=0.25 + s0 (eta025 equivalent).
GATE: beat base on BOTH mlps, mean >= 2%. Else kill Lane F permanently.
"""
import numpy as np
from scipy.stats import norm
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP

S0 = 0.998319
LAM = 0.50
C_C = (1 - LAM) * 0.20 * 0.07957747154594767
C_T = LAM * 0.5
N = 4200
K = 12
SHRINK = 0.9


def pilot(mlp):
    d = mlp.width
    half = N // 2
    rng = np.random.default_rng(mlp.seed)
    xh = rng.standard_normal((half, d))
    x = np.concatenate([xh, -xh], axis=0)
    gram = (x.T @ x) / float(N)
    ev, U = np.linalg.eigh(gram)
    ev = np.maximum(ev, 1e-6)
    Wht = (U * (ev ** -0.5)) @ U.T
    act = np.maximum(x @ (Wht @ np.array(mlp.weights[0], dtype=np.float64)), 0.0)
    means = [act.mean(axis=0)]
    for l in range(1, mlp.depth):
        w = np.array(mlp.weights[l], dtype=np.float64)
        z = act @ w
        if l == mlp.depth - 1:
            Zfin = z
        act = np.maximum(z, 0.0)
        means.append(act.mean(axis=0))
    return means, Zfin


def run(mlp, means, Zfin, restart):
    width = mlp.width
    mu = np.zeros(width)
    cov = np.eye(width)
    sig_last = a_last = phi_last = None
    for li, w_ in enumerate(mlp.weights):
        w = np.array(w_, dtype=np.float64)
        mu_pre = w.T @ mu
        cov_pre = w.T @ (cov @ w)
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sig = np.sqrt(var_pre)
        a = mu_pre / sig
        phi = norm.pdf(a)
        cdf = norm.cdf(a)
        mu = mu_pre * cdf + sig * phi
        if restart and li == K:
            mu = SHRINK * mu + (1 - SHRINK) * means[li]
        second = (mu_pre ** 2 + var_pre) * cdf + mu_pre * sig * phi
        var_post = np.maximum(second - mu * mu, 0.0)
        cov = (cdf[:, None] * cdf[None, :]) * cov_pre
        inv = 1.0 / sig
        u = phi / sig
        KK = C_C * np.outer(inv, inv) + C_T * np.outer(u, u)
        cov = cov + KK * (cov_pre * cov_pre)
        np.fill_diagonal(cov, var_post)
        sig_last, a_last, phi_last = sig, a, phi
    zc = Zfin - Zfin.mean(axis=0)
    mv = np.maximum((zc ** 2).mean(axis=0), 1e-12)
    g1f = (zc ** 3).mean(axis=0) / (mv ** 1.5)
    mu = mu - 0.25 * sig_last * g1f * a_last * phi_last / 6.0
    return S0 * mu


def main():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)
    print(f"=== P4.6-05 Lane F smoke: shrunk mean-restart k={K} ({SHRINK}/{(1-SHRINK):.1f}) ===")
    for i in [0, 1]:
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = np.array(row["final_means"], dtype=np.float64)
        means, Zfin = pilot(mlp)
        mb = float(np.mean((run(mlp, means, Zfin, False) - y) ** 2))
        mf = float(np.mean((run(mlp, means, Zfin, True) - y) ** 2))
        print(f" {row['mlp_name']}: base={mb:.4e} restart={mf:.4e} ({(mf-mb)/mb*100:+.2f}%)", flush=True)

if __name__ == "__main__":
    main()
