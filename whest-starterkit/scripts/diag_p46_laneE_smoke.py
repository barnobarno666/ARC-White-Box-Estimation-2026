"""P4.6-03 Lane E smoke: final-layer-only Edgeworth kurtosis correction.

P4-12 is NOT a valid kill of this idea [derived]: it used a fabricated schedule
gamma_2(l) = 0.0042*l (guess, nothing measured), injected it recurrently at all
16 layers (16x compounding), and fed corrected mu into an uncorrected `second`
moment (inconsistent variance). This smoke tests the honest version:
  mu_15 = mu_Gauss_15 + sigma_pre * (gamma2/24) * (a^2-1) * phi(a)
applied ONCE at the final layer, fixed universal gamma2 (measured terminal
~0.05-0.07), s0 held fixed. Skew term derived as -sigma*gamma1*a*phi/6 but
measured mean skew ~0, so gamma1 = 0 here.

SMOKE GATE (aggressive, per user): candidate cov-branch must beat control
cov-branch on BOTH smoke MLPs with mean improvement >= 3%. Else Lane E dies.
"""
import numpy as np
from scipy.stats import norm
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP

S0 = 0.998319

def blended_cov_final(mlp, lam=0.20, gamma2_final=None):
    """Returns s0-scaled final mean. gamma2_final=None -> control."""
    width = mlp.width
    mu = np.zeros(width)
    cov = np.eye(width)
    C_C = (1 - lam) * 0.20 * 0.07957747154594767
    C_T = lam * 0.5
    L = mlp.depth
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
        if gamma2_final is not None and li == L - 1:
            mu = mu + sig * (gamma2_final / 24.0) * (a * a - 1.0) * phi
        second = (mu_pre ** 2 + var_pre) * cdf + mu_pre * sig * phi
        var_post = np.maximum(second - mu * mu, 0.0)
        cov = (cdf[:, None] * cdf[None, :]) * cov_pre
        inv = 1.0 / sig
        u = phi / sig
        K = C_C * np.outer(inv, inv) + C_T * np.outer(u, u)
        cov = cov + K * (cov_pre * cov_pre)
        np.fill_diagonal(cov, var_post)
    return S0 * mu

def main():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)
    print("=== P4.6 Lane E smoke: final-layer-only kurtosis (2 MLPs) ===")
    for lam, lname in [(0.20, "champ-branch"), (0.50, "bestmean-branch")]:
        print(f"--- base lam={lam} ({lname}) ---")
        for i in [0, 1]:
            row = ds[i]
            mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
            y = np.array(row["final_means"], dtype=np.float64)
            c0 = blended_cov_final(mlp, lam=lam)
            m0 = float(np.mean((c0 - y) ** 2))
            line = f" {row['mlp_name']}: control={m0:.4e}"
            for g2 in [0.03, 0.057, 0.10, -0.057, -0.10]:
                cg = blended_cov_final(mlp, lam=lam, gamma2_final=g2)
                mg = float(np.mean((cg - y) ** 2))
                line += f" | g2={g2}: {mg:.4e} ({(mg-m0)/m0*100:+.2f}%)"
            print(line)

if __name__ == "__main__":
    main()
