"""P5-01: Layer-wise analytic error growth + C4 residual law + within-sample R2."""
import numpy as np
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
import sys
sys.path.insert(0, ".")
from estimator import _S_PRIOR
from scipy.stats import norm

def blended_cov_numpy(mlp, lam=0.20):
    """Numpy replica of blended hermite covariance (returns mus, sigmas, a per layer)."""
    width = mlp.width
    mu = np.zeros(width, dtype=np.float64)
    cov = np.eye(width, dtype=np.float64)
    mus = []
    sigmas = []
    avals = []
    C_CENTERED = 0.80 * 0.20 * 0.07957747154594767
    C_THRESH = 0.20 * 0.5
    for weight in mlp.weights:
        w = np.array(weight, dtype=np.float64)
        mu_pre = w.T @ mu
        cov_pre = w.T @ (cov @ w)
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sigma_pre = np.sqrt(var_pre)
        a = mu_pre / sigma_pre
        phi = norm.pdf(a)
        cdf = norm.cdf(a)
        mu = mu_pre * cdf + sigma_pre * phi
        second = (mu_pre**2 + var_pre) * cdf + mu_pre * sigma_pre * phi
        var_post = np.maximum(second - mu**2, 0.0)
        gain = np.where(sigma_pre > 1e-12, cdf, 0.0)
        cov_linear = (gain[:, None] * gain[None, :]) * cov_pre
        inv_sigma = np.where(sigma_pre > 1e-12, 1.0/sigma_pre, 0.0)
        u_thresh = np.where(sigma_pre > 1e-12, phi/sigma_pre, 0.0)
        kernel = C_CENTERED * np.outer(inv_sigma, inv_sigma) + C_THRESH * np.outer(u_thresh, u_thresh)
        cov_quad = kernel * (cov_pre * cov_pre)
        cov = cov_linear + cov_quad
        np.fill_diagonal(cov, var_post)
        mus.append(mu.copy()); sigmas.append(sigma_pre.copy()); avals.append(a.copy())
    return np.stack(mus), np.stack(sigmas), np.stack(avals)

def main():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)
    s0 = 0.998319
    print("=== P5-01 Layer-wise analytic MSE (s0 scaled) ===")
    layer_mses = []
    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_true_all = np.array(row["all_layer_means"], dtype=np.float64)  # (16,1024)
        mus, sigmas, avals = blended_cov_numpy(mlp)
        mses = [float(np.mean((s0*mus[l] - y_true_all[l])**2)) for l in range(16)]
        layer_mses.append(mses)
        print(f"{row['mlp_name']}: " + " ".join(f"L{l}={m:.2e}" for l, m in enumerate(mses)))
    mean_mses = np.mean(np.array(layer_mses), axis=0)
    print("MEAN per layer: " + " ".join(f"L{l}={m:.2e}" for l, m in enumerate(mean_mses)))
    # ratio of L14 to L15
    print(f"L14 mean={mean_mses[14]:.4e} L15 mean={mean_mses[15]:.4e} ratio={mean_mses[14]/mean_mses[15]:.3f}")

    print("\n=== C4 residual law: residual vs a, sigma bins (final layer) ===")
    # pool all neurons across 8 MLPs: features a, sigma, phi, firing prob=cdf; target residual r = y - s0*c
    all_a, all_sig, all_cdf, all_r, all_c = [], [], [], [], []
    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_true = np.array(row["all_layer_means"][-1], dtype=np.float64)
        mus, sigmas, avals = blended_cov_numpy(mlp)
        c = s0 * mus[-1]; a = avals[-1]; sig = sigmas[-1]; cdf = norm.cdf(a)
        r = y_true - c
        all_a.append(a); all_sig.append(sig); all_cdf.append(cdf); all_r.append(r); all_c.append(c)
    A = np.concatenate(all_a); S = np.concatenate(all_sig); P = np.concatenate(all_cdf)
    R = np.concatenate(all_r); C = np.concatenate(all_c)
    print(f"resid std={np.std(R):.4e} mean={np.mean(R):.4e} |c| rms={np.sqrt(np.mean(C**2)):.4f}")
    # bin by a
    for lo, hi, name in [(-99,-2.5,"dead"), (-2.5,2.5,"kink"), (2.5,99,"on")]:
        m = (A>=lo)&(A<hi)
        print(f" bin {name}: n={m.sum()} mean_r={np.mean(R[m]):+.4e} std_r={np.std(R[m]):.4e} mse_contrib={np.mean(R[m]**2):.4e} frac_E={np.sum(R[m]**2)/np.sum(R**2):.3f}")
    # correlation of r with features
    for fname, F in [("a",A),("sigma",S),("Phi(a)",P),("c",C),("a*phi",A*norm.pdf(A))]:
        corr = np.corrcoef(F, R)[0,1]
        print(f" corr(r,{fname})={corr:+.4f}")
    # linear fit r ~ b0 + b1*c (global, all-MLP) and LOO
    print("\n--- LOO test: global additive r~p(c) vs multiplicative s0 ---")
    base_mses, corr_mses = [], []
    for hold in range(8):
        tr_r = np.concatenate([all_r[j] for j in range(8) if j!=hold])
        tr_c = np.concatenate([all_c[j] for j in range(8) if j!=hold])
        # fit r = b0 + b1*c (2 params) on train
        Xtr = np.stack([np.ones_like(tr_c), tr_c], axis=1)
        coef, *_ = np.linalg.lstsq(Xtr, tr_r, rcond=None)
        te_r, te_c = all_r[hold], all_c[hold]
        base = float(np.mean(te_r**2))
        pred = te_r - (coef[0] + coef[1]*te_c)
        corr_mse = float(np.mean(pred**2))
        base_mses.append(base); corr_mses.append(corr_mse)
        print(f" hold {hold}: base={base:.4e} corrected={corr_mse:.4e} diff={(corr_mse-base)/base*100:+.2f}% coef={coef}")
    print(f"MEAN base={np.mean(base_mses):.4e} corr={np.mean(corr_mses):.4e} gain={(np.mean(base_mses)-np.mean(corr_mses))/np.mean(base_mses)*100:+.2f}%")

if __name__ == "__main__":
    main()
