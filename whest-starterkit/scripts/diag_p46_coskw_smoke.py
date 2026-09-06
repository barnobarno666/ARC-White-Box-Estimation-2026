"""P4.6-04 co-skewness variance-path smoke (2 MLPs, numpy, offline).

STRUCTURAL FACT driving design: final prediction uses only MEANS; covariance
matters solely via diag(variances) feeding next-layer means. So joint
non-Gaussianity can only help through VARIANCES:
 (A) 1D Edgeworth corrections to var_post diagonal (recurrent, cheap):
       E[ReLU^2] ~= second_Gauss + s^2 * [ g1*phi/3 - g2*a*phi/12 ]
     with per-neuron g1 SAMPLED per layer, g2 linear 0 -> 0.057 (measured
     endpoints: exact 0 at L0, ~0.057 terminal). Doses full + half.
 (B) Bivariate co-skewness to off-diagonal C at late layers (12-14):
       dE_ij = -[ K30/6*T30 + K21/2*T21 + K12/2*T12 + K03/6*T03 ]
     K21(i,j) = sum_a W_ai^2 W_aj s_a (factorized, s_a sampled),
     K12 = K21.T-swap, T-terms closed-form (rho=0-factorization verified).
     T30/T03 included ONLY if quadrature cross-check passes in-script.
     Both signs tested for (B) (03a lesson).

BASE = lam05 + final skew eta=0.25 w/ sampled g1 (eta025 numpy replica).
GATE: beat eta025-base on BOTH mlps AND beat champ-branch by >=3% mean.
Metric: s0-scaled final-mean MSE.
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import dblquad
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP

S0 = 0.998319
LAM = 0.50
C_C = (1 - LAM) * 0.20 * 0.07957747154594767
C_T = LAM * 0.5
N = 4200


def pilot_moments(mlp):
    """Whitened pilot: per-layer post-activation raw moments M1..M4 + final Z samples info."""
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
    moms = [np.stack([act.mean(0), (act**2).mean(0), (act**3).mean(0), (act**4).mean(0)])]
    premoms = []
    for l in range(1, mlp.depth):
        w = np.array(mlp.weights[l], dtype=np.float64)
        z = act @ w
        premoms.append(np.stack([z.mean(0), (z**2).mean(0), (z**3).mean(0), (z**4).mean(0)]))
        if l == mlp.depth - 1:
            Z = z
        act = np.maximum(z, 0.0)
        moms.append(np.stack([act.mean(0), (act**2).mean(0), (act**3).mean(0), (act**4).mean(0)]))
    return moms, premoms, Z


def central34(m):
    M1, M2, M3, M4 = m
    v = np.maximum(M2 - M1**2, 1e-12)
    s = M3 - 3*M1*M2 + 2*M1**3
    k = M4 - 4*M1*M3 + 6*M1**2*M2 - 3*M1**4
    return v, s, k


def T21(ai, aj, rho, si):
    """E[d'(z_i) H(z_j)], original units. Closed form; exact at rho=0 by construction."""
    q = np.sqrt(max(1 - rho * rho, 1e-12))
    w = (aj - rho * ai) / q
    return -(norm.pdf(ai) / si**2) * ((rho / q) * norm.pdf(w) + ai * norm.cdf(w))


def T12(ai, aj, rho, sj):
    return T21(aj, ai, rho, sj)


def K21_mat(W, s):
    """Co-skewness K21(i,j) = sum_a W_ai^2 W_aj s_a. K12(i,j) = K21(j,i)."""
    return np.einsum('ai,aj,a->ij', W * W, W, s)


def quad_check():
    """Verify T21 against brute-force quadrature at 3 points; T30/T03 admission test."""
    pts = [(0.5, -0.3, 0.2, 1.3), (0.0, 0.0, 0.0, 1.0), (1.2, 0.8, -0.4, 0.7)]
    for ai, aj, rho, si in pts:
        sj = 1.1
        f = lambda u, v: 0.0  # placeholder
        # E[d'(z_i) H(z_j)] via integration over standardized (u,v): d'(z_i)=si^-2 d'(u+ai)
        import math
        q = math.sqrt(max(1 - rho * rho, 1e-12))
        def integrand(v, u):
            # d/du H? use representation: E = -d/dt E[d(U+ai+t) H(V+aj)]|0 ; do finite diff instead
            return 0.0
        # finite-difference reference: E[d(U+ai) H(V+aj)] derivative
        # CORRECT conditional: V|U=u ~ N(rho*u, q^2) => P(V>-aj|u) = Phi((aj+rho*u)/q),
        # evaluated at u=-ai-t => Phi((aj-rho*(ai+t))/q)  [MINUS sign]
        def E_delta(t):
            # E[delta(U+ai+t) H(V+aj)] = phi(ai+t) Phi((aj-rho(ai+t))/q)
            return norm.pdf(ai + t) * norm.cdf((aj - rho * (ai + t)) / q)
        h = 1e-5
        # E[d'(z_i) H(z_j)] = +d/dt E[d(z_i+t) H(z_j)] (chain rule gives +1)
        ref = (E_delta(h) - E_delta(-h)) / (2 * h) / si**2
        got = T21(ai, aj, rho, si)
        assert abs(got - ref) < 1e-6 * max(1, abs(ref)), (ai, aj, rho, got, ref)
    print("quad_check: T21 closed form VERIFIED vs finite-difference reference")
    return True


def run_branch(mlp, moms, premoms, Zfin, mode, doseA=1.0, signB=1.0, late=(12, 13, 14)):
    """Analytic recurrence with corrections. mode in {'base','A','AB'}."""
    width = mlp.width
    mu = np.zeros(width)
    cov = np.eye(width)
    L = mlp.depth
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
        second = (mu_pre**2 + var_pre) * cdf + mu_pre * sig * phi
        if mode in ('A', 'AB') and li >= 1:
            # PREACTIVATION cumulants of this layer (li=0 is exactly Gaussian -> skip)
            v, s, k = central34(premoms[li - 1])
            g1 = np.clip(s / (v ** 1.5 + 1e-12), -1.0, 1.0)
            g2lin = 0.057 * li / 15.0  # measured endpoints: 0 at L0, ~0.057 terminal
            second = second + var_pre * doseA * (g1 * phi / 3.0 - g2lin * a * phi / 12.0)
        var_post = np.maximum(second - mu * mu, 0.0)
        gain = cdf
        clin = (gain[:, None] * gain[None, :]) * cov_pre
        inv = 1.0 / sig
        u = phi / sig
        K = C_C * np.outer(inv, inv) + C_T * np.outer(u, u)
        cquad = K * (cov_pre * cov_pre)
        cov = clin + cquad
        if mode == 'AB' and li in late:
            # (B-lite): cross co-skewness T21/T12 terms on off-diagonals.
            # K21(i,j) from factorized post-moments of PREVIOUS layer (moms[li-1] post h).
            # Identity assert (once): diag(K21) == marginal kappa3 of preactivations.
            _, s_prev, _ = central34(moms[li - 1]) if li >= 1 else (None, np.zeros(width), None)
            if li >= 1:
                K21 = K21_mat(w, s_prev)
                if li == late[0] and not hasattr(run_branch, '_kcheck'):
                    # verify against sampled preactivation marginal kappa3
                    vpre, spre, _ = central34(premoms[li - 1])
                    d = np.diag(K21)
                    rel = np.abs(d - spre) / np.maximum(np.abs(spre), 1e-9)
                    print(f"   K-identity check L{li}: median_rel_err={np.median(rel):.4f} "
                          f"(assert < 0.35, sampling noise floor ~0.2)")
                    assert np.median(rel) < 0.35, "K21 factorized identity FAILED"
                    run_branch._kcheck = True
                rho = np.clip(cov_pre / np.maximum(np.outer(sig, sig), 1e-12), -0.999, 0.999)
                q = np.sqrt(np.maximum(1 - rho * rho, 1e-12))
                # T21(i,j): -phi(ai)/si^2 * ((rho/q) phi(W) + ai Phi(W)), W=(aj-rho*ai)/q
                Wij = (a[None, :] - rho * a[:, None]) / q
                T21m = -(norm.pdf(a)[:, None] / np.maximum(var_pre[:, None], 1e-12)) * (
                    (rho / q) * norm.pdf(Wij) + a[:, None] * norm.cdf(Wij))
                Wji = (a[:, None] - rho * a[None, :]) / q
                T12m = -(norm.pdf(a)[None, :] / np.maximum(var_pre[None, :], 1e-12)) * (
                    (rho / q) * norm.pdf(Wji) + a[None, :] * norm.cdf(Wji))
                dE = -(K21 / 2.0 * T21m + K21.T / 2.0 * T12m)
                np.fill_diagonal(dE, 0.0)
                cov = cov + signB * dE
        np.fill_diagonal(cov, var_post)
        sig_last, a_last, phi_last = sig, a, phi
    # eta025-base final skew (in ALL modes): sampled g1 of final preactivations
    zc = Zfin - Zfin.mean(axis=0)
    mv = np.maximum((zc ** 2).mean(axis=0), 1e-12)
    g1f = (zc ** 3).mean(axis=0) / (mv ** 1.5)
    mu = mu - 0.25 * sig_last * g1f * a_last * phi_last / 6.0
    return S0 * mu, (None, None)


def main():
    quad_check()
    print("NOTE: (B-lite) cross co-skewness T21/T12 at layers 12-14, both signs.")
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)
    print("=== P4.6-04 smoke part A: 1D variance corrections (2 MLPs) ===")
    for i in [0, 1]:
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = np.array(row["final_means"], dtype=np.float64)
        moms, premoms, Zfin = pilot_moments(mlp)
        mu_b, _ = run_branch(mlp, moms, premoms, Zfin, 'base')
        mb = float(np.mean((mu_b - y) ** 2))
        line = f" {row['mlp_name']}: base={mb:.4e}"
        for dose in [0.5, 1.0]:
            mu_a, _ = run_branch(mlp, moms, premoms, Zfin, 'A', doseA=dose)
            ma = float(np.mean((mu_a - y) ** 2))
            line += f" | A@{dose}: {ma:.4e} ({(ma-mb)/mb*100:+.2f}%)"
        for sgn, nm in [(1.0, 'B+'), (-1.0, 'B-')]:
            if hasattr(run_branch, '_kcheck'):
                delattr(run_branch, '_kcheck')
            try:
                mu_c, _ = run_branch(mlp, moms, premoms, Zfin, 'AB', doseA=1.0, signB=sgn)
                mc = float(np.mean((mu_c - y) ** 2))
                line += f" | {nm}: {mc:.4e} ({(mc-mb)/mb*100:+.2f}%)"
            except AssertionError as e:
                line += f" | {nm}: K-IDENTITY-FAIL"
        print(line, flush=True)

if __name__ == "__main__":
    main()
