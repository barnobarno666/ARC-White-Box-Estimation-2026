"""P4.6 Lane F deployable: eta025 stack + shrunk mean-restart at L12 (0.9/0.1).

Smoke P4.6-05 (2 MLPs, numpy): restart cut cov-branch MSE -5.94%/-0.95%
(mean -3.45% >= 2% gate -> PASS). Restart center = MC branch's OWN layer-12
mean (zero extra sampling FLOPs, lawful, no truth). Covariance untouched.
Design frozen a priori: k=12, keep=0.9. Alpha stays 0.110 (isolate mechanism).
"""
from __future__ import annotations
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_TOTAL_SAMPLES = 4_200
_ZERO = fnp.asarray(0.0, dtype=fnp.float32)
_HALF = fnp.asarray(0.5, dtype=fnp.float32)
_ALPHA_MC = fnp.asarray(0.110, dtype=fnp.float32)
_ONE_MINUS_ALPHA = fnp.asarray(1.0 - 0.110, dtype=fnp.float32)
_S_PRIOR = fnp.asarray(0.998319, dtype=fnp.float32)
_ETA_SKEW = fnp.asarray(0.25, dtype=fnp.float32)
_RESTART_LAYER = 12
_RESTART_KEEP = fnp.asarray(0.9, dtype=fnp.float32)
_SIX = fnp.asarray(6.0, dtype=fnp.float32)
_EPS = fnp.asarray(1e-12, dtype=fnp.float32)

_C_CENTERED = fnp.asarray(0.50 * 0.20 * 0.07957747154594767, dtype=fnp.float32)
_C_THRESH = fnp.asarray(0.50 * 0.5, dtype=fnp.float32)


def _covariance_with_final_stats(mlp: MLP, restart_mean=None):
    """Blended lam=0.50 + exact Cho-Saul L0 + shrunk mean-restart at L12.
    restart_mean: sampled layer-12 mean (or None). Returns (stack, s_pre_L, a_L, phi_L)."""
    width = mlp.width
    rows = []
    w0 = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
    cov_pre_0 = w0.T @ w0
    var_0 = fnp.maximum(fnp.diag(cov_pre_0), fnp.asarray(1e-12, dtype=fnp.float32))
    sigma_0 = fnp.sqrt(var_0)
    mu = sigma_0 / fnp.sqrt(fnp.asarray(6.283185307179586, dtype=fnp.float32))
    inv_s0 = 1.0 / sigma_0
    rho_0 = fnp.clip((inv_s0[:, None] * cov_pre_0) * inv_s0[None, :],
                     fnp.asarray(-0.999999, dtype=fnp.float32),
                     fnp.asarray(0.999999, dtype=fnp.float32))
    term1 = fnp.sqrt(fnp.asarray(1.0, dtype=fnp.float32) - rho_0 * rho_0)
    term2 = rho_0 * (fnp.asarray(1.5707963267948966, dtype=fnp.float32) + fnp.arcsin(rho_0))
    e_joint = (term1 + term2) * fnp.asarray(0.15915494309189535, dtype=fnp.float32)
    cov = (sigma_0[:, None] * e_joint) * sigma_0[None, :] - mu[:, None] * mu[None, :]
    fnp.fill_diagonal(cov, 0.5 * var_0 - mu * mu)
    cov = flops.as_symmetric(cov, symmetry=(0, 1))
    rows.append(mu)
    sigma_pre_last = a_last = phi_last = None
    for li, weight in enumerate(mlp.weights[1:], start=1):
        w = fnp.asarray(weight, dtype=fnp.float32)
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), fnp.asarray(1e-12, dtype=fnp.float32))
        sigma_pre = fnp.sqrt(var_pre)
        a = mu_pre / sigma_pre
        phi = flops.stats.norm.pdf(a).astype(fnp.float32)
        cdf = flops.stats.norm.cdf(a).astype(fnp.float32)
        mu = mu_pre * cdf + sigma_pre * phi
        if restart_mean is not None and li == _RESTART_LAYER:
            mu = _RESTART_KEEP * mu + (fnp.asarray(1.0, dtype=fnp.float32) - _RESTART_KEEP) * restart_mean
        second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
        var_post = fnp.maximum(second - mu * mu, _ZERO)
        gain = fnp.where(sigma_pre > fnp.asarray(1e-12, dtype=fnp.float32), cdf, _ZERO)
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
        sigma_pre_last, a_last, phi_last = sigma_pre, a, phi
    return fnp.stack(rows, axis=0), sigma_pre_last, a_last, phi_last


def _mc_with_final_preact(mlp: MLP):
    """Whitened antithetic MC. Returns (stack, Z_last preactivations)."""
    width, depth = mlp.width, mlp.depth
    count = _TOTAL_SAMPLES
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
    rows = [fnp.sum(activations, axis=0) * scale]
    z_last = None
    for layer in range(1, depth):
        weight = fnp.asarray(mlp.weights[layer], dtype=fnp.float32)
        pre = activations @ weight
        if layer == depth - 1:
            z_last = pre
        activations = fnp.maximum(pre, _ZERO)
        rows.append(fnp.sum(activations, axis=0) * scale)
    return fnp.stack(rows, axis=0), z_last


class Estimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:
        dummy_w = fnp.eye(32, dtype=fnp.float32)
        dummy_cov = flops.as_symmetric(fnp.eye(32, dtype=fnp.float32), symmetry=(0, 1))
        _ = fnp.einsum("ij,ia,jb->ab", dummy_cov, dummy_w, dummy_w)
        _ = flops.stats.norm.cdf(fnp.zeros(32, dtype=fnp.float32))
        _, _ = fnp.linalg.eigh(fnp.eye(32, dtype=fnp.float32))

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        m, z_last = _mc_with_final_preact(mlp)
        rm = m[_RESTART_LAYER] if mlp.depth > _RESTART_LAYER else None
        c, sigma_pre, a, phi = _covariance_with_final_stats(mlp, restart_mean=rm)
        # Sampled-skewness Edgeworth correction (frozen eta=0.25)
        n = fnp.asarray(float(_TOTAL_SAMPLES), dtype=fnp.float32)
        z_bar = fnp.sum(z_last, axis=0) / n
        zc = z_last - z_bar
        m2 = fnp.maximum(fnp.sum(zc * zc, axis=0) / n, _EPS)
        m3 = fnp.sum(zc * zc * zc, axis=0) / n
        g1 = m3 / (m2 * fnp.sqrt(m2) + _EPS)
        delta = -_ETA_SKEW * sigma_pre * g1 * a * phi / _SIX
        c_corr = _S_PRIOR * (c[-1] + delta)
        final_pred = _ONE_MINUS_ALPHA * c_corr + _ALPHA_MC * m[-1]
        rows = []
        for l in range(mlp.depth - 1):
            rows.append(c[l])
        rows.append(final_pred)
        return fnp.stack(rows, axis=0)
