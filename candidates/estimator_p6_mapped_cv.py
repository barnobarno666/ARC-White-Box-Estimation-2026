"""estimator_p6_mapped_cv.py

Phase 6 Experimental Candidate: Layer-14 Mapped Control Variate Estimator.
Combines:
1. Dual-Kernel Blended Hermite Covariance Propagation.
2. Layer-14 Error Response Mapping:
   delta_14 = mean(H_14) - mu_14
   correction_15 = (delta_14 @ W_15) * p_15
   m_corr = m_raw - correction_15
3. Universal Scale Projection Calibration (s_0 = 0.998319).
4. Gauss-Markov Harmonic Precision Blending (alpha = 0.110).

Target utilization: ~9.81% (multiplier floor 0.1000).
Residual time: < 0.25s.
All arithmetic implemented in allowed flopscope.numpy operations.

Status & Verification Note:
- Actual metered whest run score: 1.383347e-07 (1W - 7L vs frozen control 1.223643e-07).
- Cause: Online K3 center generation is computationally heavy (~60s/MLP in Python),
  so this self-contained candidate runs the baseline Hermite covariance center. The baseline
  center bias (1.8e-07) propagates into Layer 15, causing a regression.
- Offline diagnostic with precomputed Phase 6 K3 centers showed 4.3702e-08, but that
  requires precomputed prediction tables which cannot be deployed into a standalone candidate.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_TOTAL_SAMPLES = 4_200
_ZERO = fnp.asarray(0.0, dtype=fnp.float32)
_ALPHA_MC = fnp.asarray(0.110, dtype=fnp.float32)
_ONE_MINUS_ALPHA = fnp.asarray(1.0 - 0.110, dtype=fnp.float32)
_S_PRIOR = fnp.asarray(0.998319, dtype=fnp.float32)

# Blended kernel coefficients for lambda = 0.20:
_C_CENTERED = fnp.asarray(0.80 * 0.20 * 0.07957747154594767, dtype=fnp.float32)
_C_THRESH = fnp.asarray(0.20 * 0.5, dtype=fnp.float32)


def _blended_hermite_covariance(mlp: MLP) -> fnp.ndarray:
    """Gain-covariance propagation with single-pass dual-kernel blended Hermite quadratic term."""
    width = mlp.width
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
    rows = []

    for weight in mlp.weights:
        w = fnp.asarray(weight, dtype=fnp.float32)
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), fnp.asarray(1e-12, dtype=fnp.float32))
        sigma_pre = fnp.sqrt(var_pre)
        a = mu_pre / sigma_pre
        phi = flops.stats.norm.pdf(a).astype(fnp.float32)
        cdf = flops.stats.norm.cdf(a).astype(fnp.float32)

        mu = mu_pre * cdf + sigma_pre * phi
        second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
        var_post = fnp.maximum(second - mu * mu, _ZERO)
        gain = fnp.where(sigma_pre > fnp.asarray(1e-12, dtype=fnp.float32), cdf, _ZERO)

        # 1. Base linear term
        cov_linear = fnp.multiply(fnp.outer(gain, gain), cov_pre)

        # 2. Dual-kernel blended Hermite quadratic term
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


def _mapped_cv_wmc(mlp: MLP, analytic_stack: fnp.ndarray) -> fnp.ndarray:
    """Whitened antithetic MC with Layer-14 error mapping control variate."""
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
    h14_mean = None
    p15 = None

    for layer in range(1, depth):
        weight = fnp.asarray(mlp.weights[layer], dtype=fnp.float32)
        pre = activations @ weight
        if layer == 15:
            # Measure layer 15 gate probabilities
            gate15 = fnp.where(pre > _ZERO, fnp.asarray(1.0, dtype=fnp.float32), _ZERO)
            p15 = fnp.sum(gate15, axis=0) * scale
        activations = fnp.maximum(pre, _ZERO)
        act_mean = fnp.sum(activations, axis=0) * scale
        rows.append(act_mean)

        if layer == 14:
            h14_mean = act_mean

    # Compute Layer 14 error response mapping to Layer 15
    w15 = fnp.asarray(mlp.weights[15], dtype=fnp.float32)
    delta14 = h14_mean - analytic_stack[14]
    response15 = (delta14 @ w15) * p15

    # Correct final layer MC estimate
    raw_final_mc = rows[-1]
    corrected_final_mc = raw_final_mc - response15

    rows[-1] = corrected_final_mc
    return fnp.stack(rows, axis=0)


class Estimator(BaseEstimator):
    """Phase 6 Layer-14 Mapped Control Variate Estimator."""

    def setup(self, ctx: SetupContext) -> None:
        dummy_w = fnp.eye(32, dtype=fnp.float32)
        dummy_cov = flops.as_symmetric(fnp.eye(32, dtype=fnp.float32), symmetry=(0, 1))
        _ = fnp.einsum("ij,ia,jb->ab", dummy_cov, dummy_w, dummy_w)
        _ = flops.stats.norm.cdf(fnp.zeros(32, dtype=fnp.float32))
        _, _ = fnp.linalg.eigh(fnp.eye(32, dtype=fnp.float32))

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        c = _blended_hermite_covariance(mlp)
        m = _mapped_cv_wmc(mlp, c)

        c_final = c[-1]
        m_final = m[-1]

        c_calibrated_final = _S_PRIOR * c_final
        final_pred = _ONE_MINUS_ALPHA * c_calibrated_final + _ALPHA_MC * m_final

        rows = []
        for layer_idx in range(mlp.depth - 1):
            rows.append(c[layer_idx])
        rows.append(final_pred)
        return fnp.stack(rows, axis=0)
