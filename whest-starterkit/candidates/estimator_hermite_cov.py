"""Phase 2: Second-order Hermite-corrected Covariance + WMC (N=4200).

Adds the quadratic Hermite term (1/(4*pi)) * rho^2 to the post-ReLU covariance
matrix, compensating for the systematic underestimation of the linear gain formula.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator
from whestbench.domain import MLP

_TOTAL_SAMPLES = 4_200
_COVARIANCE_WEIGHT = fnp.asarray(0.70, dtype=fnp.float32)
_MC_WEIGHT = fnp.asarray(0.30, dtype=fnp.float32)
_GAMMA = fnp.asarray(0.50, dtype=fnp.float32)
_INV_4PI = fnp.asarray(0.07957747154594767, dtype=fnp.float32)  # 1 / (4 * pi)
_ZERO = fnp.asarray(0.0, dtype=fnp.float32)


def _hermite_gain_covariance(mlp: MLP) -> fnp.ndarray:
    """Gain-covariance propagation with 2nd-order Hermite cross-covariance correction."""
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
        alpha = mu_pre / sigma_pre
        phi = flops.stats.norm.pdf(alpha).astype(fnp.float32)
        cdf = flops.stats.norm.cdf(alpha).astype(fnp.float32)

        mu = mu_pre * cdf + sigma_pre * phi
        second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
        var_post = fnp.maximum(second - mu * mu, _ZERO)
        gain = fnp.where(sigma_pre > fnp.asarray(1e-12, dtype=fnp.float32), cdf, _ZERO)

        # Base linear term
        cov_linear = fnp.multiply(fnp.outer(gain, gain), cov_pre)

        # 2nd-order Hermite quadratic term: gamma * (1/(4*pi)) * rho^2 * (sigma_i * sigma_j)
        inv_sigma = fnp.where(sigma_pre > 1e-12, 1.0 / sigma_pre, _ZERO)
        rho = fnp.multiply(fnp.outer(inv_sigma, inv_sigma), cov_pre)
        rho2 = rho * rho
        sigma_outer = fnp.outer(sigma_pre, sigma_pre)
        cov_quad = _INV_4PI * rho2 * sigma_outer

        cov = cov_linear + _GAMMA * cov_quad
        fnp.fill_diagonal(cov, var_post)
        cov = flops.as_symmetric(cov, symmetry=(0, 1))
        rows.append(mu)

    return fnp.stack(rows, axis=0)


def _whitened_antithetic_mc(mlp: MLP) -> fnp.ndarray:
    """Moment-matched MC, with the whitening transform fused into layer one."""
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
    for layer in range(1, depth):
        weight = fnp.asarray(mlp.weights[layer], dtype=fnp.float32)
        activations = fnp.maximum(activations @ weight, _ZERO)
        rows.append(fnp.sum(activations, axis=0) * scale)
    return fnp.stack(rows, axis=0)


class Estimator(BaseEstimator):
    """Hermite-corrected covariance + WMC blend."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        covariance = _hermite_gain_covariance(mlp)
        monte_carlo = _whitened_antithetic_mc(mlp)
        return _COVARIANCE_WEIGHT * covariance + _MC_WEIGHT * monte_carlo
