"""Phase 2: Randomized Hadamard Spherical Cubature + Hermite Covariance.

Combines:
1. Second-order Hermite cross-covariance correction on the analytical branch.
2. Randomized Hadamard frames on S^1023 with exact Chi(1024) expected radius
   and antipodal symmetry (4,096 samples, zero eigh FLOPs).
"""

from __future__ import annotations

import math
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_COVARIANCE_WEIGHT = fnp.asarray(0.75, dtype=fnp.float32)
_MC_WEIGHT = fnp.asarray(0.25, dtype=fnp.float32)
_GAMMA = fnp.asarray(0.45, dtype=fnp.float32)
_INV_4PI = fnp.asarray(0.07957747154594767, dtype=fnp.float32)
_ZERO = fnp.asarray(0.0, dtype=fnp.float32)
_MU_R = fnp.asarray(31.9921884548, dtype=fnp.float32)

# Precomputed Sylvester Hadamard matrix
_HADAMARD_1024: fnp.ndarray | None = None


def _init_hadamard() -> fnp.ndarray:
    global _HADAMARD_1024
    if _HADAMARD_1024 is None:
        h = np.array([[1.0]], dtype=np.float32)
        for _ in range(10):
            h = np.block([[h, h], [h, -h]])
        h *= (1.0 / 32.0)
        _HADAMARD_1024 = fnp.asarray(h, dtype=fnp.float32)
    return _HADAMARD_1024


def _hermite_gain_covariance(mlp: MLP) -> fnp.ndarray:
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

        cov_linear = fnp.multiply(fnp.outer(gain, gain), cov_pre)
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


def _hadamard_spherical_mc(mlp: MLP) -> fnp.ndarray:
    width, depth = mlp.width, mlp.depth
    rng = fnp.random.default_rng(mlp.seed)

    if width == 1024:
        H = _init_hadamard()
        # Draw random Rademacher signs (+1/-1)
        s1_1 = fnp.where(rng.standard_normal((width, 1)) >= 0, 1.0, -1.0).astype(fnp.float32)
        s2_1 = fnp.where(rng.standard_normal((1, width)) >= 0, 1.0, -1.0).astype(fnp.float32)
        q1 = (s1_1 * H * s2_1) * _MU_R

        s1_2 = fnp.where(rng.standard_normal((width, 1)) >= 0, 1.0, -1.0).astype(fnp.float32)
        s2_2 = fnp.where(rng.standard_normal((1, width)) >= 0, 1.0, -1.0).astype(fnp.float32)
        q2 = (s1_2 * H * s2_2) * _MU_R

        x = fnp.concatenate((q1, -q1, q2, -q2), axis=0)
        count = 4_096
    else:
        # Fallback for validation dummy MLP shapes
        count = 64
        half = count // 2
        x_half = fnp.asarray(rng.standard_normal((half, width)), dtype=fnp.float32)
        x = fnp.concatenate((x_half, -x_half), axis=0)

    scale = fnp.asarray(1.0 / count, dtype=fnp.float32)
    first_weight = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
    activations = fnp.maximum(x @ first_weight, _ZERO)

    rows = [fnp.sum(activations, axis=0) * scale]
    for layer in range(1, depth):
        weight = fnp.asarray(mlp.weights[layer], dtype=fnp.float32)
        activations = fnp.maximum(activations @ weight, _ZERO)
        rows.append(fnp.sum(activations, axis=0) * scale)
    return fnp.stack(rows, axis=0)


class Estimator(BaseEstimator):
    def setup(self, context: SetupContext) -> None:
        if context.width == 1024:
            _init_hadamard()

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        covariance = _hermite_gain_covariance(mlp)
        cubature = _hadamard_spherical_mc(mlp)
        return _COVARIANCE_WEIGHT * covariance + _MC_WEIGHT * cubature
