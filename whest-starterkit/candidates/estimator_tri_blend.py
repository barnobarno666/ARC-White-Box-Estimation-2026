"""Phase 2: Tri-Blend (Hermite Covariance + Isotropic WMC + Hadamard Cubature).

Combines:
1. Analytical Branch (72%): 2nd-order Hermite-corrected gain covariance.
2. Isotropic MC (14%): Whitened antithetic Gaussian sampling (N=2,500).
3. Cubature (14%): Randomized Hadamard frame on S^1023 (N=2,048).
"""

from __future__ import annotations

import math
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_W_COV = fnp.asarray(0.72, dtype=fnp.float32)
_W_WMC = fnp.asarray(0.14, dtype=fnp.float32)
_W_HAD = fnp.asarray(0.14, dtype=fnp.float32)

_GAMMA = fnp.asarray(0.45, dtype=fnp.float32)
_INV_4PI = fnp.asarray(0.07957747154594767, dtype=fnp.float32)
_ZERO = fnp.asarray(0.0, dtype=fnp.float32)
_MU_R = fnp.asarray(31.9921884548, dtype=fnp.float32)

_N_WMC = 2_500
_N_HAD = 2_048

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


def _whitened_antithetic_mc(mlp: MLP) -> fnp.ndarray:
    width, depth = mlp.width, mlp.depth
    count = _N_WMC
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


def _hadamard_spherical_mc(mlp: MLP) -> fnp.ndarray:
    width, depth = mlp.width, mlp.depth
    rng = fnp.random.default_rng(mlp.seed + 100)

    if width == 1024:
        H = _init_hadamard()
        s1 = fnp.where(rng.standard_normal((width, 1)) >= 0, 1.0, -1.0).astype(fnp.float32)
        s2 = fnp.where(rng.standard_normal((1, width)) >= 0, 1.0, -1.0).astype(fnp.float32)
        q = (s1 * H * s2) * _MU_R
        x = fnp.concatenate((q, -q), axis=0)
        count = _N_HAD
    else:
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
        cov = _hermite_gain_covariance(mlp)
        wmc = _whitened_antithetic_mc(mlp)
        had = _hadamard_spherical_mc(mlp)
        return _W_COV * cov + _W_WMC * wmc + _W_HAD * had
