"""Phase 2: Pure Whitened Antithetic MC (weight = 1.0, N=4200)."""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator
from whestbench.domain import MLP

_TOTAL_SAMPLES = 4_200
_ZERO = fnp.asarray(0.0, dtype=fnp.float32)


def _whitened_antithetic_mc(mlp: MLP) -> fnp.ndarray:
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
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        return _whitened_antithetic_mc(mlp)
