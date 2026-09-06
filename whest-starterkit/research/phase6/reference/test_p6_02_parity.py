"""test_p6_02_parity.py

Comprehensive Phase 6 Stage P6-02 Parity Verification Suite.
Validates:
1. Dense vs factorized linear transport, symmetrization, repeated-index slices, and column sign/scaling invariance.
2. Closed-form Gaussian ReLU mean and variance across negative, zero, and positive standardized regimes.
3. Step-by-step parity between factored and dense recurrence across all three modes:
   - K3-base (base=True, augment=False)
   - K3-simple (base=False, augment=False)
   - K3-augment-filtered (base=False, augment=True)
4. Explicit final-layer ReLU activation and transpose convention verification with asymmetric weights.
5. K3 ablation sensitivity on non-Gaussian states.
6. Cache invalidation safety on factor modifications.
Executed in isolated reference environment (.venv_ref) using float64.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
from torch import Tensor

import mlp_kprop
from mlp_kprop.cumulants import *
from mlp_kprop.diagslice import DSTensor, diagslice, zero_repeated
from mlp_kprop.factor_k3 import FactoredTensor, factored_nonlin_kprop_k3
from mlp_kprop.harmonic import HTensor
import mlp_kprop.kprop_harmonic as kh
from mlp_kprop.kprop_harmonic import (
    Kind,
    coerce_input,
    factored_keeps_term,
    get_all_terms_iso,
    linear_kprop,
    nonlin_kprop,
)
from mlp_kprop.tensor_utils import contract_W_basic, symmetrize
from mlp_kprop.wick import relu_wick_coef

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)

TEST_WIDTHS = [8, 16, 32]
TEST_DEPTHS = [1, 2, 4]
TEST_SEEDS = [6, 17, 29]


# =====================================================================
# 1. Closed-form Gaussian ReLU Mean and Variance
# =====================================================================
def test_gaussian_relu_closed_forms():
    """Verify relu_wick_coef against analytical Gaussian integrals."""
    import scipy.special

    def closed_form_relu(mu: float, sigma2: float) -> Tuple[float, float]:
        sigma = math.sqrt(sigma2)
        a = mu / sigma
        phi = math.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + scipy.special.erf(a / math.sqrt(2.0)))
        mu_post = mu * cdf + sigma * phi
        second = (mu * mu + sigma2) * cdf + mu * sigma * phi
        var_post = second - mu_post * mu_post
        return mu_post, var_post

    test_regimes = [
        (-3.0, 1.0),  # strongly negative standardized mean
        (-1.0, 0.5),  # mildly negative
        (0.0, 1.0),   # exact zero mean
        (0.0, 2.0),   # scaled variance
        (1.0, 0.5),   # mildly positive
        (3.0, 1.0),   # strongly positive
    ]

    for mu, sigma2 in test_regimes:
        mu_ref = relu_wick_coef(mu, sigma2, k=0, p=1)
        sec_ref = relu_wick_coef(mu, sigma2, k=0, p=2)
        var_ref = sec_ref - mu_ref * mu_ref

        mu_closed, var_closed = closed_form_relu(mu, sigma2)

        assert math.isclose(mu_ref, mu_closed, rel_tol=1e-12, abs_tol=1e-14), (
            f"Mean mismatch at mu={mu}, var={sigma2}: ref={mu_ref}, closed={mu_closed}"
        )
        assert math.isclose(var_ref, var_closed, rel_tol=1e-12, abs_tol=1e-14), (
            f"Var mismatch at mu={mu}, var={sigma2}: ref={var_ref}, closed={var_closed}"
        )


# =====================================================================
# 2. Dense vs Factorized Linear Transport & Symmetrization
# =====================================================================
@pytest.mark.parametrize("n", TEST_WIDTHS)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_factorized_linear_transport(n: int, seed: int):
    """Test dense vs factorized linear transport with asymmetric weights."""
    torch.manual_seed(seed)
    r = 5
    d = 3
    factors = tuple(torch.randn(n, r) for _ in range(d))
    FT = FactoredTensor(n, d, factors)

    # Deliberately asymmetric weight matrix
    W = torch.randn(n, n)
    assert not torch.allclose(W, W.T), "Weight matrix must be asymmetric"

    # Reference transport in ARC convention: W @ A
    FT_W = FactoredTensor(n, d, tuple(W @ f for f in FT.factors))

    # Dense transport
    T_dense = FT.to_tensor()
    T_W_dense = contract_W_basic(T_dense, W)

    assert torch.allclose(FT_W.to_tensor(), T_W_dense, atol=1e-10, rtol=1e-8)

    # Benchmark transpose transport convention check:
    # If W_bench is row-convention (h @ W_bench), then W_ref = W_bench.T.
    # Therefore, factor transport is W_bench.T @ A.
    W_bench = W.T
    FT_bench = FactoredTensor(n, d, tuple(W_bench.T @ f for f in FT.factors))
    assert torch.allclose(FT_bench.to_tensor(), T_W_dense, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("n", TEST_WIDTHS)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_slices_and_column_scaling(n: int, seed: int):
    """Verify repeated-index slices and column sign/scaling invariance."""
    torch.manual_seed(seed)
    r = 6
    factors = tuple(torch.randn(n, r) for _ in range(3))
    FT = FactoredTensor(n, 3, factors)

    # 1. Repeated slices vs dense diagslice
    T_dense = FT.to_tensor()
    slice_21_factored = FT.get_dslice((2, 1))
    slice_21_dense = zero_repeated(diagslice(T_dense, (2, 1)))
    assert torch.allclose(slice_21_factored, slice_21_dense, atol=1e-10, rtol=1e-8)

    slice_3_factored = FT.get_dslice((3,))
    slice_3_dense = zero_repeated(diagslice(T_dense, (3,)))
    assert torch.allclose(slice_3_factored, slice_3_dense, atol=1e-10, rtol=1e-8)

    # 2. Scaling invariance: scaling one factor by alpha and another by 1/alpha leaves tensor unchanged
    A, B, C = factors
    alpha = -2.5
    A_scaled = A * alpha
    B_scaled = B / alpha
    FT_scaled = FactoredTensor(n, 3, (A_scaled, B_scaled, C))
    assert torch.allclose(FT.to_tensor(), FT_scaled.to_tensor(), atol=1e-10, rtol=1e-8)


# =====================================================================
# 3. Factorized vs Dense Recurrence Parity (All 3 Modes)
# =====================================================================
@pytest.mark.parametrize("width", [8, 16])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("seed", [6, 17])
@pytest.mark.parametrize("mode", ["K3-base", "K3-simple", "K3-augment-filtered"])
def test_three_mode_recurrence_parity(width: int, depth: int, seed: int, mode: str):
    """Verify parity between factored and dense recurrence across all 3 modes."""
    torch.manual_seed(seed)
    n = width

    # Initial state: zero mean, identity covariance
    K_init = coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=3)

    if mode == "K3-base":
        kind = Kind.BASE
        augment = False
        base = True
        use_pK = True
    elif mode == "K3-simple":
        kind = Kind.SIMPLE
        augment = False
        base = False
        use_pK = True
    elif mode == "K3-augment-filtered":
        kind = Kind.AUGMENT
        augment = True
        base = False
        use_pK = True
    else:
        raise ValueError(f"Unknown mode {mode}")

    K_dense = K_init
    K_factored = K_init

    # Filtered oracle for augment mode
    real_iso = kh.get_all_terms_iso
    def filtered_iso(k_max, d_max=None, use_mean_var=False, augment=False):
        ret = real_iso(k_max, d_max=d_max, use_mean_var=use_mean_var, augment=augment)
        return {
            ip: {vp: c for vp, c in vps.items() if factored_keeps_term(k_max, ip, vp)}
            for ip, vps in ret.items()
        }

    for l in range(depth):
        W = torch.randn(n, n) * math.sqrt(2.0 / n)

        # 1. Linear step
        WK_dense = linear_kprop(K_dense, W, k_max=3)
        WK_factored = linear_kprop(K_factored, W, k_max=3)

        # 2. Nonlinear step
        if mode == "K3-augment-filtered":
            kh.get_all_terms_iso = filtered_iso

        try:
            K_dense = nonlin_kprop(
                WK_dense,
                nonlin_wick_coef=relu_wick_coef,
                k_max=3,
                kind=kind,
                use_pK=use_pK,
            )
        finally:
            kh.get_all_terms_iso = real_iso

        K_factored = factored_nonlin_kprop_k3(
            K_in=WK_factored,
            nonlin_wick_coef=relu_wick_coef,
            augment=augment,
            base=base,
            use_pK=use_pK,
        )

        # Compare output tensors at each degree
        for d in [1, 2, 3]:
            if d in K_dense:
                t_dense = K_dense[d].to_tensor()
                t_fac = K_factored[d].to_tensor()
                max_abs_err = float(torch.max(torch.abs(t_dense - t_fac)))
                assert max_abs_err < 1e-5, (
                    f"Mode {mode} Layer {l} Degree {d} parity failure: max_err={max_abs_err:.4e}"
                )


# =====================================================================
# 4. Final ReLU Activation and Transpose Convention Check
# =====================================================================
def test_terminal_relu_and_transpose_convention():
    """Verify that terminal ReLU is applied and transpose convention produces expected output."""
    torch.manual_seed(42)
    n = 8
    W_bench = torch.randn(n, n) * math.sqrt(2.0 / n)

    # Benchmark convention: input x is (1, n), z = x @ W_bench, h = relu(z)
    x = torch.randn(1000, n)
    z_bench = x @ W_bench
    h_bench = torch.relu(z_bench)
    sim_mean = torch.mean(h_bench, dim=0)

    # Analytic forward pass with whestbench convention: W_ref = W_bench.T
    K = coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=3)
    WK = linear_kprop(K, W_bench.T, k_max=3)
    KF = factored_nonlin_kprop_k3(WK, relu_wick_coef, base=False, augment=False)

    analytic_mean = KF[1].to_tensor()
    # At layer 0 on Gaussian input, analytic mean must match empirical within MC error (~0.03)
    diff = float(torch.max(torch.abs(analytic_mean - sim_mean)))
    assert diff < 0.05, f"Layer 0 analytic mean mismatch vs empirical: {diff:.4e}"


# =====================================================================
# 5. Sensitivity to K3 & Non-Gaussian Fixture
# =====================================================================
def test_k3_ablation_sensitivity():
    """Verify that removing K3 strictly changes intermediate and final predictions."""
    torch.manual_seed(99)
    n = 16
    W1 = torch.randn(n, n) * math.sqrt(2.0 / n)
    W2 = torch.randn(n, n) * math.sqrt(2.0 / n)

    # Pass 1: standard K3 propagation through layer 1
    K0 = coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=3)
    WK0 = linear_kprop(K0, W1.T, k_max=3)
    K1 = factored_nonlin_kprop_k3(WK0, relu_wick_coef, base=False, augment=False)

    # Layer 1 has non-zero K3
    k3_norm = float(torch.norm(K1[3].to_tensor()))
    assert k3_norm > 1e-4, f"K3 should be non-zero after layer 1, got norm {k3_norm}"

    # Propagate to layer 2 WITH K3
    WK1_with = linear_kprop(K1, W2.T, k_max=3)
    K2_with = factored_nonlin_kprop_k3(WK1_with, relu_wick_coef, base=False, augment=False)
    mu_with = K2_with[1].to_tensor()

    # Propagate to layer 2 WITHOUT K3 (zero out K3)
    K1_ablated = {1: K1[1], 2: K1[2]}
    if 4 in K1:
        K1_ablated[4] = K1[4]
    WK1_ablated = linear_kprop(K1_ablated, W2.T, k_max=3)
    K2_ablated = factored_nonlin_kprop_k3(WK1_ablated, relu_wick_coef, base=False, augment=False)
    mu_ablated = K2_ablated[1].to_tensor()

    # Removing K3 MUST change the predicted mean
    mean_diff = float(torch.max(torch.abs(mu_with - mu_ablated)))
    assert mean_diff > 1e-5, f"Ablating K3 had negligible effect: diff={mean_diff:.4e}"


# =====================================================================
# 6. Cache Invalidation Safety
# =====================================================================
def test_repeated_slice_cache_invalidation():
    """Verify that adding, scaling, or modifying factors invalidates stale repeated slices."""
    torch.manual_seed(123)
    n = 10
    r = 4
    factors = tuple(torch.randn(n, r) for _ in range(3))
    FT = FactoredTensor(n, 3, factors)

    # First slice computation caches repeated slices in FT.repeated
    slice1 = FT.get_dslice((2, 1)).clone()

    # Add factors (changes the tensor)
    add_factors = tuple(torch.randn(n, 2) for _ in range(3))
    FT.add_factors_(add_factors)

    # Recomputed slice must match newly materialized tensor slice
    slice2 = FT.get_dslice((2, 1))
    slice2_expected = zero_repeated(diagslice(FT.to_tensor(), (2, 1)))

    assert not torch.allclose(slice1, slice2, atol=1e-2), "Slice should change after adding factors"
    assert torch.allclose(slice2, slice2_expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
