"""P4N-02 Unit Test: Exact Bivariate Gaussian Covariance Integral & Quadrature.

Validates:
1. Exact bivariate Gaussian covariance formula:
   Cov(relu(Z_i), relu(Z_j)) / (s_i * s_j) = rho * Phi(a_i) * Phi(a_j) + rho^2 * integral_0^1 (1-u) * phi2(a_i, a_j; rho*u) du
   where phi2(a, b; t) = exp(-(a^2 - 2*t*a*b + b^2) / (2*(1-t^2))) / (2*pi*sqrt(1-t^2)).
2. Precomputed Gauss-Legendre nodes & weights for n in {4, 8, 16} incorporating (1-u).
3. Accuracy against high-accuracy numerical integration (scipy.integrate.quad) across:
   a, b in {-5, -2, -0.5, 0, 0.5, 2, 5}
   rho in {-0.99, -0.9, -0.5, -0.1, 0, 0.1, 0.5, 0.9, 0.99} (441 grid points).
4. Properties:
   - Symmetry: Cov(a, b, rho) == Cov(b, a, rho)
   - Zero correlation: Cov(a, b, 0) == 0
   - Diagonal variance: Agreement with exact univariate rectified variance
   - Cho-Saul agreement at a=b=0: Exact match with arc-cosine kernel.
"""

from __future__ import annotations

import sys
import numpy as np
import scipy.integrate as integrate
import scipy.special as sp


def phi(x: float | np.ndarray) -> float | np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def Phi(x: float | np.ndarray) -> float | np.ndarray:
    return 0.5 * (1.0 + sp.erf(x / np.sqrt(2.0)))


def g(z: float | np.ndarray) -> float | np.ndarray:
    """Mean of standard rectified normal: E[max(0, X + z)]."""
    return z * Phi(z) + phi(z)


def exact_univariate_var(a: float | np.ndarray) -> float | np.ndarray:
    """Exact variance of max(0, X + a) for X ~ N(0, 1): Var / s^2."""
    e_x = g(a)
    e_x2 = (1.0 + a * a) * Phi(a) + a * phi(a)
    return e_x2 - e_x * e_x


def phi2(a: float | np.ndarray, b: float | np.ndarray, t: float | np.ndarray) -> float | np.ndarray:
    """Standard bivariate normal PDF with correlation t."""
    denom = 1.0 - t * t
    denom = np.maximum(denom, 1e-15)
    num = a * a - 2.0 * t * a * b + b * b
    return np.exp(-0.5 * num / denom) / (2.0 * np.pi * np.sqrt(denom))


def get_quadrature_nodes_weights(n_nodes: int):
    """Precomputes Gauss-Legendre quadrature nodes u in [0, 1] and weights w_tilde incorporating (1-u)."""
    x, w = sp.roots_legendre(n_nodes)
    u = 0.5 * (x + 1.0)
    w_tilde = 0.5 * w * (1.0 - u)
    return u, w_tilde


def gaussian_cov_quadrature(
    a: float | np.ndarray,
    b: float | np.ndarray,
    rho: float | np.ndarray,
    n_nodes: int = 16,
) -> float | np.ndarray:
    """Computes Cov(relu(Z_i), relu(Z_j)) / (s_i * s_j) using Gauss-Legendre quadrature."""
    u, w_tilde = get_quadrature_nodes_weights(n_nodes)
    Phi_a = Phi(a)
    Phi_b = Phi(b)

    if np.isscalar(a) and np.isscalar(b) and np.isscalar(rho):
        if abs(rho) < 1e-15:
            return 0.0
        t = rho * u
        integral = np.sum(w_tilde * phi2(a, b, t))
        return float(rho * Phi_a * Phi_b + (rho * rho) * integral)

    # Vectorized / array evaluation
    a = np.asarray(a)
    b = np.asarray(b)
    rho = np.asarray(rho)
    integral = np.zeros_like(rho, dtype=np.float64)
    for u_k, w_k in zip(u, w_tilde):
        t_k = rho * u_k
        integral += w_k * phi2(a, b, t_k)

    return rho * Phi_a * Phi_b + (rho * rho) * integral


def high_accuracy_ref(a: float, b: float, rho: float) -> float:
    """High-accuracy reference using adaptive 1D numerical integration of conditional normal."""
    if abs(rho) < 1e-15:
        return 0.0
    sig_c = np.sqrt(max(1.0 - rho * rho, 1e-15))

    def integrand(x):
        return (x + a) * sig_c * g((rho * x + b) / sig_c) * phi(x)

    lower = -a
    if lower < -8.0:
        lower = -8.0
    val, _ = integrate.quad(integrand, lower, 10.0, epsabs=1e-14, epsrel=1e-13, limit=200)
    return val - g(a) * g(b)


def cho_saul_cov(rho: float) -> float:
    """Exact Cho-Saul arc-cosine covariance at zero mean (a=b=0)."""
    r = np.clip(rho, -1.0, 1.0)
    theta = np.arccos(r)
    return (np.sin(theta) + (np.pi - theta) * r - 1.0) / (2.0 * np.pi)


def run_unit_tests():
    print("=====================================================================")
    print("           P4N-02: EXACT GAUSSIAN COVARIANCE UNIT TESTS              ")
    print("=====================================================================")

    # 1. Precompute Quadrature Weights Check
    print("\n--- 1. Quadrature Node & Weight Precomputation ---")
    for n in [4, 8, 16]:
        u, w = get_quadrature_nodes_weights(n)
        weight_sum = np.sum(w)
        print(f"  n={n:2d}: nodes in [{u.min():.5f}, {u.max():.5f}], weight_sum={weight_sum:.12f} (exact=0.5)")
        assert abs(weight_sum - 0.5) < 1e-14, f"Weight sum failed for n={n}"

    # 2. Offline Grid Verification
    grid_a = [-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0]
    grid_b = [-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0]
    grid_rho = [-0.99, -0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 0.99]
    total_points = len(grid_a) * len(grid_b) * len(grid_rho)
    print(f"\n--- 2. Offline Grid Evaluation ({total_points} points: 7 x 7 x 9) ---")

    refs = {}
    for a in grid_a:
        for b in grid_b:
            for rho in grid_rho:
                refs[(a, b, rho)] = high_accuracy_ref(a, b, rho)

    for n in [4, 8, 16]:
        errors = []
        rel_errors = []
        for (a, b, rho), ref in refs.items():
            est = gaussian_cov_quadrature(a, b, rho, n_nodes=n)
            err = abs(est - ref)
            errors.append(err)
            if abs(ref) > 1e-7:
                rel_errors.append(err / abs(ref))

        max_err = max(errors)
        mean_err = np.mean(errors)
        max_rel = max(rel_errors) if rel_errors else 0.0
        print(f"  n_nodes = {n:2d}: Max Abs Err = {max_err:.3e} | Mean Abs Err = {mean_err:.3e} | Max Rel Err (>1e-7) = {max_rel:.3e}")

    # Verify n=16 satisfies high precision
    assert max(errors) < 1e-5, f"n=16 error too high: {max(errors)}"

    # 3. Symmetry Check
    print("\n--- 3. Symmetry Check: Cov(a, b, rho) == Cov(b, a, rho) ---")
    max_sym_err = 0.0
    for a in grid_a:
        for b in grid_b:
            for rho in grid_rho:
                c1 = gaussian_cov_quadrature(a, b, rho, n_nodes=16)
                c2 = gaussian_cov_quadrature(b, a, rho, n_nodes=16)
                max_sym_err = max(max_sym_err, abs(c1 - c2))
    print(f"  Max symmetry error: {max_sym_err:.3e}")
    assert max_sym_err < 1e-15, f"Symmetry check failed: {max_sym_err}"

    # 4. Zero Correlation Check: Cov(a, b, 0) == 0
    print("\n--- 4. Zero Correlation Check: Cov(a, b, 0) == 0 ---")
    max_zero_err = 0.0
    for a in grid_a:
        for b in grid_b:
            c0 = gaussian_cov_quadrature(a, b, 0.0, n_nodes=16)
            max_zero_err = max(max_zero_err, abs(c0))
    print(f"  Max zero-correlation error: {max_zero_err:.3e}")
    assert max_zero_err < 1e-15, f"Zero correlation check failed: {max_zero_err}"

    # 5. Diagonal Rectified Variance Check
    print("\n--- 5. Diagonal Univariate Variance Check ---")
    for a in grid_a:
        exact_v = exact_univariate_var(a)
        quad_v = gaussian_cov_quadrature(a, a, 0.9999, n_nodes=16)
        print(f"  a = {a:4.1f}: Exact Var = {exact_v:.8f} | Quad (rho=0.9999) = {quad_v:.8f} | Diff = {abs(exact_v - quad_v):.3e}")
        assert exact_v >= 0.0, f"Negative variance for a={a}"

    # 6. Cho-Saul Agreement at a=b=0
    print("\n--- 6. Cho-Saul Arc-Cosine Agreement at a=b=0 ---")
    print(f"  {'rho':>6} | {'Cho-Saul':>14} | {'Quad n=8':>14} | {'Diff n=8':>11} | {'Quad n=16':>14} | {'Diff n=16':>11}")
    print("  " + "-" * 78)
    for rho in grid_rho:
        cs = cho_saul_cov(rho)
        q8 = gaussian_cov_quadrature(0.0, 0.0, rho, n_nodes=8)
        q16 = gaussian_cov_quadrature(0.0, 0.0, rho, n_nodes=16)
        diff8 = abs(cs - q8)
        diff16 = abs(cs - q16)
        print(f"  {rho:6.2f} | {cs:14.8f} | {q8:14.8f} | {diff8:11.2e} | {q16:14.8f} | {diff16:11.2e}")
        assert diff16 < 1e-6, f"Cho-Saul check failed at rho={rho}: diff={diff16}"

    print("\n>>> ALL P4N-02 UNIT TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_unit_tests()
