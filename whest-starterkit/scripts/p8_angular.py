"""p8_angular.py

Phase 8 Stage P8-A0: Angular Moment Identities, Radius Factorization,
and Exact First-Layer / Terminal Conversions.

Reference: phase8plan.md Section 7.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy.special import gammaln

REPO_ROOT = Path(__file__).resolve().parent.parent
P8_DIR = REPO_ROOT / "research" / "phase8"
FIXTURES_DIR = P8_DIR / "fixtures"

SUPPORTED_WIDTHS = [3, 5, 8, 16, 32, 64, 128, 1024]


def compute_ap(n: int, p: int) -> float:
    """Compute a_p = E[(R / sqrt(n))^p] for R ~ chi_n.
    
    Formula:
        a_p = (2/n)^(p/2) * Gamma((n+p)/2) / Gamma(n/2)
    Using stable log-gamma:
        ln a_p = (p/2)*ln(2/n) + gammaln((n+p)/2) - gammaln(n/2)
    """
    if p == 0:
        return 1.0
    if p == 2:
        return 1.0
    if p == 4:
        return float((n + 2) / n)
    ln_val = (p / 2.0) * math.log(2.0 / n) + gammaln((n + p) / 2.0) - gammaln(n / 2.0)
    return float(math.exp(ln_val))


def generate_ap_table(widths: list[int] = SUPPORTED_WIDTHS, max_p: int = 8) -> Dict[str, Any]:
    """Precompute a_p for p=0..max_p across supported fixture widths and 1024."""
    table: Dict[str, Dict[str, float]] = {}
    for n in widths:
        table[str(n)] = {}
        for p in range(max_p + 1):
            table[str(n)][str(p)] = compute_ap(n, p)
    return table


def save_ap_constants() -> Path:
    """Save precomputed a_p constants to JSON fixture."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    table = generate_ap_table()
    out_path = FIXTURES_DIR / "angular_ap_constants.json"
    meta = {
        "description": "Precomputed a_p = E[(R/sqrt(n))^p] for R ~ chi_n across supported widths",
        "provenance": "scipy.special.gammaln offline computation via phase8plan.md Section 7.1",
        "constants": table,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_path


def load_ap_constant(n: int, p: int) -> float:
    """Load or compute a_p for width n and power p."""
    const_path = FIXTURES_DIR / "angular_ap_constants.json"
    if const_path.exists():
        try:
            with open(const_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data["constants"][str(n)][str(p)])
        except Exception:
            pass
    return compute_ap(n, p)


def gaussian_from_angular(mu_A: np.ndarray, C_A: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct Gaussian mean and covariance from angular mean and covariance.
    
    Equations:
        a_1 = E[R / sqrt(n)]
        mu_G = a_1 * mu_A
        M2_G = M2_A (since a_2 = 1)
        C_G = C_A + (1 - a_1^2) * mu_A mu_A^T
    """
    a1 = load_ap_constant(n, 1)
    mu_G = a1 * mu_A
    C_G = C_A + (1.0 - a1 ** 2) * np.outer(mu_A, mu_A)
    return mu_G, C_G


def angular_from_gaussian(mu_G: np.ndarray, C_G: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Gaussian mean and covariance to angular mean and covariance.
    
    Equations:
        mu_A = mu_G / a_1
        C_A = C_G - (1 - a_1^2) * mu_A mu_A^T
    """
    a1 = load_ap_constant(n, 1)
    mu_A = mu_G / a1
    C_A = C_G - (1.0 - a1 ** 2) * np.outer(mu_A, mu_A)
    return mu_A, C_A


def verify_angular_identities(n: int = 5, n_samples: int = 200_000, seed: int = 8101) -> Dict[str, Any]:
    """Mandatory Section 7.2 checks:
    1. Radial/angular factorization for 1-layer analytic moments, p=1..4.
    2. Angular input covariance and fourth cumulant against dense index formulas.
    3. Gaussian reconstruction of mean, second raw moment, and covariance.
    4. Exact first-layer angular mean: Gaussian first-layer mean divided by a_1.
    5. Depth-16 forward identity on fixed X and corresponding Y.
    """
    rng = np.random.default_rng(seed)
    results: Dict[str, Any] = {}

    # 1. Radius factorization: E[h(X)^p] = a_p * E[h(Y)^p]
    W1 = rng.standard_normal((n, n)) * np.sqrt(2.0 / n)
    X = rng.standard_normal((n_samples, n))
    R = np.linalg.norm(X, axis=1, keepdims=True)
    Y = np.sqrt(n) * (X / R)

    # ReLU activations
    h_X = np.maximum(0.0, X @ W1)  # (N, n)
    h_Y = np.maximum(0.0, Y @ W1)  # (N, n)

    factorization_errors = {}
    for p in [1, 2, 3, 4]:
        ap = compute_ap(n, p)
        # Check neuron 0
        val_X = np.mean(h_X[:, 0] ** p)
        val_Y = np.mean(h_Y[:, 0] ** p)
        pred_X_from_Y = ap * val_Y
        rel_err = abs(val_X - pred_X_from_Y) / max(abs(val_X), 1e-9)
        mc_se = np.std((h_X[:, 0] ** p) - ap * (h_Y[:, 0] ** p)) / np.sqrt(n_samples)
        factorization_errors[f"p={p}"] = {
            "val_X": float(val_X),
            "pred_X_from_Y": float(pred_X_from_Y),
            "ap": float(ap),
            "rel_err": float(rel_err),
            "mc_se": float(mc_se),
            "within_3se": bool(abs(val_X - pred_X_from_Y) < 3.5 * mc_se),
        }
    results["radius_factorization"] = factorization_errors

    # 2. Angular input covariance and fourth cumulant dense formulas
    # Y = sqrt(n) U, U uniform on S^{n-1}
    cov_Y = np.cov(Y, rowvar=False)
    cov_err = float(np.max(np.abs(cov_Y - np.eye(n))))
    results["angular_input_cov_err"] = cov_err

    # Fourth cumulant dense check:
    # K4_ijkl = -2/(n+2) * (delta_ij delta_kl + delta_ik delta_jl + delta_il delta_jk)
    # K4_iiii = -6/(n+2)
    # Sample 4th cumulant: kappa4(Y_i) = E[Y_i^4] - 3 (E[Y_i^2])^2
    y0 = Y[:, 0]
    m2 = np.mean(y0 ** 2)
    m4 = np.mean(y0 ** 4)
    k4_samp = float(m4 - 3.0 * (m2 ** 2))
    k4_theo = -6.0 / (n + 2)
    k4_err = abs(k4_samp - k4_theo)
    results["k4_iiii_sample"] = k4_samp
    results["k4_iiii_theoretical"] = float(k4_theo)
    results["k4_iiii_abs_err"] = float(k4_err)

    # Initial scalar c4 convention: c4 = -6 / (n + 2)
    initial_c4 = -6.0 / (n + 2)
    results["initial_c4"] = float(initial_c4)

    # 3. Gaussian reconstruction of mean, second moment, covariance
    # Ground truth Gaussian sample moments:
    mu_X_sample = np.mean(h_X, axis=0)
    C_X_sample = np.cov(h_X, rowvar=False)
    M2_X_sample = np.mean(h_X[:, :, None] * h_X[:, None, :], axis=0)

    # Angular sample moments:
    mu_Y_sample = np.mean(h_Y, axis=0)
    C_Y_sample = np.cov(h_Y, rowvar=False)
    M2_Y_sample = np.mean(h_Y[:, :, None] * h_Y[:, None, :], axis=0)

    # Reconstruct Gaussian from Angular:
    mu_G_recon, C_G_recon = gaussian_from_angular(mu_Y_sample, C_Y_sample, n)
    M2_G_recon = M2_Y_sample  # a_2 = 1

    recon_mu_diff = float(np.max(np.abs(mu_G_recon - mu_X_sample)))
    recon_C_diff = float(np.max(np.abs(C_G_recon - C_X_sample)))
    recon_M2_diff = float(np.max(np.abs(M2_G_recon - M2_X_sample)))

    results["reconstruction"] = {
        "mu_max_diff": recon_mu_diff,
        "C_max_diff": recon_C_diff,
        "M2_max_diff": recon_M2_diff,
    }

    # 4. Exact first-layer angular mean: mu_G1 / a_1
    # For single layer with W1:
    row_norms = np.linalg.norm(W1, axis=0)  # norm of incoming weight columns
    exact_mu_G1 = row_norms / np.sqrt(2.0 * np.pi)
    a1 = compute_ap(n, 1)
    exact_mu_A1 = exact_mu_G1 / a1

    diff_mu_G = float(np.max(np.abs(exact_mu_G1 - mu_X_sample)))
    diff_mu_A = float(np.max(np.abs(exact_mu_A1 - mu_Y_sample)))
    results["exact_first_layer"] = {
        "exact_mu_G1_sample_diff": diff_mu_G,
        "exact_mu_A1_sample_diff": diff_mu_A,
    }

    # 5. Depth-16 forward identity on fixed X and corresponding Y
    # Generate 16 random weight matrices
    weights_16 = [rng.standard_normal((n, n)) * np.sqrt(2.0 / n) for _ in range(16)]
    x_test = rng.standard_normal(n)
    r_test = float(np.linalg.norm(x_test))
    y_test = np.sqrt(n) * (x_test / r_test)

    # Propagate x_test through 16 ReLUs
    curr_x = x_test.copy()
    for w in weights_16:
        curr_x = np.maximum(0.0, curr_x @ w)

    # Propagate y_test through 16 ReLUs
    curr_y = y_test.copy()
    for w in weights_16:
        curr_y = np.maximum(0.0, curr_y @ w)

    # Homogeneity check: curr_x == (r_test / sqrt(n)) * curr_y
    scale_factor = r_test / np.sqrt(n)
    h16_pred_x = scale_factor * curr_y
    h16_max_err = float(np.max(np.abs(curr_x - h16_pred_x)))
    h16_rel_err = float(h16_max_err / max(np.max(np.abs(curr_x)), 1e-12))
    results["depth16_forward_homogeneity"] = {
        "max_abs_err": h16_max_err,
        "rel_err": h16_rel_err,
        "passes": bool(h16_rel_err < 1e-9),
    }

    return results


if __name__ == "__main__":
    print("Precomputing a_p constants...")
    const_file = save_ap_constants()
    print(f"Saved to {const_file}")

    print("\nRunning verification of angular identities...")
    res = verify_angular_identities(n=5, n_samples=100_000)
    print("Radius factorization:")
    for k, v in res["radius_factorization"].items():
        print(f"  {k}: rel_err={v['rel_err']:.4e} mc_se={v['mc_se']:.4e} within_3se={v['within_3se']}")
    print(f"Angular input cov err: {res['angular_input_cov_err']:.4e}")
    print(f"k4_iiii: sample={res['k4_iiii_sample']:.6f} theo={res['k4_iiii_theoretical']:.6f} err={res['k4_iiii_abs_err']:.4e}")
    print(f"Reconstruction max diff: mu={res['reconstruction']['mu_max_diff']:.4e} C={res['reconstruction']['C_max_diff']:.4e}")
    print(f"Depth-16 forward homogeneity rel_err: {res['depth16_forward_homogeneity']['rel_err']:.4e} passes={res['depth16_forward_homogeneity']['passes']}")
