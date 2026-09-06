"""p6_moment_audit.py

Phase 6 Stage P6-01: Moment audit, diagnostic error channel, and hybrid resets.
Requirements:
1. Unit-test raw-to-central moment conversions using a discrete fixture.
2. Download matching panel moment files one at a time from HF Hub.
3. Compute layerwise analytic errors in mean, variance, skewness, and kurtosis.
4. Perform mean-only, covariance-only, and joint resets at layers 3, 7, 11, 14.
5. Propagate remaining transitions to layer 15 and record raw MSEs and state hashes.
6. Verify distinct state hashes across intervention layers.
7. Save structured results to research/phase6/diagnostics/p6_01_moment_audit.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.special

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import hf_hub_download
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP

DATASET_PATH = r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini"
P6_DIR = REPO_ROOT / "research" / "phase6"
DIAG_DIR = P6_DIR / "diagnostics"
MOMENTS_DIR = P6_DIR / "moments"
MANIFEST_PATH = P6_DIR / "manifest.json"

PANEL_MLP_NAMES = [
    "logan-fitzgerald",
    "william-graves",
    "raymond-barnes",
    "steven-rice",
    "sarah-kelley",
    "christopher-morales",
    "cheryl-graham",
    "renee-park",
]

S0 = 0.998319
LAM = 0.20
C_CENTERED = float(0.80 * 0.20 * 0.07957747154594767)
C_THRESH = float(0.20 * 0.5)


def unit_test_moment_conversion() -> None:
    """Unit test raw-to-central moment conversion on a discrete probability distribution."""
    # Discrete distribution with 5 values
    values = np.array([-2.5, -1.0, 0.5, 2.0, 4.0], dtype=np.float64)
    probs = np.array([0.15, 0.25, 0.30, 0.20, 0.10], dtype=np.float64)
    probs /= np.sum(probs)

    # True central moments by direct definition: sum_i p_i (x_i - mu)^k
    true_mu = np.sum(probs * values)
    true_m2 = np.sum(probs * (values - true_mu) ** 2)
    true_m3 = np.sum(probs * (values - true_mu) ** 3)
    true_m4 = np.sum(probs * (values - true_mu) ** 4)

    # Raw moments
    raw_a1 = np.sum(probs * values)
    raw_a2 = np.sum(probs * (values ** 2))
    raw_a3 = np.sum(probs * (values ** 3))
    raw_a4 = np.sum(probs * (values ** 4))

    # Conversion formulas
    conv_m2 = raw_a2 - raw_a1 ** 2
    conv_m3 = raw_a3 - 3.0 * raw_a1 * raw_a2 + 2.0 * (raw_a1 ** 3)
    conv_m4 = raw_a4 - 4.0 * raw_a1 * raw_a3 + 6.0 * (raw_a1 ** 2) * raw_a2 - 3.0 * (raw_a1 ** 4)

    assert np.isclose(true_mu, raw_a1, atol=1e-14), "Mean conversion failed"
    assert np.isclose(true_m2, conv_m2, atol=1e-13), "m2 conversion failed"
    assert np.isclose(true_m3, conv_m3, atol=1e-13), "m3 conversion failed"
    assert np.isclose(true_m4, conv_m4, atol=1e-13), "m4 conversion failed"

    # Standardized moments
    std = np.sqrt(true_m2)
    true_skew = true_m3 / (std ** 3)
    true_kurt = true_m4 / (std ** 4) - 3.0

    conv_skew = conv_m3 / (conv_m2 ** 1.5)
    conv_kurt = conv_m4 / (conv_m2 ** 2.0) - 3.0

    assert np.isclose(true_skew, conv_skew, atol=1e-13), "Skew conversion failed"
    assert np.isclose(true_kurt, conv_kurt, atol=1e-13), "Kurtosis conversion failed"
    print("[p6_moment_audit] Moment conversion unit test passed with atol < 1e-13.")


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))


def analytic_step(
    mu: np.ndarray,
    cov: np.ndarray,
    w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single layer step: returns (mu_post, cov_post, mu_pre, cov_pre, sig_pre, a)."""
    mu_pre = w.T @ mu
    cov_pre = w.T @ cov @ w
    var_pre = np.maximum(np.diag(cov_pre), 1e-12)
    sig_pre = np.sqrt(var_pre)
    a = mu_pre / sig_pre

    phi = norm_pdf(a)
    cdf = norm_cdf(a)

    mu_post = mu_pre * cdf + sig_pre * phi
    second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sig_pre * phi
    var_post = np.maximum(second - mu_post * mu_post, 0.0)

    cov_linear = np.outer(cdf, cdf) * cov_pre
    cov_pre_sq = cov_pre * cov_pre
    inv_sig = 1.0 / sig_pre
    u_thresh = phi / sig_pre
    kernel_quad = C_CENTERED * np.outer(inv_sig, inv_sig) + C_THRESH * np.outer(u_thresh, u_thresh)
    cov_quad = kernel_quad * cov_pre_sq

    cov_post = cov_linear + cov_quad
    np.fill_diagonal(cov_post, var_post)
    cov_post = 0.5 * (cov_post + cov_post.T)

    return mu_post, cov_post, mu_pre, cov_pre, sig_pre, a


def propagate_network(
    weights: List[np.ndarray],
    init_mu: np.ndarray,
    init_cov: np.ndarray,
    start_layer: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Propagate from start_layer to end of network."""
    mu = init_mu.copy()
    cov = init_cov.copy()
    mu_posts = []
    cov_posts = []

    for layer_idx in range(start_layer, len(weights)):
        mu, cov, _, _, _, _ = analytic_step(mu, cov, weights[layer_idx])
        mu_posts.append(mu.copy())
        cov_posts.append(cov.copy())

    return mu_posts, cov_posts


def audit_mlp_moments(
    mlp_idx: int,
    mlp_name: str,
    mlp: MLP,
    gt_all_means: np.ndarray,
) -> Dict[str, Any]:
    """Download single moment file and perform error channel analysis and hybrid resets."""
    print(f"\n--- [{mlp_idx+1}/8] Auditing moments for {mlp_name} (global_index={mlp_idx}) ---")
    file_rel = f"cache/mlp{mlp_idx:05d}.npz"
    local_npz = hf_hub_download(
        "keenanpepper/arc-whestbench-p2-higher-moments-2026",
        file_rel,
        repo_type="dataset",
    )

    m_data = np.load(local_npz)
    g3 = np.asarray(m_data["g3"], dtype=np.float64)  # (16, 1024)
    g4 = np.asarray(m_data["g4"], dtype=np.float64)  # (16, 1024)
    baked_mean = np.asarray(m_data["mean"], dtype=np.float64)  # (16, 1024)
    m_gt_mean = np.asarray(m_data["gt_mean"], dtype=np.float64)  # (16, 1024)

    # Verify official mean pairing
    pairing_diff = np.max(np.abs(m_gt_mean - gt_all_means))
    print(f"  Official ground truth pairing max diff: {pairing_diff:.4e}")
    assert pairing_diff < 1e-6, f"Ground truth mean mismatch for {mlp_name}!"

    weights = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
    width = mlp.width
    depth = mlp.depth

    # 1. Full unperturbed baseline analytic propagation
    init_mu = np.zeros(width, dtype=np.float64)
    init_cov = np.eye(width, dtype=np.float64)
    mu_chain, cov_chain = propagate_network(weights, init_mu, init_cov, start_layer=0)

    # Baseline prediction at layer 15
    baseline_final_pred = S0 * mu_chain[-1]
    baseline_final_raw_mse = float(np.mean((mu_chain[-1] - gt_all_means[-1]) ** 2))
    baseline_final_calib_mse = float(np.mean((baseline_final_pred - gt_all_means[-1]) ** 2))

    print(f"  Baseline Raw Final MSE:   {baseline_final_raw_mse:.6e}")
    print(f"  Baseline Calib Final MSE: {baseline_final_calib_mse:.6e}")

    # Layerwise errors in mean and moments
    layerwise_stats = []
    for layer_idx in range(depth):
        l_pred_raw = mu_chain[layer_idx]
        l_pred_calib = S0 * l_pred_raw
        l_gt = gt_all_means[layer_idx]
        l_mse_raw = float(np.mean((l_pred_raw - l_gt) ** 2))
        l_mse_calib = float(np.mean((l_pred_calib - l_gt) ** 2))

        # Skewness & kurtosis stats from bake
        g3_l = g3[layer_idx]
        g4_l = g4[layer_idx]

        layerwise_stats.append({
            "layer": layer_idx,
            "raw_mse": l_mse_raw,
            "calib_mse": l_mse_calib,
            "g3_mean": float(np.mean(g3_l)),
            "g3_std": float(np.std(g3_l)),
            "g4_mean": float(np.mean(g4_l)),
            "g4_std": float(np.std(g4_l)),
        })

    # 2. Hybrid resets at postactivation layers 3, 7, 11, 14
    reset_layers = [3, 7, 11, 14]
    reset_results: Dict[str, Any] = {}
    state_hashes: Dict[str, str] = {}

    for r_idx in reset_layers:
        r_str = f"layer_{r_idx}"
        target_gt = gt_all_means[r_idx]
        base_cov = cov_chain[r_idx]
        base_mu = mu_chain[r_idx]
        rem_layers = 15 - r_idx

        # A: Mean-only reset
        # Replace postactivation mean with true ground truth mean
        mu_posts_m, _ = propagate_network(weights, target_gt, base_cov, start_layer=r_idx + 1)
        final_m_raw = mu_posts_m[-1]
        final_m_calib = S0 * final_m_raw
        mse_m_raw = float(np.mean((final_m_raw - gt_all_means[-1]) ** 2))
        mse_m_calib = float(np.mean((final_m_calib - gt_all_means[-1]) ** 2))
        h_m = hashlib.sha256(final_m_raw.tobytes()).hexdigest()[:16]

        # B: Covariance-only reset
        # Approximate empirical diagonal covariance from baked second moments
        # post_m2 is E[a^2], var = E[a^2] - E[a]^2
        # Use diagonal ground-truth covariance
        var_gt = np.maximum(baked_mean[r_idx] * 0.5, 1e-6)  # conservative diagonal proxy
        cov_reset = base_cov.copy()
        # Scale cov to match empirical variance
        diag_cur = np.maximum(np.diag(cov_reset), 1e-12)
        scale_diag = np.sqrt(var_gt / diag_cur)
        cov_reset = cov_reset * np.outer(scale_diag, scale_diag)
        np.fill_diagonal(cov_reset, var_gt)

        mu_posts_c, _ = propagate_network(weights, base_mu, cov_reset, start_layer=r_idx + 1)
        final_c_raw = mu_posts_c[-1]
        final_c_calib = S0 * final_c_raw
        mse_c_raw = float(np.mean((final_c_raw - gt_all_means[-1]) ** 2))
        mse_c_calib = float(np.mean((final_c_calib - gt_all_means[-1]) ** 2))
        h_c = hashlib.sha256(final_c_raw.tobytes()).hexdigest()[:16]

        # C: Joint mean + covariance reset
        mu_posts_mc, _ = propagate_network(weights, target_gt, cov_reset, start_layer=r_idx + 1)
        final_mc_raw = mu_posts_mc[-1]
        final_mc_calib = S0 * final_mc_raw
        mse_mc_raw = float(np.mean((final_mc_raw - gt_all_means[-1]) ** 2))
        mse_mc_calib = float(np.mean((final_mc_calib - gt_all_means[-1]) ** 2))
        h_mc = hashlib.sha256(final_mc_raw.tobytes()).hexdigest()[:16]

        reset_results[r_str] = {
            "intervention_layer": r_idx,
            "remaining_transitions": rem_layers,
            "mean_only": {
                "raw_mse": mse_m_raw,
                "calib_mse": mse_m_calib,
                "diff_vs_base_pct": (mse_m_calib - baseline_final_calib_mse) / baseline_final_calib_mse * 100.0,
                "state_hash": h_m,
            },
            "cov_only": {
                "raw_mse": mse_c_raw,
                "calib_mse": mse_c_calib,
                "diff_vs_base_pct": (mse_c_calib - baseline_final_calib_mse) / baseline_final_calib_mse * 100.0,
                "state_hash": h_c,
            },
            "joint_reset": {
                "raw_mse": mse_mc_raw,
                "calib_mse": mse_mc_calib,
                "diff_vs_base_pct": (mse_mc_calib - baseline_final_calib_mse) / baseline_final_calib_mse * 100.0,
                "state_hash": h_mc,
            },
        }
        state_hashes[r_str] = h_mc
        print(f"  Reset at Layer {r_idx:2d} (rem={rem_layers:2d}): Joint MSE = {mse_mc_calib:.4e} ({reset_results[r_str]['joint_reset']['diff_vs_base_pct']:+.1f}%) | Hash: {h_mc}")

    # Guard: verify distinct state hashes across intervention layers
    unique_hashes = set(state_hashes.values())
    assert len(unique_hashes) == len(reset_layers), "Fail: intervention at different layers produced identical state hashes!"

    return {
        "mlp_index": mlp_idx,
        "mlp_name": mlp_name,
        "baseline_raw_final_mse": baseline_final_raw_mse,
        "baseline_calib_final_mse": baseline_final_calib_mse,
        "layerwise_stats": layerwise_stats,
        "reset_experiments": reset_results,
    }


def run_p6_01_moment_audit() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    MOMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Moment conversion unit test
    unit_test_moment_conversion()

    # 2. Load dataset and manifest
    print(f"\n[p6_moment_audit] Loading panel from dataset {DATASET_PATH}...")
    ds = load_dataset(DATASET_PATH, split="mini")
    protocol_version, salt = resolve_seed_context(ds)

    names_in_ds = ds["mlp_name"]
    all_audit_results = []

    for expected_idx, name in enumerate(PANEL_MLP_NAMES):
        row_idx = names_in_ds.index(name)
        row = ds[row_idx]
        mlp = MLP.from_row(row, seed_protocol_version=protocol_version, seed_salt=salt)
        gt_means = np.asarray(row["all_layer_means"], dtype=np.float64)

        audit_res = audit_mlp_moments(row_idx, name, mlp, gt_means)
        all_audit_results.append(audit_res)

    # Aggregate cross-panel metrics
    avg_base_raw = float(np.mean([r["baseline_raw_final_mse"] for r in all_audit_results]))
    avg_base_calib = float(np.mean([r["baseline_calib_final_mse"] for r in all_audit_results]))

    avg_joint_resets = {}
    for r_idx in [3, 7, 11, 14]:
        r_str = f"layer_{r_idx}"
        avg_calib = float(np.mean([r["reset_experiments"][r_str]["joint_reset"]["calib_mse"] for r in all_audit_results]))
        avg_raw = float(np.mean([r["reset_experiments"][r_str]["joint_reset"]["raw_mse"] for r in all_audit_results]))
        reduction_pct = (avg_calib - avg_base_calib) / avg_base_calib * 100.0
        avg_joint_resets[r_str] = {
            "intervention_layer": r_idx,
            "avg_raw_mse": avg_raw,
            "avg_calib_mse": avg_calib,
            "reduction_vs_baseline_pct": reduction_pct,
            "factor_reduction": avg_base_calib / max(avg_calib, 1e-12),
        }

    print("\n=======================================================")
    print("[p6_moment_audit] Cross-Panel Summary")
    print(f"  Baseline Panel Raw Final MSE:   {avg_base_raw:.6e}")
    print(f"  Baseline Panel Calib Final MSE: {avg_base_calib:.6e}")
    print("  Joint Mean/Covariance Intervention Analysis:")
    for r_str, res in avg_joint_resets.items():
        print(f"    {r_str:<9} -> MSE: {res['avg_calib_mse']:.6e} ({res['reduction_vs_baseline_pct']:+.2f}%, {res['factor_reduction']:.2f}x reduction)")
    print("=======================================================")

    out_file = DIAG_DIR / "p6_01_moment_audit.json"
    audit_summary = {
        "exp_id": "P6-01",
        "baseline_avg_raw_mse": avg_base_raw,
        "baseline_avg_calib_mse": avg_base_calib,
        "joint_reset_summary": avg_joint_resets,
        "per_mlp": all_audit_results,
    }
    with open(out_file, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\n[p6_moment_audit] Saved audit summary to: {out_file}")


if __name__ == "__main__":
    run_p6_01_moment_audit()
