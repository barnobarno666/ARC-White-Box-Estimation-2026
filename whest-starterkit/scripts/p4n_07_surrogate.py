"""P4N-07 Diagnostic: Cheap Surrogate + Coupled Residual on 8 Phase 2 MLPs.

Implements Experiment P4N-07 from PHASE5_NEXT_EXPERIMENTS.md (Section 13):
1. Evaluates across all 8 Phase 2 MLPs (using whestbench load_dataset, split='mini').
2. Uses an independent 1024-row construction pilot (512 antipodal pairs) to build/freeze surrogates:
   - S1: Prefix through layer 7 + pilot-gated affine suffix
   - S1-Ridge: Prefix through layer 7 + fitted ridge suffix (rank 256)
   - S2: Prefix through layer 11 + pilot-gated affine suffix
   - S2-Ridge: Prefix through layer 11 + fitted ridge suffix (rank 256)
   - S2b: Prefix through layer 13 + pilot-gated affine suffix
   - S3: Width-512 pruned network from layer 4 onward (importance-scored, mean-folded bias)
   - S4: Width-256 pruned network from layer 4 onward (importance-scored, mean-folded bias)
3. For each surrogate, measures pair-averaged quantities:
   - Vg = coordinate variance of g(X)
   - Vd = coordinate variance of f(X) - g(X)
   - Vf = coordinate variance of f(X)
   - FLOP cost cg (cost per pair of g), cd (cost per pair of coupled f-g)
   - Theoretical optimal allocation Ng / Nd = sqrt(Vg * cd / (Vd * cg))
   - Optimal theoretical variance vs matched-cost ordinary sampling
4. Empirically evaluates the two-level estimator across budget utilization levels u in {0.098, 0.15, 0.20}:
   mu_hat = mean_A[g(X)] + mean_B[f(X) - g(X)]
   where streams A and B are independent, and difference f-g is evaluated on coupled X in stream B.
5. Evaluates standalone MSE, LOO fusion with control analytic prior, and scored error:
   S = MSE * max(0.10, compute_utilization).
6. Checks Pre-port Gate: >=30% gain over matched-cost ordinary sampling or >=20% scored gain over control.
7. Saves complete diagnostic records to research/phase4_next/diagnostics/p4n_07_surrogate.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.special

# Ensure immediate line buffering so logs are visible in real time
sys.stdout.reconfigure(line_buffering=True)

from whestbench.domain import MLP
from whestbench.dataset import load_dataset, resolve_seed_context

DATASET_PATH = r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini"
DIAGNOSTICS_DIR = Path("research/phase4_next/diagnostics")
CONTROL_RECEIPT_PATH = Path("scripts/champion_8mlp.json")

_S_PRIOR = 0.998319
_CONTROL_SCORE = 1.223643e-07
_BUDGET_FLOPS = float(2**41)


def numpy_json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))


def compute_analytic_mean(mlp: MLP, weights: List[np.ndarray]) -> np.ndarray:
    """Computes exact dual-kernel blended Hermite analytic mean (1024,)."""
    width = mlp.width
    mu = np.zeros(width, dtype=np.float32)
    cov = np.eye(width, dtype=np.float32)

    c_centered = np.float32(0.80 * 0.20 * 0.07957747154594767)
    c_thresh = np.float32(0.20 * 0.5)

    for w in weights:
        mu_pre = w.T @ mu
        cov_pre = w.T @ cov @ w
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sigma_pre = np.sqrt(var_pre)
        a = mu_pre / sigma_pre
        phi = norm_pdf(a).astype(np.float32)
        cdf = norm_cdf(a).astype(np.float32)

        mu = (mu_pre * cdf + sigma_pre * phi).astype(np.float32)
        second = ((mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi).astype(np.float32)
        var_post = np.maximum(second - mu * mu, 0.0)
        gain = np.where(sigma_pre > 1e-12, cdf, 0.0).astype(np.float32)

        cov_linear = np.outer(gain, gain) * cov_pre
        cov_pre_sq = cov_pre * cov_pre
        inv_sigma = np.where(sigma_pre > 1e-12, 1.0 / sigma_pre, 0.0)
        u_thresh = np.where(sigma_pre > 1e-12, phi / sigma_pre, 0.0)

        kernel_quad = c_centered * np.outer(inv_sigma, inv_sigma) + c_thresh * np.outer(u_thresh, u_thresh)
        cov_quad = kernel_quad * cov_pre_sq

        cov = cov_linear + cov_quad
        np.fill_diagonal(cov, var_post)

    return mu


def forward_full(X: np.ndarray, weights: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Runs full forward pass, returning final output H15 and all post-ReLU activations [H0, ..., H15]."""
    activations = []
    H = X
    for l, W in enumerate(weights):
        Z = H @ W
        H = np.maximum(Z, 0.0)
        activations.append(H)
    return H, activations


def forward_prefix(X: np.ndarray, weights: List[np.ndarray], k: int) -> np.ndarray:
    """Runs forward pass through layer k inclusive, returning Hk."""
    H = X
    for l in range(k + 1):
        H = np.maximum(H @ weights[l], 0.0)
    return H


def make_antipodal_pairs(rng: np.random.Generator, n_pairs: int, width: int = 1024) -> np.ndarray:
    """Generates n_pairs antipodal samples (2*n_pairs total rows)."""
    X_pos = rng.standard_normal((n_pairs, width), dtype=np.float32)
    X = np.empty((2 * n_pairs, width), dtype=np.float32)
    X[:n_pairs] = X_pos
    X[n_pairs:] = -X_pos
    return X


def pair_average(Y: np.ndarray, n_pairs: int) -> np.ndarray:
    """Averages rows [0..n_pairs-1] and [n_pairs..2*n_pairs-1]. Output shape (n_pairs, d)."""
    return 0.5 * (Y[:n_pairs] + Y[n_pairs:])


# ==============================================================================
# Surrogate Builders
# ==============================================================================

class AffineSuffixSurrogate:
    """Surrogate with exact prefix through layer k + affine suffix B_k, offset b_k."""

    def __init__(self, k: int, weights: List[np.ndarray], B: np.ndarray, b: np.ndarray, label: str):
        self.k = k
        self.prefix_weights = weights[:k + 1]
        self.B = B.astype(np.float32)
        self.b = b.astype(np.float32)
        self.label = label
        # FLOPs per sample: (k+1)*2*1024^2 (prefix) + 2*1024^2 (suffix)
        self.flops_per_sample = (k + 2) * 2 * (1024 ** 2)
        # Suffix-only FLOPs when prefix is already computed: 2*1024^2
        self.suffix_flops = 2 * (1024 ** 2)

    def evaluate_from_X(self, X: np.ndarray) -> np.ndarray:
        H = X
        for W in self.prefix_weights:
            H = np.maximum(H @ W, 0.0)
        return H @ self.B + self.b

    def evaluate_from_Hk(self, Hk: np.ndarray) -> np.ndarray:
        return Hk @ self.B + self.b


class PrunedSuffixSurrogate:
    """Surrogate with exact prefix through layer 3 (layers 0..3 full width),
    and width-pruned layers 4..14 to width K_retained, with mean-folded biases.
    """

    def __init__(
        self,
        weights: List[np.ndarray],
        pilot_activations: List[np.ndarray],
        K_retained: int,
        label: str,
    ):
        self.K = K_retained
        self.label = label
        self.weights_0_3 = weights[:4]

        # Select neurons for layers 4..14
        self.indices = []
        self.sub_weights = []
        self.folded_biases = []

        # Layer 4 receives 1024 inputs from layer 3, outputs K_retained neurons
        W4 = weights[4]  # (1024, 1024)
        H4_pilot = pilot_activations[4]
        # Score neurons of layer 4: Var(H4) * ||W5_col||
        W5 = weights[5]
        var_H4 = np.var(H4_pilot, axis=0)
        col_norm_W5 = np.linalg.norm(W5, axis=1)  # energy going into layer 5
        score_4 = var_H4 * col_norm_W5
        idx_4 = np.argsort(score_4)[-K_retained:]
        idx_4.sort()
        self.indices.append(idx_4)
        # Layer 4 sub-weight: (1024, K)
        self.sub_weights.append(W4[:, idx_4].astype(np.float32))
        self.folded_biases.append(np.zeros(K_retained, dtype=np.float32))

        # Layers 5..14
        for l in range(5, 15):
            prev_idx = self.indices[-1]
            W_l = weights[l]  # (1024, 1024)
            H_l_pilot = pilot_activations[l]
            W_next = weights[l + 1]
            var_H_l = np.var(H_l_pilot, axis=0)
            col_norm_next = np.linalg.norm(W_next, axis=1)
            score_l = var_H_l * col_norm_next
            idx_l = np.argsort(score_l)[-K_retained:]
            idx_l.sort()
            self.indices.append(idx_l)

            # Omitted neurons from previous layer
            all_prev = np.arange(1024)
            omitted_prev = np.setdiff1d(all_prev, prev_idx)
            H_prev_omitted_mean = np.mean(pilot_activations[l - 1][:, omitted_prev], axis=0)  # (|omitted|,)
            W_prev_omitted = W_l[omitted_prev, :][:, idx_l]  # (|omitted|, K)
            bias_fold = H_prev_omitted_mean @ W_prev_omitted  # (K,)

            # Sub-weight: from prev retained to curr retained (K, K)
            sub_W = W_l[prev_idx, :][:, idx_l]
            self.sub_weights.append(sub_W.astype(np.float32))
            self.folded_biases.append(bias_fold.astype(np.float32))

        # Layer 15 (final layer): maps from last retained K to full 1024 output
        last_idx = self.indices[-1]
        all_last = np.arange(1024)
        omitted_last = np.setdiff1d(all_last, last_idx)
        H_14_omitted_mean = np.mean(pilot_activations[14][:, omitted_last], axis=0)
        W_15 = weights[15]
        bias_15 = H_14_omitted_mean @ W_15[omitted_last, :]  # (1024,)
        sub_W_15 = W_15[last_idx, :]  # (K, 1024)
        self.sub_weights.append(sub_W_15.astype(np.float32))
        self.folded_biases.append(bias_15.astype(np.float32))

        # Calculate FLOPs per sample:
        f_0_3 = 4 * 2 * (1024 ** 2)
        f_4 = 2 * 1024 * K_retained
        f_5_14 = 10 * 2 * (K_retained ** 2)
        f_15 = 2 * K_retained * 1024
        self.flops_per_sample = f_0_3 + f_4 + f_5_14 + f_15

    def evaluate_from_X(self, X: np.ndarray) -> np.ndarray:
        H = X
        for W in self.weights_0_3:
            H = np.maximum(H @ W, 0.0)

        H = np.maximum(H @ self.sub_weights[0] + self.folded_biases[0], 0.0)

        for sub_W, bias in zip(self.sub_weights[1:-1], self.folded_biases[1:-1]):
            H = np.maximum(H @ sub_W + bias, 0.0)

        H = np.maximum(H @ self.sub_weights[-1] + self.folded_biases[-1], 0.0)
        return H


def build_surrogates_for_mlp(mlp: MLP, weights: List[np.ndarray], rng: np.random.Generator) -> Dict[str, Any]:
    """Builds all candidate surrogates using an independent 1024-row pilot."""
    width = mlp.width
    X_pilot = make_antipodal_pairs(rng, n_pairs=512, width=width)
    f_pilot, act_pilot = forward_full(X_pilot, weights)

    surrogates = {}

    # S1: Depth 7 + Weight-Aware Gated Suffix
    k = 7
    B_7 = np.eye(width, dtype=np.float32)
    for l in range(k + 1, 16):
        gate_probs = np.mean(act_pilot[l] > 0.0, axis=0).astype(np.float32)
        B_7 = (B_7 @ weights[l]) * gate_probs[None, :]

    mean_f_pilot = np.mean(f_pilot, axis=0)
    mean_H7_pilot = np.mean(act_pilot[k], axis=0)
    b_7 = mean_f_pilot - mean_H7_pilot @ B_7
    surrogates["S1_prefix7_gated"] = AffineSuffixSurrogate(k=7, weights=weights, B=B_7, b=b_7, label="S1_prefix7_gated")

    # S1-Ridge: Depth 7 + Fitted Ridge Suffix (r=256)
    H7_pairs = pair_average(act_pilot[k], 512)
    f_pairs = pair_average(f_pilot, 512)
    H7_cent = H7_pairs - np.mean(H7_pairs, axis=0)
    f_cent = f_pairs - np.mean(f_pairs, axis=0)

    C_cross = H7_cent.T @ f_cent
    _, _, Vt_cross = np.linalg.svd(C_cross, full_matrices=False)
    V_r = Vt_cross[:256, :].T
    Z_proj = H7_cent @ V_r
    G = Z_proj.T @ Z_proj
    lam_eff = 1e-2 * float(np.trace(G) / 256)
    R = np.linalg.solve(G + lam_eff * np.eye(256, dtype=np.float32), Z_proj.T @ f_cent)
    B_ridge_7 = (V_r @ R).astype(np.float32)
    b_ridge_7 = np.mean(f_pairs, axis=0) - np.mean(H7_pairs, axis=0) @ B_ridge_7
    surrogates["S1_prefix7_ridge256"] = AffineSuffixSurrogate(k=7, weights=weights, B=B_ridge_7, b=b_ridge_7, label="S1_prefix7_ridge256")

    # S2: Depth 11 + Weight-Aware Gated Suffix
    k = 11
    B_11 = np.eye(width, dtype=np.float32)
    for l in range(k + 1, 16):
        gate_probs = np.mean(act_pilot[l] > 0.0, axis=0).astype(np.float32)
        B_11 = (B_11 @ weights[l]) * gate_probs[None, :]

    mean_H11_pilot = np.mean(act_pilot[k], axis=0)
    b_11 = mean_f_pilot - mean_H11_pilot @ B_11
    surrogates["S2_prefix11_gated"] = AffineSuffixSurrogate(k=11, weights=weights, B=B_11, b=b_11, label="S2_prefix11_gated")

    # S2-Ridge: Depth 11 + Fitted Ridge Suffix (r=256)
    H11_pairs = pair_average(act_pilot[k], 512)
    H11_cent = H11_pairs - np.mean(H11_pairs, axis=0)
    C_cross_11 = H11_cent.T @ f_cent
    _, _, Vt_cross_11 = np.linalg.svd(C_cross_11, full_matrices=False)
    V_r_11 = Vt_cross_11[:256, :].T
    Z_proj_11 = H11_cent @ V_r_11
    G_11 = Z_proj_11.T @ Z_proj_11
    lam_eff_11 = 1e-2 * float(np.trace(G_11) / 256)
    R_11 = np.linalg.solve(G_11 + lam_eff_11 * np.eye(256, dtype=np.float32), Z_proj_11.T @ f_cent)
    B_ridge_11 = (V_r_11 @ R_11).astype(np.float32)
    b_ridge_11 = np.mean(f_pairs, axis=0) - np.mean(H11_pairs, axis=0) @ B_ridge_11
    surrogates["S2_prefix11_ridge256"] = AffineSuffixSurrogate(k=11, weights=weights, B=B_ridge_11, b=b_ridge_11, label="S2_prefix11_ridge256")

    # S2b: Depth 13 + Weight-Aware Gated Suffix
    k = 13
    B_13 = np.eye(width, dtype=np.float32)
    for l in range(k + 1, 16):
        gate_probs = np.mean(act_pilot[l] > 0.0, axis=0).astype(np.float32)
        B_13 = (B_13 @ weights[l]) * gate_probs[None, :]

    mean_H13_pilot = np.mean(act_pilot[k], axis=0)
    b_13 = mean_f_pilot - mean_H13_pilot @ B_13
    surrogates["S2b_prefix13_gated"] = AffineSuffixSurrogate(k=13, weights=weights, B=B_13, b=b_13, label="S2b_prefix13_gated")

    # S3: Width-512 Pruned Suffix from Layer 4
    surrogates["S3_pruned_w512"] = PrunedSuffixSurrogate(
        weights=weights,
        pilot_activations=act_pilot,
        K_retained=512,
        label="S3_pruned_w512",
    )

    # S4: Width-256 Pruned Suffix from Layer 4
    surrogates["S4_pruned_w256"] = PrunedSuffixSurrogate(
        weights=weights,
        pilot_activations=act_pilot,
        K_retained=256,
        label="S4_pruned_w256",
    )

    return surrogates


def evaluate_loo_blend(
    y_calib_list: List[np.ndarray],
    y_sample_list: List[np.ndarray],
    y_true_list: List[np.ndarray],
) -> Tuple[float, List[float], float]:
    """Evaluates outer leave-one-MLP-out (LOO) optimal scalar blending."""
    n_mlps = len(y_true_list)
    loo_errors = []
    alphas_fitted = []

    for i in range(n_mlps):
        c_train = [y_calib_list[j] for j in range(n_mlps) if j != i]
        s_train = [y_sample_list[j] for j in range(n_mlps) if j != i]
        y_train = [y_true_list[j] for j in range(n_mlps) if j != i]

        def get_alpha(c_tr, s_tr, y_tr):
            num = 0.0
            den = 0.0
            for c, s, y in zip(c_tr, s_tr, y_tr):
                diff = s - c
                num += np.dot(y - c, diff)
                den += np.dot(diff, diff)
            if den < 1e-12:
                return 0.110
            return float(np.clip(num / den, 0.0, 1.0))

        alpha_loo = get_alpha(c_train, s_train, y_train)
        alphas_fitted.append(alpha_loo)

        pred = (1.0 - alpha_loo) * y_calib_list[i] + alpha_loo * y_sample_list[i]
        err = float(np.mean((pred - y_true_list[i]) ** 2))
        loo_errors.append(err)

    mean_alpha = float(np.mean(alphas_fitted))
    return float(np.mean(loo_errors)), loo_errors, mean_alpha


def run_p4n_07():
    print("=" * 90)
    print("P4N-07: CHEAP SURROGATE + COUPLED RESIDUAL ON 8 PHASE 2 MLPS")
    print("=" * 90)

    ds = load_dataset(DATASET_PATH, split="mini")
    protocol_version, salt = resolve_seed_context(ds)

    mlp_records = []
    ds_8 = [ds[i] for i in range(8)]
    for idx, row in enumerate(ds_8):
        mlp = MLP.from_row(row, seed_protocol_version=protocol_version, seed_salt=salt)
        mlp_name = row.get("mlp_name", f"mlp_{idx}")
        weights_np = [np.array(w, dtype=np.float32) for w in mlp.weights]
        y_true = np.asarray(row["all_layer_means"][-1], dtype=np.float32)
        c_analytic = compute_analytic_mean(mlp, weights_np)
        c_calib = c_analytic * _S_PRIOR
        mlp_records.append({
            "name": mlp_name,
            "mlp": mlp,
            "weights": weights_np,
            "y_true": y_true,
            "c_analytic": c_analytic,
            "c_calib": c_calib,
        })

    # Full f FLOPs per sample: 16 * 2 * 1024^2
    f_flops_per_sample = 16 * 2 * (1024 ** 2)
    f_flops_per_pair = 2 * f_flops_per_sample
    pilot_cost_per_mlp = 1024 * f_flops_per_sample

    print(f"\nEvaluating on {len(mlp_records)} Phase 2 MLPs.")
    print(f"Full f FLOPs/sample: {f_flops_per_sample:,.0f} | FLOPs/pair: {f_flops_per_pair:,.0f}")
    print(f"Pilot construction FLOPs: {pilot_cost_per_mlp:,.0f} ({pilot_cost_per_mlp / _BUDGET_FLOPS * 100:.2f}% of budget)")

    # --------------------------------------------------------------------------
    # Step 1: Measure Variance and FLOPs for all Surrogates (N=2048 held pairs)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("STEP 1: MEASURE VARIANCE, REDUCTION RATIOS, AND THEORETICAL ALLOCATION")
    print("=" * 90)

    surrogate_names = [
        "S1_prefix7_gated",
        "S1_prefix7_ridge256",
        "S2_prefix11_gated",
        "S2_prefix11_ridge256",
        "S2b_prefix13_gated",
        "S3_pruned_w512",
        "S4_pruned_w256",
    ]

    N_MEASURE_PAIRS = 2048
    measure_draws = []
    for item in mlp_records:
        mlp = item["mlp"]
        rng = np.random.default_rng(mlp.seed + 70001)
        X_eval = make_antipodal_pairs(rng, n_pairs=N_MEASURE_PAIRS, width=mlp.width)
        f_eval, act_eval = forward_full(X_eval, item["weights"])
        f_pairs = pair_average(f_eval, N_MEASURE_PAIRS)
        measure_draws.append({
            "X_eval": X_eval,
            "f_pairs": f_pairs,
            "act_eval": act_eval,
        })

    all_surrogates: List[Dict[str, Any]] = []
    for item in mlp_records:
        mlp = item["mlp"]
        rng_pilot = np.random.default_rng(mlp.seed + 70000)
        surrs = build_surrogates_for_mlp(mlp, item["weights"], rng_pilot)
        all_surrogates.append(surrs)

    metrics_by_surrogate = {}

    for s_name in surrogate_names:
        vg_list = []
        vd_list = []
        vf_list = []
        corr_list = []
        cg_list = []
        cd_list = []

        for i, item in enumerate(mlp_records):
            surr = all_surrogates[i][s_name]
            X_eval = measure_draws[i]["X_eval"]
            f_pairs = measure_draws[i]["f_pairs"]

            if isinstance(surr, AffineSuffixSurrogate):
                Hk = measure_draws[i]["act_eval"][surr.k]
                g_eval = surr.evaluate_from_Hk(Hk)
                cg_sample = surr.flops_per_sample
                cd_sample = f_flops_per_sample + surr.suffix_flops
            else:
                g_eval = surr.evaluate_from_X(X_eval)
                cg_sample = surr.flops_per_sample
                cd_sample = f_flops_per_sample + surr.flops_per_sample

            g_pairs = pair_average(g_eval, N_MEASURE_PAIRS)
            d_pairs = f_pairs - g_pairs

            var_f = float(np.mean(np.var(f_pairs, axis=0)))
            var_g = float(np.mean(np.var(g_pairs, axis=0)))
            var_d = float(np.mean(np.var(d_pairs, axis=0)))

            cov_fg = float(np.mean(np.mean((f_pairs - np.mean(f_pairs, axis=0)) * (g_pairs - np.mean(g_pairs, axis=0)), axis=0)))
            corr = cov_fg / max(np.sqrt(var_f * var_g), 1e-12)

            vg_list.append(var_g)
            vd_list.append(var_d)
            vf_list.append(var_f)
            corr_list.append(corr)
            cg_list.append(2 * cg_sample)
            cd_list.append(2 * cd_sample)

        mean_vg = float(np.mean(vg_list))
        mean_vd = float(np.mean(vd_list))
        mean_vf = float(np.mean(vf_list))
        mean_corr = float(np.mean(corr_list))
        mean_cg = float(np.mean(cg_list))
        mean_cd = float(np.mean(cd_list))

        ratio_vd_vf = mean_vd / mean_vf
        cg_fraction = mean_cg / f_flops_per_pair
        cd_fraction = mean_cd / f_flops_per_pair

        optimal_ratio_ng_nd = np.sqrt((mean_vg * mean_cd) / max(mean_vd * mean_cg, 1e-12))
        v_opt_factor = (np.sqrt(mean_vg * mean_cg) + np.sqrt(mean_vd * mean_cd)) ** 2
        v_ord_factor = mean_vf * f_flops_per_pair
        predicted_gain_pct = (1.0 - v_opt_factor / v_ord_factor) * 100.0

        metrics_by_surrogate[s_name] = {
            "s_name": s_name,
            "mean_vf": mean_vf,
            "mean_vg": mean_vg,
            "mean_vd": mean_vd,
            "ratio_vd_vf": ratio_vd_vf,
            "mean_corr": mean_corr,
            "cg_fraction": cg_fraction,
            "cd_fraction": cd_fraction,
            "optimal_ratio_ng_nd": float(optimal_ratio_ng_nd),
            "predicted_gain_pct": float(predicted_gain_pct),
        }

    print(f"{'Surrogate':<22} | {'Vd / Vf':<10} | {'Corr(f,g)':<10} | {'cg/cf':<8} | {'cd/cf':<8} | {'Ng/Nd':<8} | {'Pred Gain (%)':<14}")
    print("-" * 90)
    for s_name in surrogate_names:
        m = metrics_by_surrogate[s_name]
        print(f"{s_name:<22} | {m['ratio_vd_vf']*100:>8.2f}% | {m['mean_corr']:>10.4f} | {m['cg_fraction']:>8.2f} | {m['cd_fraction']:>8.2f} | {m['optimal_ratio_ng_nd']:>8.2f} | {m['predicted_gain_pct']:>+13.2f}%")

    # --------------------------------------------------------------------------
    # Step 2: Empirical Evaluation of Two-Level Estimator at Target Budgets
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("STEP 2: EMPIRICAL EVALUATION OF TWO-LEVEL ESTIMATOR AT OPERATING BUDGETS")
    print("=" * 90)

    BUDGET_UTILIZATIONS = [0.098, 0.150, 0.200]
    empirical_results = {}

    for s_name in surrogate_names:
        empirical_results[s_name] = {}
        m = metrics_by_surrogate[s_name]
        ratio_ng_nd = m["optimal_ratio_ng_nd"]
        cg = m["cg_fraction"] * f_flops_per_pair
        cd = m["cd_fraction"] * f_flops_per_pair

        for util in BUDGET_UTILIZATIONS:
            total_target_flops = util * _BUDGET_FLOPS
            avail_flops = total_target_flops - pilot_cost_per_mlp
            if avail_flops <= 0:
                continue

            Nd_exact = avail_flops / (ratio_ng_nd * cg + cd)
            Ng_exact = ratio_ng_nd * Nd_exact

            nd_pairs = max(int(np.round(Nd_exact)), 16)
            ng_pairs = max(int(np.round(Ng_exact)), 16)

            actual_flops = pilot_cost_per_mlp + ng_pairs * cg + nd_pairs * cd
            actual_util = actual_flops / _BUDGET_FLOPS
            multiplier = max(0.10, actual_util)

            avail_ord_flops = total_target_flops
            n_ord_pairs = max(int(avail_ord_flops / f_flops_per_pair), 32)

            y_two_level_list = []
            y_ord_list = []
            y_true_list = []
            y_calib_list = []

            for i, item in enumerate(mlp_records):
                mlp = item["mlp"]
                surr = all_surrogates[i][s_name]
                weights = item["weights"]
                y_true = item["y_true"]
                c_calib = item["c_calib"]

                # STREAM A: Independent draws for g(X)
                rng_A = np.random.default_rng(mlp.seed + 100000 + int(util * 1000))
                X_A = make_antipodal_pairs(rng_A, n_pairs=ng_pairs, width=mlp.width)
                if isinstance(surr, AffineSuffixSurrogate):
                    Hk_A = forward_prefix(X_A, weights, surr.k)
                    g_A = surr.evaluate_from_Hk(Hk_A)
                else:
                    g_A = surr.evaluate_from_X(X_A)
                mu_g = np.mean(g_A, axis=0)

                # STREAM B: Independent draws for difference f(X) - g(X)
                rng_B = np.random.default_rng(mlp.seed + 200000 + int(util * 1000))
                X_B = make_antipodal_pairs(rng_B, n_pairs=nd_pairs, width=mlp.width)
                if isinstance(surr, AffineSuffixSurrogate):
                    f_B, act_B = forward_full(X_B, weights)
                    g_B = surr.evaluate_from_Hk(act_B[surr.k])
                else:
                    f_B, _ = forward_full(X_B, weights)
                    g_B = surr.evaluate_from_X(X_B)

                diff_B = f_B - g_B
                mu_diff = np.mean(diff_B, axis=0)

                y_two_level = mu_g + mu_diff

                # Ordinary MC at matched cost
                rng_ord = np.random.default_rng(mlp.seed + 300000 + int(util * 1000))
                X_ord = make_antipodal_pairs(rng_ord, n_pairs=n_ord_pairs, width=mlp.width)
                f_ord, _ = forward_full(X_ord, weights)
                y_ord = np.mean(f_ord, axis=0)

                y_two_level_list.append(y_two_level)
                y_ord_list.append(y_ord)
                y_true_list.append(y_true)
                y_calib_list.append(c_calib)

            raw_mse_two_level = float(np.mean([np.mean((y - yt) ** 2) for y, yt in zip(y_two_level_list, y_true_list)]))
            raw_mse_ord = float(np.mean([np.mean((y - yt) ** 2) for y, yt in zip(y_ord_list, y_true_list)]))

            loo_mse_tl, _, alpha_tl = evaluate_loo_blend(y_calib_list, y_two_level_list, y_true_list)
            loo_mse_ord, _, alpha_ord = evaluate_loo_blend(y_calib_list, y_ord_list, y_true_list)

            scored_tl = loo_mse_tl * multiplier
            scored_ord = loo_mse_ord * max(0.10, (n_ord_pairs * f_flops_per_pair) / _BUDGET_FLOPS)

            gain_vs_ord_raw = (1.0 - raw_mse_two_level / raw_mse_ord) * 100.0
            gain_vs_ord_scored = (1.0 - scored_tl / scored_ord) * 100.0
            gain_vs_control = (1.0 - scored_tl / _CONTROL_SCORE) * 100.0

            empirical_results[s_name][str(util)] = {
                "util": util,
                "actual_util": float(actual_util),
                "multiplier": float(multiplier),
                "ng_pairs": ng_pairs,
                "nd_pairs": nd_pairs,
                "raw_mse_two_level": raw_mse_two_level,
                "raw_mse_ord": raw_mse_ord,
                "loo_mse_tl": loo_mse_tl,
                "loo_mse_ord": loo_mse_ord,
                "alpha_tl": float(alpha_tl),
                "scored_tl": scored_tl,
                "scored_ord": scored_ord,
                "gain_vs_ord_raw_pct": float(gain_vs_ord_raw),
                "gain_vs_ord_scored_pct": float(gain_vs_ord_scored),
                "gain_vs_control_pct": float(gain_vs_control),
            }

            print(f"[{s_name} | util={util:.3f}] (Ng={ng_pairs}, Nd={nd_pairs})")
            print(f"  Raw MSE:  Two-Level={raw_mse_two_level:.6e} vs Ord={raw_mse_ord:.6e} ({gain_vs_ord_raw:+.2f}%)")
            print(f"  LOO MSE:  Two-Level={loo_mse_tl:.6e} (alpha={alpha_tl:.3f}) vs Ord={loo_mse_ord:.6e}")
            print(f"  Scored:   Two-Level={scored_tl:.6e} vs Control={_CONTROL_SCORE:.6e} ({gain_vs_control:+.2f}%)")

    # --------------------------------------------------------------------------
    # Step 3: Check Pre-Port Gate
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("STEP 3: GATE AUDIT & SUMMARY")
    print("=" * 90)

    best_s_name = None
    best_scored = float("inf")
    best_config = None

    for s_name, res_dict in empirical_results.items():
        for util_str, r in res_dict.items():
            if r["scored_tl"] < best_scored:
                best_scored = r["scored_tl"]
                best_s_name = s_name
                best_config = r

    print(f"Best Configuration: {best_s_name} at util={best_config['util']}")
    print(f"  Scored Error: {best_scored:.6e} (Control: {_CONTROL_SCORE:.6e}, Gain: {best_config['gain_vs_control_pct']:+.2f}%)")
    print(f"  Raw MSE:      {best_config['raw_mse_two_level']:.6e} (vs Ord: {best_config['raw_mse_ord']:.6e}, Gain: {best_config['gain_vs_ord_raw_pct']:+.2f}%)")

    pass_gate = (best_config["gain_vs_ord_raw_pct"] >= 30.0) or (best_config["gain_vs_control_pct"] >= 20.0)

    print(f"\nPre-Port Gate Status: {'PASSED' if pass_gate else 'FAILED'}")
    print(f"  - Gain over matched-cost sampling >= 30%: {best_config['gain_vs_ord_raw_pct']:.2f}% (Threshold: 30.0%)")
    print(f"  - Scored gain over control >= 20%:        {best_config['gain_vs_control_pct']:.2f}% (Threshold: 20.0%)")

    # --------------------------------------------------------------------------
    # Save complete diagnostic records
    # --------------------------------------------------------------------------
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DIAGNOSTICS_DIR / "p4n_07_surrogate.json"

    diagnostic_payload = {
        "metrics_by_surrogate": metrics_by_surrogate,
        "empirical_results": empirical_results,
        "best_configuration": {
            "surrogate": best_s_name,
            "metrics": best_config,
        },
        "gate_passed": pass_gate,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diagnostic_payload, f, indent=2, default=numpy_json_default)

    print(f"\nDiagnostic results saved to: {out_file}")


if __name__ == "__main__":
    run_p4n_07()
