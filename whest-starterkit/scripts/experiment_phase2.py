import json
import sys
from pathlib import Path
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from candidates.estimator_calibrated_hermite import _hermite_gain_covariance

def get_final_moments_and_mc(mlp: MLP, count: int = 4200, seed: int = None):
    """Compute Hermite analytic moments and sampled preactivations + activations."""
    width = mlp.width
    depth = mlp.depth
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
    zero = fnp.asarray(0.0, dtype=fnp.float32)
    inv_4pi = fnp.asarray(0.07957747154594767, dtype=fnp.float32)
    gamma = fnp.asarray(0.50, dtype=fnp.float32)

    # Propagate Hermite covariance up to layer 15
    for layer_idx, weight in enumerate(mlp.weights):
        w = fnp.asarray(weight, dtype=fnp.float32)
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), fnp.asarray(1e-12, dtype=fnp.float32))
        sigma_pre = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        phi = flops.stats.norm.pdf(alpha).astype(fnp.float32)
        cdf = flops.stats.norm.cdf(alpha).astype(fnp.float32)

        mu_post = mu_pre * cdf + sigma_pre * phi
        second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
        var_post = fnp.maximum(second - mu_post * mu_post, zero)
        gain = fnp.where(sigma_pre > fnp.asarray(1e-12, dtype=fnp.float32), cdf, zero)

        cov_linear = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        inv_sigma = fnp.where(sigma_pre > 1e-12, 1.0 / sigma_pre, zero)
        rho = fnp.multiply(fnp.outer(inv_sigma, inv_sigma), cov_pre)
        rho2 = rho * rho
        sigma_outer = fnp.outer(sigma_pre, sigma_pre)
        cov_quad = inv_4pi * rho2 * sigma_outer

        cov = cov_linear + gamma * cov_quad
        fnp.fill_diagonal(cov, var_post)
        cov = flops.as_symmetric(cov, symmetry=(0, 1))
        mu = mu_post

        if layer_idx == depth - 1:
            # Save final preactivation moments and components
            final_mu_pre = mu_pre
            final_var_pre = var_pre
            final_sigma_pre = sigma_pre
            final_alpha = alpha
            final_phi = phi
            final_cdf = cdf
            final_c = mu_post

    # Whitened MC sampling with preactivation extraction
    half = count // 2
    scale = fnp.asarray(1.0 / count, dtype=fnp.float32)
    rng = fnp.random.default_rng(seed if seed is not None else mlp.seed)

    x_half = fnp.asarray(rng.standard_normal((half, width)), dtype=fnp.float32)
    x = fnp.concatenate((x_half, -x_half), axis=0)
    gram = (x.T @ x) / fnp.asarray(float(count), dtype=fnp.float32)
    eigenvalues, eigenvectors = fnp.linalg.eigh(gram)
    eigenvalues = fnp.maximum(eigenvalues, fnp.asarray(1e-6, dtype=fnp.float32))
    whitener = (eigenvectors * fnp.power(eigenvalues, -0.5)) @ eigenvectors.T
    first_weight = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
    act = fnp.maximum(x @ (whitener @ first_weight), zero)

    for layer in range(1, depth - 1):
        weight = fnp.asarray(mlp.weights[layer], dtype=fnp.float32)
        act = fnp.maximum(act @ weight, zero)

    # Final layer: extract pre-activations z
    final_w = fnp.asarray(mlp.weights[depth - 1], dtype=fnp.float32)
    z_final = act @ final_w
    act_final = fnp.maximum(z_final, zero)

    # Sample statistics
    m_raw = fnp.sum(act_final, axis=0) * scale
    z_bar = fnp.sum(z_final, axis=0) * scale
    z_sq_bar = fnp.sum((z_final - final_mu_pre)**2, axis=0) * scale

    return {
        "c": final_c,
        "mu_pre": final_mu_pre,
        "var_pre": final_var_pre,
        "sigma_pre": final_sigma_pre,
        "alpha": final_alpha,
        "phi": final_phi,
        "cdf": final_cdf,
        "m_raw": m_raw,
        "z_bar": z_bar,
        "z_sq_bar": z_sq_bar,
    }

def run_experiments():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    with open("scripts/baseline_8mlp.json") as f:
        base = json.load(f)
    base_scores = [m["adjusted_final_layer_score"] for m in base["per_mlp"]]

    print("=" * 80)
    print("PHASE 2A & 2B: ADVANCED PROJECTIONS & FINAL-LAYER HERMITE CONTROLS")
    print("=" * 80)

    # Data collection for all 8 MLPs
    data = []
    truths = []
    names = []
    for i in range(8):
        row = ds[i]
        names.append(row["mlp_name"])
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = fnp.asarray(row["final_means"], dtype=fnp.float32)
        truths.append(y)
        d = get_final_moments_and_mc(mlp)
        data.append(d)

    print("\n--- EXPERIMENT 1: FINAL-LAYER HERMITE CONTROL VARIATES (Phase 2B) ---")
    # Evaluate raw sampling variance reduction
    mse_m_raw = []
    mse_m_cv1 = []
    mse_m_cv2 = []

    for i in range(8):
        d = data[i]
        y = truths[i]

        # Raw MC
        m_raw = d["m_raw"]
        mse_raw = float(fnp.mean((m_raw - y)**2))
        mse_m_raw.append(mse_raw)

        # Order-1 Hermite Control Variate:
        # m_cv1 = ReLU(z) - Phi(a) * (z_bar - mu_pre)
        m_cv1 = m_raw - d["cdf"] * (d["z_bar"] - d["mu_pre"])
        mse_cv1 = float(fnp.mean((m_cv1 - y)**2))
        mse_m_cv1.append(mse_cv1)

        # Order-2 Hermite Control Variate:
        # m_cv2 = m_cv1 - phi(a) / (2 * sigma_pre) * ((z - mu)^2 - var_pre)
        m_cv2 = m_cv1 - (d["phi"] / (2.0 * d["sigma_pre"])) * (d["z_sq_bar"] - d["var_pre"])
        mse_cv2 = float(fnp.mean((m_cv2 - y)**2))
        mse_m_cv2.append(mse_cv2)

        print(f"[{i+1}/8] {names[i]:<22}: Raw MC = {mse_raw:.4e} | Hermite CV1 = {mse_cv1:.4e} ({((mse_cv1-mse_raw)/mse_raw)*100:+.1f}%) | Hermite CV2 = {mse_cv2:.4e} ({((mse_cv2-mse_raw)/mse_raw)*100:+.1f}%)")

    print(f"\nAverage Sample Variance Reduction across 8 MLPs:")
    print(f"  Raw WMC Mean MSE:       {sum(mse_m_raw)/8:.6e}")
    print(f"  Order-1 Hermite CV MSE: {sum(mse_m_cv1)/8:.6e} ({((sum(mse_m_cv1)-sum(mse_m_raw))/sum(mse_m_raw))*100:+.2f}%)")
    print(f"  Order-2 Hermite CV MSE: {sum(mse_m_cv2)/8:.6e} ({((sum(mse_m_cv2)-sum(mse_m_raw))/sum(mse_m_raw))*100:+.2f}%)")

    print("\n" + "=" * 80)
    print("--- EXPERIMENT 2: BLENDING CALIBRATED COVARIANCE WITH CONTROLLED SAMPLERS ---")
    print("=" * 80)

    # For each sampler (m_raw, m_cv1, m_cv2), find optimal scale projection & blend
    samplers = {
        "Raw WMC": [d["m_raw"] for d in data],
        "Order-1 Hermite CV": [d["m_raw"] - d["cdf"] * (d["z_bar"] - d["mu_pre"]) for d in data],
        "Order-2 Hermite CV": [d["m_raw"] - d["cdf"] * (d["z_bar"] - d["mu_pre"]) - (d["phi"] / (2.0 * d["sigma_pre"])) * (d["z_sq_bar"] - d["var_pre"]) for d in data],
    }

    for s_name, s_list in samplers.items():
        print(f"\nEvaluating Blends with Sampler: {s_name}")
        print(f"{'Alpha':<7} | {'Avg Score':<12} | {'Avg MSE':<12} | {'Wins vs Base':<14} | {'Worst Score':<14}")
        print("-" * 65)

        best_alpha = None
        best_score = 999.0
        for alpha in [0.00, 0.05, 0.08, 0.10, 0.12, 0.13, 0.15, 0.18, 0.20, 0.25, 0.30]:
            scores = []
            wins = 0
            for i in range(8):
                c = data[i]["c"]
                m = s_list[i]
                y = truths[i]

                denom = float(fnp.sum(c * c))
                scale = float(fnp.sum(m * c) / denom)
                c_cal = scale * c
                pred = (1.0 - alpha) * c_cal + alpha * m
                mse = float(fnp.mean((pred - y)**2))
                score = mse * 0.1000
                scores.append(score)
                if score < base_scores[i]:
                    wins += 1
            avg_score = sum(scores) / 8.0
            worst_score = max(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha
            print(f"{alpha:<7.3f} | {avg_score:.6e} | {avg_score/0.1:.6e} | {wins}W-{8-wins}L        | {worst_score:.4e}")
        print(f"--> Best for {s_name}: alpha={best_alpha:.3f} -> Score: {best_score:.6e}")

    print("\n" + "=" * 80)
    print("--- EXPERIMENT 3: MULTI-TEMPLATE PROJECTION BASES (Phase 2A) ---")
    print("=" * 80)
    # Test U = [c], U = [c, 1], U = [c, mu_phi], U = [c, 1, mu_phi, sigma_phi]
    for b_name, make_U in [
        ("1D: [c]", lambda d: fnp.stack([d["c"]], axis=1)),
        ("2D: [c, 1]", lambda d: fnp.stack([d["c"], fnp.ones_like(d["c"])], axis=1)),
        ("2D: [c, mu*Phi]", lambda d: fnp.stack([d["c"], d["mu_pre"] * d["cdf"]], axis=1)),
        ("2D: [c, sigma*phi]", lambda d: fnp.stack([d["c"], d["sigma_pre"] * d["phi"]], axis=1)),
        ("4D: [c, 1, mu*Phi, sigma*phi]", lambda d: fnp.stack([d["c"], fnp.ones_like(d["c"]), d["mu_pre"] * d["cdf"], d["sigma_pre"] * d["phi"]], axis=1)),
    ]:
        for ridge in [0.0, 1e-4, 1e-2, 1.0, 10.0]:
            scores = []
            wins = 0
            for i in range(8):
                U = make_U(data[i])  # (1024, K)
                m = samplers["Order-1 Hermite CV"][i]
                y = truths[i]
                # Regularized projection coefficients: w = (U.T @ U + lambda I)^{-1} @ U.T @ m
                K = U.shape[1]
                reg = ridge * fnp.eye(K, dtype=fnp.float32)
                w = fnp.linalg.solve(U.T @ U + reg, U.T @ m)
                c_proj = U @ w
                pred = 0.87 * c_proj + 0.13 * m
                mse = float(fnp.mean((pred - y)**2))
                score = mse * 0.1000
                scores.append(score)
                if score < base_scores[i]:
                    wins += 1
            avg_s = sum(scores) / 8.0
            print(f"Basis {b_name:<30} | Ridge={ridge:<5} | Avg Score: {avg_s:.6e} | {wins}W-{8-wins}L | Worst: {max(scores):.4e}")

if __name__ == "__main__":
    run_experiments()
