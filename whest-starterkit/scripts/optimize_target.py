import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from candidates.estimator_calibrated_hermite import _whitened_antithetic_mc

def hermite_cov_advanced(mlp: MLP, gamma: float = 0.50, delta: float = 0.0) -> fnp.ndarray:
    """Gain-covariance propagation with 2nd-order and optional 4th-order Hermite corrections."""
    width = mlp.width
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
    zero = fnp.asarray(0.0, dtype=fnp.float32)
    inv_4pi = fnp.asarray(0.07957747154594767, dtype=fnp.float32)  # 1 / (4 * pi)
    inv_96pi = fnp.asarray(0.003315727981081153, dtype=fnp.float32) # 1 / (96 * pi)

    g_val = fnp.asarray(gamma, dtype=fnp.float32)
    d_val = fnp.asarray(delta, dtype=fnp.float32)

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
        var_post = fnp.maximum(second - mu * mu, zero)
        gain = fnp.where(sigma_pre > fnp.asarray(1e-12, dtype=fnp.float32), cdf, zero)

        cov_linear = fnp.multiply(fnp.outer(gain, gain), cov_pre)

        inv_sigma = fnp.where(sigma_pre > 1e-12, 1.0 / sigma_pre, zero)
        rho = fnp.multiply(fnp.outer(inv_sigma, inv_sigma), cov_pre)
        rho2 = rho * rho
        sigma_outer = fnp.outer(sigma_pre, sigma_pre)
        cov_quad = inv_4pi * rho2 * sigma_outer

        cov = cov_linear + g_val * cov_quad
        if delta > 0.0:
            rho4 = rho2 * rho2
            cov_quartic = inv_96pi * rho4 * sigma_outer
            cov = cov + d_val * cov_quartic

        fnp.fill_diagonal(cov, var_post)
        cov = flops.as_symmetric(cov, symmetry=(0, 1))

    return mu

def run_optimization():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    with open("scripts/baseline_8mlp.json") as f:
        base = json.load(f)
    base_scores = [m["adjusted_final_layer_score"] for m in base["per_mlp"]]

    print("=" * 80)
    print("TARGET < 1.24e-7 OPTIMIZATION & LEAVE-ONE-OUT VALIDATION")
    print("=" * 80)

    mlps = []
    truths = []
    mcs = []
    names = []
    for i in range(8):
        row = ds[i]
        names.append(row["mlp_name"])
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        mlps.append(mlp)
        truths.append(fnp.asarray(row["final_means"], dtype=fnp.float32))
        mcs.append(_whitened_antithetic_mc(mlp)[15])

    print("\n1. SWEEPING GAMMA (Hermite 2nd-order weight) with Optimal Scale & Blend:")
    best_gamma_score = 999.0
    best_gamma = None
    best_gamma_covs = None

    for gamma in [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]:
        covs = [hermite_cov_advanced(mlp, gamma=gamma, delta=0.0) for mlp in mlps]
        # Calculate true sy for each network
        sy_list = [float(fnp.sum(truths[i] * covs[i]) / fnp.sum(covs[i] * covs[i])) for i in range(8)]
        
        # Test Leave-one-out CV
        scores = []
        for i in range(8):
            other_sy = [sy_list[j] for j in range(8) if j != i]
            s_0_loo = sum(other_sy) / 7.0
            pred = 0.88 * (s_0_loo * covs[i]) + 0.12 * mcs[i]
            mse = float(fnp.mean((pred - truths[i])**2))
            scores.append(mse * 0.1000)
        
        avg_s = sum(scores) / 8.0
        print(f"Gamma={gamma:.2f} | Mean sy={sum(sy_list)/8:.6f} | LOO Score: {avg_s:.6e} | Max: {max(scores):.4e}")
        if avg_s < best_gamma_score:
            best_gamma_score = avg_s
            best_gamma = gamma
            best_gamma_covs = covs

    print(f"\nBest Gamma: {best_gamma:.2f} with LOO Score: {best_gamma_score:.6e}")

    print("\n2. SWEEPING DELTA (Hermite 4th-order quartic weight):")
    for delta in [0.0, 0.2, 0.5, 1.0, 2.0]:
        covs = [hermite_cov_advanced(mlp, gamma=best_gamma, delta=delta) for mlp in mlps]
        sy_list = [float(fnp.sum(truths[i] * covs[i]) / fnp.sum(covs[i] * covs[i])) for i in range(8)]
        scores = []
        for i in range(8):
            other_sy = [sy_list[j] for j in range(8) if j != i]
            s_0_loo = sum(other_sy) / 7.0
            pred = 0.88 * (s_0_loo * covs[i]) + 0.12 * mcs[i]
            mse = float(fnp.mean((pred - truths[i])**2))
            scores.append(mse * 0.1000)
        print(f"Delta={delta:.1f} | Mean sy={sum(sy_list)/8:.6f} | LOO Score: {sum(scores)/8:.6e} | Max: {max(scores):.4e}")

    print("\n3. FINE-GRID OPTIMIZATION: Alpha & Scale Prior on Best Covariance:")
    # Using best_gamma
    covs = best_gamma_covs
    sy_list = [float(fnp.sum(truths[i] * covs[i]) / fnp.sum(covs[i] * covs[i])) for i in range(8)]
    s_0 = sum(sy_list) / 8.0
    print(f"Grand Mean Scale s_0 = {s_0:.6f}")

    print(f"\n{'Alpha':<7} | {'Scale Offset':<14} | {'Avg Score':<12} | {'Wins':<6} | {'Worst Score':<14}")
    print("-" * 65)

    best_final_score = 999.0
    best_final_config = None

    for alpha in [0.08, 0.09, 0.10, 0.11, 0.115, 0.12, 0.125, 0.13, 0.14]:
        for s_mult in [0.9997, 0.9998, 0.9999, 1.0000, 1.0001, 1.0002, 1.0003]:
            scores = []
            wins = 0
            for i in range(8):
                # Strict LOO
                other_sy = [sy_list[j] for j in range(8) if j != i]
                s_loo = (sum(other_sy) / 7.0) * s_mult
                pred = (1.0 - alpha) * (s_loo * covs[i]) + alpha * mcs[i]
                mse = float(fnp.mean((pred - truths[i])**2))
                score = mse * 0.1000
                scores.append(score)
                if score < base_scores[i]:
                    wins += 1
            avg_s = sum(scores) / 8.0
            if avg_s < best_final_score:
                best_final_score = avg_s
                best_final_config = (alpha, s_mult, wins, max(scores))
            if s_mult == 1.0000:
                print(f"{alpha:<7.3f} | {s_mult:<14.4f} | {avg_s:.6e} | {wins}W-{8-wins}L | {max(scores):.4e}")

    print("\n" + "=" * 80)
    print("CHAMPION CANDIDATE DISCOVERED:")
    print(f"Gamma:        {best_gamma:.2f}")
    print(f"Alpha:        {best_final_config[0]:.3f}")
    print(f"Scale Mult:   {best_final_config[1]:.4f}")
    print(f"LOO Score:    {best_final_score:.6e} ({best_final_config[2]}W-{8-best_final_config[2]}L)")
    print(f"Worst MLP:    {best_final_config[3]:.4e}")
    print("=" * 80)

if __name__ == "__main__":
    run_optimization()
