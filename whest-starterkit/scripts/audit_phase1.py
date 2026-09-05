import json
import sys
from pathlib import Path
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from candidates.estimator_calibrated_hermite import _hermite_gain_covariance, _whitened_antithetic_mc

def ordinary_gain_covariance(mlp: MLP) -> fnp.ndarray:
    """Standard gain-covariance without Hermite correction (gamma=0)."""
    width = mlp.width
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
    zero = fnp.asarray(0.0, dtype=fnp.float32)

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
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)
        cov = flops.as_symmetric(cov, symmetry=(0, 1))

    return mu

def plain_antithetic_mc(mlp: MLP, count: int = 4200, seed: int = None) -> fnp.ndarray:
    """Antithetic MC without whitening."""
    width, depth = mlp.width, mlp.depth
    half = count // 2
    scale = fnp.asarray(1.0 / count, dtype=fnp.float32)
    rng = fnp.random.default_rng(seed if seed is not None else mlp.seed)
    zero = fnp.asarray(0.0, dtype=fnp.float32)

    x_half = fnp.asarray(rng.standard_normal((half, width)), dtype=fnp.float32)
    x = fnp.concatenate((x_half, -x_half), axis=0)
    activations = fnp.maximum(x @ fnp.asarray(mlp.weights[0], dtype=fnp.float32), zero)

    for layer in range(1, depth):
        weight = fnp.asarray(mlp.weights[layer], dtype=fnp.float32)
        activations = fnp.maximum(activations @ weight, zero)
    return fnp.sum(activations, axis=0) * scale

def whitened_mc_seeded(mlp: MLP, count: int = 4200, seed: int = None) -> fnp.ndarray:
    """Whitened antithetic MC with explicit seed."""
    width, depth = mlp.width, mlp.depth
    half = count // 2
    scale = fnp.asarray(1.0 / count, dtype=fnp.float32)
    rng = fnp.random.default_rng(seed if seed is not None else mlp.seed)
    zero = fnp.asarray(0.0, dtype=fnp.float32)

    x_half = fnp.asarray(rng.standard_normal((half, width)), dtype=fnp.float32)
    x = fnp.concatenate((x_half, -x_half), axis=0)
    gram = (x.T @ x) / fnp.asarray(float(count), dtype=fnp.float32)
    eigenvalues, eigenvectors = fnp.linalg.eigh(gram)
    eigenvalues = fnp.maximum(eigenvalues, fnp.asarray(1e-6, dtype=fnp.float32))
    whitener = (eigenvectors * fnp.power(eigenvalues, -0.5)) @ eigenvectors.T
    first_weight = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
    activations = fnp.maximum(x @ (whitener @ first_weight), zero)

    for layer in range(1, depth):
        weight = fnp.asarray(mlp.weights[layer], dtype=fnp.float32)
        activations = fnp.maximum(activations @ weight, zero)
    return fnp.sum(activations, axis=0) * scale

def run_audit():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 80)
    print("PHASE 1 AUDIT: EVIDENCE BASE & SYSTEMATIC MEASUREMENTS (8 MLPs)")
    print("=" * 80)

    sm_list = []
    sy_list = []
    sm_minus_sy = []
    cos_errors = []
    
    mse_ord_cov = []
    mse_herm_cov = []
    mse_unscaled_wmc = []
    mse_unscaled_plain = []
    mse_oracle_scaled = []
    mse_mc_scaled = []
    mse_blend = []

    mlp_names = []

    for i in range(8):
        row = ds[i]
        name = row["mlp_name"]
        mlp_names.append(name)
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = fnp.asarray(row["final_means"], dtype=fnp.float32)

        # 1. Ordinary Covariance
        c_ord = ordinary_gain_covariance(mlp)
        mse_ord = float(fnp.mean((c_ord - y)**2))
        mse_ord_cov.append(mse_ord)

        # 2. Hermite Covariance
        c_herm = _hermite_gain_covariance(mlp)[15]
        mse_herm = float(fnp.mean((c_herm - y)**2))
        mse_herm_cov.append(mse_herm)

        # 3. Unscaled WMC (protocol seed)
        m_wmc = whitened_mc_seeded(mlp)
        mse_wmc = float(fnp.mean((m_wmc - y)**2))
        mse_unscaled_wmc.append(mse_wmc)

        # Plain antithetic MC
        m_plain = plain_antithetic_mc(mlp)
        mse_p = float(fnp.mean((m_plain - y)**2))
        mse_unscaled_plain.append(mse_p)

        # Scales: sm and sy
        denom = float(fnp.sum(c_herm * c_herm))
        sm = float(fnp.sum(m_wmc * c_herm) / denom)
        sy = float(fnp.sum(y * c_herm) / denom)
        sm_list.append(sm)
        sy_list.append(sy)
        sm_minus_sy.append(sm - sy)

        # 4. Oracle-scaled covariance
        c_oracle = sy * c_herm
        mse_oracle = float(fnp.mean((c_oracle - y)**2))
        mse_oracle_scaled.append(mse_oracle)

        # 5. MC-scaled covariance
        c_mc_scaled = sm * c_herm
        mse_mc_scale = float(fnp.mean((c_mc_scaled - y)**2))
        mse_mc_scaled.append(mse_mc_scale)

        # 6. Current blend (alpha=0.13)
        pred_blend = (1.0 - 0.13) * c_mc_scaled + 0.13 * m_wmc
        mse_bl = float(fnp.mean((pred_blend - y)**2))
        mse_blend.append(mse_bl)

        # Cosine / Covariance between e_c and e_m
        e_c = c_mc_scaled - y
        e_m = m_wmc - y
        dot_err = float(fnp.sum(e_c * e_m))
        norm_ec = float(fnp.sqrt(fnp.sum(e_c * e_c)))
        norm_em = float(fnp.sqrt(fnp.sum(e_m * e_m)))
        cos_sim = dot_err / (norm_ec * norm_em + 1e-12)
        cos_errors.append(cos_sim)

        print(f"[{i+1}/8] {name:<22}:")
        print(f"      sm={sm:.6f} | sy={sy:.6f} | sm - sy = {sm - sy:+.6e}")
        print(f"      Error Cosine sim(e_c, e_m) = {cos_sim:+.4f} (dot = {dot_err:+.4e})")
        print(f"      MSEs: OrdCov={mse_ord:.4e} | HermCov={mse_herm:.4e} | WMC={mse_wmc:.4e} | PlainMC={mse_p:.4e}")
        print(f"            OracleScaledCov={mse_oracle:.4e} | MCScaledCov={mse_mc_scale:.4e} | Blend={mse_bl:.4e}")

    print("\n" + "=" * 80)
    print("SUMMARY METRICS ACROSS 8 MLPs:")
    print("=" * 80)
    print(f"Mean sm:                 {sum(sm_list)/8:.6f} +/- {fnp.std(fnp.asarray(sm_list)):.6f}")
    print(f"Mean sy:                 {sum(sy_list)/8:.6f} +/- {fnp.std(fnp.asarray(sy_list)):.6f}")
    print(f"Mean (sm - sy):          {sum(sm_minus_sy)/8:+.6e} (std: {fnp.std(fnp.asarray(sm_minus_sy)):.6e})")
    print(f"Mean Error Cosine:       {sum(cos_errors)/8:+.4f} (range: [{min(cos_errors):+.4f}, {max(cos_errors):+.4f}])")
    print("-" * 80)
    print(f"Mean Raw MSE Ord Cov:    {sum(mse_ord_cov)/8:.6e} (Score: {sum(mse_ord_cov)*0.1/8:.6e})")
    print(f"Mean Raw MSE Herm Cov:   {sum(mse_herm_cov)/8:.6e} (Score: {sum(mse_herm_cov)*0.1/8:.6e})")
    print(f"Mean Raw MSE Plain MC:   {sum(mse_unscaled_plain)/8:.6e} (Score: {sum(mse_unscaled_plain)*0.1/8:.6e})")
    print(f"Mean Raw MSE WMC:        {sum(mse_unscaled_wmc)/8:.6e} (Score: {sum(mse_unscaled_wmc)*0.1/8:.6e})")
    print(f"Mean Raw MSE Oracle-Sc:  {sum(mse_oracle_scaled)/8:.6e} (Score: {sum(mse_oracle_scaled)*0.1/8:.6e})")
    print(f"Mean Raw MSE MC-Scaled:  {sum(mse_mc_scaled)/8:.6e} (Score: {sum(mse_mc_scaled)*0.1/8:.6e})")
    print(f"Mean Raw MSE Blend:      {sum(mse_blend)/8:.6e} (Score: {sum(mse_blend)*0.1/8:.6e})")
    print("-" * 80)
    print(f"Whitening Variance Reduction: WMC vs Plain: {((sum(mse_unscaled_wmc) - sum(mse_unscaled_plain))/sum(mse_unscaled_plain))*100:+.2f}%")
    print(f"Calibration Effect: Hermite -> MC-Scaled: {((sum(mse_mc_scaled) - sum(mse_herm_cov))/sum(mse_herm_cov))*100:+.2f}% MSE reduction")
    print(f"Residual Sub-optimality: MC-Scaled vs Oracle-Scaled: {(sum(mse_mc_scaled) - sum(mse_oracle_scaled)):.4e}")

if __name__ == "__main__":
    run_audit()
