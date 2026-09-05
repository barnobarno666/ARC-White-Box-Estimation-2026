import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _whitened_antithetic_mc, _hermite_gain_covariance

def compute_thresh_cov(mlp: MLP, eta_2: float, eta_3: float = 0.0, eta_4: float = 0.0):
    width = mlp.width
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
    zero = fnp.asarray(0.0, dtype=fnp.float32)
    half = fnp.asarray(0.5, dtype=fnp.float32)
    one_sixth = fnp.asarray(1.0 / 6.0, dtype=fnp.float32)
    one_24th = fnp.asarray(1.0 / 24.0, dtype=fnp.float32)

    eta2_fnp = fnp.asarray(eta_2, dtype=fnp.float32)
    eta3_fnp = fnp.asarray(eta_3, dtype=fnp.float32)
    eta4_fnp = fnp.asarray(eta_4, dtype=fnp.float32)

    for weight in mlp.weights:
        w = fnp.asarray(weight, dtype=fnp.float32)
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), fnp.asarray(1e-12, dtype=fnp.float32))
        sigma_pre = fnp.sqrt(var_pre)
        a = mu_pre / sigma_pre
        phi = flops.stats.norm.pdf(a).astype(fnp.float32)
        cdf = flops.stats.norm.cdf(a).astype(fnp.float32)

        mu = mu_pre * cdf + sigma_pre * phi
        second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
        var_post = fnp.maximum(second - mu * mu, zero)
        gain = fnp.where(sigma_pre > 1e-12, cdf, zero)

        # 1. Linear term
        cov_linear = fnp.multiply(fnp.outer(gain, gain), cov_pre)

        # 2. Threshold-aware quadratic
        u = fnp.where(sigma_pre > 1e-12, phi / sigma_pre, zero)
        cov_quad = half * fnp.multiply(fnp.outer(u, u), cov_pre * cov_pre)

        cov_total = cov_linear + eta2_fnp * cov_quad

        # 3. Threshold-aware cubic (if eta_3 > 0)
        if eta_3 > 0:
            u_cube = fnp.where(sigma_pre > 1e-12, (a * phi) / (sigma_pre * sigma_pre), zero)
            cov_cube = one_sixth * fnp.multiply(fnp.outer(u_cube, u_cube), cov_pre * cov_pre * cov_pre)
            cov_total = cov_total + eta3_fnp * cov_cube

        # 4. Threshold-aware quartic (if eta_4 > 0)
        if eta_4 > 0:
            u_quart = fnp.where(sigma_pre > 1e-12, ((a * a - 1.0) * phi) / (sigma_pre * sigma_pre * sigma_pre), zero)
            cov_quart = one_24th * fnp.multiply(fnp.outer(u_quart, u_quart), (cov_pre * cov_pre) * (cov_pre * cov_pre))
            cov_total = cov_total + eta4_fnp * cov_quart

        fnp.fill_diagonal(cov_total, var_post)
        cov = flops.as_symmetric(cov_total, symmetry=(0, 1))

    return np.array(mu, dtype=np.float64)

def run_diagnostics():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("LANE A1 DIAGNOSTICS: SCALE, COVARIANCE MSE, AND FUSED MSE")
    print("=" * 90)

    # Preload ground truth and MC
    y_trues = []
    m_list = []
    mlps = []
    for i in range(8):
        row = ds[i]
        mlps.append(MLP.from_row(row, seed_protocol_version=v, seed_salt=s))
        y_trues.append(np.array(row["final_means"], dtype=np.float64))
        m_list.append(np.array(_whitened_antithetic_mc(mlps[-1])[-1], dtype=np.float64))

    # Test settings
    configs = [
        ("Centered Champion (gamma=0.20)", None, 0.20, 0.0, 0.0, 0.998319),
        ("Thresh Quad eta=0.20 (s0=0.998319)", "thresh", 0.20, 0.0, 0.0, 0.998319),
        ("Thresh Quad eta=0.50 (s0=0.998319)", "thresh", 0.50, 0.0, 0.0, 0.998319),
        ("Thresh Quad eta=1.00 (s0=0.998319)", "thresh", 1.00, 0.0, 0.0, 0.998319),
        ("Thresh Quad+Cube (eta2=0.5, eta3=0.5)", "thresh", 0.50, 0.5, 0.0, 0.998319),
        ("Thresh Quad+Cube+Quart (eta2=0.5, eta3=0.5, eta4=0.5)", "thresh", 0.50, 0.5, 0.5, 0.998319),
    ]

    for name, mode, eta2, eta3, eta4, s_prior in configs:
        sy_list = []
        cov_mse_list = []
        prior_fused_mse_list = []
        oracle_fused_mse_list = []

        for i in range(8):
            mlp = mlps[i]
            y = y_trues[i]
            m = m_list[i]

            if mode is None:
                c = np.array(_hermite_gain_covariance(mlp)[-1], dtype=np.float64)
            else:
                c = compute_thresh_cov(mlp, eta2, eta3, eta4)

            # Measure empirical scale sy
            sy = float(np.sum(y * c) / np.sum(c * c))
            sy_list.append(sy)

            # Raw unscaled cov MSE
            cov_mse = float(np.mean((c - y) ** 2))
            cov_mse_list.append(cov_mse)

            # Prior scaled fused MSE (0.89 * s_prior * c + 0.11 * m)
            fused_prior = 0.89 * (s_prior * c) + 0.11 * m
            prior_fused_mse_list.append(float(np.mean((fused_prior - y) ** 2)))

            # Mean-sy scaled fused MSE (using average sy across other 7 MLPs - LOO)
            # We'll compute LOO after gathering sy_list

        mean_sy = np.mean(sy_list)
        std_sy = np.std(sy_list)
        mean_cov_mse = np.mean(cov_mse_list)
        mean_prior_fused = np.mean(prior_fused_mse_list)

        # Compute LOO fused score
        loo_fused = []
        for i in range(8):
            loo_s = np.mean([sy_list[j] for j in range(8) if j != i])
            c = np.array(_hermite_gain_covariance(mlps[i])[-1] if mode is None else compute_thresh_cov(mlps[i], eta2, eta3, eta4), dtype=np.float64)
            fused_loo = 0.89 * (loo_s * c) + 0.11 * m_list[i]
            loo_fused.append(float(np.mean((fused_loo - y_trues[i]) ** 2)))
        mean_loo_fused = np.mean(loo_fused)

        print(f"\n--- {name} ---")
        print(f"Empirical sy:    {mean_sy:.6f} +/- {std_sy:.6f}")
        print(f"Raw Cov MSE:     {mean_cov_mse:.4e}")
        print(f"Prior Fused MSE: {mean_prior_fused:.6e} (Adj: {mean_prior_fused*0.1:.6e})")
        print(f"LOO-s Fused MSE: {mean_loo_fused:.6e} (Adj: {mean_loo_fused*0.1:.6e})")

if __name__ == "__main__":
    run_diagnostics()
