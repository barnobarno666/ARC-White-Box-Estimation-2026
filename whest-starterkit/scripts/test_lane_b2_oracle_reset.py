import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _hermite_gain_covariance, _whitened_antithetic_mc, _S_PRIOR
from candidates.estimator_lane_a1_thresh_quad_eta1 import _threshold_aware_hermite_covariance

def run_oracle_moment_reset():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("LANE B2 MANDATORY ORACLE GATE: TRUE/HIGH-ACCURACY MOMENT RESET")
    print("=" * 90)

    count_oracle = 16384  # High sample count for oracle moments at layer k
    half_oracle = count_oracle // 2
    d = 1024

    for k_reset in [4, 6, 8, 10, 12]:
        print(f"\n--- Testing Oracle Moment Reset at Layer {k_reset} ---")
        reset_mses = []
        champ_mses = []
        reset_fused_mses = []

        for i in range(8):
            row = ds[i]
            mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
            y_true = np.array(row["final_means"], dtype=np.float64)

            # 1. Forward pass large sample batch to get oracle moments at layer k_reset
            rng = np.random.default_rng(mlp.seed + 9999)
            x_half = rng.standard_normal((half_oracle, d)).astype(np.float32)
            x = np.concatenate((x_half, -x_half), axis=0)

            # Whitening at layer 0
            gram = (x.T @ x) / float(count_oracle)
            eigvals, eigvecs = np.linalg.eigh(gram)
            eigvals = np.maximum(eigvals, 1e-6)
            whitener = (eigvecs * (eigvals ** -0.5)) @ eigvecs.T

            w0 = np.array(mlp.weights[0], dtype=np.float32)
            act = np.maximum(x @ (whitener @ w0), 0.0)

            for l in range(1, k_reset + 1):
                w = np.array(mlp.weights[l], dtype=np.float32)
                act = np.maximum(act @ w, 0.0)

            # Oracle moments at layer k_reset
            oracle_mu_k = np.mean(act, axis=0)
            act_centered = act - oracle_mu_k
            oracle_cov_k = (act_centered.T @ act_centered) / float(count_oracle)

            # 2. Restart threshold-aware covariance propagation from layer k_reset+1 to 15
            mu = fnp.asarray(oracle_mu_k, dtype=fnp.float32)
            cov = flops.as_symmetric(fnp.asarray(oracle_cov_k, dtype=fnp.float32), symmetry=(0, 1))

            zero = fnp.asarray(0.0, dtype=fnp.float32)
            half = fnp.asarray(0.5, dtype=fnp.float32)

            for l in range(k_reset + 1, mlp.depth):
                weight = mlp.weights[l]
                w_l = fnp.asarray(weight, dtype=fnp.float32)
                mu_pre = w_l.T @ mu
                cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w_l, w_l)
                var_pre = fnp.maximum(fnp.diag(cov_pre), fnp.asarray(1e-12, dtype=fnp.float32))
                sigma_pre = fnp.sqrt(var_pre)
                a = mu_pre / sigma_pre
                phi = flops.stats.norm.pdf(a).astype(fnp.float32)
                cdf = flops.stats.norm.cdf(a).astype(fnp.float32)

                mu = mu_pre * cdf + sigma_pre * phi
                second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
                var_post = fnp.maximum(second - mu * mu, zero)
                gain = fnp.where(sigma_pre > 1e-12, cdf, zero)
                cov_linear = fnp.multiply(fnp.outer(gain, gain), cov_pre)
                u = fnp.where(sigma_pre > 1e-12, phi / sigma_pre, zero)
                cov_quad = half * fnp.multiply(fnp.outer(u, u), cov_pre * cov_pre)
                cov_tot = cov_linear + 1.0 * cov_quad
                fnp.fill_diagonal(cov_tot, var_post)
                cov = flops.as_symmetric(cov_tot, symmetry=(0, 1))

            c_reset = np.array(mu, dtype=np.float64)

            # Compute scale for reset covariance
            sy_reset = float(np.sum(y_true * c_reset) / np.sum(c_reset * c_reset))
            c_reset_calib = sy_reset * c_reset
            mse_reset = float(np.mean((c_reset_calib - y_true) ** 2))

            # Fused with standard WMC
            m_wmc = np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64)
            fused_reset = 0.89 * c_reset_calib + 0.11 * m_wmc
            mse_fused = float(np.mean((fused_reset - y_true) ** 2))

            # Champion MSE
            c_champ = np.array(_hermite_gain_covariance(mlp)[-1], dtype=np.float64)
            c_champ_calib = float(_S_PRIOR) * c_champ
            fused_champ = 0.89 * c_champ_calib + 0.11 * m_wmc
            mse_champ = float(np.mean((fused_champ - y_true) ** 2))

            reset_mses.append(mse_reset)
            champ_mses.append(mse_champ)
            reset_fused_mses.append(mse_fused)

            print(f"MLP {i+1} ({row['mlp_name']:<20}): Reset Fused={mse_fused:.4e} | Champ Fused={mse_champ:.4e} | Diff={((mse_fused-mse_champ)/mse_champ)*100:+.2f}%")

        mean_fused_reset = np.mean(reset_fused_mses)
        mean_champ = np.mean(champ_mses)
        gain_pct = ((mean_champ - mean_fused_reset) / mean_champ) * 100.0
        print(f"--> Layer {k_reset} Reset Fused Mean MSE: {mean_fused_reset:.6e} (Adj: {mean_fused_reset*0.1:.6e}) | Gain vs Champ: {gain_pct:+.2f}%")

if __name__ == "__main__":
    run_oracle_moment_reset()
