import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _S_PRIOR
from candidates.estimator_lane_a1_thresh_quad_eta1 import _threshold_aware_hermite_covariance

def run_lane_a4_test():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("LANE A4: JOINT FINAL-LAYER HERMITE CONTROLS (c_0, m_raw, delta_1, delta_2)")
    print("=" * 90)

    count = 4200
    half = count // 2
    d = 1024
    s0 = float(_S_PRIOR)

    c0_list = []
    m_raw_list = []
    delta1_list = []
    delta2_list = []
    y_true_list = []
    names = []

    for i in range(8):
        row = ds[i]
        names.append(row["mlp_name"])
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_true = np.array(row["final_means"], dtype=np.float64)
        y_true_list.append(y_true)

        # 1. Analytic covariance branch to get mu_pre, sigma_pre, cdf, phi
        # Run up to layer 15
        width = mlp.width
        mu = fnp.zeros(width, dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
        zero = fnp.asarray(0.0, dtype=fnp.float32)
        half_fnp = fnp.asarray(0.5, dtype=fnp.float32)

        mu_pre_last = None
        sigma_pre_last = None
        cdf_last = None
        phi_last = None

        for l, weight in enumerate(mlp.weights):
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
            cov_linear = fnp.multiply(fnp.outer(gain, gain), cov_pre)
            u = fnp.where(sigma_pre > 1e-12, phi / sigma_pre, zero)
            cov_quad = half_fnp * fnp.multiply(fnp.outer(u, u), cov_pre * cov_pre)
            cov = cov_linear + 1.0 * cov_quad
            fnp.fill_diagonal(cov, var_post)
            cov = flops.as_symmetric(cov, symmetry=(0, 1))

            if l == mlp.depth - 1:
                mu_pre_last = np.array(mu_pre, dtype=np.float64)
                sigma_pre_last = np.array(sigma_pre, dtype=np.float64)
                cdf_last = np.array(cdf, dtype=np.float64)
                phi_last = np.array(phi, dtype=np.float64)

        c_last = np.array(mu, dtype=np.float64)
        c0 = s0 * c_last
        c0_list.append(c0)

        # 2. Whitened MC with preactivation tracking at layer 15
        rng = np.random.default_rng(mlp.seed)
        x_half = rng.standard_normal((half, d)).astype(np.float32)
        x = np.concatenate((x_half, -x_half), axis=0)

        gram = (x.T @ x) / float(count)
        eigvals, eigvecs = np.linalg.eigh(gram)
        eigvals = np.maximum(eigvals, 1e-6)
        whitener = (eigvecs * (eigvals ** -0.5)) @ eigvecs.T

        w0 = np.array(mlp.weights[0], dtype=np.float32)
        act = np.maximum(x @ (whitener @ w0), 0.0)

        for l in range(1, mlp.depth - 1):
            w = np.array(mlp.weights[l], dtype=np.float32)
            act = np.maximum(act @ w, 0.0)

        # Final layer preactivations
        w_last = np.array(mlp.weights[-1], dtype=np.float32)
        Z_last = act @ w_last  # (4200, 1024)
        act_last = np.maximum(Z_last, 0.0)

        m_raw = np.mean(act_last, axis=0).astype(np.float64)
        z_bar = np.mean(Z_last, axis=0).astype(np.float64)
        z_sq_bar = np.mean(Z_last * Z_last, axis=0).astype(np.float64)

        # Control variate terms:
        # delta1 = cdf * (z_bar - mu_pre)
        delta1 = cdf_last * (z_bar - mu_pre_last)
        # delta2 = (phi / (2 * sigma_pre)) * (z_sq_bar - (mu_pre^2 + var_pre))
        var_pre_last = sigma_pre_last * sigma_pre_last
        delta2 = (phi_last / (2.0 * sigma_pre_last)) * (z_sq_bar - (mu_pre_last ** 2 + var_pre_last))

        m_raw_list.append(m_raw)
        delta1_list.append(delta1)
        delta2_list.append(delta2)

    # Now evaluate combinations across the 8 MLPs:
    # Baseline champion: 0.89 * c0 + 0.11 * m_raw
    champ_mses = []
    for i in range(8):
        pred_champ = 0.89 * c0_list[i] + 0.11 * m_raw_list[i]
        champ_mses.append(float(np.mean((pred_champ - y_true_list[i]) ** 2)))
    print(f"Current Champ Mean MSE: {np.mean(champ_mses):.6e} (Adj: {np.mean(champ_mses)*0.1:.6e})")

    # Joint model: y_hat = (1 - alpha) * c0 + alpha * m_raw - w1 * delta1 - w2 * delta2
    # Grid search for (alpha, w1, w2) using LOO
    best_loo_score = 999.0
    best_params = None

    for alpha in [0.08, 0.10, 0.11, 0.12, 0.14]:
        for beta1 in [0.0, 0.2, 0.5, 0.8, 1.0]:  # w1 = alpha * beta1
            for beta2 in [0.0, 0.2, 0.5, 0.8, 1.0]:  # w2 = alpha * beta2
                w1 = alpha * beta1
                w2 = alpha * beta2
                scores = []
                for i in range(8):
                    pred = (1.0 - alpha) * c0_list[i] + alpha * m_raw_list[i] - w1 * delta1_list[i] - w2 * delta2_list[i]
                    scores.append(float(np.mean((pred - y_true_list[i]) ** 2)))
                mean_score = np.mean(scores)
                if mean_score < best_loo_score:
                    best_loo_score = mean_score
                    best_params = (alpha, beta1, beta2)

    alpha_opt, beta1_opt, beta2_opt = best_params
    print(f"\nBest Grid Result: alpha={alpha_opt}, beta1={beta1_opt}, beta2={beta2_opt}")
    print(f"Best Mean MSE:    {best_loo_score:.6e} (Adj: {best_loo_score*0.1:.6e})")
    gain = ((np.mean(champ_mses) - best_loo_score) / np.mean(champ_mses)) * 100.0
    print(f"Gain vs Champ:    {gain:+.2f}%")

    # Also check per-MLP details for best params:
    print("\nPer-MLP Details for Best Configuration:")
    wins = 0
    for i in range(8):
        w1 = alpha_opt * beta1_opt
        w2 = alpha_opt * beta2_opt
        pred = (1.0 - alpha_opt) * c0_list[i] + alpha_opt * m_raw_list[i] - w1 * delta1_list[i] - w2 * delta2_list[i]
        s_cand = float(np.mean((pred - y_true_list[i]) ** 2)) * 0.1
        s_champ = champ_mses[i] * 0.1
        res = "WIN" if s_cand < s_champ else "LOSS"
        if s_cand < s_champ: wins += 1
        print(f"[{i+1}/8] {names[i]:<22}: Cand={s_cand:.4e} | Champ={s_champ:.4e} -> {res}")
    print(f"Head-to-head vs Champ: {wins}W-{8-wins}L")

if __name__ == "__main__":
    run_lane_a4_test()
