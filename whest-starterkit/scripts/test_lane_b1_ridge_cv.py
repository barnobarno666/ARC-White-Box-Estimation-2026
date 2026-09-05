import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _hermite_gain_covariance, _whitened_antithetic_mc, _S_PRIOR

def evaluate_mid_cv(k_layer: int = 8, rank: int = 16, ridge_lam: float = 1e-2):
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print(f"LANE B1 DIAGNOSTIC: MID-NETWORK RIDGE CV (Layer {k_layer}, Rank {rank}, Ridge {ridge_lam})")
    print("=" * 90)

    count = 4200
    half = count // 2

    raw_mc_mses = []
    cv_oracle_mses = []
    cv_analytic_mses = []
    fused_champ_mses = []
    fused_cv_mses = []

    s0 = float(_S_PRIOR)

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_true = np.array(row["final_means"], dtype=np.float64)

        # 1. Generate whitened MC samples and store activations at layer k and layer 15
        rng = np.random.default_rng(mlp.seed)
        x_half = rng.standard_normal((half, mlp.width)).astype(np.float32)
        x = np.concatenate((x_half, -x_half), axis=0)

        # Whitening
        gram = (x.T @ x) / float(count)
        eigvals, eigvecs = np.linalg.eigh(gram)
        eigvals = np.maximum(eigvals, 1e-6)
        whitener = (eigvecs * (eigvals ** -0.5)) @ eigvecs.T

        # Forward pass tracking
        first_weight = np.array(mlp.weights[0], dtype=np.float32)
        act = np.maximum(x @ (whitener @ first_weight), 0.0)

        H_k = None
        for l in range(1, mlp.depth):
            if l == k_layer:
                H_k = act.copy()
            w = np.array(mlp.weights[l], dtype=np.float32)
            act = np.maximum(act @ w, 0.0)
        H_L = act  # layer 15

        m_raw = np.mean(H_L, axis=0)
        m_k = np.mean(H_k, axis=0)

        # 2. Get analytic covariance predictions at layer k and layer 15
        # For simplicity in diagnostic, compute layer k mean and layer 15 mean
        # Using threshold-aware or standard Hermite cov
        width = mlp.width
        mu = np.zeros(width, dtype=np.float32)
        cov = np.eye(width, dtype=np.float32)
        c_k = None
        c_L = None
        for l, weight in enumerate(mlp.weights):
            w = np.array(weight, dtype=np.float32)
            mu_pre = w.T @ mu
            cov_pre = w.T @ cov @ w
            var_pre = np.maximum(np.diag(cov_pre), 1e-12)
            sigma_pre = np.sqrt(var_pre)
            a = mu_pre / sigma_pre
            # Gaussian moments
            phi = np.array(flops.stats.norm.pdf(fnp.asarray(a, dtype=fnp.float32)), dtype=np.float32)
            cdf = np.array(flops.stats.norm.cdf(fnp.asarray(a, dtype=fnp.float32)), dtype=np.float32)
            mu = (mu_pre * cdf + sigma_pre * phi).astype(np.float32)
            second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
            var_post = np.maximum(second - mu * mu, 0.0)
            gain = np.where(sigma_pre > 1e-12, cdf, 0.0)
            cov_linear = np.outer(gain, gain) * cov_pre
            u = np.where(sigma_pre > 1e-12, phi / sigma_pre, 0.0)
            cov_quad = 0.5 * np.outer(u, u) * (cov_pre * cov_pre)
            cov = cov_linear + 1.0 * cov_quad
            np.fill_diagonal(cov, var_post)
            if l == k_layer:
                c_k = mu.copy()
            if l == mlp.depth - 1:
                c_L = mu.copy()

        # 3. Two-block cross-fitting of Ridge response B_k
        # Block 1: first half, Block 2: second half
        idx1 = np.arange(0, count, 2)
        idx2 = np.arange(1, count, 2)

        def fit_and_predict(H_train_k, H_train_L, H_test_k, H_test_L, c_target):
            # Center train
            m_train_k = np.mean(H_train_k, axis=0)
            m_train_L = np.mean(H_train_L, axis=0)
            Z_k = H_train_k - m_train_k
            Z_L = H_train_L - m_train_L

            # SVD of Z_k to get rank-r basis
            # Randomized SVD / Truncated SVD
            u, s_vals, vt = np.linalg.svd(Z_k, full_matrices=False)
            V_r = vt[:rank].T  # (1024, rank)
            proj_k = Z_k @ V_r  # (N/2, rank)

            # Ridge regression of Z_L on proj_k
            # proj_k^T proj_k is (rank, rank)
            reg = ridge_lam * np.eye(rank)
            Gamma = np.linalg.solve(proj_k.T @ proj_k + reg, proj_k.T @ Z_L)  # (rank, 1024)

            # Apply to test
            m_test_k = np.mean(H_test_k, axis=0)
            m_test_L = np.mean(H_test_L, axis=0)
            # Correction: (m_test_k - c_target) @ V_r @ Gamma
            delta = ((m_test_k - c_target) @ V_r) @ Gamma
            return m_test_L - delta

        # Cross-predict
        pred1 = fit_and_predict(H_k[idx1], H_L[idx1], H_k[idx2], H_L[idx2], c_k)
        pred2 = fit_and_predict(H_k[idx2], H_L[idx2], H_k[idx1], H_L[idx1], c_k)
        m_cv_analytic = 0.5 * (pred1 + pred2)

        # Compute MSEs
        mse_raw_mc = float(np.mean((m_raw - y_true) ** 2))
        mse_cv_analytic = float(np.mean((m_cv_analytic - y_true) ** 2))

        # Champion fused with raw MC
        c_calib = s0 * c_L
        fused_champ = 0.89 * c_calib + 0.11 * m_raw
        mse_fused_champ = float(np.mean((fused_champ - y_true) ** 2))

        # Fused with CV MC
        fused_cv = 0.89 * c_calib + 0.11 * m_cv_analytic
        mse_fused_cv = float(np.mean((fused_cv - y_true) ** 2))

        raw_mc_mses.append(mse_raw_mc)
        cv_analytic_mses.append(mse_cv_analytic)
        fused_champ_mses.append(mse_fused_champ)
        fused_cv_mses.append(mse_fused_cv)

        print(f"{row['mlp_name']:<22} | Raw MC: {mse_raw_mc:.4e} -> CV MC: {mse_cv_analytic:.4e} | Fused Champ: {mse_fused_champ:.4e} -> Fused CV: {mse_fused_cv:.4e}")

    mean_raw = np.mean(raw_mc_mses)
    mean_cv = np.mean(cv_analytic_mses)
    mean_fused_champ = np.mean(fused_champ_mses)
    mean_fused_cv = np.mean(fused_cv_mses)

    print("-" * 90)
    print(f"MEAN RAW MC MSE:      {mean_raw:.4e}")
    print(f"MEAN CV MC MSE:       {mean_cv:.4e} (Change: {((mean_cv - mean_raw)/mean_raw)*100:+.2f}%)")
    print(f"MEAN FUSED CHAMP MSE: {mean_fused_champ:.6e} (Adj: {mean_fused_champ*0.1:.6e})")
    print(f"MEAN FUSED CV MSE:    {mean_fused_cv:.6e} (Adj: {mean_fused_cv*0.1:.6e}) (Change: {((mean_fused_cv - mean_fused_champ)/mean_fused_champ)*100:+.2f}%)")

if __name__ == "__main__":
    for k in [6, 8, 10]:
        for r in [16, 32]:
            evaluate_mid_cv(k_layer=k, rank=r, ridge_lam=1.0)
