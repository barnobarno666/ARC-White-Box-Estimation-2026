import numpy as np
import scipy.linalg as la
from scipy.stats import qmc
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _S_PRIOR

def get_hadamard_matrix(n: int) -> np.ndarray:
    """Construct Sylvester Hadamard matrix of size n (n is power of 2)."""
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)

def sample_hadamard(mlp: MLP, count: int = 2048) -> fnp.ndarray:
    """Sample using Scrambled Sylvester Hadamard frames scaled to standard Gaussian radius."""
    d = mlp.width
    rng = np.random.default_rng(mlp.seed + 101)
    
    # Generate random orthogonal rotation Q in R^{d x d}
    G = rng.standard_normal((d, d)).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    
    # Scrambled Hadamard frame
    H_full = get_hadamard_matrix(count)
    cols = rng.choice(count, size=d, replace=False)
    H_sub = H_full[:, cols]  # (count, d)
    
    # Scale to Chi(d) radius: E[||x||^2] = d
    # Chi(1024) mean radius is approx sqrt(1024 - 0.5) = 31.992
    r_mean = float(np.sqrt(d - 0.5))
    x_had = H_sub @ Q * r_mean  # (count, d)
    
    # Forward pass
    act = fnp.maximum(fnp.asarray(x_had, dtype=fnp.float32) @ fnp.asarray(mlp.weights[0], dtype=fnp.float32), 0.0)
    for l in range(1, mlp.depth):
        act = fnp.maximum(act @ fnp.asarray(mlp.weights[l], dtype=fnp.float32), 0.0)
    return fnp.mean(act, axis=0)

def sample_wmc(mlp: MLP, count: int = 2048) -> fnp.ndarray:
    """Standard Whitened Antithetic MC."""
    d = mlp.width
    half = count // 2
    rng = np.random.default_rng(mlp.seed + 202)
    z = rng.standard_normal((half, d)).astype(np.float32)
    x = np.concatenate((z, -z), axis=0)
    gram = (x.T @ x) / float(count)
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 1e-6)
    whitener = (eigvecs * (eigvals ** -0.5)) @ eigvecs.T
    x_white = x @ whitener
    
    act = fnp.maximum(fnp.asarray(x_white, dtype=fnp.float32) @ fnp.asarray(mlp.weights[0], dtype=fnp.float32), 0.0)
    for l in range(1, mlp.depth):
        act = fnp.maximum(act @ fnp.asarray(mlp.weights[l], dtype=fnp.float32), 0.0)
    return fnp.mean(act, axis=0)

def sample_qmc_lattice(mlp: MLP, count: int = 2048) -> fnp.ndarray:
    """Rank-1 Shifted Lattice Rule mapped to Gaussian."""
    d = mlp.width
    rng = np.random.default_rng(mlp.seed + 303)
    
    # Generate lattice generating vector z in {1, ..., count-1} coprime to count
    # Korobov lattice or random generator
    gen = rng.integers(1, count, size=d, endpoint=False)
    shift = rng.uniform(0.0, 1.0, size=d)
    
    i_idx = np.arange(count)[:, None]
    u = np.mod((i_idx * gen) / float(count) + shift, 1.0)
    # Clip to avoid exact 0 or 1
    u = np.clip(u, 1e-7, 1.0 - 1e-7)
    from scipy.stats import norm
    x_lattice = norm.ppf(u).astype(np.float32)
    
    # Whiten
    gram = (x_lattice.T @ x_lattice) / float(count)
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 1e-6)
    whitener = (eigvecs * (eigvals ** -0.5)) @ eigvecs.T
    x_white = x_lattice @ whitener

    act = fnp.maximum(fnp.asarray(x_white, dtype=fnp.float32) @ fnp.asarray(mlp.weights[0], dtype=fnp.float32), 0.0)
    for l in range(1, mlp.depth):
        act = fnp.maximum(act @ fnp.asarray(mlp.weights[l], dtype=fnp.float32), 0.0)
    return fnp.mean(act, axis=0)

def evaluate_dual_carriers():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 95)
    print("THRUST 2: ANTAGONISTIC DUAL-CARRIER SCREEN ON 8-MLP PANEL")
    print("=" * 95)

    s0 = float(_S_PRIOR)
    c_list = []
    wmc4200_list = []
    had2048_list = []
    wmc2048_list = []
    qmc2048_list = []
    y_true_list = []

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_true = np.array(row["all_layer_means"][-1], dtype=np.float64)
        y_true_list.append(y_true)

        c = np.array(_blended_hermite_covariance(mlp)[-1], dtype=np.float64) * s0
        c_list.append(c)

        wmc4200 = np.array(sample_wmc(mlp, count=4200), dtype=np.float64)
        wmc4200_list.append(wmc4200)

        had2048 = np.array(sample_hadamard(mlp, count=2048), dtype=np.float64)
        had2048_list.append(had2048)

        wmc2048 = np.array(sample_wmc(mlp, count=2048), dtype=np.float64)
        wmc2048_list.append(wmc2048)

        qmc2048 = np.array(sample_qmc_lattice(mlp, count=2048), dtype=np.float64)
        qmc2048_list.append(qmc2048)

    # 1. Baseline Champion: 0.89 * c + 0.11 * wmc4200
    champ_mses = [float(np.mean((0.89 * c_list[i] + 0.11 * wmc4200_list[i] - y_true_list[i])**2)) for i in range(8)]
    mean_champ = np.mean(champ_mses)
    print(f"Control Champion MSE: {mean_champ:.6e} (Adj: {mean_champ * 0.1:.6e})")

    # 2. Dual Carrier: Hadamard(2048) + WMC(2048) (Total N=4096)
    dual_had_wmc = [0.5 * had2048_list[i] + 0.5 * wmc2048_list[i] for i in range(8)]
    had_wmc_raw_mses = [float(np.mean((dual_had_wmc[i] - y_true_list[i])**2)) for i in range(8)]
    print(f"\nDual Carrier Had+WMC (N=4096) Raw MSE: {np.mean(had_wmc_raw_mses):.6e}")

    # Optimize blend with c
    best_hw_score = 999.0
    best_hw_alpha = 0.0
    for alpha in np.linspace(0.05, 0.20, 31):
        mses = [float(np.mean(((1.0 - alpha) * c_list[i] + alpha * dual_had_wmc[i] - y_true_list[i])**2)) for i in range(8)]
        if np.mean(mses) < best_hw_score:
            best_hw_score = np.mean(mses)
            best_hw_alpha = alpha
    print(f"Blended Had+WMC Best MSE: {best_hw_score:.6e} (Adj: {best_hw_score * 0.1:.6e}, alpha={best_hw_alpha:.3f}, Diff: {((best_hw_score - mean_champ)/mean_champ)*100:+.2f}%)")

    # 3. Dual Carrier: Hadamard(2048) + QMC(2048) (Total N=4096)
    dual_had_qmc = [0.5 * had2048_list[i] + 0.5 * qmc2048_list[i] for i in range(8)]
    had_qmc_raw_mses = [float(np.mean((dual_had_qmc[i] - y_true_list[i])**2)) for i in range(8)]
    print(f"\nDual Carrier Had+QMC (N=4096) Raw MSE: {np.mean(had_qmc_raw_mses):.6e}")

    best_hq_score = 999.0
    best_hq_alpha = 0.0
    for alpha in np.linspace(0.05, 0.20, 31):
        mses = [float(np.mean(((1.0 - alpha) * c_list[i] + alpha * dual_had_qmc[i] - y_true_list[i])**2)) for i in range(8)]
        if np.mean(mses) < best_hq_score:
            best_hq_score = np.mean(mses)
            best_hq_alpha = alpha
    print(f"Blended Had+QMC Best MSE: {best_hq_score:.6e} (Adj: {best_hq_score * 0.1:.6e}, alpha={best_hq_alpha:.3f}, Diff: {((best_hq_score - mean_champ)/mean_champ)*100:+.2f}%)")

    # 4. Antagonistic Tri-Weighting: c, had, qmc, wmc
    # Solve unconstrained linear regression on errors to find the theoretical bound of combining all 4 branches
    opt_tri_mses = []
    for i in range(8):
        y = y_true_list[i]
        A = np.column_stack([c_list[i], had2048_list[i], qmc2048_list[i], wmc2048_list[i]])
        # Constrained weights summing to 1
        # Solve min ||A w - y||^2 s.t. sum(w) = 1
        # Quadratic programming or Lagrange multiplier
        C = np.ones((1, 4))
        d = np.array([1.0])
        AtA = A.T @ A
        Aty = A.T @ y
        # KKT: [2 AtA, C.T; C, 0] [w; lambda] = [2 Aty; 1]
        KKT = np.block([[2.0 * AtA, C.T], [C, np.zeros((1, 1))]])
        rhs = np.concatenate([2.0 * Aty, d])
        sol = np.linalg.solve(KKT, rhs)
        w_opt = sol[:4]
        pred_opt = A @ w_opt
        opt_tri_mses.append(float(np.mean((pred_opt - y)**2)))

    print(f"\nTheoretical Upper Bound (Oracle Combined c + Had + QMC + WMC): {np.mean(opt_tri_mses):.6e} (Adj: {np.mean(opt_tri_mses) * 0.1:.6e}, Diff: {((np.mean(opt_tri_mses) - mean_champ)/mean_champ)*100:+.2f}%)")

if __name__ == "__main__":
    evaluate_dual_carriers()
