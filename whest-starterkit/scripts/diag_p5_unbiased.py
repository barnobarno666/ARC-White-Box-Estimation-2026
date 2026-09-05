"""P5-02: Unbiased split-sample control potential + active subspace + Stein-input."""
import numpy as np
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP

def whitened_samples(mlp, n=4200, seed=None):
    d = mlp.width
    half = n//2
    rng = np.random.default_rng(seed if seed is not None else mlp.seed)
    xh = rng.standard_normal((half, d)).astype(np.float64)
    x = np.concatenate([xh, -xh], axis=0)
    gram = (x.T @ x)/float(n)
    ev, U = np.linalg.eigh(gram)
    ev = np.maximum(ev, 1e-6)
    W = (U * (ev**-0.5)) @ U.T
    w0 = np.array(mlp.weights[0], dtype=np.float64)
    act = np.maximum(x @ (W @ w0), 0.0)
    acts = [act]
    for l in range(1, mlp.depth):
        w = np.array(mlp.weights[l], dtype=np.float64)
        act = np.maximum(act @ w, 0.0)
        acts.append(act)
    # also return input x and preactivations at final layer
    return x, acts

def main():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)
    print("=== P5-02a: unbiased split-sample linear control (final preactivations) ===")
    # For each MLP: split 4200 into two halves A,B (2100 each, each antithetic internally? simplify random split)
    # Control: mA_corrected = mbarA - beta*(zbarA - zbarB), beta = Cov(y,z)/Var(z) per neuron estimated from A (cross-fit: estimate beta from B to avoid bias, then apply)
    mses_raw, mses_cv = [], []
    for i in range(8):
        row = ds[i]; mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_true = np.array(row["final_means"], dtype=np.float64)
        x, acts = whitened_samples(mlp, n=4200)
        H = acts[-2]  # penultimate post-activations (4200,1024)
        wL = np.array(mlp.weights[-1], dtype=np.float64)
        Z = H @ wL  # final preactivations
        Y = np.maximum(Z, 0.0)
        m_raw = Y.mean(axis=0)
        mse_raw = float(np.mean((m_raw - y_true)**2))
        # split
        idx = np.random.default_rng(0).permutation(4200)
        A_idx, B_idx = idx[:2100], idx[2100:]
        YA, ZA = Y[A_idx], Z[A_idx]
        YB, ZB = Y[B_idx], Z[B_idx]
        # estimate beta per neuron from B (independent of A correction)
        zbarB = ZB.mean(axis=0); ybarB = YB.mean(axis=0)
        var_zB = ((ZB - zbarB)**2).mean(axis=0) + 1e-12
        cov_yzB = ((YB - ybarB)*(ZB - zbarB)).mean(axis=0)
        beta = cov_yzB / var_zB
        # clip beta to [0,1] (since d ReLU/dz in [0,1])
        beta = np.clip(beta, 0.0, 1.0)
        mA_corr = YA.mean(axis=0) - beta*(ZA.mean(axis=0) - ZB.mean(axis=0))
        # symmetric: also correct B using beta from A, average
        zbarA = ZA.mean(axis=0); ybarA = YA.mean(axis=0)
        var_zA = ((ZA - zbarA)**2).mean(axis=0) + 1e-12
        cov_yzA = ((YA - ybarA)*(ZA - zbarA)).mean(axis=0)
        beta2 = np.clip(cov_yzA/var_zA, 0.0, 1.0)
        mB_corr = ybarB - beta2*(zbarB - zbarA)
        m_cv = 0.5*(mA_corr + mB_corr)
        mse_cv = float(np.mean((m_cv - y_true)**2))
        mses_raw.append(mse_raw); mses_cv.append(mse_cv)
        print(f" {row['mlp_name']}: raw={mse_raw:.4e} cv={mse_cv:.4e} ratio={mse_cv/mse_raw:.3f}")
    print(f"MEAN raw={np.mean(mses_raw):.4e} cv={np.mean(mses_cv):.4e} reduction={(1-np.mean(mses_cv)/np.mean(mses_raw))*100:+.2f}%")

    print("\n=== P5-02b: active subspace - variance of final explained by top input dirs ===")
    # For 2 MLPs: regress final activations (centered samples) on input X projected onto top-k adjoint dirs
    # Adjoint dirs: propagate ones vector backwards through expected gates? Simplify: top right singular vectors of W0@W1@...@W15 product (linearized map)
    for i in range(2):
        row = ds[i]; mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        Ws = [np.array(w, dtype=np.float64) for w in mlp.weights]
        # linearized map (product) - crude but white-box legal
        P = Ws[0]
        for w in Ws[1:]:
            P = P @ w
        # top-4 right singular vectors of P (input dirs)
        U, S, Vt = np.linalg.svd(P, full_matrices=False)
        V4 = Vt[:4, :].T  # (1024,4)
        x, acts = whitened_samples(mlp, n=4000)
        Y = acts[-1]  # (4000,1024) final post-activations
        Yc = Y - Y.mean(axis=0)
        tot_var = float((Yc**2).mean())
        # project inputs
        Z4 = x @ V4  # (4000,4)
        # regress each output neuron on Z4 (linear) -> R2
        # solve least squares: B = (Z4^T Z4)^{-1} Z4^T Yc
        B, *_ = np.linalg.lstsq(Z4, Yc, rcond=None)
        Yhat = Z4 @ B
        r2 = float((Yhat**2).mean() / tot_var)
        print(f" {row['mlp_name']}: total_var={tot_var:.4e} R2_top4_linear={r2:.4f} (kill if <0.15)")

    print("\n=== P5-02c: Stein input control (X has exact zero mean) ===")
    mses_raw2, mses_stein = [], []
    for i in range(8):
        row = ds[i]; mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_true = np.array(row["final_means"], dtype=np.float64)
        x, acts = whitened_samples(mlp, n=4200)
        Y = acts[-1]
        m_raw = Y.mean(axis=0)
        # regress Y on x (1024-dim -> 1024-dim) with ridge, cross-fit halves
        idx = np.random.default_rng(1).permutation(4200)
        A_idx, B_idx = idx[:2100], idx[2100:]
        XA, YA = x[A_idx], Y[A_idx]
        XB, YB = x[B_idx], Y[B_idx]
        # low-rank: use top-8 PCs of XA to avoid 1024x1024 regression noise; ridge lam
        # Simplified: per-output univariate regression on best single input? Instead full ridge rank-8 via SVD
        Uu, Ss, Vtt = np.linalg.svd(XA - XA.mean(axis=0), full_matrices=False)
        Vk = Vtt[:8, :].T
        ZAk = (XA - XA.mean(axis=0)) @ Vk
        ZBk = (XB - XB.mean(axis=0)) @ Vk
        # beta from A, apply to B and vice versa
        Bk_A, *_ = np.linalg.lstsq(ZAk, YA - YA.mean(axis=0), rcond=None)
        mB = YB.mean(axis=0) - (ZBk.mean(axis=0) - ZAk.mean(axis=0)) @ Bk_A  # E[Z]=0? No, E[ZAk-ZBk]=0
        # Actually unbiased correction: mbarB - (zbarB - zbarA)@beta_A ; E=mu
        Bk_B, *_ = np.linalg.lstsq(ZBk, YB - YB.mean(axis=0), rcond=None)
        mA = YA.mean(axis=0) - (ZAk.mean(axis=0) - ZBk.mean(axis=0)) @ Bk_B
        m_st = 0.5*(mA+mB)
        mses_raw2.append(float(np.mean((m_raw-y_true)**2)))
        mses_stein.append(float(np.mean((m_st-y_true)**2)))
        print(f" {row['mlp_name']}: raw={mses_raw2[-1]:.4e} stein8={mses_stein[-1]:.4e} ratio={mses_stein[-1]/mses_raw2[-1]:.3f}")
    print(f"MEAN raw={np.mean(mses_raw2):.4e} stein={np.mean(mses_stein):.4e} reduction={(1-np.mean(mses_stein)/np.mean(mses_raw2))*100:+.2f}%")

if __name__ == "__main__":
    main()
