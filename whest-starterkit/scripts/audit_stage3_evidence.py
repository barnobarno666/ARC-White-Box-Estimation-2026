import json
import numpy as np
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _hermite_gain_covariance, _whitened_antithetic_mc, _S_PRIOR

def run_stage3_audit():
    print("=" * 80)
    print("STAGE 3 EVIDENCE AUDIT: FIXED-SCALE COVARIANCE ERRORS & CROSS-ERROR COVARIANCE")
    print("=" * 80)

    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    A_list = []
    B_list = []
    C_list = []
    cos_list = []
    alpha_opt_list = []
    mse_current_list = []
    mse_opt_list = []

    s0 = float(_S_PRIOR)

    print(f"{'MLP':<22} | {'A (Cov MSE)':<11} | {'B (MC MSE)':<11} | {'C (Cross)':<11} | {'Cos(ec,em)':<10} | {'alpha*':<8} | {'Cur MSE (0.11)':<14} | {'Opt MSE':<11} | {'Gain %':<7}")
    print("-" * 118)

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_true = np.array(row["final_means"], dtype=np.float64)  # final layer (layer 15)

        # Compute predictions
        c = np.array(_hermite_gain_covariance(mlp)[-1], dtype=np.float64)
        m = np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64)

        c0 = s0 * c

        ec = c0 - y_true
        em = m - y_true

        A = float(np.mean(ec ** 2))
        B = float(np.mean(em ** 2))
        C = float(np.mean(ec * em))
        cos_sim = C / np.sqrt(A * B) if (A * B) > 0 else 0.0

        alpha_opt = (A - C) / (A + B - 2 * C)

        pred_curr = (1.0 - 0.110) * c0 + 0.110 * m
        mse_curr = float(np.mean((pred_curr - y_true) ** 2))

        pred_opt = (1.0 - alpha_opt) * c0 + alpha_opt * m
        mse_opt = float(np.mean((pred_opt - y_true) ** 2))

        gain_pct = ((mse_curr - mse_opt) / mse_curr) * 100.0

        A_list.append(A)
        B_list.append(B)
        C_list.append(C)
        cos_list.append(cos_sim)
        alpha_opt_list.append(alpha_opt)
        mse_current_list.append(mse_curr)
        mse_opt_list.append(mse_opt)

        print(f"{row['mlp_name']:<22} | {A:.4e} | {B:.4e} | {C:+.4e} | {cos_sim:+.4f}    | {alpha_opt:.4f}   | {mse_curr:.4e}     | {mse_opt:.4e} | {gain_pct:+.2f}%")

    mean_A = np.mean(A_list)
    mean_B = np.mean(B_list)
    mean_C = np.mean(C_list)
    mean_cos = np.mean(cos_list)
    global_alpha_opt = (mean_A - mean_C) / (mean_A + mean_B - 2 * mean_C)
    mean_curr = np.mean(mse_current_list)
    mean_opt = np.mean(mse_opt_list)
    global_gain = ((mean_curr - mean_opt) / mean_curr) * 100.0

    print("-" * 118)
    print(f"{'PANEL MEAN':<22} | {mean_A:.4e} | {mean_B:.4e} | {mean_C:+.4e} | {mean_cos:+.4f}    | {global_alpha_opt:.4f}   | {mean_curr:.4e}     | {mean_opt:.4e} | {global_gain:+.2f}%")
    print("=" * 80)
    print(f"[measured] Global Risk-Optimal alpha*: {global_alpha_opt:.4f}")
    print(f"[measured] Panel Mean Error Cosine: {mean_cos:+.4f}")
    print(f"[measured] Oracle Headroom of Per-MLP alpha*: {global_gain:.2f}%")

if __name__ == "__main__":
    run_stage3_audit()
