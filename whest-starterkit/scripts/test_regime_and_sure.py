import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from candidates.estimator_calibrated_hermite import _hermite_gain_covariance, _whitened_antithetic_mc

def test_regimes_and_adaptive_weights():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    with open("scripts/baseline_8mlp.json") as f:
        base = json.load(f)
    base_scores = [m["adjusted_final_layer_score"] for m in base["per_mlp"]]

    print("=" * 80)
    print("REGIME-SPECIFIC SHRINKAGE & NEURON-WISE PRECISION WEIGHTING (Target < 1.24e-7)")
    print("=" * 80)

    mlps = []
    cov_rows = []
    mcs = []
    truths = []
    names = []

    for i in range(8):
        row = ds[i]
        names.append(row["mlp_name"])
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = fnp.asarray(row["final_means"], dtype=fnp.float32)
        c_full = _hermite_gain_covariance(mlp)
        m = _whitened_antithetic_mc(mlp)[15]
        cov_rows.append(c_full)
        mcs.append(m)
        truths.append(y)

    c_list = [c[15] for c in cov_rows]

    # Baseline with LOO global scale shrinkage
    # Mean s_y is ~0.998412
    print("\n--- BASELINE REFERENCE: Global Shrunk Scale s_0=0.998412, Scalar Alpha ---")
    s_0 = 0.998412
    for alpha in [0.08, 0.10, 0.11, 0.12, 0.13, 0.14]:
        scores = []
        for i in range(8):
            c = c_list[i]
            m = mcs[i]
            y = truths[i]
            pred = (1.0 - alpha) * (s_0 * c) + alpha * m
            mse = float(fnp.mean((pred - y)**2))
            scores.append(mse * 0.1000)
        print(f"Alpha={alpha:.3f} -> Avg Score: {sum(scores)/8:.6e} | Max: {max(scores):.4e}")

    print("\n--- EXPERIMENT A: 2-Regime & 4-Regime Scale Calibration ---")
    # Group neurons by predicted activation magnitude or firing probability Phi(a)
    for n_groups in [2, 4]:
        for alpha in [0.10, 0.11, 0.12, 0.13]:
            scores = []
            wins = 0
            for i in range(8):
                c = c_list[i]
                m = mcs[i]
                y = truths[i]
                
                # Quantile thresholds of c
                qs = fnp.quantile(c, fnp.linspace(0.0, 1.0, n_groups + 1))
                parts = []
                for g in range(n_groups):
                    mask = (c >= qs[g]) & (c <= qs[g+1]) if g == n_groups - 1 else (c >= qs[g]) & (c < qs[g+1])
                    c_g = fnp.where(mask, c, 0.0)
                    m_g = fnp.where(mask, m, 0.0)
                    denom = float(fnp.sum(c_g * c_g))
                    s_g = float(fnp.sum(m_g * c_g) / (denom + 1e-12))
                    s_g_shrunk = 0.5 * s_0 + 0.5 * s_g
                    part = fnp.where(mask, (1.0 - alpha) * (s_g_shrunk * c) + alpha * m, 0.0)
                    parts.append(part)
                pred = sum(parts)

                mse = float(fnp.mean((pred - y)**2))
                score = mse * 0.1000
                scores.append(score)
                if score < base_scores[i]:
                    wins += 1
            print(f"Groups={n_groups} | Alpha={alpha:.3f} -> Avg Score: {sum(scores)/8:.6e} ({wins}W-{8-wins}L) | Worst: {max(scores):.4e}")

    print("\n--- EXPERIMENT B: Neuron-Wise Variance-Adaptive Kalman Blend ---")
    # In WMC with N=4200, each neuron's sample variance is proportional to its activation variance
    # Neurons with large activation variance have higher MC noise!
    # Formula: alpha_i = tau^2 / (tau^2 + var_i / N)
    for tau_scale in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040]:
        scores = []
        wins = 0
        for i in range(8):
            c = c_list[i]
            m = mcs[i]
            y = truths[i]

            # Proxy for neuron variance: c_i^2 or c_i (since for ReLU, E[h^2] ~ mu^2 + var)
            # Actually, c_i is the mean activation; variance is approximately prop to c_i^2
            # Let's compute weights:
            # High activation neurons have high MC noise -> lower alpha
            # Low activation neurons have low MC noise -> higher alpha
            v_proxy = fnp.maximum(c * c, fnp.asarray(1e-4, dtype=fnp.float32))
            tau2 = (tau_scale)**2
            alpha_i = tau2 / (tau2 + v_proxy / 4200.0)
            alpha_i = fnp.clip(alpha_i, 0.02, 0.25)

            c_cal = s_0 * c
            pred = (1.0 - alpha_i) * c_cal + alpha_i * m
            mse = float(fnp.mean((pred - y)**2))
            score = mse * 0.1000
            scores.append(score)
            if score < base_scores[i]:
                wins += 1
        avg_score = sum(scores) / 8.0
        print(f"Tau={tau_scale:.3f} -> Avg Score: {avg_score:.6e} ({wins}W-{8-wins}L) | Worst: {max(scores):.4e}")

    print("\n--- EXPERIMENT C: Polynomial Scale Correction (Affine Calibration: s * c + b) ---")
    # What if the scale is an affine transform s * c + b?
    for b_reg in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        scores = []
        wins = 0
        for i in range(8):
            c = c_list[i]
            m = mcs[i]
            y = truths[i]
            
            # Fit m ~ s * c + b with L2 regularization on b towards 0
            # U = [c, 1]
            U = fnp.stack([c, fnp.ones_like(c)], axis=1)
            reg = fnp.diag(fnp.asarray([0.0, b_reg * 1024.0], dtype=fnp.float32))
            w = fnp.linalg.solve(U.T @ U + reg, U.T @ m)
            # w[0] is slope, w[1] is intercept
            s_fit = 0.5 * s_0 + 0.5 * float(w[0])
            b_fit = float(w[1])
            c_cal = s_fit * c + b_fit
            
            pred = 0.88 * c_cal + 0.12 * m
            mse = float(fnp.mean((pred - y)**2))
            score = mse * 0.1000
            scores.append(score)
            if score < base_scores[i]:
                wins += 1
        print(f"b_reg={b_reg:<6} -> Avg Score: {sum(scores)/8:.6e} ({wins}W-{8-wins}L) | Worst: {max(scores):.4e}")

if __name__ == "__main__":
    test_regimes_and_adaptive_weights()
