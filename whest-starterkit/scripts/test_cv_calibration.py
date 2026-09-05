import sys
from pathlib import Path
sys.path.insert(0, ".")
import json
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from scripts.experiment_phase2 import get_final_moments_and_mc

def test_calibration():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    with open("scripts/baseline_8mlp.json") as f:
        base = json.load(f)
    base_scores = [m["adjusted_final_layer_score"] for m in base["per_mlp"]]

    print("=" * 80)
    print("CONTROL VARIATE CALIBRATION & ORTHOGONAL RESIDUAL PROJECTION")
    print("=" * 80)

    data = []
    truths = []
    names = []
    for i in range(8):
        row = ds[i]
        names.append(row["mlp_name"])
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = fnp.asarray(row["final_means"], dtype=fnp.float32)
        truths.append(y)
        d = get_final_moments_and_mc(mlp)
        data.append(d)

    print("\n1. BIAS AUDIT: Comparing Mean Residuals of m_raw vs m_cv1 vs m_cv2")
    for i in range(8):
        d = data[i]
        y = truths[i]
        c = d["c"]
        m_raw = d["m_raw"]
        
        # True scale and MC scale
        denom = float(fnp.sum(c * c))
        s_y = float(fnp.sum(y * c) / denom)
        s_m = float(fnp.sum(m_raw * c) / denom)

        # Uncalibrated Hermite CV1
        m_cv1_uncal = m_raw - d["cdf"] * (d["z_bar"] - d["mu_pre"])
        
        # Calibrated Hermite CV1 (using s_m to scale mu_pre!)
        m_cv1_cal = m_raw - d["cdf"] * (d["z_bar"] - s_m * d["mu_pre"])

        # What if we center at the true E[z] = y_pre?
        # Let's check bias:
        bias_raw = float(fnp.mean(m_raw - y))
        bias_cv1_uncal = float(fnp.mean(m_cv1_uncal - y))
        bias_cv1_cal = float(fnp.mean(m_cv1_cal - y))

        mse_raw = float(fnp.mean((m_raw - y)**2))
        mse_cv1_uncal = float(fnp.mean((m_cv1_uncal - y)**2))
        mse_cv1_cal = float(fnp.mean((m_cv1_cal - y)**2))

        print(f"[{i+1}/8] {names[i]:<20}: sy={s_y:.6f}, sm={s_m:.6f}")
        print(f"      Bias: Raw={bias_raw:+.2e} | CV1_uncal={bias_cv1_uncal:+.2e} | CV1_cal={bias_cv1_cal:+.2e}")
        print(f"      MSE : Raw={mse_raw:.4e} | CV1_uncal={mse_cv1_uncal:.4e} | CV1_cal={mse_cv1_cal:.4e}")

    print("\n" + "=" * 80)
    print("2. HYBRID BLEND: s_m * c + alpha * (m_cv - E[m_cv])")
    print("=" * 80)

    # Let's test blending calibrated covariance s_m * c with different forms of MC:
    # Form A: Current Champion: (1 - alpha) * (s_m * c) + alpha * m_raw
    # Form B: (s_m * c) + alpha * (m_raw - s_m * c)  (algebraically identical to Form A!)
    # Form C: (s_m * c) + alpha * (m_cv1 - s_m * c)  
    # Form D: What is the optimal regression coefficient beta on (z_bar - mu_pre)?
    # Notice: m_cv = m_raw - beta * (z_bar - s_m * mu_pre)
    # The discrepancy between sample preactivations and predicted preactivations!

    for beta_scale in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        print(f"\n--- Testing CV coefficient beta = {beta_scale:.1f} * Phi(a) ---")
        for alpha in [0.08, 0.10, 0.12, 0.13, 0.14, 0.16]:
            scores = []
            wins = 0
            for i in range(8):
                d = data[i]
                y = truths[i]
                c = d["c"]
                m_raw = d["m_raw"]
                denom = float(fnp.sum(c * c))
                s_m = float(fnp.sum(m_raw * c) / denom)
                c_cal = s_m * c

                # Control variate correction on the preactivations:
                cv_term = (beta_scale * d["cdf"]) * (d["z_bar"] - s_m * d["mu_pre"])
                m_corr = m_raw - cv_term

                pred = (1.0 - alpha) * c_cal + alpha * m_corr
                mse = float(fnp.mean((pred - y)**2))
                score = mse * 0.1000
                scores.append(score)
                if score < base_scores[i]:
                    wins += 1
            avg_score = sum(scores) / 8.0
            print(f"Beta={beta_scale:.1f} | Alpha={alpha:.3f} | Avg Score: {avg_score:.6e} | {wins}W-{8-wins}L | Worst: {max(scores):.4e}")

if __name__ == "__main__":
    test_calibration()
