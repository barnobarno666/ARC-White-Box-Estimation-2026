import numpy as np
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _whitened_antithetic_mc
from candidates.estimator_lane_a1_thresh_quad_eta1 import _threshold_aware_hermite_covariance
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from diagnose_lane_a1 import compute_thresh_cov

def find_optimal_thresh_params():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("SWEEP & OPTIMIZATION FOR THRESHOLD-AWARE HERMITE (eta_2, s_0, alpha)")
    print("=" * 90)

    mlps = []
    y_trues = []
    m_list = []

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        mlps.append(mlp)
        y_trues.append(np.array(row["final_means"], dtype=np.float64))
        m_list.append(np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64))

    # Test eta_2 sweep: 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0
    for eta2 in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
        covs = []
        sy_list = []
        for i in range(8):
            c = compute_thresh_cov(mlps[i], eta2)
            covs.append(c)
            sy = float(np.sum(y_trues[i] * c) / np.sum(c * c))
            sy_list.append(sy)

        mean_sy = np.mean(sy_list)

        # LOO evaluation of (s_loo * c) blended with m
        loo_fused_list = []
        # Find best global alpha on LOO
        for alpha in [0.08, 0.10, 0.11, 0.12, 0.14]:
            scores = []
            for i in range(8):
                loo_s = np.mean([sy_list[j] for j in range(8) if j != i])
                pred = (1.0 - alpha) * (loo_s * covs[i]) + alpha * m_list[i]
                scores.append(float(np.mean((pred - y_trues[i]) ** 2)))
            print(f"eta_2={eta2:<4} | s_loo={mean_sy:.6f} | alpha={alpha:.2f} | Mean Fused MSE: {np.mean(scores):.6e} (Adj: {np.mean(scores)*0.1:.6e})")

if __name__ == "__main__":
    find_optimal_thresh_params()
