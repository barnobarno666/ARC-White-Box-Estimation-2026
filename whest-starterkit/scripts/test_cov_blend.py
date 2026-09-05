import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _hermite_gain_covariance, _whitened_antithetic_mc, _S_PRIOR
from candidates.estimator_lane_a1_thresh_quad_eta1 import _threshold_aware_hermite_covariance

def test_cov_blend():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("TESTING CONVEX BLEND OF CENTERED & THRESHOLD-AWARE HERMITE COVARIANCES")
    print("=" * 90)

    mlps = []
    y_trues = []
    c_centered_list = []
    c_thresh_list = []
    m_list = []
    s0 = float(_S_PRIOR)

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        mlps.append(mlp)
        y_trues.append(np.array(row["final_means"], dtype=np.float64))
        c_centered_list.append(np.array(_hermite_gain_covariance(mlp)[-1], dtype=np.float64))
        c_thresh_list.append(np.array(_threshold_aware_hermite_covariance(mlp)[-1], dtype=np.float64))
        m_list.append(np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64))

    champ_scores = []
    for i in range(8):
        pred_champ = 0.89 * (s0 * c_centered_list[i]) + 0.11 * m_list[i]
        champ_scores.append(float(np.mean((pred_champ - y_trues[i]) ** 2)) * 0.1)

    print(f"Current Champion Mean Score: {np.mean(champ_scores):.6e}")

    for lam in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        scores = []
        wins = 0
        for i in range(8):
            c_mix = (1.0 - lam) * c_centered_list[i] + lam * c_thresh_list[i]
            pred = 0.89 * (s0 * c_mix) + 0.11 * m_list[i]
            s_cand = float(np.mean((pred - y_trues[i]) ** 2)) * 0.1
            scores.append(s_cand)
            if s_cand < champ_scores[i]:
                wins += 1
        mean_score = np.mean(scores)
        gain = ((np.mean(champ_scores) - mean_score) / np.mean(champ_scores)) * 100.0
        print(f"lambda={lam:<4.2f} | Mean Score: {mean_score:.6e} | Gain: {gain:+.3f}% | Wins vs Champ: {wins}W-{8-wins}L")

if __name__ == "__main__":
    test_cov_blend()
