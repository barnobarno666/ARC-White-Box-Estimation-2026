import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from candidates.estimator_calibrated_hermite import _hermite_gain_covariance, _whitened_antithetic_mc

def test_scale_shrinkage():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    with open("scripts/baseline_8mlp.json") as f:
        base = json.load(f)
    base_scores = [m["adjusted_final_layer_score"] for m in base["per_mlp"]]

    print("=" * 80)
    print("EMPIRICAL BAYES & LEAVE-ONE-OUT SCALE SHRINKAGE (Phase 2A)")
    print("=" * 80)

    mlps = []
    covs = []
    mcs = []
    truths = []
    names = []

    for i in range(8):
        row = ds[i]
        names.append(row["mlp_name"])
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = fnp.asarray(row["final_means"], dtype=fnp.float32)
        c = _hermite_gain_covariance(mlp)[15]
        m = _whitened_antithetic_mc(mlp)[15]
        covs.append(c)
        mcs.append(m)
        truths.append(y)

    # Compute raw s_m and true s_y
    sm_list = []
    sy_list = []
    for i in range(8):
        c = covs[i]
        m = mcs[i]
        y = truths[i]
        denom = float(fnp.sum(c * c))
        sm = float(fnp.sum(m * c) / denom)
        sy = float(fnp.sum(y * c) / denom)
        sm_list.append(sm)
        sy_list.append(sy)

    print(f"Empirical s_m values: {[round(x, 6) for x in sm_list]}")
    print(f"True s_y values:      {[round(x, 6) for x in sy_list]}")

    print("\n--- ABLATION: Scale Shrinkage weight lambda in s_hat = (1 - lam) * s_0 + lam * s_m ---")
    print("Testing with Leave-One-MLP-Out (s_0 computed strictly on the other 7 MLPs):")
    print(f"{'Lambda':<8} | {'Avg Score':<12} | {'Avg MSE':<12} | {'Wins vs Base':<14} | {'Worst Score':<14}")
    print("-" * 65)

    for lam in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00]:
        scores = []
        wins = 0
        for i in range(8):
            # Leave-one-out: s_0 is the mean of the other 7 s_m values!
            # Note: s_0 is estimated from MC samples of the other 7, NEVER ground truth!
            other_sm = [sm_list[j] for j in range(8) if j != i]
            s_0_loo = sum(other_sm) / 7.0

            s_shrunk = (1.0 - lam) * s_0_loo + lam * sm_list[i]
            c = covs[i]
            m = mcs[i]
            y = truths[i]

            pred = 0.87 * (s_shrunk * c) + 0.13 * m
            mse = float(fnp.mean((pred - y)**2))
            score = mse * 0.1000
            scores.append(score)
            if score < base_scores[i]:
                wins += 1

        avg_score = sum(scores) / 8.0
        worst_score = max(scores)
        print(f"{lam:<8.2f} | {avg_score:.6e} | {avg_score/0.1:.6e} | {wins}W-{8-wins}L        | {worst_score:.4e}")

    # Now let's test joint grid of lambda and alpha under strict Leave-One-MLP-Out
    print("\n--- JOINT GRID: (Lambda, Alpha) under Strict Leave-One-Out CV ---")
    best_loo_score = 999.0
    best_config = None

    for lam in [0.0, 0.10, 0.20, 0.30, 0.50, 1.0]:
        for alpha in [0.06, 0.08, 0.10, 0.12, 0.13, 0.14, 0.16]:
            scores = []
            wins = 0
            for i in range(8):
                other_sm = [sm_list[j] for j in range(8) if j != i]
                s_0_loo = sum(other_sm) / 7.0
                s_shrunk = (1.0 - lam) * s_0_loo + lam * sm_list[i]
                c = covs[i]
                m = mcs[i]
                y = truths[i]

                pred = (1.0 - alpha) * (s_shrunk * c) + alpha * m
                mse = float(fnp.mean((pred - y)**2))
                score = mse * 0.1000
                scores.append(score)
                if score < base_scores[i]:
                    wins += 1
            avg_score = sum(scores) / 8.0
            if avg_score < best_loo_score:
                best_loo_score = avg_score
                best_config = (lam, alpha, wins, max(scores))

    print(f"Optimal LOO Configuration: Lambda = {best_config[0]:.2f}, Alpha = {best_config[1]:.2f}")
    print(f"  LOO Score: {best_loo_score:.6e} ({best_config[2]}W-{8-best_config[2]}L, Worst: {best_config[3]:.4e})")

if __name__ == "__main__":
    test_scale_shrinkage()
