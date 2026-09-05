import sys
sys.path.insert(0, ".")
import flopscope.numpy as fnp
from whestbench.domain import MLP
from whestbench.dataset import load_dataset
from candidates.estimator_calibrated_hermite import _hermite_gain_covariance, _whitened_antithetic_mc
import json

ds = load_dataset(r'D:\ALL CODES\AICROWD COMPETITION\datasets\mini', split='mini')

mlps = []
truths = []
covs = []
mcs = []

print("Precomputing cov and mc for all 8 MLPs...")
for i in range(8):
    item = ds[i]
    weights = [fnp.asarray(w) for w in item['weights']]
    mlp = MLP(width=1024, depth=16, weights=weights, seed=item['mlp_seed'])
    truth = fnp.asarray(item['all_layer_means'])
    c = _hermite_gain_covariance(mlp)[15]
    m = _whitened_antithetic_mc(mlp)[15]
    covs.append(c)
    mcs.append(m)
    truths.append(truth[15])

with open("scripts/baseline_8mlp.json") as f:
    base = json.load(f)
base_scores = [m["adjusted_final_layer_score"] for m in base["per_mlp"]]

print(f"\n{'Alpha':<7} | {'Avg Score':<12} | {'Avg MSE':<12} | {'Wins':<6} | {'Steven-Rice Score':<18}")
print("-" * 65)

for alpha in [0.06, 0.08, 0.10, 0.11, 0.12, 0.13, 0.14, 0.145, 0.15, 0.16, 0.18, 0.20]:
    scores = []
    wins = 0
    steven_score = None
    for i in range(8):
        c = covs[i]
        m = mcs[i]
        y = truths[i]
        denom = fnp.sum(c * c)
        scale = fnp.sum(m * c) / denom
        c_cal = scale * c
        pred = (1.0 - alpha) * c_cal + alpha * m
        mse = float(fnp.mean((pred - y)**2))
        score = mse * 0.1000
        scores.append(score)
        if score < base_scores[i]:
            wins += 1
        if i == 3:
            steven_score = score
    avg_s = sum(scores) / 8.0
    avg_mse = avg_s / 0.1000
    st_str = f"{steven_score:.4e} ({'WIN' if steven_score < base_scores[3] else 'LOSS'})"
    print(f"{alpha:<7.3f} | {avg_s:.6e} | {avg_mse:.6e} | {wins}W-{8-wins}L | {st_str}")
