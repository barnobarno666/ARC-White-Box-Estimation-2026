import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from scripts.optimize_target import hermite_cov_advanced
from candidates.estimator_calibrated_hermite import _whitened_antithetic_mc

def test_lower_gamma():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    with open("scripts/baseline_8mlp.json") as f:
        base = json.load(f)
    base_scores = [m["adjusted_final_layer_score"] for m in base["per_mlp"]]

    print("=" * 80)
    print("EXPLORING LOWER GAMMA RANGE (0.00 to 0.40) UNDER CALIBRATION")
    print("=" * 80)

    mlps = []
    truths = []
    mcs = []
    names = []
    for i in range(8):
        row = ds[i]
        names.append(row["mlp_name"])
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        mlps.append(mlp)
        truths.append(fnp.asarray(row["final_means"], dtype=fnp.float32))
        mcs.append(_whitened_antithetic_mc(mlp)[15])

    best_score = 999.0
    best_config = None

    for gamma in [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.38, 0.40]:
        covs = [hermite_cov_advanced(mlp, gamma=gamma, delta=0.0) for mlp in mlps]
        sy_list = [float(fnp.sum(truths[i] * covs[i]) / fnp.sum(covs[i] * covs[i])) for i in range(8)]
        
        for alpha in [0.09, 0.10, 0.11, 0.12]:
            scores = []
            for i in range(8):
                other_sy = [sy_list[j] for j in range(8) if j != i]
                s_0_loo = sum(other_sy) / 7.0
                pred = (1.0 - alpha) * (s_0_loo * covs[i]) + alpha * mcs[i]
                mse = float(fnp.mean((pred - truths[i])**2))
                scores.append(mse * 0.1000)
            avg_s = sum(scores) / 8.0
            if avg_s < best_score:
                best_score = avg_s
                best_config = (gamma, alpha, sum(sy_list)/8.0, max(scores))
            if alpha == 0.11:
                print(f"Gamma={gamma:<4.2f} | Alpha={alpha:.2f} | Mean sy={sum(sy_list)/8:.6f} | LOO Score: {avg_s:.6e} | Max: {max(scores):.4e}")

    print("\n" + "=" * 80)
    print(f"ABSOLUTE BEST CONFIGURATION:")
    print(f"Gamma:     {best_config[0]:.2f}")
    print(f"Alpha:     {best_config[1]:.2f}")
    print(f"Mean sy:   {best_config[2]:.6f}")
    print(f"LOO Score: {best_score:.6e}")
    print(f"Worst MLP: {best_config[3]:.4e}")
    print("=" * 80)

if __name__ == "__main__":
    test_lower_gamma()
