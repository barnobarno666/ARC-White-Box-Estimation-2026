import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _whitened_antithetic_mc

def test_regime_ratios():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("EMPIRICAL RATIOS y* / c ACROSS DEAD, KINK, AND ON NEURONS")
    print("=" * 90)

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = np.array(row["all_layer_means"][-1], dtype=np.float64)

        c = np.array(_blended_hermite_covariance(mlp)[-1], dtype=np.float64)

        # Get a_15
        # Layer 15 preactivations
        w = mlp.weights[-1]
        c14 = np.array(_blended_hermite_covariance(mlp)[-2], dtype=np.float64)
        
        dead_mask = y < 0.005  # dead neurons
        on_mask = y > 1.2      # highly active neurons
        kink_mask = ~dead_mask & ~on_mask

        ratio_on = y[on_mask] / c[on_mask]
        ratio_kink = y[kink_mask] / c[kink_mask]

        print(f"MLP {i+1} ({row['mlp_name']}):")
        print(f"  On Neurons   (count={np.sum(on_mask):3d}): mean(y*/c) = {np.mean(ratio_on):.6f}, std = {np.std(ratio_on):.6f}")
        print(f"  Kink Neurons (count={np.sum(kink_mask):3d}): mean(y*/c) = {np.mean(ratio_kink):.6f}, std = {np.std(ratio_kink):.6f}")
        print(f"  Total Neurons: mean(y*/c) = {np.mean(y/c):.6f}")

if __name__ == "__main__":
    test_regime_ratios()
