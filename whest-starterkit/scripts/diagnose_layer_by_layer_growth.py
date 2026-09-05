import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance

def inspect_layer_by_layer_error():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 80)
    print("LAYER-BY-LAYER ANALYTICAL COVARIANCE ERROR")
    print("=" * 80)

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_all = np.array(row["all_layer_means"], dtype=np.float64)  # (16, 1024)
        c_all = np.array(_blended_hermite_covariance(mlp), dtype=np.float64)  # (16, 1024)

        print(f"\nMLP {i} ({row['mlp_name']}):")
        for l in range(16):
            err = c_all[l] - y_all[l]
            mse = float(np.mean(err**2))
            mean_y = float(np.mean(y_all[l]))
            mean_c = float(np.mean(c_all[l]))
            ratio = mean_c / mean_y if mean_y != 0 else 1.0
            print(f"  Layer {l:2d}: MSE = {mse:.6e} | Mean True = {mean_y:.6f} | Mean Cov = {mean_c:.6f} | Ratio = {ratio:.6f}")

if __name__ == "__main__":
    inspect_layer_by_layer_error()
