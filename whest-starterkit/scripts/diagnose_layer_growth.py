"""diagnose_layer_growth.py

Investigates the layer-by-layer error growth and closure accumulation:
1. Measures the empirical scale ratio s_l = <y_l, c_l> / ||c_l||^2 at every layer l in 0..15
2. Tests Layer-by-Layer Scale Calibration (recursive prior vs single terminal scale)
3. Tests Higher-Order Hermite Kernels (Cubic & Quartic terms) across depth
4. Evaluates if multi-layer scale calibration breaks the 1.22e-06 floor.
"""

import numpy as np
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance

def diagnose_layer_growth():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("LAYER-BY-LAYER ERROR GROWTH & SCALE ACCUMULATION")
    print("=" * 90)

    # Collect layer means for all 8 MLPs
    all_trues = []
    all_covs = []

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        all_trues.append(np.array(row["all_layer_means"], dtype=np.float64))
        cov_all = np.array(_blended_hermite_covariance(mlp), dtype=np.float64)
        all_covs.append(cov_all)

    # For each layer l in 0..15:
    print(f"{'Layer':<6} | {'Mean Raw MSE':<14} | {'Mean Scale s_l':<16} | {'Scaled MSE':<14} | {'Var Ratio'}")
    print("-" * 75)
    
    mean_scales = []
    for l in range(16):
        raw_mses = []
        scaled_mses = []
        scales = []
        var_ratios = []

        for i in range(8):
            y_l = all_trues[i][l]
            c_l = all_covs[i][l]

            raw_mse = np.mean((c_l - y_l)**2)
            raw_mses.append(raw_mse)

            s_l = float(np.sum(y_l * c_l) / np.sum(c_l * c_l))
            scales.append(s_l)

            scaled_mse = np.mean((s_l * c_l - y_l)**2)
            scaled_mses.append(scaled_mse)

            var_ratio = np.mean(c_l**2) / np.mean(y_l**2)
            var_ratios.append(var_ratio)

        s_mean = np.mean(scales)
        mean_scales.append(s_mean)
        print(f"L{l:<5} | {np.mean(raw_mses):.6e} | {s_mean:.6f} +/- {np.std(scales):.4f} | {np.mean(scaled_mses):.6e} | {np.mean(var_ratios):.6f}")

    print("-" * 75)
    print("Layer scales sequence s_0..s_15:")
    print(", ".join([f"{s:.6f}" for s in mean_scales]))

if __name__ == "__main__":
    diagnose_layer_growth()
