import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from estimator import _whitened_antithetic_mc, _blended_hermite_covariance, _S_PRIOR
from diagnose_cho_saul_layer0 import exact_cho_saul_cov_0

def test_unified_blend():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("UNIFIED OPTIMAL COMBINATION EVALUATION ON 8-MLP PANEL")
    print("=" * 90)

    s0 = float(_S_PRIOR)
    y_trues = []
    c_base_list = []
    c_chosaul_list = []
    wmc_list = []

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = np.array(row["all_layer_means"][-1], dtype=np.float64)
        y_trues.append(y)

        c_base = np.array(_blended_hermite_covariance(mlp)[-1], dtype=np.float64) * s0
        c_base_list.append(c_base)

        c_cs = np.array(exact_cho_saul_cov_0(mlp)[-1], dtype=np.float64) * s0
        c_chosaul_list.append(c_cs)

        wmc = np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64)
        wmc_list.append(wmc)

    # Control Champion
    champ_mses = [float(np.mean((0.89 * c_base_list[i] + 0.11 * wmc_list[i] - y_trues[i])**2)) for i in range(8)]
    mean_champ = np.mean(champ_mses)
    print(f"Control Champion MSE: {mean_champ:.6e} (Adj: {mean_champ * 0.1:.6e})")

    # Evaluate Cho-Saul + WMC across alpha
    for alpha in [0.09, 0.10, 0.105, 0.110, 0.115, 0.120, 0.130]:
        test_mses = []
        for i in range(8):
            pred = (1.0 - alpha) * c_chosaul_list[i] + alpha * wmc_list[i]
            test_mses.append(float(np.mean((pred - y_trues[i])**2)))
        
        mean_test = np.mean(test_mses)
        diff = ((mean_test - mean_champ) / mean_champ) * 100
        wins = sum(1 for a, b in zip(test_mses, champ_mses) if a < b)
        print(f"Cho-Saul L0 + WMC (alpha={alpha:.3f}): MSE = {mean_test:.6e} (Adj: {mean_test * 0.1:.6e}, Diff: {diff:+.3f}%, Wins: {wins}/8)")

if __name__ == "__main__":
    test_unified_blend()
