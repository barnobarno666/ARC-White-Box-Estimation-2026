import numpy as np
from scipy.stats import skew, kurtosis
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP

def measure_cumulants_all_layers():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("MEASURING PREACTIVATION CUMULANTS (SKEWNESS & KURTOSIS) ACROSS ALL 16 LAYERS")
    print("=" * 90)

    count = 20000
    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        rng = np.random.default_rng(mlp.seed)
        x = rng.standard_normal((count, mlp.width)).astype(np.float32)

        print(f"\nMLP {i+1} ({row['mlp_name']}):")
        act = x
        for l in range(mlp.depth):
            w = np.array(mlp.weights[l], dtype=np.float32)
            z = act @ w
            sk = skew(z, axis=0)
            kt = kurtosis(z, axis=0)
            mean_sk = float(np.mean(np.abs(sk)))
            mean_kt = float(np.mean(kt))
            std_kt = float(np.std(kt))
            print(f"  Layer {l:2d}: Mean |Skew| = {mean_sk:.5f} | Mean Excess Kurtosis = {mean_kt:+.5f} +/- {std_kt:.5f}")
            act = np.maximum(z, 0.0)

if __name__ == "__main__":
    measure_cumulants_all_layers()
