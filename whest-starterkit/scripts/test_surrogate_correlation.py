import numpy as np
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP

def test_surrogates():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    count = 10000
    row = ds[0]
    mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
    rng = np.random.default_rng(mlp.seed)
    x = rng.standard_normal((count, mlp.width)).astype(np.float32)

    act = x
    acts = [x]
    for l in range(mlp.depth):
        w = np.array(mlp.weights[l], dtype=np.float32)
        act = np.maximum(act @ w, 0.0)
        acts.append(act)
    h15 = acts[-1]  # (count, 1024)

    # Filter active neurons (variance > 1e-4)
    var_h15 = np.var(h15, axis=0)
    active = var_h15 > 1e-4
    print(f"Active neurons: {np.sum(active)} / 1024")

    # Layer 14 linear transport to layer 15: acts[14] @ W15
    w15 = np.array(mlp.weights[15], dtype=np.float32)
    z15 = acts[14] @ w15
    # ReLU(z15) IS h15!
    # What about z15 itself (preactivation)?
    r_z15 = [np.corrcoef(h15[:, j], z15[:, j])[0, 1] for j in np.where(active)[0]]
    print(f"Preactivation z15 vs h15: Mean Corr = {np.mean(r_z15):.4f}")

    # What about layer 13 linear transport: acts[13] @ W14 @ W15
    w14 = np.array(mlp.weights[14], dtype=np.float32)
    z14_lin = (acts[13] @ w14) @ w15
    r_z14 = [np.corrcoef(h15[:, j], z14_lin[:, j])[0, 1] for j in np.where(active)[0]]
    print(f"Layer 13 2-step linear transport: Mean Corr = {np.mean(r_z14):.4f}")

    # What about layer 12 3-step linear transport:
    w13 = np.array(mlp.weights[13], dtype=np.float32)
    z13_lin = ((acts[12] @ w13) @ w14) @ w15
    r_z13 = [np.corrcoef(h15[:, j], z13_lin[:, j])[0, 1] for j in np.where(active)[0]]
    print(f"Layer 12 3-step linear transport: Mean Corr = {np.mean(r_z13):.4f}")

    # What about layer 10:
    t = acts[10]
    for l in range(11, 16):
        t = t @ np.array(mlp.weights[l], dtype=np.float32)
    r_z10 = [np.corrcoef(h15[:, j], t[:, j])[0, 1] for j in np.where(active)[0]]
    print(f"Layer 10 5-step linear transport: Mean Corr = {np.mean(r_z10):.4f}")

    # What about layer 0 linear transport (all 16 layers):
    t = x
    for l in range(16):
        t = t @ np.array(mlp.weights[l], dtype=np.float32)
    r_z0 = [np.corrcoef(h15[:, j], t[:, j])[0, 1] for j in np.where(active)[0]]
    print(f"Layer  0 16-step linear transport: Mean Corr = {np.mean(r_z0):.4f}")

if __name__ == "__main__":
    test_surrogates()
