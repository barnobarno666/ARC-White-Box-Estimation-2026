import numpy as np
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _hermite_gain_covariance, _whitened_antithetic_mc, _S_PRIOR
from candidates.estimator_blended_hermite_lam02 import _blended_hermite_covariance as cov_lam02
from candidates.estimator_blended_hermite_lam05 import _blended_hermite_covariance as cov_lam05

def run_robustness_screen():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("STAGE 3 ROBUSTNESS SCREEN: 8 MLPS ACROSS 3 SAMPLER SALTS")
    print("=" * 90)

    s0 = float(_S_PRIOR)
    d = 1024
    count = 4200
    half = count // 2

    # 3 deterministic sampler salt offsets
    salts = [0, 1337, 8888]

    for cand_name, cov_fn in [("Blended Hermite (lam=0.20)", cov_lam02), ("Blended Hermite (lam=0.50)", cov_lam05)]:
        print(f"\n--- Testing Candidate: {cand_name} ---")
        salt_scores = []
        salt_wins = []

        for salt in salts:
            cand_scores = []
            champ_scores = []
            wins = 0

            for i in range(8):
                row = ds[i]
                mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
                y_true = np.array(row["final_means"], dtype=np.float64)

                # Covariance predictions
                c_champ = np.array(_hermite_gain_covariance(mlp)[-1], dtype=np.float64)
                c_cand = np.array(cov_fn(mlp)[-1], dtype=np.float64)

                # WMC with explicit seed offset
                rng = np.random.default_rng(mlp.seed + salt)
                x_half = rng.standard_normal((half, d)).astype(np.float32)
                x = np.concatenate((x_half, -x_half), axis=0)

                gram = (x.T @ x) / float(count)
                eigvals, eigvecs = np.linalg.eigh(gram)
                eigvals = np.maximum(eigvals, 1e-6)
                whitener = (eigvecs * (eigvals ** -0.5)) @ eigvecs.T

                w0 = np.array(mlp.weights[0], dtype=np.float32)
                act = np.maximum(x @ (whitener @ w0), 0.0)
                for l in range(1, mlp.depth):
                    w = np.array(mlp.weights[l], dtype=np.float32)
                    act = np.maximum(act @ w, 0.0)
                m = np.mean(act, axis=0)

                pred_champ = 0.89 * (s0 * c_champ) + 0.11 * m
                score_champ = float(np.mean((pred_champ - y_true) ** 2)) * 0.1

                pred_cand = 0.89 * (s0 * c_cand) + 0.11 * m
                score_cand = float(np.mean((pred_cand - y_true) ** 2)) * 0.1

                cand_scores.append(score_cand)
                champ_scores.append(score_champ)

                if score_cand < score_champ:
                    wins += 1

            mean_cand = np.mean(cand_scores)
            mean_champ = np.mean(champ_scores)
            diff = ((mean_cand - mean_champ) / mean_champ) * 100.0
            salt_scores.append(mean_cand)
            salt_wins.append(wins)
            print(f"Salt +{salt:<4}: Cand={mean_cand:.6e} | Champ={mean_champ:.6e} | Diff={diff:+.2f}% | Wins={wins}W-{8-wins}L")

        print(f"--> Summary for {cand_name}: Mean={np.mean(salt_scores):.6e}, Avg Wins={np.mean(salt_wins):.1f}/8")

if __name__ == "__main__":
    run_robustness_screen()
