import numpy as np
import scipy.optimize as opt
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _blended_hermite_covariance, _whitened_antithetic_mc, _S_PRIOR

def test_two_scale():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("TESTING CONTINUOUS REGIME-AWARE CALIBRATION s(c_i)")
    print("=" * 90)

    # In test_regime_ratios, we found:
    # Large c_i (On neurons, c > 1.2): ratio y*/c is ~0.9983
    # Medium c_i (Kink neurons, 0.01 < c < 1.2): ratio y*/c is ~0.9951
    # Small c_i (Dead neurons, c < 0.01): ratio y*/c is ~0.0 (or damped)

    c_raw_list = []
    y_true_list = []
    wmc_list = []

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y = np.array(row["all_layer_means"][-1], dtype=np.float64)
        c = np.array(_blended_hermite_covariance(mlp)[-1], dtype=np.float64)
        w = np.array(_whitened_antithetic_mc(mlp)[-1], dtype=np.float64)

        c_raw_list.append(c)
        y_true_list.append(y)
        wmc_list.append(w)

    s0 = float(_S_PRIOR)
    champ_mses = [float(np.mean((0.89 * (s0 * c_raw_list[i]) + 0.11 * wmc_list[i] - y_true_list[i])**2)) for i in range(8)]
    mean_champ = np.mean(champ_mses)
    print(f"Current Champ Mean MSE: {mean_champ:.6e} (Adj: {mean_champ * 0.1:.6e})")

    # Let s(c) be a smooth function of c:
    # s(c) = s_inf + (s_0 - s_inf) / (1 + (c / c_mid)^p)
    # or a piecewise linear / polynomial in c:
    # s(c) = s_0 + s_1 * c + s_2 * c^2
    # Let's fit s(c) using Leave-One-Out CV across the 8 MLPs!

    for deg in [1, 2, 3]:
        loo_mses = []
        for test_idx in range(8):
            # Train indices
            train_c = np.concatenate([c_raw_list[j] for j in range(8) if j != test_idx])
            train_y = np.concatenate([y_true_list[j] for j in range(8) if j != test_idx])
            train_w = np.concatenate([wmc_list[j] for j in range(8) if j != test_idx])

            # We want to find s(c) such that 0.89 * (c * s(c)) + 0.11 * w approximates y
            # Equivalent to: c * s(c) approximates y_target = (y - 0.11 * w) / 0.89
            y_target = (train_y - 0.11 * train_w) / 0.89

            # Design matrix for polynomial: [c, c^2, c^3, ...]
            # pred_c = s_0 * c + s_1 * c^2 + s_2 * c^3
            X_train = np.column_stack([train_c ** p for p in range(1, deg + 2)])
            # Solve least squares
            coeffs, _, _, _ = np.linalg.lstsq(X_train, y_target, rcond=None)

            # Test on held-out MLP
            test_c = c_raw_list[test_idx]
            test_y = y_true_list[test_idx]
            test_w = wmc_list[test_idx]

            X_test = np.column_stack([test_c ** p for p in range(1, deg + 2)])
            c_cal_test = X_test @ coeffs
            pred_test = 0.89 * c_cal_test + 0.11 * test_w
            loo_mses.append(float(np.mean((pred_test - test_y)**2)))

        mean_loo = np.mean(loo_mses)
        diff = ((mean_loo - mean_champ) / mean_champ) * 100
        print(f"Degree {deg} Polynomial Scaling: LOO MSE = {mean_loo:.6e} (Adj: {mean_loo*0.1:.6e}, Diff: {diff:+.3f}%)")

if __name__ == "__main__":
    test_two_scale()
