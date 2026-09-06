"""P4N-00 Algebra and Coordinate Fixtures.

Validates:
1. Indexing conventions:
   H[-1] = X
   Z[l] = H[l-1] @ W[l]
   H[l] = relu(Z[l])  for l = 0..15
   Verify mean(Z[l]) == mean(H[l-1]) @ W[l] within numerical tolerance.
2. Coordinate permutation invariance:
   Permuting neurons at layer l and matching row/column weights preserves subsequent activations.
3. Row-vector mapping convention & SVD:
   h is row vector (1 x 1024), W has shape (1024, 1024).
   If W = U @ S @ V.T, then h @ W = h @ U @ S @ V.T, so output coordinates lie in V, NOT U!
4. Synthetic cancellation proof of old split-sample CV:
   0.5 * [(Y_A - b*(Z_A - Z_B)) + (Y_B - b*(Z_B - Z_A))] == 0.5 * (Y_A + Y_B) identically!
5. Centered Stein feature cancellation under antipodal pairing:
   Demonstrates that subtracting separate block means vanishes and proves why odd linear controls died.
"""

import sys
import numpy as np

def test_algebra_fixtures():
    print("=== Running P4N-00 Algebra Fixtures ===")
    rng = np.random.default_rng(42)
    
    # Fixture 1: Indexing & Linearity Check
    # H[-1] = X (N, 1024), W (1024, 1024)
    N = 128
    width = 64  # small dimension fixture for fast algebraic proof
    X = rng.standard_normal((N, width), dtype=np.float32)
    W0 = rng.standard_normal((width, width), dtype=np.float32) * np.sqrt(2.0 / width).astype(np.float32)
    
    H_minus_1 = X
    Z0 = H_minus_1 @ W0
    H0 = np.maximum(Z0, 0.0)
    
    mean_Z0 = np.mean(Z0, axis=0)
    expected_mean_Z0 = np.mean(H_minus_1, axis=0) @ W0
    err_linear = np.max(np.abs(mean_Z0 - expected_mean_Z0))
    print(f"[Fixture 1] mean(Z[0]) vs mean(H[-1]) @ W[0] max diff: {err_linear:.3e}")
    assert err_linear < 1e-5, f"Linearity failed: {err_linear}"
    
    # Fixture 2: Neuron Permutation Invariance
    # Permute neurons at layer 0: P is permutation matrix
    perm = rng.permutation(width)
    P = np.eye(width, dtype=np.float32)[:, perm]  # permutes columns: A @ P permutes cols
    # If Z0_p = H[-1] @ (W0 @ P) = Z0[:, perm]
    # Then H0_p = H0[:, perm]
    # If W1 has matching row permutation P.T @ W1, then H0_p @ (P.T @ W1) = H0 @ W1 !
    W1 = rng.standard_normal((width, width), dtype=np.float32) * np.sqrt(2.0 / width).astype(np.float32)
    
    W0_perm = W0 @ P
    W1_perm = P.T @ W1
    
    Z0_perm = H_minus_1 @ W0_perm
    H0_perm = np.maximum(Z0_perm, 0.0)
    Z1_perm = H0_perm @ W1_perm
    
    Z1_orig = H0 @ W1
    perm_diff = np.max(np.abs(Z1_orig - Z1_perm))
    print(f"[Fixture 2] Permutation invariance check max diff: {perm_diff:.3e}")
    assert perm_diff < 1e-5, f"Permutation invariance failed: {perm_diff}"
    
    # Fixture 3: SVD Row Vector Orientation Check
    # For a row vector h in R^{1xK}, mapping through W in R^{KxM}:
    # y = h @ W.
    # SVD of W: W = U @ np.diag(s) @ Vt. (U: KxK, Vt: MxM)
    # y = h @ U @ np.diag(s) @ Vt.
    # Therefore, the output space coordinates are spanned by rows of Vt (columns of V), NOT columns of U!
    K, M = 32, 32
    h = rng.standard_normal((1, K), dtype=np.float32)
    W = rng.standard_normal((K, M), dtype=np.float32)
    U, s, Vt = np.linalg.svd(W, full_matrices=True)
    V = Vt.T
    
    # Projecting output y onto Vt[k]:
    y = h @ W
    proj_V = y @ V  # coordinates in output singular basis V
    reconstructed = proj_V @ Vt
    svd_diff = np.max(np.abs(y - reconstructed))
    print(f"[Fixture 3] SVD row-vector output coordinates in V max diff: {svd_diff:.3e}")
    assert svd_diff < 1e-5, f"SVD orientation failed: {svd_diff}"
    
    # Fixture 4: Synthetic Cancellation Proof of Old Split-Sample CV
    # Old script did: 0.5 * [(Y_A - b*(Z_A - Z_B)) + (Y_B - b*(Z_B - Z_A))]
    # Algebra: = 0.5 * [Y_A + Y_B - b*(Z_A - Z_B + Z_B - Z_A)] = 0.5 * (Y_A + Y_B) identically!
    Y_A = rng.standard_normal(10)
    Y_B = rng.standard_normal(10)
    Z_A = rng.standard_normal(10)
    Z_B = rng.standard_normal(10)
    b = 0.75
    old_split = 0.5 * ((Y_A - b * (Z_A - Z_B)) + (Y_B - b * (Z_B - Z_A)))
    raw_mean = 0.5 * (Y_A + Y_B)
    split_cancel_diff = np.max(np.abs(old_split - raw_mean))
    print(f"[Fixture 4] Old split-sample CV algebraic cancellation diff: {split_cancel_diff:.3e}")
    assert split_cancel_diff < 1e-15, f"Algebraic cancellation failed: {split_cancel_diff}"
    
    # Fixture 5: Antipodal Pairing and Odd Feature Annihilation
    # For antipodal pairs x and -x:
    # Any odd linear function f(x) = v^T x satisfies f(-x) = -v^T x, so (f(x) + f(-x))/2 = 0 exactly.
    # Therefore, linear input controls provide exactly ZERO variance reduction for antipodal MC!
    x = rng.standard_normal((N, width))
    v = rng.standard_normal(width)
    v /= np.linalg.norm(v)
    f_pos = x @ v
    f_neg = (-x) @ v
    pair_avg_linear = 0.5 * (f_pos + f_neg)
    max_linear_pair = np.max(np.abs(pair_avg_linear))
    print(f"[Fixture 5] Antipodal linear feature pair average max: {max_linear_pair:.3e}")
    assert max_linear_pair < 1e-15, f"Antipodal pairing property failed: {max_linear_pair}"
    
    print("=== All P4N-00 Algebra Fixtures PASSED Successfully! ===")

if __name__ == "__main__":
    test_algebra_fixtures()
