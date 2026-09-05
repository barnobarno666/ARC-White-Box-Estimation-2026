import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from estimator import _hermite_gain_covariance, _S_PRIOR
from candidates.estimator_lane_a1_thresh_quad_eta1 import _threshold_aware_hermite_covariance

def test_residual_basis():
    ds = load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini", split="mini")
    v, s = resolve_seed_context(ds)

    print("=" * 90)
    print("LANE A3 DIAGNOSTIC: ORACLE CAPTURED RESIDUAL FRACTION R_U^2")
    print("=" * 90)

    s0 = float(_S_PRIOR)

    R2_c_list = []
    R2_quad_list = []
    R2_thresh_list = []
    R2_geom_list = []
    R2_combined_list = []

    for i in range(8):
        row = ds[i]
        mlp = MLP.from_row(row, seed_protocol_version=v, seed_salt=s)
        y_true = np.array(row["final_means"], dtype=np.float64)

        c_champ = np.array(_hermite_gain_covariance(mlp)[-1], dtype=np.float64)
        c0 = s0 * c_champ
        res = y_true - c0
        norm_res_sq = float(np.sum(res ** 2))

        # Vector 1: c0 itself (normalized)
        u1 = c0 / np.linalg.norm(c0)

        # Vector 2: Threshold-aware delta
        c_thresh = np.array(_threshold_aware_hermite_covariance(mlp)[-1], dtype=np.float64)
        delta_thresh = c_thresh - c_champ
        norm_dt = np.linalg.norm(delta_thresh)
        u2 = delta_thresh / norm_dt if norm_dt > 1e-12 else np.zeros_like(u1)

        # Vector 3: Layer 15 incoming weight norm per neuron
        w15 = np.array(mlp.weights[-1], dtype=np.float64)  # (1024, 1024)
        col_norms = np.linalg.norm(w15, axis=0)
        u3 = col_norms - np.mean(col_norms)
        u3 = u3 / np.linalg.norm(u3)

        # Vector 4: Adjoint sensitivity direction w15 @ 1
        adj = np.sum(w15, axis=0)
        u4 = adj - np.mean(adj)
        u4 = u4 / np.linalg.norm(u4)

        # Basis 1: U = [u1] (should be 0 since c0 was orthogonalized by s0!)
        P1_res = (res @ u1) * u1
        r2_1 = float(np.sum(P1_res ** 2) / norm_res_sq)

        # Basis 2: U = [u1, u2]
        # Orthonormalize via QR
        Q2, _ = np.linalg.qr(np.column_stack([u1, u2]))
        P2_res = Q2 @ (Q2.T @ res)
        r2_2 = float(np.sum(P2_res ** 2) / norm_res_sq)

        # Basis 4: U = [u1, u2, u3, u4]
        Q4, _ = np.linalg.qr(np.column_stack([u1, u2, u3, u4]))
        P4_res = Q4 @ (Q4.T @ res)
        r2_4 = float(np.sum(P4_res ** 2) / norm_res_sq)

        # Also let's check SVD of residuals across the 8 MLPs:
        # What is the rank-1 oracle across the panel?

        R2_c_list.append(r2_1)
        R2_thresh_list.append(r2_2)
        R2_combined_list.append(r2_4)

        print(f"{row['mlp_name']:<22} | ||res||^2: {norm_res_sq/1024:.4e} | R^2(c): {r2_1*100:.2f}% | R^2(c, dThresh): {r2_2*100:.2f}% | R^2(Rank 4): {r2_4*100:.2f}%")

    print("-" * 90)
    print(f"PANEL MEAN R^2 (c):         {np.mean(R2_c_list)*100:.2f}%")
    print(f"PANEL MEAN R^2 (c, dThresh): {np.mean(R2_thresh_list)*100:.2f}%")
    print(f"PANEL MEAN R^2 (Rank 4):     {np.mean(R2_combined_list)*100:.2f}%")
    print("=" * 90)

if __name__ == "__main__":
    test_residual_basis()
