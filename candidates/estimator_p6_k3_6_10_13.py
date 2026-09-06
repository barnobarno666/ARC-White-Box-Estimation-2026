"""Phase 6 Pure flopscope.numpy K3-Simple Estimator (Schedule [6, 10, 13], ret=0.62).

Analytical joint cumulant propagation (third-order cumulants + fourth-order harmonic core)
with exact closed-form Step 5 simplification and norm-based factor compression.
Strictly zero torch, scipy, or external numpy dependencies.
Evaluated entirely within flopscope.numpy and flopscope native stats.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

try:
    flops.configure(symmetry_warnings=False)
except Exception:
    pass


def _relu_wick(mean: fnp.ndarray, var: fnp.ndarray, k: int, p: int) -> fnp.ndarray:
    """Closed-form Gaussian ReLU Wick expectation E[d^k ReLU(Z)^p] in float32."""
    sigma = fnp.sqrt(fnp.maximum(var, fnp.asarray(1e-12, dtype=fnp.float32)))
    alpha = mean / sigma
    phi = flops.stats.norm.pdf(alpha).astype(fnp.float32)
    cdf = flops.stats.norm.cdf(alpha).astype(fnp.float32)

    if p == 1:
        if k == 0:
            return sigma * phi + mean * cdf
        if k == 1:
            return cdf
        if k == 2:
            return phi / sigma
        if k == 3:
            return -alpha * phi / (sigma**2)
        if k == 4:
            return (alpha**2 - 1.0) * phi / (sigma**3)
    elif p == 2:
        if k == 0:
            return (mean**2 + var) * cdf + mean * sigma * phi
        if k == 1:
            return 2.0 * (sigma * phi + mean * cdf)
        if k == 2:
            return 2.0 * cdf
        if k == 3:
            return 2.0 * phi / sigma
        if k == 4:
            return -2.0 * alpha * phi / (sigma**2)
    elif p == 3:
        if k == 0:
            return (sigma**3) * ((2.0 + alpha**2) * phi + (3.0 * alpha + alpha**3) * cdf)
        if k == 1:
            return 3.0 * (sigma**2) * (alpha * phi + (1.0 + alpha**2) * cdf)
        if k == 2:
            return 6.0 * (sigma * phi + mean * cdf)
        if k == 3:
            return 6.0 * cdf
        if k == 4:
            return 6.0 * phi / sigma
    elif p == 4:
        if k == 0:
            return (sigma**4) * ((5.0 * alpha + alpha**3) * phi + (3.0 + 6.0 * alpha**2 + alpha**4) * cdf)
        if k == 1:
            return 4.0 * (sigma**3) * ((2.0 + alpha**2) * phi + (3.0 * alpha + alpha**3) * cdf)
        if k == 2:
            return 12.0 * (sigma**2) * (alpha * phi + (1.0 + alpha**2) * cdf)
        if k == 3:
            return 24.0 * (sigma * phi + mean * cdf)
        if k == 4:
            return 24.0 * cdf

    raise ValueError(f"Unsupported Wick pair (k={k}, p={p})")


def _zero_diag(mat: fnp.ndarray) -> fnp.ndarray:
    res = mat.copy()
    fnp.fill_diagonal(res, 0.0)
    return res


def _sym(mat: fnp.ndarray) -> fnp.ndarray:
    return 0.5 * (mat + mat.T)


class Estimator(BaseEstimator):
    """Pure flopscope.numpy K3-Simple Estimator with schedule [6, 10, 13] (ret=0.62)."""

    def setup(self, ctx: SetupContext) -> None:
        _ = ctx
        dummy_mean = fnp.zeros(4, dtype=fnp.float32)
        dummy_var = fnp.ones(4, dtype=fnp.float32)
        _ = _relu_wick(dummy_mean, dummy_var, 0, 1)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        depth = mlp.depth
        n = mlp.width
        eye_n = fnp.eye(n, dtype=fnp.float32)

        mu = fnp.zeros(n, dtype=fnp.float32)
        cov = fnp.eye(n, dtype=fnp.float32)
        A = fnp.zeros((n, 0), dtype=fnp.float32)
        B = fnp.zeros((n, 0), dtype=fnp.float32)
        C = fnp.zeros((n, 0), dtype=fnp.float32)
        c4 = 0.0

        col_birth_layers: list[int] = []
        comp_layers = {6, 10, 13}
        retention = 0.62

        preds = []

        for l in range(depth):
            W_ref = fnp.asarray(mlp.weights[l], dtype=fnp.float32).T

            # 1. Linear step
            mu_pre = W_ref @ mu
            cov_pre = _sym(W_ref @ cov @ W_ref.T)

            R = A.shape[1]
            if R > 0:
                A_pre = W_ref @ A
                B_pre = W_ref @ B
                C_pre = W_ref @ C
                s3 = fnp.sum(A_pre * B_pre * C_pre, axis=1)
                T1 = (A_pre * B_pre) @ C_pre.T
                T2 = (A_pre * C_pre) @ B_pre.T
                T3 = (B_pre * C_pre) @ A_pre.T
                sum_pre = T1 + T2 + T3
                s21 = (1.0 / 3.0) * _zero_diag(sum_pre)
            else:
                A_pre = fnp.zeros((n, 0), dtype=fnp.float32)
                B_pre = fnp.zeros((n, 0), dtype=fnp.float32)
                C_pre = fnp.zeros((n, 0), dtype=fnp.float32)
                s3 = fnp.zeros(n, dtype=fnp.float32)
                s21 = fnp.zeros((n, n), dtype=fnp.float32)
                sum_pre = fnp.zeros((n, n), dtype=fnp.float32)

            M = W_ref @ W_ref.T
            d_M = fnp.diag(M)

            # 2. Nonlinear step
            var_pre = fnp.diag(cov_pre)
            w = lambda k, p: _relu_wick(mu_pre, var_pre, k, p)

            s4 = c4 * (d_M**2)
            s22 = (1.0 / 3.0) * c4 * _zero_diag(d_M[:, None] * d_M[None, :] + 2.0 * (M**2))
            cov11 = _zero_diag(cov_pre)

            # 3. pK slices
            pk = {}
            for p in (1, 2, 3, 4):
                pk[(p,)] = w(0, p) + (1.0 / 6.0) * w(3, p) * s3 + (1.0 / 24.0) * w(4, p) * s4

            pk[(1, 1)] = _sym(
                cov11 * (w(1, 1)[:, None] * w(1, 1)[None, :])
                + s21.T * (w(1, 1)[:, None] * w(2, 1)[None, :])
                + 0.5 * (cov11**2) * (w(2, 1)[:, None] * w(2, 1)[None, :])
                + 0.25 * s22 * (w(2, 1)[:, None] * w(2, 1)[None, :])
            )

            pk[(2, 1)] = (
                cov11 * (w(1, 2)[:, None] * w(1, 1)[None, :])
                + 0.5 * s21.T * (w(1, 2)[:, None] * w(2, 1)[None, :])
                + 0.5 * s21 * (w(2, 2)[:, None] * w(1, 1)[None, :])
                + 0.5 * (cov11**2) * (w(2, 2)[:, None] * w(2, 1)[None, :])
                + 0.25 * s22 * (w(2, 2)[:, None] * w(2, 1)[None, :])
            )

            pk[(2, 2)] = _sym(
                cov11 * (w(1, 2)[:, None] * w(1, 2)[None, :])
                + s21.T * (w(1, 2)[:, None] * w(2, 2)[None, :])
                + 0.5 * (cov11**2) * (w(2, 2)[:, None] * w(2, 2)[None, :])
                + 0.25 * s22 * (w(2, 2)[:, None] * w(2, 2)[None, :])
            )

            # 4. Invert pK to K
            pk1, pk2, pk3, pk4 = pk[(1,)], pk[(2,)], pk[(3,)], pk[(4,)]
            pk11, pk21, pk22 = pk[(1, 1)], pk[(2, 1)], pk[(2, 2)]

            k1 = pk1
            k2 = pk2 - pk1**2
            k3 = pk3 - 3.0 * pk1 * pk2 + 2.0 * (pk1**3)
            k4 = pk4 - 4.0 * pk1 * pk3 - 3.0 * (pk2**2) + 12.0 * (pk1**2) * pk2 - 6.0 * (pk1**4)
            k11 = pk11
            k21 = _zero_diag(pk21 - 2.0 * (pk1[:, None] * pk11))
            k22 = _zero_diag(
                pk22
                - 2.0 * (pk1[None, :] * pk21 + pk1[:, None] * pk21.T)
                - 2.0 * (pk11**2)
                + 4.0 * (pk1[:, None] * pk1[None, :] * pk11)
            )

            w11 = w(1, 1)
            w21 = w(2, 1)

            # 5. Closed-form exact Step 5
            s3_pk = s3 * (w11**3)
            A2 = w11[:, None] * cov11
            C2 = (w21[:, None] * w11[None, :] * cov11).T
            s21_pk = (1.0 / 3.0) * _zero_diag(
                sum_pre * (w11[:, None]**2) * (w11[None, :]) + 3.0 * (A2 * C2)
            )

            k3_ds = k3 - s3_pk
            k21_ds = k21 - s21_pk

            A_ds = fnp.diag(k3_ds) + 3.0 * k21_ds.T
            B_ds = eye_n
            C_ds = eye_n

            if R > 0:
                A_pk = fnp.concatenate([A_pre * w11[:, None], A2], axis=1)
                B_pk = fnp.concatenate([B_pre * w11[:, None], 3.0 * eye_n], axis=1)
                C_pk = fnp.concatenate([C_pre * w11[:, None], C2], axis=1)
            else:
                A_pk = A2
                B_pk = 3.0 * eye_n
                C_pk = C2

            A = fnp.concatenate([A_pk, A_ds], axis=1)
            B = fnp.concatenate([B_pk, B_ds], axis=1)
            C = fnp.concatenate([C_pk, C_ds], axis=1)

            num_new_cols = A.shape[1] - len(col_birth_layers)
            col_birth_layers.extend([l] * num_new_cols)

            # 6. Norm pruning at scheduled layers
            if l in comp_layers:
                old_idx = [i for i, b_l in enumerate(col_birth_layers) if b_l < l]
                new_idx = [i for i, b_l in enumerate(col_birth_layers) if b_l == l]

                if len(old_idx) > 0:
                    norm_a = fnp.sqrt(fnp.sum(A[:, old_idx]**2, axis=0))
                    norm_b = fnp.sqrt(fnp.sum(B[:, old_idx]**2, axis=0))
                    norm_c = fnp.sqrt(fnp.sum(C[:, old_idx]**2, axis=0))
                    scores = norm_a * norm_b * norm_c
                    scores_list = [float(scores[i]) for i in range(len(old_idx))]
                    sorted_indices = sorted(range(len(old_idx)), key=lambda i: scores_list[i])
                    num_to_keep = max(1, int(len(old_idx) * retention))
                    top_local = sorted_indices[-num_to_keep:]
                    kept_old_idx = [old_idx[i] for i in top_local]
                    kept_all = sorted(kept_old_idx + new_idx)

                    A = A[:, kept_all]
                    B = B[:, kept_all]
                    C = C[:, kept_all]
                    col_birth_layers = [col_birth_layers[i] for i in kept_all]

            # 7. Update state
            mu = k1
            cov = k11.copy()
            fnp.fill_diagonal(cov, k2)
            cov = _sym(cov)

            c4 = float((3.0 / (n * (n + 2))) * (fnp.sum(k4) + fnp.sum(k22)))

            preds.append(mu)

        return fnp.stack(preds, axis=0)
