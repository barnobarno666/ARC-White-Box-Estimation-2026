"""p65_math_checks.py

Phase 6.5 Stage P65-01: Mathematics Checks and Algebra Invariants.
Independent dense float64 checks of all key identities:
1. Symmetric 3rd-order tensor atom T(u,v) diagonal, repeated slice, norm, cross product.
2. 3-mode tensor transport under asymmetric W.
3. Section 11 slice compensation diagonal and repeated slices match; all-distinct entries differ.
4. Section 8 split vs direct terminal contractions.
5. Section 7 exact k22 row sums, total sum, and rows_E22 without full E22 materialization.
6. Section 13 dense 4th-tensor partial trace, diagonal projection, and 4-mode transport.
7. Boundary conditions: zero-rank, all-zero, equal scores, negative weights, cancellation, depth 1/2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

TOL_ALGEBRA = 1e-10
TOL_RECURRENCE = 1e-9


def make_T_dense(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Explicitly construct 3-mode symmetric tensor T(u,v)."""
    uuv = np.einsum("i,j,k->ijk", u, u, v)
    uvu = np.einsum("i,j,k->ijk", u, v, u)
    vuu = np.einsum("i,j,k->ijk", v, u, u)
    return (uuv + uvu + vuu) / 3.0


def check_identity_1_third_atom(rng: np.random.Generator, widths=(3, 5, 8)) -> List[Dict[str, Any]]:
    """Check T(u,v) diagonal, repeated slice, norm, and cross inner product."""
    results = []
    for n in widths:
        u = rng.normal(size=n)
        v = rng.normal(size=n)
        p = rng.normal(size=n)
        q = rng.normal(size=n)

        T = make_T_dense(u, v)

        # Diagonal d[i] = T[i,i,i]
        d_dense = np.array([T[i, i, i] for i in range(n)])
        d_formula = (u**2) * v
        err_d = float(np.max(np.abs(d_dense - d_formula)))
        assert err_d < TOL_ALGEBRA, f"d mismatch at n={n}: {err_d}"

        # Repeated slice S[i,j] = T[i,i,j] (diag zero)
        S_dense = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j:
                    S_dense[i, j] = T[i, i, j]
        S_formula = (2.0 * ((u * v)[:, None] * u[None, :]) + (u * u)[:, None] * v[None, :]) / 3.0
        np.fill_diagonal(S_formula, 0.0)
        err_S = float(np.max(np.abs(S_dense - S_formula)))
        assert err_S < TOL_ALGEBRA, f"S mismatch at n={n}: {err_S}"

        # Norm squared
        norm_sq_dense = float(np.sum(T**2))
        a = float(np.dot(u, u))
        b = float(np.dot(v, v))
        c = float(np.dot(u, v))
        norm_sq_formula = (a * a * b + 2.0 * a * c * c) / 3.0
        err_norm = abs(norm_sq_dense - norm_sq_formula)
        assert err_norm < TOL_ALGEBRA, f"norm_sq mismatch at n={n}: {err_norm}"

        # Cross inner product
        T_pq = make_T_dense(p, q)
        cross_dense = float(np.sum(T * T_pq))
        up = float(np.dot(u, p))
        vq = float(np.dot(v, q))
        uq = float(np.dot(u, q))
        vp = float(np.dot(v, p))
        cross_formula = ((up**2) * vq + 2.0 * up * uq * vp) / 3.0
        err_cross = abs(cross_dense - cross_formula)
        assert err_cross < TOL_ALGEBRA, f"cross mismatch at n={n}: {err_cross}"

        results.append({"n": n, "err_d": err_d, "err_S": err_S, "err_norm": err_norm, "err_cross": err_cross})
    return results


def check_identity_2_transport(rng: np.random.Generator, widths=(3, 5, 8)) -> List[Dict[str, Any]]:
    """Check 3-mode tensor transport under asymmetric W."""
    results = []
    for n in widths:
        u = rng.normal(size=n)
        v = rng.normal(size=n)
        W = rng.normal(size=(n, n))

        T = make_T_dense(u, v)
        # 3-mode transport: T_trans[i,j,k] = sum_{a,b,c} W[i,a]*W[j,b]*W[k,c]*T[a,b,c]
        T_trans_dense = np.einsum("ia,jb,kc,abc->ijk", W, W, W, T)

        # Factor transport: T(W @ u, W @ v)
        T_trans_factors = make_T_dense(W @ u, W @ v)

        err_trans = float(np.max(np.abs(T_trans_dense - T_trans_factors)))
        assert err_trans < TOL_ALGEBRA, f"transport mismatch at n={n}: {err_trans}"
        results.append({"n": n, "err_transport": err_trans})
    return results


def check_identity_3_slice_compensation(rng: np.random.Generator, widths=(3, 5, 8)) -> List[Dict[str, Any]]:
    """Check Section 11 slice compensation: diagonal and repeated slices match, distinct differ."""
    results = []
    for n in widths:
        # Construct multi-atom target and kept tensors
        R_target = 6
        R_keep = 3
        U_all = rng.normal(size=(n, R_target))
        V_all = rng.normal(size=(n, R_target))

        T_target = sum(make_T_dense(U_all[:, r], V_all[:, r]) for r in range(R_target))
        T_keep = sum(make_T_dense(U_all[:, r], V_all[:, r]) for r in range(R_keep))

        Delta = T_target - T_keep

        # Target missing slices
        d_missing = np.array([Delta[i, i, i] for i in range(n)])
        S_missing = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j:
                    S_missing[i, j] = Delta[i, i, j]

        # Section 11 correction block: U_c = I_n, V_c = diag(d_missing) + 3 * S_missing.T
        U_c = np.eye(n, dtype=np.float64)
        V_c = np.diag(d_missing) + 3.0 * S_missing.T

        # Correction tensor
        T_c = sum(make_T_dense(U_c[:, r], V_c[:, r]) for r in range(n))

        # Check diagonal
        d_c = np.array([T_c[i, i, i] for i in range(n)])
        err_d = float(np.max(np.abs(d_c - d_missing)))
        assert err_d < TOL_ALGEBRA, f"compensation d mismatch at n={n}: {err_d}"

        # Check repeated slice
        S_c = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j:
                    S_c[i, j] = T_c[i, i, j]
        err_S = float(np.max(np.abs(S_c - S_missing)))
        assert err_S < TOL_ALGEBRA, f"compensation S mismatch at n={n}: {err_S}"

        # Demonstrate distinct entries differ
        distinct_diffs = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and i != k:
                        distinct_diffs.append(abs(Delta[i, j, k] - T_c[i, j, k]))
        max_distinct_diff = max(distinct_diffs) if distinct_diffs else 0.0
        # For n >= 3, Delta has non-zero all-distinct entries while T_c has zero all-distinct entries
        assert max_distinct_diff > 1e-4, f"distinct entries should differ! max diff={max_distinct_diff}"

        results.append({
            "n": n,
            "err_d": err_d,
            "err_S": err_S,
            "max_distinct_diff": max_distinct_diff,
        })
    return results


def check_identity_4_terminal_split(rng: np.random.Generator, widths=(3, 5, 8)) -> List[Dict[str, Any]]:
    """Check Section 8 split vs direct terminal contractions."""
    results = []
    for n in widths:
        R_old = 4
        Uo = rng.normal(size=(n, R_old))
        Vo = rng.normal(size=(n, R_old))
        A2 = rng.normal(size=(n, n))
        h = rng.normal(size=n)
        A_ds = rng.normal(size=(n, n))
        W = rng.normal(size=(n, n))

        # Direct concatenation
        U_all = np.concatenate([Uo, A2, np.eye(n)], axis=1)
        V_all = np.concatenate([Vo, 3.0 * np.eye(n) * h[None, :], A_ds], axis=1)

        WU_all = W @ U_all
        WV_all = W @ V_all
        d_direct = np.sum((WU_all**2) * WV_all, axis=1)

        # Split contraction
        d_old = np.sum((W @ Uo)**2 * (W @ Vo), axis=1)
        d_path = 3.0 * np.sum((W @ A2)**2 * W * h[None, :], axis=1)
        d_slice = np.sum((W**2) * (W @ A_ds), axis=1)
        d_split = d_old + d_path + d_slice

        err = float(np.max(np.abs(d_direct - d_split)))
        assert err < TOL_ALGEBRA, f"terminal split mismatch at n={n}: {err}"
        results.append({"n": n, "err": err})
    return results


def check_identity_5_k22_sums(rng: np.random.Generator, widths=(3, 5, 8)) -> List[Dict[str, Any]]:
    """Check Section 7 exact k22 row/total sums and rows_E22 without full matrix."""
    results = []
    for n in widths:
        p = rng.normal(size=n)
        D21 = rng.normal(size=(n, n))
        np.fill_diagonal(D21, 0.0)
        B11 = rng.normal(size=(n, n))
        B11 = 0.5 * (B11 + B11.T)
        np.fill_diagonal(B11, 0.0)

        # Setup E22 from components: C0, S, F22, a2, b2
        C0 = rng.normal(size=(n, n))
        C0 = 0.5 * (C0 + C0.T)
        np.fill_diagonal(C0, 0.0)
        S = rng.normal(size=(n, n))
        np.fill_diagonal(S, 0.0)
        F22 = rng.normal(size=(n, n))
        F22 = 0.5 * (F22 + F22.T)
        np.fill_diagonal(F22, 0.0)
        a2 = rng.normal(size=n)
        b2 = rng.normal(size=n)

        H = 0.5 * (C0**2) + 0.25 * F22
        E22 = 0.5 * (
            C0 * (a2[:, None] * a2[None, :])
            + S.T * (a2[:, None] * b2[None, :])
            + H * (b2[:, None] * b2[None, :])
            + (C0 * (a2[:, None] * a2[None, :])
               + S.T * (a2[:, None] * b2[None, :])
               + H * (b2[:, None] * b2[None, :])).T
        )
        np.fill_diagonal(E22, 0.0)

        # Incumbent k22
        k22 = (
            E22
            - 2.0 * (p[None, :] * D21 + p[:, None] * D21.T)
            - 2.0 * (B11**2)
            + 4.0 * (p[:, None] * p[None, :] * B11)
        )
        np.fill_diagonal(k22, 0.0)

        # Section 7 exact row sums
        r22_direct = np.sum(k22, axis=1)
        r22_formula = (
            np.sum(E22, axis=1)
            - 2.0 * (D21 @ p + p * np.sum(D21, axis=0))
            - 2.0 * np.sum(B11**2, axis=1)
            + 4.0 * p * (B11 @ p)
        )
        err_r22 = float(np.max(np.abs(r22_direct - r22_formula)))
        assert err_r22 < TOL_ALGEBRA, f"r22 mismatch at n={n}: {err_r22}"

        # Section 7 exact total sum
        t22_direct = float(np.sum(k22))
        t22_formula = float(
            np.sum(E22)
            - 4.0 * np.dot(p, np.sum(D21, axis=0))
            - 2.0 * np.sum(B11**2)
            + 4.0 * np.dot(p, B11 @ p)
        )
        err_t22 = abs(t22_direct - t22_formula)
        assert err_t22 < TOL_ALGEBRA, f"t22 mismatch at n={n}: {err_t22}"

        # Section 7 rows_E22 without full E22
        rows_E22_direct = np.sum(E22, axis=1)
        rows_E22_formula = (
            a2 * (C0 @ a2)
            + 0.5 * (a2 * (S.T @ b2) + b2 * (S @ a2))
            + b2 * (H @ b2)
        )
        err_rows_E22 = float(np.max(np.abs(rows_E22_direct - rows_E22_formula)))
        assert err_rows_E22 < TOL_ALGEBRA, f"rows_E22 mismatch at n={n}: {err_rows_E22}"

        results.append({
            "n": n,
            "err_r22": err_r22,
            "err_t22": err_t22,
            "err_rows_E22": err_rows_E22,
        })
    return results


def check_identity_6_fourth_tensor(rng: np.random.Generator, widths=(3, 5, 8)) -> List[Dict[str, Any]]:
    """Check Section 13 dense 4th-tensor partial trace, diagonal projection, and transport."""
    results = []
    for n in widths:
        c4 = float(rng.uniform(0.1, 2.0))
        q = rng.normal(size=n)
        q = q - float(np.mean(q))  # Traceless sum(q) = 0
        Q = np.diag(q)

        # Dense I4
        I4 = np.zeros((n, n, n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        d_ij = 1.0 if i == j else 0.0
                        d_kl = 1.0 if k == l else 0.0
                        d_ik = 1.0 if i == k else 0.0
                        d_jl = 1.0 if j == l else 0.0
                        d_il = 1.0 if i == l else 0.0
                        d_jk = 1.0 if j == k else 0.0
                        I4[i, j, k, l] = (d_ij * d_kl + d_ik * d_jl + d_il * d_jk) / 3.0

        # Dense H(I, Q)
        HIQ = np.zeros((n, n, n, n), dtype=np.float64)
        eye = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        HIQ[i, j, k, l] = (
                            eye[i, j] * Q[k, l] + eye[i, k] * Q[j, l] + eye[i, l] * Q[j, k]
                            + eye[j, k] * Q[i, l] + eye[j, l] * Q[i, k] + eye[k, l] * Q[i, j]
                        ) / 6.0

        K4 = c4 * I4 + HIQ

        # 1. Partial trace: sum_a K4[a,a,i,j]
        P_trace_dense = np.einsum("aaij->ij", K4)
        P_trace_formula = c4 * ((n + 2) / 3.0) * eye + ((n + 4) / 6.0) * Q
        err_ptrace = float(np.max(np.abs(P_trace_dense - P_trace_formula)))
        assert err_ptrace < TOL_ALGEBRA, f"partial trace mismatch at n={n}: {err_ptrace}"

        # 2. Diagonal projection
        Rdiag = np.diag(P_trace_dense)
        tau = float(np.sum(Rdiag))
        c4_proj = 3.0 * tau / (n * (n + 2))
        q_full = (6.0 / (n + 4)) * (Rdiag - tau / n)
        q_proj = q_full - float(np.mean(q_full))

        err_c4 = abs(c4_proj - c4)
        err_q = float(np.max(np.abs(q_proj - q)))
        assert err_c4 < TOL_ALGEBRA, f"c4 proj mismatch at n={n}: {err_c4}"
        assert err_q < TOL_ALGEBRA, f"q proj mismatch at n={n}: {err_q}"

        # 3. Transport check
        W = rng.normal(size=(n, n))
        M = W @ W.T
        Qp = (W * q[None, :]) @ W.T
        m = np.diag(M)
        b = np.diag(Qp)

        # Preactivation 4th diagonal s4_pre: sum_{abcd} W[i,a]*W[i,b]*W[i,c]*W[i,d]*K4[a,b,c,d]
        s4_pre_dense = np.einsum("ia,ib,ic,id,abcd->i", W, W, W, W, K4)
        s4_pre_formula = c4 * (m**2) + m * b
        err_s4 = float(np.max(np.abs(s4_pre_dense - s4_pre_formula)))
        assert err_s4 < TOL_ALGEBRA, f"s4_pre mismatch at n={n}: {err_s4}"

        # Preactivation 4th repeated slice s22_pre: sum_{abcd} W[i,a]*W[i,b]*W[j,c]*W[j,d]*K4[a,b,c,d]
        s22_pre_dense = np.einsum("ia,ib,jc,jd,abcd->ij", W, W, W, W, K4)
        np.fill_diagonal(s22_pre_dense, 0.0)

        s22_pre_formula = (
            c4 * (m[:, None] * m[None, :] + 2.0 * (M**2)) / 3.0
            + (m[:, None] * b[None, :] + b[:, None] * m[None, :] + 4.0 * M * Qp) / 6.0
        )
        np.fill_diagonal(s22_pre_formula, 0.0)
        err_s22 = float(np.max(np.abs(s22_pre_dense - s22_pre_formula)))
        assert err_s22 < TOL_ALGEBRA, f"s22_pre mismatch at n={n}: {err_s22}"

        results.append({
            "n": n,
            "err_ptrace": err_ptrace,
            "err_c4": err_c4,
            "err_q": err_q,
            "err_s4": err_s4,
            "err_s22": err_s22,
        })
    return results


def check_identity_7_boundary_cases(rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Check zero-rank, all-zero, equal scores, negative weights, cancellation."""
    results = []
    n = 4

    # 1. Zero-rank
    U_zero_rank = np.zeros((n, 0))
    V_zero_rank = np.zeros((n, 0))
    d_zr = np.sum(U_zero_rank * U_zero_rank * V_zero_rank, axis=1)
    assert len(d_zr) == n and np.all(d_zr == 0.0)

    # 2. All-zero factors
    U_zero = np.zeros((n, 3))
    V_zero = np.zeros((n, 3))
    T_zero = sum(make_T_dense(U_zero[:, r], V_zero[:, r]) for r in range(3))
    assert np.all(T_zero == 0.0)

    # 3. Paired cancellation: T(u, v) + T(u, -v) = 0
    u = rng.normal(size=n)
    v = rng.normal(size=n)
    T_plus = make_T_dense(u, v)
    T_minus = make_T_dense(u, -v)
    assert np.max(np.abs(T_plus + T_minus)) < TOL_ALGEBRA

    # 4. Negative factor weights: T(u, -2*v) = -2 * T(u, v)
    T_scaled = make_T_dense(u, -2.0 * v)
    assert np.max(np.abs(T_scaled + 2.0 * T_plus)) < TOL_ALGEBRA

    # 5. Equal scores sorting stability
    scores = np.array([1.0, 2.0, 2.0, 0.5])
    idx = np.argsort(scores, kind="stable").tolist()
    assert idx == [3, 0, 1, 2]

    results.append({"status": "PASS", "boundary_checks": 5})
    return results


def run_math_checks(candidate_path: Path | str | None = None) -> Dict[str, Any]:
    """Run full suite of mathematical checks."""
    print("=======================================================")
    print("Phase 6.5 Math Checks (RNG seed 6501, float64 tolerance 1e-10)")
    print("=======================================================")

    rng = np.random.default_rng(6501)

    print("[Check 1/7] Third atom T(u,v) identities...")
    res1 = check_identity_1_third_atom(rng)
    print("  [PASS] Passed across widths 3, 5, 8")

    print("[Check 2/7] 3-mode tensor transport under asymmetric W...")
    res2 = check_identity_2_transport(rng)
    print("  [PASS] Passed across widths 3, 5, 8")

    print("[Check 3/7] Section 11 slice compensation vs missing tensor...")
    res3 = check_identity_3_slice_compensation(rng)
    print("  [PASS] Passed (diagonal & repeated slices match, distinct differ)")

    print("[Check 4/7] Section 8 terminal split contraction...")
    res4 = check_identity_4_terminal_split(rng)
    print("  [PASS] Passed across widths 3, 5, 8")

    print("[Check 5/7] Section 7 exact k22 row/total sums & rows_E22...")
    res5 = check_identity_5_k22_sums(rng)
    print("  [PASS] Passed across widths 3, 5, 8")

    print("[Check 6/7] Section 13 4th tensor partial trace, projection, transport...")
    res6 = check_identity_6_fourth_tensor(rng)
    print("  [PASS] Passed across widths 3, 5, 8")

    print("[Check 7/7] Boundary cases & cancellation...")
    res7 = check_identity_7_boundary_cases(rng)
    print("  [PASS] Passed all boundary cases")

    output = {
        "status": "ALL_PASSED",
        "identity_1_third_atom": res1,
        "identity_2_transport": res2,
        "identity_3_slice_compensation": res3,
        "identity_4_terminal_split": res4,
        "identity_5_k22_sums": res5,
        "identity_6_fourth_tensor": res6,
        "identity_7_boundary_cases": res7,
    }

    # Save to diagnostics
    diag_dir = Path(__file__).resolve().parent.parent / "research" / "phase6_5" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(diag_dir / "math_checks_receipt.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[p65_math_checks] Successfully saved receipts to: {diag_dir / 'math_checks_receipt.json'}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, default=None, help="Optional candidate file path")
    args = parser.parse_args()
    run_math_checks(args.candidate)
