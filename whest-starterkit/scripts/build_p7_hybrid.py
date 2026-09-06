"""Emit one standalone, fixed rank-64 DS/Tucker candidate from the frozen control.

This builder preserves the control's Wick and nonlinear moment formulas verbatim.
It does not evaluate networks or change the Phase 7 research plan.
"""
from pathlib import Path
import hashlib
import textwrap

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "candidates/estimator_p65_final.py"
OUTPUT = ROOT / "candidates/estimator_p7_hybrid_k64.py"
EXPECTED = "b06d91bc155f19e696c690c2b9956c2ba2426ba142de3c7d6dcf7f078344255e"


HELPERS = '''
CORE_RANK = 64
COMPRESS_FROM = 3


def _factor_slices(U, V):
    d = fnp.sum(U * U * V, axis=1)
    S = _zero_diag((2.0 * ((U * V) @ U.T) + (U * U) @ V.T) / 3.0)
    return d, S


def _core_slices(Q, G, terminal=False):
    n, k = Q.shape
    pairs = (Q[:, :, None] * Q[:, None, :]).reshape(n, k * k)
    L = pairs @ G.reshape(k * k, k)
    d = fnp.sum(L * Q, axis=1)
    return d, None if terminal else _zero_diag(L @ Q.T)


def _factor_contract2(U, V, Z1, Z2):
    a, b = U.T @ Z1, U.T @ Z2
    c, d = V.T @ Z1, V.T @ Z2
    return (U @ (a * d + b * c) + V @ (a * b)) / 3.0


def _ds_contract2(D, Z1, Z2):
    return (Z1 * (D.T @ Z2) + Z2 * (D.T @ Z1) + D @ (Z1 * Z2)) / 3.0


def _core_contract2(Q, G, Z1, Z2):
    k = Q.shape[1]
    a, b = Q.T @ Z1, Q.T @ Z2
    pairs = (a[:, None, :] * b[None, :, :]).reshape(k * k, Z1.shape[1])
    return Q @ (G.reshape(k * k, k).T @ pairs)


def _project_factors(P, U, V):
    A, B = P.T @ U, P.T @ V
    k, r = A.shape
    # Largest temporary: k*k*r, not n*n*n or n*n*k*k.
    pairs = (A[:, None, :] * A[None, :, :]).reshape(k * k, r)
    C = (pairs @ B.T).reshape(k, k, k)
    return (C + C.transpose(0, 2, 1) + C.transpose(2, 0, 1)) / 3.0


def _project_ds(P, D):
    # U=I, V=D; avoid multiplying the projection by a dense identity.
    A, B = P.T, P.T @ D
    k, n = A.shape
    pairs = (A[:, None, :] * A[None, :, :]).reshape(k * k, n)
    C = (pairs @ B.T).reshape(k, k, k)
    return (C + C.transpose(0, 2, 1) + C.transpose(2, 0, 1)) / 3.0


def _project_core(P, Q, G):
    H = P.T @ Q
    old_k, new_k = Q.shape[1], P.shape[1]
    C = (H @ G.reshape(old_k, old_k * old_k)).reshape(new_k, old_k, old_k)
    C = (H @ C.transpose(1, 0, 2).reshape(old_k, new_k * old_k))
    C = C.reshape(new_k, new_k, old_k).transpose(1, 0, 2)
    C = H @ C.transpose(2, 0, 1).reshape(old_k, new_k * new_k)
    return C.reshape(new_k, new_k, new_k).transpose(1, 2, 0)


def _sym_core(G):
    return (G + G.transpose(0, 2, 1) + G.transpose(1, 0, 2)
            + G.transpose(1, 2, 0) + G.transpose(2, 0, 1)
            + G.transpose(2, 1, 0)) / 6.0


def _compress_old(U, V, Q, G, d_old, S_old, rng):
    n = U.shape[0]
    k = min(CORE_RANK, n)
    p = min(n, k + 8)
    D_old = fnp.diag(d_old) + 3.0 * S_old.T
    scale = fnp.sqrt(fnp.asarray(1.0 / n, dtype=fnp.float32))
    Z1 = rng.standard_normal((n, p), dtype=fnp.float32) * scale
    Z2 = rng.standard_normal((n, p), dtype=fnp.float32) * scale
    Y = _factor_contract2(U, V, Z1, Z2) - _ds_contract2(D_old, Z1, Z2)
    if Q is not None:
        Y = Y + _core_contract2(Q, G, Z1, Z2)
    left, _, _ = fnp.linalg.svd(Y, full_matrices=False)
    P = left[:, :k]
    core = _project_factors(P, U, V) - _project_ds(P, D_old)
    if Q is not None:
        core = core + _project_core(P, Q, G)
    core = _sym_core(core)
    d_core, S_core = _core_slices(P, core)
    D_summary = fnp.diag(d_old - d_core) + 3.0 * (S_old - S_core).T
    return P, core, D_summary


class Estimator(BaseEstimator):
    """Preserve repeated K3 slices; compress older all-distinct state at rank 64.

    One fixed setting: compression starts after layer 3 and ends at depth-3.
    The current path birth remains explicit. The slice birth and old DS summary
    merge exactly. No labels, cached centers, pilot network, or rank sweep.
    """

    def setup(self, ctx: SetupContext) -> None:
        # No numerical precomputation outside the predict budget.
        pass

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        n, depth = mlp.width, mlp.depth
        mu = fnp.zeros(n, dtype=fnp.float32)
        cov = fnp.eye(n, dtype=fnp.float32)
        c4 = 0.0
        U = fnp.zeros((n, 0), dtype=fnp.float32)
        V = fnp.zeros((n, 0), dtype=fnp.float32)
        Q = G = D = path_A = path_h = None
        rng = fnp.random.default_rng(mlp.seed)
        predictions = []

        for layer in range(depth):
            W = fnp.asarray(mlp.weights[layer], dtype=fnp.float32).T
            terminal = layer == depth - 1
            mu_pre = W @ mu
            if terminal:
                var_pre = fnp.sum((W @ cov) * W, axis=1)
                metric_diag = fnp.sum(W * W, axis=1)
            elif layer == 0:
                metric = W @ W.T
                cov_pre = _sym(metric)
                metric_diag = fnp.diag(metric)
            else:
                cov_pre = _sym(W @ cov @ W.T)
                metric = W @ W.T
                metric_diag = fnp.diag(metric)

            u_parts, v_parts = [], []
            if U.shape[1]:
                u_parts.append(W @ U)
                v_parts.append(W @ V)
            if D is not None:
                u_parts.append(W)
                v_parts.append(W @ D)
            if path_A is not None:
                u_parts.append(W @ path_A)
                v_parts.append(3.0 * W * path_h[None, :])
            if u_parts:
                Up = fnp.concatenate(u_parts, axis=1)
                Vp = fnp.concatenate(v_parts, axis=1)
                if terminal:
                    s3 = fnp.sum(Up * Up * Vp, axis=1)
                else:
                    s3, s21 = _factor_slices(Up, Vp)
            else:
                Up = fnp.zeros((n, 0), dtype=fnp.float32)
                Vp = fnp.zeros((n, 0), dtype=fnp.float32)
                s3 = fnp.zeros(n, dtype=fnp.float32)
                if not terminal:
                    s21 = fnp.zeros((n, n), dtype=fnp.float32)
            Qp = None
            if Q is not None:
                Qp = W @ Q
                dc, Sc = _core_slices(Qp, G, terminal=terminal)
                s3 = s3 + dc
                if not terminal:
                    s21 = s21 + Sc
            s4 = c4 * metric_diag**2

            if terminal:
                base = _wick_base(mu_pre, var_pre)
                mu = (_relu_wick(mu_pre, var_pre, 0, 1, base)
                      + _relu_wick(mu_pre, var_pre, 3, 1, base) * s3 / 6.0
                      + _relu_wick(mu_pre, var_pre, 4, 1, base) * s4 / 24.0)
                predictions.append(mu)
                break

            s22 = (c4 / 3.0) * _zero_diag(
                metric_diag[:, None] * metric_diag[None, :] + 2.0 * metric**2)
            mu_next, cov_next, k3, k21, gate, h_wick, c4_next = _nonlinear(
                mu_pre, cov_pre, s3, s21, s4, s22)
            A2 = gate[:, None] * _zero_diag(cov_pre)
            d_old = s3 * gate**3
            S_old = s21 * gate[:, None]**2 * gate[None, :]
            path_S = _zero_diag(A2**2 * h_wick[None, :])
            D_birth = fnp.diag(k3 - d_old) + 3.0 * (k21 - S_old - path_S).T
            Ug, Vg = Up * gate[:, None], Vp * gate[:, None]
            Qg = None if Qp is None else Qp * gate[:, None]

            if COMPRESS_FROM <= layer < depth - 2:
                Q, G, D_summary = _compress_old(Ug, Vg, Qg, G, d_old, S_old, rng)
                U = fnp.zeros((n, 0), dtype=fnp.float32)
                V = fnp.zeros((n, 0), dtype=fnp.float32)
                # Both summaries have the same U=I and merge exactly.
                D = D_summary + D_birth
            else:
                U, V, Q, D = Ug, Vg, Qg, D_birth
            path_A, path_h = A2, h_wick
            mu, cov, c4 = mu_next, cov_next, c4_next
            predictions.append(mu)

        return fnp.stack(predictions, axis=0)
'''


def main():
    data = CONTROL.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise RuntimeError("Frozen CONTROL65 hash mismatch")
    source = data.decode("utf-8")
    prefix = source[source.index("from __future__"):source.index("class Estimator(")]
    start = source.index("            var_pre = fnp.diag(cov_pre)")
    stop = source.index("            # 5. Closed-form exact Step 5", start)
    nonlinear = "\n".join(line[12:] if line.startswith(" " * 12) else line
                              for line in source[start:stop].splitlines())
    start = source.index("            # Section 7 exact k22 sum")
    stop = source.index("            preds.append(mu)", start)
    scalar = "\n".join(line[12:] if line.startswith(" " * 12) else line
                          for line in source[start:stop].splitlines())
    body = nonlinear + "\ncov = _sym(k11)\nfnp.fill_diagonal(cov, k2)\n" + scalar
    body += "\nreturn k1, cov, k3, k21, w11, w21, c4\n"
    function = "\ndef _nonlinear(mu_pre, cov_pre, s3, s21, s4, s22):\n"
    function += "    n = mu_pre.shape[0]\n" + textwrap.indent(body, "    ")
    header = '"""Fixed rank-64 DS/Tucker experiment; generated by build_p7_hybrid.py."""\n'
    OUTPUT.write_text(header + prefix + function + HELPERS, encoding="utf-8")
    print(OUTPUT)
    print(hashlib.sha256(OUTPUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
