"""Independent offline dense-tensor and no-truncation recurrence checks."""
from pathlib import Path
import hashlib
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "whest-starterkit/research/p7_hybrid_single"


def atoms(U, V):
    return (np.einsum("ir,jr,kr->ijk", U, U, V)
            + np.einsum("ir,jr,kr->ijk", U, V, U)
            + np.einsum("ir,jr,kr->ijk", V, U, U)) / 3


def dense_core(Q, G):
    return np.einsum("ia,jb,kc,abc->ijk", Q, Q, Q, G)


def slices(T):
    d = np.einsum("iii->i", T)
    S = np.einsum("iij->ij", T).copy()
    np.fill_diagonal(S, 0)
    return d, S


def main():
    candidate = ROOT / "candidates/estimator_p7_hybrid_k64.py"
    spec = importlib.util.spec_from_file_location("hybrid", candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only this offline checker replaces numerical APIs, never the candidate file.
    numeric = SimpleNamespace(**{name: getattr(np, name) for name in dir(np)})
    numeric.float32 = np.float64
    module.fnp = numeric
    module.flops = SimpleNamespace(stats=stats)
    rng = np.random.default_rng(7101)
    errors = {}

    def check(name, actual, expected, tolerance=1e-10):
        error = float(np.max(np.abs(actual - expected)))
        errors[name] = error
        assert error <= tolerance * max(1.0, float(np.max(np.abs(expected)))), (name, error)

    n, r, k = 5, 4, 3
    U, V = rng.normal(size=(n, r)), rng.normal(size=(n, r))
    Q = np.linalg.qr(rng.normal(size=(n, k)))[0]
    G = module._sym_core(rng.normal(size=(k, k, k)))
    Z1, Z2 = rng.normal(size=(n, 7)), rng.normal(size=(n, 7))
    T, Tc = atoms(U, V), dense_core(Q, G)
    d, S = slices(T)
    D = np.diag(d) + 3 * S.T
    Tds = atoms(np.eye(n), D)
    check("factor_diag", module._factor_slices(U, V)[0], d)
    check("factor_repeat", module._factor_slices(U, V)[1], S)
    check("factor_contract2", module._factor_contract2(U, V, Z1, Z2),
          np.einsum("it,jt,ijk->kt", Z1, Z2, T))
    check("ds_contract2", module._ds_contract2(D, Z1, Z2),
          np.einsum("it,jt,ijk->kt", Z1, Z2, Tds))
    check("core_contract2", module._core_contract2(Q, G, Z1, Z2),
          np.einsum("it,jt,ijk->kt", Z1, Z2, Tc))
    check("core_repeat", module._core_slices(Q, G)[1], slices(Tc)[1])
    P = np.linalg.qr(rng.normal(size=(n, 2)))[0]
    def projection(T):
        return np.einsum("ia,jb,kc,ijk->abc", P, P, P, T)
    check("project_factors", module._project_factors(P, U, V), projection(T))
    check("project_ds", module._project_ds(P, D), projection(Tds))
    check("project_core", module._project_core(P, Q, G), projection(Tc))
    total = T + Tc
    dt, St = slices(total)
    for rank in (2, n):
        module.CORE_RANK = rank
        Pn, Gn, Dn = module._compress_old(U, V, Q, G, dt, St, rng)
        result = dense_core(Pn, Gn) + atoms(np.eye(n), Dn)
        check(f"hybrid_r{rank}_diag", slices(result)[0], dt)
        check(f"hybrid_r{rank}_repeat", slices(result)[1], St)
        if rank == n:
            check("hybrid_full_basis", result, total)

    baseline_path = ROOT / "candidates/estimator_p65_final.py"
    baseline_source = baseline_path.read_text(encoding="utf-8")
    needle = "comp_layers = {10, 13, 6}"
    assert baseline_source.count(needle) == 1
    baseline_source = baseline_source.replace(needle, "comp_layers = set()")
    baseline = {"__name__": "offline_unpruned_control"}
    exec(compile(baseline_source, str(baseline_path), "exec"), baseline)
    baseline["fnp"] = numeric
    baseline["flops"] = SimpleNamespace(stats=stats)
    module.CORE_RANK = 64
    for n, depth in ((5, 1), (8, 2), (8, 8), (16, 16)):
        weights = [rng.normal(size=(n, n)) * np.sqrt(2 / n) for _ in range(depth)]
        mlp = SimpleNamespace(width=n, depth=depth, weights=weights, seed=7101)
        pred = module.Estimator().predict(mlp, 2**41)
        expected = baseline["Estimator"]().predict(mlp, 2**41)
        assert pred.shape == (depth, n) and np.isfinite(pred).all()
        # The production candidate is float32 and draws an online range basis;
        # a full-rank recurrence is therefore an approximate parity diagnostic.
        # Require a bounded finite result, while exact tensor identities above
        # carry the strict algebraic tolerance.
        drift = float(np.max(np.abs(pred - expected)))
        errors[f"full_basis_recurrence_n{n}_d{depth}"] = drift
        assert np.isfinite(pred).all() and drift < 5e-6, (n, depth, drift)

    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "PASSED", "scope": "offline float64 dense identities and full-basis recurrence; not metered production",
               "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(), "errors": errors}
    (OUT / "math_checks.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
