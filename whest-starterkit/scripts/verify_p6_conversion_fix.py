"""Offline algebra and numerical audit; never included in the submission."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "whest-starterkit/research/phase6_conversion_fix"
OLD = ROOT / "candidates/estimator_p6_k3_6_10_13.py"
NEW = ROOT / "candidates/estimator_p6_k3_twofactor_terminal.py"


def estimator(path, precision):
    source = path.read_text()
    if precision == "float64":
        source = source.replace("fnp.float32", "fnp.float64")
    scope = {"__name__": "parity_fixture"}
    exec(compile(source, str(path), "exec"), scope)
    return scope["Estimator"]()


def main():
    warnings.filterwarnings("ignore")
    rows = []
    for precision in ("float64", "float32"):
        old, new = estimator(OLD, precision), estimator(NEW, precision)
        for n, depth in ((8, 1), (16, 2), (32, 8), (32, 16)):
            rng = np.random.default_rng(29)
            mlp = SimpleNamespace(width=n, depth=depth, weights=[
                (rng.normal(size=(n, n)) * np.sqrt(2/n)).astype("float32")
                for _ in range(depth)])
            a = np.asarray(old.predict(mlp, 2**41))
            b = np.asarray(new.predict(mlp, 2**41))
            assert np.isfinite(b).all()
            row = dict(precision=precision, width=n, depth=depth,
                       max_abs=float(np.max(np.abs(a-b))),
                       rms=float(np.sqrt(np.mean((a-b)**2))))
            rows.append(row)
            print(row, flush=True)
            # Float64 checks the algebra; float32 differences are measured and
            # separately judged on the actual fixed eight-network panel.
            if precision == "float64":
                np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9)
    OUT.mkdir(exist_ok=True)
    result = {"source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (OLD, NEW)}, "tiny_parity": rows,
              "note": "Initial float32 2e-5 check failed on width32/depth16; float64 isolates algebra from rounding. Full-size panel required."}
    (OUT / "tiny_parity.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
