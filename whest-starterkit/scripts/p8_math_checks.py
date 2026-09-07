"""p8_math_checks.py

Phase 8 Mathematical Identities & Primitives Verification Suite.
Supports:
    uv run python scripts/p8_math_checks.py --lane angular
    uv run python scripts/p8_math_checks.py --lane direct
    uv run python scripts/p8_math_checks.py --lane gate
    uv run python scripts/p8_math_checks.py --lane all

Enforces Section 5.2 tolerances (dense float64: absolute 1e-10, relative 1e-9).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.p8_angular import verify_angular_identities, save_ap_constants
from scripts.p8_direct_sources import verify_d0_identities, forecast_direct_sources_cost
from scripts.p8_gate_residuals import verify_g0_identities


def run_angular_checks() -> bool:
    print("\n=======================================================")
    print("PHASE 8 MATH CHECKS: LANE A (ANGULAR MOMENTS)")
    print("=======================================================")
    save_ap_constants()
    passed_all = True
    for n in [3, 5, 8]:
        res = verify_angular_identities(n=n, n_samples=100_000, seed=8101 + n)
        homog = res["depth16_forward_homogeneity"]
        print(f"[n={n}] Depth-16 Forward Homogeneity: rel_err={homog['rel_err']:.4e} | Passes: {homog['passes']}")
        if not homog["passes"]:
            passed_all = False

        cov_err = res["angular_input_cov_err"]
        k4_err = res["k4_iiii_abs_err"]
        print(f"[n={n}] Angular Input Cov Err: {cov_err:.4e} | K4_iiii Abs Err: {k4_err:.4e}")

        for k, v in res["radius_factorization"].items():
            if not v["within_3se"]:
                print(f"  Warning: {k} outside 3.5 SE (rel_err={v['rel_err']:.4e})")

    print(f"\nLane A Status: {'ALL PASSED' if passed_all else 'FAILED'}")
    return passed_all


def run_direct_checks() -> bool:
    print("\n=======================================================")
    print("PHASE 8 MATH CHECKS: LANE D (DIRECT K3 CONTRACTION)")
    print("=======================================================")
    passed_all = True
    for n in [3, 5, 8]:
        for depth in [2, 4]:
            res = verify_d0_identities(n=n, depth=depth, seed=8201 + n * 10 + depth)
            bf_pass = res["backward_vs_forward_passes"]
            cs_pass = res["concat_vs_separate_passes"]
            dd_pass = res["dense_tensor_diag_passes"]
            print(f"[n={n}, depth={depth}] Bwd vs Fwd: {res['backward_vs_forward_rel_err']:.4e} (pass={bf_pass}) | "
                  f"Concat vs Sep: {res['concat_vs_separate_rel_err']:.4e} (pass={cs_pass}) | "
                  f"Dense Diag: {res['dense_tensor_diag_rel_err']:.4e} (pass={dd_pass})")
            if not (bf_pass and cs_pass and dd_pass):
                passed_all = False

    cost = forecast_direct_sources_cost(n=1024, depth=16)
    print(f"\nPhase 2 Cost Forecast:")
    print(f"  D1-TERM: {cost['d1_term_flops']:.4e} FLOPs ({cost['d1_term_util']*100:.2f}%) Passes < 0.95B: {cost['d1_term_passes_cost_gate']}")
    print(f"  D2-ALL:  {cost['d2_all_flops']:.4e} FLOPs ({cost['d2_all_util']*100:.2f}%) Passes < 0.95B: {cost['d2_all_passes_cost_gate']}")

    print(f"\nLane D Status: {'ALL PASSED' if passed_all else 'FAILED'}")
    return passed_all


def run_gate_checks() -> bool:
    print("\n=======================================================")
    print("PHASE 8 MATH CHECKS: LANE G (GATE-RESIDUAL DECOMPOSITION)")
    print("=======================================================")
    passed_all = True
    for n in [3, 5, 8]:
        for depth in [2, 4]:
            res = verify_g0_identities(n=n, depth=depth, n_samples=100_000, seed=8301 + n * 10 + depth)
            id_pass = res["samplewise_passes"]
            print(f"[n={n}, depth={depth}] Samplewise Identity rel_err: {res['samplewise_rel_err']:.4e} | Passes: {id_pass}")
            if not id_pass:
                passed_all = False

    print(f"\nLane G Status: {'ALL PASSED' if passed_all else 'FAILED'}")
    return passed_all


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Mathematical Identity Verification")
    parser.add_argument("--lane", choices=["angular", "direct", "gate", "all"], default="all")
    args = parser.parse_args()

    success = True
    if args.lane in ("angular", "all"):
        if not run_angular_checks():
            success = False
    if args.lane in ("direct", "all"):
        if not run_direct_checks():
            success = False
    if args.lane in ("gate", "all"):
        if not run_gate_checks():
            success = False

    if success:
        print("\n>>> ALL PHASE 8 MATHEMATICAL CHECKS PASSED SUCCESSFULLY <<<")
        sys.exit(0)
    else:
        print("\n>>> ONE OR MORE PHASE 8 MATHEMATICAL CHECKS FAILED <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
