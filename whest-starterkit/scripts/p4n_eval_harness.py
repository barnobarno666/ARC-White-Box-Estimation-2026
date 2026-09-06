"""P4N Rigorous Evaluation Harness and Gate Checker.

Features:
- Executes candidates using official `whest run` with explicit dataset path and `--n-mlps 8`.
- Computes exact scored error: S = mean_m [ e_m * max(0.10, u_m) ].
- Matches MLPs by identity/name, not list position alone.
- Replays against immutable control receipt `scripts/champion_8mlp.json`.
- Saves parsed results to `research/phase4_next/results/{exp_id}_{timestamp}.json`.
- Evaluates complete conjunction of promotion gates:
  1. Panel adjusted score improvement >= 10% vs control (< 1.101278e-07).
  2. Wins vs control >= 6/8.
  3. Max per-MLP regression <= 10%.
  4. Max residual wall time < 0.35s (prefer < 0.30s).
  5. Failures == 0.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CONTROL_RECEIPT_PATH = Path("scripts/champion_8mlp.json")
RESULTS_DIR = Path("research/phase4_next/results")
DATASET_PATH = r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini"


def compute_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_candidate(
    estimator_path: str | Path,
    exp_id: str,
    n_mlps: int = 8,
    offset: int = 0,
    save_receipt: bool = True,
) -> Dict[str, Any] | None:
    estimator_path = Path(estimator_path)
    if not estimator_path.exists():
        print(f"Error: Estimator file {estimator_path} does not exist.")
        return None

    cand_sha = compute_sha256(estimator_path)
    print(f"\n=======================================================")
    print(f"Running P4N Evaluation: Exp ID [{exp_id}]")
    print(f"Estimator: {estimator_path} (SHA: {cand_sha[:16]}...)")
    print(f"Panel: {n_mlps} MLPs | Sampler Offset: {offset}")
    print(f"=======================================================")

    cmd = [
        "uv", "run", "whest", "run",
        "--estimator", str(estimator_path),
        "--dataset", DATASET_PATH,
        "--split", "mini",
        "--n-mlps", str(n_mlps),
        "--runner", "local",
        "--format", "json",
    ]
    if offset != 0:
        cmd.extend(["--seed-offset", str(offset)])

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"Execution FAILED with return code {proc.returncode}")
        print("STDERR:")
        print(proc.stderr)
        return None

    stdout = proc.stdout
    json_start = stdout.find("{")
    if json_start == -1:
        print("Could not find JSON output in stdout:")
        print(stdout)
        return None

    try:
        raw_data = json.loads(stdout[json_start:])
    except Exception as e:
        print(f"JSON parsing error: {e}")
        print(stdout[json_start:json_start+500])
        return None

    results = raw_data["results"]
    per_mlp = results["per_mlp"]

    # Compute exact S = mean_m [ e_m * max(0.10, u_m) ]
    scored_errors = []
    max_res_time = 0.0
    for m in per_mlp:
        e_m = m["final_layer_mse"]
        u_m = m.get("compute_utilization", m.get("effective_compute", 0.0) / (2**41))
        mult_m = max(0.10, u_m)
        s_m = e_m * mult_m
        m["calculated_adjusted_score"] = s_m
        scored_errors.append(s_m)
        res_t = m.get("residual_wall_time_s", 0.0)
        if res_t > max_res_time:
            max_res_time = res_t

    calculated_mean_score = sum(scored_errors) / len(scored_errors)
    official_score = results.get("adjusted_final_layer_score", calculated_mean_score)
    raw_mse = results.get("final_layer_mse", 0.0)
    mean_util = results.get("mean_compute_utilization", 0.0)
    failures = results.get("n_failed_mlps", 0)

    print(f"\n[Summary Results]")
    print(f"  Adjusted Score (Exact):    {calculated_mean_score:.6e} (Official: {official_score:.6e})")
    print(f"  Raw Final MSE:             {raw_mse:.6e}")
    print(f"  Mean Compute Utilization:  {mean_util * 100:.2f}% (Mult: {max(0.10, mean_util):.4f})")
    print(f"  Max Residual Wall Time:    {max_res_time:.4f}s")
    print(f"  Failures:                  {failures}")

    # Compare against control
    control_data = None
    if CONTROL_RECEIPT_PATH.exists():
        with open(CONTROL_RECEIPT_PATH, "r") as f:
            c_json = json.load(f)
            control_data = c_json.get("results", c_json)

    wins = losses = ties = 0
    max_regress_pct = -999.0
    control_score = 1.223643e-07

    if control_data and "per_mlp" in control_data:
        c_by_name = {m["mlp_name"]: m for m in control_data["per_mlp"]}
        control_score = control_data.get("adjusted_final_layer_score", 1.223643e-07)
        rel_improvement = ((calculated_mean_score - control_score) / control_score) * 100.0

        print(f"\n[Comparison vs Frozen Control ({control_score:.6e})]")
        print(f"  Relative Score Diff:       {rel_improvement:+.2f}%")
        print("\n  Per-MLP Breakdown:")
        print("  " + "-" * 75)
        print(f"  {'MLP Name':<24} | {'Candidate':<12} | {'Control':<12} | {'Diff (%)':<9} | {'W/L'}")
        print("  " + "-" * 75)

        for m in per_mlp:
            name = m["mlp_name"]
            cand_s = m["calculated_adjusted_score"]
            c_m = c_by_name.get(name)
            if c_m:
                c_s = c_m.get("adjusted_final_layer_score", c_m["final_layer_mse"] * 0.10)
                diff_pct = ((cand_s - c_s) / c_s) * 100.0
                if diff_pct > max_regress_pct:
                    max_regress_pct = diff_pct
                if cand_s < c_s:
                    outcome = "WIN"
                    wins += 1
                elif cand_s > c_s:
                    outcome = "LOSS"
                    losses += 1
                else:
                    outcome = "TIE"
                    ties += 1
                print(f"  {name:<24} | {cand_s:.6e} | {c_s:.6e} | {diff_pct:+8.2f}% | {outcome}")
            else:
                print(f"  {name:<24} | {cand_s:.6e} | [No control match]")

        print("  " + "-" * 75)
        print(f"  Head-to-Head Record:       {wins}W - {losses}L - {ties}T")
        print(f"  Worst MLP Regression:      {max_regress_pct:+.2f}%")

    # Gate verification
    gate_10pct = (calculated_mean_score <= control_score * 0.90)
    gate_wins = (wins >= 6)
    gate_regress = (max_regress_pct <= 10.0)
    gate_resid = (max_res_time < 0.35)
    gate_failures = (failures == 0)

    promotion_pass = all([gate_10pct, gate_wins, gate_regress, gate_resid, gate_failures])

    print(f"\n[Promotion Gate Conjunction Check]")
    print(f"  [1] Mean Score >= 10% Gain (< {control_score * 0.90:.6e}): {'PASS' if gate_10pct else 'FAIL'} ({calculated_mean_score:.6e})")
    print(f"  [2] Wins >= 6/8:                                         {'PASS' if gate_wins else 'FAIL'} ({wins}/8)")
    print(f"  [3] Worst Regression <= 10%:                             {'PASS' if gate_regress else 'FAIL'} ({max_regress_pct:+.2f}%)")
    print(f"  [4] Residual Wall Time < 0.35s:                         {'PASS' if gate_resid else 'FAIL'} ({max_res_time:.4f}s)")
    print(f"  [5] Zero Failures:                                       {'PASS' if gate_failures else 'FAIL'} ({failures})")
    print(f"  --> OVERALL PROMOTION DECISION: {'*** PROMOTED ***' if promotion_pass else 'REJECT PROMOTION'}")

    receipt = {
        "exp_id": exp_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "estimator_path": str(estimator_path),
        "estimator_sha256": cand_sha,
        "offset": offset,
        "n_mlps": n_mlps,
        "calculated_adjusted_score": calculated_mean_score,
        "official_adjusted_score": official_score,
        "raw_final_mse": raw_mse,
        "mean_compute_utilization": mean_util,
        "max_residual_wall_time_s": max_res_time,
        "n_failed_mlps": failures,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "max_regression_pct": max_regress_pct,
        "relative_improvement_pct": ((calculated_mean_score - control_score) / control_score) * 100.0,
        "promotion_pass": promotion_pass,
        "per_mlp": per_mlp,
    }

    if save_receipt:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts_slug = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = RESULTS_DIR / f"{exp_id}_{ts_slug}.json"
        with open(out_file, "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"\nSaved complete receipt to: {out_file}")

    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4N Evaluation Harness")
    parser.add_argument("estimator", help="Path to estimator candidate")
    parser.add_argument("exp_id", help="Experiment ID (e.g. P4N-00, P4N-01)")
    parser.add_argument("--n-mlps", type=int, default=8, help="Number of MLPs (default: 8)")
    parser.add_argument("--offset", type=int, default=0, help="Sampler seed offset (default: 0)")
    args = parser.parse_args()

    run_candidate(args.estimator, args.exp_id, n_mlps=args.n_mlps, offset=args.offset)
