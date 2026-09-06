"""p65_summarize.py

Phase 6.5 Summarizer and Decision Engine.
Reads ledger.jsonl and predictions, generates head-to-head comparisons,
leave-one-out cross validation, and updates phase6.5report.md & Current thing.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
P65_DIR = REPO_ROOT / "research" / "phase6_5"
RESULTS_DIR = P65_DIR / "results"
LEDGER_PATH = P65_DIR / "ledger.jsonl"
REPORT_PATH = WORKSPACE_ROOT / "phase6.5report.md"
CURRENT_THING_PATH = WORKSPACE_ROOT / "Current thing.md"
GT_PATH = P65_DIR / "fixtures" / "ground_truth.npy"

PANEL_MLP_NAMES = [
    "logan-fitzgerald",
    "william-graves",
    "raymond-barnes",
    "steven-rice",
    "sarah-kelley",
    "christopher-morales",
    "cheryl-graham",
    "renee-park",
]


def load_ledger() -> List[Dict[str, Any]]:
    if not LEDGER_PATH.exists():
        return []
    records = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def get_latest_receipt(exp_id: str) -> Optional[Dict[str, Any]]:
    records = load_ledger()
    for r in reversed(records):
        if r.get("exp_id") == exp_id:
            return r
    return None


def compare_candidates(
    cand_receipt: Dict[str, Any],
    parent_receipt: Dict[str, Any],
) -> Dict[str, Any]:
    cand_per = (cand_receipt.get("captured_metrics") or {}).get("per_mlp", [])
    parent_per = (parent_receipt.get("captured_metrics") or {}).get("per_mlp", [])
    n_mlps = min(len(cand_per), len(parent_per))

    wins = 0
    losses = 0
    ties = 0
    score_ratios = []
    raw_ratios = []
    comparisons = []

    for i in range(n_mlps):
        c_score = cand_per[i]["adjusted_score"]
        p_score = parent_per[i]["adjusted_score"]
        c_raw = cand_per[i]["final_layer_mse"]
        p_raw = parent_per[i]["final_layer_mse"]

        ratio = c_score / p_score if p_score > 0 else 1.0
        score_ratios.append(ratio)
        raw_ratios.append(c_raw / p_raw if p_raw > 0 else 1.0)

        if c_score < p_score - 1e-15:
            wins += 1
            decision = "WIN"
        elif c_score > p_score + 1e-15:
            losses += 1
            decision = "LOSS"
        else:
            ties += 1
            decision = "TIE"

        comparisons.append({
            "mlp_name": PANEL_MLP_NAMES[i] if i < len(PANEL_MLP_NAMES) else f"mlp_{i}",
            "cand_score": c_score,
            "parent_score": p_score,
            "ratio": ratio,
            "decision": decision,
        })

    cand_mean_score = cand_receipt.get("captured_metrics", {}).get("mean_adjusted_score", 0.0)
    parent_mean_score = parent_receipt.get("captured_metrics", {}).get("mean_adjusted_score", 0.0)
    overall_ratio = cand_mean_score / parent_mean_score if parent_mean_score > 0 else 1.0
    worst_regression = max(score_ratios) if score_ratios else 1.0

    return {
        "n_mlps": n_mlps,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "overall_score_ratio": overall_ratio,
        "mean_improvement_pct": (1.0 - overall_ratio) * 100.0,
        "worst_regression_ratio": worst_regression,
        "per_mlp": comparisons,
    }


def update_current_thing(
    method_name: str,
    short_desc: str,
    raw_final_mse: float,
    mean_mult: float,
    adjusted_score: float,
    max_res_time: float,
    failures: int,
    wins_summary: str,
    decision: str,
) -> None:
    """Append a validated row to Current thing.md keeping the table clean and separate."""
    if not CURRENT_THING_PATH.exists():
        return

    content = CURRENT_THING_PATH.read_text(encoding="utf-8")

    # Format new row
    new_row = (
        f"| {method_name} | {short_desc} | {raw_final_mse:.4e} | "
        f"{mean_mult:.4f} | {adjusted_score:.4e} | {max_res_time:.4f}s | "
        f"{failures} | {wins_summary} | {decision} |\n"
    )

    # Check if section exists
    p65_header = "## Phase 6.5 Sequential Runbook Validation Results"
    if p65_header not in content:
        section = f"\n\n{p65_header}\n\n"
        section += "| Method / Variant | Short Description | Raw Final MSE | Mean Score Mult | Adjusted Score | Max Residual Time | Failures | Wins vs Incumbent | Decision |\n"
        section += "|---|---|---|---|---|---|---|---|---|\n"
        section += new_row
        content += section
    else:
        # Append row right before the next section or at end
        content += new_row

    CURRENT_THING_PATH.write_text(content, encoding="utf-8")
    print(f"[p65_summarize] Updated Current thing.md with row for {method_name}")


def append_to_report(
    cand_receipt: Dict[str, Any],
    comp_incumbent: Optional[Dict[str, Any]] = None,
    comp_parent: Optional[Dict[str, Any]] = None,
    decision: str = "RECORDED",
) -> None:
    """Append experiment entry to phase6.5report.md."""
    if not REPORT_PATH.exists():
        return

    exp_id = cand_receipt.get("exp_id")
    cm = cand_receipt.get("captured_metrics", {})
    if not cm:
        return

    raw_mse = cm.get("mean_raw_final_mse", 0.0)
    util = cm.get("mean_compute_utilization", 0.0)
    adj = cm.get("mean_adjusted_score", 0.0)
    status = cand_receipt.get("status", "RESEARCH_ONLY")

    wins_str = "-"
    if comp_parent:
        wins_str = f"{comp_parent.get('wins', 0)}W-{comp_parent.get('losses', 0)}L"

    notes = f"Diff vs Parent: {comp_parent.get('mean_improvement_pct', 0.0):+.2f}%" if comp_parent else "Baseline"

    table_row = f"| `{exp_id}` | `{cand_receipt.get('parent_sha256', 'none')[:8]}` | {cand_receipt.get('candidate_file', '').split('/')[-1]} | {raw_mse:.4e} | {util*100:.2f}% | {adj:.4e} | {wins_str} | `{status}` | {notes} |\n"

    content = REPORT_PATH.read_text(encoding="utf-8")
    content += table_row
    REPORT_PATH.write_text(content, encoding="utf-8")
    print(f"[p65_summarize] Appended {exp_id} to phase6.5report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=str, required=True, help="Experiment ID to summarize")
    parser.add_argument("--parent-id", type=str, default="P65-00-incumbent", help="Parent experiment ID")
    parser.add_argument("--incumbent-id", type=str, default="P65-00-incumbent", help="Incumbent experiment ID")
    args = parser.parse_args()

    cand = get_latest_receipt(args.exp_id)
    if not cand:
        print(f"Error: Receipt for {args.exp_id} not found in ledger.")
        sys.exit(1)

    parent = get_latest_receipt(args.parent_id)
    comp_parent = compare_candidates(cand, parent) if parent else None

    incumbent = get_latest_receipt(args.incumbent_id)
    comp_incumbent = compare_candidates(cand, incumbent) if incumbent else None

    print("\n=======================================================")
    print(f"Summary for: {args.exp_id}")
    cm = cand.get("captured_metrics", {})
    if cm:
        print(f"  Raw Final MSE:   {cm['mean_raw_final_mse']:.6e}")
        print(f"  Compute Util:    {cm['mean_compute_utilization']*100:.2f}%")
        print(f"  Adjusted Score:  {cm['mean_adjusted_score']:.6e}")
    if comp_parent:
        print(f"  vs Parent ({args.parent_id}):")
        print(f"    Record:        {comp_parent['wins']}W - {comp_parent['losses']}L - {comp_parent['ties']}T")
        print(f"    Ratio:         {comp_parent['overall_score_ratio']:.4f} ({comp_parent['mean_improvement_pct']:+.2f}%)")
        print(f"    Worst Reg:     {comp_parent['worst_regression_ratio']:.4f}")
    print("=======================================================")
