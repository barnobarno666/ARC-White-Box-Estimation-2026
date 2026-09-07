"""p8_select.py

Phase 8 Candidate Selection, Leave-One-Out (LOO) Analysis, and Frontier Selection.
Evaluates candidates from ledger.jsonl against CONTROL7 pursuant to Phase 8 Plan Section 16:
1. Candidate A: Lowest eligible mean projected product / adjusted score.
2. Candidate B: Best eligible candidate from distinct mechanism (if within 1.25*A).
3. LOO network stability across the 8-MLP dev panel.
4. Paired bootstrap statistics on Confirmation Panel (conf12).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
P8_DIR = REPO_ROOT / "research" / "phase8"
LEDGER_PATH = P8_DIR / "ledger.jsonl"
PROGRESS_PATH = P8_DIR / "progress.json"

CONTROL7_RAW_MSE = 3.785300961567373e-08
CONTROL7_ADJUSTED = 2.633706248957546e-08
CONTROL7_UTIL = 0.6957719546471708


def load_completed_candidates() -> List[Dict[str, Any]]:
    if not LEDGER_PATH.exists():
        return []
    records = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("captured_metrics"):
                    records.append(r)
            except Exception:
                pass
    return records


def run_dev8_selection(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select finalist candidates on dev8 panel."""
    dev_records = [
        r for r in records
        if r.get("panel") == "dev8" and r.get("captured_metrics", {}).get("n_mlps") == 8
    ]
    if not dev_records:
        return {"error": "No dev8 records found"}

    c7_record = next((r for r in dev_records if r["exp_id"] == "CONTROL7"), None)
    if not c7_record:
        c7_scores = [CONTROL7_ADJUSTED] * 8
    else:
        c7_scores = c7_record["captured_metrics"]["adjusted_scores"]

    summary = []
    for r in dev_records:
        exp_id = r["exp_id"]
        metrics = r["captured_metrics"]
        cand_scores = metrics["adjusted_scores"]
        mean_adj = metrics["mean_adjusted_score"]
        mean_raw = metrics["mean_raw_final_mse"]
        util = metrics["mean_compute_utilization"]

        ratios = [c / base for c, base in zip(cand_scores, c7_scores)]
        wins = sum(1 for r_i in ratios if r_i < 1.0)
        worst_ratio = max(ratios)
        imp = (1.0 - mean_adj / CONTROL7_ADJUSTED) * 100.0

        summary.append({
            "exp_id": exp_id,
            "mean_adjusted_score": mean_adj,
            "mean_raw_final_mse": mean_raw,
            "mean_compute_utilization": util,
            "wins_vs_c7": f"{wins}/8",
            "worst_ratio": worst_ratio,
            "improvement_pct": imp,
            "candidate_file": r.get("candidate_file", ""),
            "candidate_sha256": r.get("candidate_sha256", ""),
        })

    # Sort by lowest mean adjusted score
    summary.sort(key=lambda x: x["mean_adjusted_score"])

    # Best candidate A
    cand_a = summary[0]

    # Candidate B from different mechanism
    # Mechanisms: angular ('A1-A' or 'ANGULAR'), direct ('D1-TERM'), strassen-angular ('X-STRASSEN')
    cand_b = None
    for item in summary[1:]:
        # Check mechanism differentiation
        is_diff = False
        if "STRASSEN" in cand_a["exp_id"] and "STRASSEN" not in item["exp_id"]:
            is_diff = True
        elif "A1-A" in cand_a["exp_id"] and "A1-A" not in item["exp_id"]:
            is_diff = True
        
        if is_diff and item["mean_adjusted_score"] <= 1.25 * cand_a["mean_adjusted_score"]:
            cand_b = item
            break

    return {
        "ranked_candidates": summary,
        "candidate_a": cand_a,
        "candidate_b": cand_b,
    }


def run_loo_analysis(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Leave-one-out network analysis across 8 dev MLPs."""
    dev_records = [
        r for r in records
        if r.get("panel") == "dev8" and r.get("captured_metrics", {}).get("n_mlps") == 8
        and r.get("exp_id") in ["A1-G-K3C65", "A1-A-K3C65", "X-STRASSEN-ANGULAR", "CONTROL7"]
    ]
    if not dev_records:
        return {}

    loo_winners = []
    for held_out in range(8):
        best_exp = None
        best_score = float("inf")
        for r in dev_records:
            scores = r["captured_metrics"]["adjusted_scores"]
            train_scores = [scores[i] for i in range(8) if i != held_out]
            mean_train = float(np.mean(train_scores))
            if mean_train < best_score:
                best_score = mean_train
                best_exp = r["exp_id"]
        loo_winners.append({"held_out": held_out, "winner": best_exp, "score": best_score})

    return {"loo_folds": loo_winners}


def run_confirmation_analysis(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate Conf-12 panel results with bootstrap confidence interval."""
    conf_c7 = next((r for r in records if r["exp_id"] == "CONTROL7-CONF12"), None)
    conf_f8 = next((r for r in records if r["exp_id"] == "FINAL8-CONF12"), None)

    if not conf_c7 or not conf_f8:
        return {"error": "Confirmation panel records missing"}

    c7_scores = np.array(conf_c7["captured_metrics"]["adjusted_scores"])
    f8_scores = np.array(conf_f8["captured_metrics"]["adjusted_scores"])
    n = len(c7_scores)

    diffs = f8_scores - c7_scores
    ratios = f8_scores / c7_scores
    wins = int(np.sum(ratios < 1.0))

    # Paired bootstrap (2000 resamples, seed 8199 per Section 16)
    rng = np.random.RandomState(8199)
    boot_diffs = []
    boot_ratios = []
    for _ in range(2000):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs.append(np.mean(diffs[idx]))
        boot_ratios.append(np.mean(f8_scores[idx]) / np.mean(c7_scores[idx]))

    ci_diff = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]
    ci_ratio = [float(np.percentile(boot_ratios, 2.5)), float(np.percentile(boot_ratios, 97.5))]

    return {
        "n_mlps": n,
        "c7_mean_adjusted": float(np.mean(c7_scores)),
        "final8_mean_adjusted": float(np.mean(f8_scores)),
        "c7_mean_raw_mse": conf_c7["captured_metrics"]["mean_raw_final_mse"],
        "final8_mean_raw_mse": conf_f8["captured_metrics"]["mean_raw_final_mse"],
        "utilization": conf_f8["captured_metrics"]["mean_compute_utilization"],
        "wins": f"{wins}/{n}",
        "mean_ratio": float(np.mean(ratios)),
        "ratio_95_ci": ci_ratio,
        "mean_diff": float(np.mean(diffs)),
        "diff_95_ci": ci_diff,
        "confirmed": wins >= 11 and ci_ratio[1] < 1.0,
    }


def main():
    records = load_completed_candidates()
    print(f"Loaded {len(records)} ledger records.")

    dev_res = run_dev8_selection(records)
    print("\n=== DEV-8 SELECTION SUMMARY ===")
    for c in dev_res.get("ranked_candidates", []):
        print(f"  {c['exp_id']:<22} MSE={c['mean_raw_final_mse']:.4e} Util={c['mean_compute_utilization']*100:.1f}% Adj={c['mean_adjusted_score']:.4e} Wins={c['wins_vs_c7']} Imp={c['improvement_pct']:+.2f}%")

    if dev_res.get("candidate_a"):
        print(f"\nCandidate A: {dev_res['candidate_a']['exp_id']}")
    if dev_res.get("candidate_b"):
        print(f"Candidate B: {dev_res['candidate_b']['exp_id']}")

    loo = run_loo_analysis(records)
    print("\n=== LOO STABILITY ===")
    for fold in loo.get("loo_folds", []):
        print(f"  Fold {fold['held_out']}: Winner = {fold['winner']}")

    conf = run_confirmation_analysis(records)
    if "error" not in conf:
        print("\n=== CONFIRMATION PANEL (12 MLPs) ===")
        print(f"  CONTROL7 Score:  {conf['c7_mean_adjusted']:.6e} (Raw MSE: {conf['c7_mean_raw_mse']:.6e})")
        print(f"  FINAL8 Score:    {conf['final8_mean_adjusted']:.6e} (Raw MSE: {conf['final8_mean_raw_mse']:.6e})")
        print(f"  Wins:            {conf['wins']} (Win Rate: {int(conf['wins'].split('/')[0])/conf['n_mlps']*100:.1f}%)")
        print(f"  Score Ratio:     {conf['mean_ratio']:.4f} [95% CI: {conf['ratio_95_ci'][0]:.4f}, {conf['ratio_95_ci'][1]:.4f}]")
        print(f"  Confirmed:       {conf['confirmed']}")


if __name__ == "__main__":
    main()
