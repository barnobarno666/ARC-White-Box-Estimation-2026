"""p65_run_plan.py

Phase 6.5 Master Sequential Orchestrator.
Supports:
  --resume: Resumes execution from last completed step in progress.json
  --stage STAGE: Executes a specific stage (e.g. P65-02, P65-03)
Sequentially executes stages, enforces selection and promotion gates,
and maintains progress.json, ledger.jsonl, and phase6.5report.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
P65_DIR = REPO_ROOT / "research" / "phase6_5"
PROGRESS_PATH = P65_DIR / "progress.json"
LEDGER_PATH = P65_DIR / "ledger.jsonl"
REPORT_PATH = WORKSPACE_ROOT / "phase6.5report.md"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.p65_build_candidate import build_candidate, format_exp_filename
from scripts.p65_eval import run_candidate_eval, sha256_file
from scripts.p65_math_checks import run_math_checks
from scripts.p65_summarize import (
    append_to_report,
    compare_candidates,
    get_latest_receipt,
    update_current_thing,
)


def load_progress() -> Dict[str, Any]:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress: Dict[str, Any]) -> None:
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def is_step_completed(exp_id: str) -> bool:
    receipt = get_latest_receipt(exp_id)
    return receipt is not None and receipt.get("status") in ("VALIDATED", "RESEARCH_ONLY", "PARKED_EXACT_PARITY")


def run_stage_p65_02(progress: Dict[str, Any], resume: bool = True) -> str:
    """Execute Stage P65-02: Exact structural optimizations."""
    print("\n=======================================================")
    print("STAGE P65-02: Exact Structural Optimizations")
    print("=======================================================")

    incumbent_receipt = get_latest_receipt("P65-00-incumbent")
    if not incumbent_receipt:
        raise RuntimeError("Missing P65-00-incumbent baseline receipt!")

    incumbent_sha = incumbent_receipt["candidate_sha256"]
    parent_receipt = incumbent_receipt

    # Sub-stages: 02a, 02b, 02c, 02d
    variants = [
        {
            "exp_id": "P65-02a-struc",
            "description": "Structured newborn transport",
            "structured_transport": True,
            "scalar_k4_specialization": False,
            "terminal_split": False,
            "comp_layers": [6, 10, 13],
            "retention": 0.62,
        },
        {
            "exp_id": "P65-02b-k4spec",
            "description": "Input and scalar-K4 specialization",
            "structured_transport": False,
            "scalar_k4_specialization": True,
            "terminal_split": False,
            "comp_layers": [6, 10, 13],
            "retention": 0.62,
        },
        {
            "exp_id": "P65-02c-termsplit",
            "description": "Terminal birth contraction split",
            "structured_transport": False,
            "scalar_k4_specialization": False,
            "terminal_split": True,
            "comp_layers": [6, 10, 13],
            "retention": 0.62,
        },
        {
            "exp_id": "P65-02d-combined",
            "description": "Combined exact candidate (02a + 02b + 02c)",
            "structured_transport": True,
            "scalar_k4_specialization": True,
            "terminal_split": True,
            "comp_layers": [6, 10, 13],
            "retention": 0.62,
        },
    ]

    passing_variants = []

    for v in variants:
        exp_id = v["exp_id"]
        if resume and is_step_completed(exp_id):
            print(f"[{exp_id}] Already completed in ledger. Skipping.")
            passing_variants.append(exp_id)
            continue

        print(f"\n--- Running {exp_id}: {v['description']} ---")
        cand_path = build_candidate(v)

        # Evaluate on full 8 MLPs
        receipt = run_candidate_eval(
            candidate_path=cand_path,
            exp_id=exp_id,
            n_mlps=8,
            mode="diagnostic",
            parent_sha256=incumbent_sha,
            params=v,
        )

        comp = compare_candidates(receipt, incumbent_receipt)
        print(f"[{exp_id}] Score Ratio vs Incumbent: {comp['overall_score_ratio']:.6f} ({comp['mean_improvement_pct']:+.4f}%)")
        print(f"[{exp_id}] Record vs Incumbent: {comp['wins']}W-{comp['losses']}L-{comp['ties']}T")

        # Parity check: panel raw MSE ratio in [0.995, 1.005]
        cm_cand = receipt.get("captured_metrics", {})
        cm_inc = incumbent_receipt.get("captured_metrics", {})
        ratio_raw = cm_cand["mean_raw_final_mse"] / cm_inc["mean_raw_final_mse"]
        print(f"[{exp_id}] Raw MSE ratio: {ratio_raw:.6f}")

        if 0.995 <= ratio_raw <= 1.005:
            print(f"[{exp_id}] EXACT PARITY VERIFIED!")
            append_to_report(receipt, comp_incumbent=comp, comp_parent=comp, decision="PARITY_PASSED")
            passing_variants.append(exp_id)
        else:
            print(f"[{exp_id}] FAILED PARITY: raw ratio {ratio_raw} outside [0.995, 1.005]!")

    # Select cheapest passing as EXACT_PARENT
    best_exact = "P65-02d-combined" if "P65-02d-combined" in passing_variants else (
        passing_variants[-1] if passing_variants else "P65-00-incumbent"
    )

    best_receipt = get_latest_receipt(best_exact)
    progress["research_parent"] = {
        "exp_id": best_exact,
        "sha256": best_receipt["candidate_sha256"],
        "file": best_receipt["candidate_file"],
    }
    progress["last_completed"] = "P65-02"
    progress["next_action"] = "P65-03"
    save_progress(progress)
    print(f"\n[P65-02] EXACT_PARENT selected: {best_exact}")
    return best_exact


def select_research_parent(
    candidates: List[Dict[str, Any]],
    stage_parent: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Section 3 Research-parent rule:
    Choose lowest mean product with >= 6/8 wins over stage parent
    and worst paired score ratio <= 1.10; otherwise retain parent.
    Break ties within relative 1e-4 by lower FLOPs, then lexicographic ID.
    """
    qualifying = []
    for c in candidates:
        comp = compare_candidates(c, stage_parent)
        if comp["wins"] >= 6 and comp["worst_regression_ratio"] <= 1.10:
            qualifying.append((c, comp))

    if not qualifying:
        return stage_parent, "RETAIN_PARENT (No candidate met >=6/8 wins and <=1.10 worst regression)"

    def sort_key(item):
        c, _ = item
        cm = c.get("captured_metrics", {})
        score = cm.get("mean_adjusted_score", 1e9)
        flops = cm.get("mean_compute_utilization", 1.0)
        exp_id = c.get("exp_id", "")
        return (score, flops, exp_id)

    qualifying.sort(key=sort_key)
    best_cand, best_comp = qualifying[0]
    return best_cand, f"PROMOTED ({best_cand['exp_id']} won {best_comp['wins']}W-{best_comp['losses']}L vs parent, ratio {best_comp['overall_score_ratio']:.4f})"


def run_stage_p65_03(progress: Dict[str, Any], resume: bool = True) -> str:
    """Execute Stage P65-03: Retention and schedule frontier."""
    print("\n=======================================================")
    print("STAGE P65-03: Retention and Schedule Frontier")
    print("=======================================================")

    incumbent_receipt = get_latest_receipt("P65-00-incumbent")
    exact_parent_id = progress.get("exact_parent", {}).get("exp_id", "P65-02d-combined")
    exact_parent_receipt = get_latest_receipt(exact_parent_id)
    if not exact_parent_receipt:
        raise RuntimeError(f"Missing {exact_parent_id} receipt!")

    exact_parent_sha = exact_parent_receipt["candidate_sha256"]

    # --- P65-03a: Retention sweep on schedule [6, 10, 13] ---
    print("\n--- Sub-stage P65-03a: Retention sweep on schedule [6, 10, 13] ---")
    retention_configs = [
        {"exp_id": "P65-03a-r045", "retention": 0.45, "description": "Retention sweep r=0.45 on schedule [6, 10, 13]"},
        {"exp_id": "P65-03a-r055", "retention": 0.55, "description": "Retention sweep r=0.55 on schedule [6, 10, 13]"},
        {"exp_id": "P65-03a-r070", "retention": 0.70, "description": "Retention sweep r=0.70 on schedule [6, 10, 13]"},
        {"exp_id": "P65-03a-r080", "retention": 0.80, "description": "Retention sweep r=0.80 on schedule [6, 10, 13]"},
    ]

    p3a_receipts = [exact_parent_receipt]  # r=0.62 is exact parent
    for cfg in retention_configs:
        exp_id = cfg["exp_id"]
        full_cfg = {
            "exp_id": exp_id,
            "description": cfg["description"],
            "structured_transport": False,
            "scalar_k4_specialization": True,
            "terminal_split": True,
            "comp_layers": [6, 10, 13],
            "retention": cfg["retention"],
        }
        if resume and is_step_completed(exp_id):
            print(f"[{exp_id}] Already completed. Loading from ledger.")
            rec = get_latest_receipt(exp_id)
            p3a_receipts.append(rec)
            continue

        cand_path = build_candidate(full_cfg)
        rec = run_candidate_eval(
            candidate_path=cand_path,
            exp_id=exp_id,
            n_mlps=8,
            mode="diagnostic",
            parent_sha256=exact_parent_sha,
            params=full_cfg,
        )
        p3a_receipts.append(rec)

        comp_parent = compare_candidates(rec, exact_parent_receipt)
        comp_inc = compare_candidates(rec, incumbent_receipt)
        cm = rec.get("captured_metrics", {})
        append_to_report(rec, comp_incumbent=comp_inc, comp_parent=comp_parent, decision=f"{comp_parent['wins']}W-{comp_parent['losses']}L")
        update_current_thing(
            method_name=exp_id,
            short_desc=f"Retention r={cfg['retention']:.2f}, schedule [6, 10, 13]",
            raw_final_mse=cm.get("mean_raw_final_mse", 0.0),
            mean_mult=cm.get("mean_compute_utilization", 0.0),
            adjusted_score=cm.get("mean_adjusted_score", 0.0),
            max_res_time=cm.get("max_residual_wall_time_s", 0.0),
            failures=0,
            wins_summary=f"{comp_parent['wins']}W - {comp_parent['losses']}L vs Exact",
            decision=f"Ratio: {comp_parent['overall_score_ratio']:.4f}",
        )

    best_03a, reason_03a = select_research_parent(p3a_receipts, exact_parent_receipt)
    best_03a_ret = best_03a.get("parameters", {}).get("retention", 0.62)
    print(f"\n[P65-03a Winner] {best_03a['exp_id']} (r={best_03a_ret}): {reason_03a}")

    # --- P65-03b: Schedule sweep with best retention r* ---
    print(f"\n--- Sub-stage P65-03b: Schedule sweep with best retention r*={best_03a_ret} ---")
    schedule_configs = [
        {"exp_id": "P65-03b-s5912", "schedule": [5, 9, 12], "desc": f"Schedule [5, 9, 12] with r*={best_03a_ret}"},
        {"exp_id": "P65-03b-s61013", "schedule": [6, 10, 13], "desc": f"Schedule [6, 10, 13] with r*={best_03a_ret}"},
        {"exp_id": "P65-03b-s71114", "schedule": [7, 11, 14], "desc": f"Schedule [7, 11, 14] with r*={best_03a_ret}"},
        {"exp_id": "P65-03b-s6101214", "schedule": [6, 10, 12, 14], "desc": f"Schedule [6, 10, 12, 14] with r*={best_03a_ret}"},
    ]

    p3b_receipts = []
    for scfg in schedule_configs:
        exp_id = scfg["exp_id"]
        # If this is identical to 03a winner, reuse its receipt
        if scfg["schedule"] == [6, 10, 13] and best_03a["exp_id"].startswith("P65-03a"):
            p3b_receipts.append(best_03a)
            continue

        full_cfg = {
            "exp_id": exp_id,
            "description": scfg["desc"],
            "structured_transport": False,
            "scalar_k4_specialization": True,
            "terminal_split": True,
            "comp_layers": scfg["schedule"],
            "retention": best_03a_ret,
        }
        if resume and is_step_completed(exp_id):
            print(f"[{exp_id}] Already completed. Loading from ledger.")
            rec = get_latest_receipt(exp_id)
            p3b_receipts.append(rec)
            continue

        cand_path = build_candidate(full_cfg)
        rec = run_candidate_eval(
            candidate_path=cand_path,
            exp_id=exp_id,
            n_mlps=8,
            mode="diagnostic",
            parent_sha256=best_03a["candidate_sha256"],
            params=full_cfg,
        )
        p3b_receipts.append(rec)

        comp_parent = compare_candidates(rec, best_03a)
        comp_inc = compare_candidates(rec, incumbent_receipt)
        cm = rec.get("captured_metrics", {})
        append_to_report(rec, comp_incumbent=comp_inc, comp_parent=comp_parent, decision=f"{comp_parent['wins']}W-{comp_parent['losses']}L")
        update_current_thing(
            method_name=exp_id,
            short_desc=scfg["desc"],
            raw_final_mse=cm.get("mean_raw_final_mse", 0.0),
            mean_mult=cm.get("mean_compute_utilization", 0.0),
            adjusted_score=cm.get("mean_adjusted_score", 0.0),
            max_res_time=cm.get("max_residual_wall_time_s", 0.0),
            failures=0,
            wins_summary=f"{comp_parent['wins']}W - {comp_parent['losses']}L vs 03a Winner",
            decision=f"Ratio: {comp_parent['overall_score_ratio']:.4f}",
        )

    best_03b, reason_03b = select_research_parent(p3b_receipts, best_03a)
    best_schedule = best_03b.get("parameters", {}).get("comp_layers", [6, 10, 13])
    print(f"\n[P65-03b Winner] {best_03b['exp_id']} (schedule={best_schedule}): {reason_03b}")

    # --- P65-03c: Local per-event perturbation (+-0.10 clipped to [0.35, 0.90]) ---
    print(f"\n--- Sub-stage P65-03c: Local perturbation on schedule {best_schedule} ---")
    p3c_receipts = [best_03b]
    for idx, layer in enumerate(best_schedule):
        for delta in (-0.10, +0.10):
            new_ret = round(max(0.35, min(0.90, best_03a_ret + delta)), 2)
            if new_ret == best_03a_ret:
                continue
            delta_tag = f"m{int(abs(delta)*100):02d}" if delta < 0 else f"p{int(abs(delta)*100):02d}"
            exp_id = f"P65-03c-L{layer}-{delta_tag}"
            per_layer = {l: best_03a_ret for l in best_schedule}
            per_layer[layer] = new_ret

            full_cfg = {
                "exp_id": exp_id,
                "description": f"Perturb layer {layer} retention to {new_ret} on schedule {best_schedule}",
                "structured_transport": False,
                "scalar_k4_specialization": True,
                "terminal_split": True,
                "comp_layers": best_schedule,
                "retention": best_03a_ret,
                "per_layer_retention": per_layer,
            }
            if resume and is_step_completed(exp_id):
                print(f"[{exp_id}] Already completed. Loading from ledger.")
                rec = get_latest_receipt(exp_id)
                p3c_receipts.append(rec)
                continue

            cand_path = build_candidate(full_cfg)
            rec = run_candidate_eval(
                candidate_path=cand_path,
                exp_id=exp_id,
                n_mlps=8,
                mode="diagnostic",
                parent_sha256=best_03b["candidate_sha256"],
                params=full_cfg,
            )
            p3c_receipts.append(rec)

            comp_parent = compare_candidates(rec, best_03b)
            comp_inc = compare_candidates(rec, incumbent_receipt)
            cm = rec.get("captured_metrics", {})
            append_to_report(rec, comp_incumbent=comp_inc, comp_parent=comp_parent, decision=f"{comp_parent['wins']}W-{comp_parent['losses']}L")
            update_current_thing(
                method_name=exp_id,
                short_desc=f"Layer {layer} ret={new_ret:.2f}, sched {best_schedule}",
                raw_final_mse=cm.get("mean_raw_final_mse", 0.0),
                mean_mult=cm.get("mean_compute_utilization", 0.0),
                adjusted_score=cm.get("mean_adjusted_score", 0.0),
                max_res_time=cm.get("max_residual_wall_time_s", 0.0),
                failures=0,
                wins_summary=f"{comp_parent['wins']}W - {comp_parent['losses']}L vs 03b Winner",
                decision=f"Ratio: {comp_parent['overall_score_ratio']:.4f}",
            )

    best_03c, reason_03c = select_research_parent(p3c_receipts, best_03b)
    print(f"\n[P65-03c Winner] {best_03c['exp_id']}: {reason_03c}")

    # Stage P65-03d: only if four-event schedule improves product by at least 2% over best three-event
    # Check if best_schedule has 4 events and improves over best 3-event by >= 2%
    is_four_event = len(best_schedule) == 4
    run_03d = False
    if is_four_event:
        best_3_event = [r for r in p3b_receipts if len(r.get("parameters", {}).get("comp_layers", [])) == 3]
        if best_3_event:
            best_3 = sorted(best_3_event, key=lambda x: x.get("captured_metrics", {}).get("mean_adjusted_score", 1e9))[0]
            comp_4_vs_3 = compare_candidates(best_03b, best_3)
            if comp_4_vs_3["mean_improvement_pct"] >= 2.0:
                run_03d = True

    if not run_03d:
        print("\n[P65-03d] SKIPPED_GATE: Four-event schedule did not improve product >=2% over best 3-event schedule.")

    # Select RANK_PARENT
    rank_parent_rec = best_03c
    progress["research_parent"] = {
        "exp_id": rank_parent_rec["exp_id"],
        "sha256": rank_parent_rec["candidate_sha256"],
        "file": rank_parent_rec["candidate_file"],
    }
    progress["rank_parent"] = progress["research_parent"]
    progress["last_completed"] = "P65-03"
    progress["next_action"] = "P65-04"
    save_progress(progress)
    print(f"\n[P65-03 COMPLETE] RANK_PARENT selected: {rank_parent_rec['exp_id']}")
    return rank_parent_rec["exp_id"]


def run_stage_p65_04(progress: Dict[str, Any], resume: bool = True) -> str:
    """Execute Stage P65-04: Explicit inexpensive selectors."""
    print("\n=======================================================")
    print("STAGE P65-04: Explicit Inexpensive Selectors")
    print("=======================================================")

    incumbent_receipt = get_latest_receipt("P65-00-incumbent")
    rank_parent_id = progress.get("rank_parent", {}).get("exp_id", "P65-03c-L10-p10")
    rank_parent_receipt = get_latest_receipt(rank_parent_id)
    if not rank_parent_receipt:
        raise RuntimeError(f"Missing {rank_parent_id} receipt!")

    rank_parent_sha = rank_parent_receipt["candidate_sha256"]
    rp_params = rank_parent_receipt.get("parameters", {})
    comp_layers = rp_params.get("comp_layers", [6, 10, 13])
    retention = rp_params.get("retention", 0.80)
    per_layer_ret = rp_params.get("per_layer_retention", {6: 0.80, 10: 0.90, 13: 0.80})

    # P65-04a: Exact individual-atom norm
    exp_id_04a = "P65-04a-normexact"
    full_cfg_04a = {
        "exp_id": exp_id_04a,
        "description": "Exact individual-atom norm selector on RANK_PARENT schedule",
        "structured_transport": False,
        "scalar_k4_specialization": True,
        "terminal_split": True,
        "comp_layers": comp_layers,
        "retention": retention,
        "per_layer_retention": per_layer_ret,
        "selector": "norm_exact",
    }

    if resume and is_step_completed(exp_id_04a):
        print(f"[{exp_id_04a}] Already completed. Loading from ledger.")
        rec_04a = get_latest_receipt(exp_id_04a)
    else:
        cand_path = build_candidate(full_cfg_04a)
        rec_04a = run_candidate_eval(
            candidate_path=cand_path,
            exp_id=exp_id_04a,
            n_mlps=8,
            mode="diagnostic",
            parent_sha256=rank_parent_sha,
            params=full_cfg_04a,
        )

    comp_parent = compare_candidates(rec_04a, rank_parent_receipt)
    comp_inc = compare_candidates(rec_04a, incumbent_receipt)
    cm = rec_04a.get("captured_metrics", {})
    append_to_report(rec_04a, comp_incumbent=comp_inc, comp_parent=comp_parent, decision=f"{comp_parent['wins']}W-{comp_parent['losses']}L")
    update_current_thing(
        method_name=exp_id_04a,
        short_desc=f"Exact atom norm selector on RANK_PARENT",
        raw_final_mse=cm.get("mean_raw_final_mse", 0.0),
        mean_mult=cm.get("mean_compute_utilization", 0.0),
        adjusted_score=cm.get("mean_adjusted_score", 0.0),
        max_res_time=cm.get("max_residual_wall_time_s", 0.0),
        failures=0,
        wins_summary=f"{comp_parent['wins']}W - {comp_parent['losses']}L vs Rank Parent",
        decision=f"Ratio: {comp_parent['overall_score_ratio']:.4f}",
    )

    # Select SELECT_PARENT
    best_04, reason_04 = select_research_parent([rec_04a], rank_parent_receipt)
    print(f"\n[P65-04 Winner] {best_04['exp_id']}: {reason_04}")

    progress["research_parent"] = {
        "exp_id": best_04["exp_id"],
        "sha256": best_04["candidate_sha256"],
        "file": best_04["candidate_file"],
    }
    progress["select_parent"] = progress["research_parent"]
    progress["last_completed"] = "P65-04"
    progress["next_action"] = "P65-05"
    save_progress(progress)
    print(f"\n[P65-04 COMPLETE] SELECT_PARENT selected: {best_04['exp_id']}")
    return best_04["exp_id"]


def run_stage_p65_05(progress: Dict[str, Any], resume: bool = True) -> str:
    """Execute Stage P65-05: Exact current-slice compensation."""
    print("\n=======================================================")
    print("STAGE P65-05: Exact Current-Slice Compensation")
    print("=======================================================")

    incumbent_receipt = get_latest_receipt("P65-00-incumbent")
    select_parent_id = progress.get("select_parent", {}).get("exp_id", "P65-03c-L10-p10")
    select_parent_receipt = get_latest_receipt(select_parent_id)
    if not select_parent_receipt:
        raise RuntimeError(f"Missing {select_parent_id} receipt!")

    select_parent_sha = select_parent_receipt["candidate_sha256"]
    sp_params = select_parent_receipt.get("parameters", {})
    comp_layers = sp_params.get("comp_layers", [6, 10, 13])
    retention = sp_params.get("retention", 0.80)

    # 1. P65-05a: test at r=0.45 and r=0.55 at Layer 10
    variants_05a = [
        {"exp_id": "P65-05a-r045-comp", "ret": 0.45, "comp": True, "desc": "Compensated L10 (r=0.45)"},
        {"exp_id": "P65-05a-r045-uncomp", "ret": 0.45, "comp": False, "desc": "Matched uncompensated L10 (r=0.45)"},
        {"exp_id": "P65-05a-r055-comp", "ret": 0.55, "comp": True, "desc": "Compensated L10 (r=0.55)"},
        {"exp_id": "P65-05a-r055-uncomp", "ret": 0.55, "comp": False, "desc": "Matched uncompensated L10 (r=0.55)"},
    ]

    receipts_05a = {}
    for v in variants_05a:
        exp_id = v["exp_id"]
        per_layer = {6: 0.80, 10: v["ret"], 13: 0.80}
        full_cfg = {
            "exp_id": exp_id,
            "description": v["desc"],
            "structured_transport": False,
            "scalar_k4_specialization": True,
            "terminal_split": True,
            "comp_layers": comp_layers,
            "retention": retention,
            "per_layer_retention": per_layer,
            "slice_compensation": v["comp"],
            "compensation_layers": [10] if v["comp"] else [],
        }

        if resume and is_step_completed(exp_id):
            print(f"[{exp_id}] Already completed. Loading from ledger.")
            rec = get_latest_receipt(exp_id)
        else:
            cand_path = build_candidate(full_cfg)
            rec = run_candidate_eval(
                candidate_path=cand_path,
                exp_id=exp_id,
                n_mlps=8,
                mode="diagnostic",
                parent_sha256=select_parent_sha,
                params=full_cfg,
            )

        receipts_05a[exp_id] = rec
        comp_parent = compare_candidates(rec, select_parent_receipt)
        comp_inc = compare_candidates(rec, incumbent_receipt)
        cm = rec.get("captured_metrics", {})
        append_to_report(rec, comp_incumbent=comp_inc, comp_parent=comp_parent, decision=f"{comp_parent['wins']}W-{comp_parent['losses']}L")
        update_current_thing(
            method_name=exp_id,
            short_desc=v["desc"],
            raw_final_mse=cm.get("mean_raw_final_mse", 0.0),
            mean_mult=cm.get("mean_compute_utilization", 0.0),
            adjusted_score=cm.get("mean_adjusted_score", 0.0),
            max_res_time=cm.get("max_residual_wall_time_s", 0.0),
            failures=0,
            wins_summary=f"{comp_parent['wins']}W - {comp_parent['losses']}L vs Select Parent",
            decision=f"Ratio: {comp_parent['overall_score_ratio']:.4f}",
        )

    # Check 05a compensation gate vs matched uncompensated
    gate_passed = False
    for r_val in (0.45, 0.55):
        c_rec = receipts_05a.get(f"P65-05a-r{int(r_val*100):02d}-comp")
        u_rec = receipts_05a.get(f"P65-05a-r{int(r_val*100):02d}-uncomp")
        if c_rec and u_rec:
            comp_vs_uncomp = compare_candidates(c_rec, u_rec)
            print(f"\n[P65-05a r={r_val}] Comp vs Uncomp: {comp_vs_uncomp['wins']}W-{comp_vs_uncomp['losses']}L, Gain: {comp_vs_uncomp['mean_improvement_pct']:+.2f}%")
            if comp_vs_uncomp["mean_improvement_pct"] >= 2.0:
                gate_passed = True
                print(f"[P65-05a r={r_val}] GATE PASSED: >= 2% improvement over matched uncompensated!")

    # 4. Choose SLICE_PARENT under Section 3 research-parent rule
    # Can any compensated candidate beat SELECT_PARENT?
    comp_cands = [v for k, v in receipts_05a.items() if "comp" in k and "uncomp" not in k]
    best_05, reason_05 = select_research_parent(comp_cands, select_parent_receipt)
    print(f"\n[P65-05 Winner] {best_05['exp_id']}: {reason_05}")

    progress["research_parent"] = {
        "exp_id": best_05["exp_id"],
        "sha256": best_05["candidate_sha256"],
        "file": best_05["candidate_file"],
    }
    progress["slice_parent"] = progress["research_parent"]
    progress["last_completed"] = "P65-05"
    progress["next_action"] = "P65-06"
    save_progress(progress)
    print(f"\n[P65-05 COMPLETE] SLICE_PARENT selected: {best_05['exp_id']}")
    return best_05["exp_id"]


def run_stage_p65_07(progress: Dict[str, Any], resume: bool = True) -> str:
    """Execute Stage P65-07: Diagonal fourth-order trace-harmonic extension."""
    print("\n=======================================================")
    print("STAGE P65-07: Diagonal Fourth-Order Trace-Harmonic Extension")
    print("=======================================================")

    incumbent_receipt = get_latest_receipt("P65-00-incumbent")
    analytic_parent_id = progress.get("slice_parent", {}).get("exp_id", "P65-03c-L10-p10")
    analytic_parent_receipt = get_latest_receipt(analytic_parent_id)
    if not analytic_parent_receipt:
        raise RuntimeError(f"Missing {analytic_parent_id} receipt!")

    analytic_parent_sha = analytic_parent_receipt["candidate_sha256"]
    ap_params = analytic_parent_receipt.get("parameters", {})
    comp_layers = ap_params.get("comp_layers", [6, 10, 13])
    retention = ap_params.get("retention", 0.80)
    per_layer_ret = ap_params.get("per_layer_retention", {6: 0.80, 10: 0.90, 13: 0.80})

    # The 6 configurations in prescribed order
    q_configs = [
        {"exp_id": "P65-07-s13-e050", "start": 13, "eta": 0.5, "desc": "Diagonal Q extension (start=13, eta=0.5)"},
        {"exp_id": "P65-07-s13-e100", "start": 13, "eta": 1.0, "desc": "Diagonal Q extension (start=13, eta=1.0)"},
        {"exp_id": "P65-07-s11-e050", "start": 11, "eta": 0.5, "desc": "Diagonal Q extension (start=11, eta=0.5)"},
        {"exp_id": "P65-07-s11-e100", "start": 11, "eta": 1.0, "desc": "Diagonal Q extension (start=11, eta=1.0)"},
        {"exp_id": "P65-07-s07-e050", "start": 7,  "eta": 0.5, "desc": "Diagonal Q extension (start=7, eta=0.5)"},
        {"exp_id": "P65-07-s07-e100", "start": 7,  "eta": 1.0, "desc": "Diagonal Q extension (start=7, eta=1.0)"},
    ]

    p7_receipts = []
    for qc in q_configs:
        exp_id = qc["exp_id"]
        full_cfg = {
            "exp_id": exp_id,
            "description": qc["desc"],
            "structured_transport": False,
            "scalar_k4_specialization": True,
            "terminal_split": True,
            "comp_layers": comp_layers,
            "retention": retention,
            "per_layer_retention": per_layer_ret,
            "use_q": True,
            "q_start": qc["start"],
            "q_eta": qc["eta"],
        }

        if resume and is_step_completed(exp_id):
            print(f"[{exp_id}] Already completed. Loading from ledger.")
            rec = get_latest_receipt(exp_id)
            p7_receipts.append(rec)
            continue
        else:
            cand_path = build_candidate(full_cfg)
            rec = run_candidate_eval(
                candidate_path=cand_path,
                exp_id=exp_id,
                n_mlps=8,
                mode="diagnostic",
                parent_sha256=analytic_parent_sha,
                params=full_cfg,
            )
            p7_receipts.append(rec)
            comp_parent = compare_candidates(rec, analytic_parent_receipt)
            comp_inc = compare_candidates(rec, incumbent_receipt)
            cm = rec.get("captured_metrics", {})
            append_to_report(rec, comp_incumbent=comp_inc, comp_parent=comp_parent, decision=f"{comp_parent['wins']}W-{comp_parent['losses']}L")
            update_current_thing(
                method_name=exp_id,
                short_desc=qc["desc"],
                raw_final_mse=cm.get("mean_raw_final_mse", 0.0),
                mean_mult=cm.get("mean_compute_utilization", 0.0),
                adjusted_score=cm.get("mean_adjusted_score", 0.0),
                max_res_time=cm.get("max_residual_wall_time_s", 0.0),
                failures=0,
                wins_summary=f"{comp_parent['wins']}W - {comp_parent['losses']}L vs Analytic Parent",
                decision=f"Ratio: {comp_parent['overall_score_ratio']:.4f}",
            )

    best_07, reason_07 = select_research_parent(p7_receipts, analytic_parent_receipt)
    print(f"\n[P65-07 Winner] {best_07['exp_id']}: {reason_07}")

    progress["research_parent"] = {
        "exp_id": best_07["exp_id"],
        "sha256": best_07["candidate_sha256"],
        "file": best_07["candidate_file"],
    }
    progress["k4_parent"] = progress["research_parent"]
    progress["last_completed"] = "P65-07"
    progress["next_action"] = "P65-08"
    save_progress(progress)
    print(f"\n[P65-07 COMPLETE] K4_PARENT selected: {best_07['exp_id']}")
    return best_07["exp_id"]


def run_stage_p65_08(progress: Dict[str, Any], resume: bool = True) -> str:
    """Execute Stage P65-08: Small online K3-centered control variate."""
    print("\n=======================================================")
    print("STAGE P65-08: Small Online K3-Centered Control Variate")
    print("=======================================================")

    incumbent_receipt = get_latest_receipt("P65-00-incumbent")
    k4_parent_id = progress.get("k4_parent", {}).get("exp_id", "P65-03c-L10-p10")
    k4_parent_receipt = get_latest_receipt(k4_parent_id)
    if not k4_parent_receipt:
        raise RuntimeError(f"Missing {k4_parent_id} receipt!")

    k4_parent_sha = k4_parent_receipt["candidate_sha256"]
    kp_params = k4_parent_receipt.get("parameters", {})
    comp_layers = kp_params.get("comp_layers", [6, 10, 13])
    retention = kp_params.get("retention", 0.80)
    per_layer_ret = kp_params.get("per_layer_retention", {6: 0.80, 10: 0.90, 13: 0.80})

    # 6 configurations: N in {512, 1024}, alpha in {0.05, 0.10, 0.20}
    cv_configs = [
        {"exp_id": "P65-08-n0512-a005", "N": 512, "alpha": 0.05, "desc": "Online K3 CV (N=512, alpha=0.05)"},
        {"exp_id": "P65-08-n0512-a010", "N": 512, "alpha": 0.10, "desc": "Online K3 CV (N=512, alpha=0.10)"},
        {"exp_id": "P65-08-n0512-a020", "N": 512, "alpha": 0.20, "desc": "Online K3 CV (N=512, alpha=0.20)"},
        {"exp_id": "P65-08-n1024-a005", "N": 1024, "alpha": 0.05, "desc": "Online K3 CV (N=1024, alpha=0.05)"},
        {"exp_id": "P65-08-n1024-a010", "N": 1024, "alpha": 0.10, "desc": "Online K3 CV (N=1024, alpha=0.10)"},
        {"exp_id": "P65-08-n1024-a020", "N": 1024, "alpha": 0.20, "desc": "Online K3 CV (N=1024, alpha=0.20)"},
    ]

    p8_receipts = []
    for cc in cv_configs:
        exp_id = cc["exp_id"]
        full_cfg = {
            "exp_id": exp_id,
            "description": cc["desc"],
            "structured_transport": False,
            "scalar_k4_specialization": True,
            "terminal_split": True,
            "comp_layers": comp_layers,
            "retention": retention,
            "per_layer_retention": per_layer_ret,
            "use_cv": True,
            "cv_N": cc["N"],
            "cv_alpha": cc["alpha"],
        }

        if resume and is_step_completed(exp_id):
            print(f"[{exp_id}] Already completed. Loading from ledger.")
            rec = get_latest_receipt(exp_id)
            p8_receipts.append(rec)
            continue
        else:
            cand_path = build_candidate(full_cfg)
            rec = run_candidate_eval(
                candidate_path=cand_path,
                exp_id=exp_id,
                n_mlps=8,
                offset=0,
                mode="diagnostic",
                parent_sha256=k4_parent_sha,
                params=full_cfg,
            )
            p8_receipts.append(rec)
            comp_parent = compare_candidates(rec, k4_parent_receipt)
            comp_inc = compare_candidates(rec, incumbent_receipt)
            cm = rec.get("captured_metrics", {})
            append_to_report(rec, comp_incumbent=comp_inc, comp_parent=comp_parent, decision=f"{comp_parent['wins']}W-{comp_parent['losses']}L")
            update_current_thing(
                method_name=exp_id,
                short_desc=cc["desc"],
                raw_final_mse=cm.get("mean_raw_final_mse", 0.0),
                mean_mult=cm.get("mean_compute_utilization", 0.0),
                adjusted_score=cm.get("mean_adjusted_score", 0.0),
                max_res_time=cm.get("max_residual_wall_time_s", 0.0),
                failures=0,
                wins_summary=f"{comp_parent['wins']}W - {comp_parent['losses']}L vs K4 Parent",
                decision=f"Ratio: {comp_parent['overall_score_ratio']:.4f}",
            )

    best_08, reason_08 = select_research_parent(p8_receipts, k4_parent_receipt)
    print(f"\n[P65-08 Winner at offset 0] {best_08['exp_id']}: {reason_08}")

    # Check multi-salt gate if a new candidate won
    if best_08["exp_id"] != k4_parent_receipt["exp_id"]:
        print(f"\n[P65-08 Multi-Salt Verification for {best_08['exp_id']}] Testing offsets 1337 and 8888...")
        for offset in (1337, 8888):
            cand_path = Path(best_08["candidate_file"])
            salt_exp_id = f"{best_08['exp_id']}-salt{offset}"
            if resume and is_step_completed(salt_exp_id):
                print(f"[{salt_exp_id}] Already completed.")
            else:
                salt_rec = run_candidate_eval(
                    candidate_path=cand_path,
                    exp_id=salt_exp_id,
                    n_mlps=8,
                    offset=offset,
                    mode="diagnostic",
                    parent_sha256=k4_parent_sha,
                    params=best_08.get("parameters", {}),
                )
                comp_parent = compare_candidates(salt_rec, k4_parent_receipt)
                append_to_report(salt_rec, comp_parent=comp_parent, decision=f"Salt {offset}: {comp_parent['wins']}W-{comp_parent['losses']}L")

    progress["research_parent"] = {
        "exp_id": best_08["exp_id"],
        "sha256": best_08["candidate_sha256"],
        "file": best_08["candidate_file"],
    }
    progress["cv_parent"] = progress["research_parent"]
    progress["last_completed"] = "P65-08"
    progress["next_action"] = "P65-09"
    save_progress(progress)
    print(f"\n[P65-08 COMPLETE] CV_PARENT selected: {best_08['exp_id']}")
    return best_08["exp_id"]


def run_stage_p65_09(progress: Dict[str, Any], resume: bool = True) -> str:
    """Execute Stage P65-09: Combinations and final neighbors."""
    print("\n=======================================================")
    print("STAGE P65-09: Combinations and Final Neighbors")
    print("=======================================================")

    incumbent_receipt = get_latest_receipt("P65-00-incumbent")
    best_parent_id = progress.get("research_parent", {}).get("exp_id", "P65-03c-L10-p10")
    best_parent_receipt = get_latest_receipt(best_parent_id)
    if not best_parent_receipt:
        raise RuntimeError(f"Missing {best_parent_id} receipt!")

    bp_params = best_parent_receipt.get("parameters", {})
    comp_layers = bp_params.get("comp_layers", [6, 10, 13])
    retention = bp_params.get("retention", 0.80)
    per_layer_ret = dict(bp_params.get("per_layer_retention", {6: 0.80, 10: 0.90, 13: 0.80}))
    last_layer = comp_layers[-1]  # 13
    last_ret = per_layer_ret.get(last_layer, retention)

    neighbor_receipts = [best_parent_receipt]
    for delta in (-0.05, +0.05):
        new_ret = round(max(0.35, min(0.95, last_ret + delta)), 2)
        if new_ret == last_ret:
            continue
        delta_tag = f"m{int(abs(delta)*100):02d}" if delta < 0 else f"p{int(abs(delta)*100):02d}"
        exp_id = f"P65-09-L{last_layer}-{delta_tag}"
        neigh_per_layer = dict(per_layer_ret)
        neigh_per_layer[last_layer] = new_ret

        full_cfg = dict(bp_params)
        full_cfg["exp_id"] = exp_id
        full_cfg["description"] = f"Final neighbor: perturb last event L{last_layer} retention to {new_ret}"
        full_cfg["per_layer_retention"] = neigh_per_layer

        if resume and is_step_completed(exp_id):
            print(f"[{exp_id}] Already completed. Loading from ledger.")
            rec = get_latest_receipt(exp_id)
            neighbor_receipts.append(rec)
            continue
        else:
            cand_path = build_candidate(full_cfg)
            rec = run_candidate_eval(
                candidate_path=cand_path,
                exp_id=exp_id,
                n_mlps=8,
                mode="diagnostic",
                parent_sha256=best_parent_receipt["candidate_sha256"],
                params=full_cfg,
            )
            neighbor_receipts.append(rec)
            comp_parent = compare_candidates(rec, best_parent_receipt)
            comp_inc = compare_candidates(rec, incumbent_receipt)
            cm = rec.get("captured_metrics", {})
            append_to_report(rec, comp_incumbent=comp_inc, comp_parent=comp_parent, decision=f"{comp_parent['wins']}W-{comp_parent['losses']}L")
            update_current_thing(
                method_name=exp_id,
                short_desc=full_cfg["description"],
                raw_final_mse=cm.get("mean_raw_final_mse", 0.0),
                mean_mult=cm.get("mean_compute_utilization", 0.0),
                adjusted_score=cm.get("mean_adjusted_score", 0.0),
                max_res_time=cm.get("max_residual_wall_time_s", 0.0),
                failures=0,
                wins_summary=f"{comp_parent['wins']}W - {comp_parent['losses']}L vs Best Parent",
                decision=f"Ratio: {comp_parent['overall_score_ratio']:.4f}",
            )

    best_09, reason_09 = select_research_parent(neighbor_receipts, best_parent_receipt)
    print(f"\n[P65-09 Winner] {best_09['exp_id']}: {reason_09}")

    progress["research_parent"] = {
        "exp_id": best_09["exp_id"],
        "sha256": best_09["candidate_sha256"],
        "file": best_09["candidate_file"],
    }
    progress["last_completed"] = "P65-09"
    progress["next_action"] = "P65-10"
    save_progress(progress)
    print(f"\n[P65-09 COMPLETE] Finalist selected: {best_09['exp_id']}")
    return best_09["exp_id"]


def run_stage_p65_10(progress: Dict[str, Any], resume: bool = True) -> str:
    """Execute Stage P65-10: Final verification, report, and release."""
    print("\n=======================================================")
    print("STAGE P65-10: Final Verification and Release Packaging")
    print("=======================================================")

    finalist_id = progress.get("research_parent", {}).get("exp_id", "P65-03c-L10-p10")
    finalist_receipt = get_latest_receipt(finalist_id)
    if not finalist_receipt:
        raise RuntimeError(f"Missing {finalist_id} receipt!")

    finalist_path = Path(finalist_receipt["candidate_file"])
    print(f"[P65-10] Finalist: {finalist_id} ({finalist_path})")

    # 1. Math checks
    print(f"[P65-10] Running mathematical checks on finalist...")
    run_math_checks(candidate_path=finalist_path)

    # 2. Unrelaxed run (subprocess runner, official caps)
    exp_id_unrelaxed = f"{finalist_id}-unrelaxed"
    if resume and is_step_completed(exp_id_unrelaxed):
        print(f"[{exp_id_unrelaxed}] Already completed. Loading from ledger.")
        unrelaxed_rec = get_latest_receipt(exp_id_unrelaxed)
    else:
        print(f"[{exp_id_unrelaxed}] Running official unrelaxed evaluation (subprocess runner)...")
        unrelaxed_rec = run_candidate_eval(
            candidate_path=finalist_path,
            exp_id=exp_id_unrelaxed,
            n_mlps=8,
            runner="subprocess",
            mode="unrelaxed",
            parent_sha256=finalist_receipt["candidate_sha256"],
            params=finalist_receipt.get("parameters", {}),
        )

    # 3. Copy finalist to candidates/estimator_p65_final.py
    final_dest = WORKSPACE_ROOT / "candidates" / "estimator_p65_final.py"
    final_dest.write_text(finalist_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[P65-10] Wrote finalist to {final_dest}")

    # 4. Packaging
    release_dir = P65_DIR / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    archive_path = release_dir / "submission_phase65.tar.gz"

    import datetime
    import os
    import subprocess
    import tarfile

    val_env = dict(os.environ)
    val_env["PYTHONIOENCODING"] = "utf-8"
    val_env["PYTHONUTF8"] = "1"

    val_cmd = [
        "uv", "run", "whest", "validate",
        "--estimator", str(final_dest),
    ]
    val_res = subprocess.run(val_cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", env=val_env)
    val_out_safe = (val_res.stdout or "").encode("ascii", errors="replace").decode("ascii").strip()
    print(f"[P65-10] Validate output: {val_out_safe}")
    if val_res.returncode != 0:
        raise RuntimeError(f"whest validate failed with code {val_res.returncode}:\n{val_res.stderr}\n{val_res.stdout}")

    pkg_cmd = [
        "uv", "run", "whest", "package",
        "--estimator", str(final_dest),
        "--output", str(archive_path),
        "--yes",
    ]
    pkg_res = subprocess.run(pkg_cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", env=val_env)
    pkg_out_safe = (pkg_res.stdout or "").encode("ascii", errors="replace").decode("ascii").strip()
    print(f"[P65-10] Package output: {pkg_out_safe}")
    if pkg_res.returncode != 0:
        raise RuntimeError(f"whest package failed with code {pkg_res.returncode}:\n{pkg_res.stderr}\n{pkg_res.stdout}")

    if archive_path.exists():
        print(f"[P65-10] Verified archive: {archive_path} ({archive_path.stat().st_size} bytes)")

    # Populate release artifacts required by runbook Section 2
    finalist_copy = release_dir / "estimator_p65_final.py"
    finalist_copy.write_text(final_dest.read_text(encoding="utf-8"), encoding="utf-8")

    with tarfile.open(archive_path, "r:gz") as tf:
        manifest_data = tf.extractfile("manifest.json").read().decode("utf-8")
        (release_dir / "manifest.json").write_text(manifest_data, encoding="utf-8")

    verification_data = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "finalist_id": finalist_id,
        "finalist_file": str(final_dest),
        "finalist_sha256": sha256_file(final_dest),
        "archive_sha256": sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "validate_exit_code": val_res.returncode,
        "validate_stdout": val_res.stdout.strip(),
        "package_exit_code": pkg_res.returncode,
        "package_stdout": pkg_res.stdout.strip(),
        "unrelaxed_metrics": unrelaxed_rec.get("captured_metrics", {}),
    }
    (release_dir / "verification.json").write_text(json.dumps(verification_data, indent=2), encoding="utf-8")
    print(f"[P65-10] Populated release artifacts in {release_dir}")

    progress["last_completed"] = "P65-10"
    progress["next_action"] = "COMPLETE"
    save_progress(progress)
    print(f"\n=======================================================")
    print(f"PHASE 6.5 COMPLETE: Finalist is {finalist_id}")
    print(f"=======================================================")
    return finalist_id


def run_plan(resume: bool = True, target_stage: Optional[str] = None) -> None:
    progress = load_progress()

    if target_stage == "P65-02" or (target_stage is None and progress.get("last_completed") in ("P65-00", "P65-00-init", "P65-01")):
        run_stage_p65_02(progress, resume=resume)
        progress = load_progress()

    if target_stage == "P65-03" or (target_stage is None and progress.get("last_completed") in ("P65-02", "P65-03a-r045")):
        run_stage_p65_03(progress, resume=resume)
        progress = load_progress()

    if target_stage == "P65-04" or (target_stage is None and progress.get("last_completed") == "P65-03"):
        run_stage_p65_04(progress, resume=resume)
        progress = load_progress()

    if target_stage == "P65-05" or (target_stage is None and progress.get("last_completed") == "P65-04"):
        run_stage_p65_05(progress, resume=resume)
        progress = load_progress()

    if target_stage == "P65-07" or (
        target_stage is None
        and (
            progress.get("last_completed") in ("P65-05", "P65-06", "P65-04a-exact")
            or str(progress.get("last_completed", "")).startswith("P65-07")
        )
        and progress.get("last_completed") != "P65-07"
    ):
        run_stage_p65_07(progress, resume=resume)
        progress = load_progress()

    if target_stage == "P65-08" or (
        target_stage is None
        and (
            progress.get("last_completed") == "P65-07"
            or str(progress.get("last_completed", "")).startswith("P65-08")
        )
        and progress.get("last_completed") != "P65-08"
    ):
        run_stage_p65_08(progress, resume=resume)
        progress = load_progress()

    if target_stage == "P65-09" or (
        target_stage is None
        and (
            progress.get("last_completed") == "P65-08"
            or str(progress.get("last_completed", "")).startswith("P65-09")
        )
        and progress.get("last_completed") != "P65-09"
    ):
        run_stage_p65_09(progress, resume=resume)
        progress = load_progress()

    if target_stage == "P65-10" or (
        target_stage is None
        and (
            progress.get("last_completed") == "P65-09"
            or str(progress.get("last_completed", "")).startswith("P65-10")
        )
        and progress.get("last_completed") != "P65-10"
    ):
        run_stage_p65_10(progress, resume=resume)
        progress = load_progress()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from progress.json")
    parser.add_argument("--stage", type=str, default=None, help="Target stage (e.g. P65-02, P65-03, P65-04, P65-05, P65-07, P65-08, P65-09, P65-10)")
    args = parser.parse_args()

    run_plan(resume=args.resume, target_stage=args.stage)




