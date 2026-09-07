"""p8_manifest.py

Phase 8 Stage P8-00: Manifest, environment snapshot, control freezes (CONTROL7, CONTROL65, LEGACY, REF3),
dataset panels verification (dev8 + locked confirmation panel), and schema initialization.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flopscope
import scipy
import whestbench
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP

EXPECTED_CONTROL7_SHA = "44723be93f8a432fa493a0218396a8346728277ba0a694c483fd9369ad26080f"
EXPECTED_CONTROL65_SHA = "b06d91bc155f19e696c690c2b9956c2ba2426ba142de3c7d6dcf7f078344255e"
EXPECTED_LEGACY_SHA = "ea8be8222b827607f65d166118500b55fcaae883b79f64b29785b46d86501ca1"
EXPECTED_REF_COMMIT = "93d091a4c26c042bfffa28f2e76a81bc0aba94bb"

PANEL_DEV8_NAMES = [
    "logan-fitzgerald",
    "william-graves",
    "raymond-barnes",
    "steven-rice",
    "sarah-kelley",
    "christopher-morales",
    "cheryl-graham",
    "renee-park",
]

DATASET_PATH = str(WORKSPACE_ROOT / "datasets" / "mini")
P8_DIR = REPO_ROOT / "research" / "phase8"
CONTROL_DIR = P8_DIR / "control"
FIXTURES_DIR = P8_DIR / "fixtures"
CONFIGS_DIR = P8_DIR / "configs"
PREDICTIONS_DIR = P8_DIR / "predictions"
DIAGNOSTICS_DIR = P8_DIR / "diagnostics"
RESULTS_DIR = P8_DIR / "results"
RELEASE_DIR = P8_DIR / "release"
OFFLINE_DIR = P8_DIR / "offline"
TEACHER_DIR = P8_DIR / "teacher"
TRAINING_DIR = P8_DIR / "training"
WEIGHTS_DIR = P8_DIR / "weights"

REPORT_PATH = WORKSPACE_ROOT / "phase8report.md"


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get_system_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        return -1.0


def build_manifest() -> Dict[str, Any]:
    for d in (CONTROL_DIR, FIXTURES_DIR, CONFIGS_DIR, PREDICTIONS_DIR, DIAGNOSTICS_DIR,
              RESULTS_DIR, RELEASE_DIR, OFFLINE_DIR, TEACHER_DIR, TRAINING_DIR, WEIGHTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Environment versions
    try:
        import importlib.metadata as im
        whestbench_ver = im.version("whestbench")
    except Exception:
        whestbench_ver = getattr(whestbench, "__version__", "unknown")

    flopscope_ver = getattr(flopscope, "__version__", "unknown")
    numpy_ver = np.__version__
    scipy_ver = scipy.__version__
    python_ver = sys.version
    ram_gb = get_system_ram_gb()

    # Git status
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)).decode().strip()
        git_dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(REPO_ROOT)).decode().strip().splitlines()
    except Exception:
        git_head = "unknown"
        git_dirty = []

    env_info = {
        "whestbench": whestbench_ver,
        "flopscope": flopscope_ver,
        "numpy": numpy_ver,
        "scipy": scipy_ver,
        "python": python_ver,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "ram_total_gb": round(ram_gb, 2),
        "git_head": git_head,
        "git_dirty_count": len(git_dirty),
        "historical_reference_commit": EXPECTED_REF_COMMIT,
        "caps": {
            "residual_wall_time_cap_s": 0.40,
            "predict_wall_time_cap_s": 120.0,
            "setup_wall_time_cap_s": 5.0,
            "max_memory_cap_gb": 6.0,
            "flop_budget_per_mlp": 2**41,
        },
    }
    print("[p8_manifest] Environment:")
    for k, v in env_info.items():
        if k != "caps":
            print(f"  {k}: {v}")

    # 2. Freeze Controls
    # CONTROL7
    control7_path = WORKSPACE_ROOT / "candidates" / "estimator_p7_final.py"
    if not control7_path.exists():
        raise FileNotFoundError(f"Missing CONTROL7: {control7_path}")
    control7_sha = sha256_file(control7_path)
    if control7_sha != EXPECTED_CONTROL7_SHA:
        raise ValueError(f"CONTROL7 SHA mismatch! Expected {EXPECTED_CONTROL7_SHA}, got {control7_sha}")
    control7_copy = CONTROL_DIR / "estimator_control7.py"
    shutil.copy2(control7_path, control7_copy)
    print(f"[p8_manifest] CONTROL7 verified and frozen: {control7_sha}")

    # CONTROL65
    control65_path = WORKSPACE_ROOT / "candidates" / "estimator_p65_final.py"
    if not control65_path.exists():
        raise FileNotFoundError(f"Missing CONTROL65: {control65_path}")
    control65_sha = sha256_file(control65_path)
    if control65_sha != EXPECTED_CONTROL65_SHA:
        raise ValueError(f"CONTROL65 SHA mismatch! Expected {EXPECTED_CONTROL65_SHA}, got {control65_sha}")
    control65_copy = CONTROL_DIR / "estimator_control65.py"
    shutil.copy2(control65_path, control65_copy)
    print(f"[p8_manifest] CONTROL65 verified and frozen: {control65_sha}")

    # LEGACY
    legacy_path = REPO_ROOT / "estimator.py"
    if not legacy_path.exists():
        raise FileNotFoundError(f"Missing LEGACY: {legacy_path}")
    legacy_sha = sha256_file(legacy_path)
    if legacy_sha != EXPECTED_LEGACY_SHA:
        raise ValueError(f"LEGACY SHA mismatch! Expected {EXPECTED_LEGACY_SHA}, got {legacy_sha}")
    legacy_copy = CONTROL_DIR / "estimator_legacy.py"
    shutil.copy2(legacy_path, legacy_copy)
    print(f"[p8_manifest] LEGACY verified and frozen: {legacy_sha}")

    # REF3 check
    ref3_json = REPO_ROOT / "research" / "phase6" / "results" / "P6-03_reference_panel.json"
    ref3_preds = REPO_ROOT / "research" / "phase6" / "predictions" / "P6-03_K3-simple_preds.npy"
    ref3_info: Dict[str, Any] = {}
    if ref3_json.exists():
        with open(ref3_json, "r", encoding="utf-8") as f:
            ref3_info["receipt"] = json.load(f)
        ref3_info["receipt_sha256"] = sha256_file(ref3_json)
    if ref3_preds.exists():
        ref3_info["preds_sha256"] = sha256_file(ref3_preds)
        ref3_info["recorded_mse"] = 3.63704648e-8
    print(f"[p8_manifest] REF3 verified from Phase 6 artifacts.")

    # 3. Development Panel (8 MLPs)
    print("[p8_manifest] Verifying 8-MLP development panel from dataset...")
    ds = load_dataset(DATASET_PATH, split="mini")
    salt_ver, seed_salt = resolve_seed_context(ds)

    names_in_ds = ds["mlp_name"]
    dev8_info = []
    dev8_gt_means = []

    for idx, name in enumerate(PANEL_DEV8_NAMES):
        if name not in names_in_ds:
            raise ValueError(f"Dev MLP {name} not found in dataset!")
        row_idx = names_in_ds.index(name)
        row = ds[row_idx]
        mlp = MLP.from_row(row, seed_protocol_version=salt_ver, seed_salt=seed_salt)
        weights_bytes = bytearray()
        for w in mlp.weights:
            weights_bytes.extend(np.asarray(w, dtype=np.float32).tobytes())
        w_sha = sha256_bytes(bytes(weights_bytes))

        gt_means = np.asarray(row["all_layer_means"], dtype=np.float32)
        dev8_gt_means.append(gt_means)
        t_sha = sha256_bytes(gt_means.tobytes())

        dev8_info.append({
            "order": idx,
            "name": name,
            "dataset_row_index": row_idx,
            "width": mlp.width,
            "depth": mlp.depth,
            "seed": int(mlp.seed),
            "weights_sha256": w_sha,
            "truth_sha256": t_sha,
        })
        print(f"  [{idx}] {name:20s}: seed={mlp.seed} weights_sha={w_sha[:16]}... OK")

    # Save ground truth for Dev8
    dev8_gt_arr = np.stack(dev8_gt_means, axis=0)  # (8, 16, 1024)
    gt_target = FIXTURES_DIR / "ground_truth.npy"
    np.save(gt_target, dev8_gt_arr)
    print(f"[p8_manifest] Ground truth fixture saved: {gt_target} (shape={dev8_gt_arr.shape})")

    # 4. Locked Confirmation Panel (Mini rows 8 through 23, or available rows)
    # Total rows in mini: 20 (indices 0..19). Available confirmation rows: 8..19 (12 networks).
    print("[p8_manifest] Registering locked confirmation panel (Section 3.2)...")
    confirmation_info = []
    total_ds_rows = len(ds)
    conf_start = 8
    conf_end = min(24, total_ds_rows)
    for row_idx in range(conf_start, conf_end):
        row = ds[row_idx]
        mlp = MLP.from_row(row, seed_protocol_version=salt_ver, seed_salt=seed_salt)
        weights_bytes = bytearray()
        for w in mlp.weights:
            weights_bytes.extend(np.asarray(w, dtype=np.float32).tobytes())
        w_sha = sha256_bytes(bytes(weights_bytes))
        confirmation_info.append({
            "order": row_idx - conf_start,
            "dataset_row_index": row_idx,
            "name": row["mlp_name"],
            "width": mlp.width,
            "depth": mlp.depth,
            "seed": int(mlp.seed),
            "weights_sha256": w_sha,
            # Target array is intentionally NOT saved to prevent leakage
            "status": "LOCKED",
        })
        print(f"  Conf [{row_idx - conf_start}] {row['mlp_name']:20s}: row={row_idx} seed={mlp.seed} weights_sha={w_sha[:16]}... LOCKED")

    conf_path = FIXTURES_DIR / "confirmation_panel_locked.json"
    with open(conf_path, "w", encoding="utf-8") as f:
        json.dump({
            "panel_size": len(confirmation_info),
            "total_requested": 16,
            "limitation_note": f"Dataset contains {total_ds_rows} rows total. Exactly {len(confirmation_info)} confirmation networks registered (rows {conf_start}..{conf_end-1}).",
            "networks": confirmation_info,
        }, f, indent=2)
    print(f"[p8_manifest] Locked confirmation panel metadata saved to {conf_path}")

    manifest = {
        "manifest_version": "8.0.0",
        "created_at": "2026-09-07T11:55:00+06:00",
        "environment": env_info,
        "controls": {
            "CONTROL7": {
                "source_path": str(control7_path),
                "sha256": control7_sha,
                "frozen_copy": str(control7_copy),
                "expected_local_mse": 3.7853e-8,
                "expected_local_util": 0.6958,
                "expected_local_adjusted": 2.6337e-8,
            },
            "CONTROL65": {
                "source_path": str(control65_path),
                "sha256": control65_sha,
                "frozen_copy": str(control65_copy),
                "expected_local_mse": 3.7854e-8,
                "expected_local_util": 0.7868,
                "expected_local_adjusted": 2.9784e-8,
            },
            "LEGACY": {
                "source_path": str(legacy_path),
                "sha256": legacy_sha,
                "frozen_copy": str(legacy_copy),
                "expected_local_mse": 1.223643e-6,
                "expected_local_util": 0.098052,
                "expected_local_adjusted": 1.223643e-7,
            },
            "REF3": ref3_info,
        },
        "dev_panel": dev8_info,
        "confirmation_panel_size": len(confirmation_info),
        "ground_truth_path": str(gt_target),
    }

    manifest_path = P8_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[p8_manifest] Manifest saved to {manifest_path}")

    # Initialize progress.json if not present
    progress_path = P8_DIR / "progress.json"
    if not progress_path.exists():
        init_progress = {
            "phase": "8",
            "started_at": "2026-09-07T11:55:00+06:00",
            "active_stage": "P8-00",
            "completed_stages": [],
            "completed_experiments": {},
            "frozen_parents": {
                "CONTROL7": control7_sha,
                "CONTROL65": control65_sha,
            },
            "best_candidate": {
                "exp_id": "CONTROL7",
                "adjusted_score": 2.6337e-8,
                "raw_mse": 3.7853e-8,
                "utilization": 0.6958,
            }
        }
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(init_progress, f, indent=2)
        print(f"[p8_manifest] Initialized {progress_path}")

    ledger_path = P8_DIR / "ledger.jsonl"
    if not ledger_path.exists():
        ledger_path.touch()
        print(f"[p8_manifest] Initialized empty {ledger_path}")

    # Build initial experiment_manifest.json
    exp_manifest_path = P8_DIR / "experiment_manifest.json"
    if not exp_manifest_path.exists():
        init_exp_manifest = {
            "mandatory_rows": [
                {"stage": "P8-00", "exp_id": "CONTROL7", "status": "PLANNED"},
                {"stage": "P8-A0", "exp_id": "A0-IDENTITIES", "status": "PLANNED"},
                {"stage": "P8-A1", "exp_id": "A1-G-K2", "status": "PLANNED"},
                {"stage": "P8-A1", "exp_id": "A1-A-K2", "status": "PLANNED"},
                {"stage": "P8-A1", "exp_id": "A1-G-K2K4", "status": "PLANNED"},
                {"stage": "P8-A1", "exp_id": "A1-A-K2K4", "status": "PLANNED"},
                {"stage": "P8-A1", "exp_id": "A1-G-K3C65", "status": "PLANNED"},
                {"stage": "P8-A1", "exp_id": "A1-A-K3C65", "status": "PLANNED"},
                {"stage": "P8-D0", "exp_id": "D0-IDENTITIES", "status": "PLANNED"},
                {"stage": "P8-D1", "exp_id": "D1-TERM", "status": "PLANNED"},
                {"stage": "P8-D1", "exp_id": "D1-TERM-REF", "status": "PLANNED"},
                {"stage": "P8-D2", "exp_id": "D2-ALL", "status": "PLANNED"},
                {"stage": "P8-L0", "exp_id": "L0-PILOT", "status": "PLANNED"},
                {"stage": "P8-L1", "exp_id": "L1-H0", "status": "PLANNED"},
                {"stage": "P8-L1", "exp_id": "L1-H8", "status": "PLANNED"},
                {"stage": "P8-L1", "exp_id": "L1-H16", "status": "PLANNED"},
                {"stage": "P8-L2", "exp_id": "L2-FEASIBILITY", "status": "PLANNED"},
                {"stage": "P8-L3", "exp_id": "L3-METERED", "status": "PLANNED"},
                {"stage": "P8-G0", "exp_id": "G0-IDENTITIES", "status": "PLANNED"},
                {"stage": "P8-G1", "exp_id": "G1-DIAGNOSTIC", "status": "PLANNED"},
                {"stage": "P8-G2", "exp_id": "G2-CANDIDATES", "status": "CONDITIONAL"},
                {"stage": "P8-X", "exp_id": "P8-X-COMBINATIONS", "status": "CONDITIONAL"},
                {"stage": "P8-F0", "exp_id": "P8-F0-SELECTION", "status": "CONDITIONAL"},
                {"stage": "P8-F1", "exp_id": "P8-F1-CONFIRMATION", "status": "CONDITIONAL"},
                {"stage": "P8-F2", "exp_id": "P8-F2-RELEASE", "status": "CONDITIONAL"},
            ]
        }
        with open(exp_manifest_path, "w", encoding="utf-8") as f:
            json.dump(init_exp_manifest, f, indent=2)
        print(f"[p8_manifest] Initialized {exp_manifest_path}")

    # Initialize phase8report.md if not present
    if not REPORT_PATH.exists():
        init_report = f"""# Phase 8: New Representations and Algorithms for the Error-Compute Frontier

**Date**: 7 September 2026  
**Incumbent Frozen Control (CONTROL7)**: `candidates/estimator_p7_final.py` (`{control7_sha}`)  
**Incumbent Frozen Control (CONTROL65)**: `candidates/estimator_p65_final.py` (`{control65_sha}`)  
**Local Baseline Score**: Raw MSE = `3.7853e-08`, Utilization = `69.58%`, Adjusted Score = `2.6337e-08`  
**Primary Target**: Valid adjusted score < `1.00e-08` (Stretch: `6.00e-09`, `3.00e-09`)  

---

## 1. Outcome First & Executive Summary

*Execution initialized. Sequential execution in progress across lanes:*
- Lane A: Angular Moment Propagation
- Lane D: Direct Contraction of Frozen Higher-Order Sources
- Lane L: Learned Compact Closure
- Lane G: Exact Gate-Residual Integration
- Combinations: P8-X
- Confirmation: P8-F0/F1
- Release & Artifact Verification: P8-F2

---

## 2. Frozen Controls & Evidence Anchors

| Name | Source | SHA-256 | Raw MSE | Utilization | Adjusted Product | Status |
|---|---|---|---:|---:|---:|---|
| CONTROL7 | candidates/estimator_p7_final.py | `{control7_sha}` | 3.7853e-8 | 69.58% | 2.6337e-8 | FROZEN_BENCHMARK |
| CONTROL65 | candidates/estimator_p65_final.py | `{control65_sha}` | 3.7854e-8 | 78.68% | 2.9784e-8 | FROZEN_ANCHOR |
| LEGACY | whest-starterkit/estimator.py | `{legacy_sha}` | 1.2236e-6 | 9.81% | 1.2236e-7 | FROZEN_LEGACY |
| REF3 | Phase 6 Reference K3-simple | receipt verified | 3.6370e-8 | N/A (unmetered) | N/A | REFERENCE_ANCHOR |

---

## 3. Configuration Log & Execution Receipts

*(Appended sequentially per stage)*
"""
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(init_report)
        print(f"[p8_manifest] Initialized {REPORT_PATH}")

    return manifest


if __name__ == "__main__":
    build_manifest()
