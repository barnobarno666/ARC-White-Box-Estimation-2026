"""p65_manifest.py

Phase 6.5 Stage P65-00: Manifest, environment snapshot, control freeze, and dataset verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

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

EXPECTED_INCUMBENT_SHA = "8498085d5b90646c6d65ce23087b2e9331980a77cb44bfa6d762341f68dd05ac"
HISTORICAL_CONTROL_SHA = "ea8be8222b827607f65d166118500b55fcaae883b79f64b29785b46d86501ca1"

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

DATASET_PATH = str(WORKSPACE_ROOT / "datasets" / "mini")
P65_DIR = REPO_ROOT / "research" / "phase6_5"
CONTROL_DIR = P65_DIR / "control"
FIXTURES_DIR = P65_DIR / "fixtures"
CONFIGS_DIR = P65_DIR / "configs"
PREDICTIONS_DIR = P65_DIR / "predictions"
DIAGNOSTICS_DIR = P65_DIR / "diagnostics"
RESULTS_DIR = P65_DIR / "results"
RELEASE_DIR = P65_DIR / "release"


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
    for d in (CONTROL_DIR, FIXTURES_DIR, CONFIGS_DIR, PREDICTIONS_DIR, DIAGNOSTICS_DIR, RESULTS_DIR, RELEASE_DIR):
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
        "caps": {
            "residual_wall_time_cap_s": 0.40,
            "predict_wall_time_cap_s": 120.0,
            "setup_wall_time_cap_s": 5.0,
            "max_memory_cap_gb": 6.0,
            "flop_budget_per_mlp": 2**41,
        },
    }
    print("[p65_manifest] Environment:")
    for k, v in env_info.items():
        print(f"  {k}: {v}")

    # 2. Controls & Incumbent
    incumbent_path = WORKSPACE_ROOT / "candidates" / "estimator_p6_k3_twofactor_terminal.py"
    if not incumbent_path.exists():
        raise FileNotFoundError(f"Missing incumbent: {incumbent_path}")
    incumbent_sha = sha256_file(incumbent_path)
    if incumbent_sha != EXPECTED_INCUMBENT_SHA:
        raise ValueError(
            f"Incumbent SHA mismatch!\nExpected: {EXPECTED_INCUMBENT_SHA}\nActual:   {incumbent_sha}"
        )
    print(f"[p65_manifest] Incumbent SHA verified: {incumbent_sha}")

    incumbent_copy_path = CONTROL_DIR / "estimator_incumbent.py"
    shutil.copy2(incumbent_path, incumbent_copy_path)

    # Historical control
    hist_ctrl_path = REPO_ROOT / "estimator.py"
    hist_ctrl_sha = sha256_file(hist_ctrl_path) if hist_ctrl_path.exists() else "unknown"
    print(f"[p65_manifest] Historical control SHA: {hist_ctrl_sha} (expected: {HISTORICAL_CONTROL_SHA})")

    # 3. Dataset Configuration & Panel Resolution
    print(f"[p65_manifest] Loading dataset from: {DATASET_PATH} (split='mini')...")
    ds = load_dataset(DATASET_PATH, split="mini")
    protocol_version, salt = resolve_seed_context(ds)

    dataset_cfg = {
        "path": DATASET_PATH,
        "split": "mini",
        "total_rows": len(ds),
        "seed_protocol_version": str(protocol_version),
        "seed_salt_hex": salt.hex() if isinstance(salt, bytes) else str(salt),
    }

    # Load Phase 6 manifest if present for verification
    p6_manifest_path = REPO_ROOT / "research" / "phase6" / "manifest.json"
    p6_panel_lookup = {}
    if p6_manifest_path.exists():
        with open(p6_manifest_path, "r", encoding="utf-8") as f:
            p6_data = json.load(f)
            for m in p6_data.get("panel", []):
                p6_panel_lookup[m["mlp_name"]] = m

    names_in_ds = ds["mlp_name"]
    panel_mlps: List[Dict[str, Any]] = []
    for expected_idx, name in enumerate(PANEL_MLP_NAMES):
        if name not in names_in_ds:
            raise ValueError(f"Required panel MLP {name} not found in dataset {DATASET_PATH} split mini!")
        row_idx = names_in_ds.index(name)
        row = ds[row_idx]
        mlp = MLP.from_row(row, seed_protocol_version=protocol_version, seed_salt=salt)

        if mlp.width != 1024 or mlp.depth != 16:
            raise ValueError(
                f"MLP {name} dimensions mismatch: width={mlp.width}, depth={mlp.depth} (expected 1024, 16)"
            )

        combined_h = hashlib.sha256()
        layer_hashes = []
        for l_idx, w in enumerate(mlp.weights):
            w_arr = np.asarray(w, dtype=np.float32)
            w_bytes = w_arr.tobytes()
            lh = sha256_bytes(w_bytes)
            layer_hashes.append(lh)
            combined_h.update(w_bytes)
        combined_weight_sha = combined_h.hexdigest()

        gt_means = np.asarray(row["all_layer_means"], dtype=np.float32)
        gt_sha = sha256_bytes(gt_means.tobytes())

        # Verify against Phase 6 manifest if available
        if name in p6_panel_lookup:
            exp_w_sha = p6_panel_lookup[name]["combined_weight_sha256"]
            exp_gt_sha = p6_panel_lookup[name]["ground_truth_means_sha256"]
            if combined_weight_sha != exp_w_sha:
                raise ValueError(f"Weight SHA mismatch for {name}: {combined_weight_sha} vs {exp_w_sha}")
            if gt_sha != exp_gt_sha:
                raise ValueError(f"Ground truth SHA mismatch for {name}: {gt_sha} vs {exp_gt_sha}")

        mlp_record = {
            "panel_order": expected_idx,
            "dataset_row_index": row_idx,
            "mlp_name": name,
            "input_seed": int(row.get("seed", -1)),
            "actual_estimator_seed": int(mlp.seed),
            "width": mlp.width,
            "depth": mlp.depth,
            "combined_weight_sha256": combined_weight_sha,
            "layer_weight_sha256": layer_hashes,
            "ground_truth_means_sha256": gt_sha,
            "ground_truth_final_mean_norm": float(np.linalg.norm(gt_means[-1])),
        }
        panel_mlps.append(mlp_record)
        print(f"  [{expected_idx+1}/8] {name}: seed={mlp.seed}, weight_sha={combined_weight_sha[:12]}..., gt_sha={gt_sha[:12]}...")

    manifest = {
        "manifest_version": "1.0",
        "phase": "phase6.5",
        "timestamp_utc": whestbench.metadata(ds).get("created_at_utc") if hasattr(whestbench, "metadata") else None,
        "environment": env_info,
        "controls": {
            "phase65_incumbent": {
                "file": "candidates/estimator_p6_k3_twofactor_terminal.py",
                "sha256": incumbent_sha,
                "copied_to": str(incumbent_copy_path.relative_to(REPO_ROOT)),
            },
            "historical_control": {
                "file": "estimator.py",
                "sha256": hist_ctrl_sha,
                "expected_sha256": HISTORICAL_CONTROL_SHA,
            },
        },
        "dataset": dataset_cfg,
        "panel": panel_mlps,
    }

    out_file = P65_DIR / "manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[p65_manifest] Successfully wrote panel manifest to: {out_file}")

    # Initialize progress.json if not present
    progress_file = P65_DIR / "progress.json"
    if not progress_file.exists():
        initial_progress = {
            "last_completed": "P65-00-init",
            "next_action": "P65-00-evaluate-incumbent",
            "research_parent": {
                "exp_id": "P65-00-incumbent",
                "sha256": incumbent_sha,
                "file": "research/phase6_5/control/estimator_incumbent.py",
            },
            "release_champion": {
                "exp_id": "P6-330018-incumbent",
                "sha256": incumbent_sha,
                "status": "HISTORICAL_OFFICIAL_#330018",
            },
        }
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(initial_progress, f, indent=2)
        print(f"[p65_manifest] Initialized {progress_file}")

    # Initialize ledger.jsonl if not present
    ledger_file = P65_DIR / "ledger.jsonl"
    if not ledger_file.exists():
        ledger_file.touch()
        print(f"[p65_manifest] Initialized {ledger_file}")

    return manifest


if __name__ == "__main__":
    build_manifest()
