"""p6_manifest.py

Phase 6 Stage P6-00: Establish reproducible starting point.
Builds and saves the complete panel manifest, environment versions, controls,
source hashes, and dataset configuration.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure starterkit root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flopscope
import scipy
import whestbench
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP

EXPECTED_CONTROL_SHA = "ea8be8222b827607f65d166118500b55fcaae883b79f64b29785b46d86501ca1"
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

DATASET_PATH = r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini"
P6_DIR = REPO_ROOT / "research" / "phase6"
CONTROL_DIR = P6_DIR / "control"
SAVED_CONTROL_RECEIPT = REPO_ROOT / "research" / "phase4_next" / "results" / "P4N-00_20260906_000232.json"


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_manifest() -> Dict[str, Any]:
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)

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

    env_info = {
        "whestbench": whestbench_ver,
        "flopscope": flopscope_ver,
        "numpy": numpy_ver,
        "scipy": scipy_ver,
        "python": python_ver,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    print("[p6_manifest] Environment:")
    for k, v in env_info.items():
        print(f"  {k}: {v}")

    # 2. Immutable Controls
    ctrl_estimator_path = REPO_ROOT / "estimator.py"
    if not ctrl_estimator_path.exists():
        raise FileNotFoundError(f"Missing immutable control: {ctrl_estimator_path}")
    ctrl_sha = sha256_file(ctrl_estimator_path)
    if ctrl_sha != EXPECTED_CONTROL_SHA:
        raise ValueError(
            f"Control estimator SHA mismatch!\nExpected: {EXPECTED_CONTROL_SHA}\nActual:   {ctrl_sha}"
        )
    print(f"[p6_manifest] Immutable control SHA verified: {ctrl_sha}")

    # Copy unmodified control to research/phase6/control
    ctrl_copy_path = CONTROL_DIR / "estimator_control.py"
    shutil.copy2(ctrl_estimator_path, ctrl_copy_path)

    # Secondary comparator: candidates/estimator_p46_laneE_skew_e025.py
    p46_candidate_path = REPO_ROOT / "candidates" / "estimator_p46_laneE_skew_e025.py"
    p46_sha = None
    if p46_candidate_path.exists():
        p46_sha = sha256_file(p46_candidate_path)
        shutil.copy2(p46_candidate_path, CONTROL_DIR / "estimator_p46_laneE_skew_e025.py")
        print(f"[p6_manifest] Secondary comparator p46 SHA: {p46_sha}")

    # Reference receipt
    receipt_info = {}
    if SAVED_CONTROL_RECEIPT.exists():
        with open(SAVED_CONTROL_RECEIPT, "r") as f:
            rcpt = json.load(f)
            receipt_info = {
                "path": str(SAVED_CONTROL_RECEIPT),
                "exp_id": rcpt.get("exp_id"),
                "timestamp": rcpt.get("timestamp"),
                "calculated_adjusted_score": rcpt.get("calculated_adjusted_score"),
                "official_adjusted_score": rcpt.get("official_adjusted_score"),
                "raw_final_mse": rcpt.get("raw_final_mse"),
                "mean_compute_utilization": rcpt.get("mean_compute_utilization"),
                "max_residual_wall_time_s": rcpt.get("max_residual_wall_time_s"),
                "n_failed_mlps": rcpt.get("n_failed_mlps"),
            }
        print(f"[p6_manifest] Reference receipt verified: adjusted={receipt_info.get('calculated_adjusted_score')}")
    else:
        print(f"[p6_manifest] WARNING: Reference receipt {SAVED_CONTROL_RECEIPT} not found!")

    # 3. Dataset Configuration & Panel Resolution
    print(f"[p6_manifest] Loading dataset from: {DATASET_PATH} (split='mini')...")
    ds = load_dataset(DATASET_PATH, split="mini")
    protocol_version, salt = resolve_seed_context(ds)

    dataset_cfg = {
        "path": DATASET_PATH,
        "split": "mini",
        "total_rows": len(ds),
        "seed_protocol_version": str(protocol_version),
        "seed_salt_hex": salt.hex() if isinstance(salt, bytes) else str(salt),
    }

    names_in_ds = ds["mlp_name"]
    for required_name in PANEL_MLP_NAMES:
        if required_name not in names_in_ds:
            raise ValueError(f"Required panel MLP {required_name} not found in dataset {DATASET_PATH} split mini!")

    panel_mlps: List[Dict[str, Any]] = []
    for expected_idx, name in enumerate(PANEL_MLP_NAMES):
        row_idx = names_in_ds.index(name)
        row = ds[row_idx]
        mlp = MLP.from_row(row, seed_protocol_version=protocol_version, seed_salt=salt)

        if mlp.width != 1024 or mlp.depth != 16:
            raise ValueError(
                f"MLP {name} dimensions mismatch: width={mlp.width}, depth={mlp.depth} (expected 1024, 16)"
            )

        # Compute weight hashes
        combined_h = hashlib.sha256()
        layer_hashes = []
        for l_idx, w in enumerate(mlp.weights):
            w_arr = np.asarray(w, dtype=np.float32)
            w_bytes = w_arr.tobytes()
            lh = sha256_bytes(w_bytes)
            layer_hashes.append(lh)
            combined_h.update(w_bytes)
        combined_weight_sha = combined_h.hexdigest()

        # Reference ground truth means
        gt_means = np.asarray(row["all_layer_means"], dtype=np.float32)
        gt_sha = sha256_bytes(gt_means.tobytes())

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
        "phase": "phase6",
        "timestamp_utc": whestbench.metadata(ds).get("created_at_utc") if hasattr(whestbench, "metadata") else None,
        "environment": env_info,
        "controls": {
            "immutable_control": {
                "file": "estimator.py",
                "sha256": ctrl_sha,
                "expected_sha256": EXPECTED_CONTROL_SHA,
                "copied_to": str(ctrl_copy_path.relative_to(REPO_ROOT)),
            },
            "p46_candidate": {
                "file": str(p46_candidate_path.relative_to(REPO_ROOT)) if p46_candidate_path.exists() else None,
                "sha256": p46_sha,
            },
            "saved_control_receipt": receipt_info,
        },
        "dataset": dataset_cfg,
        "panel": panel_mlps,
    }

    out_file = P6_DIR / "manifest.json"
    with open(out_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[p6_manifest] Successfully wrote panel manifest to: {out_file}")

    return manifest


if __name__ == "__main__":
    build_manifest()
