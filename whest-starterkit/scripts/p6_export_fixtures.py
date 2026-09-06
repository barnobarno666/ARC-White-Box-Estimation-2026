"""p6_export_fixtures.py

Exports the 8 panel MLPs (weights and ground-truth means) to a single fast
fixture: research/phase6/fixtures/panel_8mlp_weights.npz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP

DATASET_PATH = r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini"
OUT_DIR = REPO_ROOT / "research" / "phase6" / "fixtures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def export_fixtures():
    print(f"[p6_export_fixtures] Loading dataset from {DATASET_PATH}...")
    ds = load_dataset(DATASET_PATH, split="mini")
    proto, salt = resolve_seed_context(ds)

    names_in_ds = ds["mlp_name"]
    export_dict = {}

    for idx, name in enumerate(PANEL_MLP_NAMES):
        row_idx = names_in_ds.index(name)
        row = ds[row_idx]
        mlp = MLP.from_row(row, seed_protocol_version=proto, seed_salt=salt)
        print(f"  [{idx+1}/8] Exporting {name} (seed={mlp.seed})...")

        weights = np.stack([np.asarray(w, dtype=np.float32) for w in mlp.weights], axis=0)  # (16, 1024, 1024)
        gt_means = np.asarray(row["all_layer_means"], dtype=np.float32)  # (16, 1024)

        export_dict[f"weights_{idx}"] = weights
        export_dict[f"gt_means_{idx}"] = gt_means
        export_dict[f"seed_{idx}"] = np.int64(mlp.seed)
        export_dict[f"name_{idx}"] = np.str_(name)

    out_file = OUT_DIR / "panel_8mlp_weights.npz"
    np.savez_compressed(out_file, **export_dict)
    print(f"[p6_export_fixtures] Saved panel fixtures to: {out_file}")

if __name__ == "__main__":
    export_fixtures()
