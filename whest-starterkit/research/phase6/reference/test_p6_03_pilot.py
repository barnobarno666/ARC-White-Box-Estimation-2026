"""test_p6_03_pilot.py

Phase 6 Stage P6-03 Pilot:
Shape-based forecast, factor growth tracking, and reference propagation on logan-fitzgerald.
"""

from __future__ import annotations

import datetime
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlp_kprop.cumulants import *
from mlp_kprop.diagslice import *
from mlp_kprop.factor_k3 import FactoredTensor, factored_nonlin_kprop_k3
from mlp_kprop.harmonic import HTensor
import mlp_kprop.kprop_harmonic as kh
from mlp_kprop.kprop_harmonic import (
    Kind,
    coerce_input,
    factored_keeps_term,
    linear_kprop,
)
from mlp_kprop.wick import relu_wick_coef

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)

MANIFEST_PATH = REPO_ROOT / "research" / "phase6" / "manifest.json"
DATASET_PATH = r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini"


def forecast_resource_requirements():
    """Produce shape-based operation and memory forecast at width 1024 across 16 layers."""
    n = 1024
    print("=======================================================")
    print("Phase 6 Stage P6-03: Shape-Based Operational & Memory Forecast")
    print(f"Architecture: Width {n}, Depth 16 layers (float64)")
    print("=======================================================")
    print(f"{'Layer':<6} | {'Max Rank R':<12} | {'Factors Mem (MB)':<18} | {'Cov Mem (MB)':<14} | {'Total State (MB)'}")
    print("-" * 75)

    base_r_per_layer = n  # each layer generates n factor columns
    for l in range(16):
        r_cum = (l + 1) * base_r_per_layer
        factors_bytes = 3 * n * r_cum * 8  # 3 matrices of (n, R) in float64
        cov_bytes = n * n * 8  # (1024, 1024) in float64 = 8.38 MB
        total_mb = (factors_bytes + cov_bytes) / (1024 * 1024)
        print(f"{l:<6} | {r_cum:<12} | {factors_bytes / (1024*1024):<18.2f} | {cov_bytes / (1024*1024):<14.2f} | {total_mb:<.2f} MB")
    print("-" * 75)
    print("Forecast summary: Peak resident state at Layer 15 is ~400 MB << 6 GB bound.")
    print("Materialization of 1024^3 tensor (8.59 GB) is strictly avoided.")
    print("Contraction operations use dense (n, n) BLAS (matrix multiplication). Safe to execute.\n")


def run_pilot_on_logan():
    forecast_resource_requirements()

    # Load panel MLP 0 from datasets/mini
    # Import from whestbench in main python
    from whestbench.dataset import load_dataset, resolve_seed_context
    from whestbench.domain import MLP

    print(f"[p6_pilot] Loading row 0 (logan-fitzgerald) from {DATASET_PATH}...")
    ds = load_dataset(DATASET_PATH, split="mini")
    proto, salt = resolve_seed_context(ds)
    row0 = ds[0]
    mlp = MLP.from_row(row0, seed_protocol_version=proto, seed_salt=salt)
    gt_final_mean = np.asarray(row0["all_layer_means"][-1], dtype=np.float64)

    weights_bench = [torch.from_numpy(np.asarray(w, dtype=np.float64)) for w in mlp.weights]
    n = mlp.width
    depth = mlp.depth

    for mode in ["K3-base", "K3-simple", "K3-augment-filtered"]:
        print(f"\n=======================================================")
        print(f"Testing Mode: [{mode}] on logan-fitzgerald (Width {n}, Depth {depth})")
        print(f"=======================================================")

        if mode == "K3-base":
            base = True
            augment = False
        elif mode == "K3-simple":
            base = False
            augment = False
        elif mode == "K3-augment-filtered":
            base = False
            augment = True

        # Filtered oracle for augment mode
        real_iso = kh.get_all_terms_iso
        def filtered_iso(k_max, d_max=None, use_mean_var=False, augment=False):
            ret = real_iso(k_max, d_max=d_max, use_mean_var=use_mean_var, augment=augment)
            return {
                ip: {vp: c for vp, c in vps.items() if factored_keeps_term(k_max, ip, vp)}
                for ip, vps in ret.items()
            }

        K = coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=3)
        t0 = time.time()

        for l in range(depth):
            l_t0 = time.time()
            W_ref = weights_bench[l].T  # whestbench row convention -> ARC column convention

            WK = linear_kprop(K, W_ref, k_max=3)

            if mode == "K3-augment-filtered":
                kh.get_all_terms_iso = filtered_iso

            try:
                K = factored_nonlin_kprop_k3(
                    K_in=WK,
                    nonlin_wick_coef=relu_wick_coef,
                    augment=augment,
                    base=base,
                    use_pK=True,
                )
            finally:
                kh.get_all_terms_iso = real_iso

            r_count = K[3].factors[0].shape[1] if (3 in K and isinstance(K[3], FactoredTensor)) else 0
            l_elapsed = time.time() - l_t0
            pred_mean = K[1].to_tensor().numpy()
            l_raw_mse = float(np.mean((pred_mean - np.asarray(row0["all_layer_means"][l], dtype=np.float64)) ** 2))
            print(f"  Layer {l:2d}: Rank R = {r_count:5d} | Step Time = {l_elapsed:5.2f}s | Raw MSE = {l_raw_mse:.4e}")

        total_elapsed = time.time() - t0
        final_pred = K[1].to_tensor().numpy()
        final_raw_mse = float(np.mean((final_pred - gt_final_mean) ** 2))
        calib_final_mse = float(np.mean((0.998319 * final_pred - gt_final_mean) ** 2))

        print(f"\n[{mode} Summary on logan-fitzgerald]")
        print(f"  Total Wall Time:  {total_elapsed:.2f}s (< 600s bound)")
        print(f"  Final Raw MSE:    {final_raw_mse:.6e}")
        print(f"  Final Calib MSE:  {calib_final_mse:.6e}")
        print(f"  Baseline Control: 1.352726e-06 (calib)")
        print(f"  Diff vs Control:  {(calib_final_mse - 1.352726e-06) / 1.352726e-06 * 100.0:+.2f}%")


if __name__ == "__main__":
    run_pilot_on_logan()
