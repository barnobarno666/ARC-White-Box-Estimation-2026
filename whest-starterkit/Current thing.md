CURRENT SUBMISSION SCORE: 3e-7
CURRENT BEST VALIDATION SCORE: 1.2319e-07 (TARGET < 1.24e-7 ACHIEVED)
score to beat: 3e-7, target < 2.5e-7 (Achieved 1.6500e-07), target < 1.24e-7 (Achieved 1.2319e-07)

## Phase 2 (1024x16, Budget 2^41) Validation Results (Fixed 8-MLP Panel)

| Method / Variant | Short Description | Raw Final MSE | Mean Score Mult | Adjusted Score | Max Residual Time | Failures | 8-MLP Wins vs Baseline | Decision |
|---|---|---|---|---|---|---|---|---|
| Baseline (Blend 75/25) | Gain Cov (0.75) + WMC (0.25, N=4038) | 3.1749e-06 | 0.1000 | 3.1749e-07 | 0.1309s | 0 | - | Baseline Reference |
| Variant N=4200 | Gain Cov (0.75) + WMC (0.25, N=4200) | 3.1575e-06 | 0.1000 | 3.1575e-07 | 0.1340s | 0 | 3W-5L | Keep (Minor Gain) |
| Mid-Network CV (Layer 8, beta=0.6) | Layer-8 Covariance Anchor CV on WMC (N=4200) | 3.3391e-06 | 0.1000 | 3.3391e-07 | 0.1178s | 0 | 2W-6L | REJECT (Linear Jacobian drift) |
| Pure Covariance | Gain Cov (1.00) alone | 4.3942e-06 | 0.1000 | 4.3942e-07 | 0.0712s | 0 | 0W-8L | REJECT (High baseline bias) |
| Pure WMC | WMC alone (N=4200) | 1.1524e-05 | 0.1000 | 1.1524e-06 | 0.0994s | 0 | 0W-8L | REJECT (High sampling variance) |
| Blend 80/20 (N=4200) | Gain Cov (0.80) + WMC (0.20, N=4200) | 3.2439e-06 | 0.1000 | 3.2439e-07 | 0.1236s | 0 | 3W-5L | REJECT |
| Blend 70/30 (N=4200) | Gain Cov (0.70) + WMC (0.30, N=4200) | 3.1517e-06 | 0.1000 | 3.1517e-07 | 0.1206s | 0 | 5W-3L | Keep (5W-3L) |
| Blend 65/35 (N=4200) | Gain Cov (0.65) + WMC (0.35, N=4200) | 3.2264e-06 | 0.1000 | 3.2264e-07 | 0.1494s | 0 | 3W-5L | REJECT |
| Hermite Cov (gamma=0.5) + WMC | 2nd-order Hermite Cov (0.70) + WMC (0.30, N=4200) | 2.9362e-06 | 0.1000 | 2.9362e-07 | 0.2455s | 0 | 8W-0L | KEEP (8W-0L Clean Sweep) |
| Hermite (g=0.45, w=0.72) | Hermite gamma=0.45, blend 0.72/0.28 (N=4200) | 2.9288e-06 | 0.1000 | 2.9288e-07 | 0.1724s | 0 | 8W-0L | KEEP (8W-0L Clean Sweep) |
| Hermite (g=0.50, w=0.75) | Hermite gamma=0.50, blend 0.75/0.25 (N=4200) | 2.9078e-06 | 0.1000 | 2.9078e-07 | 0.1761s | 0 | 8W-0L | KEEP (8W-0L, New Best 2.908e-7) |
| Hermite (g=0.48, w=0.75) | Hermite gamma=0.48, blend 0.75/0.25 (N=4200) | 2.9133e-06 | 0.1000 | 2.9133e-07 | 0.2224s | 0 | 8W-0L | KEEP (8W-0L) |
| Hermite (g=0.47, w=0.75) | Hermite gamma=0.47, blend 0.75/0.25 (N=4200) | 2.9162e-06 | 0.1000 | 2.9162e-07 | 0.2269s | 0 | 8W-0L | KEEP (8W-0L) |
| Hermite (g=0.46, w=0.75) | Hermite gamma=0.46, blend 0.75/0.25 (N=4200) | 2.9191e-06 | 0.1000 | 2.9191e-07 | 0.2359s | 0 | 8W-0L | KEEP (8W-0L) |
| Hermite (g=0.45, w=0.75) | Hermite gamma=0.45, blend 0.75/0.25 (N=4200) | 2.9223e-06 | 0.1000 | 2.9223e-07 | 0.3241s | 0 | 8W-0L | KEEP (8W-0L) |
| Hermite (g=0.38, w=0.75) | Hermite gamma=0.38, blend 0.75/0.25 (N=4200) | 2.9467e-06 | 0.1000 | 2.9467e-07 | 0.3467s | 0 | 8W-0L | KEEP (8W-0L) |
| Hermite (g=0.45, w=0.78) | Hermite gamma=0.45, blend 0.78/0.22 (N=4200) | 2.9439e-06 | 0.1000 | 2.9439e-07 | 0.1685s | 0 | 7W-1L | REJECT (Overshot optimal blend) |
| Hadamard Cubature + Hermite Cov | Sylvester Hadamard (N=4096) + Hermite Cov (0.75/0.25) | 2.9442e-06 | 0.1000 | 2.9442e-07 | 0.1901s | 0 | 4W-4L | REJECT (Anisotropic hypercube bias) |
| Tri-Blend Architecture | Hermite Cov (0.72) + WMC (0.14, N=2500) + Hadamard (0.14, N=2048) | 3.0696e-06 | 0.1017 | 3.1228e-07 | 0.3638s | 0 | 5W-3L | REJECT (Compute 10.17% > 10% floor) |
| Self-Calibrated Hermite Cov (alpha=0.13) | Hermite Cov with MC scale calibration + precision blend (alpha=0.13, N=4200) | 1.6500e-06 | 0.1000 | 1.6500e-07 | 0.1463s | 0 | 7W-1L | TARGET ACHIEVED (<2.5e-7) |
| EB-Shrunk Hermite Cov (gamma=0.20, lam=0.05) | EB Scale-Shrunk Hermite (g=0.20, lam=0.05, alpha=0.11, N=4200) | 1.2376e-06 | 0.1000 | 1.2376e-07 | 0.2215s | 0 | 8W-0L | TARGET ACHIEVED (<1.24e-7) |
| Prior-Calibrated Hermite (gamma=0.20, lam=0.0) | Pure Prior Scale (s_0=0.998319, g=0.20, alpha=0.11, N=4200) | 1.2319e-06 | 0.1000 | 1.2319e-07 | 0.1873s | 0 | 8W-0L | NEW CHAMPION (<1.24e-7) |


