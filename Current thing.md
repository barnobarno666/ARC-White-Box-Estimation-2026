CURRENT SUBMISSION ID: #329939 (Track: https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/329939) [P5-ChoSaul-lam05, local 1.2153e-07, 6W-2L vs champ] — prev #329769
CURRENT BEST VALIDATION SCORE: 1.2236e-07 (STAGE 3 CHAMPION: 8W-0L CLEAN SWEEP)
score to beat: 3e-7, target < 2.5e-7 (Achieved 1.6500e-07), target < 1.24e-7 (Achieved 1.2319e-07 -> 1.2236e-07)


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
| Lane A1: Thresh Quad (eta=0.2) | Exact threshold-aware phi(a_i)phi(a_j)/2 quadratic Hermite with eta_2=0.20 | 1.2453e-06 | 0.1000 | 1.2453e-07 | 0.2784s | 0 | 3W-5L vs Champ (8W-0L vs Base) | REJECT (3W-5L vs Champ) |
| Lane A1: Thresh Quad (eta=1.0) | Threshold-aware phi(a_i)phi(a_j)/2 quadratic Hermite with eta_2=1.00 | 1.2219e-06 | 0.1000 | 1.2219e-07 | 0.2375s | 0 | 5W-3L vs Champ (8W-0L vs Base) | POTENTIAL (5W-3L vs Champ) |
| Blended Hermite Champion (lam=0.20) | Dual-kernel centered + threshold-aware Hermite (lam=0.20) | 1.2236e-06 | 0.1000 | 1.2236e-07 | 0.2037s | 0 | 8W-0L vs Champ (8W-0L vs Base) | NEW CHAMPION (8W-0L Clean Sweep) |
| Blended Hermite (lam=0.50) | Dual-kernel centered + threshold-aware Hermite (lam=0.50) | 1.2171e-06 | 0.1000 | 1.2171e-07 | 0.1695s | 0 | 6W-2L vs Champ (8W-0L vs Base) | Lowest Panel Mean (6W-2L) |
| Blended Hermite (lam=0.35) | Dual-kernel centered + threshold-aware Hermite (lam=0.35) | 1.2194e-06 | 0.1000 | 1.2194e-07 | 0.1857s | 0 | 6W-2L vs Champ (8W-0L vs Base) | Robust Blend (6W-2L) |


## Phase 4 Validation Results (Target < 8.00e-8, Stretch < 4.00e-8, Baseline Control = 1.2236e-07)

| Method / Variant | Short Description | Raw Final MSE | Mean Score Mult | Adjusted Score | Diff vs Control | 8-MLP Wins vs Control | Max Residual Time | Decision |
|---|---|---|---|---|---|---|---|---|
| Phase 4 Control | Blended Dual-Kernel Hermite (lam=0.20, s0=0.998319, alpha=0.11, N=4200) | 1.2236e-06 | 0.1000 | 1.2236e-07 | +0.00% | Benchmark | 0.2281s | FROZEN CONTROL (ea8be822) |
| Pure WMC (P4-01) | Whitened Antithetic MC alone (N=4096) | 1.1779e-05 | 0.1000 | 1.1779e-06 | +862.6% | 0W-8L | 0.0910s | Baseline (Carrier screen) |
| Pure Hadamard (P4-01) | Scrambled Hadamard frames alone (N=4096, mu_R=31.992) | 9.9703e-06 | 0.1000 | 9.9703e-07 | +714.8% | 0W-8L | 0.0754s | 15.35% lower raw variance than WMC |
| Multi-Basis Bank K=4 (P4-02/03) | 4 Orthogonal bases reweighted (N=8192, util 12.5%) | 1.0195e-06 | 0.1250 | 1.2744e-07 | +4.15% | 2W-6L | 0.1620s | REJECT (Multiplier penalty 0.1250 > 0.1000) |
| Multi-Basis Bank K=8 (P4-02/03) | 8 Orthogonal bases reweighted (N=16384, util 25.0%) | 8.4891e-07 | 0.2500 | 2.1223e-07 | +73.4% | 0W-8L | 0.2840s | REJECT (Multiplier penalty 0.2500 doubles score) |
| Mid-Network CV (P4-05) | Perfect-Centering Oracle across layers 4,6,8,10,12,14 | 1.2176e-06 | 0.1000 | 1.2176e-07 | -0.49% | Oracle | 0.1520s | KILL & ARCHIVE (Headroom 0.49% << 20% gate, r<0.016) |
| Tri-Blend Portfolio (P4-07) | Hermite (0.88) + WMC2048 (0.06) + Had2048 (0.06) | 1.2173e-06 | 0.1000 | 1.2173e-07 | -0.52% | 4W-4L | 0.1840s | REJECT (Gain 0.52% << 10% gate, 4W-4L) |
| Higher Hermite (P4-07) | Cubic & Quartic Hermite terms (eta3=0.05, eta4=0.05) | 1.2235e-06 | 0.1000 | 1.2235e-07 | -0.01% | 4W-4L | 0.2310s | REJECT (Higher-order terms O(1/n^2) negligible) |
| Recursive Scale (P4-08) | Recursive per-layer scale rho=(0.998319)^(1/16) | 1.2236e-06 | 0.1000 | 1.2236e-07 | -0.00% | 4W-4L | 0.2290s | Equivalent to terminal prior (Diff: -0.00%) |
| Regime Calibration (P4-09) | Dead/On/Kink preactivation regime calibration | 1.2219e-06 | 0.1000 | 1.2219e-07 | -0.14% | 5W-3L | 0.2150s | REJECT (Gain 0.14% << 10% gate, 5W-3L) |
| Cho-Saul Layer 0 (P4-10) | Exact closed-form arc-cosine kernel at Layer 0 | 1.3744e-06 | 0.1000 | 1.3744e-07 | -0.33% vs Cov | 7W-1L vs Cov | 0.2350s | Accurate L0 kernel, but L15 error is multi-layer |
| Dual-Carrier Antagonistic (P4-11) | Oracle blend c + Had(2048) + QMC(2048) + WMC(2048) | 1.2184e-06 | 0.1000 | 1.2184e-07 | -1.08% | Oracle | 0.2450s | REJECT (Oracle headroom 1.08% << 10% gate) |
| Edgeworth Kurtosis (P4-12) | Gram-Charlier excess kurtosis correction (a^2-1)phi(a) | 1.3811e-06 | 0.1000 | 1.3811e-07 | +0.16% | 4W-4L | 0.2420s | REJECT (Recurrent compounding increases MSE) |
| Continuous Scaling (P4-13) | LOO polynomial scale calibration s(c) degrees 1-3 | 1.2310e-06 | 0.1000 | 1.2310e-07 | +0.60% | 2W-6L | 0.2310s | REJECT (Global s0 is minimax optimal; poly overfits) |
| Cho-Saul L0 + WMC (P4-14) | Exact Cho-Saul L0 kernel + WMC (alpha=0.110, N=4200) | 1.2206e-06 | 0.1000 | 1.2206e-07 | -0.25% | 6W-2L | 0.2380s | Minor gain (-0.25%, 6W-2L), but falls short of 10% gate |
| P5-ChoSaul-lam05 (N=4200) | Blended lam=0.50 + exact L0 arc-cosine (alpha=0.11) | 1.2153e-06 | 0.1000 | 1.2153e-07 | -0.68% | 6W-2L | 0.2055s | New lowest stable mean (-0.68%, 6W-2L), REJECT (<10% gate) |
| P5-ChoSaul-lam05 (N=4350) | Same stack, N=4350 at 10.00% util boundary | 1.2032e-06 | 0.1000 | 1.2032e-07 | -1.67% | 5W-3L | 0.1772s | REJECT (5W-3L, boundary util, seed-fragile noise) |




