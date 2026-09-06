CURRENT SUBMISSION ID: #329946 (Track: https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/329946) [P46-skew-eta025, local 1.1894e-07, 8W-0L vs champ] — prev #329939, #329769
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
| P46-skew-eta05 | P5stack + sampled-skew final correction (eta=0.5 frozen) | 1.1929e-06 | 0.1000 | 1.1929e-07 | -2.51% | 7W-1L | 0.1951s | PROMISING (7W-1L); ablation needed |
| P46-skew-eta025 | Same, eta=0.25 (LOO-unanimous 8/8 folds) | 1.1894e-06 | 0.1000 | 1.1894e-07 | -2.80% | 8W-0L | 0.1947s | BEST STABLE MEAN, 8W-0L, zero regressions; REJECT promotion (<10% gate) |
| P46-skew-eta075 | Same, eta=0.75 | 1.2259e-06 | 0.1000 | 1.2259e-07 | +0.18% | 4W-4L | 0.2027s | REJECT (overdose; sarah +5.97%) |


## Phase 5 Continuation (P4N) Validation Results (Target < 1.00e-8, Control = 1.2236e-07)

| Method / Variant | Short Description | Raw Final MSE | Mean Score Mult | Adjusted Score | Diff vs Control | 8-MLP Wins vs Control | Max Residual Time | Decision |
|---|---|---|---|---|---|---|---|---|
| P4N-00 Control Replay | Baseline Control Replay (Dual-Kernel Hermite, s0, WMC N=4200) | 1.2236e-06 | 0.1000 | 1.2236e-07 | +0.00% | Benchmark (8T) | 0.1746s | VERIFIED CONTROL REPLAY |
| P4N-01 Mapped Oracle (L14) | Mapped Weight-Aware Control (L14 oracle center, N=4096) | 1.1017e-06 (fused) | 0.1000 | 1.1017e-07 | -9.97% | 8W-0L (Oracle) | N/A (diag) | GATE PASSED (8/8 MLPs >30% raw cut, 95.2% max) |
| P4N-01 Ridge L14-r256 | Multivariate Ridge (r=256 cross-basis, oracle center) | 2.2176e-06 (raw) | 0.1000 | 1.1186e-07 | -8.59% | 8W-0L (Oracle) | N/A (diag) | CONFIRMED HEADROOM (87.5% var cut, center error bottleneck) |
| P4N-02 Full Gaussian Cov | Exact Gauss-Legendre quadrature (n=4, 8, 16) across all depths | 3.593e-06 (raw) | 0.1000 | 1.448e-07 (calib) | +1.8% vs Hermite | 2W-6L vs Hermite | N/A (diag) | REJECT (<10% gate; quad matches Hermite; blend incurs FLOP penalty) |
| P4N-04 Early Sample Repair | Layer-0 Mean Matching & Covariance Transport (N=2048) | 2.5996e-05 (raw) | 0.1000 | 1.3063e-07 (fused) | +6.75% vs Ctrl | 0W-8L vs Ctrl | N/A (diag) | ARCHIVE TESTED BRANCH (24.4% raw cut, but downstream bias breaks LOO blend) |
| P4N-05 Exact Nonlinear Controls | First-layer q(X)=ReLU(XW0) & Shifted Even Ridge Features | 1.6700e-05 (raw) | 0.1000 | 1.2877e-07 | +5.24% | 1W-7L | N/A (diag) | ARCHIVE TESTED BRANCH (L0 features decorrelated with L15; <20% gate) |
| P4N-06 Response-Weighted Center Opt | Optimal scaling s_k* (L11,13,14) & Pilot Shrinkage (Outer LOO) | 3.2500e-06 (raw) | 0.1000 | 1.2441e-07 (fused) | -1.67% vs Ctrl | 2W-6L vs Ctrl | N/A (diag) | GATE FAILED (Cuts center loss by 77.2%, but residual center bias 6.5e-7 prevents gain) |
| P4N-07 Cheap Surrogate + Coupled Res | Two-level estimator mu = mean_A[g] + mean_B[f-g] (Prefix 7,11,13 & Pruned 512,256) | 2.0188e-05 (raw) | 0.1000 | 1.2655e-07 (fused) | +3.42% vs Ctrl | 0W-8L vs Ctrl | N/A (diag) | GATE FAILED (-74.3% vs Ord; Stream A sampling costs 56-94% of f with full Vg) |
| P4N-08 Cond Gaussian Mixtures | 1D Gauss-Hermite mixture (n=3,5) on sensitivity, response mode, random | 1.3738e-06 (calib) | 0.1623 | 2.1126e-07 (scored) | +72.65% vs Ctrl | 0W-8L vs Ctrl | N/A (diag) | GATE FAILED (+0.37% center reduction << 30% gate; FLOP penalty doubles score) |
| P4N-09 Nontrivial Stein Controls | Exact zero-mean q_t(X)=s*ReLU(s-t)-1_{s>t} across 16 configs (t in {0,0.5,1,2}) | 1.8720e-05 (raw) | 0.1000 | 1.2965e-07 (fused) | -0.18% vs Ctrl | 5W-3L vs Ctrl | N/A (diag) | GATE FAILED (Var red -2.52% to -6.93% << 20% gate; pilot fitting injects noise) |
| P4N-10 Certified Carrier & Calibration | Scrambled Had/Haar/Sobol + Signed Exact L0 Moment Calibration (M=32,64,128) | 1.5436e-05 (raw) | 0.1000 | 1.2783e-07 (fused) | +2.45% vs Ctrl | 2W-6L vs Ctrl | N/A (diag) | GATE FAILED (MUB coherence 5.2x limit; Had/Sobol/weights regress vs Gaussian) |
| P4N-11 1D Conditional Integration | Gauss-Hermite line quad (K=8,16,32) along response/W0/rand directions | 2.4015e-04 (raw) | 0.1000 | 3.9829e-07 (scored) | +220.1% vs Ctrl | 0W-8L vs Ctrl | N/A (diag) | GATE FAILED (+635% to +3404% error vs matched Ord; starves remaining 1023 dims) |
| P4N-12 Cumulants & Response Modes | Terminal Edgeworth Skew/Kurt + Low-Rank Modes + Center-Corrected Mapped CV | 1.3693e-06 (raw) | 0.1000 | 1.2814e-07 (fused) | -1.34% vs Ctrl | 5W-3L vs Ctrl | N/A (diag) | GATE PASSED (Center error cut by 82.74% >= 30% gate; deployable 5W-3L) |
| P4N-13 Complementary Combinations | Convex Multi-Branch Blend LOO (Analytic + MC Sampler + Mapped CV + Edgeworth) | 1.2584e-06 (raw) | 0.1000 | 1.2584e-07 (fused) | -3.11% vs Ctrl | 7W-1L vs Ctrl | N/A (diag) | GATE FAILED (Oracle headroom 3.11% << 10% gate, LOO deploy gain 3.11% < 5% gate) |
| P4N-14 Execution Optimization | Metered FLOP audit, array compaction, 9.81% multiplier floor defense | 1.2236e-06 | 0.1000 | 1.2236e-07 | +0.00% | Benchmark | 0.1746s | OPTIMIZED & VERIFIED (Zero penalty, clean 9.81% util) |
| P4N-15 Final Verification & Synthesis | Bit-identical replay of frozen champion (ea8be822...), 8/8 contract pass | 1.2236e-06 | 0.1000 | 1.2236e-07 | +0.00% | 8T (Bit-identical) | 0.1746s | CAMPAIGN COMPLETE: Retain champion estimator.py |


## Phase 6 Validation Results (Target < 1.20e-8, Stretch < 5.00e-9, Control = 1.2236e-07)

| Method / Variant | Short Description | Raw Final MSE | Mean Score Mult | Adjusted Score | Diff vs Control | 8-MLP Wins vs Control | Max Residual Time | Decision |
|---|---|---|---|---|---|---|---|---|
| P6-00 Control Replay | Bit-identical replay of frozen immutable control (ea8be822...), offset 0 | 1.2236e-06 | 0.1000 | 1.2236e-07 | +0.00% | Benchmark (8T) | 0.2084s | VERIFIED CONTROL REPLAY |
| P6-03 Reference K3-base | Uncompressed K3 cumulants (base=True, R=32768 at L15, isolated .venv_ref) | 4.2782e-07 | 0.1000 (nom) | 4.2782e-08 (nom) | -65.04% | 8W-0L | N/A (unmetered ref) | GATE PASSED (8W-0L, 65% raw error cut) |
| P6-03 Reference K3-simple | Uncompressed K3 cumulants + 4th-order state (base=False, isolated .venv_ref) | 3.6370e-08 | 0.1000 (nom) | 3.6370e-09 (nom) | -97.03% | 8W-0L | N/A (unmetered ref) | DECISIVELY PASSED (<1e-7 gate: representation breaks plateau) |
| P6-04 Norm-1/2 (50% retention) | Downstream Factor Compression: norm ranking at L3,7,11,14 (R=10624) | 5.7059e-07 | 0.1000 (nom) | 5.7059e-08 (nom) | -53.37% | 8W-0L | N/A (ref env) | GATE PASSED (67.6% rank cut, 8W-0L; influence failed 2x gate, retain norm) |
| P6-05 Omitted Diagrams | Tiny fixture diagnostic (171 terms, 164 kept, 7 dropped; 3.1% effect) | N/A (diag) | N/A | N/A | N/A | N/A | N/A | DIAGNOSTIC COMPLETE (K3-simple is minimax optimal; contractions O(n^4)) |
| P6-06 Mapped CV (Offline P6 Oracle) | Layer-14 Mapped CV with precomputed P6 center loaded from disk (oracle check) | 4.3702e-08 | 0.1000 (nom) | 4.3702e-09 (nom) | -96.43% | 8W-0L (Oracle) | N/A (offline sim) | ORACLE PROOF (Unblocks mapped CV if center known, but illegal for deployment) |
| P6-06 Deployable Candidate | Official whest run of candidates/estimator_p6_mapped_cv.py (pure flopscope.numpy) | 1.3833e-06 | 0.1000 (9.81%)| 1.3833e-07 | +13.05% | 1W-7L | 0.1405s | REJECT (1W-7L; online K3 center unresolved in flopscope; base center causes regression) |
| P6-07 Multi-Salt Oracle Headroom | Multi-salt oracle simulation across offsets 0, 1337, 8888 using offline P6 centers | 4.3594e-08 | 0.1000 (nom) | 4.3594e-09 (nom) | -96.44% | 24W-0L (Oracle) | N/A (offline sim) | THEORETICAL HEADROOM ONLY (Cannot deploy without online center computation) |


