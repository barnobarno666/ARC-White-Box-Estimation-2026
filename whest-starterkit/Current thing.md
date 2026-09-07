CURRENT SUBMISSION ID: #330160 (https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/330160) [Phase 7 Finalist P7-02b-strassen1 / estimator_p7_final.py, official score 2.5335e-08] | Prev Graded #330088, #330018 (official score 4.35e-08, raw MSE 6.82e-08, util 63.73%), #330015 (7.1370e-08), #330016 (7.5121e-08)
CURRENT BEST LOCAL VALIDATION SCORE: 2.6337e-08 (Phase 7 Finalist P7-02b-strassen1 / estimator_p7_final.py — 8W-0L CLEAN SWEEP VS CONTROL65, -11.57% SCORE REDUCTION, UTILIZATION 69.58%)
score to beat: 2.9784e-08 (Phase 6.5 finalist CONTROL65) -> ACHIEVED 2.6337e-08 (Raw MSE 3.7853e-08, Util 69.58%, -11.57% adjusted score reduction; Official Score 2.5335e-08)



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
| **P6-FNP-K3 [6, 10, 13] (Candidate 1)** | Pure `flopscope.numpy` K3-simple with fast Step 5 & [6, 10, 13] prune (ret=0.62) | **7.3186e-08** | **0.9752** (97.52% util) | **7.1370e-08** | **-41.67%** | **8W - 0L** | < 0.20s | **NEW LEADERBOARD #1 CHAMPION (8W-0L Clean Sweep, -37.3% vs live #1)** |
| **P6-FNP-K3 [7, 11, 14] (Candidate 2)** | Pure `flopscope.numpy` K3-simple with fast Step 5 & [7, 11, 14] prune (ret=0.58) | **7.6274e-08** | **0.9849** (98.49% util) | **7.5121e-08** | **-38.61%** | **8W - 0L** | < 0.20s | **NEW LEADERBOARD #2 RUNNER-UP (8W-0L Clean Sweep, -34.0% vs live #1)** |


## Phase 6.5 Sequential Runbook Validation Results (Target < 4.00e-8, Stretch < 2.00e-8, Incumbent = 4.6655e-08 / 4.35e-08)

| Method / Variant | Short Description | Raw Final MSE | Mean Score Mult | Adjusted Score | Max Residual Time | Failures | Wins vs Incumbent | Decision |
|---|---|---|---|---|---|---|---|---|
| P65-00 Incumbent Baseline | Frozen Incumbent (`estimator_p6_k3_twofactor_terminal.py`, ret=0.62) | 7.3203e-08 | 0.6373 (63.73%) | 4.6655e-08 | 1.3759s | 0 (diag) | Benchmark (8T) | VERIFIED BASELINE (RESEARCH_ONLY) |
| **P65-02d Combined Exact** | Scalar-K4 & L0 spec + terminal split (ret=0.62) | **7.3203e-08** | **0.6334 (63.34%)** | **4.6369e-08** | 2.1034s | 0 (diag) | **8W - 0L** | **NEW EXACT_PARENT (8W-0L Clean Sweep, -8.6B FLOPs)** |
| P65-03a-r045 | Retention r=0.45, schedule [6, 10, 13] | 2.0263e-07 | 0.5349 (53.49%) | 1.0839e-07 | 1.5478s | 0 (diag) | 0W - 8L | REJECT (+133.8% score regression vs Exact Parent) |
| P65-03a-r055 | Retention r=0.55, schedule [6, 10, 13] | 1.1100e-07 | 0.5908 (59.08%) | 6.5575e-08 | 1.6790s | 0 (diag) | 0W - 8L | REJECT (+41.4% score regression vs Exact Parent) |
| P65-03a-r070 | Retention r=0.70, schedule [6, 10, 13] | 5.0707e-08 | 0.6860 (68.60%) | 3.4787e-08 | 1.8853s | 0 (diag) | 8W - 0L | STRONG (+25.0% score reduction vs Exact Parent) |
| **P65-03a-r080** | Retention r=0.80, schedule [6, 10, 13] | **4.0542e-08** | **0.7579 (75.79%)** | **3.0726e-08** | 2.1035s | 0 (diag) | **8W - 0L** | **NEW RETENTION CHAMPION (+33.7% score reduction, 8W-0L Clean Sweep)** |
| P65-03b-s5912 | Schedule [5, 9, 12] with r*=0.80 | 4.1849e-08 | 0.7432 (74.32%) | 3.1101e-08 | 2.4452s | 0 (diag) | 5W - 3L vs r080 | Close contender (Ratio 1.0122 vs s61013) |
| P65-03c-L6-m10 | Schedule [6, 10, 13], r=[0.70, 0.80, 0.80] | 4.3499e-08 | 0.7233 (72.33%) | 3.1462e-08 | 1.8195s | 0 (diag) | 4W - 4L vs r080 | REJECT (Ratio 1.0239) |
| P65-03c-L6-p10 | Schedule [6, 10, 13], r=[0.90, 0.80, 0.80] | 3.8480e-08 | 0.7925 (79.25%) | 3.0494e-08 | 2.2880s | 0 (diag) | 4W - 4L vs r080 | Minor gain, 4W-4L tie (Ratio 0.9925) |
| P65-03c-L10-m10 | Schedule [6, 10, 13], r=[0.80, 0.70, 0.80] | 4.3412e-08 | 0.7289 (72.89%) | 3.1644e-08 | 2.2625s | 0 (diag) | 1W - 7L vs r080 | REJECT (Ratio 1.0299) |
| **P65-03c-L10-p10** | Schedule [6, 10, 13], r=[0.80, 0.90, 0.80] | **3.7854e-08** | **0.7868 (78.68%)** | **2.9784e-08** | 1.6050s | 0 (diag) | **8W - 0L vs r080** | **NEW RANK_PARENT (8W-0L Clean Sweep, Sub-3.0e-8 Barrier Broken!)** |
| P65-03c-L13-m10 | Schedule [6, 10, 13], r=[0.80, 0.80, 0.70] | 4.1960e-08 | 0.7461 (74.61%) | 3.1305e-08 | 1.4709s | 0 (diag) | 1W - 7L vs r080 | REJECT (Ratio 1.0189) |
| P65-03c-L13-p10 | Schedule [6, 10, 13], r=[0.80, 0.80, 0.90] | 3.9059e-08 | 0.7697 (76.97%) | 3.0062e-08 | 1.9731s | 0 (diag) | 7W - 1L vs r080 | Strong runner-up (Ratio 0.9784) |
| P65-04a-normexact | Exact atom norm selector on RANK_PARENT | 4.0665e-08 | 0.7572 (75.72%) | 3.0792e-08 | 2.3742s | 0 (diag) | 0W - 8L vs Rank Parent | REJECT (Ratio: 1.0338) |
| P65-04a-exact | Exact individual-atom norm squared | 3.6755e-08 | 0.8096 (80.96%) | 2.9757e-08 | 2.9949s | 0 (diag) | 4W - 4L vs Rank Parent | REJECT (4W-4L, <6W gate) |
| P65-05a-r045-comp | Compensated L10 (r=0.45) | 4.6834e-08 | 0.6740 (67.40%) | 3.1565e-08 | 1.9730s | 0 (diag) | 3W - 5L vs Select Parent | REJECT (Ratio: 1.0598) |
| P65-05a-r045-uncomp | Matched uncompensated L10 (r=0.45) | 7.2664e-08 | 0.6565 (65.65%) | 4.7706e-08 | 2.3547s | 0 (diag) | 0W - 8L vs Select Parent | REJECT (Ratio: 1.6017) |
| P65-05a-r055-comp | Compensated L10 (r=0.55) | 4.3168e-08 | 0.7064 (70.64%) | 3.0493e-08 | 2.2320s | 0 (diag) | 2W - 6L vs Select Parent | REJECT (Ratio: 1.0238) |
| P65-05a-r055-uncomp | Matched uncompensated L10 (r=0.55) | 5.3915e-08 | 0.6855 (68.55%) | 3.6958e-08 | 2.3652s | 0 (diag) | 0W - 8L vs Select Parent | REJECT (Ratio: 1.2409) |
| P65-07-s13-e050 | Diagonal Q extension (start=13, eta=0.5) | 4.0298e-08 | 0.7598 (75.98%) | 3.0620e-08 | 2.0984s | 0 (diag) | 1W - 7L vs Analytic Parent | REJECT (Ratio: 1.0281) |
| P65-07-s13-e100 | Diagonal Q extension (start=13, eta=1.0) | 4.0159e-08 | 0.7598 (75.98%) | 3.0514e-08 | 2.9495s | 0 (diag) | 1W - 7L vs Analytic Parent | REJECT (Ratio: 1.0245) |
| P65-07-s11-e050 | Diagonal Q extension (start=11, eta=0.5) | 3.9947e-08 | 0.7618 (76.18%) | 3.0431e-08 | 3.2065s | 0 (diag) | 1W - 7L vs Analytic Parent | REJECT (Ratio: 1.0217) |
| P65-07-s11-e100 | Diagonal Q extension (start=11, eta=1.0) | 3.9568e-08 | 0.7618 (76.18%) | 3.0142e-08 | 3.4799s | 0 (diag) | 2W - 6L vs Analytic Parent | REJECT (Ratio: 1.0120) |
| P65-07-s07-e050 | Diagonal Q extension (start=7, eta=0.5) | 3.9756e-08 | 0.7657 (76.57%) | 3.0442e-08 | 1.9214s | 0 (diag) | 1W - 7L vs Analytic Parent | REJECT (Ratio: 1.0221) |
| P65-07-s07-e100 | Diagonal Q extension (start=7, eta=1.0) | 3.9422e-08 | 0.7657 (76.57%) | 3.0186e-08 | 2.9497s | 0 (diag) | 1W - 7L vs Analytic Parent | REJECT (Ratio: 1.0135) |
| P65-08-n0512-a005 | Online K3 CV (N=512, alpha=0.05) | 5.6348e-08 | 0.7657 (76.57%) | 4.3144e-08 | 2.7757s | 0 (diag) | 0W - 8L vs K4 Parent | REJECT (Ratio: 1.4486) |
| P65-08-n0512-a010 | Online K3 CV (N=512, alpha=0.10) | 1.0459e-07 | 0.7657 (76.57%) | 8.0082e-08 | 2.5143s | 0 (diag) | 0W - 8L vs K4 Parent | REJECT (Ratio: 2.6887) |
| P65-08-n0512-a020 | Online K3 CV (N=512, alpha=0.20) | 2.9838e-07 | 0.7657 (76.57%) | 2.2846e-07 | 1.6924s | 0 (diag) | 0W - 8L vs K4 Parent | REJECT (Ratio: 7.6707) |
| P65-08-n1024-a005 | Online K3 CV (N=1024, alpha=0.05) | 4.7499e-08 | 0.7735 (77.35%) | 3.6740e-08 | 1.6979s | 0 (diag) | 0W - 8L vs K4 Parent | REJECT (Ratio: 1.2336) |
| P65-08-n1024-a010 | Online K3 CV (N=1024, alpha=0.10) | 7.0748e-08 | 0.7735 (77.35%) | 5.4723e-08 | 1.9009s | 0 (diag) | 0W - 8L vs K4 Parent | REJECT (Ratio: 1.8373) |
| P65-08-n1024-a020 | Online K3 CV (N=1024, alpha=0.20) | 1.6612e-07 | 0.7735 (77.35%) | 1.2849e-07 | 1.8989s | 0 (diag) | 0W - 8L vs K4 Parent | REJECT (Ratio: 4.3142) |
| P65-09-L13-m05 | Final neighbor: perturb last event L13 retention to 0.75 | 4.1162e-08 | 0.7520 (75.20%) | 3.0953e-08 | 1.8086s | 0 (diag) | 0W - 8L vs Best Parent | REJECT (Ratio: 1.0392) |
| P65-09-L13-p05 | Final neighbor: perturb last event L13 retention to 0.85 | 3.9828e-08 | 0.7638 (76.38%) | 3.0419e-08 | 2.0463s | 0 (diag) | 1W - 7L vs Best Parent | REJECT (Ratio: 1.0213) |
| **P65-10-unrelaxed** | Finalist official unrelaxed run (`estimator_p65_final.py`) | **3.7854e-08** | **0.7868 (78.68%)** | **2.9784e-08** | 1.6429s | 8 (local teardown) | **8W - 0L vs Incumbent** | **PHASE 6.5 FINALIST VERIFIED (Packaged archive: submission_phase65.tar.gz)** |

## Phase 7 (1024x16, Budget 2^41) Validation Results (Fixed 8-MLP Panel)

| Method / Variant | Short Description | Raw Final MSE | Mean Score Mult | Adjusted Score | Max Residual Time | Failures | Wins vs Baseline | Decision |
|---|---|---|---|---|---|---|---|---|
| **P7-12-strassen-q13e05** | Full Strassen1 + Corrected diagonal Q (start=13, eta=0.5) | **3.7854e-08** | **0.6958 (69.58%)** | **2.6338e-08** | 5.3081s | 0 | **8W-0L** | **QUALIFIED** |
| **P7-11a-Extrapolation** | Paired-resolution extrapolation screen (c6n/c8n, k32/k64) | **N/A** | **N/A** | **N/A** | 0.0000s | 0 | **N/A** | **SKIPPED_GATE (FLOP budget overflow or MSE ~1e-6)** |
| **P7-11b-Edgeworth** | Bounded terminal Edgeworth screen on finalist regime | **3.7853e-08** | **0.6958 (69.58%)** | **2.6337e-08** | 0.0000s | 0 | **0W-8L** | **SKIPPED_GATE (Delta < 1e-12, gain < 1% gate)** |
| **P7-10-lin-N1024-a05** | Linear residual control N=1024, alpha=0.05 | **4.5004e-08** | **0.8025 (80.25%)** | **3.6114e-08** | 1.8107s | 0 | **0W-8L** | **EVALUATED** |
| **P7-10-quad-N1024-a05** | Quadratic residual control N=1024, alpha=0.05 | **3.9749e-08** | **0.8025 (80.25%)** | **3.1898e-08** | 1.9350s | 0 | **0W-8L** | **EVALUATED** |
| **P7-08-q11e05** | Corrected diagonal Q extension (start=11, eta=0.5) | **3.7863e-08** | **0.7868 (78.68%)** | **2.9792e-08** | 1.9116s | 0 | **2W-6L** | **EVALUATED** |
| **P7-08-q13e10** | Corrected diagonal Q extension (start=13, eta=1.0) | **3.7861e-08** | **0.7868 (78.68%)** | **2.9791e-08** | 1.8160s | 0 | **3W-5L** | **EVALUATED** |
| **P7-08-q13e05** | Corrected diagonal Q extension (start=13, eta=0.5) | **3.7859e-08** | **0.7868 (78.68%)** | **2.9789e-08** | 2.1832s | 0 | **3W-5L** | **EVALUATED** |
| **P7-07-tail-c6n-h05** | Cap 6n, randomized atom-tail sampling at L10 (head=50%) | **4.7594e-07** | **0.4811 (48.11%)** | **2.2899e-07** | 2.1253s | 0 | **0W-8L** | **EVALUATED** |
| **P7-07-tail-c8n-h05** | Cap 8n, randomized atom-tail sampling at L10 (head=50%) | **2.3592e-07** | **0.5477 (54.77%)** | **1.2922e-07** | 2.0003s | 0 | **0W-8L** | **EVALUATED** |
| **P7-06b-onetime-k64** | One-time Tucker rank 64 at layer 10 with exact slice restoration | **3.0091e+00** | **0.5285 (52.85%)** | **1.5903e+00** | 1.4867s | 0 | **0W-8L** | **EVALUATED** |
| **P7-06b-recur-k32-s3** | Recurring DS + Tucker rank 32, start layer 3, h=0 | **3.5880e-06** | **0.1426 (14.26%)** | **5.1155e-07** | 1.0554s | 0 | **0W-8L** | **EVALUATED** |
| **P7-06b-recur-k64-s3** | Recurring DS + Tucker rank 64, start layer 3, h=0 | **3.5996e-06** | **0.1707 (17.07%)** | **6.1445e-07** | 1.0897s | 0 | **0W-8L** | **EVALUATED** |
| **P7-06b-recur-k64-s7** | Recurring DS + Tucker rank 64, start layer 7, h=0 | **2.4042e-04** | **0.3117 (31.17%)** | **7.4946e-05** | 1.5256s | 0 | **0W-8L** | **EVALUATED** |
| **P7-05b-reset-p2-s3** | Preactivation reset period=2, start=3 (rank 2n post-reset) | **3.7859e-08** | **0.7868 (78.68%)** | **2.9788e-08** | 1.6605s | 0 | **4W-4L** | **EVALUATED** |
| **P7-05a-ds-h1-s3** | Fixed-memory DS summary closure (h=1 cohorts kept, start=3) | **1.1780e+06** | **0.2164 (21.64%)** | **2.5491e+05** | 0.9327s | 0 | **0W-8L** | **EVALUATED** |
| **P7-05a-ds-h0-s3** | Fixed-memory DS summary closure (h=0 cohorts kept, start=3) | **3.5209e-06** | **0.1265 (12.65%)** | **4.4533e-07** | 0.8638s | 0 | **0W-8L** | **EVALUATED** |
| **P7-04-reweight-c6n-r1e3** | Cap 6n, signed group reweighting at L10 (G=32, lam=1e-3) | **3.1940e-07** | **0.4890 (48.90%)** | **1.5619e-07** | 1.5640s | 0 | **0W-8L** | **EVALUATED** |
| **P7-04-reweight-c8n-r1e3** | Cap 8n, signed group reweighting at L10 (G=32, lam=1e-3) | **1.5966e-07** | **0.5576 (55.76%)** | **8.9026e-08** | 1.6983s | 0 | **0W-8L** | **EVALUATED** |
| **P7-03-normexact-c4n** | Cap 4n (4096 cols) with norm_exact selector at [6, 10, 13] | **6.6153e-07** | **0.4144 (41.44%)** | **2.7417e-07** | 1.6168s | 0 | **0W-8L** | **EVALUATED** |
| **P7-03-normexact-c6n** | Cap 6n (6144 cols) with norm_exact selector at [6, 10, 13] | **3.3026e-07** | **0.4810 (48.10%)** | **1.5887e-07** | 1.4523s | 0 | **0W-8L** | **EVALUATED** |
| **P7-03-exactpairs-c8n** | Cap 8n (8192 cols) with exact_pairs selector at [6, 10, 13] | **1.9891e-07** | **0.5477 (54.77%)** | **1.0895e-07** | 1.6668s | 0 | **0W-8L** | **EVALUATED** |
| **P7-03-normupper-c8n** | Cap 8n (8192 cols) with norm_upper selector at [6, 10, 13] | **1.7806e-07** | **0.5475 (54.75%)** | **9.7481e-08** | 1.3598s | 0 | **0W-8L** | **EVALUATED** |
| **P7-03-normexact-c8n** | Cap 8n (8192 cols) with norm_exact selector at [6, 10, 13] | **1.6907e-07** | **0.5476 (54.76%)** | **9.2588e-08** | 2.5548s | 0 | **0W-8L** | **EVALUATED** |
| **P7-02b-strassen1** | 1-Level Strassen on dimensions >= 512 (8-MLP Full) | **3.7853e-08** | **0.6958 (69.58%)** | **2.6337e-08** | 4.2869s | 0 | **8W-0L** | **STRASSEN EVALUATED (-11.57% score reduction, 8W-0L clean sweep)** |
| **P7-02b-strassen1-full** | Full Strassen on factor transport + repeated slices (>=512) | **3.8017e-08** | **0.6958 (69.58%)** | **2.6451e-08** | 3.9393s | 0 (diag) | **1W-0L vs CONTROL** | **NEW PHASE 7 FINALIST (-11.57% compute, >200B FLOPs saved)** |
| **P7-02b-strassen1-trans**| Strassen on factor transport only (>=512) | **3.8026e-08** | **0.7439 (74.39%)** | **2.8287e-08** | 2.9602s | 0 (diag) | **1W-0L vs CONTROL** | **STRASSEN GATE PASSED (-5.46% compute, 94.4B FLOPs saved)** |
| **P7-02a-combined** | Combined exact structured transport + reuse | **3.7854e-08** | **0.7868 (78.68%)** | **2.9784e-08** | 2.0857s | 0 | **0W-0L** | **PARITY** |
| **P7-02a-reuse** | Exact profile-identified operation reuse | **3.7854e-08** | **0.7868 (78.68%)** | **2.9784e-08** | 2.3088s | 0 | **1W-0L** | **PARITY** |
| **P7-09-Covariance** | Covariance gate audit (5.34% < 10%) | **0.0000e+00** | **0.0000 (0.00%)** | **0.0000e+00** | 0.0000s | 0 | **N/A** | **SKIPPED_GATE (<10% cost)** |
| **P7-02a-struc** | Exact structured newborn descriptors (A2, h, A_ds) | **3.7854e-08** | **0.7868 (78.68%)** | **2.9784e-08** | 2.3906s | 0 | **0W-0L** | **PARITY** |
| **CONTROL65 Baseline** | Frozen Phase 6.5 Finalist (estimator_p65_final.py) anchor | **3.7854e-08** | **0.7868 (78.68%)** | **2.9784e-08** | 1.6429s | 8 | **Benchmark (8T)** | **FROZEN CONTROL (b06d91bc)** |

## Phase 8 (1024x16, Budget 2^41) Validation Results (Fixed 8-MLP Panel)

| Method / Variant | Short Description | Raw Final MSE | Mean Score Mult | Adjusted Score | Max Residual Time | Failures | Wins vs CONTROL7 | Decision |
|---|---|---|---|---|---|---|---|---|
| **P8-00-ctrl7-smoke1** | P8-00-ctrl7-smoke1 | **3.8017e-08** | **0.6958 (69.58%)** | **2.6451e-08** | 3.3675s | 1 | **0W-0L vs CONTROL7** | **EVALUATED** |
| **A1-G-K2** | Gaussian input, incoming K3 slices zero, scalar K4 off | **4.1257e-06** | **0.1000 (10.00%)** | **4.1257e-07** | 1.6222s | 0 | **0W-8L vs CONTROL7** | **EVALUATED** |
| **A1-A-K2** | Angular input, incoming K3 slices zero, scalar K4 off | **3.4236e-06** | **0.1000 (10.00%)** | **3.4236e-07** | 0.5360s | 0 | **0W-8L vs CONTROL7** | **EVALUATED** |
| **A1-G-K2K4** | Gaussian input, incoming K3 zero, scalar K4 on (initial c4=0) | **4.0639e-06** | **0.1000 (10.00%)** | **4.0639e-07** | 0.6068s | 0 | **0W-8L vs CONTROL7** | **EVALUATED** |
| **A1-A-K2K4** | Angular input, incoming K3 zero, scalar K4 on (initial c4=-6/(n+2)) | **3.4303e-06** | **0.1000 (10.00%)** | **3.4303e-07** | 1.1005s | 0 | **0W-8L vs CONTROL7** | **EVALUATED** |
| **A1-G-K3C65** | Gaussian input, CONTROL65 retention/schedule, scalar K4 on | **3.7859e-08** | **0.7868 (78.68%)** | **2.9788e-08** | 1.8241s | 0 | **0W-8L vs CONTROL7** | **EVALUATED** |
| **A1-A-K3C65** | Angular input, CONTROL65 retention/schedule, scalar K4 on | **3.4859e-08** | **0.7868 (78.68%)** | **2.7429e-08** | 1.7239s | 0 | **2W-6L vs CONTROL7** | **EVALUATED** |
| **D1-TERM** | Terminal direct-source contraction from frozen A1-G-K2K4 pilot | **3.8288e-06** | **0.1164 (11.64%)** | **4.4577e-07** | 0.8051s | 0 | **0W-8L vs CONTROL7** | **EVALUATED** |
| **D1-TERM-A** | D1-TERM-A | **3.2211e-06** | **0.1164 (11.64%)** | **3.7502e-07** | 1.1539s | 0 | **0W-8L vs CONTROL7** | **EVALUATED** |
| **P8-X-strassen-angular-smoke1** | P8-X-strassen-angular-smoke1 | **9.4348e-01** | **0.1000 (10.00%)** | **9.4348e-01** | 0.0000s | 1 | **0W-0L vs CONTROL7** | **EVALUATED** |
| **P8-X-smoke2** | P8-X-smoke2 | **3.5131e-08** | **0.6958 (69.58%)** | **2.4443e-08** | 3.1102s | 0 | **0W-0L vs CONTROL7** | **EVALUATED** |
| **X-STRASSEN-ANGULAR** | X-STRASSEN-ANGULAR | **3.4860e-08** | **0.6958 (69.58%)** | **2.4255e-08** | 3.4167s | 0 | **8W-0L vs CONTROL7** | **EVALUATED** |
| **CONTROL7-CONF12** | CONTROL7-CONF12 | **3.8505e-08** | **0.6958 (69.58%)** | **2.6790e-08** | 4.1179s | 0 | **0W-0L vs CONTROL7-CONF12** | **EVALUATED** |
| **FINAL8-CONF12** | FINAL8-CONF12 | **3.5630e-08** | **0.6958 (69.58%)** | **2.4790e-08** | 2.9928s | 0 | **11W-1L vs CONTROL7-CONF12** | **CONFIRMED** |
