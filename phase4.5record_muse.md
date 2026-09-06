# Phase 4.5 Experiment Log: Stacking Micro-Wins Toward e-8 (Muse Session, 2026-09-05)

This document records every experiment, ablation, and oracle diagnostic executed
during the Phase 4.5 follow-through session: a systematic hunt for the e-8 range
(adjusted score `< 8.00e-08`) after Phase 4 closed all four lanes with negative
results. Format mirrors `PHASE4_RECORD.md`.

## Operational Anchors & Phase 4.5 Reference Points

* **Immutable Phase 4.5 Control**: `whest-starterkit/estimator.py` (Blended Dual-Kernel Hermite $\lambda=0.20$, $s_0=0.998319$, $\alpha=0.110$, $N=4200$)
* **Control SHA-256**: `ea8be8222b827607f65d166118500b55fcaae883b79f64b29785b46d86501ca1`
* **Control Adjusted Score**: `1.223643e-07` (Raw MSE: `1.223643e-06`, Multiplier: `0.1000`)
* **Control Compute Utilization**: `9.81%` (strictly under the 10% floor)
* **Control Max Residual Time**: `0.1830s` (re-measured this session; Limit: `0.4000s`)
* **Phase 4.5 Primary Goal**: `< 8.00e-08` (~34.6% reduction from Control)
* **Phase 4.5 Stretch Goal**: `< 4.00e-08` (~67.3% reduction from Control)
* **Promotion Gate (unchanged from Phase 4)**: Adjusted score improvement $\ge 10\%$ over Control (i.e. $< 1.1013\times 10^{-7}$), $\ge 6/8$ wins vs Control, 0 failures, max residual time $< 0.35$s, no worst-MLP regression $> 10\%$, passes 3 sampler salts.
* **Verification Budget**: 8-MLP `mini` panel only (per user constraint).

---

## Phase 4.5 Experiment Index & Results Summary

| Exp ID | Lane / Family | Description | Raw Final MSE | Score Mult | Adjusted Score | vs Control Diff | 8-MLP Wins vs Control | Max Res Time | Decision |
|---|---|---|---|---|---|---|---|---|---|
| P5-00 | Infrastructure | Control Reproduction (`run_eval.py`) | 1.2236e-06 | 0.1000 | 1.2236e-07 | +0.00% | 8T (identical) | 0.1830s | FROZEN CONTROL |
| P5-01 | Analytic / C4 | Layer-wise error growth + shared additive residual law r(a,sigma) | 1.3789e-06 (cov branch) | - (offline) | - (offline) | L14/L15 = 0.981 | Oracle | - | KILL late-jump + C4 |
| P5-02a | Sampling CV | Per-neuron split-sample (unbiased) linear control, cross-fitted | 1.1523e-05 (sampler only) | - (offline) | - (offline) | +0.01% vs raw WMC | 8-MLP raw screen | - | KILL unbiased per-neuron CV |
| P5-02b | Lane C2 | Active-subspace top-4 linear R2 (linearized product map) | - (offline) | - (offline) | - (offline) | R2 = 0.0013-0.0014 | Oracle | - | KILL Lane C2 |
| P5-02c | Sampling CV | Stein rank-8 input control (exact-zero-mean center) | 1.1524e-05 (sampler only) | - (offline) | - (offline) | -0.00% | 8-MLP raw screen | - | KILL Stein-input CV |
| P5-03 | Lane D | Full-depth exact Cho-Saul arc-cosine kernel (offline prototype) | 4.9672e+00 (buggy) | - (offline) | - (offline) | +3.6e8% (formula bug) | Diagnostic | - | ARCHIVE as non-deployable |
| P5-04a | Sampling CV | Global single-beta split-sample control (pooled) | 1.1525e-05 (sampler only) | - (offline) | - (offline) | -0.01% | 8-MLP raw screen | - | KILL global unbiased CV |
| P5-04b | Carriers | WMC vs Hadamard lawful selection via analytic distance | - (offline) | - (offline) | - (offline) | Invalid (N bug: Had 2048 vs WMC 4200) | Diagnostic | - | NO NEW VERDICT (P4-01/P4-07 stand) |
| P5-05 | Stacked deployable | Blended lam=0.50 + Exact Cho-Saul L0, N=4200 (`estimator_p5_chosaul_lam05.py`, `93ee3c28...`) | 1.2153e-06 | 0.1000 | 1.2153e-07 | -0.68% | 6W-2L vs Champ | 0.2055s | RECORD best stable mean; REJECT promotion |
| P5-06 | Stacked deployable | Same stack at N=4350 (10.00% util boundary, `2319b7cd...`) | 1.2032e-06 | 0.1000 | 1.2032e-07 | -1.67% | 5W-3L vs Champ | 0.1772s | REJECT (seed-fragile, boundary util) |
| P5-SUB | Submission | Package + validate + AIcrowd submit of P5-05 | - | - | - | - | - | validate 130ms | SUBMITTED #329939 |

---

## Detailed Experiment Reports

### Experiment P5-00: Phase 4.5 Control Reproduction
* **Date/time**: 2026-09-05
* **Lane and method family**: Infrastructure / Calibration
* **Hypothesis**: `estimator.py` reproduces `1.223643e-07` identically under `scripts/run_eval.py` before any new research.
* **Mathematical mechanism**: Dual-Kernel Blended Hermite Covariance ($\lambda=0.20$) + Whitened Antithetic MC ($N=4200$).
* **Candidate file and SHA-256**: `estimator.py` (`ea8be8222b827607f65d166118500b55fcaae883b79f64b29785b46d86501ca1`)
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`, identical)
* **Parameters frozen before run**: $\lambda=0.20, s_0=0.998319, \alpha=0.110, N=4200$.
* **MLP panel and sampler salts**: Official 8-MLP panel (`mini` split, Seed Protocol 4.0), official seed.
* **Control adjusted score**: `1.223643e-07`
* **Candidate adjusted score**: `1.223643e-07`
* **Relative adjusted improvement**: `+0.00%`
* **Raw final-layer MSE**: `1.223643e-06`
* **Compute utilization and multiplier**: `9.81%`, Multiplier: `0.1000`
* **Max residual wall time**: `0.1830s`
* **Failures**: 0
* **Per-MLP Scores**: all 8 TIE vs champion file (identical binary).
* **Evidence label**: `[measured]`
* **Decision**: Frozen Phase 4.5 Control.
* **Exact gate applied**: P5-00 control benchmark freeze.
* **Next experiment justified**: P5-01 layer-wise / residual-law diagnostics.

### Experiment P5-01: Layer-Wise Error Growth & Shared Residual Law (C4)
* **Date/time**: 2026-09-05
* **Lane and method family**: Analytic / Lane C4 (small shared mechanistic residual law)
* **Hypothesis**: (a) Analytic error concentrates in the last layer(s), so jumping from an analytic L14 state with massive final-layer sampling wins big; (b) the deterministic residual $r = y - s_0 c$ has a stable shared law in $(a, \sigma, \Phi(a), c)$ that a $\le 2$-parameter global correction monetizes under LOO.
* **Mathematical mechanism**:
  - Numpy replica of blended Hermite covariance ($\lambda=0.20$); per-layer scaled MSE $\|s_0 \mu_l - y^*_l\|^2/1024$ vs `all_layer_means` truth.
  - Pooled residual bins Dead ($a<-2.5$) / Kink / On ($a>+2.5$); correlations of $r$ with features; LOO fit $r \approx b_0 + b_1 c$ (2 params, 7-train/1-test).
* **Candidate file and SHA-256**: `scripts/diag_p5_layerwise.py` (offline diagnostic, no deployable)
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Parameters frozen before run**: $s_0 = 0.998319$, 8 MLPs pooled (8192 neurons).
* **Results Measured**:
  - Mean per-layer MSE: L0=`9.00e-07` (note: includes terminal $s_0$ mis-scaling of early layers; unscaled L0 is ~1e-10 per prior runs), L8=`1.04e-06`, L12=`1.27e-06`, L13=`1.34e-06`, L14=`1.3528e-06`, L15=`1.3789e-06`. Ratio L14/L15 = **0.981**.
  - Residual: std=`1.1732e-03`, mean=`-4.9334e-05`, $|c|$ rms=`0.9751`.
  - Bins: dead n=1932, MSE contrib `1.23e-09`, energy frac 0.000; kink n=4326, frac 0.548; on n=1934, frac 0.452.
  - Correlations: corr(r,a)=`+0.0044`, corr(r,sigma)=`-0.0292`, corr(r,Phi)=`+0.0135`, corr(r,c)=`+0.0071`, corr(r,a*phi)=`+0.0068`.
  - LOO $b_0+b_1 c$: mean base=`1.3789e-06`, corrected=`1.3840e-06`, gain=`-0.37\%` (hurts; 2 of 8 folds regress >1%).
* **Evidence label**: `[measured]`
* **Interpretation**: Analytic error accumulates gradually over depth; the last layer adds only ~1.9%. A perfect L14 jump ceilings at ~1.9% fused gain before sampling noise — kills the late-jump family. Residual is isotropic w.r.t. all tested white-box features; the only stable structure is dead≈0 (already exploited).
* **Decision**: KILL late-jump sampling and shared additive residual law (Lane C4).
* **Exact gate applied**: Oracle headroom <10% (1.9%) / deployable LOO negative.
* **Next experiment justified**: P5-02 unbiased (analytic-free) control variates — the A4-correlation fix.

### Experiment P5-02: Unbiased Split-Sample Controls, Active Subspace, Stein-Input
* **Date/time**: 2026-09-05
* **Lane and method family**: Sampling control variates (Lane A4 follow-up) / Lane C2 active subspace / Stein zero-mean controls
* **Hypothesis**: Prior final-layer Hermite CV cut raw B by 65% but its analytic center correlated with covariance error, zeroing fused gain. A split-sample center ($E[\bar z_A - \bar z_B] \equiv 0$, no analytic contact) keeps the variance cut while staying orthogonal to the analytic branch. Separately, top-4 adjoint input directions explain $\ge 15\%$ of final variance (C2 gate), and rank-8 Stein input controls cut B for free.
* **Mathematical mechanism**:
  - (a) Per-neuron cross-fitted linear CV: $\hat m_A = \bar y_A - \hat\beta_B(\bar z_A - \bar z_B)$, $\hat\beta$ clipped to $[0,1]$, symmetrized A/B, $N=4200$ whitened samples, numpy offline.
  - (b) Active subspace: linearized map $P = W_0 \cdots W_{15}$, top-4 right singular vectors $V_4$, $R^2$ of final centered activations on $X V_4$ ($N=4000$).
  - (c) Stein-input: rank-8 PCA-ridge regression of $Y$ on $X$, cross-fitted halves correction.
* **Candidate file and SHA-256**: `scripts/diag_p5_unbiased.py` (offline diagnostic)
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Results Measured**:
  - (a) Per-neuron unbiased CV: mean raw=`1.1524e-05`, CV=`1.1523e-05`, reduction=`+0.01\%` (per-MLP ratios 0.998–1.001). Zero.
  - (b) Active subspace: logan-fitzgerald total var=`7.8476e-02`, R2_top4=`0.0013`; william-graves total var=`8.5778e-02`, R2_top4=`0.0014`. Both $\ll 0.15$.
  - (c) Stein rank-8: ratio exactly `1.000` on all 8 MLPs; mean raw=`1.1524e-05`, stein=`1.1524e-05`, reduction=`-0.00\%`.
* **Evidence label**: `[measured]`
* **Interpretation**: Split-center variance ($\mathrm{Var}(\bar z_A - \bar z_B) = 2\mathrm{Var}(\bar z)$) plus per-neuron $\hat\beta$ estimation noise from 2100 samples exactly consume the textbook control gain. Deep chaos (16-layer He ReLU) erases input-output linear signal ($R^2 \sim 10^{-3}$), so no input-anchored control can help.
* **Decision**: KILL unbiased per-neuron CV, KILL Lane C2 active subspace, KILL Stein-input CV.
* **Exact gate applied**: Deployable gain <3% with no strong oracle (all ~0%).
* **Next experiment justified**: P5-03 full-depth exact kernel (last deterministic analytic stone unturned).

### Experiment P5-03: Full-Depth Exact Cho-Saul Arc-Cosine Kernel
* **Date/time**: 2026-09-05
* **Lane and method family**: Lane D / Analytical Kernel Refinement (exact, all 16 layers)
* **Hypothesis**: Replacing the Taylor (linear + dual-kernel quadratic) covariance shape with the exact bivariate-ReLU kernel at every layer removes all truncation error and cuts A by double digits.
* **Mathematical mechanism**: Exact identity $\mathrm{Cov} = \sigma_i\sigma_j \int_0^{\rho} \Phi_2(a_i,a_j;t)\,dt$ (Price's theorem; verified against $J_1$ at $a=0$). Offline prototype attempted a closed-form $J_1$-plus-threshold-modulation shortcut.
* **Candidate file and SHA-256**: `scripts/diag_p5_chosaul.py` (offline prototype, buggy modulation `4\Phi_i\Phi_j`)
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Results Measured**: Prototype MSE `4.9672e+00` vs control-branch `1.3789e-06` (diff `+3.6e8\%`) — the modulation double-counts the mean and is mathematically wrong. Correct form needs per-pair $\Phi_2$ quadrature (~1M pairs x 16 layers), which has no deployable path under the 40% Lane-D util gate. Bounded instead by Taylor evidence: cubic/quartic terms are $\mathcal{O}(1/n^2)$ (P4-07b, -0.01%) and exact L0 alone gives -0.33% on the cov branch (P4-10) — full-depth exact projects to <1% fused.
* **Evidence label**: `[mathematically derived & measured]` (identity derived; prototype measured-buggy; bound supported by P4-07b/P4-10)
* **Interpretation**: Exact kernel is a real but tiny correction at $n=1024$; its deployment cost (bivariate-CDF quadrature per pair per layer) violates the scored-compute gate by orders of magnitude.
* **Decision**: ARCHIVE as non-deployable (Lane D gate fails); salvage exactness only at Layer 0, where it is closed-form and cheap (P5-05).
* **Exact gate applied**: Lane D — no plausible sub-40% implementation for <20% oracle headroom.
* **Next experiment justified**: P5-04 cheap lawful diagnostics (global control, carrier selection).

### Experiment P5-04: Global Single-Beta Control + WMC-vs-Hadamard Lawful Selection
* **Date/time**: 2026-09-05
* **Lane and method family**: Sampling CV (pooled, 1-param) / Lane A-B carrier selection
* **Hypothesis**: (a) A single pooled $\beta$ across all 1024 neurons has negligible estimation noise and salvages split-sample CV; (b) per-MLP lawful selection between WMC and Hadamard via analytic distance $\|m - c_0\|$ (no target contact) captures the best carrier per network.
* **Mathematical mechanism**: (a) Pooled $\hat\beta = \sum yc/\sum zc$ per half, cross-fitted correction. (b) Numpy WMC ($N=4200$) vs single-scramble Hadamard ($S^{1023}$, $\mu_R=31.992187$) compared on raw MSE and analytic distance.
* **Candidate file and SHA-256**: `scripts/diag_p5_global.py` (offline diagnostic)
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Results Measured**:
  - (a) Pooled betas stable ($\approx 0.47$–$0.52$ both halves, all MLPs) but gain=`-0.01\%` (mean raw `1.1524e-05` vs GCV `1.1525e-05`). Even noiseless shared coefficients cannot beat the $2\times$ center-noise penalty.
  - (b) INVALID: Hadamard branch used only 2048 samples (single $1024+1024$ antithetic block sliced to `[:4096]`) vs WMC 4200, so Had raw (`1.7$–$2.7e-05`) > WMC (`0.8$–$1.4e-05`) and lawful-pick trivially always chose WMC. Prior correct-budget P4-01/P4-07 numbers stand (Had $-15.35\%$ raw, tri-blend $-0.52\%$ fused).
* **Evidence label**: `[measured]`
* **Interpretation**: (a) closes even the 1-parameter unbiased-CV escape: the failure is structural (noisy center), not estimation noise. (b) yields no new information; keep prior verdicts.
* **Decision**: KILL pooled unbiased CV; NO NEW VERDICT on carriers.
* **Exact gate applied**: Deployable gain <3%, oracle weak/invalid.
* **Next experiment justified**: P5-05 stacked deployable combining the two surviving micro-wins (lam=0.50 lowest mean + exact L0).

### Experiment P5-05: Stacked Deployable — Blended lam=0.50 + Exact Cho-Saul L0 (N=4200)
* **Date/time**: 2026-09-05
* **Lane and method family**: Analytic stack (Lane A1 synthesis of Run-3 lam=0.50 lowest-mean kernel + P4-10 exact L0)
* **Hypothesis**: The two surviving sub-1% deterministic wins stack additively (independent mechanisms: downstream correlation shape + exact input-layer kernel) into a new lowest stable panel mean with 6W-2L, zero extra FLOP/time cost (exact L0 replaces a costlier einsum).
* **Mathematical mechanism**: Exact Layer-0 mean $\mu_0 = \sigma_0/\sqrt{2\pi}$ and Cho-Saul $J_1(\rho)$ covariance from $W_0^\top W_0$; layers 1–15 single-pass dual-kernel blended Hermite with $\lambda=0.50$ ($(1-\lambda)\cdot0.20/(4\pi)$ centered + $\lambda\cdot0.5\,\phi_i\phi_j$ threshold-aware); prior scale $s_0=0.998319$; fixed blend $\alpha=0.110$ with WMC $N=4200$. Zero fitted parameters.
* **Candidate file and SHA-256**: `candidates/estimator_p5_chosaul_lam05.py` (`93ee3c2871012f46f1b7b4ee1a1657de6505d3d79bd20e0a44bd16932bc14378`)
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Parameters frozen before run**: $\lambda=0.50$, exact-L0 on, $s_0=0.998319$, $\alpha=0.110$, $N=4200$.
* **MLP panel and sampler salts**: Official 8-MLP panel, official seed (salt robustness not run: gain <5% per protocol).
* **Control adjusted score**: `1.223643e-07`
* **Candidate adjusted score**: `1.215291e-07`
* **Relative adjusted improvement**: `-0.68\%`
* **Raw final-layer MSE**: `1.215291e-06`
* **Compute utilization and multiplier**: `9.76\%`, Multiplier: `0.1000`
* **Max residual wall time**: `0.2055s`
* **Failures**: 0
* **Per-MLP Scores**:
  1. logan-fitzgerald: `1.3516e-07` (Champ `1.3527e-07`, Diff `-0.08\%` -> WIN)
  2. william-graves: `1.2872e-07` (Champ `1.3030e-07`, Diff `-1.22\%` -> WIN)
  3. raymond-barnes: `1.1790e-07` (Champ `1.1857e-07`, Diff `-0.56\%` -> WIN)
  4. steven-rice: `1.1154e-07` (Champ `1.1380e-07`, Diff `-1.99\%` -> WIN)
  5. sarah-kelley: `1.2318e-07` (Champ `1.2267e-07`, Diff `+0.41\%` -> LOSS)
  6. christopher-morales: `1.2304e-07` (Champ `1.2416e-07`, Diff `-0.90\%` -> WIN)
  7. cheryl-graham: `1.0958e-07` (Champ `1.1136e-07`, Diff `-1.60\%` -> WIN)
  8. renee-park: `1.2312e-07` (Champ `1.2278e-07`, Diff `+0.27\%` -> LOSS)
* **Wins/losses/ties**: 6W-2L vs Champ; 8W-0L vs Baseline.
* **Median and worst result**: Median `~1.23e-07`; worst regression `+0.41\%` (sarah-kelley, safe).
* **Evidence label**: `[measured]`
* **Interpretation**: New lowest *stable* panel mean (beats lam=0.50-alone `1.2171e-07`); wins meet the count gate but the mean gain is $15\times$ short of promotion. Losses concentrate on the two MLPs where the centered kernel prior was already strongest — coherent mechanism, tiny magnitude.
* **Decision**: RECORD as best-mean variant; REJECT promotion (0.68% << 10% gate). Control preserved. File kept at N=4200.
* **Exact gate applied**: Phase 4 promotion gate (mean $\ge 10\%$).
* **Next experiment justified**: P5-06 boundary-N probe (spend the 0.24% slack from cheaper exact L0).

### Experiment P5-06: Same Stack at N=4350 (10.00% Utilization Boundary)
* **Date/time**: 2026-09-05
* **Lane and method family**: Stacked deployable + compute-boundary probe
* **Hypothesis**: Exact L0 lowered utilization to 9.76%, leaving 0.24% slack (~5.3B FLOPs ≈ 150 full-trajectory samples). Spending it (N=4350) stays nominally at the floor and converts to ~-0.4% fused gain on top of P5-05.
* **Mathematical mechanism**: Identical to P5-05 with `_TOTAL_SAMPLES = 4_350`.
* **Candidate file and SHA-256**: `candidates/estimator_p5_chosaul_lam05.py` at N=4350 (`2319b7cd0fa04a2d405d2d9980078395b4cf36e872248e1fcb83b5c9d8c73dea`); **reverted to N=4200 after the run**.
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Parameters frozen before run**: Same as P5-05 with $N=4350$.
* **Control adjusted score**: `1.223643e-07`
* **Candidate adjusted score**: `1.203229e-07`
* **Relative adjusted improvement**: `-1.67\%`
* **Raw final-layer MSE**: `1.203156e-06`
* **Compute utilization and multiplier**: `10.00\%`, Multiplier: `0.1000` (exactly on the boundary — zero grader margin)
* **Max residual wall time**: `0.1772s`
* **Failures**: 0
* **Per-MLP Scores**:
  1. logan-fitzgerald: `1.3560e-07` (Diff `+0.24\%` -> LOSS)
  2. william-graves: `1.2509e-07` (Diff `-4.01\%` -> WIN)
  3. raymond-barnes: `1.1131e-07` (Diff `-6.12\%` -> WIN)
  4. steven-rice: `1.1239e-07` (Diff `-1.24\%` -> WIN)
  5. sarah-kelley: `1.2155e-07` (Diff `-0.91\%` -> WIN)
  6. christopher-morales: `1.2814e-07` (Diff `+3.21\%` -> LOSS)
  7. cheryl-graham: `1.0447e-07` (Diff `-6.18\%` -> WIN)
  8. renee-park: `1.2403e-07` (Diff `+1.02\%` -> LOSS)
* **Wins/losses/ties**: 5W-3L vs Champ (fails 6W gate).
* **Median and worst result**: Worst `+3.21\%` (christopher-morales).
* **Evidence label**: `[measured]`
* **Interpretation**: Lower mean than P5-05 but the win pattern (three $\ge 4\%$ wins + three losses including a $+3.21\%$ tail) plus changed sample draws is the textbook signature of MC seed noise, not mechanism. Utilization exactly 10.00% means any grader-side FLOP jitter flips the multiplier and erases the gain.
* **Decision**: REJECT (fails 6W gate, boundary util, seed-fragile). File reverted to the N=4200 (`93ee3c28...`) state.
* **Exact gate applied**: Promotion gate (wins $\ge 6/8$) + operational margin rule.
* **Next experiment justified**: Package and submit P5-05 (best stable) per explicit user approval; then stop tuning this family.

### Experiment P5-SUB: Validation, Packaging, AIcrowd Submission of P5-05
* **Date/time**: 2026-09-05
* **Lane and method family**: Release engineering (no scientific claim)
* **Hypothesis**: The P5-05 file passes contract validation and packages reproducibly for the leaderboard.
* **Candidate file and SHA-256**: `candidates/estimator_p5_chosaul_lam05.py` (`93ee3c2871012f46f1b7b4ee1a1657de6505d3d79bd20e0a44bd16932bc14378`, N=4200)
* **Results Measured**:
  - `whest validate`: success in 130ms (class resolved, setup 0.03s < 5s cap, shape/values OK; float32-cast warnings identical to champion pattern).
  - `whest package`: `submission-20260905-160825.tar.gz` (2.3 KB, contains `estimator.py` + `manifest.json`).
  - `whest login`: `subarno_sadat_barno` (API key from workspace `api_aicrowd.json`).
  - `whest submit`: **submission #329939** (`https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/329939`).
  - Note: no `knowledge.md` exists anywhere in the repo; workflow followed `MEMORY.md` §10 + `docs/how-to` instead.
* **Evidence label**: `[measured]`
* **Decision**: SUBMITTED. `Current thing.md` header updated to #329939 (prev #329769). This is a best-mean submission, NOT a promoted champion (fails 10% gate); hidden-set transfer uncertain — watch grading before any champion claim.
* **Next experiment justified**: None in this family. See synthesis.

---

## Phase 4.5 Scientific Synthesis & Definitive Conclusions

1. **The e-8 gap is structural, not a tuning deficit.**
   - Fused error arithmetic: with analytic-branch $A \approx 1.389\times10^{-6}$ (89% of fused error) and sampling-branch $B \approx 1.15\times10^{-5}$, the harmonic optimum gives $1.22\times10^{-6}$ raw. Reaching raw $0.80\times10^{-6}$ (for adj $8.00\times10^{-8}$) needs $A$ down 38%, or $B$ down 83% (6.1×), or a combo (e.g. $A\to1.0\times10^{-6}$ *and* $B\to5\times10^{-6}$).
   - Every analytic refinement ever measured here (Hermite orders, Cho-Saul L0/full, Edgeworth, regime/recursive/continuous scaling, low-rank $R^2_U=3.78\%$, isotropic SVD) moves $A$ by <1%.
   - Every sampling refinement (Hadamard $-15\%$ raw, Hermite-CV $-65\%$ raw, QMC, portfolios) either correlates away in fusion (best deployable $-0.52\%$, best oracle $-1.08\%$) or dies to split-center/estimation noise (P5-02/P5-04: $\approx 0.00\%$).
   - Late-jump (P5-01: 1.9% ceiling), active subspace (P5-02b: $R^2\sim10^{-3}$), and full-exact kernels (P5-03: <1% at infeasible cost) close the remaining deterministic escapes.
   - Stacking the only two surviving micro-wins (P5-05) yields $-0.68\%$ stable — roughly $1/50$ of the required $-34.6\%$.

2. **The scored-compute ceiling holds.**
   - P5-05 sits at 9.76% (exact L0 is *cheaper* than the einsum it replaces). The P5-06 probe proves the boundary is unusable: 10.00% util with 5W-3L seed-fragile pattern and zero grader margin.
   - Above-floor scaling was already disproven in P4-02/03 (raw $-30\%$ at $K=8$ becomes adj $+73\%$ after the $0.25$ multiplier).

3. **Fitting is punished; determinism wins (and is exhausted).**
   - Pattern across Run 2 → P5: every fitted per-neuron/per-MLP scheme (regime scales, Kalman $\alpha_i$, affine $s\,c+b$, ridge $B_k$, LOO polynomials, split-sample $\beta$) loses to fixed white-box constants ($s_0$, $\alpha=0.110$, $\Phi$-weighted kernels). P5-02/P5-04a extend this to 1-parameter pooled controls: still $0.00\%$.
   - All surviving gains are zero-fit deterministic (kernel shapes, exact L0). That well is now dry at the $\sim 1\%$ level.

4. **Operational benchmark (unchanged promotion status).**
   - Champion remains `estimator.py` (`ea8be822...`, `1.223643e-07`) — the only file clearing all promotion gates historically.
   - Best stable mean: `candidates/estimator_p5_chosaul_lam05.py` (`93ee3c28...`, `1.215291e-07`, $-0.68\%$, 6W-2L) — submitted as **#329939** per explicit user approval; champion claim pending hidden-set grading.
   - Rejected boundary variant N=4350 (`2319b7cd...`, `1.203229e-07`, 5W-3L) kept on record only.

5. **What would actually be needed for e-8.**
   - A deterministic non-Gaussian closure correction with $\ge 30\%$ $A$-reduction at $< +2\%$ util (no current candidate exceeds 1%); or
   - A zero-fit sampler with $\ge 4\times$ $B$-reduction uncorrelated with analytic error at fixed $N$ (no current carrier exceeds $1.2\times$ fused); or
   - A lawful low-dimensional selector recovering $\ge 20\%$ of a large oracle (all measured oracles: $0.5$–$6\%$, and selectors recover $\approx 0\%$).
   - Absent one of these — genuinely new mathematics, not stacking — further tuning in this family is scientifically closed. Recommended next bets (untested, high-risk): offline-universal shipped-table priors trained on 90+ held-aside public MLPs (legal data-file path), or a distilled tiny shipped corrector applied via flopscope ops.

### Experiment P4.6-03a: Lane E Flagship Smoke — Final-Layer-Only Kurtosis
* **Date/time**: 2026-09-05
* **Lane and method family**: Lane E (TAP/Onsager program, first concrete mechanism)
* **Hypothesis**: P4-12 did NOT kill cumulant corrections: it used a fabricated schedule $\gamma_2(l)=0.0042\,l$, injected recurrently at all 16 layers (16x compounding) into an inconsistent variance. The honest version — measured universal $\gamma_2$ applied ONCE at the final layer — cuts cov-branch MSE.
* **Mathematical mechanism**: $\mu_{15} = \mu^{G}_{15} + \sigma_{\rm pre}(\gamma_2/24)(a^2-1)\phi(a)$ (Edgeworth $\kappa_4$ term re-derived by direct Hermite integration; also derived skew term $-\sigma\gamma_1 a\phi(a)/6$, unused since measured mean skew $\approx 0$). Numpy offline, $s_0$ fixed, 2 smoke MLPs, bases lam=0.20 and lam=0.50.
* **Candidate file and SHA-256**: `scripts/diag_p46_laneE_smoke.py` (offline diagnostic)
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Results Measured** (cov-branch MSE, both bases):
  - logan: control `1.5498e-06`; g2=+0.03 +1.51%; +0.057 +3.46%; +0.10 +7.74%; g2=-0.057 -0.94%; -0.10 +0.01%.
  - william: control `1.4770e-06`; +0.03 +0.72%; +0.057 +2.04%; +0.10 +5.48%; -0.057 +0.82%; -0.10 +3.34%.
* **Evidence label**: `[measured]`
* **Interpretation**: Positive (theoretically-correct-sign) doses hurt MONOTONICALLY on both MLPs and both branches; negative doses split 1W-1L (fitting noise). The 4th-order Edgeworth mean-shift family is falsified as a mechanism, not a dose — per-neuron tracking of the same family cannot help.
* **Decision**: KILL fixed-kurtosis correction. Grant exactly one last stand (P4.6-03b, the untested skew cumulant) before closing Lane E's cumulant wing.
* **Exact gate applied**: User smoke gate (both MLPs + mean $\ge 3\%$); failed (all-positive hurt).

### Experiment P4.6-03b: Lane E Last Stand — Sampled-Skewness Smoke
* **Date/time**: 2026-09-05
* **Lane and method family**: Lane E cumulant wing (skewness, the one untested cumulant)
* **Hypothesis**: Per-neuron preactivation skewness is real scatter (P4-06: std 0.11 across neurons vs sampling SE $\approx 0.038$), and the derived skew correction $-\sigma g_1 a\phi(a)/6$ with $g_1$ from the MC branch's OWN whitened samples (zero extra sampling FLOPs, lawful) cuts cov-branch MSE at damped dose.
* **Mathematical mechanism**: $g_1$ = bias-aware sample skewness of final preactivations $Z_{15}$ ($N=4200$ whitened, same draws as MC branch); $\mu_{15} = \mu^G_{15} - \eta\,\sigma_{\rm pre}\,g_1\,a\,\phi(a)/6$, $\eta \in \{0.5, 1.0\}$.
* **Candidate file and SHA-256**: `scripts/diag_p46_laneE_skew.py` (offline diagnostic)
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Results Measured**:
  - logan: control `1.5498e-06`; skew_std=0.1140; eta=0.5 `1.4974e-06` (-3.38%); eta=1.0 +2.59%.
  - william: control `1.4770e-06`; skew_std=0.1076; eta=0.5 `1.4247e-06` (-3.54%); eta=1.0 +3.44%.
  - skew_mean $\approx +0.003$ both (matches P4-06 mean~0); full dose overshoots (noise + truncation pushback), half dose lands — both MLPs identically.
* **Evidence label**: `[measured]`
* **Interpretation**: PASS of the aggressive smoke gate (both win, mean -3.46% $\ge$ 3%). Coherent dose-response (0.5 helps both / 1.0 hurts both) signals mechanism, not noise. ETA=0.5 FROZEN a priori for deployable.
* **Decision**: CONTINUE to deployable full-panel (P4.6-03c). First Lane E signal with a pulse.
* **Exact gate applied**: User smoke gate; passed.

### Experiment P4.6-03c: Lane E Deployable — P5 Stack + Sampled-Skew (eta=0.5)
* **Date/time**: 2026-09-05
* **Lane and method family**: Lane E deployable (P5-05 stack + skew correction)
* **Hypothesis**: The 03b cov-branch cut (-3.5%) survives fusion with the MC branch (A4-correlation lesson: must measure, not assume) into a multi-% fused gain at unchanged compute.
* **Mathematical mechanism**: `candidates/estimator_p46_laneE_skew.py`: lam=0.50 + exact Cho-Saul L0 + $s_0$ + sampled-skew final correction ($\eta=0.5$ frozen) + WMC blend ($\alpha=0.110$, $N=4200$). Added cost O(N·n) reductions only.
* **Candidate file and SHA-256**: `candidates/estimator_p46_laneE_skew.py` (`a1bf6a0a73a4ba126d9c2140973a2da3640c879514ac149e22f4b01a8f73fe57`)
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Parameters frozen before run**: $\lambda=0.50$, exact-L0 on, $\eta=0.5$ (smoke-frozen), $s_0=0.998319$, $\alpha=0.110$, $N=4200$.
* **Control adjusted score**: `1.223643e-07`
* **Candidate adjusted score**: `1.192905e-07`
* **Relative adjusted improvement**: `-2.51\%`
* **Raw final-layer MSE**: `1.192905e-06`
* **Compute utilization and multiplier**: `9.76\%` (unchanged — extra ops unmeasurable), Multiplier: `0.1000`
* **Max residual wall time**: `0.1951s`
* **Failures**: 0
* **Per-MLP Scores**: 7W-1L (wins -0.77% to -4.38%; sole loss sarah-kelley +1.57%). Full table in run output.
* **Evidence label**: `[measured]`
* **Interpretation**: Strongest fused gain of the entire Phase 4/4.5/4.6 program; correction/MC-branch correlation costs only ~1% of the cov-branch gain (fusion-friendly, unlike A4 controls). Still $4\times$ short of the 10% interim gate.
* **Decision**: CONTINUE for exactly one principled extension (eta ablation + LOO), then switch lanes per plan.
* **Exact gate applied**: Plan decision tree (2.51% in the 3–10% bracket modulo epsilon, 7W-1L, coherent mechanism).

### Experiment P4.6-03d: Eta Ablation {0, 0.25, 0.5, 0.75} + LOO Selection
* **Date/time**: 2026-09-05
* **Lane and method family**: Lane E dose-response mapping (the one allowed extension)
* **Hypothesis**: The smoke-tested $\eta=0.5$ is near-optimal; mapping {0, 0.25, 0.5, 0.75} on the 8-panel with LOO selection confirms the dose generalizes rather than smoke-overfits.
* **Mathematical mechanism**: Same deployable with `_ETA_SKEW` ∈ {0.25, 0.75} (`estimator_p46_laneE_skew_e025.py` `cb42829e...`, `estimator_p46_laneE_skew_e075.py` `149c92b8...`); $\eta=0$ ≡ P5-05 (`93ee3c28...`).
* **Results Measured**:
  - eta=0.00: `1.215291e-07` (-0.68%, 6W-2L).
  - eta=0.25: `1.189367e-07` (**-2.80\%, 8W-0L**, worst -0.29% sarah-kelley — ZERO regressions; resid 0.1947s; reproduced bit-identically on rerun).
  - eta=0.50: `1.192905e-07` (-2.51%, 7W-1L, worst +1.57%).
  - eta=0.75: `1.225902e-07` (+0.18%, 4W-4L, worst +5.97% sarah-kelley).
  - Clean UNIMODAL dose-response peaking near 0.25; sarah-kelley monotonic canary (+0.41 → -0.29 → +1.57 → +5.97).
  - LOO over {0, 0.25, 0.5, 0.75}: **8/8 folds pick eta=0.25** (including folds holding out the smoke MLPs logan/william — circularity mitigated); LOO mean = refit mean = `1.189367e-07`. Unanimous agreement.
* **Evidence label**: `[measured]` (LOO `[derived]` from measured tables)
* **Interpretation**: Dose confirmed generalizing, not smoke-overfit. First 8W-0L vs the current champ in the program with zero regressed MLPs. New best stable mean by far — but still $3.6\times$ short of the 10% gate and $\sim 12\times$ short of e-8.
* **Decision**: RECORD eta=0.25 file as best-mean variant; REJECT promotion (2.80% << 10%). Lane E cumulant wing now fully mapped — SWITCH LANES per plan (Lane F next).
* **Submission (2026-09-05, explicit user approval)**: validated 86ms, packaged `submission-20260905-165613.tar.gz` (2.3 KB), submitted as **#329946** (`.../submissions/329946`). LB score pending — watch grading before any champion claim.
* **Exact gate applied**: Promotion gate (mean $\ge 10\%$); plan lane-switch rule (extension consumed).

---

## Updated Synthesis (appends §4–5 of prior synthesis)

6. **Lane E outcome: mechanism confirmed, magnitude insufficient.**
   - Kurtosis wing (both signs, both branches, 2 MLPs): dead — positive doses hurt monotonically, negative split.
   - Skewness wing: REAL signal (scatter 0.11 » noise 0.038), dose-response unimodal peaking at $\eta=0.25$, LOO-unanimous, 8W-0L, util unchanged at 9.76%.
   - Best stable mean overall: `1.189367e-07` (-2.80%). The e-8 gap closed from $34.6\%$ to $32.7\%$ remaining — progress, not escape.
   - Remaining Lane E idea (full TAP reaction derivation beyond Edgeworth) is now heavily disfavored: the exact cumulant family it would extend just measured out at single-digit %. Deprioritize unless derivation shows an effect OUTSIDE the cumulant hierarchy.
7. **Next per plan: Lane F (shrunk moment-restart, one fixed-design eval), then Lane G anatomy.** The stacked winner (skew eta=0.25 file) becomes the new deployable base for all downstream lanes.

### Experiment P4.6-04: Co-Skewness Variance-Path Smoke (A: 1D var corrections; B-lite: cross T21/T12)
* **Date/time**: 2026-09-05
* **Lane and method family**: Lane E2 / joint non-Gaussianity (the one untested half of $A$)
* **Hypothesis**: Final means use only means, but covariance matters via diag(variances) → next-layer means. 1D Edgeworth corrections to var_post (A, recurrent) and factorized bivariate co-skewness to off-diagonals at layers 12–14 (B-lite, T21/T12 terms with $K_{21}(i,j)=\sum_a W_{ai}^2W_{aj}s_a$) cut final-mean MSE.
* **Mathematical mechanism**: (A) $E[\mathrm{ReLU}^2] \approx \mathrm{second}_G + s^2[g_1\phi/3 - g_{2,\mathrm{lin}}a\phi/12]$, $g_1$ sampled preactivation skew per layer, $g_{2,\mathrm{lin}} = 0.057\,l/15$ (measured endpoints); (B) $\Delta E_{ij} = -[K_{21}T_{21} + K_{12}T_{12}]/2$ with closed-form $T_{21}$ (verified vs finite-difference reference after fixing TWO sign bugs — conditional minus and chain-rule plus — caught by the in-script assert).
* **Candidate file and SHA-256**: `scripts/diag_p46_coskw_smoke.py` (offline diagnostic; numpy)
* **Control file and SHA-256**: eta025-base equivalent (lam=0.50 + final skew $\eta=0.25$), s0 fixed
* **Results Measured** (cov-branch final-mean MSE, 2 MLPs):
  - (A): logan base `1.5030e-06`, A@0.5 +1.11%, A@1.0 +4.33%; william base `1.4248e-06`, A@0.5 -0.45%, A@1.0 -0.45% (dose-flat → incidental). Mean +0.33%. FAIL gate.
  - (B-lite): K-identity assert (diag $K_{21}$ vs sampled marginal $\kappa_3$) median rel err **0.99** both MLPs (noise floor ~0.2) — factorized third cumulants are pure noise: post-activations are too dependent for independence-factorization (the cumulant hierarchy does not close at affordable order).
  - Debugging footnote: first run exploded (+1e8%) from feeding POST-activation skew into a PREACTIVATION formula; fixed by tracking preactivation moments. Second saga: two sign errors in T21 derivation, both caught by the verification assert before any conclusion was drawn.
* **Evidence label**: `[measured]`
* **Interpretation**: (A) 1W-1L, mean-harmful — the variance channel at this magnitude does not move final means. (B) structurally falsified — no affordable third-order state exists. The joint-cumulant road ends here (full-tensor tracking is $O(n^3)$ STATE/layer, budget-impossible).
* **Decision**: KILL P4.6-04 entirely (A measured-dead, B falsified, T30/T03 moot). Lane E fully closed including E2.
* **Exact gate applied**: User smoke gate (both + mean $\ge 3\%$); failed on both wings.
* **Next experiment justified**: Lane F shrunk moment-restart (P4.6-05) — fully designed, cheap, binary, tests the manifold story.

(End of file)
