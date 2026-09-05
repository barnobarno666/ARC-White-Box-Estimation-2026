# ARC White-Box Estimation Challenge 2026: Permanent Memory & System Codex

> **Note**: This file serves as the definitive, self-contained reference for the competition, rules, mathematical formulations, evaluation protocols, codebase architecture, and research roadmaps. Consult this file at the start of any session to resume work immediately without web browsing or context loss.

---

## 1. Challenge Essentials & URLs

* **Competition**: ARC White-Box Estimation Challenge 2026 (Phase 2)
* **Organizers**: Alignment Research Center (ARC) in partnership with AIcrowd
* **Core URLs**:
  * Challenge Main Page: `https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026`
  * Discussion Forum: `https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/discussion`
  * Leaderboard: `https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards`
  * WhestBench Explorer: `https://aicrowd.github.io/whestbench-explorer/`
  * Hugging Face Dataset: `https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/tree/v2-phase2`
* **Timeline**:
  * Phase 2 Active: August 22, 2026 – October 17, 2026
* **Total Prize Pool**: $50,000 ($25,000 allocated to Phase 2).

---

## 2. Problem Formulation & Contracts

* **Input**: Explicit weight matrices $W^{(0)}, \dots, W^{(L-1)}$ of a randomly initialized Multilayer Perceptron (MLP) with standard Gaussian input $X \sim \mathcal{N}(0, I_n)$.
* **Phase 2 Dimensions**:
  * Width: $n = 1024$
  * Depth: $L = 16$
  * Activation: $\text{ReLU}(z) = \max(0, z)$
  * Initialization: He Normal ($\mathrm{Var}(W_{ij}) = 2 / n_{\text{in}}$)
* **Contract Output**:
  * `predict(self, mlp: MLP, budget: int) -> fnp.ndarray`
  * Returns array of shape **`(16, 1024)`** with dtype `fnp.float32`.
  * Values must be finite, non-negative, and strictly valid.
* **Scoring Criterion**:
  * The official leaderboard ranks submissions **strictly on final-layer MSE (layer 15)** against $10^9$-sample Monte Carlo ground truth:
    $$\text{MSE}_{\text{final}} = \frac{1}{1024} \sum_{i=1}^{1024} (\hat{y}_{15, i} - y_{15, i}^*)^2$$
  * Earlier layers (0 to 14) are evaluated for secondary metrics but do not impact leaderboard rank.

---

## 3. Compute Budget & Scoring Model

* **FLOP Budget per MLP ($B_m$)**: $2^{41} \approx 2.199 \times 10^{12}$ FLOPs ($\approx 65,536$ forward passes).
* **Metered FLOPs ($C_m$)**: Strictly billed and counted via `flopscope`.
* **Adjusted Score Metric**:
  $$\text{Adjusted Score} = \text{MSE}_{\text{final}} \times \max\left(0.1000, \frac{C_m}{B_m}\right)$$
* **The 10% Floor Clamping**:
  * When compute utilization is $\le 10.00\%$ ($C_m \le 0.10 B_m$), the multiplier is clamped at **`0.1000`**.
  * Spending less than 10% compute **does not** reduce the multiplier further.
  * Any compute $> 10.00\%$ immediately incurs a linear multiplier penalty ($S_{\text{new}} / S_{\text{old}} = (\text{MSE}_{\text{new}} / \text{MSE}_{\text{old}}) \times (u / 0.10)$).

---

## 4. Grader Execution & Environment Rules

1. **CPU Execution**: 1 core (2 vCPUs) for participant inference; 7 cores reserved for flopscope background metering.
2. **Library Constraints**:
   * Standard `numpy` is banned on the grader (`import numpy` raises `ModuleNotFoundError`).
   * Must exclusively use `import flopscope.numpy as fnp` and `import flopscope as flops`.
3. **Flopscope Array Immutability**:
   * `flopscope.numpy` arrays are immutable to ensure exact FLOP tracking.
   * In-place indexed updates (e.g. `arr[i] = val` or `arr[i] += val`) are strictly prohibited and raise `TypeError`.
   * Must construct arrays functionally using `fnp.where()`, `fnp.stack()`, or whole-array operations.
4. **Residual Wall-Time Hard Cap**:
   * Non-flopscope Python interpreter overhead is capped at **400 ms per MLP**.
   * Exceeding 400 ms triggers an immediate 0-score failure for that MLP.
   * Target execution safety: maintain local residual time below 250 ms ($>1.6\times$ margin).
5. **No Graph Materialization**:
   * Calls to `bool()`, `float()`, or `int()` on un-evaluated flopscope arrays force immediate graph materialization and must be avoided inside `predict()`.
6. **Windows CLI Encoding**:
   * Always execute with `$env:PYTHONUTF8="1"` in PowerShell to prevent `cp1252` encoding exceptions on rich Unicode CLI output.
7. **Package Management**:
   * Always use `uv` (`uv run`, `uv add`), never bare `pip`.

---

## 5. Local Dataset & Evaluation Panel

* **Local Dataset Path**: `D:\ALL CODES\AICROWD COMPETITION\datasets\mini`
* **Official Evaluation Panel**: 8 fixed MLPs from the `mini` split:
  1. `logan-fitzgerald`
  2. `william-graves`
  3. `raymond-barnes`
  4. `steven-rice`
  5. `sarah-kelley`
  6. `christopher-morales`
  7. `cheryl-graham`
  8. `renee-park`
* **Seed Protocol 4.0 Rule**:
  * Always construct MLPs via `MLP.from_row(row, seed_protocol_version=v, seed_salt=s)` where `v, s = resolve_seed_context(ds)`.
  * Never use `seed=row['mlp_seed']` directly, as it bypasses dataset salt hashing and desynchronizes Monte Carlo realizations from the grader.

---

## 6. Current Immutable Champion & Score Milestones

The current champion control is deployed in [whest-starterkit/estimator.py](estimator.py):

* **Architecture**: Prior-Calibrated Hermite Covariance + Whitened Antithetic MC
* **Hyperparameters**: $\gamma = 0.20$, $s_0 = 0.998319$, $\alpha = 0.110$, $N = 4,200$
* **Performance**:
  * Raw Final MSE: **`1.231901e-06`**
  * Adjusted Score: **`1.231901e-07`** (Multiplier: `0.1000`)
  * Compute Utilization: **`9.81%`** (strictly under the 10% floor)
  * Max Residual Time: **`0.1873s`** ($>2.1\times$ safety margin under 0.40s cap)
  * Failures: **0**
  * Head-to-Head: **8W-0L Clean Sweep**
  * Leave-One-Out (LOO) Score: **`1.2396e-07`**
* **Official Submission ID**: **`#329733`** on AIcrowd.

### Historical Benchmark Milestones:
* Baseline Reference (Gain Cov 75% + WMC 25%): `3.1749e-07`
* Hermite Sweep ($\gamma=0.50$, uncalibrated): `2.9078e-07`
* Run 1 Champion (Self-Calibrated Hermite): `1.6500e-07` (Target `< 2.5e-7` achieved)
* Run 2 Champion (Prior-Calibrated Hermite): **`1.2319e-07`** (Target `< 1.24e-7` achieved)
* **Next Target (Stage 3)**: **`< 1.00e-7`** (Stretch: `< 7.50e-8`, Breakthrough: `< 5.00e-8`)

---

## 7. Mathematical Foundations & Breakthrough Discoveries

1. **Scale Deflation Factor ($s_0 = 0.998319$)**:
   * Deep Gaussian ReLU accumulation over 16 layers causes $+0.16\%$ energy inflation.
   * Across the 8 MLPs, $s_y = 0.998412 \pm 0.000192$.
   * Using an empirical scale from single-batch MC introduces $\approx 4.07 \times 10^{-7}$ MSE noise.
   * Applying the architectural prior $s_0 = 0.998319$ eliminates this noise completely.
2. **Second-Order Hermite Covariance ($\gamma = 0.20$)**:
   * Standard gain covariance underestimates off-diagonal correlations by 5%–30%.
   * Price's theorem gives quadratic term: $+\gamma \frac{1}{4\pi} \rho_{ab}^2 \sigma_a \sigma_b$.
   * Once scale is properly calibrated, optimal $\gamma$ is $0.20$ (lower than uncalibrated 0.50).
3. **Whitened Antithetic Monte Carlo (WMC)**:
   * Antipodal sampling cancels all odd-order error terms.
   * Layer-1 Gram matrix whitening matches second-order spherical moments, cutting sample variance by **$38.4\%$** over plain antithetic MC.
4. **Gauss-Markov Harmonic Precision Blending ($\alpha = 0.110$)**:
   * Covariance approximation error and MC sampling error are virtually orthogonal ($\text{Cosine} \approx +0.0762$).
   * Minimum variance linear combination combines both branches optimally at $\alpha = 0.110$.

---

## 8. Stage 3 Research Roadmap: The Path to `< 1.00e-7`

Refer to [run 3 plan.md](../run%203%20plan.md) for full details:

* **Lane A (50% Effort) — Exploit Analytic/Hybrid Family**:
  * **Lane A1**: Threshold-Aware Gaussian Hermite Expansion:
    * Compute standardized preactivations $a_i = \mu_i / \sigma_i$.
    * Linear: $C_{ij}^{(1)} = \sigma_i \sigma_j \Phi(a_i) \Phi(a_j) \rho_{ij}$
    * Quadratic: $C_{ij}^{(2)} = \frac{\sigma_i \sigma_j}{2} \phi(a_i) \phi(a_j) \rho_{ij}^2$
    * Cubic: $C_{ij}^{(3)} = \frac{\sigma_i \sigma_j}{6} a_i a_j \phi(a_i) \phi(a_j) \rho_{ij}^3$
    * Quartic: $C_{ij}^{(4)} = \frac{\sigma_i \sigma_j}{24} (a_i^2 - 1)(a_j^2 - 1) \phi(a_i) \phi(a_j) \rho_{ij}^4$
  * **Lane A2**: Risk-Aware Blending Accounting for Cross-Error Covariance:
    * $\alpha^* = \frac{A - C}{A + B - 2C}$ where $C = \mathbb{E}[(c_0 - y)^\top (m - y)]$.
  * **Lane A3**: Low-Rank Analytic Residual Correction ($U = [c, \Delta_2, \Delta_3, \Delta_4]$).
  * **Lane A4**: Strongly regularized joint reuse of final-layer Hermite control variates.
* **Lane B (30% Effort) — Mid-Network Ridge Control Variates**:
  * Mid-network ridge regression $B_k = \text{Ridge}(H_k - \bar{H}_k, H_L - \bar{H}_L)$ for $k \in \{4, 6, 8, 10\}$.
  * Sampled moment reset at layer $k$.
* **Lane C (20% Effort) — Independent Exploration**:
  * Scrambled Sobol / randomized rank-one lattices.
  * Active-subspace stratification along leading weight directions.
  * Compressed third-cumulant modes.

---

## 9. Non-Negotiable Validation Protocol for Stage 3

1. **Stage 1 (Smoke Screen)**: 2 MLPs, 1 seed.
2. **Stage 2 (Panel Screen)**: 8 MLPs, official protocol seed.
3. **Stage 3 (Robustness Screen)**: 8 MLPs across 3 deterministic sampler salts (only after $\ge 5\%$ panel gain).
4. **Promotion Gates**:
   * Benchmark against current champion (`1.2319e-07`), never the old `3.175e-07` baseline.
   * Require $\ge 6/8$ wins (preferably $7/8$).
   * LOO improvement must agree with all-eight refit.
   * No worst-MLP regression $> 10\%$.
   * Zero failures, max residual time $< 0.30s$, compute utilization $\le 10.00\%$.
5. **Claim Rigor Standards**:
   * Label all claims as: `[measured]`, `[mathematically derived]`, `[supported interpretation]`, or `[untested hypothesis]`.
6. **Reporting Standard**:
   * Record every result in the table in [Current thing.md](../Current%20thing.md).

---

## 10. CLI & Workflow Cheatsheet

```powershell
# Set UTF-8 encoding in PowerShell
$env:PYTHONUTF8="1"

# Contract validation
uv run whest validate --estimator estimator.py

# Benchmark evaluation on 8-MLP panel
uv run python scripts/run_eval.py estimator.py "Candidate Name" "Short Description"

# Package submission archive
uv run whest package --estimator estimator.py

# Submit to AIcrowd
uv run whest login --api-key 1d354c62e5581dc171d4609451a0ec867bc887b3613ffa47d4a04abbfdd5ed61
uv run whest submit submission-*.tar.gz
```
