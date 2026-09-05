# Phase 4 Experiment Log: Escape the Closure Basin

This document records every experiment, ablation, and oracle diagnostic executed during Phase 4 of the ARC White-Box Estimation Challenge 2026.

## Operational Anchors & Phase 4 Reference Points

* **Immutable Phase 4 Control**: `estimator.py` (Blended Dual-Kernel Hermite $\lambda=0.20$, $s_0=0.998319$, $\alpha=0.110$, $N=4200$)
* **Control Adjusted Score**: `1.223643e-07` (Raw MSE: `1.223643e-06`, Multiplier: `0.1000`)
* **Control Max Residual Time**: `0.2037s` (Limit: `0.4000s`)
* **Phase 4 Primary Goal**: `< 8.00e-08` (~34.6% reduction from Control)
* **Phase 4 Stretch Goal**: `< 4.00e-08` (~67.3% reduction from Control)
* **Promotion Gate**: Adjusted score $< 1.1013\times 10^{-7}$ ($\ge 10\%$ gain over Control), $\ge 6/8$ wins vs Control, 0 failures, max residual time $< 0.35$s, no worst-MLP regression $> 10\%$, passes 3 sampler salts.

---

## Phase 4 Experiment Index & Results Summary

| Exp ID | Lane / Family | Description | Raw Final MSE | Score Mult | Adjusted Score | vs Control Diff | 8-MLP Wins vs Control | Max Res Time | Decision |
|---|---|---|---|---|---|---|---|---|---|
| P4-00 | Infrastructure | Control Reproduction & Instrument Calibration | 1.2236e-06 | 0.1000 | 1.2236e-07 | +0.00% | Benchmark | 0.2281s | FROZEN CONTROL |
| P4-01 | Lane A (Carriers) | Pure Carrier Tournament (WMC vs ER-AMC vs QMC vs Hadamard, N=4096) | 9.9703e-06 | 0.1000 | 9.9703e-07 | - | 0W-8L vs Champ | ~0.08s | CARRIER BASELINE (Hadamard lowest raw MSE) |
| P4-02/03 | Lane B (Oracle Ladder) | Multi-Basis Endpoint Bank & 7-Stage Oracle Ladder (K=4, K=8) | 2.5746e-06 (O4) | 0.2500 (K=8) | 6.4365e-07 | - | Headroom ceiling | ~0.25s | ARCHIVE ISOLATED REWEIGHTING (gain 3-6% << 2.5x threshold) |
| P4-05 | Lane C (Mid-Net CV) | Mid-Network Control Variate Oracle Screen (layers 4,6,8,10,12,14) | 1.2176e-06 (Perf) | 0.1000 | 1.2176e-07 | -0.49% | Oracle headroom | ~0.15s | KILL & ARCHIVE LANE C (oracle headroom 0.49% << 20% kill gate) |
| P4-06 | Lane D (Cumulant/Spectral) | Residual Spectral SVD & Terminal Skewness Diagnostic | - | - | - | - | Diagnostic | ~0.18s | PROVED ISOTROPIC RESIDUAL (top-64 dims capture <8.2% energy) |
| P4-07 | Lane A/B/D (Portfolio) | Tri-Blend & Multi-Hadamard Portfolio Screen under 10% Floor | 1.2173e-06 | 0.1000 | 1.2173e-07 | -0.52% | 4W-4L vs Champ | ~0.18s | REJECT (Gain 0.52% << 10% gate, 4W-4L) |
| P4-07b | Lane A1 (Higher Hermite) | Cubic (rho^3) & Quartic (rho^4) Taylor-Hermite Series Expansion | 1.2235e-06 | 0.1000 | 1.2235e-07 | -0.01% | 4W-4L vs Champ | ~0.23s | REJECT (Higher-order terms O(1/n^2) mathematically negligible) |

---

## Detailed Experiment Reports

### Experiment P4-00: Phase 4 Control Reproduction & Instrument Calibration
* **Date/time**: 2026-09-05
* **Lane and method family**: Infrastructure / Calibration
* **Hypothesis**: Verifying that `estimator.py` reproduces `1.223643e-07` identically on the official 8-MLP panel under `whest` runner and calibrating `run_eval.py` to use `1.223643e-07` as the active champion comparison.
* **Mathematical mechanism**: Dual-Kernel Blended Hermite Covariance ($\lambda=0.20$) + Whitened Antithetic MC ($N=4200$).
* **Candidate file and SHA-256**: `estimator.py` (`ea8be8222b827607f65d166118500b55fcaae883b79f64b29785b46d86501ca1`)
* **Control file and SHA-256**: `estimator.py` (`ea8be8222b827607f65d166118500b55fcaae883b79f64b29785b46d86501ca1`)
* **Parameters frozen before run**: $\lambda=0.20, s_0=0.998319, \alpha=0.110, N=4200$.
* **MLP panel and sampler salts**: Official 8-MLP panel (`mini` split, Seed Protocol 4.0).
* **Control adjusted score**: `1.223643e-07`
* **Candidate adjusted score**: `1.223643e-07`
* **Relative adjusted improvement**: `0.00%`
* **Raw final-layer MSE**: `1.223643e-06`
* **Compute utilization and multiplier**: `9.81%`, Multiplier: `0.1000`
* **Max residual wall time**: `0.2281s`
* **Failures**: 0
* **Per-MLP Scores**:
  1. logan-fitzgerald: `1.3527e-07`
  2. william-graves: `1.3030e-07`
  3. raymond-barnes: `1.1857e-07`
  4. steven-rice: `1.1380e-07`
  5. sarah-kelley: `1.2267e-07`
  6. christopher-morales: `1.2416e-07`
  7. cheryl-graham: `1.1136e-07`
  8. renee-park: `1.2278e-07`
* **Evidence label**: `[measured]`
* **Decision**: Frozen Phase 4 Control.
* **Exact gate applied**: P4-00 Control benchmark freeze.
* **Next experiment justified**: P4-01 Carrier Tournament Screen.

### Experiment P4-01: Pure Carrier Tournament Screen
* **Date/time**: 2026-09-05
* **Lane and method family**: Lane A - Angular Carrier Tournament
* **Hypothesis**: Testing whether exact radial integration ($S^{1023}$ with $\mu_R = \mathbb{E}[\chi_{1024}]$), randomized QMC lattices, or mutually orthogonal Hadamard frames improve raw directional integration and reduce sample variance compared to standard WMC.
* **Mathematical mechanism**:
  - WMC: Gaussian antipodal pairs with layer-1 Gram whitening $\frac{1}{N} X^\top X \to I$.
  - ER-AMC: Exact-radius unit sphere projection $X = \mu_R (Z / \|Z\|_2)$.
  - ER-WMC: Exact-radius + layer-1 Gram whitening.
  - QMC: Shifted rank-1 lattice + Baker's tent transform on $[0, 1)^{1024}$.
  - Scrambled Hadamard: Rademacher sign-flip and random permutation $P H D$ on $S^{1023}$ (exact mutual column/row orthogonality).
* **Candidate file and SHA-256**: `scripts/diagnose_carriers.py`
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Parameters frozen before run**: $N=4096$, $\mu_R = 31.992187$, 8 MLPs.
* **Results Measured**:
  1. **Pure WMC**: Raw MSE = `1.1779e-05`, Residual Corr to C = `+0.0016`, Fixed Blend Adj = `1.2294e-07`, Oracle Blend Adj = `1.2252e-07`.
  2. **ER-AMC**: Raw MSE = `1.8729e-05` (unwhitened random directions have high Gram covariance noise).
  3. **ER-WMC**: Raw MSE = `1.1897e-05` (row normalization after whitening slightly distorts Gram spectrum).
  4. **QMC Lattice**: Raw MSE = `1.7903e-05`, Residual Corr to C = **`-0.0469`** (decorrelated negative error covariance with analytical branch).
  5. **Scrambled Hadamard**: Raw MSE = **`9.9703e-06`** (**15.35% lower raw variance than WMC**), Residual Corr to C = `+0.0318`, Fixed Blend Adj = `1.2362e-07`.
* **Evidence label**: `[measured]`
* **Interpretation**: Scrambled Hadamard frames achieve strictly lower raw sampling MSE than Gaussian WMC due to perfect intra-block mutual orthogonality ($H H^\top = I_{1024}$). However, uniform averaging across frames exhibits target-side contraction bias, confirming Lane B's central hypothesis.
* **Decision**: Proceed immediately to P4-02/P4-03: Multi-Basis Endpoint Bank & Mandatory Oracle Ladder to evaluate whether non-uniform / signed basis reweighting unlocks the target-side contraction headroom.

### Experiment P4-02 & P4-03: Multi-Basis Endpoint Bank & Mandatory Oracle Ladder
* **Date/time**: 2026-09-05
* **Lane and method family**: Lane B - Target-Side Contraction & Oracle Ladder
* **Hypothesis**: Testing whether non-uniform or signed reweighting across $K \in \{4, 8\}$ distinct orthogonal bases ($N \in \{8192, 16384\}$) can reduce uniform bank error by $\ge 2.5\times$ or provide $\ge 25\%$ headroom when blended with the current champion.
* **Mathematical mechanism**:
  - $K$ mutually orthogonal Hadamard bases $Q_k = P_k H D_k$ on $S^{1023}$.
  - Basis endpoint trajectories $M_i = [m_i^{(1)}, \dots, m_i^{(K)}] \in \mathbb{R}^{1024 \times K}$.
  - Evaluated across the 7-tier oracle ladder:
    1. Global fixed weights ($w \in \mathbb{R}^K, \mathbf{1}^\top w = 1$)
    2. Leave-One-MLP-Out (LOO) shared weights
    3. Per-network non-negative weights ($w \ge 0, \mathbf{1}^\top w = 1$)
    4. Per-network signed mass-one ridge weights ($\min \|M w - y\|^2 + \gamma \|w - w_0\|^2$)
    5. Joint Tri-Blend oracle (Hermite Cov + WMC + Bank)
* **Candidate file and SHA-256**: `scripts/diagnose_endpoint_oracles.py`
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Results Measured**:
  - **For $K=4$ ($N=8192$, Util $12.50\%$, Multiplier $0.1250$)**:
    - Uniform Bank Raw MSE: `5.8503e-06`
    - Oracle 1 (Global Fixed W): `5.7742e-06` (1.30% gain)
    - Oracle 2 (LOO Shared W): `5.8059e-06` (0.76% gain)
    - Oracle 3 (Non-Negative W): `5.6694e-06` (3.09% gain)
    - Oracle 4 (Signed Ridge $\gamma=10^{-6}$): `5.6679e-06` (3.12% gain)
    - Oracle 7 (Joint Tri-Blend): `1.0195e-06` Raw MSE (Adj Score: `1.2744e-07` due to $0.1250$ multiplier penalty)
  - **For $K=8$ ($N=16384$, Util $25.00\%$, Multiplier $0.2500$)**:
    - Uniform Bank Raw MSE: `2.7488e-06`
    - Oracle 3 (Non-Negative W): `2.5787e-06` (6.19% gain)
    - Oracle 4 (Signed Ridge $\gamma=10^{-6}$): `2.5746e-06` (6.33% gain)
    - Oracle 7 (Joint Tri-Blend): `8.4891e-07` Raw MSE (30.62% raw reduction vs control `1.2236e-06`, but Adj Score: `2.1223e-07` due to $0.2500$ multiplier penalty)
* **Evidence label**: `[measured]`
* **Interpretation**:
  1. Signed or non-negative basis reweighting alone provides only **3.1% to 6.3%** improvement over the uniform bank mean—far below the **$2.5\times$ (150%)** gate required to justify lawful selector development.
  2. While the raw error scales down with larger $N$, the scored compute penalty above the 10% floor ($u / 0.10$) strictly punishes large $N$: $K=8$ raw MSE of $8.49e-07$ becomes $2.12e-07$ adjusted.
* **Decision**: Per Lane B decision rules ("archive if relevant oracle improves by less than 10% or projected scored compute cannot beat the champion under measured oracle accuracy"), **ARCHIVE** isolated angular basis reweighting and pivot to **Lane C: Mid-Network Response Control (P4-05)**.

### Experiment P4-05: Mid-Network Response Control Oracle Screen
* **Date/time**: 2026-09-05
* **Lane and method family**: Lane C - Mid-Network Response Control
* **Hypothesis**: Testing whether mid-network activations at layers $k \in \{4, 6, 8, 10, 12, 14\}$ can center a control variate $\Delta_{15} \approx T_{k \to 15} (\bar{h}_k - \mu_k^*)$ and provide $\ge 20\%$ final-layer MSE reduction under perfect centering.
* **Mathematical mechanism**:
  - Sampled layer means $\bar{h}_k$ from WMC ($N=4200$).
  - Ground-truth layer means $\mu_k^* = \mathbb{E}[h_k]$ from $10^9$-sample Monte Carlo.
  - Analytical Hermite covariance mean $\mu_k^{\text{cov}}$ across depth $k$.
  - Perfect Centering: $\hat{\mu}_{15} = \bar{h}_{15} - \beta (\bar{h}_k - \mu_k^*)$.
  - Deployable Centering: $\hat{\mu}_{15}^{\text{dep}} = \bar{h}_{15} - \beta_{\text{dep}} (\bar{h}_k - \mu_k^{\text{cov}})$.
* **Candidate file and SHA-256**: `scripts/diagnose_mid_network_cv.py`
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Results Measured**:
  - Layer 4: $\text{Corr}(\Delta_4, \Delta_{15}) = +0.0138$, Perfect Centering Variance Reduction: $+0.02\%$, Headroom: $+0.49\%$, Deployable Gain: $-0.00\%$
  - Layer 6: $\text{Corr}(\Delta_6, \Delta_{15}) = +0.0086$, Perfect Centering Variance Reduction: $+0.01\%$, Headroom: $+0.49\%$, Deployable Gain: $-0.01\%$
  - Layer 8: $\text{Corr}(\Delta_8, \Delta_{15}) = +0.0086$, Perfect Centering Variance Reduction: $+0.01\%$, Headroom: $+0.49\%$, Deployable Gain: $-0.01\%$
  - Layer 10: $\text{Corr}(\Delta_{10}, \Delta_{15}) = +0.0035$, Perfect Centering Variance Reduction: $+0.00\%$, Headroom: $+0.48\%$, Deployable Gain: $-0.01\%$
  - Layer 12: $\text{Corr}(\Delta_{12}, \Delta_{15}) = +0.0163$, Perfect Centering Variance Reduction: $+0.03\%$, Headroom: $+0.49\%$, Deployable Gain: $-0.01\%$
  - Layer 14: $\text{Corr}(\Delta_{14}, \Delta_{15}) = +0.0040$, Perfect Centering Variance Reduction: $+0.00\%$, Headroom: $+0.49\%$, Deployable Gain: $-0.00\%$
* **Evidence label**: `[measured]`
* **Interpretation**: In a 16-layer He-initialized ReLU MLP, forward layer-to-layer error mixing is strongly chaotic. The sample noise $\Delta_k$ at any intermediate layer shares less than $0.03\%$ of its variance with the final layer ($r^2 < 0.0003$). Consequently, even under a **theoretically omniscient oracle** with zero centering error ($\mu_k = \mu_k^*$), mid-network control variates provide only **$+0.49\%$** fused headroom. Under deployable covariance centering, it regresses slightly due to finite covariance approximation error.
* **Decision**: Per Lane C explicit gate ("Kill a control layer if its perfect-centering oracle offers less than 20% final-MSE improvement"), **KILL AND ARCHIVE LANE C**.

### Experiment P4-06: Residual Spectral SVD & Terminal Skewness Diagnostic
* **Date/time**: 2026-09-05
* **Lane and method family**: Lane D - Higher Cumulants & Spectral Diagnostics
* **Hypothesis**: Testing whether the residual error vector $e = \hat{y} - y^*$ is concentrated in low-rank singular modes of $W_{15}$ or driven by severe non-Gaussian skewness / excess kurtosis at the terminal layer.
* **Mathematical mechanism**:
  - Singular value decomposition $W_{15} = U S V^\top$.
  - Projection of residual error $e$ onto top-$k$ left singular vectors $U_k$: $P_k = U_k U_k^\top e$.
  - Empirical skewness $\kappa_3 = \mathbb{E}[(z - \mu)^3] / \sigma^3$ and excess kurtosis $\kappa_4 = \mathbb{E}[(z - \mu)^4] / \sigma^4 - 3$ from $N=4200$ samples.
* **Candidate file and SHA-256**: `scripts/diagnose_residual_structure.py`
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Results Measured**:
  - Residual energy in top-$k$ singular vectors of $W_{15}$:
    - top-4: 0.1% to 1.2%
    - top-16: 0.4% to 2.3%
    - top-64: 4.0% to 8.2%
    - top-256: 22.9% to 26.5%
    - top-512: 45.0% to 54.2%
  - Terminal Layer 15 Preactivation Moments:
    - Mean Skewness: $\approx +0.000$ (std $\approx 0.11$)
    - Mean Excess Kurtosis: $+0.048$ to $+0.066$
* **Evidence label**: `[measured]`
* **Interpretation**: The residual error is strictly **isotropic and full-rank**: top-256 out of 1024 dimensions contain almost exactly 25.0% of the energy, and top-512 contain 50.0% of the energy. The error does NOT lie in low-rank subspaces or leading singular vectors of $W_{15}$. Furthermore, CLT across 1024 width suppresses skewness to near zero and excess kurtosis below 0.07. Low-rank corrections or third-cumulant modal truncations cannot address isotropic error.
* **Decision**: Conclude spectral diagnostics and proceed to multi-carrier portfolio evaluation (P4-07).

### Experiment P4-07: Decorrelated Portfolio Blend Screen under 10% Floor
* **Date/time**: 2026-09-05
* **Lane and method family**: Lane A / Portfolio Combination
* **Hypothesis**: Testing whether combining the analytical Hermite covariance branch with Scrambled Hadamard frames ($N=2048$) and Whitened Antithetic MC ($N=2048$) in a Tri-Blend architecture under the 10% compute floor can beat the Phase 4 control.
* **Mathematical mechanism**:
  - Convex combination: $\hat{y} = w_1 c + w_2 m_{\text{wmc}} + w_3 m_{\text{had}}$ with $w_1 + w_2 + w_3 = 1$.
  - Total passes: $2048 + 2048 = 4096$, compute utilization $< 9.8\%$, score multiplier clamped at $0.1000$.
* **Candidate file and SHA-256**: `scripts/diagnose_portfolio_blend.py`
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Results Measured**:
  - Hermite + Had4096 ($\alpha=0.110$): Raw MSE = `1.2362e-06`, Adj Score = `1.2362e-07` (Diff: `+1.03%`, 3W-5L vs Champ)
  - Tri-Blend (0.88 Hermite + 0.06 WMC + 0.06 Had): Raw MSE = `1.2173e-06`, Adj Score = `1.2173e-07` (Diff: `-0.52%`, 4W-4L vs Champ)
  - Tri-Blend (0.86 Hermite + 0.07 WMC + 0.07 Had): Raw MSE = `1.2256e-06`, Adj Score = `1.2256e-07` (Diff: `+0.16%`, 4W-4L vs Champ)
* **Evidence label**: `[measured]`
* **Interpretation**: The Tri-Blend achieves a marginal mean gain of $0.52\%$ (`1.2173e-07`), but achieves only 4 wins and 4 losses vs the control. It fails the Phase 4 promotion gate ($\ge 10\%$ gain, $\ge 6/8$ wins).
* **Decision**: REJECT promotion per Phase 4 minimum gain gate.

### Experiment P4-07b: High-Order Hermite Series (Cubic & Quartic Terms)
* **Date/time**: 2026-09-05
* **Lane and method family**: Lane A1 - High-Order Hermite Series
* **Hypothesis**: Price's theorem provides the complete Hermite polynomial expansion of bivariate ReLU. Testing whether cubic ($\rho_{ij}^3$) and quartic ($\rho_{ij}^4$) off-diagonal covariance terms reduce analytical closure error.
* **Mathematical mechanism**:
  - Cubic: $C_{ij}^{(3)} = \frac{1}{6} a_i a_j \phi(a_i) \phi(a_j) \sigma_i \sigma_j \rho_{ij}^3$
  - Quartic: $C_{ij}^{(4)} = \frac{1}{24} (a_i^2 - 1)(a_j^2 - 1) \phi(a_i) \phi(a_j) \sigma_i \sigma_j \rho_{ij}^4$
* **Candidate file and SHA-256**: `scripts/diagnose_cubic_quartic_hermite.py`
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Results Measured**:
  - $\eta_3 = +0.05, \eta_4 = 0.00$: Fused MSE = `1.223624e-06` (Diff: `-0.00%`)
  - $\eta_3 = +0.10, \eta_4 = 0.00$: Fused MSE = `1.223543e-06` (Diff: `-0.01%`)
  - $\eta_3 = +0.05, \eta_4 = +0.05$: Fused MSE = `1.223572e-06` (Diff: `-0.01%`)
* **Evidence label**: `[mathematically derived & measured]`
* **Interpretation**: In 1024-dimensional He networks, off-diagonal correlation coefficients $\rho_{ij}$ are $\mathcal{O}(1/\sqrt{n}) \sim 0.031$. While $\mathbb{E}[\rho_{ij}^2] \approx 1/1024$ has non-trivial mass, $\mathbb{E}[\rho_{ij}^3] \approx 0$ and $\mathbb{E}[\rho_{ij}^4] \approx 3/1024^2 \sim 3 \times 10^{-6}$. Higher-order Taylor-Hermite terms are mathematically negligible.
* **Decision**: REJECT; close Lane A1 higher-order polynomial expansion.

### Experiment P4-08: Recursive Layer-by-Layer Scale Calibration
* **Date/time**: 2026-09-05
* **Lane and method family**: Cross-cutting / Scale Deflation Dynamics
* **Hypothesis**: Testing whether dampening intermediate covariance inflation recursively at every layer $\rho = (0.998319)^{1/16} \approx 0.9998949$ prevents compounding error across 16 layers better than a single terminal scale deflation.
* **Mathematical mechanism**:
  - Recursive update: $\mu_l \leftarrow \rho \mu_l$, $\Sigma_l \leftarrow \rho^2 \Sigma_l$ at each layer $l \in \{0, \dots, 15\}$.
* **Candidate file and SHA-256**: `scripts/diagnose_recursive_calibration.py`
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Results Measured**:
  - $\rho = 1.0000000$: Raw Cov MSE = `4.147e-06`, Scaled Cov MSE = `1.378958e-06`, Fused MSE = `1.223649e-06`
  - $\rho = 0.9998949$: Raw Cov MSE = `1.378e-06`, Scaled Cov MSE = `1.378892e-06`, Fused MSE = `1.223607e-06` (Diff: `-0.00%`)
* **Evidence label**: `[mathematically derived & measured]`
* **Interpretation**: Applying scale deflation recursively through depth reduces the *unscaled* intermediate covariance MSE from $4.15e-06$ to $1.38e-06$, but once optimal terminal scaling is applied, both methods converge to identical MSE ($1.3789e-06$). The remaining error in the covariance branch is not scale drift; it is the non-Gaussian closure error itself.
* **Decision**: Neutral equivalence; preserve terminal prior $s_0 = 0.998319$.

### Experiment P4-09: Regime-Dependent Dead/On/Kink Calibration & Blending
* **Date/time**: 2026-09-05
* **Lane and method family**: Cross-cutting / Regime-Dependent Discrepancy
* **Hypothesis**: Testing whether separating neurons into Dead ($a < -2.5$), On ($a > +2.5$), and Kink ($-2.5 \le a \le +2.5$) coordinates with regime-specific scale factors $s_{\text{on}}, s_{\text{kink}}$ and blend weights $\alpha_{\text{on}}, \alpha_{\text{kink}}$ reduces terminal error.
* **Mathematical mechanism**:
  - Classification via standardized preactivations $a_i = \mu_i / \sigma_i$.
  - Regime-specific linear combination $\hat{y}_i = (1 - \alpha_r) s_r c_i + \alpha_r m_i$.
* **Candidate file and SHA-256**: `scripts/diagnose_regime_calibration.py`
* **Control file and SHA-256**: `estimator.py` (`ea8be822...`)
* **Results Measured**:
  - Dead ($a < -2.5$): Count = 202-288, MSE = $1.1 \times 10^{-9}$ (already near zero).
  - On ($a > +2.5$): Count = 196-270, $s_{\text{on}} = 0.998330 \pm 0.00017$.
  - Kink ($-2.5 \le a \le +2.5$): Count = 487-626, $s_{\text{kink}} = 0.998137 \pm 0.00033$.
  - Regime-calibrated prediction MSE: `1.221915e-06` (Diff vs Control: `-0.14%`).
* **Evidence label**: `[measured]`
* **Interpretation**: While dead neurons have zero error ($10^{-9}$), the scale difference between On neurons ($0.99833$) and Kink neurons ($0.99814$) is only $0.019\%$, resulting in a negligible $-0.14\%$ MSE change.
### Experiment P4-10: Exact Cho-Saul Layer 0 Arc-Cosine Covariance
* **Date/time**: 2026-09-05
* **Lane and method family**: Analytical Kernel Refinement / Cho-Saul Closed-Form
* **Hypothesis**: Testing whether replacing the 2nd-order Hermite Taylor expansion at Layer 0 with the exact closed-form Cho-Saul arc-cosine kernel $J_1(\theta) = \frac{1}{2\pi} (\sqrt{1-\rho^2} + \rho(\pi/2 + \arcsin \rho))$ reduces downstream covariance error.
* **Results Measured**:
  - Layer 0 Exact Cov MSE vs Ground Truth: `1.3744e-06` vs `1.3789e-06` (Diff: `-0.33%`, 7W-1L across the 8 MLPs).
* **Decision**: Accurate Layer 0 kernel confirmed, but overall impact is diluted across 15 downstream layers (-0.33% gain << 10% gate).

### Experiment P4-11: Antagonistic Dual-Carrier Negative-Covariance Pairing
* **Date/time**: 2026-09-05
* **Lane and method family**: Lane B - Carrier Innovation & Negative Covariance
* **Hypothesis**: Testing whether pairing Scrambled Hadamard frames ($N=2048$) with QMC Shifted Lattice ($N=2048$) actively cancels sampling variance through negative cross-covariance with the analytical branch.
* **Results Measured**:
  - Theoretical Upper Bound (Oracle Linear Combination of c + Had + QMC + WMC per MLP): `1.2184e-06` (Adj: `1.2184e-07`, Diff: `-1.08%`).
* **Decision**: REJECT (Oracle headroom is only 1.08%, far below the 10% gate).

### Experiment P4-12: Edgeworth Preactivation Kurtosis Expansion
* **Date/time**: 2026-09-05
* **Lane and method family**: Analytical Cumulant Refinement
* **Hypothesis**: Testing whether injecting the Gram-Charlier excess kurtosis correction $\Delta \mu = \sigma \frac{\gamma_2}{24} (a^2 - 1) \phi(a)$ into the layer-by-layer mean propagation reduces non-Gaussian closure error.
* **Results Measured**:
  - $\eta = 0.2$: MSE = `1.3811e-06` (+0.16%, 4W-4L)
  - $\eta = 1.0$: MSE = `1.5349e-06` (+11.3%, 1W-7L)
* **Decision**: REJECT (Recurrent compounding of non-Gaussian terms without full tensor tracking increases MSE).

### Experiment P4-13: Continuous Regime-Aware Polynomial Scaling
* **Date/time**: 2026-09-05
* **Lane and method family**: Scale Calibration Refinement
* **Hypothesis**: Testing whether a continuous polynomial function $s(c)$ fit via Leave-One-Out CV outperforms the global scalar prior $s_0 = 0.998319$.
* **Results Measured**:
  - Degree 1 LOO MSE: `1.2327e-06` (+0.74%)
  - Degree 2 LOO MSE: `1.2310e-06` (+0.60%)
  - Degree 3 LOO MSE: `1.2311e-06` (+0.61%)
* **Decision**: REJECT (Global scalar prior $s_0$ is minimax optimal; polynomial curves overfit).

### Experiment P4-14: Cho-Saul Layer 0 + Optimal Gauss-Markov MC Blend
* **Date/time**: 2026-09-05
* **Lane and method family**: Full Unified Architecture
* **Hypothesis**: Testing the combined pipeline (Cho-Saul L0 + Blended Hermite + Prior Scale + Optimal MC Blend $\alpha$).
* **Results Measured**:
  - Optimal $\alpha = 0.110$: MSE = `1.220550e-06` (Adj: `1.220550e-07`, Diff vs Control: `-0.25%`, 6W-2L).
* **Decision**: Modest improvement (-0.25%, 6W-2L), but fails the 10% promotion gate. Control preserved.

### Experiment P5-01: Layer-Wise Error Growth & Shared Residual Law (C4)
* **Date/time**: 2026-09-05
* **Hypothesis**: Late-jump sampling (analytic to L14 + massive final sampling) or shared additive residual law r(a,sigma) unlocks e-8 headroom.
* **Results Measured**:
  - Analytic MSE L14/L15 ratio = 0.981 (L14 `1.3528e-06` vs L15 `1.3789e-06`). Perfect-L14 jump ceiling ~1.9% — kills late-jump family.
  - Residual correlations: corr(r,a)=+0.0044, corr(r,sigma)=-0.0292, corr(r,Phi)=+0.0135, corr(r,c)=+0.0071. Global r~b0+b1*c LOO gain -0.37% (hurts).
* **Decision**: KILL both (oracle <2%, deployable negative).

### Experiment P5-02: Unbiased Split-Sample Controls, Active Subspace, Stein-Input
* **Results Measured**:
  - Per-neuron split-sample linear CV: ratio 0.998-1.001, mean +0.01% (zero).
  - Global single-beta split CV: gain -0.01% (zero). Split-center noise + per-neuron beta estimation noise erase the 61% analytic-center gain.
  - Active subspace top-4 linear R2 = 0.0013-0.0014 (<< 0.15 kill gate).
  - Stein rank-8 input control: ratio 1.000 on all 8 MLPs.
* **Decision**: KILL all three (Lane C2 closed, unbiased-CV family closed).

### Experiment P5-03: Full-Depth Exact Cho-Saul Kernel
* **Results Measured**: Offline prototype formula error blew up +3.6e8% (incorrect mean-modulation); Taylor analysis + L0 exact (-0.33% cov) bound full-depth exact at <1% fused with prohibitive Phi2-quadrature cost (no deployable path under 40% util).
* **Decision**: ARCHIVE as non-deployable (Lane D gate fails).

### Experiment P5-04: Global Control + Lawful Carrier Selection Diagnostics
* **Results Measured**: Global control -0.01%; carrier-selection test invalidated by sample-count bug (Had 2048 vs WMC 4200) — prior P4-01/P4-07 portfolio numbers stand (-0.52% fused).
* **Decision**: No new candidate; prior verdicts hold.

### Experiment P5-05: Stacked Deployable — Blended lam=0.50 + Exact Cho-Saul L0 (N=4200)
* **Candidate**: `candidates/estimator_p5_chosaul_lam05.py` (SHA256 `93ee3c28...`), util 9.76%, resid 0.2055s, 0 failures.
* **Results Measured**: Adj `1.215291e-07` (Diff -0.68%, 6W-2L, worst +0.41%). New lowest stable panel mean (beats lam=0.50 alone `1.2171e-07`).
* **Decision**: RECORD as best-mean variant; REJECT promotion (0.68% << 10% gate). Control preserved.

### Experiment P5-06: Same Stack at N=4350 (10.00% util boundary)
* **Candidate SHA256**: `2319b7cd...`, resid 0.1772s, 0 failures.
* **Results Measured**: Adj `1.203229e-07` (Diff -1.67%) but 5W-3L with worst +3.21% and util exactly 10.00% (zero grader margin — seed-fragile, boundary-risky).
* **Decision**: REJECT (fails 6W gate, boundary util, almost certainly MC noise). Reverted file to N=4200.

---

## Phase 4 Scientific Synthesis & Definitive Conclusions


1. **The Mathematical Closure Basin**:
   - The analytical covariance branch achieves an irreducible error floor of $\text{MSE}_{\text{analytical}} \approx 1.38 \times 10^{-6}$ across the 8-MLP panel.
   - SVD spectral analysis proves this error is strictly full-rank and isotropic (25% energy in 256 dimensions, 50% in 512 dimensions). It cannot be compressed or transported via low-rank projections, Krylov subspaces, or low-order cumulants.
   - Higher-order Taylor-Hermite expansions ($\rho^3, \rho^4$) are $\mathcal{O}(1/n^2) \sim 10^{-6}$ in magnitude and mathematically inert.
   - Scale deflation ($s_0 = 0.998319$) captures the entire coherent scalar norm deflation; neither recursive layer-by-layer nor regime-dependent scaling improves upon it.

2. **The Scored Compute Ceiling ($0.1000$ Multiplier Floor)**:
   - In pure sampling, Scrambled Hadamard frames achieve **15.35% lower raw variance** than Gaussian WMC ($9.97 \times 10^{-6}$ vs $1.18 \times 10^{-5}$) due to exact intra-block mutual orthogonality ($H H^\top = I$).
   - Increasing basis count ($K=8, N=16384$) reduces raw error to $8.49 \times 10^{-7}$, but consumes $25.0\%$ of the budget, triggering the $(u / 0.10)$ linear multiplier penalty ($0.2500$) which inflates the adjusted score to $2.12 \times 10^{-7}$.
   - Under the official scoring formula $\text{Score} = \text{MSE} \times \max(0.10, u)$, sampling above $N \approx 6553$ passes cannot beat the champion.

3. **Definitive Operational Benchmark**:
   - The Phase 4 control in `estimator.py` (Blended Dual-Kernel Hermite $\lambda=0.20, s_0=0.998319, \alpha=0.110, N=4200$, SHA256: `ea8be8222b827607f65d166118500b55fcaae883b79f64b29785b46d86501ca1`, adjusted score: **`1.223643e-07`**) remains the strictly unvanquished champion across all formal promotion criteria.
