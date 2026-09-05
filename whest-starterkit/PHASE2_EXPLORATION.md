# Phase 2 Exploration & Method Evolution

## Strategy Map: Transitioning from Phase 1 (256x32) to Phase 2 (1024x16)

This document maps reproducible methods from Phase 1 to Phase 2, explicitly accounting for the fourfold increase in width, halving of depth, `2**41` FLOP budget, and the 400 ms residual time hard cap ($\lambda = 0$).

---

### 1. Structured Randomized Cubature & Spherical Conditioning (Family 1)

* **Phase 1 Mechanism**:
  Positive homogeneity of ReLU ($h(\alpha x) = \alpha h(x)$) separates Gaussian radial length $r \sim \chi(d)$ from direction $u \in \mathcal{S}^{d-1}$. Integrating $r$ analytically eliminates radial variance. In $d=256$, full Kerdock/MUB sets (66,048 directions across 129 bases) provided a spherical 5-design where antipodal pairing cancels all odd-degree polynomial errors.
* **Phase 2 Architectural Shift (256x32 -> 1024x16)**:
  In $d=1024$, the complete real Kerdock construction would require $2^{10-1} \times (2^{10} + 1) \approx 524,800$ directions, which far exceeds the $\approx 65,536$ maximum forward passes allowed by the $2^{41}$ FLOP budget.
* **Phase 2 Hypothesis**:
  Exact radial conditioning ($\mathbb{E}[R] = \sqrt{2}\frac{\Gamma(1025/2)}{\Gamma(1024/2)} \approx 31.992186$) eliminates 100% of radial variance. Generating randomized orthogonal basis blocks via Sylvester-Hadamard matrices with random Rademacher sign flips ($D_1 H D_2$) paired antipodally provides an exact orthogonal frame on $S^{1023}$ requiring zero eigendecomposition (`eigh`) FLOPs.
* **Smallest Viable Prototype**:
  `candidates/estimator_hadamard_hermite.py`: Fused Sylvester-Hadamard frame ($N=4,096$) with Hermite-corrected covariance.
* **8-MLP Result & Decision**:
  - Raw MSE: `2.9442e-06`, Adjusted Score: `2.9442e-07`, Compute: `8.61%`, Max Residual Time: `0.1901s`.
  - Head-to-head: **4W-4L**. Achieved massive gains on specific MLPs (e.g. Christopher-Morales: -24.49% WIN, Sarah-Kelley: -20.27% WIN, Raymond-Barnes: -15.34% WIN), but suffered on networks sensitive to hypercube-vertex diagonal anisotropy.

---

### 2. ReLU Activation-Region Routing & Dead Neuron Analysis (Family 2)

* **Phase 1 Mechanism**:
  Neuron variance concentrates heavily around the kink ($x=0$). Confirmed dead neurons ($h_{l,i} \equiv 0$) and always-active linear neurons ($h_{l,i} > 0$) were detected via cheap pilot samples and analytic bounds, omitting dead columns from matrix multiplies.
* **Phase 2 Architectural Shift (256x32 -> 1024x16)**:
  Depth is halved (16 vs 32), but width is $1024$. Analysis showed that by layer 10 over 116 neurons are dead, and by layer 15 nearly 200 neurons (~20% of the layer) are completely dead.
* **Phase 2 Hypothesis**:
  Pruning dead columns can save ~15-20% GEMM FLOPs in the late layers.
* **8-MLP Evaluation & Decision**:
  - Python dynamic slicing overhead (`act[:, active]`) introduces memory allocation latency that risks violating the strict 400 ms residual time hard cap. With the 10% compute multiplier floor already clamping the score penalty, physical wall-time safety was prioritized over dynamic indexing.

---

### 3. Mid-Network Covariance-Anchored Control Variates (Family 3)

* **Phase 1 Mechanism**:
  In #327723 (mliston), cumulant propagation was compared against sample means at layer 22, and the difference was propagated to the final layer as a control variate, removing ~45% of final sampling variance.
* **Phase 2 Architectural Shift (256x32 -> 1024x16)**:
  In Phase 2 with 16 layers, covariance propagation is exceptionally accurate up to layers 6–8 (error $< 10^{-7}$).
* **Phase 2 Hypothesis**:
  Using covariance at layer $k \approx 8$ as an exact control variate:
  $$\hat{\mu}_{16} = \hat{\mu}_{16}^{\text{mc}} - \beta \mathbf{W}_{k \to 16} (\hat{h}_k^{\text{mc}} - \mu_k^{\text{cov}})$$
* **Smallest Viable Prototype**:
  `candidates/estimator_mid_cv.py`: Anchored at layer 8 with linear Jacobian forwarding.
* **8-MLP Result & Decision**:
  - Raw MSE: `3.3391e-06`, Adjusted Score: `3.3391e-07`, Compute: `9.80%`, Max Residual Time: `0.1178s`.
  - Head-to-head: **2W-6L** -> **REJECTED**. The tangent linear Jacobian approximation diverges across 8 non-linear ReLU layers on sensitive MLPs, introducing extrapolation noise.

---

### 4. Correlation-Aware Hermite Covariance Propagation (Family 4 - BREAKTHROUGH)

* **Phase 1 Mechanism**:
  Standard Gain-Covariance baseline approximated post-ReLU covariance via statistical linearization (equivalent gain):
  $$\text{Cov}(\text{ReLU}(z_a), \text{ReLU}(z_b)) \approx \Phi(\alpha_a) \Phi(\alpha_b) \Sigma_{ab}$$
* **Mathematical Discovery**:
  Rigorous Hermite polynomial expansion of the bivariate Gaussian arc-cosine kernel proved that the standard gain approximation systematically underestimates cross-covariance by **5% to 30%** because it truncates the series at degree 1. The exact second-order term derived from Price's theorem is strictly positive:
  $$+ \frac{1}{4\pi} \rho_{ab}^2 \sigma_a \sigma_b \approx + 0.079577 \rho_{ab}^2 \sigma_a \sigma_b$$
* **Phase 2 Hypothesis**:
  Adding the quadratic Hermite term $+ \gamma \cdot \frac{1}{4\pi} \rho_{ab}^2 \sigma_a \sigma_b$ directly repairs the covariance shrinkage across 16 layers, dramatically lowering covariance bias.
* **Smallest Viable Prototype**:
  `candidates/estimator_sweep_hermite.py` ($\gamma = 0.45$, blend $0.75 / 0.25$, $N=4,200$).
* **8-MLP Result & Decision**:
  - Raw MSE: **`2.9223e-06`**, Adjusted Score: **`2.9223e-07`** (Baseline: `3.1749e-07`).
  - Head-to-head: **8W-0L CLEAN SWEEP** (all 8 official MLPs beat baseline).
  - Compute Utilization: **`9.81%`** (safely below 10% floor, multiplier = 0.1000).
  - Max Residual Time: **`0.3241s`** (safely below 0.40s hard cap).
  - **DECISION: KEPT AS CURRENT CHAMPION (BROKE 3.0e-7 BARRIER)**.

---

### 5. Multi-Branch Tri-Blend Architecture (Family 6)

* **Phase 2 Hypothesis**:
  Combine 3 complementary estimators: 72% Hermite Covariance + 14% Isotropic Gaussian WMC ($N=2500$) + 14% Rigid Hadamard Spherical Cubature ($N=2048$).
* **8-MLP Result & Decision**:
  - Raw MSE: `3.0696e-06`, Compute: `10.17%`, Multiplier: `0.1017` (slight penalty exceeding 10% floor), Adjusted Score: `3.1228e-07`.
  - Head-to-head: **5W-3L** -> **REJECTED** in favor of the clean 8W-0L Hermite champion.

---

### 6. Unbiased Scale Projection & Precision Shrinkage (Family 7 - MAJOR BREAKTHROUGH)

* **Mathematical Discovery**:
  Across 16 layers, covariance propagation maintains a near-perfect correlation ($\rho \approx 0.999999$) with ground-truth activations, but suffers from systematic global scale inflation ($\approx +0.16\%$, scale factor $s \approx 0.9984$). Because the Monte Carlo sample mean $\mathbf{m}$ is strictly unbiased ($\mathbb{E}[\mathbf{m}] = \mathbf{y}$), the inner product projection:
  $$\hat{s} = \frac{\langle \mathbf{m}, \mathbf{c} \rangle}{\langle \mathbf{c}, \mathbf{c} \rangle}$$
  provides an unbiased, variance-negligible estimator of the exact scale deflation.
* **Precision Weighting (Kalman / Gauss-Markov Blend)**:
  Once covariance is calibrated ($s \mathbf{c}$ has MSE $\approx 1.5 \times 10^{-6}$ and near-zero bias), blending with unbiased Monte Carlo ($\text{Var}(\mathbf{m}) \approx 8.8 \times 10^{-6}$) using the harmonic precision weight:
  $$\alpha^* = \frac{\text{MSE}(s \mathbf{c})}{\text{MSE}(s \mathbf{c}) + \text{Var}(\mathbf{m})} \approx 0.13$$
  yields the minimal error estimator:
  $$\hat{\mathbf{y}} = (1 - \alpha^*) (s \mathbf{c}) + \alpha^* \mathbf{m}$$
* **Smallest Viable Prototype**:
  `candidates/estimator_calibrated_hermite.py` ($\alpha = 0.130$, $N=4,200$, $\gamma = 0.50$).
* **8-MLP Result & Decision**:
  - Raw Final MSE: **`1.6500e-06`** (Baseline: `3.1749e-06`, -48.0% error reduction).
  - Adjusted Score: **`1.6500e-07`** (Baseline: `3.1749e-07`, Prior best: `2.9078e-07`).
  - Head-to-head: **7W-1L** (6 MLPs won by 50% to 65%).
  - Compute Utilization: **`9.81%`** (comfortably under 10% floor, multiplier = 0.1000).
  - Max Residual Time: **`0.1463s`** (well below 0.40s hard cap).
  - **DECISION: NEW CHAMPION. TARGET < 2.5e-7 DECISIVELY ACHIEVED.**

