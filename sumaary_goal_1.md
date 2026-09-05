# ARC White-Box Estimation Challenge 2026 (Phase 2): Research & Experimentation Synthesis

This document provides a comprehensive technical retrospective of the exploration and algorithmic developments conducted on the Phase 2 benchmark (`1024 x 16`, FLOP budget $2^{41} \approx 2.199 \times 10^{12}$, 400 ms residual wall-time cap, 10% compute score floor).

---

## 1. Executive Summary & Benchmark Score Trajectory

* **Baseline Reference**: Public Phase 2 submission recipe (75% Gain Covariance + 25% Whitened Antithetic MC, $N=4,038$).
  * **Score**: `3.1749e-07` (Raw Final MSE: `3.1749e-06`, Multiplier: `0.1000`, Head-to-Head: Baseline).
* **Initial Exploration**: Micro-tuning samples ($N=4,200$) and 2nd-order Hermite expansion ($\gamma \in [0.38, 0.50]$).
  * **Score**: `2.9078e-07` (Plateaued: parameter tweaks could not achieve the target).
* **Target Set by User**: Break through the plateau, beat `3.0e-7`, and achieve **`< 2.5e-7`**.
* **Final Champion Score**: **`1.6500e-07`** (Raw Final MSE: **`1.6500e-06`**, Multiplier: `0.1000`, 7W-1L on 8-MLP panel).
  * **Net Improvement**: **-48.0% error reduction** over baseline, decisively achieving the target.

---

## 2. What Worked (The Core Breakthroughs)

### A. The Calibration Discovery: Unbiased Monte Carlo Scale Projection
* **The Root Cause**: Covariance propagation tracks the relative activations across neurons with near-perfect correlation ($\rho \approx 0.999999$). However, over 16 non-linear ReLU transformations, Gaussian linearization systematically overestimates the overall energy scale by approximately $+0.16\%$ (scale factor $s \approx 0.9984$). Due to the large magnitude of activation vectors ($\|\mathbf{y}\| \approx 12.8$), this tiny scale inflation accounted for over $65\%$ of the total Covariance MSE!
* **The Solution**: The Whitened Antithetic Monte Carlo mean $\mathbf{m}$ is strictly unbiased ($\mathbb{E}[\mathbf{m}] = \mathbf{y}$). By computing the orthogonal projection of $\mathbf{m}$ onto the covariance prediction $\mathbf{c}$:
  $$\hat{s} = \frac{\langle \mathbf{m}, \mathbf{c} \rangle}{\langle \mathbf{c}, \mathbf{c} \rangle} = \frac{\sum_{i=1}^{1024} m_i c_i}{\sum_{i=1}^{1024} c_i^2}$$
  we obtain an unbiased estimator of the scale factor with negligible standard error ($\sigma_s \approx 0.00025$).
* **Impact**: Simply scaling the covariance prediction by $\hat{s}$ dropped the raw Covariance MSE from **`4.5e-06` down to `1.5e-06`** across all MLPs with zero extra forward passes.

### B. Gauss-Markov / Harmonic Precision Blending
* Because the Monte Carlo sampling error $\mathbf{\epsilon}_{\text{mc}} = \mathbf{m} - \mathbf{y}$ is generated from independent pseudo-random standard normal samples, it is mathematically uncorrelated with the deterministic covariance approximation error ($\text{Cov}(\hat{s}\mathbf{c} - \mathbf{y}, \mathbf{m} - \mathbf{y}) = 0$).
* The optimal minimum-variance linear combination (Kalman / Gauss-Markov precision weight) is:
  $$\alpha^* = \frac{\text{MSE}(\hat{s}\mathbf{c})}{\text{MSE}(\hat{s}\mathbf{c}) + \text{Var}(\mathbf{m})}$$
  Plugging in the empirical values ($\text{MSE}(\hat{s}\mathbf{c}) \approx 1.5 \times 10^{-6}$, $\text{Var}(\mathbf{m}) \approx 8.8 \times 10^{-6}$):
  $$\alpha^* \approx \frac{1.5}{1.5 + 8.8} \approx 0.130$$
* Resulting in the precision-blended estimator:
  $$\hat{\mathbf{y}} = (1 - \alpha^*)(\hat{s}\mathbf{c}) + \alpha^* \mathbf{m} \quad (\text{Score: } \mathbf{1.6500 \times 10^{-7}})$$

### C. Hermite Second-Order Cross-Covariance Correction
* Standard gain linearization truncates the bivariate Gaussian arc-cosine kernel at degree 1.
* By adding the quadratic Hermite polynomial term derived from Price's theorem:
  $$\mathbf{\Sigma}_{ab}^{\text{post}} = g_a g_b \mathbf{\Sigma}_{ab}^{\text{pre}} + \gamma \cdot \frac{1}{4\pi} \rho_{ab}^2 \sigma_a \sigma_b \quad (\gamma = 0.50)$$
  the cross-neuron correlation matrix is prevented from artificial shrinkage through depth.

### D. Setup & Threadpool Warmup Scaffold
* Python import latency and dynamic BLAS threadpool spawning originally caused the first MLP evaluated (`logan-fitzgerald`) to spike to `0.4090s` residual time, exceeding the 0.4000s hard cap and triggering an automatic zero-score penalty.
* Introducing a lightweight 2ms warmup routine inside `setup(self, ctx)` pre-allocated the threadpool and caches, bringing all 8 MLPs down to a consistent, safe `0.1463s` residual wall time.

---

## 3. What Didn't Work (and Why)

| Attempted Method | Hypothesis | Outcome | Failure Rationale |
| :--- | :--- | :--- | :--- |
| **Mid-Network Control Variate (`estimator_mid_cv.py`)** | Anchor covariance at layer 8 and propagate the discrepancy to layer 16 via linear Jacobian. | **REJECTED** (Score: `3.3391e-07`, 2W-6L) | Tangent linear Jacobian extrapolation diverges across 8 consecutive non-linear ReLU layers, introducing massive extrapolation noise on sensitive networks. |
| **Hadamard Spherical Cubature (`estimator_hadamard_hermite.py`)** | Replace standard Gaussian sampling with Sylvester-Hadamard orthogonal frames ($D_1 H D_2$) paired antipodally. | **REJECTED** (Score: `2.9442e-07`, 4W-4L) | Fixed hypercube vertex orientations suffer from diagonal anisotropy; networks sensitive to coordinate rotations lost accuracy. |
| **Multi-Branch Tri-Blend (`estimator_tri_blend.py`)** | Blend 72% Hermite Covariance + 14% WMC ($N=2500$) + 14% Hadamard ($N=2048$). | **REJECTED** (Score: `3.1228e-07`, 5W-3L) | Running two distinct sampling branches pushed compute to 10.17% ($> 10\%$ floor), incurring a score multiplier penalty ($0.1017$). |
| **Pure Covariance (`estimator_pure_cov.py`)** | Rely entirely on analytical propagation with zero sampling. | **REJECTED** (Score: `4.3942e-07`, 0W-8L) | Uncalibrated covariance suffers from systematic non-Gaussian scale drift over 16 layers. |
| **Pure WMC (`estimator_pure_wmc.py`)** | Rely entirely on whitened antithetic sampling ($N=4,200$). | **REJECTED** (Score: `1.1524e-06`, 0W-8L) | Pure sampling variance ($\sigma^2 / N$) is too high under the 10% compute budget floor. |
| **Dead-Neuron Dynamic Slicing** | Slice out dead neuron columns ($h_{l, i} \equiv 0$) dynamically in GEMM. | **REJECTED** (Engineering Risk) | Dynamic slicing in Python introduces memory allocation overhead and indexing latency, risking the 400ms residual wall-time cap. |
| **Dead-Neuron Hard Gating / Thresholding** | Set predictions to analytical covariance for all neurons where $\mu < 5 \times 10^{-4}$. | **NO EFFECT** (+0.00%) | At $N=4,200$, neurons that are analytically dead had zero empirical firings in Monte Carlo as well, rendering a post-hoc threshold redundant. |

---

## 4. What Other Diagnostic Tests Explained

1. **Exact Radial Conditioning ($R = \mathbb{E}[R]$)**:
   * Tested evaluating unit sphere directions scaled by $\mathbb{E}[R] = \sqrt{2}\frac{\Gamma(1025/2)}{\Gamma(512)} \approx 31.992186$.
   * **Result**: WMC MSE moved from `8.8458e-06` to `8.8075e-06` (only 0.4% difference).
   * **Explanation**: In $d=1024$, the concentration of measure theorem dictates that radial variance is $\frac{1}{2d} = \frac{1}{2048} \approx 0.048\%$. Gaussian vectors in 1024-D are already naturally concentrated on a thin spherical shell, so eliminating radial variance yields negligible gains in Phase 2 (unlike Phase 1 where $d=256$).
2. **Layer-by-Layer Error Progression**:
   * Measuring MSE per layer revealed that Covariance propagation is virtually exact at early layers (Layer 0 MSE: $6.97 \times 10^{-10}$; Layer 1: $1.57 \times 10^{-7}$).
   * In contrast, Monte Carlo sample variance is flat at $\sim 10^{-5}$ across all layers. Blending Monte Carlo into early layers actually degrades their accuracy by $16,000\times$!
   * Because leaderboard evaluation is scored **strictly on the final layer (Layer 15)**, preserving pure covariance on early layers and applying the calibrated blend only to the final layer is theoretically optimal.
3. **Correlation vs. Bias**:
   * Covariance correlation with ground truth: **`0.999999`**.
   * Monte Carlo correlation with ground truth: **`0.999993`**.
   * Covariance is fundamentally a cleaner "shape" of the network's state than Monte Carlo, but was held back by the scale bias that our projection method resolved.

---

## 5. Summary of Validation Table (from `Current thing.md`)

| Method / Variant | Short Description | Raw Final MSE | Score Mult | Adjusted Score | Residual Time | Failures | 8-MLP Wins | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Blend 75/25)** | Gain Cov (0.75) + WMC (0.25, N=4038) | 3.1749e-06 | 0.1000 | 3.1749e-07 | 0.1309s | 0 | - | Baseline Reference |
| **Variant N=4200** | Gain Cov (0.75) + WMC (0.25, N=4200) | 3.1575e-06 | 0.1000 | 3.1575e-07 | 0.1340s | 0 | 3W-5L | Minor Gain |
| **Mid-Network CV** | Layer-8 Covariance Anchor CV on WMC | 3.3391e-06 | 0.1000 | 3.3391e-07 | 0.1178s | 0 | 2W-6L | REJECT |
| **Pure Covariance** | Gain Cov (1.00) alone | 4.3942e-06 | 0.1000 | 4.3942e-07 | 0.0712s | 0 | 0W-8L | REJECT |
| **Pure WMC** | WMC alone (N=4200) | 1.1524e-05 | 0.1000 | 1.1524e-06 | 0.0994s | 0 | 0W-8L | REJECT |
| **Hermite Cov (g=0.50, w=0.75)** | 2nd-order Hermite Cov (0.75) + WMC (0.25) | 2.9078e-06 | 0.1000 | 2.9078e-07 | 0.1761s | 0 | 8W-0L | Previous Best |
| **Hadamard Cubature** | Sylvester Hadamard + Hermite Cov | 2.9442e-06 | 0.1000 | 2.9442e-07 | 0.1901s | 0 | 4W-4L | REJECT |
| **Tri-Blend Architecture** | Hermite Cov + WMC + Hadamard | 3.0696e-06 | 0.1017 | 3.1228e-07 | 0.3638s | 0 | 5W-3L | REJECT |
| **Self-Calibrated Hermite Cov** | **Hermite Cov + MC scale projection + precision blend** | **1.6500e-06** | **0.1000** | **1.6500e-07** | **0.1463s** | **0** | **7W-1L** | **TARGET ACHIEVED (<2.5e-7)** |

---

## 6. What to Do Next (Recommended Roadmap)

1. **Neuron-Wise Variance-Adaptive Precision Weighting**:
   * Currently, the blend weight $\alpha = 0.130$ is scalar across all 1024 neurons.
   * However, Covariance propagation already computes the post-activation marginal variance $\sigma_i^2 = \text{var\_post}[15, i]$ for every single neuron.
   * Formulating neuron-specific Kalman gains:
     $$\alpha_i^* = \frac{\tau^2}{\tau^2 + \sigma_i^2 / N}$$
     where $\tau^2 \approx 1.5 \times 10^{-6}$ will place higher trust in Monte Carlo for neurons with tiny variance and higher trust in Covariance for neurons with wide variance.
2. **Third- and Fourth-Order Hermite Tensor Terms**:
   * Investigate whether adding the degree-4 Hermite term ($\rho_{ab}^4$) can account for the non-Gaussian excess kurtosis without exceeding FLOP or wall-time constraints.
3. **Packaging for Submission (Stage 5)**:
   * The current [estimator.py](file:///d:/ALL%20CODES/AICROWD%20COMPETITION/whest-starterkit/estimator.py) is fully validated and ready. Run the packaging validation tools to verify archive building and submit to the AIcrowd Phase 2 leaderboard to record official public score.
