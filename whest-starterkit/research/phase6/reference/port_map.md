# Phase 6 Reference Port Map: `mlp_kprop` -> Benchmark Architecture

## 1. Provenance & Pinned Source
- **Repository:** `alignment-research-center/mlp_cumulant_propagation`
- **Pinned Commit:** `93d091a4c26c042bfffa28f2e76a81bc0aba94bb`
- **Import Package:** `mlp_kprop`
- **License & Attribution:** MIT License / Alignment Research Center. All rights and mathematical definitions preserved.
- **Isolated Environment:** `research/phase6/reference/.venv_ref` (CPython 3.12.10, PyTorch 2.14.0+cpu, NumPy 2.5.2, SciPy 1.18.1). Isolated completely from the whestbench production environment.

---

## 2. Mathematical Objects & Data Structures

| `mlp_kprop` Class / Type | Mathematical Role | Description |
|---|---|---|
| `FactoredTensor` (`factor_k3.py`) | Factorized symmetric cumulant $\kappa_3$ | Represents $T = \text{Sym}\left(\sum_{r=1}^R a_r \otimes b_r \otimes c_r\right) + T_{\text{repeated}}$, storing factor columns `(A, B, C)` of shape `(n, R)` and diagonal repeated slices in a `DSTensor`. |
| `DSTensor` (`diagslice.py`) | Diagonal-slice tensor | Stores slices with repeated indices (partitions like `(2, 1)`, `(3,)`, `(2, 2)`). |
| `HTensor` (`harmonic.py`) | Harmonic tensor representation | Harmonic decomposition with core tensor and metric $g$, preventing unmetered metric accumulation when $r=0$. |
| `HTower` / `DSTower` | Multi-degree container | Dictionary mapping degree $d \in \{1, 2, 3, 4\}$ to `HTensor` or `FactoredTensor`. |
| `IntPartition` / `VecPartition` (`partitions.py`) | Diagram combinatorics | Multi-indices and partition graphs for Wick expansions and moment contractions. |

---

## 3. Function Mapping & Conventions

| Reference Function (`mlp_kprop`) | Phase 6 Port / Benchmark Wrapper | Convention Differences & Critical Notes |
|---|---|---|
| `linear_kprop(K, W, k_max=3)` | `p6_linear_kprop(K, W_bench)` | **Transpose Convention:** Reference assumes column vector $z = W_{\text{ref}} x$. Benchmark uses row vector $z = x W_{\text{bench}}$. Therefore, $W_{\text{ref}} = W_{\text{bench}}^T$. Factor columns are transported via $W_{\text{bench}}^T A$. |
| `factored_nonlin_kprop_k3(..., base=True, augment=False)` | `p6_nonlin_kprop_k3_base` (K3-base) | Reference base third-order mode without augmented terms or fourth-order state. |
| `factored_nonlin_kprop_k3(..., base=False, augment=False)` | `p6_nonlin_kprop_k3_simple` (K3-simple) | Reference mode with selected fourth-order state (tracked degree-4 core and metric). |
| `factored_nonlin_kprop_k3(..., base=False, augment=True)` | `p6_nonlin_kprop_k3_augment` (K3-augment-filtered) | Reference factorized augmented term set. **Critical Caveat:** Drops non-hypertree diagrams (covariance-triangle and kappa3-edge products). Output matches filtered dense reference, NOT full unfiltered augmented output. |
| `relu_wick_coef` (`wick.py`) | Exact Gaussian ReLU Wick coefficient calculator | Evaluates $E_{Z \sim \mathcal{N}(\mu, \sigma^2)}[\partial^k \text{ReLU}(Z)^p]$. |
| `factored_keeps_term` (`kprop_harmonic.py`) | Term filtration oracle | Defines exact boolean predicate for hypertree diagrams kept by the factored engine. |

---

## 4. Activation and Transpose Parity Contract
1. **Row Vector Convention:**
   - In whestbench: $h_l = \text{ReLU}(h_{l-1} W_l)$.
   - For state $\mu$: $\mu_{\text{pre}} = W_l^T \mu_{\text{post}, l-1}$.
   - For covariance $\Sigma$: $\Sigma_{\text{pre}} = W_l^T \Sigma_{\text{post}, l-1} W_l$.
   - For factor matrix $A$: $A_{\text{pre}} = W_l^T A_{\text{post}, l-1}$.
2. **Terminal Activation:**
   - In whestbench: All 16 layers (layers 0 through 15) end in ReLU. The reference wrapper omits activation on the last layer; Phase 6 applies explicit nonlinear propagation after **all 16 layers**.
