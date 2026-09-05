# Phase 6 ideas: what could plausibly change the accuracy scale?

Research sketch, 6 September 2026. Updated after Phase 5 was reported complete through P4N-15. This revision checks the saved numerical artifacts and generating code rather than adopting the report's explanations. These are hypotheses and research judgments, not an agent runbook or a claim that any proposed method works. Verification remains capped at **eight MLPs**. No new MLP evaluations were run for this revision.

## My present view

**The strongest next bet is a better representation of the joint activation distribution, used to improve a late-layer control. The alternative with substantial public evidence of untapped capacity is network-specific signed angular integration—but its cheap implementation is an unsolved problem, not an off-the-shelf method.**

I would give these two directions most of the serious mathematical attention. A different sampler, another scalar calibration, or a more accurate Gaussian covariance formula may improve the current result, but the evidence does not suggest they individually close the gap to the leaders.

**Research priority clarified:** the user's current best leaderboard score is `1.14e-7` (user-reported). A 10% reduction gives `1.026e-7`; even a 50% reduction gives `5.7e-8`. Neither addresses the approximately 33-fold gap to the earlier `3.5e-9` leaderboard snapshot. Phase 6 should prioritize mechanisms with a credible path to an order-of-magnitude improvement and ultimately the full gap. This is a requirement on the mechanism's plausible capacity, not a demand that its first prototype already achieve tenfold improvement. Optimizing the existing blend is deferred; it is not a prerequisite for pursuing a new representation or integration method.

The most useful shift is from “which named method has not been tried?” to **“what information does the current estimator discard, and can we retain the useful part at an affordable cost?”** Many appealing names already appear in Phase 1 negative-result archives. Several also appear in our own earlier plans. Reopening a direction needs a changed mathematical mechanism, not a renamed implementation.

## What completion of Phase 5 changes

**The principal research directions remain, but the evidence does not justify closing as many alternatives as the final report claims.** The new closing sections report the unchanged control score; they do not introduce a better candidate. The clearest change to this sketch is to separate measured unsuccessful variants from unresolved experiments and unverified optimization claims.

- **The control is intact.** Its current SHA-256 still matches `ea8be8222b827607f65d166118500b55fcaae883b79f64b29785b46d86501ca1`. The saved P4N-00 receipt independently supports adjusted `1.2236425135e-7`, raw `1.2236425135e-6`, eight valid networks, 9.8052% mean utilization and maximum residual `0.1746443 s`. The P4N-14/15 sections repeat these values. I did not locate a separate closing-run receipt, new sampler-salt verification, or a measured optimization improvement; this limits what can be verified from the files, rather than proving that no closing command ran. [Saved receipt](<D:/ALL CODES/AICROWD COMPETITION/whest-starterkit/research/phase4_next/results/P4N-00_20260906_000232.json>).
- **P4N-03 remains an evidence gap.** The completed report has no high-accuracy intervention result for this experiment, and no corresponding result artifact was located. Its key question—whether accurate mean, covariance or mixed-cumulant information changes the final error—therefore remains open. The small-sample P4N-12 calculation is not a replacement.
- **The four-way blend result is a measured mixture, not an established optimum.** P4N-13 saves exactly its starting weights `[0.80, 0.05, 0.10, 0.05]`; its reported global optimum and leave-one-out scores agree to every saved digit. The code optimizes an unscaled approximately `1e-6` loss with default tolerances and saves no convergence diagnostics or fold weights. These facts do not certify the claimed 3.11% maximum headroom. Its fourth branch is also just `0.999 * analytic_prediction`, not an Edgeworth calculation. [Code](<D:/ALL CODES/AICROWD COMPETITION/whest-starterkit/scripts/p4n_13_complementary_combinations.py:146>), [saved scores and weights](<D:/ALL CODES/AICROWD COMPETITION/whest-starterkit/research/phase4_next/diagnostics/p4n_13_complementary_combinations.json>).
- **The report supplies no evidence for its explanation of the leaders.** Its assertion that Puffi, J2W and marius_binner reached their scores by implementing ARC's `kprop` is unsupported by a linked disclosure or reproduced result. Nor does finite-order cumulant propagation eliminate approximation bias. The paper gives factorized runtime scaling of order `L² n^K` for `K ≥ 3`, with constants and qualifications; describing the entire method as cheap `O(nR)` third/fourth-order transport is insufficient. ARC remains a research direction, not a verified reconstruction of the winning methods. [ARC paper, Sections 4–5](https://arxiv.org/html/2605.05179v2).

The final report's “absolute sampling floor” also overreaches. A variance-over-sample-count calculation describes the specified sampler and its per-sample cost. It cannot rule out a new control, coupling or integration rule that changes that variance–cost product. Likewise, preserving 10% utilization in the unchanged control does not establish that all better methods must stay below 10%.

For research prioritization, this means **more confidence in avoiding the exact failed implementations, not more confidence that the surrounding mathematical families are impossible**. Identifying the missing error channel remains valuable because it can direct a fundamentally different estimator. Establishing the precise ceiling of the old blend is deferred: an unsupported closure claim does not itself make that investigation worth doing. This is not a request to restart the Phase 5 sweep.

## The actual gap, and what Phase 5 currently tells us

The formal local control remains `1.223643e-7` adjusted, with raw MSE `1.223643e-6`. The skew-eta-0.25 candidate recorded in `Current thing.md` has local adjusted `1.189367e-7`, an eight-of-eight improvement of about 2.8%; submission #329946 is recorded, but I have not verified its official outcome.

The leaderboard snapshot checked on 6 September for the original sketch showed Puffi and J2W at displayed adjusted `3.5e-9`, with raw MSEs `2.23e-8` and `1.97e-8`, respectively, at roughly 16–18% utilization. That is about **34 times lower adjusted error** than our best recorded local result, and roughly **53–60 times lower raw error**. These are different evaluation panels, so the ratios describe scale, not a paired performance comparison. The board does not disclose the algorithms responsible. [Official leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards).

For a single network, writing the score schematically as

\[
S=e\max(0.1,u),
\]

a `3.5e-9` score requires raw MSE about `3.5e-8` at the floor, or `1.75e-8` at 20% utilization. At aggregate level the benchmark averages the per-network products; multiplying two reported averages is not generally identical. The implication is practical: **a sufficiently large accuracy gain can justify exceeding 10% utilization**. Staying below the floor is useful for the current estimator, not a universal design requirement.

My reading of the local evidence is:

| Observation | What it supports | What it does not establish |
|---|---|---|
| P4N-01: a correctly mapped layer-14 control with the true activation mean removes about 95.2% of ordinary antithetic sampling MSE. | Deep internal fluctuations contain a strong control signal. | That a deployable center exists, or that 95% raw reduction becomes 95% adjusted improvement. |
| P4N-06: projected center error falls from about `2.87e-6` to `6.55e-7`; best nominal fused result is `1.2441e-7`. | Calibration removes a large systematic component. | A win over the immutable `1.223643e-7` control. |
| P4N-12: reported projected center reduction is 82.74%, to `5.6730e-7`; nominal fusion is `1.281359e-7`. | The reused center scaling is valuable relative to an unscaled center. | A new cumulant breakthrough: the center is literally `0.99834 * c_stack[14]`, inherited from P4N-06. The fused result is about 4.7% worse than the immutable control. |
| P4N-13: the saved starting mixture scores `1.258365e-7`, with 7/8 wins against its internal comparator. Pairwise LOO scores are `1.281267e-7` and `1.271732e-7`. | The evaluated mixture and pairwise refits fail to beat the immutable control: approximately 2.84%, 4.71% and 3.93% worse, respectively. | A valid four-branch optimum, independent four-method complementarity, or a ceiling on better combinations. The pairwise LOO loops exist, but the global headroom claim is unverified. |
| P4N-14/15: the closing report gives unchanged adjusted `1.223643e-7`; estimator hash is unchanged. | Retention of the existing control, whose score is supported by the saved P4N-00 receipt. | A new accuracy gain, independently documented closing replay, or a newly measured compute-frontier improvement. |
| P4N-02: resolving the Gaussian covariance integral more accurately brings no large gain. | Numerical Gaussian-kernel accuracy is unlikely to be the main missing ingredient. | That the true non-Gaussian joint covariance correction is unimportant. |
| P4N-07: surrogate costs are 56–94% of a full pass; even the best residual does not repay them. | Nearly full-depth surrogate networks have poor economics here. | That an exceptionally cheap nonlinear surrogate or analytically integrated surrogate cannot work. |
| P4N-08/09/10/11: input-direction mixtures, shallow Stein features, tested frame substitutions, early signed calibration, and single-line quadrature fail to beat the control. | These particular ways of spending the budget are weak. | A general impossibility theorem for joint conditioning, certified MUB constructions, or signed integration. |

The diagnostic scripts often use ordinary antithetic Gaussian draws, different sample counts, and a nominal `0.1` score multiplier. Their numbers are useful for mechanism comparisons, but are not all metered end-to-end replacements for the WMC control. Several report percentages use an internal comparator rather than the immutable champion.

There is also a boundary on the P4N-12 conclusion. Its “oracle” moments come from 16,384 samples, not the public high-accuracy moment bake. The nominal one-, two-, and four-layer variants produce identical results because the code recomputes the same terminal readout. Its covariance-mode branch subtracts an outer mean from a raw second moment and applies a fixed `0.1` variance-to-mean correction; it does not compare a propagated covariance against the correct analytic covariance, nor implement the full threshold-dependent response. **This does not test ARC's factorized cumulant recurrence or Puffi's recurring covariance-response state.** These limitations remain in the code inspected after the report declared completion.

Local evidence: [progress](<D:/ALL CODES/AICROWD COMPETITION/Current thing.md>), [Phase 5 report](<D:/ALL CODES/AICROWD COMPETITION/phase5report.md>), [P4N-12 implementation](<D:/ALL CODES/AICROWD COMPETITION/whest-starterkit/scripts/p4n_12_cumulant_propagation.py>), [earlier Phase 4 record](<D:/ALL CODES/AICROWD COMPETITION/PHASE4_RECORD.md>), and [Phase 4.5/4.6 record](<D:/ALL CODES/AICROWD COMPETITION/phase4.5record_muse.md>).

### A useful calculation from the saved oracle results

P4N-01's quoted approximately 10% oracle score gain retains the old sampling weight `alpha = 0.11`. It is not the best achievable blend of those saved predictions.

Let `C` and `V` be the analytic and oracle-control error MSEs, and `D` their mean error inner product. Then

\[
E(\alpha)=(1-\alpha)^2C+\alpha^2V+2\alpha(1-\alpha)D,
\qquad
\alpha_*={C-D\over C+V-2D}.
\]

Recovering `D` algebraically from the saved fixed-blend MSE gives, averaged over the eight existing records:

- `C = 1.378863e-6`, `V = 8.246605e-7`, `D = -2.488769e-9`;
- the post-hoc common optimum is `alpha ≈ 0.6255`;
- the resulting raw MSE is approximately `5.148685e-7`, or nominal adjusted `5.148685e-8` at the floor.

This is approximately 58% below the immutable control, but still roughly fifteen times the leading adjusted score. It uses true control centers, target-fitted fusion, and excludes any additional cost of making those centers available. It is **headroom analysis, not a candidate result**. No MLP was rerun for this calculation. [Saved diagnostic](<D:/ALL CODES/AICROWD COMPETITION/whest-starterkit/research/phase4_next/diagnostics/p4n_01_mapped_control.json>).

The important conclusion is that **solving the center problem alone does not yet buy the leader's score with this particular control and sample budget**. A competitive design probably needs both much better centers and substantially better residual integration.

## What the public Phase 1 work changes

The competition moved from width 256/depth 32 to width 1024/depth 16. Our internal “Phase 6” still concerns competition Phase 2. The ratio depth/width is eight times smaller, and neither error constants nor costs transfer unchanged. Public measurements in keenanpepper's discussion put uncorrected analytic propagation much further ahead of matched ordinary MC at 1024×16 than at 256×32. That makes analytic research more plausible here, without promising that the already calibrated baseline has comparable remaining headroom. [Primary discussion and crossover measurements](https://discourse.aicrowd.com/t/bitter-lesson-pilled-sampling-free-propagator-yielding-1-5e-5-raw-mse/18177).

Three lessons survive the dimension change:

1. **The center problem is established prior art.** Hyojun's Phase 1 method already combined a mid-network control with an analytic mean; exact late centers offered much more than deployable ones. More correlation did not reliably improve the fused estimate. Our P4N-01/06 pattern is a local recurrence of that finding. [Primary write-up](https://discourse.aicrowd.com/t/phase-1-write-up-monetizing-the-approach-to-the-moment-propagation-frontier-an-intra-network-control-variate-and-a-measured-map-of-the-wall-behind-it-submission-326024/18154).
2. **Generic input geometry is heavily explored.** The published MUB result concerns unions of orthonormal bases under an infinite-width ReLU kernel. A separate bound nearly saturates fixed positive rules at the studied Phase 1 support. Those results discourage another generic frame search, but do not prove optimality for every finite, weight-dependent, signed rule at 1024×16. Random Hadamard frames failing P4N-10 also do not constitute a test of a correctly constructed MUB family. [MUB theorem and experiments](https://discourse.aicrowd.com/t/phase-1-write-up-the-missing-basis-was-not-the-mechanism-submission-327749/18176), [fixed-positive-rule bound](https://discourse.aicrowd.com/t/phase-1-writeup-fixed-positive-sampling-is-nearly-saturated-submission-327786/18171).
3. **A local improvement can damage a calibrated recursion.** Trim_qewas documents substantial cancellation between approximation errors; keenanpepper reports that accurate local replacements can fail when composed. Their broad “closed family” language should be read against the exact substituted state, fitted system, and operating point—not adopted as a universal theorem. [Measured closure interventions](https://discourse.aicrowd.com/t/write-up-submission-326725-measuring-the-ceiling-of-a-method-class-how-we-closed-54-families-of-estimators-without-building-them/18182).

The negative-result ledgers matter as much as the papers. For example, `01-1/arc-wbe` already records failures of spherical Stein, cross-output shrinkage, several conditional readouts, and full-vector prefix controls. These are useful objections to obvious variants, rather than evidence that every possible implementation was exhausted. [Primary experiment archive](https://raw.githubusercontent.com/01-1/arc-wbe/main/history/03-rejected-and-guarded-ideas.md).

## Idea 1 — Carry joint non-Gaussian information through layers

**Highest priority; plausible structural improvement, unproven route to the required scale.**

ARC's factorization is not an independence approximation. Its third-order representation uses symmetrized products of three factor matrices, with linear transport applied to each factor; the nonlinear update creates additional terms. Repeated-neuron power cumulants and the fourth-order trace associated with odd truncation orders are part of the construction. It is an algebraic representation of a selected recurrence, not simply `sum W^3 * marginal_skew`. The paper's asymptotic guarantees also have activation and fixed-depth qualifications. [ARC paper, Sections 4 and S.4.3](https://arxiv.org/html/2605.05179v2).

That distinction matters because our independent-neuron co-skew calculation failed badly. It shows that the omitted cross-neuron terms matter; it does not show that a structured joint state cannot be affordable.

The version worth investigating is **a compressed recurrence that preserves the contractions used by the next nonlinear layer and the eventual control center**. There are several possible representations: selected tensor factors, covariance-response matrices, or a small joint state with persistent cross-layer memory. Marginal skew alone is too narrow.

Puffi supplies an actual precedent. Their SSC method reconstructs a fourth-cumulant approximation from a few live covariance-response matrices coupled by a small signed matrix. It improves substantially over their factorized third-order reference, yet still loses to sampling at Phase 1 depth. Their paper also reports a subsequent benefit from fitting the correction actually consumed by the next mean update. This supports studying the projection, not assuming their published estimator is already competitive here. [Puffi's mechanistic paper, Sections 3–4 and Appendix D](https://discourse.aicrowd.com/uploads/short-url/yQsl71y2a3zDgb4SpYQFnt9EpoE.pdf).

For a row-vector hidden mean `m_k` and downstream response `B_k`, the relevant center loss is

\[
\mathcal L_k={1\over1024}\left\|(\widehat m_k-m_k)B_k\right\|^2.
\]

Thus compression might preserve tensor contractions important to this loss rather than minimizing the tensor's global reconstruction error. In principle, a large joint tensor can have many irrelevant entries and comparatively few important contractions. The important qualification is that the nonlinear recurrence can later rotate information into relevant directions: a basis chosen only from the final weight matrix may miss it.

**What needs figuring out:** whether those useful contractions remain compressible after several genuine nonlinear updates, and whether preserving them improves the actual late center beyond the current roughly `5–7e-7` projected-error scale. A substantial gain in an unconsumed marginal cumulant is not sufficient. Likewise, full uncompressed K=3 may be a scientific reference without fitting the competition budget.

The public Phase 2 high-moment bake provides independent `N=1e8` marginal and pairwise moments, including mixed third/fourth moments. It can distinguish a bad state representation from a noisy 2,048-sample diagnostic on the same eight networks. It remains approximate and supplies diagnostic truth, not an inference-time feature. [Dataset and provenance](https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-higher-moments-2026).

## Idea 2 — Estimate the distributional error while preserving cancellation

**A more ambitious companion to Idea 1. The mathematical identity is straightforward; the cheap representation is the research problem.**

Rather than retaining more scalar moments, consider retaining the signed difference between the true activation law and its approximation.

Let `T_l` be the actual linear-plus-ReLU layer, `nu_l` its true output law, and `q_l` a tractable approximate law. The notation `T#q` means pushing the distribution through `T`. Define

\[
d_l=T_l\#q_{l-1}-q_l,\qquad
\delta_l=\nu_l-q_l.
\]

If `q_0` is the exact input law, then

\[
\delta_l=T_l\#\delta_{l-1}+d_l.
\]

This is exact bookkeeping for any chosen `q_l`. At the final layer it expresses the mean error as the sum of transported local distributional defects. Unlike a terminal scale correction, it exposes where the approximation loses information. A possible representation is a small, coupled collection of positive and negative components whose differences are propagated together.

The trap is that this can become a very expensive way of recovering ordinary sampling. If each source is sampled independently, its variance may greatly exceed the small net correction. Bulzan's related layer-source experiment found strong cancellation: the sum of source norms was nearly ten times the norm of their sum, and sampling a few layers failed badly. [Layer-grouped source verdict](https://raw.githubusercontent.com/AndreiBulzan/arc-whitebox-submission-and-documentation/main/ledger/experiments/layer_grouped_gate_source_20260808/VERDICT_LAYER_GROUPED_GATE_SOURCE_L1_R1_20260808.md).

So the interesting extension is **collective propagation of the cancelling terms**, possibly as coupled multi-layer blocks or a shared joint state. Independent source selection, independent particles for each correction, and arbitrary dropping of small local terms are poor starting points.

There is also a useful warning from the public tangent-cloud experiments. Repairing their first-layer quadrature exactly did not save later layers: replacing the nonlinear covariance law with only its first Hermite term lost essential off-diagonal information almost immediately. That points toward retaining a richer shared covariance response before allocating more particles. [Orthogonal-tangent follow-up](https://raw.githubusercontent.com/AndreiBulzan/arc-whitebox-submission-and-documentation/main/ledger/experiments/packet_connected_tangent_cloud_20260809/VERDICT_PACKET_ORTHOGONAL_TANGENT_CLOUD_C2_R1_20260809.md).

**What would make this promising:** a coupled correction with small variance because the cancellation is represented algebraically, rather than rediscovered by many samples. The open issue is whether this representation survives ReLU composition without expanding back into a large trajectory population.

## Idea 3 — Recover the signed angular integration error of this network

**The strongest public evidence for an alternative source of accuracy, with a severe unresolved cost problem.**

For a homogeneous ReLU network, radial integration can be separated from angular integration. The new question is not whether this identity exists—it has already been used—but whether the quadrature error on the sphere can be predicted for the particular weights.

Puffi's score paper found useful target-selected frame subsets, while observable-feature herding did not reliably recover them. Even downstream-response matching had already been considered. Therefore “select frames using the Jacobian” is not a fresh proposal by itself. [Puffi's score paper, Appendix B.2](https://discourse.aicrowd.com/uploads/short-url/iDdUNo66HzI7ovRqhQqpLOFFugI.pdf).

More interestingly, Bulzan's signed Poisson-band experiment obtained a correction **without accessing the true expectation**, reducing raw error by about 38.8% on its sixteen-network confirmation panel. It required 832 additional complete smoothed populations, so it was an overpowered observability result, not a viable estimator. [Primary result](https://raw.githubusercontent.com/AndreiBulzan/arc-whitebox-submission-and-documentation/main/ledger/experiments/design_defect_tangent_arc_20260806/VERDICT_SIGNED_POISSON_BAND_ORACLE_R2_20260806.md).

The mathematical attraction is that spherical smoothing separates angular frequencies. If the even network output is decomposed into harmonic bands `f_j`, a Poisson smoothing operator satisfies `P_r f_j = r^j f_j`. Consequently, for quadrature `Q`,

\[
Q(P_rf)-Q(f)=\sum_{j\ge1}(r^j-1)Q(f_j).
\]

The differences expose the **signed error contribution**, whereas a discrepancy or energy statistic can reveal its magnitude but miss the correction direction.

Unfortunately, the cheap bridges were tested too. Pooled gate statistics, basis-resolved checkpoint means, and sparse smoothed probes all failed to reproduce the successful correction. The sparse inversion amplified noise by orders of magnitude. [Compact-reconstruction follow-up](https://raw.githubusercontent.com/AndreiBulzan/arc-whitebox-submission-and-documentation/main/ledger/experiments/signed_band_gatestate_student_20260806/VERDICT_COMPACT_SIGNED_BAND_RECONSTRUCTION_R1_R3_20260806.md).

The remaining idea is narrower: **an analytic or strongly coupled approximation to the signed band response that retains within-frame directional information through the network**. Possible mathematical objects are query-specific higher-order contractions or multi-scale coupled trajectories with exact low-order cancellations. The distinction from P4N-10 is that weights or corrections would respond to a predicted deep integration defect, not just match exact first-layer moments.

**What needs figuring out:** whether the successful expensive response can be represented with far fewer full network evaluations. Learning a larger decoder from the same pooled summaries has poor supporting evidence. A 39% gain alone also does not close our gap; this would need to accompany a much stronger base estimator or reveal larger headroom at 1024×16.

## Idea 4 — Model amplitude and direction jointly inside the network

**A less explored representation change; attractive mathematically, weakly supported as an estimator.**

Martinot's Phase 1 notes separate layerwise log-amplitude and angular dynamics and describe angular concentration under repeated normalized network maps. This is useful inspiration, not evidence that a one-direction approximation integrates the finite 16-layer network accurately. [Original notes, Section 4](https://discourse.aicrowd.com/uploads/short-url/lvOeElcEqNma5IXU1iaXJVhTdzX.pdf).

Write an internal activation as `h = R u`, with `u` a unit direction. Then

\[
\mathbb E[h]=\mathbb E[R]\,\mathbb E[u]+\operatorname{Cov}(R,u).
\]

After nonlinear layers, amplitude and direction are generally dependent. An input radial correction does not capture that dependence. A possible state would retain amplitude, a small set of centered angular coordinates, their cross-moments, and the joint switching behavior of the neurons most affected by them.

This might give a more appropriate latent variable for conditional integration than a fixed input projection. P4N-08 conditions on input directions and repeats a Gaussian closure over the complement; a joint internal amplitude/angular state asks a different question. A public rank-eight input-response mixture also failed despite capturing nearly all mean-gated Jacobian energy, illustrating why an input Jacobian is not enough. [Primary rank-eight mixture diagnostic](https://raw.githubusercontent.com/AndreiBulzan/arc-whitebox-submission-and-documentation/main/ledger/experiments/response_rank8_covariance_20260729/VERDICT_RESPONSE_RANK8_CONDITIONAL_MIXTURE_R2_20260729.md).

The danger is confusing a dominant mean with low-dimensional fluctuations. A cloud of positive activations can look almost rank one before centering and still contain many important error directions. Our old final-weight SVD observations likewise do not prove that every state-dependent representation is full rank.

**What needs figuring out:** whether centered angular fluctuations together with amplitude explain the small mean error, not merely the large activation vector. The representation would need to retain near-threshold dependence and respect the positive-orthant geometry. A Gaussian tangent approximation that drops those features is likely to repeat the previous failures.

## Idea 5 — Integrate boundary contributions analytically

**High risk; an algebraic research question, not a recommendation for another finite-difference sweep.**

Gaussian integration by parts and degree-one homogeneity give, componentwise and in the weak/distributional sense,

\[
\mathbb E[f(X)]
=\mathbb E[X\cdot\nabla f(X)]
=\int \phi_d(x)\,d(\Delta f)(x).
\]

For a piecewise-linear network, the distributional Laplacian is supported on switching boundaries. This replaces an expectation of values by a Gaussian-weighted integral over changes in slope. It is essential to use the distributional derivative: ordinary automatic differentiation returns zero second derivatives almost everywhere and misses the boundary measure.

This route is also prior art. Bulzan explicitly tested Stein–Euler boundary chords and found that finite differences amplified rare boundary crossings; large chord scales reverted toward ordinary antithetic sampling. Their conclusion leaves analytic boundary integration open. [Boundary-chord verdict](https://raw.githubusercontent.com/AndreiBulzan/arc-whitebox-submission-and-documentation/main/ledger/experiments/stein_euler_boundary_chord_20260810/VERDICT_STEIN_EULER_BOUNDARY_CHORD_B1_R1_20260810.md). Deep spherical Stein controls had also failed to predict complete-frame quadrature errors. [Deep Stein verdict](https://raw.githubusercontent.com/AndreiBulzan/arc-whitebox-submission-and-documentation/main/ledger/experiments/deep_stein_control_foray_20260729/VERDICT_DEEP_STEIN_CONTROL_R1_20260729.md).

The distinct possibility is to integrate a switching surface conditionally, including the neighboring gate constraints, rather than estimate a delta function by a thin finite-difference band. A selected small gate cluster might admit a low-dimensional truncated-Gaussian calculation, while the remaining response is represented approximately.

**What needs figuring out:** whether those gate clusters have enough collective influence and sufficiently simple conditional laws. There are many boundaries, and deep boundaries are pieces of hyperplanes with additional gate constraints; treating them as unconstrained Gaussian hyperplanes would be wrong. Their abundance undermines the simple “only a few rare kinks matter” story. This remains a lower-priority possibility unless a tractable analytic boundary object can first be identified.

## Deferred supporting idea — Use the geometry of estimator errors

**Deferred until a new mechanism produces a substantial improvement. Not a standalone Phase 6 research priority.**

The recomputed oracle blend shows why fusion should change when a component changes. Beyond that, different output directions might have different analytic bias and sampling variance. An estimator of the form

\[
\widehat\mu=c+A(m-c)
\]

allows a matrix `A` instead of a scalar sampling weight. If analytic and sampling errors were independent with known second-moment matrices `B` and `V`, the ideal linear rule would be `A = B(B+V)^{-1}`. In reality their cross-error terms matter too, especially when they share samples.

The useful version would constrain this rule to a few physically motivated response subspaces. An unconstrained 1024×1024 fit from eight networks is not credible. Previous scalar/per-neuron blends and public cross-output shrinkage failures make generic empirical-Bayes tuning a weak proposal; the changed ingredient would have to be a reliable structural model of the **error covariance**, not just the activation covariance.

P4N-13 does not establish that existing branches have only 3.11% combination headroom, and 10% is not a proven upper bound either. Blending can theoretically yield large gains when errors cancel strongly. For two equal-MSE estimates with normalized error inner product `rho`, their equal blend has MSE `E(1+rho)/2`; a thirtyfold gain would require `rho` about `-0.933`. There is no evidence here for cancellation of that strength among the existing branches. The unresolved optimizer issue is therefore an evidence caveat, not a reason to spend Phase 6 finding a few more percentage points. Fusion becomes relevant again when a genuinely different component changes the error structure.

**What needs figuring out:** whether analytic error and residual sampling error actually occupy sufficiently different directions within each network. A post-hoc oracle fit can price the opportunity, but it cannot identify deployable coefficients. This is most useful after Ideas 1–3 change the estimator, not as the main Phase 6 bet.

## The economic obstacle that every high-upside idea faces

Suppose a cheap surrogate `g` is estimated with many samples and the coupled residual `f-g` with fewer. For independent mean/residual streams, optimal allocation has variance–cost product

\[
\left(\sqrt{V_g c_g}+\sqrt{V_{f-g}c_{f-g}}\right)^2.
\]

Relative to direct MC, if `V_g ≈ V_f`, `q = c_g/c_f`, `r = V_{f-g}/V_f`, and the paired residual costs approximately `c_f+c_g`, the ratio is

\[
\left(\sqrt q+\sqrt{r(1+q)}\right)^2.
\]

In this model, tenfold improvement requires `q < 0.1` even with a perfect residual; thirty-fivefold requires `q < 0.029`. P4N-07's 56–94% costs cannot satisfy that requirement. This is a model-specific calculation: exact integration of `g`, cheaper shared residual evaluation, or different variance structure changes it.

It nevertheless explains why “distill a slightly narrower network” is unlikely to be enough. The interesting surrogate would be extraordinarily cheap, analytically integrable, or avoid paying for a second long propagation. A small nonlinear core with an analytically handled background is one possible construction, but neither low rank of the mean nor a cheap linear student demonstrates that such a core exists.

## What I would want to understand before choosing the next major direction

These are research questions, not an execution sequence:

- **Which error channel remains after correct high-accuracy interventions?** Mean, variance, mixed cumulants, and their interactions should be distinguished. P4N-03 has no located result despite the report's completion statement. P4N-12's small-sample terminal calculation does not answer it. This uncertainty determines whether a joint-state implementation is addressing the right problem.
- **Does the proposed mechanism have enough capacity to matter?** A mechanism-specific idealization should suggest a plausible order-of-magnitude gain or a credible combination of structural gains. A route whose best imaginable outcome is another 5–10% improvement is outside the current research priority, even if it is easy to implement. Precise retuning of the existing blend can wait.
- **How much of the late-center error is recoverable from a carried joint state?** The relevant target is projected center error and final fusion, after allowing the surrounding calibration to adapt.
- **Does a correction retain its sign through actual nonlinear propagation?** This separates a useful joint representation from a local fit that merely predicts magnitudes.
- **Can cancellation be preserved without a large ensemble?** This is the common difficulty behind signed defects, angular-band responses, and boundary corrections.
- **Does the method improve the error–compute tradeoff enough?** A good unmetered oracle and a fast local run do not establish an affordable inference algorithm.

Within the eight-MLP verification limit, independent sampler realizations can clarify Monte Carlo uncertainty, and leave-one-network-out fits can reveal some sensitivity to calibration. Neither supplies new network diversity. After many adaptive experiments on the same eight, a small winning margin should remain a local observation. A learned representation trained on many independent networks would be a separate training-data decision; this sketch does not assume such a campaign.

## Where I would put the research attention

| Direction | Why it deserves attention | Main reason it could fail |
|---|---|---|
| Joint cumulant/response state for late centers | Directly addresses the locally measured bottleneck; has an implementable mathematical precedent. | Required state rank and repeated nonlinear corrections may cost too much. |
| Coupled distributional-defect propagation | Could retain information that scalar and marginal corrections lose. | Cancellation may require almost the full trajectory population. |
| Signed angular-error transport | Public target-free headroom exists beyond simple feature matching. | All cheap observed bridges so far were inadequate. |
| Internal amplitude/angular state | Changes the representation and conditions on an internally meaningful variable. | Apparent concentration may disappear when examining centered error. |
| Analytic boundary integration | Directly addresses the nonsmooth part of the integrand. | Conditional gate geometry may be as hard as the original expectation. |
| Structured error fusion — deferred | May help incorporate a successful new mechanism later. | No evidence that retuning the existing branches can close the scale gap; not worth a standalone campaign now. |

I would deprioritize more generic Sobol/Hadamard variants, scalar skew/kurtosis sweeps, extra Gaussian quadrature nodes, finite-difference boundary controls, and larger learners on unchanged pooled summaries. Their public and local evidence is already substantial. A certified MUB comparison is still an uncompleted factual question locally, but its expected role is a stronger sampling reference, not an assumed thirtyfold advance.

The central uncertainty is whether **the useful joint correction can be made small in representation and cost, while staying accurate through composition**. That is where a result could change the scale of this project. The literature supplies reasons to investigate it and reasons to be skeptical; it does not supply a verified recipe for the current leaders' scores.
