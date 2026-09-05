# Phase 3 Research Plan: Break the `1e-7` Barrier

## Mission

Develop a Phase 2 estimator that achieves an adjusted score below `1.00e-7` while remaining private-safe, legal, reproducible, and robust across the fixed eight-MLP validation panel.

This is the primary Stage 3 goal. It is deliberately more demanding than a small improvement:

- current local eight-MLP score: `1.231901e-7`;
- current local raw final-layer MSE: `1.231901e-6`;
- current leaderboard score: approximately `1.18e-7`;
- current leaderboard raw final-layer MSE: approximately `1.17e-6`;
- current leaderboard rank when Stage 3 was written: 57;
- primary target: adjusted score `< 1.00e-7`;
- equivalent raw-MSE target at the `0.1` multiplier floor: `< 1.00e-6`;
- stretch target: `< 7.50e-8`;
- exploration-breakthrough target: `< 5.00e-8`.

The primary target requires approximately a 19% reduction from the current local benchmark and approximately a 15% reduction from the current leaderboard result. Therefore, tiny parameter sweeps do not count as completing Stage 3.

## Current champion

The immutable control is the current `estimator.py`:

- threshold-independent damped quadratic Hermite covariance;
- `gamma = 0.20`;
- fixed prior scale `s_0 = 0.998319`;
- WMC blend weight `alpha = 0.110`;
- `N = 4,200` whitened antithetic samples;
- local compute utilization `9.81%`;
- local maximum residual time `0.1873s`;
- zero failures;
- local eight-MLP result `8W-0L` against the old covariance/WMC baseline.

Do not overwrite `estimator.py` until a candidate passes all promotion gates. Copy it into a new candidate file for every method family.

## Important evidence corrections

The agent must not repeat the following overclaims from the Run 2 summary:

1. `1.2319e-7` was a new personal benchmark, not a new state of the art.
2. Eight MLPs do not prove that the scale factor is a universal invariant. The eight-MLP result plus leaderboard transfer are evidence that it is stable enough to exploit.
3. Error cosine `+0.0762` did not rigorously prove independence because the audited covariance error used an MC-derived scale and therefore shared noise with the MC estimator.
4. The rejected standalone `rho^4` patch did not reject the complete threshold-aware Gaussian Hermite expansion.

All future reports must label claims as one of:

- measured;
- mathematically derived;
- supported interpretation;
- untested hypothesis.

## Non-negotiable validation protocol

Only the fixed eight MLPs will be used for local validation.

### Evaluation stages

1. **Smoke screen:** two MLPs, one sampling seed.
2. **Panel screen:** all eight MLPs, official protocol seed.
3. **Robustness screen:** all eight MLPs with three deterministic sampler salts, only after a candidate improves the panel mean by at least 5%.
4. **Promotion evaluation:** compare the candidate against the current champion, never only against the old `3.1749e-7` baseline.

### Statistical rules

- Use common random numbers between champion and candidate wherever possible.
- Treat the MLP as the independent generalization unit. Sampler salts are repeated measurements, not additional MLPs.
- Any parameter fitted from targets must use leave-one-MLP-out selection.
- Report both the LOO-selected result and the all-eight refitted deployment result.
- Require at least 6/8 wins for continued research and preferably 7/8 for submission.
- Reject a candidate if either of the two worst MLPs regresses by more than 10% without a compelling mean improvement.
- Report mean, median, worst MLP, win count, and per-MLP paired differences.
- Record code hash, candidate path, sampler salt, FLOPs, utilization, residual time, and failures.
- Do not describe a result as private-safe solely because it wins on the same eight repeatedly.

### Operational rules

- Zero failures are mandatory.
- Keep utilization below 10% unless the raw-MSE reduction beats the exact multiplier penalty.
- Prefer a local residual time below `0.30s` and never accept a path close to the `0.40s` hard cap.
- Preserve finite, nonnegative, correctly shaped `float32` output.
- Run the official contract validation before promotion.
- Use the official seed-resolution path. Do not instantiate an MLP from the raw 64-bit dataset seed.

## Research portfolio

Stage 3 must operate three lanes in parallel conceptually, even if experiments run sequentially:

- **Lane A — exploit the working analytic/hybrid family:** 50% of research effort.
- **Lane B — repair the sampled residual using mid-network information:** 30%.
- **Lane C — independent exploration:** 20%.

After every two Lane A experiments, run at least one experiment from Lane B or C. Do not spend more than three consecutive experiments tuning one mechanism unless results improve monotonically and by a meaningful amount.

## Lane A1 — Threshold-aware Gaussian Hermite covariance

### Why this is first

The current implementation computes the standardized preactivation mean

$$
a_i=\frac{\mu_i}{\sigma_i},
$$

but its quadratic covariance correction ignores `a_i` and uses the centered coefficient for every pair of neurons.

For jointly Gaussian standardized preactivations with correlation `rho_ij`, the first four off-diagonal ReLU covariance terms are:

$$
C_{ij}^{(1)}
=
\sigma_i\sigma_j
\Phi(a_i)\Phi(a_j)\rho_{ij},
$$

$$
C_{ij}^{(2)}
=
\frac{\sigma_i\sigma_j}{2}
\phi(a_i)\phi(a_j)\rho_{ij}^{2},
$$

$$
C_{ij}^{(3)}
=
\frac{\sigma_i\sigma_j}{6}
a_i a_j\phi(a_i)\phi(a_j)\rho_{ij}^{3},
$$

$$
C_{ij}^{(4)}
=
\frac{\sigma_i\sigma_j}{24}
(a_i^2-1)(a_j^2-1)
\phi(a_i)\phi(a_j)\rho_{ij}^{4}.
$$

Continue to replace the diagonal with the exact marginal ReLU variance.

These are Gaussian Hermite covariance terms. They are not third- or fourth-cumulant propagation.

### Why the previous quartic rejection is not decisive

The previous experiment:

- used a threshold-independent coefficient;
- omitted the generally nonzero cubic term;
- did not include the factors involving `a_i` and `a_j`;
- used `1/(96*pi)` for the centered quartic coefficient, while the consistent expansion gives `1/(48*pi)` when `a_i=a_j=0`;
- tested an isolated quartic patch rather than a coherent expansion.

Therefore, do not cite that experiment as evidence against this family.

### Required ablations

Run in this order:

1. Replace the constant quadratic shape with the threshold-aware quadratic term while retaining damping `eta_2 = 0.20`.
2. Coarsely test `eta_2` in `{0.20, 0.50, 1.00}`. Do not begin with a fine sweep.
3. Add the cubic term using `eta_3` in `{0.00, 0.50, 1.00}` while keeping the best cross-fitted quadratic setting.
4. Add the consistent quartic term with `eta_4` tied to `eta_3` initially.
5. Only if step 4 wins, test one shared damping parameter for all terms versus separate `eta_2` and `eta_3=eta_4`.

### Diagnostics

For each layer and MLP, record:

- distribution of `a_i`;
- mean absolute size of every Hermite contribution;
- ratio of each contribution to the linear covariance term;
- minimum and maximum correlation before clipping;
- final scale projection against truth;
- covariance-only MSE;
- fused champion MSE.

### Success and stopping rules

Promote within this family only if the fused result improves by at least 5% and wins on at least 6/8 MLPs.

Kill the family if:

- threshold-aware quadratic loses on at least 5/8 MLPs;
- the oracle best combination of terms improves less than 3%; or
- improved covariance-only MSE does not improve the final fused estimator.

## Lane A2 — Correct risk-aware blending

### Hypothesis

The current `alpha = 0.110` assumes that calibrated covariance error and WMC error are effectively independent. The correct risk includes their cross term.

For deployed calibrated covariance `c_0 = s_0 c`, define:

$$
A=\mathbb E\|c_0-y\|^2,
\qquad
B=\mathbb E\|m-y\|^2,
$$

$$
C=\mathbb E[(c_0-y)^\top(m-y)].
$$

For

$$
\hat y=(1-\alpha)c_0+\alpha m,
$$

the correct optimum is

$$
\alpha^*=\frac{A-C}{A+B-2C}.
$$

The usual harmonic precision formula is only the special case `C = 0`.

### Required experiments

1. Recompute `A`, `B`, and `C` using the fixed deployed `s_0`, not an MC-derived `s_m`.
2. Measure the oracle global `alpha`, per-MLP oracle `alpha`, and LOO global `alpha`.
3. If oracle headroom over `0.110` is below 2%, stop.
4. If headroom is meaningful, split the 4,200 samples into two independently whitened 2,100-point antithetic estimators `m_1` and `m_2`.
5. Estimate analytic error and full-average MC noise by

$$
\widehat A
=
\frac{(m_1-c_0)^\top(m_2-c_0)}{d},
$$

$$
\widehat B
=
\frac{\|m_1-m_2\|^2}{4d}.
$$

6. Shrink the resulting per-MLP blend strength strongly toward the global LOO prior. Do not use the raw ratio without shrinkage.

### Caution

Two separate whitening decompositions add compute. If this crosses the 10% floor, first test a cheaper decomposition or reduce sample count only enough to restore the floor. Compare the exact score trade, not raw MSE alone.

### Promotion gate

Require at least 3% adjusted-score improvement, 6/8 wins, and no material worst-MLP regression.

## Lane A3 — Low-rank analytic residual correction

### Hypothesis

The successful scale prior corrected analytic error along one deterministic direction, `c`. More analytic error may live in a tiny subspace spanned by principled correction patterns.

Construct a basis such as

$$
U=[c,\Delta_2,\Delta_3,\Delta_4],
$$

where `Delta_k` is the change in the final analytic prediction induced by adding Hermite order `k`. Optionally include one analytically chosen layer-sensitivity or covariance-response mode.

First measure the oracle captured residual fraction:

$$
R_U^2
=
\frac{\|P_U(y-c_0)\|^2}{\|y-c_0\|^2}.
$$

If rank four captures less than 15% of analytic residual energy on average, stop.

Estimate correction coefficients from the sampled residual:

$$
\hat\theta
=
(U^\top U+\lambda I)^{-1}U^\top(m-c_0).
$$

Use

$$
\hat y
=
c_0+U\hat\theta
+\alpha_\perp(I-P_U)(m-c_0).
$$

### Required ablations

- basis ranks 1, 2, 4, and at most 8;
- LOO ridge strength;
- fixed global coefficients versus per-MLP sampled coefficients;
- with and without the complementary residual shrinkage term.

Do not fit 1,024 neuron-specific coefficients.

### Promotion gate

Require at least 8% mean improvement, 6/8 wins, and stable coefficients across LOO folds.

## Lane A4 — Jointly reuse the final-layer Hermite controls

The order-one and order-two final-layer controls reduced standalone sampling MSE substantially, but the previous simple covariance blend did not monetize the improvement.

Do not conclude that the controlled estimators are useless. Test a joint, strongly regularized combination of:

- calibrated analytic covariance;
- raw WMC;
- order-one final-layer control;
- order-two final-layer control.

Use no more than two free combination weights. Estimate their complete error-covariance matrix rather than treating the branches as independent. Fit all weights with LOO evaluation.

Kill the family if the oracle joint combination improves by less than 5% or the cross-fitted result wins on fewer than 6/8 MLPs.

## Lane B1 — Proper mid-network ridge control variate

### Why the family remains open

The rejected layer-8 candidate propagated a mean discrepancy through a deterministic linear Jacobian across eight ReLU layers. That rejected the transport approximation, not the full mid-network-control family.

For a candidate control layer `k`, retain sampled activations `H_k` and final activations `H_L`. Estimate a ridge response from centered sample fluctuations:

$$
B_k
=
\operatorname{Ridge}(H_k-\bar H_k,\ H_L-\bar H_L).
$$

Then correct the final sample mean using an analytic control mean `c_k`:

$$
\hat m_L^{CV}
=
\bar H_L-(\bar H_k-c_k)B_k.
$$

The response must be learned from within-network samples, not replaced with a chain of mean Jacobians.

### Required design

1. Test layers `k in {4, 6, 8, 10}`.
2. Begin with response ranks 16 and 32 rather than a full unconstrained map.
3. Select response modes from sampled covariance or deterministic analytic sensitivity.
4. Use ridge regularization.
5. If same-sample fitting creates bias, use two-block cross-fitting: fit on block one and correct block two, then swap and average.
6. Test the controlled sampler alone.
7. Test it inside the champion analytic blend. The fused score is the decision metric.
8. Repeat with the improved Lane A1 analytic mean as the control center.

### Promotion and stopping

Promote only with at least 10% fused improvement and 6/8 wins.

Kill a layer/rank combination if:

- raw sampler variance falls but fused score does not;
- analytic control-mean bias is larger than the variance removed;
- regression coefficients are unstable across sample splits;
- compute crosses the score floor without sufficient raw-MSE gain.

## Lane B2 — Sampled-moment reset

### Hypothesis

Analytic bias accumulates through the recurrence. Injecting shrunken sampled moments at one middle layer may restart the analytic chain from a more accurate state without using an unreliable long-range linear Jacobian.

For `k in {6, 8, 10}`:

1. Compute sampled mean and covariance at layer `k`.
2. Shrink them toward the analytic mean and covariance.
3. Restart threshold-aware covariance propagation from the corrected state.
4. Compare the restarted analytic prediction, raw sample mean, and fused output.

### Mandatory oracle gate

Before implementing the deployable version, substitute high-accuracy true moments at layer `k`. If this oracle reset does not improve the champion by at least 20%, stop the family.

### Promotion gate

Require at least 10% fused improvement and no more than one MLP loss.

## Lane C1 — Alternative sampling carriers

This lane is intentionally independent of covariance refinement.

Test, in order:

1. scrambled Sobol or a randomized rank-one lattice transformed to Gaussian inputs;
2. spherical normalization with the Gaussian radial expectation integrated exactly;
3. randomly rotated orthogonal frames rather than another axis-aligned Hadamard cloud;
4. one-dimensional stratification along a white-box-selected input direction.

Use the same sample/FLOP budget as WMC and common randomizations where meaningful.

### Decision rule

An alternative carrier must achieve:

- at least 30% lower raw sampler MSE;
- at least 5% lower fused champion MSE;
- at least 6/8 fused wins.

If it improves the sampler but not the fused estimator, record it and stop. Do not assume variance reduction survives fusion.

## Lane C2 — Active-subspace stratification

Use the weights and analytic gates to construct one to four important input directions. Possible directions include:

- adjoint propagation of the final mean direction;
- leading directions of a linearized input-to-final map;
- directions selected from a small deterministic probe set.

Before building a production estimator, use stored samples to regress final activations on these coordinates. Measure the fraction of final activation variance explained.

Kill the family if four directions explain less than 15% of the output variance. If they explain substantially more, stratify Gaussian quantiles or apply low-order Gauss-Hermite integration along those directions while sampling the orthogonal complement.

## Lane C3 — Compressed higher-cumulant propagation

Do not begin with full factorized `K=3` propagation.

Its leading published cost is approximately

$$
30n^3L^2+39n^3L.
$$

At `n=1024` and `L=16`, this is several times the complete Phase 2 FLOP budget before combining it with sampling.

Explore only after an oracle test, using one of:

- third-cumulant propagation over the final 2–4 layers;
- 2–8 covariance-response modes;
- an adjoint-selected low-rank third-cumulant representation;
- a higher-cumulant result used as one residual mode or control center rather than as the full estimator.

Stop unless an offline oracle demonstrates at least 20% final-MSE headroom and the projected scored implementation has a plausible budget path.

## Lane C4 — Small shared mechanistic residual law

Inspect whether the deterministic analytic residual `y-c_0` has a stable conditional mean as a function of analytic neuron features:

- `a = mu/sigma`;
- `sigma`;
- predicted firing probability;
- final incoming-weight norm;
- threshold-aware Hermite contribution sizes;
- one or two covariance row statistics.

Use bin plots first. Only if the same pattern appears in most LOO folds, fit a shared low-degree polynomial or tiny ridge model with at most four coefficients.

This is not permission to fit a flexible 8-MLP residual predictor. Reject any model whose gain disappears when an entire MLP is held out.

## Explicitly deprioritized work

Do not spend a major cycle on:

- fine sweeps around `gamma = 0.20` before testing the corrected Hermite shape;
- fine sweeps around `alpha = 0.110` without measuring the cross-error term;
- fine sweeps around `s_0 = 0.998319`;
- increasing WMC sample count slightly past the score floor;
- neuron-wise scales or blend weights;
- another unrotated Hadamard construction;
- another long-range mean-Jacobian control;
- the same standalone `rho^4` patch;
- dense third- or fourth-order tensors;
- full factorized `K=3` without a budget calculation;
- opaque nonlinear correction trained only on eight MLPs;
- improvements measured only against the obsolete `3.1749e-7` baseline;
- compute-accounting tricks or computation outside Flopscope.

## Promotion gates

A candidate may replace the current champion only if all of these hold:

- adjusted eight-MLP mean improves by at least 5%;
- at least 6/8 MLPs improve, preferably 7/8;
- LOO and all-eight refit agree on the improvement direction;
- the improvement survives three deterministic sampler salts when the method is stochastic;
- no worst-MLP regression exceeds 10%;
- zero failures;
- correct shape, finite values, nonnegative output, and `float32`;
- exact score trade remains favorable after the compute multiplier;
- residual-time safety remains adequate;
- official validation and package reproduction pass.

A candidate may be submitted as the Stage 3 target only if:

- local adjusted score is below `1.00e-7`; or
- it improves the current local champion by at least 10% with 7/8 wins and represents a genuinely different method worth testing on the grader.

Do not use repeated leaderboard submissions to tune scalar constants.

## Mandatory execution order

1. Freeze and hash the current champion.
2. Correct the Run 2 evidence audit using fixed-scale covariance errors.
3. Run Lane A1 threshold-aware quadratic Hermite covariance.
4. Add the consistent cubic and quartic Gaussian terms.
5. Run Lane A2 covariance-aware blend diagnosis.
6. Run Lane B1 proper mid-network ridge control.
7. Run Lane A3 low-rank analytic residual correction.
8. Run one independent sampling experiment from Lane C1 or C2.
9. Run Lane A4 joint final-layer-control blending.
10. Run the Lane B2 moment-reset oracle.
11. Attempt compressed higher cumulants only if earlier diagnostics identify analytic non-Gaussian bias as the binding error.
12. Promote and package only after every gate passes.

## Agent decision tree

After every experiment, answer these questions:

1. Did raw final-layer MSE improve?
2. Did adjusted score improve after exact compute pricing?
3. Did at least 6/8 MLPs improve?
4. Did the two worst MLPs remain safe?
5. Did the improvement survive LOO selection?
6. For a stochastic method, did it survive alternative sampler salts?
7. Did the experiment improve the complete fused estimator, not only one branch?
8. Does the evidence identify the next mathematical bottleneck?

If questions 1–4 are not all yes, reject the candidate. If question 5 or 6 is no, classify it as overfit or seed-fragile. If question 7 is no, record the branch result but do not continue optimizing it. If question 8 is no after three family variants, move to a different lane.

## Required experiment report format

Every completed experiment must append a record containing:

```text
Experiment ID:
Lane and method family:
Hypothesis:
Mathematical mechanism:
Candidate file and code hash:
Parameters fixed before run:
Expected failure mode:
Champion mean adjusted score:
Candidate mean adjusted score:
Relative improvement:
Raw final-layer MSE:
Per-MLP scores:
Wins/losses:
Median and worst result:
LOO result:
Sampler-salt result, if applicable:
FLOP utilization:
Maximum residual time:
Failures:
Interpretation status: measured / derived / supported / speculative
Decision: promote / continue family / reject / archive
Next experiment justified by this result:
```

## Definition of Stage 3 completion

Stage 3 is complete only when one of the following happens:

### Success

A legal, reproducible candidate achieves:

- local adjusted score `< 1.00e-7` on the eight-MLP panel;
- at least 6/8 wins against the current champion;
- robustness across three sampler salts where applicable;
- zero failures and safe compute/time margins;
- successful packaging and grader evaluation.

### Scientifically useful stop

All high-priority lanes have been tested with their stated oracle and kill gates, none reaches the target, and the final report identifies which of these is now dominant:

- Gaussian covariance-map bias;
- non-Gaussian analytic bias;
- Monte Carlo variance;
- analytic/MC error correlation;
- mid-network control-centering bias;
- inability to observe analytic residual directions without target leakage.

Do not declare Stage 3 complete merely because a new personal best was obtained above `1.00e-7`.

## References

- [Live Phase 2 leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-2)
- [Official Phase 2 announcement](https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197)
- [Published Phase 2 covariance/WMC hybrid](https://discourse.aicrowd.com/t/phase-2-write-up-complementary-errors-beat-either-branch-a-fixed-0-75-0-25-covariance-whitened-antithetic-mc-blend-submission-328274/18200)
- [Phase 1 technique census](https://discourse.aicrowd.com/t/a-technique-census-of-phase-1-what-the-mathematics-is-doing-what-walls-it-hit-and-whats-still-open/18157/11)
- [Mid-network control-variate write-up](https://discourse.aicrowd.com/t/phase-1-write-up-monetizing-the-approach-to-the-moment-propagation-frontier-an-intra-network-control-variate-and-a-measured-map-of-the-wall-behind-it-submission-326024/18154)
- [Moment-matched sampling write-up](https://discourse.aicrowd.com/t/phase-1-write-up-variance-reduction-beats-closure-refinement-a-moment-matched-sampling-estimator-submission-324497/18201)
- [ARC cumulant-propagation paper](https://arxiv.org/html/2605.05179v2)
