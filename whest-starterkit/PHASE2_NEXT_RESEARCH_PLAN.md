# Phase 2 next research goal

## Objective

Develop a private-safe Phase 2 estimator that:

- first verifies the current projective-shrinkage result on the available eight-MLP panel;
- targets adjusted score below `1.24e-7`, not merely below `2.5e-7`;
- stays within the legitimate Phase 2 computation rules;
- has zero failures and substantial residual-time safety margin;
- minimizes overfitting to those eight MLPs through paired repetition and leave-one-MLP-out tuning.

Do not overwrite the existing estimator until a candidate passes every promotion gate.

## Phase 0 — Repair the evidence base

Only eight MLPs will be used for validation. They have already influenced gamma, alpha, sample count, and architecture choices, so they are not a true untouched validation set. Do not claim that results on these eight prove private-set generalization.

Use all eight as one fixed evaluation panel. Do not split them into a tiny development set and an even less informative lockbox. Instead:

- evaluate every candidate on all eight MLPs;
- use at least five independent legal sampling randomizations per MLP;
- use paired randomizations between the baseline and candidate;
- predeclare each candidate family, hyperparameter grid, success threshold, and stopping rule before running it;
- select any fitted hyperparameter by leave-one-MLP-out cross-validation: for each held-out MLP, choose the setting using only the other seven and score it only on the held-out MLP;
- report both the leave-one-MLP-out selection score and the score of the final deployment setting refitted on all eight;
- treat repeated redesign after viewing the same eight results as additional selection pressure and record the number of candidate families tested.

The eight MLPs provide 40 paired observations when five sampling randomizations are used, but the MLP—not the randomization—is the independent generalization unit. Confidence intervals and hypothesis tests must therefore resample or permute at the MLP level, with sampling-randomization variability summarized inside each MLP.

Record:

- mean raw final-layer MSE;
- exact mean adjusted score, computed per MLP;
- paired improvement and bootstrap confidence interval;
- win/loss count;
- median, 90th percentile, and worst MLP;
- FLOP utilization per MLP;
- residual time and forced-materialization count;
- failures;
- sampler seed and estimator hash.

With only eight MLPs, require at least six wins out of eight for promotion, inspect the worst two networks explicitly, and prefer effects that are large and consistent rather than relying on a nominal p-value. An exact paired sign/permutation test and an MLP-level bootstrap interval should both be reported, but neither removes the uncertainty caused by the small panel.

First reconcile why the recorded score is `1.650e-7` while the stored alpha-sweep calculation currently gives approximately `1.539e-7`.

## Phase 1 — Audit the present breakthrough

For every MLP, separately measure:

1. ordinary gain covariance;
2. damped Hermite covariance;
3. unscaled WMC;
4. oracle-scaled covariance using ground truth, for diagnosis only;
5. MC-scaled covariance;
6. current MC-scaled blend.

Measure:

$$
s_m=\frac{m^\top c}{c^\top c},
\qquad
s_y=\frac{y^\top c}{c^\top c}.
$$

Report the distribution of $s_m-s_y$, not merely the mean scale.

Measure the covariance or cosine between:

$$
e_c=s_mc-y,\qquad e_m=m-y.
$$

Do not use the independent-error precision formula unless this cross term is negligible across the eight-MLP panel and stable under leave-one-MLP-out analysis.

Also compare plain antithetic Gaussian sampling against whitened antithetic sampling. This separates variance reduction from whitening-induced finite-sample bias.

Promotion gate: the calibrated estimator must improve the leave-one-MLP-out mean MSE by at least 5%, win on at least six of eight MLPs, and have an MLP-level paired bootstrap interval excluding zero.

## Phase 2A — Replace fixed alpha with projection-aware shrinkage

Interpret the 1,024 output coordinates as simultaneous noisy observations.

Start with the current one-dimensional template $U=[c]$, then test only small deterministic bases:

$$
U=
[c,\ \mathbf 1,\ \mu\Phi(a),\ \sigma\phi(a)]
$$

or a subset of these columns. Never start with hundreds of neuron-specific parameters.

For a basis $U$, form the regularized projection:

$$
P_U=U(U^\top U+\lambda I)^{-1}U^\top.
$$

Use the estimator:

$$
\hat y=P_Um+\alpha(I-P_U)m.
$$

Test basis dimensions 1, 2, 4, and 8. Choose the dimension by leave-one-MLP-out evaluation. Stop increasing it when the cross-validated improvement fails to exceed 2% or the worst-two-MLP result deteriorates materially.

### Online alpha without ground-truth fitting

Split the sampling budget into two independent, moment-matched blocks $m_1,m_2$, if this does not materially damage whitening quality. Let

$$
m=\frac{m_1+m_2}{2}, \qquad Q=I-P_U.
$$

Estimate orthogonal signal and noise energies by

$$
\widehat A=\frac{(Qm_1)^\top(Qm_2)}{d},
\qquad
\widehat B=\frac{\|Q(m_1-m_2)\|^2}{4d}.
$$

Then use

$$
\hat\alpha=
\operatorname{clip}
\left(
\frac{\max(0,\widehat A)}
{\max(0,\widehat A)+\widehat B},
0,1
\right).
$$

Compare this with positive-part James–Stein/SURE shrinkage and the fixed `0.13` control. The aim is a private-safe per-MLP shrinkage strength, not another public-data-tuned constant.

If two-block whitening loses too much accuracy, estimate noise from independent antithetic pair batches, but account explicitly for the dependence introduced by global whitening.

### Regime-specific calibration

If a single scale is insufficient, divide neurons into only 2 or 4 large groups based on analytic activation regime:

- standardized preactivation $a=\mu/\sigma$;
- predicted firing probability $\Phi(a)$;
- predicted marginal variance.

Estimate one scale and one shrinkage strength per group. Require at least 128 neurons per group. Do not attempt 1,024 independent blend weights.

## Phase 2B — Final-layer Hermite control variate

This is the highest-upside statistical experiment.

Let $z_{ri}$ be the final preactivation for sample $r$, and let the analytic propagation provide $\mu_i$, variance $v_i$, $\sigma_i=\sqrt{v_i}$, and $a_i=\mu_i/\sigma_i$.

Test the order-one estimator:

$$
\hat y_i^{(1)}
=
\overline{\operatorname{ReLU}(z_i)}
-
\Phi(a_i)(\bar z_i-\mu_i).
$$

Then test the order-two estimator:

$$
\hat y_i^{(2)}
=
\overline{\operatorname{ReLU}(z_i)}
-
\Phi(a_i)(\bar z_i-\mu_i)
-
\frac{\phi(a_i)}{2\sigma_i}
\left(
\overline{(z_i-\mu_i)^2}-v_i
\right).
$$

These subtract the degree-one and degree-two Gaussian Hermite components from the sampled final-layer function while adding their analytic expectations implicitly.

For an exactly Gaussian centered preactivation with exact moments:

- total ReLU variance is approximately $0.34085\sigma^2$;
- the first Hermite component accounts for $0.25\sigma^2$;
- the second accounts for $0.07958\sigma^2$;
- only about 3.3% of the original variance remains.

This is an oracle ceiling, not an expected real result. Non-Gaussianity and analytic moment error will determine how much survives. Even a modest fraction of this theoretical reduction could be decisive.

Run these ablations in order:

1. order-one control with analytic moments;
2. order-one plus order-two;
3. the same controls with globally calibrated analytic moments;
4. cross-fitted moment calibration using two sampling blocks;
5. combine the best controlled sampler with projection/SURE shrinkage.

Do not add cubic or quartic controls until the first two orders demonstrate held-out improvement. Their expectations require additional trustworthy moments, and uncontrolled high-order corrections have repeatedly failed in Phase 1.

Kill this family if an oracle using true final-preactivation first and second moments does not improve the current estimator by at least 20%. That means moment-centering fidelity is not the relevant bottleneck.

## Phase 3 — Principled analytic upgrade

Do not implement an arbitrary $\rho^4$ covariance correction and call it fourth-order cumulant propagation. These are different mathematical objects.

First distinguish:

- Mehler/Hermite refinement of a bivariate Gaussian covariance, involving powers of $\rho$;
- genuine propagation of non-Gaussian third and fourth cumulants.

If the controlled sampler remains limited by analytic moment bias, reproduce the ARC factorized $K=3$ method faithfully:

- propagate power cumulants;
- retain the extra fourth-order full trace required for odd $K$;
- use the factorized representation rather than materializing a $1024^3$ tensor;
- verify it against the authors’ implementation at small width and depth;
- measure its standalone MSE and its error correlation with the existing covariance template.

Phase 2 has $L/n=16/1024=1/64$, much more favorable than Phase 1’s $32/256=1/8$. The published theory predicts approximate $c_K(L/n)^K$ behavior, although constants and ReLU depth effects must be measured. Use the factorized $K=3$ result first as:

1. a replacement analytic template;
2. an additional column in $U$;
3. the centering mean for the final-layer control variate.

Only use it as a standalone estimator if its leave-one-MLP-out MSE independently wins on the eight-MLP panel.

If full factorized $K=3$ is too expensive, explore 2–8 covariance-response modes or an adjoint-selected low-rank third-cumulant representation. Select modes using only weights and analytic sensitivity, never target answers.

## Phase 4 — Signed adaptive calibration, only after the above

The Phase 1 census indicates fixed nonnegative cubature is essentially exhausted, while signed or target-adaptive weighting remains open.

Given sampled output rows $f_r$ and a small set of internal features $g_r$ with analytically known expectations $g_0$, test regularized calibration weights:

$$
\min_w
\left\|w-\frac1N\mathbf1\right\|^2
+
\lambda\|w\|^2
$$

subject approximately to

$$
\mathbf1^\top w=1,
\qquad
\sum_r w_rg_r\approx g_0.
$$

Use no more than 4–16 late-layer features. Prefer final-preactivation Hermite features and adjoint-selected intermediate modes. Track effective sample size and maximum absolute weight; reject unstable signed solutions.

Recognize that this is algebraically related to control variates. Do not run it unless it offers a genuinely new feature space.

## Directions to deprioritize

Do not spend another major cycle on:

- fine sweeps of gamma around `0.5`;
- fine sweeps of scalar alpha around `0.13`;
- increasing sample count slightly past the 10% floor;
- per-neuron weights fitted on eight MLPs;
- exact radial conditioning on top of whitening;
- another unrotated Hadamard construction;
- long-range layer-8 linear Jacobian correction;
- dense third- or fourth-order tensors;
- offline nonlinear correctors trained only on public Mini;
- compute-accounting tricks or computation outside flopscope.

The public Phase 1 work already heavily tested fixed cubature, generic low-order controls, closure tweaks, radial conditioning, and long-range control propagation. Reopen one only when Phase 2 supplies a specific new mechanism.

## Promotion and submission rules

A candidate may be promoted only if:

- leave-one-MLP-out mean improves by at least 5% across the eight MLPs;
- an MLP-level paired 95% bootstrap interval excludes zero;
- at least six of eight MLPs improve;
- the result remains favorable across the paired sampling randomizations;
- no numerical or budget failures occur;
- neither of the worst two MLPs is catastrophically worse;
- the estimator remains finite, nonnegative, float32, and correctly shaped.

A promoted candidate may be submitted only if:

- its all-eight refit preserves the leave-one-MLP-out improvement direction;
- utilization is at most 9.5%, or its raw-MSE gain exceeds the multiplier penalty from crossing 10%;
- residual time has at least a 2× local safety margin;
- no `bool()`, `float()`, or `int()` forces materialization of a pending flopscope graph;
- the packaged archive reproduces the local result.

Because there is no untouched MLP lockbox, consider the public competition submission itself the only genuinely new-network check. Do not make repeated submissions for tiny apparent gains; submit only changes that clear the conservative gate above.

At utilization $u>0.1$, evaluate the exact trade:

$$
\frac{S_{\text{new}}}{S_{\text{old}}}
=
\frac{\mathrm{MSE}_{\text{new}}}{\mathrm{MSE}_{\text{old}}}
\frac{u}{0.1}.
$$

For example, 10.5% utilization requires more than a 4.76% raw-MSE reduction merely to break even.

## Recommended execution order

1. Full eight-MLP paired audit, leave-one-MLP-out setup, and score reconciliation.
2. One-dimensional adaptive James–Stein/SURE shrinkage.
3. Two- and four-dimensional template bases or regime-grouped shrinkage.
4. Final-layer order-one Hermite control.
5. Final-layer order-one plus order-two control.
6. Combine the best controlled sampler with the best shrinkage estimator.
7. Submit the first fully validated candidate.
8. Begin faithful factorized $K=3$ only if analytic moment bias is still the measured bottleneck.
9. Try signed adaptive calibration only after a useful late-layer feature basis exists.

Every experiment must state beforehand: hypothesis, mathematical mechanism, expected failure mode, success threshold, and stopping rule.

## Sources

- [Official Phase 2 announcement](https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197)
- [Published Phase 2 covariance/WMC hybrid](https://discourse.aicrowd.com/t/phase-2-write-up-complementary-errors-beat-either-branch-a-fixed-0-75-0-25-covariance-whitened-antithetic-mc-blend-submission-328274/18200)
- [Residual-cap investigation and documented 1.24e-7 entry](https://discourse.aicrowd.com/t/shipped-covariance-propagation-example-trips-the-phase-2-residual-cap-locally-0-16-0-but-grades-fine-what-happens-when-the-evaluator-upgrades/18202)
- [Phase 1 technique census](https://discourse.aicrowd.com/t/a-technique-census-of-phase-1-what-the-mathematics-is-doing-what-walls-it-hit-and-whats-still-open/18157)
- [Phase 1 intra-network control variate](https://discourse.aicrowd.com/t/phase-1-write-up-monetizing-the-approach-to-the-moment-propagation-frontier-an-intra-network-control-variate-and-a-measured-map-of-the-wall-behind-it-submission-326024/18154)
- [ARC cumulant-propagation paper](https://arxiv.org/html/2605.05179v2)
- [Stein’s multivariate risk-estimation paper](https://www.stat.yale.edu/~hz68/619/Stein-1981.pdf)
