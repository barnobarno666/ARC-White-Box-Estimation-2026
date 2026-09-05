# Phase 4 Research Plan: Escape the Closure Basin

## Mission

Develop a genuinely different Phase 2 estimator that reaches an adjusted score
below `8.00e-8` on the fixed eight-MLP validation panel while remaining legal,
private-safe, reproducible, and failure-free.

The stretch target is `< 4.00e-8`. The target is deliberately large enough that
micro-tuning the current Hermite/covariance estimator cannot complete Phase 4.

Current reference point:

- current promoted estimator: `estimator.py`;
- current local eight-MLP adjusted score: `1.223643e-7`;
- current local raw final-layer MSE: `1.223643e-6`;
- current compute utilization: approximately `9.81%`;
- current maximum measured residual time: approximately `0.20s`;
- current AIcrowd submission: `329749`;
- current public leaderboard score when this plan was written: approximately
  `1.1679e-7`;
- current public rank reported by the user: 55;
- Phase 4 primary target: `< 8.00e-8`, a reduction of approximately `34.6%`
  from the local champion;
- Phase 4 stretch target: `< 4.00e-8`, a reduction of approximately `67.3%`
  from the local champion.

Only the fixed eight MLPs are in scope for this run. Eight-MLP results are a
fast research surface, not proof of private-set generalization. Broader
confirmation is deferred until the user explicitly requests it.

Do not submit anything to AIcrowd without explicit user approval.

## Why Phase 4 is a reset

Run 3 improved the local score from `1.231901e-7` to `1.223643e-7`, only
`0.67%`. The result was directionally real, but it did not satisfy Run 3's
written `5%` promotion gate or its `< 1.00e-7` / `10% different-family`
submission gate. The evaluation helper also omitted the written minimum-gain
condition and treated any lower mean with six wins as promotable.

Therefore:

1. The current estimator remains the immutable operational control.
2. Tiny improvements may be recorded, but they cannot become Phase 4 champion.
3. Further sweeps of `lambda`, `s_0`, `alpha`, sample count near `4,200`, or
   closely related Hermite coefficients are not primary Phase 4 research.
4. `8W-0L` is not sufficient when the gains are numerically tiny.
5. Sampler salts measure stochastic stability; they are not additional MLPs.
6. A failed prototype does not close a broader family unless its oracle or
   mechanism-level gate also fails.

## Agent operating contract

Work autonomously through the experiment queue. Do not stop after the first
negative result, and do not declare success for a new personal best above the
primary target.

The agent must:

- use `uv` for every Python command and dependency action;
- preserve `estimator.py` until all Phase 4 promotion gates pass;
- create one candidate file per distinct method or ablation under `candidates/`;
- use only Phase 2 `v2-phase2`, width `1024`, depth `16` MLPs for keep/reject
  decisions;
- use the same fixed eight MLPs and the same ordering as prior runs;
- use common random numbers for paired comparisons whenever meaningful;
- keep oracle calculations in offline research scripts and never leak target
  values, network IDs, names, or target-derived per-network choices into a
  deployable estimator;
- run the official estimator/package contract checks before promotion;
- record every completed experiment, including negative and failed ones;
- label conclusions as `measured`, `derived`, `supported interpretation`, or
  `untested hypothesis`;
- continue across method families according to the decision tree below rather
  than repeatedly tuning whichever family most recently improved by a fraction
  of a percent.

## Fixed eight-MLP evaluation protocol

### Evaluation stages

1. **Mathematical pre-smoke**
   - Use small arrays or one Phase 2 MLP to catch shape, algebra, and FLOP issues.
   - This stage cannot justify keeping or rejecting a method.

2. **Two-MLP smoke**
   - Run two fixed panel MLPs with one deterministic sampler seed.
   - Stop only for contract failure, catastrophic error, impossible cost, or a
     falsified mathematical identity.

3. **Eight-MLP screen**
   - Run all eight MLPs under the official local protocol.
   - Compare with the immutable current champion using paired per-MLP results.

4. **Salt robustness**
   - Run all eight MLPs under three deterministic sampler salts only when a
     stochastic candidate improves the official-seed mean by at least `5%`, or
     when a strong oracle result justifies diagnosing a noisy implementation.

5. **Promotion reproduction**
   - Re-run the final candidate from a clean process.
   - Run official validation, calculate a code hash, build the package, and
     reproduce the package result locally.
   - Packaging does not authorize an AIcrowd submission.

### Required statistics

For every eight-MLP experiment, report:

- adjusted final-layer score;
- raw final-layer MSE;
- exact compute utilization and score multiplier;
- mean, median, best, and worst per-MLP score;
- paired percent change on every MLP;
- win/loss/tie count against the current champion;
- maximum residual time and wall time;
- failures and fallback events;
- sampler seed or salt;
- candidate path and SHA-256 hash;
- residual correlation with each existing estimator branch when predictions are
  available;
- oracle headroom separately from deployable performance.

Any target-fitted shared model must use leave-one-MLP-out fitting. Report both
the eight LOO predictions and the all-eight refit intended for deployment. The
all-eight refit is not independent validation.

## Decision gates

### Continue a family when any one condition holds

- deployable eight-MLP adjusted score improves by at least `3%` with a coherent
  mechanism and at least `5/8` wins;
- an offline oracle shows at least `20%` achievable final-MSE reduction;
- a mathematical or execution prototype proves a necessary component works and
  the remaining failure is isolated;
- the candidate residual is sufficiently decorrelated that its oracle blend with
  the champion improves by at least `10%`.

### Kill or archive a branch when

- the relevant oracle improves by less than `10%`;
- projected scored compute cannot beat the champion even under the measured
  oracle accuracy;
- the mechanism fails on both the official seed and two principled variants;
- LOO performance reverses the fitted gain;
- the method needs target values, MLP identity, or seed lookup at inference;
- residual-time safety cannot be restored without destroying the measured gain.

A high-upside family with a strong oracle may receive up to three materially
different lawful attempts to recover that oracle. Do not spend those attempts on
scalar sweeps of the same feature or regularization constant.

### Promote a new Phase 4 champion only if all hold

- adjusted mean improves by at least `10%` over `1.223643e-7`;
- at least `6/8` MLPs improve;
- no MLP regresses by more than `10%` unless the mean gain exceeds `25%` and the
  regression has a documented cause;
- LOO and all-eight refit agree on direction for any fitted component;
- stochastic gains survive three salts;
- exact score remains favorable after the compute multiplier;
- zero failures;
- output is finite, nonnegative, correctly shaped, and `float32`;
- residual time remains comfortably below `0.40s`, preferably `< 0.30s`;
- clean-process and packaged reproduction pass.

### Complete Phase 4 only if

- the eight-MLP adjusted score is `< 8.00e-8`;
- the result has at least `6/8` wins against the original Phase 4 control;
- stochastic robustness, contract validation, package reproduction, and all
  operational checks pass;
- the improvement comes from a substantive method change rather than a hidden
  target lookup or repeated eight-panel overfit.

A score between `8.00e-8` and the current champion may be promoted as an interim
champion if it clears the `10%` promotion gate, but it does not complete the
goal. The stretch result is `< 4.00e-8`.

## Research portfolio

Allocate effort approximately as follows:

- **Lane A - angular carrier bank:** 30%;
- **Lane B - lawful adaptive or signed endpoint combination:** 35%;
- **Lane C - mid-network response control:** 20%;
- **Lane D - compressed higher-cumulant centering:** 15%.

Execution improvements such as routing, compaction, FWHT, and stable-neuron
bypasses are cross-cutting infrastructure. After at most two consecutive
experiments in one lane, run an experiment or oracle from another lane unless
the current sequence is a required ablation of a gain larger than `10%`.

## Lane A - Build an angular endpoint bank

### Hypothesis

Positive homogeneity separates Gaussian radius from direction exactly. The
current WMC branch spends samples on a geometry that only matches low moments
approximately. Mutually unbiased angular blocks, randomized QMC, or BCH-like
orthogonal arrays may produce more informative final-layer estimates at the same
or moderately higher scored compute.

The earlier `estimator_hadamard_hermite.py` result does not close this family. It
tested a small randomized Hadamard construction, not a proper multi-basis MUB or
BCH bank with endpoint diagnostics, exact radial treatment, and structure-aware
execution.

### Required carrier tournament

Build and compare, without an analytic blend first:

1. current whitened antithetic MC;
2. exact-radius antithetic random directions;
3. Owen-scrambled Sobol or the closest legal flopscope implementation;
4. randomly shifted rank-one lattice with a tent/baker transform;
5. randomized orthogonal/Hadamard frames;
6. partial `1024`-dimensional Kerdock/MUB-style banks;
7. a BCH/orthogonal-array bank if a lawful compact generator is feasible.

For partial structured banks, start with approximately `4,096`, `8,192`, and
`16,384` antipodal line representatives or the nearest complete block counts.
Measure the exact adjusted score at utilization targets near `10%`, `15%`,
`20%`, and `30%`; do not compare raw MSE without applying the multiplier.

Use exact radial integration wherever the carrier lives on the unit sphere.
Keep the identity or coordinate basis as a separate ablation when the
construction provides one. Apply random sign/permutation scrambles derived only
from the legal estimator seed.

### Endpoint instrumentation

Each angular basis or block must expose an offline endpoint record:

- final mean vector for that basis;
- selected intermediate mean vectors, initially layers `4`, `6`, `8`, `12`,
  and `16`;
- per-layer firing fraction and stable/dead/kink summaries;
- cost of the block;
- deviation from the uniform bank mean;
- deviation from the analytic champion branch;
- leave-one-block-out influence on the final estimate.

The production estimator need not return or retain these diagnostics. They are
research data for Lane B.

### Lane A gate

Do not discard a structured bank before running the Lane B endpoint oracle.
Uniform averaging is not the only intended estimator.

Prefer a carrier for production when it has one or more of:

- at least `30%` lower raw sampler MSE than WMC at comparable scored compute;
- at least `10%` lower adjusted score by itself;
- an oracle blend with the analytic champion exceeding `15%`;
- substantially larger endpoint-reweighting headroom than the competing bank.

## Lane B - Solve target-side contraction

### Central hypothesis

Different angular bases produce different final estimates. Their contrast
directions can contain a large correction even when their uniform average is
already near the limit of fixed positive cubature. A winning method may require
predicting, from lawful white-box observables, which endpoints are high or low
for the current network and combining them with constrained nonuniform or signed
weights.

This is the primary high-upside Phase 4 lane.

### Mandatory oracle ladder

Using saved endpoint predictions and local truths, calculate separately:

1. best global fixed weights shared across all eight MLPs;
2. leave-one-MLP-out shared weights;
3. per-network nonnegative mass-one oracle weights;
4. per-network signed mass-one ridge weights;
5. low-rank signed weights of the form
   `w = uniform + U @ a`, with each column of `U` summing to zero;
6. subset-selection or top-k basis oracle;
7. oracle convex blends of the reweighted bank, current analytic branch, and
   current WMC branch.

Per-network oracle numbers are ceilings only. They may never directly choose
production weights.

### Lawful selector feature families

Try mechanistically distinct feature families rather than repeated scalar
tuning:

1. **Cross-bank uncertainty**
   - split the bases into two disjoint banks;
   - estimate block covariance and cross-half disagreement;
   - derive generalized least-squares or jackknife weights;
   - test whether minimizing held-bank disagreement predicts lower true error.

2. **Late-layer analytic discrepancy**
   - compare each basis endpoint with analytic means at layers `4/6/8/12`;
   - use changes across depth, not only the discrepancy at the selection layer;
   - identify which contrast directions persist, rotate, or regenerate.

3. **Response/Krylov features**
   - propagate a small set of discrepancy modes through expected ReLU gates;
   - construct up to eight weight-derived Krylov or adjoint response modes;
   - regress only the coefficients, never a full `1024 x 1024` map.

4. **Stable-neuron pseudo-labels**
   - identify conservative stable-on or exact-linear terminal coordinates;
   - use identities available for those coordinates to score endpoint weights;
   - test whether their basis-error direction predicts the kink-coordinate
     correction.

5. **Shared low-capacity selector**
   - fit a small constrained ridge or low-rank linear rule across MLPs;
   - train on the exact noisy features available in the deployed path;
   - use eight-fold leave-one-MLP-out validation;
   - cap signed-weight norm and effective-sample-size collapse.

6. **Weight-only structural features**
   - gate probabilities and near-kink fractions;
   - weight row/column norms and low-order spectral probes;
   - response-mode energies;
   - final-layer stable/dead/kink composition;
   - covariance or cumulant disagreement summaries.

Do not use `mlp.name`, raw dataset IDs, a target lookup table, or a fitted rule
whose only useful input is the public seed.

### Lane B gates

Continue the lane if the signed or subset oracle reduces the uniform-bank MSE by
at least `2.5x`, or if its blend with the current champion shows at least `25%`
headroom.

A lawful selector is promising when it:

- recovers at least `20%` of the oracle improvement under leave-one-MLP-out;
- improves at least `6/8` MLPs;
- maintains bounded signed weights and stable effective sample size;
- survives a basis scramble or disjoint bank construction.

If three genuinely different selector mechanisms recover less than `10%` of a
large oracle gain, archive the lane with the oracle and feature-alignment
diagnostics. Do not continue with regularization sweeps alone.

## Lane C - Mid-network response control

### Hypothesis

The current analytic branch is not accurate enough to replace sampling at the
final layer, but it may estimate a mid-layer mean well enough to center a control
variate. The previous mid-network experiment used a crude single Jacobian and
does not close response-aware, low-rank, or endpoint-trained transport.

### Oracle sequence

For candidate control layers `4`, `6`, `8`, and `10`:

1. measure the sampled or angular mean discrepancy at the control layer;
2. fit the best offline linear transport from that discrepancy to final error;
3. repeat with discrepancy projected onto `2`, `4`, `8`, and `16` modes;
4. compare an expected-gate Jacobian, endpoint-fitted response, and Krylov
   response basis;
5. calculate the oracle benefit using the true control-layer mean to separate
   centering error from response-transport error;
6. price the complete scored implementation.

### Deployable variants

- analytic covariance mean as the deterministic center;
- compressed K=3/K=4 center from Lane D;
- two independent angular banks, one supplying a frozen control center and the
  other supplying the corrected estimate;
- a shared ridge response trained on noisy endpoint trajectories;
- shrinkage of the transported correction fixed by LOO, not per-network truth.

### Lane C gate

- Kill a control layer if its perfect-centering oracle offers less than `20%`
  final-MSE improvement.
- Continue when a deployable center recovers at least one quarter of the oracle
  gain.
- Promote the component only when the fused estimator improves by at least
  `10%`, wins at least `6/8`, and pays for its multiplier increase.

## Lane D - Compressed higher-cumulant centering

### Scope

Do not implement the full factorized K=3 tensor at `1024 x 16`. Its leading
published cost is approximately

`30*n^3*L^2 + 39*n^3*L`,

which is about `8.9e12` FLOPs at Phase 2 dimensions before the rest of the
estimator, roughly four complete Phase 2 budgets.

Test only compressed or localized forms:

1. correct-sign diagonal skewness/kurtosis recurrence as a cheap diagnostic;
2. K=3 propagation over the final `2`, `3`, or `4` layers;
3. rank `2/4/8` covariance-response or third-cumulant modes;
4. adjoint-selected contractions aimed at final-layer means;
5. a cumulant state used as the Lane C control center;
6. a small universal pair-response table concentrated on near-kink neurons.

Use the ARC cumulant repository as an offline reference implementation and
oracle. Validate every approximation on small dimensions before Phase 2 runs.

### Lane D gate

Continue only if an offline oracle shows at least `20%` final-MSE headroom and a
scored implementation can plausibly remain below `40%` utilization. Above the
`10%` score floor, the proportional raw-MSE improvement must exceed the
proportional utilization increase.

Do not inject a correction at every layer merely because it improves that
layer's local moments. Measure the complete final-layer effect; closure errors
can compensate across depth.

## Cross-cutting execution chassis

Apply these only when they support a statistically promising estimator:

1. exact Gaussian radial expectation for spherical carriers;
2. antipodal identities, including an exact first-layer fold where available;
3. FWHT or structured first-layer evaluation for compatible bases;
4. conservative pilot-frozen dead/on/kink classification;
5. prefix or fixed-bucket compaction rather than repeated host-side dynamic
   decisions;
6. stable-on final-layer linear bypass and stable-off zeros;
7. final-layer-only expensive sampling, with cheap valid values for diagnostic
   intermediate output rows;
8. `float32` throughout unless a measured numerical failure requires otherwise;
9. no `bool()`, `float()`, `int()`, assertion, or repeated materialization of a
   pending flopscope value inside `predict()`;
10. conservative worst-case FLOP, residual-time, memory, and wall-time sizing.

Compute reduction below `10%` utilization does not improve the score. Execution
work becomes valuable when it creates room for a more accurate branch or lowers
the multiplier of an estimator operating above the floor.

## Combination rules

Do not blend methods merely because both are individually good. For every pair
of serious candidates:

1. save final residual vectors on all eight MLPs;
2. calculate pooled and per-MLP residual correlations;
3. calculate the fixed global convex oracle;
4. calculate a signed two-branch oracle only as a diagnostic;
5. compare LOO-fixed blend weights with the all-eight optimum;
6. reject the blend if oracle improvement is below `10%`;
7. require the deployed blend to improve by at least `5%` beyond its strongest
   component before adding complexity.

The current Hermite/covariance estimator should become an anchor, control, or
blend component. It should not dictate the architecture of every new method.

## Mandatory experiment order

### P4-00 - Repair the research instrument

- Correct `scripts/run_eval.py` so its labels implement this plan's exact gates.
- Freeze the Phase 4 control hash and reproduce `1.223643e-7`.
- Add fields for relative improvement, oracle headroom, branch correlations, and
  exact multiplier penalty.
- Do not modify the control estimator.

### P4-01 - Carrier baselines

- Reproduce pure WMC and the existing Hadamard candidate.
- Build exact-radius random and randomized-QMC controls.
- Compare raw and adjusted errors at matched cost.

### P4-02 - Partial MUB/structured endpoint bank

- Build the smallest lawful structured bank.
- Validate its moments and antipodal/radial identities.
- Run `4k`, `8k`, and `16k`-class bank sizes where feasible.
- Save per-basis endpoint records.

### P4-03 - Endpoint oracle

- Run the complete Lane B oracle ladder before optimizing production code.
- If oracle headroom is weak, deprioritize adaptive reweighting.
- If oracle headroom is strong, the next experiments must target the contraction
  problem rather than uniform-bank micro-optimization.

### P4-04 - Lawful selector attempts

Try, in order:

1. cross-bank GLS/jackknife;
2. late-layer analytic-discrepancy weighting;
3. low-rank response/Krylov weighting;
4. stable-neuron pseudo-label calibration;
5. a shared LOO ridge using combinations of the successful features.

Each numbered item is one mechanism family, not permission for an unlimited
hyperparameter sweep.

### P4-05 - Mid-network control oracle and prototype

- Screen layers `4/6/8/10` by oracle.
- Build only the best one or two control layers.
- Prefer endpoint-trained or low-rank response transport over the previously
  failed crude Jacobian.

### P4-06 - Compressed cumulant center

- Run cheap diagonal and final-two-layer diagnostics.
- Scale to response ranks `2/4/8` only if the oracle justifies it.
- Test both standalone and as the Lane C center.

### P4-07 - Decorrelated portfolio

- Compare the best angular, adaptive, control, cumulant, and current champion
  residuals.
- Fuse only branches with at least `10%` oracle blend headroom.
- Re-run the complete eight-MLP and salt protocol.

### P4-08 - Execution optimization and reproduction

- Add routing/FWHT/compaction only to the winning statistical design.
- Recalculate score after every cost optimization.
- Validate, hash, package, and reproduce without submitting.

## Agent decision tree

After each experiment:

1. **Contract or mathematical failure**
   - diagnose and repair once;
   - if repaired, repeat the same experiment ID with a revision suffix;
   - if fundamental, record and move lanes.

2. **Actual gain at least 10%**
   - perform required ablations and salts immediately;
   - compare against the original Phase 4 control as well as any interim
     champion;
   - promote only after all gates pass.

3. **Actual gain 3-10%**
   - continue only with a clear mechanism, strong oracle, or decorrelated blend
     value;
   - allow one principled extension, then switch lanes.

4. **Actual gain below 3%, oracle strong**
   - isolate whether the loss is carrier geometry, target-side contraction,
     centering, response transport, or compute price;
   - try a materially different lawful mechanism, not a scalar sweep.

5. **Actual gain below 3%, oracle weak**
   - archive the family and move on.

6. **Primary target reached**
   - run clean reproduction, three salts, validation, package, and final report;
   - do not submit;
   - mark Phase 4 successful only after every gate passes.

## Required experiment record

Append every experiment to a Phase 4 report using this schema:

```text
Experiment ID:
Date/time:
Lane and method family:
Hypothesis:
Mathematical mechanism:
Candidate file and SHA-256:
Control file and SHA-256:
Parameters frozen before run:
MLP panel and sampler salts:
Expected failure mode:

Control adjusted score:
Candidate adjusted score:
Relative adjusted improvement:
Raw final-layer MSE:
Compute utilization and multiplier:
Per-MLP scores and paired percent changes:
Wins/losses/ties:
Median and worst result:
Residual correlations with available branches:
Oracle ceiling and definition of oracle:
LOO result, if fitted:
All-eight refit result, if fitted:
Maximum residual/wall time:
Failures:

Evidence label:
Interpretation:
Decision: promote / continue family / reject / archive
Exact gate applied:
Next experiment justified by this result:
```

Do not reconstruct missing metrics from memory. Record them directly from the
completed evaluation outputs.

## Explicitly deprioritized work

Do not spend primary Phase 4 time on:

- more `lambda`, `s_0`, global `alpha`, or nearby Hermite coefficient sweeps;
- standalone final-layer scale or affine calibration;
- merely increasing ordinary MC sample count;
- another axis-aligned Hadamard cloud without multi-basis endpoint analysis;
- a full `1024`-dimensional complete MUB bank that cannot fit the budget;
- a full K=3 or K=4 tensor;
- flexible target-trained models on only eight MLPs;
- per-network target-aware selection disguised as adaptivity;
- leaderboard hill-climbing;
- execution optimization of a statistically inert method;
- declaring victory for `8W-0L` when the mean gain is below the Phase 4 gates.

## Scientifically useful outcomes that do not complete the goal

Record these carefully but continue working:

- a partial MUB/BCH implementation with validated angular identities;
- a large endpoint-reweighting oracle with a failed lawful selector;
- a mid-network control oracle that identifies the best layer but lacks a good
  center;
- a compressed cumulant state that improves centering but not the final score;
- a new interim champion above `8.00e-8`;
- a proof that a broad family cannot reach the target under its measured oracle
  and cost ceiling.

These are valuable findings and potential write-up material. They are not Phase
4 completion.

## References

- [Official Phase 2 announcement and contract](https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197)
- [Official scoring model](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/concepts/scoring-model.md)
- [Official allowed-code policy](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/concepts/allowed-code.md)
- [Official algorithm ideas](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/how-to/algorithm-ideas.md)
- [Official Phase 2 dataset](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026)
- [ARC cumulant-propagation paper](https://arxiv.org/html/2605.05179v2)
- [ARC cumulant reference code](https://github.com/alignment-research-center/mlp_cumulant_propagation)
- [Public Kerdock/MUB implementation and research archive](https://github.com/SkyeNygaard/arc-whitebox)
- [Phase 1 technique census and open problems](https://discourse.aicrowd.com/t/a-technique-census-of-phase-1-what-the-mathematics-is-doing-what-walls-it-hit-and-whats-still-open/18157)
- `PHASE1_STRATEGIES.md`
- `PHASE2_EXPLORATION.md`
- `PHASE3_RESEARCH_PLAN.md`
- `phase 2 summary run 2.md`
- `Current thing.md`
