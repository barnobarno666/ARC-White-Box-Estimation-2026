# ARC White-Box Estimation Challenge 2026: Phase 1 review and strategy map

Research snapshot: **2026-08-21**. This note separates official rules, the frozen
public leaderboard, and participant-reported methods. Those are not the same
thing: Phase 1 prizes are decided by a fresh private re-evaluation, and no
official Phase 1 winner announcement was available when this review was written.

## Executive take

- This repository is the clean, official starter kit, not a competition
  solution. [`estimator.py`](estimator.py) still returns an all-zero prediction.
- The checked-out `main` is Phase 1-era. A newly published official-repository
  branch signals a different Phase 2 target (`1024 x 16`, `v2-phase2`, and a
  `2**41` budget), so Phase 1 code and timings must not be carried over blindly.
- Phase 1 asks for the expected post-ReLU activation of every neuron in a
  random, bias-free `256 x 32` MLP under standard-normal input. Only the final
  layer determines rank.
- The metric rewards both accuracy and low effective compute. Above the 10%
  compute floor, merely changing a sampler's number of rows usually does almost
  nothing to its score; useful work must reduce variance per row or cost per row.
- The strongest *documented and transferable* Phase 1 family was structured
  sampling/cubature plus ReLU-aware routing: exact radial handling, Sobol or
  Kerdock/MUB directions, dead/on/kink classification, pilot pruning, sparse
  continuation, and careful metered linear algebra.
- Pure Gaussian/cumulant propagation is very cheap but loses too much joint
  dependence over 32 ReLU layers. Its best role is as a pilot, routing model,
  control-variate anchor, or feature generator—not the only estimator.
- A sensible implementation order is: measurement harness -> honest randomized
  QMC baseline -> structural pruning/routing -> mid-network control variate.
  Treat a learned joint-state propagator as a separate high-risk research lane.
- On this laptop, a warmed one-thread Phase 1 prediction takes about `0.067 s`
  for mean propagation, `0.38 s` for full covariance propagation, and `1.29 s`
  for a 10,000-row Monte Carlo estimator. The 100-MLP cached Mini split is
  therefore seconds to a few minutes depending on the method. Regenerating
  ground truth is radically slower and should be avoided.

## 1. What is actually in this repository

The checked-out branch is `main` of
[`AIcrowd/whest-starterkit`](https://github.com/AIcrowd/whest-starterkit), pinned
to Phase 1-compatible packages:

- Python `>=3.10`
- `flopscope >=0.10,<0.11`
- `whestbench >=0.14,<0.15`
- the lock file currently resolves `flopscope 0.10.0`, `whestbench 0.14.0`, and
  NumPy `2.2.6`

The useful pieces are:

| Path | What it provides |
|---|---|
| [`estimator.py`](estimator.py) | Submission entry point; currently the zero baseline |
| [`local_engine.py`](local_engine.py) | One-file pedagogical MLP and Monte Carlo harness |
| [`examples/02_mean_propagation.py`](examples/02_mean_propagation.py) | Diagonal Gaussian mean/variance propagation, `O(depth * width^2)` |
| [`examples/03_covariance_propagation.py`](examples/03_covariance_propagation.py) | Approximate full covariance propagation, `O(depth * width^3)` |
| [`docs/concepts/scoring-model.md`](docs/concepts/scoring-model.md) | Current scoring formula and baseline calibration |
| [`docs/how-to/manage-flop-budget.md`](docs/how-to/manage-flop-budget.md) | Metering and profiling details |
| [`docs/how-to/use-evaluation-datasets.md`](docs/how-to/use-evaluation-datasets.md) | Baked Mini/Full dataset workflow |

There are no competition weights, trained correction model, experiment logs, or
participant estimator in the tree. Any serious entry must therefore be built
from the baselines or ported from a public write-up/repository.

The official remote also contained an unmerged branch named
`origin/chore/flopscope-main-pin` at
[commit `ce4b84a`](https://github.com/AIcrowd/whest-starterkit/commit/ce4b84ad5aeb01cff71255248843d8522e0c4821)
on 2026-08-21. Its stated
Phase 2 configuration is width 1,024, depth 16, dataset revision `v2-phase2`,
budget `2**41`, `flopscope 0.11`, and `whestbench 0.15`. This document keeps its
Phase 1 analysis pinned to the checked-out `main`; verify that the official
Phase 2 changes have landed before implementing a live submission.

## 2. The exact Phase 1 problem

For a fixed randomly initialized MLP,

```text
h[0] = X,                         X ~ N(0, I_256)
h[l] = ReLU(h[l-1] @ W[l]),      l = 1, ..., 32
```

where every `W[l]` is `256 x 256`, has no bias, and is He-Gaussian initialized
with variance `2/256`. The estimator receives all weights and must return a
`32 x 256` matrix whose entry `(l, i)` estimates `E[h[l, i]]`.

The target matrices in the official datasets were baked using `N = 1e9`
Monte Carlo inputs. Mini contains 100 MLPs (about 850 MB) and Full contains
1,000 MLPs (about 8.5 GB). The baked target matters: a benchmark against it
loads the answer; it does **not** rerun a billion samples.

Phase 1 changed the warm-up shape from `256 x 8` to `256 x 32` and the budget
from `6.8e10` to `2.72e11` FLOPs per MLP. The official
[Phase 1 launch post](https://discourse.aicrowd.com/t/phase-1-launch-deeper-models-and-increased-prizes/18026)
is the cleanest historical statement of those values.

### Scoring

For MLP `m`,

```text
final_mse[m] = mean_i((prediction[31, i] - truth[31, i])^2)
C[m]         = F[m] + 1e11 * R[m]
score[m]     = final_mse[m] * max(0.1, C[m] / 2.72e11)
leaderboard  = mean_m(score[m])
```

`F` is analytically metered arithmetic. `R` is residual participant wall time
outside metered kernels. Going over budget is a cliff: the prediction is
replaced by zeros and receives no compute discount. Earlier-layer MSE is a
diagnostic and tie-break input, not the primary score.

The challenge landing page still contained warm-up-era examples when this note
was written (eight layers and a `0.5` multiplier floor). For Phase 1 work, use
the pinned evaluator, starter-kit documentation, and the
[v0.10.0 Phase 1 update](https://discourse.aicrowd.com/t/phase-1-update-flopscope-v0-10-0-cost-model-fixes-residual-time-safeguards-and-updated-deadlines/18125).

### The sampler invariance that should drive strategy

For a sampler above the multiplier floor, approximately

```text
MSE(N) = a / N
C(N)   = p * N
score  = (a / N) * (p * N / B) = a * p / B
```

Changing `N` moves raw error and compute in opposite directions but largely
cancels in the ranked score. A sampling strategy improves by reducing:

1. `a`: variance/error per sample, through better point geometry, whitening,
   conditioning, or control variates; or
2. `p`: metered price per sample, through exact pruning, sparse routing, fast
   products, fewer data movements, and fewer calls.

Below 10% budget the multiplier is fixed at `0.1`, so underspending further is
not rewarded. This is why a very cheap but biased analytic method can still
lose, and why blindly increasing sample count is not a strategy.

## 3. What “top Phase 1 solutions” currently means

### Public board: scores are visible, methods mostly are not

The frozen [public leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards)
showed the following leading rows when checked on 2026-08-21:

| Public rank | Participant/team | Adjusted score | Raw final-layer MSE |
|---:|---|---:|---:|
| 1 | `ednacob` | `1.84e-8` | `3.63e-8` |
| 2 | `J2W` | `5.39e-8` | `2.784e-7` |
| 3 | `dstepanov` | `6.22e-8` | `2.199e-7` |
| 4 | `dpskv5` | `7.17e-8` | `1.504e-7` |
| 5 | `oabuod` | `7.23e-8` | `2.357e-7` |

These rows do **not** reveal their algorithms. Organizers explicitly warned
that some entries fit the 50 public MLPs or their identifiers; that was allowed
on the public board but should not generalize to fresh seeds. The board also
contains entries affected by evaluator-version changes and disclosed metering
defects. Public rank is therefore useful evidence of what score was possible on
that fixed set, but not evidence of a reusable solution or final prize rank.

Phase 1 closed on 2026-08-10. Teams selected up to two artifacts for a fresh
private re-evaluation. See the official
[selection announcement](https://discourse.aicrowd.com/t/phase-1-submission-selection-phase-2-launch-date/18143)
and the [compute/ranking update](https://discourse.aicrowd.com/t/phase-1-update-flopscope-v0-10-0-cost-model-fixes-residual-time-safeguards-and-updated-deadlines/18125).
As of this snapshot, the public forum and challenge page did not contain an
official private-re-evaluation winner announcement.

### Strong documented methods

This table describes reproducible ideas rather than claiming final placement.
Adjusted scores should only be compared directly when evaluator versions match;
raw MSE is more stable across repricing.

| Method / submission | Reported result | Main idea | Main caveat |
|---|---|---|---|
| `mliston`, #327723 | adjusted `1.334e-7` | Antithetic sampling plus a layer-22 control variate. A cumulant/Hermite anchor is corrected by a small offline-trained GRU before propagating the discrepancy. | The learned gain was measured on small panels; bias and private-seed transfer are the risk. The author excluded a better-looking row affected by a metering issue. |
| `thylinao`, honest twin #326694 | adjusted `1.382e-7`, raw `1.7238e-7`, multiplier `0.8017` | Six scrambled Sobol nets, final-layer-only output, pilot pruning, sparsity tiers, two-level Strassen, and exact/certified linear routes. | Still sample-heavy and slightly biased by pruning/certification. A separate public-board twin used a disclosed undercount, so use #326694 as the honest reference. |
| `konstantin_baltsat`, #326680 | adjusted `1.4391e-7`, raw `2.4258e-7`, multiplier `0.5933` | 66,048 Kerdock/MUB antipodal directions, exact radial conditioning, FWHT first layer, conservative pilot compaction, dead/on/kink folding, exact compiler reductions. | Complex implementation; raw MSE was about 11.3% worse than SOX and the advantage was cost-driven. |
| Team SOX, #319341 | adjusted `1.551e-7`, raw `2.18e-7` in its write-up | Moment-based dead/on/kink classification, pilot reclassification, antithetic Sobol sampling, linear handling of final always-on neurons, sparse row/column routing. | Quoted under an earlier evaluator epoch; authors warned it does not transfer easily to changed architectures. |
| `evaaaz` RQMC baseline | adjusted about `4.10e-7`, `C/B about 0.42` | Randomly shifted rank-1 lattice plus exact analytic layer-1 means. Unbiased aside from numerical clipping. | Valuable baseline, but deep ReLU kinks erase much of the low-discrepancy advantage. |
| `keenanpepper`, #327838 | raw `1.50e-5`, adjusted `1.95e-6` | Fully sample-free learned recurrent propagator carrying means, second moments, a compressed third-cumulant plane, and latent state. | Scientifically interesting but not score-frontier; an expensive learned branch was found inert after submission. |

Primary method descriptions:

- [Structure-aware SOX estimator](https://discourse.aicrowd.com/t/phase-i-submission-structure-aware-estimation-for-random-relu-mlps/18106)
- [Kerdock/MUB plus metered compaction](https://discourse.aicrowd.com/t/phase-1-write-up-metered-dead-compacted-kerdock-cubature-with-exact-compiler-reductions-submission-326680/18149)
- [Mid-network learned control variate](https://discourse.aicrowd.com/t/phase-1-write-up-bias-variance-tradeo-across-control-layers-submission-327723/18170)
- [Unbiased randomized-QMC baseline](https://discourse.aicrowd.com/t/unbiased-randomized-qmc-rao-blackwell-for-post-relu-activation-means-method-unbiasedness-proof-and-where-the-frontier-is/18053)
- [Learned sample-free propagator](https://discourse.aicrowd.com/t/bitter-lesson-pilled-sampling-free-propagator-yielding-1-5e-5-raw-mse/18177)
- [Phase 1 technique census](https://discourse.aicrowd.com/t/a-technique-census-of-phase-1-what-the-mathematics-is-doing-what-walls-it-hit-and-whats-still-open/18157)
- [Expanded participant field guide and honest sampler](https://thylinao1.github.io/projects/white-box-estimation.html)

Participant numbers above are self-reported unless tied to an AIcrowd submission
record. They were not all independently reproduced for this review.

## 4. The durable Phase 1 lessons

### 4.1 Cheap marginal propagation loses the important state

The bundled diagonal and covariance baselines report approximately:

| Starter estimator | Raw final-layer MSE | Metered FLOPs per MLP |
|---|---:|---:|
| Mean/diagonal propagation | `9.5e-4` | `20,333,056` |
| Full covariance propagation | `8.4e-5` | `3,262,575,998` |

Advanced participant experiments put the idealized two-moment/Gaussian readout
floor near `8.6e-7` to `9.3e-7` raw even when moments are supplied almost
perfectly. The missing object is cross-neuron, non-Gaussian dependence, not a
slightly better final formula. ReLU introduces an atom at zero and correlations
compound through depth; layerwise errors can also cancel, so locally “correcting”
each layer may make the final answer worse.

Consequence: do not spend a long iteration cycle on another small marginal
readout tweak without an oracle upper-bound test showing that lane can reach the
target score.

### 4.2 ReLU geometry gives exact structure to exploit

- Positive homogeneity separates Gaussian radius from direction exactly. The
  radial expectation can be integrated analytically, leaving a spherical
  directional problem.
- Far-negative neurons are effectively dead; far-positive neurons are locally
  linear; only neurons near the kink at zero need expensive treatment.
- Published diagnostics suggest error is concentrated in roughly 66 of 256
  near-kink neurons, while later activations have very low effective rank.
- A small pilot can route computation. Deliberately dropping extremely rare
  firing events may improve MSE per FLOP even when it introduces tiny bias.

This explains the convergence on dead/on/kink routing in SOX, Kerdock, and
optimized Sobol submissions.

### 4.3 Structured point sets help, but input geometry is not enough

Scrambled Sobol/RQMC, antipodal pairs, batch whitening, and Kerdock/MUB designs
reduce avoidable sampling error. Kerdock directions are especially attractive
because the complete real construction is nearly minimal for its design class
and permits a fast Walsh-Hadamard first layer.

However, after 32 nonlinear layers, deep “kink variance” dominates. Input-level
controls often overlap with the variance already removed by a digital net, so
their gains do not add. Controls placed mid-network are more promising because
they target the propagated error that remains.

### 4.4 Billing is part of the algorithm

- Keep hot paths in `float32`; `float64` is billed at 2x and can appear through
  silent promotion.
- Genuine symmetric/Gram contractions can receive about half-price accounting.
- Batch work into fewer, larger calls: residual wall-time behaves partly like a
  per-call tax.
- Prune actual arithmetic and data movement. Merely spelling the same dense
  contraction as `matmul`, `einsum`, or `tensordot` does not create a legitimate
  discount.
- Keep margin below `2.72e11`; one over-budget MLP is zeroed and can dominate an
  aggregate result.
- Do not build around a meter bug. Phase 1 defects were patched and entries were
  regraded.

## 5. Strategy options for this repository

### Strategy A — establish an honest baseline ladder

Effort: low. Risk: low. Expected competitiveness: low, but mandatory.

1. Preserve the zero estimator as a smoke-test reference.
2. Copy mean propagation into a new experimental estimator and make all hot
   arrays explicitly `float32`.
3. Reproduce the documented `20.33M` FLOPs and Mini raw MSE around `9.5e-4`.
4. Reproduce covariance propagation around `3.263B` FLOPs and `8.4e-5` raw MSE.
5. Record raw MSE, adjusted score, multiplier, `flops_used`, residual time, and
   wall time per MLP—not just a single mean score.

This creates trusted measurement infrastructure and detects Phase 1/Phase 2
configuration drift before more complex work starts.

### Strategy B — honest randomized QMC with exact conditioning

Effort: medium. Risk: medium. Expected competitiveness: solid baseline.

- Use independently scrambled Sobol blocks or a randomized lattice.
- Exploit positive homogeneity: sample directions, integrate radius exactly.
- Use the exact first-layer Gaussian mean instead of estimating it.
- Test batch whitening as an alternative carrier transformation; participant
  reports claim roughly `2.0x` to `2.34x` in suitable settings, but it may be a
  substitute for QMC rather than an additive gain.
- Use several independent scrambles so the estimator produces an internal
  error estimate and experiments can use paired comparisons.

This is the best first nontrivial implementation because it is understandable,
can be kept nearly unbiased, and supplies the sampler needed by later hybrids.

### Strategy C — structure-aware QMC routing (recommended core)

Effort: high. Risk: medium. Expected competitiveness: best proven public lane.

Build Strategy B, then add:

1. a cheap diagonal/moment pilot for preliminary dead/on/kink labels;
2. a small sample pilot that can override uncertain labels;
3. dead-column compaction and firing-rate row buckets;
4. exact linear propagation for confidently always-on terminal paths;
5. full ReLU evaluation only for kink paths;
6. a small sweep over rare-fire pruning thresholds with a protected holdout;
7. exact cost reductions such as profitable Strassen/Winograd levels, symmetric
   products, and fewer data movements/calls.

Promote each component only on `MSE * effective_compute`, not raw MSE alone.
Rare-fire pruning is a sharp cliff in published experiments, so thresholds need
paired, out-of-sample tests and a conservative fallback.

### Strategy D — Kerdock/MUB deterministic spherical cubature

Effort: very high. Risk: medium-high. Expected competitiveness: strong,
independent carrier.

- Construct 129 real mutually unbiased bases and their antipodes: 66,048
  directions in dimension 256.
- Integrate Gaussian radius exactly.
- Evaluate the structured first layer with a signed fast Walsh-Hadamard
  transform.
- Reuse an initial prefix as a support pilot and compact later work.
- Add terminal dead/on/kink folding.

This is valuable even if it does not replace Sobol: it provides an independent
error geometry for blending and ablation. Validate the algebraic construction
with moment/design tests before trusting end-to-end scores. Do not assume the
entire gain comes from exact degree-5 design status; Phase 1 evidence points to
the mutually unbiased basis geometry itself as the more important mechanism.

### Strategy E — mid-network control variate

Effort: high. Risk: high. Expected upside: high if the anchor is accurate.

Run a sample estimator and a cheap analytic anchor together. At an intermediate
layer, compare their states and propagate a controlled fraction of that
difference to the output:

```text
estimate = sample_output - beta * propagated(sample_anchor - analytic_anchor)
```

The control should be placed deep enough to remain correlated with final error
but early enough that the analytic anchor is not badly biased. Phase 1 reports
put useful points around layers 6 and 22, depending on anchor quality. A small
offline-trained correction may improve centering, but it must use only legal,
weight-derived or already-computed features and be validated on entirely unseen
MLP seeds.

Test this first on ordinary/antithetic sampling. A low-order input-anchored
control may add nothing on top of a scrambled digital net because the net has
already removed the same variance component.

### Strategy F — learned joint-state mechanistic propagator

Effort: research-scale. Risk: very high. Expected score: uncertain.

Instead of another per-neuron correction, carry a compressed representation of
joint dependence: low-rank covariance responses, a third-cumulant plane,
Hermite coefficients, or recurrent latent state. Train the complete 32-layer
recursion end to end on freshly generated MLPs, then export only frozen weights
and `flopscope` inference code.

This lane addresses the real analytic failure but has two hard gates:

1. an oracle using the proposed state must beat the structure-aware sampler;
2. an ablation must show each expensive state branch materially changes held-out
   MSE.

Without both gates, the likely result is a sophisticated but inert branch like
several Phase 1 learned corrections.

### Recommended portfolio

| Priority | Workstream | Why |
|---:|---|---|
| 1 | A: baseline ladder and measurement | Makes every later result trustworthy |
| 2 | B: randomized QMC + radial split/whitening | Clean first competitive baseline |
| 3 | C: dead/on/kink routing and compaction | Most transferable Phase 1 score/cost gains |
| 4 | E: one mid-network control ablation | Best measured variance-side upside |
| 5 | D: Kerdock carrier in parallel | Independent deterministic geometry and strong published result |
| 6 | F: learned joint state | Only after oracle tests justify the cost |

## 6. If the real target is Phase 2

Phase 1's lessons transfer, but its constants and some constructions do not.
Based on the official repository's pending Phase 2 restatement:

| Parameter | Phase 1 | Pending Phase 2 restatement |
|---|---:|---:|
| Width | 256 | 1,024 |
| Depth | 32 | 16 |
| Budget per MLP | `2.72e11` | `2**41` (about `2.199e12`) |
| Dataset revision | `v1-phase1` | `v2-phase2` |
| Meter / harness | `flopscope 0.10` / `whestbench 0.14` | `0.11` / `0.15` |

The budget is calibrated to remain near 65,000 dense forward rows. The strategic
changes are more important than the similar row count:

- **Re-test analytic propagation.** The ARC companion paper predicts better
  relative behavior at larger width and fixed depth, while Phase 2 is also half
  as deep. A cumulant or low-rank joint-state method that lost at `256 x 32` may
  become competitive at `1024 x 16`.
- **Do not directly reuse the 256-dimensional Kerdock asset.** Its 66,048
  directions and FWHT construction are dimension-specific. A complete
  1,024-dimensional analogue would be far too large to propagate naively; use
  a principled subset or a newly priced structured transform.
- **Re-measure dead/on/kink sparsity.** With half the depth, activation collapse
  and terminal dead/on counts may be weaker. Phase 1 pruning thresholds are not
  portable constants.
- **Prefer low rank over full covariance.** Relative to Phase 1, a dense
  covariance step grows by roughly `32x` (`width^3 * depth`) while budget grows
  about `8x`; full covariance becomes about four times more expensive as a share
  of budget.
- **Keep the measurement protocol.** Radial conditioning, paired carrier
  replicates, oracle substitutions, float32 discipline, and private-seed
  lockboxes remain valid.

Rough one-core extrapolations from the Phase 1 timings are about `0.54 s` per
MLP for mean propagation, `12 s` for full covariance, and `10 s` for a 10,000-row
dense sampler on `1024 x 16`. These are scaling estimates, not measurements;
rerun them after switching to the official Phase 2 dependencies and dataset.

## 7. Experiment design that will survive private seeds

1. **Use three disjoint panels.** Development, validation, and untouched
   lockbox MLP seeds. The public Mini split is a final external check, not the
   training set.
2. **Pair every comparison.** Same MLPs, same carrier scramble/orientation, and
   same ground truth for baseline and candidate.
3. **Cross carrier randomness.** A single network's error can move several-fold
   across orientations/scrambles. Use multiple replicates and report the paired
   distribution, not one lucky run.
4. **Use oracle substitutions early.** Replace an internal approximate state
   with high-sample truth. If the resulting upper bound still misses the target,
   kill that whole family before implementing it.
5. **Predeclare promotion gates.** For example: no failures, no budget tail over
   90%, positive improvement on both validation and lockbox, and a bootstrap
   interval excluding zero for effects smaller than 5%.
6. **Track two frontiers.** Raw MSE versus FLOPs and adjusted score versus
   effective compute. A cheaper method can win with worse raw MSE.
7. **Test package parity.** Validate shape/dtype/finite outputs, run the local
   runner, run the subprocess runner, then replay the built archive.
8. **Measure the tail.** Report worst-MLP compute and error. Means can hide the
   exact network that is zeroed by a budget or wall-time cliff.

## 8. One-core CPU timing on this machine

### Measurement conditions

- CPU: Intel Core i5-1135G7, 4 physical cores / 8 logical processors
- OS: Windows 11
- Python: 3.10.20
- NumPy/OpenBLAS: NumPy 2.2.6, OpenBLAS 0.3.29
- `flopscope`: 0.10.0
- `whestbench`: 0.14.0
- Process affinity pinned to one logical processor; OpenBLAS, OMP, MKL, and
  NumExpr thread pools limited to one
- Warmed medians; dataset download, CLI startup, and dataset decoding excluded

The repository does not contain a trained “small model.” For reference, a
small `64 x 4` MLP took about `0.006 s` with mean propagation and `0.038 s` with
covariance propagation. A Phase 1 `256 x 32` MLP has 2,097,152 weights (about
8 MiB in float32) and produced:

| Estimator on one Phase 1 MLP | Median | Observed range | Serial 100 MLPs | Serial 1,000 MLPs |
|---|---:|---:|---:|---:|
| Current zero baseline | `0.00005 s` | negligible | `0.005 s` | `0.05 s` |
| Mean propagation | `0.067 s` | `0.054-0.095 s` | `6.7 s` | `1.1 min` |
| Full covariance propagation | `0.379 s` | `0.333-0.437 s` | `38 s` | `6.3 min` |
| Monte Carlo, 1,000 rows | `0.172 s` | run-dependent | `17 s` | `2.9 min` |
| Monte Carlo, 10,000 rows | `1.287 s` | `1.230-1.353 s` | `2.1 min` | `21.5 min` |

Allow at least 20-30% for laptop thermals and background load. Different CPUs
and BLAS builds can differ by 2x or more. Local one-core serial time is also not
the grader's elapsed time: the official Phase 1 update gave participant Python
one physical core while allowing the metered backend more cores, and the grader
evaluates MLPs in parallel.

For a cached baked dataset,

```text
T(N_mlps) ~= T_startup + T_dataset_load
             + N_mlps * (T_predict + T_score)
```

Cold Python/CLI startup here was about 2-4 seconds. The table is therefore most
useful for comparing estimator revisions after the dataset is cached.

### Do not accidentally benchmark ground-truth generation

The normal challenge benchmark reads baked `N=1e9` targets. If `--dataset` is
omitted, `whest run` generates a fresh local reference using 2.56 million
samples per MLP. Based on the one-core sampler:

| Ground truth job | Approximate one-core serial time |
|---|---:|
| Local default, `N=2.56M`, one MLP | `6.35 min` |
| Local default, 10 MLPs | `about 64 min` |
| Rebuild official `N=1e9`, one MLP | `about 41 h` |
| Rebuild official `N=1e9`, Mini 100 | `about 172 days` |

Those are linear extrapolations, not completed billion-sample runs. They explain
why the published dataset is essential.

### Reproducible PowerShell benchmark

Use `uv`; do not create another virtual environment.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

uv run --frozen whest profile-simulation `
  --preset super-quick `
  --max-threads 1 `
  --format json

uv run --frozen whest run `
  --estimator examples/02_mean_propagation.py `
  --dataset "hf://aicrowd/arc-whestbench-public-2026@v1-phase1" `
  --split mini `
  --n-mlps 10 `
  --runner local `
  --max-threads 1 `
  --profile `
  --format json
```

Run the scored command three times and use the median per-MLP wall time. Remove
`--n-mlps 10` for all 100 Mini MLPs. Use `--split full` for all 1,000 Full MLPs.
The first run downloads/caches the dataset; do not include that transfer in
inference timing.

For plain Monte Carlo, measured analytical FLOPs were approximately

```text
F(n_samples) = 4,231,424 * n_samples + 32,768
```

The FLOP-only budget permits about 64,280 rows, but residual charging lowers the
safe limit. On this strict one-thread run, roughly 52,000 rows approached the
full effective budget and roughly 5,200 rows approached the 10% multiplier
floor. Always verify with the per-MLP profile rather than hard-coding either
number.

## 9. Immediate next actions

1. Cache the Phase 1 Mini dataset and record three one-thread runs of the mean
   and covariance examples.
2. Create a separate experimental estimator rather than overwriting the zero
   smoke test immediately.
3. Implement randomized QMC with an exact layer-1 row and radial conditioning.
4. Establish paired development/validation/lockbox panels before tuning pruning
   or learned constants.
5. Add dead/on/kink pilot routing one lever at a time, measuring adjusted score
   and worst-MLP budget after every change.
6. Only then compare a mid-network control and a Kerdock carrier against the
   same protected panels.

## 10. Core references

- [Official challenge](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026)
- [Phase 1 launch: shape, budget, and datasets](https://discourse.aicrowd.com/t/phase-1-launch-deeper-models-and-increased-prizes/18026)
- [Phase 1 v0.10.0 compute and private-re-evaluation update](https://discourse.aicrowd.com/t/phase-1-update-flopscope-v0-10-0-cost-model-fixes-residual-time-safeguards-and-updated-deadlines/18125)
- [Phase 1 public dataset](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/tree/v1-phase1)
- [Official starter-kit Phase 2 restatement commit](https://github.com/AIcrowd/whest-starterkit/commit/ce4b84ad5aeb01cff71255248843d8522e0c4821)
- [ARC companion paper](https://arxiv.org/abs/2605.05179)
- [Companion cumulant-propagation code](https://github.com/alignment-research-center/mlp_cumulant_propagation)
- [Published Phase 1 technique census](https://discourse.aicrowd.com/t/a-technique-census-of-phase-1-what-the-mathematics-is-doing-what-walls-it-hit-and-whats-still-open/18157)
- [Expanded Phase 1 field guide](https://thylinao1.github.io/projects/white-box-estimation.html)
