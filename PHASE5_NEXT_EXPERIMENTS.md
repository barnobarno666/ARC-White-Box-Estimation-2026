# Phase 4 continuation: repair the evidence, then pursue a different estimator

Prepared 2026-09-05. This is the next research runbook, not an experiment report.
No candidate was implemented or benchmarked while preparing it. No goal agent
was started. The intended executor is the user's other coding agent.

## 1. Mission and recommended goal

**Pursue an adjusted score below `1.00e-8` on the same eight Phase 2 MLPs, with
all verification gates below.** Treat `<8.00e-8` and `<4.00e-8` as intermediate
milestones. Treat `<4.00e-9` as the frontier stretch target. Do not stop at a
fractional improvement around `1.2e-7`, or equate a milestone with the final goal.

The main research bet is **a better control center plus a correctly transported
sampling residual**, with **early-layer sample repair** and **a cheap surrogate
with an independently sampled correction** as separate routes. Give those
routes real implementations before returning to scalar calibration or carrier
weight sweeps. None is promised to reach the target.

There is a more immediate problem than a lack of ideas: several local scripts
do not test the mechanisms their reports say they tested. Consequently, the
claims that controls, low-rank structure, and moment resets are universally
exhausted are not established. Repairing those tests can change the research
direction materially; merely repeating them cannot.

Suggested text for the other agent's goal:

> Execute PHASE4_NEXT_EXPERIMENTS.md in order. Seek a legal estimator with
> adjusted final-layer score below 1e-8 on the fixed eight v2-phase2 MLPs.
> Verify on at most those eight distinct MLPs, with paired sampler replicates
> and grouped fitting where required. Preserve the original estimator and
> reports. Repair invalid diagnostics before using their rejection gates.
> Complete the specified mechanism screens and justified extensions; do not
> replace them with nearby scalar sweeps. Save every result and exact artifact.
> Do not submit to AIcrowd. Report a partial outcome honestly if the experiment
> queue is exhausted without reaching the target.

This document supplies a goal specification; it does not authorize creating a
separate Codex task, scheduling an automation, or starting a goal by itself.

## 2. What the evidence currently supports

### Local reference points

Working directory for all commands and candidate paths:

`D:\ALL CODES\AICROWD COMPETITION\whest-starterkit`

| Item | Evidence available at planning time |
|---|---|
| Immutable control | `estimator.py` |
| SHA-256, checked directly | `ea8be8222b827607f65d166118500b55fcaae883b79f64b29785b46d86501ca1` |
| Stored eight-MLP adjusted score | `1.22364251353702e-7` |
| Stored raw final-layer MSE | `1.22364251353702e-6` |
| Stored mean utilization / multiplier | `0.0980522865670537` / `0.1000` |
| Stored failures | `0` |
| Parameters | dual-kernel lambda `0.20`, terminal scale `0.998319`, MC weight `0.110`, total sample rows `4200` |
| Lowest reported stable variant | `candidates/estimator_p5_chosaul_lam05.py`, reported `1.215291e-7`; not promoted |
| Versions installed, checked directly | `whestbench 0.16.1`, `flopscope 0.12.1`, NumPy `2.2.6`, SciPy `1.15.3` |

The control score above was read from `scripts/champion_8mlp.json`; it was not
re-measured for this plan. Residual times vary between recorded runs, roughly
`0.17–0.23 s`. Reproduce them once at execution start. Earlier prose contains
different submission IDs; this plan does not infer which is the user's current
official submission.

### The actual competitive gap

The live [Phase 2 leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards)
was inspected on 2026-09-05. Its displayed leading rows were:

| Participant | Adjusted score | Raw final MSE | Mean compute utilization |
|---|---:|---:|---:|
| Puffi | `3.5e-9` | `2.23e-8` | `15.87%` |
| J2W | `3.6e-9` | `1.94e-8` | `18.45%` |
| marius_binner | `3.7e-9` | `1.85e-8` | `20.12%` |

These are rounded public-board values, not evidence of their algorithms or
private-set performance. The local control is roughly **35 times** the leading
displayed adjusted score and **55 times** its raw error, on different panels.
That is a scale reference, not a paired benchmark. A 1% improvement is useful
bookkeeping but does not substantially close this gap.

### Reading map and historical disposition

The planning audit used the Phase 4 plan and entire experiment record, including
its subsequently appended `P5-01` through `P5-06` sections. It also reviewed the
Run 1/2/3 summaries and prior strategy/phase plans, and inspected the scripts
behind the most consequential negative conclusions.

| Local source, relative to workspace root | What to retain for this continuation |
|---|---|
| `sumaary_goal_1.md`, `whest-starterkit/PHASE2_EXPLORATION.md` | Covariance correction and scale calibration helped; the first Jacobian and Hadamard blends did not. Dynamic compaction was largely deferred for engineering reasons. |
| `phase 2 summary run 2.md`, `whest-starterkit/PHASE2_NEXT_RESEARCH_PLAN.md` | Fixed prior scale beat noisy MC scale estimation. Final preactivation controls reduced sampling error, but their biased analytic centers limited fusion. |
| `phase 2 summary run 3.md`, `run 3 plan.md`, `whest-starterkit/PHASE3_RESEARCH_PLAN.md` | Dual-kernel blending gave a small gain. Preserve the deployed artifact, but re-audit the ridge and moment-reset conclusions. |
| `whest-starterkit/PHASE4_RESEARCH_PLAN.md` | Intended broader angular/response/cumulant program; implementation did not fully realize every listed mechanism. |
| `PHASE4_RECORD.md` | Preserve all reported numeric results as historical measurements, subject to script and cost audit; supersede its universal impossibility claims. |
| `Current thing.md`, `whest-starterkit/Current thing.md` | Historical result tables; do not use prose labels as a promotion engine. |
| `whest-starterkit/PHASE1_STRATEGIES.md` | Mechanism ideas and references. Phase 1 scores, dimensions, costs, and empirical ceilings do not transfer automatically. |

Keep the following low priority: global scale/alpha/lambda refinement,
regime-specific terminal calibration, another raw WMC count near 4200,
standalone damped cubic/quartic kernel additions, and blends of nearly identical
analytic predictions. They have already received substantial effort.

## 3. Correct these interpretation and implementation errors first

These findings are from source inspection, not newly measured improvements.
Copy and repair research scripts under a new prefix; retain old files and logs.

| Finding | Direct local evidence | Correct consequence |
|---|---|---|
| The Phase 4 mid-network “oracle” uses only `beta * I`. | `scripts/diagnose_mid_network_cv.py:102–120` concatenates same-index errors across different layers and fits one scalar. The docstring promises more transports than the body implements. | This tests an identity-coordinate control. Random weights mix coordinates. Near-zero correlation here does not bound a mapped multivariate control. |
| The earlier ridge test compares different layers. | `scripts/test_lane_b1_ridge_cv.py:49–50` stores `act` before weight `k_layer`; line 85 stores the analytic mean after that weight. | `H_k` is post-layer `k-1`, whereas `c_k` is post-layer `k`. Correct the indexing before attributing its blow-up to analytic bias. The current script also imports a removed estimator helper. |
| The residual spectrum was projected onto the wrong side of the last weight. | `scripts/diagnose_residual_structure.py:47–52` uses `U` from `W15=U S V^T`. Forward convention is row vectors `H @ W15`. | Output coordinates correspond to `V`, not `U`. Even a corrected projection onto `V` tests only that chosen basis; it cannot prove universal isotropy or uncompressibility. Include output gates. |
| The “perfect” moment reset used 16,384 sampled rows. | `scripts/test_lane_b2_oracle_reset.py:17,53` estimates a full 1024-dimensional covariance and mean from that batch. | It is a noisy reset diagnostic, not a true-moment oracle. Its reset branch also uses a different kernel and per-MLP target-fitted scale. Separate these effects. |
| The claimed 1.9% late-jump ceiling is a ratio of errors in two coordinate systems. | `scripts/diag_p5_layerwise.py:60` prints `MSE(L14)/MSE(L15)`. It never injects an accurate L14 state and evaluates the resulting output. | Similar adjacent-layer MSEs do not measure causal headroom. Propagate the same controlled state through the actual remaining weights. |
| The symmetric split-center correction nearly cancels by construction. | `scripts/diag_p5_unbiased.py:55–62` swaps the two sample means and averages both corrections. | With equal coefficients, `0.5[(YA-b(ZA-ZB))+(YB-b(ZB-ZA))]` is exactly the raw mean. Small gains/losses measure coefficient noise, not all unbiased controls. |
| The reported Stein test subtracts centered sample features whose means are zero already. | `scripts/diag_p5_unbiased.py:111–112` centers each block separately before taking its feature mean. | This is an algebraic near-no-op. Moreover an odd linear input control vanishes under exact antipodes; it is not a test of nonlinear even controls. |
| The active-input test has orientation and sample-consistency problems. | The same file uses right singular vectors as input directions and returns unwhitened `x` alongside activations evaluated at whitened `x`. | Correct input directions are left singular vectors for a row-input map. Always return the actual evaluated inputs. Test even/nonlinear information, not only linear R-squared on two networks. |
| The endpoint bank is a small random Hadamard bank. | `scripts/diagnose_endpoint_oracles.py:49–59` constructs signed Sylvester frames. No inter-basis MUB certificate is checked. | Within-basis orthogonality is not mutual unbiasedness between bases. K=4/8 global scalar basis weights have only 3/7 contrast degrees of freedom. A weak oracle here does not close pointwise or output-aware correction. |
| The all-depth exact Gaussian kernel was not successfully tested. | `PHASE4_RECORD.md`, P5-03, explicitly records a wrong mean-modulation formula and blow-up. | Correct it once using Section 8. A failed formula and layer-0 gain do not prove a full-depth cost/accuracy bound. |
| The helper can label success without all gates. | `scripts/run_eval.py` checks stretch success before residual/worst-case/salt gates and can overwrite the champion record before comparison. | Use one conjunction of all mandatory gates; keep an immutable control receipt and match rows by identity, never list position alone. |
| Crossing 10% is being treated like failure. | P5-06 and the final Phase 4 synthesis conflate a multiplier boundary with a hard cap. | The multiplier is continuous at 10%. `10.01%` costs only 0.1% more than the floor; it is not a validity failure. The hard FLOP cap is 100%. |

Two further scientific corrections:

- Eight tested MLPs do not establish that `s0` is universal or minimax-optimal.
  They support keeping it as a strong fixed local control parameter.
- Under Gaussian input, fixed-radius spherical integration is exact for a
  positively homogeneous network when directions have the correct spherical
  measure. Normalizing a finite whitened cloud is a different construction;
  one unsuccessful version does not invalidate the identity. Likewise WMC
  moment matching does not automatically give unbiased Gaussian marginals.

## 4. Optimize scored error, with an explicit compute frontier

For every network use its own raw error `e_m` and measured utilization `u_m`:

`S = mean_m[e_m * max(0.10, u_m)]`.

Do not replace this by `mean(e) * max(0.10, mean(u))`. The latter can mis-rank
adaptive implementations. A failed network uses the harness's fallback and
penalty; do not silently omit it or call it a score of zero. See the
[official scoring model](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/concepts/scoring-model.md).

For an approximately fixed multiplier, these are useful design targets:

| Utilization / multiplier | Raw MSE needed for `8e-8` | For `1e-8` | For `4e-9` |
|---:|---:|---:|---:|
| `<=10% / 0.10` | `8.00e-7` | `1.00e-7` | `4.00e-8` |
| `15% / 0.15` | `5.33e-7` | `6.67e-8` | `2.67e-8` |
| `20% / 0.20` | `4.00e-7` | `5.00e-8` | `2.00e-8` |
| `30% / 0.30` | `2.67e-7` | `3.33e-8` | `1.33e-8` |
| `40% / 0.40` | `2.00e-7` | `2.50e-8` | `1.00e-8` |

Screen serious new architectures near utilization `0.098, 0.15, 0.20, 0.30,
0.40`, selecting only three operating points initially. Extend to `0.60` only
after a measured favorable curve. Leave a conservative margin to the true
budget and time caps. Cheap methods can remain well below 10%; they do not need
padding compute to reach the floor.

For a pure fixed-cost sampler with variance `V/N`, increasing `N` above the floor
eventually leaves `S` approximately constant. This explains why brute sampling
is unattractive. It does **not** constrain a method that changes `V`, reduces
the cost per informative sample, removes bias, or estimates only a residual.

At a doubled multiplier, halving raw MSE merely breaks even. For promotion,
require the actual scored gain after paying for pilot samples, response fitting,
all centers, every branch, and production numerical operations. Offline oracle
scores must be labelled projected; multiplying by `0.1` is not cost measurement.

The [official ground-truth documentation](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/concepts/ground-truth.md)
puts the Phase 2 Mini reference-noise floor around `7.5e-11` raw. Thus the
present gap is not plausibly explained by the label floor. Use the cached
official references, not a small new MC bake, for candidate ranking.

## 5. Eight-MLP protocol and execution contract

### Fixed panel, including final verification

Dataset: `D:\ALL CODES\AICROWD COMPETITION\datasets\mini`.

Use exactly the existing eight networks:

1. `logan-fitzgerald`
2. `william-graves`
3. `raymond-barnes`
4. `steven-rice`
5. `sarah-kelley`
6. `christopher-morales`
7. `cheryl-graham`
8. `renee-park`

**No 32-, 64-, or 100-MLP confirmation. No hidden expansion of the network panel
through an offline oracle or learned-model training job.** More sample draws,
sampler salts, parameter settings, and mathematical unit tests are permitted;
they are not more distinct verification MLPs. Tiny synthetic algebra fixtures
are not competition evidence. Do not generate additional full-size networks
unless the user changes this constraint.

Construct `MLP` using `MLP.from_row` and `resolve_seed_context(ds)`. Pin dataset
revision `v2-phase2`, row identities, weight hashes, reference hashes, protocol
and salt. Do not substitute the raw `mlp_seed` column for the derived seed.
Preserve the distinction between `ctx.seed` and `mlp.seed`; changing a CLI seed
with a dataset does not automatically create a new per-network sampler draw.

Use this schedule:

- One/two existing panel MLPs for shape, math, and catastrophic-cost smoke only.
- All eight for any statistical keep/reject conclusion.
- Initial full screen at sampler offset `0`.
- Promising candidate: offsets `0, 1337, 8888`, each paired with the control
  using its corresponding offset. Keep the original zero-offset receipt too.
- Finalist only: frozen extra offsets `104729, 130363, 155921, 196613, 262147`.
  Do not tune to these extra offsets; report their result even if it reverses
  the gain. Deterministic candidates need no redundant identical reruns.

Implement the offset through an explicit sampler helper such as
`(int(mlp.seed) + offset) % 2**32`; record the resulting seed. Use distinct,
recorded stream roles for pilot, fit, calibration, and evaluation. Independence
claims require genuinely independent batches, not slices of one jointly
whitened batch. Keep antipodal pairs in the same block.

### Fitting and repeated-search discipline

Fitting a response from sampled `(H_k(X), f(X))` pairs of the current network is
legitimate white-box numerical work; it uses no reference mean. Its compute
must be included at inference. Fitting a shared rule to reference errors is a
different operation and requires grouped leave-one-MLP-out (LOO) evaluation.

For a reference-fitted rule:

- Each outer fold holds out one entire MLP: all its layers, neurons, pairs,
  samples, salts, moment files, and derived labels.
- If tuning a hyperparameter, select it using inner grouped folds within the
  remaining seven MLPs. Fit preprocessing only on those training folds.
- Limit the first shared model to at most 12 interpretable coefficients, at
  most three depth groups, and no network-ID or seed features.
- Report outer-LOO predictions separately from all-eight refit predictions.
  The refit is the deployment artifact, not independent validation.
- Preserve candidate count, the original fixed grid, and every failed setting.
  All eight networks have already been repeatedly inspected; relabelling two
  as “fresh holdout” does not undo that exposure.

Many fast runs reduce implementation uncertainty and sampler noise. They do
not turn eight MLPs into a large sample of networks. Use “eight-panel research
result,” not “proven generalization,” even after salt and LOO checks.

### File and runtime rules

- Use `uv`; do not create a separate `venv` or use bare `pip`.
- Freeze copies under `research/phase4_next/control/`, with hashes. Do not
  overwrite `baseline_estimator.py`, `estimator.py`, or historical reports.
- New scripts: `scripts/p4n_*.py`; candidates: `candidates/estimator_p4n_*.py`.
  Suffix the experiment ID and frozen configuration. Do not edit a candidate
  after recording its score; create a new revision instead.
- Use offline NumPy/SciPy for diagnostics only. Submitted numerical work must
  go through flopscope, including fitting, transforms, selectors and loaded
  learned corrections. Do not port Phase 1 bundled execution backends.
- The runtime caps are `2**41` FLOPs, 120 s predict wall time, 0.4 s residual,
  5 s setup, 8 GB process memory. Finalist targets: residual `<0.30 s` preferred
  and always `<0.35 s`, wall `<90 s`, setup `<4 s`, peak memory `<6 GB`.
- A shared-data artifact is allowed by the
  [Phase 2 code policy](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/concepts/allowed-code.md).
  Ship only generic coefficients, geometry, or quadrature data; no panel truth
  tables, weight hashes, per-network oracle weights, or reference moments.
- Do not treat scalar extraction from pending flopscope values as free.
  Batch diagnostics; keep neuron arithmetic and gate masks metered.
- Validate float32 production calculations, especially subtractive residuals
  and covariance centering. Use higher precision selectively if its measured
  value exceeds its compute cost.
- Git metadata was absent in this checkout and its workspace ancestors at
  planning time. If a real GitHub-backed repo exists when editing, follow the
  user's instruction to commit completed changes and push to `main`; do not
  invent a remote, initialize a replacement repo, or stage unrelated files.

Useful existing commands, run from the starter-kit directory:

```powershell
$env:PYTHONUTF8 = '1'
uv run --no-sync python -c "import importlib.metadata as m; print({k:m.version(k) for k in ['whestbench','flopscope']})"
uv run whest validate --estimator estimator.py
uv run whest run --estimator estimator.py --dataset 'D:\ALL CODES\AICROWD COMPETITION\datasets\mini' --split mini --n-mlps 8 --runner local --format json
```

Save stdout and stderr separately through the new wrapper and preserve the
complete parsed report. Every future `whest run` command must explicitly include
the cached dataset and `--n-mlps 8` or a smaller smoke count. Some old scripts
import removed helpers; fix their imports against an explicitly frozen version.

## 6. Ordered experiment queue

Execute these in the listed order. Optional rows are conditional on a stated
gate, not on the agent's preference for familiar code. Core budget estimates
below count configurations, not unique networks. Saved trajectories can serve
many offline diagnostics without rerunning a network forward pass.

| ID | Mechanism / task | Initial scope | Required outcome before moving on |
|---|---|---|---|
| P4N-00 | Repair instrument and causal diagnostics | 1 control replay; algebra fixtures | Correct seeds, dimensions, index conventions, gates and receipts |
| P4N-01 | Mapped control oracle and corrected ridge | 9 depths with cheap transport; rank expansion at best 3 | Locate response headroom and required center accuracy |
| P4N-02 | Correct nonzero-mean Gaussian covariance | 4/8/16 quadrature nodes; at most 2 damping variants | Measure a valid exact-Gaussian branch and its cost |
| P4N-03 | Accurate moment interventions | Matched moments for the same eight; selected depths | Separate center error, Gaussian readout error, covariance error |
| P4N-04 | Early-layer sample repair | 3 repair operators × 3 strengths, then best 2 count ladders | Test an analytic target inside the sampler |
| P4N-05 | Exact-mean nonlinear controls | First-layer controls + thresholded ridge features | Obtain a control without an approximate center |
| P4N-06 | Response-weighted control-center improvement | At most 12 shared parameters / 6 initial variants | Convert P4N-01/03 headroom to deployable score |
| P4N-07 | Cheap surrogate + coupled residual | 6 surrogate designs, then 3 budget allocations | Change sample cost without accepting surrogate bias |
| P4N-08 | Conditional Gaussian mixture propagation | 1–2 input directions; at most 12 settings | Preserve conditional structure beyond one Gaussian |
| P4N-09 | Nontrivial even Stein controls | 2 feature bases × 3 threshold sets | Test controls not annihilated by the carrier |
| P4N-10 | Certified carrier and signed-calibration extension | At most 8 initial settings | Test what the random K=4/8 bank did not test |
| P4N-11 | One-dimensional conditional integration | 2 directions × 3 quadrature sizes | Test a structurally different integration path |
| P4N-12 | Local compressed cumulant/response state | Oracle-gated; at most 9 initial variants | Carry only contractions shown useful in P4N-03 |
| P4N-13 | Complementary combinations | At most 3 branch pairs, 3 budgets each | Monetize decorrelated improvements at full cost |
| P4N-14 | Execution optimization | Winning architecture only | Recover score through actual arithmetic reduction |
| P4N-15 | Frozen final verification and handoff | Same eight only | Full receipts, package, limitations, next unresolved mechanism |

Expect roughly **100–180 meaningful first-pass configurations/diagnostics** if
every lane is reached, with fewer if valid gates eliminate branches. The user's
sub-minute panel runs justify this breadth. Porting, solving and oracle I/O can
take longer; do not promise that the whole campaign will finish in 180 minutes.
The counts are initial grid bounds, not an instruction to waste runs after a
mechanism has been falsified.

After any substantive improvement, verify it promptly and save it as an interim
candidate, then resume the remaining independent lanes. Give at least half the
post-audit effort to methods that change the statistical estimator, rather than
coefficient refinement. A strong oracle permits up to three genuinely different
implementations, each followed by its specified extension. Additional work must
state the new mechanism and expected gain before running.

## 7. P4N-00/01: put controls in the correct coordinates

### Index and algebra fixtures

Use one convention everywhere:

`H[-1] = X; Z[l] = H[l-1] @ W[l]; H[l] = relu(Z[l])`, for `l=0..15`.

Cache after the indicated ReLU. Verify that `mean(Z[l])` equals
`mean(H[l-1]) @ W[l]` up to accumulation tolerance. Test the mapping on a small
linear suffix where the exact response is known. Permuting hidden neurons and
the matching rows/columns of adjacent weights must preserve the final output.
These tests would have caught the existing coordinate bugs.

Also test synthetic cancellation of the old split-CV formula and the centered
Stein features; include their zero-control output as a negative control. Do not
rerun the old ineffective estimators on eight MLPs to rediscover their algebra.

### Response-aware oracle

Let `F(X)=H[15](X)`. The row-vector control is

`mu_hat = mean(F) - (mean(H[k]) - center[k]) @ B[k]`.

The dimensions are `B[k]: 1024 × 1024`, or a factored equivalent. `B[k]` maps
layer-k coordinates into final-output coordinates. It is not `beta * I`.

Start with independently generated antithetic Gaussian pilot/evaluation
batches: 2048 pilot rows and 4096 evaluation rows. These are totals including
both signs. All pilot and fit cost belongs to the eventual estimator.

1. For `k={0,1,3,5,7,9,11,13,14}`, use a cheap weight-aware response:
   `B = W[k+1] D[k+1] ... W[15] D[15]`, where each `D` is the diagonal vector
   of pilot gate probabilities. Apply products to the discrepancy without
   materializing a full matrix when possible.
2. At each k compare three centers, with the SAME response and evaluation draws:
   - exact first-layer center or official `all_layer_means[k]` (oracle);
   - the unscaled analytic layer-k center;
   - a properly depth-calibrated/LOO center, if already available.
   Never apply the terminal `s0` blindly to every layer.
3. Compare raw sampling, corrected sampling, and their lawful fusion with the
   control analytic branch. A target-optimized blend is an oracle only.
4. Select the best three depths by cost-adjusted oracle headroom, not by
   same-index correlation. At those depths fit multivariate responses from
   pilot activation/output pairs.

Fit the response to **antipodal pair averages** of H and F if evaluation uses
pair averages. This targets the variance that survives antipodal cancellation.
Use centered pilot features `Z` and targets `Y`; find a rank-r feature basis V,
then fit

`B = V @ solve((Z@V).T@(Z@V) + lambda_eff*I, (Z@V).T@Y)`.

Use `lambda_eff = lambda * trace(Gram)/r`, with lambda initially `1e-2`.
Initial ranks: `16, 64, 256`. Expand the best depth to `512, 1024` if the oracle
improves consistently with rank. For the latter ranks, increase the independent
pilot to 4096 rows (2048 pairs); do not call an underdetermined interpolation
fit a reliable response. Standardize features using pilot statistics.

At rank selection, compare variance PCs with an output-oriented subspace
obtained from the pilot feature/output cross-covariance. The earlier projection
onto a few arbitrary weight singular vectors is not an adequate substitute.
Only the winning rank gets lambda extensions `1e-3, 1e-1`.

Report:

- held-evaluation residual variance after mapped control;
- `||(center - true_mean) @ B||² / 1024`, the center error actually consumed;
- standalone and fused raw error, per-network cost and adjusted error;
- pilot-fit versus evaluation error; response norm and conditioning;
- oracle versus deployable headroom, without mixing their labels.

Use corrected output-space singular vectors `V` and gate-weighted response
subspaces for residual diagnostics. Do not pool raw neuron indices across
unrelated networks as if their coordinate systems were aligned.

**Continue gate:** a fitted/structured response with oracle centering cuts raw
sampler error by at least 30% on 6/8 networks, or the costed oracle fusion beats
the control by at least 20%. A smaller gain may survive as a cheap component if
P4N-13 has at least 10% headroom. If no response passes, archive only these
tested maps/depths; still run the independent early-repair and surrogate lanes.

For the winning response, sweep controlled center contamination:

`center(t) = true_mean + t*(analytic_mean - true_mean)`,
`t = {0, 0.125, 0.25, 0.5, 1}`.

This is a diagnostic only. It quantifies how much centering accuracy is needed
before spending time on a learned or cumulant center. If success requires an
unrealistic 100-fold center improvement, redirect effort to exact controls or
an independently sampled cheap center.

## 8. P4N-02: a correct full Gaussian covariance baseline

The following identity is derived for a jointly Gaussian pair; it is not a
claim that deep MLP preactivations are Gaussian. It provides a correct
replacement for the failed all-depth Cho-Saul generalization.

Let means be `mu_i, mu_j`, standard deviations `s_i,s_j`, standardized means
`a_i=mu_i/s_i`, `a_j=mu_j/s_j`, and correlation rho. Then

```text
Cov(relu(Z_i), relu(Z_j)) / (s_i*s_j)
  = rho*Phi(a_i)*Phi(a_j)
    + rho² * integral_0^1 (1-u) * phi2(a_i,a_j; rho*u) du

phi2(a,b;t)
  = exp(-(a² - 2*t*a*b + b²)/(2*(1-t²)))
    / (2*pi*sqrt(1-t²))
```

One way to check the identity: differentiating covariance with respect to rho
gives the joint gate probability; its derivative is the bivariate Gaussian
density. Covariance is zero at rho=0. Integrating twice gives the expression,
including negative rho. No nested bivariate-CDF quadrature is required.

Planning-time numerical check: this identity agreed with a separate scalar
conditional-normal integration on 25 nonzero/zero-mean, positive/negative-
correlation cases to maximum absolute discrepancy `4.1e-15`. The skewness and
kurtosis integrals in Section 9 were also checked numerically, to `7.3e-16`.
These are formula checks only; no competition MLP was evaluated for this plan.

Implementation instructions:

1. Use Gauss-Legendre nodes on `[0,1]`, including the explicit `(1-u)` weight.
   Precompute generic nodes/weights offline. Try 4, 8, 16 nodes.
2. Handle the diagonal with the exact univariate rectified variance. Handle
   zero variances separately. For off-diagonals near `|rho|=1`, use a stable
   limiting evaluation or adaptive reference check; don't hide invalid
   correlations by indiscriminate clipping.
3. Compare with high-accuracy scalar integration on an offline grid:
   `a,b in {-5,-2,-0.5,0,0.5,2,5}` and
   `rho in {-0.99,-0.9,-0.5,-0.1,0,0.1,0.5,0.9,0.99}`.
   Check symmetry, rho=0, equal-variable variance, and zero-mean arc-cosine
   agreement. Use float64 here to distinguish algebra from production rounding.
4. In production use float32 and meter the complete chain. Do not assume the
   quadrature makes it prohibitive: its elementwise work need not dominate the
   existing dense covariance propagation.
5. Run the complete chain standalone and as a new control center. Compare raw
   unscaled outputs first. If it has useful shape accuracy, fit only a single
   scale with outer LOO before considering a blend.
6. If the exact kernel regresses, allow only covariance blends with the frozen
   kernel at strengths `0.25, 0.5`. Record PSD defects and accumulated error.

**Gate:** continue if the branch gives 10% scored gain, reduces transported
center error by 20%, or enables the P4N-03 causal diagnostic. Otherwise retain
the correct kernel as a reference and stop tuning it. Exact Gaussian pair
integration cannot, by itself, repair arbitrary higher-order dependence.

## 9. P4N-03: actual state interventions and moment-noise accounting

There is now a public Phase 2
[higher-moment dataset](https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-higher-moments-2026).
It contains independently estimated moments from `1e8` samples for the same
Mini networks, with per-network files around 322 MB. Use only the eight
matching files, one at a time; do not download its complete roughly 49 GB
collection. Source metadata and the all-layer mean must match the local MLP.

These are high-accuracy empirical moments, not mathematical truth. Keep their
sampling error separate from the official `1e9` reference error. The card
defines post-layer means and `M11`, and preactivation marginal moments and
pair moments. Its derived feature cache contains truth-derived quantities;
none is a production input.

Use `C = M11 - mean*mean.T` with consistent centers and precision. Inspect
symmetry, diagonals and smallest eigenvalue. Because second moments are stored
in float32, subtractive cancellation can create small negative eigenvalues.
Record any PSD repair and repeat key conclusions with and without it.

At the top three P4N-01 control depths, plus `k=0,1,14`, intervene separately:

1. Replace only the analytic mean with the official reference mean.
2. Replace only covariance with the independent high-accuracy covariance.
3. Replace both with the matched high-accuracy mean/covariance.
4. Supply high-accuracy final preactivation mean/variance directly to the
   Gaussian ReLU readout.
5. Add high-accuracy preactivation skewness/kurtosis to that readout.
6. For controls, hold B fixed and compare the accurate and deployable centers.

Use the same downstream recurrence in each paired test. Report raw/unscaled
results first; any target-fitted scaling is a separate oracle. Do not compare
different layers' scalar MSEs as a causal intervention.

For standardized skewness gamma3 and excess kurtosis gamma4, the first
Edgeworth mean correction is

`delta_mean = s*phi(a)*[-gamma3*a/6 + gamma4*(a²-1)/24]`.

This sign and polynomial distinction matters: the old residual script's
docstring gives the kurtosis polynomial for skewness. Unit-check this expression
by integrating simple perturbed one-dimensional Gaussian densities. This is an
asymptotic approximation, so use the high-accuracy-moment result as a diagnostic,
not as a guarantee that recursive truncated cumulants will work.

Branch decisions:

- **Mean replacement helps:** prioritize P4N-06 centering and P4N-04 repair.
- **Covariance replacement helps:** prioritize projected covariance/response
  state in P4N-12; a marginal scale curve is the wrong target.
- **Accurate moments still give a poor readout:** prioritize non-Gaussian
  conditional structure, actual continuation, or the surrogate residual.
- **Only high-order accurate moments help:** determine which contractions are
  needed and whether a legal estimator can estimate them with enough precision.
- **Interventions regress:** inspect error cancellation and projected error;
  do not conclude “truth is harmful” as a universal property.

If moment files cannot be obtained or matched, mark the affected oracle
inconclusive. Continue label-only mean interventions and independent lanes.
Do not rebake `1e8` samples locally or replace that oracle with 16k samples.

## 10. P4N-04: use exact early moments inside sampled trajectories

This lane moves analytic information into the samples before the long suffix,
instead of relying on an inaccurate analytic final prediction.

For Gaussian input, the exact first post-ReLU mean is

`m0[j] = norm(W0[:,j]) / sqrt(2*pi)`.

The complete first-layer covariance is available from the zero-mean arc-cosine
kernel. These are weight-derived values, not fitted reference labels.

Start at post-layer `0` only, with unchanged actual-network continuation after
the repair. For each operator try strength `t={0.25,0.5,1}`:

1. **Positive mean matching:**
   `H0_repaired = H0 * [(1-t) + t*m0/max(sample_mean(H0),epsilon)]`.
   Use conservative handling of tiny columns; cap the ratio in `[0.8,1.2]`
   initially, and record which columns hit the cap.
2. **Additive mean matching diagnostic:**
   `H0_repaired = H0 + t*(m0-sample_mean(H0))`.
   This can make intermediate values negative. Label it as a distribution
   approximation, not exact ReLU samples. Compare against the positive version;
   do not silently clip and claim the mean stayed matched.
3. **Mean plus covariance transport:** center H0; map empirical covariance to
   the exact C0 using symmetric square roots/pseudoinverse on supported modes;
   then add m0. Interpolate between identity and this map using t.
   First try ranks `32,128`; full covariance only if the rank trend helps.
   Price the factorization and every multiplication.

Each operator is a **biased sample transformation**, unless separately
corrected. Keeping sample moments exact does not keep the full distribution
exact. Use counts `2048,8192,16384` on the best two repairs to expose its bias
trend. Pair samples with the unrepaired path and save the difference.

Next, only if layer-0 repair passes: test post-layer `1` using the P4N-02
analytic mean at strength `0.25`, and a short schedule of layers `0,1,3` at
strengths `(1,0.25,0.125)`. Compare accurate-center diagnostic versus deployable
center to identify accumulating centering error.

Test the repaired sampler standalone, then with a newly estimated LOO-fixed
blend. Do not force the original alpha=0.110 onto a different error profile.

**Gate:** 15% adjusted improvement or at least 30% reduction in the dominant
sampling residual at matched total cost. If bias remains, use the repaired
sampler as a surrogate in P4N-07, estimating the repair error on paired actual
paths. A biased transformation is not a dead end if its error is cheaply
correctable.

## 11. P4N-05: controls with an exact, lawful mean

This is the route around the approximate-center bottleneck. Start with ordinary
antithetic Gaussian sampling, where the control expectation is known under the
actual sampling law. Only later test WMC; finite-cloud whitening can alter that
law and must not inherit an unbiasedness claim.

### A. First-layer nonlinear feature control

Use `q(X)=relu(X @ W0)` with exact vector expectation m0. Fit a response to F
using an independent pilot, as in P4N-01, and return

`mean(F_eval) - (mean(q_eval)-m0) @ B`.

Pilot choices: `2048,4096` rows; initial ranks `64,256,512`. All coefficients
are frozen before evaluating the residual. Use paired-even features with an
antithetic carrier. Include the full `1024` rank only if the rank trend and
cost justify it. This is a multivariate nonlinear control, not an odd linear
input control that antipodes annihilate.

### B. Shifted even ridge-feature surrogate

For unit input directions v and nonnegative thresholds t, define

```text
q_v,t(x) = 0.5 * [relu(v^T*x - t) + relu(-v^T*x - t)]
E[q_v,t] = phi(t) - t*Phi(-t)
```

Use directions from normalized first-weight columns or a pilot-derived input
response basis. Initial feature banks:

- 64 directions, `t={0,1}`;
- 128 directions, `t={0,1}`;
- 128 directions, `t={0,0.5,1.5}`.

Fit B with the same normalized ridge as P4N-01. Do not learn per-network
coefficients from the reference final mean. Conditional on the independent
pilot, a finite linear combination of these features has an exactly known
Gaussian mean. A surrogate `g(x)=b+q(x)B` therefore permits

`mu_hat = b + E[q]B + mean(F_eval - g(X_eval))`.

Compute the residual samplewise before averaging, with stable accumulation.
Compare error variance on held evaluation samples. R-squared of the raw output
can be misleading because the large mean is easy to predict; use centered
paired residual variance and final mean error.

**Gate:** residual variance reduction must pay for feature evaluation and fit.
Pursue a branch if the projected scored gain is at least 20%, then verify it
with the real harness. If a feature mean is constant under the chosen carrier,
remove that feature: it contributes no sample-mean correction.

## 12. P4N-06: improve the mean in the directions the control consumes

Only pursue this after P4N-01 identifies a useful response and P4N-03 identifies
the missing state. The objective is not the smallest unweighted center MSE:

`loss = mean_m ||(center_m(theta)-mu_true_m) @ B_m||² / 1024`.

This directly targets the downstream control error. Use at most 12 shared
coefficients, with the grouped nested LOO protocol. First implementations:

1. Correct P4N-02 early/mid-layer means, with one shared scale per selected
   depth group; retain unscaled layer 0 exactly.
2. Add the correct skew/kurtosis response from Section 9, using deployable
   weight-derived recurrences or independent pilot estimates. Test multipliers
   `0,0.25,0.5,1` for ONE selected correction at a time; do not pretend pilot
   cumulants equal the high-accuracy oracle cumulants.
3. Use two to four pilot-observable response features selected by P4N-03,
   projected through B. Fit a small shared linear correction on training MLPs.
   No hidden truth-derived covariance, cumulants or residuals may enter these
   features on the held-out network.

When correction changes a recursive moment trajectory, train/evaluate it on
the trajectory it actually produces. A fit to perfect intermediate states is
teacher-forced evidence only. Compare an uncorrected and rolled-forward model.

Use the control coefficient strengths `beta={0,0.25,0.5,0.75,1}` initially;
choose beta by grouped LOO, not by each evaluated target. Refit the final
analytic/sampling blend jointly rather than counting two overlapping gains.

**Gate:** the deployable rule must recover at least 25% of the measured oracle
improvement and pass the 10% scored promotion gate. If three feature mechanisms
cannot reduce projected centering error, move to exact or sampled-surrogate
centers. Do not escalate to a flexible GRU/GNN trained on eight networks.

## 13. P4N-07: cheap surrogate plus a coupled correction

This is the principal independent high-upside architecture. It changes where
compute is spent while retaining a correction for an imperfect approximation.

For a pilot-frozen cheap function g and exact network f, use independent
evaluation streams:

`mu_hat = mean_A[g(X)] + mean_B[f(X)-g(X)]`.

Use the **same X inside each difference** on stream B. Streams A and B are
independent of each other and of surrogate construction. Under ordinary
Gaussian sampling this identity removes g's integration bias, regardless of
whether g has a good analytic mean. If the P4N-05 g has an exact mean, replace
the first term by that exact mean.

Try these six initial surrogates, with a 1024-row independent construction
pilot and cached actual outputs for residual diagnostics:

1. Exact prefix through layer 7, then a pilot-gated affine suffix.
2. Exact prefix through layer 11, then a pilot-gated affine suffix.
3. Retain 512 neurons per layer from layer 4 onward; fold omitted neurons into
   a pilot mean contribution in the following affine map.
4. Same, retaining 256 neurons.
5. P4N-04's best early-repaired trajectory with a cheaper late suffix.
6. P4N-05's best shallow nonlinear feature surrogate.

For surrogate 5, estimate every repair mean, ratio and covariance transform
from the independent construction pilot, then freeze them. Do not recompute
sample-matching transforms separately on evaluation streams A and B: that would
make g depend on different batch sizes/distributions and invalidate the stated
two-level cancellation. The repaired sampler from P4N-04 and this fixed-function
surrogate are distinct ablations.

Neuron selection is based on pilot activation variance times estimated
downstream response energy, with a near-kink protection term. Start by retaining
all flagged unstable neurons within the width allowance. Mean-folding creates
biases in g; that is allowed for the surrogate because `f-g` corrects them.
Never present a pilot-stable gate as mathematically always-on/off.

For each surrogate measure the pair-averaged quantities:

- `Vg = mean coordinate variance of g`;
- `Vd = mean coordinate variance of f-g`;
- `cg` and `cd`, costs per sample/pair of g and the coupled difference;
- construction/fit cost, and residual bias caused by numerical approximation.

With independent estimators, optimize

`MSE approximately Vg/Ng + Vd/Nd`,
`cost = Cpilot + Ng*cg + Nd*cd`.

At a fixed available compute C, optimal allocation satisfies

```text
Ng / Nd = sqrt(Vg*cd / (Vd*cg))
minimum variance = (sqrt(Vg*cg)+sqrt(Vd*cd))² / C
```

Use pilot variances to set counts. Round to complete antipodal pairs, then
recompute exact cost. Initial utilization targets: `0.10,0.20,0.40`.

**Pre-port gate:** the measured variance-cost expression predicts at least
30% gain over matched-cost ordinary sampling, or a costed fusion predicts at
least 20% over the control. A surrogate that looks close pointwise but costs
almost as much as f is not useful. An early prefix that remains the dominant
cost limits the possible gain; record that limitation explicitly.

Only after a two-level success try the three-level version

`E[g0] + E[g1-g0] + E[f-g1]`,

with independent streams across levels and paired inputs within differences.
Apply the same variance-cost allocation to all terms. Do not reuse one noisy
center symmetrically in a way that collapses back to raw MC.

Bias checks must distinguish the exact identity from carrier approximations:
using WMC, heuristic spherical clouds, clipping the final estimate, or a
pilot-dependent evaluation design can alter unbiasedness. Establish the
ordinary-Gaussian version first, then compare faster variants empirically.

## 14. P4N-08: conditional mixtures rather than one global Gaussian

This is a distinct analytic hypothesis, not another terminal calibration curve.
Represent a Gaussian input as

`X = t*v + X_perp`, with `t~N(0,1)`, `X_perp~N(0,I-vv^T)`.

For each quadrature node t, run conditional moment propagation starting from
mean `t*v`, covariance `I-vv^T`, then average final predictions with the
quadrature weights. For two orthonormal directions use the corresponding
rank-two conditional covariance. The input decomposition is exact; replacing
each conditional distribution by Gaussian closure through depth is approximate.

Pick directions without target labels:

1. A normalized first-layer weight column selected for large pilot downstream
   sensitivity.
2. The leading input mode of a gate-aware input/output response. For a row
   input map use its LEFT singular vectors. Include a random-direction control.

Initial settings: one direction with 3/5/9 Gauss-Hermite nodes; the best two
directions with a 3×3 rule. Normalize nodes for standard-normal expectation:
physicists' Hermite nodes need the sqrt(2) conversion and weights / sqrt(pi).
Validate moments of t before running an MLP.

Use the correct nonzero-mean pair covariance of P4N-02, not the broken
mean-modulated zero-mean kernel. Compare with the unconditioned branch and its
costed LOO blend. At most one extension: start conditional splitting at an
early layer and keep between-component covariance, clearly labelled approximate.

**Gate:** 15% scored gain or a demonstrable 30% reduction in transported center
error. If components merely reproduce the original closure at extra cost,
archive. Do not enumerate a tensor grid in four or more latent dimensions.

## 15. P4N-09: test an actual nonlinear Stein control

For `X~N(0,I)`, unit v, `s=v^T X`, and a suitable scalar function psi:

`q(X) = s*psi(s) - psi'(s)` has expectation zero.

Take `psi(s)=relu(s-t)`; the weak derivative is `1{s>t}` almost everywhere:

`q_t(X) = s*relu(s-t) - 1{s>t}`.

Average q over each antipodal pair before fitting. At t=0 this reduces to a
quadratic control under pairing and is annihilated by exact covariance matching.
It is a **negative control**, not the main experiment. Use thresholds
`{0.5,1}`, `{1,2}`, and `{0.5,1,2}` with 32 and 64 directions drawn from the
P4N-05 banks. Fit a low-rank pilot response using the same independent-batch
protocol, and subtract the control mean (zero), not the other half's noisy mean.

Measure variance of the pair-averaged q before doing a full score run. If it is
numerically zero under the chosen carrier, the control cannot help. First use
ordinary Gaussian sampling; Stein's Gaussian identity must not be asserted
unchanged for a whitened or fixed-radius cloud.

**Gate:** at least 20% cost-adjusted residual variance reduction on eight MLPs
or at least 10% costed blend headroom. This is a bounded exploratory lane; even
valid nonlinear controls may carry little information about the deep residual.

## 16. P4N-10: a limited but valid carrier/selection revisit

Do not repeat the old “Hadamard beats WMC by 15%” tournament. Test the specific
missing structures, and apply them to the best residual integrand if available.

### Geometry and scaling

- A basis contains `1024` line representatives; including antipodes gives
  `2048` actual network evaluations. Use these terms consistently.
- Verify `Q Q^T = I` for each normalized frame. To call a bank mutually
  unbiased, verify `abs(Qa Qb^T)=1/sqrt(1024)=1/32` for distinct bases.
- Row permutation of a complete frame only reorders sample points. It does not
  create a new direction set. Sign changes in input coordinates can change it.
- The complete real-MUB construction in dimension 1024 has `513` bases,
  `525312` lines and `1050624` signed points. It is not an 8k/16k bank, and
  the older local count conflated directions and antipodes. Do not build/run
  the full bank under dense evaluation.

Use a verified partial MUB construction for K=`2,4,8` if an appropriate
1024-dimensional generator can be validated. The
[Nygaard release](https://github.com/SkyeNygaard/ARC-Whitebox)
is a Phase 1 reference, not a ready-made Phase 2 solution or a certification of
random signed-Hadamard frames. Start with small-dimension geometry tests.
Generic tables can be generated offline and loaded as data; runtime numerical
generation must be metered.

Compare matched-cost alternatives once: independently rotated orthogonal
frames, properly scrambled Sobol normals, and the existing signed-Hadamard
bank. Do not label a single arbitrary Korobov generator as a test of all QMC.
Use full valid point-set sizes and audit marginal/Gram/fourth-moment errors.

### Signed calibration from exact moment constraints

If P4N-05 exact-mean features have predictive signal, test minimum-deviation
sample/block weights subject to their known expectations and mass one. Use
32/64/128 features, regularization `1e-2` first, signed weight squared-norm
inflation capped at 2× uniform. Test a positivity-constrained variant only once.

This is the dual of a regression control for suitable choices; do not count
algebraically equivalent weighting and CV formulas as independent discoveries.
Use an independent pilot/frozen rule; if evaluation-sample calibration makes
the estimator nonlinear, label and measure its finite-sample bias.

Oracles must report their degrees of freedom. An unconstrained point-level
target fit with enough weights can interpolate 1024 outputs and is a useless
ceiling. Require a bounded weight norm, fixed feature subspace and robustness
on independently regenerated endpoint banks before funding a selector.

**Gate:** 20% variance reduction at matched cost or 15% costed blend headroom.
If neither geometry nor exact-feature calibration helps, archive after this
bounded revisit. Do not spend most of Phase 4 selecting among eight random
basis means.

## 17. P4N-11: condition on one input direction and integrate it

This is a lower-probability, genuinely different integration experiment.
For a unit direction v, sample `xi~N(0,I-vv^T)` and estimate

`mu = E_xi [ E_t f(xi+t*v) ]`, with `t~N(0,1)`.

Select v from the target-free input-response basis above; compare a random v.
First use Gauss-Hermite inner rules of `8,16,32` nodes with 256 independent
outer draws. Compare with ordinary sampling at the same total cost. Include
the node cost and any construction/fitting cost. Quadrature is deterministic
approximation, not an unbiased integral; nested-rule disagreement estimates
resolution, not proof of error against the true conditional integral.

Use a small one-hidden-layer network to verify the rule against exact
one-dimensional piecewise-linear integration. Only if Phase 2 shows useful
variance reduction, implement adaptive piecewise-linear integration along the
chosen line. A line through a deep ReLU net can cross many activation regions;
cap the initial offline prototype at 256 segments and mark unresolved lines
inconclusive or fall back to a defined quadrature. Do not silently truncate
regions and claim exactness.

**Gate:** 30% lower cost-adjusted error, or integration of the P4N-07 residual
has a favorable measured variance-cost curve. A poor random direction alone
does not reject a response-directed direction; test both listed variants.

## 18. P4N-12: compressed cumulants only where an intervention values them

The previous diagonal Edgeworth test does not evaluate all joint cumulant
closures. Equally, a correct K=3/K=4 formula is not a reason to allocate a full
1024-dimensional tensor.

Proceed only if P4N-03 shows at least 30% raw headroom from a specific accurate
moment/cumulant intervention and a projected implementation can improve score
at utilization `<=0.40` initially.

Use the [ARC reference paper](https://arxiv.org/html/2605.05179v2) and
[author code](https://github.com/alignment-research-center/mlp_cumulant_propagation)
as small-dimension algebra references. They are not evidence that a compressed
version at 1024×16 will reach the frontier.

Try, in order:

1. Correct-sign marginal skew/kurtosis readout in only the final 1/2/4 layers,
   with a deployable starting state. Oracle-start results remain separate.
2. Rank `2,4,8` covariance-response modes selected using a target-free response
   or pilot fluctuation, retaining signed mode couplings where required.
3. The same state used only to improve the P4N-01 control center, instead of
   replacing the full final prediction.

Do not ask a weaker agent to invent a tensor projection. Before implementation,
write the exact carried contractions, update equations, and their shapes in
the experiment preregistration. They must reproduce dense reference updates
on a small fixture to an explicit approximation tolerance. If no valid
contraction rule can be written from the reference and chosen projection,
record this sublane as unspecified and proceed; do not fabricate an update.

For the response-mode option, a concrete first diagnostic representation is
`delta_C = U Lambda U^T`, with U selected from pilot covariance fluctuations and
downstream response energy, rank `2/4/8`. Test high-accuracy state projections
offline first, then estimate coefficients from pilot moments with shrinkage.
This represents a covariance correction, not a full third/fourth cumulant
tensor. Do not relabel it as exact K=3 propagation.

Project noisy pilot corrections onto the supported subspace and maintain valid
marginal variances. Measure final score and transported center error, not only
moment reconstruction. If the accurate projected state helps but the sampled
state fails, report an estimation-noise bottleneck rather than declaring the
subspace intrinsically useless.

## 19. P4N-13/14: combine and optimize the architecture that earned it

For surviving branches, save per-network/per-salt final residual vectors and
full costs. Consider no more than three pairs initially:

1. best improved center + mapped control;
2. best carrier/conditional rule + best residual integrand;
3. best analytic branch + best independent corrected sampler.

Compute a global convex blend oracle as a headroom diagnostic, then a grouped
LOO-fixed blend. Always charge both branches and shared work correctly. Require
at least 10% costed oracle headroom and at least 5% deployable scored gain over
the strongest constituent before retaining a combination. A sample-level
control and an endpoint blend can remove the same error; their gains do not
multiply automatically.

Profile the winning design at the three selected budget points. Then optimize
in this order:

1. Share the actual sampled prefix and reuse already computed moments.
2. Apply the first-layer antipodal identity; use a verified FWHT for compatible
   frames rather than multiplying a dense Hadamard matrix.
3. Avoid sample-by-sample terminal computation where an affine surrogate plus
   an explicitly sampled nonlinear correction gives the same estimator.
4. Freeze routing from an independent pilot; use metered fixed buckets and
   compact arrays. Distinguish exact zeros from rare-firing approximations.
5. Use justified symmetric/Gram contractions and float32 states. Compare
   against dense reference values; do not use a meter discrepancy as a gain.
6. Reallocate saved compute to the residual term or lower the multiplier.

Every approximation needs a paired accuracy ablation. Every exact rewrite
needs numerical equivalence and a fresh measured cost. The expected benefit of
ordinary late dead-column pruning alone is modest; do not advertise it as a
35× estimator breakthrough.

## 20. Gates, artifacts, and honest stopping

### Screening decisions

After each experiment, make exactly one disposition:

- **Invalid:** failed algebra, wrong index/seed, unsupported oracle, target
  leakage, or unmeasured cost. Repair once in a new revision. An invalid result
  cannot reject the mechanism.
- **Continue:** at least 5% adjusted gain with 5/8 wins; or at least 20% costed
  oracle headroom with an identified, addressable bottleneck; or the lane's
  stricter gate is met. Run the listed extension, not an arbitrary sweep.
- **Useful component:** below the full promotion gate, but a defined combined
  architecture has at least 10% costed oracle gain. Test that combination.
- **Archive tested branch:** valid eight-network result, weak relevant oracle
  and failed specified extensions, or measured variance-cost impossibility
  for that implementation. Name the exact scope of the rejection.
- **Interim finalist:** passes all promotion conditions below.

Do not stop merely because three methods failed, because an old report says
“irreducible,” or because the control still wins. Traverse the independent
queue. Conversely, do not generate dozens of scalar variants of a rejected
mechanism to inflate the experiment count.

### Promotion: one conjunction, no shortcut success branch

All must hold:

1. Eight-network adjusted mean improves at least 10% against the immutable
   control, and beats the current interim candidate on its paired comparison.
2. At least 6/8 network means across the three confirmation salts improve;
   the default-offset panel also has at least 6/8 wins.
3. No network's salt-averaged adjusted score regresses more than 10% against
   the control. If a high-upside prototype violates this, keep it experimental
   and seek a target-free guard; do not auto-promote.
4. Every confirmation salt improves the panel mean by at least 5%; overall
   three-salt adjusted improvement is at least 10%. Deterministic candidates
   compare to the corresponding control-salt results without pretending to
   have independent candidate randomization.
5. Zero failures, correct output contract, actual per-network scored compute,
   residual `<0.35 s`, and the other operational margins in Section 5.
6. Any reference-fitted rule passes outer grouped LOO in the same direction;
   no hidden per-MLP target choices survive into the candidate.
7. A clean-process and packaged reproduction agree within a tolerance stated
   before the replay. Investigate material drift rather than averaging it away.

The 10% interim threshold is approximately `1.1012783e-7` on the stored default
control receipt; calculate comparisons from complete fresh receipts rather
than rounded prose. A `<1e-8` score still has to pass every other gate.

### Final verification, still only eight networks

Freeze the candidate code/data and all fitted coefficients before the five
extra offsets. Report the three-salt and five-extra-salt aggregates separately.
If a target is passed on only one lucky seed, the target is not complete.

Use the existing
[validation and packaging workflow](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/how-to/validate-run-package.md)
with `--n-mlps 8` explicitly retained in all full-size checks. Include one
isolated subprocess replay of the frozen candidate to check the memory and
execution boundary. Verify any fixed-width candidate also satisfies the tiny
`whest validate` contract through a general fallback.

Package only the candidate and required generic assets in a clean staging
directory. Do not include research truth caches or whole-workspace contents.
Do not submit to AIcrowd. Preserve the original control even when a new
`best_candidate` is selected.

### Required output tree

The executing agent should create:

```text
whest-starterkit/
  research/phase4_next/
    control/                     # frozen estimator, data manifest, receipts
    prereg/                      # experiment hypothesis/config before running
    results/                     # complete raw JSON, stderr, prediction arrays
    diagnostics/                 # oracle arrays, projected errors, cost curves
    fits/                        # outer-fold coefficients and training manifests
    release/                     # frozen best candidate and minimal package
    experiment_ledger.jsonl
    PHASE4_NEXT_REPORT.md
```

All results must retain:

```text
Experiment ID and revision; status; hypothesis; mechanism;
candidate path and complete SHA-256; control hash;
dataset revision, eight identities and hashes; package/runtime versions;
parameters frozen before evaluation; pilot/evaluation stream roles and seeds;
oracle information used (if any), separate from deployable inputs;
raw MSE vector; utilization vector; multiplier vector; adjusted-score vector;
aggregate mean/median; per-network paired percent changes; wins/losses;
all confirmation salt results; outer-LOO result and all-eight refit separately;
residual/wall/setup time; memory; failure/fallback counts;
response rank/norm; projected center error; residual variance; actual cost;
decision and exact gate; next specified experiment or reason for branch closure.
```

### When the target is not reached

Report the best fully verified candidate, the best unverified prototype, and
the best oracle headroom as three separate items. Give the remaining gap to
`1e-8`, not merely the gain over an obsolete baseline. Provide the next three
experiments justified by the measured bottleneck and state whether the initial
queue and justified extensions were exhausted.

An exhausted research queue with no target-reaching estimator is a legitimate
negative outcome. It is not goal completion, a proof of impossibility, or a
reason to claim leaderboard competitiveness. This plan deliberately leaves
room for another direction if the corrected measurements identify one.

## 21. Source boundaries and why this order was chosen

The strongest immediate evidence comes from the local scripts: the current
control is valuable, but several supposed mechanism ceilings were never
measured correctly. That is why mapped controls and causal state interventions
precede another large carrier tournament.

The [original intra-network-control write-up](https://discourse.aicrowd.com/t/phase-1-write-up-monetizing-the-approach-to-the-moment-propagation-frontier-an-intra-network-control-variate-and-a-measured-map-of-the-wall-behind-it-submission-326024/18154)
also distinguishes a useful multivariate response from the accuracy of its
center. It is Phase 1 evidence for investigating the mechanism, not a forecast
of this run's score. Its discussion of projected center error motivates the
loss in P4N-06.

The [01-1 research release](https://github.com/01-1/arc-wbe) documents prior
rejected ideas and corrections to its own gate interpretations. Use its raw
evidence to avoid duplicating exact Phase 1 tests, while retaining the width,
depth, carrier and evaluator boundaries. No Phase 1 no-go statement is imported
as a theorem for this Phase 2 estimator.

The covariance integral, split-correction cancellation, orientation audit,
exact-feature expectations and variance-cost allocations above are explicit
mathematical checks/derivations in this plan. The proposed combinations and
their numeric experiment gates are research choices, not published benchmark
results. The live leaderboard establishes a competitive scale; it does not
identify the leaders' private methods.

This continuation's central question is testable: **can we obtain a sufficiently
accurate, sufficiently cheap center or surrogate so that most computation
estimates a small residual?** The existing work has not answered that question
with the corrected implementations specified here.
