# Phase 6 agent runbook: faithful joint cumulants, then compression by downstream effect

Prepared 6 September 2026. This is an execution plan, not an executed result. Internal Phase 6 still targets competition **v2-phase2, width 1024, depth 16**. Read `phase6ideas.md` for the broader research argument; this file narrows the work to one mechanism.

## 1. Objective and boundaries

Find whether recurring joint third-order cumulants, with the reference method's selected fourth-order state, can change the accuracy scale. First reproduce the recurrence faithfully. Then reduce its cost while preserving its effect on future predictions. Do not replace this with marginal skew corrections, scalar calibration, Gaussian Hermite-order sweeps, or another blend search.

The user's official best is **user-reported `1.14e-7`**. The earlier leading-score snapshot was approximately `3.5e-9`; it is context, not a paired experimental comparator or evidence of the leaders' algorithms. A 10% gain does not meet the research objective. The milestones below distinguish a useful prototype from a magnitude-level result.

**Hard instructions:**

- Use at most the eight full-size MLPs listed below, including diagnostics and fitting. Do not quietly add a training panel. Tiny algebra fixtures are allowed.
- Use `uv` for environments and execution. Keep reference dependencies isolated from the benchmark environment.
- Do not launch another goal, delegate tasks, submit to AIcrowd, or overwrite the incumbent estimator.
- Execute the stages and bounded alternatives below. If a required operation cannot be implemented faithfully, record the specific obstruction; do not substitute an unrelated heuristic and mark the stage complete.
- Preserve existing reports and candidates. At completion, commit only your own completed artifacts and push to main, following workspace instructions. Inspect the diff first; another agent's files may be dirty.
- A failed implementation, unaffordable implementation, and unsuccessful mathematical approximation are three different outcomes.

## 2. Fixed panel, controls, and evidence contract

Workspace root: `D:\ALL CODES\AICROWD COMPETITION`.
Benchmark root: `whest-starterkit`.
Dataset: `D:\ALL CODES\AICROWD COMPETITION\datasets\mini`.

Use these names in this order:

1. logan-fitzgerald
2. william-graves
3. raymond-barnes
4. steven-rice
5. sarah-kelley
6. christopher-morales
7. cheryl-graham
8. renee-park

Resolve rows using the benchmark's `MLP.from_row` and `resolve_seed_context(ds)` path. Save row identifiers, dataset revision, weight hashes, reference hashes, and actual seeds. Abort on a missing name or version mismatch. Do not fall back to the first eight rows. Every benchmark call must use the existing dataset; do not accidentally invoke on-the-fly reference generation.

Immutable control: `whest-starterkit/estimator.py`, expected SHA-256:

```text
ea8be8222b827607f65d166118500b55fcaae883b79f64b29785b46d86501ca1
```

This is the dual-Hermite/WMC control with lambda `0.20`, scale `0.998319`, alpha `0.110`, and N `4200`. The saved receipt `whest-starterkit/research/phase4_next/results/P4N-00_20260906_000232.json` records adjusted `1.2236425135370156e-7`, raw `1.2236425135370155e-6`, mean utilization `0.09805228656705367`, maximum residual `0.17464429963729344 s`, and eight valid networks. Verify the file and hash; do not silently relabel a different estimator as this control.

Also snapshot `candidates/estimator_p46_laneE_skew_e025.py` if present. Its recorded local adjusted score is `1.189367e-7`. Reproduce before using it as a second comparator. Do not assume it is the source of the user's official `1.14e-7` without a matching receipt.

Every experiment must save predictions `(8,16,1024)`, per-network raw and adjusted scores, all-layer errors, complete parameters, source hash, status, elapsed time, and paths to raw receipts. Metered runs additionally save FLOPs, utilization, setup/predict/residual times, and measured peak memory or an explicit “unmeasured.” Never fabricate resource numbers or copy old receipts into new experiment IDs.

Use `phase6report.md` at the workspace root, and `whest-starterkit/research/phase6/{control,reference,fixtures,moments,states,predictions,diagnostics,results,release}`. Candidate names start `estimator_p6_`. Proposed helper names below are files to implement, not claims that those tools already exist.

## 3. Scoring and execution limits

For each valid network, score `MSE * max(0.1, FLOPs / 2**41)`; average the eight products. Residual time is capped, not converted to FLOPs. A cap violation zeroes predictions and uses multiplier 1.0. Read the installed scoring and allowed-code docs before implementing production arithmetic.

Enforce the benchmark's 120-second predict, 0.4-second residual, 5-second setup, and configured memory caps. Finalist targets: predict below 90 seconds, residual below 0.30 seconds preferred and below 0.35 required, setup below 4 seconds, peak memory below 6 GB and the actual configured cap. Log machine resources first. Never start a reference run whose estimated peak exceeds available safe memory.

Do not constrain all research to 10% utilization. At 20%, a raw error of `1.75e-8` gives `3.5e-9`; a costly but much more accurate method can win. Use actual flopscope accounting, including rank selection and compression. ARC's internal operation estimates are not official benchmark receipts. Production meaningful arithmetic must use allowed metered operations; Torch/NumPy reference code is diagnostic only. Setup is not a place to hide network-dependent calculation.

## 4. Why this experiment is still open

Phase 5 explored many variants, but the inspected P4N-12 code did not implement recurring factorized joint cumulants. Its nominal one-, two-, and four-layer variants repeated a terminal calculation; its center reused an existing scalar calibration. P4N-13's fourth branch was another scaled analytic prediction. These results reject those implementations, not the method below.

The saved P4N-01 oracle-center calculation also does not establish that improving the old center alone closes the gap: its post-hoc optimal common blend is only nominally about `5.15e-8`, unmetered and target-fitted. Therefore the **direct analytic estimator is the primary output of this phase**. A new control using its centers is a secondary, tightly bounded check.

The paper motivates retaining more distributional information, but does not promise a fixed improvement at this width, depth, activation, or budget. Factorization preserves a specified truncated recurrence; compression adds another approximation. Keep those errors separate. [ARC paper](https://arxiv.org/html/2605.05179v2).

## 5. P6-00 — Establish a reproducible starting point

Implement `scripts/p6_manifest.py` and a shared evaluation wrapper. Save the panel manifest, environment versions, controls, source hashes, and dataset configuration. Current inspected versions were whestbench `0.16.1`, flopscope `0.12.1`, NumPy `2.2.6`, SciPy `1.15.3`; report actual versions rather than forcing an upgrade.

Replay the immutable control once at sampler offset 0. Compare predictions and receipts to the saved reference. Explain any discrepancy before scoring candidates. Save an unmodified control copy in the new research folder.

For later stochastic confirmation, use offsets `0,1337,8888`, then finalist offsets `104729,130363,155921,196613,262147`. Implement real sampler variation and save the effective seeds. Do not assume an existing command-line flag changes the sampler. Different offsets must produce different samples; deterministic methods need only one accuracy evaluation, with timing replays if needed.

**Exit:** working evaluation wrapper and immutable control receipt. If unavailable, other algebra work can continue, but no candidate receives a verified score label.

## 6. P6-01 — Acquire diagnostic moments and locate the error channel

Use the public [Phase 2 higher-moment dataset](https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-higher-moments-2026), matching only the eight panel identities. It supplies an independent 100-million-sample bake, including marginal moments and selected mixed pair moments. Download only matching files after inspecting the manifest; process one at a time. Do not download the full dataset or regenerate it locally. These are noisy diagnostics, not perfect truth or deployable features.

Implement `scripts/p6_moment_audit.py`. Verify names, global indices, dimensions, layer conventions, sample counts, and official mean pairing. Convert raw to central moments in float64, recording that stored float32 mixed moments retain their original precision limits. Unit-test the conversion using tiny discrete distributions. Preserve separate preactivation and postactivation coordinates.

For the existing analytic path, save per-layer errors in mean, covariance, marginal kappa3/kappa4 and available mixed kappa21/kappa31/kappa22. Report both ordinary error and its change to the next predicted mean after inserting the available moment correction into the verified local formula.

At postactivation layers `3,7,11,14`, perform separate mean-only, covariance-only, and joint mean/covariance resets, then actually propagate the remaining `15-layer_index` transitions. Label these hybrid interventions; they are not necessarily physically realizable distributions. Save changed state hashes and remaining transition counts. If changing the intervention layer produces identical paths, fail the test.

Do not synthesize an arbitrary full third-order tensor from pair slices. Do not call marginal Edgeworth terms the full joint recurrence. Do not inject the final answer and count it as evidence of an estimator. Mixed-moment interventions are permitted only where the verified formula explicitly consumes those slices.

**Exit:** a layerwise diagnostic map. Missing downloads or noisy resets do not veto faithful reference reproduction. A reduction of at least 4x in final raw MSE is a prioritization signal; weaker results are evidence, not a proof of no higher-order headroom.

## 7. P6-02 — Reproduce the actual reference, including its omissions

Use the repository [alignment-research-center/mlp_cumulant_propagation](https://github.com/alignment-research-center/mlp_cumulant_propagation), pinned to commit:

```text
93d091a4c26c042bfffa28f2e76a81bc0aba94bb
```

The import package is `mlp_kprop`. Create an isolated uv reference project; do not install its newer scientific dependencies into the benchmark environment. Read its license and preserve attribution. Inspect resource requirements before syncing dependencies. Do not run its width-sweep or sampling scripts.

Read `factor_k3.py`, `kprop_harmonic.py`, `diagslice.py`, `harmonic.py`, `wick.py`, and `tests/test_factor_k3.py`. Document a source-to-port function map in `reference/port_map.md`.

Implement three distinct modes using `factored_nonlin_kprop_k3` with `use_pK=True`:

| ID | Flags | Meaning |
|---|---|---|
| K3-base | `base=True, augment=False` | Reference base third-order mode |
| K3-simple | `base=False, augment=False` | Reference mode with its selected fourth-order state |
| K3-augment-filtered | `base=False, augment=True` | Reference factorized augmented term set |

**Critical caveat:** the pinned source explicitly drops some augmented diagrams, including covariance-triangle and kappa3-edge terms. Augmented factorized output must match the **same filtered** dense reference, not full augmented output. Also compare filtered versus full augmented on tiny fixtures to quantify this separate omission. Do not rename the filtered variant “full K4.” [Pinned implementation and warning](https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py).

Use explicit linear/nonlinear steps rather than blindly calling the reference MLP wrapper: its last linear layer has no activation, while this benchmark has ReLU after **all 16 matrices**. For benchmark row convention `h @ W`, reference factor transport uses `W.T @ A`. Verify this once with an asymmetric fixture. Input state is Gaussian mean zero, covariance identity, higher cumulants zero.

Required tests, widths `8,16,32`, depths `1,2,4`, seeds `6,17,29`:

- Dense versus factorized linear transport and symmetrization; all repeated-index slices; signs and scaling of factor columns.
- Gaussian ReLU mean/variance against closed forms, including negative, zero, and positive standardized means.
- Factorized recurrence versus the corresponding dense term set after every transition.
- Final activation and transpose tests with deliberately asymmetric weights.
- Removing K3 must change a non-Gaussian fixture; restoring it must recover the reference.
- No stale repeated-slice cache after adding, removing, scaling, or transporting factors.

Use float64. Algebra identities target `atol=1e-10, rtol=1e-8`; nonlinear comparisons record maximum absolute and relative errors and inherited reference tolerances. If tolerances need relaxation, investigate and document the cause before accepting. Negative variances and nonfinite states are failures, not permission to silently clamp everything.

**Exit:** parity receipts for all three modes. Do not proceed with a mismatched “approximate port.”

## 8. P6-03 — Measure the uncompressed method before changing it

Implement `scripts/p6_reference_panel.py`. First produce a shape-based operation and memory forecast at width 1024, all 16 layers, including growing factor counts, cached slices, temporary buffers and fourth-order state. Never materialize a `1024^3` or `1024^4` tensor. Block contractions where necessary.

If safe, run each reference mode on the first panel MLP, then all eight. An unmetered reference may exceed competition compute limits but must obey a declared local resource bound: below 6 GB peak and no more than 10 minutes per MLP initially. Terminate cleanly at the limit and record the unfinished layer. Do not spend hours on an extrapolated infeasible pass.

Save each layer's state summaries, factor counts, means and available mixed slices. Compare direct final means to the same official reference. Preserve full states only where storage is reasonable; otherwise use reproducible checkpoints and hashes.

Decision:

- Raw MSE at or below `1e-7`: substantial analytic headroom; prioritize production cost reduction.
- Raw MSE between `1e-7` and `5e-7`: useful but incomplete; inspect where the error first appears and compare base/simple/filtered-augment.
- Raw MSE above `5e-7`: check parity and conventions, then run P6-05's omission diagnostic before investing in compression tuning.
- Resource failure: establish exact short-block parity and factor-growth receipts; proceed to bounded compression. Do not claim the uncompressed method's full-size accuracy is known.

These are research allocation thresholds, not claims that all networks must cross them. Report the complete eight-vector and median, mean, worst, and wins.

## 9. P6-04 — Main modification: remove factors according to downstream damage

This stage is a new research proposal, not an ARC guarantee. Start with the best faithful mode from P6-03. Preserve its nonlinear recurrence and fourth-order representation. Modify **only K3 factor retention** first.

Write the symmetric tensor as a sum of factor atoms `T = sum_j Sym(a_j ⊗ b_j ⊗ c_j)`. Pruning or weighting an atom must operate on one factor only; multiplying all three by a coefficient incorrectly cubes that coefficient. The first experiment uses deletion only, with no fitted weights.

Implement this explicit bounded prototype:

1. Preserve factor provenance: birth layer, recurrence term family, and original column index. Divide each birth-layer/term-family block into consecutive groups of 128 columns; keep a shorter final group.
2. At compression points after layers `3,7,11,14`, target old-factor retention fractions `1/2` and `1/4`. Keep newly generated factors from the current transition intact for this first experiment. Retention fractions refer to eligible old columns, and actual total columns must be reported.
3. Produce the control compression by ranking groups on a computable upper bound: sum over columns of `||a_j|| ||b_j|| ||c_j||`. Retain highest-bound groups. This is explicitly a norm-based heuristic, not exact tensor error minimization.
4. Produce the proposed compression by greedy deletion. For each eligible group, remove it temporarily, invalidate caches, and roll the **entire coupled recurrence** forward for two more transitions, or the remaining transitions if fewer. Compare the resulting means with the unpruned state rolled forward from that same checkpoint.
5. Define deletion loss as the average of squared mean differences across both lookahead layers and all neurons. Delete the group with smallest loss, recompute losses after each accepted deletion, and stop at the requested retention fraction. Break ties by provenance order. This tests nonlinear downstream damage and cancellation directly; do not replace it with the final weight matrix or a mean-gate-only map.
6. Save the retained mask and independently continue the compressed trajectory through the actual remaining layers. Measure final prediction distortion relative to an uncompressed trajectory where available, plus error against official means. Short lookahead is only a proxy for final influence.

**This greedy version is an offline mechanism experiment and may be too expensive for deployment.** Cap it at 30 minutes per network and 6 GB, using only these eight networks. Begin with the first network. If it exceeds the cap, use a single scoring pass from the unpruned checkpoint, delete groups in that fixed order, and label the result “one-shot influence,” not greedy. Do not store oracle-selected masks for deployment.

Run the four combinations of retention fraction and norm/influence ranking. Repeat only the winning influence setting with lookahead `4` to test whether local influence was misleading. No rank-grid expansion yet.

Advance influence selection if it either halves final distortion relative to norm selection at comparable retained columns, or supports at least twice the compression at similar final distortion. If neither happens, report that this compression proposal lacks evidence and retain the faithful method or norm baseline; do not endlessly tune masks.

For deployment, every mask must be generated from the current network's predicted states inside the metered path. First meter the one-shot selector. If unaffordable, implement a directional-derivative approximation to the same two-step loss, using the reference recurrence as the derivative oracle. Verify its directional predictions against centered finite differences with steps `1e-3,1e-4,1e-5` on tiny fixtures and against exact deletion losses on the first panel network. Port all derivative arithmetic to allowed operations. This is the only permitted selector acceleration in this phase; if it still fails cost or accuracy gates, mark output-influence compression operationally unresolved.

Never export panel-specific masks, moment files, target errors, or network-name lookups into the candidate. A reference-fitted mask is evidence of capacity only. Recompute dependent fourth-order updates normally after compression; do not freeze their original trajectory to conceal accumulated error.

## 10. P6-05 — Diagnose selected fourth-order state and omitted diagrams

Run this stage if the faithful method has insufficient accuracy or filtered augmentation unexpectedly underperforms simple mode.

First use the tiny fixtures to compare full augmented, filtered augmented, and simple recurrence, separating mean, covariance, and K3 differences. Inspect the source's `factored_keeps_term` filter and list exactly the omitted diagram families. Rank their measured contributions to future mean error; do not invent a covariance-scale correction and call it a missing diagram.

Attempt **at most one** additional family: the largest-impact omitted family on the tiny diagnostics that admits a derived blocked contraction without a dense width-cubed allocation. Write the contraction and operation count before porting. Verify it against the dense reference, then evaluate on the same eight networks. If no family has a safe contraction, stop this branch with the obstruction documented.

Do not launch full width-1024 K4 propagation, a new SSC coefficient-fitting campaign, or another sampling family in this phase. Public Phase 1 SSC is relevant prior art, but it is not a drop-in replacement and its larger training sets cannot be copied under this eight-network limit. A future phase can address those alternatives after these results establish the missing channel.

## 11. P6-06 — Meter the frontier, then allow one control check

Port the surviving direct methods to `candidates/estimator_p6_<mode>_<compression>.py`. Keep scientific reference code separate. Validate production/reference parity before measuring scores.

Evaluate the actual frontier at total utilization approximately `0.10,0.20,0.35`, where feasible by retaining more factors or choosing a surviving recurrence mode. These are ceilings for separate candidates, not multipliers applied after an unmetered run. Skip duplicate or infeasible configurations. Choose by the true average adjusted score and failures, not raw accuracy alone.

Only after direct analytic results exist, test one secondary construction: replace the analytic center in the already verified layer-14 mapped control with the new method's center. Preserve coordinate mapping and compare to the same control using the old center. Use N `4200` if the measured total budget permits; otherwise calculate and document the affordable count before running. Keep independent pilot and evaluation batches, including independent whitening. Never estimate the control map on its evaluation batch.

Test fixed mixture weights `0,0.5,1` first. Permit one scalar leave-one-MLP-out mixture fit only if the new center materially improves the control error. Hold out all layers and salts of that network together; save fold weights. Do not reopen four-way blend optimization. A winning blend of weak branches does not substitute for demonstrating the new mechanism.

## 12. P6-07 — Confirmation and outcome classification

Confirm the best at most two deployable candidates on all eight networks. Stochastic methods use the paired offsets in P6-00. Deterministic methods reuse their prediction vectors and need resource rechecks, not fake salt replication.

Require zero failures, finite float32 output of shape `(16,1024)`, actual resource margins, and at least 6/8 wins against the immutable control on the mean across stochastic salts. Require no network worse by more than 25% for a replacement recommendation; if a large aggregate gain misses this, preserve it as an experimental candidate with the outlier clearly identified. Compare the second local incumbent where reproducible.

Classify final evidence:

| Outcome | Meaning |
|---|---|
| Adjusted `<=1.2e-8` | Approximately an order-of-magnitude local improvement; magnitude objective supported on this panel |
| Adjusted `<=5e-9` | Leader-scale local candidate; not a prediction of official rank |
| Adjusted `<=6e-8` | Strong intermediate gain; gap remains |
| Smaller improvement | Preserve if valid, but do not call Phase 6's magnitude objective achieved |
| Accurate reference, infeasible production | Representation works locally; cost reduction remains unresolved |
| Faithful reference weak | Report tested recurrence and omissions; do not generalize to all joint-distribution methods |

Repeated selection on eight networks remains exploratory even with LOO. No claim of fresh generalization or official improvement without an actual official result. Do not submit automatically.

## 13. Required final report and stopping rules

`phase6report.md` must open with the best **verified** outcome and the remaining gap. Include:

- A receipt-linked table for P6-00 through P6-07, with completed, failed, skipped-by-gate, or blocked status.
- Source pin, recurrence flags, omitted diagrams, parity results, and final-ReLU/transpose checks.
- Per-network scores and resource values, distinguishing diagnostic, reference, and deployable runs.
- Four separate explanations: baseline approximation error; omitted-diagram error where measured; compression distortion; production numerical/cost effects. Do not subtract MSEs and present the result as an exact additive decomposition.
- All candidate hashes, parameter choices, selection history, and confirmation salts.
- Whether joint cumulants improved direct prediction; whether influence selection beat norm selection; whether a stronger center still left a residual-integration bottleneck.
- A concrete next recommendation supported by this phase's results, including a clear negative result when appropriate.

Stop a branch after its specified alternatives fail; continue other eligible stages. Do not spend the remainder of the run on scalar calibration to manufacture a success. If the method cannot meet the target, a faithful implementation, measured error channel, and precise computational obstruction are useful completed research artifacts. They are not a completed accuracy goal.

At handoff, provide the report, best safe experimental candidate if one exists, and reproducible commands for the helpers actually implemented. Keep the incumbent unchanged. Commit and push only the completed work owned by this execution.
