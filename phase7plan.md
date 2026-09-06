# Phase 7: reduce the cost of joint-cumulant estimation without losing its accuracy

Prepared 7 September 2026. **This is a future execution runbook for the user's autoagent. Preparing it did not run competition experiments or authorize an AIcrowd submission.** It extends [phase6.5plan.md](phase6.5plan.md), using the actual generated sources and receipts as evidence rather than accepting every conclusion in [phase6.5report.md](phase6.5report.md).

## 0. Mission and execution boundaries

Deliver a measured change in the cost–accuracy frontier of the Phase 2, width-1024, depth-16 estimator. The main objective is **valid adjusted score below `1.00e-8`**, with stretch targets **`6.00e-9` and `3.00e-9`**. Preserve roughly the current raw accuracy while making the method much cheaper; retain improvements in raw accuracy when they repay their added compute. These are targets, not promised outcomes.

Execute all mandatory stages and every triggered conditional stage in order. A long run is expected. Do not end after a small win, the first failed compression method, or a timing failure that still permits useful diagnostic work. Each stage must end with receipts and a decision. Resume interrupted work. Complete the final report even if the target is missed.

**Implement the equations and grids in this file. Do not replace a method with a similarly named heuristic, silently remove experiments, or claim an unimplemented switch was tested.** Routine debugging and contraction scheduling are the executing agent's responsibility. If an identity fails, repair the implementation; if the specified mathematics or permitted API cannot support it, record the precise obstruction, stop that branch, and continue independent branches. The optional extensions have explicit gates; do not invent gates that close an entire mathematical family.

- Work sequentially; do not launch other agents, goals, or concurrent numerical jobs. Long-running means resumable execution, not unrestricted machine load.
- Use `uv` for Python and dependency management. Run benchmark helpers from `D:\ALL CODES\AICROWD COMPETITION\whest-starterkit`. Preserve the existing environment pins unless a documented incompatibility requires a separately recorded environment.
- Keep the established eight full-size networks, including diagnostics and fitting. Tiny synthetic fixtures are allowed. Do not generate/download additional full-size networks under this plan.
- Preserve existing estimators, Phase 6/6.5 reports, predictions, reference code, and unrelated dirty files. Create Phase 7 files separately. The legacy `whest-starterkit/estimator.py` is immutable.
- Candidate numerical work, including compression, sorting, randomness, solves, and decompositions, must use permitted flopscope operations. Ordinary NumPy/SciPy are for offline diagnostic helpers only. No accounting modifications, unmetered kernels, or runtime research-data access.
- Package a qualifying finalist. **Do not submit to AIcrowd without a separate user instruction.** At completion, commit only completed Phase 7 artifacts and push to `main`, following the workspace instruction.

Current official contract: [allowed code](https://raw.githubusercontent.com/AIcrowd/whest-starterkit/main/docs/concepts/allowed-code.md), [scoring](https://raw.githubusercontent.com/AIcrowd/whest-starterkit/main/docs/concepts/scoring-model.md), [performance guidance](https://raw.githubusercontent.com/AIcrowd/whest-starterkit/main/docs/how-to/performance-tips.md). Verify these and installed runner settings at P7-00; preserve the version snapshot. Meaningful computation remains metered even when it would fit inside residual time.

## 1. Starting point and arithmetic targets

Freeze [candidates/estimator_p65_final.py](candidates/estimator_p65_final.py), identical to `estimator_p65_03c_L10_p10.py`, as `CONTROL65`:

```text
SHA256 = b06d91bc155f19e696c690c2b9956c2ba2426ba142de3c7d6dcf7f078344255e
pruning after zero-based layers [6,10,13]
old-column retention [0.80,0.90,0.80]
local mean raw final MSE = 3.78537768e-8
local mean utilization = 0.7868188718994134
local projected adjusted score = 2.97840659e-8
local unrelaxed maximum residual = 1.6429321 s; all eight fail the residual gate
```

Evidence: [unrelaxed receipt](whest-starterkit/research/phase6_5/results/P65-03c-L10-p10-unrelaxed_20260907_012338.json), [ledger](whest-starterkit/research/phase6_5/ledger.jsonl), and [progress](whest-starterkit/research/phase6_5/progress.json). Hash the actual source before use. If it differs, locate the recorded source and establish which bytes produced the receipt before scoring anything.

The user's newer screenshot displays official rank 40, MSE `3.64e-8`, utilization `78.68%`. Treat this as user-supplied leaderboard evidence; verify its source/package association before attaching a submission ID. Do not substitute its raw MSE into the local paired panel. The historical Phase 6 submission #330018 is also distinct. The uncompressed local reference reports raw `3.63704648e-8`; this is an approximation, not truth or an accuracy lower bound.

For valid network `m`:

```text
B = 2**41 = 2199023255552
E_m = mean((prediction_final_m - truth_final_m)**2)
u_m = billed_FLOPs_m / B
S_m = E_m * max(0.1,u_m)
S_panel = mean_m(S_m)
```

Use actual per-network products. A cap failure uses the official failed score, not discounted pre-failure predictions. Expected caps are 0.4 s residual, 120 s predict, 5 s setup, and 8 GB memory; verify the installed and graded contract. Aim for margins of 0.35 s residual, 90 s predict, 4 s setup, and 6 GB peak memory.

At the **local** `CONTROL65` raw error, unchanged-accuracy scenarios are:

| Utilization | Hypothetical adjusted score | Meaning |
|---:|---:|---|
| 0.40 | `1.5142e-8` | Useful, short of the main goal |
| 0.25 | `9.4634e-9` | Main breakthrough scale |
| 0.15 | `5.6781e-9` | Stretch scale |
| 0.10 or lower | `3.7854e-9` | Further compute cuts no longer help |

At utilization 0.15, `3e-9` requires raw `2e-8`; at the floor it requires raw `3e-8`. Therefore retain an accuracy lane even if compression succeeds. A faster wall clock with unchanged billed FLOPs improves reliability, not the ranked product.

Maintain two research frontiers: **accuracy-preserving** (mean raw MSE at most `1.05 * CONTROL65`, worst raw ratio at most 1.15) and **best product** (may sacrifice more raw accuracy). Neither frontier overrides validity. A 50% compute reduction with 10% higher error is a useful 45% product reduction.

## 2. Why Phase 7 has these branches

The current factors encode `sum_r Sym(u_r,u_r,v_r)`. Their number grows with depth. Ordinary norm pruning deletes information; carrying more columns nearly recovers the uncompressed error but consumes about 79% of the budget. The desired change is a compact representation of the *sum*, not another percentage adjustment to the number of surviving columns.

Research hypotheses, ordered by importance:

1. **Preserve the diagonal/repeated slices and compress the all-distinct remainder separately.** These slices directly feed the nonlinear recurrence, while the remainder matters after rotation by later weights. A matrix slice state plus a small Tucker core may carry both economically. This is the main structural experiment, not an established low-rank property of the networks.
2. **Replace old factor cohorts with a summary instead of deleting them.** Short explicit memory, slice summaries, and preactivation resets offer different cost/error tradeoffs. Their errors must be measured after subsequent mixing.
3. **Reweight or sample the atom sum.** Related atoms can cancel. Group-aware fitting and an unbiased current-tensor tail estimator may preserve information that positive norm deletion loses. Nonlinear recursion can still turn local unbiasedness into final bias.
4. **Repair the evidence gaps before declaring simpler methods closed.** Several Phase 6.5 configurations did not execute their documented settings.
5. **Use the freed compute to repair identifiable approximation errors.** Corrected diagonal fourth-order tests, a specified richer fourth-order state, and shared-prefix extrapolation are separate experiments.

Conceptual references: [ARC paper](https://arxiv.org/abs/2605.05179), [ARC implementation](https://github.com/alignment-research-center/mlp_cumulant_propagation), [Sun et al., low-rank Tucker sketching](https://arxiv.org/abs/1904.10951), and [Wang et al., tensor contractions through sketches](https://arxiv.org/abs/1506.04448). Tensor sketching literature motivates compact contractions; it does **not** establish small Tucker rank for this recurrence. The specific hybrid and fourth-order approximations below are proposed experiments, not claims to reproduce those papers or a leader's undisclosed method.

## 3. Fixed panel, artifacts, and evaluator

Use these names in precisely this order, width 1024/depth 16, from `../datasets/mini`, split `mini`:

```text
logan-fitzgerald
william-graves
raymond-barnes
steven-rice
sarah-kelley
christopher-morales
cheryl-graham
renee-park
```

Verify weight, truth, seed, and ordering hashes against [Phase 6.5 manifest](whest-starterkit/research/phase6_5/manifest.json). Reuse exact cached outputs where source, effective configuration, panel, seed, arithmetic backend, and runner settings match. A nominal experiment ID is not a cache key.

Create during execution:

```text
phase7report.md
candidates/estimator_p7_<id_parameters>.py
whest-starterkit/scripts/p7_manifest.py
whest-starterkit/scripts/p7_build_candidate.py
whest-starterkit/scripts/p7_math_checks.py
whest-starterkit/scripts/p7_profile.py
whest-starterkit/scripts/p7_state_audit.py
whest-starterkit/scripts/p7_eval.py
whest-starterkit/scripts/p7_select.py
whest-starterkit/scripts/p7_run_plan.py
whest-starterkit/research/phase7/
  control/ configs/ fixtures/ predictions/ diagnostics/ results/ release/
  manifest.json
  experiment_manifest.json
  progress.json
  ledger.jsonl
```

The helpers above are specifications, not existing commands. Implement them before invoking them. Required orchestration interface:

```powershell
$env:PYTHONUTF8 = '1'
uv run python scripts/p7_run_plan.py --resume
uv run python scripts/p7_run_plan.py --stage P7-06 --resume
uv run python scripts/p7_math_checks.py --candidate ../candidates/estimator_p7_<id>.py
uv run python scripts/p7_eval.py --candidate ../candidates/estimator_p7_<id>.py --exp-id <ID> --offset 0
```

Build every candidate as a standalone, explicit implementation. Reject unknown configuration keys and unsupported switches. Normalize all layer-indexed maps with `int(key)` on configuration ingestion. Export an effective-configuration record from the built source. Require equality between requested and effective settings, including defaults, before evaluation. Test JSON save/load/build round trips. Materialize derived candidate grids and their parent hashes in `experiment_manifest.json` before running each stage.

Every record includes full source/config/panel hashes, representation, layer policies, retention/caps, scalar/vector/core dtype, rank history, seed offset, backend versions, command, complete per-network metric vector, output paths, all-layer errors, and status. For approximations include reconstruction diagnostics and the error decomposition in Section 5. Update progress atomically after each network and each configuration. Use unique temporary capture paths; wait for the child process to exit and validate output shape/checksum before publishing a receipt. Never overwrite one candidate's outputs from another evaluation.

Research-only logs, state copies, norms, and reconstruction probes run in a separate offline replay, with their own diagnostic RNG and cost record. They must not change the production candidate's random stream or be added to its reported production FLOPs. Conversely, every probe, decomposition, validation check, or fallback that **affects a prediction** belongs inside the candidate and is charged. Verify replay predictions match production before using its state diagnostics. Release code neither writes research files nor accesses truth. A missing branch winner is recorded as null; never resolve an absent winner by loading a similarly named stale Phase 6.5 parent.

States: `PLANNED`, `INCOMPLETE`, `MATH_FAILED`, `API_BLOCKED`, `NUMERIC_FAILED`, `SKIPPED_COST`, `SKIPPED_GATE`, `REJECTED_ACCURACY`, `RESEARCH_ONLY`, `VALIDATED`. Record resource validity and scientific selection separately: a numerically sound losing candidate is different from a broken implementation. A nominal completion flag without the mandatory receipts is incomplete.

## 4. Shared experiment and promotion rules

For each prescribed configuration:

1. Pass independent tiny identities and small-depth contract checks.
2. Forecast billed work and peak live storage from the **actual contraction order**, including sketches, fitting, and fallback costs. Query available RAM. Reject a production forecast above `0.95B` or the smaller of 6 GB and 60% of currently available RAM; document shapes and arithmetic. An offline reference diagnostic can exceed the FLOP cap but never the machine memory gate. Do not run dense `1024^3`/`1024^4` tensors.
3. Operational smoke on the first fixed MLP. A single-network loss does not reject a family. Stop here only for a mathematical, numerical, or resource failure.
4. Evaluate all eight serially. No published full-panel mean until all eight complete. Keep direct unrelaxed official results separate from diagnostic captures.
5. Compare to both `CONTROL65` and the frozen stage parent. Store the raw errors and actual products, wins/losses, worst ratio, costs, and uncertainty where estimated.

For exact rewrites, require float64 identity error `<=1e-10 * max(1,max(abs(reference)))`; full-panel float32 raw ratio in `[0.995,1.005]`, worst raw ratio <=1.01, and final prediction RMS drift <=`2e-6`. Investigate violations rather than calling a drifting rewrite exact. Floating-point equality is not required. Accept verified cost saving even if small.

For a non-exact research-parent promotion: at least 2% lower mean product, at least 6/8 paired wins, and worst score ratio <=1.10. Keep other non-dominated cost/error points for later combinations even when they miss this promotion rule. Break ties within relative `1e-4` by lower billed compute, then lexicographic ID.

Stage expansion gate, unless explicitly overridden below: either a supported product improvement of >=5%, **or** at least 25% less compute with mean raw ratio <=1.25 and worst raw ratio <=1.50. The second gate deliberately allows a promising cheap prototype to receive work before it beats the high-retention parent. Do not use a fit-residual improvement alone as a final promotion.

Stochastic fits/sketches/sampling use offsets `{0,1337,8888}` with identical network weights/truth. Run offset 0 for all mandatory configurations. Run the other offsets for every candidate meeting the expansion gate and every would-be finalist; compare against matching-offset parents. A retained randomized finalist must improve the parent mean product on each offset and satisfy the paired rule on the 24-observation mean. Report worst offset and per-network dispersion; three salts do not prove unbiasedness or provide an independent network test.

The offline harness may replace only the seed field passed to the estimator by `(original_seed+offset) mod 2**32`; assert weight/truth hashes remain unchanged. Do not use a CLI seed option that regenerates network weights. Keep seed/capture-wrapper timing separate from direct release validation.

Use a single `fnp.random.default_rng(mlp.seed)` per prediction. Fixed stage-specific draw order: compression fit probes, check probes, decomposition probes, optional tensor-tail draws, optional input samples. Draw only the active mechanisms' arrays, in a documented order. For nested rank grids generate the stage's common maximum probe bank and charge it in each comparator, or explicitly bill separate banks and report the lost coupling. No free pilot, reseeded duplicate cloud, or label-dependent choice inside `predict`.

For repeated layer operations, apply that order within each layer: K3 compression draws first, then active K4 and covariance decomposition draws. Draw any input sample cloud only after finishing the analytic loop. An offline diagnostic RNG never advances this production generator.

LOO: recompute held-out selection from saved per-network products for every complete grid, using only the other seven to choose. Exclude incomplete, failed, and mismatched-source records. Save all chosen IDs and fold values. Since earlier stages adapt to these same eight, LOO is a selection-stability diagnostic, **not independent generalization**. No full-size holdout claim is possible under the eight-network limit. Any fitted global coefficient must be fitted inside each fold; report the all-eight deployable coefficient separately.

Final release promotion: unrelaxed validity on all eight, >=10% mean product improvement over `CONTROL65`, 6/8 wins, worst ratio <=1.10, plus all applicable stochastic checks. A smaller exact improvement may be released only as an exact maintenance improvement, not as the Phase 7 breakthrough. Continue all remaining independent stages even if `<1e-8` is reached early.

## 5. P7-00 and P7-01: repair the evidence, measure the cost and error channels

### P7-00: mandatory source/configuration audit

The following issues were observed while preparing this plan. Preserve the old files and write corrections in Phase 7 records.

| Evidence | Required action |
|---|---|
| `estimator_p65_07_s13_e050.py` and `estimator_p65_08_n0512_a005.py` contain `per_layer_ret = {'6':0.8,'10':0.9,'13':0.8}` but call `.get(l,default_retention)` for integer `l`. | Audit every generated source for this bug. These examples execute retention 0.8 at layer 10, not 0.9. Regenerate affected comparisons with canonical integer keys. |
| `p65_build_candidate.py` reads `reweighting`, `birth_pairs`, and `diag_response` options, but the inspected selector branch implements only `norm_exact` versus the default; reweighting has no generated solve. | Treat switches as unimplemented until the generated source and behavioral test demonstrate the requested operation. Implement them explicitly in Phase 7. |
| `estimator_p65_04a_exact.py` uses `[7,11,14]` with overrides at 11/14, whereas the main report describes it against `[6,10,13]`. | Reconstruct the true parent from bytes/receipts. Run the missing controlled comparison; do not attribute a schedule change to a selector. |
| Terminal-split state stores penultimate births outside `U,V`, while some variants prune at layer 14 using a separate provenance list. | Assert column/provenance agreement, and logical inclusion of protected births. Test a layer-14 event with and without split storage. Do not assume it is correct or incorrect without parity. |
| The P65-06 gate refers to distortion or product, but the report does not supply the required distortion comparison and no reweighting implementation was found. | Reopen reweighting unconditionally in P7-04. Compare reconstruction and score independently. |
| Phase 6.5 sources/ledgers contain more candidates than the main report's final narrative, and a capture race is described in `phase6muse.md`. | Rebuild the ledger and LOO from hashes and complete receipts; do not use a manually transcribed winners table as the authority. |
| `phase6muse.md` records a terminal `k3**2` Edgeworth pilot with negligible gain. | Reuse its matching receipt and source; do not relabel it a new breakthrough or rerun its unchanged parent. |

`P7-00-audit` produces `diagnostics/phase65_audit.json` and a table in `phase7report.md`: actual source, intended/effective parameters, reliable comparisons, missing experiments, and repaired IDs. `P7-00-control` freezes `CONTROL65`, verifies cached predictions, and records a direct unrelaxed baseline. First compare direct and capture paths on one MLP; only replay the eight if cache integrity/settings require it. Do not repeat unchanged timing checks after that.

### P7-01a: billed-compute profile and attainable cost calculations

Use documented profiling on the first network and confirm total count on all eight. Produce per-layer totals for:

- mean/covariance and `M=W@W.T` transport;
- old `W@U`, `W@V` factor transport;
- repeated-slice products `(U*V)@U.T`, `(U*U)@V.T`;
- newborn transport, nonlinear moment algebra, K4 scalar/extension work;
- selection/recompression, decompositions, sampling, array copies, scalar conversions, and orchestration overhead.

Reconcile the sum against the official total; report attribution uncertainty if the profiler groups operations. Do not patch client internals to obtain a profile. Profile instrumentation stays outside release source and its cost is separately recorded.

Let `F_fixed` be the measured work outside factor transport and factor-slice contractions, and `F_factor=F_total-F_fixed`. Tabulate `F_fixed + a*F_factor` for `a={0,0.10,0.25,0.50}`. These are counterfactual cost calculations, **not feasible algorithms or lower bounds**. They answer whether even eliminating this particular work would reach 25%/15% utilization. If a necessary cost fraction is `a=(u_target*B-F_fixed)/F_factor<0`, add the covariance stage in Section 14; factor compression alone cannot meet that cost target with the other operations unchanged.

### P7-01b: saved-prediction decomposition and state diagnostics

Use the correctly paired uncompressed reference predictions from `research/phase6/predictions/P6-03_K3-simple_preds.npy` and its source/receipt. Verify row order, dtype, and truth hashes. For every layer/network define `a=reference-truth`, `b=candidate-reference`:

```text
MSE(candidate,truth) = mean(a*a) + mean(b*b) + 2*mean(a*b)
```

Report all three terms. The ~4% difference between reference and candidate MSE does not bound compression distortion: a large distortion can partly cancel reference error. Quantify this before calling the recurrence saturated.

Save pre- and postactivation `mu,C,d,S,c4`, factors/provenance or compressed state, and actual gates at layers `{3,6,10,13,14}` from a separate offline capture of the fixed reference/control. Stream one network at a time; save one state at a time and release it. Never keep eight full factor trajectories in RAM. On each compression candidate record immediate diagonal/slice error, independent contraction-probe error, next-layer slice error, final reference distortion, and actual truth error. The last two choose whether a locally better summary is useful.

Mandatory `P7-01-intervene`: at layers `{6,10,13}`, replace exactly one of `{mu,C,K3,c4}` in the control state by the corresponding correctly paired **uncompressed approximate state**, then continue the suffix with control rules. This is 12 offline diagnostics, eight networks each. Transfer full K3 factors with consistent provenance and ordinary future births; do not replace only its diagonal while claiming full K3 replacement. These are reference interventions, **not true-moment oracles**, and omit the cost of obtaining the donor. Their purpose is to identify compression/port interactions; a failed intervention does not rule out joint corrections.

No new high-sample Monte Carlo moment bake is required here. Existing high-accuracy moments may be used only with verified pre/postactivation semantics and sampling uncertainty. Missing true mixed moments remain an evidence gap. First-layer exact Gaussian mean versus baked truth provides a noise sanity check for that layer only; do not extrapolate it into a final-layer noise floor.

## 6. Mathematical primitives shared by the new representations

Use the unchanged Wick coefficients, `pk -> k` conversion, mean/covariance update, and scalar K4 update from `CONTROL65`, except where a stage explicitly changes them. In column notation `W=mlp.weights[l].T`; row samples propagate as `H@mlp.weights[l]`. Layer indices are zero-based, pruning/recompression happens after nonlinear layer `l`, and no future state is constructed after the terminal layer.

For `T(u,v)=Sym(u,u,v)` normalized over three terms:

```text
T[i,j,k] = (u[i]*u[j]*v[k] + u[i]*v[j]*u[k] + v[i]*u[j]*u[k])/3
K = sum_r T(U[:,r],V[:,r])
d(K) = sum(U*U*V,axis=1)
S(K) = Z((2*(U*V)@U.T + (U*U)@V.T)/3)
K(a,a,a) = sum_r (a@U[:,r])**2 * (a@V[:,r])
K(a,a,b) = sum_r ((aU)**2*(bV) + 2*(aU)*(aV)*(bU))/3
K(a,b,:) = sum_r [U_r*((aU_r)*(bV_r)+(aV_r)*(bU_r))
                        + V_r*(aU_r)*(bU_r)]/3
transport_W(K) = sum_r T(W@U_r,W@V_r)
```

`Z` sets a matrix diagonal to zero; `S[i,j]=K[i,i,j]`. All tensor sums and numerical scalar reductions in candidates remain metered.

For two atoms, with `A=u·p, B=v·q, C=u·q, D=v·p`:

```text
inner(T(u,v),T(p,q)) = (A*A*B + 2*A*C*D)/3
norm2(T(u,v)) = ((u·u)**2*(v·v) + 2*(u·u)*(u·v)**2)/3
```

Define the **diagonal-slice tensor** `DS(d,S)` by factors:

```text
U_DS = I
V_DS = diag(d) + 3*S.T
```

It reproduces `d,S` exactly and has zero all-distinct entries. Therefore `R=K-DS(d(K),S(K))` has zero repeated entries, including its diagonal. DS is a linear projection in the current neuron basis, not a rotation-invariant closure. A transformed DS tensor is generally not DS in the new basis.

For a Tucker tensor with symmetric core `G` of shape `(k,k,k)` and basis `Q` `(n,k)`:

```text
K_T[i,j,t] = sum_abc Q[i,a]*Q[j,b]*Q[t,c]*G[a,b,c]
L[i,c] = sum_ab Q[i,a]*Q[i,b]*G[a,b,c]
d_T = sum(L*Q,axis=1)
S_T = Z(L@Q.T)
transport_W: Q <- W@Q, G unchanged
diagonal_gate_g: Q <- g[:,None]*Q, G unchanged
K_T(a,b,:) = Q @ contract(G,Q.T@a,Q.T@b)
project_P(K_T) = G x1(P.T@Q) x2(P.T@Q) x3(P.T@Q)
```

No orthonormality is needed to *evaluate* these formulas; the new projection basis is orthonormal. Evaluate `L` in blocks of at most 64 neurons using a `(block,k,k)` temporary, then a matrix product with reshaped `G`. Never form `(n,n,k,k)` or `(n,n,n)`. Project a factor tensor with `A=P.T@U,B=P.T@V` and sum `Sym(A_r,A_r,B_r)` into a small core, accumulating columns in blocks of 64. Old-core transport costs order `k^4` through sequential mode products, not a six-index naive contraction.

Required operator API in the Phase 7 template: `diag`, `repeat`, `contract2`, `project`, `linear_transport`, `gate_transport`, and sum/difference of components. Implement explicit formulas for factor, DS, and Tucker components. A sum is evaluated componentwise; it need not concatenate every component into huge factors.

**Do not apply one random projection to U and V and claim it preserves the cubic tensor.** Matrix identities for `UV.T` do not apply to `sum u*u*v`; cross terms appear. Every proposed compression below has a separately stated target and error.

Tiny identity suite: widths `{3,5,8}`, seed `7001`, ranks `{0,1,5}`, asymmetric transports, signed factors, antipodal cancellation, zero tensors, and cores with negative entries. Compare each operator to independently materialized dense tensors. Include full-basis `k=n` recovery, DS+residual decomposition, and `S` transpose checks. Small recurrence fixtures `(n,depth)={(8,1),(16,2),(24,8),(32,16)}`, seeds `{6,17,29}`, test schedules including the penultimate layer. Exact formulas require the Section 4 float64 tolerance; truncated representations require their stated invariants, not equality to the original tensor.

## 7. P7-02: exact savings and corrected Phase 6.5 controls

Keep `CONTROL65` as the comparison anchor throughout this stage; accumulate accepted exact changes as `EXACT7`. All P7-02a variants start from CONTROL65. P7-02b starts from their accepted combined implementation; P7-02c freezes the best verified exact implementation after P7-02b.

### P7-02a: exact structured births and boundary handling

Mandatory variants: `structured`, `reuse`, `combined`.

Carry newborn path `(A2,3*diag(h))` and slice `(I,A_ds)` as descriptors until the next transport:

```text
W@I = W
W@(3*diag(h)) = 3*W*h[None,:]
path transported factors = (W@A2, 3*W*h[None,:])
slice transported factors = (W, W@A_ds)
```

Preserve logical column order, source-neuron pairing, and the exact keep set. Compute selection scores from descriptors when a current birth is protected; when it becomes eligible, give it the same score as its dense form. At every event assert logical rank equals provenance length even when some factors are stored separately. For terminal pruning compare split and concatenated tiny implementations under identical protected-column rules.

`reuse`: only profile-identified repeated numerical operations, fourth-metric products, powers, and avoidable materializations. Preserve the existing scalar-K4 sum and terminal contractions. Do not claim float32 initialization or the already implemented terminal specialization as a new saving. Reconcile every claimed delta with measured FLOPs.

### P7-02b: metered Strassen screen, limited scope

This is a bounded exact-arithmetic screen for large factor products, not the primary breakthrough hypothesis. Only after ordinary structured products pass, replace dense matmuls in old factor transport and repeated-slice contractions. Keep covariance and small products conventional initially. For `(m,k)@(k,p)`, split each even dimension in half; pad each odd dimension by one if needed and charge padding/cropping. Use standard seven products:

```text
M1=(A11+A22)@(B11+B22); M2=(A21+A22)@B11
M3=A11@(B12-B22);       M4=A22@(B21-B11)
M5=(A11+A12)@B22;      M6=(A21-A11)@(B11+B12)
M7=(A12-A22)@(B21+B22)
C11=M1+M4-M5+M7; C12=M3+M5
C21=M2+M4;       C22=M1-M2+M3+M6
```

`strassen1`: one level only when every matrix dimension >=512. Keep conventional matmul otherwise. Test non-square and odd-sized tiny identities. If this saves >=5% of total billed work, passes float32 parity, and does not increase maximum residual by >0.05 s, run `strassen2` with at most two levels and leaf minimum dimension 256. Otherwise record the two-level gate. No deeper recursion. A theoretical multiplication count is not evidence of a cheaper metered call. Sum/copy/cancellation overhead and residual time may kill this branch.

### P7-02c: corrected comparison set

On `EXACT7`, with canonical integer maps and an effective-state trace, run these mandatory repaired controls. Reuse a receipt only if it matches **actual source behavior** and parent.

- Exact atom-norm selector at `[6,10,13]`, retentions `[.8,.9,.8]`.
- Corrected diagonal-Q extension: `(start,eta)={(13,.5),(13,1),(11,.5),(11,1),(7,.5),(7,1)}` using Section 13 of Phase 6.5 verbatim, with q generated after `start` and first consumed at `start+1`. Include `eta=0` parity to the parent.
- Corrected last-event retentions `.75,.85` at layer 13; leave layer 10 at `.9`.
- One repaired ordinary-CV diagnostic at `(N,alpha)=(1024,.05)` on this parent. Run its other five original grid points only if it improves the mean product or if its measured bias/variance decomposition predicts a positive best fixed blend. This limited repair does not gate the independent residual-CV stage later.

Do not conflate these parent-corrected measurements with old mismatched ones. The richer compression stages below start from `EXACT7`; a winning q policy is saved separately as `Q7` for later controlled combinations, so compression's effect remains identifiable.

## 8. P7-03: paired and response-aware selection at smaller ranks

Purpose: implement the missed selectors correctly and establish deletion controls for every subsequent compression experiment. This is a bounded stage, not another broad schedule sweep.

Use total post-compression rank caps `{4n,6n,8n}` after `[6,10,13]`, protecting the `2n` ordinary newborn columns. Compare these four policies at each cap: `upper_individual`, `exact_individual`, `exact_pairs`, `response_pairs` (12 configurations). A cap is total columns, including protected and corrective components; never silently exceed it. Before the first cap keep the parent's state/pruning policy. Once this stage starts caps replace retention at those events.

Pair ordinary path and slice births with the same `(birth_layer,source_neuron)`; synthetic summaries are singleton groups with explicit provenance. Pair score is the squared norm of the **signed tensor sum**, including twice the cross inner product in Section 6. Select whole pairs; fill a remaining odd column slot only from true singleton groups, or leave it unused and record the count. Do not split a pair to manufacture a cost match. All candidates report actual cost as well as rank.

For `response_pairs`, compute a cheap diagonal Gaussian pilot online:

```text
m0=0, v0=1
for layer l: mpre=W_l@m; vpre=(W_l**2)@v
             sigma=sqrt(max(vpre,1e-12)); a=mpre/sigma
             g_l=Phi(a)
             m=sigma*phi(a)+mpre*Phi(a)
             v=(mpre**2+vpre)*Phi(a)+mpre*sigma*phi(a)-m**2
b[depth-1]=ones(n)
for l from depth-2 down to 0:
    b[l]=(W_(l+1)**2).T @ (g_(l+1)**2 * b[l+1])
normalize b[l] by max(mean(b[l]),1e-12)
D_l=diag(sqrt(max(b[l],1e-12)))
```

Score atoms/pairs as `T(D_l u,D_l v)` using the same norm. This is a declared diagonal approximation to a suffix sensitivity metric, not an exact adjoint or a bound on output error. Pilot, backward pass, and ranking are charged. Keep the norm-only paired comparator. No dense suffix Jacobian or per-group suffix rerun is part of this selector.

For each policy qualifying under the expansion gate, test `{3n,5n}` caps at the same events, then the best supported cap at events `[4,8,12]` (at most three extra configurations per policy). Save the best supported selector as `SELECT7`; if none wins, retain `EXACT7`. Preserve the cheap losing controls for matched comparisons below.

## 9. P7-04: signed group reweighting with current and next-layer contractions

This stage is mandatory even if slice compensation failed in Phase 6.5. It fits the *analytic current tensor*, never labels. It does not add an `n`-column correction block.

Start at layer 10 only, using total caps `{4n,6n}`, protecting ordinary newborns and using the best supported Section 8 selector (default exact pairs if no selector qualifies). Form old survivors and partition selection units in stable provenance order into `G=min(32,number_of_units)` consecutive nonempty groups of nearly equal size. Use ridge `{1e-3,1e-2}` and two feature banks `isotropic` and `next_mixed`: eight fits, plus the two unweighted same-survivor controls. Each target contains **all eligible old atoms before pruning**; protected newborns appear in neither fit side.

Use 128 fit and 128 independent check rows. An isotropic row is `f_r(z)=K_r(z,z,z)`, `z~N(0,I/n)`. For `next_mixed`, 64 rows are the same isotropic probes and 64 are contractions in next-layer weight-row directions. Draw a seeded permutation of output row indices; take 32 distinct `i` and 32 disjoint `j` for fit, separate sets for check. Use one diagonal row `K(w_i,w_i,w_i)` and one mixed row `K(w_i,w_i,w_j)` for each pair, with `w_i=W_(l+1)[i,:]`. Charge random permutations/index generation through the permitted API and both old/full and survivor projections. The prescribed events always have a next layer; disable the fit at terminal depth.

For tiny contract fixtures with n<128, fill each index bank by cycling fresh seeded permutations until it reaches its required length; repetition is allowed only for these small widths. Those checks verify plumbing/algebra, not independence of weight-row directions. Full-width evaluations use the disjoint index sets above.

Let `F[t,g]` be the sum of group contractions, `y[t]` the all-old target. Normalize and solve in metered arithmetic:

```text
eps=max(1e-12,1e-6*sqrt(mean(y*y)))
scale[t]=max(sqrt(sum_g F[t,g]**2),eps)
A=F/scale[:,None]; b=y/scale
delta=solve(A.T@A/128 + lambda*I, A.T@(b-A@ones(G))/128)
weights=1+delta
```

This is ridge toward **one**, not zero. On independent check rows compare `sum((F_check@weights-y_check)**2)` with all-one weights. Reject the solve if nonfinite, `max(abs(weights))>4`, or check error increases. Use all-one weights on rejection and keep its billed solve/probe cost. Apply each accepted weight to the group's **V columns only**. Negative weights are allowed; do not multiply U and V together or clip an unstable solution and call it the same solve.

Record fit/check errors, extrema, rejected solves, slice and next-layer errors, tensor-to-reference final distortion, actual MSE, and compute. Compare to both the exact same survivors with all-one weights and `EXACT7`. If a combination meets the expansion gate, repeat it at `[6,10,13]` with fresh probes at each event; add group count 64 with 256 fit/check rows at the successful cap/ridge/feature bank. At most two such expansions per surviving bank, chosen by product. Use all three offsets for survivors. Save `REWEIGHT7` only through actual product gates.

## 10. P7-05: fixed-memory tensor closures and future-basis reset

This stage tests whether growth in factor count is necessary. It includes deliberate failure boundaries; do not infer that all compact representations fail if the slice-only boundary loses.

### P7-05a: recent explicit cohorts plus one DS summary

Use exact ordinary births and keep the newest `h` birth cohorts explicitly. At every nonlinear layer after `start`, merge all older explicit cohorts and the previous summary into **one** new DS summary. If `K_target` is the uncompressed nonlinear target from the *current approximate state* and `K_recent` contains the ordinary births with age `<h`:

```text
K_old=K_target-K_recent
K_next=K_recent+DS(d(K_old),S(K_old))
```

The summary is transported as a tensor at the next layer; never transport only its recorded slices. It is not protected as a newly born cohort and cannot accumulate into many summaries. Track ordinary cohorts by their original provenance; if the prefix already pruned some columns, keep only those actually present. Never recreate missing columns for free when switching policy. No norm pruning is used after this policy starts. Before `start`, use `EXACT7`. Grids: `h={0,1,2,4}`, `start={3,7,11}` (12 candidates). For `h=0`, all target information is summarized into DS; mark it as the all-distinct-discarding boundary. Charge summary construction and next transport. At nonterminal layers the post-compression rank is at most `2hn+n`; early layers can have fewer cohorts. The same prefix/provenance rule applies to Section 11.

### P7-05b: preactivation reset while protecting the nonlinear path birth

At a reset layer, first transport the whole current state and obtain its exact `d_pre,S_pre`. Replace that transported K3 by `DS(d_pre,S_pre)` **before** nonlinear factor construction, while keeping the existing marginal/mixed moment calculations based on the same `d_pre,S_pre`. The current-layer means, covariance, and target `k3,k21,c4` are unchanged by the reset itself; the future tensor differs.

After the reset, gated DS stays DS, so combine it with the ordinary slice birth exactly. Let the ordinary nonlinear path tensor be `K_path=sum_j T(A2_j,3*h_j*e_j)`. The output representation is:

```text
K_next = K_path + DS(k3-d(K_path), k21-S(K_path))
```

Thus total rank is `2n`, and ordinary path all-distinct information survives. This is distinct from postactivation DS-only reset. The full input transport already paid at the reset layer is not refunded. Between resets use the ordinary recurrence with exact births; do not use retention pruning inside this experiment after its first reset.

Grids: reset period `{1,2,4}`, first reset `{3,7}`; reset at `first,first+period,...` strictly before terminal (six candidates). Record matching current-layer invariants and next-layer error. Save `MEMORY7` from the 18 candidates by the ordinary gates. For the best two qualifying memory policies, test one earlier start (subtract 2, floor 1) and one later start (add 2, cap depth-2), keeping other settings fixed.

The Section 11 hybrid is mandatory regardless of this stage's outcome: it restores an all-distinct state that this stage discards.

## 11. P7-06: DS plus a compressed all-distinct tensor — main breakthrough lane

### P7-06a: representation and exact construction of the approximation

Keep `h` recent ordinary birth cohorts explicitly, plus a Tucker tensor and one DS tensor for older information:

```text
K_stored = K_recent + Tucker(Q,G) + DS(d_c,S_c)
```

This stores order `hn**2 + n*k + k**3 + n**2` numbers instead of factors growing with depth. It still uses dense covariance and DS transport; it is not an `O(nk)` estimator. The recurrence may have high multilinear rank, so this is a hypothesis to test, not an assumed property.

At each nonlinear step obtain `K_target` from the current compressed state by the **same** factorized nonlinear rule as the parent:

```text
K_pre = transport_W(K_stored)
compute d_pre,S_pre and the unchanged moment algebra -> k3,k21,g,h_wick
K_gate = gate_transport_g(K_pre)
K_path = sum_j T(A2_j, 3*h_wick[j]*e_j)
K_birth_slice = DS(k3-d(K_gate+K_path), k21-S(K_gate+K_path))
K_target = K_gate + K_path + K_birth_slice
```

Here `h_wick` is the second Wick coefficient, not memory length `h`. Components remain symbolic factor/DS/Tucker sums. Propagate each recent cohort explicitly, label the two ordinary births with their layer/neuron, and separate `K_recent` at the requested age cutoff. Compression operates on:

```text
K_old = K_target-K_recent
d_old=d(K_old); S_old=S(K_old)
R_old=K_old-DS(d_old,S_old)
```

To find a basis, draw independent matrices `Z1,Z2` with `p=k+8` columns, each entry `N(0,1/n)`. Build `Y[:,t]=R_old(Z1[:,t],Z2[:,t],:)` with the Section 6 operators. Take the first `k` left singular vectors of **thin** `svd(Y,full_matrices=False)` as `Q`; use a deterministic column-sign convention and stable ordering. If Y is identically zero, set a zero core and skip its decomposition. If numerical rank is smaller than k (singular values below `1e-6 * largest`), use the smaller active rank and record it; zero directions carry no information. No complete n-by-n SVD is requested.

Project and restore the slices:

```text
G = project_Q(R_old)
G = average of its six index permutations
K_T = Tucker(Q,G)
d_c = d_old-d(K_T)
S_c = S_old-S(K_T)
K_next = K_recent+K_T+DS(d_c,S_c)
```

The new tensor reproduces `k3,k21` in the current layer exactly up to rounding. Its old all-distinct component is approximate. **The residual approximation is `K_T-DS(d(K_T),S(K_T))`, not K_T alone.** Use this expression for residual reconstruction metrics; the DS correction removes the repeated entries that projection reintroduced.

Do not first construct a full uncompressed trajectory and then compress it while claiming a cheap prediction. Every later target must come from the candidate's own previously compressed state. Offline exact-state snapshots are diagnostic inputs only. This distinction is enforced by a standalone candidate with no file access to the snapshots.

### P7-06b: mandatory grids and controls

Start with one-time replacement at layer 10 to debug the representation. Use `k={16,32,64}`, `h=0`, then follow ordinary transport/nonlinear births on the suffix **without any further pruning or recompression** (three experiments). Retain the Tucker component and add subsequent explicit births; a factor-column selector must not be applied to a core. Add one matched no-replacement control with that same unpruned suffix, as well as comparison to `EXACT7`. All three proceed to the full eight once the tiny/math and resource checks pass. This intervention tests recoverable information before introducing recurrent approximation.

The **mandatory recurring grid** is `k={16,32,64}`, `h={0,1}`, `start={3,7}`: 12 candidates. Before start use `EXACT7`; from start onward compress after every nonterminal layer. Use scalar K4, unchanged covariance, and no norm pruning once this policy starts. Protecting recent cohorts applies to the hybrid's explicit memory, not to the old tensor being sketched.

Controls, on `start=3,h=0`:

1. DS-only (`k=0`), reusing the matching P7-05 receipt if bytes/semantics agree.
2. Pure Tucker of the **full** K_target, `k={32,64}`, without slice restoration; basis comes from full K contractions. This tests whether protecting slices matters.
3. Hybrid with `k=32`, a fixed random orthonormal basis drawn once per prediction and reused; projections remain network-specific and metered. This checks whether online range finding earns its overhead.
4. Hybrid with `k=32`, basis and core formed from full K_old rather than R_old, followed by the same slice restoration. This compares residual-focused and whole-tensor subspaces.

No control is allowed to reuse the tested candidate's basis or projected core for free. Count its actual dataflow. The h=0 versus h=1 comparison also tests whether protecting recent births is worth its extra matrices.

### P7-06c: required diagnostics and triggered extensions

At each recompression log active k, old component sizes, DS/core norms, singular values, and independent check-probe error using 32 fresh pairs `a,b~N(0,I/n)`. Check tensor-vector contractions, not only cubic scalar probes; the former contain n responses per pair. Report immediate slices (which should be restored), next-layer slices, and final truth/reference errors. Probe observations within one network are not independent networks.

For each of the two best hybrid policies meeting the Section 4 expansion gate:

- Test `k=96` at the same h/start if the 16-to-64 progression improves mean raw MSE or final reference distortion by >=20%; otherwise skip this larger core with the actual trend recorded.
- Test `h=2` at the best k/start if h=1 improves raw error by >=10% over h=0 at matched k/start. Forecast the extra cohort transport first.
- Test refresh every second layer: compress at `start,start+2,...`; on intervening layers retain the existing core and exact new birth components, then absorb them at the next refresh. Do not silently project twice or leave an unbounded component list.
- Test a depth-varying k schedule: k/2 (minimum 8) before layer 8, k from layers 8–11, and `min(2k,96)` from layer 12 onward. The same h/start applies. Rank changes occur only at recompression. This is one specified allocation variant, not a grid over all layers.

For the best qualifying fixed-rank hybrid, one optional **online check-driven rank** variant uses k `{16,32,64}`. Generate/projection-build the k=64 basis/core once at each refresh, evaluate prefixes of that basis, and choose the smallest whose residual check error is no more than `1.10` times the k=64 error. Use 32 independent check pairs, choose only on analytic tensor contractions, and charge the full k=64 build and all checks. It can save future transport, not projection work already paid. If this overhead cannot be recovered, record failure. Do not use true output means to choose k inside predict.

Run all applicable salts before selecting `HYBRID7`. A flat small-core spectrum or poor k=64 result is evidence against this low-rank approximation at these ranks, not proof that every tensor sketch is impossible. If the hybrid fails, the reweighting, finite-memory, and randomized-tail branches still stand independently.

## 12. P7-07: randomized atom-tail compression with a deterministic head

This is a different mechanism from both input Monte Carlo and low-rank Tucker projection. It preserves the discarded tensor **in conditional expectation at the compression event**. It does not promise an unbiased final nonlinear estimate.

At each compression event, protect ordinary newborns. Keep a deterministic head of old selection units using the best supported selector. Let the remaining tail be individual atoms, or intact ordinary pairs in the paired variant. For unit `j` with tensor `T_j`, let `a_j=sqrt(max(norm2(T_j),0))`. Use exact pair norm when applicable. Investigate materially negative computed squared norms; small cancellation roundoff may be floored only at the documented float32 tolerance. Skip an empty tail. If all computed scores vanish for a nonempty tail, sample it uniformly rather than assuming its tensor is exactly zero; otherwise define:

```text
p_j = 0.95*a_j/sum(a) + 0.05/number_of_tail_units
draw K independent tail units J_s with replacement according to p
K_tail_hat = (1/K)*sum_s T_(J_s)/p_(J_s)
```

Use metered uniform draws and cumulative-probability/searchsorted sampling. Apply `1/(K*p_j)` to V only; for a pair, scale both member V columns by that same weight. Keeping duplicate sampled units is permitted and is the default; do not assume deduplication is free. The expectation identity follows from summing over the categorical draw. Future nonlinear transformations generally introduce bias, and variance may be prohibitive.

Mandatory layer-10-only grid: total output columns `{4n,6n}`, deterministic old-head fraction `{0.5,0.75}`, units `{individual,pairs}`: eight configurations. Of the columns remaining after the `2n` newborns, allocate the specified fraction to the head, rounded down to the unit size, and use all remaining unit slots for draws. In paired mode a draw costs two columns; enforce exact total cap. Compare to unweighted deterministic survivors at the same total budget and to `EXACT7`.

Tiny stochastic identity check: for a fixed signed tensor at n=5, verify the analytic expectation by summing over all categorical outcomes for K=1; separately verify sampled mean/variance with enough offline repetitions and a stated Monte Carlo interval. Do not require a random realization to equal the original tensor.

For every policy meeting the expansion gate at offset 0, run the two other offsets; for the best two salt-supported policies, repeat compression at `[6,10,13]` and test cap `3n` at the same head fraction. Do not expand a policy whose apparent win disappears on additional salts. Save `TAIL7` only by actual products and numerical stability.

## 13. P7-08: richer fourth-order state with a complete projection rule

This stage is mandatory and independent of compression success. It tests an explicit approximation that uses more of k4/k22 than the scalar or diagonal trace-harmonic state. It is **not full fourth-cumulant propagation**, and it does not require inventing missing off-diagonal fourth-order partial traces.

### P7-08a: state and normalization

Define `I4(M)` for symmetric M:

```text
I4(M)[i,j,k,l]=(M[i,j]*M[k,l]+M[i,k]*M[j,l]+M[i,l]*M[j,k])/3
K4_state = c4*I4(I) + 3*sum_t lambda[t]*I4(diag(v_t)) + Diag4(delta)
Diag4(delta)[i,i,i,i]=delta[i]; other entries zero
```

The vectors v_t are orthonormal projection directions from a symmetric matrix decomposition, but lambdas can be negative. This is a signed cumulant model; clipping negative eigenvalues would change the method. Here `r4` is the number of spectral modes, distinct from K3 rank.

After the existing nonlinear conversion, construct the symmetric zero-diagonal `k22` using the same pK algebra as the Phase 6 source. For complete clarity:

```text
a=w(1,2); b=w(2,2); H=0.5*C0**2+0.25*s22_pre
pk22=C0*(a[:,None]*a[None,:])
      +0.5*(S_pre.T*(a[:,None]*b[None,:])
            +S_pre*(b[:,None]*a[None,:]))
      +H*(b[:,None]*b[None,:])
k22=Z(sym(pk22-2*(pk1[None,:]*pk21+pk1[:,None]*pk21.T)
              -2*pk11**2+4*pk1[:,None]*pk1[None,:]*pk11))
c4=3*(sum(k4)+sum(k22))/(n*(n+2))
B4=Z(k22-c4/3)
diag(B4)=(k4-c4)/3
```

`C0=Z(C_pre)`, `S_pre=K3_pre[i,i,j]`. The `pk11` quantity is already the mixed power cumulant used by the parent; do not subtract a second mean outer product. Verify the matrix row sums against the parent's scalar-K4 sum before using this extension.

Approximate B4 by a signed rank-r4 matrix. Use `p=min(n,r4+8)`, Gaussian `Omega` `(n,p)` with variance 1/n, and `Y=B4@Omega`. For the specified one-power variant, replace Y by `B4@(B4@Y)` (charge both products). Use thin SVD of Y and retain left singular directions above `1e-6` times its largest singular value to form Q; if Y is zero use no directions. Compute `H4=sym(Q.T@B4@Q)`, diagonalize H4, select up to r4 eigenvalues by descending absolute value with a stable tie rule, and set `v=Q@eigenvectors_selected`. Keep signed lambdas. Do not use an n-by-n eigensolve in production. Fix each direction's sign by making its largest-absolute entry positive, breaking a tie at the lowest index. Apply the same sign convention to Section 11 bases.

For r4=0 skip decomposition. Define:

```text
delta=k4-c4-3*sum_t lambda[t]*v_t**2
```

Before damping, this restores the exact current-layer k4, while approximate off-diagonal k22 is `c4/3+sum lambda*v_i*v_j`. The state discards other fourth-order information. With full eigendecomposition and r4=n on tiny fixtures, it reproduces k4 and k22 exactly, but still does not reproduce an arbitrary full K4 tensor.

Optional damping beta scales **both** lambdas and delta after constructing them; c4 is unchanged. At beta=0 this is precisely the scalar parent. At beta=1 the diagonal k4 is preserved. beta=.5 deliberately gives up half the residual correction and is not an exact projection.

### P7-08b: transported slices without a dense fourth tensor

At a linear layer, `M=W@W.T`, `m=diag(M)`, and for each mode `B_t=(W*v_t[None,:])@W.T`, `b_t=diag(B_t)`:

```text
s4_pre = c4*m**2 + 3*sum_t lambda[t]*b_t**2 + (W**4)@delta
s22_pre = Z(c4*(m[:,None]*m[None,:]+2*M**2)/3
            +sum_t lambda[t]*(b_t[:,None]*b_t[None,:]+2*B_t**2)
            +(W**2*delta[None,:])@(W**2).T)
```

Accumulate one B_t at a time; never retain an `(r4,n,n,n)` intermediate. At the terminal layer only s4 is consumed: use `b_t=(W**2)@v_t` and row norms m, avoiding the full B_t matrices and s22. Feed the new s4/s22 into **all** the existing nonlinear uses, including mean, variance, covariance, K3 births, and the next fourth-state projection. Updating only the terminal mean is a separate ablation, not the full recurrence.

Initialize residual modes and delta to zero. For a suffix starting after nonlinear layer `start`, first generate them at that layer, and consume them at `start+1`. Tiny tests must compare dense four-mode transport with these formulas for signed lambda, arbitrary delta, and non-orthogonal W. Use r4=n projection checks at n<=8, scalar beta=0 parity, and beta=1 k4 preservation.

### P7-08c: experiments

Use `EXACT7` first: `r4={0,1,2,4}`, `start={11,13}`, beta=1, zero power iterations (eight candidates). The r4=0 case is **scalar plus diagonal residual**, not the parent; include beta=0 as parity control. If a configuration improves mean raw error by >=5% or mean product by >=2%, run beta=.5 and one-power variants at that same r4/start. For the best two qualifying configurations, test start=7. Do not infer full-width sensitivity from tiny tensor differences.

Compare every surviving extension to the scalar parent and to a scalar-K3 frontier point within 5% of its billed cost, if one exists. If none exists, add at most two scalar rank comparators total using +.05 and +.10 at the parent's earliest pruning event, clipped to .95; for a compact representation use its next declared rank instead. Report unmatched costs honestly. Promote on products, not accuracy alone.

Select up to two cheap K3 parents for combination: the lowest-product supported compact candidate and the lowest-cost point satisfying the accuracy-preserving frontier. Add the best two supported fourth-state policies (including corrected Q7 if it survives) to those parents: at most four combinations. Re-evaluate all feedback channels; do not add separately measured gains as percentages. Save the resulting `ANALYTIC7` and retain an unextended compact comparator.

## 14. P7-09: covariance compression only when profiling justifies it

Run this stage if covariance/metric transport constitutes >=10% of billed work in the cheapest supported compact candidate, or P7-01's counterfactual shows factor-only savings cannot reach the utilization target. Otherwise record `SKIPPED_GATE` with the profile values; do not spend a phase optimizing a negligible component.

This is a signed diagonal-plus-low-rank covariance approximation that preserves current marginal variances:

```text
K=Z(sym(C))
approximate K by Q*diag(lambda)*Q.T using symmetric randomized projection
d=diag(C)-sum_t lambda[t]*Q[:,t]**2
C_hat=diag(d)+Q*diag(lambda)*Q.T
V=W@Q
C_pre=(W*d[None,:])@W.T + (V*lambda[None,:])@V.T
```

Use the Section 13 matrix range/decomposition recipe with ranks `{16,32,64}`, oversampling 8, zero power iterations, and largest-absolute eigenvalues. It is not guaranteed PSD. Preserve the exact diagonal at compression; do not clip signed modes or d and hide an instability. Reject a new variance below `-1e-6*max(1,mean(abs(variance)))` or a covariance whose negative eigenvalue is materially outside the parent's diagnostic range on tiny/offline checks. The inherited tiny variance floor stays in place and activation count is logged.

Grid: ranks `{16,32,64}`, start `{3,7}`, recompress after every nonterminal layer (six candidates). Scalar K4's M and other channels are unchanged. At the terminal layer use the corresponding variance-only contraction. The actual saving replaces dense `W@C@W.T` with one diagonal transport plus low-rank terms, but decomposition may erase it. Charge the full range finder and projection each layer. If any qualifies, test refresh every second layer for its best rank/start; on intervening steps maintain a dense C and use ordinary transport until recompression, rather than pretending dense updates stay diagonal-plus-low-rank.

## 15. P7-10: higher-order residual control after changing the analytic center

Use the cheapest supported analytic candidate with raw error <=`1.25*CONTROL65` as parent; if none exists, use `ANALYTIC7` or `EXACT7`. This stage is mandatory because the quadratic residual is a different control from the failed small linear add-on. Gate its expansion on measured variance and cost.

At the terminal layer preserve analytic preactivation mean m, variance v, `g=Phi(m/sqrt(v))`, and `h=phi(m/sqrt(v))/sqrt(v)`. Generate ordinary standard-Gaussian antithetic input pairs and forward them through the **actual whole network** to terminal preactivations Z and final outputs Y=relu(Z). No Gaussian resampling at layer 14 is equivalent to this experiment.

For each output coordinate define:

```text
V_linear = mean(Y) - g*(mean(Z)-m)
V_quadratic = mean(Y) - g*(mean(Z)-m)
              -0.5*h*(mean((Z-m)**2)-v)
prediction = (1-alpha)*mu_analytic + alpha*V
```

All centers are computed online. The quadratic coefficient is a Gaussian/Wick approximation to a useful control, not an exact Taylor remainder bound. If `delta_m=m-E[Z]`, `delta_v=v-Var(Z)`, then:

```text
E[V_linear]-E[Y] = g*delta_m
E[V_quadratic]-E[Y] = g*delta_m + 0.5*h*(delta_v-delta_m**2)
```

Therefore reducing sampling variance can increase center bias. Do not declare this unbiased. For each antithetic pair form the average of its two residuals, then estimate the variance of the overall residual mean as sample variance across independent pair averages divided by the number of pairs and output width. The blend variance term is multiplied by alpha squared. Do not count endpoints as independent. If true terminal variance is unavailable, report the analytic bias formula and sampling estimates with their uncertainty; do not fabricate an exact bias vector.

Grid: total endpoints `N={1024,4096}`, control `{linear,quadratic}`, alpha `{.02,.05,.10,.25}`: 16 candidates at offset 0. Use nested clouds from a common maximum half-cloud (2048 rows), charging generation in each point. alpha=0 is the unsampled parent and allocates no cloud. Forward in blocks of 512 endpoints if necessary for memory, maintaining pair identity for variance estimation. Charge all forwarding and reductions. Preserve earlier analytic output rows unchanged.

Run other offsets for qualifying points. If quadratic control lowers residual trace variance by >=50% versus linear at equal N but no tested alpha wins, calculate the fixed-grid product optimum from the saved predictions across the seven training networks in each LOO fold, restricted to alpha `[0,.25]`; report it as a fitting diagnostic. Only add a deployable common-alpha candidate if the LOO mean product improves by >=5% and 6/8 folds win; freeze the all-eight coefficient and disclose its fitting provenance. No neuronwise label-fitted control matrix is allowed. If residual variance remains too high, stop this branch rather than increasing N indefinitely.

## 16. P7-11: paired-resolution extrapolation and cheap accuracy corrections

### P7-11a: fixed-coefficient paired-resolution extrapolation

Compression can have systematic bias. Use saved predictions to screen **adjacent declared resolutions of the same family**, with all other settings identical: Tucker k pairs `(16,32),(32,64)`, memory h pairs `(1,2),(2,4)`, or cap pairs `(4n,6n),(6n,8n)`. Do not combine unrelated families under the name extrapolation.

For each available pair, with expensive/finer output p_H and cheaper/coarser p_L:

```text
p_ext=p_H+gamma*(p_H-p_L), gamma in {0.25,0.5,1.0}
```

Screen all declared pairs offline using **sum of their billed costs** as a conservative standalone implementation cost; this screen is not a free ensemble candidate. Also record a shared-prefix cost forecast only when source/state traces prove an identical prefix before the divergence. Rank pairs by product including the forecast cost. Do not assume a known asymptotic error exponent or target-fit gamma.

Implement at most the best three pairs whose projected product improves the parent by >=10%, 6/8 wins, worst ratio <=1.10, and resource forecast passes. For each, compute the common prefix once, copy the state with billed operations, then execute both suffixes and combine outputs. If no common prefix exists, bill two full passes. Earlier returned rows use the finer branch. This is a deterministic approximation; randomized pairs use common documented draws where appropriate and still require all salts. Repeat selection stability calculations using the actual measured combined cost. Keep only genuine end-to-end wins.

### P7-11b: bounded terminal Edgeworth check on the new regime

The Muse pilot already tested `k3**2*w6/72` near the uncompressed parent and found negligible gain. Do not rerun it unchanged. On a **new** best compact parent, save its terminal m,v,k3,k4 and calculate these additional formal terms offline:

```text
w_j = (-1)**(j-2) * He_(j-2)(m/sigma) * phi(m/sigma) / sigma**(j-1), j>=2
He_0=1; He_1=a; He_(j+1)=a*He_j-j*He_(j-1)
Delta = k3**2*w6/72 + k3*k4*w7/144 + k4**2*w8/1152
```

These coefficients follow by expanding `exp(k3*D**3/6+k4*D**4/24)` through second order. They omit genuine higher cumulants and mixed effects elsewhere in the recurrence; no claim of exactness follows. Tiny checks use analytic Hermite recurrence/derivatives and known smooth one-dimensional moment examples, not an unstable high-order finite-difference stencil as sole evidence.

If any fixed variant `{k3sq_only,all_three}` predicts >=1% raw improvement on the eight with 6/8 wins, implement that variant at the terminal only and charge it. Otherwise record `SKIPPED_GATE` with its actual tiny effect. Do not propagate higher-order terms into hidden layers without updating the matching power-moment algebra; that unsupplied recurrence is outside this stage.

## 17. P7-12: controlled combinations and final utilization frontier

Freeze all component winners before this stage. Keep exact arithmetic savings from P7-02. Select at most three **different mechanisms** from `{SELECT7,REWEIGHT7,MEMORY7,HYBRID7,TAIL7}`: the best product, best accuracy-preserving cost, and best raw error under the resource cap. Duplicates count once. Do not stack overlapping compression corrections on the same discarded tensor.

Run these combinations where components have separately qualified:

1. Best supported fourth-order policy on each selected K3 representation (reuse Section 13 matches).
2. Best supported covariance policy on at most two selected representations, including its measured fourth-order setting if present.
3. Best supported residual control on the best two resulting analytic candidates, retaining its fixed sample count/alpha and rerunning actual sampling/centers.
4. Best supported exact Strassen implementation on the new dominant products only if shape/meter/float32 checks still pass; a smaller core changes its economics.

Maximum 12 new combinations; materialize their exact IDs before execution. Only combinations of compatible states are allowed. If two methods operate on the same missing K3, choose one; do not double-count its compensation. A cheap method that misses the 5% raw guardrail can still win the product frontier, with that tradeoff reported.

For the best two combined configurations, test two neighbors each:

- factor cap: minus n and plus n, with protected columns counted;
- Tucker: the nearest lower/higher **already declared** k at the final two refreshes;
- finite memory: h minus one and plus one, clipped to `[0,4]`, only after layer 11;
- sampled tail: shift deterministic-head fraction by minus/plus .125 within `[.25,.875]` at fixed total cap;
- residual control: alpha times .5 and 1.5, clipped to `[0,.25]`.

Use the first applicable parameter in the order above; exactly two neighbors per parent, not a Cartesian grid. No further unrecorded tuning after these neighbors. Rebuild final per-network cost/error frontier and mark whether `1e-8`, `6e-9`, or `3e-9` was actually crossed, and under which validity status.

## 18. P7-13: final verification, report, package, and version control

Freeze finalist bytes and configuration before verification. Re-run the candidate-relevant tiny identities/contract checks and direct **unrelaxed** runner on all eight; all salts if stochastic. Record full metric vectors and exact source/archive hashes. Separate:

```text
historical official leaderboard evidence
new diagnostic projected products
new valid local runner results
future official submission results (none created by this plan)
```

If the local residual discrepancy persists, leave the preferred result `RESEARCH_ONLY`; preserve any verified official release as release champion. A marked research archive is allowed for review, but do not call it release-ready or silently submit to resolve uncertainty. A documented local failure does not erase historical official success, and historical success does not validate new bytes.

Required report sections:

1. Outcome first: measured best score, validity, baseline comparison, reached/missed targets.
2. Corrected Phase 6.5 audit, including string-key settings, unsupported switches, penultimate provenance, source races, and report omissions.
3. One row for **every materialized mandatory/triggered configuration**, including skips with numeric gates and incomplete rows. List planned-but-untriggered stages separately.
4. Complete paired metric vectors, all layers, FLOPs/utilization, residual/predict/setup times, peak memory or explicit missing measurement, rank/state history, all salts, worst regressions.
5. Cost profiles and reconciled savings: transport, contractions, projections/fits, new state, sampling, shared work. No credit for optimizations already in the parent.
6. Error decomposition into reference error, compression distortion, and cross term; current-slice fidelity versus next-layer/final error.
7. LOO selected IDs/fold values and coefficients where fitted; explicit eight-network adaptivity limitation.
8. Successful and failed mathematical invariants, API/numerical/resource failures distinguished from accurate but expensive methods.
9. Final frontier chart of raw MSE versus utilization with actual per-network-product aggregate labels; show hypothetical target curves separately from measured points.
10. What remains open: failure at the tested ranks is not a family-wide ceiling; a target not reached is not evidence of impossibility. State the next unresolved mechanism in concrete terms, without inventing a winning-method explanation.

For a single-file validated finalist, create a byte-identical copy at `candidates/estimator_p7_final.py`, then from `whest-starterkit`:

```powershell
uv run whest validate --estimator ../candidates/estimator_p7_final.py
uv run whest package --estimator ../candidates/estimator_p7_final.py --output research/phase7/release/submission_phase7.tar.gz
```

Verify archive contents and source hash. Include no datasets, true means, cached reference centers, instrumentation, or credentials. If a candidate requires legitimately shipped global parameters, package a minimal folder with those exact files and their provenance; no network-identity lookup or cached per-network prediction belongs in this plan's estimator.

Inspect `git status`, diffs, and remote branch state. Stage only Phase 7 sources/configurations, compact receipts, report, and plan amendments that were actually completed. Keep large generated states/predictions out of Git unless the repository's existing artifact policy explicitly requires them. Commit with a descriptive message and push to `main`. If main advanced, fetch and reconcile the new work without overwriting others; never force-push or reset unrelated changes. Do not stage the user's existing untracked PDFs, datasets, or Muse files merely because they are present.

## 19. Literal execution schedule and completion checklist

The order is intentional: the expensive representation work starts after the evidence audit and cheap controls, but does not depend on those controls winning.

| Order | Stage | Mandatory work | Conditional work |
|---:|---|---|---|
| 1 | P7-00 | Source/configuration audit; immutable baseline; capture/runner integrity | Replay caches only when integrity/settings fail |
| 2 | P7-01 | Cost profile; reference/error decomposition; 12 channel interventions | Existing true-moment diagnostics with verified pairing |
| 3 | P7-02 | Three exact rewrites; one-level Strassen; repaired selector/Q/neighbors/CV | Two-level Strassen; remaining repaired CV grid |
| 4 | P7-03 | 12 selector/cap variants | Smaller caps and one alternative event schedule per qualifying policy |
| 5 | P7-04 | Eight signed fits plus two unweighted controls | Multi-event fits, 64-group expansion, salts |
| 6 | P7-05 | 12 finite-memory closures and six preactivation resets | Two start neighbors for best two supported policies |
| 7 | P7-06 | Three one-time hybrid replacements and matched control; 12 recurring hybrids; five recurring controls including reused k=0 | Larger core/memory, refresh interval, depth allocation, one adaptive policy |
| 8 | P7-07 | Eight head-plus-random-tail variants | Multi-event, smaller cap, salts |
| 9 | P7-08 | Eight richer K4 variants plus scalar parity | Damping/power/earlier start/equal-cost controls and compact-parent combinations |
| 10 | P7-09 | Numeric profile gate recorded | Six covariance variants and one refresh extension if gate passes |
| 11 | P7-10 | 16 linear/quadratic residual-control variants | Additional salts and one LOO-supported alpha fit |
| 12 | P7-11 | Saved-output extrapolation and changed-regime Edgeworth screens | Up to three online extrapolations; gated terminal correction |
| 13 | P7-12 | Frontier selection and compatibility review | Up to 12 combinations and four final neighbors |
| 14 | P7-13 | Exact-byte checks, full report, release-status decision, commit/push | Package validated finalist or explicitly marked research archive |

This is a substantial serial research campaign, with more than 100 specified candidate settings plus diagnostics and triggered expansions. Count exact candidate IDs in `experiment_manifest.json`; do not use this approximate count as a completion criterion. Offline profile/intervention work can be long. Take the following approach after interruption:

1. Read this file, progress, latest ledger rows, and manifest hashes.
2. Verify parent source and effective settings for `next_action`.
3. Resume missing networks/salts/configurations rather than rerunning completed ones.
4. Complete the remaining mandatory stages and each triggered expansion.
5. End only with a self-contained evidence-backed report, or a concrete external obstruction that prevents all remaining useful work.

Do not spend the run merely searching for a named method or rewriting this plan. Do not stop at scaffolding. A failed first version requires routine debugging; a correctly implemented negative experiment requires a recorded result and continuation to the next independent mechanism. The desired outcome is a valid e-9-range estimator, but the obligation is rigorous execution and honest reporting, not a fabricated breakthrough.

## 20. Plan-author checks and evidence limitations

While drafting, the installed flopscope namespace was inspected and exposed `linalg.svd`, `qr`, `eigh`, and `solve`. This is API availability only; their dtype, shape behavior, metered cost, and current backend support still require the tiny execution checks in this runbook.

Independent ordinary-NumPy n=5 checks with seed 7001 verified the Section 6 mixed contraction and Tucker slice formulas, hybrid slice restoration, and Section 13 signed fourth-harmonic transport against explicitly materialized dense tensors. Maximum errors were `5.6e-16` for mixed contraction, `4.5e-16` for hybrid repeated slices, and `1.5e-13` for fourth-order transport. These are **plan-author algebra checks**, not production implementation tests, full-panel experiments, or evidence of a score improvement. The executing agent must implement and preserve its own reproducible independent checks.

Evidence priority: actual candidate bytes and complete unmodified receipts; verified paired predictions; effective configuration traces; then derived reports and historical prose. The user's screenshot describes the competitive direction, not the evaluation data for fitting. Public tensor literature supplies mathematical background, not a disclosed explanation of the leaderboard.
