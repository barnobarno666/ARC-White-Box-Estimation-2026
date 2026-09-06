# Phase 6.5 ideas: exploit the working joint-cumulant method

Prepared 6 September 2026. This is a research plan and brainstorming document, not an executed experiment report. Internal Phase 6.5 still targets competition **v2-phase2, width 1024, depth 16**. Layer numbers below are zero-based; layer 15 includes the terminal ReLU.

## 1. The direction I would take

**Keep the Phase 6 joint-cumulant engine. First make its useful information cheaper to carry; then spend selectively on information that it currently discards.** The best immediate bets are additional exact algebra, better rank allocation, and pruning that preserves important tensor slices. Selected fourth-order state is the higher-upside extension. A Monte Carlo add-on belongs behind an explicit cost-versus-error gate.

Phase 6 established two useful facts: the coupled recurrence changes the accuracy scale, and a better representation of the same recurrence can improve the ranked score substantially. We now have something worth exploiting. Reopening every Phase 4/5 method would dilute that advantage.

My suggested effort allocation is **50% exact implementation and rank frontier, 30% better compression, 15% selected fourth-order extensions, 5% a tightly bounded control-variate check**. These are research priorities, not promised gains.

## 2. The starting evidence

### Latest official result

The public leaderboard checked on 6 September shows `subarno_sadat_barno` with:

| Metric | Displayed official value |
|---|---:|
| Adjusted score | **4.35e-8** |
| Final-layer raw MSE | **6.82e-8** |
| Compute utilization | **0.6373366929** |
| Failed MLPs | **0** |
| Position at this snapshot | **44** |

The leading displayed score is approximately `3.0e-9`. The remaining adjusted-score gap is about 14.5-fold. The board establishes performance, not which algorithm anyone used. [Official leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards).

The local submission receipt records **#330018**, created at `2026-09-06T05:50:03.757Z`, matching the board row's displayed last-submission minute. Its packaged source is [estimator_p6_k3_twofactor_terminal.py](candidates/estimator_p6_k3_twofactor_terminal.py), SHA-256:

```text
8498085d5b90646c6d65ce23087b2e9331980a77cb44bfa6d762341f68dd05ac
```

This is the Phase 6.5 working incumbent. The source hash was checked while preparing this document. The saved receipt itself still says `submitted` with a null score; the **current board independently confirms the account result**, while the submission-detail page was unavailable. Preserve that distinction in the evidence ledger. [Submission receipt](whest-starterkit/research/phase6_conversion_fix/submission_receipt.json).

### What the local artifacts actually support

| Evidence | Finding | Consequence for Phase 6.5 |
|---|---|---|
| Uncompressed K3-simple | Eight-network mean raw MSE `3.63704648e-8` in the reference implementation. | A useful accuracy target for compression; not a proven lower bound or a valid floor-discounted submission. |
| Production K3 with pruning `[6,10,13]`, retention `0.62` | Raw MSE about `7.32e-8` on the local eight. | There is a substantial gap to the reference, but it must be separated into compression, precision, and port differences. |
| Two-factor representation plus terminal specialization | Local raw MSE `7.32033447e-8`; utilization `0.63732906`, down from about `0.9752`; final prediction RMS difference from the parent about `6.28e-7`. | Exact structural optimization already paid off; look for more before changing the approximation. |
| Earlier aggressive pruning | Norm-half at `[3,7,11,14]` gave `5.70594e-7` raw MSE. | A large rank cut can destroy the mechanism. Do not restart from this schedule. |
| Earlier rollout-based influence ranking | On MLP 0, only about 15.5% less distortion than norm-half, taking 2319.5 seconds versus 57.3 seconds. | Reject this expensive selector. Cheap joint or algebraic selectors remain open. |
| Mapped control with saved reference centers | About `4.36e-8` raw MSE across the saved multi-salt aggregate, versus `3.64e-8` for direct uncompressed K3. | Beating the old Hermite control did not establish a win over direct K3. Adding this branch is not an automatic upgrade. |

Sources: [reference receipt](whest-starterkit/research/phase6/results/P6-03_reference_panel.json), [compression receipt](whest-starterkit/research/phase6/results/P6-04_compression_receipt.json), [conversion audit](PHASE6_CONVERSION_FIX_REPORT.md), [conversion summary](whest-starterkit/research/phase6_conversion_fix/summary.json), [control implementation](whest-starterkit/scripts/p6_mapped_control_check.py), and [multi-salt receipt](whest-starterkit/research/phase6/results/P6-07_confirmation_receipt.json).

**Local timing remains unresolved.** The final conversion diagnostic disabled the residual gate and measured a maximum residual of `1.564788 s`. Its `4.66546191e-8` accuracy-times-utilization value is a diagnostic projection, even though that relaxed receipt reports zero failed MLPs. It is not a passing local score. The official board now demonstrates zero failed MLPs under official execution. Reconcile backend/version/timing differences before using local pass/fail receipts for new promotion decisions; do not erase either observation.

Several conclusions in [phase6report.md](phase6report.md) are too strong: tiny omitted-diagram comparisons do not prove omissions negligible at width 1024, and they do not establish that K3-simple is “minimax optimal.” The reference is also an approximation, not a mathematical upper bound. Those claims should not close the extensions below.

## 3. What worked, in mechanism terms

The successful state couples the mean, covariance, a recurring factorized third cumulant, and a selected scalar fourth-order harmonic component. It transports non-Gaussian information through the network instead of attaching an independent marginal correction to an otherwise unchanged Gaussian calculation.

The implementation then made three useful reductions:

1. The Step 5 repeated slices were simplified algebraically instead of rebuilt through redundant matrix products.
2. Each third-order atom became `Sym(U,U,V)`, so two factor matrices carry what previously required three.
3. The terminal layer computes the mean from marginal preactivation moments without building covariance and factor state that no later layer consumes.

Pruning currently keeps all newborn columns and ranks older individual columns by `||U||² ||V||` at layers `[6,10,13]`, retaining 62% of eligible columns. **That is a workable policy, not evidence that those three depths, that percentage, or that norm are optimal.**

The reference method and its layer conventions are documented in [ARC's implementation](https://github.com/alignment-research-center/mlp_cumulant_propagation) and [paper](https://arxiv.org/abs/2605.05179). Keep the pinned local reference commit `93d091a4c26c042bfffa28f2e76a81bc0aba94bb` for parity; changing the reference and the compression simultaneously would obscure the result.

## 4. Optimize the product, not accuracy alone

For a valid network, `score = raw_MSE * max(0.1, utilization)`. Average the per-network products. A failed run loses its predictions and discount. Official limits include `2**41` billed FLOPs, 0.4 seconds residual time, 120 seconds predict time, and 5 seconds setup. [Official scoring model](https://raw.githubusercontent.com/AIcrowd/whest-starterkit/main/docs/concepts/scoring-model.md).

These are arithmetic scenarios using the two **local** raw-error levels, not measured candidates:

| Utilization | Score if raw MSE stays `7.3203e-8` | Score if raw MSE reaches `3.6370e-8` |
|---:|---:|---:|
| 0.6373 | `4.665e-8` | `2.318e-8` |
| 0.50 | `3.660e-8` | `1.819e-8` |
| 0.35 | `2.562e-8` | `1.273e-8` |
| 0.20 | `1.464e-8` | `7.274e-9` |
| 0.10 | `7.320e-9` | `3.637e-9` |

Recovering the full reference accuracy at today's utilization is roughly a twofold local opportunity. Approaching the leaders also requires a cheaper representation or accuracy beyond this reference. That is why Phase 6.5 needs both exploitation and a bounded higher-upside lane.

For a branch adding utilization `du` above the floor, it helps only if `E_new/E_old < u_old/(u_old+du)`. Adding 0.10 utilization to 0.6373 requires more than about **13.6% raw-error reduction** just to break even. A 20% compute reduction, conversely, can tolerate less than 25% raw-error growth and still improve the product. Apply these comparisons per network when costs vary.

Planning milestones: a useful first result is a reproducible **10% paired improvement over the Phase 6 incumbent**; `3.0e-8` is an exploitation target, `2.0e-8` a strong extension target, and `<1.0e-8` a larger breakthrough target. Local and official achievement must be reported separately.

## 5. Lane A — More exact savings from the existing algebra

**Priority: first. Confidence in the mechanism: high; remaining savings unmeasured.**

### A1. Keep newborn factors structured for their first transport

The source births two blocks: `(U,V)=(I,A_ds)` and `(A2,3*diag(w21))`. On their next linear step, with the code's column-vector transport matrix `W`:

```text
W @ I = W
W @ (3*diag(w21)) = 3*W*w21[None,:]
```

The current code concatenates these blocks into dense factor arrays before transport. Carry a small birth-block descriptor until that next step, and replace those identity/diagonal matrix products by exact copies or broadcasts. Older factors still need dense transport. Check whether the actual meter already recognizes the structure; count only newly saved work.

This can remove up to two dense width-by-width products per eligible transition before accounting for existing symmetry discounts. It will not remove all rank-dependent work. Preserve column order, provenance, and the existing norm-selection result during this first comparison.

### A2. Specialize the Gaussian input and the final two layers

At layer 0, `mu=0`, `cov=I`, and K3/K4 vanish. Reuse `W @ W.T` for both the preactivation covariance and the fourth-order metric matrix instead of obtaining it through multiple paths. Preserve the same nonlinear closure initially.

At layer 14, the current code builds `k4`, `k22`, and the next scalar `c4`, although layer 15 consumes only that scalar fourth-order information. Derive its contraction directly from the existing terms where possible; retain the covariance needed for terminal variance. Likewise, project newborn factors into the terminal readout without first materializing concatenated state that is immediately consumed.

The final mean-only branch is already implemented. Credit only further work removed, and test small depths so that special cases do not reintroduce the old hardcoded-depth failure.

### A3. Reduce repeated operations and allocations

Reuse `cov11²`, coefficient outer products, powers, and zero-diagonal results when lifetimes permit. Investigate the scalar contraction of `k22` before allocating its full matrix. Keep all meaningful arithmetic in metered operations. Default float32 seeding and coefficient casting already exist; “switch to float32” is not a new experiment.

Measure FLOPs, operation count, allocation/memory behavior, and residual time separately. Faster plumbing may solve reliability without changing the adjusted score; lower billed FLOPs directly change the score above the floor.

**First experiments:** A1 alone, A2 alone, then their combined version with A3 only where profiling identifies duplication. Require float64 tiny-fixture parity and measured float32 prediction drift on the fixed panel. Aim for raw-MSE change within 0.5% for an exact rewrite; investigate failures rather than calling a drifting rewrite equivalent. Retain safe small savings, but do not spend the whole phase on sub-percent ones.

## 6. Lane B — Find the useful rank-versus-cost frontier

**Priority: immediately after the exact baseline.** The old schedule was chosen under a much more expensive representation. Reuse of its constants after the conversion is convenient, but untested as an optimum.

Use bounded coordinate sweeps rather than a Cartesian search:

1. Keep `[6,10,13]`; screen common retention `{0.45, 0.55, 0.62, 0.70, 0.80}`. Preflight cost before evaluating high-retention variants. The `0.45` point is a deliberate cheap boundary, not the preferred default.
2. At the best supported retention, compare schedules `[5,9,12]`, `[6,10,13]`, `[7,11,14]`, and `[6,10,12,14]`.
3. Change retention at one pruning event at a time: `r-0.10` or `r+0.10`, clipped to `[0.35,0.90]`. Maximum six such neighbors around the selected three-event schedule.
4. Only if more frequent pruning helps, compare an absolute **post-pruning total-rank cap** of `{4n,6n,8n,10n}` after layers 6 through 14. Newborn protected columns count toward the cap; reject an infeasible cap rather than silently changing its meaning.

Keep both the lowest adjusted-error configuration and the lowest raw-error configuration. A more accurate expensive point may support another lane, but should not displace the cheaper champion on raw MSE alone.

Report the rank before/after each pruning event and per-layer cost. A final-rank number alone hides the cost of carrying factors through earlier layers. Use the uncompressed reference for distortion diagnostics, while selecting on actual ground-truth error and charged compute.

## 7. Lane C — Improve selection without expensive deletion rollouts

**Priority: high. The current selector is cheap, but it ignores within-atom geometry and cancellation between atoms.**

### C1. Use the actual symmetric-atom norm

For the normalized three-term symmetrization `T=Sym(u,u,v)`, define `a=u·u`, `b=v·v`, and `c=u·v`. Then:

```text
||T||_F² = (a²*b + 2*a*c²)/3
```

The current `a*sqrt(b)` score is an upper bound, not this exact norm. Ranking by the squared exact norm needs one extra columnwise dot product and no square root. First verify this formula against dense tiny tensors, then compare it with the incumbent norm at **identical retained ranks**. It still ignores cancellation across different atoms and is not automatically a better output-error predictor. If most factor pairs have nearly the same angle, it will barely change the ranking; keep this screen short.

### C2. Preserve related birth terms together

Track `(birth layer, source family, source neuron)`. The path block and diagonal-slice correction are constructed together; ranking them independently can remove a compensating term while retaining the term it corrects.

Compare individual-atom selection with pairs sharing birth layer and source neuron. For two atoms, score their **combined signed tensor norm**, including their cross inner product; summing two positive norm scores misses cancellation. Normalize decisions to the same column budget. If pairs are too rigid, permit singletons only after the pairwise screen.

This is different from the earlier arbitrary 128-column deletion groups. The new grouping follows the construction's algebra, and its score is computed directly rather than by rerunning the suffix for every group.

### C3. Cheap response weighting, only if C1/C2 leave headroom

Use one common suffix-response approximation, not one suffix rollout per candidate group. Start with squared diagonal sensitivity propagated backward through squared weights and approximate ReLU gains. This deliberately omits cross-neuron response terms. Obtain any required future gains from one charged cheap pass, or use a declared fixed approximation; do not assume future gains are free.

Compare plain and response-weighted norm scores at one already selected rank. Charge the pilot, backward pass, and ranking. Stop if their overhead consumes the compression saving or if they fail to improve the paired adjusted result. Full per-group deletion rollouts stay rejected.

## 8. Lane D — Replace discarded factors with a small corrective state

**Priority: highest-upside compression experiment.** Deleting a tensor column removes all of its information. Some discarded information can be summarized much more cheaply than carrying the original columns.

### D1. Preserve the third-cumulant repeated slices at pruning

After selecting retained columns, calculate the missing diagonal `d_i = DeltaK_iii` and off-diagonal repeated slice `S_ij = DeltaK_iij`, with `diag(S)=0`. They can be obtained from the pre-pruning target slices and the retained tensor's slices; account for those contractions.

Construct an `n`-column correction:

```text
U_c = I
V_c = diag(d) + 3*S.T
K_c = Sym(U_c,U_c,V_c)
```

With the code's slice convention, this exactly restores those **current-layer** diagonal and repeated slices. It sets the discarded all-distinct component to zero; it does not reproduce the whole discarded tensor or guarantee correctness after the next linear mixing.

This turns “keep or delete” into “keep explicitly or summarize.” It is a particularly natural add-on because the existing recurrence already uses this diagonal-slice representation when creating factors.

**First test:** apply compensation at layer 10 only, with old-column retention `{0.45,0.55}`. Compare against uncompensated candidates at matched total rank/cost, not merely matched retention. Include the extra `n` columns and their future transport. If either survives, test compensation at `[6,10,13]`. Require at least 25% lower final distortion versus uncompensated pruning at similar cost to justify further development, followed by a real adjusted-score win versus the incumbent.

Avoid stacking unlimited correction blocks. Track them explicitly, and include them in later compression. If recomputing retained slices costs more than the saved future transport, reject that configuration.

### D2. Reweight a small retained subset to match useful contractions

If D1 shows the missing slices matter but its extra columns are expensive, fit signed weights on a small set of retained groups to match selected tensor contractions of the pre-pruning state. Start with 32 groups and 64 projected constraints, fixed ridge values `{1e-3,1e-2}` after normalizing features. Use a documented, seeded projection and all metered computation.

Targets are calculated from the current network's analytic state, not its ground truth. Save reconstruction residual and weight magnitudes. Include an unweighted subset comparator; constrain or reject unstable weights rather than accepting cancellation that only works in float64.

This is approximate quadrature over the **cumulant atoms**, not another input-space sampler. It only earns further work if the cost of target formation and solving is recovered by cheaper future transport. Do not run it as a large general tensor-decomposition search.

## 9. Lane E — Retain selected fourth-order information that the scalar loses

**Priority: bounded exploration after a useful compressed baseline exists.** The current nonlinear step computes `k4` and `k22`, then keeps only:

```text
c4 = 3/[n*(n+2)] * (sum(k4) + sum(k22))
```

Different fourth-order patterns can have the same scalar. Future weights can distinguish them. That identifies a concrete information loss worth testing; it does not prove it dominates the current error.

### E1. Diagnose anisotropy before designing its full transport

Use the already cached matching higher-moment data to inspect marginal fourth-cumulant error at layers 7, 11, and 14. Separate reference error from compression error, and assess the sampling uncertainty of the moment bake. A layer-14 diagnostic replacement of the preactivation fourth-cumulant vector, holding other states fixed, estimates sensitivity only. It is not a coherent deployable recurrence or a promise that the same gain is attainable online. Confirm the files contain the required preactivation quantities; postactivation marginal moments alone cannot reconstruct the next mixed fourth cumulant. If the required data are absent, record that limitation rather than relabeling a different intervention.

Prioritize this lane only if the controlled diagnostic suggests at least 15% raw-error headroom beyond the selected K3 baseline. If reference-versus-compressed K3 differences dominate, return effort to Lane D.

### E2. Scalar plus a traceless matrix harmonic

A candidate richer state is `K4 approximately c4*Sym(I,I) + Sym(I,Q)`, where `Q` is symmetric and traceless and symmetrization is normalized. For a linear transition define `M=W@W.T` and `Qp=W@Q@W.T`. The matrix-harmonic contributions needed by the present code are:

```text
Delta s4[i]   = M[i,i]*Qp[i,i]
Delta s22[i,j] = (M[i,i]*Qp[j,j] + M[j,j]*Qp[i,i]
                  + 4*M[i,j]*Qp[i,j])/6,  i != j
```

Verify normalization and dense-tensor parity before using these expressions. This representation has matrix storage and matrix-product transport rather than a dense fourth-order tensor.

**The unresolved part is the nonlinear projection back to Q.** In general it needs partial traces involving fourth-order mixed slices, including information beyond the currently stored `k4/k22`. Keeping only the diagonal of `Q` is a different approximation, and arbitrary nodewise kurtosis is not the full harmonic recurrence. Derive the missing contractions and their cost explicitly; stop with that obstruction if they cannot be obtained economically.

The first prototype should add the state only over a two-layer suffix, then a four-layer suffix if justified. Compare against spending exactly that extra compute on K3 rank. Do not initialize the new state from ground truth in a candidate.

### E3. One omitted diagram family at a time

The Phase 6 tiny-fixture omission tests leave this open, but provide no reason to start with a full augmented K4 implementation. For one family, derive only the contractions needed for final mean/covariance or the chosen compact state. Start with the last two layers and dense tiny algebra checks; forecast width-1024 cost before running it there.

Do not materialize a `1024^3` tensor or full Kronecker factor pairs. If a blocked/sketched contraction is proposed, specify the approximation and charge the sketch. A small-fixture difference demonstrates sensitivity, not an accuracy gain on the competition panel.

## 10. Lane F — A control-variate add-on, now centered on the real incumbent

**Priority: conditional and low.** The earlier offline K3-center control produced roughly `4.36e-8` raw error, worse than the direct uncompressed K3 reference. Its apparent `e-9` adjusted result assumed a free reference center and a 0.1 multiplier. The deployable old mapped candidate used a weaker center. Neither result answers whether an affordable branch helps the current compressed K3 estimator.

Test that actual question with:

```text
V = mean(Y) - J*(mean(H14) - mu14_K3)
prediction = (1-alpha)*mu15_K3 + alpha*V
```

All K3 centers must be computed online. Start with `J` from K3's analytic final-layer Gaussian gate probabilities, acknowledging that this is approximate. This avoids the previous 2048-sample pilot. Use ordinary Gaussian antithetic sample totals `{512,1024}` and alpha `{0,0.05,0.10,0.20}`; keep `alpha=0` as the direct-K3 control. A whitened cloud is a separate, potentially biased variant, not an unbiased Gaussian sampling substitute.

Record the projected center bias and the residual sampling variance separately. The correction inherits `J*(mu14_K3 - E[H14])`; a good marginal center can still have an expensive projected error. Use three existing sanctioned sampler salts on the same eight networks if a stochastic finalist survives.

Do not optimize a large blend portfolio. Kill this lane if its best honest paired result fails the cost-product break-even test. The most plausible reopening is **after cheaper K3 compression**, where a small sampled residual might repair compression errors economically.

## 11. Ideas to park unless the diagnostics point there

- **More pruning-schedule extrapolation or scalar shrinkage:** a bounded final polish, not the main research mechanism. Any fitted coefficient needs network-level held-out selection.
- **Mixed precision:** only after profiling finds meaningful cost in accidental promotion. Current float32 prediction drift is much smaller than current MSE; lower precision may hurt cancellation, and cheaper billing must be demonstrated.
- **A second full K3 estimator blended with the first:** shared-prefix branching might be cheap enough; two independent full passes probably are not. Test prediction complementarity from saved outputs before implementation, then charge both continuations.
- **Low-rank covariance transport:** potentially useful only if profiling shows it matters after K3 compression. Preserve exact marginal variances and test downstream drift; the current breakthrough does not justify returning to a diagonal Gaussian covariance.
- **Exact first-layer Gaussian covariance:** at most one coherent initialization experiment if layerwise diagnostics point there. Earlier Gaussian-kernel refinements did not produce the main gain; do not restart that sweep.
- **Full K4, broad input samplers, large learned surrogates, or leader-method imitation without a disclosure:** reserve for a later phase if the concrete state-compression and selected-K4 paths fail.

## 12. Suggested order and evidence gates

Use these as bounded experiment identifiers. Candidate paths are **proposed**, not files created by this plan.

| Stage | Action | Candidate suffix / decision |
|---|---|---|
| P65-00 | Freeze source, panel manifest, current official evidence; reconcile local timing and scoring status. | `incumbent`; no baseline substitution. |
| P65-01 | Structured births, exact boundary contractions, then verified operation reuse. | `exact`; preserve numerical quality and count savings. |
| P65-02 | Bounded retention/schedule sweep on that implementation. | `rank`; retain the measured accuracy/cost frontier. |
| P65-03 | Exact atom norm, then algebraic pairing; one cheap response test only if warranted. | `selector`; compare at matched rank and charged cost. |
| P65-04 | Slice-preserving discarded-state compensation. | `slice`; first layer 10, then three events if successful. |
| P65-05 | Only if compensation motivates it: small contraction-matching reweighting. | `reweight`; stop when selection overhead defeats savings. |
| P65-06 | Fourth-order sensitivity diagnostic, then one derived compact extension. | `k4`; compare against equal-cost K3 rank. |
| P65-07 | One small online K3-centered control check. | `cv`; enforce the product break-even condition. |
| P65-08 | Combine only independently supported changes; reproduce the finalist. | `final`; retain incumbent unless evidence clears the gate. |

An execution session should create candidates under `candidates/estimator_p65_<suffix>.py` and artifacts under `whest-starterkit/research/phase6_5/{control,diagnostics,predictions,results}`. Preserve Phase 6 source and reports, plus the historical `whest-starterkit/estimator.py` control. Never compare only against the much weaker historical control and call that a Phase 6.5 improvement.

Keep the established **eight full-size MLP limit**, including diagnostics and fitting: logan-fitzgerald, william-graves, raymond-barnes, steven-rice, sarah-kelley, christopher-morales, cheryl-graham, renee-park. Resolve by name and verify hashes against the existing manifest; do not default to eight arbitrary rows or download another training panel. Tiny algebra fixtures are separate. Eight repeatedly explored networks support local decisions, not a claim of broad generalization.

For parameter selection, report network-level leave-one-out selection: select among the already declared configurations on seven networks and score the held-out eighth. Neurons are correlated coordinates, not thousands of independent validation networks. This reduces immediate selection bias but does not undo repeated adaptation to the same eight; label that limitation. Disclose both the all-eight selected configuration and the leave-one-out selection result.

**Promotion gates:** zero failures under an unrelaxed applicable grader; lower mean adjusted score; at least 6/8 paired wins for a mathematical variant; no network more than 10% worse without an explicit explanation and a separate decision. Seek at least 10% mean improvement to promote a research variant. An exact optimization may be retained for smaller robust savings with unchanged accuracy. For stochastic finalists, the mean improvement must hold across three salts rather than one fortunate run. If local execution still disagrees with official execution, mark local deployability unresolved instead of awarding a fictitious pass.

Save raw and adjusted per-network values, all-layer errors, candidate hash, parameters, FLOPs, utilization, rank history, setup/predict/residual times, backend versions, and peak memory or “unmeasured.” Preserve failure status before computing any discount. In a new diagnostic harness, fix the old failed-run score recomputation without rewriting historical receipts.

Use `uv` for execution and any isolated reference environment. Production arithmetic, sorting, fitting, and sketches must use permitted metered operations; Python handles bookkeeping. Shipped generic fitted data may be allowed, but public-network lookup tables cannot produce predictions for unknown networks. Do not import diagnostic NumPy/Torch code or alter accounting/caps in a candidate. [Official allowed-code rules](https://raw.githubusercontent.com/AIcrowd/whest-starterkit/main/docs/concepts/allowed-code.md).

Preflight hardware, rank growth, and memory before a capacity-heavy reference run. Preserve the existing 0.4-second residual cap; target below 0.35 seconds with margin, below 90 seconds predict, below 4 seconds setup, and memory below the configured cap with headroom. A plan is not authorization to start experiments or submit. After an explicitly delegated execution session, commit only its completed changes and push to main according to the workspace instruction; AIcrowd submission remains a separate user action/authorization.

## 13. My shortlist

If only three ideas receive serious implementation effort, choose **structured newborn factors**, **a fresh retention/schedule frontier**, and **slice-preserving pruning**. The first follows the proven conversion win, the second tests constants inherited from a different cost regime, and the third offers a new way to retain useful joint information without paying for every original factor.

Keep the **matrix fourth-order harmonic** as the ambitious extension, with its missing nonlinear projection stated honestly. Phase 6.5 succeeds if it turns the working recurrence into a better cost–accuracy tradeoff, while learning whether the next limiting error comes from discarded K3 structure or genuinely missing fourth-order information.
