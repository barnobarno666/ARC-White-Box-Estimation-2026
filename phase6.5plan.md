# Phase 6.5 sequential autocoding runbook

Prepared 6 September 2026 from [plan6.5 ideas.md](<plan6.5 ideas.md>). This file specifies future execution. Creating it did not run competition experiments. When the user delegates execution of this file, follow the stages below in order until every stage has a recorded outcome and the final report is complete.

## 0. Instructions to the executing agent

Improve the working Phase 6 joint-cumulant estimator by reducing its cost, improving compression, and testing the explicitly specified extensions. Long research execution is acceptable. Do not stop because a successful stage took a long time, because the first improvement is small, or because an early method failed. Resume from saved receipts after interruption.

**Execute the listed methods and equations. Do not derive replacement methods, silently change the experiment grid, or substitute a heuristic when an equation fails.** Routine implementation and debugging are your responsibility. If a required mathematical identity fails its tiny check, stop that branch, record the failure, and continue the next independent stage from the last passing parent. Stop the whole run only for a corrupted baseline/panel, unsafe available resources, or a dependency that prevents all remaining useful work.

Boundaries:

- Work sequentially. Do not launch another goal or other agents.
- Use `uv` for environments and Python execution. Keep any reference dependencies isolated.
- Use only the existing eight full-size MLPs in Section 2, including fitting and diagnostics. Tiny synthetic algebra fixtures are allowed. Do not generate or download additional full-size MLPs.
- Preserve Phase 6 candidates, reports, predictions, and `whest-starterkit/estimator.py`.
- All new production arithmetic, sorting, solves, sketches, and random arrays use permitted flopscope APIs. Python handles indices, configuration, and bookkeeping. Do not modify accounting, client transport, or caps. [Official allowed-code rules](https://raw.githubusercontent.com/AIcrowd/whest-starterkit/main/docs/concepts/allowed-code.md).
- Prepare a verified release artifact if a candidate qualifies. **Do not submit to AIcrowd without a separate user instruction.**
- At the end of the edit session, commit only your completed artifacts and push to `main`, as the workspace instructions require. Inspect the staged diff; unrelated files are already dirty.

## 1. Objective, incumbent, and score interpretation

The incumbent is [candidates/estimator_p6_k3_twofactor_terminal.py](candidates/estimator_p6_k3_twofactor_terminal.py), with pruning layers `[6,10,13]` and old-column retention `0.62`. Expected SHA-256:

```text
8498085d5b90646c6d65ce23087b2e9331980a77cb44bfa6d762341f68dd05ac
```

The leaderboard snapshot verified earlier in this conversation showed adjusted `4.35e-8`, raw MSE `6.82e-8`, utilization `0.6373366929`, and zero failures for the user's account. This is historical official evidence, not the local paired comparator. The corresponding local package/receipt is #330018; its saved receipt predates grading. [Evidence and qualification](PHASE6_CONVERSION_FIX_REPORT.md), [saved receipt](whest-starterkit/research/phase6_conversion_fix/submission_receipt.json).

Saved local diagnostic values are raw MSE `7.320334471927481e-8`, utilization `0.6373290634664954`, and projected adjusted value `4.6654619132550436e-8`. That diagnostic disabled the residual gate and measured up to `1.564788 s` residual. It is **not a passing local score**, despite the relaxed receipt's zero-failure field. [Diagnostic summary](whest-starterkit/research/phase6_conversion_fix/summary.json).

For each valid network `m`, with final prediction `p_m`, truth `y_m`, width `n`, and billed FLOPs `F_m`:

```text
E_m = sum_i (p_m[i] - y_m[i])² / n
B   = 2**41 = 2199023255552
u_m = F_m / B
S_m = E_m * max(0.1, u_m)
panel_score = sum_m S_m / 8
```

Use the official failure result when a cap is breached. Do not discount pre-failure predictions. Residual time is capped, not added to FLOPs. Official caps include residual `0.4 s`, predict `120 s`, and setup `5 s`. [Official scoring model](https://raw.githubusercontent.com/AIcrowd/whest-starterkit/main/docs/concepts/scoring-model.md).

The ideas table's `7.320e-9` and `3.637e-9` at 10% utilization are **hypothetical products**. No submission has demonstrated those combinations. The uncompressed reference's `3.63704648e-8` raw MSE is a compression target, not an exact answer, lower bound, or free online center.

Goals: first obtain at least a 10% paired improvement over the current Phase 6 incumbent; then pursue local adjusted `<3.0e-8`, `<2.0e-8`, and `<1.0e-8`. A low error with invalid resources does not achieve any deployable goal. Do not stop subsequent independent stages merely because the first target was met.

## 2. Workspace, fixed panel, and artifacts

Workspace: `D:\ALL CODES\AICROWD COMPETITION`. Benchmark project: `whest-starterkit`. Dataset: `D:\ALL CODES\AICROWD COMPETITION\datasets\mini`, split `mini`.

Use these names in this exact order:

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

Resolve using `load_dataset`, `resolve_seed_context`, and `MLP.from_row`; verify width 1024, depth 16, weight hashes, truth hashes, and actual seeds against [Phase 6 manifest](whest-starterkit/research/phase6/manifest.json). Check content, not the manifest's timestamp. Never assume a bare `--n-mlps 8` selects the right names: assert its dataset order first. If the order differs, create a name-selected view using the installed dataset API and verify it before scoring; never substitute rows.

Create this new tree during execution:

```text
phase6.5report.md
candidates/estimator_p65_<experiment_id>.py
whest-starterkit/research/phase6_5/
  control/          immutable incumbent copy and hashes
  fixtures/         tiny algebra fixtures and panel index metadata
  configs/          full configuration JSON for each candidate
  predictions/      captured outputs, keyed by source and panel hashes
  diagnostics/      state errors, profiles, algebra receipts
  results/          unmodified runner output and derived comparisons
  release/          finalist candidate, manifest, archive, verification
  manifest.json
  progress.json
  ledger.jsonl
```

Create helpers `whest-starterkit/scripts/p65_manifest.py`, `p65_math_checks.py`, `p65_build_candidate.py`, `p65_eval.py`, and `p65_summarize.py`. These names are specifications, not claims that helpers already exist. The candidate builder must emit standalone files with explicit constants; no reference-data imports or runtime dependency on research helpers. Keep one readable implementation template with tested switches for the prescribed variants.

Each receipt records experiment ID, parent source hash, candidate hash, complete parameters, panel hashes, seed offset, commands, package versions/backend, output paths, and status. Store per-network raw error, score, FLOPs, utilization, times, failures, all-layer errors, rank history, and memory measured or explicitly unmeasured. Cache reuse requires matching candidate bytes, configuration, panel, seed, backend, and relevant runner settings.

After every candidate, append its receipt and update `progress.json` atomically with `last_completed`, `next_action`, `research_parent`, and `release_champion`. Long duration is not a kill criterion for offline mathematics, but production and machine resource limits still apply.

## 3. Shared evaluation and selection procedure

Apply this sequence to each new algorithm/configuration:

1. Check its tiny invariants and output contract. Never use a full-size run to debug a transpose.
2. Forecast rank, billed work, and live memory from shapes. Query available RAM before a large reference run. Do not run a job whose safe estimated peak exceeds available memory. Avoid dense third/fourth tensors at width 1024.
3. Run one of the existing eight as an operational smoke. Reject only for contract, numerical, or resource failure at this step; one-network accuracy does not select a winner.
4. Evaluate the full eight in order, serially. Record incomplete runs as incomplete, preserving successful rows. Resume missing rows without inventing a complete mean.
5. Compare against both the stage parent and the immutable Phase 6 incumbent. Compute errors in float64 outside the measured estimator call.
6. Evaluate all declared configurations for a stage unless an explicit gate skips them. Do not stop the grid at its first win.

Maintain two statuses:

- `RESEARCH_ONLY`: captured accuracy and billed compute are meaningful, but the applicable timing/resource pass is unresolved or failed. A projected score may order further research configurations, with that status carried forward.
- `VALIDATED`: unrelaxed authoritative runner passes all eight; only these may become the release champion.

The known local/official timing discrepancy must not freeze all mathematical research. After P65-00's bounded diagnosis, continue informative diagnostics if necessary, keeping them `RESEARCH_ONLY`. Never relabel them as valid or silently relax caps in the release candidate.

**Research-parent rule:** among tested feasible configurations, choose the lowest mean product with at least 6/8 wins over the stage parent and worst paired score ratio at most `1.10`; otherwise retain the parent. Small credible improvements may carry into later stages. Break a score tie within relative `1e-4` by lower FLOPs, then lexicographic configuration ID. A purely exact optimization uses the separate parity gate below and need not win 6/8 in raw error.

**Final promotion rule:** require a valid unrelaxed run, at least 10% lower panel score than the immutable incumbent, 6/8 paired wins, and maximum paired regression at most 10%. A safe exact rewrite with unchanged accuracy may be retained for smaller savings, explicitly reported as such. If a proposal fails, retain it as a result, not as champion.

For parameter grids, report leave-one-network-out selection: for each held-out network, choose among the same declared candidates using the other seven and the same tie-breaker, then evaluate the held-out value. Do not select using that eighth network. Adaptive stages have already seen the panel; this is a selection-bias diagnostic, not an independent generalization estimate. Save selected configuration IDs for all eight folds.

For stochastic candidates use offsets `{0,1337,8888}`. Offset changes only the supplied estimator seed, modulo `2**32`, in the evaluation harness; weights, truth, and names remain fixed. Production randomness starts from `fnp.random.default_rng(mlp.seed)`, not from hardcoded dataset seeds. Every retained stochastic finalist must show a lower mean product on each offset; report all 24 per-network observations.

## 4. Mathematical contract: shapes and inherited recurrence

Definitions throughout this file:

| Symbol | Shape | Meaning |
|---|---|---|
| `A_l = mlp.weights[l]` | `(n,n)` | Row-sample forward matrix: `H_next = relu(H @ A_l)` |
| `W_l = A_l.T` | `(n,n)` | Column-moment transport matrix used in the recurrence |
| `mu`, `C` | `(n,)`, `(n,n)` | Postactivation mean and covariance of the current layer |
| `U`, `V` | `(n,R)` | Two-factor symmetric third-cumulant representation |
| `g`, `h` | `(n,)` | Gaussian ReLU Wick coefficients `w(1,1)`, `w(2,1)` |
| `S` | `(n,n)` | Third-cumulant repeated slice `K3[i,i,j]`, diagonal zero |
| `d` | `(n,)` | Third-cumulant diagonal `K3[i,i,i]` |
| `c4` | scalar | Isotropic fourth-cumulant coefficient |
| `q` | `(n,)` | Optional diagonal trace-harmonic correction, Section 13 |

`*` between arrays means elementwise multiplication; `@` means matrix multiplication. `Z(X)` copies `X` and sets its diagonal to zero. `sym(X)=(X+X.T)/2`. A vector `x[:,None]` scales rows; `x[None,:]` scales columns. Do not interchange them.

An individual third-order atom is the normalized symmetric tensor:

```text
T(u,v)[i,j,k] = (u[i]*u[j]*v[k]
                + u[i]*v[j]*u[k]
                + v[i]*u[j]*u[k]) / 3
K3 = sum_r T(U[:,r], V[:,r])
```

Therefore:

```text
d = sum(U*U*V, axis=1)
S = Z((2*(U*V) @ U.T + (U*U) @ V.T) / 3)
U_pre = W @ U
V_pre = W @ V
mu_pre = W @ mu
C_pre = sym(W @ C @ W.T)
```

After transport, calculate `d_pre,S_pre` from `U_pre,V_pre` using those same formulas. Do not use `K3[i,j,j]` where `S[i,j]` is required; it is `S.T` off diagonal.

For preactivation mean `m`, variance `v`, let:

```text
sigma = sqrt(max(v, 1e-12))
a = m/sigma
phi = standard_normal_pdf(a)
Phi = standard_normal_cdf(a)
w(0,1) = sigma*phi + m*Phi
g = w(1,1) = Phi
h = w(2,1) = phi/sigma
w(3,1) = -a*phi/sigma²
w(4,1) = (a²-1)*phi/sigma³
mean_out = w(0,1) + w(3,1)*d_pre/6 + w(4,1)*s4_pre/24
```

Preserve the incumbent `_relu_wick` definitions for powers `p=1,2,3,4` and all its `pk -> k` formulas. Do not replace power-cumulant mixed slices with ordinary raw cross moments. In particular `pk11` is already a mixed power-cumulant quantity; do not subtract an additional mean outer product from it.

The incumbent's new factors satisfy:

```text
C0 = Z(C_pre)
A2 = g[:,None] * C0
S_propagated = S_pre*g[:,None]²*g[None,:]
               + g[:,None]²*h[None,:]*(C0*C0)
d_propagated = d_pre*g³
d_res = k3 - d_propagated
S_res = Z(k21 - S_propagated)
A_ds = diag(d_res) + 3*S_res.T

U_out = concat(U_pre*g[:,None], A2, I, axis=1)
V_out = concat(V_pre*g[:,None], 3*diag(h), A_ds, axis=1)
```

Here `k3,k21` are the desired postactivation slices from the unchanged nonlinear conversion. `diag(C0)=0` is necessary for the birth-block identity. Before pruning, the constructed tensor has diagonal `k3` and repeated slice `k21`. That invariant is central to Section 11.

Initialize at the input with `mu=0`, `C=I`, `R=0`, `c4=0`, and optional `q=0`. Return `(depth,width)` finite float32 means, including the last ReLU. For tiny depths, ignore scheduled pruning indices outside `0..depth-2`; never create future factors after the terminal layer.

## 5. P65-00 — Freeze and establish trustworthy evaluation

1. Verify the incumbent hash. Copy it unchanged to `research/phase6_5/control/estimator_incumbent.py`. Record the actual historical control hash separately; never use it as the Phase 6.5 parent.
2. Build the fixed manifest and snapshot package/backend versions, CPU, RAM, configured caps, and relevant runner settings. The saved Phase 6 stack was whestbench `0.16.1`, flopscope `0.12.1+np2.2.6`; record the actual versions rather than assuming those are still installed.
3. Implement the new evaluator without modifying historical `p6_eval_harness.py` or its receipts. Preserve official failure status. Distinguish captured pre-failure predictions from scored predictions. Array conversion, saving files, and full-panel stacking belong outside the measured `predict` interval.
4. Compare the direct incumbent with the prediction-capture path on the first fixed MLP using normal caps. If the wrapper adds timing, fix the wrapper. Then compare documented local and subprocess runners. Do not patch the meter or install undocumented backends.
5. If the direct incumbent still fails residual timing, record the discrepancy and continue with separate capture diagnostics. After these three paths are documented, stop repeating unchanged timing probes until candidate changes justify another check.
6. Obtain paired baseline predictions for all eight and a separate unrelaxed score receipt. Verify the approximate saved local raw MSE; investigate a change exceeding 2% before treating it as the same comparator.

Use the installed official CLI after verifying dataset order, from the benchmark directory:

```powershell
$env:PYTHONUTF8 = '1'
uv run whest validate --estimator ../candidates/estimator_p6_k3_twofactor_terminal.py
uv run whest run --estimator ../candidates/estimator_p6_k3_twofactor_terminal.py --dataset '../datasets/mini' --split mini --n-mlps 8 --runner subprocess --format json
```

The first command is only a contract check. Do not run `whest run` without `--dataset`. Keep subprocess orchestration in research helpers, never in the estimator.

## 6. P65-01 — Implement and pass the mathematics checks

Create independent dense reference checks, not tests that call the same helper on both sides. Use NumPy in the diagnostic script only; the candidate uses flopscope. Fix tiny RNG seed `6501`; widths `{3,5,8}` for explicit tensors, plus recurrence fixtures `(8,1),(16,2),(32,8),(32,16)` with seeds `{6,17,29}`.

Required identity tests:

- Explicitly construct `T(u,v)` by its three outer products. Check its diagonal, repeated slice, norm, and cross inner product against Sections 4 and 10.
- With asymmetric `W`, compare dense three-mode tensor transport to `T(Wu,Wv)`.
- Compare the Section 11 correction's diagonal and repeated slices with the missing tensor; separately demonstrate that all-distinct entries generally differ. Do not write a false whole-tensor equality test.
- Compare direct versus split terminal contractions in Section 8.
- Compare explicit `k22` row and total sums with Section 7.
- Construct the dense fourth tensor in Section 13, then check partial trace, diagonal trace projection, and transport formulas. Check that projection discards off-diagonal trace information rather than falsely restoring it.
- Check zero-rank, all-zero factors, equal scores, negative factor weights, paired cancellation, depth 1/2, and a pruning event with no eligible old factors.

Float64 algebra tolerance: `atol=rtol=1e-10` for the tiny contraction identities and `1e-9` for complete recurrence comparisons. Do not force float32 to match float64 bitwise.

Before accepting an exact production rewrite, require finite output, unchanged full-size shapes, all-layer maximum absolute prediction drift at most `1e-5`, final-layer RMS drift at most `2e-6`, panel raw-MSE ratio in `[0.995,1.005]`, and no individual raw MSE more than 1% worse. On failure, save the first differing state/index and debug. Do not widen tolerances without reporting why.

At plan-writing time, independent dense checks of the principal new identities passed at widths 3/5/8 with maximum absolute error `5.69e-14`. This validates the written algebra only. The executing agent must test its own implementation, gating behavior, and recurrence.

## 7. Exact contraction formulas for fourth-cumulant reduction

Use these in P65-02 and the later `q` extension. All mixed matrices below have zero diagonal. Define:

```text
p = pk1              # n
B11 = pk11           # n,n symmetric
D21 = pk21           # n,n, generally asymmetric
E22 = pk22           # n,n symmetric

k22 = E22 - 2*(p[None,:]*D21 + p[:,None]*D21.T)
            - 2*(B11*B11) + 4*p[:,None]*p[None,:]*B11
```

Its exact row sums are:

```text
r22 = sum(E22,axis=1)
      - 2*(D21 @ p + p*sum(D21,axis=0))
      - 2*sum(B11*B11,axis=1)
      + 4*p*(B11 @ p)
```

Its exact total sum is:

```text
t22 = sum(E22)
      - 4*dot(p, sum(D21,axis=0))
      - 2*sum(B11*B11)
      + 4*dot(p, B11 @ p)
```

The `axis=0` on `D21` is intentional. Changing it to `axis=1` changes the recurrence. For current scalar K4, `c4_next = 3*(sum(k4)+t22)/(n*(n+2))`.

Avoid forming `E22` itself as follows. Let `C0=Z(C_pre)`, `S=S_pre`, `F22=s22_pre`, `a2=w(1,2)`, `b2=w(2,2)`, and `H=.5*C0²+.25*F22`. Then:

```text
rows_E22 = a2*(C0 @ a2)
           + .5*(a2*(S.T @ b2) + b2*(S @ a2))
           + b2*(H @ b2)
```

Substitute `rows_E22` into `r22`; use `sum(rows_E22)` for the total. This preserves the symmetrization in the incumbent. Do not remove `D21`, `B11`, or the marginal `k4` calculation: later equations still need them. Preserve the diagonal-zero convention before reducing.

## 8. P65-02 — Exact structural optimizations

Parent is the frozen incumbent. Create and evaluate these candidates independently, then combine passing changes:

### P65-02a: Structured newborn transport

At the end of layer `l`, retain descriptors for the two fresh blocks `(A2,3*diag(h))` and `(I,A_ds)` instead of concatenating them into the dense older-factor block. At the next transport:

```text
path_U_pre = W @ A2
path_V_pre = 3*W*h[None,:]
slice_U_pre = W
slice_V_pre = W @ A_ds
```

After this step these columns become ordinary old factors, with the usual row gain at the nonlinear step. Preserve ordering: old factors, path block, slice block. If an identity block has retained source indices `J`, use `W[:,J]`, not all of `W`. Do not repeatedly treat an already transported block as diagonal.

All newborn columns are protected at their birth event. For exact parity, use the incumbent norm and retention decisions. Equivalent norms for a fresh full block are `sum(A2²,axis=0)*3*abs(h)` for path atoms and `sqrt(sum(A_ds²,axis=0))` for slice atoms. Preserve the existing stable tie order.

### P65-02b: Input and scalar-K4 specialization

At layer 0 compute `M=W@W.T` once, and use its symmetric form as `C_pre`; `mu_pre=0`. Incoming K3/K4 corrections are zero. At nonterminal layers use Section 7 to obtain `c4_next` without allocating `pk22/k22` solely for a final reduction. Leave every other closure coefficient unchanged.

### P65-02c: Terminal birth contraction

The incumbent already has a mean-only terminal layer. Further split its K3 diagonal. Let `Uo,Vo` be the retained old postactivation factors, `A2,h,A_ds` the penultimate births, and `W` the terminal transport:

```text
d_terminal_old = sum((W@Uo)² * (W@Vo), axis=1)
d_terminal_path = 3*sum((W@A2)² * W * h[None,:], axis=1)
d_terminal_slice = sum(W² * (W@A_ds), axis=1)
d_terminal = d_terminal_old + d_terminal_path + d_terminal_slice
```

Apply any penultimate pruning to the old factors before this formula. Newborn blocks remain protected. The terminal preactivation variance is still `sum((W@C)*W,axis=1)`. The scalar K4 contribution is still `c4*sum(W²,axis=1)²`. No final covariance or birth state is needed. For depth 1 use the ordinary zero-rank terminal case.

### P65-02d: Combined exact candidate

Combine only passing 02a/02b/02c changes. Cache repeated `C0²`, Wick coefficients, and outer products only when profiling shows duplication. Do not redesign formulas to chase a wall-time artifact. Record FLOPs and residual time separately.

Select the cheapest passing exact candidate as `EXACT_PARENT`; if none passes, `EXACT_PARENT` remains the incumbent. Every later branch starts from a verified exact implementation.

## 9. P65-03 — Retention and schedule frontier

Use `EXACT_PARENT`, the original norm selector, and no compensation/K4/CV. At a nonterminal pruning event:

```text
R_old = count(columns with birth_layer < current_layer)
K_old = max(1, floor(retention * R_old)) if R_old > 0 else 0
R_after = K_old + 2*n
```

Select the highest scores using metered stable `argsort`, taking the last `K_old` indices, then restore original provenance order. Do not accidentally retain a fraction of newborn columns. Outside pruning events, add exactly `2n` births. The terminal layer adds none.

Execute these sub-stages in order:

| IDs | Fixed parameters | Values to evaluate |
|---|---|---|
| P65-03a-r045/r055/r062/r070/r080 | schedule `[6,10,13]` | retention `0.45,0.55,0.62,0.70,0.80` |
| P65-03b-s5912/s61013/s71114/s6101214 | best 03a retention | schedules `[5,9,12]`, `[6,10,13]`, `[7,11,14]`, `[6,10,12,14]` |
| P65-03c | selected 03b schedule | change each event's retention separately by `-0.10`, then `+0.10`, clipped to `[0.35,0.90]`; hold other events fixed |
| P65-03d-c4/c6/c8/c10 | only if the four-event schedule improves product by at least 2% over best three-event schedule | total-rank caps `4n,6n,8n,10n` at every layer 6 through 14 |

For 03c, evaluate exactly two neighbors per selected event; there are six neighbors for three events and eight for four. Do not iterate this local search indefinitely. For 03d, `K_old=min(R_old,cap-2n)`; preserve the `2n` newborns. Reject negative capacity and skip pruning when everything fits.

Forecast expensive variants before running. Mark a predicted over-budget variant `SKIPPED_COST` with the shape/cost calculation rather than exhausting the meter on all eight. Keep the lowest-product and lowest-raw-error frontier points. Apply Section 3 selection to choose `RANK_PARENT`.

## 10. P65-04 — Explicit inexpensive selectors

Keep `RANK_PARENT`'s rank budgets and schedules fixed. Compare selectors at those budgets; none may use true means or reference predictions.

### P65-04a: Exact individual-atom norm

For one column pair `u,v`:

```text
a = dot(u,u); b = dot(v,v); c = dot(u,v)
norm_squared = (a*a*b + 2*a*c*c)/3
```

Vectorize over columns and rank by `norm_squared`; no square root is needed. Compare against the original upper-bound score. This is an exact tensor norm, not an exact prediction-damage measure.

### P65-04b: Algebraic birth pairs

Group path and slice atoms from the same `(birth_layer,source_neuron)`. Keep pairs intact throughout this candidate. Round each old-column budget down to an even number, and evaluate an individual-selector control at the same even budget. If fewer than two old columns can be kept, skip that configuration.

For `T1=T(u,v)` and `T2=T(p,qv)`, the exact cross inner product is:

```text
cross = ((dot(u,p)**2)*dot(v,qv)
         + 2*dot(u,p)*dot(u,qv)*dot(v,p)) / 3
pair_norm_squared = norm_squared(u,v) + norm_squared(p,qv) + 2*cross
```

Here `qv` is a vector factor, unrelated to the K4 state `q`. Keep signed `cross`: replacing it by its absolute value destroys cancellation. A tiny negative squared norm from rounding may be clamped to zero after confirming its magnitude is at most `1e-6` times the sum of absolute contributing terms; a larger negative result is a failed numerical check.

Rank pairs by their combined squared norm. Retain `K_old/2` pairs. Tie-break with stable provenance order. Do not group arbitrary consecutive 128-column blocks or run deletion rollouts.

### P65-04c: One fixed diagonal response approximation

Run this once with the best supported 04a/04b selector. First perform a charged diagonal-Gaussian pilot, initialized `m=0`, `v=1`. For each layer with column transport `W`:

```text
m_pre = W@m
v_pre = (W*W)@v
sigma = sqrt(max(v_pre,1e-12)); a=m_pre/sigma
g_pilot = Phi(a)
m_next = sigma*phi(a) + m_pre*g_pilot
second = (m_pre²+v_pre)*g_pilot + m_pre*sigma*phi(a)
v_next = max(second-m_next²,1e-12)
```

Store the `g_pilot` vector for each layer. Then start `h_resp[depth-1]=ones(n)/n` and propagate backward:

```text
h_resp[l] = (W[l+1]*W[l+1]).T @ (g_pilot[l+1]²*h_resp[l+1])
```

For ranking at layer `l`, normalize `h_resp[l]` to mean one, clamp positive entries below `1e-12`, and let `z=sqrt(h_resp[l])`. Compute the individual/pair norm after replacing every factor `u,v` by `z*u,z*v`. This is the tensor norm in a diagonal approximation to a downstream metric; it is not an exact mean-gradient or deletion-loss formula. Use the same `z` for every group at that event.

Charge the pilot, backward recurrence, and ranking. Reject if no lower paired product results. Do not add lookahead depth, per-group rollouts, or a second learned response model. Choose `SELECT_PARENT` by Section 3; retain `RANK_PARENT` if no selector qualifies.
