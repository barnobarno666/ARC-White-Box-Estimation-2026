# Phase 8: new representations and algorithms for the error-compute frontier

Prepared 7 September 2026. **This is a future execution runbook for the user's autoagent. Writing this plan does not execute experiments, train models, or authorize an AIcrowd submission.**

This follows the structure of [phase7plan.md](phase7plan.md), but changes the research objective from factor pruning to four different mechanisms: angular moment propagation, direct contraction of frozen higher-order sources, learned compact closure, and gate-residual integration. Use Phase 1-7 records for configurations, measured numbers, source provenance, and receipts. **Do not inherit their interpretations, declarations of impossibility, family-wide rejection claims, or unsupported explanations of leaderboard methods.**

## 0. Mission and execution boundaries

Produce an estimator with a substantially better measured error-compute tradeoff on the Phase 2 width-1024, depth-16 task. Primary target: **valid adjusted score below 1.00e-8**. Stretch targets: **6.00e-9 and 3.00e-9**. These are research targets, not forecasts.

Execute the mandatory diagnostics for all four lanes. Execute each expansion only when its stated gate passes. A gate failure closes that configuration or declared expansion, not the mathematical family. Do not spend the phase on retention percentages, a larger Tucker grid, more sampler salts for losing candidates, or another terminal scalar calibration campaign.

**Implement the specified algorithms and candidate grids. Do not replace an equation with a similarly named heuristic. Do not declare a configuration evaluated unless its effective settings and branch execution are verified.** Repair routine implementation defects. When a mathematical or API obstruction remains, identify the exact failing identity or unsupported operation, mark that branch accordingly, and continue independent work.

Execution rules:

- Run sequentially. Do not spawn agents or concurrent numerical jobs. Long work must checkpoint and resume.
- Use uv for Python and dependency management. Benchmark commands run from D:\ALL CODES\AICROWD COMPETITION\whest-starterkit.
- Preserve the benchmark environment pins. Put offline teacher/training dependencies in a separate uv-managed project under research/phase8/offline. Do not install PyTorch into or modify the production environment merely for training.
- Preserve all prior estimators, reports, predictions, and unrelated dirty files. The legacy whest-starterkit/estimator.py, CONTROL65, and CONTROL7 are immutable.
- This plan permits tiny synthetic fixtures and the bounded independent teacher corpus in Section 10 during a separately authorized execution. That is an explicit change from Phase 7's eight-network-only scope. No unbounded dataset generation or training.
- Keep offline compute and prediction-time compute separate. Teacher generation, diagnostics, and training may use ordinary numerical libraries offline. Every numerical operation affecting a submitted prediction must use the permitted metered API.
- No network-identity lookup, cached per-network answer, evaluation-label access, or research-state loading inside a candidate. Ship shared learned weights only with their training provenance.
- Keep a valid official release distinct from a local research winner. Package a qualifying candidate; do not submit without a separate user instruction.
- At completion, commit completed Phase 8 artifacts and push to main, preserving unrelated work. Do not force-push.

Read and snapshot the current [allowed-code rules](https://raw.githubusercontent.com/AIcrowd/whest-starterkit/main/docs/concepts/allowed-code.md), [scoring](https://raw.githubusercontent.com/AIcrowd/whest-starterkit/main/docs/concepts/scoring-model.md), and [shipped-weights guidance](https://raw.githubusercontent.com/AIcrowd/whest-starterkit/main/docs/how-to/ship-weights.md) at P8-00. The plan author checked these on 7 September; execution must record the then-current version.

## 1. Frozen controls, numerical anchors, and targets

Freeze the actual bytes before using any cached result:

| Name | Source | SHA-256 | Recorded local raw MSE | Utilization | Projected product |
|---|---|---|---:|---:|---:|
| CONTROL7 | candidates/estimator_p7_final.py | 44723be93f8a432fa493a0218396a8346728277ba0a694c483fd9369ad26080f | 3.7853e-8 | 69.58% | 2.6337e-8 |
| CONTROL65 | candidates/estimator_p65_final.py | b06d91bc155f19e696c690c2b9956c2ba2426ba142de3c7d6dcf7f078344255e | 3.7854e-8 | 78.68% | 2.9784e-8 |
| LEGACY | whest-starterkit/estimator.py | ea8be8222b827607f65d166118500b55fcaae883b79f64b29785b46d86501ca1 | 1.223643e-6 | 9.8052% | 1.223643e-7 |
| REF3 | uncompressed Phase 6 K3-simple reference | Resolve source and receipt together | 3.63704648e-8 | Not inferred from reference timing | No valid metered score implied |

CONTROL7's recorded eight-network result is diagnostic: maximum residual approximately 4.2869 s. CONTROL65's unrelaxed maximum is 1.6429321 s. Contract validation or a relaxed run does not change that classification.

If a source hash differs, do not overwrite it. Locate the historical bytes or archive, verify the receipt association, and freeze a separate copy. Record any unresolved provenance as missing. A currently changing source is not a reproducible control.

Evidence anchors:

- [Phase 7 ledger](whest-starterkit/research/phase7/ledger.jsonl).
- [Phase 6.5 unrelaxed receipt](whest-starterkit/research/phase6_5/results/P65-03c-L10-p10-unrelaxed_20260907_012338.json).
- [Phase 6 reference panel](whest-starterkit/research/phase6/results/P6-03_reference_panel.json).
- [Reference predictions](whest-starterkit/research/phase6/predictions/P6-03_K3-simple_preds.npy).

The user's leaderboard image shows their point at rank 40, raw MSE about 3.64e-8 and utilization 78.68%. Leading points are around raw 2e-8 at 15-20% utilization. Treat these as user-supplied leaderboard evidence, not local paired observations, a training target for individual neurons, or a disclosure of the leaders' algorithms.

For each valid network m:

~~~text
B = 2**41 = 2199023255552
E_m = mean((prediction_final_m - truth_final_m)**2)
u_m = billed_FLOPs_m / B
S_m = E_m * max(0.1, u_m)
S_panel = mean_m(S_m)
~~~

Use the mean of actual per-network products. Never multiply aggregate means when utilization varies. A failed run uses the official failure result; keep pre-failure diagnostic predictions in a separate field.

| Utilization | Raw MSE needed for S=1e-8 | Raw MSE needed for S=6e-9 | Raw MSE needed for S=3e-9 |
|---:|---:|---:|---:|
| 10% or less | 1.00e-7 | 6.00e-8 | 3.00e-8 |
| 15% | 6.6667e-8 | 4.00e-8 | 2.00e-8 |
| 25% | 4.00e-8 | 2.40e-8 | 1.20e-8 |
| 35% | 2.8571e-8 | 1.7143e-8 | 8.5714e-9 |

Do not insist that a cheap candidate match CONTROL7 raw accuracy. For example, raw 6e-8 at utilization 0.15 has product 9e-9. Conversely, a cheaper candidate with much larger error may be worse.

## 2. What this phase changes

Use this table to prevent accidental repeats:

| Previous tested mechanism | Phase 8 distinction |
|---|---|
| Exact-radius sampling, spherical carriers | Carry angular analytical moments and reconstruct radius contributions exactly; changing sample radii alone is not lane A. |
| Norm/response pruning, Tucker projection, sampled atom tails | Lane D contracts frozen source contributions directly into requested statistics without maintaining the growing historical factor bank. |
| Scalar calibration, per-neuron blending, small residual regressions | Lane L learns the missing internal K3 slice update and carries a shared recurrent feature state across depth. |
| Sampled cheap surrogate with a costly estimated mean | Lane G uses an exact gate-residual decomposition, integrates the first source exactly, and measures source-specific variance and prefix cost. |
| Terminal Edgeworth corrections | Lane D includes a declared all-depth readout variant; it must distinguish terminal-only effects from changes to intermediate means and covariance. |

Four hypotheses, none established:

1. Angular state may remove an analytically known source of shared fluctuations and reduce the amount of higher-order state needed.
2. Frozen non-Gaussian source contributions may be sufficiently accurate while admitting cheaper contractions than a recurrent factor bank.
3. A small shared recurrent closure may learn the information needed by the moment update without reproducing the whole tensor.
4. Gate-residual contributions may admit cheaper integration than sampling the entire network.

Do not call any of these a reproduction of an undisclosed leaderboard solution.

## 3. Panels, splits, seeds, and claims

### 3.1 Development panel

Use the established eight networks from ../datasets/mini, split mini, in this exact order:

~~~text
0 logan-fitzgerald
1 william-graves
2 raymond-barnes
3 steven-rice
4 sarah-kelley
5 christopher-morales
6 cheryl-graham
7 renee-park
~~~

Resolve networks with the installed official dataset/MLP seed-context path. Verify weights, truth, order, width, and depth against [Phase 7 manifest](whest-starterkit/research/phase7/manifest.json). Do not reconstruct networks from an unrelated raw seed field.

### 3.2 Locked confirmation panel

Reserve mini rows 8 through 23, exactly 16 networks, for final confirmation. Save names and hashes at P8-00 without exposing their target arrays to fitting or candidate selection.

Check whether these rows were used in earlier work. If prior evaluation or fitting is found, mark them PREVIOUSLY_EXPOSED and select the next 16 unused mini rows in ascending row order, excluding every discovered used row. Record the substitution before experiments. If 16 unused rows do not exist, use available unused rows and report the limitation; do not invent independence.

The confirmation panel is opened only after Section 16 freezes at most two candidates. No tuning after opening it. A new revision requires a new, explicitly reserved confirmation set; this plan does not authorize an unlimited sequence of holdouts.

### 3.3 Teacher corpus

Training networks are independently generated with the benchmark's bias-free He-Gaussian architecture and the explicit seed list in Section 10. They are never mini evaluation rows. Teacher labels are approximate analytical states, not true means. Held-out teacher agreement and true benchmark accuracy are separate measurements.

### 3.4 Randomness

Production stochastic candidates use salts {0,1337,8888}; hold network weights and truth fixed. Use one documented production RNG stream with explicit draw order. Change only the estimator sampling seed, never the network-generating seed.

Training initialization seeds are {8101,8102,8103}. These are different from production sampling salts. Initial model screens use 8101; retrain qualifying architectures with 8102 and 8103.

Leave-one-network-out selection on the eight development networks is a stability diagnostic, not independent validation. Split by whole network, never by neuron, pair, layer, or trajectory segment.

## 4. Artifacts, runner contract, and honest execution records

Create during execution:

~~~text
phase8report.md
candidates/estimator_p8_<explicit_id>.py
whest-starterkit/scripts/p8_manifest.py
whest-starterkit/scripts/p8_build_candidate.py
whest-starterkit/scripts/p8_math_checks.py
whest-starterkit/scripts/p8_eval.py
whest-starterkit/scripts/p8_run_plan.py
whest-starterkit/scripts/p8_angular.py
whest-starterkit/scripts/p8_direct_sources.py
whest-starterkit/scripts/p8_gate_residuals.py
whest-starterkit/scripts/p8_select.py
whest-starterkit/research/phase8/
  control/ configs/ fixtures/ diagnostics/ predictions/ results/ release/
  offline/ teacher/ training/ weights/
  manifest.json
  experiment_manifest.json
  progress.json
  ledger.jsonl
~~~

These are implementation requirements, not claims that the helpers already exist. Provide these interfaces before using them:

~~~powershell
$env:PYTHONUTF8 = '1'
uv run python scripts/p8_run_plan.py --resume
uv run python scripts/p8_run_plan.py --stage P8-A1 --resume
uv run python scripts/p8_math_checks.py --lane angular
uv run python scripts/p8_eval.py --candidate ../candidates/estimator_p8_<id>.py --exp-id <ID> --panel dev8 --mode diagnostic --salt 0
uv run python scripts/p8_eval.py --candidate ../candidates/estimator_p8_<id>.py --exp-id <ID> --panel dev8 --mode unrelaxed --salt 0
~~~

The evaluator must use the official runner, preserve its complete raw output, and separately capture permitted diagnostic predictions. Compare direct and capture paths once on the control. A discrepancy must be resolved before attributing results to an algorithm.

Build explicit candidates, preferably without string-template code generation. Reject unknown configuration keys. Convert layer-indexed JSON keys to integers. Record requested and effective configuration plus its hash.

Mandatory behavioral verification:

- Scan generated source for unresolved placeholders, including literal {memory_mode}, {reset_start}, and {reset_period}. The matching Phase 7 reset source contained an always-false literal-string comparison; do not repeat that failure.
- Every nontrivial switch must have a fixture where toggling it changes the intended state, branch counter, or operation trace. An output difference alone is insufficient to show the intended algorithm executed.
- Check shape/provenance consistency at every representation transition.
- Record active layer events, source counts, pair counts, model dimensions, and RNG offsets in an offline trace whose predictions match production.
- A configuration description without numerical outputs is not a completed diagnostic. In particular, Phase 7 channel_interventions.json contains descriptions; do not treat them as measured intervention effects.

One ledger row per evaluation includes: source and weight hashes, config, parent, panel, backend/version, seed, full command, runner mode, stage, status, per-network raw and adjusted metrics, per-layer MSE, FLOPs, utilization, predict/setup/residual time, measured peak memory or MISSING, prediction checksum, and all artifact paths.

Use unique temporary files and atomically publish receipts only after the process exits and outputs validate. Never edit raw official receipts. Never write fake zero values for skipped scores. Use null and a reason.

States: PLANNED, INCOMPLETE, MATH_FAILED, API_BLOCKED, NUMERIC_FAILED, SCALE_BLOCKED, SKIPPED_COST, SKIPPED_GATE, REJECTED_ACCURACY, RESEARCH_ONLY, VALIDATED.

## 5. Shared cost, test, and promotion rules

### 5.1 Resources

Expected official caps: residual 0.4 s, prediction 120 s, setup 5 s, memory 8 GB, FLOPs B. Verify installed and graded settings. Engineering margins: 0.35 s, 90 s, 4 s, and 6 GB.

Before each new representation, calculate live tensor shapes and actual contraction costs. Production forecast above 0.95B is SKIPPED_COST unless the stage explicitly requests a bounded offline diagnostic. Never construct full 1024^3 or 1024^4 tensors.

Local live-memory ceiling is min(6 GB, 60% of currently available RAM). Process networks serially and release teacher arrays between them. Before a large teacher/training batch, run its prescribed pilot and estimate runtime, disk, and memory. Section 10 defines the local scale gate. Do not silently exceed it.

All online pilot passes, feature extraction, projections, model evaluation, decompositions, sample selection, and fallbacks are charged. Offline diagnostic work receives its own runtime/cost record and cannot be passed off as free online work.

### 5.2 Mathematical tests

Use n={3,5,8}, depth={1,2,4} for dense float64 identities; n={16,32}, depth=16 for floating-point stability. Include asymmetric weights, zero and negative weights, nearly zero variance, nonzero means, mixed activation regimes, neuron permutations, and positive scaling.

Dense identity tolerances: absolute 1e-10 and relative 1e-9, with a scale-aware denominator. Record errors rather than rounding to zero. For float32 production/reference comparison, report absolute drift and final RMS drift; do not infer bit equality from similar MSE.

### 5.3 Evaluation and expansion

Every mandatory production configuration that passes math/API/cost gates gets all eight development networks at salt 0. One-network smoke is operational only and cannot establish scientific rejection.

Common expansion gate, unless a lane specifies otherwise:

- mean product <=0.80*CONTROL7 with >=6/8 wins and worst score ratio <=1.25; OR
- utilization <=0.25 and raw MSE <=1.00e-7, with no numerical failures.

These are research gates, not release criteria. Preserve all nondominated measured points.

Common research-parent promotion: >=5% mean product improvement, >=6/8 wins, worst ratio <=1.15. Break relative ties within 1e-4 by lower cost, then lexicographic ID.

Stochastic qualifiers must pass all three salts before expansion into combinations. Report per-salt means and all 24 paired results. Deterministic CONTROL7 needs no repeated stochastic draws.

A preferred candidate can remain RESEARCH_ONLY if the local resource discrepancy persists. Do not repeatedly replay unchanged timing failures; continue scientific diagnostics and retain the distinction.

## 6. P8-00: freeze evidence and install the execution guardrails

Execute in order:

1. Inspect Git status and current source hashes. Record pre-existing dirty files. Freeze CONTROL7, CONTROL65, LEGACY, REF3 source and receipts.
2. Snapshot dataset version, seed context, evaluation panels, runtime versions, official rules, and the exact reference commit. The historical reference commit was 93d091a4c26c042bfffa28f2e76a81bc0aba94bb; verify.
3. Build the schema-valid ledger, strict configuration parser, effective-config round trip, and source placeholder checks.
4. Verify cached control predictions and costs. Reuse a cache only when source, weights, truth, backend, config, and runner settings match.
5. Direct unrelaxed first-network CONTROL7 check and direct/capture parity. Complete an eight-network baseline only if the matching receipt is missing or settings changed.
6. Record prior methods as measured configurations, not conclusions. Flag oracle means, assumed multipliers, incomplete runs, and source mismatches.
7. Build an exact candidate manifest for the mandatory rows in Sections 7-14. Expand conditional rows only when their parent gate is recorded.

Do not repair or rerun all of Phase 7. Audit only shared primitives and evidence needed by Phase 8.

## 7. P8-A0: angular identities and exact initialization

### 7.1 Representation

Let X be standard Gaussian in R^n. Write X=R U, with U uniform on the unit sphere and independent R~chi_n. Define Y=sqrt(n) U and a_p=E[(R/sqrt(n))^p].

For the bias-free ReLU network h_l:

~~~text
h_l(X) = (R/sqrt(n)) h_l(Y)
E[product of p activations under X]
    = a_p * E[the same product under Y]
a_p = (2/n)^(p/2) * Gamma((n+p)/2) / Gamma(n/2)
a_0=1; a_2=1; a_4=(n+2)/n
mu_G = a_1 * mu_A
M2_G = M2_A
C_G = C_A + (1-a_1^2) * mu_A mu_A^T
~~~

G denotes Gaussian input and A denotes angular input Y. Reconstruct raw moments before converting to centered cumulants. Multiplying every cumulant by a_p is incorrect.

For angular input:

~~~text
mu=0; C=I; K3=0
K4_ijkl = -2/(n+2) *
    (delta_ij delta_kl + delta_ik delta_jl + delta_il delta_jk)
K4_iiii = -6/(n+2)
~~~

The current scalar convention s4=c4*diag(M)^2 therefore requires initial c4=-6/(n+2), if the same metric convention is retained. Verify against the dense tensor; do not infer conventions from a variable name.

Precompute a_p for p=0..8 at supported fixture widths and 1024 using stable log-gamma offline. Store numerical constants as data with generation provenance. For unlisted widths, either provide a metered/general implementation or explicitly record unsupported widths; pass the official smoke shapes.

### 7.2 Required checks

- Radial/angular factorization for one-layer analytic moments, p=1..4.
- Angular input covariance and fourth cumulant against dense index formulas.
- Gaussian reconstruction of mean, second raw moment, and covariance, including nonzero activation means.
- Exact first-layer angular mean: Gaussian first-layer mean divided by a_1. Exact second raw moments agree because a_2=1.
- Mean-only final conversion must not apply an extra scale at every layer.
- Depth-16 forward identity on fixed X and corresponding Y, with their actual samplewise radius. This checks homogeneity, not expectation accuracy.
- Angular simulation checks use their own offline random stream and state their Monte Carlo uncertainty.

## 8. P8-A1/A2: matched angular closures

Implement one shared nonlinear moment primitive from verified REF3, with independent dense checks. Do not copy an unchecked builder. Gaussian and angular candidates differ only in the declared initialization, exact first-layer rule, and final conversion.

Mandatory configurations, eight networks each:

| ID suffix | Input representation | K3 state | Scalar K4 |
|---|---|---|---|
| A1-G-K2 | Gaussian | Set all incoming K3 slices to zero | Off |
| A1-A-K2 | Angular | Set all incoming K3 slices to zero | Off; intentionally crude angular closure |
| A1-G-K2K4 | Gaussian | Incoming K3 zero | Verified REF3 scalar recurrence; initial 0 |
| A1-A-K2K4 | Angular | Incoming K3 zero | Same recurrence; initial -6/(n+2) |
| A1-G-K3C65 | Gaussian | CONTROL65 retention/schedule | On |
| A1-A-K3C65 | Angular | Identical retention/schedule | On with angular initialization |

For K2 and K2K4 rows, do not allocate or transport U/V. Compute only the mean/covariance and scalar moments actually required. Set mixed K3 inputs to zero explicitly. Higher angular cumulants beyond the represented order remain omitted; report the approximation.

For the first layer, use exact Gaussian ReLU mean and bivariate second raw moment, converted to angular moments where relevant. Apply this exact first-layer treatment in both matched parents. Later layers use the same declared nonlinear formulas.

Record raw truth error, paired Gaussian/angular differences, covariance PSD diagnostics, final conversion size, and actual costs. Compare first to the matched Gaussian candidate, then to CONTROL7.

Conditional A2 gate: angular version improves its matched Gaussian product by >=10% with >=6/8 wins, OR passes the common expansion gate.

If triggered, test exactly:

- A2-A-LATE: angular K2K4 through layer 7, initialize K3 to zero there, carry new K3 births from layer 8 onward without pruning.
- A2-G-LATE: matched Gaussian control.
- A2-A-CAP4N and A2-A-CAP8N: angular K3 with upper-norm caps 4n and 8n after layers {6,10,13}; protect current births and count them in the cap.
- A2-G-CAP4N and A2-G-CAP8N: matched Gaussian controls.

This is a six-configuration interaction test, not a new retention sweep. Continue to lane D regardless of A's outcome. Freeze ANGULAR8 as the best qualifying angular product, otherwise null.

## 9. P8-D0/D1/D2: direct contractions of frozen K3 sources

This lane has an explicitly defined terminal approximation and a more expensive all-depth extension. Do not describe it as the full nonlinear recurrence or assume it is cheaper before counting.

### 9.1 Frozen pilot and sources

Use A1-G-K2K4 as pilot P. If lane A qualifies, also use A1-A-K2K4 as a second pilot. No teacher states or truth may enter a production pilot.

At every postactivation layer j=0..14, obtain the two newborn source blocks from the verified K3 nonlinear primitive, evaluated at that pilot's mean, covariance, and scalar K4 with incoming K3=0:

~~~text
B_j = sum_r Sym(u_jr, u_jr, v_jr)
source block 1: path births
source block 2: diagonal/repeated-slice restoration births
g_j = E_Gaussian[ReLU'(z_j)] = Phi(mu_pre_j/sigma_pre_j)
~~~

Use the actual source formulas from REF3 and verify a dense n=5 reconstruction. Source blocks are generated from the pilot, not read from a saved per-network teacher.

For target preactivation t>j, define the deterministic transport:

~~~text
Q_(j+1 <- j) = W_(j+1)^T
Q_(t <- j) = W_t^T diag(g_(t-1)) ... W_(j+1)^T
u' = Q_(t <- j) u
v' = Q_(t <- j) v
d_t += sum_r u'^2 * v'
S_t += offdiag((2*(u'*v') @ u'^T + (u'*u') @ v'^T)/3)
~~~

Here S_t[i,j]=K3_t[i,i,j], with row/column convention checked against dense tensors. For terminal-only evaluation, do not form S_t because only d_t is consumed.

Compute suffix matrices backward for a fixed target. Generate and contract each source block, then discard its factors. Store only the pilot, suffix matrices needed for the active target, and accumulated slices. This eliminates a persistent growing factor bank; it does not automatically eliminate quadratic dependence on depth.

### 9.2 D0 identities and implementation-cost gate

Mandatory tiny checks:

1. Frozen direct-source contraction equals forward transport of the same frozen source list.
2. Source order and summation order change only floating-point drift.
3. Terminal d-only contraction equals the diagonal of the dense transported tensor.
4. Separate path/slice contractions sum to concatenated-factor contractions.
5. Gaussian/Angular pilot conventions and final radial conversion remain distinct.

Cost profile must include every suffix matrix product, pilot pass, birth generation, source transport, and contraction. For all-target evaluation, count each target's work; no free shared prefixes or backward responses.

### 9.3 Mandatory D1 terminal screen

For each available pilot, evaluate exactly:

- D1-TERM: replace only terminal incoming d_15 with the frozen-source sum; keep the pilot terminal covariance, mean preactivation, and scalar K4. Recompute the terminal ReLU mean.
- D1-TERM-REF: offline diagnostic using matched REF3 terminal d_15 in the same pilot terminal state. This is an oracle substitution, never a candidate or a valid score.

D1-TERM-REF measures whether the terminal channel has enough headroom in this pilot. A negative D1 result does not close the all-depth variant.

### 9.4 Mandatory D2 all-depth frozen-slice screen

For t=1..15 calculate frozen d_t and S_t from the same pilot, independently of the later corrected trajectory. Then run a second forward mean/covariance/scalar-K4 pass that consumes these fixed slices in the verified nonlinear primitive. No recurrent K3 factors are used during this second pass.

This is a one-shot frozen-slice correction, not a full Jacobian/adjoint expansion. It includes intermediate mean/covariance changes but ignores the effect of those changes on the previously generated source slices.

Evaluate D2-ALL for each available pilot. First forecast cost. If >0.95B, run the source construction as a bounded offline diagnostic only when memory passes, and record an operation-count estimate as diagnostic. Do not assign a 0.1 multiplier.

If a D2 candidate passes the common expansion gate, test D2-REFRESH2: use its corrected trajectory as the new pilot, regenerate all sources/slices once, and run the corrected pass again. Charge both rounds. No convergence loop, adaptive iteration count, or more than two rounds.

If the offline D2 accuracy is promising but cost fails, try exactly one scheduling optimization: block targets in ascending groups of four, reuse identical computed products only when algebraically identical, and record the actual saved operations. Do not invent an adjoint approximation without adding its precise equations and parity tests to the branch record. An unresolved contraction is API_BLOCKED or SKIPPED_COST, not a successful method.

Freeze DIRECT8 as the best qualifying metered candidate, otherwise null. If only offline accuracy succeeds, preserve it as a mathematical lead with its measured cost deficit.

## 10. P8-L0: independent teacher corpus and hardware gate

Lane L must execute its bounded feasibility stage even if A and D fail. Its purpose is to learn a compact update, not fit eight public answers.

### 10.1 Teacher definition and saved targets

Teacher T is the source-verified, uncompressed REF3 K3-simple recurrence, with all 16 ReLUs and correct transpose convention. Preserve its approximation label.

For each network save per-layer:

- Input weights or their deterministic generator recipe and hashes.
- Preactivation mu, C, s3, s21, scalar c4, and the metric convention needed by the nonlinear update.
- Postactivation mu, C, scalar c4.
- The existing cheap pilot trajectory.

Store float32 compressed arrays in per-network shards. Do not save full dense K3 or all factor banks across all layers. Teacher factor banks may exist for one live network only.

### 10.2 Seed schedule and split

Use a production-independent offline generator with standard-normal weights scaled sqrt(2/n), no biases, depth 16. Pin generator/library version. Record all weight hashes and ensure no match with evaluation networks.

For each n in {64,128}, generate:

~~~text
train: network seeds 8100000 + 10000*n + i, i=0..31
validation: same base + i, i=32..39
teacher-test: same base + i, i=40..47
~~~

The different-width sets are independent. Do not call small-width success a width-1024 result.

Before this batch, pilot two n=128 teacher networks. Local batch gate: forecast <=60 minutes total, live memory below Section 5, and <=10 GB new disk. If exceeded, reduce only the initial smoke corpus to 8 train/2 validation/2 teacher-test per width and record SMOKE_ONLY; preserve the full corpus stage as SCALE_BLOCKED and prepare the same resumable offline job for a separately available execution environment. Continue other lanes.

After the small-width gate in Section 12 passes, generate the width-1024 extension:

~~~text
train: seeds 8200000+i, i=0..15
validation: i=16..19
teacher-test: i=20..27
~~~

Pilot two full-width training networks first and apply the same 60-minute batch and memory/disk gate. No evaluation rows may substitute for this corpus. If external capacity is needed, report the exact forecast and required environment; do not silently launch a capacity-heavy job.

No high-sample Monte Carlo teacher generation is part of this plan. Teacher agreement is evaluated on generated networks; actual mean accuracy uses baked evaluation labels only through the declared evaluation procedure.

## 11. P8-L1: literal learned closure architecture

### 11.1 State and prediction target

Keep analytical mu, full C, and the verified scalar c4 update. Replace the transported K3 factors by a dimensionless neuron feature matrix H of shape n by d.

The model predicts **incoming preactivation slices s3_i and s21_ij** consumed by the existing nonlinear moment primitive. It does not output the final answer directly. s21 is generally asymmetric because it represents K3[i,i,j]. Set its diagonal to zero; s3 owns the diagonal.

At layer 0 use exact Gaussian mean/covariance and zero incoming K3; initialize H from features of that state. First learned slice prediction is at layer 1. Default training/input distribution is Gaussian. Angular retraining is conditional in Section 15.

Given W_ref=W^T and incoming H:

~~~text
sigma_i = sqrt(max(C_pre_ii, 1e-12))
a_i = mu_pre_i / sigma_i
R_ij = C_pre_ij / (sigma_i sigma_j)
sigma_rms = sqrt(mean(sigma^2))
J = (W_ref @ H) / sqrt(max(row_sum(W_ref^2),1e-12))[:,None]
K = R @ J / sqrt(n)
~~~

Build per-neuron feature vector x_i in this exact order:

~~~text
clip(a_i,-8,8)
log(max(sigma_i/sigma_rms,1e-6))
Phi(a_i)
phi(a_i)
mean_j(R_ij^2)
mean_j(R_ij^3)
row_sum(W_ref^2)_i / mean(row_sum(W_ref^2))
clip(c4,-1,1)
l/15
1/sqrt(n)
J_i[0:d]
K_i[0:d]
~~~

Use shared affine layers with tanh activation. H_new = tanh(MLP_H(x_i)); all layers share weights, with l/15 supplied as a feature. Initialize H at the input to zero.

Node head: b3_i = MLP_3([x_i,H_new_i]).

Pair head, evaluated in row blocks of 64, uses:

~~~text
[H_new_i, H_new_j, a_i, a_j,
 log(max(sigma_i/sigma_rms,1e-6)),
 log(max(sigma_j/sigma_rms,1e-6)),
 R_ij, R_ij^2, l/15, 1/sqrt(n)]
~~~

Pair ordering must be preserved. Do not symmetrize s21. The pair head output is b21_ij.

Denormalize with fixed training-only scales t3 and t21:

~~~text
s3_i = sigma_i^3 * t3 * b3_i
s21_ij = sigma_i^2 * sigma_j * t21 * b21_ij, i != j
~~~

t3/t21 are RMS standardized teacher slices over training networks and layers, each floored at 1e-6. Freeze them before validation. Input feature standardization, if used, is fitted on training only and stored. Use a linear final head; no prediction clipping other than the explicitly declared a/c4 features and variance floor.

Feed these slices into the shared nonlinear primitive, calculate the next mu,C,c4, and carry H_new forward. Validate positive scaling and permutation behavior. A covariance PSD failure is a failure; do not silently repair it by a new decomposition.

### 11.2 Mandatory model grid

All MLPs have two hidden layers, equal hidden width h. Hidden activation tanh; output linear except H_new's final tanh.

| ID | H width d | Hidden width h | Recurrence |
|---|---:|---:|---|
| L1-H0 | 0 | 16 | No H/J/K; local and pair features only |
| L1-H8 | 8 | 16 | Full declared state |
| L1-H16 | 16 | 32 | Full declared state |

H0 is a required ablation, not a substitute for the recurrent models. The pair computation is O(n^2) times fixed model width, not a tensor-rank bank that grows with depth. Still bill its actual operations and memory.

### 11.3 Training recipe

Use offline float32 PyTorch, deterministic seeds and one active training job. AdamW: learning rate 1e-3, weight decay 1e-5, gradient norm clip 1.0. One network per step, accumulate four network losses before each optimizer step. Shuffle whole networks using the training seed.

Stage TF: 50 epochs teacher-forced states, using all neurons for s3 and 4096 off-diagonal pairs per network/layer sampled without replacement where possible. Loss is mean squared standardized s3 error plus mean squared standardized s21 error. Pair losses get equal aggregate weight to node losses.

Stage RO: 100 epochs full 16-layer free rollout. Learning rate 3e-4. Retain slice losses and add:

~~~text
L_mu = mean_l,i [ (mu_student-mu_teacher)^2 / q_l ]
L_C  = mean_l,selected_pairs [
    (C_student-C_teacher)^2 / (sigma_teacher_i^2*sigma_teacher_j^2 + 1e-12)]
q_l = mean_i(mu_teacher_i^2 + C_teacher_ii) + 1e-12
L = L_slice3 + L_slice21 + 10*L_mu + L_C
~~~

For L_C include all diagonals plus the same 4096 off-diagonal pairs. Preserve the full covariance during the forward pass; pair subsampling changes the training loss, not the state transition.

Check validation every five epochs. Select the checkpoint with lowest validation normalized final-mean teacher MSE; tie-break by lower validation full-rollout loss, then earlier epoch. No evaluation truth in early stopping. Keep all fixed epochs unless a numerical failure occurs; do not search learning rates.

Use checkpointing to bound memory. If full-gradient rollout exceeds the hardware gate, use truncated backpropagation of four layers with detached boundary state, explicitly named TBPTT4 for all compared architectures. Do not silently mix training procedures.

## 12. P8-L2/L3: feasibility gates, full-width training, and metered conversion

### 12.1 Small-width feasibility

Measure each model on all held-out teacher-test networks at both widths. Required reports:

- One-step standardized slice MSE relative to zero-slice predictor.
- Full-rollout final teacher MSE relative to the matched analytical K2K4 pilot.
- Per-layer drift and covariance PSD diagnostics.
- Permutation and positive-scaling error.
- Estimated and measured feature/model FLOPs.

Full-width training gate: >=50% reduction in final teacher-discrepancy MSE against the pilot at both widths, >=75% teacher-test network wins, no divergent rollout, and forecast utilization <=0.25 at width 1024. Use this only as a feasibility gate; small-width gains do not establish competition accuracy.

If none qualifies, execute one declared repair: rollout loss with L_mu weight 30 instead of 10 for H8 only, same seeds/epochs. If it also fails, stop expansion and retain the measured negative results. Do not replace the architecture with an unspecified larger network.

### 12.2 Full-width continuation

For at most two qualifying architectures ranked by validation final teacher MSE, fine-tune on the Section 10 width-1024 training corpus:

- 30 teacher-forced epochs at 3e-4.
- 60 rollout epochs at 1e-4.
- Same loss, batch accumulation, pairs, weight decay, and checkpoint selection.
- Width-1024 validation alone selects the final checkpoint; teacher-test remains evaluation only.

For qualifying architecture(s), repeat the full training recipe with initialization seeds 8102 and 8103. Report variation; do not select the best seed using development truth. Choose a single seed by teacher-validation loss, or the fixed arithmetic mean of the three models if its fully metered cost qualifies. An ensemble is a separate candidate.

### 12.3 Conversion

Export all arrays as pickle-free numerical data with an explicit layer/shape schema. Implement only affine matmul, tanh, feature contractions, and the existing analytical primitive through flopscope. No runtime torch, autograd, trainer, or dataset files.

Verify exported/in-memory predictions on tiny networks and two full-size independent teacher-test networks before public development scoring. Then evaluate all eight development networks. Report teacher error and true-target error side by side.

Freeze LEARNED8 only if it passes the common gate on actual true-target products. A good teacher fit with poor target accuracy is not a breakthrough. Do not claim improvement beyond the teacher from distillation alone.

## 13. P8-G0/G1: exact gate-residual decomposition and variance-cost diagnostic

### 13.1 Exact identity

Use Gaussian K2K4 pilot P to define fixed gates:

~~~text
D_l = diag(1[mu_pre_l >= 0])
A_l = D_l W_l^T
r_l(X) = ReLU(z_l(X)) - D_l z_l(X)
h_l(X) = A_l h_(l-1)(X) + r_l(X)
h_(-1)(X)=X
Q_15 = I
Q_l = A_15 A_14 ... A_(l+1), l=0..14
h_15(X) = A_15 ... A_0 X + sum_l Q_l r_l(X)
E[h_15] = sum_l Q_l E[r_l]
~~~

Gates are deterministic functions of the online analytical pilot. They must not be refitted on each sampled trajectory. Residuals can be nonnegative locally while their transported contributions have mixed signs.

For l=0, the Gaussian residual expectation is exact:

~~~text
E[r_0] = row_norm(W_0^T) / sqrt(2*pi)
~~~

because the preactivation mean is zero and E[D_0 z_0]=0. Its output contribution Q_0 E[r_0] requires no sample averaging.

Verify the samplewise identity on asymmetric tiny networks, both D=0 and D=I controls, and mixed gates. Check the omitted linear-input term samplewise; it vanishes in expectation, not per finite sample.

### 13.2 Mandatory diagnostic

Use 2048 independent Gaussian pilot samples and 4096 independent evaluation samples per development network, streamed in batches of 256. Antithetic evaluation is a separately named diagnostic; do not assume pair averages are independent individual samples.

Measure output-space contributions q_l=Q_l r_l for l=1..15, plus the full output and the exact l=0 source. Save per-source means, variances, and the 15x15 covariance matrix averaged across output coordinates.

For diagnostic allocation use pilot sample statistics only. Measure the cost c_l of computing a sample's prefix through l, its residual, and transporting it by Q_l; include Q/pilot construction as fixed overhead. Do not equate prefix depth with measured billed cost.

Compute two forecast classes:

1. Independent source streams: variance sum_l V_l/N_l and cost sum_l c_l*N_l plus fixed overhead.
2. Shared-prefix groups with boundaries {3,7,11,15}: use the measured variance of each group's summed transported contribution, including cross-covariances inside the group. Four groups are layers {1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13,14,15}.

For each class allocate N_l proportional to sqrt(V_l/c_l) under remaining total budgets {0.15B,0.25B}. Round down to complete batches of 32, with minimum 32 per active source/group; recompute true cost and adjust by removing batches from the smallest variance-reduction-per-FLOP allocation until it fits. Use pilot variances with floor 1e-12.

The predictor's pilot samples are additional cost unless reused under an explicitly unbiased sample-splitting design. Default: discard them from the production mean estimate.

Use evaluation-stream observations to assess the forecast; do not score the theoretical variance as measured MSE.

Expansion gate: an affordable allocation forecasts product <=0.70*CONTROL7 using measured residual variance and all overhead, and its independent evaluation diagnostic confirms >=30% improvement over matched-cost ordinary sampling on >=6/8 networks. If not, stop before online implementation. An exact algebraic identity alone earns no promotion.

## 14. P8-G2: metered gate-residual candidates

If G1 triggers, implement exactly the qualifying combinations of:

~~~text
allocation={independent_sources, four_shared_prefix_groups}
total_budget={0.15B,0.25B}
online_pilot_N=512
pilot_batch=256
allocation_rounding=32
~~~

Use a new independent evaluation stream after the online pilot. The smaller online pilot's allocations must be tested as such; offline 2048-sample diagnostics cannot secretly supply allocation parameters. Every prediction computes its own pilot features, gates, suffix matrices, allocation, and means from the received network.

Return earlier layer rows from the analytical pilot; final row is the exact first-source contribution plus estimated remaining contributions. All-layer error remains reported.

Compare with matched-cost ordinary sampling, LEGACY, and CONTROL7. Run all three production salts for a qualifying candidate.

As a tiny implementation invariant, subtract fixed Gaussian analytical centers from each residual sample and add them back exactly. This must give the same estimator within numerical tolerance. Subtracting constants does not reduce variance; do not spend another full panel evaluating this as an improvement.

Do not add a biased analytic blend, importance sampler, or neural control without a new explicit experiment specification. Freeze GATE8 if a candidate meets the common gate, otherwise null.

## 15. P8-X: restricted combinations

Freeze individual lane winners and configurations before combining them. No combination may use an oracle, cached teacher center, or unmetered online feature.

Execute only these triggered combinations:

1. If ANGULAR8 and a recurrent learned closure qualify independently: retrain the best learned architecture on angular teacher trajectories with the exact A0 initialization/conversion. Reuse architecture and training hyperparameters; generate angular teachers only for the same independent seed splits. Gaussian-trained weights applied to angular states are not an acceptable substitute. One model family, three initialization seeds only if the first qualifies.
2. If DIRECT8 qualifies: apply the existing one-level Strassen kernel to products with min dimension >=512, after tiny shape/numerical checks. Measure actual FLOPs and residual time. No assumption that the original 11.6% saving transfers to these shapes.
3. If LEARNED8 qualifies and has utilization <=0.20: add direct frozen-source correction only to its final preactivation s3, using the learned trajectory as the pilot. Compare against the identical learned parent; count the whole additional source pass.
4. If GATE8 qualifies: do not automatically blend it with a deterministic lane. First compute a development-only costed blend screen from independent predictions, accounting for the sum of both candidate costs. If costed headroom >=15%, fit one global alpha in {0,.05,.10,.20,.40,.60,.80,1} by outer network LOO and run the resulting fully metered combination. Include all three salts.

Maximum four mechanism combinations, excluding repeated initialization/salts. No Cartesian parameter search or final-neighbor tuning on the confirmation panel.

## 16. P8-F0/F1: candidate freeze and independent confirmation

Select at most two candidates using development data:

- Candidate A: lowest eligible mean projected product.
- Candidate B: best eligible candidate from a different mechanism with product <=1.25*A. If none, only A.

Eligibility: complete eight-network results, >=20% product improvement over CONTROL7, >=6/8 wins, worst ratio <=1.15, required training/salt checks, and no unresolved mathematical or provenance issue. Resource-failed but scientifically eligible candidates may be selected for research confirmation with that label.

Freeze source, weights, configuration, constants, and package hashes before opening confirmation targets. Selection is independent of whether a headline target was reached.

Evaluate candidate(s) and CONTROL7 on the locked 16 networks, using official baked truth and all required production salts. Same runner modes/settings for paired comparisons. Record results even when worse. No retraining, alpha adjustment, gate-threshold change, or candidate replacement after opening the panel.

Confirmation success: >=20% lower mean product, >=12/16 wins, worst per-network ratio <=1.25, and no numerical failures. Target crossing is reported separately. If fewer than 16 unused networks were available, report the actual panel size and do not assert the 12/16 criterion.

Use network-level paired bootstrap, 2000 resamples with fixed offline seed 8199, to report an interval for mean score difference and score ratio. Average a network's salts before resampling; do not treat salts or neurons as independent networks.

For a release claim, require unrelaxed official-runner validity on every tested network and applicable salt. A diagnostic success with failed caps remains RESEARCH_ONLY.

## 17. P8-F2: packaging, final report, and Git

Verify the exact finalist bytes, export parity, source/config event checks, and all candidate-relevant identities. Repeat full evaluations only if source/weights/settings changed or a specific unresolved failure requires it.

Single-file candidate:

~~~powershell
uv run whest validate --estimator ../candidates/estimator_p8_final.py
uv run whest package --estimator ../candidates/estimator_p8_final.py --output research/phase8/release/submission_phase8.tar.gz
~~~

Learned/multifile candidate: package a dedicated folder containing estimator.py, shared numerical weights, and required helper modules only. Validate the folder and the resulting archive with the installed official CLI. Do not package training code, teacher trajectories, public labels, logs, or per-network predictions.

Check the current package size/file-count limits. Archive extraction must reproduce the source and array hashes; verify shape, dtype, inference order, and pickle-free loading.

Required phase8report.md sections:

1. Best measured raw MSE, billed utilization, actual product, validity, comparison with CONTROL7, and reached/missed targets.
2. Complete experiment manifest with every mandatory row and triggered row accounted for; missing work is INCOMPLETE or SCALE_BLOCKED.
3. Full paired development and confirmation vectors, all layers, all seeds/salts, costs, times, and memory.
4. Requested/effective settings, behavioral branch checks, source/weight hashes, and source provenance.
5. Angular raw-moment identities and the measured effect of the representation change.
6. Direct-source frozen/forward parity, terminal versus all-depth outcomes, and actual contraction costs.
7. Learned model definition, corpus splits/hashes, teacher limitations, one-step versus rollout performance, held-network validation, training cost, and export parity.
8. Gate-source variance/covariance and cost tables, actual online allocations, and forecast-versus-observed errors.
9. Frontier of raw MSE versus utilization; hypothetical target curves clearly separate from measured points.
10. Concrete unresolved mechanisms. Do not repeat reports' unsupported claims of impossibility, universality, or leaderboard-method attribution.

Keep these labels separate throughout:

~~~text
USER_SUPPLIED_LEADERBOARD
OFFLINE_TEACHER_OR_ORACLE
DIAGNOSTIC_PROJECTED_PRODUCT
VALID_LOCAL_RUNNER_RESULT
OFFICIAL_SUBMISSION_RESULT
~~~

A passing contract smoke is neither resource validation nor a leaderboard result. This plan creates no official submission result.

Inspect Git status and diffs. Stage only completed Phase 8 files and necessary explicitly scoped dependencies. Preserve earlier untracked Phase 7 material and unrelated edits. Commit with a descriptive message and push to main. If main advanced, fetch and reconcile without force, resetting unrelated files, or broad staging.

## 18. Literal execution order and completion checklist

| Order | Stage | Mandatory work | Triggered expansion |
|---:|---|---|---|
| 1 | P8-00 | Freeze controls/panels; rules snapshot; evaluator/config guards | Replay only invalid caches |
| 2 | P8-A0 | Angular identities and exact initialization | None |
| 3 | P8-A1 | Six matched Gaussian/angular configurations | Six A2 configurations if gate passes |
| 4 | P8-D0 | Frozen-source identities and explicit cost forecast | None |
| 5 | P8-D1 | Terminal direct-source and oracle-channel screens | Second angular pilot only when available |
| 6 | P8-D2 | All-depth frozen-slice screen, metered or explicitly offline | One refresh round; one blocked scheduling rewrite if eligible |
| 7 | P8-L0 | Teacher parity, two-network hardware pilot, bounded small-width corpus | Width-1024 corpus only after learned feasibility |
| 8 | P8-L1 | H0, H8, H16 models; teacher-forced and rollout training | One fixed H8 loss-weight repair if all fail |
| 9 | P8-L2 | Held-network small-width gate | At most two full-width architectures and required seeds |
| 10 | P8-L3 | Export and metered parity for qualifying learned models | Three-model ensemble only as separately costed candidate |
| 11 | P8-G0/G1 | Exact identity; source covariance/cost diagnosis | Up to four G2 candidates if gate passes |
| 12 | P8-X | Compatibility review and frozen winners | At most four declared combinations |
| 13 | P8-F0/F1 | Freeze candidates and decision record | Locked confirmation for at most two qualifiers |
| 14 | P8-F2 | Complete report, artifact/release status, commit/push | Package qualified finalist or clearly marked research archive |

Before ending, every mandatory configuration must have a valid receipt or a specific documented obstruction. A skipped conditional expansion must cite the actual numeric gate. A long expected runtime is not permission to mark missing work complete.

After interruption:

1. Read this plan, progress, experiment manifest, and latest ledger entries.
2. Verify frozen parent/source/weights/panel hashes.
3. Resume the first incomplete mandatory row or triggered row; preserve finished networks/checkpoints.
4. Continue independent branches when one is blocked.
5. Finish with the self-contained report, including remaining blocked work and the exact resource/input needed.

Do not stop at code scaffolding, a named-paper search, or a single negative smoke. Do not turn a failed implementation into a scientific conclusion. The objective is a better algorithm, and the obligation is measurable, reproducible execution even if the target is missed.

## 19. Evidence, references, and plan-author limitations

Primary background:

- [Wu et al., Estimating the expected output of wide random MLPs more efficiently than sampling](https://arxiv.org/abs/2605.05179) and [official code](https://github.com/alignment-research-center/mlp_cumulant_propagation): reference cumulant machinery, not proof that any Phase 8 approximation works.
- [Decomposing neural networks as mappings of correlation functions](https://doi.org/10.1103/PhysRevResearch.4.043143): background for organizing higher-order contributions. Do not substitute statistics averaged over random weights for the fixed-network integration target.
- [Han et al., Uniformly accurate machine learning-based hydrodynamic models for kinetic equations](https://doi.org/10.1073/PNAS.1909854116): precedent for learned closure methodology in another domain, not benchmark validation.
- Official code, scoring, and shipped-weight links in Section 0 govern the execution contract.

The angular homogeneity and gate-residual identities are exact mathematical starting points. Their approximate closures and integration economies are hypotheses. Direct frozen-source propagation is an explicitly approximate computational graph. The learned architecture is a specified proposed model, not a trained artifact.

During plan preparation, current local controls, selected receipts, prior experiment configurations, Phase 7 plan structure, and official rules were inspected. CONTROL7's source hash was checked. **No Phase 8 identities were executed, no estimator was built, no training corpus was generated, and no new score was measured.** The executing agent must implement and independently verify the equations before drawing conclusions.

Evidence priority: actual source/weight bytes and complete unmodified receipts; paired predictions and state traces; effective configuration; derived tables; historical prose last. The user's instruction to use prior numbers rather than prior report analysis applies throughout.
