# Phase 4.6 Research Plan: Escaping the Closure Basin (Muse Session)

## Mission

Develop a Phase 2 estimator that reaches an adjusted score below `8.00e-08` on
the fixed eight-MLP validation panel (stretch `< 4.00e-08`), while remaining
legal, private-safe, reproducible, and failure-free.

Reference point:

- current control: `whest-starterkit/estimator.py` (`ea8be822...`,
  Blended Dual-Kernel Hermite $\lambda=0.20$, $s_0=0.998319$, $\alpha=0.110$,
  $N=4200$);
- current local eight-MLP adjusted score: `1.223643e-07` (raw `1.223643e-06`,
  util `9.81%`, resid `~0.18s`);
- best stable mean variant: `candidates/estimator_p5_chosaul_lam05.py`
  (`93ee3c28...`, lam=0.50 + exact Cho-Saul L0, `1.215291e-07`, `-0.68%`,
  6W-2L) — submitted as **#329939**, NOT a promoted champion;
- Phase 4.6 primary target: `< 8.00e-8` (`-34.6%` from control);
- Phase 4.6 stretch target: `< 4.00e-8` (`-67.3%` from control).

Only the fixed eight MLPs are in scope for verification decisions.
Broader confirmation is deferred until the user explicitly requests it.

Do not submit anything to AIcrowd without explicit user approval.
(#329939 went out under explicit approval as a best-mean submission.)

## Why Phase 4.6 exists (what Phase 4 + 4.5 closed)

Fused-error arithmetic (all `[measured]` on the 8-panel):

- analytic branch $A \approx 1.389\times10^{-6}$ (89% of fused error);
- sampling branch $B \approx 1.15\times10^{-5}$ (WMC, $N=4200$);
- optimum $\alpha^* = A/(A+B) \approx 0.11$, fused $\approx 1.22\times10^{-6}$ raw.
- e-8 needs raw $\le 0.80\times10^{-6}$: $A$ down 38%, or $B$ down 83%
  (6.1x), or a combo (e.g. $A \to 1.0\times10^{-6}$ and $B \to 5\times10^{-6}$).

Closed with measurements (see `PHASE4_RECORD.md`, `phase4.5record_muse.md`):

1. All analytic refinements move $A$ by <1%: higher Hermite ($\rho^3,\rho^4$),
   full-exact Cho-Saul (infeasible cost, Taylor-bound <1%), Edgeworth
   recurrence (hurts), regime/recursive/continuous scaling ($\le 0.1\%$),
   low-rank residual ($R^2_U = 3.78\% \ll 15\%$), isotropic SVD residual.
2. All sampling refinements correlate away or die to noise: Hadamard
   $-15\%$ raw but tri-blend $-0.52\%$ fused; Hermite-CV $-65\%$ raw but
   $+0.17\%$ fused (analytic-center correlation); unbiased split-sample CV
   (per-neuron AND pooled single-$\beta$) $0.00\%$ (center noise);
   Stein-input $0.00\%$; active subspace $R^2 \sim 10^{-3}$ (Lane C2 dead).
3. Late-jump ceiling 1.9% (L14/L15 error ratio 0.981); shared residual law
   negative under LOO; boundary-N (10.00% util) seed-fragile 5W-3L.
4. Above-floor scaling disproven (raw $-30\%$ at $K=8$ becomes adj $+73\%$).

Stacking the only two surviving micro-wins (lam=0.50 + exact L0) gives
$-0.68\%$ stable — ~1/50 of what e-8 needs. Further tuning in the old
families is scientifically closed. Phase 4.6 therefore bets on: (a) one
piece of genuinely new mathematics, (b) one mis-gated oracle revisited,
(c) variance anatomy before construction, (d) a methodology fix (stop
fitting on 8 MLPs), (e) engineering that funds samples inside the floor.

## Agent operating contract

- Use `uv` for every Python command; `$env:PYTHONUTF8="1"` on Windows.
- Preserve `estimator.py` until all promotion gates pass; one candidate
  file per method under `candidates/`.
- Only Phase 2 `v2-phase2` width-1024 depth-16 MLPs; same fixed eight and
  ordering; common random numbers for paired comparisons.
- Oracle calculations stay in offline research scripts; never leak target
  values, network IDs/names, or target-derived per-network choices into a
  deployable estimator.
- No `numpy`/`scipy`/BLAS/compiled kernels/threads/FFI in deployables —
  `flopscope.numpy` only (see `docs/concepts/allowed-code.md`). Residual
  400 ms is plumbing, not compute.
- Shipped data files (`.npz`, tables, weights) via `setup()` are the legal
  way to bring offline compute (see `docs/how-to/ship-weights.md`).
- Record every completed experiment (including negative/failed) in
  `phase4.5record_muse.md` (continued) with the standard schema; label
  conclusions `measured` / `derived` / `supported interpretation` /
  `untested hypothesis`.
- Promotion gate (unchanged): $\ge 10\%$ mean improvement over
  `1.223643e-7`, $\ge 6/8$ wins, worst regression $\le 10\%$, 0 failures,
  resid $< 0.35$s (prefer $< 0.30$s), util $\le 10.00\%$ with grader margin
  (target $\le 9.85\%$), LOO + refit agree, 3-salt survival for stochastic
  gains, clean-process + package reproduction.

## Research portfolio

- **Lane E — TAP/Onsager correction (new math): 40%.**
- **Lane F — shrunk moment-restart (mis-gated oracle revisited): 20%.**
- **Lane G — variance anatomy + late-layer branching: 15%.**
- **Lane H — 92-MLP universal-prior program (methodology fix): 15%.**
- **Lane I — compaction engineering (fund samples inside floor): 10%.**

After at most two consecutive experiments in one lane, run one from another
lane unless the current sequence is ablating a >10% gain.

---

## Lane E — TAP/Onsager reaction correction (headline bet)

### Hypothesis

Every analytic propagator used so far is *naive mean-field* Gaussian
closure. In statistical physics (spin glasses, AMP/compressed sensing) the
deterministic correction to naive mean-field is the Onsager reaction term:
each layer's mean is adjusted by a cheap susceptibility-weighted feedback,
computable from quantities already tracked (gains $\Phi(a)$, variances,
weight Gram diagonals) at $O(n^2)$ extra cost, zero fit, zero sampling.

Phase 4.5 evidence it fits this hole `[supported interpretation]`:

- the systematic $+0.16\%$ energy inflation ($s_0 = 0.998319$) is exactly the
  signature of a missing $O(1/n)$ reaction term accumulating over depth;
- Lane B2 proved the recurrence lives on a self-consistent "Gaussianized
  manifold" that true moments kick it *off* — i.e. the fixed point itself is
  shifted, which is what a reaction term moves;
- the residual is isotropic/full-rank (SVD), consistent with a missing
  *scalar-per-layer* feedback rather than a low-rank direction.

### Required derivation (offline, before any 8-panel run)

1. Write one ReLU layer $z = W^\top h$, $h' = \mathrm{ReLU}(z)$ with
   cavity/leave-one-out argument; extract the reaction term for the mean
   as a function of $(\mu_{\rm pre}, \sigma_{\rm pre}, \bar\Phi, \|W\|_F^2/n)$.
2. Show it reduces to a per-layer scalar-plus-gain-weighted correction
   (must be $O(n^2)$, must use only white-box observables).
3. Pre-gate (offline, training MLPs — see Lane H for the pool): $\ge 5\%$
   layer-wise bias reduction vs the uncorrected recurrence. If <5%, KILL
   Lane E in one experiment — do not build the estimator.

### Deployable form (only if pre-gate passes)

Single extra elementwise plateau per layer inside `_blended_hermite_covariance`
($O(n)$ FLOPs, no util change), then standard $s_0$/blend. Fused decision
metric only.

### Lane E gate

- Continue to full 8-panel iff offline pre-gate $\ge 5\%$ on training pool.
- Promote component iff fused $\ge 10\%$, $\ge 6/8$ wins.
- This is the only lane with a headroom story for 30%+. Everything else is
  stacked single-digits.

## Lane F — shrunk moment-restart (the B2 oracle tested the wrong endpoint)

### Hypothesis

Lane B2's kill (`+60$–$104\%` regression from injecting *pure true* moments)
tested shrinkage $\alpha = 0$ (all-sample, off-manifold). The deployable was
always $\hat\mu_k = 0.9\,c_k + 0.1\,\bar h_k$ (near-manifold): mostly the
self-consistent analytic state with a small unbiased drift correction,
then Gaussian propagation for the remaining layers. Small corrections stay
on-manifold and compound *for* us instead of against us.

### Required experiment (one full eval, shrinkage fixed a priori)

1. Pilot: single late layer $k = 12$, shrinkage $0.9/0.1$ fixed BEFORE the
   run (not tuned). Shrink covariance identically toward analytic
   ($0.9\,\Sigma_k^{\rm cov} + 0.1\,\hat\Sigma_k^{\rm samp}$ with Ledoit-Wolf
   style diagonal loading for PSD safety).
2. Resume blended-Hermite recurrence $k+1 \to 15$; standard $s_0$/blend.
3. Cost check first: one extra sampled covariance at layer 12 costs
   $O(N n^2)$ for that layer only (~1/16 of MC) — must hold util $\le 9.85\%$;
   fund by dropping $N$ 4200 $\to$ 4050 if needed (price exactly).

### Lane F gate

- Promote/continue iff fused $\ge 3\%$ with $\ge 6/8$ wins (mechanism win).
- If negative or 3W-5L style split: KILL shrunk-restart permanently (both
  endpoints — truth AND shrinkage — then failed).
- Exactly one follow-up allowed (try $k = 10$ or $0.95/0.05$) and only if
  the first run is directionally positive ($\ge 1\%$).

## Lane G — variance anatomy, then late-layer branching

### Hypothesis

Nobody has measured *where* MC variance is born by depth. If late layers
dominate final-mean variance, antithetic branching (each pilot particle
spawns pairs for the last $R$ layers only) buys large $B$-cuts at fraction
cost. If variance is born early/-uniformly, branching cannot help and the
lane dies in the anatomy step without building anything.

### Required sequence (anatomy before construction)

1. **Anatomy (offline, 8 MLPs, fixed seed):** functional ANOVA of final-mean
   MC error by layer — freeze layers $0..k$ at truth (baked
   `all_layer_means`-derived conditional replays are NOT targets at
   inference; this is offline attribution only) and re-sample the rest;
   report variance share of early/mid/late thirds.
2. **Build only if** late third owns $\ge 50\%$ of $B$: pilot $N=4200$ full
   trajectories, branch each penultimate particle into $b \in \{2, 4\}$
   antithetic children for the last $R \in \{2, 4\}$ layers; cost
   $= C_{\rm full}(4200) + b \times C_{\rm late}(4200, R)$ — must price
   $\le 9.85\%$ (reduce pilot to 3600 if needed and compare exactly).
3. Blend with analytic at re-optimized fixed $\alpha$ (single scalar from
   the 92-pool or prior $0.11$ — not per-MLP).

### Lane G gate

- Build iff anatomy shows late $\ge 50\%$.
- Keep iff fused $\ge 3\%$, $\ge 5/8$ wins with coherent mechanism.
- Expected value 3–5% fused; a funder for the stack, not an escape alone.

## Lane H — 92-MLP universal-prior program (methodology fix)

### Hypothesis

Phase 4.5 proved (down to 1-parameter pooled controls) that 8 MLPs cannot
support ANY fitting. The mini split has ~100 public MLPs; training
low-capacity universals on ~92 and validating LOO + frozen-8 converts
several dead families (depth-varying kernels, scale laws, kernel ensembles)
into statistically viable deterministic priors — all shipped as data files
at zero inference FLOP cost beyond application.

### Required program (offline-first; ≤8-panel only for validation)

1. Assemble the training pool: all public `mini`-split MLPs EXCLUDING the
   frozen eight; ground truth = baked `all_layer_means`/`final_means`
   (public data, legal).
2. Fit, in order (each $\le 10$ params, ridge-regularized, training-pool CV):
   a. depth-varying quadratic weights $\gamma_l$ / blend $\lambda_l$ in
      4 depth stages (4–8 params);
   b. scale law $s(W) = s_0 + \beta^\top \phi(W)$ with $\phi$ =
      spectral-norm / mean-fan-in-variance / gate-saturation summaries
      ($\le 4$ params);
   c. analytic kernel ensemble weights over {centered, threshold-aware,
      Cho-Saul-L0} ($\le 3$ params, simplex).
3. Ship winners as `.npz` constants loaded in `setup()`; inference applies
   them as fixed white-box formulas (still `fnp`-only ops).
4. Validation: training-pool CV gain AND frozen-8 gain must agree in sign;
   frozen-8 needs $\ge 6/8$ wins for any ship decision.

### Lane H gate

- Ship a prior iff training CV $\ge 3\%$ AND frozen-8 direction agrees with
  $\ge 5/8$ wins.
- Expected 2–4% transferable (not 30%, but real where 8-panel tuning
  overfits). Killed entirely if training CV itself <2%: the signal truly
  is not there.

## Lane I — compaction engineering (fund samples inside the floor)

### Hypothesis

~25% of terminal neurons are dead ($\approx 10^{-9}$ MSE already) yet dense
GEMMs price them fully. A cheap pilot (256 samples) freezes dead columns
with high confidence; prefix-bucket compaction skips them in the remaining
$\approx 4000$-sample GEMMs; FWHT covers Hadamard first layers;
diagonal-early/full-late covariance halves the $O(n^3)$ bill. Savings buy
$+600$–$800$ samples inside 10% → $-1$ to $-3\%$ fused. Engineering funds
statistics; never the reverse.

### Required order (only behind a statistical winner)

1. Implement behind a feature flag; verify bit-parity of predictions with
   flag on/off on 2 MLPs (pre-smoke).
2. Measure exact util delta and $N$ bought; convert to fused delta on the
   8-panel with the SAME statistical design (no confounding).
3. Residual-time budget: compaction lives in Python plumbing — hard ceiling
   $0.30$s local; kill any variant breaching it.

### Lane I gate

- Keep iff same-design fused improves $\ge 1\%$ with zero new failures and
  util $\le 9.85\%$.
- Never optimize a statistically inert design (explicitly deprioritized).

---

## Combination rules

- For every serious pair: pooled + per-MLP residual correlations, fixed
  global convex oracle, LOO-fixed vs all-eight weights; reject blends with
  oracle <10%; require deployed $\ge 5\%$ beyond strongest component.
- The champion stays an anchor/blend component, never the architecture
  dictator — except proven: anything correlating with its error at
  $|\rho| > 0.1$ must justify itself (A4 precedent).

## Mandatory experiment order

1. **P4.6-01 (Lane G anatomy):** layer-wise variance attribution (offline).
   Go/no-go for branching.
2. **P4.6-02 (Lane F):** shrunk restart $k=12$, $0.9/0.1$ fixed (one full
   8-eval). Kill or continue the family permanently.
3. **P4.6-03 (Lane E pre-gate):** TAP derivation + offline $\ge 5\%$
   layer-bias gate on the training pool. No 8-panel before it passes.
4. **P4.6-04 (Lane E deployable):** full 8-eval iff P4.6-03 passes.
5. **P4.6-05 (Lane H):** 92-pool universal priors, training CV then
   frozen-8 validation; ship as `.npz` only on dual agreement.
6. **P4.6-06 (Lane G build):** branching iff anatomy passed; exact scored
   comparison at matched util.
7. **P4.6-07 (portfolio):** fuse only $\ge 10\%$-oracle branches; full
   8-panel + salts + validation + package reproduction.
8. **P4.6-08:** Lane I compaction behind the winning design; re-price every
   step; never submit without a fresh clean-process reproduction.

## Explicitly deprioritized work

- More $\lambda$, $s_0$, global $\alpha$, $N$-near-4200 sweeps.
- Per-neuron / per-MLP fitted weights of any kind on $\le 8$ MLPs.
- Another axis-aligned Hadamard, rank-1 lattice, or unbiased-CV variant.
- Full K=3/K=4 tensors; flexible 8-MLP residual models; leaderboard
  hill-climbing; residual-time-external compute (disqualifiable —
  `docs/concepts/allowed-code.md`).
- Declaring victory for 8W-0L at <10% mean gain.

## Honest calibration (read before working)

- Lanes F+G+H+I plausibly stack to ~8–12%: an interim promotion, NOT e-8.
- Only Lane E has the headroom profile for 30%+. Sequence accordingly:
  cheap falsifications first (G-anatomy, F single-eval), TAP pre-gate
  before any 8-panel spend, 92-pool before further scalar claims.
- If Lane E fails its 5% pre-gate AND Lane F fails outright, declare this
  problem family scientifically closed at ~1.2e-07 and pivot to shipping
  robustness (sub-10% margin, residual-time headroom, salt stability)
  rather than further tuning. A closed hole documented with oracle ceilings
  is a result — see `phase4.5record_muse.md` §Synthesis.

## Required experiment record

Append every experiment to `phase4.5record_muse.md` (continued) using the
Phase 4 schema: ID / date / lane+family / hypothesis / mechanism /
candidate+SHA / control+SHA / frozen params / panel+salts / expected
failure mode / control score / candidate score / relative improvement /
raw MSE / util+multiplier / per-MLP paired changes / W-L-T / median+worst /
residual correlations / oracle ceiling+definition / LOO / refit /
max resid+wall time / failures / evidence label / interpretation /
decision / exact gate / next experiment. Record metrics from outputs, never
from memory.

## References

- `MEMORY.md` (control, contracts, gates, CLI cheatsheet)
- `PHASE4_RECORD.md` (Phase 4 oracles: carriers, endpoint ladder, mid-net
  CV, spectral/SVD, portfolios, Hermite orders, recursive/regime/Cho-Saul/
  antagonistic/Edgeworth/continuous-scale/P4-14 unified blend)
- `phase4.5record_muse.md` (P5 kills: late-jump 1.9%, unbiased-CV 0%,
  C2 $R^2\sim10^{-3}$, Stein 0%, global-CV 0%, P5-05 $-0.68\%$ 6W-2L
  submitted #329939, P5-06 boundary rejected)
- `run 3 plan.md` / `whest-starterkit/PHASE4_RESEARCH_PLAN.md` (lane
  definitions, gate arithmetic, combination rules)
- `docs/concepts/allowed-code.md` (fnp-only; residual is plumbing)
- `docs/concepts/scoring-model.md` (Score = MSE × max(0.1, C/B);
  covariance $O(n^3)$ = 51.7B / 2.35%, mean-prop 86M / 0.004%)
- `docs/how-to/ship-weights.md` (legal offline-compute path for Lane H)

(End of file)
