# Phase 6 conversion fixes — 6 September 2026

## Change

Candidate: `candidates/estimator_p6_k3_twofactor_terminal.py`.
Parent: `candidates/estimator_p6_k3_6_10_13.py`, SHA-256 `7ae6340449897468` (prefix; full hash in the parity receipt).
Pruning remains at layers `[6,10,13]`, retention `0.62`. No fitted coefficients, extra full-size networks, or reference tables were added.

1. Store each symmetric third-order atom as `Sym(U,U,V)`. Both source birth blocks admit this representation: the diagonal-slice block has two identity factors; the path block has `C[:,j] = w21[j] A[:,j]`. Absorb this scale into the remaining factor. Transport two matrices, and compute the repeated slice with `(2*(U*V)@U.T + (U*U)@V.T)/3`. This preserves the underlying tensor and norm-pruning criterion in exact arithmetic.
2. Evaluate only the final-layer mean. Compute preactivation marginal variance and third/fourth cumulants, then the mean correction. Do not construct final mixed slices, covariance, or unused future factors.

Operational cleanup: use metered `fnp.argsort` for numerical ranking, and cache the repeated Gaussian coefficients within each layer. Python continues to handle only indices and state bookkeeping. Submission code does not modify the meter or relax any caps.

## Verification and an important timing discrepancy

The float64 old/new algebra test covers widths/depths `(8,1),(16,2),(32,8),(32,16)` and passes at `atol=rtol=1e-9`; maximum difference is `1.55e-14`. A stricter initial float32 check failed on the deep, narrow fixture (maximum `4.34e-5`), so numerical fidelity was separately measured on the actual eight-network panel rather than assuming bitwise identity.

Captured production-size predictions before coefficient caching:

- Old mean raw MSE: `7.320315563167058e-8`.
- New mean raw MSE: `7.320334652408828e-8`.
- Maximum prediction difference across all layers/networks: `3.8146973e-6`.
- Final prediction RMS difference: `6.2765145e-7`.
- Utilization falls from approximately `0.9752` to `0.63737`.

**These first full-panel runs failed the local residual-time gate.** The original submitted candidate also fails this local replay: maximum residual `2.2540 s`; the first optimized version reaches `1.4215 s`. A direct subprocess check confirms that the issue is not only the prediction-capture wrapper. Caching reduces the first-network direct residual to `0.98435 s`, still above `0.4 s` locally. Therefore the calculated accuracy-times-utilization numbers are diagnostic projections, not passing local scores.

The user's original candidate nevertheless has a successful reported leaderboard score of `6.65e-8`. The local full-backend/official execution discrepancy remains unresolved. An authorized experimental submission tests the optimized implementation under official execution; acceptance or packaging alone is not grading success.

The old `p6_eval_harness.py` also incorrectly recalculates failed-run scores using the utilization discount. Its official score field correctly reflects failure. This audit uses the official failure status and independently labels captured-prediction accuracy as diagnostic. The existing harness was not edited.

## Artifacts

- Reproducible tiny checks: `whest-starterkit/scripts/verify_p6_conversion_fix.py`.
- Audit receipts: `whest-starterkit/research/phase6_conversion_fix/`.
- Earlier failed panel receipts and captured predictions: `whest-starterkit/research/phase6/results/P6-FIX-*` and `predictions/P6-FIX-*`.
- Only the candidate and its manifest are packaged. No dataset, diagnostic prediction, credential, or reference implementation is submitted.

## Final diagnostic and submission

Final source SHA-256: `8498085d5b90646c6d65ce23087b2e9331980a77cb44bfa6d762341f68dd05ac`.

Final eight-network diagnostic: raw MSE `7.320334471927481e-8`, billed FLOPs `1,401,501,432,002` per network, utilization `0.6373290634664954`. Accuracy times the compute multiplier is `4.6654619132550436e-8`, about 34.6% below the old candidate's reported local score. This is **not a passing local score**: the diagnostic explicitly disabled the residual gate, and maximum measured residual was `1.564788 s`. Maximum prediction wall time was `17.270526 s`. No such relaxation is included in the submitted code.

The package contains exactly `estimator.py` and `manifest.json`; archive source hash matches the verified candidate. Both contract validation and package validation passed. Official submission receipt is recorded separately in `whest-starterkit/research/phase6_conversion_fix/submission_receipt.json`.

**Submitted as [AIcrowd #330018](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/330018).** The returned receipt says `submitted`, one job enqueued, and no official score yet. The 60-second grading watch ended without a final grade. Do not promote this candidate as a verified leaderboard improvement until that grade is available.
