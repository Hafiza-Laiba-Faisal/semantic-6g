# M2 Protocol Exceptions — 2 Conditionally-Accepted Training Cells (2026-10-06)

**Decision (UPDATED later the same day): CONDITIONAL ACCEPT** — user-approved
2026-10-06; supervisor ratification pending. The 2 cells below are accepted
**for evaluation only** via an explicit, code-documented allowlist
(`CONDITIONAL_ACCEPT` in `tools/run_step11_m2.py`). Their
`train_report.json` files were NEVER edited (`exit_status: "incomplete"`
preserved) — the acceptance is visible in the diff itself. M2 target state:
**60/60 training rows (2 flagged incomplete), 150/150 paired evaluation
units**.

**Earlier the same day the decision was EXCLUDE + DOCUMENT (120/150);** it
was superseded when the user chose the conditional-accept route. Original
text: "M2 final evaluable state: 58/60 training runs, 120/150 paired
evaluation units. No data was fabricated; the evaluation integrity gate
was never bypassed."

## The two excluded cells

| Cell | Config hash | Failure |
|---|---|---|
| `M2_task_awgn_k30_seed43` | `d270bfa0b913b678` | `grads_finite=False` at final step |
| `M2_task_awgn_k54_seed42` | `59f19c2c451a8106` | `grads_finite=False` at final step |

## Failure evidence (both cells, twice each — deterministic)

- Frozen 9C protocol: task training, 3000 steps × 256 episodes × T=100,
  Adam 1e-3, grad clip 1.0, train SNR ~ U(0,20) dB. Retrain is deterministic
  by seed+config, so the second attempt reproduced the identical outcome.
- `nan_batches = 0` (loss finite on every batch; the NaN-skip guard never
  fired)
- Final training loss finite: 2524.58 (k30) / 5397.30 (k54)
- Saved `final_ckpt.pt` weights are fully finite (verified via the
  evaluator's 11-checkpoint integrity checks except the report-status one)
- The executor's completion gate additionally requires the **final-step**
  encoder/decoder gradient norms to be finite and strictly positive
  (`grads_finite`); these runs failed exactly that probe (exploding-then-
  clipped or zero final-step gradients)
- `exit_status: "incomplete"` in both `train_report.json` files → the
  evaluation integrity gate ("report-complete" check) refuses any lane
  containing them

## What was tried and rejected

- A second identical retrain per cell (frozen 9C policy) — identical
  deterministic failure; a third retry is futile by construction.
- An interim manual `DONE.json` marker placed on these dirs by an external
  assistant (without supervisor approval) was **removed** on 2026-10-06:
  it violated the frozen protocol and, being unaccompanied by any
  `train_report.json` change (which would be data falsification), could
  never have unlocked the integrity gate anyway. Marker state is now
  protocol-clean: `FAILED.json` + `RUNNING.json` + intact artifacts.

## Impact on the M2 matrix

- Training: 58/60 neural runs DONE; k30 and k54 AWGN task method have
  2-seed means instead of 3-seed (reconstruction and digital unaffected).
- Evaluation: 120/150 paired units complete. The 30 missing units are
  exactly the AWGN × {k30, k54} × 3 seeds × 5 SNRs task-method units that
  require the excluded checkpoints.
- `validate` status: 3 expected failures (`60-training-runs-complete`,
  `150-eval-units-complete`, `150-pooled-eval-conditions`); all other
  frozen-matrix checks PASS (matrix definition, seeds, M1 intactness,
  3-seeds-per-pooled-cell, methods-in-every-unit).

## Options that were considered

1. **EXCLUDE + DOCUMENT** (chosen first, later superseded): 120/150, honest
   documented exception.
2. **CONDITIONAL ACCEPT** (chosen finally): supervisor-approved allowlist in
   the evaluation gate for exactly these 2 runs — 150/150, with the
   acceptance visible in code + this document. **Chosen by the user on
   2026-10-06; supervisor ratification to be recorded here.**
3. Protocol amendment to the completion gate (e.g., trailing-window grad
   health): future-work item; requires supervisor approval and re-runs.

## Files

- `results/M2/M2_task_awgn_k30_seed43__d270bfa0b913b678/` (FAILED.json,
  final_ckpt.pt, train_report.json preserved)
- `results/M2/M2_task_awgn_k54_seed42__59f19c2c451a8106/` (same)
- This document records the decision trail for the thesis appendix and the
  supervisor.
