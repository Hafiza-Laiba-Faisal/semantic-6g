# M2 Protocol Exceptions — 5 Conditionally-Accepted Training Cells

**Decision: CONDITIONAL ACCEPT (amended)** — the original 2-cell decision was
user-approved 2026-10-06 (full trail preserved verbatim in
`results/_archive/2026-10-16T232046Z/M2/PROTOCOL_EXCEPTIONS.md`; the same
mechanism — an explicit code allowlist, never editing any `train_report.json`
or marker — was used).

**Amendment 2026-10-19: user-approved (this session).** The fresh full-retrain
run (`fresh` mode launched 2026-10-16T232046Z) reproduced exactly the same
failure mode on 3 MORE task cells. The user ("agar losses kam/normal hain to
chor do aur aage ka process start karo") ratified extending the conditional
accept to all 5 cells, with skip-retrain added to avoid the known-futile
deterministic retrain cost (~2.5 h per relaunch).

## Cell inventory (allowlist `CONDITIONAL_ACCEPT` in `tools/run_step11_m2.py`)

| # | Cell | Config hash | Hiistory |
|---|---|---|---|
| 1 | `M2_task_awgn_k30_seed43` | `d270bfa0b913b678` | failed in archived run AND in this run (deterministic) |
| 2 | `M2_task_awgn_k54_seed42` | `59f19c2c451a8106` | failed in archived run; **passed in this run** (kept in allowlist; harmless) |
| 3 | `M2_task_awgn_k18_seed43` | `347dd85341970c91` | new in this run — amendment 2026-10-19 |
| 4 | `M2_task_awgn_k18_seed44` | `c533b5324f220cf9` | new in this run — amendment 2026-10-19 |
| 5 | `M2_task_rayleigh_k54_seed43` | `c69033a6561fc637` | new in this run — amendment 2026-10-19 (passed in the archived run) |

## Failure evidence (identical mode, all cells)

- Frozen 9C: task training, 3000 steps × 256 episodes × T=100, Adam 1e-3,
  grad clip 1.0, train SNR ~ U(0,20) dB. Retrain is deterministic by
  seed+config; every attempt reproduces the same final-step probe failure.
- `nan_batches = 0` (the NaN-skip guard never fired; loss finite per batch).
- Final training loss finite and NOT an outlier: 5166–5229 (fails) vs
  118–5544 across completed task cells.
- Saved `final_ckpt.pt` weights are fully finite.
- The completion gate additionally requires **final-step** encoder/decoder
  gradient norms to be finite and strictly positive (`grads_finite` +
  `g>0`); all 5 failed exactly that probe.
- `train_report.json` files are NEVER edited — `exit_status: "incomplete"`
  preserved on all 5.

## Cross-run nondeterminism evidence (decision basis)

Numerically marginal probe, not a data/config defect: cells that failed in
the archived run passed after this run's retrain, and cells that passed in
the archived run failed here, with identical seed/config (same config hash).
This supports treating the final-step gradient probe as an unreliable binary
health signal for these borderline configs, while the checkpoint itself is
healthy — consistent with the 2026-10-06 analysis of the same phenomenon.

## What is changed, and where (all visible in code diff)

1. `tools/run_step11_m2.py` — `CONDITIONAL_ACCEPT` extended 2 → 5 identities.
2. `tools/run_step11_m2.py` — training queue SKIPS these ids instead of
   futile deterministic retrain (marked in the run log; avoids ~2.5 h/frame).
3. `tools/run_step11_m2.py` — `validate` counts `done + conditional-accept`
   for the `60-training-runs-complete` check (detail string records both).
4. `tools/run_wave_a.py` — governance gate `frozen_M2_artifacts_present`
   counts present allowlisted dirs alongside `DONE.json` markers.
5. `train_report.json` / `DONE.json` / `FAILED.json` markers: **untouched**
   (no data falsification; acceptance lives only in code + this document).

## Impact on the M2 matrix

- Training rows: 60/60 recorded; 56 `complete` + 4 flagged `incomplete`
  (`M2_task_awgn_k18_seed43`, `M2_task_awgn_k18_seed44`,
  `M2_task_awgn_k30_seed43`, `M2_task_rayleigh_k54_seed43`). Affected
  (k, channel, method) triples therefore have 2-seed pooled means instead
  of 3-seed: awgn × k18 (task) and awgn × k30, rayleigh × k54 (task).
  Reconstruction and digital arms throughout are unaffected.
- Evaluation target remains **150/150 paired units** (allowlisted checkpoints
  participate in eval only).
- Wave-A governance gate passes via the amended counting (documented here).

## Status caveats recorded for the thesis appendix

- m2_train stage exits 0 (all cells skip/complete); m2_validate records
  `done + conditional = 60`; m2_finalize writes the descriptive summary.
- Supervisor ratification of the 2026-10-06 + 2026-10-19 amendments to be
  recorded here when obtained; the code allowlist makes every acceptance
  auditable in the diff.
