# Protocol Amendments & Documented Exceptions — Consolidated Log

**Project:** Task-oriented semantic communication for UAV navigation
**Log opened:** 2026-10-21
**Rule used throughout:** frozen artifacts are never edited, failed cells are
never retrained because their result was inconvenient, and every deviation
from the frozen Step-9C protocol is *additive, dated, and user-approved*.
Numbers in result JSONs are produced by re-running the frozen code path — no
number in this log was typed by hand from memory.

---

## 0. Accounting at a glance

| Block | Universe | Produced | Documented exceptions | Status |
|---|---|---|---|---|
| M1 (frozen) | 36 train / 198 eval | 36 / 198 | — | complete |
| M2 (frozen) | 60 train / 150 eval | 56 complete + 4 conditional-accept / 150 | A-1 | complete (accounted 60/60) |
| M3 / M4 | — | all | — | complete |
| Wave-A | 46 train / 90 eval slots (18 units) | 44 + 2 failed / **85** | A-3 | `incomplete` **by design** |
| Wave-B | 375 cells, 15 units | **350** | A-5 | `complete` |

"Accounted" always means *evaluated or explicitly declared*. Nothing is
imputed, nothing is dropped silently, and no failed cell is replaced.

---

## A-1 — M2 conditional accept (amendment 2026-10-19, user-approved)

**Scope:** frozen M2 training, `task` method cells only.

**What happened.** The fresh full retrain reproduced the *same* failure mode
as the archived run: the final-step `grads_finite` probe returned `false`
while `nan_batches == 0`, the final task losses stayed finite (5166–5229,
inside the completed-cell range 118–5544), all 3000 steps executed, and a
finite checkpoint was saved. A deterministic retrain reproduces the identical
probe failure (frozen 9C policy: determinism by seed + config).

**Cross-run evidence that this is numerical marginality, not a config bug.**
The archived run failed `k54_seed42`, which **passed** in this run; this run
failed `k18_seed43`, `k18_seed44`, `rayleigh_k54_seed43`, which **passed** in
the archived run. A defect reproducible in one direction only is a marginal
floating-point probe, not a broken protocol.

**Declared set** (`CONDITIONAL_ACCEPT` in `tools/run_step11_m2.py`, identity
pinned by config hash):

| run_id | hash | on disk after this run |
|---|---|---|
| `M2_task_awgn_k30_seed43` | `d270bfa0b913b678` | incomplete |
| `M2_task_awgn_k54_seed42` | `59f19c2c451a8106` | **complete** (passed in this run) |
| `M2_task_awgn_k18_seed43` | `347dd85341970c91` | incomplete |
| `M2_task_awgn_k18_seed44` | `c533b5324f220cf9` | incomplete |
| `M2_task_rayleigh_k54_seed43` | `c69033a6561fc637` | incomplete |

**Terms:** evaluate-only. `train_report.json` keeps `exit_status:
"incomplete"` — it is **never edited** to say `complete`. The trainer skips
these cells with an explicit log line instead of retraining them.
Checkpoints are used for evaluation because they are finite and in-range;
the probe result is reported as-is.

**Dashboard wording:** `M2 training 56/60 (failed 4 · conditional-accept 4)`
— the 4 on-disk failures are the ones declared above, so 60/60 is accounted.

---

## A-2 — Wave-A is a non-frozen diagnostic ablation (2026-10-12)

Wave-A is *not* a reproduction of frozen M1/M2. It uses
`E = 1000` episodes per unit-SNR slot instead of the frozen 5000, so its
numbers are **not 1:1 comparable** with frozen M1/M2 tables and are labelled
that way in `step_wave_a_summary.json` (`protocol_note`) and on the dashboard.

Frozen seeds, budget, architecture, loss weights, channel model, SNR grid and
statistics are unchanged; only the new arms are additions.

---

## A-3 — Two Wave-A training failures (2026-10-20, user-approved: no retrain)

| cell directory | report | final task loss |
|---|---|---|
| `WAVE_A_long_12k_task_awgn_k12_seed42__7a6a45fdc770f9cd` | `nan_batches=0`, `grads_finite=false`, `exit_status=incomplete` | 5203.54 |
| `WAVE_A_seedvar_frozen_task_awgn_k12_seed45__20f7787a4aa60dca` | `nan_batches=0`, `grads_finite=false`, `exit_status=incomplete` | 5457.60 |

**Decision:** **not retrained.** Retraining only the two cells that failed —
after seeing that other task cells diverge — would be survivorship bias and
would make the reported arm comparison invalid. They stay failures, they stay
in the completeness accounting, and their directories are evidence-frozen.

**Consequence, stated openly:**
`k12_awgn_seed42` loses its `long_12k` arm and `k12_awgn_seed45` has no arm at
all. Therefore the Wave-A eval ceiling is **85/90 slots, 17/18 units**, not
90/90. The finalizer reports `status: incomplete` **by design**; a missing arm
is never imputed or filled from another seed.

**Evidence:** SHA256SUMS archive at
`C:\wa_evidence\WAVE_A_2026-10-20T1645Z\` (8 files, re-verified after the eval
run: all match). Pre-eval summary snapshot:
`C:\wa_evidence\WAVE_A_PRE_EVAL_2026-10-21\`.

**Tooling corollary (2026-10-21).** `run_wave_a.py train` now exits **4**
("completed with only documented failures") instead of 1 when *every* failure
in the queue is in `DOCUMENTED_FAILURES`; any undeclared failure still exits
1. `run_all.py` records exit 4 as stage status `accepted` and **does not
retry** it. Rationale: a retry cannot change a deterministic frozen outcome,
and an orchestrator that retries a 12-hour deterministic queue until it looks
green is exactly the failure mode this log exists to prevent.

---

## A-4 — Wave-B is a non-frozen, evaluation-only CSI ablation (2026-10-13)

Wave-B re-uses frozen M2 Rayleigh checkpoints and the frozen DigitalBaseline
**verbatim** (no training, no new models). It adds a CSI-estimation-error
sweep: `h_hat = h + eps`, `eps ~ CN(0, sigma^2)`, receiver equalizes and
scales LLR noise using `h_hat` only, with `eps` **paired across methods** so
the three methods see the same channel error on the same episode.

At `sigma = 0` the path must reproduce frozen M2 **bit-exactly**; this is
enforced as a hard gate on every such cell (70/70 passed).

---

## A-5 — Wave-B scope exclusion of unit k54/seed43 (2026-10-21, user-approved)

**Problem found by the gate.** `wave_b verify` refused to start because
`M2_task_rayleigh_k54_seed43` (A-1) is a documented `grads_finite=false`
cell, so one of the three arms for unit `k=54, seed=43` is not usable.

**Options considered.**

1. Retrain the M2 cell → forbidden: it is a documented failure and retraining
   it after seeing the result is exactly the bias A-3 refuses.
2. Silently drop the cell from the denominator → rejected: dishonest counting.
3. **Chosen:** evaluate Wave-B on 14 of 15 units and **declare** the excluded
   unit in machine-readable form.

**Declared in code** (`tools/run_wave_b.py`):
`EXCLUDED_UNITS = {(54, 43): "M2_task_rayleigh_k54_seed43"}`, with
`UNITS_COVERED = 14/15`, `EXPECTED_CELLS = 350` (of 375 if the unit were
complete), `EXPECTED_GATES = 70`. `cells()` filters the unit out; the
finalizer writes:

```json
"coverage": {
  "units_total": 15, "units_covered": 14,
  "cells_total_if_full": 375, "cells_expected": 350,
  "excluded_units": [{"k": 54, "seed": 43,
    "run_id": "M2_task_rayleigh_k54_seed43",
    "reason": "frozen M2 conditional-accept (documented failure); not retrained and not evaluated"}]
}
```

**Result:** 350/350 cells produced, **70/70** `sigma=0` gates PASS,
`status: complete`.

**Interpretation limit that must travel with the result:** the k=54
comparison rests on seeds 42 and 44 only, while k=12/18/24/30 rest on seeds
42/43/44. Any k=54-vs-other-k claim is therefore made on 2 seeds.

---

## A-6 — Residual risk: `caps_h256` is not actually 256-wide (open, not fixed)

`WAVE_A_caps_h256` was intended to test a 256-hidden-unit capacity arm. The
patch that sets `hidden=256` is applied **after** the encoder's default width
is bound at import time, so the trained arm is effectively **128-wide** and is
not a valid capacity ablation.

**Status:** documented, **not fixed**, no retrain performed (fixing it now
would add a 4th scientific change after seeing results). Any claim about
capacity must ignore this arm until it is re-run under a new, dated
amendment.

---

## A-7 — Tooling fixes applied 2026-10-21 (no protocol impact)

These change *how results are aggregated or loaded*, never the frozen
protocol. Each was verified to leave existing artifacts byte-identical.

| Fix | Bug | Verification |
|---|---|---|
| `run_wave_a.py evaluate` | every arm was evaluated with the **last** arm's checkpoint (`models` list built without calling `load_ckpt()`), crashing on the first unit | regression test `test_evaluate_loads_each_arm_checkpoint` (buggy code fails, fixed code passes) |
| `run_wave_a.py finalize` | `eval_units_complete` counted **slots** (reported `85/18`) | now `eval_slots_complete=85/90` and `eval_units_complete=17/18` are separate fields |
| `build_dashboard.py` | `_wave_table` crashed on non-empty pooled data (4-tuple keys unpacked as 2) | regression test `Wave-A table renders a non-empty pooled dict` |
| `build_dashboard.py` | "failed 4" hidden approved M2 exceptions; Wave-B denominator showed 375 | cards now `failed 4 · conditional-accept 4`, `0..350 (14/15 units · 1 excluded)` |
| `run_all.py` + `launch_all.bat` | launcher relaunched forever after a deterministic failure | no-progress detection → `HALT.json` + exit 3; documented failures → `accepted` (exit 4), never retried |

**Evidence of non-interference:** `step_wave_a_summary.json` and
`step_wave_b_summary.json` hashes before/after the marker refresh are
identical (`18803AC9E32C7797…`, `165E67AA8FC01B73…`).

---

## 1. What a reviewer can check independently

```powershell
python tools/run_step10_m1.py verify     # frozen M1 gates
python tools/run_step11_m2.py verify     # frozen M2 gates
python tools/run_wave_a.py verify        # wave ids, frozen-identity fingerprint
python tools/run_wave_b.py verify        # coverage contract + 70 σ=0 gates
python tools/run_all.py status           # 21 ok, 1 accepted, 0 pending

python -m tests.test_wave_a_tooling      # 60 checks
```

Hash check for the frozen evidence:

```powershell
Get-FileHash C:\wa_evidence\WAVE_A_2026-10-20T1645Z\* -Algorithm SHA256
# compare with the shipped SHA256SUMS.txt
```

## 2. Standing rule for future work

Any future change that would alter a frozen value, a seed, a step budget, a
loss weight, the channel model, the evaluation seeds, the SNR grid or the
statistics requires **a new dated entry in this file, approved by the user,
before the run** — never a silent edit, and never a retrain justified by a
result being ugly.
