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
| Wave-C | 32 cells declared, 4-cell screening | 4 / 20 eval slots (4 units) | A-8b | `complete` (screening scope only) |
| ARCH_ABL | 2 train / 10 eval slots (2 units) | 2 / 10 | scope reduced in A-9 | A-9 complete |
| AERIAL | evaluation only (no training universe) | 1 geometry, 2 models + 2 digital | - | A-10 complete |
| POLAR | evaluation only (no training universe) | 50 conditions x 2 coders | - | A-11 complete |
| AERIAL_TRAIN | 2 train / 48 method-unit-SNR slots | 2 / 48 | A-12 | A-12 complete, **negative** (training diverged; no cell retrained) |
| AERIAL_TRAIN_TRUNC | 2 train / 48 slots (+1 pre-flight, writes nothing) | 2 / 48, pre-flight PASS at +20 dB | A-13 | A-13 complete (operating point unchanged) |
| K24_SEEDS | 6 train / 30 slots | 6 / 30 | A-14 | complete (seed range ≤ 0.004; metric at ceiling) |

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

**Measured evidence (2026-10-21, from the shipped checkpoint, not memory):**

```
results/WAVE_A/WAVE_A_caps_h256_task_awgn_k12_seed42__5c1f2f0e22e5ba5e/final_ckpt.pt
  metadata.hidden                      = 256        <- the label
  encoder.backbone.0.weight            = (128, 6)   <- the actual width
  max tensor width in model_state      = 128        <- identical to the control
```

**Status:** documented, **not fixed**, no retrain performed (fixing it now
would add a 4th scientific change after seeing results). Enforced in code:

- `run_wave_a.INVALID_ARMS` is written into
  `step_wave_a_summary.json["invalid_arms"]`;
- the dashboard renders the arm as `caps_h256 ⚠` with a footnote;
- `tests/test_wave_a_tooling.py::test_invalid_arm_claim` re-reads the
  checkpoint weights and fails if the declared invalidity stops being true.

Any claim about capacity must ignore this arm until it is re-run under a new,
dated amendment.

**Update 2026-10-26 (A-9):** the *root cause* is now fixed in code (constructors
resolve the width at build time instead of binding the config default at
import). The arm itself is still **invalid and was not retrained**; this entry
stays as the historical record, and the new capacity run lives in `ARCH_ABL`.

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

## A-8 — Wave-C: lambda-shaping across bandwidth (2026-10-21, user-approved)

**Question it answers.** Wave-A's λ-shaped arms reached 0.99 task success at
k = 12, but that number sits on the task ceiling and was only trained at one
bandwidth. Wave-C asks whether the effect survives at k = 18/24/30/54, where
the frozen task arm sits at 0.02–0.05 and the digital baseline rises from
0.06 to 0.94.

**Design (pre-registered before the first cell, recorded here first).**

| field | value |
|---|---|
| arms | `lam_ctl_light` (λ_u,λ_v ×10), `lam_ctl_heavy` (×100) |
| k | 18, 24, 30, 54 |
| channels | awgn, rayleigh |
| seeds | 42, 43 |
| steps | `config.TASK_TRAIN_STEPS` = 3000 (same as Wave-A k=12 arms) |
| cells / units / slots | **32 / 16 / 80** |
| eval | E = 1000 per unit-SNR, 5-SNR grid |
| root | `results/WAVE_C/` only |
| runner | `tools/run_wave_c.py` (smoke, verify, train, evaluate, finalize, status) |

**Cost, measured not guessed.** A 100-step smoke gave ≈1.3 s/step at both
k = 18 and k = 54; Wave-A's own train reports give a 0.667 s/step median and a
37 min median per 3000-step cell. Full grid ≈ **25 h**. A cut-down 8-cell
version (~7 h) was offered; the **full 32-cell grid was approved** by the user
on 2026-10-21, so the cheaper design was not used.

**Isolation guarantees.** The runner reuses Wave-A's training/evaluation
machinery through a patched cell universe and root, but it cannot write outside
`results/WAVE_C/`. Wave-A's declared universe stays at 46 cells / 18 units /
90 slots, and `run_wave_a.py verify` still passes. Seeds, step budget,
architecture, channel model, SNR grid and statistics are unchanged; only the
loss weighting and k are new.

**Known limitation, stated up front.** Success rate is expected to saturate at
k ≥ 24 (the task is easier when more symbols are available), so the informative
comparison at high k is against the **digital baseline** (0.94 at k = 54), not
against the frozen task arm. Secondary metrics (`control_effort_mean`,
`ttg_success_mean`, `final_distance_mean`) are recorded per cell and must be
reported alongside success rate.

---

## A-8b — Wave-C screening subset (2026-10-22, pre-registered before any of its cells trained)

**Why.** The full grid in A-8 needs ≈25 h of wall clock, which the user
explicitly cannot wait for. The grid was therefore reduced **before any of the
following four cells was trained** — this is pre-registration, not a post-hoc
choice.

| field | value |
|---|---|
| arm | `lam_ctl_heavy` only (the arm that was stable on both channels at k=12) |
| k | 18 and 54 (the two decisive extremes) |
| channels | awgn, rayleigh |
| seed | 42 |
| cells | **4** |
| wall clock | 4 parallel shards × 2 threads each on 8 logical cores |
| eval | E=1000 per unit-SNR, 5-SNR grid → 4 units / 20 slots |

**What each extreme answers.** k=18: the frozen task arm is 0.02–0.05 and the
digital baseline 0.06–0.17, so λ-shaping's effect is measurable without hitting
the digital ceiling. k=54: the digital baseline is 0.92–0.94, so the question is
whether a λ-shaped task-oriented model can *beat a saturated classical link*.

**Already-existing cells that are NOT part of this design.** Two
`lam_ctl_light` cells at k=18/awgn (seeds 42, 43) were trained under A-8 before
the reduction. They are kept, never deleted or overwritten, and may only be
reported as a labelled side observation — not as part of the screening
comparison.

**Limitations that must travel with any Wave-C claim.** One seed → no variance
claim; one arm → no dose-response claim; two k values → no trend claim. The
full 32-cell grid (A-8) stays declared and its cells remain idempotent, so the
rest can be added later without touching what is reported now.

---

## A-9 - Capacity ablation and the hidden-width constructor bug (2026-10-26)

**Question.** The frozen task arm at k=18 sits at ~0.02 success with an
oscillating loss. Is that a *capacity* artefact? The frozen MLP is
6-128-128-2k, so "no" is not obvious without an experiment.

**Root cause found while wiring the ablation.** Both constructors declared

```python
hidden_dims: tuple = config.HIDDEN_DIMS,
```

which Python evaluates **once, at import**. Patching `config.HIDDEN_DIMS`
before building a model (what `WaveCell(hidden=256)` does) therefore had no
effect: the layers stayed 128-wide while the metadata said 256. This is the
exact mechanism behind the invalid Wave-A `caps_h256` arm (A-6).

**Fix (code only, 2026-10-26).** `deepjscc_encoder.py`, `deepjscc_decoder.py`
and `reconstruction_jscc.py` now take `hidden_dims: Optional[tuple] = None`
and resolve `config.HIDDEN_DIMS` **inside** `__init__`. The frozen default is
still `(128, 128)`, so every existing checkpoint loads and every existing
model is byte-identical; the patch only makes an explicit width (or a runtime
config patch) actually take effect. Guarded by
`tests/test_arch_width.py` (14 checks).

**Pre-registered ablation design (fixed before the first cell trained):**

| field | value |
|---|---|
| root | `results/ARCH_ABL/` (nothing else written) |
| arms | `frozen_task` (lambda_u = lambda_v = 0) |
| width | 256 (both layers), verified from `encoder.backbone.0.weight` before training |
| k | 18 (the failing, non-saturated condition) |
| channels | awgn, rayleigh |
| seeds | 42 |
| steps | `config.TASK_TRAIN_STEPS` = 3000 (unchanged budget) |
| cells | 2 |
| eval | E=1000, 5-SNR grid, 2 units / 10 slots |

**Scope reduction, recorded (12:25 UTC 2026-10-26).** The pre-registered design
was 8 cells (2 arms x 2 channels x 2 seeds). A 256-wide cell costs about
3-4x the wall clock of a frozen 128-wide cell on this 4-core machine, so the
run was cut to the 2 cells that answer the headline question: frozen task
arm, width 256, k=18, awgn + rayleigh, seed 42. No cell had completed and
no metric existed when the decision was taken (the cancelled start markers
are kept under `results/ARCH_ABL/_cancelled/`). The lambda arm at 256 and
both seed-43 cells stay declared but unrun; the ablation is **not** a
2x2x2 design and must not be described as one.

**What it can and cannot say.** If 256-wide still fails at k=18, capacity is
not the explanation. If it rescues the arm, the frozen failure is an
architecture limitation, not a property of task-oriented JSCC. One width and
one k: this is a point check, not a scaling law. Frozen M1/M2 and Wave-A/B/C
artifacts are untouched.

---

## A-10 - 3GPP UMa-aerial link budget on the frozen system (2026-10-26)

**Why.** The frozen benchmark is a no-pathloss, no-shadowing Rayleigh link.
`project/channels/aerial.py` implements the 3GPP TR 36.777 / TR 38.901
UMa-aerial large-scale layer but was never wired into any reported number.

**Method (evaluation only; nothing is trained).** Under perfect-CSI ZF a
deterministic transmit gain g multiplies the signal but not the noise, so

```
SNR_eff = SNR_tx - PL          (dB)
```

exactly, with the fading statistics unchanged. The trained checkpoints are
therefore evaluated unchanged on an *effective-SNR* grid, and LoS/NLoS are
combined on that one curve:

```
SR(SNR_tx) = P_LoS * SR(SNR_tx - PL_LoS)
           + (1 - P_LoS) * SR(SNR_tx - PL_NLoS)
```

This is exact for i.i.d. episodes with an independent LoS draw - it is a
mixture of two evaluations of the same curve, not a new channel simulation.
The reported aerial penalty is `required_tx - required_eff`.

**Scope, stated up front.** Large-scale path loss and LoS probability only.
No Doppler, no altitude-dependent small-scale fading, no shadowing term, no
per-episode gain dispersion. `reference_pl_db=0` means "absolute loss", i.e.
the benchmark's SNR is defined at the receiver with unit path gain; any other
anchor is a labelled sensitivity, never a silent default change.

**Grid.** d2D = 500 m, h = 50 m, fc = 2 GHz (P_LoS = 0.8887, PL_mean =
95.63 dB), E=1000, effective-SNR grid -40..+25 dB in 5 dB steps (measured in
two passes over the same deterministic stream via `--reuse`), methods:
Wave-C `lam_ctl_heavy` at k=18 and k=54, plus the frozen convolutional digital
baseline at the same effective SNRs. Results: `results/AERIAL/`.

**Tooling bug found during the first run (fixed before the reported numbers).**
The transmitter-SNR mapping originally clipped interpolation beyond the
measured grid to 1.0, which made any method with a low ceiling look as if it
"reached" 0.90 at an arbitrary SNR. It now uses constant extrapolation, so a
method whose ceiling is 0.17 reports `None` rather than a fake threshold. The
contaminated first output was overwritten, not kept alongside the fixed one.

A second runner bug was found in the same pass: the digital comparator was
evaluated at B=2 for both bandwidths, but the frozen k=54 comparator is B=8
(`M2_B_BY_K`). At k=54 that made the frozen baseline look far worse than it
is (0.17 vs the published 0.72 at 0 dB). The runner now uses `M2_B_BY_K[k]`;
the corrected numbers are the only ones shipped.

---

## A-11 - Polar-coded digital baseline on the frozen M2 grid (2026-10-26)

**Question.** Every "digital" number in this project comes from one coding
choice (K=7, rate-1/2 convolutional + QPSK). Does the conclusion survive a
different modern code?

**Baseline.** `project/baselines/digital_polar.py`: 5G NR-style CRC-8 polar
code with SC decoding, behind the same quantize / pack / QPSK / channel
interface (4/4 own gates pass). It is deterministic and untrained, so there is
nothing to train and no seed variance to average.

**Design (pre-registered before the run).**

| field | value |
|---|---|
| k grid | 12, 18, 24, 30, 54 (M2) |
| B per k | {12:1, 18:2, 24:3, 30:4, 54:8} (M2_B_BY_K, unchanged) |
| SNR grid | 0, 5, 10, 15, 20 dB (M2) |
| channels | awgn, rayleigh |
| methods | polar (new) and conv (re-run on the **same** episodes, exact pairing) |
| episodes | 1000 (default); 5000 available via `--episodes=5000` |
| test stream | `config.TEST_SEED`, one stream |
| output | `results/POLAR/` |

**Why one stream and not three "seeds".** Both baselines are deterministic.
Evaluating the same deterministic pipeline three times and calling it seed
replication would be fake variance, so this run reports one stream with a
binomial (Wilson) 95% CI.

**Known limitation, stated with the numbers.** SC decoding without a CRC-aided
list is weaker than a production 5G receiver, so the polar column is a
*lower* bound on what polar can do, not a verdict on polar coding. Frozen
M1/M2 and all Wave artifacts are untouched.

---

## A-12 - Training under the 3GPP UMa-aerial dispersion (2026-10-26)

**Question.** A-10 evaluated Rayleigh-trained checkpoints on a 3GPP aerial
link. A reviewer can object that the models were never *trained* on that
channel. Does training under the 3GPP statistics change the lambda arm's
operating point - and, in particular, does it recover any of the ~99 dB
transmit-SNR penalty?

**Channel (added, opt-in; `channel="umaaerial"`).**

* one LoS/NLoS draw per episode with the TR 36.777 P_LoS of the geometry,
* one log-normal shadow term per episode, TR 38.901 UMa sigma (LoS 4 dB,
  NLoS 6 dB),
* perfect-CSI ZF on the true `h*g`; an episode whose offset is `-o` dB is
  exactly the frozen Rayleigh channel at `SNR - o` (unit-tested),
* the gain is **centred on the geometry mean path loss** (dB), so the frozen
  nominal SNR axis keeps its meaning and the experiment isolates dispersion
  rather than an arbitrary link budget. Transmit SNR = nominal + PL_mean,
* block fading over the episode. **No Doppler** and no time-correlated
  shadowing: the environment is a 2-D double integrator with no altitude or
  velocity in the state, so a Doppler term would require a different
  benchmark, not a channel patch.

**Design (pre-registered 2026-10-26T13:10Z, before the first cell).**

| field | value |
|---|---|
| arm | `lam_ctl_heavy` (the frozen task arm sits at ~0.02 success and cannot discriminate a channel effect) |
| k | 18, 54 |
| channel | umaaerial |
| seeds | 42 |
| width / steps / lambdas | frozen (128 / 3000 / 1.0, 0.1, 0.1, 2.0) |
| cells | 2 |
| eval | E=1000, nominal grid -15..+20 dB (8 points), one shared per-episode gain vector (paired) |
| comparators | Wave-C `lam_ctl_heavy` Rayleigh checkpoints (untouched), frozen digital K=7 with B = `M2_B_BY_K[k]` |
| output | `results/AERIAL_TRAIN/` |

**Why the comparison is exact, not approximate.** Under perfect-CSI ZF the
block gain only changes the effective SNR, so evaluating the pre-generated
paired realizations with the noise divided by `g` is the UMa-aerial
realization itself - no curve interpolation, no re-training of the
comparators.

**What is NOT changed.** Frozen M1/M2, Wave-A/B/C universes, results and
checkpoints; seeds; step budget; architecture; loss weights; environment;
controller; evaluation episodes/seed/grid; statistics. Only the channel seen
during these two trainings is new. Frozen artifacts stay read-only; the new
runner writes exclusively under `results/AERIAL_TRAIN/`.

**Known limits.** Absolute link budget still depends on `reference_pl_db`
(here the geometry mean, reported explicitly); shadowing is per-episode, not
time-correlated; SC/conv digital baselines are unchanged; nothing here claims
hardware feasibility.

**Result (2026-10-27, run as registered — no retraining, no re-selection).**
Both cells completed (no NaN losses) and both training runs **diverged**. They
reached the frozen λ-arm's terminal task loss (15.4 at step 105 for k = 18,
15.6 at step 160 for k = 54, vs 16.1 for every converged Wave-C cell) and then
left it: 14–17 % of steps have loss > 10³, 203–311 steps have non-finite
encoder gradients, ~75 % of steps exceed 10⁶. The resulting checkpoints are
dead networks — 0.023–0.029 (k = 18) and 0.013–0.025 (k = 54) at *every*
nominal SNR, and 0.027 / 0.014 flat on the frozen AWGN channel as a control.
Cause: the dispersion doubles the effective-SNR spread (9.37 vs 5.59 dB at the
same mean), putting ~4 % of episodes ≥20 dB below the frozen 0–20 dB training
curriculum and 55× (vs 14×) worst-case noise amplification into every batch;
the frozen recipe has no outlier control. Consequence: A-12 **cannot** be used
to compare "trained on 3GPP" with "trained on Rayleigh", and the A-10
evaluation-only result stands unchanged (its numbers reproduce on the same
gain vector). A fair test requires a new amendment (truncated training tail,
robust loss, or SNR-curriculum shift); none was applied. Details:
`fyp_md_files/RESULTS_AERIAL_TRAIN_2026-10-26.md`.

---

## A-13 - 3GPP dispersion training with the outage tail excluded (2026-10-27)

**Approved by:** the user, 2026-10-27, after reading the A-12 negative result
and the measured tail statistics. Recorded **before the first A-13 cell**; no
A-13 number existed when this text was written.

**Why.** A-12 diverged because the frozen recipe has no protection against
rare deep-fade episodes: the dispersion doubles the effective-SNR spread
(9.37 vs 5.59 dB) and ~4.2 % of episodes land 20 dB or more below the 0-20 dB
training curriculum, so nearly every 256-episode batch contains an episode
whose squared error is ~10³× the typical one. A-13 changes **only which
training episodes are used**, never the objective, the architecture, the
budget or the evaluation.

**One-line difference from A-12.** In the training path, an episode whose
large-scale gain offset exceeds **+20 dB** (i.e. a loss 20 dB worse than the
geometry mean, i.e. an outage episode; **3.78 %** of episodes) is
**resampled**. Everything else is identical.

**Correction, 2026-10-27, before the first A-13 cell (no A-13 metric in
existence).** The first draft of this entry said "+10 dB (2.2 %)". That number
came from a sign slip: 2.2 % is the *better-than-mean* tail. Measured on the
pinned geometry, a +10 dB cap resamples **10.1 %** of episodes and keeps only
**10.6 %** of the NLoS branch - it deletes essentially all NLoS training and
would make "trained on the 3GPP dispersion" false. The cap is therefore
**+20 dB**, which removes the genuine outage tail (3.78 %) while keeping
100 % of LoS and 66 % of NLoS episodes (kept offset std 5.91 dB, versus
9.37 dB untruncated and 5.59 dB for the Rayleigh channel).

| cap | episodes resampled | NLoS episodes kept | kept offset std |
|---|---|---|---|
| +15 dB | 7.34 % | 34.3 % | 4.77 dB |
| **+20 dB (primary)** | **3.78 %** | **66.1 %** | **5.91 dB** |
| +25 dB | 1.21 % | 89.2 % | 6.90 dB |

**Declared ladder and stopping rule (fixed in advance, to avoid
cap-shopping).** Pre-flight at +20 dB. If it fails, **one** retry at +15 dB.
If that also fails, A-13 stops as a negative result and **no** cell is
trained: a +10 dB cap is not used, because it trains on the LoS branch only
and no longer answers the question. At most two pre-flights (~8 min each).

**Design.**

| field | value |
|---|---|
| arm | `lam_ctl_heavy` (λ_d 1.0, λ_u 0.1, λ_v 0.1, λ_T 2.0) - unchanged |
| k / seed / cells | 18, 54 / 42 / 2 - unchanged |
| width, steps, lr, clip, batch | 128, 3000, 1e-3, 1.0, 256 episodes - unchanged |
| channel | `umaaerial` exactly as A-12 (TR 36.777 P_LoS, TR 38.901 4/6 dB shadowing, block fading, gain centred on PL_mean) |
| **training tail** | episodes with `offset_db > +20 dB` resampled (ladder: +15 dB fallback, then stop) |
| evaluation | **full, untruncated dispersion**; E = 1000, seed 10042, nominal grid −15…+20 dB, one shared per-episode gain vector (identical to A-12's) |
| comparators | unchanged: Wave-C `lam_ctl_heavy` Rayleigh checkpoints + digital with B = `M2_B_BY_K[k]` |
| output | `results/AERIAL_TRAIN_TRUNC/` (A-12 artifacts stay untouched) |

**Why this is the *fairer* comparison, not a weaker one.** The Rayleigh-trained
comparator never trains on outage episodes either; both arms now train on
non-outage episodes, and the aerial arm additionally sees the LoS/NLoS split
and the shadowing. Excluding the tail makes the two training conditions
comparable instead of giving the 3GPP arm a corrupted optimisation problem.

**Pre-registered pre-flight gate (300 steps, k = 18, seed 42).** The full run
is spent only if all three hold: (i) zero non-finite gradient steps,
(ii) no step after step 50 with loss > 10³, (iii) mean loss of the last 20
steps ≤ 50. If any fails, A-13 stops, the pre-flight is recorded as the
result, and no cell is trained. The pre-flight writes nothing.

**What is NOT changed.** Objective and loss weights, architecture, seeds, step
budget, environment, controller, optimiser, evaluation episodes/seed/grid,
statistics, the channel formula, and the evaluation dispersion. Frozen
M1/M2/Wave-A/B/C artifacts stay read-only. A-12's failed cells remain in the
record and are not retrained or replaced.

**Known limits, stated up front.** "3GPP-trained" here means *trained on the
3GPP large-scale dispersion with the > 20 dB outage tail excluded (3.78 % of
episodes), and evaluated on the full dispersion*; that qualifier must travel
with every number. Doppler and time-correlated shadowing remain out of scope.
One seed, two k values.

**Result (2026-10-27, run as registered).** Pre-flight passed on the primary
cap (0 non-finite gradients; worst loss after step 50 = 71; last-20 mean
17.65), so the +15 dB fallback was not used. Both cells converged to the
frozen λ-arm profile (loss 16.2 / 16.1; final encoder gradient 0.92 / 1.45;
3 outlier steps each; **0** non-finite gradients) - against A-12's
4320 / 210 final loss and 311 / 203 non-finite gradients. Evaluation on the
full dispersion, same paired gain vector: success 0.661/0.900/0.985/0.999 at
-15/-5/+10/+20 dB (k = 18) and 0.800/0.930/0.995/0.999 (k = 54), against
0.647/0.898/0.978/0.995 and 0.781/0.930/0.992/0.997 for the Rayleigh-trained
comparator. Required transmit SNR at 0.90: **90.6 dB** (k = 18, digital not
reachable) and **87.3 dB** (k = 54, digital 105.3 dB, i.e. **18.0 dB** better);
at 0.99: 108.4 and 103.7 dB. Difference from the Rayleigh-trained arm:
-0.27 / -2.22 dB (k = 18) and -0.50 / -1.09 dB (k = 54) at 0.90 / 0.99, all
inside sampling error (E = 1000, Wilson intervals overlap). Conclusion: the
operating point does **not** move materially, so A-10 stands; A-13 is not
evidence of a gain and must not be quoted as one. Details:
`fyp_md_files/RESULTS_AERIAL_TRAIN_TRUNC_2026-10-27.md`; figure
`fig_aerial_training.png`.

---

## A-14 - seed variance of the lambda arm at k = 24 (2026-10-27)

**Approved by:** the user, 2026-10-27, as the remaining reviewer point
("k = 24 multi-initialisation missing"). Written **before the first A-14 cell**;
no A-14 number existed when this text was committed.

**Question.** k = 24 is the bandwidth where the task-oriented and classical
links cross over, and every k = 24 number in the thesis is currently a single
seed. Is the crossover position, and the lambda arm's success rate there, an
artefact of one initialisation?

**Design (fixed before the run).**

| field | value |
|---|---|
| arm | `lam_ctl_heavy` (λ_d 1.0, λ_u 0.1, λ_v 0.1, λ_T 2.0) - the Wave-C tag, unchanged |
| k | 24 (the crossover condition) |
| seeds | 45, 46, 47 (the Wave-A A3 seed set, reused so the numbers are comparable) |
| channels | awgn, rayleigh |
| cells | 6 |
| width / steps / lr / clip / batch | 128 / 3000 / 1e-3 / 1.0 / 256 episodes - all frozen |
| evaluation | E = 1000, 5-SNR grid 0…20 dB → 6 units / 30 slots |
| output | `results/K24_SEEDS/` (own root; no other universe is touched) |
| runner | `tools/run_k24_seeds.py` |

**Why seeds 45–47.** Wave-A arm A3 already used {45, 46, 47} for the
*instability* arm at k = 12, so these are the thesis's declared extra seeds,
not a fresh choice.

**Statistics declared up front.** Per-seed success rates are reported
individually, then mean and range (min–max) across the three seeds; the range
is reported instead of a confidence interval, because n = 3. Nothing is pooled
across seeds to produce a tighter interval, and no seed will be dropped.

**What is NOT changed.** Architecture, width, loss weights, step budget,
optimiser, environment, controller, evaluation episodes/seed/grid, statistics
of the frozen protocol. Frozen M1/M2/Wave-A/B/C and the A-9…A-13 artifacts stay
read-only. This run adds cells to a **new** root; it does not extend or
re-denominate any published universe.

**Cost (measured, not estimated).** A-13's 3000-step cells took 76–81 min each
under 2-way parallelism, so 6 cells run as 2 shards × 3 cells ≈ **4 h** wall
clock. (The "~40 min/cell" figure in `GAP_ANALYSIS_RESPONSE.md` §4 was an
estimate made before A-12/A-13 measured it; the estimate is corrected there.)

**Limits.** n = 3 seeds; two channels; one width; E = 1000. This quantifies
seed sensitivity at one bandwidth; it is not a variance model for the whole
grid, and it does not touch the 32-cell Wave-C universe.

**Result (added after the run; the text above was not edited).** All 6 cells
trained to the frozen terminal regime (final loss 16.0–17.6, minimum
13.6–14.1, **0** non-finite gradient steps, 0 NaN batches) and all 30 eval
slots completed. Success rate at 0 dB: awgn 0.9970 / 1.0000 / 0.9960, rayleigh
0.9950 / 0.9950 / 0.9990; at 20 dB: awgn 0.998 / 0.999 / 0.998, rayleigh
0.999 / 0.998 / 1.000. The seed-to-seed **range is ≤ 0.004 at every SNR**,
the same size as the binomial sampling error at E = 1000 (Wilson half-width
≈ 0.003). Conclusion: the k = 24 result is **not** an artefact of one
initialisation, but the metric is at its ceiling, so the run rules out a large
initialisation dependence rather than measuring seed variance at a
non-saturated point. The frozen task arm at k = 24 (which is not saturated)
still has no multi-seed estimate. Details:
`fyp_md_files/RESULTS_K24_SEEDS_2026-10-27.md`.

---

## 1. What a reviewer can check independently

```powershell
python tools/run_step10_m1.py verify     # frozen M1 gates
python tools/run_step11_m2.py verify     # frozen M2 gates
python tools/run_wave_a.py verify        # wave ids, frozen-identity fingerprint
python tools/run_wave_b.py verify        # coverage contract + 70 σ=0 gates
python tools/run_wave_c.py verify        # screening scope + eval-slot contract
python tools/run_arch_ablation.py verify # capacity-ablation pre-flight (A-9)
python tools/run_aerial_training.py verify # 3GPP training-channel pre-flight (A-12)
python tools/run_all.py status           # 21 ok, 1 accepted, 0 pending

python -m tests.test_wave_a_tooling      # 60 checks
python -m tests.test_aerial_link         # 26 checks (3GPP formulas + mapping)
python -m tests.test_aerial_training     # 25 checks (3GPP training channel)
python -m tests.test_arch_width          # 16 checks (capacity plumbing)
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
