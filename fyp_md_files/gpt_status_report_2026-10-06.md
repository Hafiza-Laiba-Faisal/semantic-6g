# Project Status Report — 2026-10-06, 12:00 PKT

**Project:** Task-Oriented DeepJSCC for UAV Navigation under Noisy and Fading Wireless Channels (FYP, simulation-only, Python + PyTorch)

**Purpose of this report:** a complete, factual handoff for an external assistant (GPT). Every number below is measured and stored in JSON artifacts on disk — nothing is projected or invented. Use these facts as fixed ground truth; do not extrapolate or invent numbers beyond them.

---

## 1. One-paragraph context

We compare three communication approaches for a UAV goal-reaching task with a
6-D state and a shared PD controller: (1) Task-Oriented DeepJSCC (trained on a
differentiable closed-loop navigation loss), (2) Reconstruction-Oriented
DeepJSCC (normalized-state MSE), and (3) a conventional digital baseline
(scalar quantizer + K=7 R_c=1/2 convolutional code + soft Viterbi + QPSK),
under AWGN and slow Rayleigh fading with perfect CSI and ZF equalization.
Bandwidth is strictly matched: every method uses k channel uses per state
(rho = k/6 for the neural methods). All experiments run under a frozen
protocol (Step-9C). M5 is deferred + disabled per that protocol (needs a
supervisor-approved experiment definition; nothing exists for it).

## 2. Completion overview

| Experiment | Definition | Status |
|---|---|---|
| **M1** | task vs reconstruction, k ∈ {1,2,3}, 11 SNRs {0,2,…,20} dB | ✅ **COMPLETE** (all runs + 198/198 eval units + summary) |
| **M3** | structural digital-infeasibility display at k ∈ {1,2,3} | ✅ **COMPLETE** |
| **M4** | SECONDARY digital reference, B=8 at k=54 | ✅ **COMPLETE** (10/10 conditions) |
| **M2** | primary 3-way matched-budget comparison, k ∈ {12,18,24,30,54} | 🔄 **EXECUTING (~57% training, k12 fully done incl. eval)** |
| **M5** | ablation set | ⏸️ deferred + disabled (protocol) |

## 3. M1 — COMPLETE (the key measured result so far)

- **Training:** 36/36 neural runs (task + reconstruction × k{1,2,3} ×
  AWGN/Rayleigh × seeds {42,43,44}), 0 failed, 0 NaN.
- **Evaluation:** 198/198 paired units; both methods share bitwise-identical
  episodes/noise/fading (paired design), 5000 test episodes per unit, test
  seed 10042.
- **Headline execution fact:** in all 198 paired comparisons, the
  reconstruction-oriented variant beat the task-oriented variant in the
  low-k regime (n10 > n01 in every cell). This is a measured fact at
  k ∈ {1,2,3} — **no general superiority claim** is made beyond the data,
  and M2's larger budgets are exactly where this may change.
- **Statistics:** Wilson 95% CIs; paired McNemar + paired-t. A factor-of-2
  defect in the McNemar p-value (one-sided instead of two-sided χ²(1)
  survival) was found in a numerical audit and corrected (commit `88e107a`,
  Step-10E). χ² values were unaffected; corrected p = 2 × stored p exactly;
  all corrected p-values remain far below 0.05, so every significance
  conclusion is unchanged. The 198 historical eval JSONs store the stale
  (halved) p-values; they were intentionally not modified (historical
  artifacts) — regeneration is a tracked action item, pending supervisor
  approval.

## 4. M3 — COMPLETE (structural, not a performance result)

The matched-budget digital baseline **cannot exist** at k ∈ {1,2,3}: the
smallest transmissible packet needs 6B + 6 = 12 channel uses at B = 1
(6 source scalars × 1 bit + 6 tail bits), so deficits at k = 1/2/3 are
11/10/9 channel uses. This is an architectural feasibility boundary.
Artifact: `results/M3/m3_report.json`.

## 5. M4 — COMPLETE (SECONDARY digital reference, B = 8 at k = 54)

Exact-fit (6·8 + 6 = 54). Evaluation-only, one cell per channel, M2's SNR
grid {0,5,10,15,20} dB, 5000 episodes, test seed 10042. Full table:

| Channel | SNR (dB) | Success rate | Wilson 95% CI | Final dist. |
|---|---|---|---|---|
| AWGN | 0 | 0.9736 | [0.9688, 0.9777] | 0.37 |
| AWGN | 5 | 1.0000 | [0.9992, 1.0000] | 0.03 |
| AWGN | 10 | 1.0000 | [0.9992, 1.0000] | 0.03 |
| AWGN | 15 | 1.0000 | [0.9992, 1.0000] | 0.03 |
| AWGN | 20 | 1.0000 | [0.9992, 1.0000] | 0.03 |
| Rayleigh | 0 | 0.7178 | [0.7052, 0.7301] | 1.08 |
| Rayleigh | 5 | 0.9908 | [0.9878, 0.9931] | 0.29 |
| Rayleigh | 10 | 0.9984 | [0.9968, 0.9992] | 0.13 |
| Rayleigh | 15 | 0.9998 | [0.9989, 1.0000] | 0.07 |
| Rayleigh | 20 | 1.0000 | [0.9992, 1.0000] | 0.04 |

Artifact: `results/M4/step13_m4_summary.json`. Always label this SECONDARY
(it uses 54 channel uses at k=54 neural cells — a different operating point,
frozen in the matrix validator).

## 6. M2 — EXECUTING (frozen matrix + exact current state)

**Frozen matrix:** task + reconstruction × k ∈ {12,18,24,30,54} ×
{AWGN, Rayleigh} × seeds {42,43,44} = **60 neural training runs**; digital
baseline evaluation-only at matched budgets B = {12:1, 18:2, 24:3, 30:4,
54:8} (exact fit 6B + 6 = k for all five); **150 paired evaluation units**
= 30 (k, channel, seed) unit dirs × 5 SNRs {0,5,10,15,20} dB × 3 methods
per unit × 5000 paired episodes, test seed 10042.

**Progress as of 12:00 PKT, 2026-10-06:**

- **k12: FULLY COMPLETE** — 12/12 training runs + 30/30 evaluation files.
- **k30: 11/12 training done** · **k54: 11/12 training done**.
- **k18 + k24 lanes: launched ~11:50 today** (24 runs = 40% of M2's
  training), recon runs in progress; per-measured rates below, training
  should finish ~15:00–15:30, then the self-driving watcher automatically
  launches their evaluations and the remaining Rayleigh evals.
- **34/60 neural training runs DONE (57%)** · **30/150 evaluation files done
  (20%)** · achievable today without any decision: **120/150 files** (the
  remaining 30 are blocked only by the two failed cells below).

**Measured training wall-times (3-seed means, under 3-lane CPU contention):**

| Run type | k=12 | k=30 | k=54 |
|---|---|---|---|
| Reconstruction | ~142 s | ~173 s | ~206 s |
| Task | ~1698 s | ~1822 s | ~1987 s |

Reconstruction training behaves exactly as expected: loss_final decreases
with k (AWGN 3-seed means 0.0054 → 0.0020 → 0.0012 at k = 12/30/54) and is
~4–6× harder on Rayleigh. Task-oriented training shows the same high
seed-to-seed variance already documented in M1 (e.g., M1's per-run loss
reduction ranged 4%–94% across seeds under the same frozen protocol) —
2 of 10 completed M2 task runs so far ended with a higher final training
loss than initial; no protocol change is implied or made.

**Environment:** Windows 10, Python 3.9.7 (venv over Anaconda base), torch
2.4.1+CPU, no CUDA; ~3 concurrent training lanes × 6 torch threads.

## 7. The one open issue in M2 (needs a supervisor decision — do NOT fix silently)

Two task cells failed **deterministically twice each** (identical
seed+config ⇒ identical outcome):

- `M2_task_awgn_k30_seed43` (hash d270bfa0b913b678)
- `M2_task_awgn_k54_seed42` (hash 59f19c2c451a8106)

Failure mode: the completion check requires the **final-step** encoder /
decoder gradient norms to be finite and strictly positive
(`grads_finite=False`), while everything else is healthy: `nan_batches=0`,
finite final loss (2524.6 and 5397.3), stored checkpoints with fully finite
weights, and 3000/3000 gradient steps executed under a clip-norm of 1.0.
The frozen Step-9C recovery policy only allows deterministic retrain
(no optimizer state is saved — documented limitation), which reproduces the
same result; there is no in-policy retry-mechanism.

**Blocked by this:** the per-k evaluation gates refuse a whole k-lane while
any of its cells is not "done", so k30 + k54 AWGN evaluations (30 files) are
on hold. Rayleigh k30/k54 checkpoints are DONE and unaffected.

**Options that exist (supervisor's call):**
1. **Conditional accept:** treat the frozen checkpoints as valid (weights
   finite; only the last-step grad-norm probe failed) and evaluate them.
2. **Protocol amendment:** add a documented in-policy rule (e.g., grad
   health checked over a trailing window) — requires supervisor approval +
   a new run only if rerun is demanded.
3. **Exclude & document:** report 58/60 training cells and note the two
   cells as protocol-strict failures.

## 8. Execution infrastructure (how it keeps running unattended)

A self-driving watcher (`tools/m2_pipeline_watcher.py`, 24h cap, marker-file
idempotency) drives every remaining stage automatically: per-k evaluations
when each k's training completes, the Rayleigh lane after k18+k24 finish,
then `validate` + `finalize` → `results/M2/step11_m2_summary.json`. Every
run is identity-addressed (`run_id` + full-config sha256 →
`results/M2/<run_id>__<hash>/` with DONE/RUNNING/FAILED markers); completed
runs SKIP, interrupted/failed runs retrain deterministically (frozen 9C
policy). Restarting the watcher is safe and never double-launches.

## 9. Artifacts inventory (all on disk, nothing fabricated)

- M1: `results/M1/` — 36 run dirs, `results/M1/eval/…` 198 files,
  `results/M1/step10_m1_summary.json`
- M2: `results/M2/` — run dirs with `final_ckpt.pt` + `train_report.json`
  + markers; `results/M2/eval/<unit>/eval_<snr>.json`;
  per-lane logs `lane_k*.log`, watcher `watchdog.log`
- M3: `results/M3/m3_report.json` · M4: `results/M4/step13_m4_summary.json`
- Correction & diagnostics (Step 10D/10E): McNemar p-value fix + matched-MSE
  diagnostic (`project/evaluation/statistics.py`, `diagnostics/`)
- Earlier interim report: `fyp_md_files/interim_status_2026-10-02.md`

## 10. What GPT is being asked to help with next

1. **Thesis/report prose** around these measured facts (methods chapter is
   already grounded in the frozen spec: 6-D state, double integrator, shared
   PD controller, k-complex-symbol DeepJSCC, per-codeword power
   normalization, AWGN/Rayleigh models, scalar-quantizer digital chain).
2. **Result narration with the required caveats:** M1's "reconstruction beat
   task-oriented at low k" is a measured finding, not a mechanism claim;
   M2's final three-way ranking is pending evaluation; M4/B=8 is always
   SECONDARY; M3 is structural feasibility, not performance.
3. **Statistics handling:** report corrected McNemar p-values (2 × stored),
   Wilson CIs, paired-t on continuous metrics; note the stale-p regeneration
   action item.
4. Nothing in M2 may be narrated as "final" until
   `results/M2/step11_m2_summary.json` exists (validate + finalize passed).

**ETA for full M2 (honest):** k18/k24 training ~15:00–15:30, their
evaluations + remaining Rayleigh lane by ~16:30–17:30 today → 120/150 files
with only the AWGN k30/k54 decision pending. Everything is running
unattended on the local machine; no cloud dependency.
