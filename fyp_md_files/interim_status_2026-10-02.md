# Interim Execution Status — 2026-10-02, 13:30 PKT

**Project:** Task-Oriented DeepJSCC for UAV Navigation
**Prepared for:** supervisor meeting (3:00 PM)
**Status line:** M1 COMPLETE · M3 COMPLETE · M4 COMPLETE · M2 EXECUTING (~25%)

All experiments run under the frozen Step-9C protocol (recon 5000 steps ×
batch 4096; task 3000 steps × 256 episodes × T=100; Adam 1e-3, clip 1.0;
train SNR ~ U(0,20) dB per batch; seeds {42,43,44}; test seed 10042;
5000 test episodes; early stopping disabled; M5 deferred+disabled).

---

## M1 — COMPLETE ✅ (Step 10)

- **Training:** 36/36 neural runs complete (task + reconstruction ×
  k ∈ {1,2,3} × AWGN/Rayleigh × seeds {42,43,44}), 0 failed, 0 NaN.
- **Evaluation:** 198/198 paired units complete (11 SNRs {0..20} dB ×
  5000 paired episodes per unit, test seed 10042; both methods share
  bitwise-identical episodes/noise/fading by construction).
- **Statistics:** Wilson 95% CIs per method; paired McNemar (task vs
  reconstruction) and paired-t on final/avg distance and control effort.
  The McNemar p-value implementation defect found in the numerical audit
  was corrected (commit `88e107a`); stored χ² values are unaffected and
  corrected p-values derive exactly from them. **Significance conclusions
  unchanged** (corrected p-values remain ≪ 0.05 for all 198 paired cells).
- **Headline (execution facts):** all 198 paired comparisons favour
  reconstruction-oriented DeepJSCC over the task-oriented variant in this
  low-k regime (n10 > n01 in every cell). No overall ranking claim beyond
  the measured data; full pooled tables in
  `results/M1/step10_m1_summary.json`.

## M3 — COMPLETE ✅ (Step 12)

Structural display (audit 8.5): the matched-budget digital baseline cannot
exist at k ∈ {1,2,3} — the smallest transmissible packet needs
6B + 6 = 12 channel uses at B = 1 (deficits 11/10/9 at k = 1/2/3). This is
an architectural feasibility boundary, **not** a measured performance
result. (`results/M3/m3_report.json`)

## M4 — COMPLETE ✅ (Step 13)

SECONDARY digital reference (never a primary cell): B = 8 at
k = n_dig = 54 (6·8 + 6 = 54, exact fit), evaluation-only, one cell per
channel, M2 SNR grid, 5000 episodes, test seed 10042.
(`results/M4/step13_m4_summary.json`)

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

The B=8 reference transmits the same 54 channel uses as the k=54 neural
cells but is a different operating point (secondary label frozen in the
matrix validator).

## M2 — EXECUTING 🔄 (Step 11, ~25%)

- **Frozen matrix:** task + reconstruction × k ∈ {12,18,24,30,54} ×
  AWGN/Rayleigh × seeds {42,43,44} = 60 neural runs; digital baseline
  evaluation-only at matched budgets B = {12:1, 18:2, 24:3, 30:4, 54:8};
  150 paired 3-method evaluation units (reconstruction/task/digital in one
  paired harness call per unit × 5 SNRs).
- **Progress (13:28):** 15/60 training runs complete, 0 failed, 0 NaN.
  k30 and k54 lanes on task seed-43 runs; k12 lane retraining the two
  runs lost to an environmental restart (deterministic, frozen 9C policy).
  k18/k24 queues next; all remaining evaluation queued per partition.
- **Interpretation guard:** M2 is an empirical three-way comparison. M1
  already showed task-oriented DeepJSCC **underperformed** reconstruction
  at low k — no superiority is presumed for any method at M2's larger
  budgets. Results reported only as measured, per-channel, per-k, per-SNR
  with paired statistics.
- **ETA:** training complete this afternoon; full M2 evaluation and
  summary by early evening (updated live in
  `results/M2/step11_m2_summary.json` once finalized).

---

*Execution report only — no method ranking or scientific verdict beyond
the measured M1 data. M5 remains deferred+disabled per the frozen
protocol (placeholder; requires supervisor-approved experiment definition
before any execution).*
