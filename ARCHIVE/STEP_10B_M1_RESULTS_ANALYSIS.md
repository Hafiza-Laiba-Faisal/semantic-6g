# Step 10B — M1 Scientific Results Analysis (Read-Only)

**Project:** Task-Oriented Deep Joint Source-Channel Coding for UAV Navigation
Under Noisy and Fading Wireless Channels
**Date:** 2026-10-02
**Git state at analysis time:** `HEAD = origin/main = 01338fa` (branch `main`, 0 ahead / 0 behind)
**Data basis:** `results/M1/` — 36/36 training runs, 198/198 paired evaluation units
(396 method-rows), 5,000 test episodes per method per cell, test seed `10042`,
SNR grid {0, 2, 4, ..., 20} dB, training seeds {42, 43, 44}.
**Nature:** READ-ONLY analysis. No file under `results/` was modified. No training
or evaluation was executed. All statistics below are recomputed directly from the
per-cell evaluation JSONs (Wilson CIs, McNemar counts, Spearman ranks computed
independently of the executor tool).

---

## 1. Data integrity pre-check

- 36 run directories, each with `final_ckpt.pt`, `train_report.json`, `DONE.json`;
  no `FAILED.json` markers; checkpoint integrity 36/36 (Step 10A `checkint`).
- 198/198 eval units × 11 SNR files = 198 cells, each with
  `episodes=5000`, `test_seed=10042`, `paired=true`, both method aggregates,
  success+failure = 5000 per method, Wilson and McNemar blocks internally consistent.
- No M2/M3/M4/M5 artifacts exist anywhere under `results/`.
- Pooled cells combine the 3 training seeds → 15,000 episodes per pooled condition
  (132 conditions across methods × k × channel × SNR).

## 2. Pooled success rates (%), 3 seeds × 5,000 episodes = 15,000 per cell

| Method | Channel | k | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 (dB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reconstruction | awgn | 1 | 13.9 | 15.0 | 15.6 | 16.4 | 16.2 | 16.2 | 15.5 | 15.5 | 15.4 | 15.2 | 15.2 |
| reconstruction | awgn | 2 | 27.2 | 29.2 | 30.9 | 30.4 | 29.1 | 27.6 | 26.3 | 25.4 | 24.8 | 24.4 | 24.2 |
| reconstruction | awgn | 3 | 27.8 | 29.4 | 28.9 | 27.9 | 27.0 | 26.0 | 25.1 | 24.8 | 24.3 | 24.1 | 24.0 |
| reconstruction | rayleigh | 1 | 13.0 | 13.4 | 14.7 | 15.3 | 15.5 | 15.4 | 15.6 | 15.7 | 15.8 | 15.7 | 15.5 |
| reconstruction | rayleigh | 2 | 24.8 | 27.5 | 28.9 | 29.8 | 29.1 | 28.4 | 27.4 | 26.3 | 25.8 | 25.2 | 24.7 |
| reconstruction | rayleigh | 3 | 26.6 | 29.1 | 29.1 | 28.6 | 27.7 | 26.7 | 25.9 | 24.7 | 24.0 | 23.6 | 23.5 |
| task | awgn | 1 | 1.7 | 1.7 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 |
| task | awgn | 2 | 3.5 | 3.5 | 3.3 | 3.3 | 3.0 | 2.8 | 2.8 | 2.7 | 2.8 | 2.9 | 2.7 |
| task | awgn | 3 | 2.1 | 2.2 | 2.1 | 2.1 | 2.1 | 2.1 | 2.1 | 2.1 | 2.1 | 2.1 | 2.1 |
| task | rayleigh | 1 | 4.0 | 4.3 | 4.6 | 4.6 | 4.9 | 4.9 | 4.7 | 5.0 | 5.1 | 4.9 | 5.0 |
| task | rayleigh | 2 | 2.6 | 2.7 | 2.7 | 2.7 | 2.8 | 2.7 | 2.8 | 2.8 | 2.9 | 3.0 | 3.0 |
| task | rayleigh | 3 | 3.4 | 3.4 | 3.6 | 3.5 | 3.5 | 3.5 | 3.7 | 3.6 | 3.6 | 3.6 | 3.6 |

## 3. Method effect — paired task vs reconstruction

- Reconstruction achieves a **higher success count in 198/198 evaluation cells**
  (0 reversals, 0 ties at the unit level).
- McNemar test on paired per-episode successes: **p < 0.05 in 198/198 cells**;
  worst-case p = 5.186×10⁻²⁷ (best case 7.177×10⁻³⁰⁴). Mean discordant pairs per
  cell ≈ 1,182 (max 1,674); task loses more discordant pairs than it wins in
  **198/198** cells (n10 > n01 everywhere).
- Mean risk difference (task − reconstruction) per condition, pooled over seeds and SNRs:
  - awgn: k1 = −13.82 pp, k2 = −24.20 pp, k3 = −24.19 pp
  - rayleigh: k1 = −10.34 pp, k2 = −24.27 pp, k3 = −22.76 pp
  - Task is worse in **every one of the 33 seed×SNR cells** in each of the six conditions.
- Continuous metrics (pooled over all seeds and SNRs):

| Metric | reconstruction | task |
|---|---|---|
| Final distance (mean) | **3.79** | 9.57 |
| Avg distance (mean) | **4.54** | 9.15 |
| Control effort (mean) | **691.2** | 770.2 |

  Paired t on final distance: task is farther from goal than reconstruction in
  **198/198 cells**. Task is dominated on all three axes (navigates worse while
  spending more control effort).
- Example Wilson 95% CIs (pooled): reconstruction k1 awgn @0 dB = 13.90%
  [13.36, 14.46]; task k1 awgn @0 dB = 1.75% [1.55, 1.97] — CIs are separated by
  an order of magnitude; the same holds in every condition.

## 4. Channel-use (k) effect — reconstruction

- k1 → k2 is the dominant jump: 15.5% → 27.2% (awgn), 15.1% → 27.1% (rayleigh)
  averaged over SNRs (~1.75×).
- k2 → k3 adds nothing (27.2 → 26.3 awgn; 27.1 → 26.3 rayleigh); the ordering is
  slightly non-monotonic with a k3 dip. **k = 2 is the efficiency sweet spot.**

## 5. Channel effect — reconstruction

- AWGN vs Rayleigh differ by at most ±0.4 pp at every k (e.g., k2: 27.2 vs 27.1).
  For the reconstruction method under this protocol, **channel type is practically
  immaterial** at these training budgets.

## 6. SNR trends (Spearman ρ over the 11-point grid, pooled)

| Method | Channel | k | ρ | Peak SNR | Range |
|---|---|---|---|---|---|
| reconstruction | awgn | 1 | −0.04 | 6 dB | 13.9–16.4% |
| reconstruction | awgn | 2 | −0.84 | 4 dB | 24.2–30.9% |
| reconstruction | awgn | 3 | −0.95 | 2 dB | 24.0–29.4% |
| reconstruction | rayleigh | 1 | +0.87 | 16 dB | 13.0–15.8% |
| reconstruction | rayleigh | 2 | −0.47 | 6 dB | 24.7–29.8% |
| reconstruction | rayleigh | 3 | −0.85 | 4 dB | 23.5–29.1% |
| task | awgn | 1–3 | −0.37…−0.83 | 0–2 dB | ≈1.6–3.5% |
| task | rayleigh | 1–3 | +0.81…+0.97 | 12–18 dB | 2.6–5.1% |

## 7. What the data SUPPORTS

1. **Universal, statistically overwhelming superiority of reconstruction-based
   DJSCC over the task-oriented policy under the frozen M1 protocol** — 198/198
   paired cells, all McNemar p<0.05, consistent on continuous metrics and risk
   differences across all seeds, channels, k values, and SNRs.
2. **k = 2 as the channel-use sweet spot** for reconstruction (large k1→k2 gain,
   flat k2→k3).
3. **Channel-type insensitivity for reconstruction** at these budgets.
4. **No train/test leakage signal** for the task method: train-time post-training
   eval success (2.34% / 3.12% / 3.91% for k=1/2/3, seed 42, awgn) is consistent
   with test-time averages (1.95% / 2.63% / 2.83%).

## 8. What the data does NOT support

1. Any claim of task-oriented advantage under this protocol — or that the gap
   would close at larger budgets. Both methods are likely **under-trained**:
   absolute success rates remain ≤ 31% (reconstruction) and ≤ 5.1% (task) even at
   the best operating points, against the frozen 9C budgets
   (5,000×4096 recon / 3,000×256 T=100 task steps).
2. A monotonic "higher SNR → better navigation" law: at k ≥ 2 the reconstruction
   success **declines** toward 20 dB (see Anomaly A).
3. A consistent k2-vs-k3 ordering for the task method (the order flips between
   awgn and rayleigh).
4. Any generalization of these numbers beyond M1's frozen small-budget protocol,
   single test seed (10042), and the specific 2-D navigation environment.

## 9. Anomalies flagged (for Step 10C / Step 11 review — no conclusions drawn)

- **A — Inverted SNR slope at k ≥ 2 (reconstruction).** Success peaks at 2–6 dB
  and declines toward 20 dB (ρ = −0.85…−0.95 at k3/k2 awgn, k3 rayleigh).
  Counterintuitive for a noise-limited task; primary suspect is the frozen
  U(0,20) dB per-batch training-SNR policy interacting with fixed test SNRs,
  or noise-scaling in the reconstruction decoder path. Deserves a targeted probe.
- **B — Task k1: Rayleigh ≫ AWGN (4.7% vs 1.6%).** Fading being "easier" than
  AWGN is physically odd, and the seed spread on task/k1/rayleigh is extreme:
  per-seed success ≈ **0.7% / 4.3% / 9.3%** (CV ≈ 0.68) across seeds 42/43/44 —
  a training-instability signature, not measurement noise.
- **C — Degenerate cells.** 11 pooled conditions fall below 2% success, all of
  them task-method on AWGN — near-collapsed task policies on the AWGN lane.
- **D — Task uses more control effort while navigating worse.** Consistent with
  an under-trained weighted objective (λ_terminal = 2.0, λ_distance = 1.0 at
  3,000 steps) rather than a fundamental mechanism failure; not resolvable from
  M1 data alone.

## 10. Verdict

For the **frozen Step-9C M1 protocol**, the evidence is statistically conclusive:

> **Reconstruction-based DeepJSCC outperforms the task-oriented policy in every
> one of the 198 paired evaluation cells (McNemar p<0.05 in all; max
> p = 5.2×10⁻²⁷), with k = 2 the channel-use sweet spot and channel type
> practically irrelevant for reconstruction.**

The result is directionally robust but scientifically bounded by: (i) low
absolute success rates (≤31%), (ii) the inverted high-SNR slopes at k ≥ 2,
(iii) task-method seed instability (CV ≈ 0.7 in the worst condition), and
(iv) a single test seed. These limitations must be carried forward explicitly
into Step 11 (and the thesis discussion) rather than averaged away.

## 11. Recommended next actions

1. Probe Anomaly A (noise-scaling path + per-seed reconstruction curves vs SNR).
2. Probe Anomaly B/C (task AWGN collapse; per-seed training-loss trajectories of
   the three task/k1 runs).
3. Record these findings in the thesis results chapter as descriptive, paired,
   protocol-bounded comparisons — no overall ranking claims beyond the frozen
   protocol.
4. Defer M5 (unchanged, per Step 9C freeze).

---

*Analysis tool: ad-hoc read-only script over `results/M1/eval/**/*.json`
(reproducible; no side effects). Git working tree untouched at commit `01338fa`.*
