# REPORT FOR EXTERNAL REVIEW — Step 10B: M1 Scientific Results Analysis

> NOTE TO REVIEWER: This document is self-contained. You do not need access to
> the code repository. All numbers below were computed directly from the
> completed experiment output files (JSON evaluation records). Please review
> the methodology, statistics, and interpretations, and point out any errors,
> unsupported claims, or additional analyses you would recommend.

---

## 1. PROJECT CONTEXT

**FYP Title:** Task-Oriented Deep Joint Source-Channel Coding for UAV Navigation
Under Noisy and Fading Wireless Channels

**Setup:** Simulation-only, Python + PyTorch. Two learned communication methods
are compared on a frozen 2-D UAV navigation task (reach the goal within radius
0.5 within horizon T=100, dt=0.1):

- **reconstruction** — DeepJSCC: encoder/decoder trained to reconstruct the
  transmitted state features; control policy runs on the reconstruction.
- **task** — Task-oriented DeepJSCC: encoder/decoder trained end-to-end on the
  navigation objective (weighted loss: distance 1.0, terminal 2.0, control 0.0,
  velocity 0.0).

**Channels:** AWGN and slow Rayleigh fading. **Channel uses per transmission:**
k ∈ {1, 2, 3}. **Training seeds:** {42, 43, 44}. **Training SNR policy:**
U(0,20) dB per batch. **Frozen budgets (Step 9C):** reconstruction 5,000 steps
× batch 4096 states; task 3,000 steps × batch 256 episodes, T=100. Adam,
lr=1e-3, grad-clip 1.0, no early stopping. **M5 (digital baseline): DEFERRED,
not executed.**

**Evaluation protocol (frozen):** paired comparison — for each
(k, channel, seed, SNR) cell, ONE evaluation call hands both same-seed models
the SAME test episodes, initial states, goals, noise and fading realizations
(bitwise identical pairing by construction). 5,000 test episodes per method per
cell, test seed 10042, SNR grid {0, 2, 4, ..., 20} dB.

**Matrix:** 2 methods × 3 k × 2 channels × 3 seeds = **36 training runs**;
evaluation: 18 paired units (k, channel, seed) × 11 SNRs = **198 paired cells**
(396 per-method rows; 132 pooled conditions after pooling 3 seeds).

**Success metric:** episode success = UAV reaches goal radius 0.5 within T=100.
Secondary: final distance, avg distance, control effort, MSE, time-to-goal.

**Statistics:** Wilson 95% CI per method per cell; McNemar test on paired
per-episode successes; paired t on paired continuous metrics. No SciPy —
implementations verified against closed forms.

**Execution integrity (verified before analysis):** 36/36 runs complete with
checkpoints and DONE markers; 198/198 eval cells with episodes=5000 and
test_seed=10042 everywhere; checkpoint/config-hash integrity 36/36; no missing
or duplicate run identities; no M2/M3/M4/M5 artifacts; analysis performed
read-only.

---

## 2. MAIN RESULTS — pooled success rates (%)

Each pooled cell = 3 seeds × 5,000 episodes = 15,000 episodes.

### Reconstruction (DeepJSCC)

| k | channel | 0 dB | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | awgn | 13.9 | 15.0 | 15.6 | 16.4 | 16.2 | 16.2 | 15.5 | 15.5 | 15.4 | 15.2 | 15.2 |
| 1 | rayleigh | 13.0 | 13.4 | 14.7 | 15.3 | 15.5 | 15.4 | 15.6 | 15.7 | 15.8 | 15.7 | 15.5 |
| 2 | awgn | 27.2 | 29.2 | 30.9 | 30.4 | 29.1 | 27.6 | 26.3 | 25.4 | 24.8 | 24.4 | 24.2 |
| 2 | rayleigh | 24.8 | 27.5 | 28.9 | 29.8 | 29.1 | 28.4 | 27.4 | 26.3 | 25.8 | 25.2 | 24.7 |
| 3 | awgn | 27.8 | 29.4 | 28.9 | 27.9 | 27.0 | 26.0 | 25.1 | 24.8 | 24.3 | 24.1 | 24.0 |
| 3 | rayleigh | 26.6 | 29.1 | 29.1 | 28.6 | 27.7 | 26.7 | 25.9 | 24.7 | 24.0 | 23.6 | 23.5 |

### Task-oriented DeepJSCC

| k | channel | 0 dB | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | awgn | 1.7 | 1.7 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 |
| 1 | rayleigh | 4.0 | 4.3 | 4.6 | 4.6 | 4.9 | 4.9 | 4.7 | 5.0 | 5.1 | 4.9 | 5.0 |
| 2 | awgn | 3.5 | 3.5 | 3.3 | 3.3 | 3.0 | 2.8 | 2.8 | 2.7 | 2.8 | 2.9 | 2.7 |
| 2 | rayleigh | 2.6 | 2.7 | 2.7 | 2.7 | 2.8 | 2.7 | 2.8 | 2.8 | 2.9 | 3.0 | 3.0 |
| 3 | awgn | 2.1 | 2.2 | 2.1 | 2.1 | 2.1 | 2.1 | 2.1 | 2.1 | 2.1 | 2.1 | 2.1 |
| 3 | rayleigh | 3.4 | 3.4 | 3.6 | 3.5 | 3.5 | 3.5 | 3.7 | 3.6 | 3.6 | 3.6 | 3.6 |

Example Wilson 95% CIs (pooled): reconstruction k1 awgn @0 dB = 13.90%
[13.36, 14.46]; task k1 awgn @0 dB = 1.75% [1.55, 1.97]. CIs are separated by
roughly an order of magnitude in every condition.

---

## 3. PAIRED METHOD EFFECT (task vs reconstruction)

- Reconstruction has a higher success count in **198/198 cells** (0 reversals).
- **McNemar test: p < 0.05 in 198/198 cells.** Worst-case p = 5.186×10⁻²⁷;
  best-case p = 7.18×10⁻³⁰⁴. Mean discordant pairs per cell ≈ 1,182
  (max 1,674 out of 5,000 pairs). The task method loses more discordant
  episodes than it wins in **198/198 cells**.
- Mean risk difference (task − reconstruction), pooled over seeds and SNRs:
  - awgn: k1 = −13.82 pp, k2 = −24.20 pp, k3 = −24.19 pp
  - rayleigh: k1 = −10.34 pp, k2 = −24.27 pp, k3 = −22.76 pp
  - Within each of the 6 (k, channel) conditions, task is worse in **every one
    of the 33 seed×SNR cells**.
- Continuous metrics, pooled over all seeds and SNRs:

| Metric | reconstruction | task |
|---|---|---|
| Final distance to goal | **3.79** | 9.57 |
| Average distance | **4.54** | 9.15 |
| Control effort | **691** | 770 |

  Paired t on final distance: task farther than reconstruction in **198/198
  cells**. Task is dominated on all three axes: it navigates worse AND spends
  more control effort.

---

## 4. STRUCTURAL EFFECTS

**Channel uses (k):** k1→k2 roughly doubles success (15.5% → 27.2% awgn; 15.1%
→ 27.1% rayleigh). k2→k3 adds nothing (26.3% both channels) — a plateau/slight
dip. **k = 2 is the efficiency sweet spot.**

**Channel type:** for reconstruction, AWGN vs Rayleigh differ by ≤ 0.4 pp at
every k — practically immaterial at these budgets. For task, Rayleigh is
consistently *higher* than AWGN at k1 (4.7% vs 1.6%) — see Anomaly B.

**SNR trends (Spearman ρ over the 11-point grid, pooled):**

| Method | Channel | k | ρ | Peak SNR |
|---|---|---|---|---|
| reconstruction | awgn | 1 | −0.04 (flat) | 6 dB |
| reconstruction | awgn | 2 | −0.84 | 4 dB |
| reconstruction | awgn | 3 | −0.95 | 2 dB |
| reconstruction | rayleigh | 1 | +0.87 | 16 dB |
| reconstruction | rayleigh | 2 | −0.47 | 6 dB |
| reconstruction | rayleigh | 3 | −0.85 | 4 dB |
| task | awgn | 1–3 | −0.37…−0.83 | 0–2 dB |
| task | rayleigh | 1–3 | +0.81…+0.97 | 12–18 dB |

---

## 5. WHAT THE DATA SUPPORTS

1. **Universal, statistically overwhelming superiority of reconstruction-based
   DeepJSCC over the task-oriented policy under this frozen protocol** —
   consistent across 198/198 paired cells, all seeds, channels, k values, SNRs,
   and both binary and continuous metrics.
2. **k = 2 as the channel-use sweet spot** for reconstruction.
3. **Channel-type insensitivity for reconstruction** at these budgets.
4. **No train/test leakage signal**: task models' train-time post-training eval
   success (2.34% / 3.12% / 3.91% for k=1/2/3) matches test-time averages
   (1.95% / 2.63% / 2.83%).

## 6. WHAT THE DATA DOES NOT SUPPORT

1. Any task-oriented advantage under this protocol — nor that the gap would
   close with more training (untested).
2. That the frozen budgets are sufficient: absolute success ≤ 31% (recon) and
   ≤ 5.1% (task) even at best operating points suggests **both methods are
   under-trained**; the comparison may not reflect asymptotic performance.
3. A monotonic "higher SNR → better navigation" law — violated at k ≥ 2
   (success declines toward 20 dB).
4. Generalization beyond this environment, the single test seed (10042), and
   the frozen protocol. No ranking claim beyond M1 is licensed.

---

## 7. ANOMALIES / OPEN QUESTIONS FOR REVIEW

- **A — Inverted SNR slope at k ≥ 2 (reconstruction).** Success peaks at 2–6 dB
  then *declines* toward 20 dB (ρ = −0.85…−0.95). Counterintuitive for a
  noise-limited task. Hypotheses: (i) the U(0,20) dB training-SNR distribution
  mismatches high test SNRs (models rarely trained near 20 dB behave worse
  there); (ii) noise-scaling bug in the decoder path at high SNR; (iii) more
  channel uses + low noise induces overconfidence/overfitting in the control
  loop. *Question: which diagnostic would best discriminate these?*
- **B — Task k1: Rayleigh (4.7%) ≫ AWGN (1.6%).** Fading being easier than AWGN
  is physically odd. Seed spread is extreme: per-seed success 0.7% / 4.3% /
  9.3% across seeds 42/43/44 (CV ≈ 0.68) — training instability, not
  measurement noise. *Question: is per-seed training-loss trajectory analysis
  the right next step?*
- **C — Degenerate cells.** 11 pooled conditions < 2% success, all task/AWGN —
  near-collapsed task policies on the AWGN lane.
- **D — Control-effort paradox.** Task spends MORE control effort while
  navigating worse. Consistent with an under-trained weighted objective
  (terminal weight 2.0) rather than a mechanism failure, but not resolvable
  from this data alone.
- **E — Single test seed.** All stochasticity in evaluation comes from test
  seed 10042; per-seed training variation is captured, but evaluation-seed
  sensitivity is not measured.

## 8. QUESTIONS FOR THE REVIEWER

1. Do you agree the paired McNemar + per-cell sign consistency across 33
   seed×SNR cells is sufficient to claim a decisive method comparison *within
   this protocol*?
2. For Anomaly A, what analysis would you run first: per-seed SNR curves, a
   noise-scaling unit probe, or a retrain at a fixed high training SNR?
3. Given ≤31% absolute success, is it defensible to present the reconstruction
   superiority as the paper's primary M1 finding, with under-training as an
   explicit limitation? Or would you require budget-scaling evidence first?
4. Any additional checks you would require before writing this into the thesis
   results chapter (e.g., evaluation-seed resampling, bootstrap CIs, effect
   sizes with CIs)?

---

*Source: results/M1/ per-cell evaluation JSONs (198 cells), computed read-only
at commit 01338fa. Companion internal report: STEP_10B_M1_RESULTS_ANALYSIS.md.*
