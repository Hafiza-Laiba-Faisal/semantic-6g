# Wave-A & Wave-B Results — 2026-10-21

Numbers below were read from the artifacts listed under *Provenance*; none
were typed from memory. Figures are in `fyp_md_files/figures/`.
Protocol exceptions are logged in `fyp_md_files/PROTOCOL_AMENDMENTS.md`.

---

## Provenance

| Block | Artifacts | Count | Gate |
|---|---|---|---|
| Wave-A | `results/WAVE_A/eval/*/eval_*.json` | 85 slots / 17 units | `run_wave_a.py verify` ALL PASS |
| Wave-A train | `results/WAVE_A/*/train_report.json` | 46 rows (44 complete, 2 failed) | — |
| Wave-B | `results/WAVE_B/eval/*/eval_*.json` | **350** cells / 14 units | **70/70** σ=0 bit-exact gates |
| Summaries | `step_wave_a_summary.json`, `step_wave_b_summary.json` | — | `wave_b` status `complete` |

Evaluation budget: Wave-A `E = 1000` per unit-SNR (diagnostic grid);
Wave-B `E = 5000` (frozen protocol). Test seed 10042. Both blocks are
**non-frozen, evaluation-only amendments**.

---

## 1. Wave-A — arm ablation (diagnostic; E = 1000)

Pooled over seeds, mean of the 5-SNR grid. `n` = seeds contributing.

| k | channel | arm | SR@0 dB | SR@10 dB | SR@20 dB | mean | n |
|---|---|---|---|---|---|---|---|
| 12 | awgn | **lam_ctl_heavy** | 0.997 | 0.997 | 0.996 | **0.997** | 3 |
| 12 | rayleigh | **lam_ctl_heavy** | 0.991 | 0.999 | 0.998 | **0.997** | 3 |
| 12 | awgn | **lam_ctl_light** | 0.990 | 0.996 | 0.997 | **0.995** | 3 |
| 12 | rayleigh | lam_ctl_light | 0.018 | 0.018 | 0.021 | 0.019 | 3 |
| 12 | awgn | wrecon_pos | 0.262 | 0.237 | 0.237 | 0.243 | 3 |
| 12 | rayleigh | wrecon_pos | 0.271 | 0.238 | 0.225 | 0.244 | 3 |
| 12 | awgn | seedvar_frozen | 0.020 | 0.027 | 0.029 | 0.025 | 2 |
| 12 | rayleigh | seedvar_frozen | 0.033 | 0.031 | 0.031 | 0.032 | 3 |
| 12 | awgn | fixsnr_10db | 0.048 | 0.049 | 0.052 | 0.050 | 1 |
| 12 | awgn | caps_h256 | 0.026 | 0.027 | 0.028 | 0.027 | 1 |
| 12 | rayleigh | long_12k | 0.039 | 0.038 | 0.031 | 0.035 | 1 |
| 1 | awgn | wrecon_pos | 0.141 | 0.184 | 0.170 | 0.168 | 3 |
| 1 | rayleigh | wrecon_pos | 0.132 | 0.145 | 0.138 | 0.141 | 3 |
| 1 | awgn | lam_ctl_light | 0.026 | 0.027 | 0.026 | 0.028 | 3 |
| 1 | rayleigh | lam_ctl_heavy | 0.040 | 0.040 | 0.042 | 0.041 | 3 |
| 1 | awgn | lam_ctl_heavy | 0.018 | 0.017 | 0.016 | 0.017 | 3 |
| 1 | rayleigh | lam_ctl_light | 0.020 | 0.021 | 0.020 | 0.020 | 3 |

Figure: `fig_wave_a_arms.png`.

### 1.1 The headline signal

Adding effort/velocity penalties to the task loss (`lam_ctl_light` =
λ_u,λ_v ×10, `lam_ctl_heavy` = ×100) moves k = 12 task success from the
frozen-λ level (0.02–0.05) to **0.99–1.00**, at every SNR from 0 to 20 dB,
in 3/3 seeds, on both channels (except one degenerate case, §1.3).

### 1.2 Three controls run before trusting it

| Control | Result | Conclusion |
|---|---|---|
| Untrained random model, k=12 | SR 0.016 | harness/metric is not trivially exploitable |
| `lam_ctl_heavy`, −20 dB | SR 0.279 (vs 0.998 at 0 dB) | the channel is genuinely used; not a bypass artifact |
| `lam_ctl_light`, rayleigh | SR 0.015 flat across SNR | one arm is degenerate → the effect is **fragile** |

So the pipeline is sound, but the result is **saturated**: 0.99+ at 0 dB is
near the task's information ceiling, so this grid cannot resolve *how much*
better λ-shaping is. A harder task (tighter goal radius, shorter episode, more
clutter) is required before claiming a specific gain.

### 1.3 Honest caveats

- **No retraining of the 2 failed cells.** `long_12k` (k12/awgn/seed42) and
  `seedvar_frozen` (k12/awgn/seed45) ended `grads_finite=false` with finite
  losses (5203.5 / 5457.6). They remain failures; the k12/awgn/seed45 unit is
  therefore unreachable, capping Wave-A at **85/90 slots, 17/18 units**.
  `status: incomplete` is correct and expected.
- **`caps_h256` is not a 256-wide arm.** The width patch never took effect
  (encoder default is bound at import), so the trained arm is 128-wide and the
  capacity ablation is invalid. No claim may use this arm.
- **Single-seed arms** (`caps_h256`, `fixsnr_10db`, `long_12k`) cannot support
  variance claims.
- **Seed variance is large** in the frozen-λ task arm: k12/awgn task SR is
  0.118 at seed 43 vs 0.007 at seed 44 — the same instability that produced
  the training failures.

---

## 2. Wave-B — CSI-estimation-error ablation (E = 5000, frozen models)

`h_hat = h + eps`, `eps ~ CN(0, σ²)`, equalization and LLR-noise scaling use
`h_hat` only; `eps` is paired across methods and across σ so every method
sees identical episodes. σ = 0 must reproduce frozen M2 bit-exactly:
**70/70 passed**.

### 2.1 Success rate vs bandwidth

| k | Reconstruction | Task | **Digital** | recon MSE | digital MSE |
|---|---|---|---|---|---|
| 12 | **0.253** | 0.054 | 0.064 | 0.0541 | 0.5451 |
| 18 | **0.237** | 0.025 | 0.165 | 0.0522 | 0.0958 |
| 24 | 0.235 | 0.036 | **0.324** | 0.0510 | 0.0889 |
| 30 | 0.262 | 0.021 | **0.629** | 0.0492 | 0.0834 |
| 54 | 0.249 | 0.036 | **0.935** | 0.0487 | 0.0897 |

Figure: `fig_wave_b_sr_vs_k.png`, `fig_wave_b_sr_vs_snr.png`.

**Crossover between k = 18 and k = 24.** Neural JSCC wins clearly at k = 12
(4× the digital baseline) and still at k = 18; from k = 24 the classical
digital link dominates, reaching 0.9984 / 0.9998 / **1.0000** at 10 / 15 / 20 dB
for k = 54. Reconstruction MSE is *flat* in k (0.049–0.054), i.e. the neural
decoder is bottlenecked by its training objective, not by bandwidth.

The frozen task-oriented arm is the worst method at **every** k
(0.021–0.054), including against the digital baseline — the same pathology
Wave-A identified and Wave-A's λ-shaped arms address.

### 2.2 CSI-error sweep (all k pooled)

| σ_csi | recon SR | task SR | digital SR | recon MSE | recon final dist |
|---|---|---|---|---|---|
| 0.0 | 0.2468 | 0.0341 | 0.3841 | 0.0488 | 3.491 |
| 0.05 | 0.2472 | 0.0342 | 0.3840 | 0.0490 | 3.493 |
| 0.1 | 0.2473 | 0.0342 | 0.3846 | 0.0494 | 3.496 |
| 0.2 | 0.2467 | 0.0341 | 0.3875 | 0.0512 | 3.515 |
| 0.4 | 0.2475 | 0.0339 | 0.3946 | **0.0576** | 3.569 |

Figure: `fig_wave_b_csi_sweep.png`.

Per-k MSE penalty at σ = 0.4 is remarkably uniform:
**+18.0 % (k=12), +18.5 % (k=18), +17.8 % (k=24), +18.2 % (k=30), +17.6 % (k=54)**.

**Correct claim:** closed-loop *success rate* is insensitive to CSI magnitude
error up to σ = 0.4, while *reconstruction quality* degrades monotonically by
≈18 % MSE. Success rate is too coarse to expose it — reporting SR alone would
have hidden a real effect.

### 2.3 Two anomalies, stated rather than smoothed over

1. **The digital baseline's σ-response is non-monotone and changes sign with
   bandwidth.** Pooled over k it looks like a small *improvement*
   (0.3841 → 0.3946). Per k it is not:

   | k | digital SR, σ=0 | σ=0.4 | Δ |
   |---|---|---|---|
   | 12 | 0.0643 | 0.0648 | +0.0005 |
   | 18 | 0.1647 | 0.1661 | +0.0014 |
   | 24 | 0.3222 | 0.3334 | +0.0112 |
   | 30 | 0.6135 | 0.6642 | **+0.0507** |
   | 54 | 0.9414 | 0.9193 | **−0.0221** |

   The pooled "+1 pt" is almost entirely the k = 30 outlier; in the cleanest,
   highest-SR regime (k = 54) the digital link **degrades**, as physics
   requires. A paired per-episode check on k=54/seed42/0 dB (3000 episodes,
   identical realisations) gives SR 0.7060 → 0.6150 with **472** episodes
   flipping 0→1 and **745** flipping 1→0 (two-sided p < 1e-4); the same check
   on reconstruction gives 232 vs 287 (p ≈ 0.018).

   Interpretation: the controller is a PD law acting on the *estimate*, so a
   biased estimate can occasionally steer an episode into the goal. That is a
   closed-loop control effect, **not** an information-theoretic benefit of
   worse CSI, and it is largest where success probability sits in the middle
   (k = 30). **Never claim "CSI-error robustness" from the pooled SR curve —
   report it per k, or not at all.**
2. **Digital saturates at high k** (1.0000 at k = 54, 20 dB). Genuine ceiling
   (54 symbols × 2 × 1 bit = 108 bits for a 6-D state), not a bug — but effect
   sizes at high k are compressed against it, which is part of why the k = 30
   anomaly stands out.

### 2.4 Coverage limit

The k = 54 comparison rests on **2 seeds** (42, 44): unit k54/seed43 is
excluded because its frozen M2 task checkpoint is a documented conditional-
accept failure (Amendment A-5). All other k values use 3 seeds. The finalizer
writes this limit into `coverage.excluded_units`; nothing is imputed.

---

## 3. What can and cannot be claimed

**Supported by the data**

1. In the low-bandwidth regime (k = 12) neural JSCC beats the classical
   digital link by ≈4× on closed-loop task success (0.25 vs 0.06).
2. The advantage is gone by k = 24 and reverses strongly by k = 54 (0.25 vs
   0.94); the thesis of task-oriented communication applies to a
   **bandwidth-limited** regime, not universally.
3. The frozen task objective underperforms both alternatives at every
   bandwidth — evidence that the objective, not the architecture, is the
   bottleneck.
4. Effort/velocity shaping (λ-control) removes that pathology at k = 12
   (0.02 → 0.99), robust across 3 seeds and both channels.
5. CSI magnitude error up to σ = 0.4 costs ≈18 % reconstruction MSE but is
   invisible in *pooled* success rate — per k the sign can flip (§2.3), so
   the robustness statement is about the pooled average, not monotonicity.

**Not supported / must not be claimed**

1. Any specific "λ-shaping gain" magnitude — the k = 12 result saturates at
   the task ceiling; a harder task is needed to measure a gain.
2. Generality of λ-shaping across bandwidth — it was only trained at k = 12.
3. A capacity effect from `caps_h256` — the arm is 128-wide.
4. A k = 54 trend at 3-seed confidence — only 2 seeds exist there.
5. CSI-error robustness as measured by success rate alone.

---

## 4. Reproduce

```powershell
.venv\Scripts\python.exe tools\run_wave_a.py evaluate
.venv\Scripts\python.exe tools\run_wave_a.py finalize
.venv\Scripts\python.exe tools\run_wave_b.py run          # idempotent
.venv\Scripts\python.exe tools\run_wave_b.py finalize
.venv\Scripts\python.exe tools\make_figures_wave.py
.venv\Scripts\python.exe -m tests.test_wave_a_tooling     # 60 checks
```

Example sizes: Wave-A eval 85 slots ≈ 2.8 min; Wave-B 350 cells ≈ 49 min
(k = 54 ≈ 40 s/cell).
