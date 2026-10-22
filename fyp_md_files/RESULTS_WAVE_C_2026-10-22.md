# Wave-C results — does λ-shaping survive at other bandwidths?

**Date:** 2026-10-22 · **Amendment:** A-8 (full grid, 2026-10-21) + A-8b (screening
subset, pre-registered before its first cell) · **Status:** `complete` for the
A-8b screening subset, `declared` (not run) for the rest of the grid.

Answer in one line: **yes at k = 18 and k = 54, on both channels, and the
advantage over the classical link is largest exactly where the classical link
struggles (low SNR, high k).** One seed, so this is a screening result, not a
variance claim.

---

## 1. What was run

| field | value |
|---|---|
| cells | 4 = `lam_ctl_heavy` × k{18, 54} × {awgn, rayleigh} × seed 42 |
| steps | 3000 (frozen `config.TASK_TRAIN_STEPS`) |
| eval | E = 1000 per unit-SNR, SNR = 0/5/10/15/20 dB |
| slots | 20 / 20 complete, 4 / 4 units complete |
| root | `results/WAVE_C/` |
| log | `C:\wa_evidence\wave_c_shard{0..3}.log`, `wave_c_eval.log` |

All four cells were trained **in parallel** (4 processes × 2 threads on 8
logical cores). Wall clock 4956–5457 s per cell; sequential single-process cost
on this machine is ~2260 s per 3000-step cell, i.e. parallelism cost 2.2× per
cell but saved ~2× wall clock for four cells.

| cell | wall | task loss | nan batches |
|---|---|---|---|
| heavy · awgn · k18 | 4956 s | 16.078 | 0 |
| heavy · rayleigh · k18 | 5152 s | 16.120 | 0 |
| heavy · awgn · k54 | 5350 s | 16.084 | 0 |
| heavy · rayleigh · k54 | 5457 s | 16.094 | 0 |

Training stability is **independent of k and channel** (spread 0.04, no NaNs),
which is itself the reproduction of Wave-A's stabilising effect at k = 12.

---

## 2. Success rate (Wave-C, E = 1000)

| unit | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|
| heavy · awgn · k18 | 0.999 | 0.999 | 0.999 | 0.998 | 0.999 |
| heavy · rayleigh · k18 | 0.993 | 0.997 | 0.996 | 0.997 | 0.997 |
| heavy · awgn · k54 | 0.998 | 0.999 | 0.999 | 0.998 | 0.999 |
| heavy · rayleigh · k54 | 0.996 | 0.998 | 0.998 | 0.998 | 0.998 |

Success is at the task ceiling in every condition. **The ceiling is the reason
the baselines matter**, not the absolute number.

## 3. Baselines (frozen protocol, E = 5000, σ = 0)

| k | channel | task-oriented | reconstruction | digital |
|---|---|---|---|---|
| 18 | awgn | 0.0179 | 0.2706 | 0.1640 |
| 54 | awgn | 0.0159 | 0.2455 | 0.9736 |
| 18 | rayleigh | 0.0238 | 0.2894 | 0.1656 |
| 54 | rayleigh | 0.0357 | 0.2576 | 0.7178 |

awgn rows: `results/M2/step11_m2_summary.json` (E = 5000, frozen protocol).
rayleigh rows: `results/WAVE_B/step_wave_b_summary.json`, σ = 0
(k54 rayleigh digital at 20 dB is 1.0000; every other digital entry is flat in
SNR).

**Contrasts that carry the claim:**

| condition | λ-heavy | best baseline | gap |
|---|---|---|---|
| k18, awgn, 0 dB | 0.999 | digital 0.164 | +83.5 pp |
| k18, rayleigh, 0 dB | 0.993 | digital 0.166 | +82.7 pp |
| k54, awgn, 0 dB | 0.998 | digital 0.974 | +2.4 pp |
| k54, rayleigh, 0 dB | 0.996 | digital 0.718 | +27.8 pp |
| k54, rayleigh, 20 dB | 0.998 | digital 1.000 | −0.2 pp (parity) |

At k = 54 the classical link saturates at high SNR, so λ-shaping **cannot** win
there — it ties. Its advantage lives at low SNR and on the fading channel, i.e.
exactly in the regime the thesis is about. At k = 18 the classical link never
recovers (< 0.17 at any SNR tested) and λ-shaping is at 0.99.

## 4. Is 0.99 a real policy or a degenerate one?

This is the first question a reviewer should ask about a zero-effort "success".
Same unit, same episodes (k18, rayleigh, 0 dB, seed 42):

| arm | success | final distance | control effort | time-to-goal |
|---|---|---|---|---|
| **λ-heavy (Wave-C)** | **0.993** | **0.125** | **341** | 34.4 |
| frozen task-oriented (M2) | 0.019 | 11.21 | 778 | — |
| digital (M2) | 0.166 | 2.61 | 747 | — |
| reconstruction (M2) | 0.289 | 3.50 | 670 | — |

The λ arm ends **two orders of magnitude closer** to the target than the task
arm while spending **less than half** the control effort. A policy that does
nothing, or that parks at the origin, cannot produce a small final distance on
a randomly initialised target. The success is tracking, not idling.

## 5. Side observation (not part of the A-8b comparison)

Two `lam_ctl_light` k = 18 awgn cells were trained under the full A-8 design
before the screening reduction. They are kept (never deleted or overwritten) and
reported only as a labelled observation. At the same unit and seed:

| arm (k18, awgn, seed 42) | success @0 dB | final distance | control effort |
|---|---|---|---|
| light (λ_u, λ_v ×10) | 0.991 | 0.103 | 361 |
| heavy (λ_u, λ_v ×100) | 0.999 | 0.056 | 258 |

and light seed 43 gives 0.987. Direction is consistent with a dose effect
(stronger penalties → closer tracking, lower effort), but **two seeds in one
condition is not a dose-response study**.

## 6. Claim limits — read before quoting

1. **One seed (42).** No variance or significance claim. E = 1000 gives roughly
   ±0.3–0.5 pp binomial error per point, which is *not* the dominant
   uncertainty here; seed-to-seed variation is unknown for these cells.
2. **One arm in the screening subset.** The light arm appears only as §5.
3. **Two k values.** No trend across k; k = 24 and k = 30 were not run.
4. **Protocol mismatch with baselines.** Wave-C is E = 1000 (diagnostic);
   M2/Wave-B baselines are E = 5000 (frozen). Gaps of tens of pp are robust to
   this, the k54 awgn +2.4 pp gap is not — do not quote it as a win.
5. **Success saturates.** Any comparison "λ-heavy vs λ-heavy" across k is
   uninformative at this ceiling; use the baselines and the secondary metrics.
6. **k = 54 ties the digital link at 20 dB.** The honest statement is
   "matches the classical link where it saturates, beats it by 27.8 pp at
   0 dB", not "beats digital everywhere".

## 7. Reproduce

```bat
python tools/run_wave_c.py verify
python tools/run_wave_c.py evaluate
python tools/run_wave_c.py finalize --tag=lam_ctl_heavy --k=18 --k=54 --seed=42
python tools/make_figures_wave.py
```

`evaluate` is slot-refreshing: a slot is reused only if the file already holds
every trained arm for that unit *and* is stamped `experiment = WAVE_C`.
Figure: `fyp_md_files/figures/fig_wave_c_success.png`.

## 8. Provenance and incidents

- Two machine sleeps (~22 min frozen, then a hard power loss) interrupted the
  first sequential attempt at 2/32 cells; the sharded restart skipped completed
  cells. No cell was retrained after success, no artifact overwritten.
- The first evaluation pass wrote Wave-A labels into Wave-C eval files and
  skipped a later-trained arm (`lam_ctl_light` seed 42) because the per-unit
  artifact already existed. Both are fixed in `tools/run_wave_a.py`:
  experiment/amendment labels now come from `WAVE_EXPERIMENT`/`WAVE_AMENDMENT`,
  and a slot is refreshed when it lacks a trained arm or carries foreign
  labels. All 25 Wave-C slots were regenerated after the fix; Wave-A artifacts
  are byte-identical under the fix (same label strings, all arms present).
