# Seed variance of the λ arm at k = 24 (A-14)

**Date:** 2026-10-27 · **Status:** complete — 6/6 cells trained, 30/30 eval slots
· **Pre-registered** in `PROTOCOL_AMENDMENTS.md` A-14 (commit `0753c89`) **before
the first cell** · **Runner:** `tools/run_k24_seeds.py` · **Design locked** by
`tests/test_k24_design.py` (23 checks, commit `5b9a605`) · **Results:**
`results/K24_SEEDS/step_k24_summary.json`

## The question

Every k = 24 number in the thesis was a single seed, and k = 24 is where the
task-oriented link crosses the classical ones. Is that one number an artefact
of one initialisation?

## Design (as registered)

| field | value |
|---|---|
| arm | `lam_ctl_heavy` — λ_d 1.0, λ_u 0.1, λ_v 0.1, λ_T 2.0 (unchanged) |
| k | 24 |
| seeds | 45, 46, 47 |
| channels | awgn, rayleigh |
| cells | 6 |
| width / steps / lr / clip / batch | 128 / 3000 / 1e-3 / 1.0 / 256 — frozen |
| evaluation | E = 1000, SNR 0…20 dB step 5 → 6 units / 30 slots |
| statistics | per seed, then mean and min–max range (n = 3; **not** a CI) |

Cost actually paid: 11 906 s and 11 994 s wall on the two shards (3 cells each,
2 threads per process), i.e. **≈ 3 h 20 min** plus 33 s for the whole
evaluation.

## Training health (all six cells)

| cell | loss start | final | minimum (step) | non-finite grads | NaN batches |
|---|---|---|---|---|---|
| awgn seed 45 | 3495.0 | 17.41 | 13.93 (2079) | 0 | 0 |
| awgn seed 46 | 3998.1 | 15.99 | 13.58 (2207) | 0 | 0 |
| awgn seed 47 | 4509.2 | 17.20 | 14.00 (65) | 0 | 0 |
| rayleigh seed 45 | 3599.2 | 17.63 | 13.95 (2079) | 0 | 0 |
| rayleigh seed 46 | 3887.2 | 16.03 | 13.62 (2207) | 0 | 0 |
| rayleigh seed 47 | 4445.4 | 17.43 | 14.08 (65) | 0 | 0 |

Every cell reaches the frozen terminal regime (16.0–17.6 against the frozen
reference ≈ 16.1) with **zero** non-finite gradient steps and zero NaN batches.
No cell was retried and no cell failed.

## Success rate per seed

| channel | SNR | seed 45 | seed 46 | seed 47 | mean | min–max |
|---|---|---|---|---|---|---|
| awgn | 0 dB | 0.9970 | 1.0000 | 0.9960 | **0.9977** | 0.996–1.000 |
| awgn | 5 dB | 0.9980 | 1.0000 | 0.9970 | **0.9983** | 0.997–1.000 |
| awgn | 10 dB | 0.9980 | 0.9990 | 0.9970 | **0.9980** | 0.997–0.999 |
| awgn | 15 dB | 0.9980 | 0.9990 | 0.9970 | **0.9980** | 0.997–0.999 |
| awgn | 20 dB | 0.9980 | 0.9990 | 0.9980 | **0.9983** | 0.998–0.999 |
| rayleigh | 0 dB | 0.9950 | 0.9950 | 0.9990 | **0.9963** | 0.995–0.999 |
| rayleigh | 5 dB | 0.9980 | 0.9990 | 1.0000 | **0.9990** | 0.998–1.000 |
| rayleigh | 10 dB | 0.9990 | 0.9990 | 1.0000 | **0.9993** | 0.999–1.000 |
| rayleigh | 15 dB | 0.9990 | 0.9980 | 1.0000 | **0.9990** | 0.998–1.000 |
| rayleigh | 20 dB | 0.9990 | 0.9980 | 1.0000 | **0.9990** | 0.998–1.000 |

(E = 1000 episodes per cell and SNR, so every entry is a count out of 1000.)

## Answer, with its size

**The k = 24 number is not an artefact of one initialisation.** The worst seed
is 0.995 at 0 dB; the best is 1.000. The whole seed-to-seed range is
**≤ 0.004**, which is the same size as the binomial sampling error at
E = 1000 (Wilson half-width ≈ 0.003 at p ≈ 0.998). Stated carefully: *seed
variance is not larger than sampling noise at this operating point* — not
"seed variance is zero".

The honest limit of this answer is that **the metric is at its ceiling**. All
three seeds sit at 0.995–1.000, so E = 1000 cannot resolve differences between
them; this run rules out a *large* initialisation dependence at k = 24, it does
not measure seed sensitivity at a non-saturated bandwidth. The frozen task arm
at k = 24 (which is *not* saturated) was never run with three seeds and still
has no variance estimate.

## Context (descriptive only — different protocol)

Frozen M2 at k = 24, awgn, 10 dB, E = 5000, 3 seeds: digital **0.305**,
reconstruction **0.229**, task-oriented **0.035**. The λ arm reaches **0.998**
here. These are *not* directly comparable — M2 uses E = 5000 and its own seeds,
A-14 uses E = 1000 and the Wave-C diagnostic grid — so this is context for
where k = 24 sits, **not** a ranking claim, and no dB gain is derived from it.

## What was not changed

Architecture, width, loss weights, step budget, optimiser, channel model,
evaluation seed, episode budget, SNR grid. No frozen M1/M2/Wave-A/B/C or A-9…A-13
artifact was modified; the new cells live in their own root and no published
denominator changed.

## Reproduce

```powershell
python tools/run_k24_seeds.py verify      # 12 gates
python tools/run_k24_seeds.py status      # 6/6 train, 30/30 slots
python tools/run_k24_seeds.py finalize    # writes step_k24_summary.json
python -m tests.test_k24_design           # 23 checks, design lock
```
