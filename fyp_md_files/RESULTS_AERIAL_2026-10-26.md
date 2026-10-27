# Aerial link budget - what does the system need on a 3GPP UMa-aerial link?

**Date:** 2026-10-26 - **Amendment:** A-10 - **Status:** complete
(evaluation only; nothing trained, no frozen artifact written).

Answer in one line: **at 500 m / 50 m / 2 GHz the lambda-shaped policy still
wins, and the win is large in transmit-power terms - it needs 91 dB (k = 18)
or 87 dB (k = 54) of transmit SNR to reach 0.90, while the classical link
needs 108 dB at k = 54 and cannot reach 0.90 at all at k = 18.**

---

## 1. What was run

| field | value |
|---|---|
| geometry | d2D = 500 m, h = 50 m, fc = 2 GHz, d3D = 502.49 m |
| model | 3GPP TR 36.777 / TR 38.901 UMa-aerial, large-scale layer only |
| P_LoS | 0.8887 |
| PL_LoS | 93.45 dB |
| PL_NLoS | 113.09 dB |
| PL_mean | 95.63 dB (gain 1.65e-05) |
| methods | Wave-C `lam_ctl_heavy` k = 18 / k = 54 (seed 42); frozen convolutional digital at the M2 rate (`M2_B_BY_K`) |
| episodes | E = 1000 per point, frozen test stream |
| effective-SNR grid | -40, -35, ..., +25 dB (14 points) |
| root | `results/AERIAL/` |

Nothing is retrained: the Wave-C checkpoints are evaluated unchanged, and the
aerial layer enters only as the link-budget mapping described next.

## 2. How the aerial layer enters (and what it does not model)

Under perfect-CSI ZF a deterministic transmit gain `g` multiplies the signal
but not the noise, so the received effective SNR is exactly

```
SNR_eff = SNR_tx - PL_linear_scale   (dB)
```

with the *same* Rayleigh fading statistics. LoS and NLoS are then combined on
that one measured curve:

```
SR(SNR_tx) = P_LoS * SR(SNR_tx - PL_LoS)
           + (1 - P_LoS) * SR(SNR_tx - PL_NLoS)
```

This is exact for i.i.d. episodes with an independent LoS/NLoS draw. It is a
mixture of two evaluations of the same curve, not a new channel simulation.

**Modelled:** LoS probability, LoS and NLoS path loss, and their mixture.
**Not modelled:** Doppler, altitude-dependent small-scale fading, shadowing,
per-episode gain dispersion, antenna patterns, noise figure or bandwidth.
`reference_pl_db = 0` means the reported path loss is absolute; the transmit
SNR numbers below are therefore relative to a unit-gain benchmark, not a
hardware budget.

## 3. Results

### 3.1 Required SNR

| method | SR >= 0.90 at eff. SNR | -> transmit SNR | SR >= 0.99 at eff. SNR | -> transmit SNR |
|---|---|---|---|---|
| heavy, k = 18 | -8.3 dB | 91 dB | -0.8 dB | 107 dB |
| heavy, k = 54 | -12.3 dB | 87 dB | -5.2 dB | 102 dB |
| digital, k = 18 | not reachable | not reachable | not reachable | not reachable |
| digital, k = 54 | +3.5 dB | 108 dB | +7.4 dB | 117 dB |

"Not reachable" is honest: the k = 18 digital link is payload-starved at
B = 2 (6 source bits per component after the code), so its success rate
saturates near 0.15-0.17 and no transmit power fixes that. The frozen M2
study already frames that condition that way.

### 3.2 Measured effective-SNR curves

| SNR_eff (dB) | -40 | -30 | -20 | -15 | -10 | -5 | 0 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| heavy k = 18 | 0.026 | 0.073 | 0.345 | 0.630 | 0.861 | 0.975 | 0.993 | 0.997 | 0.996 | 0.997 |
| heavy k = 54 | 0.036 | 0.139 | 0.613 | 0.844 | 0.949 | 0.992 | 0.996 | 0.998 | 0.998 | 0.998 |
| digital k = 54 | 0.013 | 0.012 | 0.010 | 0.017 | 0.021 | 0.104 | 0.717 | 0.982 | 0.999 | 1.000 |

The digital curve is flat because its failure at k = 54 is a *payload* failure
at the chosen rate, not a noise failure: extra SNR cannot repair it. This is
the same mechanism the frozen M2 report describes as the crossover.

### 3.3 Aerial penalty

| method | penalty at 0.90 | penalty at 0.99 |
|---|---|---|
| heavy, k = 18 | 99.3 dB | 107.8 dB |
| heavy, k = 54 | 99.3 dB | 107.2 dB |
| digital, k = 54 | 104.5 dB | 109.6 dB |

The penalty is slightly above `PL_mean` (95.6 dB) because the 11% NLoS tail
needs much more power than the LoS majority; the mixture is dominated by the
tail at the 0.99 threshold.

## 4. What this supports - and what it does not

**Supports.** The lambda-shaped task-oriented policy remains usable on a
3GPP-parameterised aerial link, and its transmit-power advantage over the
classical link *grows* in the regime where the classical link is rate-limited:
21 dB at 0.90 and 15 dB at 0.99 for k = 54; unbounded at k = 18 because the
classical link never reaches 0.90.

**Does not support.** (i) Any claim about Doppler, shadowing or aerial
small-scale fading - not modelled. (ii) Any claim about other geometries,
heights or carriers - one geometry was run. (iii) Any claim about absolute
hardware feasibility - the 83-108 dB figures are relative to a unit-gain
benchmark and exclude antenna gain, noise figure and bandwidth. (iv) Any
seed-variance claim - one checkpoint per cell. (v) A general "neural beats
digital" claim: the k = 18 digital link is infeasible by payload, which is a
protocol fact, not a coding result.

## 5. Incidents (both fixed before the shipped numbers)

1. **Interpolation saturation.** The transmit-SNR mapping originally clipped
   interpolation beyond the measured grid to 1.0, which made a method with a
   0.17 ceiling appear to "reach" 0.90 at an arbitrary SNR. Fixed to constant
   extrapolation; the contaminated output was overwritten.
2. **Wrong digital rate at k = 54.** The first pass evaluated the digital
   comparator at B = 2 for both bandwidths; the frozen k = 54 comparator is
   B = 8. That made it look far worse than published (0.17 vs 0.72 at 0 dB).
   Fixed to `M2_B_BY_K[k]`; only the corrected numbers are shipped.

**Grid note.** The curve was measured in two passes over the *same*
deterministic stream: the first pass covered -20..+25 dB, the second added
-40..-25 dB with `--reuse` (the runner re-uses the stored points and only
evaluates the missing ones). Nothing else changed, and the k = 54 digital
point at 0 dB reproduces the published M2 value (0.717 vs 0.718), which
validates the rate mapping.

**Non-interference evidence.** The constructor change in A-9 does not touch
these numbers; re-evaluating one frozen Wave-C slot in memory reproduces the
stored JSON exactly (k = 18/awgn/seed 42/0 dB: SR 0.999 = 0.999, final
distance 0.0559941 vs 0.0559941).

## 6. Reproduce

```powershell
python tools/run_aerial_link.py --d2d=500 --h=50 --fc=2.0 --episodes=1000
# extend an existing curve without re-evaluating stored points:
python tools/run_aerial_link.py --d2d=500 --h=50 --fc=2.0 --episodes=1000 --reuse
python -m tests.test_aerial_link     # 26 checks: formulas + mapping arithmetic
```

## 7. Follow-up: training *under* the channel (A-12)

This study changes only the evaluation channel; the checkpoints were all
trained on the frozen Rayleigh model. A-12 (same date, `results/AERIAL_TRAIN/`)
retrains the λ-shaped arm **under** the per-episode 3GPP dispersion
(LoS/NLoS draw + TR 38.901 shadowing, block fading, no Doppler) and reports
the paired operating point against the numbers above. The two studies are
complementary: A-10 = "what does the channel cost a frozen policy", A-12 =
"can training on it recover the cost".

**Outcome (2026-10-27): A-12 ran as registered and both training cells
diverged** - the checkpoints are dead networks (0.023-0.029 and 0.013-0.025 at
every nominal SNR), so A-12 yields no trained operating point to compare
against. It does **not** change anything in sections 1-6 above: the numbers
here are produced by untouched Wave-C checkpoints on the same gain vector and
reproduce exactly. What A-12 adds is a limitation, documented with its cause
(effective-SNR spread 9.37 vs 5.59 dB; ~4 % of episodes ≥20 dB below the
training curriculum). See `RESULTS_AERIAL_TRAIN_2026-10-26.md`.
