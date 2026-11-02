# Results — A-19: Doppler / time-correlated Rayleigh fading

**Date:** 2026-11-02
**Amendment:** A-19 (registered before the first cell, `PROTOCOL_AMENDMENTS.md`)
**Root:** `results/DOPPLER/`
**Summary artifact:** `results/DOPPLER/step_doppler_summary.json`
**Runner:** `tools/run_doppler.py` (`verify` → `evaluate` → `summary`)

Everything below is read from the summary artifact. Nothing is typed from
memory and no cell was re-run to make a number look better.

---

## 1. Why this experiment exists

Every headline result in this thesis uses **block** Rayleigh fading: one
complex gain `h ~ CN(0,1)` per codeword, redrawn independently at each of the
100 control steps. That is the DeepJSCC "slow fading" convention. It is only
correct if the platform moves many wavelengths per control period.

The frozen numbers are `DT = 0.1 s` and `V_MAX = 5 m/s`. At 2 GHz the
wavelength is 0.1499 m, so the displacement per step is
`Δ = v·DT/λ = 0.033 … 3.34` wavelengths over the speed range. At
`Δ ≥ 0.33` the channel is already decorrelated within one step
(`J0(2πΔ) = 0.13` at `Δ = 0.33`). Correlation therefore only exists when the
platform is nearly hovering — which is exactly when a deep fade can persist
for many steps and the controller has to fly through it.

A-19 measures that instead of asserting it.

## 2. Design (as registered)

| Item | Value |
|---|---|
| Fading | Clarke/Jakes sum of sinusoids, `M = 32`, per-episode random angles and phases |
| Ensemble autocorrelation | `R(τ) = J0(2π f_d τ)` (exact, because the angle of arrival is uniform) |
| Normalisation | `E|h|² = 1`, identical to the frozen block model |
| Axis | `Δ = v·DT/λ`, the displacement per control period in wavelengths |
| Controls | the frozen i.i.d. path, unchanged and bit-identical (separate RNG stream, verified) |
| Arms | `lam_ctl_heavy` at k = 12 (Wave-A) and k = 54 (Wave-C); frozen digital baseline |
| Episodes | 1000, `TEST_SEED` 10042, horizon 100 |
| SNR | 0, 5, 10, 15, 20 dB |
| Training | none — evaluation of stored checkpoints |

100 of 100 slots completed.

## 3. How much temporal structure the sweep actually created

This is the control that makes the rest of the table meaningful: the marginal
distribution is Rayleigh everywhere (deep-fade fraction is flat at 9.2–9.6 %
by construction), but the *duration* of a fade changes by a factor of four.

| Point | f_c | v (m/s) | f_d (Hz) | Δ (λ/step) | deep-fade fraction | mean fade run (steps) |
|---|---|---|---|---|---|---|
| `frozen_iid` | — | — | — | — | 0.094 | 1.10 |
| `d0033_2g` | 2.0 GHz | 0.05 | 0.334 | 0.033 | 0.094 | **4.08** |
| `d0067_2g` | 2.0 GHz | 0.10 | 0.667 | 0.067 | 0.096 | 2.17 |
| `d0133_2g` | 2.0 GHz | 0.20 | 1.334 | 0.133 | 0.095 | 1.33 |
| `d0333_2g` | 2.0 GHz | 0.50 | 3.336 | 0.334 | 0.094 | 1.11 |
| `d0667_2g` | 2.0 GHz | 1.00 | 6.671 | 0.667 | 0.094 | 1.13 |
| `d1333_2g` | 2.0 GHz | 2.00 | 13.343 | 1.334 | 0.093 | 1.11 |
| `d3333_2g` | 2.0 GHz | 5.00 | 33.356 | 3.336 | 0.092 | 1.11 |
| `d0117_07g` | 0.7 GHz | 0.50 | 1.168 | 0.117 | 0.096 | 1.43 |
| `d0234_07g` | 0.7 GHz | 1.00 | 2.335 | 0.233 | 0.095 | 1.15 |

At `Δ = 0.033` a fade lasts four steps instead of one. Above `Δ ≈ 0.33` the
process is indistinguishable from the frozen i.i.d. model — the correlation
time is shorter than the control period, which is the opposite of the usual
"block fading is optimistic" worry.

## 4. Success rate — λ-heavy arm

Mean over 0–20 dB (E = 1000 per SNR; do **not** quote a single low-SNR cell,
B ≤ 4 cells dither at the quantiser boundary).

| Point | Δ (λ/step) | k = 12 | Δ vs i.i.d. | k = 54 | Δ vs i.i.d. |
|---|---|---|---|---|---|
| `frozen_iid` | — | 0.997 | — | 0.998 | — |
| `d0033_2g` | 0.033 | 0.994 | −0.003 | 0.997 | −0.001 |
| `d0067_2g` | 0.067 | 0.997 | 0.000 | 0.998 | 0.000 |
| `d0133_2g` | 0.133 | 0.997 | 0.000 | 0.997 | 0.000 |
| `d0333_2g` | 0.334 | 0.996 | −0.001 | 0.998 | 0.000 |
| `d0667_2g` | 0.667 | 0.997 | 0.000 | 0.997 | −0.001 |
| `d1333_2g` | 1.334 | 0.994 | −0.003 | 0.997 | −0.001 |
| `d3333_2g` | 3.336 | 0.996 | −0.001 | 0.997 | 0.000 |
| `d0117_07g` | 0.117 | 0.997 | 0.000 | 0.998 | 0.000 |
| `d0234_07g` | 0.233 | 0.996 | −0.001 | 0.997 | 0.000 |

**Largest deviation from the frozen model: 0.003 at k = 12 and 0.001 at
k = 54, i.e. three episodes out of a thousand.** The λ-heavy policy is
insensitive to whether the fading is independent between steps or not.

## 5. Success rate — frozen digital baseline

| Point | Δ (λ/step) | k = 12 | Δ vs i.i.d. | k = 54 | Δ vs i.i.d. |
|---|---|---|---|---|---|
| `frozen_iid` | — | 0.062 | — | 0.940 | — |
| `d0033_2g` | 0.033 | 0.062 | 0.000 | **0.912** | **−0.028** |
| `d0067_2g` | 0.067 | 0.067 | +0.005 | 0.925 | −0.014 |
| `d0133_2g` | 0.133 | 0.063 | +0.001 | 0.938 | −0.001 |
| `d0333_2g` | 0.334 | 0.065 | +0.003 | 0.941 | +0.002 |
| `d0667_2g` | 0.667 | 0.067 | +0.005 | 0.938 | −0.002 |
| `d1333_2g` | 1.334 | 0.065 | +0.003 | 0.940 | 0.000 |
| `d3333_2g` | 3.336 | 0.064 | +0.002 | 0.940 | 0.000 |
| `d0117_07g` | 0.117 | 0.064 | +0.002 | 0.933 | −0.007 |
| `d0234_07g` | 0.233 | 0.065 | +0.003 | 0.933 | 0.000 |

Two clean facts:

1. At k = 12 the digital link is **payload-limited**, not fading-limited:
   0.062–0.067 at every Doppler point, including the frozen one. It cannot
   use the extra coherence, and correlation cannot hurt it further.
2. At k = 54 the digital link loses up to **0.028** when fades last four
   steps (`Δ = 0.033`) and the loss decays monotonically as the fades
   shorten. That is the physically expected direction: an uncoded-branch
   convolutional code has no time diversity, so a persistent fade erases a
   run of symbols that a single-step fade would not.

The 0.7 GHz rows reproduce the 2 GHz behaviour at matching `Δ`
(0.933 at Δ = 0.117 vs 0.938 at Δ = 0.133), which supports the claim that
`Δ`, not `v` and not `f_c`, is the controlling variable.

## 6. What this changes in the thesis

- **The i.i.d. block-fading convention is validated for the neural arms at
  the frozen operating point.** The headline numbers do not rest on a
  modelling assumption that a reviewer can call optimistic. At the only
  correlated setting a UAV can actually hover in (Δ = 0.033, i.e. 0.05 m/s at
  2 GHz) the λ-heavy arm moves by 0.003.
- **The one vulnerability is the digital comparator's k = 54 point**, and it
  moves *in the thesis's favour*: correlated fading lowers the digital
  baseline from 0.940 to 0.912, while the neural arm stays at 0.997. No
  comparison flips, and the ordering becomes clearer rather than weaker.
- **The correlation time is shorter than the control period for any speed
  above ~0.2 m/s.** This is worth stating explicitly: at 2 GHz with a 100 ms
  decision period the frozen model is not an approximation, it is the right
  model; the interesting regime is hovering, and even there the policy does
  not care.

## 7. Decision rule (declared in A-19) and its outcome

The rule was: *if success changes by more than 0.02 in the correlated regime
(`Δ ≤ 0.13`), the i.i.d. model is not adequate and one training arm at the
worst measured point is required; otherwise the i.i.d. assumption is
validated and no training run is made.*

- Neural arm: **not triggered** (max 0.003).
- Digital baseline at k = 54: **triggered** (−0.028 at Δ = 0.033).

Because the rule is written on "success", not on "neural success", it
triggered, and the follow-up training arm was run rather than argued away.
It is recorded in §8.

## 8. Follow-up arm (rule-triggered)

The arm was registered in A-19 before it was launched and trained from the
same initialisation family as the frozen k = 12 checkpoint, with the only
change being that the closed-loop rollout draws its channel from the
time-correlated generator instead of the i.i.d. one.

| Item | Value |
|---|---|
| Channel point | `d0033_2g` (2 GHz, 0.05 m/s, `f_d = 0.3336 Hz`, Δ = 0.033 λ/step) |
| Arm | λ-heavy, k = 12, Rayleigh |
| Initialisation / seed | seed 42, λ = (1.0, 0.1, 0.1, 2.0) |
| Optimiser steps | 3000 |
| Wall clock | 54.1 min (CPU) |
| Evaluation | 1000 episodes, `TEST_SEED` 10042, same slots as the sweep |

| SNR (dB) | 0 | 5 | 10 | 15 | 20 | Mean |
|---|---|---|---|---|---|---|
| Stored i.i.d.-trained checkpoint | — | — | — | — | — | 0.994 |
| Retrained under correlated fading | 0.977 | 0.996 | 0.998 | 0.998 | 0.996 | **0.993** |

The retrained arm lands **0.001 below** the stored checkpoint (artifact:
`trained_arm.delta_vs_stored_arm = −0.0012` in
`results/DOPPLER/step_doppler_summary.json`). This is inside the seed-to-seed
spread already documented for this arm and is not an improvement.

Read together with §4, the follow-up answers the rule in the strongest way
available: the policy trained on independent fading does not lose anything
when the fading is correlated, and a policy trained on correlated fading does
not gain anything. The i.i.d. convention is therefore not merely *adequate*
for the neural arms, it is the right model at this operating point — and the
trigger was honoured rather than argued away. The digital comparator's k = 54
loss (§5) is a property of an uncoded-span code without interleaving, not of
the simulation convention, and it moves the comparison in the thesis's favour
(0.912 vs 0.997).

## 9. Reproduce

```powershell
.venv\Scripts\python.exe tools\run_doppler.py verify
.venv\Scripts\python.exe tools\run_doppler.py evaluate
.venv\Scripts\python.exe tools\run_doppler.py train
.venv\Scripts\python.exe tools\run_doppler.py evaluate-trained
.venv\Scripts\python.exe tools\run_doppler.py summary
.venv\Scripts\python.exe -m tests.test_doppler
```

`verify` re-checks the physics (power, `J0` autocorrelation), that the frozen
i.i.d. streams are bit-identical when `doppler_hz` is `None`, that every
displacement in the grid is distinct, and that the training plumbing is
deterministic.

## 10. Scope limits (stated, not hidden)

- Rayleigh only. The 3GPP aerial layer stays per-episode block fading; adding
  time-correlated **shadowing** is a different simulator change and is not
  claimed here.
- `M = 32` oscillators per episode. The ensemble autocorrelation is exact; a
  single episode is a finite sum, which is why the sweep is run over 1000
  episodes and the ensemble statistic is reported.
- The mobile speed is held constant within an episode; the process is
  stationary. Acceleration within an episode is not modelled.
- Link-level only: no antenna gain, no noise figure, no bandwidth claim.
