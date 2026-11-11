# One-Page Project Summary

**Title:** Task-Oriented Deep Joint Source-Channel Coding for UAV Navigation under Noisy and Fading Wireless Channels — with a 3GPP UMa-Aerial Link-Budget and Dispersion Extension

**Paper title:** When Does Task-Oriented DeepJSCC Beat a Coded Digital Link? A Matched-Budget Study for UAV Navigation

**Authors:** Areesha Sohail (1005), Amna Shahid (1004), Hafiza Laiba Faisal (1015)
**Supervisor:** Ismat Hira — Department of Electrical Engineering, LCWU

---

## Problem

A UAV sends its 6-D navigation state (position, velocity, goal) over a noisy, fading link.
Conventional and reconstruction-oriented links optimise a generic distortion measure that does **not**
correspond to what matters operationally: reaching the goal. Under a fixed bandwidth budget, spending
channel uses on every state component with equal fidelity may be the wrong allocation.

## Method

Three links, **identical episodes, noise, fading and channel uses** (matched budget):

| Arm | Sends | Trained on |
|---|---|---|
| Task-oriented DeepJSCC | learned 6 → 2k symbols | closed-loop navigation success |
| Reconstruction DeepJSCC | learned 6 → 2k symbols | state reconstruction MSE |
| Digital baseline | quantise → K=7 conv. code → QPSK | not trained; `6B + 6 ≤ k` enforced |

Evaluated at k = 12–54, SNR = 0–20 dB, three training seeds, 5,000 paired episodes per unit.
Statistics: McNemar test and Wilson intervals (no formal equivalence test is claimed).
Aerial extension: 3GPP TR 36.777 / TR 38.901 UMa-aerial **large-scale layer**.

## Three results

1. **Bandwidth crossover.** Reconstruction arm: neural wins at k ≤ 18 (0.242 vs 0.062 at k = 12);
   coded link wins at k ≥ 24 (0.306 vs 0.233; 0.995 vs 0.246 at k = 54).
   The shaped task arm sits at the task ceiling (0.996–0.999) at every k, so it cannot be ranked there.

2. **Task loss was the failure, not capacity.** The frozen loss reaches only 2–6% success.
   Adding control and velocity penalties (λ = 0.1) recovers 0.997. The median encoder gradient
   falls by 7–10 orders of magnitude; a width-256 model still fails when unshaped.

3. **Aerial link.** At the pinned geometry (500 m, 50 m, 2 GHz) and k = 54, the shaped policy
   requires 87 dB of normalised transmit power for 0.90 success and 102 dB for 0.99;
   the coded link requires 108 dB and 117 dB. Across 12 geometries the gap is 15–21 dB at 0.90
   and 12–15 dB at 0.99. Training under the full dispersion fails (documented negative result);
   tail truncation restores stability without a measurable gain.

## Limitations (stated, not hidden)

Simulation-only; 2-D kinematics; fixed controller; 3GPP layer is large-scale only (no Doppler,
no correlated shadowing, no aerial small-scale fading); the 12-geometry sweep remaps one shared
curve rather than running independent channels; the digital comparator is fixed-rate (no adaptive
MCS/HARQ); extension blocks use a single seed. Figures are normalised link-budget dB, **not EIRP**.

## One-sentence claim

Under a matched channel-use and bandwidth budget, task-oriented DeepJSCC is the better use of a
constrained UAV link at low bandwidth and low SNR, its advantage survives a 3GPP-inspired aerial
link budget at k = 54, and the result is conditional — at high bandwidth the benchmark saturates.
