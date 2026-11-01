# M5b and M5c results — uncoded digital reference and cross-channel transfer

**Date:** 2026-11-01 · **Amendment:** A-16 (registered before the first cell) ·
**Artifacts:** `results/M5/uncoded/`, `results/M5/cross/` ·
**Runner:** `tools/run_m5.py uncoded`, `tools/run_m5.py cross` ·
**Status:** M5b 100/100 slots, M5c 10/10 slots · **Type:** evaluation only, nothing trained

All numbers below were read from the stored JSON; none were typed from memory.
Evaluation protocol: E = 1000 per cell, SNR grid 0/5/10/15/20 dB, `TEST_SEED`
= 10042, paired realizations shared across methods.

---

## 1. Why this run exists

The supervisor review made two objections that A-11/A-15 (polar, 5G NR polar)
only partly answer:

1. *"Your convolutional code is outdated. Reviewers will dismiss the digital
   baseline as a strawman."*
2. *"Is the crossover an artefact of the coding choice?"*

M5b removes the channel code **entirely**. Two variants are reported per
`(k, channel)` so that "more payload bits" and "no coding" are separable:

| Variant | Payload | Meaning |
|---|---|---|
| `uncoded_bmax` | `B = floor(k/3)` | whole bandwidth, no code |
| `uncoded_bcoded` | `B = floor((k−6)/6)` | same payload as the coded baseline, code removed |

---

## 2. M5b — uncoded QPSK, mean success over the 0–20 dB grid

### AWGN

| k | reconstruction (M2) | task, frozen λ (M2) | digital, conv (M2) | **uncoded, B = k/3** | uncoded, same-B |
|---|---|---|---|---|---|
| 12 | 0.242 | 0.028 | 0.062 | **0.526** (B = 4) | 0.064 (B = 1) |
| 18 | 0.254 | 0.018 | 0.164 | **0.880** (B = 6) | 0.149 (B = 2) |
| 24 | 0.233 | 0.034 | 0.306 | **0.879** (B = 8) | 0.297 (B = 3) |
| 30 | 0.279 | 0.021 | 0.566 | **0.885** (B = 10) | 0.528 (B = 4) |
| 54 | 0.246 | 0.016 | 0.995 | 0.884 (B = 18) | 0.881 (B = 8) |

### Rayleigh

| k | reconstruction (M2) | task, frozen λ (M2) | digital, conv (M2) | **uncoded, B = k/3** | uncoded, same-B |
|---|---|---|---|---|---|
| 12 | 0.253 | 0.053 | 0.064 | **0.581** (B = 4) | 0.068 (B = 1) |
| 18 | 0.236 | 0.025 | 0.165 | **0.853** (B = 6) | 0.156 (B = 2) |
| 24 | 0.233 | 0.036 | 0.322 | **0.854** (B = 8) | 0.289 (B = 3) |
| 30 | 0.264 | 0.021 | 0.614 | **0.861** (B = 10) | 0.586 (B = 4) |
| 54 | 0.258 | 0.030 | 0.941 | 0.867 (B = 18) | 0.864 (B = 8) |

M2 columns are means over three seeds; M5b columns are one evaluation stream
(the link is deterministic and untrained, so extra "seeds" would be fake
replication).

---

## 3. Four findings

### Finding 1 — the low-bandwidth digital ceiling is a **payload** artefact, not a separation artefact

At k = 12 the frozen coded baseline carries `B = 1` (because `6B + 6 ≤ 12`)
and reaches 0.062. The **same channel uses, uncoded**, carry `B = 4` and reach
**0.526** on AWGN (0.581 Rayleigh) — an 8× improvement with no coding at all.

So M3's "digital is infeasible at k = 1, 2, 3" and the k = 12 payload ceiling
are statements about the **frozen coded framing**, not about digital
communication per se. This must be said in exactly those words.

### Finding 2 — the plateau is set by `B` alone, which validates the implementation

| Payload | k = 12 | k = 30 | plateau |
|---|---|---|---|
| `B = 4` | 0.526 | 0.528 | **0.52** |
| `B ≥ 6` | — | 0.885 (B = 10) | **0.88** |

The same `B = 4` payload gives the same success rate at k = 12 and k = 30,
and `B ≥ 6` saturates at 0.88. Success is therefore **quantization-limited**
at fixed payload, independent of `k`. This is a strong internal consistency
check: two different `k` with the same payload land on the same number.

### Finding 3 — the non-monotone curves are real, and the frozen baseline has them too

At `B = 4` success is *higher* at 5 dB than at 20 dB (0.698 → 0.515 on AWGN
k = 12). This is not a bug: at high SNR the estimate collapses to the
deterministic quantized state, whose residual error (one quantizer step
≈ 1.07 m in position for `B = 4`) exceeds the 0.5 m goal radius, so the
controller cannot settle. At moderate SNR, bit errors **dither** the estimate
and occasionally land it inside the goal radius.

The frozen coded baseline shows the identical pattern at k = 30 Rayleigh
(0.686 at 0 dB → 0.551 at 20 dB, stored in `results/M2`). The effect is a
property of the frozen quantizer + controller, not of the uncoded link.

**Consequence for reporting:** never quote a single-SNR number for a `B ≤ 4`
cell. Quote the grid, and say "quantization-limited with noise dithering".

### Finding 4 — the crossover depends on which digital comparator you use

| Comparator | Where the classical link overtakes the λ-heavy neural arm |
|---|---|
| Frozen convolutional, `B = floor((k−6)/6)` | k = 54 (0.995 vs 0.998 — effectively a tie) |
| 5G NR polar (A-15) | k = 54 (0.909 vs 0.998 — neural still ahead) |
| **Uncoded, `B = k/3`** | **not before k = 54** (0.884 vs 0.998) |

Measured λ-heavy points: k = 12 → 0.997 (Wave-A), k = 18 → 0.999 (Wave-C),
k = 24 → 0.995–1.000 (A-14), k = 54 → 0.998 (Wave-C). **k = 30 was never
trained for the λ arm**, so no comparison is claimed there.

Against the uncoded link the neural arm is ahead at k = 12 (0.997 vs 0.526),
k = 18 (0.999 vs 0.880) and k = 24 (≈0.999 vs 0.879), and ties at k = 54
(0.998 vs 0.884 uncoded, 0.995 coded).

The thesis's headline "digital wins at k ≥ 24" is true **for the frozen coded
baseline**. Against a payload-efficient uncoded link, no digital variant beats
the λ-heavy arm below k = 54. Both statements are correct; the comparator must
travel with the sentence.

---

## 4. M5c — cross-channel transfer (evaluation only)

Wave-C λ-heavy, k = 18, seed 42, trained on one channel, evaluated on the other.

| Trained on | Evaluated on | SR @ 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|
| AWGN | **Rayleigh** | 0.991 | 0.997 | 0.999 | 0.999 | 0.999 |
| Rayleigh | **AWGN** | 0.999 | 0.998 | 0.998 | 0.997 | 0.997 |

**Reading.** Transfer is complete — but the metric is **saturated**, so this
is not evidence that the learned representation is channel-agnostic. At
k = 18 the λ-heavy arm is at the task ceiling on both channels, so
cross-channel degradation cannot be resolved. The honest claim is: *no
measurable degradation at this operating point; the experiment cannot
distinguish transfer from ceiling.*

**What would make it informative:** repeat at k = 12, where the λ-heavy arm is
0.997 AWGN / 0.997 Rayleigh (Wave-A) — still at ceiling. A harder task or a
lower-bandwidth point is needed. This is recorded as a limitation, not
papered over.

---

## 5. What this changes in the thesis

1. **§5.9 must gain the uncoded row.** The current text says the crossover
   "survives the strongest classical comparator". It does — but the strongest
   comparator is the *uncoded full-bandwidth* link, not the coded one, and it
   moves the crossover to k = 54. Add the M5b table and one sentence.
2. **§5.2.1 (M3) needs a qualifier.** "Digital infeasible at k = 1, 2, 3" is
   about the frozen coded framing. The uncoded link at k = 12 is feasible and
   reaches 0.53.
3. **§7 (Limitations) gains one line:** the digital comparator family is
   payload-limited at low k by construction; the uncoded reference is the
   payload-maximal bound.
4. **Do not claim** that the neural arm beats digital at every bandwidth. At
   k = 54 the coded link ties (0.995 vs 0.998), and the λ arm was never
   trained at k = 30.

---

## 6. Provenance

| Item | Path |
|---|---|
| M5b eval JSONs | `results/M5/uncoded/<variant>/k<k>_<channel>/eval_<snr>.json` |
| M5c eval JSONs | `results/M5/cross/train_<ch>_eval_<ch>/eval_<snr>.json` |
| Runner | `tools/run_m5.py` (`uncoded`, `cross`) |
| Amendment | `fyp_md_files/PROTOCOL_AMENDMENTS.md` A-16 |
| M2 comparison columns | `results/M2/eval/*/eval_*.json` (E = 5000, 3 seeds) |

M5a (λ term removal) is still training; it will be reported separately and
must not be mixed into this note.
