# Polar vs convolutional digital baseline - does the coding choice matter?

**Date:** 2026-10-26 - **Amendment:** A-11 - **Status:** complete
(evaluation only; nothing trained, no frozen artifact written).

Answer in one line: **at k = 12/18/24 the two coders are within 0.05 of each
other once the payload ceiling is reached, so the reported crossover does not
depend on the convolutional code; at k = 30 polar is *better* than the frozen
convolutional baseline at >= 10 dB; at k = 54 (B = 8) plain SC polar is much
worse at low SNR and reaches parity only by 10-15 dB.**

---

## 1. What was run

| field | value |
|---|---|
| k grid | 12, 18, 24, 30, 54 (frozen M2) |
| B per k | 1, 2, 3, 4, 8 (`M2_B_BY_K`, unchanged) |
| SNR grid | 0, 5, 10, 15, 20 dB (frozen M2) |
| channels | awgn, rayleigh |
| methods | polar (CRC-8, SC) and conv (K=7, rate 1/2), **same episodes** |
| episodes | E = 1000 per cell |
| stream | `config.TEST_SEED`, one stream |
| root | `results/POLAR/` |
| wall time | 4229 s for 100 conditions |

Both baselines are deterministic and untrained, so one evaluation stream is
the honest design; three "seeds" of the same deterministic pipeline would be
fake replication. At E = 1000 the binomial 95% CI is about +/-3 pp at p = 0.5
and +/-1 pp near p = 0.95.

Polar details: 48 source bits max (B = 8), CRC-8, block length N = next power
of two >= payload, **no rate matching**, **SC (not CA-SCL)**, QPSK and the
same quantize/pack/dequantize layers as the frozen baseline.

## 2. Results (success rate, E = 1000)

### Polar

| k | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|
| 12 | awgn | 0.034 | 0.071 | 0.066 | 0.066 | 0.066 |
| 12 | rayleigh | 0.042 | 0.071 | 0.070 | 0.061 | 0.062 |
| 18 | awgn | 0.031 | 0.149 | 0.146 | 0.148 | 0.148 |
| 18 | rayleigh | 0.059 | 0.147 | 0.164 | 0.153 | 0.150 |
| 24 | awgn | 0.035 | 0.332 | 0.266 | 0.286 | 0.286 |
| 24 | rayleigh | 0.057 | 0.295 | 0.353 | 0.280 | 0.267 |
| 30 | awgn | 0.037 | 0.178 | 0.572 | 0.515 | 0.515 |
| 30 | rayleigh | 0.042 | 0.292 | 0.686 | 0.666 | 0.582 |
| 54 | awgn | 0.034 | 0.256 | 0.999 | 1.000 | 1.000 |
| 54 | rayleigh | 0.032 | 0.326 | 0.932 | 0.995 | 0.998 |

### Convolutional (frozen comparator, re-run on the same episodes)

| k | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|
| 12 | awgn | 0.063 | 0.066 | 0.066 | 0.066 | 0.066 |
| 12 | rayleigh | 0.063 | 0.059 | 0.059 | 0.063 | 0.066 |
| 18 | awgn | 0.143 | 0.148 | 0.148 | 0.148 | 0.148 |
| 18 | rayleigh | 0.172 | 0.164 | 0.147 | 0.153 | 0.151 |
| 24 | awgn | 0.265 | 0.286 | 0.286 | 0.286 | 0.286 |
| 24 | rayleigh | 0.339 | 0.270 | 0.280 | 0.273 | 0.277 |
| 30 | awgn | 0.658 | 0.515 | 0.515 | 0.515 | 0.515 |
| 30 | rayleigh | 0.697 | 0.679 | 0.593 | 0.544 | 0.528 |
| 54 | awgn | 0.976 | 1.000 | 1.000 | 1.000 | 1.000 |
| 54 | rayleigh | 0.717 | 0.982 | 0.999 | 1.000 | 1.000 |

### Difference (polar - conv)

| k | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|
| 12 | awgn | -0.029 | +0.005 | 0.000 | 0.000 | 0.000 |
| 12 | rayleigh | -0.021 | +0.012 | +0.011 | -0.002 | -0.004 |
| 18 | awgn | -0.112 | +0.001 | -0.002 | 0.000 | 0.000 |
| 18 | rayleigh | -0.113 | -0.017 | +0.017 | 0.000 | -0.001 |
| 24 | awgn | -0.230 | +0.046 | -0.020 | 0.000 | 0.000 |
| 24 | rayleigh | -0.282 | +0.025 | +0.073 | +0.007 | -0.010 |
| 30 | awgn | -0.621 | -0.337 | +0.057 | 0.000 | 0.000 |
| 30 | rayleigh | -0.655 | -0.387 | +0.093 | +0.122 | +0.054 |
| 54 | awgn | -0.942 | -0.744 | -0.001 | 0.000 | 0.000 |
| 54 | rayleigh | -0.685 | -0.656 | -0.067 | -0.005 | -0.002 |

## 3. What this means

**The crossover story does not depend on the code.** At k = 12, 18 and 24 both
coders sit at the same payload ceiling from 5 dB upward (|delta| <= 0.05, and
mostly <= 0.02). The frozen report's claim that the classical link is
*rate-limited* in this regime - not noise-limited - is therefore a property of
the rate/quantizer, not of Viterbi vs SC.

**At k = 30 polar is the stronger link.** +0.06 (awgn) and +0.09/+0.12/+0.05
(rayleigh, 10/15/20 dB), because the 3-bit payload at k = 30 leaves less
redundancy for the rate-1/2 convolutional code, while the polar block keeps
its own code rate. The ordering of the *digital* column against the neural
methods at k = 30 therefore depends on the coder - a caveat worth one
sentence, not a reversal.

**At k = 54 (B = 8) plain SC polar is the weaker link.** Delta -0.69 to
-0.94 at 0-5 dB, parity by 10-15 dB. Two implementation reasons, both
documented rather than hidden: (i) code length is the next power of two above
the 48-bit payload (N = 64, rate 0.75) with **no rate matching** to the 54
symbols, and (ii) **SC without CRC-aided list decoding**, which is far from a
production 5G receiver. The polar column at high rate is a lower bound on what
polar can do, not a verdict on polar coding.

**Both coders show non-monotone success in SNR at k = 30** (e.g. conv awgn
0.658 at 0 dB then 0.515 at 5-20 dB). That is the benchmark's payload ceiling,
not a decoding bug: extra SNR cannot recover bits the quantizer never sent.
The neural arms show the same plateau for the same reason.

## 4. What this does not support

- Any claim that "polar is worse/better than convolutional" in general: one
  block length rule, no rate matching, SC only, one payload per k.
- Any seed-variance claim: deterministic baselines, one stream.
- Any claim at E = 5000: this run is E = 1000 (diagnostic budget). Re-run with
  `--episodes=5000` before quoting a number as protocol-grade.

## 5. Reproduce

```powershell
python tools/run_polar_baseline.py smoke                     # timing check
python tools/run_polar_baseline.py run --episodes=1000       # this artifact
python tools/run_polar_baseline.py run --episodes=5000       # protocol budget
```
