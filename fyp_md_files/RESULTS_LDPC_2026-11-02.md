# Results — A-20: 5G NR LDPC (BG2) comparator

**Date:** 2026-11-02
**Amendment:** A-20 (registered before the first cell, `PROTOCOL_AMENDMENTS.md`)
**Root:** `results/LDPC/`
**Summary artifact:** `results/LDPC/step_ldpc_summary.json`
**Runner:** `tools/run_ldpc.py` (`verify` → `evaluate` → `summary`)

Every number below is read from the summary artifact or from the per-slot JSON
files. Nothing is typed from memory, and no cell was re-run to move a number.

---

## 1. Why this experiment exists

A-11 answers "is the convolutional baseline weak?" with CRC-8 polar, and A-15
answers it with the full 5G NR polar chain. A reviewer can still ask why the
*other* 3GPP code family is missing. A-20 adds **5G NR LDPC base graph 2** to
the comparison, on the same episodes, the same quantiser, the same QPSK layer
and the same `E = 2k` budget. Nothing is retrained; the frozen policy cells are
untouched.

## 2. What was run (and what was not)

| Item | Value |
|---|---|
| Code | 5G NR LDPC, base graph 2, TS 38.212 Table 5.3.2-3 (197 non-zero entries, vendored, not retyped) |
| Lifting | Table 5.3.2-1 sets, `Kb = 10`, `Z` = smallest valid value with `10Z ≥ A` |
| Encoder | systematic, GF(2) solve of the parity block |
| Decoder | offset min-sum, 15 iterations, offset 0.5 |
| Filler bits | NULL, not transmitted (as in 38.212) |
| Budget | `A = 6B`, `E = 2k`, `B` from `DIGITAL_B_BY_K` (1, 2, 3, 4, 8) |
| Transmission | first `E` non-NULL mother bits (systematic payload, then parity) |
| Channels | AWGN, Rayleigh |
| SNR | 0, 5, 10, 15, 20 dB |
| Episodes | `E = 1000` per cell, `TEST_SEED` 10042 |
| Frozen comparators | the convolutional baseline in the same run, on the same episodes |

**Not implemented, and labelled as such everywhere:** the 38.212 rate-matching
chain (circular buffer, bit interleaving, redundancy version), the
transport-block CRC, and LDPC base-graph selection. The arm is therefore
"5G NR LDPC BG2 without the 38.212 rate-matching chain", never "full 5G NR
LDPC". This is the same honesty level as A-11 ("CRC-8 polar, SC, no rate
matching").

One consequence deserves to be stated in the results rather than in a
footnote: at k ≤ 24 the block is far below the code's design point. `Z = 2`
gives `N = 104` mother bits and `K = 20` information bits at k = 12, against a
payload of 6 bits. A poor LDPC number at low k is a statement about block
length and lifting size, not about LDPC as a family — the comparison is still
the right one because the frozen protocol fixes `k`, but the reading must not
be over-generalised.

## 3. Pre-flight gates

`tools/run_ldpc.py verify` passes before any evaluation slot:

- A-20 is registered in `PROTOCOL_AMENDMENTS.md`.
- Lifting sizes match Table 5.3.2-1 for every k.
- `H c^T = 0` for every encoded codeword, and `rank(H) = N − K`.
- Noiseless round-trip recovers the payload for every k.
- AWGN BER is monotone in SNR, and the k = 54 BER at 10 dB is below 10⁻².
- The link encodes `k` QPSK symbols and round-trips at bit level (k = 54, B = 8).
- The output root is isolated under `results/`.

## 4. Results

Mean over 0–20 dB is the headline; per-SNR columns are shown because the low-k
floor is flat and the k = 54 curve is not. `E = 1000` per cell gives a binomial
95 % interval of about ±3 pp at p = 0.5 and ±1 pp near p = 0.95, so differences
below ~0.01 at mid-range and ~0.005 near the ceiling are not resolvable.

### LDPC (BG2, no rate-matching chain)

| k | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | Mean |
|---|---|---|---|---|---|---|---|
| 12 | awgn | 0.064 | 0.066 | 0.066 | 0.066 | 0.066 | 0.0656 |
| 12 | rayleigh | 0.067 | 0.067 | 0.060 | 0.063 | 0.066 | 0.0646 |
| 18 | awgn | 0.165 | 0.150 | 0.148 | 0.148 | 0.148 | 0.1518 |
| 18 | rayleigh | 0.155 | 0.155 | 0.144 | 0.156 | 0.151 | 0.1522 |
| 24 | awgn | 0.313 | 0.284 | 0.286 | 0.286 | 0.286 | 0.2910 |
| 24 | rayleigh | 0.348 | 0.305 | 0.271 | 0.262 | 0.271 | 0.2914 |
| 30 | awgn | 0.717 | 0.511 | 0.515 | 0.515 | 0.515 | 0.5546 |
| 30 | rayleigh | 0.651 | 0.704 | 0.628 | 0.554 | 0.540 | 0.6154 |
| 54 | awgn | 0.890 | 1.000 | 1.000 | 1.000 | 1.000 | 0.9780 |
| 54 | rayleigh | 0.732 | 0.991 | 0.997 | 0.999 | 1.000 | 0.9438 |

### Convolutional baseline (A-11 re-run, identical episodes)

| k | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | Mean |
|---|---|---|---|---|---|---|---|
| 12 | awgn | 0.063 | 0.066 | 0.066 | 0.066 | 0.066 | 0.0654 |
| 12 | rayleigh | 0.063 | 0.059 | 0.059 | 0.063 | 0.066 | 0.0620 |
| 18 | awgn | 0.143 | 0.148 | 0.148 | 0.148 | 0.148 | 0.1470 |
| 18 | rayleigh | 0.172 | 0.164 | 0.147 | 0.153 | 0.151 | 0.1574 |
| 24 | awgn | 0.265 | 0.286 | 0.286 | 0.286 | 0.286 | 0.2818 |
| 24 | rayleigh | 0.339 | 0.270 | 0.280 | 0.273 | 0.277 | 0.2878 |
| 30 | awgn | 0.658 | 0.515 | 0.515 | 0.515 | 0.515 | 0.5436 |
| 30 | rayleigh | 0.697 | 0.679 | 0.593 | 0.544 | 0.528 | 0.6082 |
| 54 | awgn | 0.976 | 1.000 | 1.000 | 1.000 | 1.000 | 0.9952 |
| 54 | rayleigh | 0.717 | 0.982 | 0.999 | 1.000 | 1.000 | 0.9396 |

### Paired difference

| k | B | A bits | Z | N | Channel | LDPC | Conv | Δ (LDPC − Conv) |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 12 | 1 | 6 | 2 | 104 | awgn | 0.0656 | 0.0654 | +0.0002 |
| 12 | 1 | 6 | 2 | 104 | rayleigh | 0.0646 | 0.0620 | +0.0026 |
| 18 | 2 | 12 | 2 | 104 | awgn | 0.1518 | 0.1470 | +0.0048 |
| 18 | 2 | 12 | 2 | 104 | rayleigh | 0.1522 | 0.1574 | −0.0052 |
| 24 | 3 | 18 | 2 | 104 | awgn | 0.2910 | 0.2818 | +0.0092 |
| 24 | 3 | 18 | 2 | 104 | rayleigh | 0.2914 | 0.2878 | +0.0036 |
| 30 | 4 | 24 | 3 | 156 | awgn | 0.5546 | 0.5436 | +0.0110 |
| 30 | 4 | 24 | 3 | 156 | rayleigh | 0.6154 | 0.6082 | +0.0072 |
| 54 | 8 | 48 | 5 | 260 | awgn | 0.9780 | 0.9952 | −0.0172 |
| 54 | 8 | 48 | 5 | 260 | rayleigh | 0.9438 | 0.9396 | +0.0042 |

## 5. What this changes in the thesis

1. **No ordering changes.** At every `k`, LDPC and convolutional are within
   0.011 on AWGN and 0.007 on Rayleigh, except the k = 54 AWGN cell where LDPC
   is 0.017 *below* the convolutional baseline. The crossover claim (neural wins
   at k ≤ 18, coded digital at k ≥ 24) is untouched: at k = 24 both coders sit
   near 0.29, far below the λ-heavy arm's 0.999.
2. **The low-k floor is a framing property, not a code property.** At k = 12,
   18 and 24 the two code families are statistically indistinguishable
   (0.065, 0.152, 0.291). The lifting size is `Z = 2` (`N = 104`, `K = 20`),
   far below any 5G LDPC design point, and the payload is 6, 12 and 18 bits.
   No code can lift that floor, which is the same conclusion as M5b, reached
   from the code-family side.
3. **The k = 54 AWGN gap is explained by puncturing, not by LDPC.** The BG2
   mother code at `Z = 5` is `K/N = 50/260 ≈ 0.19`, and the frozen budget sends
   only the first 108 bits (48 systematic + 60 parity). The effective rate of
   0.44 is reached by *puncturing* the mother code, which costs belief
   propagation at 0 dB; the convolutional baseline at that k has rate
   `54/108 = 0.5` with a short constraint length and is stronger exactly where
   the channel is worst. At Rayleigh the two are equal (0.944 against 0.940),
   so the gap is a low-SNR artifact of the missing rate-matching chain.
4. **The non-monotone k = 30 curves reproduce a frozen effect.** LDPC goes
   0.717 → 0.515 between 0 and 5 dB, and the convolutional baseline goes
   0.658 → 0.515. This is the quantiser dithering already documented in M5b
   (B = 4, one step ≈ 1.07 m in position, goal radius 0.5 m), not a decoder
   defect: at high SNR the estimate collapses to the deterministic quantised
   state and cannot settle inside the goal radius.
5. **The defence of the digital baseline is now complete on both code
   families.** A-11 (SC polar), A-15 (full 5G NR polar), A-20 (5G NR LDPC BG2)
   and M5b (uncoded, payload-maximal) all agree: the crossover is a property of
   the frozen coded framing and the objective, not of one weak code.

## 6. Reproduce

```powershell
.venv\Scripts\python.exe tools\run_ldpc.py verify
.venv\Scripts\python.exe tools\run_ldpc.py evaluate
.venv\Scripts\python.exe tools\run_ldpc.py summary
.venv\Scripts\python.exe -m tests.test_ldpc
```

## 7. Scope limits (stated, not hidden)

- BG2 only. Base graph 1 is the high-throughput graph and is not implemented.
- No rate-matching chain, no CRC, no interleaving. The `E`-budget is filled by
  truncation, which is weaker than the standard circular buffer.
- One lifting size per k, chosen by the standard rule; no lifting-size sweep.
- Offset min-sum with a fixed offset and 15 iterations, not a tuned decoder.
- Link-level only: no bandwidth, noise figure or hardware claim.
