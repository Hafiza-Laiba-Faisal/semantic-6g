# Training under the 3GPP UMa-aerial dispersion (A-13): the operating point does not move

**Date:** 2026-10-27 · **Amendment:** A-13 (`PROTOCOL_AMENDMENTS.md`) ·
**Artifacts:** `results/AERIAL_TRAIN_TRUNC/` · **Runner:**
`tools/run_aerial_training.py` (`--variant=a13`) · **Tests:**
`tests/test_aerial_training.py` (34/34)

## Question

A-12 (identical design, full dispersion during training) diverged at both k:
the frozen recipe has no protection against the ~4 % of episodes that land
≥20 dB below its 0–20 dB training curriculum. A-13 repeats the experiment with
**one change**: training episodes whose 3GPP gain offset exceeds **+20 dB**
(the outage tail, 3.78 % of episodes) are resampled. The objective, weights,
architecture, budget, seeds, optimiser and — importantly — the **evaluation**
are unchanged: evaluation always sees the full, untruncated dispersion.

This is the *fairer* comparison, not a weaker one: the Rayleigh-trained
comparator never trained on outage episodes either.

## Design

| field | value |
|---|---|
| arm | `lam_ctl_heavy` (λ 1.0 / 0.1 / 0.1 / 2.0) |
| k / seed / cells | 18, 54 / 42 / 2 |
| width, steps, lr, clip, batch | 128, 3000, 1e-3, 1.0, 256 episodes (all frozen) |
| channel | TR 36.777 P_LoS (0.889) + TR 38.901 shadowing (4 dB LoS / 6 dB NLoS), block fading, gain centred on PL_mean = 95.63 dB |
| training tail | episodes with offset > +20 dB resampled: keeps 100 % of LoS and 66 % of NLoS episodes (kept offset std 5.91 dB) |
| evaluation | full dispersion, E = 1000, seed 10042, −15…+20 dB, one shared gain vector (mean +0.18 dB, std 7.53 dB, P_LoS 0.879) |
| comparators | Wave-C Rayleigh `lam_ctl_heavy` checkpoints (untouched), digital K=7, B = `M2_B_BY_K[k]` |

Amendment and code were committed **before the first cell** (`3e9c036`).

## Pre-registered pre-flight gate (300 steps, k = 18)

| gate | required | measured |
|---|---|---|
| non-finite gradient steps | 0 | **0** |
| loss after step 50 | ≤ 10³ | **71.0** (A-12: 1440) |
| mean of last 20 steps | ≤ 50 | **17.65** (A-12: 4256) |

449 s, 1.50 s/step; the in-training (full-dispersion) diagnostic already read
**0.992** success at 300 steps. Gate passed on the primary cap, so the
declared +15 dB fallback was not used.

## Training

| cell | wall | s/step | loss initial → final | loss min (step) | steps > 10³ | non-finite grad steps | final encoder grad | train-eval SR |
|---|---|---|---|---|---|---|---|---|
| k = 18 | 4558 s | 1.52 | 4643.2 → **16.2** | 14.5 (853) | 3 | **0** | 0.92 | 0.9844 |
| k = 54 | 4882 s | 1.63 | 5235.9 → **16.1** | 14.4 (853) | 3 | **0** | 1.45 | 0.9961 |

For scale: every converged frozen λ-arm cell ends at loss 16.1 with encoder
gradient 1–4; A-12's cells ended at 4320 / 210 with 311 / 203 non-finite
gradient steps. The `aerial_train_max_offset_db = 20.0` field is recorded in
both `train_report.json` specs, so the flag provably reached the trainer.

## Evaluation (paired dispersion, E = 1000)

| k | method | −15 | −10 | −5 | 0 | +5 | +10 | +15 | +20 dB |
|---|---|---|---|---|---|---|---|---|---|
| 18 | **trained on 3GPP (A-13)** | 0.661 | 0.818 | 0.900 | 0.941 | 0.965 | 0.985 | 0.994 | **0.999** |
| 18 | trained on Rayleigh (A-10) | 0.647 | 0.812 | 0.898 | 0.935 | 0.960 | 0.978 | 0.990 | 0.995 |
| 18 | digital conv | 0.054 | 0.088 | 0.140 | 0.152 | 0.154 | 0.151 | 0.147 | 0.151 |
| 54 | **trained on 3GPP (A-13)** | 0.800 | 0.885 | 0.930 | 0.963 | 0.982 | 0.995 | 0.998 | **0.999** |
| 54 | trained on Rayleigh (A-10) | 0.781 | 0.877 | 0.930 | 0.965 | 0.980 | 0.992 | 0.996 | 0.997 |
| 54 | digital conv | 0.030 | 0.082 | 0.353 | 0.685 | 0.854 | 0.903 | 0.936 | 0.969 |

Required nominal SNR (transmit = nominal + 95.63 dB):

| method | 0.90 | 0.99 |
|---|---|---|
| trained on 3GPP, k = 18 | −5.00 dB / **90.6 dB** | 12.78 dB / **108.4 dB** |
| trained on Rayleigh, k = 18 | −4.73 dB / 90.9 dB | 15.00 dB / 110.6 dB |
| trained on 3GPP, k = 54 | −8.33 dB / **87.3 dB** | 8.08 dB / **103.7 dB** |
| trained on Rayleigh, k = 54 | −7.83 dB / 87.8 dB | 9.17 dB / 104.8 dB |
| digital conv, k = 54 | 9.69 dB / 105.3 dB | not reachable |
| trained on 3GPP, k = 18 **vs** digital | digital not reachable (ceiling 0.154) | — |
| A-12 (diverged) | not reachable | not reachable |

Gap to the Rayleigh-trained comparator (negative = A-13 needs less power):
**−0.27 dB** at 0.90 and **−2.22 dB** at 0.99 (k = 18); **−0.50 dB** and
**−1.09 dB** (k = 54). Against the classical link at k = 54 the trained arm
keeps an **18.0 dB** advantage at 0.90, and at k = 18 the classical link never
reaches 0.90 at all.

Per-branch behaviour (same episodes, same gains) — the trained arm buys its
small edge on the NLoS branch at low SNR:

| k | SNR | LoS SR: A-13 / frozen | NLoS SR: A-13 / frozen |
|---|---|---|---|
| 18 | 0 dB | 0.9932 / 0.9920 | **0.5620** / 0.5207 |
| 18 | +10 dB | 1.0000 / 0.9966 | **0.8760** / 0.8430 |
| 54 | 0 dB | 0.9977 / 0.9954 | 0.7107 / 0.7438 |
| 54 | +10 dB | 0.9989 / 0.9977 | 0.9669 / 0.9504 |

## Reading

1. **The A-10 conclusion is insensitive to the training channel.** Training
   under the 3GPP dispersion rather than Rayleigh moves the operating point by
   at most ~0.5 dB at 0.90 and ~2 dB at 0.99, in both directions of sampling
   noise (E = 1000 Wilson intervals overlap at every grid point). The
   transmit-SNR advantage over the classical link survives.
2. **The divergence in A-12 was the tail, not the channel.** One line changed
   (resample 3.78 % of episodes), and the same design went from 311 non-finite
   gradient steps to zero, and from a dead network (0.02 flat) to 0.94–0.999.
   That is direct evidence for the diagnosis in
   `RESULTS_AERIAL_TRAIN_2026-10-26.md`.
3. **A-13 must not be quoted as "3GPP training recovers X dB".** The honest
   statement is: *training on the 3GPP dispersion with the outage tail
   excluded, and evaluating on the full dispersion, gives a policy that is
   statistically indistinguishable from the Rayleigh-trained one; neither
   gains nor loses materially.* The qualifier travels with the number.
4. **Neither arm is "the better one".** Differences are inside sampling error;
   no ranking is claimed.

## Limits

- One seed per k; E = 1000 (Wilson ±~2.6 pp at p = 0.9). The 0.3–2.2 dB
  crossing differences are interpolations of overlapping curves and are **not**
  a claimed improvement.
- Training excluded 3.78 % of episodes (outage tail). "Trained on 3GPP" here
  always means "trained on 96.2 % of the dispersion, evaluated on 100 %".
- No Doppler, no time-correlated shadowing, one geometry (500 m / 50 m /
  2 GHz), 2-D benchmark.
- The +15 dB fallback cap was never used; A-12's failed cells remain in the
  record untouched.

## Reproduce

```powershell
python tools/run_aerial_training.py verify --variant=a13
python tools/run_aerial_training.py preflight --variant=a13      # 300-step gate
python tools/run_aerial_training.py train --variant=a13 --shard=0/2
python tools/run_aerial_training.py train --variant=a13 --shard=1/2
python tools/run_aerial_training.py evaluate --variant=a13
python tools/run_aerial_training.py finalize --variant=a13
python -m tests.test_aerial_training                              # 34 checks
```

Summary: `results/AERIAL_TRAIN_TRUNC/step_aerial_training_summary.json`
(`outcome` is computed from the data, not typed in). Figure:
`fig_aerial_training.png` (A-12 flat, A-13 over the frozen curve, digital
dotted).
