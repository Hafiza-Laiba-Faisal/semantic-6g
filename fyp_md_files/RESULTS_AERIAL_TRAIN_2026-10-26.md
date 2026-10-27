# Training under the 3GPP UMa-aerial dispersion (A-12): the frozen recipe diverges

**Date:** 2026-10-26 (run), 2026-10-27 (write-up) · **Amendment:** A-12
(`PROTOCOL_AMENDMENTS.md`) · **Artifacts:** `results/AERIAL_TRAIN/` ·
**Runner:** `tools/run_aerial_training.py` · **Tests:**
`tests/test_aerial_training.py` (25/25)

## Question

A-10 evaluated the frozen Wave-C checkpoints on a 3GPP UMa-aerial link
(evaluation only). A reviewer can object that the models were never *trained*
on that channel. A-12 trains the `lam_ctl_heavy` arm under the 3GPP dispersion
and asks whether its operating point changes.

## Design (pre-registered 2026-10-26T13:10Z, before the first cell)

| field | value |
|---|---|
| arm | `lam_ctl_heavy` (λ_d 1.0, λ_u 0.1, λ_v 0.1, λ_T 2.0) |
| k / seeds / cells | 18, 54 / 42 / 2 |
| width, steps, lr, clip | 128, 3000, 1e-3, 1.0 (all frozen) |
| channel | `umaaerial`: per-episode LoS/NLoS (P_LoS 0.889) + TR 38.901 shadowing (4 dB LoS / 6 dB NLoS), block fading, gain centred on PL_mean |
| geometry | d2D 500 m, h 50 m, 2 GHz → PL_LoS 93.45, PL_NLoS 113.02, PL_mean 95.63 dB |
| eval | E = 1000, seed 10042, nominal grid −15…+20 dB (8 points), one shared gain vector (mean +0.18 dB, std 7.53 dB, empirical P_LoS 0.879) |
| comparators | Wave-C `lam_ctl_heavy` Rayleigh checkpoints (untouched), digital K=7 with B = `M2_B_BY_K[k]` |

Nothing here was retrained or re-selected: this is the run as registered.

## Result: both cells complete, both training runs collapsed

| cell | wall | s/step | loss, first 10 steps | loss minimum (step) | steps with loss > 1e3 | steps with non-finite grad | steps with grad > 1e6 | final loss |
|---|---|---|---|---|---|---|---|---|
| k = 18 | 4756 s | 1.59 | 1079 | **15.4 (step 105)** | 428 (14.3 %) | 311 | 2238 | 4320 |
| k = 54 | 5329 s | 1.78 | 1143 | **15.6 (step 160)** | 517 (17.2 %) | 203 | 2421 | 210 |

Reference: every converged λ-heavy cell in Wave-C ends at task loss **16.1**
with encoder gradient **1.3–4.3**. So both A-12 cells *reached* the frozen
arm's terminal loss within ~100–160 steps, and then the trajectory left it:
about one step in six produces a loss above 10³, and three quarters of all
steps have an encoder gradient above 10⁶ (up to `inf`).

`train_report.json` says `nan_batches: 0` and `grads_finite: true`; those
fields refer to the **final** step / to non-finite *losses* only. The step
history contains 311 and 203 non-finite gradients. Nothing in the report was
edited after the fact.

## Evaluation: the checkpoints are dead networks

Success rate vs nominal SNR, paired gain vector, E = 1000:

| k | method | −15 | −10 | −5 | 0 | +5 | +10 | +15 | +20 dB |
|---|---|---|---|---|---|---|---|---|---|
| 18 | trained on 3GPP (A-12) | 0.023 | 0.027 | 0.028 | 0.025 | 0.027 | 0.025 | 0.028 | 0.029 |
| 18 | trained on Rayleigh (A-10) | 0.647 | 0.812 | 0.898 | 0.935 | 0.960 | 0.978 | 0.990 | 0.995 |
| 18 | digital conv | 0.054 | 0.088 | 0.140 | 0.152 | 0.154 | 0.151 | 0.147 | 0.151 |
| 54 | trained on 3GPP (A-12) | 0.025 | 0.020 | 0.015 | 0.014 | 0.016 | 0.017 | 0.013 | 0.014 |
| 54 | trained on Rayleigh (A-10) | 0.781 | 0.877 | 0.930 | 0.965 | 0.980 | 0.992 | 0.996 | 0.997 |
| 54 | digital conv | 0.030 | 0.082 | 0.353 | 0.685 | 0.854 | 0.903 | 0.936 | 0.969 |

The A-12 curves are flat in SNR — the signature of a broken mapping, not of a
bandwidth or power limit. Consequently the 0.90 / 0.99 crossing points are
**not reachable** for either A-12 cell. The comparator crossings on the same
gain vector (nominal / transmit = nominal + 95.63 dB):

| method | 0.90 | 0.99 |
|---|---|---|
| trained on Rayleigh, k = 18 | −4.73 dB / 90.9 dB | 15.0 dB / 110.6 dB |
| trained on Rayleigh, k = 54 | −7.83 dB / 87.8 dB | 9.17 dB / 104.8 dB |
| digital conv, k = 54 | 9.69 dB / 105.3 dB | not reachable |
| trained on 3GPP, k = 18 and 54 | not reachable | not reachable |

The 0.90 crossings reproduce A-10 (91 / 87 dB). The 0.99 crossings do not:
A-10 used the analytic two-branch mixture on a 5-dB grid, this is the direct
paired Monte-Carlo, and the upper tail is where a mixture approximation is
weakest.

**Diagnostic — is it the channel or the network?** The A-12 checkpoints were
also evaluated on the frozen awgn and rayleigh channels with no aerial gain at
all:

| k | awgn 0/10/20 dB | rayleigh 0/10/20 dB |
|---|---|---|
| 18 | 0.027 / 0.027 / 0.027 | 0.029 / 0.027 / 0.028 |
| 54 | 0.014 / 0.012 / 0.013 | 0.014 / 0.014 / 0.013 |

They fail on AWGN. The networks themselves were destroyed; the aerial
evaluation is not what is being measured here.

## Why it diverged (measured, not asserted)

The channel implementation is not the suspect: with `aerial_gain = 1` it is
bit-identical to the frozen Rayleigh channel, and the evaluation convention
(divide the pre-generated noise by `g`) matches the model's `umaaerial`
branch exactly — both are unit tests in `tests/test_aerial_training.py`.

What changes is the *spread* of the effective SNR, at unchanged mean:

| at nominal 0 dB | Rayleigh | 3GPP UMa-aerial |
|---|---|---|
| effective-SNR mean / std | +0.50 / 5.59 dB | +0.49 / **9.37 dB** |
| P(SNR_eff ≤ −20 dB) | 0.52 % | **4.16 %** |
| P(SNR_eff ≤ −30 dB) | 0.05 % | 0.75 % |
| worst noise amplification 1/(g·h) in a 256-episode batch, median | 14× | **55×** |
| same, 99th percentile of batches | 127× | 452× |

The frozen training curriculum samples nominal SNR uniformly on 0–20 dB, so
in the aerial channel ~4 % of episodes sit 20 dB or more below the curriculum's
floor, and the worst episode in a batch is amplified ~4× more than anything
Rayleigh produces. With an unweighted mean loss, a handful of such episodes
dominate both the loss and its gradient (their squared error is ~10³× the
typical one). The frozen recipe has no mechanism for this: no loss clipping,
no per-episode normalisation, fixed learning rate, and a gradient clip that
converts a 10¹²-norm gradient into a unit-norm step in an essentially random
direction. That is the observed signature — the model learns the task in the
first 100 steps, then the outlier batches push it out of the basin and clipped
steps keep it wandering.

This is the honest reading of the evidence, and it is **not** a claim that
3GPP training is impossible — only that this recipe, unchanged, cannot do it.
Alternative explanations that this run cannot exclude: the specific mixture
bimodality (LoS vs NLoS) rather than the tail per se, and plain
hyper-parameter sensitivity at 128 width.

## What this does and does not show

- **Does not** show that training under the 3GPP channel helps or hurts the
  λ-arm. The A-12 curves are not a ranking; a reviewer must not read "3GPP
  training is worse" from them.
- **Does not** weaken A-10. That result evaluates untouched Wave-C
  checkpoints; its numbers reproduce here on the same gain vector (0.647 →
  0.995 and 0.781 → 0.997).
- **Does** show that the λ-arm's advantage is a property of the *receiver-SNR*
  operating point, and that transporting it to a heavy-tailed dispersion needs
  a robust training recipe, which is a protocol decision, not a bug fix.
- **Does** leave the design question open: the two cells hit the frozen arm's
  terminal loss (15.4 / 15.6 vs 16.1) *before* collapsing, so the channel is
  not what prevents learning.

## Not run: what a fair test would need (A-13, requires approval)

Any of these changes a pre-registered quantity, so none of them was done
silently and none is included in the numbers above:

| option | what changes | why it is defensible | cost |
|---|---|---|---|
| **truncated training tail** (recommended) | training episodes with gain offset > 10 dB are resampled (2.2 % of episodes); evaluation keeps the full dispersion | channel model, objective, eval grid and statistics all unchanged; the training set is simply restricted to non-outage episodes and this is stated | 2 cells ≈ 90 min |
| robust loss | per-episode loss clipped at the batch 99th percentile | keeps the exact channel; changes the objective, so it is a new arm, not the frozen one | 2 cells ≈ 90 min |
| SNR curriculum shift | nominal SNR drawn per episode so the *effective* SNR is uniform 0–20 dB | keeps the objective; but removes the dispersion from training, i.e. it is not a 3GPP-trained arm any more | 2 cells ≈ 90 min |

Recommendation: the truncated-tail variant, because it changes only which
episodes are used for training and leaves the channel, the objective, the
budget and the evaluation untouched — the difference from A-12 is one line and
is stated up front. A 300-step pre-flight would also be run first to confirm
that the collapse is gone before spending the full budget.

## Reproduce

```powershell
python tools/run_aerial_training.py verify
python tools/run_aerial_training.py train --shard=0/2    # k = 18
python tools/run_aerial_training.py train --shard=1/2    # k = 54
python tools/run_aerial_training.py evaluate
python tools/run_aerial_training.py finalize
python -m tests.test_aerial_training                      # 25 checks
```

Summary JSON: `results/AERIAL_TRAIN/step_aerial_training_summary.json`
(2/2 train cells, 48/48 evaluation slots). It carries `status: complete`,
which only means every declared slot was filled, **and** an explicit
`outcome: negative` — do not read `complete` as success. Figure:
`fig_aerial_training.png` — the flat red curves are the honest picture.

The AWGN/Rayleigh control above is reproducible without touching anything:

```powershell
python tools/diag_aerial_checkpoints.py
```
