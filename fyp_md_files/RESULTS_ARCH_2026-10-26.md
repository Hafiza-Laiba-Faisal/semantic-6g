# Capacity ablation (A-9): does a wider network rescue the frozen task arm?

**Date:** 2026-10-26 · **Amendment:** A-9 (`PROTOCOL_AMENDMENTS.md`) ·
**Artifacts:** `results/ARCH_ABL/` · **Runner:** `tools/run_arch_ablation.py` ·
**Tests:** `tests/test_arch_width.py` (16/16)

## Question

The frozen task-oriented arm (λ_u = λ_v = 0) fails at k = 18: about 2 %
success at every SNR against 99 % for the λ-shaped arm. Its MLP is small
(6–128–128–2k). A reviewer can ask whether this is a **capacity** artifact
rather than a property of the objective. This run trains the same loss at
**hidden width 256** and checks the failure again.

## Design (pre-registered, then reduced once, before any cell finished)

| field | value |
|---|---|
| arm | frozen task loss (λ_d = 1, λ_u = 0, λ_v = 0, λ_T = 2) |
| width | 256 (frozen is 128), verified on the real weight shape before training |
| k | 18 |
| channels | awgn, rayleigh |
| seeds | 42 |
| steps | 3000 (frozen budget), lr 1e-3, grad clip 1.0 |
| eval | E = 1000, SNR 0/5/10/15/20 dB, `TEST_SEED` = 10042 |
| cells | 2 (awgn, rayleigh) |
| root | `results/ARCH_ABL/` only |

**Scope reduction (recorded in A-9 at 12:25 UTC, no cell complete, no metric
in existence).** The pre-registered design was 8 cells (2 arms × 2 channels ×
2 seeds). A 256-wide cell costs 3–4× a 128-wide one on this 4-core machine,
so the run was cut to the two cells that answer the headline question. The
λ arm at 256 and both seed-43 cells stay **declared but unrun**; do not
describe this as a 2 × 2 × 2 design.

**Historical bug (fixed, not retrained).** Wave-A's `caps_h256` claimed width
256 but every layer was 128 wide, because the model constructors bound
`config.HIDDEN_DIMS` as a default argument at import time. That arm is
declared invalid. The constructors now resolve the width at construction
time; this runner asserts `encoder.backbone[0].weight.shape == (256, 6)`
before training. During the first evaluation attempt the shared checkpoint
loader still rebuilt a 128-wide model and refused to load the 256-wide
weights; `ArchCell.load_ckpt` now builds the real width (tooling fix, applied
before any ARCH number was reported).

## Results

Training (3000 steps each, no NaN batches, finite encoder/decoder gradients):

| cell | wall | s/step | task loss initial → final | recon MSE (normalized) initial → final | train-eval SR |
|---|---|---|---|---|---|
| awgn, seed 42 | 9439 s | 3.15 | 5025.2 → 1343.6 | 0.246 → **1605** | 0.035 |
| rayleigh, seed 42 | 3950 s | 1.32 | 5015.8 → 658.8 | 0.247 → **1344** | 0.043 |

The two wall times are not a channel cost comparison: the awgn cell ran while
other jobs shared the machine.

Success rate vs SNR (paired realizations, E = 1000):

| k | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|
| 18 | awgn | 0.027 | 0.026 | 0.024 | 0.026 | 0.027 |
| 18 | rayleigh | 0.041 | 0.044 | 0.042 | 0.042 | 0.039 |

Reference points at the same k, for scale (different protocol, stated so the
comparison is not overstated):

| arm | width | channel | E | seeds | SR |
|---|---|---|---|---|---|
| frozen task (M2) | 128 | awgn | 5000 | 3 | 0.0179 [0.0159, 0.0202] |
| frozen task (M2) | 128 | rayleigh | 5000 | 3 | 0.0238 [0.0215, 0.0264] |
| frozen task (A-9) | 256 | rayleigh | 1000 | 1 | 0.041 |
| λ-shaped (Wave-C) | 128 | rayleigh | 1000 | 1 | **0.993** at 0 dB |

## Reading

1. **Capacity is not the cause.** Doubling the width does not move the frozen
   arm out of the 2–4 % band; the λ-shaped arm at *half* the width reaches
   99 %. The gap is an objective/training-signal property.
2. **The failure has a clear signature.** While the task loss falls
   (5025 → 1344 awgn, 5016 → 659 rayleigh), the normalized reconstruction MSE
   explodes from 0.25 to ~1.3–1.6 × 10³. The network is not learning a better
   state representation; it is drifting into a degenerate high-gain regime
   that the distance/terminal terms weakly reward, and the decoder output
   leaves the data manifold entirely.
3. **It is not "underfitting".** A model with more parameters underfits less
   by construction, and this one still collapses. That is evidence for the
   diagnostic claim of the thesis rather than against it.

## Limits (explicit)

- One seed, one width, one k. This is not a general claim about residual or
  attention architectures — only that a 2× wider MLP does not fix it.
- E = 1000 here vs E = 5000 for the frozen M2 reference; Wilson intervals for
  a single seed are not seed variance.
- 3000 steps for both, so "it needs longer" is not excluded. A longer-budget
  run would be a new amendment.
- Declared but unrun: λ arm at 256, seed 43 (both channels).

## Reproduce

```powershell
python tools/run_arch_ablation.py verify
python tools/run_arch_ablation.py train --shard=0/2   # awgn
python tools/run_arch_ablation.py train --shard=1/2   # rayleigh
python tools/run_arch_ablation.py evaluate
python tools/run_arch_ablation.py finalize
python -m tests.test_arch_width                       # 16 checks
```
