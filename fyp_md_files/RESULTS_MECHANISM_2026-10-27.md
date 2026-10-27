# Why the λ-shaped arm trains stably: gradient-conditioning evidence from the stored runs

**Date:** 2026-10-27 · **Type:** analysis only — **no training, no new cell, no
amendment** (nothing pre-registered was changed) · **Inputs:** the stored
`train_report.json` histories of the runs that recorded them · **Script:**
`tools/analyse_training_stability.py` · **Output:**
`results/MECHANISM/step_stability_summary.json`,
`fyp_md_files/figures/fig_training_stability.png`

## The gap this addresses

The thesis shows *that* the λ-shaped shaping stabilises low-bandwidth
task-oriented training (Wave-A: 0 NaN batches; Wave-C: 16.1 terminal loss at
k = 18 and 54), but not *why*. A reviewer asked for the mechanism, not the
correlation. This note extracts every per-step gradient measurement the
project actually stored and reports what it supports — and, just as
importantly, what it does not.

## Data actually available

118 training reports exist; **only 18 stored a full per-step history**
(`grad_enc`, `grad_dec`, `recon_mse_diag`, loss components):

| group | cells with history | width | channels |
|---|---|---|---|
| λ-heavy (0.1/0.1) + frozen protocol | 4 (Wave-C, k = 18/54) | 128 | awgn, rayleigh |
| λ-heavy + frozen protocol, k = 24 | 6 (A-14, 3 seeds) | 128 | awgn, rayleigh |
| λ-light (0.01/0.01) + frozen protocol | 2 (Wave-C, k = 18) | 128 | awgn |
| frozen task arm (λ_u = λ_v = 0) | 2 (A-9, k = 18) | **256** | awgn, rayleigh |
| λ-heavy + 3GPP dispersion, full tail | 2 (A-12, k = 18/54) | 128 | UMa-aerial |
| λ-heavy + 3GPP dispersion, outage tail resampled | 2 (A-13, k = 18/54) | 128 | UMa-aerial |

Wave-A and M2 reports store only first/final gradient scalars and mostly store
`null`, so they are used for the coarse table below and nothing more. **No 128-wide
frozen-arm history exists** — that is a data limitation, not a choice, and it is
stated with every number.

## Result 1 — the gradient magnitude separates the arms by ~14 orders

| cell | p50 ‖g_enc‖ | p99 ‖g_enc‖ | non-finite grad steps | converged |
|---|---|---|---|---|
| λ-heavy, k = 18 awgn (Wave-C) | 3.62 | 57 | 0 | yes |
| λ-heavy, k = 18 rayleigh | 3.74 | 57 | 0 | yes |
| λ-heavy, k = 54 awgn | 3.44 | 69 | 0 | yes |
| λ-heavy, k = 54 rayleigh | 3.31 | 67 | 0 | yes |
| λ-light, k = 18 awgn, seed 42 | 3.95 | 57 | 0 | yes |
| λ-light, k = 18 awgn, seed 43 | 4.09 | 113 | 0 | yes |
| λ-heavy, k = 24, awgn, seeds 45/46/47 (A-14) | 3.69 / 3.68 / 4.06 | 92 / 107 / 77 | 0 | yes |
| λ-heavy, k = 24, rayleigh, seeds 45/46/47 (A-14) | 3.65 / 3.54 / 4.10 | 94 / 96 / 85 | 0 | yes |
| **frozen task arm, k = 18 awgn (w 256)** | **3.0 × 10¹⁰** | **2.4 × 10¹⁸** | **75** | **no** |
| **frozen task arm, k = 18 rayleigh (w 256)** | **8.1 × 10⁷** | **5.0 × 10¹⁸** | **100** | **no** |
| **λ-heavy + full 3GPP tail, k = 18 (A-12)** | **2.4 × 10¹⁴** | **2.0 × 10¹⁹** | **311** | **no** |
| **λ-heavy + full 3GPP tail, k = 54 (A-12)** | **2.2 × 10¹⁴** | **1.6 × 10¹⁹** | **203** | **no** |
| λ-heavy + tail resampled, k = 18 (A-13) | 6.18 | 102 | 0 | yes |
| λ-heavy + tail resampled, k = 54 (A-13) | 4.27 | 68 | 0 | yes |

Coarse view over all completed cells that stored a numeric final gradient
(only 16 of 118 do): λ-heavy median final ‖g‖ = **1.9** (n = 14, max 1.5 × 10⁹
from A-12), λ-light = **1.9** (n = 2), frozen task arm = **1.1 × 10⁵** (n = 2,
the 256-wide A-9 cells). The coarse view is thin and is not used for any claim
beyond "the λ cells end at O(1)".

The six A-14 cells are a third, independent replication of the same pattern at
a different bandwidth (k = 24) and three seeds: p50 3.5–4.1, p99 77–107, zero
non-finite steps, all converged — the same regime as the Wave-C and A-13 cells
and ~10¹⁴ away from the frozen arm's stored histories.

## Result 2 — a quasi-2 × 2 that isolates the trigger

| λ shaping | training episodes | outcome | p50 ‖g_enc‖ |
|---|---|---|---|
| off | standard (awgn/rayleigh) | diverges, 0/2 converged | 10⁷–10¹⁰ |
| **on** | standard (awgn/rayleigh) | converges, 10/10 | 3.3–4.1 |
| on | 3GPP, full dispersion | diverges, 0/2 | ~10¹⁴ |
| on | 3GPP, outage tail (> 20 dB) resampled | converges, 2/2 | 4.3–6.2 |

Reading:

1. **Turning λ on is what brings the gradient into O(1)** under the frozen
   protocol: p99 falls from ~10¹⁸ to ~60 and every cell converges with zero
   non-finite steps.
2. **Capacity is not the explanation.** The frozen-arm histories are the
   *wider* (256) cells, and they still explode; A-9 showed a 256-wide frozen
   cell still fails at 0.027/0.041 success. The pathology follows the loss, not
   the parameter count.
3. **λ is not a general cure.** The same λ arm explodes when the training set
   contains > 20 dB outage episodes (A-12). Resampling 3.78 % of episodes
   returns p50 ‖g‖ from 2 × 10¹⁴ to ≈ 5 (A-13). So the trigger is rare
   high-error episodes, and λ shaping buys robustness to the *frozen protocol's*
   channel, not to arbitrary ones.

## What this does not show

- **Not a causal decomposition.** It does not say whether λ_u, λ_v, λ_T or the
  interaction does the work; that needs new arms (λ_u only, λ_v only) and is
  not claimed.
- **No sharpness / loss-landscape measurement.** The evidence is gradient
  magnitude and convergence, not curvature.
- **The λ on/off contrast is confounded by width** (128 vs 256). The confound
  works *against* the claim (the wider network fails anyway), which is why the
  comparison is still informative, but it is not a clean factorial.
- **The naive within-run correlation between reconstruction MSE and gradient
  norm is negative** (median Spearman −0.43) and confounded by the early
  transient. It is stored in the JSON for transparency and is **not** used as
  evidence.
- The 3GPP cells differ in channel from the frozen-protocol cells; the clean
  contrast there is A-12 vs A-13 (identical design, one flag).

## Where this leaves the reviewer point

Partially closed, honestly: the mechanism claim is now "λ shaping bounds the
encoder gradient by ~14 orders of magnitude, and the failure it prevents is an
outlier-episode gradient blow-up that capacity does not fix" — with the stored
numbers behind it. Still open: **which** λ term matters, and a curvature-level
measurement.

## Reproduce

```powershell
python tools/analyse_training_stability.py            # table + JSON
python tools/analyse_training_stability.py --figure   # + fig_training_stability.png
```

Read-only: the script opens `train_report.json` files and writes only
`results/MECHANISM/step_stability_summary.json` and the figure.
