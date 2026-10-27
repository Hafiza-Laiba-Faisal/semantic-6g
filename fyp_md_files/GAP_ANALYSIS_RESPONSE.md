# Reviewer / roadmap gaps - what is now implemented

**Date:** 2026-10-26 - **Scope:** implementation response to the review points
(capacity, polar baseline, 3GPP channel, reproducibility, novelty). Everything
here is additive: no frozen M1/M2 artifact was edited, nothing was retrained
to improve a number, and every new run has a dated amendment (A-9, A-10, A-11,
A-12).

## At a glance

| # | gap raised | status | artifact | headline |
|---|---|---|---|---|
| 1 | channel is only AWGN/Rayleigh, no 3GPP aerial model | **done** | A-10, `results/AERIAL/`; A-12 (negative); A-13, `results/AERIAL_TRAIN_TRUNC/` | lambda policy needs 90.6 dB (k=18) / 87.3 dB (k=54) tx SNR for 0.90 - digital needs 105.3 dB at k=54 and never reaches 0.90 at k=18. Training *under* the dispersion moves this by <=2 dB (inside sampling error), so the evaluation-only result stands; A-12 (full dispersion) diverged and is reported as-is |
| 2 | digital baseline is one convolutional code | **done** | A-11, `results/POLAR/` | k=12/18/24 within 0.05 at the payload ceiling; polar better at k=30 (+0.06..+0.12 at >=10 dB), worse at k=54/B=8 below 10 dB |
| 3 | capacity ablation invalid / missing | **done** (2 cells) | A-9, `results/ARCH_ABL/` | width 256 still fails: 0.027 awgn / 0.041 rayleigh vs 0.018/0.024 at 128; the task loss falls while reconstruction MSE explodes to ~10³ |
| 4 | k=24 multi-initialisation missing | **not run** (designed, costed) | section 4 | 6 cells, ~2.5-3 h at 2-way parallelism |
| 5 | packaging / reproducibility | **partial** | `environment.yml`, Dockerfile | conda spec added; Docker CMD fixed; image not built (no docker on this machine); Zenodo needs the user |
| 6 | novelty argument | **clarified** (mechanism now measured) | section 6, `RESULTS_MECHANISM_2026-10-27.md` | four defensible claims; mechanism measured at gradient level (λ bounds ‖g‖ by ~14 orders; capacity does not fix it; outage episodes are the trigger); still open: which λ term, Doppler, competitive polar receiver |

## 1. 3GPP aerial channel (A-10)

`project/channels/aerial.py` already had the TR 36.777 / TR 38.901 UMa-aerial
large-scale formulas, but no reported number used them. `tools/run_aerial_link.py`
now wires them in **without retraining**:

- under perfect-CSI ZF a deterministic gain is exactly an effective-SNR shift
  (`SNR_eff = SNR_tx - PL`), so the trained checkpoints are evaluated unchanged;
- LoS and NLoS are mixed on the same measured curve with `P_LoS` weights,
  which is exact for i.i.d. episodes with an independent LoS draw;
- the required **transmit** SNR is reported next to the receiver-side SNR, so
  the aerial penalty is explicit.

Result (d2D = 500 m, h = 50 m, 2 GHz, `P_LoS` = 0.889, `PL_mean` = 95.6 dB,
E = 1000):

| method | SR >= 0.90 | SR >= 0.99 |
|---|---|---|
| lambda-shaped, k = 18 | 91 dB | 107 dB |
| lambda-shaped, k = 54 | 87 dB | 102 dB |
| digital conv, k = 18 | not reachable (ceiling 0.17) | not reachable |
| digital conv, k = 54 | 108 dB | 117 dB |

The lambda policy's advantage over the classical link is 21 dB at 0.90 and
15 dB at 0.99 (k = 54), and unbounded at k = 18 where the classical link is
payload-limited. **Not modelled:** Doppler, shadowing, aerial small-scale
fading, antenna gain/noise figure. One geometry only.

Details, limits and the two runner bugs found and fixed before shipping:
`fyp_md_files/RESULTS_AERIAL_2026-10-26.md`. Figure:
`fig_aerial_required_snr.png`.

### 1b. Training under the dispersion (A-12): ran as registered, and diverged

A-10 uses checkpoints trained on Rayleigh, so A-12 retrains the `lam_ctl_heavy`
arm *under* the same 3GPP dispersion (per-episode LoS/NLoS + TR 38.901
shadowing, block fading, gain centred on PL_mean) at k = 18 and 54, frozen
width/steps/lr/clip/loss, E = 1000 evaluation on the same paired gains.

**Both cells completed and both training runs collapsed.** They reached the
frozen λ-arm's terminal task loss (15.4 at step 105, 15.6 at step 160, against
16.1 for every converged Wave-C cell) and then left it: 14–17 % of steps have
loss > 10³, 203–311 steps produce non-finite encoder gradients, ~75 % of steps
exceed 10⁶. The saved checkpoints are dead networks: success rate 0.023–0.029
(k = 18) and 0.013–0.025 (k = 54) at every nominal SNR from −15 to +20 dB, and
0.027 / 0.014 flat on the frozen **AWGN** channel as a control.

**Cause, measured.** At the same mean effective SNR the dispersion more than
doubles the spread (std 9.37 vs 5.59 dB). At nominal 0 dB, 4.16 % of episodes
sit at SNR_eff ≤ −20 dB (Rayleigh: 0.52 %), so the worst episode in a
256-episode batch carries 55× noise amplification in the median batch (14× for
Rayleigh) and ~10³× the typical squared error. The frozen recipe has no
outlier control - no loss clipping, no per-episode normalisation, fixed lr,
and a gradient clip that turns a 10¹²-norm gradient into an arbitrary
unit-norm step.

**What it does not mean.** Not that 3GPP training is worse: these curves are
not a ranking. The A-10 evaluation result is untouched and reproduces on the
same gain vector. A fair test needs a dated amendment (truncated training tail
recommended; alternatives: robust loss, SNR-curriculum shift) - **none was
applied, no cell was retrained**, and the failed cells remain in the record.
Details: `fyp_md_files/RESULTS_AERIAL_TRAIN_2026-10-26.md`, figure
`fig_aerial_training.png`.

### 1c. A-13: the same design with the outage tail resampled — the operating point does not move

A-13 (dated amendment, approved 2026-10-27, code committed **before the first
cell**) repeats A-12 with one change: training episodes whose 3GPP gain offset
exceeds +20 dB (the outage tail, 3.78 % of episodes) are resampled. Evaluation
is never truncated. A pre-registered 300-step pre-flight gate passed on the
first attempt (0 non-finite gradients; worst loss after step 50 = 71 vs 1440
in A-12; last-20 mean 17.7 vs 4256), so the declared +15 dB fallback was never
used.

Both cells converged to the frozen λ-arm profile (loss 16.2 / 16.1, final
encoder gradient 0.92 / 1.45) with **zero** non-finite gradient steps. On the
full dispersion, same paired gain vector:

| method | 0.90 | 0.99 |
|---|---|---|
| trained on 3GPP (A-13), k = 18 | −5.00 dB / **90.6 dB** | 12.78 dB / 108.4 dB |
| trained on Rayleigh, k = 18 | −4.73 dB / 90.9 dB | 15.00 dB / 110.6 dB |
| trained on 3GPP (A-13), k = 54 | −8.33 dB / **87.3 dB** | 8.08 dB / 103.7 dB |
| trained on Rayleigh, k = 54 | −7.83 dB / 87.8 dB | 9.17 dB / 104.8 dB |
| digital conv, k = 54 | 9.69 dB / 105.3 dB | not reachable |

So the penalty for having been trained on Rayleigh rather than on the 3GPP
dispersion is **≤ 2.2 dB at 0.99 and ≤ 0.5 dB at 0.90**, with overlapping
Wilson intervals at every grid point — no ranking is claimed, and A-13 is not
evidence of a gain. What it does establish is that the A-10 numbers are not an
artefact of the training channel, and that A-12's failure was the tail (one
line changed: 311 → 0 non-finite gradient steps, dead network → 0.94–0.999).
The qualifier that must travel with the number: *trained on 96.2 % of the
dispersion (100 % LoS, 66 % NLoS episodes), evaluated on 100 %*. Details:
`fyp_md_files/RESULTS_AERIAL_TRAIN_TRUNC_2026-10-27.md`.

## 2. Polar baseline (A-11)

`tools/run_polar_baseline.py` puts the CRC-8/SC polar baseline
(`project/baselines/digital_polar.py`) on the frozen M2 grid and re-runs the
convolutional baseline on the **same episodes**, so the contrast is paired.

- k = 12/18/24: |delta| <= 0.05 from 5 dB upward (mostly <= 0.02) - the
  reported crossover is a rate/quantizer effect, not a coding-choice effect.
- k = 30: polar is *better* at >= 10 dB (+0.06 awgn; +0.09/+0.12/+0.05
  rayleigh).
- k = 54 (B = 8): plain SC polar is much worse at 0-5 dB (-0.66..-0.94) and
  reaches parity by 10-15 dB, because the code has no rate matching and no
  CRC-aided list decoding. That column is a lower bound on polar, not a
  verdict on polar coding.

Details: `fyp_md_files/RESULTS_POLAR_2026-10-26.md`. Figure:
`fig_polar_vs_conv.png`.

## 3. Capacity ablation (A-9)

**The bug.** `DeepJSCCEncoder`/`DeepJSCCDecoder` declared
`hidden_dims: tuple = config.HIDDEN_DIMS`, a default bound at **import** time.
`WaveCell(hidden=256)` patches the config, which therefore did nothing: the
Wave-A `caps_h256` arm trained 128-wide layers while its metadata said 256
(A-6). The width is now resolved inside `__init__`; the frozen default is still
128, existing checkpoints load unchanged, and a fresh evaluation of one Wave-C
slot reproduces the stored JSON exactly.

**The run.** Pre-registered as 8 cells (2 arms x 2 channels x 2 seeds) at
width 256, k = 18. A 256-wide cell costs ~3-4x a frozen 128-wide cell on this
4-core machine, so the run was reduced to the 2 cells that answer the headline
question: frozen task arm, width 256, k = 18, awgn + rayleigh, seed 42. The
reduction was recorded at 12:25 UTC with **no cell completed and no metric in
existence**; cancelled start markers are kept under
`results/ARCH_ABL/_cancelled/`. The lambda arm at 256 and both seed-43 cells
remain declared but unrun - this is **not** a 2x2x2 design.

<!-- ARCH_RESULT -->
**Result.** Both cells completed (no NaN batches, finite gradients). Success
rate stays in the frozen band: awgn 0.027/0.026/0.024/0.026/0.027 and rayleigh
0.041/0.044/0.042/0.042/0.039 at 0/5/10/15/20 dB (E = 1000, seed 42), against
0.018 (awgn) and 0.024 (rayleigh) for the 128-wide frozen arm in M2
(E = 5000, 3 seeds). The λ-shaped arm reaches 0.99 at **half** the width.

The failure also has a signature: the task loss falls (5025 → 1344 awgn,
5016 → 659 rayleigh) while the normalized reconstruction MSE explodes from
0.25 to ~1.3–1.6 × 10³. The wider network does not learn a better state; it
drifts into a degenerate high-gain mapping that the distance/terminal terms
weakly reward. Capacity is therefore **not** the explanation, and this is
evidence for the diagnostic claim rather than against it.

Limits: one seed, one width, one k, 3000 steps, E = 1000. Nothing here is a
claim about residual or attention architectures. Details:
`RESULTS_ARCH_2026-10-26.md`.

## 4. k = 24 multi-initialisation (A-14, running)

It needs 6 new training cells (heavy lambda, k = 24, awgn + rayleigh,
seeds 45/46/47) at the frozen 3000-step budget. Measured cost from A-12/A-13 is
76–81 min per cell under 2-way parallelism, i.e. ~4 h wall clock for 6 cells —
the earlier "~40 min/cell" figure was an estimate made before those runs
measured it.

Pre-registered in `PROTOCOL_AMENDMENTS.md` A-14 and committed
(`0753c89`) **before the first cell**, with runner `tools/run_k24_seeds.py`
(`verify` + a 20-step smoke that writes nothing came first):

```
cells   : 6 = lam_ctl_heavy x k=24 x {awgn, rayleigh} x seeds {45, 46, 47}
steps   : config.TASK_TRAIN_STEPS = 3000        (unchanged)
eval    : E=1000, 5-SNR grid -> 6 units, 30 slots
output  : results/K24_SEEDS/   (own root; no published denominator changes)
purpose : seed variance at the k where the crossover sits (seeds 45-47
          already exist for the instability arm in Wave-A A3, but not for
          k=24 heavy)
```

Statistics were declared up front: per-seed values plus mean and min–max range
across the three seeds; the range is **not** a confidence interval (n = 3), and
no seed may be dropped.

Do not describe k = 24 as having a variance estimate until this finishes.

## 5. Packaging and reproducibility

- `environment.yml` (conda, pinned to the same torch 2.4.1 CPU / numpy / matplotlib).
- `requirements.txt` unchanged (pip path still works).
- `Dockerfile`: the default command was `python tools/build_dashboard.py --open`,
  but the image ships no `results/` tree, so it could only fail. It now runs
  `python main.py validate-matrix` (dependency-free structural check) and the
  comment shows how to mount `results/` for the dashboard.
- **Docker was not built here**: no `docker` binary on this machine (same for
  `conda`). The image is therefore *unverified*; the command it runs is
  verified locally (`main.py validate-matrix` passes).
- Zenodo DOI cannot be minted by the agent: it needs the user's account. The
  repository is already self-contained (results + docs + evidence zip), so a
  GitHub release tarball can be deposited as-is.

## 6. Novelty framing

**What the evidence now supports.**

1. **Stability:** lambda-shaped control-aware shaping removes the low-bandwidth
   training instability of the frozen task arm (Wave-A, reproduced at k = 18
   and k = 54 in Wave-C with 0 NaN batches).
2. **Bandwidth crossover:** the task-oriented link beats the classical link
   where the classical link is rate-limited (k = 18: 0.99 vs 0.16), and loses
   where it is saturated (k = 54 at 20 dB: parity). This is a link-budget
   statement, and it survives a different channel code (A-11).
3. **Robustness:** the task metric degrades gracefully with CSI error while
   the reconstruction metric collapses (Wave-B, 350 cells, 70 sigma=0 gates).
4. **Deployability:** on a 3GPP UMa-aerial link the same policy keeps a
   15–21 dB transmit-power advantage over the classical link at k = 54 and is
   the only method that works at k = 18. This survives *training* on the
   dispersion (A-13): the operating point moves by ≤ 2 dB, inside sampling
   error, so the advantage is not an artefact of the training channel.

**What is still missing for a strong novelty claim.**

- A mechanism result — **partially delivered**
  (`RESULTS_MECHANISM_2026-10-27.md`): λ shaping bounds the encoder gradient
  norm by ~14 orders of magnitude (p99 ~60 vs ~10¹⁸) and every λ cell under the
  frozen protocol converges with zero non-finite gradient steps; capacity is
  ruled out (the exploding cells are the *wider* ones, and A-9's 256-wide frozen
  cell still fails); the trigger is rare outage episodes (A-12 explodes with
  the same λ arm, A-13 returns to O(1) when 3.78 % of episodes are resampled).
  Still open: **which** λ term does the work (needs λ_u-only / λ_v-only arms)
  and a curvature-level (sharpness) measurement.
- Seed variance at k = 24 (section 4) and the full 32-cell Wave-C grid.
- Doppler and time-correlated shadowing (A-13 trains and evaluates a
  block-fading dispersion only); the 3GPP training channel now exists but the
  outage tail had to be excluded, which must be stated with every number.
- A competitive 5G polar receiver (rate matching + CA-SCL) if polar is to be
  used as a serious comparator.

## 7. What changed, and what did not

**Changed (all additive):**
`project/models/{deepjscc_encoder,deepjscc_decoder,reconstruction_jscc}.py`
(width resolution; A-12 aerial branch), `project/channels/aerial.py` (A-12
dispersion sampler), `project/training/train_task_oriented.py` (channel
whitelist + block gain), `project/config.py` (two additive RNG streams),
new `tools/run_arch_ablation.py`, `tools/run_aerial_link.py`,
`tools/run_polar_baseline.py`, `tools/run_aerial_training.py`
(`--variant=a12|a13`), `tools/run_k24_seeds.py`, `tools/diag_aerial_checkpoints.py`,
`tools/analyse_training_stability.py`, `tools/make_figures_extensions.py`,
`environment.yml`, `Dockerfile`, four figures, six result docs,
A-9/A-10/A-11/A-12/A-13/A-14, and
`tests/test_aerial_link.py` (26 checks) + `tests/test_arch_width.py`
(16 checks) + `tests/test_aerial_training.py` (34 checks) +
`tests/test_k24_design.py` (23 checks).

**Not changed:** every file under `results/M1`, `results/M2`, `results/WAVE_A`,
`results/WAVE_B`, `results/WAVE_C`; all frozen config values, seeds, step
budgets, loss weights, channel conventions, evaluation seeds and SNR grids.
No cell was retrained. Existing checkpoints load and reproduce their stored
evaluations.

## 8. Declared but not run (accounting)

| block | declared | run | reason |
|---|---|---|---|
| Wave-C | 32 cells | 4 (A-8b screening) | time; screening was pre-registered |
| ARCH_ABL | 8 cells (2 arms x 2 ch x 2 seeds) | 2 (frozen arm x 2 ch x seed 42) | 256-wide wall clock |
| AERIAL_TRAIN | 2 cells x 1 seed | 2 (lam_ctl_heavy, k=18/54, umaaerial) | run as registered in full; **negative** (training diverged), kept as the record, not retrained |
| AERIAL_TRAIN_TRUNC | 2 cells x 1 seed (+1 pre-flight) | 2 (same design, outage tail > 20 dB resampled) | run as registered in full; pre-flight passed at the primary cap |
| k=24 inits | 6 cells | 0 | section 4 |
| polar at E=5000 | 50 conditions | 0 (E=1000 run) | protocol budget needs ~4x the wall clock |

## 9. Verify

```powershell
python tools/run_arch_ablation.py verify     # A-9 pre-flight (2 cells, width 256 real)
python tools/run_aerial_training.py verify   # A-12 pre-flight (3GPP training channel)
python tools/run_aerial_training.py verify --variant=a13   # A-13, tail cap +20 dB
python -m tests.test_arch_width              # 16 checks
python -m tests.test_aerial_link             # 26 checks
python -m tests.test_aerial_training         # 34 checks
python -m tests.test_k24_design              # 23 checks (A-14 design lock)
python tools/run_aerial_link.py --d2d=500 --h=50 --fc=2.0 --episodes=1000
python tools/run_polar_baseline.py run --episodes=1000
python tools/run_aerial_training.py status --variant=a13   # A-13 progress
python tools/run_k24_seeds.py verify         # A-14 pre-flight (6 cells, 30 slots)
python tools/run_k24_seeds.py status         # A-14 progress
python tools/analyse_training_stability.py   # mechanism table + JSON (read-only)
```
