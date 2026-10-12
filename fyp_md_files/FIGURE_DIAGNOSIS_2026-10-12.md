# Figure-Level Diagnosis — 2026-10-12

Supervisor review figures ko dekh kar di gayi thi; yeယ un findings ka structured analysis hai.
Per-figure observations + cross-checked numbers + impact on the Wave-A diagnosis plan.

---

## Fig. 10 — Training curves (k=24, seed 42) [supervisor's "alarming oscillation" figure]

**Observation:**
- Task loss oscillates between ≈30 and ≈5000 across all 3000 steps, *both* channels; no stable
  plateau even at step 3000. Recon (right panel) descends smoothly to a tight plateau by ≈1000 steps.
- Log-scale collapse from ~4000 start to ~30 by step ~100 (AWGN) shows fast initial descent, then
  the loss *re-expands* — this is not slow convergence, it is non-monotone instability.

**Cross-checked numbers (per-seed final losses, task, from train_report.json):**
final task loss ranges 136 → 5397 across the 30 task runs; and final loss does **not** predict
eval SR (k12 AWGN: seed44 loss 261 → SR 4.7%; seed43 loss 4459 → SR 4.3%; seed42 loss 2692 → SR 6.3%).
A converged training objective should correlate with downstream performance. It does not.

**Interpretation:** part of the oscillation is *by construction* — each batch draws SNR ~ U(0,20)
and a fresh episode set, so batch-to-batch loss variance is large. But the seed-decorrelation and
the non-settling envelope are genuine optimization instability, not just metric noise.

---

## Fig. 1 — M1 success-rate curves (k = 1..3)

- recon k=2/3: SR 0.24–0.31, *declining* at high SNR (0.31 @ 4–6 dB → 0.24 @ 20 dB). Flagged before
  in m1_analysis as unexplained non-monotonicity.
- task k=1..3: flat 0.02–0.05, essentially SNR-independent. min-distance ≈ 6.4–6.6 m —
  the UAV never meaningfully approaches the goal under the task decoder.

## Fig. 2 — M2 AWGN, 3-way (this is the sharpest figure)

- **recon is FLAT in k:** SR ≈ 0.23–0.27 for k = 12, 18, 24, 30, 54 — a 4.5× bandwidth increase
  buys *nothing*. Its MSE is also flat (0.0529 @ k=12 → 0.0461 @ k=54).
- **task is flat in k AND in SNR:** ≈ 0.02–0.03 everywhere.
- **digital:** 0.06 @ k=12 → 1.00 @ k ≥ 30 (wait: 0.55 @ k=30, 1.00 @ k=54; crossover vs recon
  between k=18 and k=24, confirmed by Fig. 4).

**Key inference:** recon's ceiling is bandwidth-independent → its error floor is set by
*training under-convergence* (fixed 5000-step budget, plateau reached), not by channel noise or
capacity. The oracle gate (noiseless, full info) reaches ~100% SR, so the plant/controller are
fine — the bottleneck is the learned decoder's remaining error, which more symbols cannot fix
if the model is already plateaued.

## Fig. 12 — Trajectories (k=24, 10 dB)

- recon: large looping paths; some reach goal, some wander/exit — *noise-like* decoding error.
- task: smooth, well-behaved curves that consistently end far from the goal with no successful
  approach in the shown samples — the signature of a **biased** goal-relative estimate
  (systematically wrong place), not of noisy jitter. Smooth-but-wrong = the decoder learned a
  consistent inversion, an optimization failure mode rather than channel sensitivity.

## Fig. 15 — Learned constellations (pre-normalization, k=24)

- recon encoder symbols: tight, similar-scale clusters.
- task encoder symbol 1: huge dynamic range (in-phase up to ≈6, quadrature ≈3.4) vs symbol 2's
  small spread. After per-codeword power normalization this concentrates power unevenly and makes
  symbol 1 noise-sensitive. This is a *structural* symptom of the unstable optimization:
  the task encoder allocates symbols pathologically.
- (Rendering footnote before thesis use: the two subplot titles overlap — "500 stat|ksencoder".
  Regenerate with constrained_layout.)

---

## Reconciled root-cause ranking (supersedes the earlier hypothesis list)

| # | Cause | Evidence | Confidence |
|---|---|---|---|
| H1 | **Task training instability / non-convergence** (possibly interacting with per-batch SNR mixing and grad-clip saturation reported in the 9D pilot) | Fig. 10 envelope; final-loss ⊥ eval-SR across 30 runs; seed scatter (evalSR 0.008–0.19 at k=24/30) | High |
| H2 | **Both neural models under-converged at the frozen budgets** → recon ceiling flat in k; task never settles | Fig. 2 flatness in k; recon MSE floor 0.044–0.053 despite 4.5× symbols; pilot plateau at smoke scale | High |
| H3 | **Task encoder symbol-allocation pathology** (downstream symptom of H1) | Fig. 15 asymmetric constellation | Medium |
| H4 | **Loss shaping (λ_u = λ_v = 0)** | No direct figure evidence yet; ablation still required to rule in/out | Open |
| H5 | recon's mild high-SNR SR decline (Fig. 1) | unexplained; likely train/test SNR-distribution mismatch (U(0,20) training vs fixed eval SNR) | Medium-low |

## Impact on Wave-A (amendments to fyp_md_files/WAVE_A_DESIGN.md)

Add **W-A4 — convergence/instability arm** (this now outranks λ ablations in priority):

1. **Extended-budget task training** at one representative cell (k=12 AWGN, 3 seeds):
   3000 → 12000 steps, same per-batch SNR law, to test "it just needed more steps" against
   "the objective cannot converge". ~4× runtime of one 9C task run per seed.
2. **Fixed-SNR curriculum variant:** train at fixed 10 dB (no U(0,20) per-batch mixing) for the
   same cell — isolates whether SNR mixing is what destabilizes the loss.
3. **LR-schedule probe:** 3000-step run with a cosine/step decay at the same cell (protocol amendment
   only; frozen arm untouched).
4. Plot per-step loss *envelope* (rolling min/mean/max over 200-step windows) for all arms —
   the direct, visual answer to Mam's Fig. 10 objection.

W-A1 (λ ablations) and W-A2 (weighted recon) remain, but their interpretation now includes
testing H4 explicitly. W-A3 (extra seeds) stays; the seed decorrelation found here makes it
mandatory, not optional.

**Framing for the paper (honest and stronger):** the frozen-budget result is real and stands
(recon > task at low k; digital wins with budget) — and the diagnosis chapter shows *why*:
optimization instability of the closed-loop task objective and under-converged neural arms at
CPU-scale budgets, with the byte-level protocol making every comparison reproducible. That is a
defensible Transactions-style diagnostic story, exactly as the supervisor's roadmap anticipated.
