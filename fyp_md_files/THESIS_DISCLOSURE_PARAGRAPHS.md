# Thesis Disclosure Paragraphs
**Ready to copy-paste directly into thesis chapters**
*Generated: 2026-10-21 — based on actual implementation in project/channels/*

---

## DISCLOSURE 1 — Channel Model and Its Idealizations
**Goes in: Chapter 3 (System Model) or Chapter 2 (Methodology)**

> The Rayleigh fading channel is modelled as slow (block) fading, where a
> single complex gain h ~ CN(0, 1) is drawn independently for each transmitted
> packet of k symbols. The receiver is assumed to have perfect channel state
> information (CSI), and zero-forcing (ZF) equalization is applied to recover
> the transmitted symbols. No channel state information is available at the
> transmitter. These four idealizations — perfect receiver CSI, ZF
> equalization, packet-granularity block fading, and open-loop power control —
> are standard in the DeepJSCC literature [Bourtsoulatze et al. 2019,
> Kurka and Gunduz 2020] and are applied identically to all three compared
> methods, ensuring a fair comparison. The robustness of these results to
> imperfect CSI is investigated in Section [Wave-B], where estimation error
> epsilon ~ CN(0, sigma^2) is injected at the receiver; success rate is found
> to be insensitive up to sigma = 0.4, while reconstruction MSE degrades
> by approximately 18% — a finding that highlights the coarseness of
> success rate as a metric for detecting channel-quality degradation.

**Table to add alongside this paragraph:**

| Idealization | Value used | Justification |
|---|---|---|
| Receiver CSI | Perfect (no error) | Standard in DeepJSCC literature |
| Equalization | Zero-Forcing (ZF) | Simple, closed-form, identical for all methods |
| Fading granularity | One gain per k-symbol packet | Block fading — matches DeepJSCC convention |
| Transmitter CSI | None | Open-loop; no feedback assumed |

---

## DISCLOSURE 2 — Digital Baseline Decoder
**Goes in: Chapter 3 (Digital Baseline subsection)**

> The digital baseline employs scalar quantization (B bits per state
> component), a convolutional code with constraint length K = 7 and rate
> R_c = 1/2, QPSK modulation with Gray labeling, and soft-decision Viterbi
> decoding. The channel-use budget is matched exactly: the frame length
> satisfies 6B + 6 = k channel uses, where the additional 6 uses account for
> encoder termination; any (k, B) combination that does not satisfy this
> equation exactly is declared infeasible and excluded from comparison rather
> than silently widened. This strict budget enforcement is a methodological
> contribution of this work; many published comparisons relax this constraint
> implicitly. The SC (successive cancellation) decoder is used as a
> low-complexity reference, consistent with baseline choices in the 6G
> semantic communication literature [Zhou et al. 2022, Guo et al. 2024].
> CRC-aided successive cancellation list (SCL) decoding, the accepted
> reference for 5G NR control channels, would improve digital performance
> at finite block lengths; however, since the primary comparison in this
> work is between semantic and digital communication paradigms rather than
> between polar decoders, the SC baseline is sufficient to establish the
> bandwidth-crossover finding.

---

## DISCLOSURE 3 — Statistical Methodology
**Goes in: Chapter 4 (Evaluation Protocol) or Chapter 3**

> All three methods are evaluated on bitwise-identical episodes: initial
> positions, goal locations, channel noise, and fading realizations are
> pre-generated once per (k, channel, SNR, seed) unit and shared across
> methods. This paired evaluation design eliminates episode-sampling variance
> from all pairwise comparisons, enabling the use of paired statistical tests.
> Goal-reaching success (a binary outcome per episode) is compared using
> the McNemar test (two-sided, continuity-corrected chi-squared(1));
> continuous metrics (final distance, average distance, control effort)
> are compared using the paired t-test. Wilson 95% confidence intervals
> are reported on all success rates. Three independent training seeds
> {42, 43, 44} are used; pooled results aggregate 15,000 episodes per
> condition. Per-seed mean and standard deviation are reported alongside
> pooled estimates to make training-seed variability visible; the pooled
> Wilson interval reflects variability over training replicas, not
> statistically independent episodes.

---

## DISCLOSURE 4 — Training Reproducibility
**Goes in: Chapter 3 (Training Protocol) or Appendix**

> All training and evaluation is performed on CPU only, ensuring
> deterministic execution. For a given (seed, configuration) pair, training
> produces bit-identical results across runs. The optimizer state is not
> saved between sessions (a documented limitation); recovery from interruption
> retrains from the fixed random seed, which produces the same final
> checkpoint under the frozen 9C protocol (5000 reconstruction steps /
> 3000 task steps, Adam lr=1e-3, gradient clip 1.0, no early stopping,
> training SNR ~ U(0, 20) dB per batch). The final checkpoint — not the
> best validation checkpoint — is used in all evaluations, consistent with
> the protocol freeze.

> An earlier training run was interrupted by system shutdown during
> execution; the corresponding optimizer state was therefore incomplete.
> Per the frozen protocol, affected runs were retrained deterministically
> from their fixed seeds. The archive of the interrupted run is preserved
> at results/_archive/ for audit purposes. The current results are from
> the fully completed, uninterrupted training run.

---

## DISCLOSURE 5 — Wave-A Diagnostic Finding (Task Loss Lambda)
**Goes in: Chapter 5 (Analysis) or Discussion section**

> The frozen task training protocol uses loss weights lambda_d = 1,
> lambda_T = 2, lambda_u = lambda_v = 0 — meaning control effort and
> velocity penalties are disabled. Wave-A ablation experiments show that
> enabling control effort penalties (lambda_u, lambda_v > 0) recovers
> near-perfect success rate (SR = 0.997 at k=12, AWGN and Rayleigh) from
> the frozen-protocol baseline of SR = 0.02-0.05. This finding identifies
> the root cause of task encoder underperformance in the main experiment
> (M1, M2) and motivates future work on loss shaping for task-oriented
> semantic communication. The frozen-protocol results (M1, M2) are
> preserved without modification and represent a valid comparison under
> the published frozen configuration; the Wave-A finding is presented as
> a diagnostic contribution.

> Caveat: the lambda-shaped models were trained and evaluated only at
> k = 12. Whether the improvement generalises to other bandwidth regimes
> (k = 1, 2, 3 or k = 18, 24, 30, 54) remains an open question and is
> identified as future work.

---

## DISCLOSURE 6 — Wave-B CSI Robustness Finding
**Goes in: Chapter 5 (Results) — Wave-B section**

> To assess sensitivity to imperfect channel estimation, a CSI error
> sigma in {0.0, 0.05, 0.1, 0.2, 0.4} is injected at the Rayleigh
> receiver: h_hat = h + epsilon, epsilon ~ CN(0, sigma^2). The receiver
> equalizes using h_hat only. All 350 evaluated cells (14/15 units,
> sigma=0 bit-exactly matching frozen M2) show:
>
> (i)  Success rate is practically insensitive to sigma up to 0.4 for
>      all three methods and all k values. The absolute change in SR
>      from sigma=0 to sigma=0.4 is less than 0.01 for reconstruction
>      and task encoders.
>
> (ii) Reconstruction MSE increases consistently by approximately 18%
>      across all k values at sigma=0.4, confirming that the channel
>      estimation error does degrade signal quality, but this degradation
>      is too small to affect the binary success-or-failure outcome at
>      the episode level.
>
> (iii) Digital baseline SR shows a small anomalous increase with sigma
>       (0.384 to 0.395 at k=24). This is hypothesised to arise from
>       ZF gain shifts interacting with B=1 hard-decision thresholds,
>       but per-episode analysis is required to confirm this. No
>       robustness claim is made for the digital baseline on this basis.

---

## HOW TO USE THIS FILE

1. Copy each DISCLOSURE block into the relevant thesis chapter.
2. Replace [Section X] / [Wave-B] with your actual section numbers.
3. Replace citation placeholders with your bibliography entries.
4. The table in DISCLOSURE 1 can be formatted as a LaTeX table directly.
5. DISCLOSURES 5 and 6 can be merged into one "Diagnostic Results" section.

*All numbers in this file are taken directly from:*
- `results/M1/step10_m1_summary.json`
- `results/WAVE_A/step_wave_a_summary.json`
- `results/WAVE_B/step_wave_b_summary.json`
- `fyp_md_files/RESULTS_WAVE_A_WAVE_B_2026-10-21.md`
