# Claim-Correction Patch List — 2026-11-07

Scope: the six pre-submission fixes agreed after the external review, applied to
`fyp_md_files/IEEE_PAPER_DRAFT.md`, plus the matching legacy-number corrections in
`fyp_md_files/THESIS_MASTER_DOCUMENT.md`.

No experiment was re-run. No numeric result was changed. All new numbers were read
from the existing artifacts (`results/M2/step11_m2_summary.json` and the mechanism
summary), not from prose.

---

## A. Paper — `IEEE_PAPER_DRAFT.md`

| # | Location | Old | New | Reason |
|---|---|---|---|---|
| A1 | Abstract | "at k = 54 it is within sampling error of the coded link, although the two are measured at different episode budgets" | "at k = 54 its confidence interval overlaps that of the coded link, although the two are measured at different episode budgets and no formal equivalence test was performed" | Overlapping Wilson intervals are not an equivalence test. |
| A2 | Abstract | "needs 15–21 dB less transmit SNR than the digital link across twelve geometries" | "requires 15–21 dB less transmit power at matched reliability … at k = 54 (0.90) and 12–15 dB less at 0.99 over twelve geometric configurations; because one shared performance curve is mapped onto those configurations, this is a link-budget sensitivity analysis rather than independent validation" | Adds the k and reliability qualifiers, names the metric correctly, and removes the implied 12-channel validation. |
| A3 | §I contributions | Four bullets ("matched-budget benchmark", "measured crossover", "diagnosis", "3GPP assessment and two robustness checks") | Five bullets: (1) closed-loop task-oriented communication, (2) controlled empirical comparison incl. baseline scope, (3) diagnosis of shaping, (4) aerial link-budget interpretation, (5) training-stability investigation | Reviewer-approved contribution framing; adds an explicit fixed-rate baseline scope statement. |
| A4 | §II, last sentence | "None of these provides a matched-budget comparison…" | Same sentence plus "To our knowledge, and subject to the search protocol in Appendix A, no prior work combines closed-loop task-level evaluation, a matched-budget comparison, an aerial link-budget mapping, and a training-stability investigation in a single study." | Replaces any priority claim with a scoped, searchable statement. |
| A5 | §III-C, digital link | — (paragraph ended at the M3 boundary) | Added: baseline is deliberately fixed-rate (no adaptive MCS, no HARQ, no soft combining); part of the gap is attributable to absent rate adaptation; quantifying the residual gap is future work. | Pre-empts the "fixed-rate straw man" objection. |
| A6 | §III-E | — | New paragraph "**Comparison protocol.**" — identical channel uses and bandwidth; success vs power; required power at a target rate; 15–21 dB is a horizontal gap at matched reliability, quoted only at the stated rates; all powers on a normalised unit-gain scale, not EIRP. | Resolves the equal-power vs power-advantage contradiction. |
| A7 | §III-E | — | Same paragraph states antenna gain, noise figure and implementation losses are excluded. | Prevents reading the dB figures as hardware EIRP. |
| A8 | §IV-C, pinned geometry | "needs 87 dB of transmit SNR … The advantage is therefore 21 dB … and unbounded at k = 18" | "requires 87 dB of normalised transmit power … The gap in required power is therefore 21 dB … at k = 18 no finite gap is defined, because the coded link is payload-limited" | Correct metric name; removes the meaningless "unbounded". |
| A9 | §IV-C, Table III text | "The advantage does not collapse…" | "Every row reuses one shared effective-SNR performance curve … reports how the required power varies with geometry, not independent validation across distinct channel realisations … carries no information about geometry-dependent small-scale statistics. The required-power gap does not collapse…" | Correct A-17 framing. |
| A10 | Table III caption | "transmit SNR (dB)" | "required normalised transmit power (dB) … obtained by mapping one shared performance curve onto each geometry" | Consistency with A8–A9. |
| A11 | Table III column | "Adv." | "Gap" | Terminology consistency. |
| A12 | Fig. 4 caption | "required transmit SNR" / "twelve-geometry sweep" | "required normalised transmit power" / "twelve-geometry sensitivity sweep" | Same. |
| A13 | §IV-C, A-12/A-13 paragraph | "final losses of 4,320 and 210 and 311 and 203 non-finite gradient steps" | "final losses are 4,320 and 210, with 311 and 203 non-finite gradient steps respectively" | Fixes an ambiguous sentence; numbers unchanged. |
| A14 | §IV-C, A-13 | "moves the operating point by at most about 2.2 dB…" | Added: single training seed, single truncation threshold (+20 dB), stability result not a variance estimate, and it does not establish that outage tails are the sole cause of A-12. | Disclosure; prevents over-reading A-13. |
| A15 | §IV-D, LDPC | "the two families are statistically indistinguishable" | "do not differ significantly (overlapping Wilson intervals; no formal equivalence test was performed)" | Statistical wording. |
| A16 | §IV-D, Doppler | "indistinguishable from the frozen model" | "practically indistinguishable … at the reported resolution" | Same. |
| A17 | §V, para 1 | "within sampling error of the coded link …" | "its interval overlaps that of the coded link … overlapping intervals are not evidence of equivalence" | Same. |
| A18 | §V, 3GPP para | "87–108 dB of transmit SNR …" | "87–108 dB of normalised transmit power … the figures are not EIRP … stable at k = 54 … obtained by mapping one shared performance curve" | Metric name + EIRP caveat + shared-curve disclosure. |
| A19 | §V, threats | "no Doppler or time-correlated shadowing, and the LDPC arm omits…" | Added: fixed-rate comparator without adaptive MCS or HARQ; twelve-geometry sweep reuses one shared curve. | Completes the limitation list. |
| A20 | §VI, conclusion | "within sampling error …" | "its interval overlaps … no formal equivalence test was performed" | Same as A1. |
| A21 | §VI, conclusion | "needs 15–21 dB less transmit SNR … across twelve geometries" | "requires 15–21 dB less transmit power at matched reliability … at k = 54 … maps a single shared curve and should be read as a link-budget sensitivity analysis" | Same as A2. |
| A22 | New Appendix A | — | Literature search protocol: databases, window, three query blocks, screening, inclusion criteria, claim form. **Screening counts and search date are marked `[to be completed]`.** | Makes the "no prior work combines" sentence auditable rather than asserted. |

### Still required before submission

1. Fill the two placeholders in Appendix A (search date, identified/screened/included counts).
2. Confirm the page budget. Appendix A adds roughly half a page; if the venue caps at six pages, move it to a supplementary file and cite it as "supplementary material".
3. The DOCX has been regenerated from the corrected markdown (`python tools/build_ieee_paper_docx.py ...`). Re-run it after any further edit; do not edit the DOCX directly.

---

## B. Thesis — `THESIS_MASTER_DOCUMENT.md`

All values below were recomputed from `results/M2/step11_m2_summary.json`
(`pooled_rows`, mean of `success_rate_pooled` over SNR 0–20 dB) or taken from the
mechanism artifact. These are corrections of stale prose, not new results.

| # | Location | Old | New |
|---|---|---|---|
| B1 | Table 5.2, all five rows | 0.253 / 0.054 / 0.064; 0.237 / 0.025 / 0.165; 0.235 / 0.036 / 0.324; 0.262 / 0.021 / 0.629; 0.249 / 0.036 / 0.935 | 0.242 / 0.028 / 0.062; 0.254 / 0.018 / 0.164; 0.233 / 0.034 / 0.306; 0.279 / 0.021 / 0.566; 0.246 / 0.016 / 0.995 |
| B2 | §5.2 text | "digital link reaches 0.935 while reconstruction stays near 0.249"; "MSE … (0.049–0.054)" | "0.995 … near 0.246"; "(0.044–0.047)" |
| B3 | §5.4 | "p50 ≈ 3.0 × 10^14 and p99 ≈ 2.4 × 10^14 … roughly fourteen orders" | "p50 ≈ 3.0 × 10^10 (AWGN) and 8.1 × 10^7 (Rayleigh), p99 ≈ 2.4 × 10^18 and 5.0 × 10^18 … median separation roughly seven to ten orders, p99 about seventeen" |
| B4 | Fig. 5.5 caption and figure list | "bounds the encoder gradient by about fourteen orders" | "lowers the encoder gradient by about seven to ten orders in the median…" |
| B5 | §5.9 (A-13) | "at most 2 dB … — inside sampling error" | "at most about 2.2 dB; the confidence intervals overlap, but no formal equivalence test was performed"; adds single seed and single threshold |
| B6 | §5.11 (LDPC) | "statistically indistinguishable"; "**puncturing** the mother code" | "do not differ significantly (… no formal equivalence test)"; "**truncating the mother code to its first 108 bits** (the 38.212 rate-matching chain is omitted)" |
| B7 | §5.13 findings | "bounded by about fourteen orders" | "falls by about seven to ten orders of magnitude in the median" |
| B8 | §5.13 findings | "transmit SNR … (≤ 2 dB, inside sampling error)" | "normalised transmit power … (≤ 2.2 dB; intervals overlap, no formal equivalence test)" |
| B9 | §5.13 findings | "statistically indistinguishable … consequence of puncturing" | "do not differ significantly … follows from truncating the mother code without the 38.212 rate-matching chain" |
| B10 | §5.15 discussion | "statistically indistinguishable"; "puncturing a rate-0.19 mother code" | "do not differ significantly (overlapping intervals; no formal equivalence test)"; "truncating … to its first 108 bits" |
| B11 | §5.15 discussion | "both came back inside sampling error" | "both came back either non-significant (overlapping intervals, with no formal equivalence test) or in the thesis's favour" |
| B12 | Ch. 6 exclusions | "because the mother code is punctured" | "because the mother code is truncated to its first 108 bits (no rate matching)" |
| B13 | Ch. 7 / FAQ | "statistically indistinguishable"; "by puncturing" | "does not differ significantly (overlapping intervals)"; "by truncating the mother code to its first 108 bits" |
| B14 | Executive summary | "2–5%"; "fourteen orders"; "statistically indistinguishable" | "2–6%"; "seven to ten orders in the median"; "does not differ significantly … (overlapping intervals)" |
| B15 | Ch. 8 conclusion | "2–5%"; "fourteen orders"; "transmit SNR" | "2–6%"; "falls by about seven to ten orders in the median"; "normalised transmit power" |
| B16 | Ch. 8 conclusion | "transmit-power advantage is stable at 15–21 dB" | "required-power gap at k = 54 is stable at 15–21 dB … although it maps one shared performance curve onto each geometry rather than running independent channels" |
| B17 | Ch. 9 implementation | "is statistically indistinguishable from the convolutional baseline at k = 12–30" | "does not differ significantly … (overlapping intervals)" |

### Not changed (deliberately)

- Thesis aerial ranges such as "74–108 dB", "78–113 dB" and "89–127 dB" were left
  as written. They could not be reconciled against the A-17 table without another
  derivation pass; they should be re-derived from `results/AERIAL_GEOM/` before the
  thesis is finalised.
- The thesis still describes the 3GPP layer as a full dispersion model in places.
  The paper's explicit "large-scale layer only" wording is the correct one; the
  thesis wording should be aligned in the same pass as the ranges above.
