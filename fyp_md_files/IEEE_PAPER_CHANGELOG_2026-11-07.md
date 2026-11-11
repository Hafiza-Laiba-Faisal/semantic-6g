# IEEE Paper Correction Changelog — 2026-11-07

**Paper:** `fyp_md_files/IEEE_PAPER_DRAFT.md` → rebuilt `IEEE_PAPER_DRAFT.docx`
**Audit source:** `fyp_md_files/IEEE_PAPER_AUDIT_2026-11-07.md`
**Approval:** P0 + P1 corrections approved by the author; P2 polish items 20–23 also applied where they were direct consequences of the approved corrections.
**Scope:** paper Markdown and rebuilt DOCX only. Thesis, experimental artifacts, frozen configurations, training/eval results, and figures were not modified.

This document records every substantive edit: the original claim, the corrected claim, and the file/field that supports the correction. Line references are to the pre-edit and post-edit Markdown.

---

## 1. Abstract and contribution bullets

| # | Original claim | Corrected claim | Evidence |
|---|---|---|---|
| A1 | "We find a bandwidth crossover: neural joint source-channel coding wins at k ≤ 18, the coded link wins at k ≥ 24, and the two are effectively tied at k = 54." | "We find a bandwidth crossover **for the reconstruction-oriented neural link**: it wins at k ≤ 18 and the coded link wins at k ≥ 24. **The shaped task-oriented link has a mean success of 0.996–0.999 at every measured bandwidth; at k = 54 it is within sampling error of the coded link, although the two are measured at different episode budgets.**" | M2 `results/M2/step11_m2_summary.json` (`pooled_rows`); Wave-C `results/WAVE_C/step_wave_c_summary.json`; K24 `results/K24_SEEDS/step_k24_summary.json`; M4 `results/M4/step13_m4_summary.json`. The old sentence silently switched from the reconstruction arm (Table I) to the shaped arm (Wave-C/K24). |
| A2 | "fails at 2–5% success" | "fails at 2–6% success" | M2 task arm range `0.0159–0.0563`; Wave-A frozen range `0.020–0.034`; rounded envelope 2–6%. |
| A3 | "adding both penalties recovers 0.997 at k = 12 and 0.993–0.999 at k = 18 and 54." | Same values, plus **"on AWGN and Rayleigh"**. | Wave-A λ-heavy 0.9966 (AWGN) / 0.9967 (Rayleigh) at k = 12; Wave-C λ-heavy 0.993–0.999 at k = 18/54. |
| A4 | "gradient measurements show a conditioning effect of roughly **fourteen orders of magnitude**" | "gradient measurements show a **median** conditioning gap of about **seven to ten orders of magnitude**" | `results/MECHANISM/step_stability_summary.json`: frozen p50 `2.9895e10` (AWGN) / `8.1149e7` (Rayleigh); λ-heavy p50 `3.3–4.1`. The old "fourteen" was not reproducible. |
| A5 | Contribution 2: "A measured bandwidth crossover… The neural link wins at k ≤ 18… a payload-maximal uncoded link… does not overtake the neural arm below k = 54." | "The **reconstruction-oriented** neural link wins at k ≤ 18… the **shaped task-oriented link sits at the task ceiling (mean 0.996–0.999)** at every measured k, so the crossover statement is about the reconstruction baseline, not about the task arm… does not overtake the **shaped** neural arm at any measured bandwidth." | Same artifacts as A1; M5 uncoded `results/M5/step_m5_summary.json` (0.884 at k = 54); Wave-C k = 54 0.998. |
| A6 | Contribution 3: "fails at 2–5%… recover 0.993–0.997… encoder gradient falls from about 10^14 to O(1)" | "fails at 2–6%… recover 0.993–0.999 **on the standard channels**… median encoder gradient falls from **3.0 × 10^10 (AWGN) and 8.1 × 10^7 (Rayleigh)** to **3.3–4.1**." | M2; Wave-A; Wave-C; `results/MECHANISM/step_stability_summary.json`. |
| A7 | "0.996–0.999" was used without an aggregation qualifier in the abstract, contribution 2, §IV-A, discussion, limitations, and conclusion, while §IV-B reports 0.993–0.999. | Qualified as **mean success** at each bandwidth; §IV-B retains the per-SNR-grid range **0.993–0.999**, and the limitations state "mean 0.996–0.999 (per-SNR minimum 0.993)." | Wave-C λ-heavy `pooled_rows`: channel means 0.9960–0.9990; per-SNR minimum 0.993 at k = 18 Rayleigh, 0 dB. Resolves an apparent abstract/§IV-B contradiction. |

---

## 2. Protocol and provenance

| # | Original claim | Corrected claim | Evidence |
|---|---|---|---|
| P1 | "Every mean reported below is recomputed from the stored per-SNR JSON artifacts rather than transcribed by hand." | "**Tables I, II, III, and IV are recomputed directly** from the stored per-SNR JSON artifacts; **diagnostic and ablation values cite their source artifacts in the text**." | The universal claim was too strong: A-12/A-13, Wave-B, and mechanism values are single-artifact diagnostics, not table recomputations. |
| P2 | Extension blocks "run at 1,000 episodes per unit and are labelled separately rather than pooled." | Added: "**most extension cells use a single training seed**." | Wave-C, A-10/A-17, A-12, A-13, LDPC are single-seed; M2 and Wave-A/K24 are multi-seed. |

---

## 3. Section IV-A — bandwidth crossover

| # | Original claim | Corrected claim | Evidence |
|---|---|---|---|
| R1 | "The reconstruction MSE is flat in k (**0.049–0.054**)" | "The **mean reconstruction MSE over the 0–20 dB grid** is flat in k (**0.044–0.047**)" | M2 `mse_mean_mean_over_seeds`, averaged over the five SNRs: 0.0468 / 0.0468 / 0.0452 / 0.0439 / 0.0448. The old 0.049–0.054 was the 0 dB slice. |
| R2 | No separate statement that the crossover concerns the reconstruction arm. | Added paragraph: shaped task arm reaches a mean of 0.997 at k = 12 and 0.996–0.999 at k = 18/24/54; "the benchmark cannot rank [the shaped link] against the coded link above k = 18." | Wave-A, Wave-C, K24. Prevents the Table I reconstruction crossover from being read as a shaped-task result. |
| R3 | Table I caption: "Success rate versus bandwidth on AWGN, pooled over three seeds (E = 5,000 per unit)." | Added: "**The task column is the frozen loss; the shaped task arm is reported in §IV-B.**" | Table I task column is M2 frozen `(1,0,0,2)`; shaped arm is Wave-C/K24 at E = 1,000. |
| R4 | Fig. 1 caption: "Success rate versus channel uses on AWGN. The crossover lies between k = 18 and k = 24…" | "Success rate versus channel uses **on AWGN at 0 dB (left) and 20 dB (right)**. The crossover lies between k = 18 and k = 24 **for the reconstruction arm; Table I reports the 0–20 dB mean**." | Figure source data; M2 per-SNR rows. The figure is two panels (0 and 20 dB), not a single mean curve. |
| R5 | "M4… reaches 0.974 on AWGN and 0.718 on Rayleigh at 0 dB… whereas the shaped neural link reaches 0.999 at k = 18." | "M4, **E = 5,000 per unit**… whereas the **shaped Wave-C arm** reaches 0.999 at k = 18 at 0 dB **(E = 1,000 per unit)**. The crossover is therefore a statement about how much bandwidth **the reconstruction objective** needs…" | M4 `conditions[]`; Wave-C `pooled_rows` at k = 18, 0 dB. Budgets are different and are now stated. |
| R6 | "the uncoded link reaches only 0.884 at k = 54 and **does not overtake the shaped neural arm at any measured bandwidth below k = 54**." | "…and **never overtakes the shaped neural arm (0.998 at k = 54)**. The headline 'digital wins at k ≥ 24' is true of the **coded baseline and the reconstruction arm** specifically." | M5 uncoded 0.884 vs Wave-C 0.998 at k = 54; the old phrase was self-contradictory (it implied a possible overtake at k = 54). |

---

## 4. Section IV-B — frozen loss, width-256 model, mechanism

| # | Original claim | Corrected claim | Evidence |
|---|---|---|---|
| B1 | "the task arm sits at **2–5%** success" | "**2–6%**" | M2 task 0.0159–0.0563; rounded envelope. |
| B2 | "A width-256 model still fails (**0.027 AWGN, 0.041 Rayleigh**), and its reconstruction MSE **explodes to about 10³** while the task loss falls: the failure is a loss pathology, not a capacity limit." | "…(**0.027 on AWGN and 0.041 on Rayleigh at 0 dB; means 0.025 and 0.042**), and its **training-time reconstruction-MSE diagnostic rises to about 1.3–1.6 × 10^3** while the task loss falls (**the evaluation-time MSE is 0.52–0.66**)." | ARCH_ABL `results/ARCH_ABL/step_arch_summary.json` (width 256) and `results/ARCH_ABL/eval/*/eval_0.json` (`aggregate_frozen_task.mse_mean` 0.5207 / 0.6600). The old text mixed the training diagnostic with evaluation MSE. |
| B3 | "Adding control and velocity penalties fixes it (Table II)." | "Adding control and velocity penalties fixes it **on AWGN and Rayleigh** (Table II)." | Wave-A λ-heavy 0.9966 / 0.9967; λ-light collapses on Rayleigh (0.019). |
| B4 | "A reconstruction auxiliary loss (0.243), fixed-SNR training (0.050), and longer training (0.035) do not recover the task, and **an untrained model scores 0.016**, so the metric is not trivially exploitable." | "…fixed-SNR training (**0.050 on AWGN**), and longer training (**0.035 on Rayleigh**) do not recover the task. **The success metric is a binary goal-reaching indicator (final distance ≤ 0.5 m), so it cannot be satisfied without reaching the goal; an untrained control was not persisted as an artifact and is therefore not reported.**" | M5 `results/M5/step_m5_summary.json`; no stored untrained artifact exists. The 0.016 row was unsupported and removed. |
| B5 | Table II caption: "Wave-A loss-component ablation at k = 12 (E = 1,000)." | "…**(E = 1,000 per seed × 3 seeds = 3,000 pooled; 85/90 slots complete, with two documented cell failures)**." | Wave-A `eval_slots_complete=85`, `eval_slots_expected=90`, `documented_failures` = `WAVE_A_long_12k_task_awgn_k12_seed42`, `WAVE_A_seedvar_frozen_task_awgn_k12_seed45`. |
| B6 | Table II frozen AWGN "**0.020–0.048**"; Rayleigh "**0.033**". | AWGN "**0.020–0.029**"; Rayleigh "**0.031–0.034 (mean 0.033)**". | Wave-A `seedvar_frozen` pooled rows: AWGN 0.0200–0.0285; Rayleigh 0.0310–0.0337. The old 0.048 was `fixsnr_10db` at 0 dB, not the frozen arm. |
| B7 | Table II row "Untrained random model 0.016". | Row removed. | No stored artifact; removing it avoids an unverifiable claim. |
| B8 | "The frozen arm's encoder gradient has p50 ≈ **3.0 × 10^14** and p99 ≈ **2.4 × 10^14**, with 75–100 non-finite steps and no convergence. The λ-heavy arm has p50 ≈ 3.3–4.1 and p99 ≈ 57–113, with zero non-finite steps and convergence in 10/10 cells. The separation is roughly **fourteen orders of magnitude**…" | "The frozen arm (**ARCH_ABL, width 256, k = 18, seed 42**) has p50 ≈ **3.0 × 10^10 on AWGN and 8.1 × 10^7 on Rayleigh**, p99 ≈ **2.4 × 10^18 and 5.0 × 10^18**, 75–100 non-finite steps, and **no convergence (0/2 cells)**. The λ-heavy arm has p50 ≈ 3.3–4.1 and p99 ≈ 57–113, zero non-finite steps, and convergence in **10/10 standard-channel cells (12/12 including A-13)**. The **median separation is about seven to ten orders of magnitude (Rayleigh/AWGN)** and the **p99 separation about seventeen**; the **maximum-to-maximum ratio reaches about fifteen**… **It does not by itself fix training under the full 3GPP dispersion (A-12), which is reported separately.**" | `results/MECHANISM/step_stability_summary.json` (`cells_with_history[]`: `grad_p50`, `grad_p99`, `nonfinite_grad_steps`, `converged`). Frozen = ARCH_ABL: p50 `2.9895e10` / `8.1149e7`, p99 `2.3945e18` / `5.0075e18`, nf 75/100, converged 0/2. λ-heavy = 10 standard cells + A-13 2 cells; A-12 is the exception. The old 3.0e14 / 2.4e14 were A-12 λ-heavy values, not the frozen arm. |
| B9 | Fig. 3 caption: "λ-shaping bounds the encoder gradient by about **fourteen orders of magnitude** and removes non-finite steps." | "λ-shaping lowers the encoder gradient by about **seven to ten orders in the median on AWGN/Rayleigh** and removes non-finite steps **there**; **under the full 3GPP dispersion (A-12, also λ-shaped) it does not, which motivates the resampling arm (A-13)**." | Same mechanism artifact; A-12 `results/AERIAL_TRAIN/step_aerial_training_summary.json` and A-13 `results/AERIAL_TRAIN_TRUNC/step_aerial_training_summary.json`. |

---

## 5. Section IV-C — 3GPP UMa-aerial

| # | Original claim | Corrected claim | Evidence |
|---|---|---|---|
| C1 | "the checkpoints sit at **0.02–0.03** success." | "…**0.01–0.03** success (**0.027 at k = 18, 0.012 at k = 54**)." | A-12 `train_rows[].eval_after_success_rate`: `0.02734375` / `0.01171875`. |
| C2 | "Resampling the >20 dB outage tail (**3.78% of episodes**) restores stability…" | "…(**3.78% of episodes, as documented in the A-13 amendment report**)…" | `fyp_md_files/GAP_ANALYSIS_RESPONSE.md` §1c (dated amendment, approved 2026-10-27) states 3.78%; no JSON field stores the computed fraction. The provenance is now explicit. |
| C3 | "moves the operating point by **at most 2 dB** relative to the Rayleigh-trained arm, **inside sampling error**." | "moves the operating point by **at most about 2.2 dB** relative to the Rayleigh-trained arm." | A-13 `gaps`: k18 0.99 −2.22 dB; k54 0.99 −1.09 dB; k54 0.90 −0.50 dB. The "inside sampling error" wording was removed because the gap is a link-budget shift, not a tested success-rate difference. |
| C4 | Fig. 4 caption: "A-10/A-17 required transmit SNR on the 3GPP UMa-aerial link. The shaped policy retains a 15–21 dB advantage across twelve geometries." | "**A-10** required transmit SNR on the **pinned** 3GPP UMa-aerial geometry (**d_2D = 500 m, h = 50 m, f_c = 2 GHz**) for SR ≥ 0.90 and ≥ 0.99 at k = 18 and 54. **The twelve-geometry sweep is reported in Table III.**" | A-10 `results/AERIAL/step_aerial_summary.json`; A-17 `results/AERIAL_GEOM/step_aerial_geometry_summary.json`. The figure is the A-10 pinned geometry, not the 12-row sweep. |

---

## 6. Section IV-D — code family and Doppler

| # | Original claim | Corrected claim | Evidence |
|---|---|---|---|
| D1 | "the 108-bit budget is obtained by **puncturing to an effective rate of 0.44**" | "the 108-bit budget is obtained by **truncating the mother code to its first 108 bits** (the 38.212 rate-matching chain is omitted), an effective rate of 0.44…" | LDPC implementation; `results/LDPC/step_ldpc_summary.json`. The operation is truncation, not puncturing. |
| D2 | "The full 5G NR polar chain behaves the same way: **identical to convolution at k = 12–30**, and weaker at k = 54 (0.909 against 0.995)." | "…**not significantly different from convolution at k = 12–30 (largest gap 0.016 at k = 30, with overlapping Wilson intervals)**, and weaker at k = 54 (0.909 against 0.995)." | `results/POLAR5G/step_polar5g_summary_E1000.json` (`pooled_5snr`, method `polar5g`): k54 AWGN 0.9092 vs conv 0.9952; k30 0.5596 vs 0.5436 (gap 0.016). |
| D3 | "Four comparators… therefore leave **the crossover** intact." | "…leave **the reconstruction crossover** intact." | Arm separation; consistent with A1. |
| D4 | "retraining under correlated fading at Δ = 0.033 gives 0.993 against 0.994 for the stored checkpoint, **inside the seed spread**." | "…(**single seed**) gives 0.993 against 0.994 for the stored checkpoint, **a difference of 0.001**." | `results/DOPPLER/step_doppler_summary.json`; the retraining is one seed, so "inside the seed spread" was not established. |
| D5 | "The i.i.d. convention is therefore **validated** for the neural arms rather than assumed." | "…**supported** for the neural arms rather than merely assumed." | Same artifact; the evidence is a 0.001 shift at one seed, which supports rather than proves the convention. |

---

## 7. Discussion and conclusion

| # | Original claim | Corrected claim | Evidence |
|---|---|---|---|
| E1 | "the coded link overtakes it at k = 54, where the two are **effectively tied**." | "the coded link overtakes **the reconstruction arm**. **The shaped task arm remains at the ceiling, and at k = 54 it is within sampling error of the coded link (0.998 against 0.995) under a smaller episode budget.**" | Wave-C 0.9986 at k = 54 (E = 1,000) vs M2 digital 0.9947 (E = 15,000 pooled). Wilson intervals overlap; budgets differ. |
| E2 | "the shaped neural arm is at the task ceiling (**0.99+**), so the benchmark **proves** that the policy works…" | "…at the task ceiling (**mean 0.996+**), so the benchmark **shows** that the policy works…" | Wave-A/Wave-C/K24 channel means ≥ 0.996; "proves" was too strong for a saturated benchmark. |
| E3 | "rare deep outages destabilise the rollout gradient…" | "…**even with λ-shaping (A-12)**, and resampling the tail restores stability…" | A-12 λ-heavy divergence (`results/AERIAL_TRAIN`); avoids implying the failure is only a frozen-loss issue. |
| E4 | Limitations: simulation-only, 2-D, fixed controller, perfect CSI, binary metric, no 3GPP Doppler/shadowing, LDPC missing rate matching. | Added: "**The task arm is saturated at a mean of 0.996–0.999 (per-SNR minimum 0.993), so the high-bandwidth comparison is inconclusive rather than a demonstrated win; most extension blocks use 1,000 episodes per unit and a single seed, so their differences are not tested for significance; Wave-A completed 85 of 90 slots with two documented cell failures, and the CSI sweep excludes one documented M2 failure.**" | Wave-A slot status; LDPC/POLAR5G E = 1,000; Wave-B `results/WAVE_B/step_wave_b_summary.json` (k=54 seed 43 excluded). |
| E5 | Conclusion: "At k ≤ 18 the neural link wins… at k ≥ 24 the coded link overtakes it and ties at k = 54… frozen loss fails at 2–5%… recover 0.993–0.997… **fourteen-order** gradient signature." | Conclusion separates the **reconstruction arm** from the **shaped task arm**; shaped ceiling **mean 0.996–0.999**; k54 tie 0.998 vs 0.995 with **different episode budgets and saturation**; **2–6%**; **0.993–0.999**; **median gradient gap about seven to ten orders**. | Same artifacts as A1–A4 and E1–E2. |
| E6 | Conclusion: Doppler "**validates** the i.i.d. fading convention." | Doppler "**supports** the i.i.d. fading convention." | Same as D5. |

---

## 8. Reference corrections

| Ref. | Original | Corrected | Source |
|---|---|---|---|
| [1] | "Study on Enhanced LTE Support for Aerial Vehicles", TR 36.777 | "**Enhanced LTE Support for Aerial Vehicles**", TR 36.777, 2017 | 3GPP TR 36.777 title page. |
| [8] | INFOCOM Workshops 2024, no pages | Added **pp. 1–6** | IEEE Xplore / Crossref record. |
| [9] | JSAC "early access, 2025" | **vol. 43, no. 7, pp. 2575–2588, 2025** | Crossref / publisher record. |
| [10] | JSAC "early access, 2025" | **vol. 43, no. 9, pp. 3041–3055, 2025** | Crossref / publisher record. |
| [11] | "arXiv:2504.18317, 2025" | **Proc. IEEE GLOBECOM 2025, pp. 1059–1064, DOI: 10.1109/GLOBECOM59602.2025.11432022** | Crossref / IEEE Xplore. The arXiv identifier was wrong for this paper. |
| [12] | TVT "early access, 2025" | **vol. 75, no. 2, pp. 3354–3359, 2026** | Crossref / publisher record. |
| [14] | "arXiv:2312.13182, 2024" | **IEEE Trans. Wireless Commun., vol. 23, no. 12, pp. 18903–18915, 2024** | Crossref / publisher record. |
| [19] | "arXiv:1910.04969, 2019" | **IEEE Wireless Commun. Lett., vol. 9, no. 6, pp. 861–865, 2020** | Crossref / publisher record. |
| [24] | Dissertation, Univ. Glasgow, 2025 | Added **DOI: 10.5525/gla.thesis.85353** | University of Glasgow thesis repository. |

References [2]–[7], [13], [15]–[18], [20]–[23] were checked and left unchanged.

---

## 9. P2 polish applied

| # | Change | Evidence |
|---|---|---|
| P2-1 | "2–5%" → "2–6%" in abstract, §IV-B, and conclusion. | M2 task range 0.0159–0.0563. |
| P2-2 | 3.78% outage-tail fraction now cites the A-13 amendment report. | `fyp_md_files/GAP_ANALYSIS_RESPONSE.md` §1c. |
| P2-3 | Wave-A's 85/90 slots and two documented failures stated in the Table II caption and limitations. | `results/WAVE_A/step_wave_a_summary.json`. |
| P2-4 | Doppler retraining explicitly labelled single-seed; "validated" → "supported". | `results/DOPPLER/step_doppler_summary.json`. |

---

## 10. Build and layout verification

**Build command**

```text
python tools/build_ieee_paper_docx.py --md fyp_md_files/IEEE_PAPER_DRAFT.md --out IEEE_PAPER_DRAFT.docx
```

**Word (print view) inspection**

| Check | Result |
|---|---|
| Page count | **6 pages** (US Letter; Word print view) |
| Word count | **5,554** |
| Sections | 2: single-column title/abstract; continuous section break into two-column body |
| Tables | 4 (Table I p. 3; Tables II–III p. 4; Table IV p. 5) |
| Figures | 4 inline shapes, each **3.35 in** wide (fits the 3.35 in column) |
| Figure pages | Fig. 1 p. 3; Figs. 2–4 p. 4 |
| Table integrity | Each table fits on a single page; no table split across pages |
| References | 24 entries, starting p. 6; all 24 cited in text |
| Old wrong values | Not present in DOCX text (checked: 3.0×10^14, 2.4×10^14, 0.049–0.054, 0.020–0.048, fourteen orders, 0.02–0.03, at most 2 dB, puncturing, below k = 54, 2–5%, 0.993–0.997) |

Note: an initial page-count query in Word's non-print view returned 18 pages; in print view with repagination the document is 6 pages. The 18-page figure was a view/pagination artifact, not the layout.

---

## 11. Final consistency audit

A read-only verification script re-derived the paper's numerical claims from the stored JSON artifacts and checked both the Markdown and the rebuilt DOCX.

| Area | Checks | Result |
|---|---|---|
| Table I (M2, AWGN, 3 seeds / 15,000 pooled) | 5 rows + seed/pool invariant | PASS |
| Table I MSE range and Rayleigh k = 54 | 3 | PASS |
| Table II (Wave-A frozen/λ-light/λ-heavy/reconstruction) | 9 | PASS |
| Table III (A-17 six printed rows, all fields) | 6 | PASS |
| Table IV (LDPC vs convolutional, 5 k values × 2 channels) | 5 | PASS |
| Wave-C / K24 / M4 / Polar5G | 4 | PASS |
| Doppler (fade duration, max Δ, digital k = 54) | 4 | PASS |
| Mechanism (frozen p50/p99/non-finite/convergence, λ-heavy) | 5 | PASS |
| A-12 / A-13 / ARCH_ABL eval | 4 | PASS |
| Corrected strings present in MD | 19 | PASS |
| Old wrong strings absent from MD | 12 | PASS |
| DOCX structural invariants (4 tables, 4 figures) | 2 | PASS |
| Corrected strings present in DOCX | 19 | PASS |
| Old wrong strings absent from DOCX | 12 | PASS |
| References 1–24 all cited | 1 | PASS |
| **Total** | **111** | **111 PASS / 0 FAIL** |

Test suite: the audit recorded `111 passed in 196.70s`; no code, configuration, or data was changed by this correction pass, so the test result remains valid.

---

## 12. Git status and diff

**Modified (intended):**

```text
 M IEEE_PAPER_DRAFT.docx
 M fyp_md_files/IEEE_PAPER_DRAFT.md
```

**New (intended documentation):**

```text
?? fyp_md_files/IEEE_PAPER_AUDIT_2026-11-07.md
?? fyp_md_files/IEEE_PAPER_CHANGELOG_2026-11-07.md
```

**Diff stat for the Markdown:**

```text
fyp_md_files/IEEE_PAPER_DRAFT.md | 71 ++++++++++++++++++++-------------------
1 file changed, 36 insertions(+), 35 deletions(-)
```

The DOCX is a binary rebuild; `git diff --stat` reports the size change. No thesis file, result JSON, figure, configuration, or tool was modified.

---

## 13. Residual caveats and non-claims

- **Thesis-only errors remain untouched**, as instructed: Table 5.2, §5.4, Table 5.4, and the Fig. 5.1/5.5 captions in `fyp_md_files/THESIS_MASTER_DOCUMENT.md` still carry the legacy values (0.935, 0.02–0.03, fourteen orders). They are outside this paper-correction scope.
- **3.78%** is documented in the A-13 amendment report but is not stored as a computed JSON field; the paper now says so.
- **The k = 54 tie** is a non-significant difference under different episode budgets (Wave-C E = 1,000 vs M2 digital E = 15,000 pooled) on a saturated metric. It is not a claim that the shaped task link is superior.
- **A-12** remains a reported negative result: full-dispersion training diverges even with λ-shaping. **A-13** restores stability but moves the operating point by up to ~2.2 dB; the paper does not claim it recovers a specific number of decibels or that the shift is "inside sampling error."
- **2–6%** is a rounded envelope of the observed 1.6–5.6% frozen-task range; the tables retain the exact per-cell values.
- **0.996–0.999** is the per-bandwidth mean range; the per-SNR minimum is 0.993 (k = 18 Rayleigh, 0 dB), which is the range quoted when the text refers to the full SNR grid.
- **No new experiments, protocol changes, retraining, or artifact edits** were performed. No IEEE Transactions readiness claim is made; this is a corrected six-page conference draft.
