# IEEE Research-Project Evaluation and Remaining Work

**Project:** Task-Oriented Deep Joint Source-Channel Coding for UAV Navigation under Noisy and Fading Wireless Channels, with a 3GPP UMa-Aerial Link-Budget and Dispersion Extension

**Repository:** `https://github.com/Hafiza-Laiba-Faisal/semantic-6g.git`

**Evaluated commit:** `82eae81` (the evaluation pass itself is committed separately)

**Date:** October 2026

**Evaluator:** Independent IEEE-style research audit (AI-assisted)

---

## Roman Urdu khulasa

Ye project technically strong aur honest hai. Iska main result — bandwidth crossover, lambda-shaping ka fix, aur 3GPP UMa-aerial extension — data se supported hai. Thesis aur figures document complete hain, aur M3/M4/M5 ke labels sahi hain: M3 structural hai, M4 secondary hai, M5 deferred hai.

Is pass mein jo cheezein fix ki gayin: `fig15_constellation.png` ke overlapping titles, `fig_aerial_training.png` ka A-13 legend/title mismatch, thesis mein **List of Figures** aur **List of Tables**, aur multiple-comparison ka honest caveat.

Jo cheezein baqi hain: **M5 ablations** (task-loss component removal, uncoded digital, cross-channel) run nahi hue; 3GPP model sirf ek geometry par hai aur Doppler/time-correlated shadowing nahi hai; IEEE conference paper ke liye numeric citations aur 6-page compression chahiye; PDF export Word COM se hang hota hai. Thesis submission ke liye ready hai, IEEE paper ke liye nahi.

---

## 1. Executive summary

**Verdict for the thesis:** Strong and submission-ready after minor formatting checks. The work has a clear question, a matched-budget design, a reproducible protocol, honest negative results, and a first-order 3GPP aerial extension. The main claims are supported by stored artifacts.

**Verdict for an IEEE conference paper:** Not yet ready. The thesis uses author-year citations, is 54 pages long, and carries more material than a 6-page paper can hold. A paper version needs numeric IEEE citations, a tighter contribution framing, 5–6 selected figures, and a 3–4 table results section.

**Biggest scientific gap:** M5 was never executed. The thesis correctly labels it deferred, but a reviewer can still ask for the task-loss component removal, the uncoded-digital reference, and the cross-channel training. These are the remaining experiments, not missing write-up.

**Biggest modelling gap:** The 3GPP UMa-aerial extension uses one geometry (`d2D = 500 m`, `h = 50 m`, `fc = 2 GHz`) and no Doppler or time-correlated shadowing. It is a large-scale plus dispersion layer, not a full 3GPP simulator. The thesis states this, but a multi-geometry evaluation would make RQ3 stronger.

**Biggest reproducibility gap:** The local environment does not have `pytest` installed in either the venv or the system Python, so the reported 57-test suite could not be re-run in this pass. The repository reports a green CI run; a fresh environment should install the test dependencies and re-run the suite.

**What was fixed in this pass:**
1. `fig15_constellation.png` — overlapping panel titles removed; titles shortened and layout tightened.
2. `fig_aerial_training.png` — the A-13 curve is now plotted (purple diamonds), the legend includes A-13, and the title matches the content.
3. `fyp_md_files/FIGURES_DOCUMENT.md` — Figure 33 description updated to match the new A-13 curve.
4. `THESIS_MASTER_DOCUMENT.md` — added **List of Figures** and **List of Tables**.
5. `THESIS_MASTER_DOCUMENT.md` — added an explicit no-family-wise-correction caveat to the statistical methodology.
6. `fig13_mse_vs_snr.png` — moved the 15-entry legend below the axes; the caveat no longer collides with the x-axis labels.
7. `fig14_rate_distortion.png` — separated the two-line tick labels, the bandwidth-ratio label, and the caveat; the bottom of the figure is now readable.
8. `tools/build_thesis_docx.py` — each numbered list now gets its own numbering instance, so the limitations and future-work lists restart at 1 instead of continuing from the previous list.
9. Regenerated both `.docx` files and refreshed the Word fields, TOC, and page numbers.

---

## 2. Evidence base

The audit used the following artifacts:

| Artifact | Use |
|---|---|
| `fyp_md_files/THESIS_MASTER_DOCUMENT.md` | Thesis source, 54 pages after this pass |
| `fyp_md_files/FIGURES_DOCUMENT.md` | 38-figure companion |
| `fyp_md_files/PROTOCOL_AMENDMENTS.md` | A-1…A-15 definitions and exceptions |
| `fyp_md_files/05-implementation-audit.md` | M1–M5 matrix, digital packet invariant |
| `fyp_md_files/RESEARCH_ROADMAP.md` | Pipeline status and experiment accounting |
| `fyp_md_files/GAP_ANALYSIS_RESPONSE.md` | A-9…A-15 response summary |
| `results/M1`, `results/M2` | Main benchmark summaries |
| `results/M3/m3_report.json` | M3 infeasibility cells |
| `results/M4/step13_m4_summary.json` | M4 B = 8 reference |
| `results/WAVE_A`, `results/WAVE_B`, `results/WAVE_C` | Ablation and robustness waves |
| `results/AERIAL`, `results/AERIAL_TRAIN`, `results/AERIAL_TRAIN_TRUNC` | 3GPP link budget and dispersion training |
| `results/POLAR`, `results/POLAR5G`, `results/K24_SEEDS`, `results/ARCH_ABL` | A-9…A-15 extensions |
| `tools/build_thesis_docx.py`, `tools/make_figures_semantic.py`, `tools/make_figures_extensions.py` | Document and figure generation |

All numbers in this report were checked against the stored JSON or the thesis source; none were typed from memory.

---

## 3. Gap matrix

| Area | What exists | IEEE expectation | Gap | Priority | Action |
|---|---|---|---|---|---|
| Novelty and contribution | Matched-budget three-way benchmark, crossover, λ-shaping mechanism, 3GPP link budget | Clear, defensible contribution | Incremental domain transfer, honestly framed | P2 | Keep the honest framing; do not claim a new JSCC paradigm |
| Research questions | RQ1–RQ3, answered in Ch. 5 and Ch. 8 | RQs mapped to evidence | RQ3 uses one 3GPP geometry | P1 | Add 2–3 geometries or state the single-geometry limit clearly |
| Literature review | Ch. 2, 24 references | Coverage, recency, positioning | Some 2025 preprints; some DOIs missing | P2 | Add DOIs where available; keep preprints labelled |
| Methodology | Ch. 3–4, frozen protocol, paired episodes, digital packet invariant | Rigorous and reproducible | 3GPP model lacks Doppler and time-correlated shadowing | P1 | State limits (done); add a multi-geometry A-10 extension |
| Experimental design | M1–M4, Waves A–C, A-9…A-15 | Baselines, ablations, seeds, CIs | M5 never run; k = 24 at ceiling | P1/P2 | Run M5 if required; add a harder task for non-ceiling variance |
| Statistical validity | McNemar, Wilson 95%, paired t, three seeds | Correct paired tests, uncertainty | No family-wise correction | P2 | Caveat added; optional Holm correction for the main table |
| Results and discussion | Ch. 5–6, negative results kept | Claims supported by data | M3/M4/M5 labels must travel with every table | P0 (labels) | Labels are present; keep them in the paper version |
| Figures and tables | 11 thesis figures, 21 tables, 38-figure companion | Legible, numbered, captioned | Four figure defects found | P0 | Fixed `fig13`, `fig14`, `fig15`, and `fig_aerial_training`; List of Figures/Tables added |
| References | Author-year, 24 entries | IEEE numeric for a paper; consistent for a thesis | Thesis style is APA-like, not IEEE numeric | P1 for paper | Convert to numeric `[1]` only for the IEEE paper |
| Writing and structure | Title, declaration, certificate, abstract, TOC, abbreviations, nomenclature, 8 chapters, appendices, viva Q&A | Complete thesis structure | Long Table 4.2; dense figures | P2 | Check the Word layout at 100% zoom |
| Reproducibility | Frozen config, deterministic seeds, Docker/CI, 57 reported tests | Code, data, commands, tests | `pytest` not installed locally in this pass | P1 | Install test dependencies and re-run the suite |
| Ethics and declarations | Simulation-only disclosure, negative results kept, no fabricated numbers | Honest reporting | None material | Done | Keep the negative results visible |
| IEEE paper readiness | Thesis only | 6-page paper, numeric citations, selected figures | No paper draft | P1 | Produce a paper plan and a 6-page draft |

---

## 4. Detailed findings

### 4.1 Strengths

1. **The comparison is genuinely matched.** All methods share episodes, noise, fading, SNR, and channel uses. The digital packet invariant `6B + 6 ≤ k` is enforced rather than silently relaxed.
2. **The negative results are kept.** A-9 (capacity ablation) and A-12 (full-dispersion training) are reported as negative, not retrained away. This is a strong signal of research integrity.
3. **The λ-shaping diagnosis has a mechanism.** The thesis does not stop at “λ works”; it measures the encoder gradient and shows a roughly fourteen-orders-of-magnitude conditioning effect.
4. **The 3GPP extension is first-class.** Chapter 3.8 defines the model, Chapter 5.8 reports the link budget, and Chapter 5.10 reports the dispersion-training result. It is not an appendix afterthought.
5. **The protocol is documented.** A-1…A-15 are dated, pre-registered, and tied to artifacts. The thesis can defend every deviation.
6. **The figures companion is complete.** All 38 figures have a description, a “what to notice” list, and a best placement.

### 4.2 Weaknesses and risks

1. **M5 is missing.** The frozen protocol declared M5 but never ran it. The thesis labels it deferred, which is honest, but a reviewer may still ask for the task-loss component removal, the uncoded-digital reference, and the cross-channel training.
2. **The 3GPP model is narrow.** One geometry, no Doppler, no time-correlated shadowing, no altitude-dependent small-scale fading. The transmit-SNR numbers are a link-budget statement, not a hardware-feasibility claim.
3. **A-12 diverges.** The full-dispersion training result is negative. A-13 restores stability by resampling the >20 dB tail, but it does not move the operating point. A principled full-dispersion training scheme is still open.
4. **The metric is at the ceiling at k = 24.** The seed-variance study (A-14) can rule out a large initialisation dependence, but it cannot resolve small differences between good methods. A harder task is needed for that.
5. **The thesis is not an IEEE paper.** Author-year citations, 54 pages, 11 figures, and 21 tables are appropriate for a thesis but not for a 6-page IEEE conference paper.
6. **Local test execution is not reproducible out of the box.** `pytest` is not installed in the venv or the system Python in this environment. The CI report is green, but a fresh local run needs the test dependencies.
7. **PDF export is unresolved.** Word COM field update and page counting work; PDF export hangs. The `.docx` files are usable, but a PDF submission still needs a different export path.

### 4.3 Honesty audit

| Label | Status | Evidence | Required handling |
|---|---|---|---|
| M3 | Structural infeasibility, not a measured result | `results/M3/m3_report.json`; `k = 1,2,3`, required 12, deficits 11/10/9 | Never quote M3 as a success rate |
| M4 | Secondary B = 8 reference, not a primary cell | `results/M4/step13_m4_summary.json`; k = 54, E = 5,000 | Keep the “secondary” label in every table and figure |
| M5 | Deferred, not run | No `results/M5`; `05-implementation-audit.md` (`M5_STATUS = DEFERRED`); `PROJECT_STATUS.md` | Do not invent M5 numbers |
| A-9 | Negative capacity ablation | `results/ARCH_ABL`; width 256 still fails | Keep as negative |
| A-12 | Negative full-dispersion training | `results/AERIAL_TRAIN`; final losses 4,320 / 210; 311 / 203 non-finite steps | Keep as negative; do not retrain for convenience |
| A-13 | Stable tail-resampled variant | `results/AERIAL_TRAIN_TRUNC`; 0 non-finite steps; ≤ 2 dB operating-point change | Do not claim a specific dB recovery |

---

## 5. What was fixed in this pass

### 5.1 `fig15_constellation.png`

**Problem:** The two panel titles overlapped (`recon encoder ...` and `task encoder ...`), making the figure unreadable at thesis width.

**Fix:** Shortened the titles to “recon encoder (k=24), 500 states” and “task encoder (k=24), 500 states”, reduced the title font, added padding, and applied `tight_layout` with a reserved top margin.

**Result:** The titles no longer overlap; the figure is legible at 5.9 inches.

### 5.2 `fig_aerial_training.png`

**Problem:** The title said “A-12 vs A-13”, but the legend only showed A-12 and the Rayleigh baseline. The A-13 curve was skipped because the summary did not contain `aerial_train_max_offset_db`.

**Fix:** Added a documented fallback (`cap = 20.0`, the “trunc20” variant) so the A-13 curve is plotted. The legend now includes “trained on 3GPP, outage tail >20 dB resampled (A-13)”.

**Result:** The figure now shows A-12 (red squares, near zero), A-13 (purple diamonds, on the Rayleigh baseline), Rayleigh (blue circles), and digital (green dotted). The title, legend, and content match.

### 5.3 Figure 33 description

**Problem:** The figure companion said “the A-13 resampled variant is reported in the results table; this figure is the A-12 diagnostic” and “Do not claim that A-13 is plotted here; it is not.” That was true before the fix, but false after.

**Fix:** Rewrote the entry to describe the A-13 curve, the >20 dB cap, and the “stable but no operating-point shift” claim.

### 5.4 List of Figures and List of Tables

**Problem:** The thesis had a TOC but no List of Figures or List of Tables.

**Fix:** Added both after the Nomenclature, with all 11 figures and all 17 numbered tables.

**Result:** The TOC now lists “List of Figures” (page 11) and “List of Tables” (page 12).

### 5.5 Multiple-comparison caveat

**Problem:** The thesis reported per-comparison McNemar and paired-t tests without a family-wise correction.

**Fix:** Added an explicit caveat: no Holm/Bonferroni correction is applied; the main effects are very large (0.02 vs 0.99), so correction would not change the qualitative conclusion; p-values are descriptive for the large effects.

**Result:** The statistical section is now honest about the scope of the inference.

### 5.6 `fig13_mse_vs_snr.png` and `fig14_rate_distortion.png`

**Problem:** Both figures had bottom-edge collisions. In `fig13`, the 15-entry legend sat on top of the task curves and the caveat touched the x-axis labels. In `fig14`, the two-line tick labels, the “bandwidth ratio” label, and the caveat overlapped each other.

**Fix:** Moved the `fig13` legend below the axes in five columns, reserved a bottom margin, and placed the caveat in its own band. In `fig14`, removed the per-axis x-label, added one shared figure-level x-label, increased the figure height, and reserved separate bands for the tick labels, x-label, and caveat.

**Result:** Both figures are readable at thesis width; no text overlaps the data or other text.

### 5.7 Regenerated documents

| Document | Pages | Words | Tables | Figures |
|---|---:|---:|---:|---:|
| `THESIS_MASTER_DOCUMENT.docx` | 54 | 14,352 | 21 | 11 |
| `FIGURES_DOCUMENT.docx` | 53 | 5,525 | 3 | 38 |

Word fields, TOC, and page numbers were refreshed with Word COM. PDF export was not attempted because the known COM export path hangs.

---

## 6. Remaining work

### 6.1 P0 — must fix before submission

There is no scientific P0 left in the thesis after this pass. The remaining P0 is logistical:

1. **Produce a PDF** through a path that does not hang (Word “Save As PDF” manually, LibreOffice, or a docx-to-PDF service). The `.docx` is the source of truth.
2. **Check the Word layout** at 100% zoom, especially Table 4.2, Table 5.10, and the 3GPP figure. Wide tables may need landscape pages or smaller font.
3. **Insert signatures and dates** on the Declaration and Certificate pages before printing.

### 6.2 P1 — should fix

1. **Run M5 if the supervisor requires it.** The design is specified below. Do not run it without a dated protocol amendment and user approval.
2. **Add a multi-geometry A-10 extension.** Evaluate the frozen policy at `d2D ∈ {200, 500, 1000} m`, `h ∈ {50, 100} m`, and `fc ∈ {2, 3.5} GHz`. This is evaluation-only and cheap; it turns RQ3 from one point into a trend.
3. **Install and run the test suite locally.** `pip install pytest` in a clean environment, then `python -m pytest tests/ -q`. Record the result in the thesis.
4. **Produce an IEEE paper plan.** Numeric citations, 6-page compression, 5–6 figures, 3–4 tables.
5. **Add DOIs** to references where available (Khuwaja 2018, McNemar 1947, Wilson 1927 are the easiest).

### 6.3 P2 — nice to have

1. **Doppler and time-correlated shadowing.** Add a small-scale fading layer and a shadowing correlation model; this is a simulator change, not a thesis rewrite.
2. **A principled full-dispersion training scheme.** A-12 diverges; A-13 only resamples the tail. A curriculum or gradient-clipping study would close this.
3. **A harder task.** Tighten the goal radius, shorten the horizon, or add obstacles to escape the ceiling and measure seed variance at a non-saturated point.
4. **λ attribution.** Isolate λ_u versus λ_v and test whether a single combined penalty is sufficient.
5. **Hardware validation.** SDR or flight test; this is a separate project, not a thesis revision.
6. **Public artifact DOI.** A Zenodo DOI for the code and result JSONs would strengthen reproducibility.

---

## 7. M5 execution plan (only if approved)

M5 was declared in the frozen protocol and then deferred; its ablation set was never specified, and no M5 result exists. If the supervisor wants it, the following plan is the smallest honest version. It is a proposal, not a recovered frozen design.

**Pre-registration:** Add A-16 to `PROTOCOL_AMENDMENTS.md` before the first cell. State the question, design, budget, seeds, and what is not changed.

### M5a — task-loss component removal

| Field | Value |
|---|---|
| Arm | λ-heavy with one term removed at a time |
| Removals | λ_u = 0; λ_v = 0; λ_T = 0 |
| k | 12 and 18 |
| Channels | AWGN, Rayleigh |
| Seeds | 42 |
| Steps | 3,000 (frozen) |
| Evaluation | E = 1,000, 5-SNR grid |
| Cells | 3 removals × 2 k × 2 channels = 12 |
| Expected cost | Roughly 8–12 hours on CPU, based on Wave-C timings |

**Question:** Which term is responsible for the λ-shaping effect? If removing λ_u alone collapses success, the control penalty is the active term. If removing λ_v alone collapses it, the velocity penalty is active. If none alone collapses it, the terms are redundant.

### M5b — uncoded digital reference

| Field | Value |
|---|---|
| Arm | Digital with no convolutional code |
| k | 12, 18, 24, 30, 54 |
| B | Chosen to fit the uncoded packet |
| Channels | AWGN, Rayleigh |
| Evaluation | E = 1,000 |
| Cells | 10 (evaluation only) |
| Expected cost | Minutes to an hour |

**Question:** Is the crossover an artefact of convolutional coding? A-11 and A-15 already test polar; the uncoded reference closes the loop.

### M5c — cross-channel training

| Field | Value |
|---|---|
| Train | AWGN, evaluate AWGN + Rayleigh; train Rayleigh, evaluate both |
| k | 18 |
| Arm | λ-heavy |
| Seeds | 42 |
| Steps | 3,000 |
| Evaluation | E = 1,000 |
| Cells | 2 training × 2 evaluation = 4 units |
| Expected cost | Roughly 2–4 hours |

**Question:** Does training on one channel transfer to the other? This is the “cross-channel” part of M5.

**Rule:** If a cell diverges, report it as a failure. Do not retrain it because the result is inconvenient.

---

## 8. 3GPP multi-geometry extension plan

The current A-10 result is one point. A cheap evaluation-only extension would strengthen RQ3 without retraining.

| Field | Current | Proposed extension |
|---|---|---|
| d2D | 500 m | 200, 500, 1000 m |
| h | 50 m | 50, 100 m |
| fc | 2 GHz | 2, 3.5 GHz |
| Methods | λ-heavy k = 18/54, digital k = 54 | Same |
| Episodes | 1,000 | 1,000 |
| Output | `results/AERIAL` | New `results/AERIAL_GEOM` |

**Deliverable:** A table of required transmit SNR for SR ≥ 0.90 and ≥ 0.99 at each geometry, plus a short discussion of how the λ advantage scales with path loss. This is a P1, not a P0.

**What not to claim:** The extension is still a large-scale layer. It does not add Doppler, time-correlated shadowing, or a full 3GPP channel simulator.

---

## 9. IEEE paper readiness plan

The thesis is not a paper. A 6-page IEEE conference paper would need:

1. **Title and abstract:** one paragraph, 150–200 words, with the crossover and the 3GPP link-budget result.
2. **Introduction:** 1 page; the gap is the matched-budget UAV-state comparison, not “new JSCC”.
3. **Related work:** 0.5 page; cite Bourtsoulatze, Kurda, Gündüz, Shao, Diao, Fang.
4. **System model and method:** 1.5 pages; state, channel, encoder/decoder, loss, digital baseline.
5. **Results:** 2 pages; Figures 5.1, 5.2/5.3 (combined), 5.4, 5.8; Tables 5.2, 5.9.
6. **Conclusion:** 0.3 page; crossover, λ mechanism, 3GPP penalty, M5/M5-like future work.
7. **References:** numeric IEEE style, 20–25 entries.

**Contribution framing:** “A matched-budget benchmark for task-oriented DeepJSCC in UAV navigation, a measured bandwidth crossover, a gradient-conditioning explanation of λ-shaping, and a first-order 3GPP UMa-aerial link-budget assessment.”

**Do not claim:** “We propose a new semantic communication paradigm.” The thesis itself says it is a domain transfer and systematic empirical study.

---

## 10. Verification

### 10.1 Commands used

```powershell
# Rebuild the thesis
python tools/build_thesis_docx.py `
  --md fyp_md_files/THESIS_MASTER_DOCUMENT.md `
  --out THESIS_MASTER_DOCUMENT.docx `
  --header "Task-Oriented DeepJSCC for UAV Navigation" `
  --title "Task-Oriented DeepJSCC for UAV Navigation"

# Rebuild the figures companion
python tools/build_thesis_docx.py `
  --md fyp_md_files/FIGURES_DOCUMENT.md `
  --out FIGURES_DOCUMENT.docx `
  --header "Thesis Figures" `
  --title "Thesis Figures - Complete Collection"

# Refresh Word fields and count pages (Word COM; do not export PDF here)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open((Resolve-Path "THESIS_MASTER_DOCUMENT.docx").Path)
$doc.Fields.Update() | Out-Null
$doc.Repaginate()
$doc.ComputeStatistics(2)
$doc.Save()
$doc.Close()
$word.Quit()
```

### 10.2 Counts

| Item | Count |
|---|---:|
| Thesis pages | 54 |
| Thesis words | 14,352 |
| Thesis tables | 21 |
| Thesis figures | 11 |
| Figure companion pages | 53 |
| Figure companion figures | 38 |
| Figure companion tables | 3 |

### 10.3 Test status

The repository reports 57 passing tests and a green Docker/CI run. In this environment, `pytest` is not installed in the venv or the system Python, so the suite could not be re-run. This is recorded as a P1 reproducibility item, not as a test failure.

---

## 11. Bottom line

**Thesis:** Strong, honest, and complete enough to submit after the P0 logistics (PDF, layout check, signatures). The main scientific gap is M5, which is correctly labelled deferred.

**IEEE paper:** Not ready. It needs a 6-page rewrite, numeric citations, and a tighter contribution framing.

**Next action:** If the supervisor approves, run M5a first (task-loss component removal). It is the highest-value remaining experiment because it turns the λ-shaping diagnosis from “λ works” into “which term works”. If M5 is not approved, keep it deferred and submit the thesis as it stands.
