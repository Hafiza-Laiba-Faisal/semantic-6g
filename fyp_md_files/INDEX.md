# fyp_md_files/ — document index

Documents accumulated across the project, plus a pointer to the archived material.
This index says what each one is and whether it is **authoritative**. When two
documents disagree, the authoritative one wins.

Paths without a prefix live in `fyp_md_files/`; `ARCHIVE/` paths are at the repo root.
Winning-order note: the corrected paper and thesis (2026-11-07 claim pass) supersede
everything under `ARCHIVE/`.

## Read these first

| File | Role | Authoritative? |
|---|---|---|
| `../SUBMISSION_KIT/ONE_PAGER.md` | Plain-language one-page answer to "what was the problem, how was it solved, was 3GPP done" | Yes, for the narrative |
| `THESIS_MASTER_DOCUMENT.md` | Thesis source (24 numbered tables, 11 figures) | **Yes — the deliverable** |
| `IEEE_PAPER_DRAFT.md` | 6-page IEEE conference-paper draft (numeric citations, 4 tables, 4 figures) | Yes, for the paper track |
| `PROTOCOL_AMENDMENTS.md` | A-1 … A-20, every dated deviation | **Yes — the protocol ledger** |
| `IEEE_PAPER_AUDIT_2026-11-07.md` | Independent claim/novelty audit and correction list | Yes, for the claim record |
| `CLAIM_FIX_PATCHLIST_2026-11-07.md` | Old → new record of every applied claim fix | Yes, for provenance |
| `ARCHIVE/IEEE_EVALUATION_AND_REMAINING_WORK.md` | Independent IEEE-style audit, gap matrix, M5 plan (pre-A-16) | Historical |
| `ARCHIVE/PROJECT_STATUS.md` | Pipeline state and frozen decisions | Historical |
| `ARCHIVE/HANDOFF_FOR_NEW_CHAT.md` | Onboarding summary for a fresh session | Historical |

## Thesis companions

| File | Role |
|---|---|
| `FIGURES_DOCUMENT.md` | 38-figure companion: description, what to notice, placement |
| `FIGURES_README.md` | Figure inventory and regeneration notes |
| `THESIS_DISCLOSURE_PARAGRAPHS.md` | Reusable honesty/disclosure paragraphs |
| `SUPERVISOR_APPROVAL_MEMO.md` | A1–A4 approval request and review response |
| `LACUNA_LITERATURE_MAP.md` | Direct-query literature map for Ch. 2 and the IEEE paper |

## Results notes (one per experiment block)

| File | Experiment |
|---|---|
| `RESULTS_WAVE_A_WAVE_B_2026-10-21.md` | λ/objective ablations + CSI error |
| `RESULTS_WAVE_C_2026-10-22.md` | λ-shaping across bandwidth |
| `RESULTS_ARCH_2026-10-26.md` | A-9 capacity ablation (negative) |
| `RESULTS_AERIAL_2026-10-26.md` | A-10 3GPP link budget |
| `RESULTS_AERIAL_TRAIN_2026-10-26.md` | A-12 full-dispersion training (negative) |
| `RESULTS_AERIAL_TRAIN_TRUNC_2026-10-27.md` | A-13 tail-resampled training |
| `RESULTS_POLAR_2026-10-26.md` | A-11 polar comparator |
| `RESULTS_K24_SEEDS_2026-10-27.md` | A-14 seed variance at k = 24 |
| `RESULTS_MECHANISM_2026-10-27.md` | Gradient-conditioning mechanism |
| `RESULTS_AERIAL_GEOM_2026-11-01.md` | A-17 3GPP multi-geometry link budget (12 geometries) |
| `RESULTS_M5B_M5C_2026-11-01.md` | M5b uncoded digital reference + M5c cross-channel |
| `RESULTS_M5A_2026-11-02.md` | M5a λ-term removal (λ_u is the active term) |
| `RESULTS_DOPPLER_2026-11-02.md` | A-19 Doppler / time-correlated Rayleigh sweep |
| `RESULTS_LDPC_2026-11-02.md` | A-20 5G NR LDPC BG2 comparator |

## Design and planning

| File | Role |
|---|---|
| `WAVE_A_DESIGN.md` | Wave-A pre-registration (arms A1–A4) |
| `WAVE_A_DEPLOYMENT.md` | Wave-A run/deploy notes |
| `EXTENSIONS_SPEC.md` | A-9…A-15 extension specification |
| `RESEARCH_ROADMAP.md` | Pipeline status and accounting |
| `GAP_ANALYSIS_RESPONSE.md` | A-9…A-15 response summary |
| `RESEARCH_STANDARDS_AUDIT.md` | Standards/honesty audit |
| `05-implementation-audit.md` | Implementation audit; M1–M5 matrix; packet invariant |
| `03-final-mathematical-implementation-specification.md` | Math spec for the implementation |

## Historical (superseded — keep for the audit trail, do not cite)

| File | Why superseded |
|---|---|
| `ARCHIVE/interim_status_2026-10-02.md` | Snapshot during M2 execution |
| `ARCHIVE/gpt_status_report_2026-10-06.md` | Snapshot during M2 execution |
| `ARCHIVE/m1_analysis_2026-10-06.md` | Superseded by thesis Ch. 5 |
| `ARCHIVE/STEP_10B_M1_RESULTS_ANALYSIS.md` | Superseded by thesis Ch. 5 |
| `ARCHIVE/STEP_10B_REPORT_FOR_GPT_REVIEW.md` | Review artifact |
| `ARCHIVE/STEP_10D_M1_NUMERICAL_RECONCILIATION.md` | Reconciliation record; conclusion is in Step 10E |
| `FIGURE_DIAGNOSIS_2026-10-12.md` | Figure defect log; fixes are in the current figures |

## Source material (raw inputs, not project output)

`01-…md.txt` + `.zip`, `02-…md.txt`, `03-uav-navigation-mathematical-model.txt`,
`04-digital-baseline-and-training-eval-protocol.txt`, `final_variables-1/2.md.txt`,
`fyp-proposal.txt`. The original brief (`We are developing an FYP titled…`), the raw
extracts (`extracted_raw*.txt`) and the WhatsApp message now live under `ARCHIVE/`.

These are the proposal and reference dumps the design was derived from. They are
**inputs**, not results. Do not cite them as evidence.

## Regenerating the Word documents

```powershell
python tools/build_thesis_docx.py --md fyp_md_files/THESIS_MASTER_DOCUMENT.md --out THESIS_MASTER_DOCUMENT.docx --header "Task-Oriented DeepJSCC for UAV Navigation"
python tools/build_thesis_docx.py --md fyp_md_files/FIGURES_DOCUMENT.md --out FIGURES_DOCUMENT.docx --header "Thesis Figures"
python tools/build_thesis_docx.py --md fyp_md_files/LACUNA_LITERATURE_MAP.md --out LACUNA_LITERATURE_MAP.docx --header "Lacuna Literature Map"
python tools/build_ieee_paper_docx.py --md fyp_md_files/IEEE_PAPER_DRAFT.md --out IEEE_PAPER_DRAFT.docx
```
