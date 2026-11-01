# Handoff — New Chat

**Project:** Task-Oriented DeepJSCC for UAV Navigation under Noisy and Fading Wireless Channels, with a 3GPP UMa-Aerial Link-Budget and Dispersion Extension  
**Supervisor:** Ismat Hira  
**Team:** Areesha Sohail; Amna Shahid; Hafiza Laiba Faisal  
**Repo:** `Hafiza-Laiba-Faisal/semantic-6g`  
**Latest commit:** `f39bb43` on `origin/main`

---

## 1. Current deliverables

| File | What it is | Status |
|---|---|---|
| `fyp_md_files/THESIS_MASTER_DOCUMENT.md` | Thesis source | Complete draft |
| `THESIS_MASTER_DOCUMENT.docx` | 54 pages, 14,352 words, 21 tables, 11 figures | Generated |
| `fyp_md_files/FIGURES_DOCUMENT.md` | 38-figure companion source | Complete |
| `FIGURES_DOCUMENT.docx` | 53 pages, 5,525 words, 38 figures | Generated |
| `fyp_md_files/IEEE_EVALUATION_AND_REMAINING_WORK.md` | IEEE-style evaluation, gap matrix, M5 plan | Complete |
| `IEEE_EVALUATION_AND_REMAINING_WORK.docx` | 11 pages, 3,295 words, 9 tables | Generated |
| `fyp_md_files/LACUNA_LITERATURE_MAP.md` | Direct-query literature map for Chapter 2 / IEEE paper | Complete |
| `LACUNA_LITERATURE_MAP.docx` | 15 pages, ~4,095 words, 7 tables | Generated |
| `fyp_md_files/PROTOCOL_AMENDMENTS.md` | Authoritative A-1…A-15 definitions | Reference |
| `fyp_md_files/SUPERVISOR_APPROVAL_MEMO.md` | Mam's review response, approvals A1–A4 | Reference |

---

## 2. Main scientific result

- **Bandwidth crossover:** neural JSCC wins at **k ≤ 18**; digital wins at **k ≥ 24**.
- **Frozen task loss fails.** It does not train a useful task-oriented codec by itself.
- **λ-shaping fixes it:** `λ_u = λ_v = 0.1` recovers the operating point. This is the mechanism finding, not a new coding paradigm.
- **M3:** at B = 1, 0.90 success needs 12 channel uses; deficits are 11/10/9 at k = 1/2/3. Structural infeasibility, not a measured failure.
- **M4:** B = 8, k = 54, E = 5000. 0 dB: AWGN 0.9736, Rayleigh 0.7178. 5 dB: 1.000 / 0.9908.
- **3GPP A-10:** λ-heavy needs 91 dB (k = 18) or 87 dB (k = 54) for 0.90; digital needs 108 dB at k = 54 and never reaches 0.90 at k = 18.
- **A-12:** full-dispersion training diverges (final losses 4,320/210; 311/203 non-finite gradients).
- **A-13:** tail-resampled training is stable and moves the operating point ≤ 2 dB.

---

## 3. What is already done

- M1–M4, Waves A/B/C, and A-9…A-15 audited against stored JSON artifacts.
- 3GPP UMa-aerial is a first-class channel in §3.8, with Figure 3.1.
- M3 §5.2.1 and M4 §5.2.2 added; M5 clearly marked deferred.
- Table 4.2 and Figure 4.1 (experiment matrix) added.
- §6.5, limitations, appendices, and viva Q&A updated.
- Figure defects fixed: `fig13`, `fig14`, `fig15`, `fig_aerial_training`.
- List of Figures and List of Tables added.
- Multiple-comparison caveat added to §4.11.
- Document builder supports CLI args, field updates, repeating table headers, per-list numbering restart, and fenced code blocks.
- Lacuna direct-query literature map completed.

---

## 4. What is not done

- **M5** (task-loss component removal, uncoded digital, cross-channel training) was never executed. There is no `results/M5`.
- **3GPP model** uses one geometry: d2D = 500 m, h = 50 m, fc = 2 GHz. No Doppler, no time-correlated shadowing.
- **`pytest`** is not installed in the project venv or system Python; the reported 57-test suite could not be re-run locally.
- **PDF export via Word COM hangs.** The `.docx` files are the usable deliverables.
- **IEEE paper** is not ready: needs 6-page compression, numeric citations, and the new related-work paragraphs.
- **Supervisor approvals** A1–A4 and any M5/A-16 amendment are pending.

---

## 5. Next actions

### P0 — thesis submission (1–3 days)

1. Add five citations from the Lacuna map to Chapter 2, Table 2.1, and References:
   - Path-SC (Zhao et al., 2025/2026)
   - SC-GIR (Wanasekara et al., 2025)
   - JTD-SC/DSSCC (Huang et al., 2023/2024)
   - Khina et al. (2016), analog JSCC for control
   - Pan et al. (2024), DRJSCC over time-varying channels
2. Verify venue, year, and DOI for each new reference.
3. Run the four exact-phrase gap checks in `LACUNA_LITERATURE_MAP.md` §8.
4. Produce a PDF through LibreOffice or manual Word “Save as PDF” (not Word COM).
5. Check Table 4.2, Table 5.10, and Figure 3.1 for layout.
6. Insert signatures/dates and do a final proofread.

### P1 — paper and robustness

7. A-10 multi-geometry sweep: d2D = 200/500/1000 m; h = 30/50/100 m; fc = 2/3.5 GHz.
8. Write A-16 and run **M5a** first (task-loss component removal), only after supervisor approval.
9. Add more seeds at k = 18 and 24; add Holm–Bonferroni or a preregistered primary endpoint.
10. Install `pytest` and re-run the test suite.
11. Draft the 6-page IEEE paper using `LACUNA_LITERATURE_MAP.md` §7.

### P2 — future work

12. Doppler and time-correlated shadowing.
13. Principled full-dispersion training.
14. Harder task and continuous control cost.
15. SDR/hardware validation.
16. Zenodo DOI for code and data.

---

## 6. Supervisor context

Mam's review raised three points:

1. **Architecture:** the implemented system is UAV-to-ground-station **uplink state transmission**, not UAV-to-self. The return command downlink is assumed ideal and out of scope.
2. **RQ1 wording:** the original RQ presumed task-oriented would win. The neutral version asks under what conditions it outperforms, matches, or underperforms.
3. **Novelty:** do not claim a new coding paradigm. Claim domain transfer + closed-loop extension, the first systematic matched-bandwidth three-way comparison for UAV navigation, the regime-mapping result, and the reproducible frozen-protocol testbed.

The memo requests approvals **A1–A4**:
- A1: revised neutral RQ1.
- A2: title/problem statement aligned with UAV navigation state uplink.
- A3: one-sentence return-link scope note.
- A4: approve extension waves W-A and W-B as post-freeze amendments, M5-expanded.

The WhatsApp message asks two questions:
1. Should the new ablation runs be included as a diagnostic-amendment chapter?
2. Thesis first, paper after?

**Do not run M5 without a dated protocol amendment and explicit approval.**

---

## 7. Ready-to-paste prompt for the new chat

```text
Project: semantic-6g — Task-Oriented DeepJSCC for UAV Navigation over AWGN, Rayleigh, and 3GPP UMa-Aerial channels.

Read these first:
- fyp_md_files/THESIS_MASTER_DOCUMENT.md
- fyp_md_files/IEEE_EVALUATION_AND_REMAINING_WORK.md
- fyp_md_files/LACUNA_LITERATURE_MAP.md
- fyp_md_files/PROTOCOL_AMENDMENTS.md
- fyp_md_files/SUPERVISOR_APPROVAL_MEMO.md

Current state:
- Thesis and figures are drafted; main result is the k=18→24 bandwidth crossover.
- λ_u = λ_v = 0.1 fixes the frozen task loss.
- M5 is deferred; no results/M5 exists. Do not invent M5 results.
- 3GPP model has one geometry; no Doppler.
- Latest commit is f39bb43.

Immediate task:
[write the specific task here, e.g. "Add the five Lacuna citations to Chapter 2 and Table 2.1, then rebuild the thesis docx."]

Rules:
- Do not invent results.
- Do not run M5 without a dated protocol amendment and supervisor approval.
- Keep changes consistent with the existing thesis style.
- Commit and push when done.
```

---

## 8. Useful commands

```powershell
# Rebuild thesis
python tools/build_thesis_docx.py --md fyp_md_files/THESIS_MASTER_DOCUMENT.md --out THESIS_MASTER_DOCUMENT.docx --header "Task-Oriented DeepJSCC for UAV Navigation"

# Rebuild figures companion
python tools/build_thesis_docx.py --md fyp_md_files/FIGURES_DOCUMENT.md --out FIGURES_DOCUMENT.docx --header "Thesis Figures"

# Rebuild literature map
python tools/build_thesis_docx.py --md fyp_md_files/LACUNA_LITERATURE_MAP.md --out LACUNA_LITERATURE_MAP.docx --header "Lacuna Literature Map"

# Git
git status --short
git add <files>
git commit -m "<message>"
git push origin main
```
