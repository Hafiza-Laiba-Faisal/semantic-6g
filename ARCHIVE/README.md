# ARCHIVE — superseded material

Yeh folder woh documents rakhta hai jo **project ka hissa hain, lekin current deliverables nahi**.
Inhe presentation, paper, ya thesis mein cite na karein.

## Zaroori warning

In documents mein **purane numbers** hain jo corrected thesis/paper se match nahi karte:

| Purana | Correct (artifact-verified) |
|---|---|
| frozen gradient `3.0 × 10^14` / `2.4 × 10^14` | `3.0 × 10^10` (AWGN) / `8.1 × 10^7` (Rayleigh); p99 `2.4 × 10^18` / `5.0 × 10^18` |
| "fourteen orders of magnitude" | 7–10 orders (median), ~17 at p99 |
| digital k = 54 AWGN `0.935` | `0.995` |
| "inside sampling error" | overlapping intervals; no formal equivalence test |
| "puncturing" (LDPC) | truncation to the first 108 mother bits |
| frozen loss "2–5%" | 2–6% |

Current sources of truth: `fyp_md_files/THESIS_MASTER_DOCUMENT.md`,
`fyp_md_files/IEEE_PAPER_DRAFT.md`, `SUBMISSION_KIT/`, and the JSON artifacts under `results/`.

## Contents

| File | Why archived |
|---|---|
| `SUPERVISOR_BRIEFING_PLAIN.md` / `.docx` | Superseded narrative; stale mechanism numbers |
| `PROJECT_STATUS.md` | Pipeline snapshot; not a deliverable |
| `HANDOFF_FOR_NEW_CHAT.md` | Session handoff note |
| `interim_status_2026-10-02.md` | Mid-M2 snapshot |
| `gpt_status_report_2026-10-06.md` | Mid-M2 snapshot |
| `m1_analysis_2026-10-06.md` | Superseded by thesis Ch. 5 |
| `STEP_10B_M1_RESULTS_ANALYSIS.md` | Superseded by thesis Ch. 5 |
| `STEP_10B_REPORT_FOR_GPT_REVIEW.md` | External review artifact |
| `STEP_10D_M1_NUMERICAL_RECONCILIATION.md` | Reconciliation record; conclusion is in Step 10E |
| `IEEE_EVALUATION_AND_REMAINING_WORK.md` / `.docx` | Pre-A-16 audit (historical) |
| `We are developing an FYP titled…` `.md` / `.pdf` | Original project brief |
| `extracted_raw.txt`, `extracted_raw_nolayout.txt` | Raw extraction dumps |
| `WHATSAPP_SUPERVISOR_MESSAGE.txt` | Informal message |

## Kya nahi move kiya

`PROTOCOL_AMENDMENTS.md`, saare `RESULTS_*.md`, aur `05-implementation-audit.md` **fyp_md_files/**
mein hi hain — yeh evidence trail hai aur thesis inhe cite karti hai.
