# SUBMISSION KIT — Task-Oriented DeepJSCC for UAV Navigation

Yeh folder **sirf woh cheezein** rakhta hai jo kisi ko project dikhane ya jama karne ke liye chahiye.
Baqi sab `ARCHIVE/` mein hai, aur source code apni jagah (`project/`, `tools/`, `results/`) par hai.

**FYP title:** *Task-Oriented Deep Joint Source-Channel Coding for UAV Navigation under Noisy and Fading Wireless Channels — with a 3GPP UMa-Aerial Link-Budget and Dispersion Extension*

**Paper title:** *When Does Task-Oriented DeepJSCC Beat a Coded Digital Link? A Matched-Budget Study for UAV Navigation*

---

## 1. Is folder mein kya hai

| File | Kya hai | Kab use karein |
|---|---|---|
| `IEEE_PAPER_DRAFT.docx` | Corrected 6-page conference paper (4 tables, 4 figures) | Submission / panel ko dena |
| `THESIS_MASTER_DOCUMENT.docx` | Corrected full thesis (Table 5.2 aur gradient numbers artifact se verified) | Viva / final submission |
| `CLAIM_FIX_PATCHLIST_2026-11-07.md` | Har claim-correction ka old → new record | Agar panel poochay "kya badla" |
| `ONE_PAGER.md` | 1-page handout: problem, method, 3 results, limitations | Har interview mein pehla page |
| `figures/fig4_bandwidth_crossover.png` | Hero figure: k ≤ 18 vs k ≥ 24 crossover | Slide 3 |
| `figures/fig_wave_a_arms.png` | Frozen loss vs λ-shaped arms | Slide 4 |
| `figures/fig_training_stability.png` | 7–10 order gradient gap | Slide 5 |
| `figures/fig_aerial_required_snr.png` | 87 dB vs 108 dB @ 0.90 | Slide 6 |
| `figures/fig_experiment_matrix.png` | Poora experiment matrix | Slide 2 (ya backup) |

---

## 2. 10-minute presentation — 8 slides

| # | Slide | File | Ek line |
|---|---|---|---|
| 1 | Title + problem | paper p.1 | "UAV state bhejna hai; reconstruction ≠ navigation." |
| 2 | System + 3 arms | `fig_experiment_matrix.png` | "Same episodes, same noise, same k — sirf objective badla." |
| 3 | **Main result** | `fig4_bandwidth_crossover.png` | "k ≤ 18 neural jeetta hai, k ≥ 24 digital." |
| 4 | Task-arm fix | `fig_wave_a_arms.png` | "Frozen loss 2–6%; λ-shaping se 0.997." |
| 5 | Mechanism | `fig_training_stability.png` | "Yeh capacity nahi — gradient conditioning hai." |
| 6 | Aerial link | `fig_aerial_required_snr.png` | "87 dB vs 108 dB @ 0.90, k = 54." |
| 7 | Robustness | paper Table IV | "4 comparators + Doppler + CSI — ordering nahi badalti." |
| 8 | Limitations + takeaway | `ONE_PAGER.md` | "Conditional design rule, not a new paradigm." |

**2-minute version:** slide 3 → slide 4 → slide 6. Bas.

---

## 3. Numbers jo yaad hone chahiye

| Kya | Value |
|---|---|
| State / budget | 6-D state, k = 12–54 channel uses |
| k = 12 | neural **0.242** vs digital **0.062** |
| k = 54 | digital **0.995** vs reconstruction **0.246** |
| Shaped task arm | **0.996–0.999** (ceiling) |
| Frozen loss → fixed | **2–6% → 0.997** |
| Gradient gap | **7–10 orders** (median) |
| Aerial @ 0.90, k = 54 | λ **87 dB** vs digital **108 dB** → **21 dB** |
| Aerial, 12 geometries | **15–21 dB** @ 0.90, **12–15 dB** @ 0.99 |

---

## 4. Limitations — khud pehle bol dein

- Simulation-only, 2-D kinematics, fixed controller.
- 3GPP = **large-scale layer only** (no Doppler, no time-correlated shadowing, no aerial small-scale fading).
- A-17 = **one shared curve remapped** to 12 geometries → sensitivity, not validation.
- Digital baseline **fixed-rate** (no adaptive MCS / HARQ); gap ka kuch hissa isi ka hai.
- Extension blocks **single seed**; A-13 ek hi threshold.
- Task **saturated** at 0.996+ → high-bandwidth comparison inconclusive.

---

## 5. Kya NAHI dikhana

`ARCHIVE/` mein woh documents hain jin mein **purane numbers** hain (3 × 10^14, "fourteen orders", 2.5%, "puncturing"). Inhe presentation mein na dikhayein:

- `ARCHIVE/SUPERVISOR_BRIEFING_PLAIN.md` / `.docx`
- `ARCHIVE/PROJECT_STATUS.md`
- `ARCHIVE/HANDOFF_FOR_NEW_CHAT.md`
- `ARCHIVE/IEEE_EVALUATION_AND_REMAINING_WORK.md` / `.docx`
- `ARCHIVE/interim_status_2026-10-02.md`, `ARCHIVE/gpt_status_report_2026-10-06.md`, `ARCHIVE/m1_analysis_2026-10-06.md`
- `ARCHIVE/STEP_10B_*`, `ARCHIVE/STEP_10D_*`

> `README.md` ka "Headline result" section bhi abhi purane numbers rakhta hai — present karne se pehle update karna hai.

---

## 6. Source of truth

| Cheez | File |
|---|---|
| Paper source | `fyp_md_files/IEEE_PAPER_DRAFT.md` |
| Thesis source | `fyp_md_files/THESIS_MASTER_DOCUMENT.md` |
| Numeric evidence | `results/M2/`, `results/WAVE_A/`, `results/AERIAL/`, `results/AERIAL_GEOM/`, `results/MECHANISM/` |
| Claim audit | `fyp_md_files/IEEE_PAPER_AUDIT_2026-11-07.md` |

DOCX regenerate karne ke liye:

```powershell
python tools/build_ieee_paper_docx.py --md fyp_md_files/IEEE_PAPER_DRAFT.md --out IEEE_PAPER_DRAFT.docx
python tools/build_thesis_docx.py --md fyp_md_files/THESIS_MASTER_DOCUMENT.md --out THESIS_MASTER_DOCUMENT.docx --header "THESIS MASTER DOCUMENT"
```

---

## 7. Pending (submission se pehle)

1. Appendix A (literature search protocol) ke counts + search date bharne hain.
2. Abstract 311 words → 250 tak trim.
3. Thesis ke aerial ranges (`74–108`, `78–113`, `89–127 dB`) ko `results/AERIAL_GEOM/` se re-derive karna.
