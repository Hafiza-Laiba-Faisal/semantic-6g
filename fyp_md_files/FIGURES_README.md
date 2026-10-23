# Thesis Figures — Complete Index
**All figures are in: `fyp_md_files/figures/`**
*Last updated: 2026-10-22*

---

## HOW TO USE
- Copy any figure file directly into your thesis (LaTeX or Word)
- All figures are 200 DPI — print quality
- Scripts to regenerate are listed — all read existing data, no retraining

---

## MAIN RESULTS — M1 & M2 (Chapter 4/5)

| File | What it shows | Use in thesis |
|---|---|---|
| `fig1_m1_sr_vs_snr.png` | M1: Reconstruction vs Task SR vs SNR, k=1,2,3, AWGN+Rayleigh | Primary M1 result |
| `fig2_m2_awgn.png` | M2: 3-way comparison (recon/task/digital), k=12–54, AWGN | **Strongest figure — use first** |
| `fig3_m2_rayleigh.png` | M2: same as fig2 but Rayleigh channel | Channel robustness |
| `fig4_bandwidth_crossover.png` | M2 AWGN: SR vs k at 0 dB and 20 dB — crossover annotated | Crossover finding |
| `fig5_full_bandwidth_axis.png` | M1+M2 combined: full k axis at 0 dB | Full story in one figure |

📄 Script: `tools/make_figures.py`

---

## SUPPORTING METRICS — M1 & M2 (Chapter 5 / Appendix)

| File | What it shows |
|---|---|
| `fig6_final_distance.png` | Final distance to goal per method |
| `fig7_control_effort.png` | Control effort — task encoder efficiency |
| `fig8_time_to_goal.png` | Time to reach goal (successful episodes) |
| `fig9_sr_heatmaps.png` | SR heatmap across (k, SNR) grid |
| `fig10_training_curves.png` | Training loss curves — oscillation explained by λ issue (Wave-A) |
| `fig11_cdf_final_dist.png` | CDF of final distance |
| `fig12_trajectories.png` | UAV trajectories — smooth-but-wrong = λ issue |
| `fig13_mse_vs_snr.png` | MSE vs SNR ⚠️ use with caveat (post-exit MSE=0) |
| `fig14_rate_distortion.png` | Rate-distortion curve |
| `fig15_constellations.png` | Learned encoder constellations ⚠️ fix layout overlap first |

📄 Script: `tools/make_figures_extra.py` / `tools/make_figures_semantic.py`

---

## STATISTICAL VALIDATION (Chapter 4 / Appendix)

| File | What it shows | Why important |
|---|---|---|
| `fig_new1_per_seed_agreement.png` | SR per seed (42/43/44) overlaid — seed robustness | Answers "lucky seed?" objection |
| `fig_new2_mcnemar_heatmap.png` | McNemar −log₁₀(p) heatmap — all 198 cells | Statistical significance visual |
| `fig_new3_seed_spread_bars.png` | Mean ± 1σ bars across seeds | Seed variance explicitly shown |
| `fig_new4_sr_table.png` | Full M1 SR table as figure | Thesis appendix table |

📄 Script: `tools/make_figures_new.py`

---

## WAVE-B — CSI ROBUSTNESS (Chapter 5 / New Finding)

| File | What it shows | Key result |
|---|---|---|
| `fig_waveb1_sr_vs_k_csi_bands.png` | SR vs bandwidth with σ=0/0.2/0.4 bands | SR insensitive to CSI error |
| `fig_waveb2_mse_vs_csi_sigma.png` | MSE vs σ per k — ~18% penalty annotated | MSE degrades, SR does not |
| `fig_waveb3_sr_vs_snr_csi.png` | SR vs SNR at k=12, all σ levels | Flat across both axes |

📄 Script: `tools/make_figures_waveb.py`

---

## WAVE-A & WAVE-C — λ DIAGNOSIS (Chapter 5 / Discussion)

| File | What it shows | Key result |
|---|---|---|
| `fig_wavec1_sr_vs_snr.png` | Wave-C: heavy/light SR vs SNR at k=18,54 | 0.99+ at all SNR points |
| `fig_wavec2_sr_vs_k.png` | Combined k=12(Wave-A) + k=18,54(Wave-C) | λ-shaping generalizes across k |
| `fig_wavec3_comparison_table.png` | Summary table: frozen vs λ-heavy, improvement factor | ~30–40× improvement |

📄 Script: `tools/make_figures_wavec.py`

---

## FIGURE COUNT SUMMARY

| Series | Count |
|---|---|
| Main results (M1/M2) | 5 |
| Supporting metrics | 10 |
| Statistical validation | 4 |
| Wave-B CSI robustness | 3 |
| Wave-A/C λ diagnosis | 3 |
| **Total** | **25** |

---

## REGENERATE ALL FIGURES

Run these in order (all read existing data — no training needed):

```bash
.venv\Scripts\python.exe tools/make_figures.py
.venv\Scripts\python.exe tools/make_figures_new.py
.venv\Scripts\python.exe tools/make_figures_waveb.py
.venv\Scripts\python.exe tools/make_figures_wavec.py
```

---

## WHICH FIGURES FOR THESIS CHAPTERS

| Chapter | Figures to use |
|---|---|
| Chapter 3 — System Model | fig12 (trajectories), fig15 (constellations) |
| Chapter 4 — Methodology | fig_new3 (seed spread), fig_new2 (McNemar heatmap) |
| Chapter 5 — M1 Results | fig1, fig_new1 (per-seed agreement) |
| Chapter 5 — M2 Results | **fig2** (strongest), fig3, fig4, fig5 |
| Chapter 5 — Wave-B | fig_waveb1, fig_waveb2 |
| Chapter 5 — Wave-C/Discussion | fig_wavec2 (key), fig_wavec1 |
| Appendix | fig_new4 (SR table), fig6, fig7, fig8, fig9, fig10, fig11 |

---

*All data from: `results/M1/`, `results/M2/`, `results/WAVE_A/`,
`results/WAVE_B/`, `results/WAVE_C/`*
*Zero training required to regenerate any figure.*

---

## ALL THESIS FILES — COMPLETE LIST

### Writing (copy-paste ready)
```
fyp_md_files/THESIS_DISCLOSURE_PARAGRAPHS.md     ← 6 disclosure paragraphs
fyp_md_files/RESEARCH_ROADMAP.md                 ← Claims, novelty, citations
fyp_md_files/SUPERVISOR_APPROVAL_MEMO.md         ← RQ revision, uplink clarification
fyp_md_files/03-final-mathematical-implementation-specification.md  ← Math
fyp_md_files/03-uav-navigation-mathematical-model.txt               ← UAV model
fyp_md_files/05-implementation-audit.md          ← All frozen decisions documented
```

### Results tables (numbers)
```
fyp_md_files/RESULTS_WAVE_A_WAVE_B_2026-10-21.md ← Wave-A + Wave-B tables
fyp_md_files/RESULTS_WAVE_C_2026-10-22.md        ← Wave-C λ results
results/M1/step10_m1_summary.json                ← M1 numbers
results/M2/step11_m2_summary.json                ← M2 numbers
results/WAVE_B/step_wave_b_summary.json          ← Wave-B CSI numbers
results/WAVE_C/step_wave_c_summary.json          ← Wave-C numbers
results/M2/PROTOCOL_EXCEPTIONS.md               ← 4 excluded cells
```

### Reference (background reading)
```
ARCHITECTURE.md
FEATURES.md
RESEARCH_STANDARDS_AUDIT.md
fyp_md_files/PROTOCOL_AMENDMENTS.md
fyp_md_files/fyp-proposal.txt
fyp_md_files/02-task-oriented-communication-papers.md.txt  ← Literature review notes
fyp_md_files/04-digital-baseline-and-training-eval-protocol.txt
```
