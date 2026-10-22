# FYP Research Roadmap & Audit Document
**Last updated: 2026-10-21 — PIPELINE COMPLETE**
**Project: Task-Oriented Semantic Communication for UAV Navigation (6G)**

---

## PIPELINE STATUS — FINAL

| Component | Status | Numbers |
|---|---|---|
| M1 Training | ✅ Complete | 36/36 |
| M1 Evaluation | ✅ Complete | 198/198 paired units |
| M2 Training | ✅ Complete | 56 DONE + 4 conditional-accept = 60/60 |
| M2 Evaluation | ✅ Complete | 150/150 units |
| Wave-A Training | ✅ Complete | 44 DONE, 2 documented failures (honest) |
| Wave-A Evaluation | ✅ Complete | 85/90 slots (5 honest gap — diverged seed) |
| Wave-B Evaluation | ✅ **COMPLETE** | **350/350 cells, 70/70 σ=0 gates PASS** |
| Tests | ✅ Pass | 57/57 |
| Infinite loop bug | ✅ Fixed | run_all now halts on stall |

**Koi bhi training dobara nahi karni. Sab results intact hain.**

---

## 1. Project Summary — Ek Line Mein

> UAV ko goal tak pohonchane ke liye, kya task-oriented DeepJSCC better hai ya
> reconstruction-oriented DeepJSCC ya classical digital communication —
> same wireless bandwidth (k channel uses) ke andar?

---

## 2. Final Results — Kya Nikla

### M1 — Low bandwidth (k=1,2,3): Task vs Reconstruction

- Reconstruction k=2,3: SR 0.24–0.31 (AWGN), beats task
- Task k=1,2,3: SR 0.02–0.05 — flat, SNR-independent
- **Finding:** Frozen λ (ld=1, lT=2, lu=lv=0) se task encoder properly converge
  nahi kiya — Wave-A ne yeh diagnose kiya

### M2 — Matched budget 3-way (k=12..54): Crossover result

```
k=12:  Recon 0.253  vs  Digital 0.064  → Neural 4× better ✅
k=18:  Recon 0.237  vs  Digital 0.165  → Neural still winning ✅
k=24:  Recon 0.235  vs  Digital 0.324  → Digital overtakes ← CROSSOVER
k=30:  Recon 0.262  vs  Digital 0.629  → Digital clearly better
k=54:  Recon 0.249  vs  Digital 0.935  → Digital dominates
```

**Thesis ka core claim:** Low bandwidth (k ≤ 18) mein semantic JSCC wins.
High bandwidth (k ≥ 24) mein classical digital wins.
Yeh 6G ka direct answer hai: "semantic comm kab use karo?"

### Wave-A — λ ablation: Root cause found

```
lam_ctl_heavy (k=12):  SR 0.997  ← 70× better than frozen baseline!
lam_ctl_light (k=12):  SR 0.995 (AWGN) / 0.019 (Rayleigh) ← fragile
wrecon_pos    (k=12):  SR 0.24   ← reconstruction loss adds nothing
fixsnr_10db   (k=12):  SR 0.05   ← fixed SNR training doesn't help
long_12k      (k=12):  SR 0.035  ← more steps alone don't fix it
```

**Root cause:** λ_u = λ_v = 0 (control + velocity weights zero) se task encoder
properly optimize nahi hota. Heavy control penalty (lam_ctl_heavy) se 99.7% SR.

⚠️ **Caveat:** k=12 tak hi tested hai. High bandwidth pe yeh effect kaam karta
hai ya nahi — abhi unknown (Wave-C scope).

### Wave-B — CSI error robustness: New finding

5 CSI error levels (σ=0.0 to σ=0.4) across all k values:

```
σ=0.0:  recon SR 0.247,  task SR 0.034,  digital SR 0.384
σ=0.4:  recon SR 0.248,  task SR 0.034,  digital SR 0.395
```

**MSE penalty across all k:** consistently ~18% at σ=0.4
**Success rate:** practically FLAT — insensitive to CSI error

**Thesis claim (honest):**
> "Closed-loop task success is insensitive to CSI magnitude error up to
> σ=0.4. Reconstruction MSE degrades monotonically by ~18% but success
> rate is too coarse to reveal this distinction."

⚠️ **One anomaly:** Digital SR slightly *improves* with CSI error (0.384→0.395).
Physically unexpected. Likely B=1 hard-decision threshold shift from ZF gain.
Do NOT claim "digital is robust to CSI error" — investigate first.

---

## 3. Kya Claim Kar Rahe Ho (Updated After All Results)

1. **Bandwidth crossover exists** — semantic JSCC wins at k≤18, digital wins
   at k≥24. This is the primary thesis contribution. ✅ Data-proven.

2. **Task-oriented encoder underperforms reconstruction** under frozen λ — but
   Wave-A shows λ tuning recovers 99.7% SR. This motivates future work. ✅ Honest.

3. **CSI error robustness** — success rate insensitive to σ≤0.4. New 6G finding.
   ✅ Wave-B proven.

4. **Statistical rigor exceeds published DeepJSCC papers** — paired testing,
   Wilson CI, 3 seeds, matched budget enforcement. ✅

---

## 4. Novelty (6G Context Mein)

| Contribution | Kya Naya Hai |
|---|---|
| Bandwidth crossover | Low-k: semantic wins. High-k: digital wins. Direct "when to use semantic comm" answer |
| Matched-budget digital comparison | 6B+6=k enforced, infeasible cells declared — field mein yeh rigor rare hai |
| Paired evaluation harness | Bitwise-identical episodes/noise — apple-to-apple, stronger than cited papers |
| CSI robustness finding | SR flat up to σ=0.4, MSE degrades 18% — Wave-B new result |
| λ diagnosis (Wave-A) | Root cause of task failure found — opens conference paper path |
| Deterministic CPU reproducibility | No GPU non-determinism — exact reproduction guaranteed |

---

## 5. Issues — Final Status

| # | Issue | Status |
|---|---|---|
| SC decoder (not SCL) | Minor — 6G context mein acceptable. Ek sentence thesis mein add karo | Open (writing) |
| Rayleigh disclosure | Minor — paragraph thesis mein add karo | Open (writing) |
| Pooled CI semantics | F-NEW-3 aur F-NEW-4 se addressed | ✅ Done |
| Pipeline failures | All fixed by OpenCode | ✅ Done |
| Task instability (Fig.10) | Wave-A ne root cause find kiya (λ issue) | ✅ Diagnosed |
| Infinite loop bug | Fixed — run_all now exits on stall | ✅ Done |
| caps_h256 arm (128-wide) | Documented residual — capacity ablation invalid | Known gap |
| Digital CSI anomaly | Per-episode analysis needed before claiming robustness | ⚠️ Do not claim |

---

## 6. Rayleigh Disclosure — Thesis Mein Yeh Paragraph Add Karo

```
Channel Model and Its Idealizations:
- Perfect receiver CSI assumed (no channel estimation error)
- Zero-Forcing (ZF) equalization (not MMSE or matched filter)
- Block fading: one fading coefficient per control packet
- No transmitter CSI (open-loop power control only)
These are standard idealizations in the DeepJSCC literature
[Bourtsoulatze 2019, Kurka & Gündüz 2020] and applied identically
to all compared methods.
```

---

## 7. Figures — Full Assessment

| Figure | Status | Notes |
|---|---|---|
| fig1_m1_sr_vs_snr | ✅ Keep | Core M1 result |
| fig2_m2_awgn | ✅ STRONGEST | Crossover clearly visible |
| fig3_m2_rayleigh | ✅ Keep | Rayleigh counterpart |
| fig4_bandwidth_crossover | ✅ Keep | Annotated crossover point |
| fig5_full_bandwidth_axis | ✅ Keep | Full story in one figure |
| fig6_final_distance | ✅ Keep | Supporting metric |
| fig7_control_effort | ✅ Keep | Task-oriented claim support |
| fig8_time_to_goal | ✅ Keep | Supporting metric |
| fig9_sr_heatmaps | ✅ Keep | Grid visualization |
| fig10_training_curves | ⚠️ Keep + explain | λ issue wajah hai oscillation ki — NOW explained by Wave-A |
| fig11_cdf_final_dist | ✅ Keep | Statistical summary |
| fig12_trajectories | ✅ Keep + explain | Smooth-but-wrong = λ issue (Wave-A diagnosed) |
| fig13_mse_vs_snr | ⚠️ Use with caveat | Post-exit MSE = 0 issue document karo |
| fig14_rate_distortion | ✅ Keep | M2 complete hai |
| fig15_constellations | ⚠️ Fix layout | constrained_layout add karo |
| fig_new1_per_seed_agreement | ✅ DONE | Seed robustness proof |
| fig_new2_mcnemar_heatmap | ✅ DONE | Statistical significance visual |
| fig_new3_seed_spread_bars | ✅ DONE | Seed variance bars |
| fig_new4_sr_table | ✅ DONE | Thesis appendix table |

**Wave-B figures — DONE:**

| Figure | File | Kya Dikhata Hai |
|---|---|---|
| F-WAVE-B-1 ✅ | `fig_waveb1_sr_vs_k_csi_bands.png` | SR vs k, 3 sigma bands overlaid per method |
| F-WAVE-B-2 ✅ | `fig_waveb2_mse_vs_csi_sigma.png` | MSE vs sigma_csi per k — ~18% penalty annotated |
| F-WAVE-B-3 ✅ | `fig_waveb3_sr_vs_snr_csi.png` | SR vs SNR at k=12, all sigma levels |

Script: `tools/make_figures_waveb.py`

---

## 8. Papers Jo Cite Karni Chahiyen

### Core DeepJSCC (must cite):
1. Bourtsoulatze et al. 2019 — arXiv:1809.01733 — original DeepJSCC
2. Kurka & Gündüz 2020 — arXiv:1911.11174 — DeepJSCC-f

### 6G Semantic Communication (must cite):
3. Zhou et al. 2022 — arXiv:2210.09372 — Task-Oriented 6G Networks
4. Guo et al. 2024 — arXiv:2408.04188 — Trustworthy Task-Oriented 6G SemCom

### Task-oriented / goal-oriented (strongly recommended):
5. Shao et al. 2021 — IEEE JSAC — Information Bottleneck task-oriented comm
6. Gündüz et al. 2022 — IEEE JSAC — Beyond Transmitting Bits
7. Strinati & Barbarossa 2021 — IEEE Wireless Commun — 6G beyond Shannon

### UAV + wireless control:
8. Ayan et al. 2019 — Age of Information vs Value of Information, NCS
9. Semantic-Aware UAV papers 2023

### Stats methods:
10. Wilson 1927 — score interval
11. McNemar 1947 — paired test

---

## 9. Action Items — Kya Bacha Hai (Thesis Submission Ke Liye)

| Priority | Action | Time | Retraining? |
|---|---|---|---|
| 1 | Thesis mein Rayleigh idealization paragraph add karo | 30 min | No |
| 2 | SC decoder disclosure sentence add karo | 5 min | No |
| 3 | Fig. 15 layout fix (constrained_layout) | 15 min | No |
| 4 | Wave-B figures banao (F-WAVE-B-1, F-WAVE-B-2) | 1-2 hours | No |
| 5 | Thesis mein Fig. 10 explain karo (λ wajah hai — Wave-A proved it) | 20 min | No |
| 6 | Thesis mein Wave-A results add karo (lam_ctl_heavy finding) | 1 hour | No |
| 7 | Digital CSI anomaly note add karo (don't claim robustness) | 10 min | No |
| 8 | Exclusion decisions ka prose record likhna (M2 4 cells) | 30 min | No |

**Total remaining: ~4-5 hours of writing and minor scripts. Zero training.**

---

## 10. Scientist-Level Paper Roadmap (FYP Ke BAAD)

### Conference paper path (IEEE ICC/WCNC 2027):
Wave-A ka lam_ctl_heavy finding + mutual information analysis add karo.
Claim: "λ-shaped task loss recovers 99.7% SR — why does it work?"
Timeline: 3-4 months post-FYP.

### Transactions paper path (IEEE TWC / JSAC):
- Second environment generalization
- Theoretical bound on crossover point
- Wave-C: lam_ctl_heavy at high k (does 99.7% hold at k=24,54?)
Timeline: 12-18 months.

---

## 11. Key Numbers — Thesis Mein Clearly State Karo

- M1: 36/36 training, 198/198 evaluation, **all McNemar tests significant**
- M2: 150/150 evaluation, 4 conditional-accept documented
- Wave-A: 44/46 training complete, 85/90 eval slots
- Wave-B: **350/350 cells, 70/70 σ=0 gates pass** (bit-exact validation)
- Total episodes evaluated: ~2.5 million
- Training: CPU-only, deterministic, seeds {42,43,44}, test seed 10042
- Crossover: **k=18→24** (AWGN), confirmed by both M2 and Wave-B

---

## 12. Training History (Important For Thesis)

**Pehli training:** System band karne se interrupted runs tha. Optimizer state save
nahi hota (documented Step-8 limitation). Incomplete runs had higher final loss
(e.g., task_loss_final=4096 vs 1929 in full run).

**Dobara training (mam ke kehne par):** Fully uninterrupted. Better convergence.
Current results yahi training ke hain. Archive preserved at `results/_archive/`.

**Supervisor ka CPU-only decision bilkul sahi tha** — GPU non-determinism se
seed-based reproducibility break ho jati. CPU pe exact same results dobara
milenge.

---

*Document prepared: 2026-10-21 | Pipeline complete | Zero pending training*
