# Independent Audit — IEEE Conference Paper (`IEEE_PAPER_DRAFT.docx` / `.md`)

**Date:** 2026-11-07
**Auditor:** Independent verification pass (read-only)
**Targets:** `fyp_md_files/IEEE_PAPER_DRAFT.md` and the built `IEEE_PAPER_DRAFT.docx`
**Reference thesis:** `fyp_md_files/THESIS_MASTER_DOCUMENT.md`
**Repository state:** `78ba7ac` (clean, `main == origin/main`)
**Changes made during audit:** none to the paper, thesis, artifacts, or configurations. This report is the only new file.

---

## 0. Kholasa (Roman Urdu)

Paper ka structure theek hai: 6 pages, 4 tables, 4 figures, 24 references, sab references in-text cited hain, aur **M2 / LDPC / POLAR / Doppler / 3GPP / M5 / CSI** ke numeric values mostly artifacts se exactly match karte hain. Lekin paper **submit-ready nahi** hai. Sab se bade masle ye hain:

1. **Mechanism ke gradient numbers galat hain.** Paper kehta hai frozen arm ka `p50 ≈ 3.0×10^14`, `p99 ≈ 2.4×10^14`. Artifact mein frozen arm ka `p50 = 3.0×10^10` (AWGN) / `8.1×10^7` (Rayleigh) aur `p99 = 2.4×10^18` / `5.0×10^18` hai. `3.0×10^14` asal mein **A-12 λ-heavy 3GPP arm** ka p50 hai, jo diverge hua. Is liye "fourteen orders" wali claim bhi apne quoted p50/p99 se nahi nikalati.
2. **Table I ka MSE claim (0.049–0.054) galat aggregation hai.** Stored M2 JSON se 0–20 dB mean `0.0439–0.0468` aata hai; sirf 0 dB column `0.0465–0.0538` hai. Dono mein se koi bhi `0.049–0.054` nahi.
3. **Table II ka frozen row `0.020–0.048` galat hai.** Wave-A `seedvar_frozen` k=12 AWGN range `0.0200–0.0285` hai; `0.048` asal mein `fixsnr_10db` arm ka 0 dB value hai. "Untrained random model 0.016" ka koi stored artifact hi nahi.
4. **Crossover ki framing misleading hai.** "Coded link wins at k ≥ 24" sirf **reconstruction** arm ke liye sach hai. Paper ka apna **λ-heavy task arm** k=12 (0.997), k=18 (0.999), k=24 (0.996–0.999) aur k=54 (0.998) par task ceiling par hai — yani task-oriented link har measured bandwidth par digital se jeet raha hai. Table I mein λ-heavy column hi nahi, is liye reader ko 0.998 nazar nahi aata.
5. **Fig. 4 ka caption jhoota hai.** Caption "A-10/A-17 ... across twelve geometries" kehta hai, magar PNG sirf **A-10 pinned geometry (500 m / 50 m / 2 GHz)** plot karta hai. Twelve geometries sirf Table III mein hain.
6. **Fig. 1 ka caption aggregation nahi batata.** Figure 0 dB aur 20 dB ke do panels hain, jabke Table I 0–20 dB mean hai. Is liye figure ke numbers (k=12: 0.264) Table I (0.242) se different lagte hain.
7. **Chhoti numeric mismatches:** "at most 2 dB" (asal 2.22 dB), "checkpoints 0.02–0.03" (asal 0.0117–0.0273), "2–5%" (asal 1.59–5.63%), "puncturing" (asal truncation).
8. **References:** 6 entries ko update karna chahiye — [11] ka arXiv ID doosra paper hai (asal GLOBECOM 2025), [9]/[10] ab early-access nahi, [12] 2026 vol/pages, [14]/[19] ka journal version maujood hai, [1] ka title.

Paper ke 111 tests claim ko verify kiya: **111 passed** (196.7 s).

---

## 1. Verification method

All numbers below were recomputed directly from the stored JSON artifacts (no hand transcription):

| Area | Source |
|---|---|
| Main matrix | `results/M2/step11_m2_summary.json` → `pooled_rows[]` |
| Loss ablation | `results/WAVE_A/step_wave_a_summary.json` → `pooled_rows[]`, `invalid_arms`, `train_*` |
| Shaped arm | `results/WAVE_C/step_wave_c_summary.json` → `pooled_rows[]` |
| k=24 seed study | `results/K24_SEEDS/step_k24_summary.json` → `spread_across_seeds[]` |
| Mechanism | `results/MECHANISM/step_stability_summary.json` → `cells_with_history[]` |
| Term removal | `results/M5/step_m5_summary.json` → `m5a_pooled_rows`, `m5b_uncoded_rows` |
| LDPC | `results/LDPC/step_ldpc_summary.json` → `mean_over_snr` |
| Polar | `results/POLAR/step_polar_summary_E1000.json`, `results/POLAR5G/step_polar5g_summary_E1000.json` |
| Doppler | `results/DOPPLER/step_doppler_summary.json` |
| 3GPP pinned | `results/AERIAL/step_aerial_summary.json` → `methods[*].required_tx_db`, `curve_eff` |
| 3GPP sweep | `results/AERIAL_GEOM/step_aerial_geometry_summary.json` → `rows[]` |
| 3GPP training | `results/AERIAL_TRAIN/step_aerial_training_summary.json`, `.../AERIAL_TRAIN_TRUNC/...` |
| Capacity | `results/ARCH_ABL/step_arch_summary.json` + `results/ARCH_ABL/eval/*` + `train_report.json` |
| CSI | `results/WAVE_B/step_wave_b_summary.json` + `results/WAVE_B/eval/*` |
| Saturated link | `results/M4/step13_m4_summary.json` → `conditions[]` |
| References | Crossref REST API + DOI content negotiation + arXiv API + Enlighten Theses |

Test suite check: `.venv\Scripts\python.exe -m pytest -q` → `111 passed in 196.70s`.

---

## 2. Critical discrepancies (must fix before submission)

### C1 — Mechanism gradient percentiles are wrong (frozen arm)

| | |
|---|---|
| **Paper location** | §IV-B, line 126; repeated in Abstract (line 7), contribution 3 (line 23), Fig. 3 caption (line 130), Conclusion (line 183) |
| **Paper claim** | "The frozen arm's encoder gradient has p50 ≈ 3.0 × 10^14 and p99 ≈ 2.4 × 10^14" |
| **Artifact value** | Frozen (ARCH_ABL, width 256, k=18, seed 42): `p50 = 2.9895×10^10` (AWGN), `p50 = 8.1149×10^7` (Rayleigh); `p99 = 2.3945×10^18` (AWGN), `p99 = 5.0075×10^18` (Rayleigh); `grad_max = 2.4236×10^19` (AWGN), `2.4398×10^19` (Rayleigh); `nonfinite_grad_steps = 75` / `100`; `converged = false` (0/2). |
| **Source** | `results/MECHANISM/step_stability_summary.json` → `cells_with_history[]` where `experiment == "ARCH_ABL"`, fields `grad_p50`, `grad_p99`, `grad_max`, `nonfinite_grad_steps`, `converged` |
| **Where 3.0×10^14 comes from** | The λ-heavy **A-12 3GPP** cells: `p50 = 2.3597×10^14` (k=18) and `2.1602×10^14` (k=54), i.e. the arm the paper itself reports as diverged. The paper has conflated A-12 λ-heavy with the frozen arm. |
| **Explanation** | `3.0×10^14` is not a frozen-arm number in any stored cell. `p99 = 2.4×10^14` matches no stored p99 either (it is close to the A-12 k=18 p50, `2.36×10^14`). |
| **Recommended correction** | Replace with: "frozen-arm p50 ≈ 3.0×10^10 (AWGN) / 8.1×10^7 (Rayleigh) and p99 ≈ 2.4×10^18 / 5.0×10^18, with 75–100 non-finite steps and no convergence." Delete `3.0×10^14`/`2.4×10^14` everywhere (abstract, contribution, §IV-B, Fig. 3 caption, conclusion). |

### C2 — "Fourteen orders of magnitude" is not supported by the quoted statistics

| | |
|---|---|
| **Paper location** | Abstract (line 7); contribution 3 (line 23); §IV-B (line 126); Fig. 3 caption (line 130); Conclusion (line 183) |
| **Paper claim** | "conditioning effect of roughly fourteen orders of magnitude"; "The separation is roughly fourteen orders of magnitude"; "λ-shaping bounds the encoder gradient by about fourteen orders" |
| **Artifact value** | Frozen vs λ-heavy: p50 ratio `3.0×10^10 / 3.6 ≈ 8×10^9` (≈9.9 orders); p99 ratio `2.4×10^18 / 57 ≈ 4.2×10^16` (≈16.6 orders); max ratio `2.42×10^19 / 4.2×10^4 ≈ 5.8×10^14` (≈14.8 orders). Only the **max-to-max** ratio is ≈14–15 orders. |
| **Source** | `results/MECHANISM/step_stability_summary.json` → `cells_with_history[]` (`grad_p50`, `grad_p99`, `grad_max`) |
| **Explanation** | The "fourteen" figure is being used as a *typical* separation but it only arises from an extreme max/max comparison; the medians differ by ~10 orders and the p99s by ~17. Fig. 3 itself plots ~10^9–10^11 (frozen) vs ~10^0.5 (λ), i.e. ~10 orders, not 14. |
| **Recommended correction** | Either (a) drop the "fourteen orders" phrase and say "roughly nine to ten orders in the median and about sixteen in the 99th percentile", or (b) keep "up to about fifteen orders at the maximum" and label it as a max-to-max ratio. Do not attach it to p50/p99. |

### C3 — Table I MSE claim does not match the stored M2 aggregation

| | |
|---|---|
| **Paper location** | §IV-A, line 90 |
| **Paper claim** | "The reconstruction MSE is flat in k (0.049–0.054)" |
| **Artifact value** | Mean over SNR 0–20 dB (AWGN, 3 seeds): k=12 `0.0468`, k=18 `0.0468`, k=24 `0.0452`, k=30 `0.0439`, k=54 `0.0448` → **0.0439–0.0468**. At 0 dB only: `0.0538`, `0.0514`, `0.0488`, `0.0469`, `0.0465` → **0.0465–0.0538**. |
| **Source** | `results/M2/step11_m2_summary.json` → `pooled_rows[]`, `mse_mean_mean_over_seeds` (channel `awgn`, method `reconstruction`, per `snr_db`) |
| **Explanation** | Neither aggregation produces `0.049–0.054`. The range `0.049–0.054` is inherited from thesis Table 5.2, whose MSE column is neither the current 0 dB values nor the SNR mean (it is a legacy column). |
| **Recommended correction** | If the intent is the paper's Table I (SNR-pooled), write "mean reconstruction MSE is flat in k (0.044–0.047)". If the intent is 0 dB, write "at 0 dB the reconstruction MSE is 0.047–0.054" and say so explicitly. Also remove/qualify the blanket sentence "Every mean reported below is recomputed from the stored per-SNR JSON artifacts" (line 84), which is contradicted by this row. |

### C4 — Table II frozen row `0.020–0.048` is not a Wave-A frozen range

| | |
|---|---|
| **Paper location** | Table II, line 118; thesis Table 5.4 line 635 |
| **Paper claim** | "Frozen task loss (λ_u = λ_v = 0) \| 0.020–0.048 \| 0.033" |
| **Artifact value** | Wave-A `seedvar_frozen`, k=12, AWGN per-SNR: `0.0200, 0.0235, 0.0270, 0.0270, 0.0285` → **0.0200–0.0285**; Rayleigh: `0.0333, 0.0337, 0.0313, 0.0317, 0.0310` → **0.0310–0.0337** (mean `0.0322`). |
| **Source** | `results/WAVE_A/step_wave_a_summary.json` → `pooled_rows[]`, tag `seedvar_frozen`, `k=12`, field `success_rate_pooled` |
| **Where 0.048 comes from** | `fixsnr_10db`, AWGN, k=12, SNR 0 dB: `success_rate_pooled = 0.048` — a **different arm**. |
| **Explanation** | The upper bound mixes the fixed-SNR training arm into the frozen row. Across M2 (AWGN + Rayleigh, all k) the frozen task range is 0.0159–0.0563, so `0.048` is not a frozen value under any reading. |
| **Recommended correction** | Table II frozen AWGN cell: `0.020–0.029`; Rayleigh `0.031–0.034 (mean 0.033)`. |

### C5 — "Untrained random model 0.016" has no stored artifact

| | |
|---|---|
| **Paper location** | Table II, line 122; §IV-B line 112 |
| **Paper claim** | "Untrained random model \| 0.016"; "an untrained model scores 0.016, so the metric is not trivially exploitable" |
| **Artifact value** | No `untrained` tag/arm/cell exists in `results/WAVE_A` (tags: `caps_h256`, `fixsnr_10db`, `lam_ctl_heavy`, `lam_ctl_light`, `long_12k`, `seedvar_frozen`, `wrecon_pos`; `caps_h256` is an invalid arm). No `untrained` eval file, no `untrained` field in the Wave-A summary. The nearest stored `0.016` is `m5_no_u`, AWGN, k=12, SNR 5 dB — an unrelated arm. |
| **Source** | `results/WAVE_A/step_wave_a_summary.json` (`pooled_rows`, `invalid_arms`, `train_rows`); `results/M5/step_m5_summary.json` → `m5a_pooled_rows` |
| **Explanation** | The number is carried over from thesis Table 5.4, where it also has no artifact. It is not reproducible from the repository. |
| **Recommended correction** | Either (a) remove the row/sentence, or (b) re-run and persist an untrained/random-init control (with the same E=1,000, 3 seeds) before claiming it. Do not keep `0.016` as-is. |

### C6 — The "crossover" headline applies to the reconstruction arm, not the task-oriented link

| | |
|---|---|
| **Paper location** | Abstract (line 7); contribution 1–2 (lines 21–22); §IV-A line 104; Discussion line 173; Conclusion line 183 |
| **Paper claim** | "neural ... wins at k ≤ 18, the coded link wins at k ≥ 24, and the two are effectively tied at k = 54"; "at k ≥ 24 the coded link overtakes it and ties at k = 54" |
| **Artifact value** | Shaped λ-heavy arm: k=12 `0.9966` (AWGN) / `0.9967` (Rayleigh); k=18 `0.9990` / `0.9960`; k=24 (K24, 3 seeds) `0.9963–0.9993`; k=54 `0.9986` / `0.9976`. Coded digital: k=12 `0.062`, k=18 `0.164`, k=24 `0.306`, k=30 `0.566`, k=54 `0.995`. |
| **Source** | `results/WAVE_A/step_wave_a_summary.json` (`lam_ctl_heavy`); `results/WAVE_C/step_wave_c_summary.json` (`pooled_rows`, k=18/54); `results/K24_SEEDS/step_k24_summary.json` (`spread_across_seeds`); `results/M2/step11_m2_summary.json` |
| **Explanation** | The crossover exists only between the **reconstruction** DeepJSCC arm (0.242/0.254/0.233/0.279/0.246) and the digital link (0.062/0.164/0.306/0.566/0.995). The paper's own task-oriented (λ-heavy) arm is at the ceiling (≈0.997–0.999) at **every** measured bandwidth, including k=24 and k=54. Table I does not contain the λ-heavy column, so the abstract's "neural vs coded" statement reads as if the task arm loses at high k, which the artifacts contradict. |
| **Recommended correction** | Split the claim into two sentences: (i) "the reconstruction-oriented DeepJSCC arm crosses over between k=18 and k=24"; (ii) "the task-oriented λ-heavy arm is at the task ceiling (0.996–0.999) at every measured k, so the benchmark is saturated and cannot rank it against the coded link above k=18." Add the λ-heavy column (or a separate table) so 0.998 is visible; the "tie at k=54" must name the λ-heavy arm and its E=1,000 protocol. |

### C7 — Fig. 4 caption claims twelve geometries; the figure is the pinned A-10 geometry only

| | |
|---|---|
| **Paper location** | Fig. 4 caption, line 149 |
| **Paper claim** | "A-10/A-17 required transmit SNR ... The shaped policy retains a 15–21 dB advantage across twelve geometries." |
| **Artifact value** | `fig_aerial_required_snr.png` is generated by `tools/make_figures_extensions.py` from `results/AERIAL/step_aerial_summary.json` only — one geometry (d=500 m, h=50 m, f_c=2 GHz), k=18/54, targets 0.90/0.99. The twelve-geometry sweep is `results/AERIAL_GEOM` (Table III), not plotted. |
| **Source** | `tools/make_figures_extensions.py` header/plot (only `results/AERIAL/step_aerial_summary.json`); `results/AERIAL_GEOM/step_aerial_geometry_summary.json` → `rows` (12 rows) |
| **Explanation** | The figure and the caption describe different datasets. The 15–21 dB range is true (A-17), but it is not in this PNG. |
| **Recommended correction** | Caption: "A-10 pinned geometry (500 m, 50 m, 2 GHz). The twelve-geometry sweep is Table III." Or regenerate the figure from A-17 (requires new plotting; not done in this audit). |

### C8 — Fig. 1 caption/aggregation mismatch with Table I

| | |
|---|---|
| **Paper location** | Fig. 1 caption, line 102 |
| **Paper claim** | "Success rate versus channel uses on AWGN." |
| **Artifact value** | The PNG has two panels at **0 dB** and **20 dB** (from `tools/make_figures.py`, `fig4_bandwidth_crossover`). Table I is the mean over 0–20 dB. At k=12 the figure shows reconstruction ≈0.264 (0 dB) and ≈0.24 (20 dB); Table I reports 0.242. Digital k=54 is 0.9736 at 0 dB but 0.995 in Table I. |
| **Source** | `tools/make_figures.py` (fig4 function: M2 AWGN at 0 and 20 dB); `results/M2/step11_m2_summary.json` |
| **Explanation** | Both are correct for their own aggregation, but the reader sees different numbers for the "same" curve. The caption must state the SNRs. |
| **Recommended correction** | Caption: "Success rate versus channel uses on AWGN at 0 dB (left) and 20 dB (right). Table I reports the 0–20 dB mean." Add one sentence in §IV-A distinguishing the per-SNR figure from the pooled table. |

### C9 — A-12 checkpoint range and "at most 2 dB"

| | |
|---|---|
| **Paper location** | §IV-C, line 151 |
| **Paper claim** | "the checkpoints sit at 0.02–0.03 success"; "moves the operating point by at most 2 dB relative to the Rayleigh-trained arm" |
| **Artifact value** | `eval_after_success_rate = 0.02734375` (k=18) and `0.01171875` (k=54) → **0.012–0.027** (not 0.02–0.03). A-13 gaps: k18 0.90 `−0.27`, 0.99 `−2.22`; k54 0.90 `−0.50`, 0.99 `−1.09` → max |gap| **2.22 dB**. |
| **Source** | `results/AERIAL_TRAIN/step_aerial_training_summary.json` → `train_rows[].eval_after_success_rate`; `results/AERIAL_TRAIN_TRUNC/step_aerial_training_summary.json` → `gaps` |
| **Explanation** | Rounding hides the k=54 checkpoint (0.012) and understates the worst gap (2.22 > 2). |
| **Recommended correction** | "checkpoints sit at 0.01–0.03 success"; "moves the operating point by at most about 2.2 dB". |

### C10 — Thesis k=54 digital `0.935` vs artifact `0.995` / `0.941` (thesis error; paper already uses 0.995)

| | |
|---|---|
| **Thesis location** | Table 5.2, line 587; narrative line 589; thesis conclusion |
| **Thesis claim** | AWGN digital at k=54 = **0.935**; task 0.036; recon 0.249 |
| **Artifact value** | M2 AWGN digital k=54 mean = **0.9947** (per-SNR `0.9736, 1.000, 1.000, 1.000, 1.000`). M2 Rayleigh digital k=54 mean = **0.9414** (per-SNR `0.7178, 0.9908, 0.9984, 0.9998, 1.000`). |
| **Source** | `results/M2/step11_m2_summary.json` → `pooled_rows[]` (`channel`, `k`, `method`, `snr_db`, `success_rate_pooled`, `n_seeds=3`, `episodes_pooled=15000`) |
| **Explanation** | No stored JSON contains an M2 k=54 digital success rate of 0.935. A repo-wide search found exactly one `success_rate = 0.935` in `results/`: `results/AERIAL_TRAIN/eval/frozen_rayleigh_k18/eval_0.json` (the A-12 frozen Rayleigh comparator at nominal 0 dB, k=18) — unrelated to M2 k=54. Git history of the M2 summary (dca0769, a5b0e17, b31feec, HEAD) never contains 0.935. The thesis Table 5.2 is closest to the **Rayleigh** column, not AWGN: recon `0.253/0.237/0.235/0.262/0.249` vs Rayleigh means `0.2525/0.2364/0.2333/0.2638/0.2577`; digital `0.064/0.165/0.324/0.629/0.935` vs `0.0643/0.1647/0.3222/0.6135/0.9414`. Even then, 0.629 and 0.935 do not match exactly. The k=54 value is closest to the Rayleigh mean (0.9414, −0.0064), not the AWGN mean (0.9947, −0.0597). |
| **Conclusion** | Thesis Table 5.2 is a legacy/mislabeled (Rayleigh-like) table, not reproducible from the frozen M2 summary. The **paper's Table I is correct** (0.995 from M2 AWGN) and should not be changed. Thesis Table 5.2 and its narrative need correction after approval. |

---

## 3. Major findings (should fix)

### M1 — λ-shaping does not remove non-finite steps under the 3GPP dispersion

* Paper: "λ-shaping bounds the encoder gradient ... and removes non-finite steps" (Fig. 3 caption, line 130); abstract says "conditioning effect ... rather than a capacity effect" without channel qualifier.
* Artifact: A-12 **is λ-heavy** (`(1, 0.1, 0.1, 2)`) and has `nonfinite_grad_steps = 311` (k=18) and `203` (k=54), `converged = false`, p50 `2.36×10^14`/`2.16×10^14`. Fig. 3 itself plots these A-12 cells (red, `analyse_training_stability.py` includes `AERIAL_TRAIN`).
* Source: `results/MECHANISM/step_stability_summary.json` (AERIAL_TRAIN cells); `tools/analyse_training_stability.py` lines 43–44, 293–390.
* Correction: scope the claim: "λ-shaping removes non-finite steps on AWGN/Rayleigh; it does **not** under the full 3GPP dispersion (A-12), which is why the resampling intervention (A-13) is needed." The paper already reports A-12 separately; the caption/abstract must not contradict it.

### M2 — "convergence in 10/10 cells" is ambiguous

* Artifact: Wave-C λ-heavy 4 cells + K24 λ-heavy 6 cells = 10 converged; A-13 adds 2 more (12/12); A-12 adds 2 failures (12/14 overall). The paper says 10/10 without saying which set.
* Source: `results/MECHANISM/step_stability_summary.json`.
* Correction: "10/10 standard-channel λ-heavy cells (Wave-C + K24); 12/12 including A-13; 0/2 under full 3GPP dispersion (A-12)."

### M3 — Table II caption `E = 1,000` but values are pooled over 3,000 episodes

* Wave-A: 3 seeds × 1,000 = 3,000 pooled. The caption says only `E = 1,000`.
* Source: `results/WAVE_A/step_wave_a_summary.json` → `pooled_rows[].n_seeds = 3`, `episodes_pooled = 3000`.
* Correction: "E = 1,000 per seed × 3 seeds (3,000 pooled); 85/90 slots complete, 2 documented cell failures."

### M4 — Wave-A incompleteness is not disclosed

* Summary `status = "incomplete"`, `train_failed = 2`, `eval_slots_complete = 85/90`, `eval_units_complete = 17/18`, 1 unreachable unit. The paper presents Wave-A as complete.
* Source: `results/WAVE_A/step_wave_a_summary.json` (`status`, `train_failed`, `eval_slots_complete`, `eval_units_unreachable`).
* Correction: add one sentence/limitation. (The λ-heavy λ-light cells used in Table II are complete; only unrelated cells failed, but the disclosure is still required.)

### M5 — "does not overtake ... below k = 54" is misleading

* Artifact: uncoded at k=54 `0.884` vs λ-heavy `0.998`; it never overtakes at k=54 either. The phrase "below k=54" implies the exception is at k=54.
* Source: `results/M5/step_m5_summary.json` → `m5b_uncoded_rows`; `results/WAVE_C/step_wave_c_summary.json`.
* Correction: "does not overtake the shaped neural arm at any measured bandwidth (0.884 vs 0.998 at k=54)."

### M6 — "identical to convolution at k = 12–30" is too strong

* Artifact: full 5G polar vs conv: k=12 `0.0654 vs 0.0654`; k=18 `0.1504 vs 0.1470`; k=24 `0.2926 vs 0.2818`; k=30 `0.5596 vs 0.5436`. Differences up to 0.016; Wilson intervals overlap but "identical" is not the same as "not significantly different".
* Source: `results/POLAR5G/step_polar5g_summary_E1000.json`; `results/POLAR/step_polar_summary_E1000.json`.
* Correction: "statistically indistinguishable in our sample" or give the numbers.

### M7 — "puncturing" vs truncation

* Paper: "108-bit budget is obtained by puncturing to an effective rate of 0.44".
* A-20 implementation: "first E non-NULL mother bits" (truncation), not the 38.212 circular-buffer rate-matching/puncturing chain.
* Source: `fyp_md_files/RESULTS_LDPC_2026-11-02.md` line 33; `results/LDPC/step_ldpc_summary.json` protocol.
* Correction: "truncating the mother code to the first E bits (rate-matching chain omitted)".

### M8 — "reconstruction MSE explodes to about 10^3" needs a metric definition

* Paper line 110. Two stored quantities disagree: training diagnostic `recon_mse_diag_final = 1605.16` (AWGN) / `1343.51` (Rayleigh) vs evaluation `aggregate_frozen_task.mse_mean = 0.5207` (AWGN 0 dB) / `0.6600` (Rayleigh 0 dB).
* Source: `results/ARCH_ABL/*/train_report.json` (`recon_mse_diag_final`); `results/ARCH_ABL/eval/*/eval_0.json` (`aggregate_frozen_task.mse_mean`).
* Correction: say "the training-time reconstruction-MSE diagnostic rises to ~1.3–1.6×10^3" and add that the evaluation MSE is 0.52–0.66. Otherwise a reader who checks the eval JSON sees 0.52, not 10^3.

### M9 — Protocol mixing in the "tie at k = 54"

* The 0.998/0.999 comes from Wave-C (E=1,000, seed 42); the 0.995 comes from M2 (E=5,000, 3 seeds). The paper's Table I says E=5,000. The comparison is not like-for-like.
* Source: `results/WAVE_C/step_wave_c_summary.json` (`n_seeds=1`, `episodes_pooled=1000`); `results/M2/step11_m2_summary.json` (`n_seeds=3`, `episodes_pooled=15000`).
* Correction: label the tie as "M2 coded link (E=5,000) vs Wave-C shaped arm (E=1,000)" or, better, quote both at the same budget.

### M10 — "Every mean reported below is recomputed from the stored per-SNR JSON artifacts" is false

* Counterexamples: MSE `0.049–0.054` (C3), Table II frozen `0.020–0.048` (C4), untrained `0.016` (C5), mechanism percentiles (C1). These are transcribed from the thesis, not recomputed.
* Source: line 84 vs the fields above.
* Correction: either remove the sentence or restrict it to the tables that are machine-generated (Table I, III, IV). Add a provenance note for the rest.

### M11 — "2–5% success" is a rounded range

* Artifact: M2 task across all cells `0.0159–0.0563`; Wave-A frozen `0.0200–0.0337`. "2–5%" misses the 1.6% (k=54) and 5.6% (Rayleigh k=12) endpoints.
* Correction: "roughly 2–5%" is acceptable prose, but the abstract/conclusion should use "2–6%" or "1.6–5.6%" if precision is claimed.

### M12 — 3.78% outage-tail fraction is doc-only

* The 3.78% figure appears in `fyp_md_files/GAP_ANALYSIS_RESPONSE.md` line 88 and in the A-13 train report `spec` narrative, but no JSON field stores the computed fraction. It is plausible and documented, but its provenance is weaker than the rest of the paper's numbers.
* Correction: either recompute/persist the fraction or cite it as "documented in the A-13 amendment report".

---

## 4. Numeric verification — table by table

### 4.1 Table I (M2 AWGN, 3 seeds, 15,000 episodes per condition)

| k | Paper recon | Artifact recon | Paper task | Artifact task | Paper digital | Artifact digital |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.242 | 0.2423 | 0.028 | 0.0276 | 0.062 | 0.0620 |
| 18 | 0.254 | 0.2543 | 0.018 | 0.0179 | 0.164 | 0.1640 |
| 24 | 0.233 | 0.2326 | 0.034 | 0.0343 | 0.306 | 0.3059 |
| 30 | 0.279 | 0.2794 | 0.021 | 0.0210 | 0.566 | 0.5656 |
| 54 | 0.246 | 0.2464 | 0.016 | 0.0162 | 0.995 | 0.9947 |

**Status: all 15 cells match** (rounding only). Source: `results/M2/step11_m2_summary.json` → `pooled_rows[]`, `success_rate_pooled`.

Rayleigh inline values: digital k=54 `0.9414` (paper 0.941 ✓); reconstruction k=54 `0.2577` (paper 0.258 ✓).

### 4.2 Table II (Wave-A, k=12)

| Arm | Paper AWGN | Artifact AWGN | Paper Rayleigh | Artifact Rayleigh | Status |
|---|---:|---:|---:|---:|---|
| Frozen | 0.020–0.048 | 0.0200–0.0285 | 0.033 | 0.0310–0.0337 (mean 0.0322) | **AWGN wrong** (C4) |
| λ-light | 0.995 | 0.9951 | 0.019 | 0.0194 | ✓ |
| λ-heavy | 0.997 | 0.9966 | 0.997 | 0.9967 | ✓ |
| Recon aux | 0.243 | 0.2425 | 0.244 | 0.2439 | ✓ |
| Untrained | 0.016 | **no artifact** | — | — | **unsupported** (C5) |

Text-only values: fixed-SNR `0.050` → artifact `0.0502` (AWGN only; Rayleigh not run) ✓; longer training `0.035` → artifact `0.0352` (**Rayleigh only**; no AWGN long_12k cell) — the paper should say which channel.

### 4.3 Table III (A-17, 12 geometries, k=54, E=1,000)

All six displayed rows match `results/AERIAL_GEOM/step_aerial_geometry_summary.json` → `rows[]`:

| d/h/fc | P_LoS paper/art. | PL_mean paper/art. | λ paper/art. | Digital paper/art. | Adv. paper/art. |
|---|---|---|---|---|---|
| 200/50/2.0 | 0.967 / 0.9672 | 85.4 / 85.42 | 74 / 74 | 89 / 89 | +15 / +15 |
| 200/50/3.5 | 0.967 / 0.9672 | 90.3 / 90.29 | 79 / 79 | 94 / 94 | +15 / +15 |
| 500/50/2.0 | 0.889 / 0.8887 | 95.6 / 95.63 | 87 / 87 | 108 / 108 | +21 / +21 |
| 500/100/3.5 | 0.945 / 0.9446 | 99.2 / 99.22 | 88 / 88 | 103 / 103 | +15 / +15 |
| 1000/50/2.0 | 0.772 / 0.7721 | 105.3 / 105.34 | 103 / 103 | 123 / 123 | +20 / +20 |
| 1000/100/3.5 | 0.853 / 0.8533 | 107.4 / 107.42 | 99 / 99 | 119 / 119 | +20 / +20 |

Other A-17 claims: 12 rows ✓; advantage 15–21 dB at 0.90 ✓; 12–15 dB at 0.99 ✓ (`lambda_advantage_k54_at_0.99_db`); 500/50/2 reproduces A-10 (87/108) ✓; distance ≈29 dB per 5× (74→103, 79→108) ✓; altitude recovers 9 dB at 1000 m (103→94) ✓; carrier +5 dB (74→79, 87→92, 103→108, etc.) ✓.

### 4.4 Table IV (LDPC BG2 vs convolutional, E=1,000)

| k | LDPC AWGN paper/art. | Conv AWGN paper/art. | LDPC Ray. paper/art. | Conv Ray. paper/art. |
|---:|---|---|---|---|
| 12 | 0.0656 / 0.0656 | 0.0654 / 0.0654 | 0.0646 / 0.0646 | 0.0620 / 0.0620 |
| 18 | 0.1518 / 0.1518 | 0.1470 / 0.1470 | 0.1522 / 0.1522 | 0.1574 / 0.1574 |
| 24 | 0.2910 / 0.2910 | 0.2818 / 0.2818 | 0.2914 / 0.2914 | 0.2878 / 0.2878 |
| 30 | 0.5546 / 0.5546 | 0.5436 / 0.5436 | 0.6154 / 0.6154 | 0.6082 / 0.6082 |
| 54 | 0.9780 / 0.9780 | 0.9952 / 0.9952 | 0.9438 / 0.9438 | 0.9396 / 0.9396 |

**Status: all 20 cells match.** Sources: `results/LDPC/step_ldpc_summary.json`; `results/POLAR/step_polar_summary_E1000.json`.

Polar: k=54 `0.9092` vs conv `0.9952` ✓ (paper 0.909/0.995); k=12–30 values 0.0654/0.1504/0.2926/0.5596 vs 0.0654/0.1470/0.2818/0.5436 (paper "identical" — see M6).

### 4.5 A-10 pinned geometry

| Claim | Artifact | Source |
|---|---|---|
| P_LoS 0.8887, PL_mean 95.63 | 0.8887485, 95.6310 | `results/AERIAL/step_aerial_summary.json` → `aerial` |
| λ k=54: 87 dB (0.90), 102 dB (0.99) | 87, 102 | `methods.lam_ctl_heavy_k54.required_tx_db` |
| Digital k=54: 108 dB (0.90), 117 dB (0.99) | 108, 117 | `methods.digital_k54.required_tx_db` |
| λ k=18: 91 dB (0.90), 107 dB (0.99) | 91, 107 | `methods.lam_ctl_heavy_k18.required_tx_db` |
| Digital k=18 ceiling 0.17 | max `curve_eff` = 0.1720 | `methods.digital_k18.curve_eff` |
| Advantage 21 dB / 15 dB | 108−87 = 21; 117−102 = 15 | derived |

### 4.6 A-12 / A-13 training

| Claim | Artifact | Source |
|---|---|---|
| Final losses 4,320 and 210 | 4320.129 / 210.321 | `AERIAL_TRAIN` → `train_rows[].task_loss_final` |
| 311 and 203 non-finite | 311 / 203 | `MECHANISM` → `cells_with_history[].nonfinite_grad_steps` |
| Checkpoints 0.02–0.03 | 0.0273 / 0.0117 | `train_rows[].eval_after_success_rate` (C9) |
| Loss 4,643 → 16.2 | initial 4643.19 → final 16.169 / 16.136 | `AERIAL_TRAIN_TRUNC/train_report.json` `history[0].total` |
| Max shift 2 dB | 2.22 dB | `AERIAL_TRAIN_TRUNC` → `gaps` (C9) |
| 3.78% tail | documented in `GAP_ANALYSIS_RESPONSE.md`, not in JSON | M12 |

### 4.7 Doppler (A-19)

| Claim | Artifact | Status |
|---|---|---|
| Fade duration 1.10 → 4.08 steps | 1.1048 → 4.0842 | ✓ |
| Deep fade 9.2–9.6% | 0.09382 / 0.09407 | ✓ |
| Neural max |Δ| 0.003 (k12), 0.001 (k54) | 0.0028 / 0.0008 | ✓ |
| Digital k54 0.912 vs 0.940 (−0.028) | 0.9116 vs 0.9396 (−0.0280) | ✓ |
| Retrained 0.993 vs 0.994 | 0.993 vs 0.994 | ✓ |
| Δ sweep 0.033–3.34 | 0.03336–3.3356 | ✓ |
| 32 oscillators, exact J0 | `M = 32`, ensemble J0 | ✓ (`project/channels/doppler.py`) |

### 4.8 CSI (Wave-B)

| Claim | Artifact | Status |
|---|---|---|
| Success flat to σ=0.4 | recon `0.2470→0.2476`; task `0.0342→0.0340`; digital `0.4212→0.4296` | ✓ |
| MSE degrades ~18% | `0.048825 → 0.057636` = **+18.04%** | ✓ |

Source: `results/WAVE_B/step_wave_b_summary.json`; per-cell `results/WAVE_B/eval/*/eval_*.json` → `aggregate_reconstruction.mse_mean`.

Note: the σ=0 row is the frozen M2 path (protocol note), and k=54 seed 43 is excluded (documented M2 failure). The paper does not mention the excluded unit; consider adding to limitations.

### 4.9 M5 term-removal and uncoded

| Claim | Artifact | Status |
|---|---|---|
| No control penalty: 0.016–0.027 | 0.016–0.027 | ✓ |
| No velocity: 0.992–0.999 | 0.994–0.997 | ✓ (range narrower) |
| No terminal: 0.992–0.999 | 0.992–0.999 | ✓ |
| k=18 no-u loss 5,213.5 | 5213.5 (declared failure) | ✓ |
| Uncoded 0.526 / 0.880 / 0.879 / 0.885 / 0.884 | 0.526 / 0.880 / 0.879 / 0.885 / 0.884 | ✓ |

### 4.10 K24, M4, M3

| Claim | Artifact | Status |
|---|---|---|
| k=24 shaped 0.995–1.000, range ≤0.004 | min 0.995, max 1.000, range ≤0.004 | ✓ |
| M4 k=54 B=8: 0.974 AWGN, 0.718 Rayleigh at 0 dB | 0.9736 / 0.7178 (E=5,000) | ✓ |
| M3 deficits 11, 10, 9 | 6·1+6−k for k=1,2,3 | ✓ |

---

## 5. k=54 digital 0.935 — investigation result

**Question:** thesis Table 5.2 reports AWGN k=54 digital = 0.935, while §5.2 M2 artifacts give 0.995. Which is right?

**Answer:** the artifact is right; the thesis value is not reproducible.

1. M2 AWGN digital k=54: mean `0.9947`; per-SNR `[0.9736, 1.000, 1.000, 1.000, 1.000]`.
2. M2 Rayleigh digital k=54: mean `0.9414`; per-SNR `[0.7178, 0.9908, 0.9984, 0.9998, 1.000]`.
3. `0.935` is neither: it is 0.0597 below the AWGN mean and 0.0064 below the Rayleigh mean.
4. A repo-wide search found exactly **one** unrelated `success_rate = 0.935`: `results/AERIAL_TRAIN/eval/frozen_rayleigh_k18/eval_0.json` (A-12 frozen Rayleigh comparator, nominal 0 dB, k=18). It is not an M2 k=54 digital value and the rest of Table 5.2 does not match that comparator.
5. Git history of the M2 summary (`dca0769`, `a5b0e17`, `b31feec`, `HEAD`) never contains 0.935 or 0.629.
6. The whole thesis Table 5.2 is closest to the **Rayleigh** column (recon 0.253/0.237/0.235/0.262/0.249 vs 0.2525/0.2364/0.2333/0.2638/0.2577; digital 0.064/0.165/0.324/0.629/0.935 vs 0.0643/0.1647/0.3222/0.6135/0.9414) — but even then k=30 (0.629 vs 0.6135) and k=54 (0.935 vs 0.9414) do not match exactly.
7. Conclusion: Table 5.2 was probably built from an older Rayleigh/Wave-B run (or a transcription of one) and labelled AWGN. The paper's Table I (0.995 AWGN) is correct and must stay. The thesis table/narrative should be corrected to 0.995 (AWGN) and 0.941 (Rayleigh) after approval, with the aggregation stated.

---

## 6. References audit (24 entries)

All 24 are cited in text; no orphan or missing citation. Bibliographic verification via Crossref/DOI/arXiv:

| # | Status | Finding / correction |
|---:|---|---|
| [1] | Minor | 3GPP TR 36.777 title is "Enhanced LTE support for aerial vehicles" (not "Study on Enhanced..."). Fix title. |
| [2] | OK | TR 38.901, 2019 ("Study on channel model for frequencies from 0.5 to 100 GHz"). |
| [3] | OK | TS 38.212, "Multiplexing and channel coding", 2023. |
| [4] | OK | TCCN 5(3) 567–579, 2019. |
| [5] | OK | TWC 20(12) 8081–8095, 2021; DOI 10.1109/TWC.2021.3090048. |
| [6] | OK | JSAC 41(1) 5–41, 2023. |
| [7] | OK | JSAC 40(1) 197–211, 2022. |
| [8] | Minor | Add pages 1–6; DOI 10.1109/INFOCOMMWKSHPS61880.2024.10620735. Authors/venue correct. |
| [9] | **Fix** | Now published: JSAC 43(7) 2575–2588, 2025 (not "early access"). |
| [10] | **Fix** | Now published: JSAC 43(9) 3041–3055, 2025 (not "early access"). |
| [11] | **Fix** | arXiv:2504.18317 is a different paper ("Task-Oriented Semantic Compression for Localization at the Network Edge"). The cited title is in **GLOBECOM 2025, pp. 1059–1064**, DOI 10.1109/GLOBECOM59602.2025.11432022. |
| [12] | **Fix** | Now published: TVT 75(2) 3354–3359, **2026** (not early access 2025). |
| [13] | OK | LCOMM 27(8) 2232–2236, 2023. |
| [14] | **Fix** | Journal version exists: TWC 23(12) 18903–18915, 2024 (not arXiv:2312.13182). |
| [15] | OK | TWC 23(10) 15002–15017, 2024; authors match Crossref. |
| [16] | OK | TCCN 8(2) 584–599, 2022. |
| [17] | OK | TCSVT 32(4) 2315–2328, 2022; author list matches. |
| [18] | OK | COMST 20(4) 2804–2821, 2018. |
| [19] | **Fix** | Published as IEEE WCL 9(6) 861–865, 2020 (not arXiv:1910.04969, 2019). |
| [20] | OK | ICLR 2015. |
| [21] | OK | NeurIPS 2019. |
| [22] | OK | Psychometrika 12(2) 153–157, 1947. |
| [23] | OK | JASA 22(158) 209–212, 1927. |
| [24] | Minor | Add DOI 10.5525/gla.thesis.85353; title/author/year verified (Enlighten Theses, Univ. Glasgow, 2025). |

In-text relevance is sound for all 24. The only content mismatch is [11]'s identifier (wrong arXiv ID for the right title).

---

## 7. Figures — provenance and caption check

| Figure | Source script | Data actually plotted | Caption issue |
|---|---|---|---|
| Fig. 1 (`fig4_bandwidth_crossover.png`) | `tools/make_figures.py` | M2 AWGN at **0 dB and 20 dB**, k=12–54, three methods | Caption omits 0/20 dB; Table I is SNR-mean → apparent mismatch (C8) |
| Fig. 2 (`fig_wave_a_arms.png`) | `tools/make_figures_wave.py` | Wave-A tags at k=1/12, both channels | Caption fine; figure includes arms beyond the table (fixed/long), acceptable |
| Fig. 3 (`fig_training_stability.png`) | `tools/analyse_training_stability.py` | All deep cells: Wave-C, K24, ARCH frozen, A-12, A-13 | "λ-shaping ... removes non-finite steps" contradicted by A-12 in the same plot (M1); "fourteen orders" not visible (C2) |
| Fig. 4 (`fig_aerial_required_snr.png`) | `tools/make_figures_extensions.py` | **A-10 only** (500 m/50 m/2 GHz; k=18/54; 0.90/0.99) | Caption claims A-10/A-17 + twelve geometries (C7) |

---

## 8. Unsupported / exaggerated claims and missing limitations

### Unsupported or overstated

1. "Fourteen orders of magnitude" attached to p50/p99 (C2).
2. Frozen p50/p99 values (C1).
3. `0.049–0.054` MSE (C3).
4. Table II frozen `0.020–0.048` (C4).
5. Untrained `0.016` (C5).
6. "coded link wins at k ≥ 24" without specifying the reconstruction arm (C6).
7. Fig. 4's "twelve geometries" (C7).
8. "Every mean ... recomputed from JSON" (M10).
9. "removes non-finite steps" as a general λ claim (M1).
10. "identical to convolution at k=12–30" (M6).
11. "at most 2 dB" / "0.02–0.03" (C9).
12. "does not overtake ... below k=54" (M5).
13. "reconstruction MSE explodes to 10^3" without saying training-diagnostic vs evaluation (M8).

### Missing limitations (should be added)

1. **Saturation:** the task arm is at 0.99+ for every measured k; the benchmark cannot rank methods above k≈18. The paper says this in the Discussion but not in the abstract/conclusion, and Table I hides it.
2. **Protocol mixing:** M2/M4 E=5,000 vs Wave-C/A-17/LDPC/A-12 E=1,000; single seed for Wave-C/A-10/A-17/A-12/A-13; 3 seeds only for M2/Wave-A/K24. No significance test for the extension comparisons.
3. **Wave-A incompleteness:** 85/90 slots, 2 failed cells, 1 unreachable unit; the untrained control is not persisted.
4. **3GPP dispersion training is λ-shaped too:** A-12 fails despite λ (311/203 non-finite), so the "λ fixes conditioning" claim is channel-scoped.
5. **Doppler retraining:** 0.993 vs 0.994 is a single-seed comparison; the "inside seed spread" claim needs the seed spread stated.
6. **LDPC:** no 38.212 rate-matching (already stated) and low-k block-length limitation (the A-20 report states it; the paper does not).
7. **CSI sweep:** the k=54 seed-43 unit is excluded; the paper does not mention it.
8. **No hardware/flight test** (already stated).

### Novelty / framing

The conditional framing ("we do not propose a new paradigm") is appropriate and should be preserved. The main framing problem is C6 (which arm the crossover describes), not the novelty claim itself.

---

## 9. Frozen vs λ diagnostic distinction — audit

| Experiment | Loss | Episodes / seeds | Paper label | Correct? |
|---|---|---|---|---|
| M2 | frozen `(1,0,0,2)` | E=5,000, seeds 42/43/44 | "frozen task" / Table I | ✓ |
| M4 | frozen, B=8 at k=54 | E=5,000 | "M4" | ✓ |
| ARCH_ABL | frozen, width 256 | E=1,000, seed 42 | "width-256 model" | ✓ (spec hidden=256) |
| Wave-A | λ light/heavy, recon, fixed, long | E=1,000, 3 seeds | "Wave-A diagnostic" | ✓, but incomplete/mislabeled rows (C4/C5/M3/M4) |
| Wave-C | λ heavy | E=1,000, seed 42 | "shaped arm" | ✓ |
| K24 | λ heavy | E=1,000, seeds 45–47 | "k=24 seed study" | ✓ |
| A-12 | λ heavy, 3GPP dispersion | E=1,000, seed 42 | "dispersion training negative result" | ✓ |
| A-13 | λ heavy + tail resampling | E=1,000, seed 42 | "resampling intervention" | ✓ |

The paper correctly separates frozen from λ in most places. The failures are: (i) the mechanism paragraph attributes A-12's p50 to the frozen arm (C1); (ii) the abstract's "neural vs coded" crossover silently switches arms (C6); (iii) the "tie" mixes E=5,000 and E=1,000 (M9).

---

## 10. Prioritized correction list (for approval; nothing applied)

### P0 — must fix (factual)

1. Mechanism p50/p99: `3.0×10^10` / `8.1×10^7` and `2.4×10^18` / `5.0×10^18`; delete `3.0×10^14`/`2.4×10^14` everywhere (C1).
2. Replace "fourteen orders" with a defensible statement (median ≈10 orders; p99 ≈17; max ≈15) or remove (C2).
3. Table I MSE: `0.044–0.047` (SNR mean) or `0.047–0.054` (0 dB) with the aggregation stated (C3).
4. Table II frozen AWGN: `0.020–0.029`; Rayleigh `0.031–0.034` (C4).
5. Remove or re-run the untrained `0.016` row (C5).
6. Rewrite the crossover claim so it names the reconstruction arm; add the λ-heavy arm/column and the saturation caveat (C6).
7. Fig. 4 caption: A-10 only, or regenerate from A-17 (C7).
8. Fig. 1 caption: 0 dB / 20 dB (C8).
9. "0.01–0.03" and "at most ~2.2 dB" (C9).
10. Scope the λ/gradient claim to AWGN/Rayleigh and acknowledge A-12 (M1).

### P1 — should fix

11. Table II caption: 3 × 1,000 = 3,000 pooled; 85/90 complete (M3/M4).
12. "does not overtake at any measured bandwidth" (M5).
13. "statistically indistinguishable" for polar (M6).
14. "truncation" not "puncturing" (M7).
15. Clarify training-diagnostic vs evaluation MSE for width-256 (M8).
16. Label the E=5,000 vs E=1,000 tie (M9).
17. Remove/qualify "every mean recomputed" (M10).
18. Add saturation/protocol/single-seed/Wave-B-exclusion limitations (§8).
19. Correct references [1], [9], [10], [11], [12], [14], [19]; add pages for [8], DOI for [24] (§6).

### P2 — polish

20. "2–5%" → "2–6%" or "1.6–5.6%" (M11).
21. 3.78% provenance note (M12).
22. Mention Wave-A's 2 failed cells and the Wave-B excluded unit.
23. State that `0.993`/`0.994` Doppler retraining is single-seed.

### Thesis-only corrections (after explicit approval)

- Table 5.2 → M2 AWGN means (0.242/0.254/0.233/0.279/0.246; task 0.028/0.018/0.034/0.021/0.016; digital 0.062/0.164/0.306/0.566/0.995) and add a Rayleigh table.
- §5.4 gradient numbers → C1 values.
- Table 5.4 frozen range and untrained row → C4/C5.
- Figure 5.1 caption → 0/20 dB (C8).
- Figure 5.5 caption → remove "fourteen orders" or restate.

---

## 11. Artifact integrity statement

- No experimental artifact, frozen configuration, thesis table, figure, or paper file was modified during this audit.
- Git status at the end of the audit: clean except this new report file.
- All recomputations used read-only parsing of the JSON files listed in §1; scratch scripts lived under the OS temp directory, not the repository.
- Test-suite claim verified: `111 passed in 196.70s`.

**Awaiting explicit approval before applying any correction.**
