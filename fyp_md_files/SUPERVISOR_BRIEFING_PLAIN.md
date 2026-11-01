# Supervisor Briefing — What the Problem Was, How It Was Solved, and What Was Done on 3GPP

**Purpose:** one plain-language document that answers the four questions raised in review:
*what was the problem, how was it solved, was the 3GPP model actually implemented, and what is still missing.*

Every number below is copied from a stored artifact. File paths are given so each claim can be checked independently.

---

## 0. Ek nazar mein (Roman Urdu)

1. **Masla:** task-oriented JSCC low bandwidth par **2.5%** success de raha tha, jabke reconstruction **23.7%** — sawal ye tha ke ye failure *task-orientation* ka qudrati masla hai ya *loss function* ka.
2. **Jawab:** loss function ka masla tha. Sirf distance penalty (`lambda_u = lambda_v = 0`) model ko useful gradient nahi de rahi thi. Control aur velocity penalties `0.1` karte hi wahi model **0.999** par chala gaya.
3. **Mechanism:** frozen arm mein encoder gradient `3 x 10^14` tak ja raha tha (14 orders of magnitude zyada), is liye training kabhi converge nahi hui. Lambda on karne se gradient `O(1)` ho gaya.
4. **3GPP:** haan, implement hua hai — `TR 36.777` UMa-aerial: LoS probability, LoS/NLoS path loss, altitude, shadowing. Link budget (A-10), dispersion training (A-12), stable variant (A-13). **Doppler nahi hai**, aur sirf ek geometry hai.
5. **Baqi:** M5 ka formal term-removal, multi-geometry, LDPC, aur naya "aligned" framework (ATROC/rateless/CIV/UEP) — ye thesis ke baad ka kaam hai.

---

## 1. The problem, in one paragraph

The frozen protocol trained the task-oriented arm with

```
L_task = 1.0 * L_distance + 0.0 * L_control + 0.0 * L_velocity + 2.0 * L_terminal
```

At `k = 18` (3 channel uses per state dimension) this arm reached **0.025** success against **0.237** for reconstruction-oriented JSCC and **0.165** for the digital link. The open question was whether this is a property of task-oriented communication itself, or an artefact of how the objective was written. The thesis answers that question with five controlled experiments.

---

## 2. Five competing explanations, five experiments

Each row removes one alternative explanation. Nothing was retrained away; negative results are kept.

| # | Alternative explanation | Experiment | Result | Verdict |
|---|---|---|---|---|
| 1 | "The objective is wrong — the model is never told that control effort and velocity matter." | Add `lambda_u = lambda_v = 0.1` (Wave-A, k = 12; Wave-C, k = 18) | `0.020` → **`0.997`** at k = 12; `0.025` → **`0.999`** at k = 18 | **This is the cause** |
| 2 | "The network is too small." | Width 128 → 256 (A-9) | `0.027` AWGN / `0.041` Rayleigh — still fails | Not capacity |
| 3 | "It just needs reconstruction help." | Position-weighted reconstruction auxiliary arm (Wave-A `wrecon_pos`) | `0.24` — does not fix it | Not the objective family |
| 4 | "It needs longer training / a better SNR curriculum." | 12,000 steps; fixed 10 dB training | `0.035`; `0.050` | Not an optimisation-budget issue |
| 5 | "The metric is easy to game." | Untrained random model | `0.016` | Control: metric is honest |

**Sources:** `results/WAVE_A/`, `results/WAVE_C/`, `results/ARCH_ABL/`, `fyp_md_files/RESULTS_WAVE_A_WAVE_B_2026-10-21.md`, `fyp_md_files/RESULTS_ARCH_2026-10-26.md`, thesis Tables 5.5 and 5.7.

---

## 3. Why the fix works (the mechanism, not just the correlation)

The project stored per-step gradient measurements for 18 training runs. Extracting them (`fyp_md_files/RESULTS_MECHANISM_2026-10-27.md`):

| Arm | median encoder gradient | 99th percentile | non-finite gradient steps | converged |
|---|---|---|---|---|
| Frozen task loss (`lambda_u = lambda_v = 0`) | `~3.0 x 10^14` | `~2.4 x 10^14` | 75–100 | 0 / 2 |
| Lambda-shaped (`0.1 / 0.1`) | `~3.3–4.1` | `~57–113` | 0 | 10 / 10 |

The gap is roughly **fourteen orders of magnitude**. The trigger is not parameter count (the width-256 frozen cells explode too) — it is **rare high-error episodes** dominating the gradient. Adding the control and velocity terms keeps those episodes from producing a degenerate gradient, so the same architecture and the same data now converge.

This is the project's mechanism finding: **task-oriented JSCC does not fail because task orientation is wrong; it fails when the task loss is myopic and unbounded.**

---

## 4. Was the 3GPP model actually implemented? — Yes, with a stated boundary

### 4.1 Implemented (`project/channels/aerial.py`)

| Component | Implementation | Standard |
|---|---|---|
| LoS probability `P_LoS` | Function of horizontal distance **and** altitude, with the three `d1/d2` branches and the `c` term | TR 36.777 aerial UMa |
| LoS path loss | `28.0 + 22 log10(d3D) + 20 log10(fc)` | TR 36.777 high-altitude branch |
| NLoS path loss | `-17.5 + (46 - 7 log10 h) log10(d3D) + 20 log10(40 pi fc / 3)` | TR 36.777 |
| Mixture | `PL_mean = P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS` | Expected loss over the LoS/NLoS state |
| Altitude | Explicit `h` in `P_LoS`, `PL_NLoS`, and `d3D = hypot(d2D, h)` | Altitude-dependent, as required |
| Shadowing | Log-normal, `sigma = 4 dB` LoS / `6 dB` NLoS, one draw per episode | Height/state-dependent dispersion |
| LoS/NLoS switching | One Bernoulli draw per episode at the episode geometry | Probabilistic switching |

### 4.2 What was measured on it

- **A-10 — link budget** (thesis Table 5.9): at `d2D = 500 m`, `h = 50 m`, `fc = 2 GHz`, `P_LoS = 0.8887`, `PL_mean = 95.63 dB`. Required transmit SNR for 0.90 success: lambda-shaped **91 dB** at k = 18 and **87 dB** at k = 54; digital **108 dB** at k = 54 and **never reaches 0.90** at k = 18 (payload-limited at 0.17). Advantage: **21 dB** at k = 54, unbounded at k = 18.
- **A-12 — train under the full dispersion:** both cells collapse (final losses 4,320 and 210; 311 and 203 non-finite gradient steps). Reported as a **negative result**, not retrained.
- **A-13 — train with the >20 dB outage tail (3.78% of episodes) resampled:** stable (4,643 → 16.2), operating point moves by **at most 2 dB** versus the Rayleigh-trained arm — inside sampling error. Honest claim: dispersion training does not move the operating point.

### 4.3 What is *not* implemented (stated in the thesis, §3.13 and Chapter 7)

- **Doppler / velocity-induced fading** — the model is block-fading with no time correlation.
- **Time-correlated shadowing** — one independent draw per episode.
- **Aerial small-scale fading, antenna patterns, noise figure, bandwidth.**
- **Multiple geometries** — only `d2D = 500 m`, `h = 50 m`, `fc = 2 GHz`.

**One-line answer:** it is a **3GPP large-scale + LoS/NLoS dispersion layer**, not a full 3GPP link-level simulator. That boundary is written in the thesis, not hidden.

---

## 5. Review points mapped to evidence

| Review point | Status | Where the evidence is |
|---|---|---|
| Replace generic Rayleigh with 3GPP TR 36.777 / 38.901 | **Done** (large-scale + dispersion) | `project/channels/aerial.py`; thesis §3.8, Fig 3.1; A-10, A-12, A-13 |
| LoS/NLoS probabilistic switching | **Done** | `los_probability()`, `sample_aerial_offsets_db()` |
| LoS/NLoS path loss with 3D distance, frequency, altitude | **Done** | `los_path_loss_db()`, `nlos_path_loss_db()` |
| Altitude and velocity effects / Doppler | **Altitude yes, Doppler no** | thesis §3.13, Ch. 7 |
| Perfect CSI → add estimation error or justify | **Done** | Wave-B, 350 cells, 70/70 bit-exact gates (A-4); thesis §5.5 |
| Run M5 loss ablation (non-zero control/velocity) | **Done as Wave-A/Wave-C**; formal term-removal remains | Tables 5.5, 5.7; A-8 |
| Architecture ablation (larger network) | **Done — negative** | A-9, `results/ARCH_ABL/` |
| Objective ablation (position-weighted reconstruction) | **Done — does not fix it** | Wave-A `wrecon_pos` = 0.24 |
| Diagnose training instability | **Done** | `RESULTS_MECHANISM_2026-10-27.md`; A-14 seed variance |
| Modern digital baseline (LDPC / Polar) | **Polar + full 5G NR polar done; LDPC not done** | A-11, A-15; thesis §5.9 |
| Aligned framework (ATROC), rateless, CIV, UEP | **Not done — new research** | Future work, Ch. 8 |
| Containerised, DOI, configs, seeds | **Container + env + configs + seeds done; DOI not yet** | `Dockerfile`, `environment.yml`, `requirements.txt`, `project/config.py` |
| Reproducibility | **Frozen protocol, deterministic seeds, 57 tests** | `PROJECT_STATUS.md`, `PROTOCOL_AMENDMENTS.md` |

---

## 6. Why the main table still shows the 0.025 number

This is the point that is easiest to misread.

The main benchmark (M2) was **pre-registered and frozen before execution**. Changing the loss after seeing the result would invalidate the benchmark. So the thesis reports the frozen result as the primary comparison, and reports the lambda-shaping fix as a **dated post-freeze amendment** (A-8, A-9, A-14) with its own experiment IDs.

That is a methodological strength: the negative result and the fix are both on the record, in the order they happened, and the fix was not silently substituted into the main table.

---

## 7. What actually remains

| Item | Type | Cost | Blocks thesis? |
|---|---|---|---|
| M5a — remove one lambda term at a time (12 cells) | compute | ~8–12 h CPU | No |
| M5b — uncoded digital reference (10 cells) | evaluation | minutes | No |
| M5c — cross-channel training (4 cells) | compute | ~2–4 h | No |
| A-10 multi-geometry (3 distances x 2 heights x 2 carriers) | evaluation | cheap | No |
| Doppler + time-correlated shadowing | simulator change | days | No |
| LDPC baseline | implementation | days | No (Polar already covers it) |
| ATROC / rateless / CIV / UEP | new research | weeks–months | No |
| IEEE paper (6 pages, numeric citations) + DOI | writing | weeks | No |

**Rule that still applies:** M5 is not run without a dated protocol amendment (A-16) and explicit written supervisor approval.

---

## 8. Six sentences that summarise the whole project

1. We compared task-oriented JSCC, reconstruction JSCC, and a digital link at **matched bandwidth** for UAV navigation state transmission.
2. The pre-registered task-oriented arm **failed at low bandwidth** — 0.025 against 0.237 for reconstruction.
3. We showed the failure is **not** capacity, not training length, and not the metric — it is the **loss function**.
4. Adding control and velocity penalties (`0.1 / 0.1`) takes the same architecture from **0.025 to 0.999**, because the encoder gradient improves by **fourteen orders of magnitude**.
5. On the **3GPP UMa-aerial** link the shaped policy needs **87–91 dB** transmit SNR for 0.90 success, while the digital link needs **108 dB** at k = 54 and cannot reach 0.90 at k = 18.
6. The remaining work is the formal M5 term-removal, a multi-geometry sweep, Doppler, and the journal extension — not a re-do of the thesis.
