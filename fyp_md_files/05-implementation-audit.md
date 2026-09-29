# Implementation Audit — Task-Oriented DeepJSCC for UAV Navigation

Scope: audit of the frozen specification against the source documents in `fyp_md_files/`
(03-final-mathematical-implementation-specification.md, 04-digital-baseline-and-training-eval-protocol.txt,
final_variables-1, final_variables-2, and the channel-verification notes). No project code written yet.

---

## 0. FROZEN DECISIONS — UPDATE (supersedes the corresponding proposals below)

- **A1 — Digital bandwidth: Option A, strict channel-use matching (PRIMARY).** For every primary experiment, Task-Oriented DeepJSCC, Reconstruction DeepJSCC, and the digital baseline use the same k complex channel uses; ρ = k/6. The digital baseline never receives additional channel uses in the primary comparison. Any larger-budget digital curve exists only as a clearly labeled SECONDARY reference and never replaces the primary matched-budget comparison.
- **A11 — Digital channel code: convolutional, K = 7, R_c = 1/2, generators (171, 133)₈, soft-decision Viterbi decoding, Gray-labeled QPSK, unit-average-power constellation (E_s = P = 1). No LDPC in the first implementation.** An uncoded chain is permitted only as a documented digital ablation, never as the primary digital system.
- **Framing rule.** The exact-fit identity 6B = k·R_c·log2(M) is NOT enforced. The enforced invariant is **6B + 6 ≤ k** (source payload + termination tail must fit inside the budget; documented padding fills any remainder — full design in §8). Infeasible (k, B) combinations are reported as infeasible at configuration time; k, bandwidth, modulation, code rate, or the DeepJSCC definition are never silently changed.
- **Controller FROZEN: Kp = 1.0, Kv = 2.0 (user decision after the oracle gate).** The damping term is explicitly −Kv·v; the acceleration clip only bounds commanded acceleration and is NOT a source of damping. Kv = 1.0 FAILED the noiseless oracle gate (SR 87.5%, boundary exits 31.3%, ~16% overshoot from ζ = 0.5). With Kv = 2.0 (ζ = 1.0) the oracle gate passed under the UNCHANGED frozen episode distribution: SR = 100.00% (5000/5000), exits 0.00%, mean final distance 0.011 m, max |u| before clip 15.9 m/s² (saturation fraction 8.4%), zero NaN/Inf.
- **A6 RESOLVED empirically:** with the frozen controller, success is 100% in every initial-distance bin up to d0 ≈ 20.6 m (full-workspace sampling), so **d_max is not required**; config keeps D_MAX = None and the full-workspace episode distribution stands.
- **Step 0–2 implemented and verified** (`project/config.py`, `data/normalization.py`, `uav/dynamics.py`, `uav/controller.py`, `uav/environment.py`; 11 unit tests + oracle gate all pass; project-local venv with torch 2.4.1+cpu / numpy 1.26.4).
- **Task-loss weights FROZEN: λ_d = 1.0, λ_u = 0.0, λ_v = 0.0, λ_T = 2.0 (user decision in Step 5).** Project design choice (A7 proposed defaults, consistent with final_variables-1 §2.7 "λ_T = 2.0, start simple"), NOT literature-derived. L_d and L_T drive optimization; L_u and L_v are diagnostics only. Rollout convention frozen (audit 1.1 row 12): L_d and L_v sum over post-transition states t = 1..T (t = 0 excluded), L_u over applied controls t = 0..T−1.
- **Step-6 soft-decision conventions FROZEN (user decision).** (i) QPSK soft demapper = EXACT log-MAP, LLR = log[P(bit=0)/P(bit=1)]; for the frozen Gray-QPSK mapping this is exactly linear: LLR_I = 4√2·Re(y_eq)/σ²_eff, LLR_Q = 4√2·Im(y_eq)/σ²_eff. (ii) Rayleigh uses PER-SYMBOL perfect-CSI scaling: after the frozen ZF equalizer, σ²_eff = σ²/max(|h|², ε_eq) per codeword — never the average σ². (iii) Uncoded QPSK chain = validation/reference path for the BER gate only; the formal uncoded ablation remains a later M5 experiment.
- **Step-7 evaluation conventions FROZEN (user decision).** Scope = `project/evaluation/` harness only (metrics, statistics, paired loop, tests) — no experiments/, no full training, no thesis-level execution. Checkpoints regenerated-by-seed (never committed); harness is checkpoint-agnostic. Time-to-goal = first t with d_t ≤ r_g; success-conditional statistics exclude failures (failures keep the T+1 sentinel, reported as failure_count). Statistics implemented: mean, std, success count/total, Wilson 95% CI, continuity-corrected McNemar (n01 == n10 handled explicitly as p = 1), paired t-test (incomplete-beta t p-value, no SciPy). Fixed SNR per (channel, method, condition) — never sampled per episode/step. Paired evaluation: episodes AND n[episode, step, symbol] AND h[episode, step] pre-generated once per condition from purpose-separated streams and handed to every method as exact tensors (bitwise-identical across methods, hard-gated). Minimal backward-compatible channel extension added for pre-generated realizations (Step-3 equations untouched).

---

## 1. SPECIFICATION CHECK

### 1.1 Mathematical components I understand (will implement exactly as frozen)

| # | Component | Frozen content |
|---|-----------|----------------|
| 1 | State | s_t = [x, y, vx, vy, x_g, y_g] ∈ R^6 |
| 2 | Dynamics | 2-D double integrator: p_{t+1} = p_t + Δt·v_t + ½Δt²·u_t ; v_{t+1} = v_t + Δt·u_t (differentiable) |
| 3 | Goal / success | e_t = p_t − p_g ; d_t = ‖p_t − p_g‖₂ ; success ⇔ d_t ≤ r_g |
| 4 | Controller | u*_t = −Kp(p_t − p_g) − Kv·v_t, then clip to ±u_max; identical for all three methods; gains in config |
| 5 | Normalization | per-component affine map to [−1,1] with fixed config ranges [a_i, b_i]; never data-derived |
| 6 | Encoder | MLP 6 → h → h → 2k real outputs, PReLU; split into k complex symbols; **k complex channel uses** (never 2k) |
| 7 | Bandwidth | ρ = k/6 (frozen prompt; resolves the ρ = 6/n_c variant in final_variables-2) |
| 8 | Power norm | per-codeword z = √(kP)·z̃/√(z̃*z̃ + ε), P = 1, ε numerics-only |
| 9 | AWGN | y = z + n, n ~ CN(0, σ²I); σ² = P·10^(−SNR_dB/10); Var(n_R) = Var(n_I) = σ²/2 |
| 10 | Rayleigh | y = h·z + n, h ~ CN(0,1), **one scalar h per codeword** (= per state transmission); E[|h|²] = 1 |
| 11 | CSI | perfect CSIR for the main Rayleigh experiment; ZF equalization before decoding; **same CSI policy for all baselines** |
| 12 | Task training | differentiable closed-loop rollout; L_task = λd·Ld + λu·Lu + λv·Lv + λT·LT with Ld = (1/T)Σ_{t=1..T}‖p_t−p_g‖², Lu = (1/T)Σ_{t=0..T−1}‖u_t‖², Lv = (1/T)Σ_{t=1..T}‖v_t‖², LT = ‖p_T−p_g‖² — a **project-specific navigation objective** (task-oriented principle from the literature; not claimed from Shao et al.) |
| 13 | Reconstruction baseline | identical capacity/channel/budget/protocol; loss L_recon = (1/6)‖ŝ̄ − s̄‖² on normalized state; evaluated through the same controller |
| 14 | Digital baseline | normalize → uniform scalar quantize → bits → channel code → modulate → channel → demodulate → decode → dequantize → controller; quantizer/code/mod explicitly documented as **project adaptation** (no JPEG/LDPC-image pretense, no Shannon-capacity pseudo-baseline) |
| 15 | Evaluation | success rate primary; final/average distance, control effort, time-to-goal, reconstruction MSE; AWGN and Rayleigh reported separately; ≥ 3 seeds; mean ± std; success count / total; identical test episodes |
| 16 | Ablations | task-vs-recon loss, AWGN-vs-Rayleigh, SNR grid, bandwidth grid, task-loss component removals |
| 17 | Infra | config-driven, seeds for Python/NumPy/Torch(+CUDA), CUDA-if-available, no internet needed |

Note on your message: the L_task formula rendered with `*` between terms (likely a paste artifact). Doc 03 is unambiguous — I read it as a **sum** of weighted terms.

### 1.2 Genuine ambiguities (no values invented; each with a proposed resolution for your sign-off)

- **A1 — Digital bandwidth matching (the critical one).** A per-step budget of k ∈ {1,2,3} complex symbols cannot carry a useful quantized 6-D payload: info bits available = k·R_c·log2(M). Even uncoded 256-QAM at k=3 gives 24 bits ≈ 4 bits/scalar ≈ 1.07 m position resolution ≫ r_g = 0.5 m. Constraint: 6·B = k·R_c·log2(M) for exact fit. See §5 and §6 for the two resolution options. → **FROZEN: Option A (strict matching) with explicit framing/padding — see §0 and §8.**
- **A2 — Task-oriented decoder output interface.** Your frozen pipeline says "decoder → estimated state → controller" (6-D ŝ, same interface as reconstruction). Earlier drafts (final_variables-1/2) proposed a 4-D task vector q_t = [p_g−p, v]. Proposal: **6-D ŝ for both neural methods** — it matches the frozen prompt, keeps the controller interface identical, and makes reconstruction MSE defined for all three methods. The 4-D variant is treated as superseded.
- **A3 — Velocity limit enforcement.** The frozen dynamics do not clip v, but "velocity limits" are required config. Proposal: enforce ‖v_{t+1}‖ ≤ v_max by scaling in the dynamics (differentiable a.e.), identically in all methods including the digital path and the oracle sanity run.
- **A4 — Success criterion.** Frozen prompt: "d_t ≤ goal_radius". Doc final_variables-1 §14: ∃t ∈ {0..T}. Proposal: **∃t (first entry counts)**; time-to-goal = first such t; final distance reported separately.
- **A5 — Boundary-exit / failure semantics.** Leaving the workspace must terminate eval episodes (counted as failure). During training rollouts, early termination breaks the uniform computation graph. Proposal: eval = stop + fail; training = fixed-horizon rollout, boundary handled by the clipped normalization (no early stop).
- **A6 — d_max (max start-goal distance) is unspecified.** Only d_min = 4 m is drafted. Worst-case in-page distance is ≈ 22.6 m; with u_max = 2, v_max = 5 and a Kp=Kv=1 PD loop (ζ ≈ 0.5, ω_n ≈ 1), far corners may time out within T = 100·0.1 s. Proposal: freeze d_max **after** the controller sanity run (§7, step 3) — e.g., d_max ≈ 14–16 m — not before.
- **A7 — λ defaults.** Proposal (matches your Step-2 draft): λd = 1, λT = 2, λu = λv = 0 for the headline model; the full four-term loss kept behind an ablation flag. Not to be presented as literature values.
- **A8 — Training-SNR law granularity.** U(0,20) dB is drafted; per-sample vs per-batch is open. Proposal: per-sample.
- **A9 — Training state distribution for the reconstruction baseline.** It cannot use closed-loop rollouts (controller excluded from its training). Proposal: states generated by rolling out the **noiseless full-information PD controller** (nominal trajectory distribution), frozen and shared as the data protocol; task-oriented trains on its own channel-perturbed rollouts by design.
- **A10 — Equalization guard form.** Raw y/h explodes when |h| → 0 (also in gradients). Proposal: ỹ = y·h*/max(|h|², ε_h) with ε_h in config — identical to ZF away from zero, bounded for training.
- **A11 — Digital channel code family.** LDPC (literature-adjacent but a from-scratch decoder is heavy) vs convolutional K=7, R_c = 1/2, generators (171,133)₈ + soft Viterbi (classic, exact, deterministic) vs uncoded reference. Proposal: convolutional + Viterbi for v1, uncoded as an ablation, all documented as project adaptation. → **FROZEN: convolutional K = 7, R_c = 1/2, soft Viterbi, QPSK — see §0 and §8.**
- **A12 — Test-episode count.** 5,000 (Step-2 draft) vs 100–500 (protocol doc). Proposal: 5,000 for final results (binomial SE ≈ 0.7% at p = 0.5), 500 for smoke tests.

### 1.3 Inconsistencies between your own documents (resolved by the frozen prompt)

1. **Bandwidth convention:** final_variables-2 defines ρ = d_s/n_c with n_c ∈ {36, 18, 12}; final_variables-1 and your frozen prompt define ρ = k/6 with k ∈ {1, 2, 3}. I implement the frozen prompt (k = ρ·6 complex uses).
2. **Task representation:** 4-D q_t (final_variables-1/2) is superseded by the frozen prompt's 6-D estimated-state pipeline (see A2).
3. **Gain notation:** Kv (frozen prompt) = Kd (earlier docs). One symbol, Kv, in code.

---

## 2. CONFIGURATION PARAMETERS

Status key: **[FROZEN]** specified by the research design (your prompt) · **[LIT]** literature-derived convention · **[PROJ]** project implementation choice (proposed value needs your sign-off unless already in your Step-2 draft, marked "(draft)").

### 2.1 UAV and episode

| Parameter | Proposed value | Status |
|---|---|---|
| dt | 0.1 s | [PROJ] (draft) |
| T_max | 100 steps (10 s) | [PROJ] (draft) |
| Workspace | [−8, 8]² | [PROJ] (draft) |
| d_min start–goal | 4 m | [PROJ] (draft) |
| d_max start–goal | not required (resolved empirically: SR = 100% in all d0 bins up to ≈ 20.6 m); D_MAX = None | [PROJ] resolved (A6) |
| v_0 | (0, 0) | [PROJ] (draft) |
| goal_radius r_g | 0.5 m | [PROJ] (draft) |
| v_max | 5 m/s | [PROJ] (draft); enforcement per A3 |
| u_max | 2 m/s² | [PROJ] (draft) |
| Boundary exit | eval: fail + stop; train: fixed horizon | [PROJ] (A5) |
| Success rule | ∃t ∈ {0..T}: d_t ≤ r_g | [PROJ] (A4) |

### 2.2 Controller

| Parameter | Proposed value | Status |
|---|---|---|
| Kp | 1.0 | [FROZEN] user decision |
| Kv | 2.0 (damping term −Kv·v; clip is not damping) | [FROZEN] user decision after oracle gate (Kv = 1.0 failed: SR 87.5%, exits 31.3%) |
| Clip | ±u_max per axis | [FROZEN] |
| Shared across all methods | yes, single implementation | [FROZEN] |

### 2.3 Normalization

| Parameter | Proposed value | Status |
|---|---|---|
| Position ranges | [−8, 8] (= workspace) | [PROJ] derived |
| Velocity ranges | [−v_max, v_max] | [PROJ] derived |
| Formula / clipping | affine to [−1,1], clip inside | [FROZEN] |
| Ranges from test data | forbidden | [FROZEN] |

### 2.4 Communication (common to all methods)

| Parameter | Proposed value | Status |
|---|---|---|
| P | 1 | [FROZEN] (matches DeepJSCC P = 1 convention) |
| ε_power | 1e-12 | [PROJ] numerics |
| ε_eq (guard) | 1e-6 | [PROJ] numerics (A10) |
| ρ grid | {1/6, 1/3, 1/2} → k ∈ {1, 2, 3} | [PROJ] (draft; ratios from literature ranges) |
| SNR test grid | {0, 2, …, 20} dB | [PROJ] (draft; proposal says 0–20 dB) |
| SNR training | U(0, 20) dB, per sample | [LIT] SNR-range training + [PROJ] range (A8) |
| Fading coherence | one h per codeword (per state transmission) | [FROZEN] |
| CSI | perfect CSIR + ZF equalization, all methods | [FROZEN] (a project extension of the original DeepJSCC no-CSI setup — never cite as Bourtsoulatze et al.) |
| Update rate | 1 transmission per control step | [PROJ] (draft) |
| Paired test realizations | same noise/h across methods per episode | [PROJ] protocol |
| E_b/N₀ | supplementary label only: SNR − 10log10(R_c·log2M) for digital | [LIT] convention |

### 2.5 Neural models

| Parameter | Proposed value | Status |
|---|---|---|
| Hidden dims | 128, 128 | [PROJ] |
| Activation | PReLU | [FROZEN] |
| Encoder out | 2k reals → k complex, stacked split z_i = a_i + j·a_{k+i} | [FROZEN]; ordering fixed as project convention |
| Decoder | 2k → 128 → 128 → 6, PReLU | [PROJ] (mirror) |
| Decoder output | 6-D normalized state (both neural methods) | [PROJ] interpretation (A2) |
| dtype | float32, channel math on real tensors [B,2k] with σ²/2 halves (complex-dtype unit check) | [PROJ] |

### 2.6 Training protocol

| Parameter | Proposed value | Status |
|---|---|---|
| Optimizer | Adam | [LIT] |
| Learning rate | 1e-3 | [LIT] default; validate on val set |
| Batch size | 256 episodes (task rollouts) / 4096 states (recon) | [PROJ] |
| Duration | ~20k gradient steps + early stop on val | [PROJ] TBD |
| Split | episodes 70/15/15 | [PROJ] protocol |
| Checkpoint rule | val success (task) / val recon MSE (recon) | [PROJ] protocol |
| Episodes | 10,000 train / 2,000 val | [PROJ] (draft) |
| Gradient clipping | global-norm 1.0 | [PROJ] numerics |

### 2.7 Digital baseline

| Parameter | Proposed value | Status |
|---|---|---|
| Quantizer | uniform scalar, endpoint-inclusive, on normalized state, Δ = 2/(2^B−1) | [PROJ] adaptation (documented) |
| Bits/scalar B | integer bits per component; feasibility per §8.4 (6B + 6 ≤ k); B = 8 for the secondary reference | [FROZEN] design rule (§8) |
| Channel code | conv K = 7, R_c = 1/2, (171,133)₈ + soft-decision Viterbi; zero-tail termination (6 bits); uncoded only as documented ablation | [FROZEN] (A11) |
| Modulation | Gray-labeled QPSK only (no QAM menu in v1) | [FROZEN] (A11) |
| Symbol power | unit-average constellation, E_s = P = 1 | [LIT] convention |
| Framing | payload 6B + tail 6 → N_coded = 12B + 12 → n_pay = 6B + 6 payload symbols; documented filler symbols pad to exactly k; invariant 6B + 6 ≤ k enforced at config time; infeasible combos reported, never silently relaxed | [FROZEN] (§8) |
| BER/BLER | digital-only diagnostics, reported separately | [PROJ] protocol |

### 2.8 Evaluation and statistics

| Parameter | Proposed value | Status |
|---|---|---|
| Test episodes | 5,000 final / 500 smoke | [PROJ] (draft; A12) |
| Training seeds | {42, 43, 44} | [PROJ] (draft) |
| Test seed | fixed, separate stream; RNG separated per purpose (episodes / init / shuffle / noise-train / noise-test) | [PROJ] protocol |
| Reporting | mean ± std, success count/total, Wilson 95% CI | [PROJ] protocol |
| Significance | McNemar (success), paired t (continuous) | [PROJ] protocol |

---

## 3. ARCHITECTURE (exact tensor shapes)

B = batch. Task-oriented training path (eval is identical under `no_grad`):

```
s_t              [B,6]   raw physical state
s̄_t = norm(s_t)  [B,6]   per-component affine, clip to [−1,1]

ENCODER   Linear(6,128) → PReLU → Linear(128,128) → PReLU → Linear(128,2k)
a_t              [B,2k]
z̃_R = a_t[:, :k] , z̃_I = a_t[:, k:2k]        (stacked I/Q convention, fixed)
z̃_t              [B,k]   complex, k channel uses, ρ = k/6

POWER NORM  z_t = √(kP) · z̃_t / √(Σ_i |z̃_t,i|² + ε)     [B,k]  per-codeword

CHANNEL (AWGN)      y_t = z_t + n_t ,  n ~ CN(0, σ²I)     [B,k]
CHANNEL (Rayleigh)  h_t [B,1] ~ CN(0,1) per codeword; y_t = h_t·z_t + n_t
EQUALIZE            ỹ_t = y_t · h* / max(|h|², ε_h)       [B,k]
TO REAL             [Re ỹ ; Im ỹ]                          [B,2k]

DECODER   Linear(2k,128) → PReLU → Linear(128,128) → PReLU → Linear(128,6)
ŝ̄_t              [B,6]
ŝ_t = denorm(ŝ̄_t) [B,6]

CONTROLLER  e_t = ŝ_t[:,0:2] − ŝ_t[:,4:6]
            u*_t = −Kp·e_t − Kv·ŝ_t[:,2:4]
            u_t = clip(u*_t, −u_max, u_max)               [B,2]

DYNAMICS    p_{t+1} = p_t + Δt·v_t + ½Δt²·u_t             [B,2]
            v_{t+1} = v_clip(v_t + Δt·u_t, v_max)          [B,2]
            s_{t+1} = [p_{t+1} ; v_{t+1} ; p_g]            [B,6]

loop t = 0 … T−1 ; losses Ld, Lu, Lv, LT (scalars) accumulate
```

Digital chain (same normalization, controller, dynamics, channel):

```
s̄_t [B,6] → uniform scalar quantize (B bits/comp) → indices [B,6] → bits [B,6B]
→ append 6 zero tail bits → info block [B, 6B+6]
→ conv encode (R_c = 1/2) → coded bits [B, 12B+12]
→ Gray map → payload QPSK symbols [B, 6B+6]
→ append n_pad = k − (6B+6) filler symbols → transmitted symbols [B, k]   (== DeepJSCC budget)
→ same channel → equalize → drop padding symbols → LLRs [B, 12B+12] → Viterbi → b̂ [B,6B] → dequantize → ŝ̄_t [B,6]
→ same controller → same dynamics
```

Budget rule enforced at config time: **6B + 6 ≤ k** (payload + termination must fit; padding fills the rest; infeasible combos reported — see §8).

---

## 4. CHANNEL VERIFICATION

Each item states the check the implementation must pass before it is trusted.

1. **Complex representation.** z̃_i = a_i + j·a_{k+i} (stacked). k complex uses; 2k is only the real width of the final linear layer. Assert: encoder output reinterpreted to [B,k] complex; config sanity print ρ = k/6. The decoder input width is 2k.
2. **Power normalization.** Per-codeword: after normalization, (1/k)Σ_i|z_i|² = P·(1 + O(ε)) — exact per sample, not merely on batch average. Individual symbols may have unequal energies (expected). Guard: ε only prevents division by ~0; verify ε ≪ typical ‖z̃‖² so it never reshapes the distribution.
3. **AWGN variance.** σ² = P·10^(−SNR_dB/10); sample n_R, n_I ~ N(0, σ²/2) each so E|n|² = σ². Empirical test: 10⁶ samples, mean |n|² within 0.1% of σ². The classic bug (σ² per I/Q component) produces a systematic 3 dB SNR shift — detectable by matching the simulated Gray-QPSK BER to Q(√(E_s/N₀)) at fixed SNRs (textbook closed form used **only** to validate the simulator, not as a baseline).
4. **Rayleigh fading.** h_R, h_I ~ N(0, 1/2) ⇒ E|h|² = 1, |h| Rayleigh, phase uniform. One scalar h per codeword: h has shape [B,1], broadcast over the k symbols; a fresh h per state transmission. Empirical tests: histogram vs Rayleigh pdf; E|h|² ≈ 1; within-codeword symbol correlation = 1 (block), across codewords ≈ 0 (rejects the per-symbol fast-fading bug).
5. **CSI / equalization.** Perfect CSIR; ỹ = y·h*/max(|h|², ε_h) ≡ y/h for |h|² ≫ ε_h. After equalization the per-codeword SNR is |h|²P/σ² (instantaneous, heavy-tailed); the config SNR axis is the **average** SNR = P/σ² since E|h|² = 1. The identical equalizer is applied in the digital baseline. The guard keeps training gradients finite when |h| → 0.
6. **SNR calculation & bandwidth ratio.** One function `snr_db → σ²` is the single source of truth for every method; no per-method recomputation, no E_b/N₀ on the comparison axis (digital-only supplementary label). Runtime asserts: ρ = k/6 with integer k; the digital chain transmits exactly k QPSK symbols, of which the first 6B + 6 carry the coded payload (padding discarded at the receiver), and 6B + 6 ≤ k must hold (misconfiguration fails loudly as infeasible rather than silently changing the budget — §8).

---

## 5. FAIRNESS CHECK

- **Single shared implementations** (not copies) of: dynamics, controller, normalization, channel models, equalizer, episode/RNG pipeline, metrics. A method can only differ in its encoder/quantizer → decoder/dequantizer map and its training objective.
- **Identical for all three methods:** 6-D state; dynamics and limits; goal and success criterion; test episodes (same start/goal per episode index, frozen seeds); paired channel realizations (same n and h per episode/step); SNR grid and definition (P/σ² per complex symbol); P = 1; budget: exactly k transmitted symbols for every method, digital payload symbols 6B + 6 ≤ k with documented padding (asserted, §8); 1 transmission per control step; perfect CSIR + ZF equalization; controller gains and clipping; horizon and metrics.
- **Legitimately different (the independent variable):** how the state is mapped to channel symbols and back, and the training objective. The digital baseline may carry fewer effective source bits at matched k — that is a *result* (bandwidth-limited separation starves), not an unfairness, provided the budget equality holds and its configuration is reported.
- **Guard rails in code:** runtime asserts (budget equality, shapes, identical seeds); an "oracle perfect-state, no channel" run allowed only as a clearly labeled upper reference, never a fourth comparator; normalization ranges from config only; no statistic ever computed on test data; digital BER/BLER reported as link diagnostics, never substituted for task metrics.

**Decision A1 — FROZEN (Option A, strict matching):** in every primary experiment the digital baseline uses exactly the same k complex channel uses as both DeepJSCC systems — no extra channel uses, ever. Larger-budget digital curves may exist only as clearly labeled SECONDARY references (§8.7, matrix row M4) and never replace the primary matched comparison. The full packet/framing design, feasibility arithmetic, and final experiment matrix are in §8.

---

## 6. IMPLEMENTATION RISKS (ways this could silently become mathematically inconsistent)

1. **I/Q variance factor-2 bug** — sampling n_R, n_I with variance σ² instead of σ²/2 shifts every SNR by 3 dB. Mitigation: the §4.3 empirical test.
2. **Channel-use miscounting** — treating 2k real outputs as 2k uses, letting the digital chain transmit n_dig ≠ k symbols, or counting padding/tail bits as source information. Mitigation: runtime asserts (exactly k transmitted symbols; payload = 6B + 6 ≤ k; padding dropped before LLR extraction) + config sanity print (§8).
3. **Wrong normalization granularity** — per-symbol or per-batch power normalization instead of per-codeword changes the SNR semantics. Mitigation: per-sample check that (1/k)Σ|z_i|² = P.
4. **Fast vs block fading** — drawing h per symbol instead of per codeword destroys the slow-fading comparison. Mitigation: h shape [B,1] and the correlation test in §4.4.
5. **Asymmetric CSI** — equalizing one method and not another, or ε_h applied in only some paths. Mitigation: one shared equalizer module.
6. **SNR-axis drift for digital** — quoting E_b/N₀ for the coded/modulated chain against E_s/N₀ for DeepJSCC. Mitigation: single σ² source of truth; E_b/N₀ only as a label.
7. **Normalization leakage or drift** — ranges computed from data, or different clip behavior between train and eval; denormalization using unclipped bounds. Mitigation: constants in config, one pair of functions used everywhere.
8. **Controller information leak** — feeding any ground-truth quantity (e.g., true goal or true position) to one method's controller. Every controller input must come from that method's received representation. Mitigation: controller consumes only the decoded/denormalized state tensor.
9. **Autograd pitfalls in the rollout** — detaching z or h, or clipping away all gradient (dead zones at saturation are physically correct but must be understood); noise/h must enter as constants (leaves), while gradients flow through z, decoder, encoder. Mitigation: gradient-flow unit test (loss.backward reaches encoder weights).
10. **Loss off-by-one** — Ld/Lv summed over the wrong index range, Lu including t = T, or LT evaluated at T−1. Mitigation: implement exactly as §1.1 row 12; one shared loss used for both training and reporting.
11. **Success/failure semantics mismatch** — ∃t vs terminal-only success, and inconsistent boundary-exit handling between training and eval. Mitigation: A4/A5 frozen in config; one episode-termination module.
12. **Digital strawman or capacity ghost** — accidentally giving the digital chain capacity-achieving performance, or crippling it via a coarse quantizer without reporting the budget arithmetic. Mitigation: Option A/B reporting rules; textbook BER-curve validation of the implemented modem.
13. **RNG contamination** — one global seed reused across episode generation, init, shuffling, and noise breaks paired evaluation. Mitigation: dedicated `torch.Generator` per purpose; test noise/h regenerated identically per (episode, step) for every method.
14. **Quantizer/normalization mismatch** — quantizing with a different range or step convention than the dequantizer assumes, or forgetting the clip before dequantization. Mitigation: quantize/dequantize round-trip unit test (error ≤ Δ/2 noiseless).
15. **Velocity handling asymmetry** — v_max enforced in one method's loop but not another's. Mitigation: clip inside the shared dynamics module (A3).

---

## 7. CODING ORDER (safest sequence; each step has a validation gate before the next)

0. **`config.py` + reproducibility utils.** All §2 parameters, seed setter (Python/NumPy/Torch/CUDA), device selection, RNG-stream separation, runtime asserts (k integer, budget identity, ranges). Gate: `python -m config` prints a validated configuration summary.
1. **`data/normalization.py`.** Normalize/denormalize pair. Gate: round-trip exactness on and off range; clip behavior matches config.
2. **`uav/dynamics.py` + `uav/controller.py` + controller sanity gate.** Noiseless full-information PD closed loop. **Freeze Kp, Kv, T_max, d_max here** (A6) based on measured success rate (target ≈ 100%). Gate: success ≈ 1, trajectories physically sane, no NaNs; pure-control validation (no comms involved).
3. **`channels/` (complex_utils, awgn, rayleigh) + verification script.** All §4 checks: noise/fading statistics, power-norm exactness, Gray-QPSK BER vs Q(√(E_s/N₀)) on AWGN, Rayleigh block structure. Gate: every §4 test passes numerically.
4. **`models/` encoder/decoder + `losses/reconstruction_loss.py` → smoke-train reconstruction DeepJSCC** at one (SNR, k) on AWGN. Gate: val MSE decreases smoothly; no NaNs; shapes correct end-to-end.
5. **`losses/task_loss.py` + differentiable rollout (`models/task_oriented.py`) → smoke-train task-oriented DeepJSCC** (AWGN, k = 3). Gate: gradient reaches encoder weights; closed-loop success > 0 and improves over training; task model beats random-encoding control.
6. **`baselines/digital.py`.** Sub-steps, each gated: (a) quantizer noiseless round-trip; (b) uncoded modem BER vs theory on AWGN; (c) coded chain (Viterbi) BER vs theory; (d) Rayleigh with the shared equalizer; (e) full chain into the shared controller/dynamics. Framing asserts per §8 are enforced at config time.
7. **`evaluation/` (metrics, statistics, evaluate.py).** Paired harness over frozen test episodes with per-purpose RNG streams. Gate: identical episodes/noise verified across methods; metrics reproduce hand-computed values on a tiny case.
8. **`training/common.py`, `train_task.py`, `train_reconstruction.py`.** Full protocol (SNR-range training, 70/15/15, checkpoint rules). Gate: one full seed runs clean end-to-end.
9. **`experiments/awgn.py`, `experiments/rayleigh.py`, `experiments/ablations.py`, `main.py`.** Smoke grid first (2 SNRs × 2 ρ × 3 methods), then the full matrix (2 channels × 11 SNRs × 3 ρ × 3 methods, 3 seeds) with results tables (mean ± std, success counts, Wilson CIs).

Rationale: every layer is validated against a closed-form reference (control theory, channel statistics, textbook BER, quantizer distortion) before the next layer is stacked on top of it, so any later inconsistency is localizable to one newly added module.

---

---

## 8. DIGITAL PACKET / FRAMING DESIGN — FROZEN (A1 + A11)

Notation: **B = integer quantization bits per state component** (the letter B denotes batch size elsewhere in this document; it does not appear in this section). Channel code: convolutional, constraint length K = 7, rate R_c = 1/2, generators G1 = (171)₈, G2 = (133)₈ (the standard K = 7, rate-1/2 code); the G1/G2 output ordering is a fixed implementation convention to be stated once in the code. Modulation: Gray-labeled QPSK (m = log2 M = 2 coded bits per complex symbol), unit-average-power constellation.

### 8.1 Packet layout (one state transmission = one control step)

```
field              bits                  notes
----------------------------------------------------------
quantized state    6B                    B bits/component × 6 components;
                                         component-major, MSB-first per component
                                         (fixed project convention)
termination tail   6                     K − 1 = 6 zero bits, zero-tail flush
----------------------------------------------------------
information block  N_info = 6B + 6       tail bits are overhead, NOT source info
conv encoding      N_coded = 12B + 12    R_c = 1/2, no puncturing
QPSK payload       n_pay = 6B + 6        Gray labels, unit average power
QPSK padding       n_pad = k − n_pay     known filler symbols, exists iff n_pay < k
----------------------------------------------------------
transmitted        k                     == DeepJSCC budget; ρ = k/6 for all methods
```

With K = 7, R_c = 1/2, QPSK and integer B, every quantity above is automatically an integer for every B (12B + 12 is always even), so a fractional-symbol case can never arise; the only possible failure mode is the budget upper bound handled in §8.3.

### 8.2 Padding and termination rules (explicit)

1. Padding is added after channel coding as filler QPSK symbols mapped from zero bits; the Viterbi block (branch metrics, traceback) is untouched.
2. The receiver discards the last n_pad equalized symbols before LLR extraction; padding never reaches the Viterbi decoder or the dequantizer.
3. Padding symbols are transmitted at the same per-symbol average power P = 1 and occupy channel uses like any other symbol, so every method transmits a k-symbol codeword with identical total energy kP.
4. Padding and tail bits carry no source information and are never counted as payload in any report (accounting per packet: 6B useful source bits; 6 overhead tail bits; 6B + 6 payload symbols; k − 6B − 6 padding symbols).
5. Zero-tail termination is chosen over tail-biting (which would give n_pay = 6B with no tail) because tail-biting requires circular Viterbi decoding; the +6-bit tail is documented overhead. Feasibility conclusions in §8.5 are unchanged under either convention.

### 8.3 Budget enforcement

- Config-time invariant: **n_pay = 6B + 6 ≤ k.** Violation raises `InfeasibleDigitalConfig(k_min = 6B + 6, requested_k, B)`; the experiment runner records the cell as INFEASIBLE. k, ρ, modulation, code rate, and the DeepJSCC definition are never silently changed.
- Runtime asserts: transmitted symbols == k for every method; payload symbols == 6B + 6; padding == k − 6B − 6 ≥ 0; Viterbi input length == 12B + 12.

### 8.4 Feasibility arithmetic

| B | N_src = 6B | N_info = 6B+6 | N_coded = 12B+12 | n_pay = 6B+6 | k_min | position resolution = 16 m/(2^B − 1) |
|---|---|---|---|---|---|---|
| 1 | 6 | 12 | 24 | 12 | 12 | 16 m (2 levels/component — degenerate) |
| 2 | 12 | 18 | 36 | 18 | 18 | ≈ 5.33 m |
| 3 | 18 | 24 | 48 | 24 | 24 | ≈ 2.29 m |
| 4 | 24 | 30 | 60 | 30 | 30 | ≈ 1.07 m (max quantization error ±0.53 m ≳ r_g = 0.5 m) |
| 8 | 48 | 54 | 108 | 54 | 54 | ≈ 0.063 m (fine quantization resolution relative to the configured workspace) |

General feasibility rule: B is feasible at budget k iff **B ≤ (k − 6)/6**, B integer ≥ 1.

### 8.5 Consequence for the primary ρ grid

k ∈ {1, 2, 3} requires k_min = 6B + 6 ≥ 12 for any B ≥ 1 (even B = 0 with zero-tail needs 6 tail bits → 6 symbols > 3) ⇒ **the digital baseline is INFEASIBLE at every primary ρ ∈ {1/6, 1/3, 1/2}**. This holds under either termination convention (tail-biting gives k_min = 6B ≥ 6 > 3). The matrix reports these cells as INFEASIBLE with the k_min arithmetic shown — a substantive finding: at ρ ≤ 1/2 a separated quantize–code–modulate chain cannot frame even a 1-bit-per-component description of the 6-D state, while DeepJSCC remains executable at low k because it does not require an explicit quantization/coding packet. No budget, rate, or modulation is relaxed to force digital into these cells. This infeasibility is an architectural/resource result of the specified separated digital baseline (this quantizer, this code, this modulation, this framing); it must not be generalized to all possible digital communication systems.

### 8.6 Feasible matched-budget operating points (primary)

| k | ρ = k/6 | B = (k−6)/6 | n_pad | assessment |
|---|---|---|---|---|
| 12 | 2 | 1 | 0 | feasible; degenerate 2-level quantizer |
| 18 | 3 | 2 | 0 | feasible; exactly fits |
| 24 | 4 | 3 | 0 | feasible |
| 30 | 5 | 4 | 0 | feasible; quantizer error already ≳ r_g |
| 54 | 9 | 8 | 0 | feasible; B = 8 provides fine quantization resolution relative to the configured workspace (navigation benefit to be verified experimentally) |

At each of these k values all three methods run with the same budget; the neural architectures are unchanged (encoder output width 2k, decoder input 2k).

### 8.7 Final experiment matrix

| Row | Purpose | Methods | k / ρ | Channels | SNR grid | Seeds |
|---|---|---|---|---|---|---|
| M1 | Task-vs-reconstruction loss ablation (neural only) | Task-O, Recon | {1, 2, 3} / {1/6, 1/3, 1/2} | AWGN, Rayleigh | {0, 2, …, 20} dB | 3 |
| M2 | Matched-budget three-way comparison (**primary**) | Digital + Task-O + Recon | {12, 18, 24, 30, 54} per §8.6 (digital B = (k−6)/6) | AWGN, Rayleigh | {0, 5, 10, 15, 20} dB (compute-bounded; extendable) | 3 |
| M3 | Digital infeasibility cells | Digital | {1, 2, 3} | — | — | — |
| M4 | SECONDARY bandwidth reference (clearly labeled, optional) | Digital B = 8 (n_dig = 54, ρ_dig = 9) vs neural curves at primary ρ | mixed | AWGN, Rayleigh | {0, 5, 10, 15, 20} dB | ≥ 1 |
| M5 | Ablations (task-loss components, uncoded digital, cross-channel) | per §1.1 row 16 | per ablation | per ablation | subset | 3 |

Rules: M2 is the primary three-way comparison (equal k, equal power, equal episodes and noise/fading realizations). M4 exists only to show how much bandwidth a separated design needs; it is labeled SECONDARY in every table and legend and reports both ρ values. M3 cells display INFEASIBLE with the §8.5 arithmetic. M1 remains the primary task-vs-reconstruction grid at the proposal's bandwidth ratios.

---

*Implementation status: Steps 0–6 complete and verified as recorded in §9 below (regression totals 11 + 43 + 21 + 10 + 28 + 33). Step 7 complete and verified: `project/evaluation/` harness — metrics.py (exact audit row-15 formulas, identical schema for oracle/recon/task/digital), statistics.py (mean/std, success counts, Wilson 95% CI, continuity-corrected McNemar, paired t via incomplete beta — dependency-free), evaluate.py (Condition dataclass + pre-generated paired episodes/noise/h handed bitwise-identically to all methods; explicit structured infeasible status for digital at k ∈ {1,2,3}); 18/18 checks (Gates A–H); minimal backward-compatible channel extension for pre-generated realizations (Step-3 equations untouched). Known smoke-scale observations for full training: encoder grad norms grow late in training (global-norm clip 1.0 does the real work — monitor/lower LR in full runs); λ_u = 0 so control effort legitimately rose as a diagnostic; reconstruction-MSE diagnostic rose while task loss fell (the expected task-vs-fidelity trade-off). Decision status: A1, A11, controller gains, task-loss λ, Step-6 LLR conventions, Step-7 evaluation conventions FROZEN; A2–A12 otherwise carry proposed defaults (A6 resolved: no d_max needed).*

---

## 9. CHECKPOINT LOG (git checkpoint policy — one entry per completed step)

Frozen task-loss weights (user decision, Step 5) — **project design choice, NOT a literature-derived value**:

```text
λ_d = 1.0
λ_u = 0.0
λ_v = 0.0
λ_T = 2.0
```

```text
Step: 0-2
Status: PASS
Commit: ddc41b0 (root commit)
Branch: main
Tests: 11/11 unit tests + oracle gate (SR 100%, 0/5000 exits)
Date: 2026-09-29

Step: 3
Status: PASS
Commit: 8517ee7
Branch: main
Tests: 43/43 channel verification checks
Date: 2026-09-29

Step: 4
Status: PASS
Commit: 8e4dc95
Branch: main
Tests: 21/21 structural + 10/10 smoke checks
Date: 2026-09-29

Step: 5
Status: PASS
Commit: bb82334584949d76719d4a035a80af44c24f7727
Branch: main
Tests: 28/28 Step-5 checks; regression 11/11 + 43/43 + 21/21 + 10/10
Date: 2026-09-29

Step: 6
Status: PASS
Commit: f58cb77b23555635b331761b52987f062130161a
Branch: main
Tests: 33/33 Step-6 checks (Gates A-E); regression 11/11 + 43/43 + 21/21 + 10/10 + 28/28
Date: 2026-09-29

Step: 7
Status: PASS
Commit: 787ca849b9bcf76ecc8091fcbb63820dadb847eb
Branch: main
Tests: 18/18 Step-7 checks (Gates A-H); full regression 11 + 43 + 21 + 10 + 28 + 33
Date: 2026-09-29
```

Step-7 gate results (tests/test_evaluation.py):

```text
Gate A (hand-computed metrics): two hand-walked trajectories — success, final
  distance, avg distance (t=0 excluded), control effort, time-to-goal (first hit;
  argmax over the hit vector), normalized-state MSE (perfect=0, offset=0.09),
  failure sentinel T+1 — all match hand values exactly.
Gate B (Wilson 95% CI): implementation == hand closed form for (7,50), (0,10),
  (10,10); n=0 safe.
Gate C (mean/std): matches reference calculation to 1e-12.
Gate D (paired realization identity, HARD): pre-generated noise and h are
  bitwise identical across reconstruction / task / digital captures; the
  materialization itself is deterministic. All methods share the same k
  (matched budget) so the noise width matches.
Gate E (determinism): full harness rerun -> bitwise-identical metric tables.
Gate F (interface parity): oracle/recon/task produce the identical 7-key
  metric schema + aggregate row; digital at k=3 (B unset) returns the explicit
  structured INFEASIBLE status (never fake metrics).
Gate G (statistical pairing): McNemar counts only discordant pairs (symmetric
  discordance = p 1 explicitly); paired t is sign-antisymmetric and preserves
  pairing; constant nonzero difference -> t = inf, p = 0.
Gate H (regression): 11 + 43 + 21 + 10 + 28 + 33 all PASS.
```

Step-6 gate results (all expected/empirical/tolerance reported by tests/test_digital.py):

```text
Gate A (quantizer): 5/5 — Δ = 2/(2^B−1) exact; max granular error ≤ Δ/2
  (e.g. B=8: 0.00392 ≤ 0.00391+1e-9); endpoints exact; clip both ways; round-trip exact.
Gate B (uncoded Gray-QPSK): empirical BER vs Q(√SNR) = 0.5·erfc(√(SNR/2)) on AWGN,
  N = 8×10^6 bits per point: 0 dB 0.158763 vs 0.158655; 6 dB 0.023003 vs 0.023007;
  10 dB 0.000783 vs 0.000783 (all within 4σ binomial tolerance). Axis: SNR = P/σ² = Es/N0.
Gate C (conv + Viterbi): impulse response matches (1+D³+D⁴+D⁵+D⁶, 1+D+D³+D⁴+D⁶) by hand
  and by reference vector; zero-tail drives state 7→0; noiseless chain = 0 bit errors
  in 7680; single-symbol corruption fully recovered.
Gate D (Rayleigh): block fading verified inside the digital packet; bitwise
  reproducible; CSI-scaled LLRs finite under the ε_eq guard (|h| = 1e-6 case).
Gate E (full chain): noiseless output == quantizer-only reconstruction (dev 0.0);
  framing 6B=48 → info 54 → coded 108 → payload 54 → pad 0 → k=54 asserted;
  all feasible k ∈ {12,18,24,30,54} execute; AWGN + Rayleigh finite and in-range;
  closed-loop digital navigation (B=8, k=54, 14 dB) reaches 0.174 m in 60 steps.
Infeasibility: k ∈ {1,2,3} raises InfeasibleDigitalConfig (k_min = 12) — reported as
  architectural infeasibility, never relaxed.
```

Continuation marker: `fyp_md_files/PROJECT_STATUS.md` (machine-independent handoff; resume procedure inside).
