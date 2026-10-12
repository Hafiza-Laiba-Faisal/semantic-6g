# Extensions Spec — Paper-Track Directions (implementation-ready)

This document turns the four research directions proposed in the supervisor's
roadmap into concrete, buildable specs anchored to the existing codebase.
Priority order reflects effort-to-payoff for the paper track. Each item lists:
objective, math, integration points (files), runtime, and the starting commit.

---

## 1. Full ATROC — Aligned Task- & Reconstruction-Oriented JSCC (Phase-3 fix)

**Objective.** Show a positive contribution: recon encoder robustness +
task-guided final stage.

**Architecture.** Keep the frozen 6→128→128→2k encoder. Add a lightweight
"information reshaper": decoder input x = y_eq ∈ R^{2k} passes through
G = a small residual MLP producing reconstructed state estimate ŝ; then a
task head R = W·ŝ ∈ R^4 outputs the goal-relative representation
(control uses R). Train with a **joint** loss:

    L_ATROC = λ_rec L_recon + λ_task L_task_rollout
            (λ_rec = 1.0, λ_task = 0.5 grid; L_recon = frozen formula from
             losses.reconstruction_loss; L_task = frozen task_loss)

This is exactly the ablation framework Wave-A arm A2 expects; the full version
uses *both* heads. Differentiability: fully differentiable (no hard ops).

**Integration points.**
- New file: `project/models/atroc.py` (subclass `ReconstructionDeepJSCC` with
  a `reshape_head: 2k→128→6` + `task_head: 6→4`).
- New trainer: `project/training/train_atroc.py` — closed-loop task rollout
  (like train_task_oriented) + weighted recon term in minibatch; reuse all
  Step-9C machinery.
- Matrix: 6 configs (λ_rec × λ_task from {1, 0.5, 0.2} × channels) × 3 seeds
  ≈ 36 runs at flux 0.37 s/step × 3000 steps ≈ 11 h CPU.

**Paper role.** The constructive fix if the diagnosis shows a loss/optimizer
artifact; also cost of replacing the pure "it's impossible" conclusion.

---

## 2. Modern digital baseline — 5G NR Polar (Phase-3 comparison)

**Objective.** De-strawman the K=7-conv digital arm.

**Design.** Reed-Muller-style 5G-Polar (using frozen CRC-24C + 3GPP clause-7.3.3
polar sequence for the matched packet sizes B ∈ {1,2,3,4,8}) decoding by soft
SC/SCL-4 with CRC; encode to BPSK/QPSK as frozen Gray-QPSK did. This is
algorithmic (no training) so matrix steps identical to the Step-6 baseline;
only the **coding block changes**. Masks:

- SC-List SCL: L=8, CRC-24C-aided.
- Use `sionna` (TF) on CPU or a 100-line NumPy polar kernel; I recommend the
  NumPy kernel to stay torch/NumPy-only.

**Integration points.**
- New: `project/baselines/digital_polar.py` with the same
  `DigitalBaseline` interface (quantize → polar encode → modulate → channel →
  demod → Viterbi→SC decode → dequantize).
- Validation: BER-vs-Eb/N0 curve must match the polar bound for B=1 at
  SNR 0–8 dB; run the 3-way eval as Step-11 did but with digital_polar
  replacing digital (config `baselines.digital_polar`).
- Runtime: eval-only (no training): ~2.5 h CPU for the full k-grid.

**Paper role.** If digital still wins with modern FEC, the claim is far
stronger; if it loses, the negative result deepens. Either way publishable.

---

## 3. Rateless DeepJSCC for closed-loop control (Direction 1, high potential)

**Core idea.** Instead of fixed k symbols per state, allow the encoder to emit
up to K_max symbols; the receiver ACKs "decoded enough for control" after
cumulative power-norm decoding. Task success probability drives the stopping.

**Math.** During training, sample stop index t_stop ∈ [1, k] and pass the
partial codeword z[:t_stop] through frozen power-norm → channel → zero-pad
decoder, with loss = task rollout + λ_head · (t_stop/k) parking term:

    L_rateless = L_task + λ_head * E[t_stop]/k
    λ_head grid {0.1, 0.2, 0.5} to trace a rate-vs-reliability curve.

**Integration points.**
- `project/models/rateless_jscc.py` (encoder with mask-based variable-length
  emission; decoder with zero-padding). Frozen `power_normalize` reused
  (normalization per ACTIVE prefix, not full codeword).
- `project/training/train_rateless.py` — closed-loop rollout but the channel
  pass samples the prefix length each step (deterministic per seed).
- Matrix: 2 cells (k=12, 24) × channels × 3 seeds × λ_head {0.1, 0.2} ≈ 24 runs
  ≈ 10 h CPU.

**Paper role.** Direct novelty claim vs RLACS (image reconstruction only) —
rateless + closed-loop control not yet published. **Strongest extension
candidate.**

---

## 4. CIV / world-model reward (Direction 2, theory weight)

**Objective.** Replace myopic distance with long-horizon value attribution.

**Simplified, implementable CIV (no full world model):**
For each transmission step t, define the empirical CIV of that state token as
the gradient of ROLLOUT-RETURN w.r.t. the information content of the state
probed by leaving the token OUT vs IN (two rollouts per step, paired seeds):

    CIV_t = J(full) - J(ablated)
    J = Σ α_t ||p_t - p_g||² + λ_T ||p_T - p_g||²   (frozen task cost)

Then train with **CIV-weighted task loss**:

    L_CIV = Σ_t w_t ||p_t - p_g||²,  w_t = softmax_t(CIV_t / τ),  τ grid {1, 5}

**Integration points.** `project/training/train_civ.py` (reuses the rollout;
CIV computed offline per 100 steps and cached per cell); matrix = 2 cells ×
3 seeds × τ {1, 5} ≈ 20 runs ≈ 6 h CPU.

**Paper role.** First DeepJSCC+CIV combination — theoretical hook, strongest
reviewer bait, moderate compute.

---

## 5. Control rate-distortion / theoretical boundary (Direction 4)

**Approach.** Formal claim: the crossover observed (digital overtake ρ_dig'
between k=18–24 AWGN) and the low-ρ task-underperformance ARE consistent with
a control rate-distortion argument: define the control distortion
D_c = E||‖p_T - p_g||² - J_min|| and the minimal semantic rate
R_sem = min I(s; ẑ) subject to D_c ≤ D*.
Use the DeepJSCC decoder outputs already saved (ŝ trajectories) to estimate
I(s; ẑ) via MINE on saved eval dumps per cell; plot the empirical boundary
R_sem(D_c) against the three methods.

**Integration points.** `project/analysis/control_rdt.py` + MINE probe:
no re-training, pure analysis on existing eval data (≈ 2 h CPU total).
**Paper role.** Explains the crossover point information-theoretically; free
on compute, carries the theory claim.

## 6. Reproducibility assets (Docker/Zenodo, one-hour item)

**Already committed:** frozen configs + seeds (git-tracked).
**Add:**
- `environment.yml` / `requirements-lock.txt` (torch 2.4.1+cpu, numpy 1.26.4
  pinned).
- `Dockerfile`: python:3.9-slim, pip install of the lock file,
  `CMD python tools/build_dashboard.py`.
- Zenodo deposit = the repo at tag `paper-v1`; README already covers setup.
**Runtime:** < 1 h; no science depends on it.

---

## Effort summary (highest → lowest priority for the paper)

| # | Item | Effort | Publishable claim |
|---|---|---|---|
| 1 | ATROC full | 36 runs ≈ 11 h | Aligned task+recon framework |
| 2 | Polar baseline | ~2.5 h eval-only | Digital baseline de-strawmanned |
| 3 | Rateless | 24 runs ≈ 10 h | Rateless + closed-loop = novel |
| 4 | CIV | 20 runs ≈ 6 h | DeepJSCC + CIV first combo |
| 5 | Control-RDT | ~2 h analysis | Theory boundary explanation |
| 6 | Docker/Zenodo | < 1 h | Reproducibility |

**Dependency note:** run order 2 (free, eval-only) → 5 (free, analysis) →
Wave-A verdict → 1/3/4 (theWave-B/A抗生素 wait for the Wave-A root cause so
the framework story is diagnosed first).
