# Wave-A Design — Diagnostic Ablations (post-freeze amendment, M5-expanded)

**Purpose:** establish the root cause of the task-oriented gap (M1/M2: task SR ≈ 0.03 vs recon SR
≈ 0.26 at k=12, 10 dB AWGN) before the paper claims anything. Three candidate causes are tested:

- **C1. Loss shaping:** frozen λ = (1, 0, 0, 2) ignores control/velocity structure.
- **C2. Objective alignment:** plain MSE treats all 6 state components equally, although the
  navigation task depends on position/goal/velocity with different sensitivities (UEP-lite).
- **C3. Training variance/instability:** MLPs trained on closed-loop rollouts may converge to
  poor local minima with high seed variance (mam ka reviewer "Figure 10 oscillation" objection bhi covered).

## W-A1: λ ablations (task method)

Modify `train_task_oriented` to take `lambdas: Tuple[float, float, float, float]`
(default = frozen config values), thread through to `task_loss` (already accepts λ parameters),
and record `"lambdas"` in the report (already does, but must read the passed-in values, not config).

**Grid (per channel × k × seed):**

| Tag | λ_d | λ_u | λ_v | λ_T | Rationale |
|---|---|---|---|---|---|
| `lam_frozen` | 1.0 | 0.0 | 0.0 | 2.0 | control arm (re-skip: identical configs skip via run identity) |
| `lam_ctl_light` | 1.0 | 0.01 | 0.01 | 2.0 | light effort/velocity shaping |
| `lam_ctl_heavy` | 1.0 | 0.1 | 0.1 | 2.0 | strong shaping |

Recommended scope (CPU budget): channels {awgn, rayleigh} × k {1, 12} × seeds {42,43,44} →
12 new training runs × λ-tags (2 new tags) = 24 runs. Training budget: same frozen
3000 steps × 256 episodes × T=100 as 9C so the comparison stays matched.

**Deliverable:** SR-parity check. If no λ tag lifts task SR materially (> 2×), loss shaping is
**not** the root cause and the paper gains a strong, citable negative ablation.

## W-A2: task-aware weighted reconstruction (new method `recon_weighted`)

Add optional per-component weights to `reconstruction_loss` (default = frozen uniform):

```
L_wrecon = Σ_j w_j (ŝ̄_j − s̄_j)² / Σ_j w_j ,   w = (2, 2, 1, 1, 2, 2)  [positions/goal ×2]
```

`train_reconstruction` gets `state_weights: Optional[Tuple[float,...]] = None`
(None → frozen uniform baseline). `recon_weighted` is a distinct method tag so run identity
(`config_hash`) differs; the frozen `reconstruction` arm stays untouched and re-used.

**Rationale:** tests C2/UEP-lite cheaply — same architecture, same budget, only the loss weights
change. If weighted-recon trained models match/beat plain recon whiletask SR stays similar,
and possibly helps task SR if the controller's goal-relative geometry is better estimated —
this is also the cheap "ATROC-lite" positive contribution for the paper.

**Scope:** channels × k {1, 12} × seeds — 12 runs.

## W-A3: training variance / instability diagnostic

For the two worst cells reported by the current M1/M2 figures (task k=12 AWGN and task k=12
rayleigh are candidates; exact cells chosen from the existing pooled tables):

- 3 extra training seeds {45, 46, 47} (same 3000-step protocol) → seed-variance of final
  eval SR (report mean ± range). If cross-seed spread is large, **training instability is a
  first-class finding**, per the supervisor review.
- 6 runs (2 cells × 3 seeds).

**Deliverable:** instability verdict + loss-trajectory overlay figures (already-scriptable from
`train_report.json` histories; the dashboard plots these automatically).

## Wave-A totals

- 42 training runs ≈ task: 30 × ~25 min (3000 steps, ~0.37 s/step) + recon: 12 × ~1.5 min ≈ **~13 h CPU**,
  each run individually resumable/marker-idempotent with the existing `results/<EXP>/` identity
  infrastructure (run_id includes the wave tag + λ/method tag so no frozen run collides).
- Evaluation: 30 units × reduced E=1000 episodes × SNR grid {0,5,10,15,20} ≈ ~2.5 h CPU.
  **Protocol note (must be respected):** reduced-episode Wave-A evals are *comparative
  diagnostics only*; no M1/M2 number is re-derived from them, and every table captions the
  E=1000 protocol. Wilson CIs at E=1000 still allow paired significance checks between
  ablation arms and their matched frozen arms on the same 3 seeds.

## Guardrails (unchanged frozen rules)

- No changes to `LAMBDA_*` defaults, `TRAIN_STEPS`, batch sizes, seeds, or SNR conventions in
  `project/config.py` — ablation values are **passed as parameters**, never written to config.
- Frozen M1–M4 artifacts: re-running any identical (experiment, method, channel, k, seed) identity
  returns SKIP (done-marker), so the amendment cannot silently overwrite the primary results.
- Wave-A dirs live beside the existing ones under `results/WAVE_A/` with `non_frozen: true`
  recorded in every report JSON.

## Implementation notes (for the other-PC run)

1. `extend` `train_task_oriented(..., lambdas=...)` and `train_reconstruction(..., state_weights=...)`
   — both are additive, default-identical changes (existing calls compile unchanged; a regression
   gate asserts bitwise-identical first steps on default args).
2. Add a `tools/run_wave_a.py` executor mirroring `tools/run_step11_m2.py` (identity/marker/DONE
   semantics reused).
3. Dashboard (`tools/dashboard/build_dashboard.py`) picks up `results/WAVE_A/eval/**` automatically
   — no dashboard changes needed except adding WAVE_A to the experiment list.
