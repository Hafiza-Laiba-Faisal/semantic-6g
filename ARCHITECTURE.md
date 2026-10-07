# Architecture

Single-repo simulation study comparing three communication approaches for a
UAV goal-reaching task under AWGN and slow Rayleigh fading, with everything
shared except the encode/decode map and the training objective.

## 1. Layered data flow (one step, all methods)

```
s_t ∈ R^6  [x, y, vx, vy, xg, yg]
   │  project/data/normalization.py      fixed-range affine → [-1, 1]
   ▼
s̄_t ∈ [-1,1]^6
   │  ───────────── method-specific segment ─────────────
   │  A. DeepJSCC (task & recon): project/models/deepjscc_encoder.py
   │        6 → 128 → 128 → 2k real (PReLU) → stacked I/Q → k complex symbols
   │        project/channels/complex_utils.py: power normalize √(kP)·z/‖z‖, P = 1
   │  B. Digital: project/baselines/digital.py
   │        scalar quantize B bits/scalar → frame/pad → conv K=7 R_c=1/2
   │        → QPSK (Gray) — exactly k channel uses (6B + 6 = k) or INFEASIBLE
   │  ─────────────────────────────────────────────────────
   ▼
channel (project/channels/)
   • awgn.py:        y = z + n,  σ² = P·10^(−SNR/10), Var(n_R) = Var(n_I) = σ²/2
   • rayleigh.py:    y = h·z + n, h ~ CN(0,1) per codeword, perfect CSI, ZF equalize
   ▼
decoder → ŝ_t ∈ R^6 (DeepJSCC)  /  dequantized ŝ_t (digital)
   │  project/data/normalization.py  denormalize
   ▼
controller (project/uav/controller.py, shared by ALL methods)
   u = clip(−Kp·(p − p_g) − Kv·v, ±u_max),  Kp = Kv = 1, u_max = 2
   ▼
dynamics (project/uav/dynamics.py, differentiable double integrator)
   p_{t+1} = p_t + Δt·v_t + ½Δt²·u_t ;  v_{t+1} = v_t + Δt·u_t ;  Δt = 0.1
   ▼
metrics (project/evaluation/metrics.py)
   success = ∃t ≤ T with ‖p_t − p_g‖ ≤ 0.5 m; final/avg/min distance;
   control effort; time-to-goal; trajectory MSE (see §5 caveat)
```

The task-oriented trainer (`project/training/train_task_oriented.py`)
backpropagates through the whole loop:

```
L_task = λd·L_d + λu·L_u + λv·L_v + λT·L_T
L_d = (1/T)Σ‖p_t − p_g‖²,  L_u = (1/T)Σ‖u_t‖²,  L_v = (1/T)Σ‖v_t‖²,
L_T = ‖p_T − p_g‖²            (project-specific loss, λ = (1, 0, 0, 2))
```

The reconstruction trainer (`project/training/train_reconstruction.py`)
optimizes `L_recon = (1/6)·‖ŝ̄ − s̄‖²` on noiseless-PD rollout states.

## 2. Module map

| Module | Responsibility |
|---|---|
| `project/config.py` | every frozen scalar (dt, T, ranges, gains, λ, seeds, SNR grids, B budgets) — no magic numbers elsewhere |
| `project/data/normalization.py` | fixed-range affine normalize/denormalize (ranges from config, never data-derived) |
| `project/channels/awgn.py` | AWGN channel with the exact σ² convention |
| `project/channels/rayleigh.py` | slow (per-codeword) Rayleigh + ZF equalization with ε_h guard |
| `project/channels/complex_utils.py` | I/Q stacking ↔ k complex symbols, per-codeword power normalization, runtime asserts (n_dig == k) |
| `project/models/deepjscc_encoder.py` | 6 → h → h → 2k MLP, PReLU; k complex symbols out (ρ = k/6, never 2k uses) |
| `project/models/deepjscc_decoder.py` | 2k → h → h → 6 MLP, PReLU |
| `project/models/reconstruction_jscc.py` | `ReconstructionDeepJSCC` = encoder + decoder container (both methods share the exact architecture) |
| `project/baselines/digital.py` | scalar quantizer, documented framing/padding, conv K=7 R_c=1/2, soft Viterbi, QPSK Gray; budget-exact or infeasible |
| `project/uav/dynamics.py` | differentiable double integrator (batched) |
| `project/uav/controller.py` | shared nominal PD controller + acceleration clip |
| `project/uav/environment.py` | `NavigationEnv`: workspace ±8 m, goal radius 0.5 m, start sampling, boundary-exit semantics |
| `project/training/train_task_oriented.py` | closed-loop task training (NaN-skip guard, grad clip, deterministic episode/SNR/noise generators) |
| `project/training/train_reconstruction.py` | reconstruction training (same optimizer protocol) |
| `project/training/common.py` | shared trainer utilities (seeding, history) |
| `project/evaluation/evaluate.py` | `Condition`, `generate_realizations`, `evaluate_condition` — paired harness: all methods see bitwise-identical episodes/noise/fading |
| `project/evaluation/metrics.py` | per-episode metrics + aggregates |
| `project/evaluation/statistics.py` | Wilson 95% CI, McNemar (corrected two-sided χ²(1)), paired-t |
| `project/experiments/config.py` | frozen M1–M5 matrix constants (k values, SNR grids, B budgets, seeds) |
| `project/experiments/identity.py` | `run_id` + full-config sha256 → run dir; `DONE/RUNNING/FAILED` markers; `cell_state` (done > failed > running) |
| `project/experiments/matrix.py` | matrix validator (M2/M4 confusion, budget mismatches, seed coverage, SNR grids) |
| `project/experiments/runner.py` | experiment runner plumbing |

## 3. Frozen 9C training/eval protocol

| Item | Value |
|---|---|
| Reconstruction training | 5000 steps × batch 4096 states |
| Task training | 3000 steps × 256 episodes × T = 100 |
| Optimizer | Adam, lr 1e-3, grad clip 1.0, no early stopping |
| Train SNR | ~ U(0, 20) dB, per batch |
| Training seeds | {42, 43, 44} |
| Test protocol | seed 10042, 5000 paired episodes per unit |
| M1 grid | k ∈ {1,2,3}, SNR {0,2,…,20} dB |
| M2 grid | k ∈ {12,18,24,30,54}, SNR {0,5,10,15,20} dB, digital B = (k−6)/6 |
| M4 | secondary digital reference B = 8 at k = 54 (always labeled SECONDARY) |
| M5 | deferred + disabled (needs supervisor-approved definition) |

## 4. Identity / recovery / pipeline

- Run directory = `<run_id>__<sha256(config)>/` under `results/<EXP>/`;
  `cell_state` precedence: **done > failed > running**.
- Executors are idempotent: complete → SKIP; stale RUNNING (crash) or FAILED
  → deterministic retrain (frozen 9C policy: optimizer state intentionally
  not saved; recovery is by re-seeding).
- `tools/m2_pipeline_watcher.py`: marker-idempotent state machine that
  launches training lanes / per-k evaluations / rayleigh lane / validate +
  finalize in order; safe to restart at any time (24 h cap).
- Full artifact trail is committed to the repository as an off-machine
  recovery backup (see SETUP.md).

## 5. Known metric caveat (documented, by design)

`mse_mean` censors exited episodes (post-exit ŝ = s ⇒ zero MSE suffix), so
MSE is a diagnostic only; success rate (paired McNemar) is the primary
metric. The train-time `recon_mse_diag` (single-step, random SNR) vs eval
trajectory MSE gap is ≈ 164× — structural, quantified by
`diagnostics/matched_mse_diagnostic.py` (Step-10E).
