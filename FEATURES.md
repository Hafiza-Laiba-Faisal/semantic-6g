# Features

What the implemented system provides, layer by layer. Everything below is
implemented and covered by gate tests in `tests/`.

## 1. UAV environment (shared, frozen)

- 6-D state `[x, y, vx, vy, xg, yg]`, workspace ±8 m, goal radius 0.5 m
- Differentiable 2-D double integrator (batched, Δt = 0.1, T = 100)
- Shared nominal PD controller `u = clip(−Kp(p−p_g) − Kv·v, ±2)` — identical
  for all three methods (no information leaks)
- Success = ∃t ≤ T with distance ≤ goal radius; boundary exit terminates an
  episode as failure
- Fixed-range normalization to [−1, 1] (ranges from config, never from data)

## 2. Communication methods (matched budgets)

- **Task-Oriented DeepJSCC** — trained on the differentiable closed-loop
  navigation loss `λd·L_d + λu·L_u + λv·L_v + λT·L_T` through encoder →
  channel → decoder → controller → dynamics
- **Reconstruction-Oriented DeepJSCC** — identical architecture, trained on
  normalized-state MSE (no navigation loss)
- **Conventional digital baseline** — scalar quantization (B bits/scalar),
  explicit framing/termination padding, convolutional code K = 7 R_c = 1/2,
  soft-decision Viterbi, QPSK with Gray labeling, unit-average-power
  constellation
- Strict channel-use matching: every method occupies exactly k complex
  channel uses per state (ρ = k/6); digital budget 6B + 6 = k exactly or the
  cell is declared INFEASIBLE (M3) — never silently widened

## 3. Channels

- AWGN with the correct complex-noise convention (σ² = P·10^(−SNR/10),
  Var(n_R) = Var(n_I) = σ²/2) — validated against textbook QPSK BER
- Slow Rayleigh fading: one h ~ CN(0,1) per codeword, perfect receiver CSI,
  ZF equalization with a numerical guard — identical CSI assumption for all
  methods

## 4. DeepJSCC model

- MLP encoder 6 → 128 → 128 → 2k real (PReLU) → stacked I/Q → k complex
  symbols (channel uses counted as k, never 2k)
- Per-codeword power normalization `√(kP)·z/√(Σ|z|² + ε)`, P = 1
- Mirrored decoder 2k → 128 → 128 → 6
- Runtime asserts: codeword power = P (within tolerance), n_dig == k

## 5. Training (frozen 9C protocol)

- Reconstruction: 5000 steps × 4096 states; task: 3000 steps × 256 episodes
  × T = 100; Adam 1e-3, grad clip 1.0, early stopping disabled
- Train SNR ~ U(0, 20) dB per batch
- Deterministic by (seed, config): fixed generators for init/episodes/SNR/
  noise; NaN-batch skip guard with counter; structured failure reporting
- Seeds {42, 43, 44}; test seed 10042

## 6. Evaluation & statistics

- **Paired harness**: all methods evaluated on bitwise-identical episodes,
  noise and fading realizations (`evaluate_condition`, 5000 episodes/unit)
- Per-episode metrics: success, final/avg/min distance, control effort,
  time-to-goal, trajectory MSE (post-exit caveat documented)
- Wilson 95% confidence intervals on every success rate
- Paired McNemar test (two-sided χ²(1) survival function — factor-2 defect
  found and fixed, Step-10E) for task-vs-reconstruction per cell
- Paired-t on final/average distance and control effort
- Pooled summaries (3 seeds → 15 000 episodes/condition) with over-seed
  mean/std

## 7. Experiment matrix & execution engine

- Frozen M1–M5 matrix (`project/experiments/config.py`) with validator gates:
  M2/M4 confusion rejection, budget mismatch rejection, seed coverage, SNR
  grids, infeasible-cell display
- **M1**: task vs reconstruction, k ∈ {1,2,3}, 11 SNRs — 36 runs, 198 paired
  units (COMPLETE)
- **M2**: matched-budget 3-way comparison, k ∈ {12,18,24,30,54} — 60 runs,
  150 paired units (120/150 complete; 2 cells excluded + documented in
  `results/M2/PROTOCOL_EXCEPTIONS.md`)
- **M3**: structural infeasibility display at k ∈ {1,2,3} (COMPLETE)
- **M4**: secondary digital reference B = 8 @ k = 54, 10 conditions (COMPLETE)
- **M5**: deferred + disabled by protocol (supervisor-gated)
- Identity-addressed runs: `results/<EXP>/<run_id>__<sha256>/` with
  DONE/RUNNING/FAILED markers — completed runs SKIP, interrupted runs
  retrain deterministically
- Parallel lane executors + self-driving watcher (marker-idempotent, 24 h
  cap, safe restart) that chains training → per-k evaluation → rayleigh lane
  → validate → finalize

## 8. Analysis & reporting tooling

- `tools/analyze_m1.py` — pooled tables, McNemar corrected-p audit
  (198/198 significant), paired-t sign counts, per-seed agreement
- `tools/make_figures.py` — 5 thesis-grade figures (SR-vs-SNR curves,
  3-method panels, bandwidth-crossover, full-bandwidth-axis view)
- `diagnostics/matched_mse_diagnostic.py` — quantifies the train-time
  diagnostic vs trajectory-MSE structural gap (~164×) and the post-exit
  bypass
- `tools/run_step11_m2.py status|validate|finalize` — machine-checkable
  completion state + `step11_m2_summary.json`
- Modal cloud fallback app for heavy lanes (detached entrypoint, volume
  snapshot upload/fetch)

## 9. Reliability

- 12 gate-test modules in `tests/` (channels, digital, deepjscc, training,
  evaluation, experiments, McNemar exact values) — all runnable in seconds
  on CPU
- Every scalar frozen in `project/config.py`; results committed to the
  repository as a recovery backup
