# FYP Project Status

## Project
TASK-ORIENTED DEEP JOINT SOURCE-CHANNEL CODING FOR UAV NAVIGATION UNDER NOISY AND FADING WIRELESS CHANNELS

Simulation-only, Python + PyTorch. Comparison of Task-Oriented DeepJSCC vs
Reconstruction DeepJSCC vs a conventional digital baseline under AWGN and
slow Rayleigh fading with a frozen UAV navigation task.

## Current Status
Step 9A — PASS (preflight only; checkpointed)

## Last Completed Step
Step 9A — Full Experiment Preflight & Compute Plan
(frozen-matrix verification 10/10, exact workload: 96 neural training runs
+ 292 eval cells + 3 infeasible + 10 secondary; environment verified
CPU-only; tiny measured benchmark; run-identity/recovery design
(project/experiments/identity.py); preflight PASS with 0 blockers;
NO training, NO execution, NO scientific results)

## Next Step
BLOCKED PENDING EXPLICIT USER APPROVAL.
Next permitted action: FULL M1–M5 experiment execution — requires the
user's explicit approval AND two open decisions:
1. full-scale training budget (steps/epochs) — NOT frozen anywhere;
   config's 70/15/15 split + N_TRAIN_EPISODES=10,000 are unused
   placeholders (trainers are step-based)
2. M5 ablation set — NOT EXECUTABLE YET (specification incomplete)
Also pending: user PAT push of the local commits (see Last Commit).
Execution order when approved: Stage A pilot → M1 → M2 → M3 (display) →
M4 (secondary) → M5 only when specified.

## Frozen Decisions
- UAV state: 6D [x, y, vx, vy, x_g, y_g]
- Dynamics: 2D double integrator, dt = 0.1 s, T_max = 100
- Workspace [-8, 8]^2, d_min = 4 m, goal radius r_g = 0.5 m, v_max = 5 (norm-rescale), u_max = 2 (per-axis clip)
- Kp = 1.0
- Kv = 2.0 (damping term -Kv*v; the clip is NOT a damping source)
- Normalization: frozen affine map to [-1, 1], config-only ranges
- DeepJSCC: 6 -> 128 -> 128 -> 2k (PReLU), decoder mirror 2k -> 128 -> 128 -> 6
- k is the number of complex channel uses (never 2k); rho = k/6
- Power normalization: Step-3 implementation (per codeword, P = 1)
- AWGN: Step-3 implementation (sigma^2 = P * 10^(-SNR_dB/10), sigma^2/2 per I/Q component)
- Rayleigh: Step-3 implementation, block fading h ~ CN(0,1) per codeword + frozen perfect-CSI ZF equalization
- Training SNR ~ U(0, 20) dB; bandwidth grid rho in {1/6, 1/3, 1/2} -> k in {1, 2, 3}
- Digital baseline: conv K=7, R_c=1/2, (171,133)_8, soft Viterbi, Gray QPSK;
  packet invariant 6B + 6 <= k (audit section 8); INFEASIBLE at k in {1,2,3}
- QPSK soft demapper: EXACT log-MAP, LLR = log[P(0)/P(1)]
  (LLR_I = 4*sqrt(2)*Re(y_eq)/sigma2_eff, LLR_Q = 4*sqrt(2)*Im(y_eq)/sigma2_eff)
- Rayleigh LLR: per-symbol perfect-CSI scaling, sigma2_eff = sigma2/max(|h|^2, eps_eq)
- Uncoded QPSK: validation/reference path only (formal ablation = later M5)
- Evaluation: paired pre-generated episodes + noise [E,T,k] + h [E,T]
  bitwise-identical across methods; one fixed SNR per condition;
  success-conditional time-to-goal (failures keep the T+1 sentinel);
  checkpoints regenerated-by-seed, never committed
- Task loss: L_task = ld*L_d + lu*L_u + lv*L_v + lT*L_T with
  - lambda_d = 1.0
  - lambda_u = 0.0
  - lambda_v = 0.0
  - lambda_T = 2.0
  (project design choice, NOT a literature-derived value; L_u/L_v are
  diagnostics only; L_d/L_v sum over post-transition states t = 1..T)
- Seeds: training {42, 43, 44}, test seed 10042, purpose-separated RNG streams

## Completed Steps
- Step 0-2: PASS (config, normalization, dynamics, controller, environment, oracle gate)
- Step 3: PASS (channel layer + 43-check verification suite)
- Step 4: PASS (DeepJSCC encoder/decoder + reconstruction pipeline + smoke training)
- Step 5: PASS (task-oriented closed-loop training with frozen task loss)
- Step 6: PASS (digital baseline: quantizer, conv coding, Gray-QPSK, Viterbi, framing)
- Step 7: PASS (evaluation harness: metrics, statistics, paired evaluation)
- Step 8: PASS (training/common.py + experiments/{config,matrix,runner}.py
  + main.py: checkpoint infra, dry-run, smoke runner, matrix validation;
  full M1–M5 execution NOT performed)
- Step 9A: PASS (preflight only: matrix verification, exact workload
  counts, environment + benchmark, identity/recovery design, per-cell
  validation; execution NOT started; M5 flagged NOT EXECUTABLE YET;
  full-scale training budget flagged as an open user decision)

## Last Verification
- Step 0-2: 11/11 unit tests + oracle gate PASS (SR 100%, 0/5000 exits)
- Step 3: 43/43 checks PASS
- Step 4: 21/21 structural + 10/10 smoke checks PASS
- Step 5: 28/28 checks PASS (incl. in-process 43 + 11 regression)
- Step 6: 33/33 checks PASS (Gates A-E incl. BER-vs-theory on 8e6 bits/point)
- Step 7: 18/18 checks PASS (Gates A-H; full regression 11 + 43 + 21 + 10 +
  28 + 33 re-verified)
- Step 8: 12/12 training-common checks (Gates A-B) + 63/63 experiments
  checks (Gates C-L); full regression 11 + 43 + 21 + 10 + 28 + 33 + 18 +
  oracle gate re-verified PASS. Determinism gate: identical reruns ->
  bitwise-equal loss histories, bitwise-equal model parameters, equal
  eval rows, bitwise-identical evaluation realizations (exact, no
  tolerance loosened). Dry-run PASS (zero side effects); smoke runs PASS
  (recon AWGN 83% loss reduction in 30 steps, task AWGN/Rayleigh, digital
  k=54, checkpoints round-trip).
- Step 9A: preflight PASS (0 blockers; results/preflight/step9a_preflight.json);
  full regression re-verified 11 + 43 + 21 + 10 + 28 + 33 + 18 + 12 + 63
  + oracle gate. Benchmark (non-scientific, measured): recon ≈ 4.6 µs/state
  (k=3) to 7.0 µs/state (k=54); task ≈ 36.5 µs/episode-step (k=3) to
  45.7 µs/episode-step (k=54); digital eval ≈ 156 µs/episode-step.
  Environment: Python 3.9.7, torch 2.4.1+CPU, NO GPU, 12 cores,
  15.83 GB RAM, 219.8 GB free disk.

## Last Commit
- step0-2: ddc41b0 (root commit)
- step3:   8517ee7
- step4:   8e4dc95
- step5:   bb82334584949d76719d4a035a80af44c24f7727
- step6:   f58cb77b23555635b331761b52987f062130161a
- step7:   787ca849b9bcf76ecc8091fcbb63820dadb847eb
- step8:   c3dd8fb7a787ecaac4ed48dac19bfd0d95e09feb (+ bookkeeping commit
  7d2306e; PUSH PENDING — user PAT required, 403 from this machine)

## Repository State
Clean after checkpoint: YES

Push state after Step-5 checkpoint: SUCCESS (pushed by the user via a
fine-grained PAT; verified with `git ls-remote origin` — origin/main =
5463aa9 = local HEAD). Push procedure for future steps: the user pushes
from their own terminal with a one-shot token URL (not stored on the
machine):
    git push https://<TOKEN>@github.com/Hafiza-Laiba-Faisal/semantic-6g.git main
origin = https://github.com/Hafiza-Laiba-Faisal/semantic-6g.git

## Environment (machine-independent setup)
- Python 3.9+ (project-local venv; no global installs)
- torch 2.4.1+cpu, numpy 1.26.4, matplotlib
- Setup: create a venv, `pip install torch numpy matplotlib`
  (torch 2.4.1+cpu wheel was fetched from download.pytorch.org/whl/cpu)
- Run tests from the repository root with the venv python, e.g.:
  `.venv/Scripts/python.exe -m tests.test_step02` (Windows) or
  `.venv/bin/python -m tests.test_step02` (POSIX)
- Test modules: tests/test_step02.py, tests/test_channels.py,
  tests/test_deepjscc.py, tests/test_reconstruction_smoke.py,
  tests/test_task_oriented.py, tests/test_digital.py,
  tests/test_evaluation.py, tests/test_training_common.py (Step 8),
  tests/test_experiments.py (Step 8); oracle gate: tests/sanity_noiseless.py
- Step-8 CLI: python main.py config | dry-run | validate-matrix |
  smoke <experiment> <method> <channel> <k> <seed> ; main.py full -> REFUSED

## Important Rules
- Do not modify frozen mathematical conventions without explicit approval.
- Do not invent missing numerical values.
- Do not start the next step before the current step is committed and pushed.
- Re-run regression tests after changes.
- Do not commit virtual environments, caches, secrets, or unrelated artifacts.
- Git checkpoint policy: after every completed step, run tests, update audit,
  commit, push, verify, record the hash, update this file.
- Incremental commits (user policy): commit every change-batch immediately
  (per module / per gate / per bug fix) with a descriptive message - do not
  batch unrelated work into one large commit; push at each completed step.

## Resume Procedure
1. Clone the repository.
2. Checkout `main`.
3. Pull the latest commit.
4. Read this file.
5. Read `fyp_md_files/05-implementation-audit.md`.
6. Run the regression suite (all five test modules above).
7. Confirm the working tree is clean.
8. Continue from the stated next step only.
