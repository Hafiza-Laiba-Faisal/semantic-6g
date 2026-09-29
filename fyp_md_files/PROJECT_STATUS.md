# FYP Project Status

## Project
TASK-ORIENTED DEEP JOINT SOURCE-CHANNEL CODING FOR UAV NAVIGATION UNDER NOISY AND FADING WIRELESS CHANNELS

Simulation-only, Python + PyTorch. Comparison of Task-Oriented DeepJSCC vs
Reconstruction DeepJSCC vs a conventional digital baseline under AWGN and
slow Rayleigh fading with a frozen UAV navigation task.

## Current Status
Step 5 — PASS (checkpointed)

## Last Completed Step
Step 5 — Task-Oriented Closed-Loop Training

## Next Step
Step 6 — NOT STARTED / awaiting specification
(candidate per coding order: digital baseline with conv K=7 R_c=1/2 +
Gray-QPSK + soft Viterbi, framing per audit section 8)

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

## Last Verification
- Step 0-2: 11/11 unit tests + oracle gate PASS (SR 100%, 0/5000 exits)
- Step 3: 43/43 checks PASS
- Step 4: 21/21 structural + 10/10 smoke checks PASS
- Step 5: 28/28 checks PASS (incl. in-process 43 + 11 regression)

## Last Commit
- step0-2: ddc41b0 (root commit)
- step3:   8517ee7
- step4:   8e4dc95
- step5:   bb82334584949d76719d4a035a80af44c24f7727

## Repository State
Clean after checkpoint: YES

Push state after Step-5 checkpoint: BLOCKED (no push credentials in the
build environment; commits are local on `main`). To publish, run once in
an interactive terminal with GitHub access:
    git push -u origin main
(origin = https://github.com/Hafiza-Laiba-Faisal/semantic-6g.git, which
was verified reachable and empty before the checkpoint; the credential
dialog completes only in an interactive session.)

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
  tests/test_task_oriented.py; oracle gate: tests/sanity_noiseless.py

## Important Rules
- Do not modify frozen mathematical conventions without explicit approval.
- Do not invent missing numerical values.
- Do not start the next step before the current step is committed and pushed.
- Re-run regression tests after changes.
- Do not commit virtual environments, caches, secrets, or unrelated artifacts.
- Git checkpoint policy: after every completed step, run tests, update audit,
  commit, push, verify, record the hash, update this file.

## Resume Procedure
1. Clone the repository.
2. Checkout `main`.
3. Pull the latest commit.
4. Read this file.
5. Read `fyp_md_files/05-implementation-audit.md`.
6. Run the regression suite (all five test modules above).
7. Confirm the working tree is clean.
8. Continue from the stated next step only.
