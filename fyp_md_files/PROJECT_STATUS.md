# FYP Project Status

## Project
TASK-ORIENTED DEEP JOINT SOURCE-CHANNEL CODING FOR UAV NAVIGATION UNDER NOISY AND FADING WIRELESS CHANNELS

Simulation-only, Python + PyTorch. Comparison of Task-Oriented DeepJSCC vs
Reconstruction DeepJSCC vs a conventional digital baseline under AWGN and
slow Rayleigh fading with a frozen UAV navigation task.

## Current Status
Step 9D — PASS (Stage-A production pilots complete; CLEARED FOR FULL
EXECUTION; M1–M4 NOT started; awaiting user approval for Step 10)

## Last Completed Step
Step 9C — Final Training Protocol FROZEN (commit d065acd)

## Next Step
**Step 10 — Full M1 Execution** — ONLY after explicit user approval.
The frozen protocol (Step 9C) and the Stage-A pilot (Step 9D) prove the
pipeline executes safely at production scale. M5 stays DEFERRED. Also
pending: user PAT push of the local commits (see Last Commit).
Execution order when approved: M1 → M2 → M3 (display) → M4 (secondary) →
M5 only when specified.

## Step 9D — Stage-A Production Pilot (2026-09-30)

Objective: prove the FROZEN Step-9C protocol executes correctly and
safely at PRODUCTION scale before launching the full matrix. Exactly two
production-scale runs (AWGN, k=3, seed 42), full frozen budgets, no
substitutions. **NON-SCIENTIFIC — no performance conclusions drawn; the
numbers below are execution/readiness evidence only.**

Pre-pilot verification (tools/pilot_step9d.py, new tool):
- `verify`: 13/13 PASS — every frozen 9C value asserted programmatically
  (5000/3000/T=100/4096/256/Adam/1e-3/clip 1.0/U(0,20)/seeds {42,43,44}/
  test 10042/early-stop off/M5 deferred+disabled); any mismatch would
  STOP the pilot.
- `identitytest`: ALL PASS — run identity distinguishes method, channel,
  k, seed and config hash; completed runs SKIP; incompatible configs
  create NEW identities; output paths deterministic; marker semantics
  correct (DONE > FAILED > RUNNING); results/ and *.pth not tracked by
  Git; no silent overwrite across seeds/configs.
- `resumetest`: ALL PASS — model + Adam moments restore bitwise; with
  data-generator state checkpointed, continuation is bitwise-identical.
  Honest limitation (documented, not hidden): the Step-4/5 trainers own
  their generators internally, so mid-run continuation of those loops is
  finite but NOT bitwise; per the FROZEN 9C policy, recovery = RETRAIN
  (deterministic by seed+config), not mid-run resume.

### Pilot A — Reconstruction (production scale)
- Config: method=reconstruction, channel=AWGN, k=3, seed=42,
  steps=5000 (RECON_TRAIN_STEPS), batch_states=4096, Adam lr 1e-3,
  grad-clip 1.0, SNR ~ U(0,20) dB per batch — frozen values, no changes.
- Runtime: 90.1 s wall (18.0 ms/step) on CPU (torch 2.4.1+cpu).
- Loss: start → final as recorded in pilotA_report.json; 0 NaN/Inf;
  gradient-finite checks PASS throughout.
- Identity: run_id STEP_9D_PILOT_reconstruction_awgn_k3_seed42,
  config_hash fb564626b2e8603f.
- Checkpoint: results/STEP_9D_PILOT/STEP_9D_PILOT/
  STEP_9D_PILOT_reconstruction_awgn_k3_seed42__fb564626b2e8603f/
  final_ckpt.pt (gitignored); exit status: complete.
- Integrity (`checkint A`): 11/11 PASS (exists, readable, stored config,
  seed, method/channel/k, final step == 5000, params finite, optimizer
  state finite, identity match, config-hash match).
- Skip test: identical re-invocation → "SKIP: completed pilot found";
  no retraining, no overwrite.
- Resources: no OOM, no runaway RAM, normal disk growth (~0.4 MB ckpt).

### Pilot B — Task-oriented (production scale)
- Config: method=task, channel=AWGN, k=3, seed=42, steps=3000
  (TASK_TRAIN_STEPS), batch_episodes=256, T=100 (TASK_TRAINING_T),
  lambdas {ld=1.0, lu=0.0, lv=0.0, lT=2.0} — frozen values, no changes.
- Runtime: attempt 2 = 1106.0 s wall (0.3687 s/step). Attempt 1 (1088 s)
  had completed all 3000 steps but was killed by a Freebuff desktop
  restart seconds before marker writing — ENVIRONMENTAL finding (desktop
  restarts kill detached background processes), not a code failure.
- Task loss: 3275.1 → 912.0 (fluctuates with per-batch U(0,20) SNR by
  frozen protocol). Diagnostics (NOT performance claims):
  recon-MSE-diag 0.253 → 36.08 (the expected task-vs-fidelity trade-off,
  seen already at smoke scale in Step 5); eval SR 0.0195 → 0.0391.
- Numerical health: 0 NaN/Inf; gradient-finite checks PASS (pre-clip
  norms large late in training; clip 1.0 bounds them, as anticipated in
  the audit).
- Identity: run_id STEP_9D_PILOT_task_awgn_k3_seed42, config_hash
  bd6f10e92d2aced3.
- Checkpoint: results/STEP_9D_PILOT/STEP_9D_PILOT/
  STEP_9D_PILOT_task_awgn_k3_seed42__bd6f10e92d2aced3/final_ckpt.pt
  (gitignored); exit status: complete.
- Integrity (`checkint B`): 11/11 PASS.
- Skip test: PASS (skip, no retraining, no overwrite).
- Resume/determinism evidence: attempt 1 and attempt 2 — independent
  processes, same seed+config — produced BITWISE-IDENTICAL final task
  loss (911.971923828125), directly demonstrating the frozen 9C
  retrain-on-interruption policy is deterministic across processes.
- Resources: no OOM, no runaway RAM, normal disk growth (~0.4 MB ckpt).

### Runtime update (measured vs Step-9B estimate)
| Run | 9B estimate | Measured | Factor |
|---|---|---|---|
| Recon, batch 4096 | 20–29 ms/step | 18.0 ms/step | ~1.1–1.6× faster |
| Task, batch 256, T=100 | 1.02–1.17 s/step | 0.3687 s/step | ~3× faster |

The task deviation is material and explained: the 9B extrapolation
assumed linear scaling from a batch-64 smoke measurement to batch 256;
the real scaling is sub-linear. No frozen value was changed because of
runtime.

Updated full-matrix estimate from MEASURED rates (single process,
CPU-only, sequential): 48 recon runs ≈ 1.2 h; 48 task runs ≈ 14.7 h;
96 neural training runs ≈ **16 h total** (vs the 9B estimate of ~46 h),
plus eval-only digital cells and 292 evaluation cells (small).

### Contamination control
All pilot artifacts live under results/STEP_9D_PILOT/ with STEP_9D_PILOT
labels in run_id, directory names and both report JSONs
("STEP_9D_PILOT": true, "non_scientific": true); results/ and *.pth are
gitignored. No pilot artifact can enter final M1–M4 tables or paper
claims. Post-pilot regression: 11 + 43 + 21 + 10 + 28 + 33 + 18 + 12 +
63 + oracle gate ALL PASS.

### Decision
Both production-scale pilots completed successfully, stayed finite,
produced valid integrity-verified checkpoints, passed skip tests, and
left the regression suite green. **STEP 9D PASS — CLEARED FOR FULL
EXECUTION. M1–M4 NOT STARTED.** Next: Step 10 — Full M1 Execution,
only on explicit user approval.

## Step 9C — Final Training Protocol (FROZEN, 2026-09-30)

The Step-9B proposals (below) were USER-APPROVED and are now FROZEN in
`project/config.py` + audit §2.6.1. Classification: PROJECT DESIGN
DECISIONS, NOT literature-derived (papers explicitly unspecified; the
"~20k + early stop" audit line was [PROJ] TBD and is superseded).

- RECON_TRAIN_STEPS = 5000; TASK_TRAIN_STEPS = 3000 (fixed-step; M1+M2,
  every channel, every seed)
- TASK_TRAINING_T = 100 (== frozen eval horizon; trainer default)
- BATCH_STATES = 4096 / BATCH_EPISODES = 256 (audit §2.6; smoke values
  remain smoke-only)
- Adam, lr 1e-3, grad-clip 1.0, SNR ~ U(0,20) dB per batch, seeds
  {42,43,44} (unchanged)
- EARLY_STOPPING = False — the audit's val protocol stays unimplemented;
  fixed-step budgets are the primary protocol (deterministic,
  reproducible)
- Checkpoints: final checkpoint per run via the Step-8
  identity/marker infrastructure; interrupted runs retrained
  (deterministic); no periodic-checkpoint system built
- M5_STATUS = DEFERRED, M5_EXECUTION_ENABLED = False — specification
  incomplete (task-loss-component removals, uncoded-digital reference,
  cross-channel definition). M1 stays the PRIMARY task-vs-reconstruction
  comparison; it is not an M5 substitute.
- Full M1–M5 execution: NOT STARTED. Next permitted action:
  **Step 9D — Final Execution Readiness / User Approval Gate** (NOT
  execution itself).

## Step 9B Planning — PROPOSED / PENDING APPROVAL (NOT frozen)

Everything in this section is a PROPOSAL from Step 9B (source review +
smoke-scale convergence probe, seed 42, non-scientific). Nothing here is
frozen until the user approves it. Audit §2.6 stays authoritative: the
"~20k steps + early stop on val" line is [PROJ] TBD, NOT a frozen value.

Evidence (measured, smoke-scale): recon k=3 plateaus by ~1000 steps
(0.243 -> 0.019 AWGN / 0.047 Rayleigh, late-vs-mid within ±3%); task
k=3 T=100 is still improving at 150 steps (Rayleigh late-vs-mid −19%;
eval SR 0.03 -> 0.88 AWGN / 0.00 -> 0.47 Rayleigh); 0 NaN/Inf anywhere.
Measured full-scale-batch costs on this machine: recon ≈ 20–29 ms/step
(batch 4096); task ≈ 1.02–1.17 s/step (batch 256, T=100).

Candidate budgets (M1+M2 = 48 recon runs + 48 task runs, single-process):
- A conservative: recon 2,000 / task 1,000 steps → ≈ 15 h total;
  risk: task (esp. Rayleigh) possibly under-trained
- B balanced:   recon 5,000 / task 3,000 steps → ≈ 46 h (~2 days);
  risk: low-moderate
- C extended:   recon 10,000 / task 6,000 steps → ≈ 91 h (~3.8 days);
  risk: lowest; audit's 20k-everywhere ≈ 12 days (impractical on CPU)

Proposed (pending approval): **Candidate B**; task training T = 100
(= frozen eval horizon; trainer default); audit §2.6 batch sizes 4096
states / 256 episodes; fixed-step budgets WITHOUT early stopping (early
stop remains unimplemented and is not silently introduced); checkpoints:
final + one rolling checkpoint every 1,000 steps (overwritten, ≤3 files
per run, ≈ 1 MB each) with the existing full metadata + config-hash
identity; M5 stays EXCLUDED from primary execution (Outcome B — the three
additional M5 items: task-loss-component removals, uncoded-digital cells,
cross-channel definition — are all under-specified in the sources).

Differently-budgeted recon vs task is justified methodologically:
recon plateaus ~10× sooner in steps AND costs ~50× less per step; both
are trained to their own convergence evidence, which is equal
methodological treatment (not equal step counts).

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
- Step 9D: verify 13/13; identitytest ALL PASS; resumetest ALL PASS (with
  the documented trainer-generator limitation); pilots A+B complete at
  production scale (0 NaN/Inf, grads finite); integrity 11/11 both;
  skip tests PASS; post-pilot regression re-verified 11 + 43 + 21 + 10 +
  28 + 33 + 18 + 12 + 63 + oracle gate ALL PASS.

## Last Commit
- step0-2: ddc41b0 (root commit)
- step3:   8517ee7
- step4:   8e4dc95
- step5:   bb82334584949d76719d4a035a80af44c24f7727
- step6:   f58cb77b23555635b331761b52987f062130161a
- step7:   787ca849b9bcf76ecc8091fcbb63820dadb847eb
- step8:   c3dd8fb7a787ecaac4ed48dac19bfd0d95e09feb (+ bookkeeping commit
  7d2306e; PUSH PENDING — user PAT required, 403 from this machine)
- step9a:  4dc4859 | step9b: cb7bdee | step9c freeze: d065acd
  (PUSH PENDING — user PAT required, 403 from this machine)
- step9d:  <see git log after the Step-9D commit; PUSH PENDING>

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
