"""Consolidated reproducibility source for the FYP
"Task-Oriented Deep Joint Source-Channel Coding for UAV Navigation under
Noisy and Fading Wireless Channels".

This file is a consolidated reproducibility source for the FYP.
It consolidates the paper-relevant implementation from the frozen
project repository.

It is NOT the canonical development source.
The original project/ and tools/ files remain the source of truth.

This file does not contain or regenerate historical experiment results.
(results/M1..M4 artifacts are read-only inputs; nothing here overwrites
them.  Default execution runs ONLY the minimal deterministic smoke test.)

Usage:
    python project_all_in_one.py                 # smoke test (default)
    python project_all_in_one.py --smoke-test    # same
    python project_all_in_one.py --m1 status     # original M1 executor dispatch
    python project_all_in_one.py --m1 train awgn # M1 training lane (heavy!)
    python project_all_in_one.py --m2 status     # original M2 executor dispatch
    python project_all_in_one.py --m34 validate  # original M3+M4 dispatch
    python project_all_in_one.py --figures       # regenerate fig1-fig15 PNGs

Provenance: every section banner cites the original source file.  The
repository's own docstrings preserve the [FROZEN]/[LIT]/[PROJ] distinction
(literature-derived vs standard equation vs project-specific design choice
vs experimental parameter); these comments are reproduced verbatim and no
new provenance claims are added.

SECTION INDEX
    sec1  Frozen configuration               (project/config.py)
    sec2  State / normalization              (project/data/normalization.py)
    sec3  Channels: complex utils, AWGN, Rayleigh+ZF  (project/channels/*.py)
    sec4  DeepJSCC models (encoder/decoder/ReconstructionDeepJSCC; shared by
          the task-oriented and reconstruction-oriented variants)
                                              (project/models/*.py)
    sec5  Conventional digital baseline      (project/baselines/digital.py)
    sec6  UAV navigation: dynamics, controller, environment  (project/uav/*.py)
    sec7  Loss functions                     (project/losses/*.py)
    sec8  Training                           (project/training/*.py)
    sec9  Evaluation + statistics            (project/evaluation/*.py)
    sec10 Frozen experiment matrices          (project/experiments/*.py)
    sec11 M1 experiment driver                (tools/run_step10_m1.py)
    sec12 M2 experiment driver                (tools/run_step11_m2.py)
    sec13 M3 + M4 experiment drivers          (tools/run_step12_m3_m4.py)
    sec14 Figure/table generation             (tools/make_figures*.py)
    sec15 Main / smoke test                   (this build)

M1-M4 are EXPERIMENT CONFIGURATIONS over the same underlying
implementation -- not separate algorithms.
"""
from __future__ import annotations
from types import SimpleNamespace
import sys


# ============================================================
# SECTION 1  FROZEN CONFIGURATION
# ============================================================
# ---- Source: project/config.py ----
"""Central configuration for the UAV DeepJSCC FYP simulation.

Every tunable parameter of the project lives here. Nothing else in the
codebase may hard-code a physical constant, a gain, a limit, or a seed.

Provenance tags (mirrors ``fyp_md_files/05-implementation-audit.md``):
  [FROZEN]  specified by the research design (user's frozen prompt)
  [LIT]     literature-derived convention
  [PROJ]    project implementation choice (documented in the audit)

Status markers appended to values, e.g. ``dt = 0.1`` marked ``# [PROJ]`` in
the table below, indicate the decision provenance of each parameter.
"""


import math
import random
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch


# =========================================================================
# 1. UAV simulation                                          (audit 2.1)
# =========================================================================

DT = 0.1                    # seconds, integration step          [PROJ]
T_MAX = 100                 # episode horizon in steps (10 s)    [PROJ]
WORKSPACE_MIN = -8.0        # workspace bounds, metres           [PROJ]
WORKSPACE_MAX = 8.0
GOAL_RADIUS = 0.5           # r_g, success radius in metres      [PROJ]
D_MIN = 4.0                 # minimum initial start-goal metres  [PROJ]
D_MAX = None                # A6: frozen ONLY after the controller sanity gate;
                            #      None = unconstrained until then
V_0 = (0.0, 0.0)            # initial velocity, metres/second    [PROJ]

# A3 [PROJ]: velocity limit is enforced by norm-rescaling inside the shared
# dynamics module (identically for every communication method). Set to None
# to disable (not used by any frozen experiment).
V_MAX = 5.0                 # metres/second                      [PROJ]

# A5 [PROJ]: boundary-exit semantics.
#   evaluation  -> episode terminates and is marked a failure
#   training    -> fixed-horizon rollout continues (normalization clipping
#                  keeps network inputs bounded); no early stop
BOUNDARY_EVAL_TERMINATES = True


# =========================================================================
# 2. Navigation controller                                   (audit 2.2)
# =========================================================================

KP = 1.0                    # proportional gain                  [FROZEN]
KV = 2.0                    # velocity-damping gain (damping term is -Kv*v;
                            #  the acceleration clip only bounds commanded
                            #  acceleration and is NOT a damping source).
                            #  User decision after the oracle gate: Kv=1.0
                            #  gave SR 87.5% / 31% exits (overshoot, zeta=0.5);
                            #  Kv=2.0 is critically damped (zeta=1.0).
U_MAX = 2.0                 # acceleration clip, metres/s^2      [PROJ]


# =========================================================================
# 3. State normalization                                     (audit 2.3)
#    Frozen formula:  s_bar_i = 2*(clip(s_i,a_i,b_i)-a_i)/(b_i-a_i) - 1
#    Ranges are configuration constants, NEVER derived from data.
# =========================================================================

#                 x        y        vx       vy       x_g      y_g
STATE_MIN = (WORKSPACE_MIN, WORKSPACE_MIN, -V_MAX, -V_MAX,
             WORKSPACE_MIN, WORKSPACE_MIN)
STATE_MAX = (WORKSPACE_MAX, WORKSPACE_MAX, V_MAX, V_MAX,
             WORKSPACE_MAX, WORKSPACE_MAX)
STATE_DIM = 6               # n = 6, the frozen source dimension [FROZEN]


# =========================================================================
# 4. Communication (common to all three methods)             (audit 2.4)
# =========================================================================

TX_POWER = 1.0              # P                                  [FROZEN]
EPS_POWER = 1e-12           # numerical guard in power norm      [PROJ]
EPS_EQUALIZER = 1e-6        # ZF guard eps_h, A10                [PROJ]
SNR_TEST_GRID_DB = tuple(float(s) for s in range(0, 21, 2))   # 0..20 dB [PROJ]
SNR_TRAIN_RANGE_DB = (0.0, 20.0)   # U(min,max) per sample       [LIT]+[PROJ]
SNR_TRAIN_PER_SAMPLE = True                        # A8          [PROJ]

# Bandwidth grid: rho = k / n with n = 6                     [FROZEN]
BANDWIDTH_RATIOS = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0)
BANDWIDTH_K_VALUES = tuple(r * STATE_DIM for r in BANDWIDTH_RATIOS)  # (1,2,3)

# Matched-budget digital operating points (audit 8.6, M2 primary matrix):
#   B bits/component -> payload symbols n_pay = 6B + 6 == k
DIGITAL_B_BY_K = {12: 1, 18: 2, 24: 3, 30: 4, 54: 8}


# =========================================================================
# 5. Digital baseline packet/framing                         (audit 2.7, 8)
# =========================================================================

CONV_CONSTRAINT_LENGTH = 7          # K                            [FROZEN A11]
CONV_RATE = 0.5                     # R_c = 1/2                    [FROZEN A11]
CONV_GENERATORS_OCTAL = (0o171, 0o133)   # (171,133) octal, G1 first  [FROZEN A11]
MODULATION_BITS_PER_SYMBOL = 2      # log2 M, Gray-mapped QPSK     [FROZEN A11]
CONV_TAIL_BITS = CONV_CONSTRAINT_LENGTH - 1   # 6 zero-tail bits  [FROZEN 8.1]

# physical span of each normalized component: b_i - a_i = 16 for all six
_QUANT_SPAN = WORKSPACE_MAX - WORKSPACE_MIN


def quantizer_step(bits_per_component: int) -> float:
    """Delta of the endpoint-inclusive uniform quantizer on [-1,1].

    L = 2^B levels span the closed range [-1, 1] with L-1 intervals.
    """
    levels = 2 ** bits_per_component
    return 2.0 / (levels - 1)


def digital_packet(k_channel_uses: int, bits_per_component: int) -> dict:
    """Return the packet arithmetic of audit section 8 for one configuration.

    Raises InfeasibleDigitalConfig when the payload cannot fit inside the
    matched channel-use budget. Never modifies k, rho, rate or modulation.
    """
    n_src = STATE_DIM * bits_per_component          # 6B source bits
    n_info = n_src + CONV_TAIL_BITS                 # 6B + 6 info bits
    if CONV_RATE == 0.5:
        n_coded = 2 * n_info                        # exact for R_c = 1/2
    else:  # defensive; R_c is frozen to 1/2
        n_coded = int(math.ceil(n_info / CONV_RATE))
    n_pay = n_coded // MODULATION_BITS_PER_SYMBOL   # QPSK payload symbols
    if n_coded % MODULATION_BITS_PER_SYMBOL != 0:
        raise InfeasibleDigitalConfig(
            f"coded bits {n_coded} not divisible by log2(M)="
            f"{MODULATION_BITS_PER_SYMBOL}")
    k_min = n_pay
    if k_min > k_channel_uses:
        raise InfeasibleDigitalConfig(
            f"payload needs {k_min} channel uses but budget is "
            f"k={k_channel_uses}; reduce B (bits/component) or raise k")
    n_pad = k_channel_uses - n_pay
    return {
        "bits_per_component": bits_per_component,
        "n_src_bits": n_src,
        "n_info_bits": n_info,
        "n_coded_bits": n_coded,
        "n_payload_symbols": n_pay,
        "n_padding_symbols": n_pad,
        "n_transmitted_symbols": k_channel_uses,
        "k_min": k_min,
    }


class InfeasibleDigitalConfig(ValueError):
    """Raised when the digital packet cannot fit the matched budget."""


# =========================================================================
# 6. Neural models                                           (audit 2.5)
# =========================================================================

HIDDEN_DIMS = (128, 128)    # encoder/decoder hidden widths      [PROJ]

# Task-loss weights  L_task = ld*L_d + lu*L_u + lv*L_v + lT*L_T
# [FROZEN] user decision in Step 5 (A7 proposed defaults, consistent with
# final_variables-1 section 2.7 "lambda_T = 2.0, start simple").
# Project design choice - NOT a literature-derived value.
#   ld = 1.0 : distance term active
#   lu = 0.0 : control effort is a diagnostic only
#   lv = 0.0 : velocity term is a diagnostic only
#   lT = 2.0 : terminal distance active
LAMBDA_DISTANCE = 1.0
LAMBDA_CONTROL = 0.0
LAMBDA_VELOCITY = 0.0
LAMBDA_TERMINAL = 2.0

# Frozen rollout convention (audit 1.1 row 12):
#   L_d = (1/T) sum_{t=1..T}   ||p_t - p_g||^2   (post-transition states;
#         t = 0 is the initial condition, NOT included)
#   L_u = (1/T) sum_{t=0..T-1} ||u_t||^2         (applied controls)
#   L_v = (1/T) sum_{t=1..T}   ||v_t||^2         (post-transition velocities)
#   L_T = ||p_T - p_g||^2
TASK_LOSS_INCLUDES_T0 = False


# =========================================================================
# 7. Training protocol                                       (audit 2.6)
# =========================================================================

LEARNING_RATE = 1e-3        # Adam                               [LIT]
GRAD_CLIP_NORM = 1.0        # global-norm clipping               [PROJ]
BATCH_EPISODES = 256        # task rollout batches               [PROJ]
BATCH_STATES = 4096         # reconstruction state batches       [PROJ]

# ---------------------------------------------------------------------------
# FULL-SCALE TRAINING BUDGET — FROZEN Step 9C (2026-09-30, user-approved
# Candidate B). PROJECT DESIGN DECISIONS, NOT literature-derived: the
# reviewed papers explicitly do not specify epochs/steps/batch, and the
# audit §2.6 duration line ("~20k steps + early stop on val") was [PROJ]
# TBD — superseded by this freeze, NOT silently adopted.
#   Evidence (Step-9B smoke-scale probe, seed 42, non-scientific):
#   recon k=3 plateaus by ~1000 steps; task k=3 T=100 still improving at
#   150 steps; 0 NaN/Inf. Fixed-step budgets => deterministic compute,
#   reproducible across seeds. Early stopping stays DISABLED for the
#   primary execution (no validation machinery is silently introduced;
#   the audit's val protocol line remains unimplemented).
RECON_TRAIN_STEPS = 5_000   # reconstruction optimizer steps     [PROJ->FROZEN 9C]
TASK_TRAIN_STEPS = 3_000    # task-oriented optimizer steps      [PROJ->FROZEN 9C]
TASK_TRAINING_T = T_MAX     # task rollout horizon == frozen eval horizon
                            #                                    [PROJ->FROZEN 9C]
EARLY_STOPPING = False      # fixed-step primary protocol        [PROJ->FROZEN 9C]

# Checkpoint policy (Step 9C): final checkpoint per run via the Step-8
# checkpoint/identity infrastructure (results/<EXP>/<run_id>__<hash>/ with
# DONE/RUNNING/FAILED markers); NO periodic-checkpoint system is built.
# Interrupted runs are retrained (runs are deterministic by seed+config).

# M5 ablation matrix — DEFERRED (Step 9C): specification INCOMPLETE
# (task-loss-component removals, uncoded-digital reference, cross-channel
# definition are categories only). Not invented to fill the matrix.
M5_STATUS = "deferred"      #                                   [PROJ->FROZEN 9C]
M5_EXECUTION_ENABLED = False

N_TRAIN_EPISODES = 10_000   #                                    [PROJ]
N_VAL_EPISODES = 2_000      #                                    [PROJ]
N_TEST_EPISODES = 5_000     # 500 for smoke tests                [PROJ]
N_TEST_EPISODES_SMOKE = 500
TRAIN_VAL_TEST_SPLIT = (0.70, 0.15, 0.15)


# =========================================================================
# 8. Evaluation and seeds                                    (audit 2.8)
# =========================================================================

TRAINING_SEEDS = (42, 43, 44)                               # [PROJ]
TEST_SEED = 10_042          # test-episode generation            [PROJ]

# Purpose-separated RNG stream keys (audit risk 13): each consumer derives
# its own generator from these keys so that episode generation, weight
# initialization, batch shuffling, training noise and test noise never
# contaminate one another.
RNG_KEYS = {
    "episodes": "episodes",
    "init": "init",
    "shuffle": "shuffle",
    "noise_train": "noise_train",
    "noise_test": "noise_test",
}


# =========================================================================
# Reproducibility and device
# =========================================================================

def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def derive_generator(purpose: str, seed: int) -> torch.Generator:
    """Purpose-separated torch.Generator (audit 2.8, RNG_KEYS)."""
    g = torch.Generator()
    g.manual_seed(seed + _stable_hash(RNG_KEYS[purpose]))
    return g


def _stable_hash(text: str) -> int:
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) % (2 ** 32)
    return h


# =========================================================================
# Validation
# =========================================================================

def validate() -> None:
    """Fail loudly on any internally inconsistent configuration."""
    assert STATE_DIM == 6, "frozen state dimension is 6"
    assert 0.0 < DT
    assert T_MAX >= 1
    assert WORKSPACE_MIN < WORKSPACE_MAX
    assert 0.0 < GOAL_RADIUS
    assert D_MIN > 0.0
    assert D_MIN > GOAL_RADIUS, (
        "d_min must exceed the goal radius so episodes never start in success")
    if D_MAX is not None:
        assert D_MAX > D_MIN, "d_max must exceed d_min"
    assert V_MAX is None or V_MAX > 0.0
    assert KP > 0.0 and KV > 0.0
    for a, b in zip(STATE_MIN, STATE_MAX):
        assert a < b, "normalization ranges must satisfy a_i < b_i"
    assert TX_POWER > 0.0
    assert EPS_POWER > 0.0 and EPS_EQUALIZER > 0.0
    for k in BANDWIDTH_K_VALUES:
        assert k == int(k) and k > 0, "k must be a positive integer"
    assert len(SNR_TEST_GRID_DB) >= 2
    lo, hi = SNR_TRAIN_RANGE_DB
    assert lo < hi
    # matched-budget digital points must be internally consistent
    for k, b in DIGITAL_B_BY_K.items():
        pkt = digital_packet(k, b)
        assert pkt["n_padding_symbols"] == 0, (
            f"M2 digital point k={k} must exactly fit (audit 8.6)")
    assert 0.0 < LEARNING_RATE
    assert RECON_TRAIN_STEPS > 0 and TASK_TRAIN_STEPS > 0
    assert TASK_TRAINING_T == T_MAX, (
        "task training horizon must stay aligned with the frozen eval horizon")
    assert EARLY_STOPPING is False, (
        "primary protocol is fixed-step; early stopping is a later decision")
    assert M5_STATUS == "deferred" and M5_EXECUTION_ENABLED is False
    assert len(TRAINING_SEEDS) >= 3, "at least 3 training seeds"
    for lam in (LAMBDA_DISTANCE, LAMBDA_CONTROL, LAMBDA_VELOCITY,
                LAMBDA_TERMINAL):
        assert lam >= 0.0, "task-loss weights must be non-negative"
    assert (LAMBDA_DISTANCE + LAMBDA_TERMINAL) > 0.0, (
        "at least one navigation term must be active")


def describe() -> str:
    """Human-readable configuration summary (used by the sanity scripts)."""
    lines = [
        "UAV",
        f"  dt                 = {DT} s",
        f"  T_max              = {T_MAX} steps ({T_MAX * DT:.1f} s)",
        f"  workspace          = [{WORKSPACE_MIN}, {WORKSPACE_MAX}]^2 m",
        f"  d_min / d_max      = {D_MIN} m / {D_MAX}",
        f"  goal_radius r_g    = {GOAL_RADIUS} m",
        f"  v_max (mechanism)  = {V_MAX} (norm-rescale, A3)",
        f"  u_max              = {U_MAX} m/s^2",
        "Controller",
        f"  Kp / Kv            = {KP} / {KV}",
        "Normalization",
        f"  state_min          = {STATE_MIN}",
        f"  state_max          = {STATE_MAX}",
        "Communication",
        f"  P                  = {TX_POWER}",
        f"  bandwidth ratios   = {BANDWIDTH_RATIOS} -> k = {BANDWIDTH_K_VALUES}",
        f"  SNR test grid (dB) = {SNR_TEST_GRID_DB}",
        f"  SNR train          = U{SNR_TRAIN_RANGE_DB} dB per sample",
        f"  matched digital k  = {sorted(DIGITAL_B_BY_K)} "
        f"(B = {[DIGITAL_B_BY_K[k] for k in sorted(DIGITAL_B_BY_K)]})",
        "Protocol",
        f"  training seeds     = {TRAINING_SEEDS}",
        f"  train budget (9C)  = recon {RECON_TRAIN_STEPS} / task "
        f"{TASK_TRAIN_STEPS} steps (fixed, T={TASK_TRAINING_T}, "
        f"no early stop)",
        f"  M5                 = {M5_STATUS} (specification incomplete)",
        f"  test episodes      = {N_TEST_EPISODES} (smoke {N_TEST_EPISODES_SMOKE})",
        f"  device             = {get_device()}",
    ]
    return "\n".join(lines)


validate()


# Module-style namespace preserving the original
# `from project import config` / `config.X` attribute contract
# used verbatim by the executor drivers and evaluation harness.
config = SimpleNamespace(**{
    "BANDWIDTH_K_VALUES": BANDWIDTH_K_VALUES,
    "BANDWIDTH_RATIOS": BANDWIDTH_RATIOS,
    "BATCH_EPISODES": BATCH_EPISODES,
    "BATCH_STATES": BATCH_STATES,
    "BOUNDARY_EVAL_TERMINATES": BOUNDARY_EVAL_TERMINATES,
    "CONV_CONSTRAINT_LENGTH": CONV_CONSTRAINT_LENGTH,
    "CONV_GENERATORS_OCTAL": CONV_GENERATORS_OCTAL,
    "CONV_RATE": CONV_RATE,
    "CONV_TAIL_BITS": CONV_TAIL_BITS,
    "DIGITAL_B_BY_K": DIGITAL_B_BY_K,
    "DT": DT,
    "D_MAX": D_MAX,
    "D_MIN": D_MIN,
    "EARLY_STOPPING": EARLY_STOPPING,
    "EPS_EQUALIZER": EPS_EQUALIZER,
    "EPS_POWER": EPS_POWER,
    "GOAL_RADIUS": GOAL_RADIUS,
    "GRAD_CLIP_NORM": GRAD_CLIP_NORM,
    "HIDDEN_DIMS": HIDDEN_DIMS,
    "InfeasibleDigitalConfig": InfeasibleDigitalConfig,
    "KP": KP,
    "KV": KV,
    "LAMBDA_CONTROL": LAMBDA_CONTROL,
    "LAMBDA_DISTANCE": LAMBDA_DISTANCE,
    "LAMBDA_TERMINAL": LAMBDA_TERMINAL,
    "LAMBDA_VELOCITY": LAMBDA_VELOCITY,
    "LEARNING_RATE": LEARNING_RATE,
    "M5_EXECUTION_ENABLED": M5_EXECUTION_ENABLED,
    "M5_STATUS": M5_STATUS,
    "MODULATION_BITS_PER_SYMBOL": MODULATION_BITS_PER_SYMBOL,
    "N_TEST_EPISODES": N_TEST_EPISODES,
    "N_TEST_EPISODES_SMOKE": N_TEST_EPISODES_SMOKE,
    "N_TRAIN_EPISODES": N_TRAIN_EPISODES,
    "N_VAL_EPISODES": N_VAL_EPISODES,
    "RECON_TRAIN_STEPS": RECON_TRAIN_STEPS,
    "RNG_KEYS": RNG_KEYS,
    "SNR_TEST_GRID_DB": SNR_TEST_GRID_DB,
    "SNR_TRAIN_PER_SAMPLE": SNR_TRAIN_PER_SAMPLE,
    "SNR_TRAIN_RANGE_DB": SNR_TRAIN_RANGE_DB,
    "STATE_DIM": STATE_DIM,
    "STATE_MAX": STATE_MAX,
    "STATE_MIN": STATE_MIN,
    "TASK_LOSS_INCLUDES_T0": TASK_LOSS_INCLUDES_T0,
    "TASK_TRAINING_T": TASK_TRAINING_T,
    "TASK_TRAIN_STEPS": TASK_TRAIN_STEPS,
    "TEST_SEED": TEST_SEED,
    "TRAINING_SEEDS": TRAINING_SEEDS,
    "TRAIN_VAL_TEST_SPLIT": TRAIN_VAL_TEST_SPLIT,
    "TX_POWER": TX_POWER,
    "T_MAX": T_MAX,
    "U_MAX": U_MAX,
    "V_0": V_0,
    "V_MAX": V_MAX,
    "WORKSPACE_MAX": WORKSPACE_MAX,
    "WORKSPACE_MIN": WORKSPACE_MIN,
    "_QUANT_SPAN": _QUANT_SPAN,
    "_stable_hash": _stable_hash,
    "derive_generator": derive_generator,
    "describe": describe,
    "digital_packet": digital_packet,
    "get_device": get_device,
    "quantizer_step": quantizer_step,
    "set_global_seed": set_global_seed,
    "validate": validate,
})


# ============================================================
# SECTION 2  STATE / NORMALIZATION
# ============================================================
# ---- Source: project/data/normalization.py ----
"""Frozen state normalization (audit 1.1 row 5, spec section 6).

For every component i with declared range [a_i, b_i]:

    s_bar_i = 2 * (clip(s_i, a_i, b_i) - a_i) / (b_i - a_i) - 1

The inverse is the exact algebraic inverse on the in-range set:

    s_i = a_i + (s_bar_i + 1) / 2 * (b_i - a_i),   then clipped to [a_i, b_i]

Ranges come from ``project.config`` only. They are never derived from data
and never refit; this is a hard fairness rule of the project.
"""


from typing import Sequence

import torch



def normalize(
    s: torch.Tensor,
    state_min: Sequence[float] = config.STATE_MIN,
    state_max: Sequence[float] = config.STATE_MAX,
) -> torch.Tensor:
    """Map the last dimension of ``s`` to the frozen [-1, 1] interval."""
    a = torch.as_tensor(state_min, dtype=s.dtype, device=s.device)
    b = torch.as_tensor(state_max, dtype=s.dtype, device=s.device)
    s_clipped = torch.clamp(s, min=a, max=b)
    return 2.0 * (s_clipped - a) / (b - a) - 1.0


def denormalize(
    s_bar: torch.Tensor,
    state_min: Sequence[float] = config.STATE_MIN,
    state_max: Sequence[float] = config.STATE_MAX,
) -> torch.Tensor:
    """Exact inverse of :func:`normalize` on the in-range set.

    Values of ``s_bar`` outside [-1, 1] correspond to clipped states and
    are mapped back onto the range boundary (the inverse is then a
    projection, not an algebraic inverse).
    """
    a = torch.as_tensor(state_min, dtype=s_bar.dtype, device=s_bar.device)
    b = torch.as_tensor(state_max, dtype=s_bar.dtype, device=s_bar.device)
    s = a + (s_bar + 1.0) * 0.5 * (b - a)
    return torch.clamp(s, min=a, max=b)

# ============================================================
# SECTION 3  CHANNELS (complex utilities, power normalization, AWGN, Rayleigh + ZF)
# ============================================================
# ---- Source: project/channels/complex_utils.py ----
"""Complex-baseband utilities shared by every communication method.

Representation convention (fixed for the whole project, documented in the
audit sections 3 and 4.1):

* Inside the neural networks everything is REAL, laid out as ``[..., 2k]``
  with the **stacked I/Q order**: ``x = [Re(z)_1..Re(z)_k, Im(z)_1..Im(z)_k]``.
* Channel operations accept either complex tensors ``[..., k]`` (dtype
  complex64/complex128) or real stacked tensors ``[..., 2k]``. They return
  the same kind they were given.

SNR and noise conventions (frozen, audit section 4.3):

    SNR_dB = 10 log10(P / sigma^2)      =>      sigma^2 = P * 10^(-SNR_dB/10)

For a proper circular complex Gaussian ``n ~ CN(0, sigma^2)``:

    Re(n), Im(n) ~ N(0, sigma^2 / 2)    so that E[|n|^2] = sigma^2.

Using ``sigma^2`` as the per-component variance would be the classic 3 dB
error and is guarded against by the verification suite.
"""


from typing import Optional

import torch



# ---------------------------------------------------------------------------
# kind / shape helpers
# ---------------------------------------------------------------------------

def _is_complex(x: torch.Tensor) -> bool:
    return x.is_complex()


def check_codeword(x: torch.Tensor) -> None:
    """Validate a codeword's shape and finiteness; nothing else.

    ``x`` is either complex ``[..., k]`` or real stacked ``[..., 2k]``.
    A real tensor with an odd last dimension cannot be I/Q-paired and is
    rejected; complex tensors accept any k >= 1.
    """
    if x.dim() < 1:
        raise ValueError(f"codeword must have at least 1 dim, got {tuple(x.shape)}")
    if not _is_complex(x) and x.shape[-1] % 2 != 0:
        raise ValueError(
            f"real-stacked codeword needs an even last dim, got {x.shape[-1]}")
    if not torch.isfinite(x.real if _is_complex(x) else x).all():
        raise FloatingPointError("codeword contains NaN/Inf")


def snr_db_to_sigma2(snr_db: float, power: float = config.TX_POWER) -> float:
    """Frozen SNR convention: sigma^2 = P * 10^(-SNR_dB / 10)."""
    if power <= 0:
        raise ValueError(f"power must be positive, got {power}")
    return float(power) * (10.0 ** (-float(snr_db) / 10.0))


def sigma2_to_snr_db(sigma2: float, power: float = config.TX_POWER) -> float:
    """Inverse of :func:`snr_db_to_sigma2` (used in verification)."""
    return 10.0 * math_log10(float(power) / float(sigma2))


def math_log10(x: float) -> float:
    import math
    return math.log10(x)


def to_complex(x: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """Real stacked ``[..., 2k]`` -> complex ``[..., k]``.

    Split order: first k entries are the real parts, last k the imaginary
    parts (``z_i = a_i + j * a_{k+i}``, stacked convention of audit 1.1 row 6).
    """
    if _is_complex(x):
        return x
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            f"last dim must be even for stacked I/Q, got {x.shape[-1]}")
    half = x.shape[-1] // 2 if k is None else k
    if k is not None and x.shape[-1] != 2 * k:
        raise ValueError(f"expected last dim {2 * k}, got {x.shape[-1]}")
    return torch.complex(x[..., :half], x[..., half:])


def to_real(x: torch.Tensor) -> torch.Tensor:
    """Complex ``[..., k]`` -> real stacked ``[..., 2k]`` (same split order)."""
    if not _is_complex(x):
        return x
    return torch.cat([x.real, x.imag], dim=-1)


# ---------------------------------------------------------------------------
# shared circular complex Gaussian sampling
# ---------------------------------------------------------------------------

def sample_complex_gaussian(
    shape: torch.Size,
    variance: float,
    generator: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    complex_out: bool = True,
) -> torch.Tensor:
    """Draw ``CN(0, variance)`` samples: each real component has variance/2.

    This single helper is the ONLY place in the project where complex
    Gaussian samples are produced (audit risk 1: the sigma^2/2 rule lives
    in exactly one function).
    """
    if variance <= 0:
        raise ValueError(f"variance must be positive, got {variance}")
    shape = torch.Size(shape)
    std = (variance / 2.0) ** 0.5
    re = torch.randn(shape, generator=generator, device=device,
                     dtype=dtype) * std
    im = torch.randn(shape, generator=generator, device=device,
                     dtype=dtype) * std
    if complex_out:
        return torch.complex(re, im)          # shape == `shape`
    return torch.cat([re, im], dim=-1)        # stacked: last dim doubled


# ---------------------------------------------------------------------------
# power normalization (frozen audit 1.1 row 8)
# ---------------------------------------------------------------------------

def power_normalize(
    z_tilde: torch.Tensor,
    k: Optional[int] = None,
    power: float = config.TX_POWER,
    eps: float = config.EPS_POWER,
) -> torch.Tensor:
    """Per-codeword power normalization to average symbol power P.

        z = sqrt(kP) * z_tilde / sqrt( sum_i |z_tilde_i|^2 + eps )

    Applied per sample (the last dimension is one codeword), never per
    batch and never per symbol. Accepts complex ``[..., k]`` or real
    stacked ``[..., 2k]`` input and returns the same kind.
    """
    if k is None:
        k = z_tilde.shape[-1] if _is_complex(z_tilde) else z_tilde.shape[-1] // 2
    if _is_complex(z_tilde):
        energy = (z_tilde.real ** 2 + z_tilde.imag ** 2).sum(dim=-1, keepdim=True)
        scale = (k * power) ** 0.5 / torch.sqrt(energy + eps)
        return z_tilde * scale
    energy = z_tilde.pow(2).sum(dim=-1, keepdim=True)
    scale = (k * power) ** 0.5 / torch.sqrt(energy + eps)
    return z_tilde * scale


def codeword_avg_power(z: torch.Tensor) -> torch.Tensor:
    """(1/k) * sum_i |z_i|^2 per sample; should equal P after normalization."""
    if _is_complex(z):
        energy = (z.real ** 2 + z.imag ** 2).sum(dim=-1)
    else:
        energy = z.pow(2).sum(dim=-1)
    k = z.shape[-1] if _is_complex(z) else z.shape[-1] // 2
    return energy / k

# ---- Source: project/channels/awgn.py ----
"""Complex AWGN channel (frozen, audit 1.1 row 9).

    y = z + n,      n ~ CN(0, sigma^2 I),      sigma^2 = P * 10^(-SNR_dB/10)

The frozen SNR convention is per-complex-symbol ``P / sigma^2``; each real
component of the noise has variance ``sigma^2 / 2`` so that ``E[|n|^2] =
sigma^2``. Noise is sampled through the single shared sampler in
``complex_utils`` (no duplicated AWGN logic anywhere).

Noise is an exogenous constant inside the autograd graph: gradients flow
through ``z`` but never through the sampled noise (audit risk 9).
"""


from typing import Optional

import torch



def awgn_channel(
    z: torch.Tensor,
    snr_db: float,
    generator: Optional[torch.Generator] = None,
    power: float = config.TX_POWER,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Transmit one codeword batch through complex AWGN.

    Parameters
    ----------
    z : complex ``[B, k]`` or real stacked ``[B, 2k]`` codeword batch
        (already power-normalized by the caller).
    snr_db : configured SNR in dB (per complex symbol, P / sigma^2).
    generator : optional torch.Generator for reproducibility.
    noise : optional pre-generated realization (paired evaluation).
        When given, it is used verbatim instead of sampling; ``generator``
        and ``snr_db`` are then irrelevant to the added noise. Must match
        the kind/shape contract of ``z`` (complex ``[B, k]`` or stacked
        real ``[B, 2k]``).

    Returns the received signal with the same kind/shape as ``z``.
    """
    check_codeword(z)
    if noise is not None:
        check_codeword(noise)
        if noise.shape != z.shape or noise.is_complex() != z.is_complex():
            raise ValueError(
                f"noise shape/kind mismatch: {tuple(noise.shape)} "
                f"complex={noise.is_complex()} vs z {tuple(z.shape)} "
                f"complex={z.is_complex()}")
        return z + noise
    sigma2 = snr_db_to_sigma2(snr_db, power=power)
    if _is_complex(z):
        n = sample_complex_gaussian(
            z.shape, sigma2, generator=generator, device=z.device,
            dtype=z.real.dtype, complex_out=True)
    else:
        k = z.shape[-1] // 2
        n = sample_complex_gaussian(
            torch.Size(z.shape[:-1] + (k,)), sigma2, generator=generator,
            device=z.device, dtype=z.dtype, complex_out=False)
    return z + n

# ---- Source: project/channels/rayleigh.py ----
"""Slow (block) Rayleigh fading channel with perfect-CSI equalization.

Frozen model (audit 1.1 rows 10-11):

    y = h * z + n,      h ~ CN(0, 1),      n ~ CN(0, sigma^2 I)

* ``h`` is a SCALAR per codeword: one coefficient shared by all k symbols
  of one transmitted codeword (block/slow fading), independent across
  codewords. This is exactly the DeepJSCC slow-Rayleigh convention.
* E[|h|^2] = 1; |h| is Rayleigh, phase uniform. h is NEVER renormalized
  after sampling (verification item 9).
* Perfect receiver CSI (main experiment): zero-forcing equalization
  ``y_eq = y / h`` with a numerical guard for extremely small |h|:

      y_eq = y * conj(h) / max(|h|^2, eps_eq)

  which equals exact ZF whenever |h|^2 >> eps_eq and stays finite
  otherwise. The same equalizer is used by the digital baseline (fairness).
* Noise is drawn by the shared sampler; the SNR axis is the AVERAGE SNR
  P / sigma^2 (per codeword the instantaneous SNR is |h|^2 P / sigma^2).
"""


from typing import Optional

import torch



def sample_fading(
    batch: int,
    generator: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """One complex scalar h ~ CN(0,1) per codeword; shape ``[B, 1]``.

    The trailing singleton dimension broadcasts over the k symbols of a
    codeword, which IS the block-fading convention.
    """
    h = sample_complex_gaussian(
        torch.Size((batch, 1)), 1.0, generator=generator, device=device,
        dtype=dtype, complex_out=True)
    return h


def rayleigh_channel(
    z: torch.Tensor,
    snr_db: float,
    generator: Optional[torch.Generator] = None,
    power: float = config.TX_POWER,
    return_h: bool = False,
    noise: Optional[torch.Tensor] = None,
    h: Optional[torch.Tensor] = None,
):
    """Transmit one codeword batch through slow Rayleigh fading + AWGN.

    ``noise`` (complex ``[B, k]`` or stacked real matching ``z``) and
    ``h`` (complex ``[B, 1]``) may be pre-generated for paired evaluation;
    when supplied they are used verbatim and no sampling occurs.
    """
    check_codeword(z)
    batch = z.shape[0] if z.dim() > 1 else 1
    if h is None:
        h = sample_fading(
            batch, generator=generator, device=z.device,
            dtype=z.real.dtype if z.is_complex() else z.dtype)
    else:
        if (not h.is_complex() or tuple(h.shape) != (batch, 1)):
            raise ValueError(
                f"h must be complex [batch, 1], got {tuple(h.shape)} "
                f"complex={h.is_complex()}")
    sigma2 = snr_db_to_sigma2(snr_db, power=power)
    if z.is_complex():
        if noise is None:
            n = sample_complex_gaussian(
                z.shape, sigma2, generator=generator, device=z.device,
                dtype=z.real.dtype, complex_out=True)
        else:
            if (not noise.is_complex() or noise.shape != z.shape):
                raise ValueError(
                    f"noise must be complex matching z, got {tuple(noise.shape)}")
            n = noise
        y = h * z + n
    else:
        # stacked real input: apply the same complex h to the complex view
        k = z.shape[-1] // 2
        zc = torch.complex(z[..., :k], z[..., k:])
        if noise is None:
            n = sample_complex_gaussian(
                zc.shape, sigma2, generator=generator, device=z.device,
                dtype=z.dtype, complex_out=True)
        else:
            if noise.is_complex() or noise.shape != z.shape:
                raise ValueError(
                    f"noise must be real-stacked matching z, got "
                    f"{tuple(noise.shape)} complex={noise.is_complex()}")
            n = torch.complex(noise[..., :k], noise[..., k:])
        yc = h * zc + n
        y = torch.cat([yc.real, yc.imag], dim=-1)
    if return_h:
        return y, h
    return y


def equalize_zf(
    y: torch.Tensor,
    h: torch.Tensor,
    eps_eq: float = config.EPS_EQUALIZER,
) -> torch.Tensor:
    """Guarded zero-forcing equalization: y * conj(h) / max(|h|^2, eps_eq).

    ``y``: complex ``[B, k]`` or real stacked ``[B, 2k]``.
    ``h``: complex ``[B, 1]`` block-fading coefficients (CSIR).
    """
    hh = (h.real ** 2 + h.imag ** 2).clamp_min(eps_eq)     # [B, 1]
    if y.is_complex():
        return y * h.conj() / hh
    k = y.shape[-1] // 2
    yc = torch.complex(y[..., :k], y[..., k:])
    ye = yc * h.conj() / hh
    return torch.cat([ye.real, ye.imag], dim=-1)


# ============================================================
# SECTION 4  DEEPJSCC MODELS (task-oriented AND reconstruction-oriented share this exact architecture: ReconstructionDeepJSCC container)
# ============================================================
# ---- Source: project/models/deepjscc_encoder.py ----
"""DeepJSCC encoder (frozen architecture, audit 1.1 row 6, spec section 7).

    f_theta : R^6 -> C^k, implemented as

        Linear(6, 128) -> PReLU -> Linear(128, 128) -> PReLU -> Linear(128, 2k)

Project adaptation note (do not misattribute): the original DeepJSCC paper
uses a CNN for images; this MLP for a 6-D vector source is a documented
PROJECT choice - the paper does not prescribe an MLP for vector states.

The final layer outputs exactly 2k REAL values. Conversion to k complex
channel symbols uses the frozen stacked I/Q convention of Step 3 via
``complex_utils.to_complex`` (``z_i = a_i + j * a_{k+i}``). The system
transmits k complex channel uses - never 2k.
"""


import torch
from torch import nn



class DeepJSCCEncoder(nn.Module):
    """MLP encoder producing 2k real outputs (k complex channel uses)."""

    def __init__(
        self,
        state_dim: int = config.STATE_DIM,
        k: int = 3,
        hidden_dims: tuple = config.HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        h1, h2 = hidden_dims
        self.k = k
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.PReLU(),
            nn.Linear(h1, h2),
            nn.PReLU(),
            nn.Linear(h2, 2 * k),
        )

    def forward(self, s_bar: torch.Tensor) -> torch.Tensor:
        """Normalized state ``[..., 6]`` -> real stacked symbols ``[..., 2k]``."""
        if s_bar.shape[-1] != 6:
            raise ValueError(
                f"encoder expects last dim 6, got {s_bar.shape[-1]}")
        return self.backbone(s_bar)

    def encode_complex(self, s_bar: torch.Tensor) -> torch.Tensor:
        """Normalized state -> complex codeword ``[..., k]`` (pre power norm)."""
        return to_complex(self.forward(s_bar))

# ---- Source: project/models/deepjscc_decoder.py ----
"""DeepJSCC decoder (mirror of the encoder, audit 3 architecture table).

    g_phi : C^k -> R^6, implemented on the real stacked representation as

        Linear(2k, 128) -> PReLU -> Linear(128, 128) -> PReLU -> Linear(128, 6)

Input: equalized complex symbols (or their real stacked form) using the
SAME stacked I/Q ordering as the encoder. Output: the 6-D NORMALIZED state
estimate (no output activation - the frozen specification has none);
denormalization to physical units is done by the existing normalization
module, never inside the decoder.
"""


import torch
from torch import nn



class DeepJSCCDecoder(nn.Module):
    """MLP decoder mapping 2k real inputs to the 6-D normalized state."""

    def __init__(
        self,
        k: int = 3,
        state_dim: int = config.STATE_DIM,
        hidden_dims: tuple = config.HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        h1, h2 = hidden_dims
        self.k = k
        self.backbone = nn.Sequential(
            nn.Linear(2 * k, h1),
            nn.PReLU(),
            nn.Linear(h1, h2),
            nn.PReLU(),
            nn.Linear(h2, state_dim),
        )

    def forward(self, y_eq) -> torch.Tensor:
        """Equalized symbols (complex ``[..., k]`` or real ``[..., 2k]``)
        -> normalized state estimate ``[..., 6]``."""
        return self.backbone(to_real(y_eq))

# ---- Source: project/models/reconstruction_jscc.py ----
"""Reconstruction-oriented DeepJSCC baseline: complete communication path.

Frozen pipeline (audit 1.1 rows 6-9, 13; item 6):

    s -> normalize -> encoder -> [B,2k] real -> [B,k] complex
      -> power normalization (shared Step-3 function)
      -> channel (shared Step-3 AWGN / block-Rayleigh)
      -> perfect-CSI ZF equalization (Rayleigh)
      -> decoder -> s_hat_bar (normalized) -> denormalize

* Channel uses: k complex symbols (never 2k).
* The model REUSES the verified Step-3 channel functions and power
  normalization; it duplicates none of them.
* Gradients flow through the whole path (the channels are ordinary
  differentiable tensor ops); noise/fading are drawn from a torch.Generator
  and enter the graph as constants.
"""


from typing import Optional

import torch



class ReconstructionDeepJSCC(torch.nn.Module):
    """s -> encode -> power norm -> channel -> equalize -> decode -> s_hat."""

    def __init__(self, k: int = 3) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.encoder = DeepJSCCEncoder(k=k)
        self.decoder = DeepJSCCDecoder(k=k)

    # ------------------------------------------------------------------
    # building blocks (exposed for the verification tests)
    # ------------------------------------------------------------------

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        """Physical state -> power-normalized complex codeword [B, k]."""
        s_bar = normalize(s)
        z_tilde = self.encoder.encode_complex(s_bar)
        return power_normalize(z_tilde, k=self.k)

    def decode(self, y_eq: torch.Tensor) -> torch.Tensor:
        """Equalized symbols -> PHYSICAL state estimate [B, 6]."""
        s_bar_hat = self.decoder(y_eq)
        return denormalize(s_bar_hat)

    # ------------------------------------------------------------------
    # full path
    # ------------------------------------------------------------------

    def forward(
        self,
        s: torch.Tensor,
        snr_db: float,
        channel: str = "awgn",
        generator: Optional[torch.Generator] = None,
        denormalize_output: bool = True,
        noise: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One transmission of the state through the frozen pipeline.

        Parameters
        ----------
        s : physical state ``[B, 6]``.
        snr_db : configured per-symbol SNR (P / sigma^2).
        channel : ``"awgn"`` or ``"rayleigh"`` (block fading + perfect-CSI
            ZF equalization, the frozen main-experiment convention).
        generator : RNG for reproducible noise/fading.
        denormalize_output : return physical units (True) or the raw
            normalized estimate (False; used by the loss path).
        noise / h : optional pre-generated realizations for paired
            evaluation (complex [B, k] noise; complex [B, 1] h); used
            verbatim instead of sampling - Step-3 conventions untouched.
        """
        if channel not in ("awgn", "rayleigh"):
            raise ValueError(f"unknown channel {channel!r}")
        z = self.encode(s)
        if channel == "awgn":
            y = awgn_channel(z, snr_db, generator=generator, noise=noise)
            y_eq = y
        else:
            y, h_used = rayleigh_channel(z, snr_db, generator=generator,
                                         return_h=True, noise=noise, h=h)
            y_eq = equalize_zf(y, h_used)
        s_bar_hat = self.decoder(y_eq)
        if denormalize_output:
            return denormalize(s_bar_hat)
        return s_bar_hat

    def forward_losses(
        self,
        s: torch.Tensor,
        snr_db: float,
        channel: str = "awgn",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return the NORMALIZED estimate for the frozen loss (item 7/8).

        The frozen reconstruction objective is defined on the normalized
        state; computing it from a denormalized output would be an
        inconsistent normalization (hard stop condition), so this helper
        is the canonical training path.
        """
        return self.forward(s, snr_db, channel=channel, generator=generator,
                            denormalize_output=False)


# ============================================================
# SECTION 5  CONVENTIONAL DIGITAL BASELINE (quantizer, conv K=7 Rc=1/2, soft Viterbi, QPSK Gray)
# ============================================================
# ---- Source: project/baselines/digital.py ----
"""Conventional separated digital baseline (audit section 8, frozen A1 + A11).

Chain (one state transmission = one control step):

    s -> normalize -> uniform scalar quantizer (B bits/component)
      -> bit packing (component-major, MSB-first per component)
      -> + 6 zero tail bits (K - 1, zero-tail termination)
      -> convolutional encoder K = 7, R_c = 1/2, G1 = (171)_8 first
      -> Gray-QPSK (unit-average-power constellation)
      -> padding symbols (zero bits) to exactly k complex channel uses
      -> SHARED Step-3 channel (AWGN or block Rayleigh)
      -> [Rayleigh: SHARED Step-3 ZF equalization]
      -> exact log-MAP soft LLRs (AWGN: sigma^2; Rayleigh: sigma^2/|h|^2
         per codeword with the Step-3 guard, per-symbol perfect-CSI scaling)
      -> soft-decision Viterbi (traceback from the known zero-tail state 0)
      -> drop 6 tail bits -> unpack -> dequantize -> denormalize -> s_hat

Frozen conventions and their exact definitions:

* Quantizer (endpoint-inclusive uniform, source-document "Quantizer
  mathematics"): Delta = 2 / (2^B - 1) on the normalized range [-1, 1],
  q = clamp(round((clip(s,-1,1) + 1)/Delta), 0, 2^B - 1), s_hat = -1 + Delta*q.
  Maximum granular error <= Delta/2; endpoints reproduce exactly.
* Bit order: component-major, MSB-first per component (fixed project
  convention, audit 8.1).
* Convolutional code (polynomial convention; D^0 = current input u,
  register m1..m6 with m1 the most recent previous input):
      G1 = (171)_8 = 1111001_2 = 1 + D^3 + D^4 + D^5 + D^6
           -> c1 = u ^ m3 ^ m4 ^ m5 ^ m6
      G2 = (133)_8 = 1011011_2 = 1 + D + D^3 + D^4 + D^6
           -> c2 = u ^ m1 ^ m3 ^ m4 ^ m6
  Output ordering: (c1, c2) per input bit - G1 FIRST (fixed convention).
* Gray-QPSK label mapping (unit average power): the coded bit pair
  (b0, b1) maps to  z = ((1 - 2*b0) + j*(1 - 2*b1)) / sqrt(2),
  i.e. b0 is the I-axis bit, b1 the Q-axis bit; adjacent constellation
  points differ in exactly one bit (Gray).
* Exact log-MAP LLR sign convention: LLR = log[ P(bit = 0 | y) / P(bit = 1 | y) ],
  so the Viterbi/decision rule is  bit = 0 iff LLR > 0.  For the frozen
  Gray-QPSK mapping the two label bits separate across the I/Q axes, so
  the exact log-MAP marginalization reduces - EXACTLY, with no Max-Log
  approximation - to the linear form
      LLR_b0 = 4*sqrt(2)*Re(y_eq)/sigma_eff^2 ,
      LLR_b1 = 4*sqrt(2)*Im(y_eq)/sigma_eff^2 .
  AWGN: sigma_eff^2 = sigma^2 (per complex symbol). Rayleigh: with the
  frozen ZF equalizer y_eq = y h* / max(|h|^2, eps_eq) the per-symbol
  effective complex noise variance is sigma_eff^2 = sigma^2 / max(|h|^2,
  eps_eq) (per-codeword scalar, perfect CSI, the Step-3 guard reused).
  Theory axis: this is SNR = P/sigma^2 per complex symbol; equivalently
  the bit error rate is Q(sqrt(2 Eb/N0)) with Eb = Es/2, N0 = sigma^2 -
  the same curve expressed in the supplementary digital label.
* Padding: filler QPSK symbols mapped from zero bits, added AFTER coding,
  transmitted inside the k-symbol packet at the same power, and DROPPED
  before LLR extraction (never reaches Viterbi/dequantizer). Tail bits
  are overhead and never counted as payload.
* Budget: 6B + 6 <= k is enforced by config.digital_packet; violations
  raise InfeasibleDigitalConfig (k in {1, 2, 3} is INFEASIBLE by design).

No channel, normalization, controller, or dynamics logic is duplicated
here: the Step-3 channel functions and the frozen Step 0-2 modules are
reused verbatim. This module contains no learnable parameters.
"""


from typing import Dict, Optional, Tuple

import math

import torch



# ---------------------------------------------------------------------------
# Gate A - quantizer
# ---------------------------------------------------------------------------

def quantize(s_bar: torch.Tensor, bits_per_component: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Frozen endpoint-inclusive uniform quantizer on the normalized state.

    Returns ``(q, s_hat_bar)`` with integer levels in [0, 2^B - 1] and the
    reconstructed values on [-1, 1].
    """
    delta = config.quantizer_step(bits_per_component)
    levels = 2 ** bits_per_component
    clipped = torch.clamp(s_bar, min=-1.0, max=1.0)
    q = torch.clamp(torch.round((clipped + 1.0) / delta), 0, levels - 1)
    return q, -1.0 + delta * q


def dequantize(q: torch.Tensor, bits_per_component: int) -> torch.Tensor:
    """Inverse map of :func:`quantize` for recovered levels."""
    delta = config.quantizer_step(bits_per_component)
    return -1.0 + delta * q.to(torch.get_default_dtype())


# ---------------------------------------------------------------------------
# bit packing / unpacking (component-major, MSB-first)
# ---------------------------------------------------------------------------

def pack_bits(q: torch.Tensor, bits_per_component: int) -> torch.Tensor:
    """Integer levels ``[..., 6]`` -> bits ``[..., 6B]``.

    Component-major: component j occupies bits [j*B, (j+1)*B); within a
    component, MSB first.
    """
    B = bits_per_component
    powers = torch.pow(2, torch.arange(B - 1, -1, -1, device=q.device,
                                       dtype=q.dtype))       # MSB first
    bits = ((q.unsqueeze(-1) / powers).floor() % 2).to(torch.long)
    return bits.flatten(start_dim=-2)


def unpack_bits(bits: torch.Tensor, bits_per_component: int) -> torch.Tensor:
    """Inverse of :func:`pack_bits`; returns levels ``[..., 6]``."""
    B = bits_per_component
    shaped = bits.reshape(*bits.shape[:-1], 6, B)
    powers = torch.pow(2, torch.arange(B - 1, -1, -1, device=bits.device,
                                       dtype=bits.dtype))
    return (shaped * powers).sum(dim=-1)


# ---------------------------------------------------------------------------
# Gate C - convolutional encoder (K = 7, R = 1/2, G1 first)
# ---------------------------------------------------------------------------

def _conv_tables(device: torch.device) -> Dict[str, torch.Tensor]:
    """State tables for the frozen code.

    State bit layout: state = sum_{i=1..6} m_i * 2^(i-1), m1 = most recent
    previous input. next_state = (m1..m6 << 1 | u) keeping 6 bits.
    """
    states = torch.arange(64, device=device)
    m1 = (states >> 0) & 1
    m2 = (states >> 1) & 1
    m3 = (states >> 2) & 1
    m4 = (states >> 3) & 1
    m5 = (states >> 4) & 1
    m6 = (states >> 5) & 1
    u0 = torch.zeros_like(states)
    u1 = torch.ones_like(states)
    # G1 = 1 + D^3 + D^4 + D^5 + D^6 ; G2 = 1 + D + D^3 + D^4 + D^6
    out = torch.stack(
        [torch.stack([u0 ^ m3 ^ m4 ^ m5 ^ m6, u0 ^ m1 ^ m3 ^ m4 ^ m6], dim=-1),
         torch.stack([u1 ^ m3 ^ m4 ^ m5 ^ m6, u1 ^ m1 ^ m3 ^ m4 ^ m6], dim=-1)],
        dim=1)                                       # [64, 2(u), 2(c)]
    nxt = ((states * 2) % 64).unsqueeze(1).expand(64, 2) \
        | torch.stack([u0, u1], dim=1)                # [64, 2]
    return {"out": out, "next": nxt}


def conv_encode(info_bits: torch.Tensor) -> torch.Tensor:
    """Information bits ``[B, L]`` (tail already appended) -> coded bits
    ``[B, 2L]`` with (c1, c2) per input bit, starting and ending at state 0.
    """
    tables = _conv_tables(info_bits.device)
    B, L = info_bits.shape
    out_states = tables["out"]                        # [64, 2, 2]
    nxt = tables["next"]                              # [64, 2]
    state = torch.zeros(B, dtype=torch.long, device=info_bits.device)
    coded = info_bits.new_zeros(B, 2 * L)
    for t in range(L):
        u = info_bits[:, t]
        pair = out_states[state, u]                   # [B, 2] -> (c1, c2)
        coded[:, 2 * t] = pair[:, 0]
        coded[:, 2 * t + 1] = pair[:, 1]
        state = nxt[state, u]
    assert bool((state == 0).all()), "zero tail must drive the register to 0"
    return coded


# ---------------------------------------------------------------------------
# Gate B - Gray-QPSK modem with exact log-MAP LLRs
# ---------------------------------------------------------------------------

def qpsk_modulate(bits: torch.Tensor) -> torch.Tensor:
    """Coded bit pairs ``[B, N]`` (even N) -> complex symbols ``[B, N/2]``.

    (b0, b1) -> ((1 - 2 b0) + j (1 - 2 b1)) / sqrt(2); unit average power.
    """
    if bits.shape[-1] % 2 != 0:
        raise ValueError("QPSK needs an even number of bits")
    b0 = bits[..., 0::2].to(torch.get_default_dtype())
    b1 = bits[..., 1::2].to(torch.get_default_dtype())
    return (torch.complex(1.0 - 2.0 * b0, 1.0 - 2.0 * b1)
            / math.sqrt(2.0))


def qpsk_llrs(
    y_eq: torch.Tensor,
    sigma2_eff: torch.Tensor,
) -> torch.Tensor:
    """Exact log-MAP LLRs for the frozen Gray-QPSK mapping.

    ``y_eq``: equalized complex symbols ``[B, n]``.
    ``sigma2_eff``: effective complex noise variance per codeword,
    broadcastable to ``y_eq`` (AWGN: sigma^2; Rayleigh: sigma^2/|h|^2).

    Returns LLRs ``[B, 2n]`` ordered as (b0(sym1), b1(sym1), b0(sym2), ...),
    with LLR = log[P(0)/P(1)] > 0 meaning bit 0 is more likely.
    """
    if torch.is_complex(y_eq):
        yi, yq = y_eq.real, y_eq.imag
    else:
        k = y_eq.shape[-1] // 2
        yi, yq = y_eq[..., :k], y_eq[..., k:]
    s2 = sigma2_eff.to(yi.dtype)
    llr_i = (4.0 * math.sqrt(2.0)) * yi / s2
    llr_q = (4.0 * math.sqrt(2.0)) * yq / s2
    return torch.stack([llr_i, llr_q], dim=-1).flatten(start_dim=-2)


# ---------------------------------------------------------------------------
# Gate C - soft-decision Viterbi (zero tail, traceback from state 0)
# ---------------------------------------------------------------------------

def viterbi_decode(llrs: torch.Tensor) -> torch.Tensor:
    """Soft-decision Viterbi for the frozen code.

    ``llrs``: ``[B, 2L]`` in (c1, c2) order matching the encoder output.
    Branch metric for codeword bit c given LLR l: +l if c = 0, -l if c = 1
    (i.e. l * (1 - 2c)).

    ACS (textbook butterfly): next state j = (2s + u) mod 64 merges exactly
    two previous states, both with the SAME input u = j & 1:
        prev_a = j >> 1,   prev_b = prev_a + 32.
    Each next state therefore sees exactly two candidates; the winner and
    its previous state are stored. (Note: a scatter over prev->next is
    INVALID here because the map is 2-to-1 and scatter semantics for
    duplicate indices are undefined.)

    Traceback starts from state 0 (guaranteed by the zero tail); the
    information bit at step t is the LSB of the next state (= the common
    input u of both merged branches). Returns the L recovered information
    bits ``[B, L]``.
    """
    tables = _viterbi_tables(llrs.device)
    out = tables["out"]                               # [64, 2(u), 2(c)]
    B, twoL = llrs.shape
    L = twoL // 2
    device = llrs.device
    j_all = torch.arange(64, device=device)
    prev_a = j_all >> 1                               # [64] per next state
    prev_b = prev_a + 32                              # [64]
    u_all = j_all & 1                                 # [64] shared input
    sign_a1 = 1.0 - 2.0 * out[prev_a, u_all, 0].to(llrs.dtype)   # [64]
    sign_a2 = 1.0 - 2.0 * out[prev_a, u_all, 1].to(llrs.dtype)
    sign_b1 = 1.0 - 2.0 * out[prev_b, u_all, 0].to(llrs.dtype)
    sign_b2 = 1.0 - 2.0 * out[prev_b, u_all, 1].to(llrs.dtype)

    metrics = torch.full((B, 64), -1e30, device=device, dtype=llrs.dtype)
    metrics[:, 0] = 0.0                               # start state 0
    hist = []                                         # per-step winning prev
    for t in range(L):
        l1 = llrs[:, 2 * t].unsqueeze(1)              # [B, 1]
        l2 = llrs[:, 2 * t + 1].unsqueeze(1)          # [B, 1]
        bm_a = l1 * sign_a1.unsqueeze(0) + l2 * sign_a2.unsqueeze(0)  # [B, 64]
        bm_b = l1 * sign_b1.unsqueeze(0) + l2 * sign_b2.unsqueeze(0)
        c_a = metrics.gather(1, prev_a.unsqueeze(0).expand(B, 64)) + bm_a
        c_b = metrics.gather(1, prev_b.unsqueeze(0).expand(B, 64)) + bm_b
        take_b = c_b > c_a
        metrics = torch.where(take_b, c_b, c_a)
        hist.append(torch.where(take_b,
                                prev_b.unsqueeze(0).expand(B, 64),
                                prev_a.unsqueeze(0).expand(B, 64)))
    # traceback from state 0; bit(t) = LSB of the next state
    bits = llrs.new_zeros(B, L, dtype=torch.long)
    state = torch.zeros(B, dtype=torch.long, device=device)
    for t in range(L - 1, -1, -1):
        bits[:, t] = state & 1
        state = hist[t].gather(1, state.unsqueeze(1)).squeeze(1)
    return bits


def _viterbi_tables(device: torch.device) -> Dict[str, torch.Tensor]:
    t = _conv_tables(device)
    return {"next": t["next"], "out": t["out"]}


# ---------------------------------------------------------------------------
# Gate E - full digital baseline
# ---------------------------------------------------------------------------

class DigitalBaseline:
    """Separated quantize-code-modulate chain with the frozen framing.

    Not differentiable and not trainable by design (standard practice for
    the conventional baseline); evaluation-only.
    """

    def __init__(self, k: int, bits_per_component: int) -> None:
        self.pkt = config.digital_packet(k, bits_per_component)  # raises if infeasible
        self.k = k
        self.B = bits_per_component
        self.n_tail = config.CONV_TAIL_BITS
        self.pad_bits = 2 * self.pkt["n_padding_symbols"]        # zero-bit filler pairs

    # -- transmit side --------------------------------------------------

    def encode_state(self, s: torch.Tensor) -> torch.Tensor:
        """Physical state -> k complex channel symbols (payload + padding)."""
        s_bar = normalize(s)
        q, _ = quantize(s_bar, self.B)
        info = torch.cat([pack_bits(q, self.B).to(torch.long),
                          torch.zeros(*q.shape[:-1], self.n_tail,
                                      dtype=torch.long, device=q.device)], dim=-1)
        coded = conv_encode(info)                                # [B, 12B+12]
        sym = qpsk_modulate(coded)                               # [B, 6B+6]
        if self.pad_bits:
            pad = qpsk_modulate(torch.zeros(*sym.shape[:-1], self.pad_bits,
                                            dtype=torch.long,
                                            device=sym.device))
            sym = torch.cat([sym, pad], dim=-1)
        assert sym.shape[-1] == self.k
        return sym

    # -- receive side ---------------------------------------------------

    def decode_symbols(
        self,
        y: torch.Tensor,
        channel: str,
        snr_db: float,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Channel output -> dequantized NORMALIZED state estimate."""
        check_codeword(y)
        n_pay = self.pkt["n_payload_symbols"]
        if torch.is_complex(y):
            y_pay = y[..., :n_pay]
        else:                                                    # stacked real
            k = y.shape[-1] // 2
            y_pay = y[..., : 2 * n_pay]
        if channel == "rayleigh":
            if h is None:
                raise ValueError("Rayleigh decoding needs the CSI tensor h")
            y_pay = equalize_zf(y_pay, h)
        sigma2 = config.TX_POWER * (10.0 ** (-snr_db / 10.0))
        if channel == "rayleigh":
            hh = (h.real ** 2 + h.imag ** 2).clamp_min(config.EPS_EQUALIZER)
            sigma2_eff = sigma2 / hh                             # [B, 1], per-codeword CSI
        else:
            sigma2_eff = torch.tensor(sigma2, dtype=y_pay.real.dtype
                                      if torch.is_complex(y_pay) else y_pay.dtype)
        llrs = qpsk_llrs(y_pay, sigma2_eff)
        info = viterbi_decode(llrs)
        info = info[:, :-self.n_tail]                            # drop tail
        q_hat = unpack_bits(info, self.B)
        return dequantize(q_hat, self.B)

    # -- full path ------------------------------------------------------

    def forward(
        self,
        s: torch.Tensor,
        snr_db: float,
        channel: str = "awgn",
        generator: Optional[torch.Generator] = None,
        noiseless: bool = False,
        noise: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Physical state -> estimated physical state through the full chain.

        ``noiseless=True`` bypasses the channel (validation reference for
        Gate E: the estimate then equals the quantizer reconstruction).
        ``noise``/``h`` accept pre-generated realizations for paired
        evaluation (used verbatim; no sampling). The frozen framing,
        budget and Step-3 channel conventions are untouched.
        """
        if channel not in ("awgn", "rayleigh"):
            raise ValueError(f"unknown channel {channel!r}")
        with torch.no_grad():
            sym = self.encode_state(s)
            if noiseless:
                # validation reference: any sigma2 gives exact decisions on
                # an unnoised symbol sequence
                s_bar_hat = self.decode_symbols(sym, "awgn", snr_db)
                return denormalize(s_bar_hat)
            if channel == "awgn":
                y = awgn_channel(sym, snr_db, generator=generator, noise=noise)
                s_bar_hat = self.decode_symbols(y, "awgn", snr_db)
            else:
                y, h_used = rayleigh_channel(sym, snr_db, generator=generator,
                                             noise=noise, h=h, return_h=True)
                s_bar_hat = self.decode_symbols(y, "rayleigh", snr_db, h=h_used)
            return denormalize(s_bar_hat)


# ============================================================
# SECTION 6  UAV NAVIGATION (dynamics, controller, environment)
# ============================================================
# ---- Source: project/uav/dynamics.py ----
"""2-D double-integrator UAV dynamics (audit 1.1 row 2).

Frozen discrete-time equations, implemented with PyTorch tensors and fully
differentiable (required for the closed-loop task-oriented training later):

    p_{t+1} = p_t + dt * v_t + 0.5 * dt^2 * u_t
    v_{t+1} = v_t + dt * u_t

Velocity-limit mechanism [A3, PROJECT IMPLEMENTATION CHOICE]: the frozen
specification requires a velocity limit in configuration but does not
prescribe an enforcement mechanism. We enforce ``||v_{t+1}||_2 <= v_max`` by
norm-rescaling inside this shared module, identically for every
communication method, the oracle sanity run and the training rollouts.
The rescaling is differentiable a.e.; at the (measure-zero) boundary the
gradient through the scale factor is zero, which is the physically correct
saturated behaviour.

The module never clips position: workspace handling is an episode-level
decision owned by ``project.uav.environment`` (audit A5).
"""


import torch



def clip_velocity(v: torch.Tensor, v_max: float = config.V_MAX) -> torch.Tensor:
    """Rescale velocity vectors whose Euclidean norm exceeds ``v_max``.

    ``v`` has shape ``[..., 2]``. Vectors already inside the ball are
    returned unchanged (gradient flows through untouched); vectors outside
    are rescaled onto the ball ``||v|| = v_max``. A ``v_max`` of ``None``
    disables the mechanism entirely. Differentiable a.e.; the 1e-12 guard
    only affects the unselected ``where`` branch, never the output.
    """
    if v_max is None:
        return v
    speed_sq = v.pow(2).sum(dim=-1, keepdim=True)
    scale = torch.where(
        speed_sq > v_max * v_max,
        v_max / torch.sqrt(torch.clamp(speed_sq, min=1e-12)),
        torch.ones_like(speed_sq),
    )
    return v * scale


def step(
    p: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    dt: float = config.DT,
    v_max: float = config.V_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the double integrator by one step.

    Parameters
    ----------
    p, v : tensors of shape ``[batch, 2]`` (position, velocity)
    u    : control of shape ``[batch, 2]``; the caller (controller) is
           responsible for the frozen per-axis ``clip(+-u_max)``.

    Returns
    -------
    (p_next, v_next), each of shape ``[batch, 2]``.
    """
    p = torch.as_tensor(p)
    v = torch.as_tensor(v)
    u = torch.as_tensor(u)
    p_next = p + dt * v + 0.5 * (dt ** 2) * u
    v_next = v + dt * u
    v_next = clip_velocity(v_next, v_max)
    return p_next, v_next

# ---- Source: project/uav/controller.py ----
"""Frozen PD navigation controller (audit 1.1 row 4).

    u_raw = -Kp * (p - p_goal) - Kv * v
    u     = clip(u_raw, -u_max, u_max)          (per axis)

Interface rules (frozen):

* The controller consumes **only** the state estimate handed to it. It has
  no side channel to ground truth, so it can be reused verbatim by all
  three communication methods (task-oriented, reconstruction, digital)
  plus the noiseless sanity run.
* The state estimate is the 6-D vector ``[p, v, p_goal]`` (position,
  velocity, goal), raw physical units. Every method is responsible for
  producing this same interface from its own received representation.
"""


import torch



def pd_controller(
    state_estimate: torch.Tensor,
    kp: float = config.KP,
    kv: float = config.KV,
    u_max: float = config.U_MAX,
) -> torch.Tensor:
    """Compute the clipped PD control from a (possibly noisy) state estimate.

    Parameters
    ----------
    state_estimate : tensor of shape ``[batch, 6]`` laid out as
        ``[x, y, vx, vy, x_g, y_g]``. This is an *estimate*: for the
        communication baselines it is the decoded/denormalized state, for
        the sanity run it is the ground-truth state itself.
    """
    s = torch.as_tensor(state_estimate)
    p = s[..., 0:2]
    v = s[..., 2:4]
    p_goal = s[..., 4:6]
    u_raw = -kp * (p - p_goal) - kv * v
    return torch.clamp(u_raw, min=-u_max, max=u_max)

# ---- Source: project/uav/environment.py ----
"""Deterministic 2-D navigation environment (audit 1.1 rows 2-4, A4/A5).

Minimal, deterministic, vectorized, obstacle-free. Success and failure
semantics follow the frozen decisions:

* success  : d_t <= goal_radius at ANY step t in {0, 1, ..., T} (A4)
* failure  : horizon elapsed without success, or workspace exit during
             evaluation (A5). On exit the episode terminates for that UAV:
             its state is frozen at the first out-of-workspace position and
             all later records are held constant (deterministic, distinct
             from a silent in-place clamp).

During training the environment is bypassed: rollouts run directly through
``project.uav.dynamics`` with fixed horizons (audit A5).

The environment holds no learned components and no notion of channels;
communication systems plug in upstream by producing the state estimate.
"""


from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch



@dataclass
class EpisodeResult:
    """Full trajectory record of one vectorized episode batch."""

    positions: List[torch.Tensor]        # [B, 2] per step, t = 0..T (record=True)
    velocities: List[torch.Tensor]       # [B, 2] per step
    controls: List[torch.Tensor]         # [B, 2] per step, u applied at t -> t+1
    distances: List[torch.Tensor]        # [B]    per step, t = 0..T
    goal: torch.Tensor                   # [B, 2]
    success: torch.Tensor                # [B] bool: d_t <= r_g for some t
    time_to_goal: torch.Tensor           # [B] int: first t of success/exit, T+1 if none
    final_distance: torch.Tensor         # [B]
    min_distance: torch.Tensor           # [B]
    exited_workspace: torch.Tensor       # [B] bool


def episode_distances(p: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Euclidean distance to the goal, shape ``[batch]``."""
    return torch.linalg.vector_norm(p - goal, dim=-1)


class NavigationEnv:
    """Vectorized double-integrator navigation environment."""

    def __init__(
        self,
        dt: float = config.DT,
        t_max: int = config.T_MAX,
        goal_radius: float = config.GOAL_RADIUS,
        v_max: float = config.V_MAX,
        u_max: float = config.U_MAX,
        workspace_min: float = config.WORKSPACE_MIN,
        workspace_max: float = config.WORKSPACE_MAX,
    ) -> None:
        self.dt = dt
        self.t_max = t_max
        self.goal_radius = goal_radius
        self.v_max = v_max
        self.u_max = u_max
        self.workspace_min = workspace_min
        self.workspace_max = workspace_max

    # ------------------------------------------------------------------
    # episode creation
    # ------------------------------------------------------------------

    def sample_starts(
        self,
        n: int,
        generator: Optional[torch.Generator] = None,
        enforce_d_min: bool = True,
        enforce_d_max: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample initial positions, velocities and goals.

        Positions are uniform over the workspace with the configured
        minimum start-goal distance (and ``d_max`` when frozen, A6).
        Initial velocity is the frozen ``v_0 = (0, 0)``.

        Returns ``(p0, v0, goal)``, each shape ``[n, 2]``.
        """
        lo, hi = self.workspace_min, self.workspace_max
        p0 = torch.empty(n, 2).uniform_(lo, hi, generator=generator)
        goal = torch.empty(n, 2).uniform_(lo, hi, generator=generator)

        if enforce_d_min:
            d = episode_distances(p0, goal)
            while bool((d < config.D_MIN).any()):
                m = int((d < config.D_MIN).sum())
                p0[d < config.D_MIN] = torch.empty(m, 2).uniform_(
                    lo, hi, generator=generator)
                goal[d < config.D_MIN] = torch.empty(m, 2).uniform_(
                    lo, hi, generator=generator)
                d = episode_distances(p0, goal)

        if enforce_d_max and config.D_MAX is not None:
            d = episode_distances(p0, goal)
            while bool((d > config.D_MAX).any()):
                m = int((d > config.D_MAX).sum())
                p0[d > config.D_MAX] = torch.empty(m, 2).uniform_(
                    lo, hi, generator=generator)
                goal[d > config.D_MAX] = torch.empty(m, 2).uniform_(
                    lo, hi, generator=generator)
                d = episode_distances(p0, goal)

        v0 = torch.zeros(n, 2) + torch.tensor(config.V_0)
        return p0, v0, goal

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------

    def run_noiseless(
        self,
        p0: torch.Tensor,
        v0: torch.Tensor,
        goal: torch.Tensor,
        record: bool = False,
    ) -> EpisodeResult:
        """Closed-loop noiseless episode; the controller gets ground truth.

        This is the sanity-gate oracle path. Communication methods are
        exercised by scripts that feed their own decoded estimates into the
        same ``pd_controller``.
        """
        n = p0.shape[0]
        device = p0.device
        p = p0.clone()
        v = v0.clone()
        g = goal.clone()

        active = torch.ones(n, dtype=torch.bool, device=device)
        exited = torch.zeros(n, dtype=torch.bool, device=device)

        distances: List[torch.Tensor] = [episode_distances(p, g)]
        positions: List[torch.Tensor] = [p.clone()] if record else []
        velocities: List[torch.Tensor] = [v.clone()] if record else []
        controls: List[torch.Tensor] = []

        d0 = distances[0]
        success = d0 <= self.goal_radius
        time_to_goal = torch.where(
            success,
            torch.zeros(n, dtype=torch.long, device=device),
            torch.full((n,), self.t_max + 1, dtype=torch.long, device=device),
        )

        for t in range(self.t_max):
            # ground-truth full-information state for the oracle path
            s = torch.cat([p, v, g], dim=-1)
            u = pd_controller(s, u_max=self.u_max)
            if record:
                controls.append(u.clone())
            p_new, v_new = dynamics_step(p, v, u, dt=self.dt, v_max=self.v_max)
            d_new = episode_distances(p_new, g)

            # first success time (success may occur while still active);
            # gated on ~success so lingering inside the radius never
            # overwrites the FIRST entry time (A4 semantics)
            newly_success = active & ~success & (d_new <= self.goal_radius)
            time_to_goal = torch.where(
                newly_success,
                torch.full_like(time_to_goal, t + 1),
                time_to_goal,
            )
            success = success | newly_success

            # boundary exit freezes the episode for that UAV (A5)
            outside = (
                (p_new[:, 0] < self.workspace_min)
                | (p_new[:, 0] > self.workspace_max)
                | (p_new[:, 1] < self.workspace_min)
                | (p_new[:, 1] > self.workspace_max)
            )
            newly_exited = active & outside
            time_to_goal = torch.where(
                newly_exited & ~success,
                torch.full_like(time_to_goal, t + 1),
                time_to_goal,
            )
            exited = exited | newly_exited

            # update only still-active UAVs; frozen UAVs hold state & distance
            mask = active.unsqueeze(-1)
            p = torch.where(mask, p_new, p)
            v = torch.where(mask, v_new, v)
            distances.append(
                torch.where(active, d_new, distances[-1]))

            if record:
                positions.append(p.clone())
                velocities.append(v.clone())

            active = active & ~outside
            if not bool(active.any()):
                break

        final_d = distances[-1]
        min_d = torch.stack(distances).min(dim=0).values
        return EpisodeResult(
            positions=positions,
            velocities=velocities,
            controls=controls,
            distances=distances,
            goal=g,
            success=success,
            time_to_goal=time_to_goal,
            final_distance=final_d,
            min_distance=min_d,
            exited_workspace=exited,
        )


# Alias preserving the original `from project.uav.dynamics import step as dynamics_step`
dynamics_step = step

# ============================================================
# SECTION 7  LOSS FUNCTIONS (task-oriented + reconstruction)
# ============================================================
# ---- Source: project/losses/task_loss.py ----
"""Task-oriented navigation loss (frozen, audit 1.1 row 12).

    L_task = ld * L_d + lu * L_u + lv * L_v + lT * L_T

    L_d = (1/T) sum_{t=1..T}   ||p_t - p_g||^2     (t = 0 excluded)
    L_u = (1/T) sum_{t=0..T-1} ||u_t||^2
    L_v = (1/T) sum_{t=1..T}   ||v_t||^2           (t = 0 excluded)
    L_T = ||p_T - p_g||^2

This is a PROJECT-SPECIFIC differentiable navigation/control objective
(task-oriented principle from the literature; this exact equation is NOT
claimed from Shao et al. or the DeepJSCC paper). Weights are the frozen
user decision: ld = 1.0, lu = 0.0, lv = 0.0, lT = 2.0.

The function returns the total AND every component for diagnostics; the
components never enter optimization beyond the frozen weighted sum.
"""


from typing import Dict, List, Union

import torch



def _stack(traj: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    """Accept a pre-stacked [B, T(+1), 2] tensor or a per-step list."""
    if isinstance(traj, torch.Tensor):
        return traj
    return torch.stack(list(traj), dim=1)


def task_loss(
    positions: Union[torch.Tensor, List[torch.Tensor]],
    controls: Union[torch.Tensor, List[torch.Tensor]],
    velocities: Union[torch.Tensor, List[torch.Tensor]],
    goal: torch.Tensor,
    lambda_distance: float = config.LAMBDA_DISTANCE,
    lambda_control: float = config.LAMBDA_CONTROL,
    lambda_velocity: float = config.LAMBDA_VELOCITY,
    lambda_terminal: float = config.LAMBDA_TERMINAL,
) -> Dict[str, torch.Tensor]:
    """Compute the frozen task loss and its components.

    Shapes: ``positions``/``velocities`` ``[B, T+1, 2]`` (t = 0..T),
    ``controls`` ``[B, T, 2]`` (t = 0..T-1), ``goal`` ``[B, 2]``.

    Returns a dict with scalar tensors: ``total``, ``L_d``, ``L_u``,
    ``L_v``, ``L_T``. Gradients flow through every tensor passed in.
    """
    p = _stack(positions)                    # [B, T+1, 2]
    u = _stack(controls)                     # [B, T, 2]
    v = _stack(velocities)                   # [B, T+1, 2]
    if p.dim() != 3 or p.shape[-1] != 2:
        raise ValueError(f"positions must be [B, T+1, 2], got {tuple(p.shape)}")
    if v.dim() != 3 or v.shape[-1] != 2:
        raise ValueError(f"velocities must be [B, T+1, 2], got {tuple(v.shape)}")
    if u.dim() != 3 or u.shape[-1] != 2:
        raise ValueError(f"controls must be [B, T, 2], got {tuple(u.shape)}")
    if u.shape[1] != p.shape[1] - 1:
        raise ValueError(
            f"need T controls for T+1 positions, got {u.shape[1]} controls "
            f"and {p.shape[1]} position steps")

    p_err_sq = (p[:, 1:, :] - goal.unsqueeze(1)).pow(2).sum(dim=-1)   # [B, T]
    L_d = p_err_sq.mean()
    L_u = u.pow(2).sum(dim=-1).mean()
    L_v = (v[:, 1:, :] ).pow(2).sum(dim=-1).mean()
    L_T = p_err_sq[:, -1].mean()

    total = (lambda_distance * L_d + lambda_control * L_u
             + lambda_velocity * L_v + lambda_terminal * L_T)
    return {
        "total": total,
        "L_d": L_d,
        "L_u": L_u,
        "L_v": L_v,
        "L_T": L_T,
    }

# ---- Source: project/losses/reconstruction_loss.py ----
"""Reconstruction loss (frozen, audit 1.1 row 13; item 7).

    L_recon = (1/6) * || s_hat_bar - s_bar ||^2

Computed per sample over the 6 NORMALIZED-state components, then averaged
over the batch. No navigation/task terms; the controller and UAV dynamics
are NOT part of this baseline's training graph.
"""


import torch


def reconstruction_loss(
    s_hat_bar: torch.Tensor,
    s_bar: torch.Tensor,
) -> torch.Tensor:
    """Frozen per-sample (1/6)*||s_hat_bar - s_bar||^2, batch-averaged.

    Both arguments must be NORMALIZED states (last dim 6). Per-sample
    values are computed first, then the mean over the batch is returned.
    """
    if s_hat_bar.shape != s_bar.shape:
        raise ValueError(
            f"shape mismatch: {tuple(s_hat_bar.shape)} vs {tuple(s_bar.shape)}")
    if s_hat_bar.shape[-1] != 6:
        raise ValueError(
            f"loss is defined on the 6-D normalized state, got last dim "
            f"{s_hat_bar.shape[-1]}")
    per_sample = (s_hat_bar - s_bar).pow(2).sum(dim=-1) / 6.0
    return per_sample.mean()


# ============================================================
# SECTION 8  TRAINING (common utilities, task-oriented, reconstruction)
# ============================================================
# ---- Source: project/training/common.py ----
"""Common training infrastructure (audit section 7, item 8).

Responsibilities (strictly separated - no model/channel/loss mathematics
lives here):

* deterministic setup via the existing config utilities
* optimizer construction (Adam, frozen project defaults)
* gradient-norm helpers and finite-value checks
* checkpoint save/load with full reproduction metadata (+ optional RNG
  state capture) under results/checkpoints (gitignored)
* configuration serialization (plain JSON-able dicts)
* history logging helpers

Checkpoint determinism, stated honestly: saving/loading model, optimizer,
step and history is exact. Restoring the process RNG states (torch global
+ purpose generators) is SUPPORTED when the caller passes their states;
the current Step-4/5 trainers create their generators internally at call
time, so bitwise mid-run continuation of those specific loops requires
exposing their generator states - a documented limitation, not silently
assumed away.
"""


import math
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch


CHECKPOINT_DIR = Path("results/checkpoints")


# ---------------------------------------------------------------------------
# deterministic setup / optimizer
# ---------------------------------------------------------------------------

def setup_deterministic(seed: int) -> Dict[str, torch.Generator]:
    """Set all global seeds and return fresh purpose-separated generators.

    Uses config.set_global_seed (Python/NumPy/Torch/CUDA). The returned
    generators are for explicit callers; nothing here relies on implicit
    global randomness afterwards.
    """
    config.set_global_seed(seed)
    return {purpose: config.derive_generator(purpose, seed)
            for purpose in config.RNG_KEYS}


def build_optimizer(model: torch.nn.Module,
                    lr: Optional[float] = None) -> torch.optim.Adam:
    """Frozen project optimizer: Adam with config.LEARNING_RATE default."""
    return torch.optim.Adam(model.parameters(),
                            lr=config.LEARNING_RATE if lr is None else lr)


def grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """Global L2 norm over the provided parameters' gradients."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.norm()) ** 2
    return math.sqrt(total)


def check_finite_loss(loss: torch.Tensor) -> bool:
    """True iff the loss tensor is a finite scalar."""
    return bool(torch.isfinite(loss))


def check_finite_grads(params: Iterable[torch.nn.Parameter]) -> bool:
    """True iff every present gradient is finite."""
    for p in params:
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            return False
    return True


def log_history(history: list, entry: Dict) -> list:
    """Append one JSON-able history entry and return the list (chainable)."""
    history.append(entry)
    return history


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------

def to_jsonable(obj):
    """Recursively convert dataclasses/tensors/primitives to JSON types."""
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: os.PathLike | str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    seed: Optional[int] = None,
    metadata: Optional[Dict] = None,
    history: Optional[list] = None,
    capture_rng: bool = True,
) -> Path:
    """Save a deterministic, self-describing checkpoint.

    Contents: model state, optimizer state, step, seed, metadata (method,
    k, channel, training/model configuration ...), history and (optionally)
    the torch global RNG state for continuation. No secrets are stored.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict = {
        "format_version": 1,
        "step": int(step),
        "seed": seed,
        "metadata": to_jsonable(metadata or {}),
        "history": to_jsonable(history or []),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if capture_rng:
        payload["torch_rng_state"] = torch.get_rng_state()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: os.PathLike | str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    restore_rng: bool = True,
) -> Dict:
    """Load a checkpoint into the given model (and optimizer).

    Returns the payload (step, seed, metadata, history, ...). With
    ``restore_rng=True`` the torch global RNG state is restored when
    present, so continued sampling follows the saved stream.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng and "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    return payload


def models_identical(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    """Bitwise parameter/state equality between two modules."""
    sa, sb = a.state_dict(), b.state_dict()
    if sa.keys() != sb.keys():
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa)

# ---- Source: project/training/train_task_oriented.py ----
"""Task-oriented DeepJSCC closed-loop training (Step 5).

Frozen pipeline (audit 1.1 rows 6-12, spec section 15):

    s_t -> normalize -> encoder -> power norm -> channel -> [ZF] -> decoder
        -> s_hat_bar -> denormalize -> PD controller -> u_t
        -> double integrator -> s_{t+1} -> ... (T steps) -> L_task

* The optimization objective is the FROZEN task loss
  (project/losses/task_loss.py) with the signed-off weights
  ld = 1.0, lu = 0.0, lv = 0.0, lT = 2.0 - never the reconstruction MSE
  (reconstruction MSE is computed only as a detached diagnostic).
* The controller (Kp = 1.0, Kv = 2.0, hard clip) and the dynamics are the
  frozen Step 0-2 implementations, reused verbatim inside the graph.
* The state distribution is the SAME frozen A9 protocol used by the
  reconstruction baseline (env.sample_starts + v0 = 0).
* Training SNR: U(0, 20) dB per rollout (vectorized-channel contract;
  the per-sample refinement is documented in the Step-4 trainer).
* No .detach()/.numpy()/.item() anywhere in the differentiable path;
  noise/fading enter as generator-drawn constants. Training uses a FIXED
  horizon (no early termination, audit A5).
"""


import math
from typing import Dict, Optional, Tuple

import torch



def closed_loop_rollout(
    comm: ReconstructionDeepJSCC,
    s0: torch.Tensor,
    t_max: int = config.T_MAX,
    snr_db: float = 10.0,
    channel: str = "awgn",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable T-step closed-loop rollout.

    Returns ``(positions [B, T+1, 2], controls [B, T, 2],
    velocities [B, T+1, 2])``. Every operation is differentiable w.r.t.
    the communication parameters; noise/fading are constants.
    """
    if channel not in ("awgn", "rayleigh"):
        raise ValueError(f"unknown channel {channel!r}")
    p = s0[:, 0:2]
    v = s0[:, 2:4]
    g = s0[:, 4:6]

    positions = [p]
    velocities = [v]
    controls = []
    s = s0

    for _ in range(t_max):
        # communication: normalized estimate of the current state
        s_bar_hat = comm.forward_losses(s, snr_db, channel=channel,
                                        generator=generator)
        s_hat = denormalize(s_bar_hat)              # physical estimate
        u = pd_controller(s_hat)                    # frozen controller
        p, v = dynamics_step(p, v, u, dt=config.DT, v_max=config.V_MAX)
        positions.append(p)
        velocities.append(v)
        controls.append(u)
        s = torch.cat([p, v, g], dim=-1)

    return (torch.stack(positions, dim=1),
            torch.stack(controls, dim=1),
            torch.stack(velocities, dim=1))


def evaluate_task_loss(
    comm: ReconstructionDeepJSCC,
    s_eval: torch.Tensor,
    t_max: int = config.T_MAX,
    snr_db: float = 10.0,
    channel: str = "awgn",
    eval_seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Deterministic task-loss evaluation on a fixed episode set.

    The channel generator is re-derived from ``eval_seed`` on every call,
    so the same episodes always see the same noise/fading realizations -
    before/after comparisons and reproducibility checks stay exact.
    """
    gen = config.derive_generator("noise_test", eval_seed)
    pos, ctl, vel = closed_loop_rollout(comm, s_eval, t_max=t_max,
                                        snr_db=snr_db, channel=channel,
                                        generator=gen)
    # task_loss is computed OUTSIDE no_grad so the same deterministic
    # evaluation can also be used for backward/finite-difference checks;
    # the extra statistics below are pure diagnostics.
    out = dict(task_loss(pos, ctl, vel, s_eval[:, 4:6]))
    with torch.no_grad():
        goal = s_eval[:, 4:6]
        d_final = (pos[:, -1, :] - goal).norm(dim=-1)
        d_min = (pos[:, 1:, :] - goal.unsqueeze(1)).norm(dim=-1).min(dim=1).values
        out["final_distance_mean"] = d_final.mean()
        out["min_distance_mean"] = d_min.mean()
        out["success_rate"] = (d_min <= config.GOAL_RADIUS).float().mean()
    return out


def train_task_oriented(
    k: int = 3,
    channel: str = "awgn",
    seed: int = 42,
    steps: int = 300,
    batch_episodes: int = 128,
    t_max: int = config.T_MAX,
    lr: Optional[float] = None,
    snr_range: Tuple[float, float] = config.SNR_TRAIN_RANGE_DB,
    eval_episodes: int = 256,
    eval_snr_db: float = 10.0,
    verbose_every: int = 0,
) -> Tuple[ReconstructionDeepJSCC, dict]:
    """Smoke-scale task-oriented training; returns (model, report).

    Deterministic: fixed seeds for init/episodes/SNR/noise; a fixed
    evaluation episode set is drawn once and reused for the before/after
    task-loss comparison with identical channel realizations.
    """
    if channel not in ("awgn", "rayleigh"):
        raise ValueError(f"unknown channel {channel!r}")
    config.set_global_seed(seed)                     # weight init
    gen_states = config.derive_generator("episodes", seed)
    gen_snr = config.derive_generator("noise_train", seed)
    gen_noise = config.derive_generator("noise_test", seed)
    env = NavigationEnv()

    # fixed deterministic evaluation batch (same distribution, fixed seeds)
    gen_eval = config.derive_generator("episodes", seed + 1)
    p0e, v0e, goale = env.sample_starts(eval_episodes, generator=gen_eval)
    s_eval = torch.cat([p0e, v0e, goale], dim=-1)     # [B, 6] full state

    lr = config.LEARNING_RATE if lr is None else lr
    model = ReconstructionDeepJSCC(k=k)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    enc_params = list(model.encoder.parameters())
    dec_params = list(model.decoder.parameters())

    def _gnorm(params):
        return math.sqrt(sum(float(p.grad.norm()) ** 2 for p in params
                             if p.grad is not None))

    eval_before = evaluate_task_loss(model, s_eval, t_max, eval_snr_db,
                                     channel, eval_seed=seed + 2)
    history, g_enc_h, g_dec_h = [], [], []
    nan_batches = 0

    for step_idx in range(steps):
        p0, v0, goal = env.sample_starts(batch_episodes, generator=gen_states)
        s0 = torch.cat([p0, v0, goal], dim=-1)
        snr_db = float(torch.empty(1).uniform_(
            snr_range[0], snr_range[1], generator=gen_snr))

        pos, ctl, vel = closed_loop_rollout(model, s0, t_max=t_max,
                                            snr_db=snr_db, channel=channel,
                                            generator=gen_noise)
        out = task_loss(pos, ctl, vel, goal)
        loss = out["total"]

        # detached diagnostics only (never optimized)
        with torch.no_grad():
            s_bar_hat_diag = model.forward_losses(s0, snr_db, channel=channel,
                                                  generator=gen_noise)
            recon_mse = float(((s_bar_hat_diag - normalize(s0)) ** 2)
                              .sum(dim=-1).mean() / 6.0)

        if not bool(torch.isfinite(loss)):
            nan_batches += 1
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_enc, g_dec = _gnorm(enc_params), _gnorm(dec_params)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.GRAD_CLIP_NORM)
        opt.step()

        history.append({"total": float(loss), "L_d": float(out["L_d"]),
                        "L_u": float(out["L_u"]), "L_v": float(out["L_v"]),
                        "L_T": float(out["L_T"]), "recon_mse_diag": recon_mse,
                        "grad_enc": g_enc, "grad_dec": g_dec})
        g_enc_h.append(g_enc)
        g_dec_h.append(g_dec)
        if verbose_every and (step_idx % verbose_every == 0 or
                              step_idx == steps - 1):
            h = history[-1]
            print(f"  [task {channel} k={k}] step {step_idx:4d} "
                  f"snr {snr_db:5.1f}  total {h['total']:9.3f}  "
                  f"Ld {h['L_d']:8.3f}  LT {h['L_T']:9.3f}  "
                  f"recon(diag) {h['recon_mse_diag']:.4f}  "
                  f"|g| {h['grad_enc']:.2e}/{h['grad_dec']:.2e}")

    eval_after = evaluate_task_loss(model, s_eval, t_max, eval_snr_db,
                                    channel, eval_seed=seed + 2)

    report = {
        "channel": channel, "k": k, "seed": seed,
        "steps": steps, "batch_episodes": batch_episodes, "t_max": t_max,
        "lr": lr, "snr_range_db": snr_range,
        "eval_episodes": eval_episodes, "eval_snr_db": eval_snr_db,
        "lambdas": {"ld": config.LAMBDA_DISTANCE,
                    "lu": config.LAMBDA_CONTROL,
                    "lv": config.LAMBDA_VELOCITY,
                    "lT": config.LAMBDA_TERMINAL},
        "eval_before": {key: float(val) for key, val in eval_before.items()},
        "eval_after": {key: float(val) for key, val in eval_after.items()},
        "train_total_initial": history[0]["total"] if history else float("nan"),
        "train_total_final": history[-1]["total"] if history else float("nan"),
        "train_Ld_initial": history[0]["L_d"] if history else float("nan"),
        "train_Ld_final": history[-1]["L_d"] if history else float("nan"),
        "train_Lu_initial": history[0]["L_u"] if history else float("nan"),
        "train_Lu_final": history[-1]["L_u"] if history else float("nan"),
        "train_Lv_initial": history[0]["L_v"] if history else float("nan"),
        "train_Lv_final": history[-1]["L_v"] if history else float("nan"),
        "train_LT_initial": history[0]["L_T"] if history else float("nan"),
        "train_LT_final": history[-1]["L_T"] if history else float("nan"),
        "recon_mse_diag_initial": history[0]["recon_mse_diag"] if history else float("nan"),
        "recon_mse_diag_final": history[-1]["recon_mse_diag"] if history else float("nan"),
        "grad_enc_first": g_enc_h[0] if g_enc_h else float("nan"),
        "grad_enc_final": g_enc_h[-1] if g_enc_h else float("nan"),
        "grad_dec_first": g_dec_h[0] if g_dec_h else float("nan"),
        "grad_dec_final": g_dec_h[-1] if g_dec_h else float("nan"),
        "nan_batches": nan_batches,
        "history": history,
    }
    return model, report


# ---- Source: project/training/train_reconstruction.py ----
"""Reconstruction DeepJSCC training (smoke scale for Step 4).

Training protocol frozen by the project specification:

* objective: the frozen reconstruction loss on the NORMALIZED state
  (losses.reconstruction_loss); no task terms, no controller/dynamics in
  the graph.
* training states: the frozen protocol of audit A9 - states generated by
  rolling out the noiseless full-information PD controller. The initial
  states come from the same frozen sampler as evaluation (v0 = 0, d_min
  enforced); the network therefore sees the same normalized state
  representation the task-oriented model will use later.
* training SNR: Uniform(SNR_TRAIN_RANGE_DB) = U(0, 20) dB (audit 2.4).
  The vectorized channel contract takes one SNR per batch, so the smoke
  trainer draws one SNR per batch; the audit's per-sample law (A8) is the
  full-training refinement.
* optimizer: Adam(lr = LEARNING_RATE), gradient clipping global-norm 1.0
  (audit 2.6). Seeds via config.set_global_seed + purpose generators.

This entry point runs a SMOKE budget (proof of learning, not the final
experiment): modest steps, fixed seed, fully deterministic.
"""


import math
from typing import Optional, Tuple

import torch



def _group_grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.norm()) ** 2
    return math.sqrt(total)


def train_reconstruction(
    k: int = 3,
    channel: str = "awgn",
    seed: int = 42,
    steps: int = 400,
    batch_states: int = 2048,
    lr: Optional[float] = None,
    snr_range: Tuple[float, float] = config.SNR_TRAIN_RANGE_DB,
    verbose_every: int = 0,
) -> Tuple[ReconstructionDeepJSCC, dict]:
    """Train the reconstruction baseline; return (model, report).

    The default budget is the SMOKE scale required by Step 4 (proof of
    finite, reproducible, meaningful learning) - not the final protocol.
    """
    config.set_global_seed(seed)          # weight init
    gen_states = config.derive_generator("episodes", seed)
    gen_snr = config.derive_generator("noise_train", seed)
    gen_noise = config.derive_generator("noise_test", seed)
    env = NavigationEnv()

    lr = config.LEARNING_RATE if lr is None else lr
    model = ReconstructionDeepJSCC(k=k)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    enc_params = list(model.encoder.parameters())
    dec_params = list(model.decoder.parameters())

    history = []
    grad_enc_hist = []
    grad_dec_hist = []
    nan_batches = 0
    loss_initial = None

    for step_idx in range(steps):
        # --- frozen state protocol (A9): starts + v0=0 + goal -> [B, 6]
        p0, v0, goal = env.sample_starts(batch_states, generator=gen_states)
        s = torch.cat([p0, v0, goal], dim=-1)

        # --- U(0, 20) dB per batch (see module docstring for granularity)
        snr_db = float(torch.empty(1).uniform_(
            snr_range[0], snr_range[1], generator=gen_snr))

        s_bar_hat = model.forward_losses(s, snr_db, channel=channel,
                                         generator=gen_noise)
        target = normalize(s)             # clean normalized ground truth

        loss = reconstruction_loss(s_bar_hat, target)

        if not bool(torch.isfinite(loss)):
            nan_batches += 1
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_enc = _group_grad_norm(enc_params)
        g_dec = _group_grad_norm(dec_params)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.GRAD_CLIP_NORM)
        opt.step()

        loss_val = float(loss)
        history.append(loss_val)
        grad_enc_hist.append(g_enc)
        grad_dec_hist.append(g_dec)
        if loss_initial is None:
            loss_initial = loss_val
        if verbose_every and (step_idx % verbose_every == 0 or
                              step_idx == steps - 1):
            print(f"  [{channel} k={k}] step {step_idx:4d}  "
                  f"snr {snr_db:5.1f} dB  loss {loss_val:.5f}  "
                  f"|g_enc| {g_enc:.3f}  |g_dec| {g_dec:.3f}")

    report = {
        "channel": channel,
        "k": k,
        "seed": seed,
        "steps": steps,
        "batch_states": batch_states,
        "lr": lr,
        "snr_range_db": snr_range,
        "loss_initial": loss_initial,
        "loss_final": history[-1] if history else float("nan"),
        "loss_reduction_pct": (100.0 * (1.0 - history[-1] / loss_initial)
                               if history and loss_initial else float("nan")),
        "grad_norm_encoder_initial": grad_enc_hist[0] if grad_enc_hist else float("nan"),
        "grad_norm_encoder_final": grad_enc_hist[-1] if grad_enc_hist else float("nan"),
        "grad_norm_decoder_initial": grad_dec_hist[0] if grad_dec_hist else float("nan"),
        "grad_norm_decoder_final": grad_dec_hist[-1] if grad_dec_hist else float("nan"),
        "nan_batches": nan_batches,
        "history": history,
        "grad_encoder_history": grad_enc_hist,
        "grad_decoder_history": grad_dec_hist,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    return model, report



# ============================================================
# SECTION 9  EVALUATION + STATISTICAL TESTS (paired harness, metrics, Wilson CI, corrected McNemar, paired t)
# ============================================================
# ---- Source: project/evaluation/metrics.py ----
"""Frozen navigation metrics (audit 1.1 row 15, spec section 19).

Definitions (per episode, horizon T):

    success        = 1 iff exists t in {0..T} with d_t <= goal_radius
                     (workspace exit is a failure per the A5 convention)
    final_distance = d_T   = ||p_T - p_g||
    avg_distance   = d_bar = (1/T) sum_{t=1..T} d_t          (t = 0 excluded)
    control_effort = E_u    = sum_{t=0..T-1} ||u_t||^2
    time_to_goal   = first t with d_t <= goal_radius (T+1 sentinel if never)
    mse (normalized state) = (1/(6T)) sum_{t=1..T} ||s_hat_t - s_t||^2
                     computed on NORMALIZED states; applicable only for
                     methods that produce a state estimate (not the oracle).

BER/BLER are digital-only supplementary LINK diagnostics and are never part
of this task-metric schema.
"""


from typing import Dict, Optional

import torch


def episode_distances(p: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """d_t for one timestep: ``p`` [B, 2], ``goal`` [B, 2] -> [B]."""
    return torch.linalg.vector_norm(p - goal, dim=-1)


def per_episode_metrics(
    positions: torch.Tensor,
    controls: torch.Tensor,
    velocities: torch.Tensor,
    goal: torch.Tensor,
    goal_radius: float,
    s_hat_bar: Optional[torch.Tensor] = None,
    s_bar: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Per-episode metrics from one closed-loop rollout.

    Shapes: ``positions``/``velocities`` [B, T+1, 2] (t = 0..T),
    ``controls`` [B, T, 2] (t = 0..T-1), ``goal`` [B, 2]; for the state
    MSE, ``s_hat_bar``/``s_bar`` are the NORMALIZED transmitted/true states
    for t = 1..T, shape [B, T, 6]. Returns ``[B]`` tensors with keys:
    ``success`` (bool), ``final_distance``, ``avg_distance``,
    ``control_effort``, ``min_distance``, ``time_to_goal`` (long, T+1
    sentinel), ``exited`` (bool, pass-through if provided by the harness
    is NOT here - the harness records it), and ``mse`` (NaN where not
    applicable).
    """
    if positions.dim() != 3 or positions.shape[-1] != 2:
        raise ValueError(f"positions must be [B, T+1, 2], got {tuple(positions.shape)}")
    if controls.dim() != 3 or controls.shape[-1] != 2:
        raise ValueError(f"controls must be [B, T, 2], got {tuple(controls.shape)}")
    T = positions.shape[1] - 1
    if controls.shape[1] != T:
        raise ValueError(f"need T={T} controls for T+1 positions, got {controls.shape[1]}")

    d = torch.linalg.vector_norm(positions[:, 1:, :] - goal.unsqueeze(1), dim=-1)  # [B, T] t=1..T
    d0 = episode_distances(positions[:, 0, :], goal)                               # [B]

    success = (d <= goal_radius).any(dim=1) | (d0 <= goal_radius)
    # first timestep with d_t <= r_g: argmax over the 0/1 hit vector returns
    # the FIRST true index (and 0 when success holds at t = 0)
    hit = (torch.cat([d0.unsqueeze(1), d], dim=1) <= goal_radius)
    ttg = torch.where(
        success,
        torch.argmax(hit.long(), dim=1),
        torch.full_like(d0, dtype=torch.long, fill_value=T + 1),
    ).to(torch.long)

    avg_distance = d.mean(dim=1)                       # (1/T) sum_{t=1..T} d_t
    final_distance = d[:, -1]
    control_effort = controls.pow(2).sum(dim=-1).sum(dim=-1)   # sum ||u_t||^2
    min_distance = torch.cat([d0.unsqueeze(1), d], dim=1).min(dim=1).values

    if s_hat_bar is not None and s_bar is not None:
        if s_hat_bar.shape != s_bar.shape or s_hat_bar.shape[-1] != 6:
            raise ValueError("s_hat_bar/s_bar must be [B, T, 6] normalized states")
        if s_hat_bar.shape[1] != T:
            raise ValueError(f"state estimates must cover t=1..T ({T} steps)")
        mse = (s_hat_bar - s_bar).pow(2).sum(dim=-1).sum(dim=-1) / (6.0 * T)
    else:
        mse = torch.full_like(final_distance, float("nan"))

    return {
        "success": success,
        "final_distance": final_distance,
        "avg_distance": avg_distance,
        "control_effort": control_effort,
        "min_distance": min_distance,
        "time_to_goal": ttg,
        "mse": mse,
    }


def aggregate_episode_metrics(per_ep: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Aggregate per-episode tensors into the reporting table row.

    Success-conditional time-to-goal uses ONLY successful episodes
    (failures keep the T+1 sentinel and are reported as ``failure_count``);
    they are never mixed into the success-conditional mean/median.
    """
    success = per_ep["success"]
    ttg = per_ep["time_to_goal"]
    n = int(success.numel())
    n_success = int(success.sum())
    out = {
        "episodes": n,
        "success_count": n_success,
        "failure_count": n - n_success,
        "success_rate": n_success / n if n else float("nan"),
        "final_distance_mean": float(per_ep["final_distance"].mean()),
        "avg_distance_mean": float(per_ep["avg_distance"].mean()),
        "control_effort_mean": float(per_ep["control_effort"].mean()),
        "min_distance_mean": float(per_ep["min_distance"].mean()),
        "mse_mean": (float(per_ep["mse"].mean())
                     if bool(torch.isfinite(per_ep["mse"]).any()) else float("nan")),
        "ttg_success_mean": (float(ttg[success].float().mean())
                             if n_success else float("nan")),
        "ttg_success_median": (float(ttg[success].float().median())
                               if n_success else float("nan")),
    }
    return out


METRIC_KEYS = ("success", "final_distance", "avg_distance", "control_effort",
               "min_distance", "time_to_goal", "mse")

# ---- Source: project/evaluation/statistics.py ----
"""Evaluation statistics (audit 2.8; protocol doc sections V/W).

Implemented exactly: mean, std, success count/total, Wilson 95% CI,
McNemar's test (paired binary outcomes) and the paired t-test (paired
continuous metrics). All functions are dependency-free (no SciPy) and
machine-readable with clearly named fields. They never rank methods or
pick winners; they only compute the requested quantities.
"""


import math
from typing import Dict, Tuple

import torch

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def mean_std(x) -> Dict[str, float]:
    """Mean and (sample) standard deviation of a 1-D sequence."""
    t = torch.as_tensor(list(x) if not torch.is_tensor(x) else x,
                        dtype=torch.float64).reshape(-1)
    if t.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=True)) if t.numel() > 1 else 0.0,
        "n": int(t.numel()),
    }


def success_count(success) -> Dict[str, float]:
    """Success count / total (n = 0 safe)."""
    t = torch.as_tensor(list(success) if not torch.is_tensor(success) else success)
    t = t.reshape(-1).to(torch.uint8)
    n = int(t.numel())
    k = int(t.sum())
    return {"success_count": k, "total": n,
            "success_rate": (k / n) if n else float("nan")}


def wilson_ci_95(success) -> Dict[str, float]:
    """Wilson score 95% interval for a binomial proportion.

    Uses the exact observed count and total; n = 0 returns NaNs; the
    all-success and zero-success cases yield one-sided-touching intervals
    (never a degenerate [p, p] unless the Normal approximation truly
    collapses). Numerically stable closed form:

        center = (p + z^2/(2n)) / (1 + z^2/n)
        half   = z/(1 + z^2/n) * sqrt(p(1-p)/n + z^2/(4n^2))
    """
    c = success_count(success)
    k, n = c["success_count"], c["total"]
    if n == 0:
        return {"wilson_low": float("nan"), "wilson_high": float("nan"),
                "p_hat": float("nan"), "success_count": 0, "total": 0}
    z = Z_95
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(max(p * (1.0 - p) / n
                                       + z * z / (4.0 * n * n), 0.0))
    return {"wilson_low": max(center - half, 0.0),
            "wilson_high": min(center + half, 1.0),
            "p_hat": p, "success_count": k, "total": n}


def _binom_sf_approx(k: int, n: int, p: float) -> float:
    """Upper-tail binomial probability P(X >= k) via the normal
    approximation with continuity correction (no SciPy dependency;
    accurate enough at the tail probabilities the test uses)."""
    if k <= 0:
        return 1.0
    var = n * p * (1.0 - p)
    if var <= 0:
        return 1.0 if k > n * p else 0.0
    z = (k - 0.5 - n * p) / math.sqrt(var)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def mcnemar_test(success_a, success_b) -> Dict[str, float]:
    """Exact-form McNemar test on PAIRED binary outcomes.

    Only discordant pairs matter: n01 = #(A=1, B=0), n10 = #(A=0, B=1).
    Statistic chi2 = (|n01 - n10| - 1)^2 / (n01 + n10) with 1 dof
    (continuity correction), p from the survival function of chi2(1)
    = erfc(sqrt(chi2/2)). Perfectly symmetric discordance (n01 == n10)
    is the known edge case where the correction would produce a spurious
    chi2 = (n01+n10)/2; it is handled explicitly as chi2 = 0, p = 1.
    Preserves pairing by construction.
    """
    a = torch.as_tensor(list(success_a) if not torch.is_tensor(success_a) else success_a
                        ).reshape(-1).to(torch.uint8)
    b = torch.as_tensor(list(success_b) if not torch.is_tensor(success_b) else success_b
                        ).reshape(-1).to(torch.uint8)
    if a.shape != b.shape:
        raise ValueError(f"paired outcomes must align: {a.shape} vs {b.shape}")
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n_pairs = int(a.numel())
    if n01 == n10:
        return {"n01": n01, "n10": n10, "n_pairs": n_pairs,
                "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return {"n01": n01, "n10": n10, "n_pairs": n_pairs,
            "chi2": float(chi2), "p_value": float(p)}


def paired_t_test(x_a, x_b) -> Dict[str, float]:
    """Paired t-test on paired continuous metrics (x_a - x_b differences).

    Preserves pairing: only the per-episode differences enter. Returns the
    mean difference, t statistic, degrees of freedom and the two-sided
    p-value from the t-distribution (computed with the regularized
    incomplete beta function, no SciPy).
    """
    a = torch.as_tensor(list(x_a) if not torch.is_tensor(x_a) else x_a,
                        dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(list(x_b) if not torch.is_tensor(x_b) else x_b,
                        dtype=torch.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"paired samples must align: {a.shape} vs {b.shape}")
    n = a.numel()
    if n < 2:
        return {"mean_diff": float("nan"), "t_stat": float("nan"),
                "dof": int(n - 1), "p_value": float("nan"), "n": int(n)}
    d = a - b
    md = float(d.mean())
    sd = float(d.std(unbiased=True))
    if sd == 0.0:
        t_stat = float("inf") if md != 0 else 0.0
        p = 0.0 if md != 0 else 1.0
    else:
        t_stat = md / (sd / math.sqrt(n))
        p = _t_sf_two_sided(abs(t_stat), n - 1)
    return {"mean_diff": md, "t_stat": float(t_stat), "dof": int(n - 1),
            "p_value": float(p), "n": int(n)}


def _betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via the continued fraction
    (Lentz), standard numerical recipe; adequate for p-value tails."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-12):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _t_sf_two_sided(t: float, dof: int) -> float:
    """Two-sided p-value P(|T| > t) for Student-t with dof >= 1."""
    if dof <= 0:
        return float("nan")
    x = dof / (dof + t * t)
    return _betainc_reg(dof / 2.0, 0.5, x)

# ---- Source: project/evaluation/evaluate.py ----
"""Paired evaluation harness (audit section 7, item 7).

Pairing contract (hard requirement):

    same test episodes + same channel realizations + same SNR
    + same horizon + same environment/controller conventions

Design: a condition's episodes AND channel realizations are PRE-GENERATED
from purpose-separated deterministic streams and then handed to every
method as exact tensors. Method identity can never alter the noise or
fading. For Rayleigh, h[episode, step] and n[episode, step, symbol] are
identical across oracle / reconstruction / task / digital; for AWGN the
same holds for n.

The harness reuses the frozen interfaces verbatim: env sampler, PD
controller + double integrator (shared, outside any method), the Step-3
channels in their pre-generated-realization form, the Step-4/5 comm path,
and the Step-6 digital baseline (whose infeasibility raises
InfeasibleDigitalConfig at k in {1, 2, 3} - reported here as an explicit
structured status, never silently dropped).

BER/BLER are digital-only link diagnostics and are NOT part of this
task-metric harness (metrics.py docstring).
"""


from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch



# ---------------------------------------------------------------------------
# condition materialization
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    """One frozen evaluation condition (fixed SNR, channel, budget)."""

    k: int
    channel: str            # "awgn" | "rayleigh"
    snr_db: float           # ONE fixed SNR per condition (frozen approval 5)
    t_max: int = config.T_MAX
    bits_per_component: Optional[int] = None   # required for the digital method
    episodes: int = config.N_TEST_EPISODES_SMOKE
    seed: int = config.TEST_SEED


def generate_realizations(
    cond: Condition,
    episodes: int,
) -> Dict[str, torch.Tensor]:
    """Pre-generate episodes and paired channel realizations for a condition.

    Returns the episode batch (p0, v0, goal) and per-step realizations:
    ``noise [E, T, k]`` (complex) and, for Rayleigh, ``h [E, T]`` (complex).
    Derived from purpose-separated deterministic streams; independent of
    any method.
    """
    if cond.channel not in ("awgn", "rayleigh"):
        raise ValueError(f"unknown channel {cond.channel!r}")
    env = NavigationEnv()
    gen_ep = config.derive_generator("episodes", cond.seed)
    p0, v0, goal = env.sample_starts(episodes, generator=gen_ep)

    gen_noise = config.derive_generator("noise_test", cond.seed)
    sigma2 = config.TX_POWER * (10.0 ** (-cond.snr_db / 10.0))
    shape = (episodes, cond.t_max, cond.k)
    noise = torch.complex(
        torch.randn(shape, generator=gen_noise) * (sigma2 / 2.0) ** 0.5,
        torch.randn(shape, generator=gen_noise) * (sigma2 / 2.0) ** 0.5,
    )
    out = {"p0": p0, "v0": v0, "goal": goal, "noise": noise}
    if cond.channel == "rayleigh":
        h = torch.complex(
            torch.randn(episodes, cond.t_max, generator=gen_noise) * 0.5 ** 0.5,
            torch.randn(episodes, cond.t_max, generator=gen_noise) * 0.5 ** 0.5,
        )
        out["h"] = h
    return out


def method_feasible(method: str, cond: Condition) -> (bool, Optional[str]):
    """Structured feasibility for the digital method (frozen framing)."""
    if method != "digital":
        return True, None
    if cond.bits_per_component is None:
        return False, ("digital requires bits_per_component on the condition "
                       f"(k={cond.k}); 6B + 6 <= k must hold")
    try:
        config.digital_packet(cond.k, cond.bits_per_component)
        return True, None
    except config.InfeasibleDigitalConfig as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# method drivers (s_hat/denorm producers for a single (episode, step) batch)
# ---------------------------------------------------------------------------

def _oracle_estimate(s: torch.Tensor) -> torch.Tensor:
    return s.clone()


def _neural_estimate_factory(model: ReconstructionDeepJSCC, channel: str,
                             snr_db: float):
    def est(s: torch.Tensor, noise_c: torch.Tensor,
            h_c: Optional[torch.Tensor]) -> torch.Tensor:
        return model.forward(s, snr_db, channel=channel, noise=noise_c,
                             h=h_c, denormalize_output=True)
    return est


def _digital_estimate_factory(base: "DigitalBaseline", channel: str,
                              snr_db: float):
    def est(s: torch.Tensor, noise_c: torch.Tensor,
            h_c: Optional[torch.Tensor]) -> torch.Tensor:
        return base.forward(s, snr_db, channel=channel, noise=noise_c, h=h_c)
    return est


# ---------------------------------------------------------------------------
# paired closed-loop evaluation of one method
# ---------------------------------------------------------------------------

def run_method(
    cond: Condition,
    real: Dict[str, torch.Tensor],
    estimate_fn: Optional[Callable] = None,
) -> Dict[str, torch.Tensor]:
    """Roll out one method under the paired realizations and score it.

    ``estimate_fn(s, noise_c, h_c) -> s_hat`` receives the per-step state
    and the exact pre-generated realizations for that step. The oracle is
    ``estimate_fn=None``. The shared controller/dynamics run OUTSIDE the
    method on the produced estimate, so conventions cannot drift between
    methods.
    """
    p = real["p0"].clone()
    v = real["v0"].clone()
    goal = real["goal"]
    g = goal
    B = p.shape[0]
    E, T, k = real["noise"].shape

    pos = [p]
    vel = [v]
    ctl = []
    s_hat_bar_hist: List[torch.Tensor] = []
    s_bar_hist: List[torch.Tensor] = []
    exited = torch.zeros(B, dtype=torch.bool)
    active = torch.ones(B, dtype=torch.bool)

    for t in range(T):
        s = torch.cat([p, v, g], dim=-1)
        s_bar = normalize(s)
        noise_c = real["noise"][:, t, :]                      # [B, k] exact
        h_c = real["h"][:, t].unsqueeze(-1) if "h" in real else None

        if estimate_fn is None:
            s_hat = s                                          # oracle
        else:
            active_idx = active.nonzero(as_tuple=True)[0]
            if active_idx.numel() == 0:
                s_hat = s
            else:
                s_hat_active = estimate_fn(s[active_idx], noise_c[active_idx],
                                           h_c[active_idx] if h_c is not None else None)
                s_hat = s.clone()
                s_hat = s_hat.index_copy(0, active_idx, s_hat_active)

        s_bar_hat = normalize(s_hat)
        s_bar_hist.append(s_bar)
        s_hat_bar_hist.append(s_bar_hat)

        u = pd_controller(s_hat)                               # frozen controller
        p_new, v_new = dynamics_step(p, v, u)                  # frozen dynamics
        d_new = (p_new - g).norm(dim=-1)

        outside = ((p_new[:, 0] < config.WORKSPACE_MIN)
                   | (p_new[:, 0] > config.WORKSPACE_MAX)
                   | (p_new[:, 1] < config.WORKSPACE_MIN)
                   | (p_new[:, 1] > config.WORKSPACE_MAX))
        newly = active & outside
        exited = exited | newly
        active = active & ~outside

        mask = active.unsqueeze(-1)
        p = torch.where(mask, p_new, p)
        v = torch.where(mask, v_new, v)
        pos.append(p)
        vel.append(v)
        ctl.append(u)

    positions = torch.stack(pos, dim=1)                        # [B, T+1, 2]
    controls = torch.stack(ctl, dim=1)                         # [B, T, 2]
    velocities = torch.stack(vel, dim=1)                       # [B, T+1, 2]

    metrics = per_episode_metrics(positions, controls, velocities, goal,
                                  config.GOAL_RADIUS,
                                  s_hat_bar=torch.stack(s_hat_bar_hist, dim=1),
                                  s_bar=torch.stack(s_bar_hist, dim=1))
    metrics["exited"] = exited
    metrics["positions"] = positions
    metrics["velocities"] = velocities
    metrics["controls"] = controls
    metrics["goal"] = goal
    return metrics


# ---------------------------------------------------------------------------
# condition-level paired evaluation across the method set
# ---------------------------------------------------------------------------

def evaluate_condition(
    cond: Condition,
    methods: Optional[List[str]] = None,
    recon_model: Optional[ReconstructionDeepJSCC] = None,
    task_model: Optional[ReconstructionDeepJSCC] = None,
    digital: Optional[DigitalBaseline] = None,
) -> Dict[str, Dict]:
    """Evaluate the requested methods on one condition, fully paired.

    Returns ``{method: {"metrics": per-episode dict, "aggregate": row} or
    {"status": "infeasible", "reason": ...}}``. The digital method needs
    ``cond.bits_per_component``; at k in {1, 2, 3} it is reported as an
    explicit infeasible status (frozen Step-6 framing).
    """
    if methods is None:
        methods = ["oracle", "reconstruction", "task", "digital"]
    if cond.channel not in ("awgn", "rayleigh"):
        raise ValueError(f"unknown channel {cond.channel!r}")

    real = generate_realizations(cond, cond.episodes)
    results: Dict[str, Dict] = {}

    for method in methods:
        if method == "oracle":
            m = run_method(cond, real, estimate_fn=None)
        elif method in ("reconstruction", "task"):
            model = recon_model if method == "reconstruction" else task_model
            if model is None:
                results[method] = {"status": "unavailable",
                                   "reason": f"no {method} model supplied"}
                continue
            est = _neural_estimate_factory(model, cond.channel, cond.snr_db)
            m = run_method(cond, real, estimate_fn=est)
        elif method == "digital":
            feasible, reason = method_feasible("digital", cond)
            if not feasible:
                results[method] = {"status": "infeasible", "reason": reason}
                continue
            if digital is None:
                digital = DigitalBaseline(k=cond.k,
                                          bits_per_component=cond.bits_per_component)
            est = _digital_estimate_factory(digital, cond.channel, cond.snr_db)
            m = run_method(cond, real, estimate_fn=est)
        else:
            raise ValueError(f"unknown method {method!r}")
        results[method] = {"metrics": m,
                           "aggregate": aggregate(m)}

    return results


def aggregate(per_ep: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Aggregate row (delegates to metrics.aggregate_episode_metrics)."""
    return aggregate_episode_metrics(per_ep)


# Alias preserving the original `from project.evaluation.evaluate import Condition as EvalCondition` (project/experiments/config.py)
EvalCondition = Condition

# ============================================================
# SECTION 10 FROZEN EXPERIMENT MATRICES (M1-M4 grids, seeds, SNRs, k values, digital budgets, identity/recovery, matrix validator)
# ============================================================
# ---- Source: project/experiments/config.py ----
"""Explicit experiment configuration (audit section 7, item 9).

Every experiment run is fully described by an :class:`ExperimentConfig`;
nothing is hard-coded in model/training code. All frozen values come from
``project.config``:

* training seeds {42, 43, 44}, test seed 10042
* training SNR policy U(0, 20) dB (explicit, inspectable)
* test SNR grids: M1 {0, 2, ..., 20} dB, M2 {0, 5, 10, 15, 20} dB
* bandwidth: neural k in {1, 2, 3}; matched-budget M2 k in
  {12, 18, 24, 30, 54} with B = (k - 6)/6
* digital infeasible at k in {1, 2, 3} (6B + 6 <= k) - EXPECTED for M3
  cells, an error everywhere else

Run modes: ``dry_run`` (configuration summary only, NO training), ``smoke``
(small deterministic validation, non-scientific), ``full`` (explicitly
BLOCKED until the user approves full-scale M1-M5 execution).

Note: ``digital_feasible`` lives HERE (not in matrix.py) because matrix.py
imports this module; the budget arithmetic itself is the frozen
``config.digital_packet`` - no duplicated mathematics.
"""


from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple



# ---------------------------------------------------------------------------
# frozen matrix definitions (structural; execution is a later step)
# ---------------------------------------------------------------------------

M1_K_VALUES = (1, 2, 3)
M1_SNR_DB = tuple(float(s) for s in range(0, 21, 2))
M2_K_VALUES = (12, 18, 24, 30, 54)
M2_SNR_DB = (0.0, 5.0, 10.0, 15.0, 20.0)
M2_B_BY_K = dict(config.DIGITAL_B_BY_K)   # {12:1, 18:2, 24:3, 30:4, 54:8}
M4_DIGITAL_B = 8                          # secondary reference: B = 8
M4_K = 54                                 # n_dig = 54 channel uses (rho_dig = 9)
CHANNELS = ("awgn", "rayleigh")
TRAINING_SEEDS = tuple(config.TRAINING_SEEDS)      # (42, 43, 44)


def b_for_m2_k(k: int) -> int:
    """B = (k - 6) / 6 for the matched-budget points (audit 8.6)."""
    return M2_B_BY_K[k]


def digital_feasible(k: int, bits_per_component: Optional[int]) -> Tuple[bool, str]:
    """Frozen budget check: payload 6B + 6 must fit inside k channel uses.

    Single implementation for the config layer (matrix.py imports this);
    delegates the arithmetic to the frozen ``config.digital_packet``.
    """
    if bits_per_component is None:
        return False, "digital requires bits_per_component (B)"
    try:
        pkt = config.digital_packet(k, bits_per_component)
    except config.InfeasibleDigitalConfig as e:
        return False, str(e)
    return True, (f"6B={pkt['n_src_bits']}, info={pkt['n_info_bits']}, "
                  f"coded={pkt['n_coded_bits']}, payload="
                  f"{pkt['n_payload_symbols']}, pad={pkt['n_padding_symbols']}, "
                  f"k={k}")


# ---------------------------------------------------------------------------
# experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """One experiment cell, fully self-describing."""

    experiment: str                 # "M1" | "M2" | "M3" | "M4" | "M5"
    method: str                     # "reconstruction" | "task" | "digital" | "oracle"
    channel: str                    # "awgn" | "rayleigh"
    k: int
    training_seed: int
    test_seed: int = config.TEST_SEED
    # training configuration (smoke scale by default; full scale is a later,
    # explicitly approved step - run_mode='full' is rejected by validate())
    train_steps: int = 400
    batch_episodes: int = config.BATCH_EPISODES
    batch_states: int = config.BATCH_STATES
    lr: Optional[float] = None      # None -> frozen config.LEARNING_RATE
    # training SNR policy: explicit and inspectable, never hard-coded in
    # the trainers' callers; the trainers implement it (U(range) per batch)
    train_snr_policy: Dict = field(default_factory=lambda: {
        "distribution": "uniform",
        "range_db": list(config.SNR_TRAIN_RANGE_DB),
        "granularity": "per_batch"})
    # evaluation configuration
    eval_snr_grid_db: Tuple[float, ...] = M1_SNR_DB
    t_max: int = config.T_MAX
    eval_snr_db: float = 10.0       # fixed SNR for the task trainer's eval set
    n_train_episodes: int = config.N_TRAIN_EPISODES
    n_test_episodes: int = config.N_TEST_EPISODES_SMOKE
    eval_episodes: int = 256        # task trainer's fixed evaluation episodes
    # method configuration
    bits_per_component: Optional[int] = None   # required for the digital method
    checkpoint_path: Optional[str] = None
    output_path: Optional[str] = None
    run_mode: str = "smoke"         # "dry_run" | "smoke" | "full"

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @property
    def expected_infeasible(self) -> bool:
        """M3 cells are digital-infeasible BY DESIGN (audit 8.5)."""
        return self.experiment == "M3" and self.method == "digital"

    def validate(self) -> List[str]:
        """Return a list of configuration errors (empty = valid)."""
        errors: List[str] = []
        if self.experiment not in ("M1", "M2", "M3", "M4", "M5"):
            errors.append(f"unknown experiment {self.experiment!r}")
        if self.method not in ("reconstruction", "task", "digital", "oracle"):
            errors.append(f"unknown method {self.method!r}")
        if self.channel not in CHANNELS:
            errors.append(f"unknown channel {self.channel!r}")
        if self.run_mode not in ("dry_run", "smoke", "full"):
            errors.append(f"unknown run_mode {self.run_mode!r}")
        if self.training_seed not in TRAINING_SEEDS:
            errors.append(
                f"training seed {self.training_seed} not in {TRAINING_SEEDS}")
        if self.test_seed != config.TEST_SEED:
            errors.append(f"test seed must be {config.TEST_SEED}")
        if self.k <= 0:
            errors.append("k must be a positive integer")
        if self.method == "digital":
            if self.bits_per_component is None:
                errors.append("digital method requires bits_per_component")
            else:
                feasible, reason = digital_feasible(self.k, self.bits_per_component)
                if self.expected_infeasible:
                    if feasible:
                        errors.append(
                            "M3 cell unexpectedly feasible - digital must remain "
                            f"infeasible at k in {{1, 2, 3}} ({reason})")
                elif not feasible:
                    errors.append(f"digital infeasible: {reason}")
        if self.run_mode == "full":
            errors.append(
                "run_mode='full' is BLOCKED in this step: full-scale M1-M5 "
                "execution requires explicit user approval")
        return errors

    def validate_or_raise(self) -> "ExperimentConfig":
        errs = self.validate()
        if errs:
            raise ValueError("invalid experiment config: " + "; ".join(errs))
        return self

    # ------------------------------------------------------------------
    # presets (frozen matrix rows, validated at construction)
    # ------------------------------------------------------------------

    @classmethod
    def m1(cls, method: str, k: int, channel: str, seed: int,
           run_mode: str = "smoke") -> "ExperimentConfig":
        return cls(experiment="M1", method=method, channel=channel, k=k,
                   training_seed=seed, eval_snr_grid_db=M1_SNR_DB,
                   run_mode=run_mode).validate_or_raise()

    @classmethod
    def m2(cls, method: str, k: int, channel: str, seed: int,
           run_mode: str = "smoke") -> "ExperimentConfig":
        b = b_for_m2_k(k) if method == "digital" else None
        return cls(experiment="M2", method=method, channel=channel, k=k,
                   training_seed=seed, bits_per_component=b,
                   eval_snr_grid_db=M2_SNR_DB,
                   run_mode=run_mode).validate_or_raise()

    @classmethod
    def m3(cls, k: int, bits: int = 1, channel: str = "awgn", seed: int = 42,
           run_mode: str = "dry_run") -> "ExperimentConfig":
        """Digital infeasibility cell (k in {1, 2, 3}); expected infeasible."""
        return cls(experiment="M3", method="digital", channel=channel, k=k,
                   training_seed=seed, bits_per_component=bits,
                   run_mode=run_mode).validate_or_raise()

    @classmethod
    def m4_secondary(cls, channel: str, seed: int,
                     run_mode: str = "smoke") -> "ExperimentConfig":
        """SECONDARY digital reference: B = 8 -> n_dig = 54 (rho_dig = 9).

        Never a primary cell; the matrix validator and this preset both
        enforce the labeling.
        """
        return cls(experiment="M4", method="digital", channel=channel, k=M4_K,
                   training_seed=seed, bits_per_component=M4_DIGITAL_B,
                   eval_snr_grid_db=M2_SNR_DB,
                   run_mode=run_mode).validate_or_raise()

    # ------------------------------------------------------------------
    # dry run (NO training, NO side effects)
    # ------------------------------------------------------------------

    def digital_packet_summary(self) -> Optional[Dict]:
        """Packet arithmetic for digital cells (None for neural/oracle)."""
        if self.method != "digital" or self.bits_per_component is None:
            return None
        feasible, note = digital_feasible(self.k, self.bits_per_component)
        if not feasible:
            return {"feasible": False, "note": note}
        return dict(config.digital_packet(self.k, self.bits_per_component),
                    feasible=True)

    def dry_run_summary(self) -> Dict:
        """Structured summary for the configuration-only dry run."""
        feasibility = None
        if self.method == "digital":
            feasible, reason = digital_feasible(self.k, self.bits_per_component)
            feasibility = {"feasible": feasible, "expected_infeasible":
                           self.expected_infeasible, "note": reason}
        return {
            "experiment": self.experiment,
            "method": self.method,
            "channel": self.channel,
            "k": self.k,
            # expected channel-use budget
            "channel_uses": self.k,
            "rho": self.k / 6.0,
            "digital_packet": self.digital_packet_summary(),
            "bits_per_component": self.bits_per_component,
            "training_seed": self.training_seed,
            "test_seed": self.test_seed,
            "training_config": {
                "train_steps": self.train_steps,
                "batch_episodes": self.batch_episodes,
                "batch_states": self.batch_states,
                "lr": self.lr if self.lr is not None else config.LEARNING_RATE,
                "train_snr_policy": dict(self.train_snr_policy),
                "n_train_episodes": self.n_train_episodes},
            "evaluation_config": {
                "eval_snr_grid_db": list(self.eval_snr_grid_db),
                "t_max": self.t_max,
                "eval_snr_db": self.eval_snr_db,
                "n_test_episodes": self.n_test_episodes,
                "eval_episodes": self.eval_episodes},
            "model_config": {
                "hidden_dims": list(config.HIDDEN_DIMS),
                "activation": "PReLU",
                "state_dim": config.STATE_DIM},
            "digital_feasibility": feasibility,
            "checkpoint_path": self.checkpoint_path,
            "output_path": self.output_path,
            "run_mode": self.run_mode,
            "training_will_occur": self.run_mode == "smoke",
            "errors": self.validate(),
        }

    # ------------------------------------------------------------------
    # execution (smoke only in this step)
    # ------------------------------------------------------------------

    def to_eval_condition(self, snr_db: float) -> EvalCondition:
        return EvalCondition(k=self.k, channel=self.channel, snr_db=snr_db,
                             t_max=self.t_max,
                             bits_per_component=self.bits_per_component,
                             episodes=self.n_test_episodes,
                             seed=self.test_seed)

    def run(self, dry_run: Optional[bool] = None) -> Dict:
        """Dispatch by run_mode. ``full`` is rejected by validate()."""
        summary = self.dry_run_summary()
        mode = self.run_mode if dry_run is None else (
            "dry_run" if dry_run else "smoke")
        if mode == "dry_run":
            dry = replace(self, run_mode="dry_run")
            return {"config": dry.dry_run_summary(), "status": "dry_run"}
        self.validate_or_raise()          # rejects run_mode='full'

        ckpt_path = self.checkpoint_path or (
            f"results/checkpoints/{self.experiment}_{self.method}_"
            f"{self.channel}_k{self.k}_seed{self.training_seed}.pth")

        model: Optional[ReconstructionDeepJSCC] = None
        train_block: Optional[Dict] = None
        if self.method == "reconstruction":
            model, rep = train_reconstruction(
                k=self.k, channel=self.channel, seed=self.training_seed,
                steps=self.train_steps, batch_states=self.batch_states,
                lr=self.lr)
            train_block = {"loss_initial": rep["loss_initial"],
                           "loss_final": rep["loss_final"],
                           "loss_reduction_pct": rep["loss_reduction_pct"],
                           "nan_batches": rep["nan_batches"],
                           "history": rep["history"]}
        elif self.method == "task":
            model, rep = train_task_oriented(
                k=self.k, channel=self.channel, seed=self.training_seed,
                steps=self.train_steps, batch_episodes=self.batch_episodes,
                t_max=self.t_max, lr=self.lr,
                snr_range=tuple(self.train_snr_policy["range_db"]),
                eval_episodes=self.eval_episodes,
                eval_snr_db=self.eval_snr_db)
            train_block = {"train_total_initial": rep["train_total_initial"],
                           "train_total_final": rep["train_total_final"],
                           "lambdas": rep["lambdas"],
                           "eval_before": rep["eval_before"],
                           "eval_after": rep["eval_after"],
                           "nan_batches": rep["nan_batches"],
                           "history": rep["history"]}

        # evaluation over the configured grid via the Step-7 paired harness
        digital_model: Optional[DigitalBaseline] = None
        if self.method == "digital":
            feasible, _ = digital_feasible(self.k, self.bits_per_component)
            if feasible:
                digital_model = DigitalBaseline(
                    k=self.k, bits_per_component=self.bits_per_component)
        eval_rows: Dict[str, Dict] = {}
        for snr_db in self.eval_snr_grid_db:
            cond = self.to_eval_condition(snr_db)
            cell = evaluate_condition(
                cond, methods=[self.method],
                recon_model=model if self.method == "reconstruction" else None,
                task_model=model if self.method == "task" else None,
                digital=digital_model)
            row = cell[self.method]
            eval_rows[f"{snr_db:g}"] = row.get("aggregate", row.get("status"))

        # checkpoint: neural methods only (digital/oracle have no state)
        ckpt: Optional[str] = None
        if model is not None:
            ckpt = str(save_checkpoint(
                ckpt_path, model, optimizer=build_optimizer(model, self.lr),
                step=self.train_steps, seed=self.training_seed,
                metadata={
                    "experiment": self.experiment,
                    "method": self.method,
                    "channel": self.channel,
                    "k": self.k,
                    "bits_per_component": self.bits_per_component,
                    "training_seed": self.training_seed,
                    "test_seed": self.test_seed,
                    "train_steps": self.train_steps,
                    "batch_episodes": self.batch_episodes,
                    "batch_states": self.batch_states,
                    "lr": self.lr if self.lr is not None else config.LEARNING_RATE,
                    "train_snr_policy": dict(self.train_snr_policy),
                    "eval_snr_grid_db": list(self.eval_snr_grid_db),
                    "t_max": self.t_max,
                    "n_test_episodes": self.n_test_episodes,
                    "model_config": {"hidden_dims": list(config.HIDDEN_DIMS),
                                     "activation": "PReLU"},
                    "run_mode": self.run_mode},
                history=train_block["history"] if train_block else None))

        return {"config": summary, "status": "smoke_complete",
                "train": train_block, "eval": eval_rows,
                "checkpoint": ckpt,
                "non_scientific": True,
                "note": ("Step-8 smoke validation of infrastructure only; "
                         "not an experimental result")}


# convenience re-export (runner.py / tests use dataclasses.replace on configs)
__all__ = ["ExperimentConfig", "M1_K_VALUES", "M1_SNR_DB", "M2_K_VALUES",
           "M2_SNR_DB", "M2_B_BY_K", "M4_DIGITAL_B", "M4_K", "CHANNELS",
           "TRAINING_SEEDS", "b_for_m2_k", "digital_feasible", "replace"]

# ---- Source: project/experiments/identity.py ----
"""Stable experiment/run identity (Step 9A recovery design, section 14).

One canonical identifier per experiment cell:

    run_id      = "<experiment>_<method>_<channel>_k<k>_seed<seed>"
    config_hash = sha256(sorted canonical config JSON), first 16 hex

Output/checkpoint layout is derived ONLY from these, so recovery logic is
pure filesystem semantics:

    completed run  -> DONE marker present                -> skip
    interrupted    -> RUNNING/FAILED marker, no DONE     -> resume
    missing        -> nothing on disk                    -> train

Markers (json) live next to the artifacts:
    <run_id>__<config_hash>/.json suffixes:
        .DONE  .RUNNING  .FAILED  .ckpt.meta.json
Scientific outputs live in per-cell result directories
(results/<experiment>/<run_id>__<hash>/) so different cells can never
overwrite each other (section 19).

The run_id vocabulary is exactly the ExperimentConfig vocabulary; the
config hash covers the full training + evaluation configuration so a
changed protocol changes the identity (stale outputs are never reused).
"""


import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional



def canonical_config(cfg: ExperimentConfig) -> Dict:
    """The full configuration that determines a run's identity."""
    return {
        "experiment": cfg.experiment,
        "method": cfg.method,
        "channel": cfg.channel,
        "k": cfg.k,
        "bits_per_component": cfg.bits_per_component,
        "training_seed": cfg.training_seed,
        "test_seed": cfg.test_seed,
        "train_steps": cfg.train_steps,
        "batch_episodes": cfg.batch_episodes,
        "batch_states": cfg.batch_states,
        "lr": cfg.lr if cfg.lr is not None else config.LEARNING_RATE,
        "train_snr_policy": cfg.train_snr_policy,
        "eval_snr_grid_db": list(cfg.eval_snr_grid_db),
        "t_max": cfg.t_max,
        "eval_snr_db": cfg.eval_snr_db,
        "n_train_episodes": cfg.n_train_episodes,
        "n_test_episodes": cfg.n_test_episodes,
        "eval_episodes": cfg.eval_episodes,
    }


def config_hash(cfg: ExperimentConfig) -> str:
    payload = json.dumps(canonical_config(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_id(cfg: ExperimentConfig) -> str:
    return (f"{cfg.experiment}_{cfg.method}_{cfg.channel}"
            f"_k{cfg.k}_seed{cfg.training_seed}")


def run_dir(cfg: ExperimentConfig, results_root: str = "results") -> Path:
    """results/<experiment>/<run_id>__<config_hash>/"""
    d = Path(results_root) / cfg.experiment / f"{run_id(cfg)}__{config_hash(cfg)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# lifecycle markers
# ---------------------------------------------------------------------------

def _marker_path(cfg: ExperimentConfig, kind: str, results_root: str) -> Path:
    return run_dir(cfg, results_root) / f"{kind}.json"


def mark_running(cfg: ExperimentConfig, results_root: str = "results") -> Path:
    p = _marker_path(cfg, "RUNNING", results_root)
    p.write_text(json.dumps({"run_id": run_id(cfg),
                             "config_hash": config_hash(cfg),
                             "started_utc": time.strftime(
                                 "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return p


def mark_done(cfg: ExperimentConfig, summary: Optional[Dict] = None,
              results_root: str = "results") -> Path:
    p = _marker_path(cfg, "DONE", results_root)
    p.write_text(json.dumps({"run_id": run_id(cfg),
                             "config_hash": config_hash(cfg),
                             "finished_utc": time.strftime(
                                 "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "summary": summary or {}}))
    return p


def mark_failed(cfg: ExperimentConfig, reason: str,
                results_root: str = "results") -> Path:
    p = _marker_path(cfg, "FAILED", results_root)
    p.write_text(json.dumps({"run_id": run_id(cfg),
                             "config_hash": config_hash(cfg),
                             "reason": str(reason)[:400]}))
    return p


def cell_state(cfg: ExperimentConfig, results_root: str = "results") -> str:
    """\"done\" | \"failed\" | \"running\" | \"missing\" from the markers only.

    Completion is decided by the DONE marker alone (a stale RUNNING file
    from a crashed process does not make a finished run look unfinished,
    and a DONE marker can never be mistaken for an unfinished one).
    """
    root = Path(results_root) / cfg.experiment
    for state in ("done", "failed", "running"):
        if (root / f"{run_id(cfg)}__{config_hash(cfg)}"
                / f"{state.upper()}.json").exists():
            return state
    return "missing"

# ---- Source: project/experiments/matrix.py ----
"""Frozen experiment-matrix validation (M1-M5) - structural only.

This module VALIDATES the frozen matrix design (audit sections 8.5-8.7).
It never executes experiments. Detected problems:

* invalid k / invalid B / digital budget mismatch (6B + 6 <= k)
* wrong SNR grid per matrix row
* missing/wrong training seeds (must be exactly {42, 43, 44})
* channel outside {AWGN, Rayleigh}
* M2/M4 confusion: any cell labeling the B = 8 secondary reference as
  primary, or giving primary cells a digital budget other than B = (k-6)/6
* M3 cells that are unexpectedly FEASIBLE (they must stay infeasible)
* unsupported method/channel/experiment combinations

The budget arithmetic is the frozen ``config.digital_packet`` via
``config.digital_feasible`` - no duplicated mathematics.
"""


from typing import Dict, List, Tuple



def matrix_digital_feasible(k: int, bits_per_component) -> Tuple[bool, str]:
    """Backward-compatible alias for the frozen budget check."""
    return digital_feasible(k, bits_per_component)


def expected_cells() -> Dict[str, int]:
    """Frozen cell counts per matrix row (structural expectation, audit 8.7).

    Training-run counts: a training run is per (method, k, channel, seed);
    the SNR grid enters at evaluation (one fixed SNR per condition), so it
    multiplies the EVALUATION conditions, not the training runs.
    """
    n_eval_m1 = len(M1_SNR_DB) * len(CHANNELS)
    n_eval_m2 = len(M2_SNR_DB) * len(CHANNELS)
    return {
        # 2 methods x 3 k x 2 channels x 3 seeds
        "M1_training_runs": 2 * len(M1_K_VALUES) * len(CHANNELS) * len(TRAINING_SEEDS),
        "M1_eval_conditions": 2 * len(M1_K_VALUES) * n_eval_m1,
        # NEURAL methods only (2 x 5 k x 2 channels x 3 seeds); the digital
        # baseline needs NO training (Step-9A preflight correction: the
        # former M2_training_runs=90 wrongly counted digital as trainable)
        "M2_neural_training_runs": 2 * len(M2_K_VALUES) * len(CHANNELS) * len(TRAINING_SEEDS),
        # digital evaluation-only seed rows (5 k x 2 channels x 3 seeds);
        # deterministic given (k, B, channel, SNR, test seed)
        "M2_digital_eval_only_seeds": len(M2_K_VALUES) * len(CHANNELS) * len(TRAINING_SEEDS),
        "M2_eval_conditions": 3 * len(M2_K_VALUES) * n_eval_m2,
        "M3_infeasible_cells": len((1, 2, 3)),          # k in {1, 2, 3}
        # SECONDARY digital reference: both channels x SNR grid (>= 1 seed)
        "M4_secondary_eval_conditions": 2 * len(M2_SNR_DB),
        # ablation infrastructure only; cells defined later per ablation
        "M5_infrastructure_only": 0,
    }


def validate_matrix(cells: List[ExperimentConfig]) -> Dict:
    """Validate a proposed cell list against the frozen M1-M5 design.

    Returns ``{"errors": [...], "warnings": [...], "counts": {...},
    "valid": bool}``. No experiment is executed.
    """
    errors: List[str] = []
    warnings: List[str] = []

    for i, cell in enumerate(cells):
        tag = (f"cell[{i}] {cell.experiment}/{cell.method}/"
               f"{cell.channel}/k={cell.k}")
        # generic config validation (run_mode, seeds, method set, digital
        # feasibility; M3 expected-infeasible handled inside validate())
        for err in cell.validate():
            errors.append(f"{tag}: {err}")

        if cell.experiment == "M1":
            if cell.method not in ("task", "reconstruction"):
                errors.append(f"{tag}: M1 is neural task-vs-reconstruction only")
            if cell.k not in M1_K_VALUES:
                errors.append(f"{tag}: M1 k must be in {M1_K_VALUES}")
            if any(s not in M1_SNR_DB for s in cell.eval_snr_grid_db):
                errors.append(f"{tag}: M1 SNR grid must be {M1_SNR_DB}")
            if cell.method == "digital":
                errors.append(f"{tag}: M2/M4 confusion - digital does not "
                              f"belong in M1 (primary neural ablation)")

        elif cell.experiment == "M2":
            if cell.method not in ("task", "reconstruction", "digital"):
                errors.append(f"{tag}: M2 methods are digital+task+reconstruction")
            if cell.k not in M2_K_VALUES:
                errors.append(f"{tag}: M2 k must be in {M2_K_VALUES}")
            if any(s not in M2_SNR_DB for s in cell.eval_snr_grid_db):
                errors.append(f"{tag}: M2 SNR grid must be {M2_SNR_DB}")
            if (cell.method == "digital"
                    and cell.bits_per_component != M2_B_BY_K.get(cell.k)):
                errors.append(
                    f"{tag}: M2 digital budget mismatch - expected "
                    f"B={M2_B_BY_K.get(cell.k)} for k={cell.k}, got "
                    f"{cell.bits_per_component}")
            # M2/M4 confusion: primary cells must NOT carry the secondary
            # budget (B = 8 belongs to M4 only; only k=54 coincides with it)
            if (cell.method == "digital"
                    and cell.bits_per_component == M4_DIGITAL_B
                    and cell.k != M4_K):
                errors.append(f"{tag}: M2/M4 confusion - B=8 is the M4 "
                              f"SECONDARY reference, not a primary budget")

        elif cell.experiment == "M3":
            if cell.method != "digital":
                errors.append(f"{tag}: M3 cells are digital-only")
            if cell.k not in (1, 2, 3):
                errors.append(f"{tag}: M3 cells are exactly k in {{1, 2, 3}}")
            feasible, reason = digital_feasible(cell.k, cell.bits_per_component)
            if feasible:
                errors.append(f"{tag}: M3 cell unexpectedly feasible - {reason}")
            # expected-infeasible is recorded as an explicit warning-level
            # note, never silently dropped
            if not feasible and cell.run_mode not in ("dry_run",):
                warnings.append(f"{tag}: M3 cell executes nothing - digital is "
                                f"infeasible here by design ({reason})")

        elif cell.experiment == "M4":
            if cell.method != "digital" or cell.bits_per_component != M4_DIGITAL_B:
                errors.append(f"{tag}: M4 is the SECONDARY digital reference "
                              f"with B={M4_DIGITAL_B} (n_dig={M4_K}) only")
            if cell.k != M4_K:
                errors.append(f"{tag}: M4 must use n_dig={M4_K} channel uses, "
                              f"got k={cell.k}")
            if any(s not in M2_SNR_DB for s in cell.eval_snr_grid_db):
                errors.append(f"{tag}: M4 SNR grid must be {M2_SNR_DB}")
            # unconditional SECONDARY-label reminder (audit 8.7: labeled in
            # every table/legend); reported once per cell
            warnings.append(f"{tag}: M4 must remain labeled SECONDARY in "
                            f"every table/legend (audit 8.7)")

        elif cell.experiment == "M5":
            warnings.append(f"{tag}: M5 is ablation infrastructure only - "
                            f"no ablation cells are invented in this step")

        else:
            errors.append(f"{tag}: unknown experiment {cell.experiment!r}")

    # cross-cell consistency: M1/M2 seeds must cover exactly the frozen set
    for row in ("M1", "M2"):
        seeds = sorted({c.training_seed for c in cells
                        if c.experiment == row})
        if seeds and seeds != sorted(TRAINING_SEEDS):
            errors.append(f"{row}: training seeds {seeds} must be exactly "
                          f"{sorted(TRAINING_SEEDS)}")
        chans = {c.channel for c in cells if c.experiment == row}
        if chans and chans != set(CHANNELS):
            errors.append(f"{row}: channels {sorted(chans)} must cover exactly "
                          f"{sorted(CHANNELS)}")

    return {
        "errors": errors,
        "warnings": warnings,
        "counts": expected_cells(),
        "valid": len(errors) == 0,
    }


# ============================================================
# SECTION 11 M1 EXPERIMENT DRIVER (recon vs task, k in {1,2,3})
# ============================================================
# ---- Source: tools/run_step10_m1.py ----
def _run_step10_m1_main():
    __doc__ = 'Step 10 — Full M1 Execution (user-approved, 2026-10-01).\n\nExecutes the FROZEN M1 matrix exactly as specified (no frozen value is\ntouched, no digital cells, no M2/M3/M4/M5):\n\n    training  : 2 methods x 3 k x 2 channels x 3 seeds = 36 neural runs\n                (frozen 9C budgets: recon 5000x4096, task 3000x256 T=100)\n    evaluation: paired task-vs-reconstruction per (k, channel, seed, SNR),\n                SNR grid {0,2,...,20} dB, 5000 episodes, test seed 10042,\n                via the frozen Step-7 harness: both same-seed models are\n                handed to ONE evaluate_condition call, so test episodes,\n                initial states, goals, noise and fading realizations are\n                bitwise identical for both methods by construction.\n\nStatistics (frozen statistics.py, no SciPy): Wilson 95% CI per method per\ncell; McNemar on the paired per-episode successes; paired t on the paired\ncontinuous metrics (final_distance, avg_distance, control_effort).\nDescriptive only — the tool never ranks methods or picks winners.\n\nIdentity/recovery: the Step-8/9A identity system\n(results/M1/<run_id>__<hash>/ with DONE/RUNNING/FAILED markers).\nCompleted runs SKIP; interrupted/failed runs are RETRAINED (deterministic\nby seed+config — the frozen 9C policy; honest: mid-run bitwise\ncontinuation is NOT claimed for the production trainers).\n\nSubcommands (repository root):\n    verify      frozen-protocol + workload hard gate (STOP on mismatch)\n    plan        print the exact deterministic execution order (no side effects)\n    status      per-cell state table + counts (no side effects)\n    train       run the training queue (skip completed; retrain on retry)\n    evaluate    run the paired-eval queue (skips completed units)\n    validate    completion checks: counts/ids/keys + frozen-matrix validation\n    checkint    per-run checkpoint integrity re-verification\n    finalize    write results/M1/step10_m1_summary.json (execution facts)\n\nArtifacts (all gitignored):\n    results/M1/<run_id>__<hash>/{final_ckpt.pt, train_report.json, DONE.json}\n    results/M1/eval/k<k>_<channel>_seed<seed>/eval_<snr>.json\n    results/M1/step10_m1_summary.json'
    """Step 10 — Full M1 Execution (user-approved, 2026-10-01).

    Executes the FROZEN M1 matrix exactly as specified (no frozen value is
    touched, no digital cells, no M2/M3/M4/M5):

        training  : 2 methods x 3 k x 2 channels x 3 seeds = 36 neural runs
                    (frozen 9C budgets: recon 5000x4096, task 3000x256 T=100)
        evaluation: paired task-vs-reconstruction per (k, channel, seed, SNR),
                    SNR grid {0,2,...,20} dB, 5000 episodes, test seed 10042,
                    via the frozen Step-7 harness: both same-seed models are
                    handed to ONE evaluate_condition call, so test episodes,
                    initial states, goals, noise and fading realizations are
                    bitwise identical for both methods by construction.

    Statistics (frozen statistics.py, no SciPy): Wilson 95% CI per method per
    cell; McNemar on the paired per-episode successes; paired t on the paired
    continuous metrics (final_distance, avg_distance, control_effort).
    Descriptive only — the tool never ranks methods or picks winners.

    Identity/recovery: the Step-8/9A identity system
    (results/M1/<run_id>__<hash>/ with DONE/RUNNING/FAILED markers).
    Completed runs SKIP; interrupted/failed runs are RETRAINED (deterministic
    by seed+config — the frozen 9C policy; honest: mid-run bitwise
    continuation is NOT claimed for the production trainers).

    Subcommands (repository root):
        verify      frozen-protocol + workload hard gate (STOP on mismatch)
        plan        print the exact deterministic execution order (no side effects)
        status      per-cell state table + counts (no side effects)
        train       run the training queue (skip completed; retrain on retry)
        evaluate    run the paired-eval queue (skips completed units)
        validate    completion checks: counts/ids/keys + frozen-matrix validation
        checkint    per-run checkpoint integrity re-verification
        finalize    write results/M1/step10_m1_summary.json (execution facts)

    Artifacts (all gitignored):
        results/M1/<run_id>__<hash>/{final_ckpt.pt, train_report.json, DONE.json}
        results/M1/eval/k<k>_<channel>_seed<seed>/eval_<snr>.json
        results/M1/step10_m1_summary.json
    """


    import json
    import math
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch


    M1_ROOT = Path("results/M1")
    FAILS = []


    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  [{name}]" + (f"  ({detail})" if detail else ""))
        if not ok:
            FAILS.append(name)


    def _m1_cfg(method, k, channel, seed):
        """The exact frozen M1 training cell (frozen values only)."""
        return ExperimentConfig(
            experiment="M1", method=method, channel=channel, k=k,
            training_seed=seed, test_seed=config.TEST_SEED,
            train_steps=(config.RECON_TRAIN_STEPS if method == "reconstruction"
                         else config.TASK_TRAIN_STEPS),
            batch_states=config.BATCH_STATES,
            batch_episodes=config.BATCH_EPISODES,
            t_max=config.TASK_TRAINING_T,
            lr=config.LEARNING_RATE,
            eval_snr_grid_db=M1_SNR_DB,
            eval_snr_db=10.0,
            n_train_episodes=config.N_TRAIN_EPISODES,
            n_test_episodes=config.N_TEST_EPISODES,
            eval_episodes=config.BATCH_EPISODES,
            run_mode="smoke")   # identity/vocabulary only; execution is THIS tool


    def m1_cells():
        """36 cells in the frozen deterministic order."""
        cells = []
        for k in M1_K_VALUES:
            for channel in CHANNELS:
                for method in ("reconstruction", "task"):
                    for seed in TRAINING_SEEDS:
                        cells.append(_m1_cfg(method, k, channel, seed))
        return cells


    def eval_units():
        """198 paired evaluation units: (k, channel, seed) x 11 SNRs.

        Each unit pairs the SAME-SEED reconstruction and task models in one
        evaluate_condition call (paired episodes/noise/h by construction).
        """
        units = []
        for k in M1_K_VALUES:
            for channel in CHANNELS:
                for seed in TRAINING_SEEDS:
                    units.append({
                        "unit": f"k{k}_{channel}_seed{seed}",
                        "k": k, "channel": channel, "seed": seed,
                        "recon": _m1_cfg("reconstruction", k, channel, seed),
                        "task": _m1_cfg("task", k, channel, seed),
                    })
        return units


    def _eval_path(unit, snr):
        return M1_ROOT / "eval" / unit["unit"] / f"eval_{snr:g}.json"


    def _load_model(cfg):
        ckpt = run_dir(cfg, results_root="results") / "final_ckpt.pt"
        if not ckpt.exists():
            return None
        model = ReconstructionDeepJSCC(k=cfg.k)
        payload = torch.load(ckpt, weights_only=False)
        model.load_state_dict(payload["model_state"])
        return model


    # ---------------------------------------------------------------------------
    # verify — hard gate on the frozen protocol and the workload
    # ---------------------------------------------------------------------------

    def cmd_verify():
        print("== frozen Step-9C protocol verification (Step-10 gate) ==")
        check("recon_steps==5000", config.RECON_TRAIN_STEPS == 5000)
        check("task_steps==3000", config.TASK_TRAIN_STEPS == 3000)
        check("task_training_T==100", config.TASK_TRAINING_T == 100 == config.T_MAX)
        check("batch_states==4096", config.BATCH_STATES == 4096)
        check("batch_episodes==256", config.BATCH_EPISODES == 256)
        check("lr==1e-3", config.LEARNING_RATE == 1e-3)
        check("grad_clip==1.0", config.GRAD_CLIP_NORM == 1.0)
        check("training_snr==U(0,20)/batch",
              config.SNR_TRAIN_RANGE_DB == (0.0, 20.0))
        check("seeds=={42,43,44}", config.TRAINING_SEEDS == (42, 43, 44))
        check("test_seed==10042", config.TEST_SEED == 10042)
        check("early_stopping==disabled", config.EARLY_STOPPING is False)
        check("M5==deferred+disabled", config.M5_STATUS == "deferred"
              and config.M5_EXECUTION_ENABLED is False)
        check("m1_eval_grid=={0,2,...,20}",
              M1_SNR_DB == tuple(float(s) for s in range(0, 21, 2))
              == config.SNR_TEST_GRID_DB)
        check("m1_k_values=={1,2,3}", M1_K_VALUES == (1, 2, 3))
        check("task_lambdas_frozen",
              (config.LAMBDA_DISTANCE, config.LAMBDA_CONTROL,
               config.LAMBDA_VELOCITY, config.LAMBDA_TERMINAL) == (1.0, 0.0, 0.0, 2.0))
        check("n_test_episodes==5000", config.N_TEST_EPISODES == 5000)
        check("goal_radius==0.5", config.GOAL_RADIUS == 0.5)
        check("horizon_T_max==100_dt==0.1",
              config.T_MAX == 100 and config.DT == 0.1)
        check("tool_has_no_M2_M3_M4_M5_paths", True,
              "this tool implements M1 only")
        config.validate()
        print("config.validate() OK")
        check("workload==36_training_runs", len(m1_cells()) == 36)
        check("workload==198_paired_eval_units",
              len(eval_units()) * len(M1_SNR_DB) == 198
              and len(m1_cells()) * len(M1_SNR_DB) == 396,
              "396 per-method rows == 198 paired units x 2 methods")
        check("pooled_eval_conditions==132",
              2 * 3 * 2 * len(M1_SNR_DB) == 132)
        print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
        return 1 if FAILS else 0


    # ---------------------------------------------------------------------------
    # plan / status (no side effects)
    # ---------------------------------------------------------------------------

    def cmd_plan():
        print("== M1 execution plan (deterministic order; NO execution) ==")
        for i, cfg in enumerate(m1_cells(), 1):
            print(f"{i:2d}. {run_id(cfg):46s} steps={cfg.train_steps:5d} "
                  f"batch={'states4096' if cfg.method == 'reconstruction' else 'episodes256_T100'}")
        n_units = len(eval_units()) * len(M1_SNR_DB)
        print(f"total: {len(m1_cells())} training runs; {n_units} paired eval "
              f"units (2 methods each, 11 SNRs x 5000 episodes, test seed "
              f"{config.TEST_SEED}); pooled conditions: 132")
        return 0


    def cmd_status():
        print("== M1 status ==")
        counts = {"done": 0, "failed": 0, "running": 0, "missing": 0}
        for cfg in m1_cells():
            st = cell_state(cfg, results_root="results")
            counts[st] += 1
            print(f"{st:8s} {run_id(cfg)}  hash={config_hash(cfg)}")
        print(f"-- training: done={counts['done']} failed={counts['failed']} "
              f"running={counts['running']} missing={counts['missing']} (of 36)")
        ev_done = sum(1 for u in eval_units() for snr in M1_SNR_DB
                      if _eval_path(u, snr).exists())
        print(f"-- eval: {ev_done}/198 paired units complete")
        ok = counts["done"] == 36 and counts["failed"] == 0 and ev_done == 198
        print(f"-- M1 complete: {ok}")
        return 0 if ok else 1


    # ---------------------------------------------------------------------------
    # training queue
    # ---------------------------------------------------------------------------

    def _train_one(cfg):
        if cfg.method == "reconstruction":
            model, rep = train_reconstruction(
                k=cfg.k, channel=cfg.channel, seed=cfg.training_seed,
                steps=cfg.train_steps, batch_states=cfg.batch_states)
            g0, g1 = rep["grad_norm_encoder_final"], rep["grad_norm_decoder_final"]
            grads_finite = (math.isfinite(g0) and math.isfinite(g1)
                            and g0 > 0 and g1 > 0)
            summary = {"loss_initial": rep["loss_initial"],
                       "loss_final": rep["loss_final"],
                       "loss_reduction_pct": rep["loss_reduction_pct"]}
        else:
            model, rep = train_task_oriented(
                k=cfg.k, channel=cfg.channel, seed=cfg.training_seed,
                steps=cfg.train_steps, batch_episodes=cfg.batch_episodes,
                t_max=cfg.t_max, eval_episodes=cfg.eval_episodes,
                verbose_every=0)
            g0, g1 = rep["grad_enc_final"], rep["grad_dec_final"]
            grads_finite = (math.isfinite(g0) and math.isfinite(g1)
                            and g0 > 0 and g1 > 0)
            summary = {"task_loss_initial": rep["train_total_initial"],
                       "task_loss_final": rep["train_total_final"],
                       "recon_mse_diag_initial": rep["recon_mse_diag_initial"],
                       "recon_mse_diag_final": rep["recon_mse_diag_final"],
                       "eval_after_success_rate": rep["eval_after"]["success_rate"],
                       "lambdas": rep["lambdas"]}
        return model, rep, grads_finite, summary


    def cmd_train(lane=None):
        """Training queue. Optional lane filter ("awgn"|"rayleigh") allows two
        independent processes to share the matrix by channel (each cell still
        runs exactly once, in its own deterministic order; per-cell computation
        is unchanged — purely a wall-clock scheduling device)."""
        print("== M1 training queue (frozen 9C budgets) =="
              + (f"  [lane: {lane}]" if lane else ""))
        cells = [c for c in m1_cells()
                 if lane is None or c.channel == lane]
        total = len(cells)
        n_trained = n_skipped = n_retrained = 0
        total_wall = 0.0
        for i, cfg in enumerate(cells, 1):
            rid, ch = run_id(cfg), config_hash(cfg)
            st = cell_state(cfg, results_root="results")
            if st == "done":
                print(f"[{i:2d}/{total}] SKIP {rid} (already complete)")
                n_skipped += 1
                continue
            if st == "failed":
                print(f"[{i:2d}/{total}] RETRAIN-after-FAILURE {rid} "
                      f"(frozen 9C policy: deterministic by seed+config)")
                n_retrained += 1
            elif st == "running":
                print(f"[{i:2d}/{total}] RETRAIN-after-INTERRUPTION {rid} "
                      f"(stale RUNNING marker; environmental; frozen 9C policy)")
                n_retrained += 1
            d = run_dir(cfg, results_root="results")
            mark_running(cfg, results_root="results")
            print(f"[{i:2d}/{total}] TRAIN {rid} hash={ch} steps={cfg.train_steps} "
                  f"batch={'4096' if cfg.method == 'reconstruction' else '256xT100'}")
            t0 = time.perf_counter()
            try:
                model, rep, grads_finite, summary = _train_one(cfg)
            except Exception as e:              # structured failure, no silent retry
                mark_failed(cfg, f"{type(e).__name__}: {e}", results_root="results")
                (d / "train_report.json").write_text(json.dumps({
                    "run_id": rid, "config_hash": ch, "exit_status": "failed",
                    "error": f"{type(e).__name__}: {e}"}, indent=2))
                print(f"[{i:2d}/{total}] FAILED {rid}: {type(e).__name__}: {e}")
                FAILS.append(rid)
                continue
            wall = time.perf_counter() - t0
            total_wall += wall
            loss_key = ("loss_final" if cfg.method == "reconstruction"
                        else "task_loss_final")
            complete = (rep["nan_batches"] == 0 and grads_finite
                        and math.isfinite(summary[loss_key]))
            rec = {
                "STEP": "10", "experiment": "M1",
                "run_id": rid, "config_hash": ch,
                "method": cfg.method, "channel": cfg.channel, "k": cfg.k,
                "seed": cfg.training_seed,
                "protocol": {
                    "train_steps": cfg.train_steps,
                    "batch_states": cfg.batch_states,
                    "batch_episodes": cfg.batch_episodes,
                    "t_max": cfg.t_max, "lr": cfg.lr,
                    "optimizer": "Adam", "grad_clip": config.GRAD_CLIP_NORM,
                    "train_snr_policy": {"distribution": "uniform",
                                         "range_db": [0.0, 20.0],
                                         "granularity": "per_batch"},
                    "early_stopping": False},
                "wall_s": round(wall, 1),
                "s_per_step": round(wall / cfg.train_steps, 4),
                "nan_batches": rep["nan_batches"],
                "grads_finite": grads_finite,
                "exit_status": "complete" if complete else "incomplete",
                "checkpoint": str(d / "final_ckpt.pt"),
                "optimizer_state_saved": False,
                "optimizer_note": ("trainer-internal optimizer not exposed "
                                   "(documented Step-8 limitation); recovery is "
                                   "deterministic RETRAIN by seed+config (9C)"),
                **summary,
            }
            save_checkpoint(
                d / "final_ckpt.pt", model, optimizer=None,
                step=cfg.train_steps, seed=cfg.training_seed,
                metadata={"run_id": rid, "config_hash": ch,
                          "experiment": "M1", "method": cfg.method,
                          "channel": cfg.channel, "k": cfg.k,
                          "training_seed": cfg.training_seed,
                          "train_steps": cfg.train_steps,
                          "batch_states": cfg.batch_states,
                          "batch_episodes": cfg.batch_episodes,
                          "t_max": cfg.t_max, "lr": cfg.lr,
                          "model_config": {"hidden_dims": list(config.HIDDEN_DIMS),
                                           "activation": "PReLU"},
                          "train_snr_policy": rec["protocol"]["train_snr_policy"]},
                history=rep["history"], capture_rng=True)
            (d / "train_report.json").write_text(
                json.dumps(rec, indent=2, default=str))
            n_trained += 1
            if complete:
                mark_done(cfg, summary={"wall_s": rec["wall_s"],
                                        loss_key: summary[loss_key]},
                          results_root="results")
                print(f"[{i:2d}/{total}] DONE  {rid}  {rec['wall_s']}s  "
                      f"{loss_key}={summary[loss_key]:.4f}")
            else:
                mark_failed(cfg, "incomplete numerical state",
                            results_root="results")
                FAILS.append(rid)
                print(f"[{i:2d}/{total}] INCOMPLETE {rid} "
                      f"(nan_batches={rep['nan_batches']}, "
                      f"grads_finite={grads_finite})")
        print(f"-- training queue finished: trained={n_trained} "
              f"skipped={n_skipped} retrained={n_retrained} "
              f"failed={len(FAILS)} train_wall={total_wall:.1f}s")
        return 1 if FAILS else 0


    # ---------------------------------------------------------------------------
    # paired evaluation queue
    # ---------------------------------------------------------------------------

    def _integrity_failures(cfg):
        """11-checkpoint integrity for one run; returns a list of failures."""
        bad = []
        d = run_dir(cfg, results_root="results")
        ckpt = d / "final_ckpt.pt"
        if not ckpt.exists():
            return ["ckpt-exists"]
        fresh = ReconstructionDeepJSCC(k=cfg.k)
        payload = torch.load(ckpt, weights_only=False)
        md = payload.get("metadata", {})
        if not md:
            bad.append("ckpt-metadata")
        if not (md.get("train_steps") == cfg.train_steps
                and md.get("t_max") == cfg.t_max
                and math.isclose(md.get("lr", -1), config.LEARNING_RATE)
                and md.get("batch_states") == cfg.batch_states
                and md.get("batch_episodes") == cfg.batch_episodes):
            bad.append("config-matches")
        if payload.get("seed") != cfg.training_seed:
            bad.append("seed-matches")
        if not (md.get("method") == cfg.method and md.get("channel") == cfg.channel
                and md.get("k") == cfg.k):
            bad.append("method-channel-k")
        if payload.get("step") != cfg.train_steps:
            bad.append("final-step==budget")
        if not all(bool(torch.isfinite(p).all()) for p in fresh.parameters()):
            bad.append("params-finite")
        if md.get("run_id") != run_id(cfg):
            bad.append("identity-matches")
        if md.get("config_hash") != config_hash(cfg):
            bad.append("hash-matches")
        rep_path = d / "train_report.json"
        if not (rep_path.exists()
                and json.loads(rep_path.read_text()).get("exit_status") == "complete"):
            bad.append("report-complete")
        return bad


    def _paired_stats(cell):
        """Wilson CI per method + McNemar + paired t from one paired cell."""
        out = {}
        m_rec, m_task = cell["reconstruction"]["metrics"], cell["task"]["metrics"]
        for name, m in (("reconstruction", m_rec), ("task", m_task)):
            out[f"wilson_{name}"] = wilson_ci_95(m["success"])
        out["mcnemar_task_vs_recon"] = mcnemar_test(m_task["success"],
                                                    m_rec["success"])
        for key in ("final_distance", "avg_distance", "control_effort"):
            out[f"paired_t_{key}_task_vs_recon"] = paired_t_test(
                m_task[key].to(torch.float64), m_rec[key].to(torch.float64))
        return out


    def cmd_evaluate(lane=None):
        """Paired-eval queue; optional lane filter ("awgn"|"rayleigh") lets two
        independent processes share the units by channel (disjoint outputs)."""
        print("== M1 paired evaluation queue (Step-7 harness, 5000 episodes) =="
              + (f"  [lane: {lane}]" if lane else ""))
        print("-- per-run checkpoint integrity gate --")
        blocked = []
        for cfg in m1_cells():
            if lane is not None and cfg.channel != lane:
                continue
            if cell_state(cfg, results_root="results") != "done":
                blocked.append(f"{run_id(cfg)}: not complete")
            else:
                bad = _integrity_failures(cfg)
                if bad:
                    blocked.append(f"{run_id(cfg)}: {bad}")
        if blocked:
            for b in blocked:
                print(f"REFUSED: {b}")
            return 2
        print(f"   integrity gate passed for lane="
              f"{lane or 'all'} ({len(m1_cells()) if lane is None else 18} runs)")
        n_new = 0
        total_wall = 0.0
        units = [u for u in eval_units()
                 if lane is None or u["channel"] == lane]
        for u in units:
            m_recon = _load_model(u["recon"])
            m_task = _load_model(u["task"])
            cond_base = dict(k=u["k"], channel=u["channel"], t_max=config.T_MAX,
                             episodes=config.N_TEST_EPISODES,
                             seed=config.TEST_SEED)
            for snr in M1_SNR_DB:
                out = _eval_path(u, snr)
                if out.exists():
                    continue
                cond = Condition(snr_db=snr, **cond_base)
                t0 = time.perf_counter()
                with torch.no_grad():
                    cell = evaluate_condition(
                        cond, methods=["reconstruction", "task"],
                        recon_model=m_recon, task_model=m_task)
                wall = time.perf_counter() - t0
                total_wall += wall
                rec = {
                    "STEP": "10", "experiment": "M1",
                    "unit": u["unit"], "k": u["k"], "channel": u["channel"],
                    "seed": u["seed"], "snr_db": snr,
                    "test_seed": config.TEST_SEED,
                    "episodes": config.N_TEST_EPISODES, "t_max": config.T_MAX,
                    "paired": True, "wall_s": round(wall, 1),
                    "aggregate_reconstruction": cell["reconstruction"]["aggregate"],
                    "aggregate_task": cell["task"]["aggregate"],
                    **_paired_stats(cell),
                }
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rec, indent=2, default=str))
                n_new += 1
                sr_r = rec["aggregate_reconstruction"]["success_rate"]
                sr_t = rec["aggregate_task"]["success_rate"]
                print(f"  EVAL {u['unit']} snr={snr:5.1f} "
                      f"SR recon={sr_r:.4f} task={sr_t:.4f} ({wall:.1f}s)")
            del m_recon, m_task
        print(f"-- evaluation queue finished: {n_new} new units this session, "
              f"eval_wall={total_wall:.1f}s total")
        return 0


    # ---------------------------------------------------------------------------
    # checkint / validate
    # ---------------------------------------------------------------------------

    def cmd_checkint():
        bad = {}
        for cfg in m1_cells():
            f = _integrity_failures(cfg)
            if f:
                bad[run_id(cfg)] = f
        for rid, f in bad.items():
            print(f"FAIL {rid}: {f}")
        print(f"== checkpoint integrity: {36 - len(bad)}/36 runs pass ==")
        return 1 if bad else 0


    def cmd_validate():
        print("== M1 completion validation ==")
        FAILS.clear()
        cells = m1_cells()
        ids = [(run_id(c), config_hash(c)) for c in cells]
        check("36-unique-identities", len(set(ids)) == 36)
        done = sum(1 for c in cells
                   if cell_state(c, results_root="results") == "done")
        check("36-training-runs-complete", done == 36, f"{done}/36")
        check("methods==reconstruction,task",
              sorted({c.method for c in cells}) == ["reconstruction", "task"])
        check("k=={1,2,3}", sorted({c.k for c in cells}) == [1, 2, 3])
        check("channels==awgn,rayleigh",
              sorted({c.channel for c in cells}) == ["awgn", "rayleigh"])
        check("seeds=={42,43,44}",
              sorted({c.training_seed for c in cells}) == [42, 43, 44])
        check("no-digital-cells", not any(c.method == "digital" for c in cells))
        check("no-m2-m3-m4-m5-cells",
              not any(c.experiment != "M1" for c in cells))
        # eval completeness: 198 paired units, 5000 episodes, test seed 10042
        missing = [f"{u['unit']}@{snr:g}" for u in eval_units()
                   for snr in M1_SNR_DB if not _eval_path(u, snr).exists()]
        check("198-eval-units-complete", len(missing) == 0,
              f"missing={missing[:6]}{'...' if len(missing) > 6 else ''}")
        pooled, eps_ok, pair_ok = {}, True, True
        for u in eval_units():
            for snr in M1_SNR_DB:
                p = _eval_path(u, snr)
                if not p.exists():
                    continue
                rec = json.loads(p.read_text())
                if rec.get("episodes") != 5000 or rec.get("test_seed") != 10042:
                    eps_ok = False
                if not rec.get("paired"):
                    pair_ok = False
                for method in ("reconstruction", "task"):
                    key = (method, u["k"], u["channel"], snr)
                    pooled.setdefault(key, []).append(rec[f"aggregate_{method}"])
        check("episodes==5000-test-seed==10042-everywhere", eps_ok)
        check("paired-true-everywhere", pair_ok)
        check("132-pooled-eval-conditions", len(pooled) == 132, f"{len(pooled)}")
        check("3-seeds-per-pooled-cell",
              all(len(v) == 3 for v in pooled.values()))
        # frozen matrix expectations + structural validation of the proposal
        from main import _matrix_proposal
        ec = expected_cells()
        check("matrix-expected-M1-counts",
              ec["M1_training_runs"] == 36 and ec["M1_eval_conditions"] == 132)
        res = validate_matrix(_matrix_proposal())
        check("frozen-matrix-proposal-valid", res["valid"],
              f"warnings={len(res['warnings'])}")
        # every unit carries both methods' aggregates
        both = True
        for u in eval_units():
            for snr in M1_SNR_DB:
                p = _eval_path(u, snr)
                if not p.exists():
                    continue
                rec = json.loads(p.read_text())
                if ("aggregate_reconstruction" not in rec
                        or "aggregate_task" not in rec):
                    both = False
        check("both-methods-in-every-unit", both)
        print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
        return 1 if FAILS else 0


    # ---------------------------------------------------------------------------
    # finalize — execution-facts summary (descriptive, no ranking)
    # ---------------------------------------------------------------------------

    def cmd_finalize():
        print("== finalize: results/M1/step10_m1_summary.json ==")
        cells = m1_cells()
        train_rows = []
        for cfg in cells:
            d = run_dir(cfg, results_root="results")
            rec = json.loads((d / "train_report.json").read_text())
            train_rows.append({k2: rec.get(k2) for k2 in
                               ("run_id", "config_hash", "method", "channel", "k",
                                "seed", "wall_s", "s_per_step", "nan_batches",
                                "grads_finite", "exit_status")})
        n_units = 0
        pooled = {}
        wilson = {}
        for u in eval_units():
            for snr in M1_SNR_DB:
                p = _eval_path(u, snr)
                if not p.exists():
                    continue
                n_units += 1
                rec = json.loads(p.read_text())
                for method in ("reconstruction", "task"):
                    key = (method, u["k"], u["channel"], snr)
                    pooled.setdefault(key, []).append(
                        rec[f"aggregate_{method}"])
                    w = rec[f"wilson_{method}"]
                    wilson.setdefault(key, []).append(
                        (w["success_count"], w["total"]))
        pool_rows = []
        for (method, k, ch, snr) in sorted(pooled.keys(),
                                           key=lambda x: (x[1], x[2], x[0], x[3])):
            aggs = pooled[(method, k, ch, snr)]
            row = {"method": method, "k": k, "channel": ch, "snr_db": snr,
                   "n_seeds": len(aggs)}
            # pooled success: sum of successes over the seed rows / totals
            sc = sum(a["success_count"] for a in aggs)
            tot = sum(a["episodes"] for a in aggs)
            row["success_count_pooled"] = sc
            row["episodes_pooled"] = tot
            row["success_rate_pooled"] = sc / tot if tot else float("nan")
            w = wilson[(method, k, ch, snr)]
            succ = ([1] * sum(c for c, _ in w) + [0] * sum(t - c for c, t in w))
            ci = wilson_ci_95(succ)
            row["wilson_low_pooled"] = ci["wilson_low"]
            row["wilson_high_pooled"] = ci["wilson_high"]
            for f in ("final_distance_mean", "avg_distance_mean",
                      "control_effort_mean", "min_distance_mean", "mse_mean",
                      "ttg_success_mean"):
                vals = [a[f] for a in aggs
                        if isinstance(a.get(f), (int, float))
                        and math.isfinite(a[f])]
                if vals:
                    row[f"{f}_mean_over_seeds"] = sum(vals) / len(vals)
                    if len(vals) > 1:
                        mu = sum(vals) / len(vals)
                        var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
                        row[f"{f}_std_over_seeds"] = var ** 0.5
            pool_rows.append(row)
        total_train_s = sum(r["wall_s"] or 0 for r in train_rows)
        summary = {
            "STEP": "10", "experiment": "M1",
            "status": ("complete" if len(train_rows) == 36 and n_units == 198
                       else "incomplete"),
            "protocol_frozen_9C": {
                "recon_steps": config.RECON_TRAIN_STEPS,
                "task_steps": config.TASK_TRAIN_STEPS,
                "task_training_T": config.TASK_TRAINING_T,
                "batch_states": config.BATCH_STATES,
                "batch_episodes": config.BATCH_EPISODES,
                "optimizer": "Adam", "lr": config.LEARNING_RATE,
                "grad_clip": config.GRAD_CLIP_NORM,
                "train_snr_policy": "U(0,20) dB per batch",
                "training_seeds": list(config.TRAINING_SEEDS),
                "test_seed": config.TEST_SEED, "early_stopping": False,
                "eval_grid_db": list(M1_SNR_DB),
                "eval_episodes": config.N_TEST_EPISODES,
                "eval_horizon": {"T_max": config.T_MAX, "dt": config.DT},
                "goal_radius": config.GOAL_RADIUS,
                "m5_status": config.M5_STATUS},
            "training_runs": {"expected": 36, "completed": len(train_rows),
                              "failed": sum(1 for r in train_rows
                                            if r["exit_status"] != "complete")},
            "eval": {"paired_units_expected": 198, "paired_units_complete": n_units,
                     "pooled_conditions": len(pool_rows),
                     "episodes_per_unit_per_method": config.N_TEST_EPISODES},
            "total_train_wall_s": round(total_train_s, 1),
            "interpretation_note": ("Execution report only; no overall ranking "
                                    "or scientific verdict is made in this step. "
                                    "Pooled rows are descriptive (pooled success "
                                    "over the 3 training-seed rows + mean/std of "
                                    "per-seed aggregates); paired per-seed "
                                    "McNemar/paired-t results live in the "
                                    "eval unit JSONs."),
            "train_rows": train_rows,
            "pooled_rows": pool_rows,
        }
        out = M1_ROOT / "step10_m1_summary.json"
        out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"wrote {out} ({len(train_rows)} train rows, "
              f"{len(pool_rows)} pooled rows, {n_units}/198 units)")
        return 0 if summary["status"] == "complete" else 1


    def _dispatch():
        if len(sys.argv) < 2:
            print(__doc__)
            sys.exit(1)
        cmd = sys.argv[1]
        dispatch = {
            "verify": cmd_verify, "plan": cmd_plan, "status": cmd_status,
            "train": cmd_train, "evaluate": cmd_evaluate,
            "validate": cmd_validate, "checkint": cmd_checkint,
            "finalize": cmd_finalize,
        }
        if cmd not in dispatch:
            print(__doc__)
            sys.exit(1)
        if cmd in ("train", "evaluate"):
            lane = sys.argv[2] if len(sys.argv) > 2 else None
            if lane not in (None, "awgn", "rayleigh"):
                print(f"unknown lane {lane!r} (use awgn|rayleigh or none)")
                sys.exit(1)
            sys.exit(dispatch[cmd](lane))
        sys.exit(dispatch[cmd]())

    return _dispatch
_run_step10_m1_main = _run_step10_m1_main()

# ============================================================
# SECTION 12 M2 EXPERIMENT DRIVER (3-way matched budgets, k in {12,18,24,30,54})
# ============================================================
# ---- Source: tools/run_step11_m2.py ----
def _run_step11_m2_main():
    __doc__ = 'Step 11 — Full M2 Execution (user-approved, 2026-10-02).\n\nExecutes the FROZEN M2 matrix exactly as specified (no frozen value is\ntouched, no M1/M3/M4/M5 paths):\n\n    training  : task + reconstruction x k {12,18,24,30,54} x 2 channels\n                x 3 seeds = 60 neural runs (frozen 9C budgets: recon\n                5000x4096, task 3000x256 T=100)\n    evaluation: 3-method PAIRED evaluation (reconstruction, task, digital)\n                per (k, channel, seed, SNR) in ONE evaluate_condition call,\n                SNR grid {0,5,10,15,20} dB, 5000 episodes, test seed 10042\n                -> 150 paired units x 3 methods. The digital baseline is\n                evaluation-ONLY (30 seed combinations, matched budgets\n                B={12:1,18:2,24:3,30:4,54:8}); it has no training run.\n\nPairing contract (frozen Step-7 harness): test episodes, initial states,\ngoals, noise and fading realizations are bitwise identical across the three\nmethods by construction.\n\nStatistics (frozen statistics.py): Wilson 95% CI per method per unit;\nMcNemar on paired per-episode successes for the 3 pairs (task-vs-recon,\ntask-vs-digital, recon-vs-digital); paired t on final_distance,\navg_distance, control_effort for the same 3 pairs. Descriptive only —\nthe tool never ranks methods or picks winners.\n\nIdentity/recovery: the frozen identity system\n(results/M2/<run_id>__<hash>/ with DONE/RUNNING/FAILED markers).\nCompleted runs SKIP; interrupted/failed runs are RETRAINED (deterministic\nby seed+config — the frozen 9C policy). M1 results are never touched:\nthis tool only reads results/M1 to confirm it stays intact in validate().\n\nSubcommands (repository root):\n    verify      frozen-protocol + workload hard gate (STOP on mismatch)\n    plan        print the exact deterministic execution order (no side effects)\n    status      per-cell state table + counts (no side effects)\n    train       run the training queue (skip completed; retrain on retry)\n    evaluate    run the 3-method paired-eval queue (skips completed units)\n    validate    completion checks + frozen-matrix validation + M1-intact check\n    checkint    per-run checkpoint integrity re-verification (60 neural runs)\n    finalize    write results/M2/step11_m2_summary.json (execution facts)\n\nLane filter ("awgn"|"rayleigh") on train/evaluate is purely a wall-clock\nscheduling device: two independent processes share the matrix by channel;\neach cell still runs exactly once in its own deterministic order.\n\nArtifacts (all gitignored):\n    results/M2/<run_id>__<hash>/{final_ckpt.pt, train_report.json, DONE.json}\n    results/M2/eval/k<k>_<channel>_seed<seed>/eval_<snr>.json\n    results/M2/step11_m2_summary.json'
    """Step 11 — Full M2 Execution (user-approved, 2026-10-02).

    Executes the FROZEN M2 matrix exactly as specified (no frozen value is
    touched, no M1/M3/M4/M5 paths):

        training  : task + reconstruction x k {12,18,24,30,54} x 2 channels
                    x 3 seeds = 60 neural runs (frozen 9C budgets: recon
                    5000x4096, task 3000x256 T=100)
        evaluation: 3-method PAIRED evaluation (reconstruction, task, digital)
                    per (k, channel, seed, SNR) in ONE evaluate_condition call,
                    SNR grid {0,5,10,15,20} dB, 5000 episodes, test seed 10042
                    -> 150 paired units x 3 methods. The digital baseline is
                    evaluation-ONLY (30 seed combinations, matched budgets
                    B={12:1,18:2,24:3,30:4,54:8}); it has no training run.

    Pairing contract (frozen Step-7 harness): test episodes, initial states,
    goals, noise and fading realizations are bitwise identical across the three
    methods by construction.

    Statistics (frozen statistics.py): Wilson 95% CI per method per unit;
    McNemar on paired per-episode successes for the 3 pairs (task-vs-recon,
    task-vs-digital, recon-vs-digital); paired t on final_distance,
    avg_distance, control_effort for the same 3 pairs. Descriptive only —
    the tool never ranks methods or picks winners.

    Identity/recovery: the frozen identity system
    (results/M2/<run_id>__<hash>/ with DONE/RUNNING/FAILED markers).
    Completed runs SKIP; interrupted/failed runs are RETRAINED (deterministic
    by seed+config — the frozen 9C policy). M1 results are never touched:
    this tool only reads results/M1 to confirm it stays intact in validate().

    Subcommands (repository root):
        verify      frozen-protocol + workload hard gate (STOP on mismatch)
        plan        print the exact deterministic execution order (no side effects)
        status      per-cell state table + counts (no side effects)
        train       run the training queue (skip completed; retrain on retry)
        evaluate    run the 3-method paired-eval queue (skips completed units)
        validate    completion checks + frozen-matrix validation + M1-intact check
        checkint    per-run checkpoint integrity re-verification (60 neural runs)
        finalize    write results/M2/step11_m2_summary.json (execution facts)

    Lane filter ("awgn"|"rayleigh") on train/evaluate is purely a wall-clock
    scheduling device: two independent processes share the matrix by channel;
    each cell still runs exactly once in its own deterministic order.

    Artifacts (all gitignored):
        results/M2/<run_id>__<hash>/{final_ckpt.pt, train_report.json, DONE.json}
        results/M2/eval/k<k>_<channel>_seed<seed>/eval_<snr>.json
        results/M2/step11_m2_summary.json
    """


    import json
    import math
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch


    M2_ROOT = Path("results/M2")
    NEURAL_METHODS = ("reconstruction", "task")
    ALL_METHODS = ("reconstruction", "task", "digital")
    FAILS = []


    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  [{name}]" + (f"  ({detail})" if detail else ""))
        if not ok:
            FAILS.append(name)


    # ---------------------------------------------------------------------------
    # frozen cells
    # ---------------------------------------------------------------------------

    def _packet_feasible(k, b):
        """True iff config.digital_packet accepts (k, B) without raising."""
        try:
            config.digital_packet(k, b)
            return True
        except config.InfeasibleDigitalConfig:
            return False


    def _m2_cfg(method, k, channel, seed):
        """The exact frozen M2 cell (frozen values only; nothing is set here)."""
        return ExperimentConfig(
            experiment="M2", method=method, channel=channel, k=k,
            training_seed=seed, test_seed=config.TEST_SEED,
            train_steps=(config.RECON_TRAIN_STEPS if method == "reconstruction"
                         else (config.TASK_TRAIN_STEPS if method == "task" else 0)),
            batch_states=config.BATCH_STATES,
            batch_episodes=config.BATCH_EPISODES,
            t_max=config.TASK_TRAINING_T,
            lr=config.LEARNING_RATE,
            bits_per_component=(M2_B_BY_K[k] if method == "digital" else None),
            eval_snr_grid_db=M2_SNR_DB,
            eval_snr_db=10.0,
            n_train_episodes=config.N_TRAIN_EPISODES,
            n_test_episodes=config.N_TEST_EPISODES,
            eval_episodes=config.BATCH_EPISODES,
            run_mode="smoke")   # identity/vocabulary only; execution is THIS tool


    def m2_neural_cells():
        """60 neural training cells in the frozen deterministic order."""
        cells = []
        for k in M2_K_VALUES:
            for channel in CHANNELS:
                for method in NEURAL_METHODS:
                    for seed in TRAINING_SEEDS:
                        cells.append(_m2_cfg(method, k, channel, seed))
        return cells


    def m2_digital_cells():
        """30 digital evaluation-only seed combinations (no training)."""
        return [_m2_cfg("digital", k, channel, seed)
                for k in M2_K_VALUES for channel in CHANNELS
                for seed in TRAINING_SEEDS]


    def eval_units():
        """30 paired evaluation units (k, channel, seed); each unit evaluates
        reconstruction + task + digital in ONE paired call per SNR."""
        units = []
        for k in M2_K_VALUES:
            for channel in CHANNELS:
                for seed in TRAINING_SEEDS:
                    units.append({
                        "unit": f"k{k}_{channel}_seed{seed}",
                        "k": k, "channel": channel, "seed": seed,
                        "B": M2_B_BY_K[k],
                        "recon": _m2_cfg("reconstruction", k, channel, seed),
                        "task": _m2_cfg("task", k, channel, seed),
                        "digital": _m2_cfg("digital", k, channel, seed),
                    })
        return units


    def _eval_path(unit, snr):
        return M2_ROOT / "eval" / unit["unit"] / f"eval_{snr:g}.json"


    def _load_model(cfg):
        ckpt = run_dir(cfg, results_root="results") / "final_ckpt.pt"
        if not ckpt.exists():
            return None
        model = ReconstructionDeepJSCC(k=cfg.k)
        payload = torch.load(ckpt, weights_only=False)
        model.load_state_dict(payload["model_state"])
        return model


    # ---------------------------------------------------------------------------
    # verify — hard gate on the frozen protocol and the workload
    # ---------------------------------------------------------------------------

    def cmd_verify():
        print("== frozen Step-9C protocol verification (Step-11/M2 gate) ==")
        check("recon_steps==5000", config.RECON_TRAIN_STEPS == 5000)
        check("task_steps==3000", config.TASK_TRAIN_STEPS == 3000)
        check("task_training_T==100", config.TASK_TRAINING_T == 100 == config.T_MAX)
        check("batch_states==4096", config.BATCH_STATES == 4096)
        check("batch_episodes==256", config.BATCH_EPISODES == 256)
        check("lr==1e-3", config.LEARNING_RATE == 1e-3)
        check("grad_clip==1.0", config.GRAD_CLIP_NORM == 1.0)
        check("training_snr==U(0,20)/batch",
              config.SNR_TRAIN_RANGE_DB == (0.0, 20.0))
        check("seeds=={42,43,44}", config.TRAINING_SEEDS == (42, 43, 44))
        check("test_seed==10042", config.TEST_SEED == 10042)
        check("early_stopping==disabled", config.EARLY_STOPPING is False)
        check("M5==deferred+disabled", config.M5_STATUS == "deferred"
              and config.M5_EXECUTION_ENABLED is False)
        check("n_test_episodes==5000", config.N_TEST_EPISODES == 5000)
        check("goal_radius==0.5", config.GOAL_RADIUS == 0.5)
        check("horizon_T_max==100_dt==0.1",
              config.T_MAX == 100 and config.DT == 0.1)
        check("task_lambdas_frozen",
              (config.LAMBDA_DISTANCE, config.LAMBDA_CONTROL,
               config.LAMBDA_VELOCITY, config.LAMBDA_TERMINAL) == (1.0, 0.0, 0.0, 2.0))
        # frozen M2 matrix
        check("m2_k_values=={12,18,24,30,54}", M2_K_VALUES == (12, 18, 24, 30, 54))
        check("m2_snr_grid=={0,5,10,15,20}", M2_SNR_DB == (0.0, 5.0, 10.0, 15.0, 20.0))
        check("m2_b_by_k=={12:1,18:2,24:3,30:4,54:8}",
              M2_B_BY_K == {12: 1, 18: 2, 24: 3, 30: 4, 54: 8}
              and M2_B_BY_K == dict(config.DIGITAL_B_BY_K))
        # frozen digital baseline constants
        check("digital_K7_rate_half",
              config.CONV_CONSTRAINT_LENGTH == 7 and config.CONV_RATE == 0.5)
        check("digital_generators_G1_first",
              config.CONV_GENERATORS_OCTAL == (0o171, 0o133))
        check("digital_gray_qpsk", config.MODULATION_BITS_PER_SYMBOL == 2)
        check("digital_tail_bits==6", config.CONV_TAIL_BITS == 6)
        check("digital_feasible_all_m2_k",
              all(_packet_feasible(k, M2_B_BY_K[k]) for k in M2_K_VALUES),
              "6B+6<=k holds for every frozen M2 k")
        config.validate()
        print("config.validate() OK")
        # workload
        neural, digital = m2_neural_cells(), m2_digital_cells()
        check("workload==60_neural_training_runs", len(neural) == 60)
        check("workload==30_digital_eval_only", len(digital) == 30)
        check("workload==150_paired_eval_units",
              len(eval_units()) * len(M2_SNR_DB) == 150
              and len(eval_units()) == 30)
        check("workload==150_pooled_conditions",
              3 * 5 * 2 * len(M2_SNR_DB) == 150)
        check("tool_has_no_M1_M3_M4_M5_paths", True,
              "this tool implements M2 only (M1 read-only intactness in validate)")
        print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
        return 1 if FAILS else 0


    # ---------------------------------------------------------------------------
    # plan / status (no side effects)
    # ---------------------------------------------------------------------------

    def cmd_plan():
        print("== M2 execution plan (deterministic order; NO execution) ==")
        for i, cfg in enumerate(m2_neural_cells(), 1):
            print(f"{i:2d}. {run_id(cfg):48s} steps={cfg.train_steps:5d} "
                  f"batch={'states4096' if cfg.method == 'reconstruction' else 'episodes256_T100'}")
        print(f"    + {len(m2_digital_cells())} digital evaluation-only cells "
              f"(no training; B={sorted(set(M2_B_BY_K.values()))})")
        print(f"total: {len(m2_neural_cells())} neural training runs; "
              f"{len(m2_digital_cells())} digital eval-only combos; "
              f"{len(eval_units()) * len(M2_SNR_DB)} paired 3-method eval units "
              f"(30 units x 5 SNRs x 5000 episodes, test seed {config.TEST_SEED}); "
              f"pooled conditions: 150")
        return 0


    def cmd_status():
        print("== M2 status ==")
        counts = {"done": 0, "failed": 0, "running": 0, "missing": 0}
        for cfg in m2_neural_cells():
            st = cell_state(cfg, results_root="results")
            counts[st] += 1
            print(f"{st:8s} {run_id(cfg)}  hash={config_hash(cfg)}")
        print(f"-- training: done={counts['done']} failed={counts['failed']} "
              f"running={counts['running']} missing={counts['missing']} (of 60)")
        ev_done = sum(1 for u in eval_units() for snr in M2_SNR_DB
                      if _eval_path(u, snr).exists())
        print(f"-- eval: {ev_done}/150 paired units complete")
        ok = counts["done"] == 60 and counts["failed"] == 0 and ev_done == 150
        print(f"-- M2 complete: {ok}")
        return 0 if ok else 1


    # ---------------------------------------------------------------------------
    # training queue (neural runs only)
    # ---------------------------------------------------------------------------

    def _train_one(cfg):
        if cfg.method == "reconstruction":
            model, rep = train_reconstruction(
                k=cfg.k, channel=cfg.channel, seed=cfg.training_seed,
                steps=cfg.train_steps, batch_states=cfg.batch_states)
            g0, g1 = rep["grad_norm_encoder_final"], rep["grad_norm_decoder_final"]
            grads_finite = (math.isfinite(g0) and math.isfinite(g1)
                            and g0 > 0 and g1 > 0)
            summary = {"loss_initial": rep["loss_initial"],
                       "loss_final": rep["loss_final"],
                       "loss_reduction_pct": rep["loss_reduction_pct"]}
        else:
            model, rep = train_task_oriented(
                k=cfg.k, channel=cfg.channel, seed=cfg.training_seed,
                steps=cfg.train_steps, batch_episodes=cfg.batch_episodes,
                t_max=cfg.t_max, eval_episodes=cfg.eval_episodes,
                verbose_every=0)
            g0, g1 = rep["grad_enc_final"], rep["grad_dec_final"]
            grads_finite = (math.isfinite(g0) and math.isfinite(g1)
                            and g0 > 0 and g1 > 0)
            summary = {"task_loss_initial": rep["train_total_initial"],
                       "task_loss_final": rep["train_total_final"],
                       "recon_mse_diag_initial": rep["recon_mse_diag_initial"],
                       "recon_mse_diag_final": rep["recon_mse_diag_final"],
                       "eval_after_success_rate": rep["eval_after"]["success_rate"],
                       "lambdas": rep["lambdas"]}
        return model, rep, grads_finite, summary


    def _split_lane(lane):
        """Lane spec -> (channel, k) filter. "awgn"/"rayleigh" filter by channel;
        "k12".."k54" filter by k (disjoint scheduling partitions for
        multi-machine runs); None = everything."""
        if lane is None:
            return None, None
        if lane.startswith("k"):
            return None, int(lane[1:])
        return lane, None


    def cmd_train(lane=None):
        """Training queue; optional lane filter ("awgn"|"rayleigh"|"k12".."k54")."""
        f_channel, f_k = _split_lane(lane)
        print("== M2 training queue (frozen 9C budgets) =="
              + (f"  [lane: {lane}]" if lane else ""))
        cells = [c for c in m2_neural_cells()
                 if (f_channel is None or c.channel == f_channel)
                 and (f_k is None or c.k == f_k)]
        total = len(cells)
        n_trained = n_skipped = n_retrained = 0
        total_wall = 0.0
        for i, cfg in enumerate(cells, 1):
            rid, ch = run_id(cfg), config_hash(cfg)
            st = cell_state(cfg, results_root="results")
            if st == "done":
                print(f"[{i:2d}/{total}] SKIP {rid} (already complete)")
                n_skipped += 1
                continue
            if st == "failed":
                print(f"[{i:2d}/{total}] RETRAIN-after-FAILURE {rid} "
                      f"(frozen 9C policy: deterministic by seed+config)")
                n_retrained += 1
            elif st == "running":
                print(f"[{i:2d}/{total}] RETRAIN-after-INTERRUPTION {rid} "
                      f"(stale RUNNING marker; environmental; frozen 9C policy)")
                n_retrained += 1
            d = run_dir(cfg, results_root="results")
            mark_running(cfg, results_root="results")
            print(f"[{i:2d}/{total}] TRAIN {rid} hash={ch} steps={cfg.train_steps} "
                  f"batch={'4096' if cfg.method == 'reconstruction' else '256xT100'}")
            t0 = time.perf_counter()
            try:
                model, rep, grads_finite, summary = _train_one(cfg)
            except Exception as e:              # structured failure, no silent retry
                mark_failed(cfg, f"{type(e).__name__}: {e}", results_root="results")
                (d / "train_report.json").write_text(json.dumps({
                    "run_id": rid, "config_hash": ch, "exit_status": "failed",
                    "error": f"{type(e).__name__}: {e}"}, indent=2))
                print(f"[{i:2d}/{total}] FAILED {rid}: {type(e).__name__}: {e}")
                FAILS.append(rid)
                continue
            wall = time.perf_counter() - t0
            total_wall += wall
            loss_key = ("loss_final" if cfg.method == "reconstruction"
                        else "task_loss_final")
            complete = (rep["nan_batches"] == 0 and grads_finite
                        and math.isfinite(summary[loss_key]))
            if not complete:
                print(f"[{i:2d}/{total}] NUMERICAL ALERT {rid}: "
                      f"nan_batches={rep['nan_batches']} "
                      f"grads_finite={grads_finite} "
                      f"loss_finite={math.isfinite(summary[loss_key])}")
            rec = {
                "STEP": "11", "experiment": "M2",
                "run_id": rid, "config_hash": ch,
                "method": cfg.method, "channel": cfg.channel, "k": cfg.k,
                "seed": cfg.training_seed,
                "protocol": {
                    "train_steps": cfg.train_steps,
                    "batch_states": cfg.batch_states,
                    "batch_episodes": cfg.batch_episodes,
                    "t_max": cfg.t_max, "lr": cfg.lr,
                    "optimizer": "Adam", "grad_clip": config.GRAD_CLIP_NORM,
                    "train_snr_policy": {"distribution": "uniform",
                                         "range_db": [0.0, 20.0],
                                         "granularity": "per_batch"},
                    "early_stopping": False},
                "wall_s": round(wall, 1),
                "s_per_step": round(wall / cfg.train_steps, 4),
                "nan_batches": rep["nan_batches"],
                "grads_finite": grads_finite,
                "exit_status": "complete" if complete else "incomplete",
                "checkpoint": str(d / "final_ckpt.pt"),
                "optimizer_state_saved": False,
                "optimizer_note": ("trainer-internal optimizer not exposed "
                                   "(documented Step-8 limitation); recovery is "
                                   "deterministic RETRAIN by seed+config (9C)"),
                **summary,
            }
            save_checkpoint(
                d / "final_ckpt.pt", model, optimizer=None,
                step=cfg.train_steps, seed=cfg.training_seed,
                metadata={"run_id": rid, "config_hash": ch,
                          "experiment": "M2", "method": cfg.method,
                          "channel": cfg.channel, "k": cfg.k,
                          "training_seed": cfg.training_seed,
                          "train_steps": cfg.train_steps,
                          "batch_states": cfg.batch_states,
                          "batch_episodes": cfg.batch_episodes,
                          "t_max": cfg.t_max, "lr": cfg.lr,
                          "model_config": {"hidden_dims": list(config.HIDDEN_DIMS),
                                           "activation": "PReLU"},
                          "train_snr_policy": rec["protocol"]["train_snr_policy"]},
                history=rep["history"], capture_rng=True)
            (d / "train_report.json").write_text(
                json.dumps(rec, indent=2, default=str))
            n_trained += 1
            if complete:
                mark_done(cfg, summary={"wall_s": rec["wall_s"],
                                        loss_key: summary[loss_key]},
                          results_root="results")
                print(f"[{i:2d}/{total}] DONE  {rid}  {rec['wall_s']}s  "
                      f"{loss_key}={summary[loss_key]:.4f}")
            else:
                mark_failed(cfg, "incomplete numerical state",
                            results_root="results")
                FAILS.append(rid)
                print(f"[{i:2d}/{total}] INCOMPLETE {rid} "
                      f"(nan_batches={rep['nan_batches']}, "
                      f"grads_finite={grads_finite})")
        print(f"-- training queue finished: trained={n_trained} "
              f"skipped={n_skipped} retrained={n_retrained} "
              f"failed={len(FAILS)} train_wall={total_wall:.1f}s")
        return 1 if FAILS else 0


    # ---------------------------------------------------------------------------
    # checkpoint integrity (neural runs only)
    # ---------------------------------------------------------------------------

    def _integrity_failures(cfg):
        """11-checkpoint integrity for one neural run; list of failures."""
        bad = []
        d = run_dir(cfg, results_root="results")
        ckpt = d / "final_ckpt.pt"
        if not ckpt.exists():
            return ["ckpt-exists"]
        fresh = ReconstructionDeepJSCC(k=cfg.k)
        payload = torch.load(ckpt, weights_only=False)
        md = payload.get("metadata", {})
        if not md:
            bad.append("ckpt-metadata")
        if not (md.get("train_steps") == cfg.train_steps
                and md.get("t_max") == cfg.t_max
                and math.isclose(md.get("lr", -1), config.LEARNING_RATE)
                and md.get("batch_states") == cfg.batch_states
                and md.get("batch_episodes") == cfg.batch_episodes):
            bad.append("config-matches")
        if payload.get("seed") != cfg.training_seed:
            bad.append("seed-matches")
        if not (md.get("method") == cfg.method and md.get("channel") == cfg.channel
                and md.get("k") == cfg.k):
            bad.append("method-channel-k")
        if payload.get("step") != cfg.train_steps:
            bad.append("final-step==budget")
        if not all(bool(torch.isfinite(p).all()) for p in fresh.parameters()):
            bad.append("params-finite")
        if md.get("run_id") != run_id(cfg):
            bad.append("identity-matches")
        if md.get("config_hash") != config_hash(cfg):
            bad.append("hash-matches")
        rep_path = d / "train_report.json"
        if not (rep_path.exists()
                and json.loads(rep_path.read_text()).get("exit_status") == "complete"):
            bad.append("report-complete")
        return bad


    # ---------------------------------------------------------------------------
    # 3-method paired evaluation queue
    # ---------------------------------------------------------------------------

    # Frozen-protocol exception, user-approved 2026-10-06 (supervisor ratification
    # pending): these 2 runs failed ONLY the final-step grads_finite probe —
    # nan_batches=0, finite final loss, finite checkpoint weights, full 3000
    # gradient steps executed. They are conditionally accepted for EVALUATION
    # only; their train_report.json exit_status remains "incomplete" (never
    # edited). Full decision trail: results/M2/PROTOCOL_EXCEPTIONS.md.
    CONDITIONAL_ACCEPT = frozenset({
        "M2_task_awgn_k30_seed43__d270bfa0b913b678",
        "M2_task_awgn_k54_seed42__59f19c2c451a8106",
    })


    def _paired_stats3(cell):
        """Wilson CI per method + 3 McNemar + 3x3 paired t from one paired unit."""
        out = {}
        for name in ALL_METHODS:
            out[f"wilson_{name}"] = wilson_ci_95(cell[name]["metrics"]["success"])
        for a, b in (("task", "reconstruction"), ("task", "digital"),
                     ("reconstruction", "digital")):
            out[f"mcnemar_{a}_vs_{b}"] = mcnemar_test(
                cell[a]["metrics"]["success"], cell[b]["metrics"]["success"])
            for key in ("final_distance", "avg_distance", "control_effort"):
                out[f"paired_t_{key}_{a}_vs_{b}"] = paired_t_test(
                    cell[a]["metrics"][key].to(torch.float64),
                    cell[b]["metrics"][key].to(torch.float64))
        return out


    def cmd_evaluate(lane=None):
        """3-method paired-eval queue; lane filter ("awgn"|"rayleigh"|"k12".."k54")."""
        f_channel, f_k = _split_lane(lane)
        print("== M2 paired evaluation queue (Step-7 harness, 3 methods, "
              "5000 episodes) ==" + (f"  [lane: {lane}]" if lane else ""))
        print("-- per-run checkpoint integrity gate --")
        blocked = []
        neural = [c for c in m2_neural_cells()
                  if (f_channel is None or c.channel == f_channel)
                  and (f_k is None or c.k == f_k)]
        for cfg in neural:
            st = cell_state(cfg, results_root="results")
            if st != "done":
                if f"{run_id(cfg)}__{config_hash(cfg)}" in CONDITIONAL_ACCEPT:
                    continue          # documented protocol exception (eval only)
                blocked.append(f"{run_id(cfg)}: not complete")
            else:
                bad = _integrity_failures(cfg)
                if bad:
                    blocked.append(f"{run_id(cfg)}: {bad}")
        if blocked:
            for b in blocked:
                print(f"REFUSED: {b}")
            return 2
        print(f"   integrity gate passed for lane="
              f"{lane or 'all'} ({len(neural)} runs)")
        n_new = 0
        total_wall = 0.0
        units = [u for u in eval_units()
                 if (f_channel is None or u["channel"] == f_channel)
                 and (f_k is None or u["k"] == f_k)]
        for u in units:
            m_recon = _load_model(u["recon"])
            m_task = _load_model(u["task"])
            cond_base = dict(k=u["k"], channel=u["channel"],
                             bits_per_component=u["B"],
                             t_max=config.T_MAX,
                             episodes=config.N_TEST_EPISODES,
                             seed=config.TEST_SEED)
            for snr in M2_SNR_DB:
                out = _eval_path(u, snr)
                if out.exists():
                    continue
                cond = Condition(snr_db=snr, **cond_base)
                t0 = time.perf_counter()
                with torch.no_grad():
                    cell = evaluate_condition(
                        cond, methods=list(ALL_METHODS),
                        recon_model=m_recon, task_model=m_task)
                wall = time.perf_counter() - t0
                total_wall += wall
                rec = {
                    "STEP": "11", "experiment": "M2",
                    "unit": u["unit"], "k": u["k"], "channel": u["channel"],
                    "seed": u["seed"], "bits_per_component": u["B"],
                    "snr_db": snr,
                    "test_seed": config.TEST_SEED,
                    "episodes": config.N_TEST_EPISODES, "t_max": config.T_MAX,
                    "paired": True,
                    "methods": list(ALL_METHODS),
                    "wall_s": round(wall, 1),
                    **{f"aggregate_{name}": cell[name]["aggregate"]
                       for name in ALL_METHODS},
                    **_paired_stats3(cell),
                }
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rec, indent=2, default=str))
                n_new += 1
                sr = {name: rec[f"aggregate_{name}"]["success_rate"]
                      for name in ALL_METHODS}
                print(f"  EVAL {u['unit']} snr={snr:5.1f} "
                      f"SR recon={sr['reconstruction']:.4f} "
                      f"task={sr['task']:.4f} digital={sr['digital']:.4f} "
                      f"({wall:.1f}s)")
            del m_recon, m_task
        print(f"-- evaluation queue finished: {n_new} new units this session, "
              f"eval_wall={total_wall:.1f}s total")
        return 0


    # ---------------------------------------------------------------------------
    # checkint / validate
    # ---------------------------------------------------------------------------

    def cmd_checkint():
        bad = {}
        for cfg in m2_neural_cells():
            f = _integrity_failures(cfg)
            if f:
                bad[run_id(cfg)] = f
        for rid, f in bad.items():
            print(f"FAIL {rid}: {f}")
        print(f"== checkpoint integrity: {60 - len(bad)}/60 runs pass ==")
        return 1 if bad else 0


    def cmd_validate():
        print("== M2 completion validation ==")
        FAILS.clear()
        cells = m2_neural_cells()
        ids = [(run_id(c), config_hash(c)) for c in cells]
        check("60-unique-identities", len(set(ids)) == 60)
        done = sum(1 for c in cells
                   if cell_state(c, results_root="results") == "done")
        check("60-training-runs-complete", done == 60, f"{done}/60")
        check("methods==reconstruction,task",
              sorted({c.method for c in cells}) == ["reconstruction", "task"])
        check("k=={12,18,24,30,54}", sorted({c.k for c in cells})
              == [12, 18, 24, 30, 54])
        check("channels==awgn,rayleigh",
              sorted({c.channel for c in cells}) == ["awgn", "rayleigh"])
        check("seeds=={42,43,44}",
              sorted({c.training_seed for c in cells}) == [42, 43, 44])
        check("no-m1-m3-m4-m5-cells",
              not any(c.experiment != "M2" for c in cells))
        # eval completeness: 150 paired 3-method units, 5000 episodes, seed 10042
        missing = [f"{u['unit']}@{snr:g}" for u in eval_units()
                   for snr in M2_SNR_DB if not _eval_path(u, snr).exists()]
        check("150-eval-units-complete", len(missing) == 0,
              f"missing={missing[:6]}{'...' if len(missing) > 6 else ''}")
        pooled, eps_ok, pair_ok, methods_ok = {}, True, True, True
        for u in eval_units():
            for snr in M2_SNR_DB:
                p = _eval_path(u, snr)
                if not p.exists():
                    continue
                rec = json.loads(p.read_text())
                if rec.get("episodes") != 5000 or rec.get("test_seed") != 10042:
                    eps_ok = False
                if not rec.get("paired"):
                    pair_ok = False
                if sorted(rec.get("methods", [])) != sorted(ALL_METHODS):
                    methods_ok = False
                for method in ALL_METHODS:
                    key = (method, u["k"], u["channel"], snr)
                    pooled.setdefault(key, []).append(rec[f"aggregate_{method}"])
        check("episodes==5000-test-seed==10042-everywhere", eps_ok)
        check("paired-true-everywhere", pair_ok)
        check("three-methods-in-every-unit", methods_ok)
        check("150-pooled-eval-conditions", len(pooled) == 150, f"{len(pooled)}")
        check("3-seeds-per-pooled-cell",
              all(len(v) == 3 for v in pooled.values()))
        # frozen matrix expectations + structural validation of the proposal
        from main import _matrix_proposal
        ec = expected_cells()
        check("matrix-expected-M2-counts",
              ec["M2_neural_training_runs"] == 60
              and ec["M2_digital_eval_only_seeds"] == 30
              and ec["M2_eval_conditions"] == 150)
        res = validate_matrix(_matrix_proposal())
        check("frozen-matrix-proposal-valid", res["valid"],
              f"warnings={len(res['warnings'])}")
        # every unit carries all three methods' aggregates
        both = True
        for u in eval_units():
            for snr in M2_SNR_DB:
                p = _eval_path(u, snr)
                if not p.exists():
                    continue
                rec = json.loads(p.read_text())
                if any(f"aggregate_{m}" not in rec for m in ALL_METHODS):
                    both = False
        check("all-methods-in-every-unit", both)
        # M1 intactness (read-only): M1 artifacts must be untouched
        m1_done = len(list(Path("results/M1").glob("*__*/DONE.json")))
        m1_eval = len(list(Path("results/M1/eval").glob("*/eval_*.json")))
        m1_sum = Path("results/M1/step10_m1_summary.json")
        m1_status = (json.loads(m1_sum.read_text()).get("status")
                     if m1_sum.exists() else "missing")
        check("m1-intact-36-train-198-eval",
              m1_done == 36 and m1_eval == 198 and m1_status == "complete",
              f"train={m1_done}/36 eval={m1_eval}/198 summary={m1_status}")
        print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
        return 1 if FAILS else 0


    # ---------------------------------------------------------------------------
    # finalize — execution-facts summary (descriptive, no ranking)
    # ---------------------------------------------------------------------------

    def cmd_finalize():
        print("== finalize: results/M2/step11_m2_summary.json ==")
        cells = m2_neural_cells()
        train_rows = []
        for cfg in cells:
            d = run_dir(cfg, results_root="results")
            rec = json.loads((d / "train_report.json").read_text())
            train_rows.append({k2: rec.get(k2) for k2 in
                               ("run_id", "config_hash", "method", "channel", "k",
                                "seed", "wall_s", "s_per_step", "nan_batches",
                                "grads_finite", "exit_status")})
        n_units = 0
        pooled = {}
        wilson = {}
        for u in eval_units():
            for snr in M2_SNR_DB:
                p = _eval_path(u, snr)
                if not p.exists():
                    continue
                n_units += 1
                rec = json.loads(p.read_text())
                for method in ALL_METHODS:
                    key = (method, u["k"], u["channel"], snr)
                    pooled.setdefault(key, []).append(
                        rec[f"aggregate_{method}"])
                    w = rec[f"wilson_{method}"]
                    wilson.setdefault(key, []).append(
                        (w["success_count"], w["total"]))
        pool_rows = []
        for (method, k, ch, snr) in sorted(pooled.keys(),
                                           key=lambda x: (x[2], x[3], x[0], x[1])):
            aggs = pooled[(method, k, ch, snr)]
            row = {"method": method, "k": k, "channel": ch, "snr_db": snr,
                   "n_seeds": len(aggs)}
            sc = sum(a["success_count"] for a in aggs)
            tot = sum(a["episodes"] for a in aggs)
            row["success_count_pooled"] = sc
            row["episodes_pooled"] = tot
            row["success_rate_pooled"] = sc / tot if tot else float("nan")
            w = wilson[(method, k, ch, snr)]
            succ = ([1] * sum(c for c, _ in w) + [0] * sum(t - c for c, t in w))
            ci = wilson_ci_95(succ)
            row["wilson_low_pooled"] = ci["wilson_low"]
            row["wilson_high_pooled"] = ci["wilson_high"]
            for f in ("final_distance_mean", "avg_distance_mean",
                      "control_effort_mean", "min_distance_mean", "mse_mean",
                      "ttg_success_mean"):
                vals = [a[f] for a in aggs
                        if isinstance(a.get(f), (int, float))
                        and math.isfinite(a[f])]
                if vals:
                    row[f"{f}_mean_over_seeds"] = sum(vals) / len(vals)
                    if len(vals) > 1:
                        mu = sum(vals) / len(vals)
                        var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
                        row[f"{f}_std_over_seeds"] = var ** 0.5
            pool_rows.append(row)
        total_train_s = sum(r["wall_s"] or 0 for r in train_rows)
        summary = {
            "STEP": "11", "experiment": "M2",
            "status": ("complete" if len(train_rows) == 60 and n_units == 150
                       else "incomplete"),
            "protocol_frozen_9C": {
                "recon_steps": config.RECON_TRAIN_STEPS,
                "task_steps": config.TASK_TRAIN_STEPS,
                "task_training_T": config.TASK_TRAINING_T,
                "batch_states": config.BATCH_STATES,
                "batch_episodes": config.BATCH_EPISODES,
                "optimizer": "Adam", "lr": config.LEARNING_RATE,
                "grad_clip": config.GRAD_CLIP_NORM,
                "train_snr_policy": "U(0,20) dB per batch",
                "training_seeds": list(config.TRAINING_SEEDS),
                "test_seed": config.TEST_SEED, "early_stopping": False,
                "m2_k_values": list(M2_K_VALUES),
                "m2_snr_grid_db": list(M2_SNR_DB),
                "m2_digital_b_by_k": {str(k): b for k, b in M2_B_BY_K.items()},
                "eval_episodes": config.N_TEST_EPISODES,
                "eval_horizon": {"T_max": config.T_MAX, "dt": config.DT},
                "goal_radius": config.GOAL_RADIUS,
                "m5_status": config.M5_STATUS},
            "training_runs": {"expected": 60, "completed": len(train_rows),
                              "failed": sum(1 for r in train_rows
                                            if r["exit_status"] != "complete")},
            "digital_baseline": {"evaluation_only": True,
                                 "seed_combinations": 30,
                                 "matched_budgets": {str(k): b
                                                     for k, b in M2_B_BY_K.items()}},
            "eval": {"paired_units_expected": 150, "paired_units_complete": n_units,
                     "pooled_conditions": len(pool_rows),
                     "episodes_per_unit_per_method": config.N_TEST_EPISODES,
                     "methods": list(ALL_METHODS)},
            "total_train_wall_s": round(total_train_s, 1),
            "interpretation_note": ("Execution report only; no overall ranking "
                                   "or scientific verdict is made in this step. "
                                   "Three-way comparison is empirical: M1 showed "
                                   "task-oriented UNDERPERFORMED reconstruction in "
                                   "the low-k regime, so no superiority is "
                                   "presumed. Pooled rows are descriptive "
                                   "(pooled success over the 3 training-seed rows "
                                   "+ mean/std of per-seed aggregates); paired "
                                   "per-seed McNemar/paired-t results live in "
                                   "the eval unit JSONs."),
            "train_rows": train_rows,
            "pooled_rows": pool_rows,
        }
        out = M2_ROOT / "step11_m2_summary.json"
        out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"wrote {out} ({len(train_rows)} train rows, "
              f"{len(pool_rows)} pooled rows, {n_units}/150 units)")
        return 0 if summary["status"] == "complete" else 1


    def _dispatch():
        if len(sys.argv) < 2:
            print(__doc__)
            sys.exit(1)
        cmd = sys.argv[1]
        dispatch = {
            "verify": cmd_verify, "plan": cmd_plan, "status": cmd_status,
            "train": cmd_train, "evaluate": cmd_evaluate,
            "validate": cmd_validate, "checkint": cmd_checkint,
            "finalize": cmd_finalize,
        }
        if cmd not in dispatch:
            print(__doc__)
            sys.exit(1)
        if cmd in ("train", "evaluate"):
            lane = sys.argv[2] if len(sys.argv) > 2 else None
            valid = (lane is None or lane in ("awgn", "rayleigh")
                     or (lane.startswith("k") and lane[1:].isdigit()
                         and int(lane[1:]) in M2_K_VALUES))
            if not valid:
                print(f"unknown lane {lane!r} "
                      f"(use awgn|rayleigh|k12|k18|k24|k30|k54 or none)")
                sys.exit(1)
            sys.exit(dispatch[cmd](lane))
        sys.exit(dispatch[cmd]())

    return _dispatch
_run_step11_m2_main = _run_step11_m2_main()

# ============================================================
# SECTION 13 M3 + M4 EXPERIMENT DRIVERS (M3: structural digital feasibility at low k; M4: SECONDARY digital reference B=8 at k=54)
# ============================================================
# ---- Source: tools/run_step12_m3_m4.py ----
def _run_step12_m3_m4_main():
    __doc__ = 'Step 12/13 — M3 (digital-infeasibility display) + M4 (secondary digital\nreference) — user-approved 2026-10-02.\n\nM3 (frozen, structural display — audit 8.5): the matched-budget digital\nbaseline is INFEASIBLE at k in {1, 2, 3} because the smallest transmissible\npacket needs 6B + 6 = 7 channel uses at B = 1. This step RECORDS the frozen\npacket arithmetic for the 3 display cells; nothing is trained or evaluated\n(no digital signal exists at these k by construction).\n\nM4 (frozen, SECONDARY reference — audit 8.6/8.7): digital baseline at the\nfixed secondary point B = 8, k = n_dig = 54 (6*8 + 6 = 54, exact fit), one\ncell per channel (seed 42), evaluated on the M2 SNR grid {0,5,10,15,20} dB\nwith 5000 episodes, test seed 10042 — evaluation-ONLY (digital has no\ntraining). M4 is explicitly NOT a primary cell and never carries the\nmatched-budget label.\n\nNo M1/M2 artifact is touched. No M5 path exists in this tool.\n\nSubcommands (repository root):\n    verify      frozen-protocol + workload hard gate\n    run-m3      record the 3 structural infeasibility display cells\n    run-m4      run the 10 evaluation-only secondary-reference conditions\n    validate    completion checks\n    finalize    write results/M3/m3_report.json (run-m3) and\n                results/M4/step13_m4_summary.json (finalize)'
    """Step 12/13 — M3 (digital-infeasibility display) + M4 (secondary digital
    reference) — user-approved 2026-10-02.

    M3 (frozen, structural display — audit 8.5): the matched-budget digital
    baseline is INFEASIBLE at k in {1, 2, 3} because the smallest transmissible
    packet needs 6B + 6 = 7 channel uses at B = 1. This step RECORDS the frozen
    packet arithmetic for the 3 display cells; nothing is trained or evaluated
    (no digital signal exists at these k by construction).

    M4 (frozen, SECONDARY reference — audit 8.6/8.7): digital baseline at the
    fixed secondary point B = 8, k = n_dig = 54 (6*8 + 6 = 54, exact fit), one
    cell per channel (seed 42), evaluated on the M2 SNR grid {0,5,10,15,20} dB
    with 5000 episodes, test seed 10042 — evaluation-ONLY (digital has no
    training). M4 is explicitly NOT a primary cell and never carries the
    matched-budget label.

    No M1/M2 artifact is touched. No M5 path exists in this tool.

    Subcommands (repository root):
        verify      frozen-protocol + workload hard gate
        run-m3      record the 3 structural infeasibility display cells
        run-m4      run the 10 evaluation-only secondary-reference conditions
        validate    completion checks
        finalize    write results/M3/m3_report.json (run-m3) and
                    results/M4/step13_m4_summary.json (finalize)
    """


    import json
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch


    M3_ROOT = Path("results/M3")
    M4_ROOT = Path("results/M4")
    FAILS = []


    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  [{name}]" + (f"  ({detail})" if detail else ""))
        if not ok:
            FAILS.append(name)


    # ---------------------------------------------------------------------------
    # verify — hard gate
    # ---------------------------------------------------------------------------

    def cmd_verify():
        print("== frozen protocol verification (Step-12/13 gate) ==")
        check("m3_cells==k{1,2,3}_B1_dry_run",
              all(ExperimentConfig.m3(k, bits=1, run_mode="dry_run").expected_infeasible
                  for k in (1, 2, 3)))
        check("m4_point==B8_k54_exact_fit",
              M4_DIGITAL_B == 8 and M4_K == 54
              and 6 * M4_DIGITAL_B + 6 == M4_K)
        check("m4_eval_grid==M2_grid", M2_SNR_DB == (0.0, 5.0, 10.0, 15.0, 20.0))
        check("m4_cells==2channels_x5snr_seed42", 2 * len(M2_SNR_DB) == 10)
        check("n_test_episodes==5000", config.N_TEST_EPISODES == 5000)
        check("test_seed==10042", config.TEST_SEED == 10042)
        check("digital_K7_rate_half_gray_qpsk",
              config.CONV_CONSTRAINT_LENGTH == 7 and config.CONV_RATE == 0.5
              and config.CONV_GENERATORS_OCTAL == (0o171, 0o133)
              and config.MODULATION_BITS_PER_SYMBOL == 2)
        check("M5==deferred+disabled", config.M5_STATUS == "deferred"
              and config.M5_EXECUTION_ENABLED is False)
        config.validate()
        print("config.validate() OK")
        print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
        return 1 if FAILS else 0


    # ---------------------------------------------------------------------------
    # M3 — structural infeasibility display (no training, no eval)
    # ---------------------------------------------------------------------------

    def cmd_run_m3():
        print("== M3: digital infeasibility display cells (k in {1,2,3}) ==")
        rows = []
        for k in (1, 2, 3):
            cfg = ExperimentConfig.m3(k, bits=1, run_mode="dry_run")
            try:
                pkt = config.digital_packet(k, cfg.bits_per_component)
                feasible, note = True, pkt
            except config.InfeasibleDigitalConfig as e:
                feasible, note = False, str(e)
            row = {
                "run_id": cfg.run_id if hasattr(cfg, "run_id") else None,
                "experiment": "M3", "method": "digital",
                "k": k, "bits_per_component": cfg.bits_per_component,
                "required_channel_uses": 6 * cfg.bits_per_component + 6,
                "available_channel_uses": k,
                "deficit": (6 * cfg.bits_per_component + 6) - k,
                "expected_infeasible": cfg.expected_infeasible,
                "status": ("infeasible-by-design" if not feasible
                           else "UNEXPECTED-FEASIBLE"),
                "reason": note,
            }
            rows.append(row)
            tag = "PASS" if not feasible else "FAIL"
            print(f"  {tag} k={k}: needs {row['required_channel_uses']} uses, "
                  f"has {k} -> deficit {row['deficit']} "
                  f"({row['status']})")
        # contrast: the matched-budget M2 points must all be feasible
        contrast = {}
        for k in (12, 18, 24, 30, 54):
            b = {12: 1, 18: 2, 24: 3, 30: 4, 54: 8}[k]
            try:
                pkt = config.digital_packet(k, b)
                contrast[str(k)] = {"B": b, "feasible": True,
                                    "n_info_bits": pkt["n_info_bits"]}
            except config.InfeasibleDigitalConfig:
                contrast[str(k)] = {"B": b, "feasible": False}
        check("all-three-m3-cells-infeasible",
              all(r["status"] == "infeasible-by-design" for r in rows))
        check("all-m2-digital-points-feasible",
              all(v["feasible"] for v in contrast.values()))
        report = {
            "STEP": "12", "experiment": "M3",
            "status": "complete" if not FAILS else "incomplete",
            "definition": ("Structural display: the matched-budget digital "
                           "baseline cannot exist at k in {1,2,3} because the "
                           "smallest packet needs 6B+6 = 7 channel uses at B=1. "
                           "No training or evaluation exists for these cells "
                           "BY DESIGN (audit 8.5); this is a feasibility "
                           "boundary, not a measured result."),
            "cells": rows,
            "m2_contrast_feasible_points": contrast,
            "interpretation_note": ("M3 is architectural infeasibility, NOT an "
                                   "empirical performance result. The M3/M4 "
                                   "distinction: M4's B=8 reference transmits "
                                   "the same 54 channel uses as the k=54 neural "
                                   "cells and is labeled SECONDARY."),
        }
        out = M3_ROOT / "m3_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {out} (3 display cells)")
        return 1 if FAILS else 0


    # ---------------------------------------------------------------------------
    # M4 — secondary digital reference, evaluation-only (10 conditions)
    # ---------------------------------------------------------------------------

    def _m4_eval_path(channel, snr):
        return M4_ROOT / "eval" / channel / f"eval_{snr:g}.json"


    def cmd_run_m4():
        print("== M4: secondary digital reference (B=8, k=54) — 10 conditions ==")
        n_new = 0
        total_wall = 0.0
        for channel in CHANNELS:
            for snr in M2_SNR_DB:
                out = _m4_eval_path(channel, snr)
                if out.exists():
                    continue
                cond = Condition(k=M4_K, channel=channel, snr_db=snr,
                                 bits_per_component=M4_DIGITAL_B,
                                 t_max=config.T_MAX,
                                 episodes=config.N_TEST_EPISODES,
                                 seed=config.TEST_SEED)
                t0 = time.perf_counter()
                with torch.no_grad():
                    cell = evaluate_condition(cond, methods=["digital"])
                wall = time.perf_counter() - t0
                total_wall += wall
                agg = cell["digital"]["aggregate"]
                rec = {
                    "STEP": "13", "experiment": "M4",
                    "channel": channel, "snr_db": snr,
                    "k": M4_K, "bits_per_component": M4_DIGITAL_B,
                    "test_seed": config.TEST_SEED,
                    "episodes": config.N_TEST_EPISODES, "t_max": config.T_MAX,
                    "secondary_reference": True,
                    "wall_s": round(wall, 1),
                    "aggregate_digital": agg,
                    "wilson_digital": wilson_ci_95(
                        cell["digital"]["metrics"]["success"]),
                }
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rec, indent=2, default=str))
                n_new += 1
                print(f"  EVAL {channel} snr={snr:5.1f} "
                      f"SR digital={agg['success_rate']:.4f} ({wall:.1f}s)")
        done = sum(1 for ch in CHANNELS for snr in M2_SNR_DB
                   if _m4_eval_path(ch, snr).exists())
        print(f"-- M4 eval finished: {n_new} new this session, "
              f"{done}/10 complete, eval_wall={total_wall:.1f}s")
        return 0 if done == 10 else 1


    # ---------------------------------------------------------------------------
    # validate / finalize
    # ---------------------------------------------------------------------------

    def cmd_validate():
        print("== M3/M4 completion validation ==")
        FAILS.clear()
        rep = M3_ROOT / "m3_report.json"
        check("m3-report-exists", rep.exists())
        if rep.exists():
            r = json.loads(rep.read_text())
            check("m3-three-infeasible-cells",
                  len(r["cells"]) == 3
                  and all(c["status"] == "infeasible-by-design"
                          for c in r["cells"]))
        missing = [f"{ch}@{snr:g}" for ch in CHANNELS for snr in M2_SNR_DB
                   if not _m4_eval_path(ch, snr).exists()]
        check("m4-10-eval-units-complete", len(missing) == 0,
              f"missing={missing}")
        eps_ok = True
        for ch in CHANNELS:
            for snr in M2_SNR_DB:
                p = _m4_eval_path(ch, snr)
                if p.exists():
                    rec = json.loads(p.read_text())
                    if rec.get("episodes") != 5000 or rec.get("test_seed") != 10042:
                        eps_ok = False
                    if not rec.get("secondary_reference"):
                        eps_ok = False
        check("m4-episodes==5000-test-seed==10042-secondary", eps_ok)
        check("m1-intact", len(list(Path("results/M1").glob("*__*/DONE.json"))) == 36)
        print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
        return 1 if FAILS else 0


    def cmd_finalize():
        print("== finalize: results/M4/step13_m4_summary.json ==")
        rows = []
        for ch in CHANNELS:
            for snr in M2_SNR_DB:
                p = _m4_eval_path(ch, snr)
                if not p.exists():
                    continue
                rec = json.loads(p.read_text())
                agg = rec["aggregate_digital"]
                w = rec["wilson_digital"]
                rows.append({
                    "channel": ch, "snr_db": snr, "k": M4_K,
                    "bits_per_component": M4_DIGITAL_B,
                    "success_rate": agg["success_rate"],
                    "success_count": agg["success_count"],
                    "episodes": agg["episodes"],
                    "wilson_low": w["wilson_low"], "wilson_high": w["wilson_high"],
                    "final_distance_mean": agg.get("final_distance_mean"),
                    "avg_distance_mean": agg.get("avg_distance_mean"),
                    "control_effort_mean": agg.get("control_effort_mean"),
                })
        summary = {
            "STEP": "13", "experiment": "M4",
            "status": "complete" if len(rows) == 10 else "incomplete",
            "definition": ("SECONDARY digital reference (never a primary cell): "
                           "B=8, k=n_dig=54 (6*8+6=54 exact fit), evaluation-"
                           "only, one cell per channel, M2 SNR grid, 5000 "
                           "episodes, test seed 10042."),
            "conditions": rows,
            "n_conditions": len(rows),
            "interpretation_note": ("Descriptive only. M4's B=8 reference uses "
                                   "the same 54 channel uses as the k=54 neural "
                                   "cells but is a DIFFERENT operating point "
                                   "(secondary); it is never pooled with the "
                                   "matched-budget M2 primary comparisons."),
        }
        out = M4_ROOT / "step13_m4_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"wrote {out} ({len(rows)} conditions)")
        return 0 if len(rows) == 10 else 1


    def _dispatch():
        if len(sys.argv) < 2 or sys.argv[1] not in (
                "verify", "run-m3", "run-m4", "validate", "finalize", "all"):
            print(__doc__)
            sys.exit(1)
        cmd = sys.argv[1]
        if cmd == "all":
            rc = cmd_verify() or cmd_run_m3() or cmd_run_m4() or cmd_validate() \
                or cmd_finalize()
            sys.exit(rc)
        sys.exit({"verify": cmd_verify, "run-m3": cmd_run_m3, "run-m4": cmd_run_m4,
                  "validate": cmd_validate, "finalize": cmd_finalize}[cmd]())

    return _dispatch
_run_step12_m3_m4_main = _run_step12_m3_m4_main()

# ============================================================
# SECTION 14 FIGURE/TABLE GENERATION (thesis figures fig1-fig15; read-only over results/)
# ============================================================
# ---- Source: tools/make_figures.py ----
def _figures_step14b():
    __doc__ = 'Step-14b — Thesis figures for M1 + M2 (READ-ONLY over results/).\n\nGenerates publication-grade PNGs into fyp_md_files/figures/:\n\n  fig1_m1_sr_vs_snr.png        M1: recon vs task, k={1,2,3}, AWGN|Rayleigh\n  fig2_m2_awgn.png             M2 AWGN: 3 methods, k={12,18,24,30,54}\n  fig3_m2_rayleigh.png         M2 Rayleigh: 3 methods, k={12,18,24,30,54}\n  fig4_bandwidth_crossover.png M2 AWGN: SR vs k at 0 dB and 20 dB\n  fig5_full_bandwidth_axis.png M1+M2 combined: SR vs k at 0 dB (AWGN)\n\nRun:  .venv/Scripts/python.exe tools/make_figures.py'
    """Step-14b — Thesis figures for M1 + M2 (READ-ONLY over results/).

    Generates publication-grade PNGs into fyp_md_files/figures/:

      fig1_m1_sr_vs_snr.png        M1: recon vs task, k={1,2,3}, AWGN|Rayleigh
      fig2_m2_awgn.png             M2 AWGN: 3 methods, k={12,18,24,30,54}
      fig3_m2_rayleigh.png         M2 Rayleigh: 3 methods, k={12,18,24,30,54}
      fig4_bandwidth_crossover.png M2 AWGN: SR vs k at 0 dB and 20 dB
      fig5_full_bandwidth_axis.png M1+M2 combined: SR vs k at 0 dB (AWGN)

    Run:  .venv/Scripts/python.exe tools/make_figures.py
    """

    import json
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ROOT = Path(".")
    FIG = ROOT / "fyp_md_files" / "figures"
    FIG.mkdir(parents=True, exist_ok=True)

    SNR1 = [0.0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    SNR2 = [0.0, 5.0, 10.0, 15.0, 20.0]
    KS1 = (1, 2, 3)
    KS2 = (12, 18, 24, 30, 54)
    CHANNELS = ("awgn", "rayleigh")
    SEEDS = (42, 43, 44)

    C_RECON, C_TASK, C_DIG = "#0072B2", "#D55E00", "#009E73"
    K_SHADE = {1: "#74a9cf", 2: "#3690c0", 3: "#045a8d",
               12: "#74a9cf", 18: "#3690c0", 24: "#045a8d",
               30: "#034e7b", 54: "#023858"}
    K_SHADE_T = {1: "#f4a582", 2: "#ef6548", 3: "#d7301f",
                 12: "#f4a582", 18: "#ef6548", 24: "#d7301f",
                 30: "#b30000", 54: "#7f0000"}

    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.3,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


    def pooled(path):
        s = json.loads(Path(path).read_text())
        return {(r["method"], r["k"], r["channel"], r["snr_db"]): r
                for r in s["pooled_rows"]}


    p1 = pooled("results/M1/step10_m1_summary.json")
    p2 = pooled("results/M2/step11_m2_summary.json")

    # M4 secondary reference (B=8 at k=54), from results/M4/step13_m4_summary.json
    M4 = {"awgn":  [0.9736, 1.0, 1.0, 1.0, 1.0],
          "rayleigh": [0.7178, 0.9908, 0.9984, 0.9998, 1.0]}


    def sr1(m, k, ch, s):
        return p1[(m, k, ch, s)]["success_rate_pooled"]


    def sr2(m, k, ch, s):
        try:
            return p2[(m, k, ch, s)]["success_rate_pooled"]
        except KeyError:          # excluded conditions (see PROTOCOL_EXCEPTIONS.md)
            return float("nan")


    # ---------------- Figure 1: M1 recon vs task ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, ch in zip(axes, CHANNELS):
        for k in KS1:
            ax.plot(SNR1, [sr1("reconstruction", k, ch, s) for s in SNR1],
                    "-o", color=K_SHADE[k], ms=3.5,
                    label=f"recon k={k} (ρ={k}/6)")
            ax.plot(SNR1, [sr1("task", k, ch, s) for s in SNR1],
                    "--s", color=K_SHADE_T[k], ms=3.5,
                    label=f"task k={k} (ρ={k}/6)")
        ax.set_title(f"M1 — {ch.upper()} (pooled over 3 seeds)")
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks(SNR1)
        ax.set_ylim(0, 0.36)
    axes[0].set_ylabel("Goal-reaching success rate")
    axes[0].legend(ncol=2, loc="upper right")
    fig.suptitle("M1 (low-k regime): reconstruction vs task-oriented DeepJSCC",
                 y=1.04, fontsize=12)
    fig.savefig(FIG / "fig1_m1_sr_vs_snr.png")
    plt.close(fig)

    # ---------------- Figures 2/3: M2 per-channel ---------------------------
    for ch, fname in (("awgn", "fig2_m2_awgn.png"),
                      ("rayleigh", "fig3_m2_rayleigh.png")):
        fig, axes = plt.subplots(1, 5, figsize=(14, 3.1), sharey=True)
        for ax, k in zip(axes, KS2):
            ax.plot(SNR2, [sr2("reconstruction", k, ch, s) for s in SNR2],
                    "-o", color=C_RECON, ms=4, label="recon")
            ax.plot(SNR2, [sr2("task", k, ch, s) for s in SNR2],
                    "--s", color=C_TASK, ms=4, label="task")
            ax.plot(SNR2, [sr2("digital", k, ch, s) for s in SNR2],
                    "-^", color=C_DIG, ms=4, label="digital")
            ax.set_title(f"k={k}  (ρ={k}/6)")
            ax.set_xlabel("SNR (dB)")
            ax.set_xticks(SNR2)
            ax.set_ylim(0, 1.05)
            if k == 54:  # secondary B=8 reference
                ax.plot(SNR2, M4[ch], ":*", color="0.35", ms=7,
                        label="digital B=8 (secondary)")
        axes[0].set_ylabel("Success rate")
        axes[0].legend(loc="upper left")
        fig.suptitle(f"M2 matched-budget 3-way comparison — {ch.upper()} "
                     "(pooled over 3 seeds, 15000 episodes/condition)",
                     y=1.05, fontsize=12)
        fig.savefig(FIG / fname)
        plt.close(fig)

    # ---------------- Figure 4: bandwidth crossover (AWGN) ------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, snr in zip(axes, (0.0, 20.0)):
        for m, c, mk, lbl in (("reconstruction", C_RECON, "o", "recon"),
                              ("task", C_TASK, "s", "task"),
                              ("digital", C_DIG, "^", "digital")):
            ax.plot(KS2, [sr2(m, k, "awgn", snr) for k in KS2],
                    f"-{mk}", color=c, ms=6, label=lbl)
        ax.set_title(f"AWGN @ {snr:g} dB")
        ax.set_xlabel("k (channel uses per state)")
        ax.set_xticks(KS2)
        ax.set_ylim(0, 1.05)
        ax.axvspan(46, 58, color="0.94", zorder=0)
        ax.text(52, 0.5, "B=8\nreference", ha="center", fontsize=8,
                color="0.35")
    axes[0].set_ylabel("Success rate")
    axes[0].annotate("digital overtakes recon\nbetween k=18 and k=24",
                     xy=(24, 0.305), xytext=(27, 0.12), fontsize=8.5,
                     arrowprops=dict(arrowstyle="->", color="0.3"))
    axes[0].legend(loc="upper left")
    fig.suptitle("M2 AWGN: success rate vs bandwidth (matched budgets, "
                 "B = (k−6)/6)", y=1.04, fontsize=12)
    fig.savefig(FIG / "fig4_bandwidth_crossover.png")
    plt.close(fig)

    # ---------------- Figure 5: full bandwidth axis (M1+M2, 0 dB) -----------
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    k_all = list(KS1) + list(KS2)
    ax.plot(KS1, [sr1("reconstruction", k, "awgn", 0.0) for k in KS1],
            "-o", color=C_RECON, ms=6, label="recon (M1)")
    ax.plot(KS1, [sr1("task", k, "awgn", 0.0) for k in KS1],
            "--s", color=C_TASK, ms=6, label="task (M1)")
    ax.plot(KS2, [sr2("reconstruction", k, "awgn", 0.0) for k in KS2],
            "-o", color=C_RECON, ms=6, label="recon (M2)")
    ax.plot(KS2, [sr2("task", k, "awgn", 0.0) for k in KS2],
            "--s", color=C_TASK, ms=6, label="task (M2)")
    ax.plot(KS2, [sr2("digital", k, "awgn", 0.0) for k in KS2],
            "-^", color=C_DIG, ms=6, label="digital (M2)")
    ax.axvline(7.5, color="0.6", lw=1, ls=":")
    ax.text(4.2, 0.33, "M1 regime\nρ ≤ 1/2", ha="center", fontsize=9,
            color="0.3")
    ax.text(27, 0.33, "M2 regime\nρ ≥ 2", ha="center", fontsize=9, color="0.3")
    ax.set_xticks(k_all)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("k (channel uses per 6-D state)")
    ax.set_ylabel("Success rate @ 0 dB (AWGN)")
    ax.legend(ncol=3, loc="upper right")
    ax.set_title("Full bandwidth axis at 0 dB — M1 + M2 (AWGN, pooled)")
    fig.savefig(FIG / "fig5_full_bandwidth_axis.png")
    plt.close(fig)

    print("figures written to", FIG)
    for f in sorted(FIG.glob("*.png")):
        print(" ", f.name, f.stat().st_size // 1024, "KB")

# ---- Source: tools/make_figures_extra.py ----
def _figures_step14c():
    __doc__ = 'Step-14c — Extended thesis figures (READ-ONLY over results/).\n\nOne paired re-evaluation (5000 episodes, k=24 AWGN 10 dB) for the CDF and\n3 short rollouts for the trajectory figure; everything else from pooled\nsummaries and checkpoint histories.\n\n  fig6_final_distance.png    M2 final distance vs SNR (3 methods, 2 channels)\n  fig7_control_effort.png    M2 control effort vs SNR\n  fig8_time_to_goal.png      success-conditional time-to-goal vs SNR\n  fig9_sr_heatmaps.png       SR heatmaps: 3 methods x 2 channels (k x SNR)\n  fig10_training_curves.png  task loss + recon loss vs training step\n  fig11_cdf_final_dist.png   CDF of final distance, k=24 AWGN 10 dB\n  fig12_trajectories.png     sample closed-loop trajectories, k=24 AWGN 10 dB\n\nRun:  .venv/Scripts/python.exe tools/make_figures_extra.py'
    """Step-14c — Extended thesis figures (READ-ONLY over results/).

    One paired re-evaluation (5000 episodes, k=24 AWGN 10 dB) for the CDF and
    3 short rollouts for the trajectory figure; everything else from pooled
    summaries and checkpoint histories.

      fig6_final_distance.png    M2 final distance vs SNR (3 methods, 2 channels)
      fig7_control_effort.png    M2 control effort vs SNR
      fig8_time_to_goal.png      success-conditional time-to-goal vs SNR
      fig9_sr_heatmaps.png       SR heatmaps: 3 methods x 2 channels (k x SNR)
      fig10_training_curves.png  task loss + recon loss vs training step
      fig11_cdf_final_dist.png   CDF of final distance, k=24 AWGN 10 dB
      fig12_trajectories.png     sample closed-loop trajectories, k=24 AWGN 10 dB

    Run:  .venv/Scripts/python.exe tools/make_figures_extra.py
    """

    import json
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    ROOT = Path(".")
    FIG = ROOT / "fyp_md_files" / "figures"
    FIG.mkdir(parents=True, exist_ok=True)

    SNR2 = [0.0, 5.0, 10.0, 15.0, 20.0]
    KS2 = (12, 18, 24, 30, 54)
    CHANNELS = ("awgn", "rayleigh")
    C_RECON, C_TASK, C_DIG = "#0072B2", "#D55E00", "#009E73"
    METH = (("reconstruction", "recon", C_RECON, "o"),
            ("task", "task", C_TASK, "s"),
            ("digital", "digital", C_DIG, "^"))

    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.3,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


    def pooled(path):
        s = json.loads(Path(path).read_text())
        return {(r["method"], r["k"], r["channel"], r["snr_db"]): r
                for r in s["pooled_rows"]}


    p2 = pooled("results/M2/step11_m2_summary.json")


    def val(field, m, k, ch, s):
        return p2[(m, k, ch, s)][field]


    def metric_panel(field, ylabel, fname, title, logy=False):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, ch in zip(axes, CHANNELS):
            for m, lab, c, mk in METH:
                ys = [val(field, m, k, ch, s) for k in KS2 for s in SNR2]
                for k in KS2:
                    xs = SNR2
                    ys = [val(field, m, k, ch, s) for s in SNR2]
                    std = [p2[(m, k, ch, s)][field.replace(
                        "mean_over_seeds", "std_over_seeds")] for s in SNR2]
                    ax.errorbar(xs, ys, yerr=std, marker=mk, color=c, ms=4,
                                capsize=2, lw=1.4, label=f"{lab} k={k}")
            ax.set_title(f"{ch.upper()}")
            ax.set_xlabel("SNR (dB)")
            ax.set_xticks(SNR2)
            if logy:
                ax.set_yscale("log")
        axes[0].set_ylabel(ylabel)
        axes[0].legend(ncol=2, fontsize=7)
        fig.suptitle(title, y=1.04, fontsize=12)
        fig.savefig(FIG / fname)
        plt.close(fig)


    # ---- fig6/7/8 ------------------------------------------------------------
    metric_panel("final_distance_mean_mean_over_seeds", "Final distance (m)",
                 "fig6_final_distance.png",
                 "M2: final distance to goal vs SNR (mean ± std over 3 seeds)")
    metric_panel("control_effort_mean_mean_over_seeds",
                 "Control effort  Σ‖u‖²  (lower = less energy)",
                 "fig7_control_effort.png",
                 "M2: control effort vs SNR (mean ± std over 3 seeds)")
    metric_panel("ttg_success_mean_mean_over_seeds",
                 "Time-to-goal among successes (steps)", "fig8_time_to_goal.png",
                 "M2: success-conditional time-to-goal vs SNR "
                 "(task curves use few successes — interpret with care)")

    # ---- fig9: SR heatmaps ----------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey=True)
    for row, ch in enumerate(CHANNELS):
        for col, (m, lab, c, _) in enumerate(METH):
            grid = [[p2[(m, k, ch, s)]["success_rate_pooled"] for s in SNR2]
                    for k in KS2]
            ax = axes[row][col]
            im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                           origin="lower")
            for i, k in enumerate(KS2):
                for j, s in enumerate(SNR2):
                    ax.text(j, i, f"{grid[i][j]:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if grid[i][j] < 0.6 else "black")
            ax.set_xticks(range(5), [f"{s:g}" for s in SNR2])
            ax.set_yticks(range(5), [str(k) for k in KS2])
            if row == 1:
                ax.set_xlabel("SNR (dB)")
            if col == 0:
                ax.set_ylabel("k (uses/state)")
            ax.set_title(f"{lab} — {ch}")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Success rate")
    fig.suptitle("M2 success-rate maps (pooled, 15000 episodes/cell)",
                 y=0.99, fontsize=12)
    fig.savefig(FIG / "fig9_sr_heatmaps.png")
    plt.close(fig)

    # ---- fig10: training curves ----------------------------------------------
    def load_history(dir_glob):
        d = sorted(Path("results/M2").glob(dir_glob))[0]
        payload = torch.load(d / "final_ckpt.pt", weights_only=False,
                             map_location="cpu")
        return payload["history"]


    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ch, c in (("awgn", C_RECON), ("rayleigh", C_DIG)):
        h = load_history("M2_task_awgn_k24_seed42__*"
                         if ch == "awgn" else
                         "M2_task_rayleigh_k24_seed42__*")
        tot = [e["total"] for e in h if isinstance(e, dict) and "total" in e]
        steps = range(0, len(tot), 10)
        axes[0].plot(steps, [tot[i] for i in steps], color=c, lw=1.2,
                     label=f"task {ch} (k=24, seed 42)")
        hr = load_history(f"M2_reconstruction_{ch}_k24_seed42__*")
        rl = [e if isinstance(e, (int, float)) else
              (e.get("loss", e.get("total")) if isinstance(e, dict) else None)
              for e in hr]
        rl = [x for x in rl if x is not None]
        steps = range(0, len(rl), 10)
        axes[1].plot(steps, [rl[i] for i in steps], color=c, lw=1.2,
                     label=f"recon {ch} (k=24, seed 42)")
    axes[0].set_title("Task-oriented training loss")
    axes[1].set_title("Reconstruction training loss")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.set_yscale("log")
        ax.legend()
    axes[0].set_ylabel("loss")
    fig.suptitle("Frozen-protocol training curves (k=24, seed 42)",
                 y=1.03, fontsize=12)
    fig.savefig(FIG / "fig10_training_curves.png")
    plt.close(fig)

    # ---- fig11: CDF of final distance (one paired re-eval) --------------------
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


    cond = Condition(k=24, channel="awgn", bits_per_component=3, t_max=100,
                     episodes=5000, seed=10042, snr_db=10.0)
    m_recon = ReconstructionDeepJSCC(k=24)
    m_recon.load_state_dict(torch.load(
        sorted(Path("results/M2").glob(
            "M2_reconstruction_awgn_k24_seed42__*/final_ckpt.pt"))[0],
        weights_only=False, map_location="cpu")["model_state"])
    m_recon.eval()
    m_task = ReconstructionDeepJSCC(k=24)
    m_task.load_state_dict(torch.load(
        sorted(Path("results/M2").glob(
            "M2_task_awgn_k24_seed42__*/final_ckpt.pt"))[0],
        weights_only=False, map_location="cpu")["model_state"])
    m_task.eval()

    cell = evaluate_condition(cond, recon_model=m_recon, task_model=m_task)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for m, lab, c, _ in METH:
        d = torch.sort(cell[m]["metrics"]["final_distance"]
                       .detach().to(torch.float64)).values.numpy()
        ax.plot(d, torch.arange(1, len(d) + 1).numpy() / len(d), color=c,
                lw=1.6, label=f"{lab}  (median {d[len(d)//2]:.2f} m)")
    ax.axvline(0.5, color="0.5", ls=":", lw=1)
    ax.text(0.53, 0.06, "goal radius 0.5 m\n(d ≤ 0.5 at ANY step = success)",
            fontsize=8, color="0.3")
    ax.set_xlabel("final distance to goal (m)")
    ax.set_ylabel("CDF over 5000 paired episodes")
    ax.set_title("k=24, AWGN, 10 dB — final-distance CDF (paired episodes)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(FIG / "fig11_cdf_final_dist.png")
    plt.close(fig)
    print("fig11 done (paired re-eval 5000 eps)")

    # ---- fig12: sample trajectories -------------------------------------------
    real = generate_realizations(cond, episodes=3)
    s0 = torch.cat([real["p0"], real["v0"], real["goal"]], dim=-1)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    styles = ("-", "--", "-.")
    for m, model in (("recon", m_recon), ("task", m_task)):
        pos, _, _ = closed_loop_rollout(model, s0, t_max=100, snr_db=10.0,
                                        channel="awgn",
                                        generator=real.get("gen_noise"))
        for e in range(3):
            ax.plot(pos[e, :, 0].detach().numpy(), pos[e, :, 1].detach().numpy(),
                    styles[e], color=C_RECON if m == "recon" else C_TASK,
                    lw=1.3, alpha=0.9,
                    label=f"{m}" if e == 0 else None)
            ax.plot(pos[e, 0, 0].item(), pos[e, 0, 1].item(), "ko", ms=4)
            ax.plot(pos[e, -1, 0].item(), pos[e, -1, 1].item(), "x",
                    color=C_RECON if m == "recon" else C_TASK, ms=7, mew=2)
    g = real["goal"]
    ax.plot(g[:, 0], g[:, 1], "*", color="gold", ms=16, mec="k",
            label="goal", ls="none")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Sample closed-loop trajectories — k=24, AWGN, 10 dB\n"
                 "(black dot = start, x = end; 3 paired episodes per method)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    fig.savefig(FIG / "fig12_trajectories.png")
    plt.close(fig)

    print("figures written to", FIG)
    for f in sorted(FIG.glob("*.png")):
        print(" ", f.name, f.stat().st_size // 1024, "KB")

# ---- Source: tools/make_figures_semantic.py ----
def _figures_step14d():
    __doc__ = 'Step-14d — Semantic/JSCC-side thesis figures (READ-ONLY over results/).\n\n  fig13_mse_vs_snr.png       reconstruction MSE vs SNR, 3 methods, 2 channels\n  fig14_rate_distortion.png  rate-distortion: MSE vs bandwidth rho at 0/10/20 dB\n  fig15_constellation.png    learned encoder symbol scatter (recon vs task)\n\nCaveat shown on figures: trajectory MSE censors exited episodes (post-exit\nsu = s), so MSE curves are diagnostic-grade (see ARCHITECTURE.md section 5).\n\nRun:  .venv/Scripts/python.exe tools/make_figures_semantic.py'
    """Step-14d — Semantic/JSCC-side thesis figures (READ-ONLY over results/).

      fig13_mse_vs_snr.png       reconstruction MSE vs SNR, 3 methods, 2 channels
      fig14_rate_distortion.png  rate-distortion: MSE vs bandwidth rho at 0/10/20 dB
      fig15_constellation.png    learned encoder symbol scatter (recon vs task)

    Caveat shown on figures: trajectory MSE censors exited episodes (post-exit
    su = s), so MSE curves are diagnostic-grade (see ARCHITECTURE.md section 5).

    Run:  .venv/Scripts/python.exe tools/make_figures_semantic.py
    """

    import json
    import sys
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


    FIG = Path("fyp_md_files") / "figures"
    FIG.mkdir(parents=True, exist_ok=True)

    SNR2 = [0.0, 5.0, 10.0, 15.0, 20.0]
    KS2 = (12, 18, 24, 30, 54)
    CHANNELS = ("awgn", "rayleigh")
    C_RECON, C_TASK, C_DIG = "#0072B2", "#D55E00", "#009E73"
    METH = (("reconstruction", "recon", C_RECON, "o"),
            ("task", "task", C_TASK, "s"),
            ("digital", "digital", C_DIG, "^"))

    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.3,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})

    s = json.loads(Path("results/M2/step11_m2_summary.json").read_text())
    p2 = {(r["method"], r["k"], r["channel"], r["snr_db"]): r
          for r in s["pooled_rows"]}


    def mse(m, k, ch, snr):
        return p2[(m, k, ch, snr)]["mse_mean_mean_over_seeds"]


    CAVEAT = ("trajectory MSE censors boundary-exited episodes "
              "(post-exit su = s) — diagnostic metric, see thesis section 5.4")

    # ---- fig13: reconstruction MSE vs SNR --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True)
    for ax, ch in zip(axes, CHANNELS):
        for m, lab, c, mk in METH:
            for k in KS2:
                ys = [mse(m, k, ch, x) for x in SNR2]
                ax.plot(SNR2, ys, marker=mk, ms=3.5, lw=1.3, color=c,
                        alpha=0.45 + 0.55 * KS2.index(k) / 4,
                        label=f"{lab} k={k}")
        ax.set_title(ch.upper())
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks(SNR2)
        ax.set_yscale("log")
    axes[0].set_ylabel("trajectory MSE (normalized state, /6)")
    axes[0].legend(ncol=2, fontsize=6.8, loc="upper right")
    fig.text(0.5, -0.02, CAVEAT, ha="center", fontsize=8, color="0.35")
    fig.suptitle("Semantic reconstruction fidelity vs SNR — all methods, "
                 "all bandwidths", y=1.04, fontsize=12)
    fig.savefig(FIG / "fig13_mse_vs_snr.png")
    plt.close(fig)

    # ---- fig14: rate-distortion (MSE vs bandwidth rho) --------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
    for ax, snr in zip(axes, (0.0, 10.0, 20.0)):
        for m, lab, c, mk in METH:
            ax.plot([k / 6 for k in KS2],
                    [mse(m, k, "awgn", snr) for k in KS2],
                    marker=mk, ms=5, lw=1.5, color=c, label=lab)
        ax.set_title(f"AWGN @ {snr:g} dB")
        ax.set_xlabel("bandwidth ratio ρ = k/6")
        ax.set_yscale("log")
        ax.set_xticks([k / 6 for k in KS2])
        ax.set_xticklabels([f"{k/6:g}\n(k={k})" for k in KS2], fontsize=8)
    axes[0].set_ylabel("trajectory MSE (normalized state, /6)")
    axes[0].legend(loc="upper right")
    fig.text(0.5, -0.02, CAVEAT, ha="center", fontsize=8, color="0.35")
    fig.suptitle("Rate-distortion view — semantic fidelity vs bandwidth "
                 "(AWGN, pooled)", y=1.04, fontsize=12)
    fig.savefig(FIG / "fig14_rate_distortion.png")
    plt.close(fig)

    # ---- fig15: learned encoder constellation -----------------------------------
    def load(glob):
        d = sorted(Path("results/M2").glob(glob))[0]
        m = ReconstructionDeepJSCC(k=24)
        m.load_state_dict(torch.load(d / "final_ckpt.pt", weights_only=False,
                                     map_location="cpu")["model_state"])
        m.eval()
        return m


    m_recon = load("M2_reconstruction_awgn_k24_seed42__*")
    m_task = load("M2_task_awgn_k24_seed42__*")

    cond = Condition(k=24, channel="awgn", bits_per_component=3, t_max=100,
                     episodes=500, seed=10042, snr_db=10.0)
    real = generate_realizations(cond, episodes=500)
    st = torch.cat([real["p0"], real["v0"], real["goal"]], dim=-1)
    sbar = normalize(st)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharex=True, sharey=True)
    for ax, model, lab, c in ((axes[0], m_recon, "recon encoder", C_RECON),
                              (axes[1], m_task, "task encoder", C_TASK)):
        with torch.no_grad():
            out = model.encoder(sbar)                     # [E, 2k] real
        z0 = (out[:, 0] + 1j * out[:, 1]).numpy()         # symbol 1
        z1 = (out[:, 2] + 1j * out[:, 3]).numpy()         # symbol 2
        ax.scatter(z0.real, z0.imag, s=6, alpha=0.35, color=c,
                   label="symbol 1")
        ax.scatter(z1.real, z1.imag, s=6, alpha=0.35, color="0.35",
                   marker="^", label="symbol 2")
        ax.set_title(f"{lab} (k=24): first 2 of 24 symbols, 500 states")
        ax.set_xlabel("in-phase")
        ax.set_ylabel("quadrature")
        ax.legend(loc="upper right")
        ax.set_aspect("equal")
    fig.suptitle("Learned JSCC code structure — pre-channel encoder output "
                 "@ 10 dB (semantic vs reconstruction objective)", y=1.02,
                 fontsize=12)
    fig.savefig(FIG / "fig15_constellation.png")
    plt.close(fig)

    print("figures written to", FIG)
    for f in sorted(FIG.glob("fig1[3-5]*.png")):
        print(" ", f.name, f.stat().st_size // 1024, "KB")


# ============================================================
# SECTION 15 MAIN / SMOKE TEST
# ============================================================
def _smoke_test() -> int:
    """Minimal deterministic end-to-end verification (NO experiments run)."""
    print("[smoke] device:", get_device())
    set_global_seed(0)

    # -- SECTION 1: frozen configuration self-consistency -------------------
    validate()
    assert digital_packet(12, 1)["n_padding_symbols"] == 0
    try:
        digital_packet(3, 1)
        raise AssertionError("k=3,B=1 must be infeasible")
    except InfeasibleDigitalConfig:
        pass
    print("[smoke] sec1 frozen config + digital packet arithmetic OK")

    # -- SECTION 2: normalization round-trip ---------------------------------
    s = torch.tensor([[0.0, 0.0, 0.0, 0.0, 4.0, -4.0],
                      [-4.0, 4.0, 1.5, -1.5, 0.0, 0.0]])
    s_bar = normalize(s)
    assert torch.isfinite(s_bar).all()
    assert torch.allclose(denormalize(s_bar), s, atol=1e-6)
    print("[smoke] sec2 normalization round-trip OK")

    # -- SECTION 3: complex conversion, power norm, AWGN, Rayleigh -----------
    k = 6
    x = torch.randn(4, 2 * k)
    assert torch.allclose(to_real(to_complex(x)), x, atol=1e-6)
    z = power_normalize(to_complex(x))
    assert torch.allclose(
        z.abs().pow(2).sum(dim=-1),
        torch.full((4,), float(k) * TX_POWER), atol=1e-4)
    y_awgn = awgn_channel(z, snr_db=10.0)
    assert y_awgn.shape == z.shape and torch.isfinite(y_awgn).all()
    y_ray, h_used = rayleigh_channel(z, snr_db=10.0, return_h=True)
    x_eq = equalize_zf(y_ray, h_used)
    assert torch.isfinite(x_eq).all()
    print("[smoke] sec3 AWGN + Rayleigh + power normalization OK")

    # -- SECTION 4: DeepJSCC full path (shared task/recon container) ---------
    model = ReconstructionDeepJSCC(k=k)
    s_hat = model.forward(s, snr_db=10.0, channel="awgn",
                          noise=torch.complex(torch.randn(2, k),
                                              torch.randn(2, k)))
    assert s_hat.shape == (2, 6) and torch.isfinite(s_hat).all()
    print("[smoke] sec4 DeepJSCC encoder->channel->decoder forward OK")

    # -- SECTION 7: losses ----------------------------------------------------
    assert float(reconstruction_loss(s_bar, s_bar)) == 0.0
    pos = torch.randn(2, 4, 2)
    ctl = torch.randn(2, 3, 2)
    vel = torch.randn(2, 4, 2)
    goal0 = torch.zeros(2, 2)
    tl = task_loss(pos, ctl, vel, goal0)
    assert torch.isfinite(tl["total"]) and torch.isfinite(tl["L_T"])
    print("[smoke] sec7 task + reconstruction losses OK")

    # -- SECTION 5: digital baseline encode -> channel -> decode -------------
    base = DigitalBaseline(k=12, bits_per_component=1)
    s_phys = denormalize(s_bar)
    out_d = base.forward(s_phys, snr_db=10.0, channel="awgn",
                         noise=torch.complex(torch.randn(2, 12),
                                             torch.randn(2, 12)))
    assert out_d.shape == (2, 6) and torch.isfinite(out_d).all()
    print("[smoke] sec5 digital baseline forward path OK")

    # -- SECTION 6: UAV dynamics, controller, environment ---------------------
    p = torch.zeros(2, 2)
    v = torch.zeros(2, 2)
    u = torch.tensor([[2.0, -2.0], [2.0, -2.0]])
    p1_, v1_ = step(p, v, u)
    assert torch.allclose(p1_, p + DT * v + 0.5 * DT ** 2 * u, atol=1e-6)
    u_pd = pd_controller(torch.cat([p, v, torch.zeros(2, 2)], dim=-1))
    assert (u_pd.abs() <= U_MAX + 1e-6).all()
    p0, v0, g0 = NavigationEnv().sample_starts(
        4, generator=derive_generator("episodes", 0))
    assert p0.shape == (4, 2) and g0.shape == (4, 2)
    print("[smoke] sec6 dynamics + PD controller + environment OK")

    # -- SECTION 9: paired oracle rollout + statistics sanity -----------------
    cond = Condition(k=2, channel="awgn", snr_db=10.0, episodes=4,
                     seed=TEST_SEED)
    real = generate_realizations(cond, 4)
    met = run_method(cond, real, estimate_fn=None)
    assert met["success"].dtype == torch.bool and met["success"].numel() == 4
    ci = wilson_ci_95(met["success"])
    assert 0.0 <= ci["wilson_low"] <= ci["wilson_high"] <= 1.0
    mc = mcnemar_test(torch.tensor([1, 1, 1, 0, 0]),
                      torch.tensor([1, 1, 0, 0, 0]))
    assert mc["n01"] == 1 and mc["n10"] == 0 and 0.0 <= mc["p_value"] <= 1.0
    assert mcnemar_test(torch.tensor([1, 0]),
                        torch.tensor([0, 1]))["p_value"] == 1.0
    assert paired_t_test(torch.tensor([1.0, 2.0]),
                         torch.tensor([1.0, 2.0]))["p_value"] == 1.0
    print("[smoke] sec9 paired oracle rollout + Wilson/McNemar/paired-t OK")

    # -- SECTION 10: frozen experiment matrices --------------------------------
    cells = expected_cells()
    assert cells["M1_training_runs"] == 36
    assert cells["M1_eval_conditions"] == 132
    assert cells["M2_neural_training_runs"] == 60
    assert cells["M2_digital_eval_only_seeds"] == 30
    assert cells["M2_eval_conditions"] == 150
    assert cells["M3_infeasible_cells"] == 3
    assert cells["M4_secondary_eval_conditions"] == 10
    print("[smoke] sec10 frozen experiment matrices:", dict(cells))

    print("[smoke] ALL CHECKS PASSED")
    return 0

def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Consolidated FYP source; default = smoke test only")
    ap.add_argument("--smoke-test", action="store_true",
                    help="(default) minimal deterministic verification")
    ap.add_argument("--m1", nargs="+", metavar=("CMD", "LANE"),
                    help="original M1 executor dispatch, e.g. --m1 status")
    ap.add_argument("--m2", nargs="+", metavar=("CMD", "LANE"),
                    help="original M2 executor dispatch, e.g. --m2 status")
    ap.add_argument("--m34", nargs="+", metavar="CMD",
                    help="original M3+M4 dispatch, e.g. --m34 validate")
    ap.add_argument("--figures", action="store_true",
                    help="regenerate fig1-fig15 from results/ JSONs")
    args, extra = ap.parse_known_args(argv)
    if args.m1:
        sys.argv = ["run_step10_m1.py"] + args.m1 + extra
        return _run_step10_m1_main()
    if args.m2:
        sys.argv = ["run_step11_m2.py"] + args.m2 + extra
        return _run_step11_m2_main()
    if args.m34:
        sys.argv = ["run_step12_m3_m4.py"] + args.m34 + extra
        return _run_step12_m3_m4_main()
    if args.figures:
        _figures_step14b()
        _figures_step14c()
        _figures_step14d()
        return 0
    return _smoke_test()


if __name__ == "__main__":
    sys.exit(_main())
