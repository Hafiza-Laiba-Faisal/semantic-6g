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

from __future__ import annotations

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
        f"  test episodes      = {N_TEST_EPISODES} (smoke {N_TEST_EPISODES_SMOKE})",
        f"  device             = {get_device()}",
    ]
    return "\n".join(lines)


validate()

if __name__ == "__main__":
    print(describe())
