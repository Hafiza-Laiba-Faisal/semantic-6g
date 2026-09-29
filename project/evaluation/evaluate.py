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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from project import config
from project.baselines.digital import DigitalBaseline
from project.data.normalization import normalize
from project.evaluation.metrics import per_episode_metrics
from project.models.reconstruction_jscc import ReconstructionDeepJSCC
from project.uav.controller import pd_controller
from project.uav.dynamics import step as dynamics_step
from project.uav.environment import NavigationEnv


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
    from project.evaluation.metrics import aggregate_episode_metrics
    return aggregate_episode_metrics(per_ep)
