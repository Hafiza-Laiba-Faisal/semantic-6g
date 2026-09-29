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

from __future__ import annotations

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
