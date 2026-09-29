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

from __future__ import annotations

from typing import Dict, List, Union

import torch

from project import config


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
