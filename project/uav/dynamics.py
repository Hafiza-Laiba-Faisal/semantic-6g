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

from __future__ import annotations

import torch

from project import config


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
