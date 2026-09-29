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

from __future__ import annotations

import torch

from project import config


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
