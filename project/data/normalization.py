"""Frozen state normalization (audit 1.1 row 5, spec section 6).

For every component i with declared range [a_i, b_i]:

    s_bar_i = 2 * (clip(s_i, a_i, b_i) - a_i) / (b_i - a_i) - 1

The inverse is the exact algebraic inverse on the in-range set:

    s_i = a_i + (s_bar_i + 1) / 2 * (b_i - a_i),   then clipped to [a_i, b_i]

Ranges come from ``project.config`` only. They are never derived from data
and never refit; this is a hard fairness rule of the project.
"""

from __future__ import annotations

from typing import Sequence

import torch

from project import config


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