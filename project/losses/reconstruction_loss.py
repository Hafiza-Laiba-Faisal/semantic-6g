"""Reconstruction loss (frozen, audit 1.1 row 13; item 7).

    L_recon = (1/6) * || s_hat_bar - s_bar ||^2

Computed per sample over the 6 NORMALIZED-state components, then averaged
over the batch. No navigation/task terms; the controller and UAV dynamics
are NOT part of this baseline's training graph.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def reconstruction_loss(
    s_hat_bar: torch.Tensor,
    s_bar: torch.Tensor,
    component_weights: Optional[Tuple[float, ...]] = None,
) -> torch.Tensor:
    """Frozen per-sample (1/6)*||s_hat_bar - s_bar||^2, batch-averaged.

    Both arguments must be NORMALIZED states (last dim 6). Per-sample
    values are computed first, then the mean over the batch is returned.

    Wave-A amendment (non-frozen, opt-in): passing ``component_weights``
    computes the task-aware weighted variant
    ``L = sum_j w_j (ŝ̄_j - s̄_j)^2 / sum_j w_j``, batch-averaged. The
    default (None) is BITWISE-IDENTICAL to the frozen uniform loss.
    """
    if s_hat_bar.shape != s_bar.shape:
        raise ValueError(
            f"shape mismatch: {tuple(s_hat_bar.shape)} vs {tuple(s_bar.shape)}")
    if s_hat_bar.shape[-1] != 6:
        raise ValueError(
            f"loss is defined on the 6-D normalized state, got last dim "
            f"{s_hat_bar.shape[-1]}")
    sq = (s_hat_bar - s_bar).pow(2)
    if component_weights is None:
        per_sample = sq.sum(dim=-1) / 6.0
    else:
        if len(component_weights) != sq.shape[-1]:
            raise ValueError(
                f"component_weights must have {sq.shape[-1]} entries, "
                f"got {len(component_weights)}")
        w = torch.as_tensor(component_weights, dtype=sq.dtype,
                            device=sq.device)
        per_sample = (sq * w).sum(dim=-1) / w.sum()
    return per_sample.mean()
