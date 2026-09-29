"""Reconstruction loss (frozen, audit 1.1 row 13; item 7).

    L_recon = (1/6) * || s_hat_bar - s_bar ||^2

Computed per sample over the 6 NORMALIZED-state components, then averaged
over the batch. No navigation/task terms; the controller and UAV dynamics
are NOT part of this baseline's training graph.
"""

from __future__ import annotations

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
