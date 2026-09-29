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

from __future__ import annotations

import torch
from torch import nn

from project import config
from project.channels.complex_utils import to_complex


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
