"""DeepJSCC decoder (mirror of the encoder, audit 3 architecture table).

    g_phi : C^k -> R^6, implemented on the real stacked representation as

        Linear(2k, 128) -> PReLU -> Linear(128, 128) -> PReLU -> Linear(128, 6)

Input: equalized complex symbols (or their real stacked form) using the
SAME stacked I/Q ordering as the encoder. Output: the 6-D NORMALIZED state
estimate (no output activation - the frozen specification has none);
denormalization to physical units is done by the existing normalization
module, never inside the decoder.
"""

from __future__ import annotations

import torch
from torch import nn

from project import config
from project.channels.complex_utils import to_real


class DeepJSCCDecoder(nn.Module):
    """MLP decoder mapping 2k real inputs to the 6-D normalized state."""

    def __init__(
        self,
        k: int = 3,
        state_dim: int = config.STATE_DIM,
        hidden_dims: tuple = config.HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        h1, h2 = hidden_dims
        self.k = k
        self.backbone = nn.Sequential(
            nn.Linear(2 * k, h1),
            nn.PReLU(),
            nn.Linear(h1, h2),
            nn.PReLU(),
            nn.Linear(h2, state_dim),
        )

    def forward(self, y_eq) -> torch.Tensor:
        """Equalized symbols (complex ``[..., k]`` or real ``[..., 2k]``)
        -> normalized state estimate ``[..., 6]``."""
        return self.backbone(to_real(y_eq))
