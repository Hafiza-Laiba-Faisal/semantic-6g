"""Reconstruction-oriented DeepJSCC baseline: complete communication path.

Frozen pipeline (audit 1.1 rows 6-9, 13; item 6):

    s -> normalize -> encoder -> [B,2k] real -> [B,k] complex
      -> power normalization (shared Step-3 function)
      -> channel (shared Step-3 AWGN / block-Rayleigh)
      -> perfect-CSI ZF equalization (Rayleigh)
      -> decoder -> s_hat_bar (normalized) -> denormalize

* Channel uses: k complex symbols (never 2k).
* The model REUSES the verified Step-3 channel functions and power
  normalization; it duplicates none of them.
* Gradients flow through the whole path (the channels are ordinary
  differentiable tensor ops); noise/fading are drawn from a torch.Generator
  and enter the graph as constants.
"""

from __future__ import annotations

from typing import Optional

import torch

from project import config
from project.channels.awgn import awgn_channel
from project.channels.complex_utils import power_normalize
from project.channels.rayleigh import equalize_zf, rayleigh_channel
from project.data.normalization import denormalize, normalize
from project.models.deepjscc_decoder import DeepJSCCDecoder
from project.models.deepjscc_encoder import DeepJSCCEncoder


class ReconstructionDeepJSCC(torch.nn.Module):
    """s -> encode -> power norm -> channel -> equalize -> decode -> s_hat."""

    def __init__(self, k: int = 3) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.encoder = DeepJSCCEncoder(k=k)
        self.decoder = DeepJSCCDecoder(k=k)

    # ------------------------------------------------------------------
    # building blocks (exposed for the verification tests)
    # ------------------------------------------------------------------

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        """Physical state -> power-normalized complex codeword [B, k]."""
        s_bar = normalize(s)
        z_tilde = self.encoder.encode_complex(s_bar)
        return power_normalize(z_tilde, k=self.k)

    def decode(self, y_eq: torch.Tensor) -> torch.Tensor:
        """Equalized symbols -> PHYSICAL state estimate [B, 6]."""
        s_bar_hat = self.decoder(y_eq)
        return denormalize(s_bar_hat)

    # ------------------------------------------------------------------
    # full path
    # ------------------------------------------------------------------

    def forward(
        self,
        s: torch.Tensor,
        snr_db: float,
        channel: str = "awgn",
        generator: Optional[torch.Generator] = None,
        denormalize_output: bool = True,
    ) -> torch.Tensor:
        """One transmission of the state through the frozen pipeline.

        Parameters
        ----------
        s : physical state ``[B, 6]``.
        snr_db : configured per-symbol SNR (P / sigma^2).
        channel : ``"awgn"`` or ``"rayleigh"`` (block fading + perfect-CSI
            ZF equalization, the frozen main-experiment convention).
        generator : RNG for reproducible noise/fading.
        denormalize_output : return physical units (True) or the raw
            normalized estimate (False; used by the loss path).
        """
        if channel not in ("awgn", "rayleigh"):
            raise ValueError(f"unknown channel {channel!r}")
        z = self.encode(s)
        if channel == "awgn":
            y = awgn_channel(z, snr_db, generator=generator)
            y_eq = y
        else:
            y, h = rayleigh_channel(z, snr_db, generator=generator,
                                    return_h=True)
            y_eq = equalize_zf(y, h)
        s_bar_hat = self.decoder(y_eq)
        if denormalize_output:
            return denormalize(s_bar_hat)
        return s_bar_hat

    def forward_losses(
        self,
        s: torch.Tensor,
        snr_db: float,
        channel: str = "awgn",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return the NORMALIZED estimate for the frozen loss (item 7/8).

        The frozen reconstruction objective is defined on the normalized
        state; computing it from a denormalized output would be an
        inconsistent normalization (hard stop condition), so this helper
        is the canonical training path.
        """
        return self.forward(s, snr_db, channel=channel, generator=generator,
                            denormalize_output=False)
