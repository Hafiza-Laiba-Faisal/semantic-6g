"""Slow (block) Rayleigh fading channel with perfect-CSI equalization.

Frozen model (audit 1.1 rows 10-11):

    y = h * z + n,      h ~ CN(0, 1),      n ~ CN(0, sigma^2 I)

* ``h`` is a SCALAR per codeword: one coefficient shared by all k symbols
  of one transmitted codeword (block/slow fading), independent across
  codewords. This is exactly the DeepJSCC slow-Rayleigh convention.
* E[|h|^2] = 1; |h| is Rayleigh, phase uniform. h is NEVER renormalized
  after sampling (verification item 9).
* Perfect receiver CSI (main experiment): zero-forcing equalization
  ``y_eq = y / h`` with a numerical guard for extremely small |h|:

      y_eq = y * conj(h) / max(|h|^2, eps_eq)

  which equals exact ZF whenever |h|^2 >> eps_eq and stays finite
  otherwise. The same equalizer is used by the digital baseline (fairness).
* Noise is drawn by the shared sampler; the SNR axis is the AVERAGE SNR
  P / sigma^2 (per codeword the instantaneous SNR is |h|^2 P / sigma^2).
"""

from __future__ import annotations

from typing import Optional

import torch

from project import config
from project.channels.complex_utils import (
    check_codeword,
    sample_complex_gaussian,
    snr_db_to_sigma2,
)


def sample_fading(
    batch: int,
    generator: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """One complex scalar h ~ CN(0,1) per codeword; shape ``[B, 1]``.

    The trailing singleton dimension broadcasts over the k symbols of a
    codeword, which IS the block-fading convention.
    """
    h = sample_complex_gaussian(
        torch.Size((batch, 1)), 1.0, generator=generator, device=device,
        dtype=dtype, complex_out=True)
    return h


def rayleigh_channel(
    z: torch.Tensor,
    snr_db: float,
    generator: Optional[torch.Generator] = None,
    power: float = config.TX_POWER,
    return_h: bool = False,
    noise: Optional[torch.Tensor] = None,
    h: Optional[torch.Tensor] = None,
):
    """Transmit one codeword batch through slow Rayleigh fading + AWGN.

    ``noise`` (complex ``[B, k]`` or stacked real matching ``z``) and
    ``h`` (complex ``[B, 1]``) may be pre-generated for paired evaluation;
    when supplied they are used verbatim and no sampling occurs.
    """
    check_codeword(z)
    batch = z.shape[0] if z.dim() > 1 else 1
    if h is None:
        h = sample_fading(
            batch, generator=generator, device=z.device,
            dtype=z.real.dtype if z.is_complex() else z.dtype)
    else:
        if (not h.is_complex() or tuple(h.shape) != (batch, 1)):
            raise ValueError(
                f"h must be complex [batch, 1], got {tuple(h.shape)} "
                f"complex={h.is_complex()}")
    sigma2 = snr_db_to_sigma2(snr_db, power=power)
    if z.is_complex():
        if noise is None:
            n = sample_complex_gaussian(
                z.shape, sigma2, generator=generator, device=z.device,
                dtype=z.real.dtype, complex_out=True)
        else:
            if (not noise.is_complex() or noise.shape != z.shape):
                raise ValueError(
                    f"noise must be complex matching z, got {tuple(noise.shape)}")
            n = noise
        y = h * z + n
    else:
        # stacked real input: apply the same complex h to the complex view
        k = z.shape[-1] // 2
        zc = torch.complex(z[..., :k], z[..., k:])
        if noise is None:
            n = sample_complex_gaussian(
                zc.shape, sigma2, generator=generator, device=z.device,
                dtype=z.dtype, complex_out=True)
        else:
            if noise.is_complex() or noise.shape != z.shape:
                raise ValueError(
                    f"noise must be real-stacked matching z, got "
                    f"{tuple(noise.shape)} complex={noise.is_complex()}")
            n = torch.complex(noise[..., :k], noise[..., k:])
        yc = h * zc + n
        y = torch.cat([yc.real, yc.imag], dim=-1)
    if return_h:
        return y, h
    return y


def equalize_zf(
    y: torch.Tensor,
    h: torch.Tensor,
    eps_eq: float = config.EPS_EQUALIZER,
) -> torch.Tensor:
    """Guarded zero-forcing equalization: y * conj(h) / max(|h|^2, eps_eq).

    ``y``: complex ``[B, k]`` or real stacked ``[B, 2k]``.
    ``h``: complex ``[B, 1]`` block-fading coefficients (CSIR).
    """
    hh = (h.real ** 2 + h.imag ** 2).clamp_min(eps_eq)     # [B, 1]
    if y.is_complex():
        return y * h.conj() / hh
    k = y.shape[-1] // 2
    yc = torch.complex(y[..., :k], y[..., k:])
    ye = yc * h.conj() / hh
    return torch.cat([ye.real, ye.imag], dim=-1)
