"""Complex AWGN channel (frozen, audit 1.1 row 9).

    y = z + n,      n ~ CN(0, sigma^2 I),      sigma^2 = P * 10^(-SNR_dB/10)

The frozen SNR convention is per-complex-symbol ``P / sigma^2``; each real
component of the noise has variance ``sigma^2 / 2`` so that ``E[|n|^2] =
sigma^2``. Noise is sampled through the single shared sampler in
``complex_utils`` (no duplicated AWGN logic anywhere).

Noise is an exogenous constant inside the autograd graph: gradients flow
through ``z`` but never through the sampled noise (audit risk 9).
"""

from __future__ import annotations

from typing import Optional

import torch

from project import config
from project.channels.complex_utils import (
    _is_complex,
    check_codeword,
    sample_complex_gaussian,
    snr_db_to_sigma2,
)


def awgn_channel(
    z: torch.Tensor,
    snr_db: float,
    generator: Optional[torch.Generator] = None,
    power: float = config.TX_POWER,
) -> torch.Tensor:
    """Transmit one codeword batch through complex AWGN.

    Parameters
    ----------
    z : complex ``[B, k]`` or real stacked ``[B, 2k]`` codeword batch
        (already power-normalized by the caller).
    snr_db : configured SNR in dB (per complex symbol, P / sigma^2).
    generator : optional torch.Generator for reproducibility.

    Returns the received signal with the same kind/shape as ``z``.
    """
    check_codeword(z)
    sigma2 = snr_db_to_sigma2(snr_db, power=power)
    if _is_complex(z):
        n = sample_complex_gaussian(
            z.shape, sigma2, generator=generator, device=z.device,
            dtype=z.real.dtype, complex_out=True)
    else:
        k = z.shape[-1] // 2
        n = sample_complex_gaussian(
            torch.Size(z.shape[:-1] + (k,)), sigma2, generator=generator,
            device=z.device, dtype=z.dtype, complex_out=False)
    return z + n
