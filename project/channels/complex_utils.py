"""Complex-baseband utilities shared by every communication method.

Representation convention (fixed for the whole project, documented in the
audit sections 3 and 4.1):

* Inside the neural networks everything is REAL, laid out as ``[..., 2k]``
  with the **stacked I/Q order**: ``x = [Re(z)_1..Re(z)_k, Im(z)_1..Im(z)_k]``.
* Channel operations accept either complex tensors ``[..., k]`` (dtype
  complex64/complex128) or real stacked tensors ``[..., 2k]``. They return
  the same kind they were given.

SNR and noise conventions (frozen, audit section 4.3):

    SNR_dB = 10 log10(P / sigma^2)      =>      sigma^2 = P * 10^(-SNR_dB/10)

For a proper circular complex Gaussian ``n ~ CN(0, sigma^2)``:

    Re(n), Im(n) ~ N(0, sigma^2 / 2)    so that E[|n|^2] = sigma^2.

Using ``sigma^2`` as the per-component variance would be the classic 3 dB
error and is guarded against by the verification suite.
"""

from __future__ import annotations

from typing import Optional

import torch

from project import config


# ---------------------------------------------------------------------------
# kind / shape helpers
# ---------------------------------------------------------------------------

def _is_complex(x: torch.Tensor) -> bool:
    return x.is_complex()


def check_codeword(x: torch.Tensor) -> None:
    """Validate a codeword's shape and finiteness; nothing else.

    ``x`` is either complex ``[..., k]`` or real stacked ``[..., 2k]``.
    A real tensor with an odd last dimension cannot be I/Q-paired and is
    rejected; complex tensors accept any k >= 1.
    """
    if x.dim() < 1:
        raise ValueError(f"codeword must have at least 1 dim, got {tuple(x.shape)}")
    if not _is_complex(x) and x.shape[-1] % 2 != 0:
        raise ValueError(
            f"real-stacked codeword needs an even last dim, got {x.shape[-1]}")
    if not torch.isfinite(x.real if _is_complex(x) else x).all():
        raise FloatingPointError("codeword contains NaN/Inf")


def snr_db_to_sigma2(snr_db: float, power: float = config.TX_POWER) -> float:
    """Frozen SNR convention: sigma^2 = P * 10^(-SNR_dB / 10)."""
    if power <= 0:
        raise ValueError(f"power must be positive, got {power}")
    return float(power) * (10.0 ** (-float(snr_db) / 10.0))


def sigma2_to_snr_db(sigma2: float, power: float = config.TX_POWER) -> float:
    """Inverse of :func:`snr_db_to_sigma2` (used in verification)."""
    return 10.0 * math_log10(float(power) / float(sigma2))


def math_log10(x: float) -> float:
    import math
    return math.log10(x)


def to_complex(x: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """Real stacked ``[..., 2k]`` -> complex ``[..., k]``.

    Split order: first k entries are the real parts, last k the imaginary
    parts (``z_i = a_i + j * a_{k+i}``, stacked convention of audit 1.1 row 6).
    """
    if _is_complex(x):
        return x
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            f"last dim must be even for stacked I/Q, got {x.shape[-1]}")
    half = x.shape[-1] // 2 if k is None else k
    if k is not None and x.shape[-1] != 2 * k:
        raise ValueError(f"expected last dim {2 * k}, got {x.shape[-1]}")
    return torch.complex(x[..., :half], x[..., half:])


def to_real(x: torch.Tensor) -> torch.Tensor:
    """Complex ``[..., k]`` -> real stacked ``[..., 2k]`` (same split order)."""
    if not _is_complex(x):
        return x
    return torch.cat([x.real, x.imag], dim=-1)


# ---------------------------------------------------------------------------
# shared circular complex Gaussian sampling
# ---------------------------------------------------------------------------

def sample_complex_gaussian(
    shape: torch.Size,
    variance: float,
    generator: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    complex_out: bool = True,
) -> torch.Tensor:
    """Draw ``CN(0, variance)`` samples: each real component has variance/2.

    This single helper is the ONLY place in the project where complex
    Gaussian samples are produced (audit risk 1: the sigma^2/2 rule lives
    in exactly one function).
    """
    if variance <= 0:
        raise ValueError(f"variance must be positive, got {variance}")
    shape = torch.Size(shape)
    std = (variance / 2.0) ** 0.5
    re = torch.randn(shape, generator=generator, device=device,
                     dtype=dtype) * std
    im = torch.randn(shape, generator=generator, device=device,
                     dtype=dtype) * std
    if complex_out:
        return torch.complex(re, im)          # shape == `shape`
    return torch.cat([re, im], dim=-1)        # stacked: last dim doubled


# ---------------------------------------------------------------------------
# power normalization (frozen audit 1.1 row 8)
# ---------------------------------------------------------------------------

def power_normalize(
    z_tilde: torch.Tensor,
    k: Optional[int] = None,
    power: float = config.TX_POWER,
    eps: float = config.EPS_POWER,
) -> torch.Tensor:
    """Per-codeword power normalization to average symbol power P.

        z = sqrt(kP) * z_tilde / sqrt( sum_i |z_tilde_i|^2 + eps )

    Applied per sample (the last dimension is one codeword), never per
    batch and never per symbol. Accepts complex ``[..., k]`` or real
    stacked ``[..., 2k]`` input and returns the same kind.
    """
    if k is None:
        k = z_tilde.shape[-1] if _is_complex(z_tilde) else z_tilde.shape[-1] // 2
    if _is_complex(z_tilde):
        energy = (z_tilde.real ** 2 + z_tilde.imag ** 2).sum(dim=-1, keepdim=True)
        scale = (k * power) ** 0.5 / torch.sqrt(energy + eps)
        return z_tilde * scale
    energy = z_tilde.pow(2).sum(dim=-1, keepdim=True)
    scale = (k * power) ** 0.5 / torch.sqrt(energy + eps)
    return z_tilde * scale


def codeword_avg_power(z: torch.Tensor) -> torch.Tensor:
    """(1/k) * sum_i |z_i|^2 per sample; should equal P after normalization."""
    if _is_complex(z):
        energy = (z.real ** 2 + z.imag ** 2).sum(dim=-1)
    else:
        energy = z.pow(2).sum(dim=-1)
    k = z.shape[-1] if _is_complex(z) else z.shape[-1] // 2
    return energy / k
