"""Time-correlated Rayleigh fading (Clarke/Jakes), A-19.

Frozen model, for contrast: ``project.channels.rayleigh`` draws one complex
gain per codeword and treats successive steps as independent.  That is only
correct if the platform moves many wavelengths per control period.

This module supplies the correlated alternative used by A-19:

    h_e(t) = (1/sqrt(M)) * sum_m exp( j(2*pi*f_d*t*cos(alpha_em) + phi_em) )

with ``alpha_em, phi_em ~ U[0, 2*pi)`` drawn per episode, ``M = 32``, and
``f_d = v*f_c/c``.  Because ``alpha`` is uniform the ensemble autocorrelation
is exactly ``J0(2*pi*f_d*tau)``; a single episode is a finite sum, so the
process is ergodic in the ensemble, not within one episode.

Normalisation: ``E[|h|^2] = 1`` by construction, so the frozen ZF equaliser
and the digital baseline consume it unchanged.

The physically meaningful quantity is the displacement per step in
wavelengths, ``delta = v*DT/lambda``: ``v`` and ``f_c`` enter only through
``delta``, which is why the A-19 grid is indexed by it.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from project import config

SPEED_OF_LIGHT = 299_792_458.0     # m/s
DEFAULT_OSCILLATORS = 32           # M in the docstring
DEFAULT_CHUNK = 256                # episodes per generation block


def wavelength_m(carrier_hz: float) -> float:
    if carrier_hz <= 0:
        raise ValueError("carrier_hz must be positive")
    return SPEED_OF_LIGHT / carrier_hz


def doppler_hz(speed_mps: float, carrier_hz: float) -> float:
    """Maximum Doppler shift ``f_d = v f_c / c`` in Hz."""
    if speed_mps < 0:
        raise ValueError("speed_mps must be non-negative")
    return speed_mps * carrier_hz / SPEED_OF_LIGHT


def displacement_per_step(speed_mps: float, carrier_hz: float,
                          dt: Optional[float] = None) -> float:
    """Platform displacement per control period, in wavelengths."""
    dt = config.DT if dt is None else dt
    return speed_mps * dt / wavelength_m(carrier_hz)


def jakes_fading(
    episodes: int,
    t_max: int,
    fd_hz: float,
    dt: Optional[float] = None,
    n_osc: int = DEFAULT_OSCILLATORS,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.complex64,
    chunk: int = DEFAULT_CHUNK,
) -> torch.Tensor:
    """Correlated complex fading, shape ``[episodes, t_max]``, ``E|h|^2 = 1``.

    ``fd_hz`` must be strictly positive: at ``fd_hz = 0`` every oscillator
    contributes a constant and the result is a single random complex gain
    (not a CN(0,1) draw), so callers wanting the uncorrelated control must
    use the frozen i.i.d. path instead.
    """
    if fd_hz <= 0:
        raise ValueError("fd_hz must be > 0; use the frozen i.i.d. path for "
                         "the uncorrelated control")
    if episodes <= 0 or t_max <= 0:
        raise ValueError("episodes and t_max must be positive")
    if n_osc < 1:
        raise ValueError("n_osc must be >= 1")
    dt = config.DT if dt is None else dt

    t = torch.arange(t_max, dtype=torch.float32) * float(dt)
    out = torch.empty(episodes, t_max, dtype=dtype)
    scale = 1.0 / math.sqrt(float(n_osc))
    step = max(1, int(chunk))

    for start in range(0, episodes, step):
        stop = min(episodes, start + step)
        n = stop - start
        alpha = torch.rand(n, n_osc, generator=generator) * (2.0 * math.pi)
        phi = torch.rand(n, n_osc, generator=generator) * (2.0 * math.pi)
        omega = (2.0 * math.pi * float(fd_hz)) * torch.cos(alpha)   # rad/s
        phase = omega.unsqueeze(1) * t.view(1, -1, 1) + phi.unsqueeze(1)
        unit = torch.polar(torch.ones_like(phase), phase)           # [n,T,M]
        h = unit.sum(dim=-1) * scale
        out[start:stop] = h.to(dtype)

    return out


def fade_statistics(h: torch.Tensor, threshold_db: float = -10.0) -> dict:
    """Descriptive temporal statistics of a ``[episodes, steps]`` fading batch.

    Reported alongside the A-19 success rates so the reader can see how much
    temporal structure the sweep actually created: the deep-fade fraction and
    the mean number of consecutive steps spent below the threshold.
    """
    if h.dim() != 2:
        raise ValueError("h must be [episodes, steps]")
    power_db = 20.0 * torch.log10(h.abs().clamp_min(1e-12))
    below = power_db < float(threshold_db)
    episodes, steps = below.shape

    run_lengths = []
    for e in range(episodes):
        row = below[e]
        run, total, count = 0, 0, 0
        for t in range(steps):
            if bool(row[t]):
                run += 1
            elif run:
                total += run
                count += 1
                run = 0
        if run:
            total += run
            count += 1
        run_lengths.extend([total / count] if count else [])

    mean_power = (h.abs() ** 2).mean().item()
    return {
        "threshold_db": float(threshold_db),
        "deep_fade_fraction": float(below.float().mean().item()),
        "mean_fade_run_steps": (float(sum(run_lengths) / len(run_lengths))
                                if run_lengths else 0.0),
        "mean_power": float(mean_power),
        "steps": int(steps),
        "episodes": int(episodes),
    }
