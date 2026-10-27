"""3GPP UMa-aerial channel layer (Wave-C; non-frozen amendment).

Source-verified formulas (Cui et al., arXiv:2105.13717, quoting 3GPP
TR 36.777 + TR 38.901 aerial models; verified 2026-10-12):

* LoS probability, h in (22.5 m, 100 m], d2D <= 4 km:
      P_L = d1/r + exp(-r/p1)*(1 - d1/r),
      p1 = 4300*log10(h) - 3800,  d1 = max(460*log10(h) - 700, 18)
  (and P_L = 1 for r <= d1; for h in [1.5, 22.5] m the 36.777-shaped
  18-exp form below is used with C'_h = ((h-13)/10)^1.5 for h > 13).
* LoS path loss (dB), h in (22.5, 300] m, d2D <= 4 km:
      PL_L = 28.0 + 22*log10(d3D) + 20*log10(f_c_GHz)
* NLoS path loss (dB), h in (22.5, 100] m:
      PL_N = -17.5 + (46 - 7*log10(h))*log10(d3D)
             + 20*log10(40*pi*f_c/3)
* Expected loss = PL_L*P_L + PL_N*(1-P_L).

How it enters THIS system (differentiability, no invented physics):
* The frozen simulators use per-codeword complex fading h ~ CN(0,1) with
  perfect-CSI ZF equalization - a no-pathloss, no-shadowing benchmark.
  The UMa-aerial layer adds a DETERMINISTIC differentiable multiplicative
  gain  g = 10^( -(PL_mean - PL_ref)/20 )  applied to the codeword:
      y = h * (g * z) + n,   then perfect-CSI ZF equalizes (g*h) as
      y_eq-yield -> the neural decoder sees the usual fading BUT the SNR
      is measurably degraded by the aerial large-scale loss.
* ``csi_error_sigma`` adds Gaussian estimation noise to the equalizer's
  channel estimate h_hat = h + eps (eps ~ CN(0, csi_sigma^2)) BEFORE
  equalization - the Wave-B item requested by the supervisor.
* Everything is implemented on top of the FROZEN rayleigh_channel
  (h sampling + noise convention reused verbatim; no duplicated RNG).

NOT USED by default; enabled only when a config passes channel="umaaerial".

Two uses of the same formulas:

* A-10 (evaluation only): a DETERMINISTIC mean path loss ``g`` evaluated at an
  effective SNR, or as a LoS/NLoS mixture of the same measured curve.
* A-12 (training): ``sample_aerial_gain`` draws one gain per episode from the
  LoS/NLoS split plus log-normal shadowing, centred on the mean path loss, so
  the model is trained *under* the 3GPP dispersion rather than evaluated on
  it.  Block fading over the episode; no Doppler.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from project import config
from project.channels.rayleigh import equalize_zf, rayleigh_channel, sample_fading

# ---------------------------------------------------------------------------
# 3GPP formulas (exact, cited above)
# ---------------------------------------------------------------------------

def los_probability(d2d_m: float, h_m: float) -> float:
    """P_LoS(r, h) per TR 36.777 aerial UMa model (meters, meters)."""
    if h_m <= 22.5:
        # 36.777 low-altitude branch (h in [1.5, 22.5] m)
        if d2d_m <= 18.0:
            p = 1.0
        else:
            c = 0.0 if h_m <= 13.0 else ((h_m - 13.0) / 10.0) ** 1.5
            p = ((18.0 / d2d_m
                  + math.exp(-d2d_m / 36.0) * (1.0 - 18.0 / d2d_m))
                 * (1.0 + c * (5.0 / 4.0)
                    * (d2d_m / 100.0) ** 3 * math.exp(-d2d_m / 150.0)))
        return min(1.0, max(0.0, p))
    p1 = 4300.0 * math.log10(h_m) - 3800.0
    d1 = max(460.0 * math.log10(h_m) - 700.0, 18.0)
    if d2d_m <= d1:
        return 1.0
    return d1 / d2d_m + math.exp(-d2d_m / p1) * (1.0 - d1 / d2d_m)


def los_path_loss_db(d3d_m: float, fc_ghz: float) -> float:
    """PL_L for the high-altitude branch: 28.0 + 22log10(d3D)+20log10(fc)."""
    return 28.0 + 22.0 * math.log10(max(d3d_m, 1.0)) + 20.0 * math.log10(fc_ghz)


def nlos_path_loss_db(d3d_m: float, h_m: float, fc_ghz: float) -> float:
    """PL_N for h in (22.5, 100] m: -17.5+(46-7log10 h)log10(d3D)+20log10(40pi fc/3)."""
    return (-17.5 + (46.0 - 7.0 * math.log10(h_m)) * math.log10(max(d3d_m, 1.0))
            + 20.0 * math.log10(40.0 * math.pi * fc_ghz / 3.0))


def expected_path_loss_db(d2d_m: float, h_m: float, fc_ghz: float) -> float:
    """Mean PL over LoS/NLoS: PL_L*P_L + PL_N*(1-P_L)."""
    p_l = los_probability(d2d_m, h_m)
    pl_los = los_path_loss_db(math.hypot(d2d_m, h_m), fc_ghz)
    pl_nlos = nlos_path_loss_db(math.hypot(d2d_m, h_m), h_m, fc_ghz)
    return p_l * pl_los + (1.0 - p_l) * pl_nlos


def aerial_gain(d2d_m: float, h_m: float, fc_ghz: float,
                reference_pl_db: float) -> float:
    """Differentiable-linear gain g = 10^(-(PL_mean - PL_ref)/20).

    ``reference_pl_db`` anchors what "1.0 gain" means so that SNR sweep
    axes stay comparable with the frozen benchmark (PL_ref excludes the
    aerial large-scale effect: all codes run through the frozen path).
    """
    pl = expected_path_loss_db(d2d_m, h_m, fc_ghz)
    return 10.0 ** (-(pl - reference_pl_db) / 20.0)


# ---------------------------------------------------------------------------
# per-episode dispersion channel (A-12: training UNDER the 3GPP statistics)
# ---------------------------------------------------------------------------

# TR 38.901 Table 7.4.1-1 (UMa): shadow-fading standard deviation, dB.
SHADOW_SIGMA_DB = {"los": 4.0, "nlos": 6.0}

# Pinned geometry of the A-12 training channel.  Recorded in every cell spec
# and in the run summary; changing it creates a different experiment.
TRAINING_GEOMETRY = {"d2d_m": 500.0, "h_m": 50.0, "fc_ghz": 2.0}


def sample_aerial_offsets_db(
    batch: int,
    generator: Optional[torch.Generator] = None,
    *,
    d2d_m: Optional[float] = None,
    h_m: Optional[float] = None,
    fc_ghz: Optional[float] = None,
    reference_pl_db: Optional[float] = None,
    max_offset_db: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-episode large-scale offset (dB), ``[B, 1]``, and the LoS mask.

    One Bernoulli LoS/NLoS draw per episode (probability ``P_LoS`` of the
    geometry) plus one log-normal shadow term ``N(0, sigma_branch^2)``:

        offset_db = PL_branch + shadow - reference_pl_db

    ``reference_pl_db=None`` centres the offset on the geometry's MEAN path
    loss, so the trained model sees the 3GPP *dispersion* (LoS/NLoS split +
    shadowing) around the frozen link budget.  A numeric reference keeps
    the absolute loss (the A-10 evaluation convention).

    ``max_offset_db`` (A-13, opt-in) rejects episodes whose extra loss is
    worse than that cap by **resampling them** (rejection sampling; the rest
    of the batch is untouched).  One-sided: the better-than-mean tail is kept,
    because it is not an outlier for the error.  ``None`` reproduces the A-12
    stream exactly.  The evaluation path never passes it, so evaluation always
    sees the full dispersion.

    Block-fading assumption: the offset is constant over one episode
    (``T_MAX`` steps).  Doppler / time-correlated shadowing are out of
    scope and are stated as such in the amendment.
    """
    geo = TRAINING_GEOMETRY
    d2d_m = geo["d2d_m"] if d2d_m is None else float(d2d_m)
    h_m = geo["h_m"] if h_m is None else float(h_m)
    fc_ghz = geo["fc_ghz"] if fc_ghz is None else float(fc_ghz)

    p_los = los_probability(d2d_m, h_m)
    d3d = math.hypot(d2d_m, h_m)
    pl_los = los_path_loss_db(d3d, fc_ghz)
    pl_nlos = nlos_path_loss_db(d3d, h_m, fc_ghz)
    if reference_pl_db is None:
        reference_pl_db = p_los * pl_los + (1.0 - p_los) * pl_nlos
    ref = float(reference_pl_db)

    def _draw(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        u = torch.rand(n, 1, generator=generator, device=device, dtype=dtype)
        los = u < p_los
        pl = torch.where(los, torch.full_like(u, pl_los),
                         torch.full_like(u, pl_nlos))
        sigma = torch.where(los, torch.full_like(u, SHADOW_SIGMA_DB["los"]),
                            torch.full_like(u, SHADOW_SIGMA_DB["nlos"]))
        shadow = torch.randn(n, 1, generator=generator, device=device,
                             dtype=dtype) * sigma
        return pl + shadow - ref, los

    offsets_db, is_los = _draw(batch)
    if max_offset_db is not None:
        cap = float(max_offset_db)
        bad = offsets_db > cap
        for _ in range(10000):                  # terminates w.p. 1; guarded
            if not bool(bad.any()):
                break
            idx = bad.nonzero(as_tuple=True)[0]
            new_off, new_los = _draw(int(idx.numel()))
            offsets_db[idx] = new_off
            is_los[idx] = new_los
            bad = offsets_db > cap
        else:                                   # pragma: no cover
            raise RuntimeError(
                f"max_offset_db={cap} dB is unreachable for this geometry")
    return offsets_db, is_los


def sample_aerial_gain(
    batch: int,
    generator: Optional[torch.Generator] = None,
    **kwargs,
) -> torch.Tensor:
    """Per-episode multiplicative gain ``g = 10^(-offset_db/20)``, ``[B, 1]``.

    See :func:`sample_aerial_offsets_db` for the sampling convention.  The
    gain multiplies the codeword before the frozen Rayleigh pass; perfect-CSI
    ZF on ``h*g`` then turns it into an effective-SNR penalty of
    ``-20 log10(g)`` dB for that episode.
    """
    offsets_db, _ = sample_aerial_offsets_db(batch, generator, **kwargs)
    return 10.0 ** (-offsets_db / 20.0)


# ---------------------------------------------------------------------------
# channel dispatch-compatible wrapper
# ---------------------------------------------------------------------------

def aerial_rayleigh_channel(
    z: torch.Tensor,
    snr_db: float,
    *,
    d2d_m: float,
    h_m: float,
    fc_ghz: float = 2.0,
    reference_pl_db: float = 0.0,     # frozen benchmark anchor (no path loss)
    csi_error_sigma: float = 0.0,     # Wave-B hook: h_hat = h + eps
    generator: Optional[torch.Generator] = None,
    noise: Optional[torch.Tensor] = None,
    h: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """UMa-aerial pass: gain*g then frozen Rayleigh fading + optional CSI error.

    Returns (y, h_eq_target) matching the frozen channel contract, where
    h_eq_target is the TRUE h*g (for diagnostics); callers following the
    frozen perfect-CSI convention equalize with h_hat = h + eps instead.
    """
    g = aerial_gain(d2d_m, h_m, fc_ghz, reference_pl_db)
    z_eff = z * g                     # differentiable scalar multiply
    y, h_used = rayleigh_channel(z_eff, snr_db, generator=generator,
                                 noise=noise, h=h, return_h=True)
    if csi_error_sigma > 0.0:
        eps = sample_fading(h_used.shape[0], generator=generator,
                            device=h_used.device,
                            dtype=h_used.real.dtype) * csi_error_sigma
        h_hat = h_used + eps                      # estimation error model
    else:
        h_hat = h_used
    # equalize with the (possibly errored) estimate, return true-h diagnostics
    y_eq = equalize_zf(y, h_hat)
    return y_eq, h_used * g


CHANNEL_INFO = {
    "umaaerial": {
        "source": "3GPP TR 36.777 / TR 38.901 UMa-aerial (via arXiv:2105.13717)",
        "los_prob": "d1/r + exp(-r/p1)(1-d1/r), p1=4300log10(h)-3800, "
                    "d1=max(460log10(h)-700,18)",
        "pl_los_db": "28.0 + 22*log10(d3D) + 20*log10(fc_GHz)",
        "pl_nlos_db": "-17.5 + (46-7*log10(h))*log10(d3D) + 20*log10(40*pi*fc/3)",
        "shadow_sigma_db": SHADOW_SIGMA_DB,
        "training_geometry": TRAINING_GEOMETRY,
        "training_channel": ("per-episode LoS/NLoS draw + log-normal "
                             "shadowing, block-fading over the episode; "
                             "no Doppler"),
        "csi_error_hook": "csi_error_sigma",
        "default_disabled": True,
        "additive_to_frozen": True,
    },
}
