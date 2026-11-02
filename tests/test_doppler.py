"""Doppler / time-correlated Rayleigh fading tests (A-19).

Covers the Clarke/Jakes generator, the frozen-path non-regression, and the
``Condition`` hook.  No training, no evaluation writes.

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_doppler
"""

from __future__ import annotations

import math
import sys

SUMMARY = []


def check(name, ok, detail=""):
    SUMMARY.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]"
          + (f"  ({detail})" if detail else ""))


def close(a, b, tol=1e-3):
    return abs(a - b) <= tol


def _bessel_j0(x: float) -> float:
    """J0 via the standard series; x up to ~25 needs the asymptotic form."""
    if abs(x) < 12.0:
        term, total, k = 1.0, 1.0, 0
        while True:
            k += 1
            term *= -(x * x) / (4.0 * k * k)
            total += term
            if abs(term) < 1e-16:
                return total
    # asymptotic: sqrt(2/(pi x)) cos(x - pi/4)
    return math.sqrt(2.0 / (math.pi * abs(x))) * math.cos(
        abs(x) - math.pi / 4.0)


def test_physics_helpers():
    from project.channels import doppler

    lam = doppler.wavelength_m(2.0e9)
    check("wavelength at 2 GHz", close(lam, 0.1499, 1e-3), f"{lam:.4f} m")

    fd = doppler.doppler_hz(5.0, 2.0e9)
    check("Doppler at 5 m/s, 2 GHz", close(fd, 33.36, 0.05), f"{fd:.2f} Hz")

    delta = doppler.displacement_per_step(5.0, 2.0e9)
    check("displacement per step at 5 m/s", close(delta, 3.335, 0.01),
          f"{delta:.3f} lambda")

    delta2 = doppler.displacement_per_step(1.0, 0.7e9)
    check("0.7 GHz 1 m/s matches 2 GHz 0.35 m/s displacement",
          close(delta2, 1.0 * 0.1 / doppler.wavelength_m(0.7e9), 1e-9),
          f"{delta2:.3f} lambda")

    for bad in (-0.5, -1.0):
        try:
            doppler.doppler_hz(bad, 2.0e9)
            check(f"negative speed {bad} rejected", False)
        except ValueError:
            check(f"negative speed {bad} rejected", True)

    check("zero speed gives zero Doppler", doppler.doppler_hz(0.0, 2.0e9) == 0.0)

    try:
        doppler.jakes_fading(4, 10, 0.0)
        check("fd = 0 rejected (use the frozen path)", False)
    except ValueError:
        check("fd = 0 rejected (use the frozen path)", True)


def test_power_and_shape():
    import torch
    from project.channels import doppler

    g = torch.Generator().manual_seed(7)
    h = doppler.jakes_fading(4000, 50, 33.0, generator=g)
    check("shape", tuple(h.shape) == (4000, 50), str(tuple(h.shape)))
    check("complex", h.is_complex())
    power = (h.abs() ** 2).mean().item()
    check("E|h|^2 = 1", close(power, 1.0, 0.03), f"{power:.4f}")

    finite = bool(torch.isfinite(h.real).all() and torch.isfinite(h.imag).all())
    check("all finite", finite)


def test_autocorrelation_matches_bessel():
    """Ensemble autocorrelation must approach J0(2*pi*fd*tau)."""
    import torch
    from project.channels import doppler

    fd = 10.0
    dt = 0.1
    g = torch.Generator().manual_seed(11)
    h = doppler.jakes_fading(6000, 20, fd, dt=dt, generator=g)
    mean_power = (h.abs() ** 2).mean().item()
    for lag in (1, 2, 3):
        lhs = (h[:, :-lag] * h[:, lag:].conj()).mean().real.item() / mean_power
        rhs = _bessel_j0(2.0 * math.pi * fd * dt * lag)
        check(f"R(lag {lag}) ~ J0", close(lhs, rhs, 0.06),
              f"empirical {lhs:+.3f} vs J0 {rhs:+.3f}")

    # strongly correlated regime: lag-1 correlation near one
    g2 = torch.Generator().manual_seed(12)
    h2 = doppler.jakes_fading(2000, 20, 0.35, dt=dt, generator=g2)
    mp2 = (h2.abs() ** 2).mean().item()
    r1 = (h2[:, :-1] * h2[:, 1:].conj()).mean().real.item() / mp2
    check("low Doppler keeps lag-1 correlation high", r1 > 0.85, f"{r1:.3f}")

    # fast fading: lag-1 correlation near zero
    g3 = torch.Generator().manual_seed(13)
    h3 = doppler.jakes_fading(2000, 20, 33.0, dt=dt, generator=g3)
    mp3 = (h3.abs() ** 2).mean().item()
    r1f = (h3[:, :-1] * h3[:, 1:].conj()).mean().real.item() / mp3
    check("high Doppler decorrelates within one step", abs(r1f) < 0.15,
          f"{r1f:+.3f}")


def test_fade_statistics():
    import torch
    from project.channels import doppler

    g = torch.Generator().manual_seed(21)
    slow = doppler.jakes_fading(200, 100, 0.35, generator=g)
    g2 = torch.Generator().manual_seed(21)
    fast = doppler.jakes_fading(200, 100, 33.0, generator=g2)

    s_slow = doppler.fade_statistics(slow)
    s_fast = doppler.fade_statistics(fast)
    check("deep-fade fraction is a probability",
          0.0 <= s_slow["deep_fade_fraction"] <= 1.0,
          f"{s_slow['deep_fade_fraction']:.3f}")
    check("slow fading has longer fades than fast fading",
          s_slow["mean_fade_run_steps"] > s_fast["mean_fade_run_steps"],
          f"{s_slow['mean_fade_run_steps']:.2f} vs "
          f"{s_fast['mean_fade_run_steps']:.2f} steps")
    check("fade statistics report power ~ 1", close(s_slow["mean_power"], 1.0, 0.1),
          f"{s_slow['mean_power']:.3f}")


def test_determinism():
    import torch
    from project.channels import doppler

    a = doppler.jakes_fading(64, 30, 5.0, generator=torch.Generator().manual_seed(3))
    b = doppler.jakes_fading(64, 30, 5.0, generator=torch.Generator().manual_seed(3))
    check("same seed is bit-identical", bool(torch.equal(a, b)))

    c = doppler.jakes_fading(64, 30, 5.0, generator=torch.Generator().manual_seed(4))
    check("different seed differs", not bool(torch.equal(a, c)))


def test_frozen_path_untouched():
    """The default Condition must reproduce the frozen i.i.d. draws exactly."""
    import torch
    from project import config
    from project.evaluation.evaluate import Condition, generate_realizations

    cond = Condition(k=12, channel="rayleigh", snr_db=10.0, t_max=20,
                     episodes=64, seed=config.TEST_SEED)
    real_a = generate_realizations(cond, 64)

    gen_noise = config.derive_generator("noise_test", cond.seed)
    shape = (64, 20, 12)
    sigma2 = config.TX_POWER * (10.0 ** (-10.0 / 10.0))
    noise = torch.complex(
        torch.randn(shape, generator=gen_noise) * (sigma2 / 2.0) ** 0.5,
        torch.randn(shape, generator=gen_noise) * (sigma2 / 2.0) ** 0.5,
    )
    h = torch.complex(
        torch.randn(64, 20, generator=gen_noise) * 0.5 ** 0.5,
        torch.randn(64, 20, generator=gen_noise) * 0.5 ** 0.5,
    )
    check("frozen noise unchanged", bool(torch.equal(real_a["noise"], noise)))
    check("frozen h unchanged", bool(torch.equal(real_a["h"], h)))

    cond_d = Condition(k=12, channel="rayleigh", snr_db=10.0, t_max=20,
                       episodes=64, seed=config.TEST_SEED, doppler_hz=5.0)
    real_d = generate_realizations(cond_d, 64)
    check("doppler path still shares the frozen noise",
          bool(torch.equal(real_d["noise"], noise)))
    check("doppler path replaces h", not bool(torch.equal(real_d["h"], h)))
    hh = real_d["h"]
    mp = (hh.abs() ** 2).mean().item()
    check("doppler path power ~ 1", close(mp, 1.0, 0.15), f"{mp:.3f}")


if __name__ == "__main__":
    print("=" * 72)
    print("DOPPLER / TIME-CORRELATED RAYLEIGH TESTS (A-19)")
    print("=" * 72)
    test_physics_helpers()
    test_power_and_shape()
    test_autocorrelation_matches_bessel()
    test_fade_statistics()
    test_determinism()
    test_frozen_path_untouched()
    n_pass, n_all = sum(SUMMARY), len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)
