"""3GPP UMa-aerial link-budget tests (A-10).

Two halves:

  A. formula tests against hand-computed reference values (no simulation),
  B. arithmetic tests for the transmitter-SNR mapping used by
     ``tools/run_aerial_link.py`` (synthetic curves, no checkpoints).

No training, no evaluation, no writes outside ``tempfile``.

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_aerial_link
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


# ---------------------------------------------------------------------------
# A. 3GPP formulas
# ---------------------------------------------------------------------------

def test_los_probability():
    from project.channels import aerial

    # high-altitude branch, r > d1: hand-computed
    # p1 = 4300*log10(50)-3800 = 3505.5709
    # d1 = max(460*log10(50)-700, 18) = 81.5263
    # P  = d1/r + exp(-r/p1)*(1-d1/r) = 0.888748512
    # (independently recomputed with numpy float64 on 2026-10-26)
    p = aerial.los_probability(500.0, 50.0)
    check("P_LoS(500 m, 50 m) reference", close(p, 0.888748512, 1e-8), f"{p:.9f}")

    # r <= d1 saturates at 1
    check("P_LoS saturates for r <= d1", aerial.los_probability(50.0, 50.0) == 1.0)

    # low-altitude branch (h = 1.5 m) approaches 18/r for large r
    p_low = aerial.los_probability(500.0, 1.5)
    check("P_LoS low-altitude branch reference",
          close(p_low, 0.0360009, 1e-6), f"{p_low:.7f}")

    # monotone decreasing in distance, bounded to [0, 1]
    vals = [aerial.los_probability(r, 50.0) for r in (100, 300, 500, 1000, 4000)]
    check("P_LoS monotone decreasing", all(a >= b for a, b in zip(vals, vals[1:])),
          str([round(v, 4) for v in vals]))
    check("P_LoS in [0, 1]", all(0.0 <= v <= 1.0 for v in vals))


def test_path_loss():
    from project.channels import aerial

    d3d = math.hypot(500.0, 50.0)  # 502.4938

    # PL_L = 28 + 22log10(d3D) + 20log10(fc) = 93.445475 dB
    pl_l = aerial.los_path_loss_db(d3d, 2.0)
    check("PL_LoS reference", close(pl_l, 93.445475120, 1e-8), f"{pl_l:.9f}")

    # PL_N = -17.5 + (46-7log10(50))*log10(d3D) + 20log10(40*pi*2/3)
    #      = 113.090404 dB
    pl_n = aerial.nlos_path_loss_db(d3d, 50.0, 2.0)
    check("PL_NLoS reference", close(pl_n, 113.090403740, 1e-8), f"{pl_n:.9f}")

    # NLoS must always be worse than LoS at the same geometry
    check("PL_NLoS > PL_LoS", pl_n > pl_l, f"{pl_n - pl_l:.2f} dB")

    # expected loss is the P_LoS-weighted mixture
    pl_mean = aerial.expected_path_loss_db(500.0, 50.0, 2.0)
    p = aerial.los_probability(500.0, 50.0)
    check("expected PL is the mixture",
          close(pl_mean, p * pl_l + (1 - p) * pl_n, 1e-6), f"{pl_mean:.4f}")

    # higher carrier -> higher loss
    check("PL increases with fc",
          aerial.los_path_loss_db(d3d, 3.5) > aerial.los_path_loss_db(d3d, 2.0))


def test_aerial_gain():
    from project.channels import aerial

    pl = aerial.expected_path_loss_db(500.0, 50.0, 2.0)
    g = aerial.aerial_gain(500.0, 50.0, 2.0, reference_pl_db=0.0)
    check("gain == 10**(-PL/20)", close(g, 10.0 ** (-pl / 20.0), 1e-12),
          f"{g:.4e}")

    # a reference loss cancels exactly
    g_ref = aerial.aerial_gain(500.0, 50.0, 2.0, reference_pl_db=pl)
    check("gain == 1 at the reference loss", close(g_ref, 1.0, 1e-12))

    # the documented default anchor gives the ~-95.6 dB loss
    check("default anchor loss in dB", close(20 * math.log10(g), -95.631003, 1e-4),
          f"{20 * math.log10(g):.4f} dB")


def test_channel_info():
    from project.channels import aerial

    info = aerial.CHANNEL_INFO["umaaerial"]
    check("CHANNEL_INFO documents the source",
          "3GPP" in info["source"])
    check("CHANNEL_INFO marks default-disabled",
          info["default_disabled"] is True and info["additive_to_frozen"] is True)


def test_channel_smoke():
    import torch
    from project.channels import aerial

    torch.manual_seed(0)
    z = torch.complex(torch.randn(2, 18), torch.randn(2, 18))
    y, h_target = aerial.aerial_rayleigh_channel(z, snr_db=10.0, d2d_m=500.0,
                                                 h_m=50.0, fc_ghz=2.0)
    check("aerial channel returns complex output", torch.is_complex(y))
    check("aerial channel output finite", bool(torch.isfinite(y).all()))
    check("aerial channel returns an equalizer target",
          torch.is_complex(h_target) and bool(torch.isfinite(h_target).all()))


# ---------------------------------------------------------------------------
# B. runner arithmetic
# ---------------------------------------------------------------------------

def _curve(points):
    return [{"snr_eff_db": float(x), "success_rate": float(y)} for x, y in points]


def test_required_interpolation():
    from tools.run_aerial_link import _required

    curve = _curve([(-10, 0.0), (0, 0.5), (10, 1.0)])
    check("required() interpolates a crossing", _required(curve, 0.9) == 8.0,
          str(_required(curve, 0.9)))
    check("required() returns None below the ceiling",
          _required(_curve([(0, 0.1), (10, 0.2)]), 0.9) is None)
    check("required() ignores input order",
          _required(list(reversed(curve)), 0.9) == 8.0)


def test_mixture_mapping():
    from tools.run_aerial_link import _mixture_curve

    # saturating-in-1-step curve: 0 below 0 dB, 1 at/above 0 dB
    curve = _curve([(-20, 0.0), (-5, 0.0), (0, 1.0), (20, 1.0)])
    rows = _mixture_curve(curve, p_los=0.5, pl_los=100.0, pl_nlos=110.0)
    by_tx = {r["snr_tx_db"]: r["success_rate"] for r in rows}

    # at tx=105 the LoS branch is at 5 dB (works), NLoS at -5 (fails) -> 0.5
    check("mixture weights LoS/NLoS", close(by_tx[105.0], 0.5, 1e-9))
    # at tx=110 both branches are at/above 0 dB -> 1.0
    check("mixture saturates past the worse branch", close(by_tx[110.0], 1.0, 1e-9))
    # at tx=100 only LoS is exactly at 0 dB -> 0.5
    check("mixture at the LoS edge", close(by_tx[100.0], 0.5, 1e-9))
    # far below both -> 0
    check("mixture below both branches", by_tx[60.0] == 0.0)

    # p_los=1 collapses to a pure shift by pl_los
    rows1 = _mixture_curve(curve, 1.0, 100.0, 110.0)
    by1 = {r["snr_tx_db"]: r["success_rate"] for r in rows1}
    check("p_los=1 is a pure shift", by1[100.0] == 1.0
          and close(by1[97.0], 0.4, 1e-9) and by1[90.0] == 0.0,
          f"{by1[97.0]:.3f}")


if __name__ == "__main__":
    print("=" * 72)
    print("3GPP UMa-AERIAL TESTS (formulas + link-budget arithmetic)")
    print("=" * 72)
    test_los_probability()
    test_path_loss()
    test_aerial_gain()
    test_channel_info()
    test_channel_smoke()
    test_required_interpolation()
    test_mixture_mapping()
    n_pass, n_all = sum(SUMMARY), len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)
