"""Step 3 channel verification suite (audit section 4, items 1-9).

Every check prints: property tested, expected value, empirical value,
tolerance, and PASS/FAIL. Random statistics use Monte-Carlo samples with
tolerances scaled to the sample size (no exact-equality fragility).

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_channels
"""

from __future__ import annotations

import math
import sys

import torch

from project import config
from project.channels.awgn import awgn_channel
from project.channels.complex_utils import (
    check_codeword,
    codeword_avg_power,
    power_normalize,
    sample_complex_gaussian,
    snr_db_to_sigma2,
    sigma2_to_snr_db,
    to_complex,
    to_real,
)
from project.channels.rayleigh import equalize_zf, rayleigh_channel, sample_fading

torch.manual_seed(0)

SUMMARY: list = []


def check(name, desc, expected, empirical, tol, ok=None):
    if ok is None:
        ok = abs(float(empirical) - float(expected)) <= tol
    SUMMARY.append(ok)
    status = "PASS" if ok else "FAIL"
    print(f"{status}  [{name}] {desc}")
    print(f"      expected={expected}  empirical={empirical:.6g}  tol={tol}")
    return ok


def check_bool(name, desc, ok, detail=""):
    SUMMARY.append(bool(ok))
    status = "PASS" if ok else "FAIL"
    print(f"{status}  [{name}] {desc}" + (f"  ({detail})" if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# 1. complex conversion
# ---------------------------------------------------------------------------

def test_conversion():
    B, k = 64, 7
    x = torch.randn(B, 2 * k)
    z = to_complex(x)
    check_bool("1a", "to_complex shape [B,2k] -> [B,k]",
               tuple(z.shape) == (B, k), f"{tuple(x.shape)} -> {tuple(z.shape)}")
    x2 = to_real(z)
    check_bool("1b", "real->complex->real round-trip exact",
               bool((x2 == x).all()))
    n2_x = float(x.pow(2).sum())
    n2_z = float((z.real ** 2 + z.imag ** 2).sum())
    check("1c", "no factor-of-2 scaling: ||z||^2 == ||x||^2 (ratio)",
          1.0, n2_z / n2_x, 1e-6)
    # stacked order: z_i = a_i + j a_{k+i}
    check_bool("1d", "stacked I/Q order z_i = a_i + j*a_{k+i}",
               bool((z.real == x[:, :k]).all() and (z.imag == x[:, k:]).all()))
    # codeword validation guards
    try:
        check_codeword(torch.randn(4, 5))
        check_bool("1e", "odd last dim rejected", False)
    except ValueError:
        check_bool("1e", "odd last dim rejected", True)
    try:
        check_codeword(torch.tensor([[float("nan"), 0.0]]))
        check_bool("1f", "NaN codeword rejected", False)
    except FloatingPointError:
        check_bool("1f", "NaN codeword rejected", True)


# ---------------------------------------------------------------------------
# 2. power normalization
# ---------------------------------------------------------------------------

def test_power_normalization():
    worst = 0.0
    for k in (1, 2, 3, 12, 54):
        for B in (1, 7, 256):
            zt = torch.randn(B, k, dtype=torch.complex64)
            z = power_normalize(zt, k=k)
            dev = float((codeword_avg_power(z) - config.TX_POWER).abs().max())
            worst = max(worst, dev)
    check("2a", "per-codeword avg power == P over k x B grid (max dev)",
          0.0, worst, 1e-5)
    # real-stacked path agrees with complex path
    x = torch.randn(32, 6)                       # k = 3
    dev = float((to_real(power_normalize(to_complex(x)))
                 - power_normalize(x)).abs().max())
    check("2b", "real-stacked path == complex path (max dev)", 0.0, dev, 1e-6)
    # eps only protects: an exactly-zero codeword stays finite (and zero)
    z0 = power_normalize(torch.zeros(4, 3, dtype=torch.complex64))
    check_bool("2c", "zero-energy codeword: finite output (eps guard)",
               bool(torch.isfinite(z0).all()))


# ---------------------------------------------------------------------------
# 3. AWGN variance
# ---------------------------------------------------------------------------

def test_awgn_variance():
    N = 2_000_000
    for snr_db in (0.0, 10.0):
        sigma2 = snr_db_to_sigma2(snr_db)
        g = torch.Generator().manual_seed(1234)
        n = sample_complex_gaussian(torch.Size((N,)), sigma2, generator=g)
        p_emp = float((n.real ** 2 + n.imag ** 2).mean())
        check(f"3a(snr={snr_db:.0f})", "E[|n|^2] == sigma^2",
              sigma2, p_emp, 0.01)
        var_re = float(n.real.var(unbiased=False))
        var_im = float(n.imag.var(unbiased=False))
        check(f"3b(snr={snr_db:.0f})", "Var(Re n) == sigma^2/2",
              sigma2 / 2, var_re, 0.01)
        check(f"3b(snr={snr_db:.0f})", "Var(Im n) == sigma^2/2",
              sigma2 / 2, var_im, 0.01)


# ---------------------------------------------------------------------------
# 4. AWGN empirical SNR + reproducibility
# ---------------------------------------------------------------------------

def test_awgn_snr():
    B, k = 100_000, 3
    zt = torch.randn(B, k, dtype=torch.complex64)
    z = power_normalize(zt, k=k)
    P_emp = float((z.real ** 2 + z.imag ** 2).mean())
    for snr_db in (0.0, 5.0, 10.0, 20.0):
        sigma2 = snr_db_to_sigma2(snr_db)
        g = torch.Generator().manual_seed(777)
        n = sample_complex_gaussian(z.shape, sigma2, generator=g)
        y = z + n
        n_emp = y - z
        sigma2_emp = float((n_emp.real ** 2 + n_emp.imag ** 2).mean())
        snr_emp = 10 * math.log10(P_emp / sigma2_emp)
        check(f"4(snr={snr_db:.0f})", "empirical received SNR (dB)",
              snr_db, snr_emp, 0.2)
    # reproducibility: same generator seed -> bitwise identical noise
    g1 = torch.Generator().manual_seed(99)
    g2 = torch.Generator().manual_seed(99)
    y1 = awgn_channel(z, 10.0, generator=g1)
    y2 = awgn_channel(z, 10.0, generator=g2)
    check_bool("4r", "AWGN reproducible with fixed seed (bitwise)",
               bool((y1 == y2).all()))
    g3 = torch.Generator().manual_seed(100)
    y3 = awgn_channel(z, 10.0, generator=g3)
    check_bool("4r", "different seed -> different noise",
               bool((y1 != y3).any()))
    # complex path == real-stacked path under aligned generators
    g1 = torch.Generator().manual_seed(55)
    g2 = torch.Generator().manual_seed(55)
    y_c = awgn_channel(z, 7.0, generator=g1)
    y_r = awgn_channel(to_real(z), 7.0, generator=g2)
    dev = float((to_real(y_c) - y_r).abs().max())
    check("4p", "complex and real-stacked AWGN paths identical (max dev)",
          0.0, dev, 0.0)


# ---------------------------------------------------------------------------
# 5. Rayleigh fading statistics
# ---------------------------------------------------------------------------

def test_rayleigh_stats():
    N = 500_000
    g = torch.Generator().manual_seed(4242)
    h = sample_fading(N, generator=g).squeeze(-1)
    check("5a", "E[|h|^2] == 1", 1.0,
          float((h.real ** 2 + h.imag ** 2).mean()), 0.01)
    check("5b", "Var(h_re) == 1/2 (CN(0,1))", 0.5,
          float(h.real.var(unbiased=False)), 0.01)
    check("5b", "Var(h_im) == 1/2 (CN(0,1))", 0.5,
          float(h.imag.var(unbiased=False)), 0.01)
    check("5b", "E[h_re] == 0", 0.0, float(h.real.mean()), 0.01)
    mag = h.abs()
    check("5c", "Rayleigh amplitude: P(|h| > 1) == e^-1", math.exp(-1),
          float((mag > 1.0).float().mean()), 0.01)
    check("5c", "Rayleigh amplitude: P(|h| > 2) == e^-4", math.exp(-4),
          float((mag > 2.0).float().mean()), 0.005)
    # h is never renormalized after sampling: two independent batches must
    # both deviate from exactly 1.0 by their own Monte-Carlo fluctuation
    g = torch.Generator().manual_seed(31337)
    m1 = float((sample_fading(200_000, generator=g).abs() ** 2).mean())
    m2 = float((sample_fading(200_000, generator=g).abs() ** 2).mean())
    check_bool("5d", "no post-sampling renormalization of h "
                     "(batch means fluctuate, not pinned to 1)",
               abs(m1 - 1.0) > 1e-6 and abs(m2 - 1.0) > 1e-6,
               f"means {m1:.6f}, {m2:.6f}")


# ---------------------------------------------------------------------------
# 6. block fading
# ---------------------------------------------------------------------------

def test_block_fading():
    B, k = 100_000, 4
    z = power_normalize(torch.randn(B, k, dtype=torch.complex64))
    g = torch.Generator().manual_seed(8)
    h = sample_fading(B, generator=g)
    y = h * z                                     # noiseless fading
    # h constant across symbols within a codeword: y_i / z_i identical for
    # two different symbol indices of the same codeword
    r0 = y[:, 0] / z[:, 0]
    r3 = y[:, 3] / z[:, 3]
    dev = float((r0 - r3).abs().max())
    check("6a", "within-codeword: y_i/z_i identical for all i (max dev)",
          0.0, dev, 1e-4)
    # across codewords: independent draws -> cross-correlation ~ 0
    h1 = sample_fading(B, generator=g).squeeze(-1)
    h2 = sample_fading(B, generator=g).squeeze(-1)
    corr = float((h1 * h2.conj()).mean().abs())
    check("6b", "cross-codeword |E[h1 conj(h2)]| ~ 0 (independence)",
          0.0, corr, 0.02)


# ---------------------------------------------------------------------------
# 7. CSI / equalization
# ---------------------------------------------------------------------------

def test_equalization():
    B, k = 64, 3
    z = power_normalize(torch.randn(B, k, dtype=torch.complex64))
    g = torch.Generator().manual_seed(21)
    h = sample_fading(B, generator=g)
    # (a) noiseless, non-negligible h: exact reconstruction
    mag2 = (h.real ** 2 + h.imag ** 2)
    keep = (mag2.squeeze(-1) > 1e-3)
    z_rec = equalize_zf(h * z, h)
    rel = float(((z_rec[keep] - z[keep]).abs() / z[keep].abs()).max())
    check("7a", "noiseless ZF: equalized h*z recovers z (max rel err)",
          0.0, rel, 1e-5)
    # (b) equals exact y/h away from zero
    y = h * z
    dev = float((equalize_zf(y, h)[keep] - (y / h)[keep]).abs().max())
    check("7b", "guard == exact ZF whenever |h|^2 >> eps (max dev)",
          0.0, dev, 1e-5)
    # (c) numerical protection for |h| -> 0: finite output
    h_tiny = torch.full((B, 1), 1e-6, dtype=torch.complex64)
    h_tiny = h_tiny * torch.exp(1j * torch.linspace(0, 3, B)).unsqueeze(-1)
    out = equalize_zf(h_tiny * z, h_tiny)
    check_bool("7c", "|h|=1e-6: equalized output finite (eps_eq guard)",
               bool(torch.isfinite(out).all()))
    # (d) real-stacked path identical
    dev = float((to_real(equalize_zf(h * z, h))
                 - equalize_zf(to_real(h * z), h)).abs().max())
    check("7d", "real-stacked equalization path identical (max dev)",
          0.0, dev, 0.0)


# ---------------------------------------------------------------------------
# 8/9. Rayleigh reproducibility + convention guards
# ---------------------------------------------------------------------------

def test_repro_and_conventions():
    B, k = 32, 3
    z = power_normalize(torch.randn(B, k, dtype=torch.complex64))
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    y1, h1 = rayleigh_channel(z, 10.0, generator=g1, return_h=True)
    y2, h2 = rayleigh_channel(z, 10.0, generator=g2, return_h=True)
    check_bool("8a", "Rayleigh reproducible with fixed seed (bitwise y, h)",
               bool((y1 == y2).all()) and bool((h1 == h2).all()))

    # 9. frozen conventions
    check("9a", "SNR math: 10 dB, P=1 -> sigma^2 == 0.1",
          0.1, snr_db_to_sigma2(10.0), 1e-12)
    check("9a", "SNR math: 0 dB, P=1 -> sigma^2 == 1.0",
          1.0, snr_db_to_sigma2(0.0), 1e-12)
    check("9a", "SNR round-trip sigma2_to_snr_db(snr_db_to_sigma2(s)) == s",
          17.5, sigma2_to_snr_db(snr_db_to_sigma2(17.5)), 1e-9)
    check_bool("9b", "complex noise uses sigma^2/2 per component "
                     "(verified separately in 3b)", True)
    check_bool("9c", "Rayleigh uses E|h|^2 = 1, block fading, no h "
                     "renormalization (verified in 5, 6)", True)
    # input validation
    try:
        snr_db_to_sigma2(10.0, power=0.0)
        check_bool("9d", "invalid power rejected", False)
    except ValueError:
        check_bool("9d", "invalid power rejected", True)
    try:
        sample_complex_gaussian(torch.Size((4,)), variance=-1.0)
        check_bool("9e", "invalid variance rejected", False)
    except ValueError:
        check_bool("9e", "invalid variance rejected", True)


if __name__ == "__main__":
    print("=" * 72)
    print("STEP 3 CHANNEL VERIFICATION SUITE")
    print(f"torch {torch.__version__} | P = {config.TX_POWER} | "
          f"eps_power = {config.EPS_POWER} | eps_eq = {config.EPS_EQUALIZER}")
    print("=" * 72)
    test_conversion()
    test_power_normalization()
    test_awgn_variance()
    test_awgn_snr()
    test_rayleigh_stats()
    test_block_fading()
    test_equalization()
    test_repro_and_conventions()
    n_pass = sum(SUMMARY)
    n_all = len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"VERIFICATION SUMMARY: {n_pass}/{n_all} checks passed")
    if n_pass == n_all:
        print("STEP 3 VERDICT: PASS")
        sys.exit(0)
    print("STEP 3 VERDICT: FAIL - fundamental channel convention broken; "
          "STOP and fix before proceeding.")
    sys.exit(1)
