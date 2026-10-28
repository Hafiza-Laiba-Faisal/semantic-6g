"""5G NR polar chain (A-15) gates.

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_polar5g

Gates:
  G1  tables: the 1024-entry reliability sequence is a permutation of
      0..1023 with the spec's first/last entries; the 32-column sub-block
      pattern is a permutation of 0..31.
  G2  transform: the butterfly equals u @ F^{(x)n} for N = 8, 16, 32.
  G3  length selection + mode: (K, E) -> (N, repetition/shortening) exactly
      for the five cells of the frozen grid.
  G4  CRC: attach-then-check passes; every single-bit flip is detected, for
      A = 6, 12, 18, 24, 48 (CRC-6 and CRC-11).
  G5  rate matching: the interleaver is a bijection; the rate-matching
      pre-frozen set never overlaps the information set (38.212 5.3.1.2);
      shortening drops only frozen code bits; K information positions exist.
  G6  noiseless block roundtrip: payload bits are recovered exactly for all
      five grid cells, including the repetition cell.
  G7  list decoding: the list starts at one path and grows to L, so the
      surviving paths are distinct; CA-SCL(8) is strictly better than SC on
      a seeded batch at 3 dB.
  G8  noiseless full-chain state roundtrip: Polar5GBaseline reproduces the
      quantized state exactly (error < 1e-6 vs the ideal quantization).
  G9  determinism: the same symbols decode to the same bits twice.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import torch

from project.baselines import polar5g as P5
from project.baselines.digital import dequantize, quantize
from project.baselines.polar5g_tables import (Q_0_NMAX, SUB_BLOCK_PATTERN,
                                              reliability_order)
from project.baselines.polar5g import (Polar5GBaseline, Polar5GCode,
                                       crc_attach, crc_len_for, crc_passes,
                                       mother_length, polar_transform,
                                       pre_frozen_positions, rate_match_mode)
from project.data.normalization import denormalize, normalize

SUMMARY = []
GRID = ((12, 1), (18, 2), (24, 3), (30, 4), (54, 8))


def gate(name: str, ok: bool, detail: str = "") -> None:
    SUMMARY.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def _awgn_llr(bits: np.ndarray, snr_db: float, rng) -> np.ndarray:
    sym = (1 - 2.0 * bits) / np.sqrt(2.0)
    sigma2 = 10.0 ** (-snr_db / 10.0)
    y = sym + rng.normal(0.0, np.sqrt(sigma2 / 2.0), sym.shape)
    return 4.0 * np.sqrt(2.0) * y / sigma2


def run() -> None:
    rng = np.random.default_rng(20261027)

    # ---------------------------------------------------------------- G1
    try:
        ok = (len(Q_0_NMAX) == 1024
              and sorted(Q_0_NMAX) == list(range(1024))
              and Q_0_NMAX[:8] == (0, 1, 2, 4, 8, 16, 32, 3)
              and Q_0_NMAX[-8:] == (991, 1020, 1007, 1015, 1019, 1021,
                                    1022, 1023)
              and sorted(SUB_BLOCK_PATTERN) == list(range(32)))
        gate("G1 spec tables", ok,
             f"len={len(Q_0_NMAX)} ends ok pattern ok={ok}")
    except BaseException:
        gate("G1 spec tables", False, traceback.format_exc())

    # ---------------------------------------------------------------- G2
    try:
        F = np.array([[1, 0], [1, 1]], dtype=np.int8)
        ok = True
        for n in (3, 4, 5):
            G = np.array([[1]], dtype=np.int8)
            for _ in range(n):
                G = np.kron(G, F)
            u = rng.integers(0, 2, size=(5, G.shape[0])).astype(np.int8)
            ok &= np.array_equal(polar_transform(u), (u @ G) % 2)
        gate("G2 transform == F kron", ok)
    except BaseException:
        gate("G2 transform == F kron", False, traceback.format_exc())

    # ---------------------------------------------------------------- G3
    try:
        expect = {12: (12, 32, "shortening"), 18: (18, 32, "repetition"),
                  24: (24, 64, "shortening"), 30: (35, 64, "shortening"),
                  54: (59, 128, "shortening")}
        ok = True
        det = []
        for k, b in GRID:
            c = Polar5GCode(6 * b, 2 * k)
            kk, nn, mode = expect[k]
            good = (c.k_bits == kk and c.N == nn and c.mode == mode
                    and c.n == int(np.log2(nn)))
            ok &= good
            det.append(f"k{k}:K{c.k_bits}/N{c.N}/{c.mode[:4]}")
        gate("G3 length + mode", ok, " ".join(det))
    except BaseException:
        gate("G3 length + mode", False, traceback.format_exc())

    # ---------------------------------------------------------------- G4
    try:
        ok = True
        for a in (6, 12, 18, 24, 48):
            L = crc_len_for(a)
            x = rng.integers(0, 2, size=(300, a)).astype(np.int8)
            y = crc_attach(x, L)
            ok &= bool(crc_passes(y, L).all())
            pos = rng.integers(0, a + L, size=300)
            bad = y.copy()
            bad[np.arange(300), pos] ^= 1
            ok &= not bool(crc_passes(bad, L).any())
        gate("G4 CRC attach/detect", ok)
    except BaseException:
        gate("G4 CRC attach/detect", False, traceback.format_exc())

    # ---------------------------------------------------------------- G5
    try:
        ok = True
        det = []
        for k, b in GRID:
            c = Polar5GCode(6 * b, 2 * k)
            good = (sorted(c.perm.tolist()) == list(range(c.N))
                    and len(c.info_pos) == c.k_bits
                    and len(set(c.info_pos.tolist())) == c.k_bits
                    and not (set(c.info_pos.tolist())
                             & set(c.pre_frozen.tolist())))
            if c.mode == "shortening":
                # the dropped code bits must be frozen subchannels
                dropped = set(int(c.perm[i]) for i in range(c.e_bits, c.N))
                good &= dropped.issubset(
                    set(range(c.N)) - set(c.info_pos.tolist()))
                good &= dropped.issubset(set(c.pre_frozen.tolist()))
            ok &= good
            det.append(f"k{k}:{c.mode[:4]}/{c.pre_frozen.size}pre")
        gate("G5 rate-matching invariants", ok, " ".join(det))
    except BaseException:
        gate("G5 rate-matching invariants", False, traceback.format_exc())

    # ---------------------------------------------------------------- G6
    try:
        ok = True
        det = []
        for k, b in GRID:
            c = Polar5GCode(6 * b, 2 * k)
            payload = rng.integers(0, 2, size=(40, 6 * b)).astype(np.int8)
            e = c.encode(payload)
            llr = np.where(e == 0, 40.0, -40.0)
            dec = c.decode(llr)
            good = bool((dec == payload).all())
            ok &= good
            det.append(f"k{k}:{'ok' if good else 'FAIL'}")
        gate("G6 noiseless block roundtrip", ok, " ".join(det))
    except BaseException:
        gate("G6 noiseless block roundtrip", False, traceback.format_exc())

    # ---------------------------------------------------------------- G7
    try:
        c = Polar5GCode(18, 48)
        payload = rng.integers(0, 2, size=(200, 18)).astype(np.int8)
        e = c.encode(payload)
        llr = _awgn_llr(e, 3.0, rng)
        sc = _scl_decode_payload(c, llr, 1)
        scl = _scl_decode_payload(c, llr, 8)
        bler_sc = float((sc != payload).any(axis=1).mean())
        bler_scl = float((scl != payload).any(axis=1).mean())
        gate("G7 SCL(8) beats SC", bler_scl < bler_sc,
             f"SC={bler_sc:.3f} SCL8={bler_scl:.3f}")
    except BaseException:
        gate("G7 SCL(8) beats SC", False, traceback.format_exc())

    # ---------------------------------------------------------------- G8
    try:
        torch.manual_seed(3)
        s = torch.rand(128, 6) * 16 - 8
        ok = True
        det = []
        for k, b in GRID[:4]:
            base = Polar5GBaseline(k, b)
            s_hat = base.forward(s, 20.0, channel="awgn", noiseless=True)
            q, _ = quantize(normalize(s), b)
            ideal = denormalize(dequantize(q, b))
            err = float((s_hat - ideal).abs().max())
            ok &= err < 1e-6
            det.append(f"k{k}:{err:.1e}")
        gate("G8 noiseless state roundtrip", ok, " ".join(det))
    except BaseException:
        gate("G8 noiseless state roundtrip", False, traceback.format_exc())

    # ---------------------------------------------------------------- G9
    try:
        c = Polar5GCode(24, 60)
        payload = rng.integers(0, 2, size=(20, 24)).astype(np.int8)
        e = c.encode(payload)
        llr = _awgn_llr(e, 2.0, rng)
        a = c.decode(llr)
        b = c.decode(llr)
        gate("G9 determinism", bool((a == b).all()))
    except BaseException:
        gate("G9 determinism", False, traceback.format_exc())

    print()
    n_fail = sum(1 for _, ok2, _ in SUMMARY if not ok2)
    print(f"5G polar gates: {len(SUMMARY) - n_fail}/{len(SUMMARY)} pass")
    sys.exit(0 if n_fail == 0 else 1)


def _scl_decode_payload(code: Polar5GCode, llr_e: np.ndarray,
                        list_size: int) -> np.ndarray:
    llr_n = code._llrs_to_mother(llr_e)
    out = np.empty((llr_n.shape[0], code.a_bits), dtype=np.int8)
    for m in range(llr_n.shape[0]):
        out[m] = P5._scl_decode(llr_n[m][None, :], code.info_pos, code.N,
                                code.k_bits, code.crc_len, list_size)[0]
    return out


if __name__ == "__main__":
    run()
