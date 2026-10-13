"""Wave-D polar digital baseline tests.

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_polar_baseline

Gates:
  P1  GF(2) generator sanity: polar_matrix well-formed, G @ Ginv == I.
  P2  Clean-channel SC identity: enc/dec roundtrip exact for K in {4,6,9,14}
      on jittered clean LLRs, plus random unequal magnitudes (no float ties).
  P3  AWGN BLER monotonicity: K=6 polar code BLER decreases with SNR and is
      ~0 at 10 dB (loose statistical bounds, seeded).
  P4  Noiseless full-chain state roundtrip: PolarDigitalBaseline at zero
      noise reproduces the quantized state EXACTLY (err vs ideal < 1e-6)
      for a spread of (k, bits) feasible packets.
"""

from __future__ import annotations

import math
import sys
import traceback

import torch

from project.baselines.digital import dequantize, quantize
from project.baselines.digital_polar import (PolarCode, PolarDigitalBaseline,
                                             _gf2_inverse, polar_matrix,
                                             polar_sc_decode)
from project.data.normalization import denormalize, normalize

SUMMARY = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    SUMMARY.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}")


def run() -> None:
    torch.manual_seed(11)

    # ------------------------------------------------------------------ P1
    try:
        all_inv = True
        for N in (16, 32):
            G = polar_matrix(N)
            Ginv = _gf2_inverse(G)
            if not torch.equal((G @ Ginv) % 2, torch.eye(N, dtype=torch.long)):
                all_inv = False
            if not torch.equal((Ginv @ G) % 2, torch.eye(N, dtype=torch.long)):
                all_inv = False
        all_units = True
        for N in (16, 32):
            G = polar_matrix(N)
            if not torch.equal((G @ G) % 2, torch.eye(N, dtype=torch.long)):
                all_units = False          # F^⊗n family is self-inverse
        gate("P1 gf2 generator sanity", all_inv and all_units)
    except BaseException:
        gate("P1 gf2 generator sanity", False, traceback.format_exc())

    # ------------------------------------------------------------------ P2
    try:
        ok = True
        details = []
        from project.baselines.digital_polar import clean_llr_magnitude
        for K in (4, 6, 9, 14):
            code = PolarCode(K)
            B, batch = 128, 4
            exact = 0
            torch.manual_seed(5 + K)
            mag_fixed = clean_llr_magnitude(code.N)
            for _ in range(4):
                info = torch.randint(0, 2, (B, K))
                x = code.encode(info)
                llr = (1 - 2 * x.double()) * mag_fixed
                if torch.equal(polar_sc_decode(code, llr), info):
                    exact += 1
            ok_k = exact == 4
            details.append(f"K={K} fixedmag {exact}/4")
            ok &= ok_k
        # random unequal magnitudes: full-batch identity, all node sizes
        for N in (2, 4, 8, 16):
            p = object.__new__(PolarCode)
            p.N, p.n_info = N, N
            p.frozen = torch.zeros(N, dtype=torch.long)
            p.info_idx = torch.arange(N)
            torch.manual_seed(N)
            B = 256
            u = torch.randint(0, 2, (B, N))
            x = (u @ polar_matrix(N)) % 2
            mag = 1e6 * (0.9 + 0.2 * torch.rand(B, N))
            got = polar_sc_decode(p, (1 - 2 * x.double()) * mag).long()
            ok &= torch.equal(got, u)
            details.append(f"N={N} {int((got == u).all(1).sum())}/{B}")
        gate("P2 clean SC identity", ok, " ".join(details))
    except BaseException:
        gate("P2 clean SC identity", False, traceback.format_exc())

    # ------------------------------------------------------------------ P3
    try:
        code = PolarCode(6)
        torch.manual_seed(123)
        bler = {}
        for snr_db in (4, 6, 8, 10, 12, 14):
            n_pow = 1.0 / (10 ** (snr_db / 10))
            fails, trials = 0, 0
            torch.manual_seed(123)
            for _ in range(20):
                info = torch.randint(0, 2, (64, 6))
                x = code.encode(info)
                amp = (1 - 2 * x.double())
                y = amp + torch.randn_like(amp) * (n_pow / 2) ** 0.5
                dec = polar_sc_decode(code, 2 * y / n_pow)
                fails += int((torch.any(dec != info, dim=1)).sum())
                trials += 64
            bler[snr_db] = fails / trials
        mono = (bler[4] > bler[6] >= bler[8] >= bler[10] >= bler[12]) \
            and bler[10] <= 0.02
        gate("P3 AWGN BLER curve",
             mono,
             " ".join(f"{k}dB:{v:.4f}" for k, v in bler.items()))
    except BaseException:
        gate("P3 AWGN BLER curve", False, traceback.format_exc())

    # ------------------------------------------------------------------ P4
    try:
        ok = True
        detail = []
        torch.manual_seed(3)
        s = torch.rand(256, 6) * 16 - 8
        for k, bits in [(12, 1), (24, 1), (30, 1), (24, 2)]:
            base = PolarDigitalBaseline(k, bits)
            s_hat = base.forward(s, 20.0, channel="awgn", noiseless=True)
            s_bar = normalize(s)
            q, _ = quantize(s_bar, bits)
            ideal = denormalize(dequantize(q, bits))
            err = float((s_hat - ideal).abs().max())
            detail.append((k, bits, err))
            ok &= err < 1e-6
        gate("P4 noiseless state roundtrip", ok,
             " ".join(f"k{x[0]}b{x[1]}:{x[2]:.3g}" for x in detail))
    except BaseException:
        gate("P4 noiseless state roundtrip", False, traceback.format_exc())

    print()
    n_fail = sum(1 for _, ok2, _ in SUMMARY if not ok2)
    print(f"Polar baseline gates: {len(SUMMARY) - n_fail}/{len(SUMMARY)} pass")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    run()
