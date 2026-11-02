"""5G NR LDPC BG2 comparator tests (A-20).

Structural gates (table, lifting, parity, rank, noiseless round-trip), a
waterfall sanity check, and interface parity with the frozen digital baseline.
No training, no evaluation writes.

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_ldpc
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


def test_base_graph_table():
    from project.baselines import ldpc5g

    check("base graph loaded", len(ldpc5g.BASE_GRAPH) == 197,
          f"{len(ldpc5g.BASE_GRAPH)} entries")
    rows = {r for r, _ in ldpc5g.BASE_GRAPH}
    cols = {c for _, c in ldpc5g.BASE_GRAPH}
    check("42 base rows", len(rows) == 42, str(len(rows)))
    check("52 base columns", max(cols) == 51 and min(cols) == 0,
          f"{min(cols)}..{max(cols)}")
    shifts = [v for tup in ldpc5g.BASE_GRAPH.values() for v in tup]
    check("every entry has 8 lifting shifts",
          all(len(t) == 8 for t in ldpc5g.BASE_GRAPH.values()))
    check("shifts are non-negative and below 384",
          min(shifts) >= 0 and max(shifts) < 384,
          f"{min(shifts)}..{max(shifts)}")

    sizes = [z for zset in ldpc5g.LIFTING_SETS for z in zset]
    check("lifting table size (8 sets)", len(ldpc5g.LIFTING_SETS) == 8)
    check("no duplicate lifting size", len(sizes) == len(set(sizes)),
          f"{len(sizes)} values")


def test_lifting_choice():
    from project.baselines.ldpc5g import LIFTING_SETS, lifting_size

    expect = {6: (2, 0), 12: (2, 0), 18: (2, 0), 24: (3, 1), 30: (3, 1),
              48: (5, 2), 54: (6, 1)}
    for a_bits, want in expect.items():
        got = lifting_size(a_bits)
        check(f"A={a_bits} -> Z={want[0]}", got == want,
              f"got Z={got[0]} i_ls={got[1]}")
        check(f"A={a_bits} Z in its own set",
              got[0] in LIFTING_SETS[got[1]], str(got))


def _rank_gf2(matrix):
    import numpy as np

    a = matrix.astype(np.uint8).copy()
    rows, cols = a.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if a[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        a[[rank, pivot]] = a[[pivot, rank]]
        for r in range(rows):
            if r != rank and a[r, col]:
                a[r] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def test_coding_structure():
    import numpy as np

    from project.baselines.ldpc5g import LDPC5GCode

    rng = np.random.default_rng(21)
    for a_bits, e_bits in ((6, 24), (18, 48), (48, 108)):
        code = LDPC5GCode(a_bits, e_bits)
        payload = rng.integers(0, 2, size=(64, a_bits), dtype=np.uint8)
        mother = code.full_codeword(payload)

        check(f"parity check holds (A={a_bits})",
              all(code.parity_ok(row) for row in mother))
        check(f"dimension is K (A={a_bits})",
              code.n_bits - _rank_gf2(
                  np.concatenate([code.h_s, code.h_p], axis=1)) == code.k_bits,
              f"N={code.n_bits} K={code.k_bits}")

        tx = code.encode(payload)
        check(f"transmitted width E (A={a_bits})",
              tx.shape == (64, e_bits), str(tx.shape))
        check(f"filler bits are not transmitted (A={a_bits})",
              e_bits + code.n_filler <= code.n_bits,
              f"E={e_bits} filler={code.n_filler} N={code.n_bits}")

        dec = code.decode(np.where(tx == 0, 20.0, -20.0))
        check(f"noiseless round-trip (A={a_bits})", bool((dec == payload).all()))

        # determinism
        check(f"encoding deterministic (A={a_bits})",
              bool((code.encode(payload) == tx).all()))


def _awgn_ber(code, snr_db, n_blocks=200, seed=5):
    import numpy as np

    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 2, size=(n_blocks, code.a_bits), dtype=np.uint8)
    coded = code.encode(payload)
    s2 = 10.0 ** (-snr_db / 10.0)
    sym = (1 - 2 * coded.astype(np.float64)) * 0.5
    pairs = sym.reshape(n_blocks, -1, 2)
    x = (pairs[..., 0] + 1j * pairs[..., 1]) / math.sqrt(2.0)
    noise = rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)
    y = x + noise * math.sqrt(s2 / 2.0)
    llr = np.stack([(4.0 * math.sqrt(2.0)) * y.real / s2,
                    (4.0 * math.sqrt(2.0)) * y.imag / s2],
                   axis=-1).reshape(n_blocks, -1)
    dec = code.decode(llr.astype(np.float32))
    return float((dec != payload).mean())


def test_waterfall():
    from project.baselines.ldpc5g import LDPC5GCode

    code = LDPC5GCode(48, 108)
    bers = [_awgn_ber(code, s) for s in (0.0, 4.0, 8.0, 12.0)]
    check("BER decreases with SNR",
          all(bers[i] >= bers[i + 1] for i in range(len(bers) - 1)),
          " ".join(f"{b:.4f}" for b in bers))
    check("BER at 12 dB is small", bers[-1] < 1e-3, f"{bers[-1]:.5f}")


def test_baseline_interface():
    import torch

    from project import config
    from project.baselines.digital import pack_bits, quantize
    from project.baselines.ldpc5g import LDPC5GBaseline
    from project.data.normalization import normalize

    base = LDPC5GBaseline(54, 8)
    torch.manual_seed(0)
    s = torch.rand(16, 6)
    span = torch.tensor(config.STATE_MAX) - torch.tensor(config.STATE_MIN)
    s = s * span + torch.tensor(config.STATE_MIN)

    sym = base.encode_state(s)
    check("encode_state -> k QPSK symbols", tuple(sym.shape) == (16, 54),
          str(tuple(sym.shape)))

    q, _ = quantize(normalize(s), 8)
    want = pack_bits(q, 8).numpy()
    tx = base.code.encode(want)
    got = base.code.decode(
        __import__("numpy").where(tx == 0, 1e6, -1e6))
    check("bit-level noiseless round-trip", bool((got == want).all()))

    err = float((base.forward(s, 20.0, channel="awgn", noiseless=True)
                 - s).abs().max())
    step_m = config.quantizer_step(8) * 8.0        # normalised step -> metres
    check("noiseless state error is quantiser-limited",
          err <= step_m, f"{err:.4f} <= {step_m:.4f} m")

    y = base.forward(s, 20.0, channel="awgn", noiseless=True)
    check("forward returns physical states", tuple(y.shape) == (16, 6),
          str(tuple(y.shape)))


def test_params_provenance():
    from project.baselines.ldpc5g import LDPC5GCode

    p = LDPC5GCode(48, 108).params()
    check("params name the standard", p["standard"] == "3GPP TS 38.212")
    check("params say BG2", p["base_graph"] == 2)
    check("params disclose the missing rate-matching chain",
          p["rate_matching_chain"] is False and p["crc"] is False)
    check("params disclose the selection rule",
          "non-NULL" in p["selection"])


if __name__ == "__main__":
    print("=" * 72)
    print("5G NR LDPC BG2 COMPARATOR TESTS (A-20)")
    print("=" * 72)
    test_base_graph_table()
    test_lifting_choice()
    test_coding_structure()
    test_waterfall()
    test_baseline_interface()
    test_params_provenance()
    n_pass, n_all = sum(SUMMARY), len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)
