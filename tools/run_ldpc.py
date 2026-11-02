"""A-20: 5G NR LDPC (base graph 2) comparator, evaluation only.

Registered in fyp_md_files/PROTOCOL_AMENDMENTS.md (A-20) on 2026-11-02 before
any evaluation ran.  Nothing frozen is edited; this tool writes under
results/LDPC/ only.

The arm is *BG2 without the 38.212 rate-matching chain*: base graph, lifting,
systematic encoding and offset min-sum decoding are the specified ones; the
circular buffer, interleaving and CRC are not.  Every reported table says so.

Usage
    python tools/run_ldpc.py verify
    python tools/run_ldpc.py evaluate [--limit N]
    python tools/run_ldpc.py summary
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project import config                                    # noqa: E402
from project.baselines.ldpc5g import (LDPC5GBaseline, LDPC5GCode,  # noqa: E402
                                      lifting_size)
from project.evaluation.evaluate import (                     # noqa: E402
    Condition, aggregate, generate_realizations, run_method,
)

LDPC_ROOT = ROOT / "results" / "LDPC"
EVAL_EPISODES = 1000
SNR_GRID = (0.0, 5.0, 10.0, 15.0, 20.0)
K_LIST = (12, 18, 24, 30, 54)
CHANNELS = ("awgn", "rayleigh")
AMENDMENT = "A-20"


def eval_path(k, channel, snr):
    return LDPC_ROOT / "eval" / f"k{k}_{channel}" / f"eval_{snr:g}.json"


def summary_path():
    return LDPC_ROOT / "step_ldpc_summary.json"


# ---------------------------------------------------------------------------
# correctness gates
# ---------------------------------------------------------------------------

def _rank_gf2(matrix):
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


def _awgn_ber(code, snr_db, n_blocks=300, seed=5):
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 2, size=(n_blocks, code.a_bits), dtype=np.uint8)
    coded = code.encode(payload)
    s2 = 10.0 ** (-snr_db / 10.0)
    sym = (1 - 2 * coded.astype(np.float64)) * 0.5
    pairs = sym.reshape(n_blocks, -1, 2)
    x = (pairs[..., 0] + 1j * pairs[..., 1]) / math.sqrt(2.0)
    noise = (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape))
    noise *= math.sqrt(s2 / 2.0)
    y = x + noise
    llr_i = (4.0 * math.sqrt(2.0)) * y.real / s2
    llr_q = (4.0 * math.sqrt(2.0)) * y.imag / s2
    llr = np.stack([llr_i, llr_q], axis=-1).reshape(n_blocks, -1)
    dec = code.decode(llr.astype(np.float32))
    return float((dec != payload).mean())


def cmd_verify():
    print("== A-20 pre-flight gates ==", flush=True)
    bad = 0

    def gate(name, ok, detail=""):
        nonlocal bad
        print(f"{'PASS' if ok else 'FAIL'}  [{name}]"
              + (f"  ({detail})" if detail else ""), flush=True)
        if not ok:
            bad += 1

    text = (ROOT / "fyp_md_files" / "PROTOCOL_AMENDMENTS.md").read_text(
        encoding="utf-8")
    gate("amendment A-20 registered", "## A-20" in text)

    # 1. lifting sizes follow 38.212 (smallest Z with Kb*Z >= A)
    expect = {12: 2, 18: 2, 24: 2, 30: 3, 54: 5}
    got = {}
    for k in K_LIST:
        a_bits = config.digital_packet(k, config.DIGITAL_B_BY_K[k])["n_src_bits"]
        got[k] = lifting_size(a_bits)[0]
    gate("lifting sizes", got == expect, str(got))

    # 2-4. per-k structural gates
    rng = np.random.default_rng(11)
    for k in K_LIST:
        B = config.DIGITAL_B_BY_K[k]
        code = LDPC5GCode(6 * B, 2 * k)
        payload = rng.integers(0, 2, size=(200, 6 * B), dtype=np.uint8)
        mother = code.full_codeword(payload)
        parity = all(code.parity_ok(row) for row in mother)
        gate(f"parity H c^T = 0, k={k}", parity)

        h = np.concatenate([code.h_s, code.h_p], axis=1)
        rank = _rank_gf2(h)
        gate(f"rank = N - K, k={k}", code.n_bits - rank == code.k_bits,
             f"N={code.n_bits} rank={rank} K={code.k_bits}")

        tx = code.encode(payload)
        llr = np.where(tx == 0, 20.0, -20.0).astype(np.float32)
        dec = code.decode(llr)
        gate(f"noiseless round-trip, k={k}", bool((dec == payload).all()))

    # 5. waterfall is monotone
    code = LDPC5GCode(48, 108)
    bers = [_awgn_ber(code, s) for s in (0.0, 4.0, 8.0, 12.0)]
    monotone = all(bers[i] >= bers[i + 1] - 1e-9 for i in range(len(bers) - 1))
    gate("AWGN BER monotone", monotone, " ".join(f"{b:.4f}" for b in bers))

    # 6. the decoder must actually work at k = 54 (the declared sanity check)
    ber_10 = _awgn_ber(code, 10.0)
    gate("k=54 BER at 10 dB < 1e-2", ber_10 < 1e-2, f"{ber_10:.5f}")

    # 7. interface parity with the frozen baseline, checked at the bit level
    from project.baselines.digital import pack_bits, quantize
    from project.data.normalization import normalize

    torch.manual_seed(0)
    s = torch.rand(64, 6)
    s = s * (torch.tensor(config.STATE_MAX) - torch.tensor(config.STATE_MIN))
    s = s + torch.tensor(config.STATE_MIN)
    base = LDPC5GBaseline(54, 8)
    gate("encode_state produces k symbols",
         base.encode_state(s).shape[-1] == 54)
    q, _ = quantize(normalize(s), 8)
    want = pack_bits(q, 8).numpy()
    # decode the exact transmitted bits through the code path
    tx = base.code.encode(want)
    got = base.code.decode(np.where(tx == 0, 1e6, -1e6))
    gate("bit-level noiseless round-trip (k=54, B=8)", bool((got == want).all()))

    gate("output root isolated", str(LDPC_ROOT).startswith(str(ROOT / "results")),
         str(LDPC_ROOT.relative_to(ROOT)))

    print("== gates OK ==" if bad == 0 else f"== {bad} gate(s) FAILED ==",
          flush=True)
    return 0 if bad == 0 else 1


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(limit=None):
    print("== A-20 5G NR LDPC BG2 comparator (evaluation only) ==", flush=True)
    n_new, total_wall = 0, 0.0
    links = {}

    for k in K_LIST:
        B = config.DIGITAL_B_BY_K[k]
        links[k] = LDPC5GBaseline(k, B)
        for channel in CHANNELS:
            for snr in SNR_GRID:
                if limit is not None and n_new >= limit:
                    print(f"-- limit {limit} reached", flush=True)
                    return 0
                out = eval_path(k, channel, snr)
                if out.exists():
                    continue
                cond = Condition(k=k, channel=channel, snr_db=snr,
                                 t_max=config.T_MAX, bits_per_component=B,
                                 episodes=EVAL_EPISODES, seed=config.TEST_SEED)
                real = generate_realizations(cond, EVAL_EPISODES)
                link = links[k]
                t0 = time.perf_counter()

                def est(s, noise_c, h_c, _link=link, _snr=snr, _ch=channel):
                    return _link.forward(s, _snr, channel=_ch,
                                         noise=noise_c, h=h_c)

                with torch.no_grad():
                    metrics = run_method(cond, real, estimate_fn=est)
                agg = aggregate(metrics)
                wall = time.perf_counter() - t0
                total_wall += wall
                rec = {
                    "experiment": "LDPC",
                    "amendment": AMENDMENT,
                    "k": k,
                    "channel": channel,
                    "snr_db": snr,
                    "bits_per_component": B,
                    "a_bits": 6 * B,
                    "e_bits": 2 * k,
                    "code": link.code.params(),
                    "episodes": EVAL_EPISODES,
                    "test_seed": config.TEST_SEED,
                    "t_max": config.T_MAX,
                    "non_frozen": True,
                    "protocol_note": ("5G NR LDPC BG2 without the 38.212 "
                                      "rate-matching chain; evaluation only"),
                    "aggregate_ldpc": agg,
                    "wall_s": round(wall, 1),
                }
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rec, indent=2, default=str))
                n_new += 1
                print(f"   k={k:>2} {channel:<8} snr={snr:5.1f} "
                      f"SR={agg['success_rate']:.3f} ({wall:.1f}s)", flush=True)

    print(f"-- finished: {n_new} new slots, wall={total_wall:.1f}s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def cmd_summary():
    rows = []
    missing = 0
    for k in K_LIST:
        for channel in CHANNELS:
            for snr in SNR_GRID:
                p = eval_path(k, channel, snr)
                if not p.exists():
                    missing += 1
                    continue
                d = json.loads(p.read_text(encoding="utf-8"))
                rows.append({"k": k, "channel": channel, "snr_db": snr,
                             "success_rate": d["aggregate_ldpc"]["success_rate"]})

    by_unit = {}
    for row in rows:
        by_unit.setdefault((row["k"], row["channel"]), []).append(row)

    means = {f"k{k}_{ch}": sum(r["success_rate"] for r in v) / len(v)
             for (k, ch), v in by_unit.items()}

    summary = {
        "experiment": "LDPC",
        "amendment": AMENDMENT,
        "registered": "2026-11-02",
        "episodes": EVAL_EPISODES,
        "snr_grid": list(SNR_GRID),
        "k_list": list(K_LIST),
        "expected_slots": len(K_LIST) * len(CHANNELS) * len(SNR_GRID),
        "completed_slots": len(rows),
        "missing_slots": missing,
        "status": "complete" if missing == 0 else "incomplete",
        "mean_over_snr": means,
        "rows": rows,
    }
    summary_path().parent.mkdir(parents=True, exist_ok=True)
    summary_path().write_text(json.dumps(summary, indent=2, default=str),
                              encoding="utf-8")

    print(f"slots: {len(rows)}/{summary['expected_slots']} "
          f"(missing {missing}) -> {summary['status']}", flush=True)
    print(f"\n{'k':>3} {'channel':<9} " + " ".join(f"{s:>6g}" for s in SNR_GRID)
          + "   mean", flush=True)
    for k in K_LIST:
        for ch in CHANNELS:
            key = (k, ch)
            if key not in by_unit:
                continue
            vals = {r["snr_db"]: r["success_rate"] for r in by_unit[key]}
            line = " ".join(f"{vals.get(s, float('nan')):6.3f}" for s in SNR_GRID)
            print(f"{k:>3} {ch:<9} {line}   {means[f'k{k}_{ch}']:.3f}",
                  flush=True)
    print(f"\nwrote {summary_path().relative_to(ROOT)}", flush=True)
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "verify":
        return cmd_verify()
    if cmd == "evaluate":
        limit = None
        for a in argv[1:]:
            if a.startswith("--limit="):
                limit = int(a.split("=", 1)[1])
        return cmd_evaluate(limit)
    if cmd == "summary":
        return cmd_summary()
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
