"""5G NR polar comparator evaluation (A-15).

The frozen "digital" comparator is a rate-1/2 K=7 convolutional code; A-11
added a CRC-8 polar with SC decoding but without the 5G construction or rate
matching.  `project/baselines/polar5g.py` implements the specified 5G NR
polar chain (TS 38.212): CRC-6/11, the 1024-entry reliability sequence, the
sub-block interleaver, bit selection with rate-matching pre-freezing, and
CRC-aided SCL decoding with list L = 8.  It is evaluated on exactly the same
payload, the same k channel uses and the same paired episodes as the frozen
comparators.

Design (pre-registered in A-15, no cell was run before that text except the
five-cell timing smoke it discloses):

  k grid        : (12, 18, 24, 30, 54)   -- M2_K_VALUES
  B per k       : {12:1, 18:2, 24:3, 30:4, 54:8}  -- M2_B_BY_K
  SNR grid      : 0, 5, 10, 15, 20 dB    -- M2_SNR_DB
  channels      : awgn, rayleigh
  methods       : polar5g (new), conv, polar (both re-run for exact pairing)
  episodes      : 1000 (default) or 5000
  test stream   : config.TEST_SEED, one stream

Nothing is trained and no frozen artifact is written: output goes to
`results/POLAR5G/` only.

Usage:
    python tools/run_polar5g.py verify
    python tools/run_polar5g.py smoke
    python tools/run_polar5g.py run --episodes=1000
    python tools/run_polar5g.py run --episodes=1000 --methods=polar5g
"""

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project import config  # noqa: E402
from project.baselines.digital import DigitalBaseline  # noqa: E402
from project.baselines.digital_polar import PolarDigitalBaseline  # noqa: E402
from project.baselines.polar5g import Polar5GCode, Polar5GBaseline  # noqa: E402
from project.evaluation.evaluate import (Condition, aggregate,  # noqa: E402
                                         generate_realizations, run_method)
from project.experiments.config import (M2_B_BY_K, M2_K_VALUES,  # noqa: E402
                                        M2_SNR_DB)

OUT = ROOT / "results" / "POLAR5G"
CHANNELS = ("awgn", "rayleigh")
CLASSES = {"polar5g": Polar5GBaseline, "conv": DigitalBaseline,
           "polar": PolarDigitalBaseline}


def _rows(episodes):
    for k in M2_K_VALUES:
        b = M2_B_BY_K[k]
        for ch in CHANNELS:
            for snr in M2_SNR_DB:
                yield k, b, ch, float(snr)


def _evaluate(k, b, ch, snr, episodes, cls):
    cond = Condition(k=k, channel=ch, snr_db=snr, t_max=config.T_MAX,
                     bits_per_component=b, episodes=episodes,
                     seed=config.TEST_SEED)
    real = generate_realizations(cond, episodes)
    base = cls(k=k, bits_per_component=b)

    def est(s, noise_c, h_c, _b=base, _snr=snr, _ch=ch):
        return _b.forward(s, _snr, channel=_ch, noise=noise_c, h=h_c)

    m = run_method(cond, real, estimate_fn=est)
    return aggregate(m)


def _wilson(p, n, z=1.96):
    if n == 0:
        return [None, None]
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _code_table():
    return {str(k): Polar5GCode(6 * M2_B_BY_K[k], 2 * k).params()
            for k in M2_K_VALUES}


def cmd_verify():
    """Structural + noiseless gates for the new link; no evaluation."""
    import numpy as np
    ok = True
    for k in M2_K_VALUES:
        c = Polar5GCode(6 * M2_B_BY_K[k], 2 * k)
        p = c.params()
        assert p["e_bits"] == 2 * k
        assert p["a_bits"] == 6 * M2_B_BY_K[k]
        rng = np.random.default_rng(k)
        payload = rng.integers(0, 2, size=(8, 6 * M2_B_BY_K[k])).astype(np.int8)
        e = c.encode(payload)
        dec = c.decode(np.where(e == 0, 40.0, -40.0))
        good = bool((dec == payload).all())
        ok &= good
        print(f"k={k:2d} A={p['a_bits']:2d} K={p['k_bits']:2d} N={p['N']:3d} "
              f"E={p['e_bits']:3d} {p['mode']:<11} "
              f"rate={p['rate']:.3f} noiseless={'ok' if good else 'FAIL'}")
    print(f"\nA-15 verify: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_smoke():
    t0 = time.perf_counter()
    a = _evaluate(18, M2_B_BY_K[18], "rayleigh", 10.0, 200, Polar5GBaseline)
    t1 = time.perf_counter()
    b = _evaluate(18, M2_B_BY_K[18], "rayleigh", 10.0, 200, DigitalBaseline)
    t2 = time.perf_counter()
    print(f"polar5g k=18 B=2 rayleigh 10 dB E=200: SR={a['success_rate']:.4f} "
          f"({t1 - t0:.2f}s)")
    print(f"conv    k=18 B=2 rayleigh 10 dB E=200: SR={b['success_rate']:.4f} "
          f"({t2 - t1:.2f}s)")
    print(f"projected E=1000, k=18: polar5g {(t1 - t0) * 5:.0f}s "
          f"(real rate is better: decoding is batched)")
    return 0


def cmd_run(argv):
    args = dict(a.split("=", 1) for a in argv if a.startswith("--"))
    episodes = int(args.get("--episodes", 1000))
    methods = args.get("--methods", "polar5g,conv,polar").split(",")
    for name in methods:
        if name not in CLASSES:
            raise SystemExit(f"unknown method {name!r}")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.perf_counter()
    n_cells = len(M2_K_VALUES) * len(CHANNELS) * len(M2_SNR_DB)
    for i, (k, b, ch, snr) in enumerate(_rows(episodes), 1):
        row = {"k": k, "bits_per_component": b, "channel": ch, "snr_db": snr,
               "episodes": episodes, "test_seed": config.TEST_SEED}
        for name in methods:
            t0 = time.perf_counter()
            agg = _evaluate(k, b, ch, snr, episodes, CLASSES[name])
            row[name] = {"success_rate": agg["success_rate"],
                         "success_count": agg["success_count"],
                         "final_distance_mean": agg["final_distance_mean"],
                         "control_effort_mean": agg["control_effort_mean"],
                         "ci95": _wilson(agg["success_rate"], episodes),
                         "wall_seconds": round(time.perf_counter() - t0, 3)}
        rows.append(row)
        msg = "  ".join(f"{n}={row[n]['success_rate']:.3f}" for n in methods)
        print(f"[{i:2d}/{n_cells}] k={k:2d} B={b} {ch:<8} {snr:4.0f} dB  {msg}",
              flush=True)

    pooled = {}
    for row in rows:
        for name in methods:
            key = (name, row["k"], row["channel"])
            acc = pooled.setdefault(key, {"hits": 0, "eps": 0})
            acc["hits"] += row[name]["success_count"]
            acc["eps"] += row["episodes"]
    summary = []
    for (name, k, ch), acc in sorted(pooled.items(),
                                     key=lambda kv: (kv[0][0], kv[0][1],
                                                     kv[0][2])):
        p = acc["hits"] / acc["eps"]
        summary.append({"method": name, "k": k, "channel": ch,
                        "success_rate_pooled_5snr": p,
                        "ci95": _wilson(p, acc["eps"])})

    out = OUT / f"step_polar5g_summary_E{episodes}.json"
    out.write_text(json.dumps({
        "experiment": "POLAR5G",
        "amendment": "A-15",
        "amendment_label": "POLAR-5G-COMPARATOR-2026-10-28",
        "question": ("does the classical-versus-neural gap depend on the "
                     "classical link being a convolutional code? 5G NR polar "
                     "(CRC-6/11, 5G construction, rate matching, CA-SCL L=8) "
                     "on the frozen grid and identical episodes"),
        "design": {"k_grid": list(M2_K_VALUES),
                   "bits_by_k": {str(k): b for k, b in M2_B_BY_K.items()},
                   "snr_grid_db": list(M2_SNR_DB),
                   "channels": list(CHANNELS),
                   "episodes": episodes,
                   "test_seed": config.TEST_SEED,
                   "methods": methods},
        "protocol_note": ("deterministic untrained baselines: one evaluation "
                          "stream, binomial (Wilson) CI, no seed relabelling"),
        "polar5g_code": _code_table(),
        "spec_deviations": [
            "input bit interleaver: downlink-only feature, not used for UCI",
            "channel interleaver: fixed permutation, no burst errors here",
            "PC bits: not implemented (also omitted by the Sionna reference)",
            "k=12 / A=6 is below the 5G polar minimum and is an extension",
            "CRC-11 for A>=20 makes the 5G rate higher than the conv rate",
        ],
        "rows": rows,
        "pooled_5snr": summary,
        "wall_seconds_total": round(time.perf_counter() - t_start, 1),
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}  "
          f"({time.perf_counter() - t_start:.1f}s total)")

    d = {(s["method"], s["k"], s["channel"]): s for s in summary}
    print(f"{'k':>3} {'ch':<9} {'5g':>7} {'conv':>7} {'polar':>7} "
          f"{'5g-conv':>8}")
    for k in M2_K_VALUES:
        for ch in CHANNELS:
            vals = {n: d[(n, k, ch)]["success_rate_pooled_5snr"]
                    for n in ("polar5g", "conv", "polar")
                    if (n, k, ch) in d}
            if "polar5g" in vals and "conv" in vals:
                print(f"{k:>3} {ch:<9} {vals['polar5g']:>7.3f} "
                      f"{vals['conv']:>7.3f} "
                      f"{vals.get('polar', float('nan')):>7.3f} "
                      f"{vals['polar5g'] - vals['conv']:>+8.3f}")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "verify":
        return cmd_verify()
    if argv[0] == "smoke":
        return cmd_smoke()
    if argv[0] == "run":
        return cmd_run(argv[1:])
    print(f"unknown command {argv[0]!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
