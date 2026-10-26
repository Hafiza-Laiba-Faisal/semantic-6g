"""Polar-coded digital baseline evaluation (A-11).

The frozen digital comparator is a rate-1/2 K=7 convolutional code.  A
reviewer's question is whether the reported "digital" numbers depend on that
one coding choice.  `project/baselines/digital_polar.py` implements a 5G
NR-style CRC-8 polar code with SC decoding behind the same quantize / pack /
QPSK / channel interface (its own 4/4 gates pass in
`tests/test_polar_baseline.py`).  This runner puts it on the frozen M2 grid.

Design:

  k grid        : (12, 18, 24, 30, 54)   -- M2_K_VALUES
  B per k       : {12:1, 18:2, 24:3, 30:4, 54:8}  -- M2_B_BY_K
  SNR grid      : 0, 5, 10, 15, 20 dB    -- M2_SNR_DB
  channels      : awgn, rayleigh
  methods       : polar (new), conv (the frozen baseline, re-run here on the
                  SAME episodes so the contrast is exactly paired)
  episodes      : 1000 (default) or 5000 (the frozen M2 protocol)
  test stream   : config.TEST_SEED, one stream

Why one stream: both baselines are deterministic and untrained.  Re-labelling
the same deterministic result as three "seeds" would be fake replication, so
this runner evaluates once and reports a binomial CI instead.

Nothing is trained and no frozen artifact is written: output goes to
`results/POLAR/` only.

Usage:
    python tools/run_polar_baseline.py smoke
    python tools/run_polar_baseline.py run --episodes=1000
    python tools/run_polar_baseline.py run --episodes=5000
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
from project.evaluation.evaluate import (Condition, aggregate,  # noqa: E402
                                         generate_realizations, run_method)
from project.experiments.config import M2_B_BY_K, M2_K_VALUES, M2_SNR_DB  # noqa: E402

OUT = ROOT / "results" / "POLAR"
CHANNELS = ("awgn", "rayleigh")


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


def cmd_smoke():
    t0 = time.perf_counter()
    a = _evaluate(18, M2_B_BY_K[18], "rayleigh", 10.0, 200,
                  PolarDigitalBaseline)
    t1 = time.perf_counter()
    b = _evaluate(18, M2_B_BY_K[18], "rayleigh", 10.0, 200,
                  DigitalBaseline)
    t2 = time.perf_counter()
    print(f"polar k=18 B=2 rayleigh 10 dB E=200: SR={a['success_rate']:.4f} "
          f"({t1 - t0:.2f}s)")
    print(f"conv  k=18 B=2 rayleigh 10 dB E=200: SR={b['success_rate']:.4f} "
          f"({t2 - t1:.2f}s)")
    print(f"projected E=1000 per condition: polar "
          f"{(t1 - t0) * 5:.1f}s, conv {(t2 - t1) * 5:.1f}s")
    return 0


def cmd_run(argv):
    args = dict(a.split("=", 1) for a in argv if a.startswith("--"))
    episodes = int(args.get("--episodes", 1000))
    methods = args.get("--methods", "polar,conv").split(",")
    classes = {"polar": PolarDigitalBaseline, "conv": DigitalBaseline}

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.perf_counter()
    for i, (k, b, ch, snr) in enumerate(_rows(episodes), 1):
        row = {"k": k, "bits_per_component": b, "channel": ch, "snr_db": snr,
               "episodes": episodes, "test_seed": config.TEST_SEED}
        for name in methods:
            t0 = time.perf_counter()
            agg = _evaluate(k, b, ch, snr, episodes, classes[name])
            row[name] = {"success_rate": agg["success_rate"],
                         "success_count": agg["success_count"],
                         "final_distance_mean": agg["final_distance_mean"],
                         "control_effort_mean": agg["control_effort_mean"],
                         "ci95": _wilson(agg["success_rate"], episodes),
                         "wall_seconds": round(time.perf_counter() - t0, 3)}
        rows.append(row)
        msg = "  ".join(f"{n}={row[n]['success_rate']:.3f}" for n in methods)
        print(f"[{i:2d}/{len(M2_K_VALUES) * len(CHANNELS) * len(M2_SNR_DB)}] "
              f"k={k:2d} B={b} {ch:<8} {snr:4.0f} dB  {msg}", flush=True)

    # pooled-by-(k, channel, method) over the SNR grid, for the report table
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

    out = OUT / f"step_polar_summary_E{episodes}.json"
    out.write_text(json.dumps({
        "experiment": "POLAR",
        "amendment": "A-11",
        "amendment_label": "POLAR-BASELINE-2026-10-26",
        "question": ("do the frozen digital numbers depend on the "
                     "convolutional coding choice? polar (CRC-8, SC) vs "
                     "conv (K=7, rate 1/2) on identical episodes"),
        "design": {"k_grid": list(M2_K_VALUES),
                   "bits_by_k": {str(k): b for k, b in M2_B_BY_K.items()},
                   "snr_grid_db": list(M2_SNR_DB),
                   "channels": list(CHANNELS),
                   "episodes": episodes,
                   "test_seed": config.TEST_SEED,
                   "methods": methods},
        "protocol_note": ("deterministic untrained baselines: one evaluation "
                          "stream, binomial (Wilson) CI, no seed relabelling"),
        "polar_code": {"n_tail_crc": 8, "decoder": "SC",
                       "source": "project/baselines/digital_polar.py"} ,
        "rows": rows,
        "pooled_5snr": summary,
        "wall_seconds_total": round(time.perf_counter() - t_start, 1),
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}  "
          f"({time.perf_counter() - t_start:.1f}s total)")
    print(f"{'k':>3} {'ch':<9} {'polar':>7} {'conv':>7} {'delta':>8}")
    for k in M2_K_VALUES:
        for ch in CHANNELS:
            d = {(s["method"], s["k"], s["channel"]): s
                 for s in summary}
            if ("polar", k, ch) in d and ("conv", k, ch) in d:
                p = d[("polar", k, ch)]["success_rate_pooled_5snr"]
                c = d[("conv", k, ch)]["success_rate_pooled_5snr"]
                print(f"{k:>3} {ch:<9} {p:>7.3f} {c:>7.3f} {p - c:>+8.3f}")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "smoke":
        return cmd_smoke()
    if argv[0] == "run":
        return cmd_run(argv[1:])
    print(f"unknown command {argv[0]!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
