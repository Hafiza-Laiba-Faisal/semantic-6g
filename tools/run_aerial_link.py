"""3GPP UMa-aerial link-budget evaluation (A-10).

Question: the frozen benchmark is a no-pathloss, no-shadowing Rayleigh link.
If the same closed-loop system is placed on a 3GPP TR 36.777 / TR 38.901
aerial UMa link, how much transmit SNR does it actually need, and does the
lambda-shaped policy keep its advantage?

Method (no training, no frozen file modified):

1. `project/channels/aerial.py` gives P_LoS(d2D, h), PL_LoS(d3D, fc),
   PL_NLoS(d3D, h, fc) and the expected loss.  These are deterministic
   large-scale quantities.
2. Under perfect-CSI ZF, a deterministic transmit gain g multiplies the
   signal and not the noise, so the received effective SNR is
       SNR_eff = SNR_tx - PL          (dB)
   exactly.  The fading statistics stay the frozen Rayleigh ones, so the
   trained checkpoints can be evaluated unchanged at SNR_eff.
3. LoS and NLoS are scored as a mixture on the SAME curve:
       SR(SNR_tx) = P_L * SR(SNR_tx - PL_L) + (1 - P_L) * SR(SNR_tx - PL_N)
   This is exact for i.i.d. episodes with an independent LoS/NLoS draw.
4. The required transmit SNR is the smallest SNR_tx reaching a target
   success rate.  The aerial penalty is required_tx - required_eff.

Scope limits (stated, not hidden): large-scale path loss and LoS probability
only.  No Doppler, no small-scale aerial-specific fading, no shadowing term,
no per-episode gain dispersion.  That is the documented scope of aerial.py.

Usage:
    python tools/run_aerial_link.py --d2d=500 --h=50 --fc=2.0
    python tools/run_aerial_link.py --d2d=1000 --h=100 --fc=3.5 --episodes=1000
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from project import config  # noqa: E402
from project.baselines.digital import DigitalBaseline  # noqa: E402
from project.channels import aerial  # noqa: E402
from project.evaluation.evaluate import (Condition, aggregate,  # noqa: E402
                                         generate_realizations, run_method)
from project.experiments.config import M2_B_BY_K  # noqa: E402

OUT = ROOT / "results" / "AERIAL"
K_GRID = (18, 54)
EFF_GRID = tuple(range(-40, 26, 5))          # dB, effective (post-pathloss) SNR
TX_GRID = np.arange(40.0, 141.0, 1.0)        # dB, transmit SNR scan
TARGETS = (0.90, 0.99)


# ---------------------------------------------------------------------------
# model loading (Wave-C heavy checkpoints; nothing is retrained)
# ---------------------------------------------------------------------------

def _wave_c_models():
    import tools.run_wave_c as wc
    out = {}
    for c in wc.cells():
        if (c.tag == "lam_ctl_heavy" and c.channel == "rayleigh"
                and c.seed == 42 and c.k in K_GRID):
            model = c.load_ckpt()
            if model is not None:
                out[c.k] = model
    return out


def _cached_points(path, method):
    """Measured points already on disk for this geometry/episodes, if any."""
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    rec = d.get("methods", {}).get(method)
    if not rec:
        return None
    if float(d["geometry"]["d2d_m"]) != D2D or float(d["geometry"]["h_m"]) != H             or float(d["geometry"]["fc_ghz"]) != FC or int(d["episodes"]) != EPISODES:
        return None
    return {float(p["snr_eff_db"]): p for p in rec["curve_eff"]}


def _curve_for(k, model, episodes, cached=None):
    """Success rate vs EFFECTIVE SNR for one model at bandwidth k."""
    pts = []
    cached = cached or {}
    for eff in EFF_GRID:
        if float(eff) in cached:
            pts.append(cached[float(eff)])
            print(f"   k={k:2d} eff={eff:+4.0f} dB  (cached)", flush=True)
            continue
        cond = Condition(k=k, channel="rayleigh", snr_db=float(eff),
                         t_max=config.T_MAX, episodes=episodes,
                         seed=config.TEST_SEED)
        real = generate_realizations(cond, episodes)

        def est(s, noise_c, h_c, _m=model, _eff=eff):
            return _m.forward(s, _eff, channel="rayleigh", noise=noise_c,
                              h=h_c, denormalize_output=True)

        m = run_method(cond, real, estimate_fn=est)
        agg = aggregate(m)
        pts.append({"snr_eff_db": float(eff),
                    "success_rate": agg["success_rate"],
                    "final_distance_mean": agg["final_distance_mean"],
                    "control_effort_mean": agg["control_effort_mean"]})
        print(f"   k={k:2d} eff={eff:+4.0f} dB  SR={agg['success_rate']:.4f}",
              flush=True)
    return pts


def _digital_curve(k, episodes, bits=2, cached=None):
    pts = []
    cached = cached or {}
    for eff in EFF_GRID:
        if float(eff) in cached:
            pts.append(cached[float(eff)])
            print(f"   digital k={k:2d} eff={eff:+4.0f} dB  (cached)", flush=True)
            continue
        cond = Condition(k=k, channel="rayleigh", snr_db=float(eff),
                         t_max=config.T_MAX, bits_per_component=bits,
                         episodes=episodes, seed=config.TEST_SEED)
        real = generate_realizations(cond, episodes)
        base = DigitalBaseline(k=k, bits_per_component=bits)

        def est(s, noise_c, h_c, _b=base, _eff=eff):
            return _b.forward(s, _eff, channel="rayleigh", noise=noise_c,
                              h=h_c)

        m = run_method(cond, real, estimate_fn=est)
        agg = aggregate(m)
        pts.append({"snr_eff_db": float(eff),
                    "success_rate": agg["success_rate"]})
        print(f"   digital k={k:2d} eff={eff:+4.0f} dB  "
              f"SR={agg['success_rate']:.4f}", flush=True)
    return pts


# ---------------------------------------------------------------------------
# aerial arithmetic
# ---------------------------------------------------------------------------

def _required(curve, target):
    """Smallest x (any axis) where the curve first reaches ``target``."""
    xs = np.array([p["snr_eff_db"] for p in curve], dtype=float)
    ys = np.array([p["success_rate"] for p in curve], dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if ys.max() < target:
        return None
    # first crossing, refined by linear interpolation
    for i in range(1, len(xs)):
        if ys[i] >= target:
            y0, y1 = ys[i - 1], ys[i]
            if y1 == y0:
                return float(xs[i])
            frac = (target - y0) / (y1 - y0)
            return float(xs[i - 1] + frac * (xs[i] - xs[i - 1]))
    return float(xs[0])


def _mixture_curve(curve, p_los, pl_los, pl_nlos):
    """SR(SNR_tx) = P_L*SR(tx-PL_L) + (1-P_L)*SR(tx-PL_N)."""
    xs = np.array([p["snr_eff_db"] for p in curve], dtype=float)
    ys = np.array([p["success_rate"] for p in curve], dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    # Below the measured grid the success rate must fall to zero, not sit at
    # the first measured value.  Add a sentinel one dB below the grid so the
    # interpolation is zero there; above the grid the measured ceiling is the
    # honest (constant) extrapolation.
    xs = np.concatenate([[xs[0] - 1.0], xs])
    ys = np.concatenate([[0.0], ys])
    rows = []
    for tx in TX_GRID:
        # constant extrapolation past the measured grid (do NOT saturate to
        # 1.0: a method whose ceiling is 0.17 must never "reach" 0.90)
        sr = (p_los * float(np.interp(tx - pl_los, xs, ys))
              + (1.0 - p_los) * float(np.interp(tx - pl_nlos, xs, ys)))
        rows.append({"snr_tx_db": float(tx), "success_rate": sr})
    return rows


D2D = H = FC = EPISODES = None


def main(argv):
    global D2D, H, FC, EPISODES
    args = dict(a.split("=", 1) for a in argv
                if a.startswith("--") and "=" in a)
    d2d = float(args.get("--d2d", 500.0))
    h = float(args.get("--h", 50.0))
    fc = float(args.get("--fc", 2.0))
    ref = float(args.get("--ref", 0.0))        # 0 = absolute path loss
    episodes = int(args.get("--episodes", 1000))
    reuse = "--reuse" in argv
    D2D, H, FC, EPISODES = d2d, h, fc, episodes
    path = OUT / "step_aerial_summary.json"

    p_los = aerial.los_probability(d2d, h)
    d3d = (d2d ** 2 + h ** 2) ** 0.5
    pl_los = aerial.los_path_loss_db(d3d, fc)
    pl_nlos = aerial.nlos_path_loss_db(d3d, h, fc)
    pl_mean = p_los * pl_los + (1.0 - p_los) * pl_nlos
    gain_mean = 10.0 ** (-(pl_mean - ref) / 20.0)

    print("== 3GPP UMa-aerial link budget ==")
    print(f"geometry: d2D={d2d:.0f} m, h={h:.0f} m, fc={fc} GHz, "
          f"d3D={d3d:.1f} m, ref={ref} dB")
    print(f"P_LoS={p_los:.4f}  PL_LoS={pl_los:.2f} dB  "
          f"PL_NLoS={pl_nlos:.2f} dB  PL_mean={pl_mean:.2f} dB  "
          f"g_mean={gain_mean:.3e}")

    OUT.mkdir(parents=True, exist_ok=True)
    models = _wave_c_models()
    print(f"checkpoints: {sorted(models)}")

    methods = {}
    for k in K_GRID:
        if k not in models:
            print(f"NOTE: no Wave-C heavy checkpoint for k={k}; skipped")
            continue
        print(f"-- heavy lambda, k={k} --")
        cached = _cached_points(path, f"lam_ctl_heavy_k{k}") if reuse else None
        curve = _curve_for(k, models[k], episodes, cached)
        methods[f"lam_ctl_heavy_k{k}"] = {
            "k": k, "curve_eff": curve,
            "required_eff_db": {f"{t:.2f}": _required(curve, t)
                                for t in TARGETS},
            "mixture_tx": _mixture_curve(curve, p_los, pl_los, pl_nlos),
        }
    for k in K_GRID:
        # the frozen k=54 comparator is B=8, not B=2 (M2_B_BY_K)
        b = M2_B_BY_K[k]
        print(f"-- digital (K=7 conv, B={b}), k={k} --")
        cached = _cached_points(path, f"digital_k{k}") if reuse else None
        curve = _digital_curve(k, episodes, bits=b, cached=cached)
        methods[f"digital_k{k}"] = {
            "k": k, "curve_eff": curve,
            "required_eff_db": {f"{t:.2f}": _required(curve, t)
                                for t in TARGETS},
            "mixture_tx": _mixture_curve(curve, p_los, pl_los, pl_nlos),
        }

    # required TX SNR = first tx in the mixture curve reaching the target
    for name, rec in methods.items():
        req_tx = {}
        for t in TARGETS:
            hit = next((r["snr_tx_db"] for r in rec["mixture_tx"]
                        if r["success_rate"] >= t), None)
            req_tx[f"{t:.2f}"] = hit
        rec["required_tx_db"] = req_tx
        rec["aerial_penalty_db"] = {
            t: (None if req_tx[t] is None or rec["required_eff_db"][t] is None
                else req_tx[t] - rec["required_eff_db"][t])
            for t in req_tx}

    out = OUT / "step_aerial_summary.json"
    out.write_text(json.dumps({
        "experiment": "AERIAL",
        "amendment": "A-10",
        "amendment_label": "AERIAL-3GPP-2026-10-26",
        "scope": ("3GPP TR 36.777 / TR 38.901 UMa-aerial large-scale layer: "
                  "P_LoS, LoS/NLoS path loss, mixture over the same curve. "
                  "No Doppler, no shadowing, no aerial-specific small-scale "
                  "fading."),
        "geometry": {"d2d_m": d2d, "h_m": h, "fc_ghz": fc,
                     "d3d_m": d3d, "reference_pl_db": ref},
        "aerial": {"p_los": p_los, "pl_los_db": pl_los,
                   "pl_nlos_db": pl_nlos, "pl_mean_db": pl_mean,
                   "gain_mean": gain_mean,
                   "formulas": aerial.CHANNEL_INFO["umaaerial"]},
        "eff_grid_db": list(EFF_GRID),
        "episodes": episodes,
        "eff_grid_measured": list(EFF_GRID),
        "reuse": bool(reuse),
        "methods": methods,
        "protocol_note": ("evaluated at effective (post-pathloss) SNR; the "
                          "mixture maps that curve onto transmit SNR with "
                          "P_LoS weights"),
    }, indent=2, default=str))

    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"{'method':<20} {'SR>=0.90 eff':>13} {'tx':>8} {'SR>=0.99 eff':>13} {'tx':>8}")
    for name, rec in methods.items():
        a = rec["required_eff_db"]; b = rec["required_tx_db"]
        print(f"{name:<20} {str(round(a['0.90'],1)) if a['0.90'] else '-':>13} "
              f"{str(b['0.90']) if b['0.90'] else '-':>8} "
              f"{str(round(a['0.99'],1)) if a['0.99'] else '-':>13} "
              f"{str(b['0.99']) if b['0.99'] else '-':>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
