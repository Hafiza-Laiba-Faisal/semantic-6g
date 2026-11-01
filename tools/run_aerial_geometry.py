"""3GPP UMa-aerial multi-geometry sweep (A-17, evaluation only).

A-10 measured the link budget at ONE geometry (d2D = 500 m, h = 50 m,
fc = 2 GHz), so RQ3 was a single point.  This driver repeats the identical
A-10 evaluation over a 3 x 2 x 2 geometry grid and writes a trend table.

Nothing is trained.  The Wave-C lambda-heavy checkpoints and the frozen
digital baseline are evaluated at effective (post-path-loss) SNR exactly as in
A-10; only the large-scale geometry changes.  The frozen A-10 artifact under
results/AERIAL/ is never touched - every geometry writes to its own
subdirectory of results/AERIAL_GEOM/.

Scope is unchanged from A-10: large-scale path loss and LoS probability only.
No Doppler, no time-correlated shadowing, no aerial-specific small-scale
fading.

Usage:
    python tools/run_aerial_geometry.py            # full 12-geometry sweep
    python tools/run_aerial_geometry.py --reuse    # reuse measured curves
    python tools/run_aerial_geometry.py --only=d2d=200,h=100
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.run_aerial_link as arl  # noqa: E402
from project.channels import aerial  # noqa: E402

OUT_ROOT = ROOT / "results" / "AERIAL_GEOM"
AMENDMENT = "A-17"

D2D_GRID = (200.0, 500.0, 1000.0)
H_GRID = (50.0, 100.0)
FC_GRID = (2.0, 3.5)
EPISODES = 1000
TARGETS = ("0.90", "0.99")


def geometries():
    return [(d, h, fc) for d in D2D_GRID for h in H_GRID for fc in FC_GRID]


def geom_dir(d2d, h, fc):
    return OUT_ROOT / f"geom_d{d2d:.0f}_h{h:.0f}_fc{fc:g}"


def geom_label(d2d, h, fc):
    return f"d2D={d2d:.0f} m, h={h:.0f} m, fc={fc:g} GHz"


def _filter(args):
    sel = next((a.split("=", 1)[1] for a in args
                if a.startswith("--only=")), None)
    if not sel:
        return geometries()
    want = dict(part.split("=", 1) for part in sel.split(","))
    out = []
    for d, h, fc in geometries():
        if "d2d" in want and float(want["d2d"]) != d:
            continue
        if "h" in want and float(want["h"]) != h:
            continue
        if "fc" in want and float(want["fc"]) != fc:
            continue
        out.append((d, h, fc))
    return out


def run_one(d2d, h, fc, reuse=False):
    out_dir = geom_dir(d2d, h, fc)
    summary = out_dir / "step_aerial_summary.json"
    if reuse and summary.exists():
        print(f"-- reuse {geom_label(d2d, h, fc)}")
        return 0
    argv = [f"--d2d={d2d:g}", f"--h={h:g}", f"--fc={fc:g}",
            f"--episodes={EPISODES}",
            f"--out={out_dir.relative_to(ROOT).as_posix()}"]
    if reuse:
        argv.append("--reuse")
    print(f"\n=== {geom_label(d2d, h, fc)} ===")
    return arl.main(argv)


def _row(d2d, h, fc):
    p = geom_dir(d2d, h, fc) / "step_aerial_summary.json"
    if not p.exists():
        return {"d2d_m": d2d, "h_m": h, "fc_ghz": fc, "status": "missing"}
    rec = json.loads(p.read_text())
    aer = rec["aerial"]
    row = {
        "d2d_m": d2d, "h_m": h, "fc_ghz": fc, "status": "measured",
        "d3d_m": rec["geometry"]["d3d_m"],
        "p_los": aer["p_los"],
        "pl_los_db": aer["pl_los_db"],
        "pl_nlos_db": aer["pl_nlos_db"],
        "pl_mean_db": aer["pl_mean_db"],
        "episodes": rec["episodes"],
        "methods": {},
    }
    for name, m in rec["methods"].items():
        row["methods"][name] = {
            "k": m["k"],
            "required_eff_db": m["required_eff_db"],
            "required_tx_db": m["required_tx_db"],
            "aerial_penalty_db": m["aerial_penalty_db"],
        }
    lam54 = row["methods"].get("lam_ctl_heavy_k54", {})
    dig54 = row["methods"].get("digital_k54", {})
    for t in TARGETS:
        a = lam54.get("required_tx_db", {}).get(t)
        b = dig54.get("required_tx_db", {}).get(t)
        row[f"lambda_advantage_k54_at_{t}_db"] = (
            None if a is None or b is None else b - a)
    lam18 = row["methods"].get("lam_ctl_heavy_k18", {})
    dig18 = row["methods"].get("digital_k18", {})
    for t in TARGETS:
        a = lam18.get("required_tx_db", {}).get(t)
        b = dig18.get("required_tx_db", {}).get(t)
        row[f"lambda_advantage_k18_at_{t}_db"] = (
            None if a is None or b is None else b - a)
    return row


def finalize():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [_row(d, h, fc) for d, h, fc in geometries()]
    out = OUT_ROOT / "step_aerial_geometry_summary.json"
    out.write_text(json.dumps({
        "experiment": "AERIAL_GEOM",
        "amendment": AMENDMENT,
        "amendment_label": "AERIAL-GEOM-2026-11-01",
        "question": ("does the A-10 lambda advantage over the classical link "
                     "hold across UAV geometry (distance, altitude, carrier)?"),
        "scope": ("evaluation only; large-scale 3GPP UMa-aerial path loss and "
                  "LoS probability; no Doppler, no shadowing correlation, no "
                  "aerial small-scale fading"),
        "grid": {"d2d_m": list(D2D_GRID), "h_m": list(H_GRID),
                 "fc_ghz": list(FC_GRID)},
        "episodes": EPISODES,
        "targets": list(TARGETS),
        "protocol_note": ("identical method to A-10: evaluate the stored "
                          "Wave-C lambda-heavy checkpoints and the frozen "
                          "digital baseline at effective SNR, then map to "
                          "transmit SNR through the LoS/NLoS mixture"),
        "rows": rows,
        "counts": {
            "geometries": len(rows),
            "measured": sum(1 for r in rows if r["status"] == "measured"),
        },
    }, indent=2, default=str))
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"{'geometry':<30} {'P_LoS':>6} {'PL_mean':>8} "
          f"{'lam k54 tx@.90':>15} {'dig k54 tx@.90':>15} {'adv':>7}")
    for r in rows:
        if r["status"] != "measured":
            continue
        lam = r["methods"].get("lam_ctl_heavy_k54", {}).get(
            "required_tx_db", {}).get("0.90")
        dig = r["methods"].get("digital_k54", {}).get(
            "required_tx_db", {}).get("0.90")
        adv = r.get("lambda_advantage_k54_at_0.90_db")
        print(f"{geom_label(r['d2d_m'], r['h_m'], r['fc_ghz']):<30} "
              f"{r['p_los']:>6.3f} {r['pl_mean_db']:>8.2f} "
              f"{('-' if lam is None else round(lam, 1))!s:>15} "
              f"{('-' if dig is None else round(dig, 1))!s:>15} "
              f"{('-' if adv is None else round(adv, 1))!s:>7}")
    return 0


def main(argv):
    reuse = "--reuse" in argv
    sel = _filter(argv)
    print(f"== 3GPP UMa-aerial multi-geometry sweep ({AMENDMENT}) ==")
    print(f"   {len(sel)} of {len(geometries())} geometries, E={EPISODES}")
    t0 = time.perf_counter()
    bad = 0
    for d, h, fc in sel:
        rc = run_one(d, h, fc, reuse=reuse)
        if rc:
            bad += 1
            print(f"!! geometry failed rc={rc}: {geom_label(d, h, fc)}")
    finalize()
    print(f"-- sweep finished in {time.perf_counter() - t0:.0f}s "
          f"({len(sel) - bad}/{len(sel)} ok)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
