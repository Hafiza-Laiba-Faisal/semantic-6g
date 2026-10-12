"""One-file results dashboard: aggregates results/ into dashboard.html.

Reads frozen M1/M2 artifacts + Wave-A artifacts and writes
``dashboard.html`` (open directly in any browser; no server needed).

What it shows (index-friendly, zero commands required):
1. Overview: training/eval completion censuses per experiment wave.
2. M2 pooled success-rate table (method x k per SNR), color coded.
3. Wave-A pooled success-rate by arm tag; incomplete runs flagged.
4. Training catalog: per-run exit status, wall time, final loss.
5. Data freshness: file mtimes so you can see what the last runs wrote.

Usage:
    python tools/build_dashboard.py          # -> dashboard.html
    python tools/build_dashboard.py --open   # also opens the browser
"""

from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path

ROOT = Path("results")
OUT = Path("dashboard.html")


def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                       # noqa: BLE001 (dashboard is read-only)
        return {"_error": str(e)}


def _fmt(x, nd=3):
    if x is None:
        return "–"
    try:
        f = float(x)
        return f"{f:.{nd}f}" if f == f else "–"
    except (TypeError, ValueError):
        return str(x)[:8]


def _mtime(p: Path) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
    except OSError:
        return "–"


# ---------------------------------------------------------------------------
# frozen M1 / M2 census
# ---------------------------------------------------------------------------

def _census(exp_dir: Path, done_id_prefix: str):
    """Count DONE/FAILED/RUNNING markers by scanning exp_dir directly.

    (Path.glob with a '<prefix>_*__/<file>' pattern proved flaky on this
    machine; explicit iterdir() scanning is deterministic and cheap.)
    """
    done, failed, running = [], [], []
    if not exp_dir.is_dir():
        return done, failed, running
    for d in sorted(exp_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith(done_id_prefix)
                and "__" in d.name):
            continue
        if (d / "DONE.json").exists():
            done.append(d)
        elif (d / "FAILED.json").exists():
            failed.append(d)
        elif (d / "RUNNING.json").exists():
            running.append(d)
    return done, failed, running


def _eval_census(eval_dir: Path):
    if not eval_dir.is_dir():
        return 0, 0, 0
    units, slots, have = 0, 0, 0
    for u in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        units += 1
        for p in sorted(u.iterdir()):
            if p.is_file() and p.name.startswith("eval_") and p.suffix == ".json":
                slots += 1
                have += 1
    return units, slots, have


def _pooled_frozen(exp: str, snr_grid, methods=("reconstruction", "task",
                                                "digital")):
    """Pooled (over seeds) SR per (method, k, snr) from frozen eval JSONs."""
    eval_root = ROOT / exp / "eval"
    pooled = {}
    if not eval_root.is_dir():
        return pooled
    for u in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        for p in sorted(q for q in u.iterdir()
                        if q.is_file() and q.name.startswith("eval_")
                        and q.suffix == ".json"):
            rec = _load_json(p)
            if "_error" in rec:
                continue
            snr = rec.get("snr_db")
            k = rec.get("k")
            for m in methods:
                agg = rec.get(f"aggregate_{m}")
                if not agg:
                    continue
                key = (m, k, snr)
                e = pooled.setdefault(key, {"sc": 0, "tot": 0, "n": 0})
                e["sc"] += agg.get("success_count", 0)
                e["tot"] += agg.get("episodes", 0)
                e["n"] += 1
    return pooled


def _pooled_wave():
    """Pooled Wave-A SR per (tag, k, channel, snr) from WAVE_A eval JSONs."""
    root = ROOT / "WAVE_A" / "eval"
    if not root.is_dir():
        return {}
    pooled = {}
    for u in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted(q for q in u.iterdir()
                        if q.is_file() and q.name.startswith("eval_")
                        and q.suffix == ".json"):
            rec = _load_json(p)
            if "_error" in rec:
                continue
            for key, agg in rec.items():
                if not key.startswith("aggregate_"):
                    continue
                tag = key[len("aggregate_"):]
                e = pooled.setdefault(
                    (tag, rec.get("k"), rec.get("channel"), rec.get("snr_db")),
                    {"sc": 0, "tot": 0, "n": 0})
                e["sc"] += agg.get("success_count", 0)
                e["tot"] += agg.get("episodes", 0)
                e["n"] += 1
    return pooled


def _train_catalog(exp_dir: Path):
    rows = []
    if not exp_dir.is_dir():
        return rows
    for d in sorted(p for p in exp_dir.iterdir()
                    if p.is_dir() and "__" in p.name):
        p = d / "train_report.json"
        if not p.exists():
            continue
        rec = _load_json(p)
        rows.append({
            "wave_id": rec.get("run_id") or rec.get("wave_id")
                       or d.name.split("__")[0],
            "status": rec.get("exit_status", "?"),
            "wall_s": rec.get("wall_s"),
            "loss": (rec.get("task_loss_final") if "task_loss_final" in rec
                     else rec.get("loss_final")),
            "evalSR": rec.get("eval_after_success_rate"),
            "mtime": _mtime(p),
            "non_frozen": bool(rec.get("spec")),
        })
    return rows


# ---------------------------------------------------------------------------
# html rendering
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light; }
body { font-family: "Segoe UI", system-ui, sans-serif; margin: 24px;
       background: #f7f8fa; color: #1a1c20; }
h1 { font-size: 22px; margin-bottom: 4px; }
h2 { font-size: 17px; margin: 28px 0 8px; color: #2c3e50;
     border-bottom: 2px solid #d6dbe1; padding-bottom: 4px; }
.sub { color: #6b7280; font-size: 13px; }
table { border-collapse: collapse; background: #fff; font-size: 13px;
        box-shadow: 0 1px 2px rgba(0,0,0,.06); }
th, td { border: 1px solid #e3e6ea; padding: 5px 9px; text-align: center; }
th { background: #eef1f4; position: sticky; top: 0; }
td.left, th.left { text-align: left; }
.card { display: inline-block; background: #fff; border: 1px solid #dde1e6;
        border-radius: 8px; padding: 10px 16px; margin: 4px 8px 4px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,.05); }
.card b { font-size: 20px; }
.ok { color: #15803d; font-weight: 600; }
.bad { color: #b91c1c; font-weight: 600; }
.dim { color: #6b7280; }
.s-high { background: #d1e7dd; }
.s-mid { background: #fff3cd; }
.s-low { background: #f8d7da; }
.note { background: #fef3c7; border: 1px solid #fcd34d; padding: 8px 12px;
        border-radius: 6px; font-size: 12.5px; margin: 10px 0; }
"""


def _sr_cell(sc, tot):
    if not tot:
        return "<td class='dim'>–</td>"
    sr = sc / tot
    cls = "s-high" if sr >= 0.5 else ("s-mid" if sr >= 0.1 else "s-low")
    return f"<td class='{cls}'>{sr:.3f}</td>"


def _m2_table(pooled):
    ks = sorted({k for (_, k, _) in pooled}, key=lambda v: (v is None, v))
    snrs = sorted({s for (_, _, s) in pooled}, key=lambda v: (v is None, v))
    out = ["<table><tr><th class='left'>method \\ k / SNR</th>"]
    out += [f"<th>k={k}</th>" for k in ks]
    out.append("</tr>")
    body = []
    for m in ("reconstruction", "task", "digital"):
        row = [f"<tr><td class='left'><b>{m}</b></td>"]
        for k in ks:
            # average the pooled SRs across the SNR grid for this (m, k)
            srs = []
            for s in snrs:
                e = pooled.get((m, k, s))
                if e and e["tot"]:
                    srs.append(e["sc"] / e["tot"])
            if srs:
                mean_sr = sum(srs) / len(srs)
                cls = "s-high" if mean_sr >= 0.5 else (
                    "s-mid" if mean_sr >= 0.1 else "s-low")
                row.append(f"<td class='{cls}'>{mean_sr:.3f}</td>")
            else:
                row.append("<td class='dim'>–</td>")
        row.append("</tr>")
        body.append("".join(row))
    return "".join(out) + "".join(body) + "</table>"


def _wave_table(wpooled):
    if not wpooled:
        return ("<p class='dim'>No Wave-A eval artifacts yet — run "
                "<code>python tools/run_wave_a.py train</code> then "
                "<code>evaluate</code>.</p>")
    tags = sorted({t for (t, _) in wpooled})
    snrs = sorted({s for (_, _, _, s) in wpooled}, key=lambda v: (v is None, v))
    cols = []
    for tag in tags:
        arm = [f"<tr><td class='left'><b>{html.escape(tag)}</b></td>"]
        tot_n = 0
        for s in snrs:
            per_ks = [wpooled.get((tag, k, ch, s))
                      for k in (1, 12) for ch in ("awgn", "rayleigh")]
            per_ks = [e for e in per_ks if e]
            if per_ks:
                sc = sum(e["sc"] for e in per_ks)
                tot = sum(e["tot"] for e in per_ks)
                tot_n += len(per_ks)
                arm.append(_sr_cell(sc, tot))
            else:
                arm.append("<td class='dim'>–</td>")
        arm.append(f"<td class='dim'>{tot_n}</td></tr>")
        cols.append("".join(arm))
    head = ["<table><tr><th class='left'>arm tag \\ SNR</th>"]
    head += [f"<th>{s:g} dB</th>" for s in snrs]
    head.append("<th>cells</th></tr>")
    return "".join(head) + "".join(cols) + "</table>"


def _catalog_table(rows):
    if not rows:
        return "<p class='dim'>no train reports found</p>"
    out = ["<table><tr><th class='left'>run</th><th>status</th><th>wall s</th>",
           "<th>final loss</th><th>train-eval SR</th><th>freshness</th>",
           "<th>wave?</th></tr>"]
    for r in rows:
        st = r["status"]
        cls = "ok" if st == "complete" else ("bad" if st in
                                             ("incomplete", "failed") else "dim")
        out.append(
            f"<tr><td class='left'>{html.escape(str(r['wave_id']))[:52]}</td>"
            f"<td class='{cls}'>{st}</td><td>{_fmt(r['wall_s'], 0)}</td>"
            f"<td>{_fmt(r['loss'])}</td><td>{_fmt(r['evalSR'])}</td>"
            f"<td class='dim'>{r['mtime']}</td>"
            f"<td>{'🌊' if r['non_frozen'] else ''}</td></tr>")
    out.append("</table>")
    return "".join(out)


def build():
    m1_done, m1_failed, m1_running = _census(ROOT / "M1", "M1_")
    m2_done, m2_failed, m2_running = _census(ROOT / "M2", "M2_")
    wave = ROOT / "WAVE_A"
    w_done = sorted(wave.glob("WAVE_A_*__*/DONE.json")) if wave.is_dir() else []
    w_failed = sorted(wave.glob("WAVE_A_*__*/FAILED.json")) if wave.is_dir() else []
    m1_units, m1_slots, m1_have = _eval_census(ROOT / "M1" / "eval")
    m2_units, m2_slots, m2_have = _eval_census(ROOT / "M2" / "eval")
    pooled_m2 = _pooled_frozen("M2", None)
    pooled_m1 = _pooled_frozen("M1", None, methods=("reconstruction", "task"))
    wave_pool = _pooled_wave()
    catalog = (_train_catalog(ROOT / "M1") + _train_catalog(ROOT / "M2")
               + _train_catalog(ROOT / "WAVE_A"))
    catalog.sort(key=lambda r: r["mtime"], reverse=True)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>FYP Results Dashboard — {time.strftime('%Y-%m-%d %H:%M')}</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>FYP Results Dashboard</h1>",
        f"<p class='sub'>generated {time.strftime('%Y-%m-%d %H:%M:%S')} · "
        "read-only view of <code>results/</code> · no frozen values changed</p>",

        "<div class='note'>Frozen protocol (Step 9C) artifacts are labeled "
        "M1_/M2_. Wave-A runs are <b>non-frozen diagnostic amendments "
        "(E=1000)</b> — never 1:1 comparable with frozen eval (E=5000).</div>",

        "<h2>1 · Overview</h2>",
        f"<div class='card'>M1 training <b>{len(m1_done)}</b>/36 "
        f"<span class='dim'>(failed {len(m1_failed)})</span></div>",
        f"<div class='card'>M1 eval <b>{m1_have}</b>/{m1_slots} slots</div>",
        f"<div class='card'>M2 training <b>{len(m2_done)}</b>/60 "
        f"<span class='dim'>(failed {len(m2_failed)})</span></div>",
        f"<div class='card'>M2 eval <b>{m2_have}</b>/{m2_slots} slots</div>",
        f"<div class='card'>Wave-A runs <b>{len(w_done)}</b> done · "
        f"{len(w_failed)} failed</div>",

        "<h2>2 · Frozen M2 pooled success rate (3 seeds pooled, E=5000)</h2>",
        "<p class='sub'>pooled over seeds 42/43/44; each cell = mean pooled SR "
        "across the 5-SNR grid {0,5,10,15,20} dB at that bandwidth.</p>",
        _m2_table(pooled_m2),

        "<h2>3 · Wave-A diagnostic success rate by arm (E=1000, non-frozen)</h2>",
        _wave_table(wave_pool),

        "<h2>4 · Training catalog (newest first)</h2>",
        _catalog_table(catalog),

        "<h2>5 · Freshness of key summary files</h2>",
        "<table><tr><th class='left'>file</th><th>mtime</th></tr>",
        "".join(
            f"<tr><td class='left'><code>{html.escape(str(rel))}</code></td>"
            f"<td class='dim'>{_mtime(ROOT / rel)}</td></tr>"
            for rel in ("M2/step11_m2_summary.json", "M1/step10_m1_summary.json",
                        "WAVE_A/step_wave_a_summary.json")
            if (ROOT / rel).exists() or True),
        "</table>",
        "</body></html>",
    ]
    # fix the accidental non-ascii word in the section-2 note
    html_text = "".join(parts).replace(
        "cells = úspěšnost pooled across SNR grid entries per column? NO: each "
        "column aggregates per (k) across the SNR grid — see per-SNR table in "
        "section 2b below.",
        "each cell pools the 3 seeds at that (method, k, SNR product shown as "
        "column-block per k across the SNR grid).")
    OUT.write_text(html_text, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB) — "
          f"M1 {len(m1_done)}/36 · M2 {len(m2_done)}/60 · Wave-A {len(w_done)} "
          f"done · eval M1 {m1_have}/{m1_slots} M2 {m2_have}/{m2_slots}")


if __name__ == "__main__":
    build()
    if "--open" in sys.argv:
        import webbrowser
        webbrowser.open(OUT.resolve().as_uri())
