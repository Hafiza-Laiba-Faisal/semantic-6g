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


def _wall_min_label(wall_min):
    """Render a run_all summary wall time without mislabelling units."""
    if not isinstance(wall_min, (int, float)):
        return "?"
    return f"{wall_min:.1f} min ({wall_min / 60:.1f} h)"


def _wave_census_readonly():
    """Exact Wave-A census from the declared cell universe (read-only).

    Unlike ``WaveCell.state()`` this never calls ``dir()`` (which would
    create missing run directories); it only stats marker files.  Marker
    precedence is DONE > FAILED > RUNNING, so a retained historical
    RUNNING marker can never be double-counted.  Returns None if the
    runner module cannot be imported (caller falls back to _census).
    """
    try:
        try:
            from tools.run_wave_a import wave_cells
        except ImportError:
            from run_wave_a import wave_cells
        cells = wave_cells()
    except Exception:                        # noqa: BLE001 (read-only view)
        return None
    counts = {"done": 0, "failed": 0, "running": 0, "missing": 0}
    for c in cells:
        d = ROOT / "WAVE_A" / c.run_dirname()
        if (d / "DONE.json").exists():
            counts["done"] += 1
        elif (d / "FAILED.json").exists():
            counts["failed"] += 1
        elif (d / "RUNNING.json").exists():
            counts["running"] += 1
        else:
            counts["missing"] += 1
    return counts


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
.pill { display: inline-block; padding: 1px 8px; border-radius: 10px;
        font-size: 12px; font-weight: 600; }
.p-ok   { background: #d1e7dd; color: #15803d; }
.p-bad  { background: #f8d7da; color: #b91c1c; }
.p-run  { background: #cfe2ff; color: #1d4ed8; }
.p-pend { background: #e5e7eb; color: #6b7280; }
.logbox { background: #0f172a; color: #d1d5db; font-family: Consolas,
          monospace; font-size: 12px; padding: 10px 12px; border-radius: 6px;
          white-space: pre-wrap; max-height: 260px; overflow-y: auto;
          line-height: 1.45; }
.live-hdr { font-size: 15px; margin: 6px 0; }
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


def _wave_invalid_arms():
    """Arms whose label does not match the trained artifact (declared in the
    Wave-A summary by run_wave_a.finalize)."""
    rec = _load_json(ROOT / "WAVE_A" / "step_wave_a_summary.json")
    inv = rec.get("invalid_arms") or {}
    return {str(k): str(v) for k, v in inv.items()} if isinstance(inv, dict) else {}


def _wave_table(wpooled):
    if not wpooled:
        return ("<p class='dim'>No Wave-A eval artifacts yet — run "
                "<code>python tools/run_wave_a.py train</code> then "
                "<code>evaluate</code>.</p>")
    invalid = _wave_invalid_arms()
    tags = sorted({key[0] for key in wpooled})
    snrs = sorted({s for (_, _, _, s) in wpooled}, key=lambda v: (v is None, v))
    cols = []
    for tag in tags:
        label = html.escape(tag)
        if tag in invalid:
            label = f"<span title='{html.escape(invalid[tag])}'>{label} ⚠</span>"
        arm = [f"<tr><td class='left'><b>{label}</b></td>"]
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
    note = ""
    if invalid:
        items = "; ".join(f"<b>{html.escape(k)}</b>: {html.escape(v)}"
                          for k, v in sorted(invalid.items()))
        note = (f"<p class='dim'>⚠ invalid arm — not a valid ablation: "
                f"{items} (see PROTOCOL_AMENDMENTS.md)</p>")
    return "".join(head) + "".join(cols) + "</table>" + note


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


# ---------------------------------------------------------------------------
# live run_all orchestrator section (frontend for "abhi kya ho raha hai")
# ---------------------------------------------------------------------------

def _stage_order():
    """Canonical 22-stage order from tools/run_all.py (never duplicated)."""
    try:
        from tools.run_all import stages as _st
    except ImportError:
        try:
            from run_all import stages as _st
        except ImportError:
            return []
    return [(s[0], s[2]) for s in _st(None)]       # (stage_id, hint_min)


def _tail(p: Path, n: int = 15) -> str:
    try:
        return "\n".join(p.read_text(encoding="utf-8",
                                     errors="replace").splitlines()[-n:])
    except OSError:
        return "(log not found)"


def _run_all_section():
    """Section 0: live queue status from results/RUN_ALL/* (read-only)."""
    ra = ROOT / "RUN_ALL"
    prog = _load_json(ra / "progress.json") if (ra / "progress.json").exists() else {}
    cur = _load_json(ra / "current.json") if (ra / "current.json").exists() else {}
    summ = (_load_json(ra / "run_all_summary.json")
            if (ra / "run_all_summary.json").exists() else {})
    if "_error" in prog:
        prog = {}
    if "_error" in cur:
        cur = {}
    if "_error" in summ:
        summ = {}
    order = _stage_order()
    if not order:
        return ("<h2>0 · Live run_all queue</h2><p class='dim'>"
                "tools/run_all.py not importable — queue section disabled.</p>")

    parts = ["<h2>0 · Live run_all queue</h2>"]
    # header cards
    n_ok = sum(1 for v in prog.values() if v.get("status") == "ok")
    n_bad = sum(1 for v in prog.values() if v.get("status") == "failed")
    parts.append(
        f"<div class='card'>stages ok <b>{n_ok}</b>/{len(order)}</div>"
        f"<div class='card'>failed <b class='{'bad' if n_bad else ''}'>{n_bad}</b></div>")
    if cur.get("stage"):
        started = cur.get("started_utc", "")
        elapsed = ""
        try:
            t0 = time.mktime(time.strptime(started, "%Y-%m-%dT%H%M%SZ"))
            elapsed = f"{max((time.time() - t0) / 60, 0):.0f} min ago"
        except ValueError:
            pass
        parts.append(
            f"<div class='card'><span class='pill p-run'>RUNNING</span> "
            f"<b>{html.escape(str(cur['stage']))}</b> "
            f"<span class='dim'>{elapsed}</span></div>")
    else:
        status_txt = (summ.get("status") or
                      ("idle" if not prog else "between stages"))
        parts.append(f"<div class='card'><span class='pill p-pend'>"
                     f"{html.escape(str(status_txt))}</span></div>")
    # stage table
    parts.append("<table><tr><th>#</th><th class='left'>stage</th>"
                 "<th>status</th><th>exit</th><th>finished</th>"
                 "<th title='static plan hint, not elapsed wall time'>"
                 "~min (plan)</th></tr>")
    for i, (sid, hint) in enumerate(order, 1):
        v = prog.get(sid, {})
        st = v.get("status", "pending")
        if cur.get("stage") == sid:
            pill, cls = "RUNNING", "p-run"
        elif st == "ok":
            pill, cls = "DONE", "p-ok"
        elif st == "failed":
            pill, cls = "FAILED", "p-bad"
        else:
            pill, cls = "pending", "p-pend"
        parts.append(
            f"<tr><td>{i}</td><td class='left'>{html.escape(sid)}</td>"
            f"<td><span class='pill {cls}'>{pill}</span></td>"
            f"<td>{v.get('exit', '–')}</td>"
            f"<td class='dim'>{html.escape(str(v.get('finished_utc', '–')))}</td>"
            f"<td>{hint}</td></tr>")
    parts.append("</table>")
    # live log tail of the currently running stage
    if cur.get("stage"):
        log_p = ra / f"run_all_{cur['stage']}.log"
        parts.append(f"<p class='live-hdr'>tail · "
                     f"<code>results/RUN_ALL/{html.escape(log_p.name)}</code>"
                     f" (last 15 lines)</p>")
        parts.append(f"<div class='logbox'>{html.escape(_tail(log_p))}</div>")
    if summ:
        parts.append(
            "<p class='sub'>last completed run_all invocation: "
            f"{html.escape(str(summ.get('status', '?')))} · "
            f"{html.escape(_wall_min_label(summ.get('wall_min')))} wall · "
            f"finished {html.escape(str(summ.get('finished_utc', '?')))} · "
            f"failed: "
            f"{html.escape(', '.join(summ.get('failed_stages', [])) or 'none')}"
            " <span class='dim'>(historical summary; live stage state is "
            "shown above)</span></p>")
    return "".join(parts)


def _m2_conditional_accept(done_names):
    """Approved frozen-M2 documented failures (single source: run_step11_m2).

    These cells are formally accepted as failures (amendment 2026-10-19,
    eval-only, not retrained).  The dashboard must show them as ACCOUNTED,
    not as silently missing work.
    """
    try:
        from run_step11_m2 import CONDITIONAL_ACCEPT
    except ImportError:
        try:
            from tools.run_step11_m2 import CONDITIONAL_ACCEPT
        except ImportError:
            return []
    return sorted(r for r in CONDITIONAL_ACCEPT if r not in done_names)


def _wave_b_scope():
    """Coverage contract: approved M2 Rayleigh failures are excluded."""
    units = set()
    for rid in _m2_conditional_accept(set()):
        parts = rid.split("__")[0].split("_")
        if len(parts) == 5 and parts[0] == "M2" and parts[2] == "rayleigh":
            try:
                units.add((int(parts[3][1:]), int(parts[4][4:])))
            except ValueError:
                pass
    total_units = 5 * 3                     # k grid x seeds (Rayleigh)
    per_unit = 5 * 5                        # SNR grid x sigma grid
    return {"excluded": len(units), "total_units": total_units,
            "covered_units": total_units - len(units),
            "expected_cells": (total_units - len(units)) * per_unit,
            "total_cells": total_units * per_unit}


def build():
    m1_done, m1_failed, m1_running = _census(ROOT / "M1", "M1_")
    m2_done, m2_failed, m2_running = _census(ROOT / "M2", "M2_")
    wave = ROOT / "WAVE_A"
    w_counts = _wave_census_readonly()
    if w_counts is None:                     # runner not importable: fallback
        w_done, w_failed, w_running = _census(wave, "WAVE_A_")
        w_counts = {"done": len(w_done), "failed": len(w_failed),
                    "running": len(w_running), "missing": "?"}
    m1_units, m1_slots, m1_have = _eval_census(ROOT / "M1" / "eval")
    m2_units, m2_slots, m2_have = _eval_census(ROOT / "M2" / "eval")
    pooled_m2 = _pooled_frozen("M2", None)
    pooled_m1 = _pooled_frozen("M1", None, methods=("reconstruction", "task"))
    wave_pool = _pooled_wave()
    catalog = (_train_catalog(ROOT / "M1") + _train_catalog(ROOT / "M2")
               + _train_catalog(ROOT / "WAVE_A"))
    catalog.sort(key=lambda r: r["mtime"], reverse=True)

    wave_b_evals = len(list((ROOT / "WAVE_B" / "eval").glob("*/*.json"))) \
        if (ROOT / "WAVE_B" / "eval").is_dir() else 0
    m2_cond = _m2_conditional_accept({p.name for p in m2_done})
    wb = _wave_b_scope()
    wa_units, wa_slots, wa_have = _eval_census(ROOT / "WAVE_A" / "eval")
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta http-equiv='refresh' content='30'>",
        f"<title>FYP Results Dashboard — {time.strftime('%Y-%m-%d %H:%M')}</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>FYP Results Dashboard</h1>",
        f"<p class='sub'>generated {time.strftime('%Y-%m-%d %H:%M:%S')} · "
        "read-only view of <code>results/</code> · no frozen values changed · "
        "auto-refreshes every 30 s</p>",

        _run_all_section(),

        "<div class='note'>Frozen protocol (Step 9C) artifacts are labeled "
        "M1_/M2_. Wave-A runs are <b>non-frozen diagnostic amendments "
        "(E=1000)</b> — never 1:1 comparable with frozen eval (E=5000).</div>",

        "<h2>1 · Overview</h2>",
        f"<div class='card'>M1 training <b>{len(m1_done)}</b>/36 "
        f"<span class='dim'>(failed {len(m1_failed)})</span></div>",
        f"<div class='card'>M1 eval <b>{m1_have}</b>/{m1_slots} slots</div>",
        f"<div class='card'>M2 training <b>{len(m2_done)}</b>/60 "
        f"<span class='dim'>(failed {len(m2_failed)} · conditional-accept "
        f"{len(m2_cond)})</span></div>",
        f"<div class='card'>M2 eval <b>{m2_have}</b>/{m2_slots} slots</div>",
        f"<div class='card'>Wave-A runs <b>{w_counts['done']}</b> done · "
        f"{w_counts['failed']} failed · {w_counts['running']} running · "
        f"{w_counts['missing']} missing</div>",
        f"<div class='card'>Wave-A eval <b>{wa_have}</b>/90 slots "
        f"<span class='dim'>({wa_units}/18 units)</span></div>",
        f"<div class='card'>Wave-B CSI evals <b>{wave_b_evals}</b>/"
        f"{wb['expected_cells']} <span class='dim'>({wb['covered_units']}/"
        f"{wb['total_units']} units · {wb['excluded']} excluded: approved M2 "
        f"failure)</span></div>",

        "<h2>2 · Frozen M2 pooled success rate (3 seeds pooled, E=5000)</h2>",
        "<p class='sub'>pooled over seeds 42/43/44; each cell = mean pooled SR "
        "across the 5-SNR grid {0,5,10,15,20} dB at that bandwidth.</p>",
        _m2_table(pooled_m2),

        "<h2>3 · Wave-A diagnostic success rate by arm (E=1000, non-frozen)</h2>",
        f"<p class='sub'>eval coverage {wa_have}/90 slots over {wa_units}/18 "
        "units. Two documented training failures leave one unit unreachable "
        "(max 85/90); an arm missing from a unit is never imputed.</p>",
        _wave_table(wave_pool),

        "<h2>4 · Training catalog (newest first)</h2>",
        _catalog_table(catalog),

        "<h2>5 · Freshness of key summary files</h2>",
        "<table><tr><th class='left'>file</th><th>mtime</th></tr>",
        "".join(
            f"<tr><td class='left'><code>{html.escape(str(rel))}</code></td>"
            f"<td class='dim'>{_mtime(ROOT / rel)}</td></tr>"
            for rel in ("M2/step11_m2_summary.json", "M1/step10_m1_summary.json",
                        "WAVE_A/step_wave_a_summary.json",
                        "WAVE_B/step_wave_b_summary.json",
                        "RUN_ALL/run_all_summary.json")
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
          f"M1 {len(m1_done)}/36 · M2 {len(m2_done)}/60 · "
          f"Wave-A {w_counts['done']} done · eval M1 {m1_have}/{m1_slots} "
          f"M2 {m2_have}/{m2_slots} Wave-A {wa_have}/{wa_slots}")


if __name__ == "__main__":
    build()
    if "--open" in sys.argv:
        import webbrowser
        webbrowser.open(OUT.resolve().as_uri())
