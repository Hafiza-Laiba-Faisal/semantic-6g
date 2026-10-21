"""Master orchestrator - run EVERYTHING unattended, on one PC, for days.

Design goals (user request 2026-10-13):
* ONE command trains + evaluates + finalizes every round in dependency
  order: M1 -> M2 -> M3/M4 -> Wave-A -> Wave-B.
* UNATTENDED-SAFE: no prompts, per-stage log files, per-stage retry once,
  failed stages never abort the remaining queue; final summary + exit
  code report what failed.
* CRASH-SAFE / RESUMABLE: every inner runner is already idempotent
  (DONE/FAILED cells SKIP; RUNNING/missing cells RETRAIN), and this
  orchestrator additionally records stage outcomes in
  results/RUN_ALL/progress.json - relaunching after a power cut simply
  continues from the first non-ok stage.
* FROZEN SAFETY: default mode never touches existing results (the inner
  runners skip completed work).  ``--fresh`` MOVES every results/M1..M4,
  WAVE_A, WAVE_B directory into results/_archive/<UTC-stamp>/ (archive,
  never delete) and clears progress, so training restarts from scratch.

Usage (repository root, inside the venv):
    python tools/run_all.py status        # stage list + completion census
    python tools/run_all.py run           # run/continue the whole queue
    python tools/run_all.py run --fresh   # archive old results, then retrain all
    python tools/run_all.py run --lane awgn   # lane-split (see note below)
    python tools/run_all.py reset         # clear progress markers only

Exit codes: 0 = complete, 1 = some stage failed (see run_all_summary.json),
3 = stalled / no progress (see HALT.json), 4 = never returned by this
orchestrator itself - it is the INNER "completed with only documented,
user-approved cell failures" code (currently Wave-A training), which this
orchestrator records as stage status "accepted" instead of "failed" and
does not retry.

Lane note: M1/M2/Wave-A accept a channel lane ("awgn"|"rayleigh") so TWO
processes can split the wall clock; run ``--lane awgn`` and ``--lane
rayleigh`` side by side.  Wave-B and M3/M4 ignore the lane (eval-only,
single cells).

For true multi-day unattended operation use tools/launch_all.bat which
relaunches this script in a loop after any crash/power loss.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ALL = ROOT / "results" / "RUN_ALL"
PROGRESS = RUN_ALL / "progress.json"
CURRENT = RUN_ALL / "current.json"        # live "what is running now" marker
HALT = RUN_ALL / "HALT.json"              # written when an invocation made no progress
FP_FILE = RUN_ALL / "last_progress_fingerprint.txt"
DASH_EVERY_S = 120                        # dashboard rebuild cadence
PY = sys.executable

NO_PROGRESS_EXIT = 3                      # "stalled" exit code for the launcher
ACCEPTED_EXIT = 4                         # "completed with only DOCUMENTED
                                          #  failures" (never retried, never
                                          #  counted as a stage failure)

ARCHIVE_DIRS = ("M1", "M2", "M3", "M4", "WAVE_A", "WAVE_B")


# ---------------------------------------------------------------------------
# stage table (dependency order; each entry is idempotent inside)
# ---------------------------------------------------------------------------

def stages(lane: str | None) -> list:
    """Ordered [(stage_id, argv, wall_hint_min)] command queue.

    Log file per stage: results/RUN_ALL/run_all_<stage_id>.log
    """

    def w(script, *args):
        argv = [PY, f"tools/{script}"] + list(args)
        if lane and args and args[0] in ("train", "evaluate", "run") \
                and script != "run_wave_b.py":
            argv.append(lane)
        return argv

    return [
        # ---- frozen M1 (36 neural runs + paired evals) -------------------
        ("m1_verify",     w("run_step10_m1.py", "verify"),    1),
        ("m1_train",      w("run_step10_m1.py", "train"),    360),
        ("m1_evaluate",   w("run_step10_m1.py", "evaluate"), 120),
        ("m1_validate",   w("run_step10_m1.py", "validate"),  10),
        ("m1_finalize",   w("run_step10_m1.py", "finalize"),   5),
        # ---- frozen M2 (60 neural runs + 3-method paired evals) ----------
        ("m2_verify",     w("run_step11_m2.py", "verify"),    1),
        ("m2_train",      w("run_step11_m2.py", "train"),    600),
        ("m2_evaluate",   w("run_step11_m2.py", "evaluate"), 180),
        ("m2_validate",   w("run_step11_m2.py", "validate"),  10),
        ("m2_finalize",   w("run_step11_m2.py", "finalize"),   5),
        # ---- M3 display + M4 secondary digital reference -----------------
        ("m3m4_verify",   w("run_step12_m3_m4.py", "verify"),    1),
        ("m3_run",        w("run_step12_m3_m4.py", "run-m3"),    1),
        ("m4_run",        w("run_step12_m3_m4.py", "run-m4"),   30),
        ("m3m4_validate", w("run_step12_m3_m4.py", "validate"),  5),
        ("m3m4_finalize", w("run_step12_m3_m4.py", "finalize"),  2),
        # ---- Wave-A ablation arms (46 training runs; the long pole) ------
        ("wave_a_verify",   w("run_wave_a.py", "verify"),      1),
        ("wave_a_train",    w("run_wave_a.py", "train"),      744),
        ("wave_a_evaluate", w("run_wave_a.py", "evaluate"),   120),
        ("wave_a_finalize", w("run_wave_a.py", "finalize"),     5),
        # ---- Wave-B CSI-error ablation (eval-only, uses M2 ckpts) --------
        ("wave_b_verify",   w("run_wave_b.py", "verify"),       1),
        ("wave_b_run",      w("run_wave_b.py", "run"),         60),
        ("wave_b_finalize", w("run_wave_b.py", "finalize"),     3),
    ]


# ---------------------------------------------------------------------------
# progress bookkeeping
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    if PROGRESS.exists():
        try:
            return json.loads(PROGRESS.read_text())
        except Exception:
            pass
    return {}


def save_progress(prog: dict) -> None:
    RUN_ALL.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(prog, indent=2))


def _progress_fingerprint(prog: dict) -> str:
    """Stage outcome + first-line-of-error fingerprint.

    Two consecutive invocations with the SAME fingerprint mean the queue made
    no progress at all (same stages, same exit codes).  Retrying forever would
    just rewrite the same logs, so the launcher must stop instead.
    """
    import hashlib

    smooth = {sid: {"status": v.get("status"), "exit": v.get("exit")}
              for sid, v in sorted(prog.items())}
    return hashlib.sha256(
        json.dumps(smooth, sort_keys=True).encode()).hexdigest()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# dashboard refresh (frontend: dashboard.html shows "abhi kya ho raha hai")
# ---------------------------------------------------------------------------

def rebuild_dashboard() -> None:
    """Best-effort rebuild of dashboard.html (read-only w.r.t. results/)."""
    try:
        try:
            from tools import build_dashboard as bd
        except ImportError:
            import build_dashboard as bd
        bd.build()
    except Exception as e:                      # noqa: BLE001 (never kill the queue)
        print(f"          (dashboard rebuild skipped: {e})")


def _dashboard_refresher(stop: threading.Event) -> None:
    """Daemon thread: rebuild dashboard.html every DASH_EVERY_S so a browser
    tab left open on dashboard.html always shows the live queue state."""
    while not stop.wait(DASH_EVERY_S):
        rebuild_dashboard()


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_status(lane=None) -> int:
    prog = load_progress()
    tot_min = 0
    print("== run_all stage queue ==")
    for sid, argv, hint in [(s[0], s[1], s[2]) for s in stages(lane)]:
        st = prog.get(sid, {}).get("status", "pending")
        mark = {"ok": "DONE  ", "accepted": "ACCEPT",
                "failed": "FAILED"}.get(st, "....")
        print(f"  [{mark}] {sid:18s} ~{hint:>4}min  {' '.join(argv[1:])}")
        if st not in ("ok", "accepted"):
            tot_min += hint
    done = sum(1 for v in prog.values() if v.get("status") == "ok")
    accepted = sum(1 for v in prog.values()
                   if v.get("status") == "accepted")
    print(f"-- {done} stages ok"
          + (f", {accepted} accepted (documented failures)" if accepted else "")
          + f"; remaining estimate ~{tot_min/60:.1f} h"
          + (f" [lane: {lane}]" if lane else ""))
    return 0


def cmd_reset() -> int:
    if PROGRESS.exists():
        PROGRESS.unlink()
    print("progress markers cleared (inner DONE markers untouched)")
    return 0


def cmd_fresh() -> int:
    """Archive every results round dir + clear progress (never delete)."""
    stamp = _utc()
    dest = ROOT / "results" / "_archive" / stamp
    moved = []
    for name in ARCHIVE_DIRS:
        src = ROOT / "results" / name
        if src.exists():
            dest.mkdir(parents=True, exist_ok=True)
            src.rename(dest / name)
            moved.append(name)
    for name in ("wave_b_run.log", "M4_run.log"):
        p = ROOT / "results" / name
        if p.exists():
            dest.mkdir(parents=True, exist_ok=True)
            p.rename(dest / name)
            moved.append(name)
    cmd_reset()
    print(f"-- archived to {dest}: {moved or '(nothing to archive)'}")
    return 0


def run_stage(sid: str, argv: list, log_name: str, attempt: int) -> int:
    log_path = RUN_ALL / f"{log_name}.log"
    RUN_ALL.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # Unbuffered child stdout: redirected files are block-buffered by default,
    # which made a multi-hour training log look empty.  This changes only
    # logging latency, never training behaviour/verbosity.
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a") as log:
        log.write(f"\n===== {_utc()} attempt={attempt} "
                  f"cmd={' '.join(argv)} =====\n")
        log.flush()
        proc = subprocess.run(argv, cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def _stage_status(code: int) -> str:
    """Map a stage exit code to a progress status.

    0                  -> "ok"
    ACCEPTED_EXIT (4)  -> "accepted"  (completed; only DOCUMENTED cell
                          failures inside; never retried, never a stage failure)
    anything else      -> "failed"
    """
    if code == 0:
        return "ok"
    if code == ACCEPTED_EXIT:
        return "accepted"
    return "failed"


def cmd_run(lane=None, fresh=False) -> int:
    if fresh:
        cmd_fresh()
    prog = load_progress()
    RUN_ALL.mkdir(parents=True, exist_ok=True)
    if CURRENT.exists():                      # stale marker from a crashed run
        CURRENT.unlink()
    queue = stages(lane)
    t_start = time.perf_counter()
    print(f"== run_all queue: {len(queue)} stages"
          + (f" [lane: {lane}]" if lane else "") + " ==")
    stop_evt = threading.Event()
    refresher = threading.Thread(target=_dashboard_refresher,
                                 args=(stop_evt,), daemon=True)
    rebuild_dashboard()                       # fresh page immediately
    refresher.start()
    failures = []
    for i, (sid, argv, _hint) in enumerate(queue, 1):
        log_name = f"run_all_{sid}"
        if prog.get(sid, {}).get("status") in ("ok", "accepted"):
            print(f"[{i:2d}/{len(queue)}] SKIP {sid} "
                  f"(already {prog[sid]['status']})")
            continue
        print(f"[{i:2d}/{len(queue)}] RUN   {sid}  ({' '.join(argv[1:])})",
              flush=True)
        CURRENT.write_text(json.dumps(
            {"stage": sid, "argv": argv[1:], "started_utc": _utc(),
             "progress": f"{i}/{len(queue)}"}))
        code = run_stage(sid, argv, log_name, attempt=1)
        if code == ACCEPTED_EXIT:
            # Completed, with only declared/documented cell failures inside.
            # Retrying cannot change a frozen deterministic outcome, so we do
            # not burn a second full run (that is exactly the "retrain until
            # the numbers look good" trap this project must avoid).
            print(f"          documented failures only -> ACCEPTED", flush=True)
        elif code != 0:
            print(f"          retry once (exit={code})...", flush=True)
            code = run_stage(sid, argv, log_name, attempt=2)
        if CURRENT.exists():
            CURRENT.unlink()
        status = _stage_status(code)
        wall = round(time.perf_counter() - t_start, 1)
        prog[sid] = {"status": status, "exit": code,
                     "finished_utc": _utc(), "wall_cum_s": wall}
        save_progress(prog)
        label = {"ok": "OK", "accepted": "ACCEPTED",
                 "failed": f"FAILED (exit={code})"}[status]
        print(f"          -> {label} "
              f"| {sid} | log: results/RUN_ALL/{log_name}.log", flush=True)
        if status == "failed":
            # hard gates STOP the queue (a broken frozen assumption makes
            # everything after it meaningless); soft stages continue.
            if sid.endswith("_verify"):
                print(f"!! {sid} is a hard gate - stopping queue")
                failures.append(sid)
                break
            failures.append(sid)
    # ---- final summary ---------------------------------------------------
    stop_evt.set()
    refresher.join(timeout=5)
    failed = [s for s, v in prog.items() if v.get("status") == "failed"]
    done = [s for s, v in prog.items() if v.get("status") == "ok"]
    accepted = [s for s, v in prog.items() if v.get("status") == "accepted"]
    mins = (time.perf_counter() - t_start) / 60
    summary = {
        "finished_utc": _utc(), "ok_stages": sorted(done),
        "accepted_stages": sorted(accepted),
        "failed_stages": sorted(failed), "wall_min": round(mins, 1),
        "lane": lane,
        "status": "complete" if not failed else "incomplete",
    }
    save_progress(prog)
    stalled = False
    if failed:
        fp = _progress_fingerprint(prog)
        prev = FP_FILE.read_text().strip() if FP_FILE.exists() else None
        stalled = prev == fp                # identical outcomes => no progress
        FP_FILE.write_text(fp)
        if stalled:
            summary["status"] = "stalled"
            summary["halt"] = "no-progress"
            HALT.write_text(json.dumps({
                "halted_utc": _utc(),
                "reason": ("two consecutive invocations produced IDENTICAL "
                           "stage outcomes; retrying cannot change anything"),
                "failed_stages": sorted(failed),
                "exit_code": NO_PROGRESS_EXIT,
                "how_to_resume": ("fix the cause, delete "
                                  "results/RUN_ALL/HALT.json, then rerun "
                                  "python tools/run_all.py run"),
            }, indent=2))
    else:
        summary["status"] = "complete"
        for p in (HALT, FP_FILE):
            if p.exists():
                p.unlink()                  # clean run clears stale halt state
    (RUN_ALL / "run_all_summary.json").write_text(json.dumps(summary, indent=2))
    rebuild_dashboard()                       # final state on the page
    print(f"\n== run_all {'COMPLETE' if not failed else 'INCOMPLETE'}: "
          f"{len(done)} ok"
          + (f", {len(accepted)} accepted" if accepted else "")
          + f", {len(failed)} failed, wall {mins/60:.1f} h ==")
    if accepted:
        print("accepted stages (documented cell failures, not retried):",
              ", ".join(sorted(accepted)))
    if failed:
        print("failed stages:", ", ".join(sorted(failed)))
        if stalled:
            print("!! NO PROGRESS since the previous invocation - halting "
                  "instead of looping forever.")
            print("   see results/RUN_ALL/HALT.json (fix, delete it, rerun)")
            return NO_PROGRESS_EXIT
        print("relaunch the SAME command to retry only the missing parts.")
        return 1
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    cmd, rest = args[0], args[1:]
    lane = None
    fresh = False
    i = 0
    while i < len(rest):
        if rest[i] == "--lane" and i + 1 < len(rest):
            lane = rest[i + 1]
            if lane not in ("awgn", "rayleigh"):
                print(f"bad lane {lane!r}")
                sys.exit(1)
            i += 2
        elif rest[i] == "--fresh":
            fresh = True
            i += 1
        else:
            print(f"unknown arg {rest[i]!r}")
            sys.exit(1)
    dispatch = {"status": lambda: cmd_status(lane),
                "run": lambda: cmd_run(lane=lane, fresh=fresh),
                "reset": cmd_reset,
                "fresh": cmd_fresh}
    if cmd not in dispatch:
        print(__doc__)
        sys.exit(1)
    sys.exit(dispatch[cmd]())
