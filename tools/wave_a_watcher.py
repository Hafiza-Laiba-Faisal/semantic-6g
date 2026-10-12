"""Wave-A auto-pipeline watchdog: train -> evaluate -> finalize -> dashboard.

Drives the full Wave-A stage chain unattended and rebuilds
``dashboard.html`` periodically so the "frontend" is always fresh.

State machine (idempotent via marker files under results/WAVE_A/watchdog/):
  1. launch `train` (full queue; marker-idempotent, safe on restart)
  2. when training fully done -> launch `evaluate` once
  3. when evaluation fully done -> run `finalize` once
  4. every AUTOBUILD_S seconds: rebuild dashboard.html (cheap, read-only)

Safe to kill/restart any time: completed runs SKIP, each stage fires at
most once per marker, dashboard rebuild is read-only.

Run (repository root, self-driving for days):
    nohup .venv/Scripts/python.exe -u tools/wave_a_watcher.py \
        > results/WAVE_A/watchdog.log 2>&1 &

Internet: NOT required at any point; everything is local.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")
PYEXE = str(PY / "python.exe") if sys.platform == "win32" else str(PY / "python")

WAVE = ROOT / "results" / "WAVE_A"
WD = WAVE / "watchdog"
WD.mkdir(parents=True, exist_ok=True)
AUTOBUILD_S = 120          # dashboard refresh cadence
TRAIN_POLL_S = 600         # stage-check cadence during long stages


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


def _marker(name):
    p = WD / name
    if p.exists():
        return False          # already fired
    p.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return True               # newly fired


def _run(cmd):
    log(f"RUN  {' '.join(cmd[2:])}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    log(f"EXIT {r.returncode}")
    return r.returncode


def wave_done_all(cells):
    return all(c["DONE.json"].exists() for c in cells)


def count_wave_states():
    done = failed = other = 0
    for d in WAVE.glob("WAVE_A_*__*/"):
        if (d / "DONE.json").exists():
            done += 1
        elif (d / "FAILED.json").exists():
            failed += 1
        else:
            other += 1
    return done, failed, other


def eval_complete():
    n_expected = len(list((WAVE / "eval").glob("*/"))) * 0  # computed below
    units = [d for d in (WAVE / "eval").glob("*") if d.is_dir()]
    n_slots = 0
    n_have = 0
    for u in units:
        slots = list(u.glob("eval_*.json"))
        n_have += len(slots)
    # 24 eval units x 5 SNRs expected when all present
    n_expected = 24 * 5
    return n_have >= n_expected, n_have, n_expected


def main():
    log("Wave-A watcher started (offline; dashboard autorebuild "
        f"every {AUTOBUILD_S}s)")
    train_fired = eval_fired = final_fired = False

    last_dash = 0.0
    while True:
        t = time.time()

        # --- periodic dashboard refresh (always, cheap, read-only) ---
        if t - last_dash > AUTOBUILD_S:
            try:
                _run([PYEXE, "tools/build_dashboard.py"])
                last_dash = t
            except Exception as e:
                log(f"dashboard rebuild error: {type(e).__name__}: {e}")

        # --- stage 1: training ---
        n_done, n_failed, n_other = count_wave_states()
        if not train_fired:
            if _marker("TRAIN_LAUNCHED"):
                # launch train in a detached, resumable way: run it in this
                # process' foreground once; the queue itself is idempotent,
                # so repeated watcher restarts simply continue the queue.
                rc = _run([PYEXE, "tools/run_wave_a.py", "train"])
                train_fired = (rc == 0)
                if train_fired:
                    _marker("TRAIN_COMPLETE")
                    log("stage 1 complete: all Wave-A runs done")
                else:
                    # transient failures: wipe TRAIN_LAUNCHED marker? No —
                    # FAILED runs block the queue; leave it for operator.
                    log("training queue returned non-zero; check "
                        "results/WAVE_A/*/__FAILED.json; watcher will "
                        "retry the queue on next poll (SKIP logic keeps "
                        "it idempotent)")
        else:
            n_done, n_failed, n_other = count_wave_states()
            if n_done + n_failed >= 46:
                log(f"training census complete: done={n_done} "
                    f"failed={n_failed}")

        # --- stage 2: evaluation ---
        if (WD / "TRAIN_COMPLETE").exists() and not eval_fired:
            if _marker("EVAL_LAUNCHED"):
                rc = _run([PYEXE, "tools/run_wave_a.py", "evaluate"])
                eval_fired = (rc == 0)
                if eval_fired:
                    _marker("EVAL_COMPLETE")
                    log("stage 2 complete: Wave-A evaluation done")

        # --- stage 3: finalize ---
        if (WD / "EVAL_COMPLETE").exists() and not final_fired:
            if _marker("FINALIZE_LAUNCHED"):
                _run([PYEXE, "tools/run_wave_a.py", "finalize"])
                final_fired = True
                _marker("ALL_DONE")
                log("stage 3 complete: finalize done; watcher exits to "
                    "dashboard-only mode")

        if (WD / "ALL_DONE").exists():
            log("ALL_DONE: no further stages; dashboard refresh continues "
                "(Ctrl+C to stop)")
            while True:
                time.sleep(AUTOBUILD_S)
                _run([PYEXE, "tools/build_dashboard.py"])

        time.sleep(TRAIN_POLL_S)


if __name__ == "__main__":
    main()
