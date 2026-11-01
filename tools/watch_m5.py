"""Wait for the M5a training queue, then run evaluation and finalize.

The training queue is idempotent and marker-driven, so this watcher does not
need to own it: it polls the on-disk markers and, once every declared cell is
terminal (DONE or FAILED), runs `evaluate` and `finalize` exactly once.

A cell is terminal when its run directory carries DONE.json or FAILED.json.
RUNNING.json is a start marker and is deliberately ignored as a liveness
signal (same convention as the Wave runners).

Usage:
    python tools/watch_m5.py                  # wait, then evaluate + finalize
    python tools/watch_m5.py --no-eval        # wait, then finalize only
    python tools/watch_m5.py --max-hours=12   # give up after 12 h
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.run_m5 as m5  # noqa: E402

POLL_SECONDS = 120


def _terminal(cell):
    d = m5.M5_ROOT / cell.run_dirname()
    return (d / "DONE.json").exists() or (d / "FAILED.json").exists()


def _progress():
    cells = m5.m5a_cells()
    done = sum(1 for c in cells if _terminal(c))
    return done, len(cells)


def _run(module_args):
    cmd = [sys.executable, str(ROOT / "tools" / "run_m5.py"), *module_args]
    print(f"[watch] running: {' '.join(module_args)}", flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT))
    print(f"[watch] rc={rc} for {' '.join(module_args)}", flush=True)
    return rc


def main(argv):
    max_hours = next((float(a.split("=", 1)[1]) for a in argv
                      if a.startswith("--max-hours=")), 12.0)
    do_eval = "--no-eval" not in argv
    deadline = time.time() + max_hours * 3600

    print(f"[watch] waiting for M5a ({max_hours:g} h cap, "
          f"poll {POLL_SECONDS}s)", flush=True)
    while time.time() < deadline:
        done, total = _progress()
        print(f"[watch] {time.strftime('%H:%M:%S')} terminal {done}/{total}",
              flush=True)
        if done == total:
            break
        time.sleep(POLL_SECONDS)
    else:
        print("[watch] deadline reached; running finalize on what exists",
              flush=True)

    rc = 0
    if do_eval:
        rc |= _run(["evaluate"])
    rc |= _run(["finalize"])
    print(f"[watch] finished rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
