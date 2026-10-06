"""M2 pipeline watchdog — drives the remaining LOCAL M2 stages unattended.

State machine (idempotent via marker files under results/M2/watchdog/):

  1. k30 training complete (12/12) -> launch `evaluate k30` once
  2. k54 training complete (12/12) -> launch `evaluate k54` once
  3. k30 AND k54 training complete -> launch `train k18` and `train k24`
     lanes (parallel, 6 threads each) once
  4. k18 training complete -> launch `evaluate k18` once
  5. k24 training complete -> launch `evaluate k24` once
  6. all 150 eval units present -> run `validate` then `finalize` once,
     write results/M2/watchdog/ALL_DONE, and exit

Every action checks a marker file first and writes it only after spawning,
so a watcher restart can never double-launch a stage. The k12 lane is NOT
driven here (its own process chains `train k12 && evaluate k12`).

Run (repository root):
    nohup .venv/Scripts/python.exe -u tools/m2_pipeline_watcher.py \
        > results/M2/watchdog.log 2>&1 &
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
M2 = ROOT / "results" / "M2"
WD = M2 / "watchdog"
WD.mkdir(parents=True, exist_ok=True)

M2_K = (12, 18, 24, 30, 54)


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def done_count(k):
    return len(list(M2.glob(f"*_k{k}_*__*/DONE.json")))


def eval_count(k):
    return len(list((M2 / "eval").glob(f"k{k}_*/eval_*.json")))


def train_complete(k):
    return done_count(k) >= 12


def eval_complete(k):
    return eval_count(k) >= 30


def launched(name):
    return (WD / f"{name}.launched").exists()


def mark(name):
    (WD / f"{name}.launched").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def spawn(name, args):
    """Spawn each arg-list as a separate independent background process."""
    logf = open(M2 / f"watch_{name}.log", "w")
    # args is a list of arg-lists e.g. [["train","k18"]] or [["evaluate","k30"]]
    for a in args:
        proc = subprocess.Popen(
            [PY, "-u", "tools/run_step11_m2.py", *a],
            cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0
        )
        log(f"LAUNCHED {name} {' '.join(a)} (pid {proc.pid})")
    mark(name)


def main():
    log("watcher started; polling every 60 s (24h cap)")
    t0 = time.time()
    while time.time() - t0 < 24 * 3600:
        try:
            # 1-2: eval k30 / k54 as soon as their training completes
            for k in (30, 54):
                if train_complete(k) and not eval_complete(k) \
                        and not launched(f"eval_k{k}"):
                    spawn(f"eval_k{k}", [["evaluate", str(k)]])
            # 3: k18 + k24 lanes once both k30/k54 finished training
            if train_complete(30) and train_complete(54):
                for k in (18, 24):
                    if not train_complete(k) and not launched(f"train_k{k}"):
                        spawn(f"train_k{k}", [["train", str(k)]])
            # 4-5: eval k18 / k24 when their training completes
            for k in (18, 24):
                if train_complete(k) and not eval_complete(k) \
                        and not launched(f"eval_k{k}"):
                    spawn(f"eval_k{k}", [["evaluate", str(k)]])
            # 4.5: k30/k54 rayleigh evals once k18+k24 trained — their per-k
            # gates stay blocked by the 2 deterministic-failed awgn cells,
            # so drive the clean rayleigh lane explicitly
            if train_complete(18) and train_complete(24) \
                    and not launched("eval_rayleigh_rest"):
                spawn("eval_rayleigh_rest", [["evaluate", "rayleigh"]])
            # 6: final validate + finalize when every k has 30 eval units
            if all(eval_complete(k) for k in M2_K) \
                    and not launched("finalize_m2"):
                r1 = subprocess.run([PY, "-u", "tools/run_step11_m2.py",
                                     "validate"], cwd=ROOT,
                                    capture_output=True, text=True)
                r2 = subprocess.run([PY, "-u", "tools/run_step11_m2.py",
                                     "finalize"], cwd=ROOT,
                                    capture_output=True, text=True)
                log("VALIDATE tail: " + r1.stdout.strip().splitlines()[-1])
                log("FINALIZE tail: " + r2.stdout.strip().splitlines()[-1])
                mark("finalize_m2")
                (WD / "ALL_DONE").write_text(
                    time.strftime("%Y-%m-%d %H:%M:%S"))
                log("ALL DONE — watcher exiting")
                return
        except Exception as e:              # keep the watcher alive
            log(f"watcher error (will retry): {type(e).__name__}: {e}")
        time.sleep(60)
    log("watcher stopped (24h safety cap)")


if __name__ == "__main__":
    main()
