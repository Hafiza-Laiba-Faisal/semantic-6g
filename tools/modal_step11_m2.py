"""Step 11 — M2 execution on Modal (heavy multi-lane cloud CPU, user-approved).

WHY: the frozen-protocol M2 run is split into DISJOINT k-partitions so the
cloud (this app) and the local machine can train SIMULTANEOUSLY without
collision. This app trains k in {12, 18, 24} (3 lanes x ~6 threads on a
32-vCPU container); the local lanes cover k in {30, 54}. The SAME code,
identity system (completed SKIP; frozen 9C RETRAIN policy) and torch
2.4.1+cpu stack run everywhere, so results merge cleanly: whatever a
partition produces is skipped by every other machine afterwards.

AUTOSAVE (watchdog): a background thread commits the cloud volume every
5 minutes, so every newly finished run is persisted even if the container
dies unexpectedly (e.g. credits exhausted). Resume is exact at RUN level:
DONE runs SKIP, interrupted runs are RETRAINED deterministically (9C).

Protocol safety: this app contains NO protocol values of its own — it only
invokes tools/run_step11_m2.py (verify/train/evaluate/validate/finalize),
whose frozen-protocol verify gate must pass in the cloud image or the run
aborts. M1 results are uploaded read-only so the executor's M1-intactness
check passes; they are never modified.

Push the LOCAL results snapshot into the cloud volume (~1 min) so that
already-completed runs SKIP instead of retraining:
    py -3.11 -m modal --profile hafiza-laiba-lcwu run tools/modal_step11_m2.py::sync_up

Launch (detached-safe: the laptop may be closed after launch):
    py -3.11 -m modal --profile hafiza-laiba-lcwu run --detach tools/modal_step11_m2.py::main

Live progress:
    py -3.11 -m modal --profile hafiza-laiba-lcwu run tools/modal_step11_m2.py::peek

Fetch results back (merges into local results/M2 via the volume CLI):
    py -3.11 -m modal --profile hafiza-laiba-lcwu volume get semantic6g-m2 M2 fetch_m2
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.4.1+cpu",
        "numpy==1.26.4",
        "matplotlib",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .add_local_dir(
        ".",
        remote_path="/repo",
        ignore=[".venv", ".git", "results", "__pycache__", "*.pyc",
                ".freebuff", "node_modules", "fetch_modal", "fetch_m1b",
                "fetch_m2", ".agents"],
    )
)

app = modal.App("semantic6g-step11-m2", image=image)

volume = modal.Volume.from_name("semantic6g-m2", create_if_missing=True)

LANES = ("k12", "k18", "k24")   # disjoint k-partitions for THIS cloud app


def _run(cmd, env_extra=None):
    import os
    import subprocess
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(cmd, cwd="/repo", capture_output=True, text=True,
                          env=env)
    tail = proc.stdout[-6000:] + ("\n[stderr] " + proc.stderr[-1500:]
                                  if proc.returncode else "")
    print(tail, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"stage failed ({' '.join(cmd)}); rc="
                           f"{proc.returncode}")
    return proc.stdout


def _counts():
    from pathlib import Path
    root = Path("/repo/results/M2")
    done = len(list(root.glob("*__*/DONE.json"))) if root.exists() else 0
    ev = len(list(root.glob("eval/*/eval_*.json"))) if root.exists() else 0
    return done, ev


@app.function(cpu=32, timeout=60 * 60 * 6, volumes={"/repo/results": volume})
def run_all():
    """3 k-partition training lanes + watchdog autosave, then paired eval."""
    import os
    import subprocess
    import threading
    import time

    env = dict(os.environ, OMP_NUM_THREADS="6", MKL_NUM_THREADS="6")
    t0 = time.time()
    print("== cloud frozen-protocol verify (M2 gate) ==", flush=True)
    _run(["python", "tools/run_step11_m2.py", "verify"])

    # watchdog: autosave the volume every 5 min so no finished run is lost
    stop = threading.Event()

    def watchdog():
        last_done, last_ev = -1, -1
        while not stop.is_set():
            stop.wait(300)
            if stop.is_set():
                return
            try:
                d, e = _counts()
                if (d, e) != (last_done, last_ev):
                    volume.commit()
                    print(f"[watchdog] autosaved: train DONE={d}/60 "
                          f"eval={e}/150", flush=True)
                    last_done, last_ev = d, e
            except Exception as ex:      # never let autosave kill the run
                print(f"[watchdog] commit skipped: {type(ex).__name__}: {ex}",
                      flush=True)

    wt = threading.Thread(target=watchdog, daemon=True)
    wt.start()

    print(f"== training: {len(LANES)} disjoint k-lanes in parallel "
          f"{LANES} ==", flush=True)
    procs = {
        lane: subprocess.Popen(
            ["python", "tools/run_step11_m2.py", "train", lane],
            cwd="/repo", stdout=open(f"/tmp/train_{lane}.log", "w"),
            stderr=subprocess.STDOUT, env=env)
        for lane in LANES
    }
    for lane, proc in procs.items():
        proc.wait()
        print(f"-- lane {lane} rc={proc.returncode}", flush=True)
        print(open(f"/tmp/train_{lane}.log").read()[-3000:], flush=True)
    stop.set()
    wt.join(timeout=10)
    if any(p.returncode for p in procs.values()):
        volume.commit()
        raise RuntimeError("a training lane failed; inspect the logs above")

    volume.commit()
    print(f"[checkpoint] training persisted: DONE={_counts()[0]}/60",
          flush=True)

    print("== 3-method paired evaluation (k-lanes) ==", flush=True)
    ev = {
        lane: subprocess.Popen(
            ["python", "tools/run_step11_m2.py", "evaluate", lane],
            cwd="/repo", stdout=open(f"/tmp/eval_{lane}.log", "w"),
            stderr=subprocess.STDOUT, env=env)
        for lane in LANES
    }
    for lane, proc in ev.items():
        proc.wait()
        print(f"-- eval lane {lane} rc={proc.returncode}", flush=True)
        print(open(f"/tmp/eval_{lane}.log").read()[-2000:], flush=True)
    if any(p.returncode for p in ev.values()):
        volume.commit()
        raise RuntimeError("an evaluation lane failed; inspect the logs above")

    volume.commit()
    print("== validate + finalize ==", flush=True)
    _run(["python", "tools/run_step11_m2.py", "validate"])
    _run(["python", "tools/run_step11_m2.py", "finalize"])
    volume.commit()
    print(f"== cloud M2 partition finished in {(time.time() - t0) / 60:.1f} "
          f"min ==", flush=True)


@app.local_entrypoint()
def main():
    # spawn(): the local client exits immediately while the cloud function
    # keeps running, so a client shutdown can never cancel the in-flight run.
    run_all.spawn()
    print("spawned detached M2 run_all (lanes k12/k18/k24); cloud continues "
          "independently of this client")
    print("\nNEXT: fetch results with")
    print("  py -3.11 -m modal --profile hafiza-laiba-lcwu volume get "
          "semantic6g-m2 M2 fetch_m2")


@app.local_entrypoint()
def sync_up():
    """Upload the LOCAL results snapshot (M1 read-only + M2 partial) so
    completed runs SKIP in cloud and the M1-intactness check passes."""
    import os
    with volume.batch_upload() as batch:
        if os.path.isdir("results/M1"):
            batch.put_directory("results/M1", "/M1")
            print("M1 snapshot uploaded (read-only, for the intactness gate)")
        if os.path.isdir("results/M2"):
            batch.put_directory("results/M2", "/M2")
            print("M2 snapshot uploaded (completed runs will SKIP)")
        else:
            print("no local results/M2 yet — starting from an empty volume")


@app.function(cpu=1, timeout=60 * 5, volumes={"/repo/results": volume})
def _peek_remote():
    from pathlib import Path
    root = Path("/repo/results/M2")
    done_ids = sorted(p.parent.name for p in root.glob("*__*/DONE.json"))
    eval_files = sorted(p.name for p in root.glob("eval/*/*.json"))
    return (f"train DONE: {len(done_ids)}/60\n"
            + "\n".join("  " + d for d in done_ids[-12:])
            + f"\neval files: {len(eval_files)}/450")


@app.local_entrypoint()
def peek():
    """Live cloud progress: DONE markers + eval files in the volume."""
    print(_peek_remote.remote())
