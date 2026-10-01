"""Step 10 — M1 execution on Modal (cloud CPU fallback, user-approved).

WHY: the local machine must be returned; the FROZEN-protocol M1 run
continues on Modal with the SAME code, the SAME identity system
(completed runs SKIP; frozen 9C RETRAIN policy) and the SAME torch
2.4.1+cpu stack. No third party receives the code: it is uploaded into
the USER'S OWN Modal workspace from their own repository, and results
live in the user's own Modal volume. Quality is unchanged: same seeds,
same budgets, same identity hashes — results are protocol-identical.

One-time local setup (repository root, ~3 min):
    .venv/Scripts/python.exe -m pip install modal
    .venv/Scripts/python.exe -m modal setup      # browser login (own account)

Push the LOCAL results snapshot into the cloud volume (~1 min) so that
already-completed runs SKIP instead of retraining:
    .venv/Scripts/python.exe -m modal run tools/modal_step10_m1.py::sync_up

Launch the FULL remaining M1 workload in the cloud (detached-safe: the
laptop may be closed after launch; the cloud function keeps running):
    .venv/Scripts/python.exe -m modal run tools/modal_step10_m1.py

Later, from any machine with the repo + Modal login, fetch results back
(merges into local results/M1):
    .venv/Scripts/python.exe -m modal run tools/modal_step10_m1.py::sync_down

Protocol safety: this app contains NO protocol values of its own — it
only invokes tools/run_step10_m1.py (verify/train/evaluate/validate/
finalize), whose frozen-protocol verify gate must pass in the cloud
image or the run aborts. The volume "semantic6g-m1" persists ~7 days.
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
        # source code only; results come from the volume, venv/git stay out
        ignore=[".venv", ".git", "results", "__pycache__", "*.pyc",
                ".freebuff", "node_modules"],
    )
)

app = modal.App("semantic6g-step10-m1", image=image)

volume = modal.Volume.from_name("semantic6g-m1", create_if_missing=True)


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


@app.function(cpu=16, timeout=60 * 60 * 8, volumes={"/repo/results": volume})
def run_all():
    """Train both channel lanes concurrently, then eval, validate, finalize.

    One serverless invocation does the whole remaining M1 workload: if the
    local laptop disconnects after launching, this function still runs to
    completion and writes everything into the Modal volume.
    """
    import os
    import subprocess
    import time

    env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    t0 = time.time()
    print("== cloud frozen-protocol verify ==", flush=True)
    _run(["python", "tools/run_step10_m1.py", "verify"])

    print("== training: two disjoint channel lanes in parallel ==", flush=True)
    procs = {
        lane: subprocess.Popen(
            ["python", "tools/run_step10_m1.py", "train", lane],
            cwd="/repo", stdout=open(f"/tmp/train_{lane}.log", "w"),
            stderr=subprocess.STDOUT, env=env)
        for lane in ("awgn", "rayleigh")
    }
    for lane, proc in procs.items():
        proc.wait()
        print(f"-- lane {lane} rc={proc.returncode}", flush=True)
        print(open(f"/tmp/train_{lane}.log").read()[-3000:], flush=True)
    if any(p.returncode for p in procs.values()):
        raise RuntimeError("a training lane failed; inspect the logs above")

    volume.commit()

    print("== paired evaluation (both lanes) ==", flush=True)
    ev = {
        lane: subprocess.Popen(
            ["python", "tools/run_step10_m1.py", "evaluate", lane],
            cwd="/repo", stdout=open(f"/tmp/eval_{lane}.log", "w"),
            stderr=subprocess.STDOUT, env=env)
        for lane in ("awgn", "rayleigh")
    }
    for lane, proc in ev.items():
        proc.wait()
        print(f"-- eval lane {lane} rc={proc.returncode}", flush=True)
        print(open(f"/tmp/eval_{lane}.log").read()[-2000:], flush=True)
    if any(p.returncode for p in ev.values()):
        raise RuntimeError("an evaluation lane failed; inspect the logs above")

    volume.commit()
    print("== validate + finalize ==", flush=True)
    _run(["python", "tools/run_step10_m1.py", "validate"])
    _run(["python", "tools/run_step10_m1.py", "finalize"])
    volume.commit()
    print(f"== cloud M1 run finished in {(time.time() - t0) / 60:.1f} min ==",
          flush=True)


@app.local_entrypoint()
def main():
    out = run_all.remote()
    print(out[-3000:])
    print("\nNEXT: fetch results with")
    print("  .venv/Scripts/python.exe -m modal run tools/modal_step10_m1.py::sync_down")


@app.local_entrypoint()
def sync_up():
    """Upload the LOCAL results snapshot so completed runs SKIP in cloud."""
    import os
    if not os.path.isdir("results/M1"):
        print("no local results/M1 yet — starting from an empty volume")
    with volume.batch_upload() as batch:
        batch.put_directory("results/M1", "/M1")
    print("snapshot uploaded to volume 'semantic6g-m1' (/M1)")


@app.local_entrypoint()
def sync_down():
    """Merge cloud results into local results/M1 (existing files kept)."""
    with volume.batch_upload() as batch:
        batch.get_directory("/M1", "results/M1")
    print("cloud results merged into results/M1")


@app.function(cpu=1, timeout=60 * 5, volumes={"/repo/results": volume})
def _peek_remote():
    from pathlib import Path
    root = Path("/repo/results/M1")
    dones = sorted(root.glob("*__*/*.json"))
    done_ids = sorted(p.parent.name for p in root.glob("*__*/DONE.json"))
    eval_files = sorted(p.name for p in root.glob("eval/*/*.json"))
    return (f"train DONE: {len(done_ids)}/36\n"
            + "\n".join("  " + d for d in done_ids)
            + f"\neval units: {len(eval_files)}/198"
            + ("\n  " + ", ".join(eval_files[:8]) if eval_files else ""))


@app.local_entrypoint()
def peek():
    """Live cloud progress: DONE markers + eval units in the volume."""
    print(_peek_remote.remote())
