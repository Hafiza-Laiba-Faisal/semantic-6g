"""Step 9A preflight benchmark (infrastructure-only, NON-scientific).

Tiny wall-clock timings on smoke-scale configs to enable CAUTIOUS
extrapolation to the full M1-M5 workload. No scientific output is
produced; nothing here is a result. Output: results/benchmark/
step9a_benchmark.json (gitignored).

Run from the repository root:
    .venv/Scripts/python.exe tools/benchmark_step9a.py
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from project import config
from project.evaluation.evaluate import Condition, evaluate_condition
from project.models.reconstruction_jscc import ReconstructionDeepJSCC
from project.training.train_reconstruction import train_reconstruction
from project.training.train_task_oriented import train_task_oriented

OUT_PATH = Path("results/benchmark/step9a_benchmark.json")


def environment_report() -> dict:
    ram_total = ram_avail = None
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        ram_total, ram_avail = st.ullTotalPhys, st.ullAvailPhys
    except Exception:
        pass

    def dir_size(path, exclude):
        total = 0
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in exclude]
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": (torch.version.cuda
                         if torch.cuda.is_available() else None),
        "cpu": (platform.processor() or platform.machine()),
        "cpu_count_logical": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "ram_total_gb": None if ram_total is None else round(ram_total / 2**30, 2),
        "ram_available_gb": None if ram_avail is None else round(ram_avail / 2**30, 2),
        "disk_free_gb": round(shutil.disk_usage(".").free / 2**30, 2),
        "code_size_mb": round(dir_size(".", {".venv", ".git", "wheels",
                                             "__pycache__", ".freebuff",
                                             ".pytest_cache", "results"})
                              / 2**20, 2),
        "results_size_mb": round(dir_size("results", set()) / 2**20, 2)
        if Path("results").exists() else 0.0,
    }


def bench(name, fn, units=None):
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    entry = {"name": name, "wall_s": round(dt, 3)}
    if units:
        entry["units"] = units
        entry["s_per_unit"] = round(dt / units, 5)
    print(f"  {name:34s} {dt:8.2f} s"
          + (f"  ({units} units, {dt / units * 1000:.1f} ms/unit)"
             if units else ""))
    return entry


def main():
    print("=" * 72)
    print("STEP 9A PREFLIGHT BENCHMARK (infrastructure-only, non-scientific)")
    print("=" * 72)
    env = environment_report()
    print(json.dumps(env, indent=2))

    results = {"environment": env, "benchmarks": []}

    print("\n-- warmup (discarded) --")
    train_reconstruction(k=1, channel="awgn", seed=42, steps=3,
                         batch_states=256)

    print("\n-- reconstruction training (30 steps x 2048 states) --")
    for channel in ("awgn", "rayleigh"):
        results["benchmarks"].append(bench(
            f"recon_{channel}_k3",
            lambda ch=channel: train_reconstruction(
                k=3, channel=ch, seed=42, steps=30, batch_states=2048),
            units=30 * 2048))
    results["benchmarks"].append(bench(
        "recon_awgn_k54",
        lambda: train_reconstruction(k=54, channel="awgn", seed=42,
                                     steps=30, batch_states=2048),
        units=30 * 2048))

    print("\n-- task training (k=3: 8 steps x T=40 x 64 eps; "
          "k=54 probe: 6 x 30) --")
    for channel in ("awgn", "rayleigh"):
        results["benchmarks"].append(bench(
            f"task_{channel}_k3",
            lambda ch=channel: train_task_oriented(
                k=3, channel=ch, seed=42, steps=8, batch_episodes=64,
                t_max=40, eval_episodes=32),
            units=8 * 64 * 40 + 2 * 32 * 40))
    results["benchmarks"].append(bench(
        "task_awgn_k54_probe",
        lambda: train_task_oriented(k=54, channel="awgn", seed=42, steps=6,
                                    batch_episodes=64, t_max=30,
                                    eval_episodes=16),
        units=6 * 64 * 30 + 2 * 16 * 30))

    print("\n-- evaluation harness (500-episode protocol scale, sampled) --")

    def digital_eval():
        cond = Condition(k=54, channel="rayleigh", snr_db=10.0, t_max=100,
                         bits_per_component=8, episodes=100, seed=config.TEST_SEED)
        return evaluate_condition(cond, methods=["digital"])

    results["benchmarks"].append(bench(
        "eval_digital_rayleigh_k54_x100ep",
        digital_eval, units=100 * 100))

    def neural_eval():
        cond = Condition(k=3, channel="awgn", snr_db=10.0, t_max=100,
                         episodes=250, seed=config.TEST_SEED)
        model = ReconstructionDeepJSCC(k=3)
        return evaluate_condition(cond, methods=["oracle", "reconstruction",
                                                 "task"],
                                  recon_model=model, task_model=model)

    results["benchmarks"].append(bench(
        "eval_neural_awgn_k3_x250ep",
        neural_eval, units=250 * 100 * 2))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {OUT_PATH} (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
