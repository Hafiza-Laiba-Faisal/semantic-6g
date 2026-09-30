"""Step 9B convergence probe (smoke-scale evidence gathering ONLY).

Purpose: determine whether existing smoke-scale training evidence can
inform the training-budget decision (section 5 of the Step-9B prompt).
NOT a scientific result; no method comparison; fixed seed 42.

Measures (deterministic, seed 42):
* reconstruction: 1000 steps, batch 2048, k=3, AWGN + Rayleigh
  -> loss quartiles, late-trend (last-100 vs mid-100 mean), NaN count,
     wall s/step
* task: 150 steps, batch 64, T=100 (the frozen episode horizon),
  k=3, AWGN + Rayleigh -> same trend stats + eval before/after + the
  T=100 per-step wall time (the number Step 9A could not measure)

Output: results/preflight/step9b_convergence_probe.json (gitignored).

Run from the repository root:
    .venv/Scripts/python.exe tools/convergence_probe_step9b.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from project.training.train_reconstruction import train_reconstruction
from project.training.train_task_oriented import train_task_oriented

OUT = Path("results/preflight/step9b_convergence_probe.json")


def trend_stats(history):
    n = len(history)
    q = lambda lo, hi: sum(history[lo:hi]) / max(1, hi - lo)
    mid = q(int(0.4 * n), int(0.6 * n))
    late = q(int(0.9 * n), n)
    return {
        "first": history[0],
        "q25": q(0, max(1, n // 4)),
        "mid": mid,
        "q75": q(int(0.75 * n), n),
        "last": history[-1],
        "late_vs_mid_pct": round(100.0 * (late - mid) / mid, 2),
        "n_steps": n,
    }


def main():
    print("=" * 72)
    print("STEP 9B CONVERGENCE PROBE (smoke-scale evidence, non-scientific)")
    print("=" * 72)
    out = {}

    for channel in ("awgn", "rayleigh"):
        t0 = time.perf_counter()
        _, rep = train_reconstruction(k=3, channel=channel, seed=42,
                                      steps=1000, batch_states=2048)
        wall = time.perf_counter() - t0
        stats = trend_stats(rep["history"])
        stats["wall_s_per_step"] = round(wall / 1000, 4)
        stats["nan_batches"] = rep["nan_batches"]
        out[f"recon_{channel}"] = stats
        print(f"recon {channel:8s} k=3: {stats['first']:.4f} -> "
              f"{stats['last']:.4f}  late-vs-mid {stats['late_vs_mid_pct']:+.1f}%  "
              f"{stats['wall_s_per_step']*1000:.1f} ms/step  nan={stats['nan_batches']}")

    for channel in ("awgn", "rayleigh"):
        t0 = time.perf_counter()
        _, rep = train_task_oriented(k=3, channel=channel, seed=42, steps=150,
                                     batch_episodes=64, t_max=100,
                                     eval_episodes=32)
        wall = time.perf_counter() - t0
        totals = [h["total"] for h in rep["history"]]
        stats = trend_stats(totals)
        stats["wall_s_per_step_T100"] = round(wall / 150, 4)
        stats["nan_batches"] = rep["nan_batches"]
        stats["eval_before_total"] = rep["eval_before"]["total"]
        stats["eval_after_total"] = rep["eval_after"]["total"]
        stats["eval_before_success"] = rep["eval_before"]["success_rate"]
        stats["eval_after_success"] = rep["eval_after"]["success_rate"]
        out[f"task_{channel}"] = stats
        print(f"task  {channel:8s} k=3 T=100: {stats['first']:.1f} -> "
              f"{stats['last']:.1f}  late-vs-mid {stats['late_vs_mid_pct']:+.1f}%  "
              f"{stats['wall_s_per_step_T100']*1000:.0f} ms/step  "
              f"eval_total {stats['eval_before_total']:.1f} -> "
              f"{stats['eval_after_total']:.1f}  "
              f"SR {stats['eval_before_success']:.2f} -> "
              f"{stats['eval_after_success']:.2f}  nan={stats['nan_batches']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {OUT} (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
