"""Diagnostic sweep for the failed sanity gate (A6 + gain assessment).

NOTHING here changes configuration or modules. Alternative settings are
injected only in memory to MEASURE their effect, so the user can decide
what (if anything) to freeze. This script exists because the oracle gate
failed at Kp = Kv = 1.0 with full-workspace goal sampling.

Run from the repository root:
    .venv/Scripts/python.exe -m tests.diagnose_configs
"""

from __future__ import annotations

import functools

import torch

from project import config
from project.uav.environment import NavigationEnv, episode_distances
import project.uav.environment as envmod


def main() -> None:
    torch.manual_seed(0)
    env = NavigationEnv()
    n = 5000
    gen = config.derive_generator("episodes", config.TEST_SEED)
    p0, v0, goal = env.sample_starts(n, generator=gen)
    d0 = episode_distances(p0, goal)
    print(f"diagnostic sweep on {n} episodes "
          f"(d0 in [{float(d0.min()):.2f}, {float(d0.max()):.2f}] m)")
    print(f"{'configuration':44s} {'SR':>8s} {'exits':>8s}")
    print("-" * 64)

    def report(name: str, res: object, mask: object = None) -> None:
        if mask is None:
            sr = float(res.success.float().mean())
            ex = float(res.exited_workspace.float().mean())
        else:
            sr = float(res.success[mask].float().mean())
            ex = float(res.exited_workspace[mask].float().mean())
        print(f"{name:44s} {sr * 100:7.2f}% {ex * 100:7.2f}%")

    orig_pd = envmod.pd_controller

    def with_kv(kv: float, res_fn) -> None:
        envmod.pd_controller = functools.partial(orig_pd, kv=kv)
        try:
            res_fn()
        finally:
            envmod.pd_controller = orig_pd

    # ---- baseline, gains unchanged ------------------------------------
    res = env.run_noiseless(p0, v0, goal)
    report("Kv=1.0 (frozen config, full workspace)", res)

    # ---- velocity-damping gain sweep ----------------------------------
    for kv in (1.5, 2.0, 2.5, 3.0):
        def run(kv=kv):
            with_kv(kv, lambda: report(
                f"Kv={kv:.1f} (full workspace)",
                env.run_noiseless(p0, v0, goal)))
        run()

    # ---- goal-margin sampling (config-level choice), Kv = 1 -----------
    for margin in (1.0, 1.5, 2.0):
        lo = config.WORKSPACE_MIN + margin
        hi = config.WORKSPACE_MAX - margin
        gm = goal.clamp(lo, hi)
        res = env.run_noiseless(p0, v0, gm)
        report(f"Kv=1.0, goal margin {margin:.1f} m", res)

    # ---- combinations --------------------------------------------------
    for kv, margin in ((1.5, 1.5), (2.0, 0.0), (2.0, 1.5), (3.0, 1.5)):
        lo = config.WORKSPACE_MIN + margin
        hi = config.WORKSPACE_MAX - margin
        gm = goal.clamp(lo, hi)
        with_kv(kv, lambda kv=kv, gm=gm, margin=margin: report(
            f"Kv={kv:.1f}, goal margin {margin:.1f} m",
            env.run_noiseless(p0, v0, gm)))

    # ---- d_max caps (A6) with each Kv ----------------------------------
    for dmax in (8.0, 12.0):
        m = d0 <= dmax
        res = env.run_noiseless(p0[m], v0[m], goal[m])
        report(f"Kv=1.0, d_max = {dmax:.0f} m (subset {int(m.sum())})", res)
        for kv in (2.0,):
            with_kv(kv, lambda kv=kv, m=m, dmax=dmax: report(
                f"Kv={kv:.1f}, d_max = {dmax:.0f} m (subset {int(m.sum())})",
                env.run_noiseless(p0[m], v0[m], goal[m])))

    print("-" * 64)
    print("Diagnostic only: nothing frozen, nothing modified.")
    print("Overshoot analysis: first-order overshoot ~exp(-pi*zeta/sqrt(1-zeta^2)):")
    import math
    for kv in (1.0, 1.5, 2.0, 3.0):
        kp = config.KP
        zeta = kv / (2.0 * math.sqrt(kp))
        ov = (math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))
              if zeta < 1 else 0.0)
        print(f"  Kv = {kv:.1f} -> zeta = {zeta:.2f}, "
              f"overshoot ~ {ov * 100:5.1f}% of d0")


if __name__ == "__main__":
    main()
