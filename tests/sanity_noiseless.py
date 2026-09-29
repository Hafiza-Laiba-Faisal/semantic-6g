"""Noiseless oracle sanity gate (audit section 7, step 2).

The controller receives PERFECT ground-truth state - no channel, no
communication, no learning. This validates the plant + controller before
any neural network exists.

Frozen controller: Kp = 1.0, Kv = 2.0 (user decision). The damping term is
explicitly ``-Kv * v``; the acceleration clip only bounds the commanded
acceleration and is NOT a source of damping.

Run from the repository root:
    .venv/Scripts/python.exe -m tests.sanity_noiseless
"""

from __future__ import annotations

import math

import torch

from project import config
from project.uav.controller import pd_controller
from project.uav.dynamics import step as dyn_step
from project.uav.environment import NavigationEnv, episode_distances


def main() -> None:
    torch.manual_seed(0)
    env = NavigationEnv()

    print("=" * 70)
    print("ORACLE SANITY GATE - noiseless, perfect ground-truth state")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 10. exact sampling rules (frozen, unchanged)
    # ------------------------------------------------------------------
    n = config.N_TEST_EPISODES          # 5000 oracle episodes
    gen = config.derive_generator("episodes", config.TEST_SEED)
    p0, v0, goal = env.sample_starts(n, generator=gen)
    d0 = episode_distances(p0, goal)

    print("\n[10] initial-state / goal sampling rules (unchanged frozen):")
    print(f"     p0 ~ U([{config.WORKSPACE_MIN},{config.WORKSPACE_MAX}]^2), "
          f"goal ~ U([{config.WORKSPACE_MIN},{config.WORKSPACE_MAX}]^2)")
    print(f"     rejection resample until ||p0 - goal|| >= d_min = {config.D_MIN} m")
    print(f"     v0 = {config.V_0} (frozen)")
    print(f"     RNG: derive_generator('episodes', TEST_SEED="
          f"{config.TEST_SEED});  episodes = {n}")
    print(f"     realized d0 in [{float(d0.min()):.2f}, {float(d0.max()):.2f}] m, "
          f"mean {float(d0.mean()):.2f} m")

    # ------------------------------------------------------------------
    # closed-loop rollout with full records for exact statistics
    # ------------------------------------------------------------------
    res = env.run_noiseless(p0, v0, goal, record=True)

    pos = torch.stack(res.positions)     # [T+1, B, 2]
    vel = torch.stack(res.velocities)    # [T+1, B, 2]
    dst = torch.stack(res.distances)     # [T+1, B]

    # max acceleration command BEFORE clipping: same PD law, u_max = inf
    s_all = torch.cat([pos[:-1], vel[:-1],
                       goal.unsqueeze(0).expand(pos[:-1].shape)], dim=-1)
    u_raw = pd_controller(s_all, u_max=float("inf"))
    sat = u_raw.abs() > config.U_MAX + 1e-9

    # 4. overshoot statistic on successful episodes
    success_mask = res.success
    d0_s = d0[success_mask]
    dmin_s = res.min_distance[success_mask]
    overshoot_frac = ((d0_s - dmin_s) / d0_s)     # 1 - dmin/d0 in [0, 1+]
    deep = float((res.min_distance[success_mask]
                  < 0.2 * config.GOAL_RADIUS).float().mean())

    nan_eps = int((~torch.isfinite(dst)).any(dim=0).sum())

    print("\n" + "=" * 70)
    print("ORACLE GATE RESULTS")
    print("=" * 70)
    print(f"[1]  Kp = {config.KP}, Kv = {config.KV}  "
          f"(damping term: u_raw = -Kp*(p-pg) - Kv*v; clip is NOT damping)")
    print(f"[2]  success rate                 = "
          f"{float(success_mask.float().mean()) * 100:6.2f}%  "
          f"({int(success_mask.sum())}/{n})")
    print(f"[3]  boundary-exit rate           = "
          f"{float(res.exited_workspace.float().mean()) * 100:6.2f}%  "
          f"({int(res.exited_workspace.sum())}/{n})")
    print(f"[4]  overshoot (successful)       = mean 1 - dmin/d0 = "
          f"{float(overshoot_frac.mean()):.4f}, "
          f"median {float(overshoot_frac.median()):.4f}; "
          f"episodes deeper than 0.2*r_g: {deep * 100:.2f}%")
    print(f"[5]  mean final distance          = "
          f"{float(res.final_distance.mean()):.4f} m")
    print(f"[6]  mean minimum distance        = "
          f"{float(res.min_distance.mean()):.4f} m")
    print(f"[7]  max velocity observed        = "
          f"{float(vel.norm(dim=-1).max()):.4f} m/s  (v_max = {config.V_MAX})")
    print(f"[8]  max |u| before clipping      = "
          f"{float(u_raw.abs().max()):.4f} m/s^2  (u_max = {config.U_MAX}); "
          f"saturated step fraction = {float(sat.float().mean()):.4f}")
    print(f"[9]  NaN/Inf episodes             = {nan_eps}")
    print(f"     time-to-goal (successful)    = mean "
          f"{float(res.time_to_goal[success_mask].float().mean()):.1f}, "
          f"median "
          f"{float(res.time_to_goal[success_mask].float().median()):.1f} steps")

    sr = float(success_mask.float().mean())
    ex = float(res.exited_workspace.float().mean())
    gate = sr >= 0.99 and ex <= 0.01 and nan_eps == 0

    # ------------------------------------------------------------------
    # per-bin breakdown (kept for traceability)
    # ------------------------------------------------------------------
    print("\n     breakdown by initial distance:")
    edges = [4, 6, 8, 10, 12, 14, 16, 24]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d0 >= lo) & (d0 < hi)
        if int(m.sum()):
            srb = float(success_mask[m].float().mean())
            exb = float(res.exited_workspace[m].float().mean())
            print(f"       d0 in [{lo:2d},{hi:2d}) m : {int(m.sum()):5d} eps, "
                  f"SR = {srb * 100:6.2f}%, exits = {exb * 100:5.2f}%")

    print("\n" + "=" * 70)
    if gate:
        print("ORACLE GATE: PASS")
        print(f"Kp = {config.KP}, Kv = {config.KV} under the unchanged frozen")
        print("episode distribution: SR >= 99%, exits <= 1%, no NaN/Inf.")
        print("The controller is now FROZEN for every communication method;")
        print("this SR is the noiseless ceiling all methods are measured against.")
    else:
        print("ORACLE GATE: FAIL - do not proceed; report, do not adapt.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
