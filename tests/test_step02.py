"""Step 0-2 unit tests (config, normalization, dynamics, controller, env).

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_step02

Every check prints one PASS line; the script exits non-zero on any failure.
"""

from __future__ import annotations

import math
import sys

import torch

from project import config
from project.data.normalization import denormalize, normalize
from project.uav.controller import pd_controller
from project.uav.dynamics import clip_velocity, step as dyn_step
from project.uav.environment import NavigationEnv, episode_distances

torch.manual_seed(0)


def test_config() -> None:
    config.validate()
    assert config.DT == 0.1 and config.T_MAX == 100
    assert config.GOAL_RADIUS == 0.5 and config.D_MIN == 4.0
    assert config.KP == 1.0 and config.KV == 2.0
    assert config.U_MAX == 2.0 and config.V_MAX == 5.0
    assert config.TRAINING_SEEDS == (42, 43, 44)
    assert config.BANDWIDTH_K_VALUES == (1.0, 2.0, 3.0)
    # matched-budget digital points reproduce audit 8.6 exactly
    for k, b in config.DIGITAL_B_BY_K.items():
        pkt = config.digital_packet(k, b)
        assert pkt["n_src_bits"] == 6 * b
        assert pkt["n_info_bits"] == 6 * b + 6
        assert pkt["n_coded_bits"] == 12 * b + 12
        assert pkt["n_payload_symbols"] == 6 * b + 6
        assert pkt["n_padding_symbols"] == 0
        assert pkt["n_transmitted_symbols"] == k
    # infeasible combination must raise, not adapt
    try:
        config.digital_packet(3, 8)
        raise AssertionError("expected InfeasibleDigitalConfig")
    except config.InfeasibleDigitalConfig:
        pass
    print("PASS  config: values, packet arithmetic, infeasibility handling")


def test_normalization_roundtrip() -> None:
    a = torch.tensor(config.STATE_MIN)
    b = torch.tensor(config.STATE_MAX)
    s = a + torch.rand(512, 6) * (b - a)          # in-range states
    s_bar = normalize(s)
    assert bool((s_bar >= -1.0 - 1e-6).all()) and bool((s_bar <= 1.0 + 1e-6).all())
    s_rec = denormalize(s_bar)
    assert torch.allclose(s, s_rec, atol=1e-6), "round-trip must be exact in-range"
    print("PASS  normalization: round-trip exact within 1e-6 on 512 in-range states")


def test_normalization_endpoints_and_clip() -> None:
    a = torch.tensor(config.STATE_MIN)
    b = torch.tensor(config.STATE_MAX)
    s_lo = a.repeat(2, 1)
    s_hi = b.repeat(2, 1)
    s = torch.cat([s_lo, s_hi])
    s_bar = normalize(s)
    assert torch.allclose(s_bar[:2], -torch.ones(2, 6), atol=1e-7)
    assert torch.allclose(s_bar[2:], torch.ones(2, 6), atol=1e-7)
    # out-of-range inputs clip to +-1
    s_out = torch.cat([a.unsqueeze(0) - 3.0, b.unsqueeze(0) + 3.0])
    s_bar_out = normalize(s_out)
    assert torch.allclose(s_bar_out[0], -torch.ones(6), atol=1e-7)
    assert torch.allclose(s_bar_out[1], torch.ones(6), atol=1e-7)
    # inverse clips out-of-interval values back onto the range:
    #   s_bar = +1.5 -> range max,  s_bar = -2.0 -> range min,  s_bar = 0 -> midpoint
    s_back = denormalize(torch.tensor([[+1.5, -2.0, 0.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(s_back[0, 0:1], b[0:1], atol=1e-6), "+1.5 must map to the range max"
    assert torch.allclose(s_back[0, 1:2], a[1:2], atol=1e-6), "-2.0 must map to the range min"
    assert torch.allclose(s_back[0, 2:4], torch.zeros(2), atol=1e-6), "0 must map to the mid-range value"
    print("PASS  normalization: endpoints map to +-1; clipping verified both ways")


def test_dynamics_closed_form() -> None:
    p = torch.randn(256, 2)
    v = torch.randn(256, 2)
    u = torch.randn(256, 2)
    p2, v2 = dyn_step(p, v, u, v_max=None)        # pure equations, no clip
    p_ref = p + config.DT * v + 0.5 * config.DT ** 2 * u
    v_ref = v + config.DT * u
    assert torch.allclose(p2, p_ref, atol=1e-6)
    assert torch.allclose(v2, v_ref, atol=1e-6)
    print("PASS  dynamics: matches closed-form double integrator within 1e-6")


def test_dynamics_autograd() -> None:
    p = torch.randn(4, 2, requires_grad=True)
    v = torch.randn(4, 2, requires_grad=True)
    u = torch.randn(4, 2)                          # exogenous input, constant
    p2, v2 = dyn_step(p, v, u, v_max=None)
    p2.sum().backward(retain_graph=True)
    assert p.grad is not None and bool((p.grad == 1.0).all())
    v.grad = None          # autograd accumulates across backward calls; reset
    v2.sum().backward()
    assert v.grad is not None and bool((v.grad == 1.0).all())
    # gradient w.r.t. control must flow (needed for task-oriented training)
    p = torch.randn(4, 2, requires_grad=True)
    v = torch.randn(4, 2, requires_grad=True)
    u = torch.randn(4, 2, requires_grad=True)
    p2, v2 = dyn_step(p, v, u, v_max=None)
    (p2.sum() + v2.sum()).backward()
    assert u.grad is not None and torch.allclose(
        u.grad, torch.full((4, 2), config.DT * (1.0 + 0.5 * config.DT)), atol=1e-6)
    print("PASS  dynamics: autograd reaches p, v and u with exact analytic grads")


def test_velocity_clip() -> None:
    v = torch.tensor([[3.0, 4.0], [0.3, 0.4]])     # norms 5 and 0.5
    out = clip_velocity(v, v_max=5.0)
    assert math.isclose(float(out[0].norm()), 5.0, rel_tol=1e-6)
    assert torch.allclose(out[1], v[1])            # inside ball untouched
    assert clip_velocity(v, v_max=None) is v       # mechanism disable
    print("PASS  dynamics: v_max rescaling correct; untouched inside the ball")


def test_controller_formula() -> None:
    s = torch.tensor([[2.0, 1.0, 0.5, -0.25, 0.0, 0.0]])
    u = pd_controller(s)                # Kp = 1, Kv = 2 (frozen), u_max = 2
    u_raw = torch.tensor(
        [[-(2.0 - 0.0) - 2.0 * 0.5, -(1.0 - 0.0) - 2.0 * (-0.25)]])
    u_ref = torch.clamp(u_raw, -2.0, 2.0)
    assert torch.allclose(u, u_ref, atol=1e-7)
    print("PASS  controller: matches u = clip(-Kp(p-pg) - Kv v, +-u_max)")


def test_controller_saturation_and_gradient() -> None:
    s = torch.tensor([[50.0, -50.0, 10.0, -10.0, 0.0, 0.0]])
    u = pd_controller(s)
    assert bool((u.abs() <= config.U_MAX + 1e-7).all())
    # x: -(50) - (10) = -60 -> -u_max ; y: -(-50) - (-10) = +60 -> +u_max
    assert torch.allclose(
        u, torch.tensor([[-config.U_MAX, config.U_MAX]]), atol=1e-7)
    # gradient reaches the estimate on the unsaturated components
    s = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    u = pd_controller(s)
    u.sum().backward()
    assert s.grad is not None and float(s.grad.abs().sum()) > 0
    print("PASS  controller: saturation at +-u_max exact; gradient flows to estimate")


def test_env_success_semantics() -> None:
    env = NavigationEnv()
    p0 = torch.tensor([[0.0, 0.0]])
    v0 = torch.zeros(1, 2)
    goal = torch.tensor([[5.0, 0.0]])               # d0 = 5 m >= d_min
    res = env.run_noiseless(p0, v0, goal, record=True)
    assert bool(res.success[0]), "UAV must reach the 0.5 m goal radius"
    assert bool(torch.isfinite(res.final_distance))
    # success == (min realized distance <= r_g); time-to-goal is the first hit
    assert bool(res.min_distance[0] <= env.goal_radius)
    assert int(res.time_to_goal[0]) == int(
        (torch.stack(res.distances) <= env.goal_radius).float().argmax(dim=0))
    assert res.final_distance.shape == (1,)
    print(
        "PASS  environment: success/first-entry semantics consistent "
        f"(t_goal={int(res.time_to_goal[0])}, d_final={float(res.final_distance[0]):.3f} m)"
    )


def test_env_exit_freeze_and_determinism() -> None:
    env = NavigationEnv()
    # goal outside the workspace forces a deterministic boundary exit
    p0 = torch.tensor([[0.0, 0.0]])
    v0 = torch.zeros(1, 2)
    goal = torch.tensor([[50.0, 0.0]])
    res = env.run_noiseless(p0, v0, goal, record=True)
    assert bool(res.exited_workspace[0]) and not bool(res.success[0])
    pos = torch.stack(res.positions)
    t_exit = int(res.time_to_goal[0])
    # frozen after exit: all later positions identical to the exit position
    assert bool((pos[t_exit:] == pos[t_exit]).all())
    # determinism: bitwise identical rerun
    res2 = env.run_noiseless(p0, v0, goal, record=True)
    pos2 = torch.stack(res2.positions)
    assert bool((pos == pos2).all())
    print("PASS  environment: exit freezes the episode deterministically (A5)")


def test_env_sampling_and_nans() -> None:
    env = NavigationEnv()
    for seed in config.TRAINING_SEEDS:
        g = config.derive_generator("episodes", seed)
        p0, v0, goal = env.sample_starts(2000, generator=g)
        d = episode_distances(p0, goal)
        assert bool((d >= config.D_MIN - 1e-9).all()), "d_min violated"
        assert bool((v0 == 0).all()), "v_0 must be (0, 0)"
        res = env.run_noiseless(p0, v0, goal)
        all_d = torch.stack(res.distances)
        assert bool(torch.isfinite(all_d).all()), "NaN/Inf in trajectory"
        assert bool(torch.isfinite(res.final_distance).all())
        # internal consistency: success iff realized min distance <= r_g
        consist = (res.min_distance <= env.goal_radius) == res.success
        assert bool(consist.all())
    print("PASS  environment: seeds 42/43/44, d_min, finiteness, success consistency")


if __name__ == "__main__":
    test_config()
    test_normalization_roundtrip()
    test_normalization_endpoints_and_clip()
    test_dynamics_closed_form()
    test_dynamics_autograd()
    test_velocity_clip()
    test_controller_formula()
    test_controller_saturation_and_gradient()
    test_env_success_semantics()
    test_env_exit_freeze_and_determinism()
    test_env_sampling_and_nans()
    print("\nALL STEP 0-2 TESTS PASSED")
    sys.exit(0)
