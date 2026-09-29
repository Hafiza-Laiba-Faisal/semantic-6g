"""Step 5 task-oriented closed-loop tests (items 14, 15).

Covers:
  A  task-loss algebra vs hand-computed values (incl. t=0 exclusion)
  B  rollout dimensions for (B, T)
  C  zero-error trajectory -> all loss components exactly zero
  D  gradient flow through the full closed loop (AWGN + Rayleigh)
  E  controller/dynamics consistency with the frozen equations
  F  in-process regression of the full Step-3 (43) and Step 0-2 (11) suites
  15 finite-difference vs autograd on one encoder parameter

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_task_oriented
"""

from __future__ import annotations

import math
import sys

import torch

from project import config
from project.data.normalization import denormalize, normalize
from project.losses.task_loss import task_loss
from project.models.reconstruction_jscc import ReconstructionDeepJSCC
from project.training.train_task_oriented import (
    closed_loop_rollout,
    evaluate_task_loss,
    train_task_oriented,
)
from project.uav.controller import pd_controller
from project.uav.dynamics import step as dynamics_step

SUMMARY = []


def check(name, ok, detail=""):
    SUMMARY.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]" +
          (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
# A. task-loss algebra
# ---------------------------------------------------------------------------

def test_a_loss_algebra():
    p = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])     # t=0..2
    u = torch.tensor([[[0.5, -0.25], [-1.0, 2.0]]])              # t=0..1
    v = torch.tensor([[[1.0, 1.0], [0.0, -2.0], [3.0, 1.0]]])    # t=0..2
    g = torch.tensor([[2.0, 0.0]])
    out = task_loss(p, u, v, g)
    # L_d = (1/2)[||p1-g||^2 + ||p2-g||^2] = (1/2)[1 + 0] = 0.5
    #       (t=0 term ||p0-g||^2 = 4 must be EXCLUDED)
    check("A: L_d (t0 excluded)", math.isclose(float(out["L_d"]), 0.5, abs_tol=1e-7),
          f"{float(out['L_d']):.7f} vs 0.5")
    # L_u = (1/2)[(0.25+0.0625) + (1+4)] = 2.65625
    check("A: L_u", math.isclose(float(out["L_u"]), 2.65625, abs_tol=1e-7),
          f"{float(out['L_u']):.7f} vs 2.65625")
    # L_v = (1/2)[||v1||^2 + ||v2||^2] = (1/2)[4 + 10] = 7
    check("A: L_v (t0 excluded)", math.isclose(float(out["L_v"]), 7.0, abs_tol=1e-7),
          f"{float(out['L_v']):.7f} vs 7.0")
    # L_T = ||p2-g||^2 = 0
    check("A: L_T", math.isclose(float(out["L_T"]), 0.0, abs_tol=1e-9),
          f"{float(out['L_T']):.7f} vs 0")
    # frozen weights (1, 0, 0, 2): total = 0.5
    check("A: total (frozen lambdas)", math.isclose(float(out["total"]), 0.5, abs_tol=1e-7),
          f"{float(out['total']):.7f} vs 0.5")
    # explicit non-default weights verify the weighted sum
    out2 = task_loss(p, u, v, g, lambda_distance=2.0, lambda_control=1.0,
                     lambda_velocity=0.5, lambda_terminal=3.0)
    check("A: weighted sum (2,1,0.5,3)",
          math.isclose(float(out2["total"]), 2 * 0.5 + 2.65625 + 0.5 * 7.0 + 3 * 0.0,
                       abs_tol=1e-7),
          f"{float(out2['total']):.7f} vs 7.15625")


# ---------------------------------------------------------------------------
# B. rollout dimensions
# ---------------------------------------------------------------------------

def test_b_rollout_dims():
    torch.manual_seed(0)
    model = ReconstructionDeepJSCC(k=2)
    B, T = 4, 10
    s0 = torch.cat([torch.rand(B, 2) * 16 - 8, torch.zeros(B, 2),
                    torch.rand(B, 2) * 16 - 8], dim=-1)
    pos, ctl, vel = closed_loop_rollout(model, s0, t_max=T, snr_db=10.0,
                                        channel="awgn",
                                        generator=torch.Generator().manual_seed(1))
    ok = (tuple(pos.shape) == (B, T + 1, 2) and tuple(ctl.shape) == (B, T, 2)
          and tuple(vel.shape) == (B, T + 1, 2))
    check("B: rollout shapes", ok, f"pos {tuple(pos.shape)}, ctl {tuple(ctl.shape)}, "
          f"vel {tuple(vel.shape)}")
    out = evaluate_task_loss(model, s0, t_max=T, snr_db=10.0, channel="awgn",
                             eval_seed=3)
    check("B: scalar loss + diagnostics",
          out["total"].dim() == 0 and out["success_rate"].dim() == 0)


# ---------------------------------------------------------------------------
# C. zero-error case
# ---------------------------------------------------------------------------

def test_c_zero_error():
    T = 5
    g = torch.tensor([[1.5, -2.0]])
    B = 1
    p = g.repeat(B, T + 1, 1)
    v = torch.zeros(B, T + 1, 2)
    u = torch.zeros(B, T, 2)
    out = task_loss(p, u, v, g.repeat(B, 1))
    ok = all(math.isclose(float(out[key]), 0.0, abs_tol=1e-12)
             for key in ("L_d", "L_u", "L_v", "L_T", "total"))
    check("C: stay-at-goal trajectory -> all components 0", ok,
          f"L_d={float(out['L_d']):.2e} L_u={float(out['L_u']):.2e} "
          f"L_v={float(out['L_v']):.2e} L_T={float(out['L_T']):.2e} "
          f"total={float(out['total']):.2e}")


# ---------------------------------------------------------------------------
# D. gradient flow through the full closed loop
# ---------------------------------------------------------------------------

def test_d_gradient_flow():
    torch.manual_seed(0)
    for channel in ("awgn", "rayleigh"):
        model = ReconstructionDeepJSCC(k=3)
        B, T = 8, 8
        s0 = torch.cat([torch.rand(B, 2) * 16 - 8, torch.zeros(B, 2),
                        torch.rand(B, 2) * 16 - 8], dim=-1)
        pos, ctl, vel = closed_loop_rollout(model, s0, t_max=T, snr_db=10.0,
                                            channel=channel,
                                            generator=torch.Generator().manual_seed(2))
        out = task_loss(pos, ctl, vel, s0[:, 4:6])
        out["total"].backward()
        enc = [float(pr.grad.norm()) for pr in model.encoder.parameters()
               if pr.grad is not None]
        dec = [float(pr.grad.norm()) for pr in model.decoder.parameters()
               if pr.grad is not None]
        n_enc = len(list(model.encoder.parameters()))
        n_dec = len(list(model.decoder.parameters()))
        finite = all(math.isfinite(x) for x in enc + dec)
        ok = (len(enc) == n_enc and len(dec) == n_dec and finite
              and sum(enc) > 0 and sum(dec) > 0)
        check(f"D: closed-loop grads {channel}", ok,
              f"enc {len(enc)}/{n_enc} params |g|={sum(enc):.3e}, "
              f"dec {len(dec)}/{n_dec} params |g|={sum(dec):.3e}")


# ---------------------------------------------------------------------------
# E. controller/dynamics consistency (frozen equations, manual state)
# ---------------------------------------------------------------------------

def test_e_controller_dynamics():
    s = torch.tensor([[1.0, 2.0, 0.5, -1.0, 3.0, 0.0]])
    u = pd_controller(s)
    # u_raw = -1*(p-g) - 2*v = -(-2,2) - (1,-2) = (1, 0); inside clip region
    check("E: controller manual state", torch.allclose(u, torch.tensor([[1.0, 0.0]])),
      f"u = {u.tolist()}")
    # saturation region: p far from goal
    s_far = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    u_far = pd_controller(s_far)
    check("E: clip intact (hard clamp, no surrogate)",
          torch.allclose(u_far, torch.tensor([[-config.U_MAX, 0.0]])),
          f"u = {u_far.tolist()}")
    # dynamics: p' = p + dt v + 0.5 dt^2 u ; v' = v + dt u
    p = torch.tensor([[1.0, 2.0]])
    v = torch.tensor([[0.5, -1.0]])
    u1 = torch.tensor([[1.0, 0.0]])
    p2, v2 = dynamics_step(p, v, u1)
    p_ref = p + config.DT * v + 0.5 * config.DT ** 2 * u1
    v_ref = v + config.DT * u1
    check("E: dynamics frozen update", torch.allclose(p2, p_ref, atol=1e-7)
          and torch.allclose(v2, v_ref, atol=1e-7),
          f"p' = {p2.tolist()}, v' = {v2.tolist()}")


# ---------------------------------------------------------------------------
# 15. finite-difference vs autograd (deterministic channel realization)
# ---------------------------------------------------------------------------

def test_finite_difference():
    # float64 diagnostic: the task loss is O(10) and the finite difference
    # is O(gradient * eps); in float32 the L-noise (~1e-5) swamps the
    # difference, so this DIAGNOSTIC runs in double precision. The actual
    # training pipeline stays float32 and fully stochastic.
    torch.manual_seed(0)
    for channel in ("awgn", "rayleigh"):
        model = ReconstructionDeepJSCC(k=2).double()
        B, T = 16, 8
        gen = torch.Generator().manual_seed(11)
        s0 = torch.cat([torch.rand(B, 2, generator=gen, dtype=torch.float64) * 16 - 8,
                        torch.zeros(B, 2, dtype=torch.float64),
                        torch.rand(B, 2, generator=gen, dtype=torch.float64) * 16 - 8],
                       dim=-1)
        eval_seed = 77
        out = evaluate_task_loss(model, s0, t_max=T, snr_db=10.0,
                                 channel=channel, eval_seed=eval_seed)
        out["total"].backward()
        param = next(model.encoder.parameters())
        idx = (0, 0)
        g_auto = float(param.grad[idx])
        param.grad = None

        eps = 1e-5
        with torch.no_grad():
            orig = float(param[idx])
            param[idx] = orig + eps
            L_plus = float(evaluate_task_loss(
                model, s0, t_max=T, snr_db=10.0, channel=channel,
                eval_seed=eval_seed)["total"])
            param[idx] = orig - eps
            L_minus = float(evaluate_task_loss(
                model, s0, t_max=T, snr_db=10.0, channel=channel,
                eval_seed=eval_seed)["total"])
            param[idx] = orig
        g_fd = (L_plus - L_minus) / (2 * eps)
        same_sign = (g_auto > 0) == (g_fd > 0)
        denom = max(abs(g_auto), abs(g_fd), 1e-8)
        rel = abs(g_auto - g_fd) / denom
        ok = same_sign and rel < 0.25
        check(f"15: finite-difference vs autograd ({channel}, fp64)", ok,
              f"g_auto={g_auto:.4e}, g_fd={g_fd:.4e}, rel_dev={rel:.3f}")


# ---------------------------------------------------------------------------
# F. regression of the full Step-3 and Step 0-2 suites (in-process)
# ---------------------------------------------------------------------------

def test_f_regression():
    import tests.test_channels as tc
    tc.SUMMARY.clear()
    tc.test_conversion(); tc.test_power_normalization(); tc.test_awgn_variance()
    tc.test_awgn_snr(); tc.test_rayleigh_stats(); tc.test_block_fading()
    tc.test_equalization(); tc.test_repro_and_conventions()
    check("F: Step-3 channel suite", all(tc.SUMMARY),
          f"{sum(tc.SUMMARY)}/{len(tc.SUMMARY)} checks")

    import tests.test_step02 as ts
    ts_test_fns = [ts.test_config, ts.test_normalization_roundtrip,
                   ts.test_normalization_endpoints_and_clip,
                   ts.test_dynamics_closed_form, ts.test_dynamics_autograd,
                   ts.test_velocity_clip, ts.test_controller_formula,
                   ts.test_controller_saturation_and_gradient,
                   ts.test_env_success_semantics,
                   ts.test_env_exit_freeze_and_determinism,
                   ts.test_env_sampling_and_nans]
    ok = True
    for fn in ts_test_fns:
        try:
            fn()
        except AssertionError:
            ok = False
    check("F: Step 0-2 suite (11 tests)", ok)


# ---------------------------------------------------------------------------
# smoke train + reproducibility
# ---------------------------------------------------------------------------

def test_smoke_train():
    for channel in ("awgn", "rayleigh"):
        _, rep = train_task_oriented(channel=channel, k=3, seed=42,
                                     steps=200, batch_episodes=64, t_max=50,
                                     eval_episodes=128, verbose_every=0)
        ev, ea = rep["eval_before"], rep["eval_after"]
        red = 100.0 * (1.0 - rep["train_total_final"] / rep["train_total_initial"])
        print(f"\n--- {channel.upper()} task smoke (item 13 report) ---")
        print(f"  lambdas                     = {rep['lambdas']}")
        print(f"  seed / k / SNR range        = {rep['seed']} / {rep['k']} / "
              f"U{tuple(rep['snr_range_db'])}")
        print(f"  steps x batch x T           = {rep['steps']} x "
              f"{rep['batch_episodes']} x {rep['t_max']}")
        print(f"  train total initial/final   = {rep['train_total_initial']:.4f} -> "
              f"{rep['train_total_final']:.4f}  ({red:.1f}% reduction)")
        print(f"  L_d  initial/final          = {rep['train_Ld_initial']:.4f} -> "
              f"{rep['train_Ld_final']:.4f}")
        print(f"  L_u  initial/final (diag)   = {rep['train_Lu_initial']:.4f} -> "
              f"{rep['train_Lu_final']:.4f}")
        print(f"  L_v  initial/final (diag)   = {rep['train_Lv_initial']:.4f} -> "
              f"{rep['train_Lv_final']:.4f}")
        print(f"  L_T  initial/final          = {rep['train_LT_initial']:.4f} -> "
              f"{rep['train_LT_final']:.4f}")
        print(f"  recon MSE diag (not opt.)   = {rep['recon_mse_diag_initial']:.4f} -> "
              f"{rep['recon_mse_diag_final']:.4f}")
        print(f"  encoder grad norm first/fin = {rep['grad_enc_first']:.3e} / "
              f"{rep['grad_enc_final']:.3e}")
        print(f"  decoder grad norm first/fin = {rep['grad_dec_first']:.3e} / "
              f"{rep['grad_dec_final']:.3e}")
        print(f"  NaN/Inf batches             = {rep['nan_batches']}")
        print(f"  fixed-eval success rate     = {ev['success_rate']:.3f} -> "
              f"{ea['success_rate']:.3f}")
        check(f"13/{channel}: finite", all(math.isfinite(v) for v in
              (rep["train_total_initial"], rep["train_total_final"],
               rep["grad_enc_final"], rep["grad_dec_final"])))
        check(f"13/{channel}: no NaN batches", rep["nan_batches"] == 0)
        check(f"13/{channel}: meaningful task-loss reduction", red >= 25.0,
              f"{red:.1f}%")
        check(f"13/{channel}: grads alive", rep["grad_enc_final"] > 0
              and rep["grad_dec_final"] > 0)
        # reproducibility: identical short run twice
        _, ra = train_task_oriented(channel=channel, k=3, seed=42, steps=40,
                                    batch_episodes=32, t_max=20,
                                    eval_episodes=64)
        _, rb = train_task_oriented(channel=channel, k=3, seed=42, steps=40,
                                    batch_episodes=32, t_max=20,
                                    eval_episodes=64)
        check(f"13/{channel}: reproducible (bitwise histories)",
              ra["history"] == rb["history"])


if __name__ == "__main__":
    print("=" * 72)
    print("STEP 5 TASK-ORIENTED CLOSED-LOOP TESTS")
    print(f"lambdas (frozen): ld={config.LAMBDA_DISTANCE}, "
          f"lu={config.LAMBDA_CONTROL}, lv={config.LAMBDA_VELOCITY}, "
          f"lT={config.LAMBDA_TERMINAL}")
    print("=" * 72)
    test_a_loss_algebra()
    test_b_rollout_dims()
    test_c_zero_error()
    test_d_gradient_flow()
    test_e_controller_dynamics()
    test_finite_difference()
    test_smoke_train()
    test_f_regression()
    n_pass, n_all = sum(SUMMARY), len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)
