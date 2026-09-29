"""Task-oriented DeepJSCC closed-loop training (Step 5).

Frozen pipeline (audit 1.1 rows 6-12, spec section 15):

    s_t -> normalize -> encoder -> power norm -> channel -> [ZF] -> decoder
        -> s_hat_bar -> denormalize -> PD controller -> u_t
        -> double integrator -> s_{t+1} -> ... (T steps) -> L_task

* The optimization objective is the FROZEN task loss
  (project/losses/task_loss.py) with the signed-off weights
  ld = 1.0, lu = 0.0, lv = 0.0, lT = 2.0 - never the reconstruction MSE
  (reconstruction MSE is computed only as a detached diagnostic).
* The controller (Kp = 1.0, Kv = 2.0, hard clip) and the dynamics are the
  frozen Step 0-2 implementations, reused verbatim inside the graph.
* The state distribution is the SAME frozen A9 protocol used by the
  reconstruction baseline (env.sample_starts + v0 = 0).
* Training SNR: U(0, 20) dB per rollout (vectorized-channel contract;
  the per-sample refinement is documented in the Step-4 trainer).
* No .detach()/.numpy()/.item() anywhere in the differentiable path;
  noise/fading enter as generator-drawn constants. Training uses a FIXED
  horizon (no early termination, audit A5).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from project import config
from project.data.normalization import denormalize, normalize
from project.losses.task_loss import task_loss
from project.models.reconstruction_jscc import ReconstructionDeepJSCC
from project.uav.controller import pd_controller
from project.uav.dynamics import step as dynamics_step
from project.uav.environment import NavigationEnv


def closed_loop_rollout(
    comm: ReconstructionDeepJSCC,
    s0: torch.Tensor,
    t_max: int = config.T_MAX,
    snr_db: float = 10.0,
    channel: str = "awgn",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable T-step closed-loop rollout.

    Returns ``(positions [B, T+1, 2], controls [B, T, 2],
    velocities [B, T+1, 2])``. Every operation is differentiable w.r.t.
    the communication parameters; noise/fading are constants.
    """
    if channel not in ("awgn", "rayleigh"):
        raise ValueError(f"unknown channel {channel!r}")
    p = s0[:, 0:2]
    v = s0[:, 2:4]
    g = s0[:, 4:6]

    positions = [p]
    velocities = [v]
    controls = []
    s = s0

    for _ in range(t_max):
        # communication: normalized estimate of the current state
        s_bar_hat = comm.forward_losses(s, snr_db, channel=channel,
                                        generator=generator)
        s_hat = denormalize(s_bar_hat)              # physical estimate
        u = pd_controller(s_hat)                    # frozen controller
        p, v = dynamics_step(p, v, u, dt=config.DT, v_max=config.V_MAX)
        positions.append(p)
        velocities.append(v)
        controls.append(u)
        s = torch.cat([p, v, g], dim=-1)

    return (torch.stack(positions, dim=1),
            torch.stack(controls, dim=1),
            torch.stack(velocities, dim=1))


def evaluate_task_loss(
    comm: ReconstructionDeepJSCC,
    s_eval: torch.Tensor,
    t_max: int = config.T_MAX,
    snr_db: float = 10.0,
    channel: str = "awgn",
    eval_seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Deterministic task-loss evaluation on a fixed episode set.

    The channel generator is re-derived from ``eval_seed`` on every call,
    so the same episodes always see the same noise/fading realizations -
    before/after comparisons and reproducibility checks stay exact.
    """
    gen = config.derive_generator("noise_test", eval_seed)
    pos, ctl, vel = closed_loop_rollout(comm, s_eval, t_max=t_max,
                                        snr_db=snr_db, channel=channel,
                                        generator=gen)
    # task_loss is computed OUTSIDE no_grad so the same deterministic
    # evaluation can also be used for backward/finite-difference checks;
    # the extra statistics below are pure diagnostics.
    out = dict(task_loss(pos, ctl, vel, s_eval[:, 4:6]))
    with torch.no_grad():
        goal = s_eval[:, 4:6]
        d_final = (pos[:, -1, :] - goal).norm(dim=-1)
        d_min = (pos[:, 1:, :] - goal.unsqueeze(1)).norm(dim=-1).min(dim=1).values
        out["final_distance_mean"] = d_final.mean()
        out["min_distance_mean"] = d_min.mean()
        out["success_rate"] = (d_min <= config.GOAL_RADIUS).float().mean()
    return out


def train_task_oriented(
    k: int = 3,
    channel: str = "awgn",
    seed: int = 42,
    steps: int = 300,
    batch_episodes: int = 128,
    t_max: int = config.T_MAX,
    lr: Optional[float] = None,
    snr_range: Tuple[float, float] = config.SNR_TRAIN_RANGE_DB,
    eval_episodes: int = 256,
    eval_snr_db: float = 10.0,
    verbose_every: int = 0,
) -> Tuple[ReconstructionDeepJSCC, dict]:
    """Smoke-scale task-oriented training; returns (model, report).

    Deterministic: fixed seeds for init/episodes/SNR/noise; a fixed
    evaluation episode set is drawn once and reused for the before/after
    task-loss comparison with identical channel realizations.
    """
    if channel not in ("awgn", "rayleigh"):
        raise ValueError(f"unknown channel {channel!r}")
    config.set_global_seed(seed)                     # weight init
    gen_states = config.derive_generator("episodes", seed)
    gen_snr = config.derive_generator("noise_train", seed)
    gen_noise = config.derive_generator("noise_test", seed)
    env = NavigationEnv()

    # fixed deterministic evaluation batch (same distribution, fixed seeds)
    gen_eval = config.derive_generator("episodes", seed + 1)
    p0e, v0e, goale = env.sample_starts(eval_episodes, generator=gen_eval)
    s_eval = torch.cat([p0e, v0e, goale], dim=-1)     # [B, 6] full state

    lr = config.LEARNING_RATE if lr is None else lr
    model = ReconstructionDeepJSCC(k=k)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    enc_params = list(model.encoder.parameters())
    dec_params = list(model.decoder.parameters())

    def _gnorm(params):
        return math.sqrt(sum(float(p.grad.norm()) ** 2 for p in params
                             if p.grad is not None))

    eval_before = evaluate_task_loss(model, s_eval, t_max, eval_snr_db,
                                     channel, eval_seed=seed + 2)
    history, g_enc_h, g_dec_h = [], [], []
    nan_batches = 0

    for step_idx in range(steps):
        p0, v0, goal = env.sample_starts(batch_episodes, generator=gen_states)
        s0 = torch.cat([p0, v0, goal], dim=-1)
        snr_db = float(torch.empty(1).uniform_(
            snr_range[0], snr_range[1], generator=gen_snr))

        pos, ctl, vel = closed_loop_rollout(model, s0, t_max=t_max,
                                            snr_db=snr_db, channel=channel,
                                            generator=gen_noise)
        out = task_loss(pos, ctl, vel, goal)
        loss = out["total"]

        # detached diagnostics only (never optimized)
        with torch.no_grad():
            s_bar_hat_diag = model.forward_losses(s0, snr_db, channel=channel,
                                                  generator=gen_noise)
            recon_mse = float(((s_bar_hat_diag - normalize(s0)) ** 2)
                              .sum(dim=-1).mean() / 6.0)

        if not bool(torch.isfinite(loss)):
            nan_batches += 1
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_enc, g_dec = _gnorm(enc_params), _gnorm(dec_params)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.GRAD_CLIP_NORM)
        opt.step()

        history.append({"total": float(loss), "L_d": float(out["L_d"]),
                        "L_u": float(out["L_u"]), "L_v": float(out["L_v"]),
                        "L_T": float(out["L_T"]), "recon_mse_diag": recon_mse,
                        "grad_enc": g_enc, "grad_dec": g_dec})
        g_enc_h.append(g_enc)
        g_dec_h.append(g_dec)
        if verbose_every and (step_idx % verbose_every == 0 or
                              step_idx == steps - 1):
            h = history[-1]
            print(f"  [task {channel} k={k}] step {step_idx:4d} "
                  f"snr {snr_db:5.1f}  total {h['total']:9.3f}  "
                  f"Ld {h['L_d']:8.3f}  LT {h['L_T']:9.3f}  "
                  f"recon(diag) {h['recon_mse_diag']:.4f}  "
                  f"|g| {h['grad_enc']:.2e}/{h['grad_dec']:.2e}")

    eval_after = evaluate_task_loss(model, s_eval, t_max, eval_snr_db,
                                    channel, eval_seed=seed + 2)

    report = {
        "channel": channel, "k": k, "seed": seed,
        "steps": steps, "batch_episodes": batch_episodes, "t_max": t_max,
        "lr": lr, "snr_range_db": snr_range,
        "eval_episodes": eval_episodes, "eval_snr_db": eval_snr_db,
        "lambdas": {"ld": config.LAMBDA_DISTANCE,
                    "lu": config.LAMBDA_CONTROL,
                    "lv": config.LAMBDA_VELOCITY,
                    "lT": config.LAMBDA_TERMINAL},
        "eval_before": {key: float(val) for key, val in eval_before.items()},
        "eval_after": {key: float(val) for key, val in eval_after.items()},
        "train_total_initial": history[0]["total"] if history else float("nan"),
        "train_total_final": history[-1]["total"] if history else float("nan"),
        "train_Ld_initial": history[0]["L_d"] if history else float("nan"),
        "train_Ld_final": history[-1]["L_d"] if history else float("nan"),
        "train_Lu_initial": history[0]["L_u"] if history else float("nan"),
        "train_Lu_final": history[-1]["L_u"] if history else float("nan"),
        "train_Lv_initial": history[0]["L_v"] if history else float("nan"),
        "train_Lv_final": history[-1]["L_v"] if history else float("nan"),
        "train_LT_initial": history[0]["L_T"] if history else float("nan"),
        "train_LT_final": history[-1]["L_T"] if history else float("nan"),
        "recon_mse_diag_initial": history[0]["recon_mse_diag"] if history else float("nan"),
        "recon_mse_diag_final": history[-1]["recon_mse_diag"] if history else float("nan"),
        "grad_enc_first": g_enc_h[0] if g_enc_h else float("nan"),
        "grad_enc_final": g_enc_h[-1] if g_enc_h else float("nan"),
        "grad_dec_first": g_dec_h[0] if g_dec_h else float("nan"),
        "grad_dec_final": g_dec_h[-1] if g_dec_h else float("nan"),
        "nan_batches": nan_batches,
        "history": history,
    }
    return model, report


if __name__ == "__main__":
    for ch in ("awgn", "rayleigh"):
        _, rep = train_task_oriented(channel=ch, verbose_every=50)
        print(f"\n=== TASK-ORIENTED {ch.upper()} smoke ===")
        for key in ("train_total_initial", "train_total_final",
                    "train_Ld_initial", "train_Ld_final",
                    "train_LT_initial", "train_LT_final",
                    "grad_enc_first", "grad_enc_final",
                    "grad_dec_first", "grad_dec_final",
                    "nan_batches"):
            print(f"  {key}: {rep[key]}")
        print("  eval (fixed episodes, fixed channel realizations):")
        for key in sorted(rep["eval_before"]):
            print(f"    {key:20s} before={rep['eval_before'][key]:10.4f}  "
                  f"after={rep['eval_after'][key]:10.4f}")
