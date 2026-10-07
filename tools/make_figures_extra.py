"""Step-14c — Extended thesis figures (READ-ONLY over results/).

One paired re-evaluation (5000 episodes, k=24 AWGN 10 dB) for the CDF and
3 short rollouts for the trajectory figure; everything else from pooled
summaries and checkpoint histories.

  fig6_final_distance.png    M2 final distance vs SNR (3 methods, 2 channels)
  fig7_control_effort.png    M2 control effort vs SNR
  fig8_time_to_goal.png      success-conditional time-to-goal vs SNR
  fig9_sr_heatmaps.png       SR heatmaps: 3 methods x 2 channels (k x SNR)
  fig10_training_curves.png  task loss + recon loss vs training step
  fig11_cdf_final_dist.png   CDF of final distance, k=24 AWGN 10 dB
  fig12_trajectories.png     sample closed-loop trajectories, k=24 AWGN 10 dB

Run:  .venv/Scripts/python.exe tools/make_figures_extra.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(".")
FIG = ROOT / "fyp_md_files" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SNR2 = [0.0, 5.0, 10.0, 15.0, 20.0]
KS2 = (12, 18, 24, 30, 54)
CHANNELS = ("awgn", "rayleigh")
C_RECON, C_TASK, C_DIG = "#0072B2", "#D55E00", "#009E73"
METH = (("reconstruction", "recon", C_RECON, "o"),
        ("task", "task", C_TASK, "s"),
        ("digital", "digital", C_DIG, "^"))

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


def pooled(path):
    s = json.loads(Path(path).read_text())
    return {(r["method"], r["k"], r["channel"], r["snr_db"]): r
            for r in s["pooled_rows"]}


p2 = pooled("results/M2/step11_m2_summary.json")


def val(field, m, k, ch, s):
    return p2[(m, k, ch, s)][field]


def metric_panel(field, ylabel, fname, title, logy=False):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, ch in zip(axes, CHANNELS):
        for m, lab, c, mk in METH:
            ys = [val(field, m, k, ch, s) for k in KS2 for s in SNR2]
            for k in KS2:
                xs = SNR2
                ys = [val(field, m, k, ch, s) for s in SNR2]
                std = [p2[(m, k, ch, s)][field.replace(
                    "mean_over_seeds", "std_over_seeds")] for s in SNR2]
                ax.errorbar(xs, ys, yerr=std, marker=mk, color=c, ms=4,
                            capsize=2, lw=1.4, label=f"{lab} k={k}")
        ax.set_title(f"{ch.upper()}")
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks(SNR2)
        if logy:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(ncol=2, fontsize=7)
    fig.suptitle(title, y=1.04, fontsize=12)
    fig.savefig(FIG / fname)
    plt.close(fig)


# ---- fig6/7/8 ------------------------------------------------------------
metric_panel("final_distance_mean_mean_over_seeds", "Final distance (m)",
             "fig6_final_distance.png",
             "M2: final distance to goal vs SNR (mean ± std over 3 seeds)")
metric_panel("control_effort_mean_mean_over_seeds",
             "Control effort  Σ‖u‖²  (lower = less energy)",
             "fig7_control_effort.png",
             "M2: control effort vs SNR (mean ± std over 3 seeds)")
metric_panel("ttg_success_mean_mean_over_seeds",
             "Time-to-goal among successes (steps)", "fig8_time_to_goal.png",
             "M2: success-conditional time-to-goal vs SNR "
             "(task curves use few successes — interpret with care)")

# ---- fig9: SR heatmaps ----------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey=True)
for row, ch in enumerate(CHANNELS):
    for col, (m, lab, c, _) in enumerate(METH):
        grid = [[p2[(m, k, ch, s)]["success_rate_pooled"] for s in SNR2]
                for k in KS2]
        ax = axes[row][col]
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       origin="lower")
        for i, k in enumerate(KS2):
            for j, s in enumerate(SNR2):
                ax.text(j, i, f"{grid[i][j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if grid[i][j] < 0.6 else "black")
        ax.set_xticks(range(5), [f"{s:g}" for s in SNR2])
        ax.set_yticks(range(5), [str(k) for k in KS2])
        if row == 1:
            ax.set_xlabel("SNR (dB)")
        if col == 0:
            ax.set_ylabel("k (uses/state)")
        ax.set_title(f"{lab} — {ch}")
fig.colorbar(im, ax=axes, shrink=0.8, label="Success rate")
fig.suptitle("M2 success-rate maps (pooled, 15000 episodes/cell)",
             y=0.99, fontsize=12)
fig.savefig(FIG / "fig9_sr_heatmaps.png")
plt.close(fig)

# ---- fig10: training curves ----------------------------------------------
def load_history(dir_glob):
    d = sorted(Path("results/M2").glob(dir_glob))[0]
    payload = torch.load(d / "final_ckpt.pt", weights_only=False,
                         map_location="cpu")
    return payload["history"]


fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ch, c in (("awgn", C_RECON), ("rayleigh", C_DIG)):
    h = load_history("M2_task_awgn_k24_seed42__*"
                     if ch == "awgn" else
                     "M2_task_rayleigh_k24_seed42__*")
    tot = [e["total"] for e in h if isinstance(e, dict) and "total" in e]
    steps = range(0, len(tot), 10)
    axes[0].plot(steps, [tot[i] for i in steps], color=c, lw=1.2,
                 label=f"task {ch} (k=24, seed 42)")
    hr = load_history(f"M2_reconstruction_{ch}_k24_seed42__*")
    rl = [e if isinstance(e, (int, float)) else
          (e.get("loss", e.get("total")) if isinstance(e, dict) else None)
          for e in hr]
    rl = [x for x in rl if x is not None]
    steps = range(0, len(rl), 10)
    axes[1].plot(steps, [rl[i] for i in steps], color=c, lw=1.2,
                 label=f"recon {ch} (k=24, seed 42)")
axes[0].set_title("Task-oriented training loss")
axes[1].set_title("Reconstruction training loss")
for ax in axes:
    ax.set_xlabel("training step")
    ax.set_yscale("log")
    ax.legend()
axes[0].set_ylabel("loss")
fig.suptitle("Frozen-protocol training curves (k=24, seed 42)",
             y=1.03, fontsize=12)
fig.savefig(FIG / "fig10_training_curves.png")
plt.close(fig)

# ---- fig11: CDF of final distance (one paired re-eval) --------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project.evaluation.evaluate import (Condition, evaluate_condition,
                                         generate_realizations)
from project.models.reconstruction_jscc import ReconstructionDeepJSCC
from project.training.train_task_oriented import closed_loop_rollout

cond = Condition(k=24, channel="awgn", bits_per_component=3, t_max=100,
                 episodes=5000, seed=10042, snr_db=10.0)
m_recon = ReconstructionDeepJSCC(k=24)
m_recon.load_state_dict(torch.load(
    sorted(Path("results/M2").glob(
        "M2_reconstruction_awgn_k24_seed42__*/final_ckpt.pt"))[0],
    weights_only=False, map_location="cpu")["model_state"])
m_recon.eval()
m_task = ReconstructionDeepJSCC(k=24)
m_task.load_state_dict(torch.load(
    sorted(Path("results/M2").glob(
        "M2_task_awgn_k24_seed42__*/final_ckpt.pt"))[0],
    weights_only=False, map_location="cpu")["model_state"])
m_task.eval()

cell = evaluate_condition(cond, recon_model=m_recon, task_model=m_task)
fig, ax = plt.subplots(figsize=(6.2, 4))
for m, lab, c, _ in METH:
    d = torch.sort(cell[m]["metrics"]["final_distance"]
                   .detach().to(torch.float64)).values.numpy()
    ax.plot(d, torch.arange(1, len(d) + 1).numpy() / len(d), color=c,
            lw=1.6, label=f"{lab}  (median {d[len(d)//2]:.2f} m)")
ax.axvline(0.5, color="0.5", ls=":", lw=1)
ax.text(0.53, 0.06, "goal radius 0.5 m\n(d ≤ 0.5 at ANY step = success)",
        fontsize=8, color="0.3")
ax.set_xlabel("final distance to goal (m)")
ax.set_ylabel("CDF over 5000 paired episodes")
ax.set_title("k=24, AWGN, 10 dB — final-distance CDF (paired episodes)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.savefig(FIG / "fig11_cdf_final_dist.png")
plt.close(fig)
print("fig11 done (paired re-eval 5000 eps)")

# ---- fig12: sample trajectories -------------------------------------------
real = generate_realizations(cond, episodes=3)
s0 = torch.cat([real["p0"], real["v0"], real["goal"]], dim=-1)
fig, ax = plt.subplots(figsize=(6.2, 6))
styles = ("-", "--", "-.")
for m, model in (("recon", m_recon), ("task", m_task)):
    pos, _, _ = closed_loop_rollout(model, s0, t_max=100, snr_db=10.0,
                                    channel="awgn",
                                    generator=real.get("gen_noise"))
    for e in range(3):
        ax.plot(pos[e, :, 0].detach().numpy(), pos[e, :, 1].detach().numpy(),
                styles[e], color=C_RECON if m == "recon" else C_TASK,
                lw=1.3, alpha=0.9,
                label=f"{m}" if e == 0 else None)
        ax.plot(pos[e, 0, 0].item(), pos[e, 0, 1].item(), "ko", ms=4)
        ax.plot(pos[e, -1, 0].item(), pos[e, -1, 1].item(), "x",
                color=C_RECON if m == "recon" else C_TASK, ms=7, mew=2)
g = real["goal"]
ax.plot(g[:, 0], g[:, 1], "*", color="gold", ms=16, mec="k",
        label="goal", ls="none")
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Sample closed-loop trajectories — k=24, AWGN, 10 dB\n"
             "(black dot = start, x = end; 3 paired episodes per method)")
ax.legend(loc="upper right")
ax.set_aspect("equal")
fig.savefig(FIG / "fig12_trajectories.png")
plt.close(fig)

print("figures written to", FIG)
for f in sorted(FIG.glob("*.png")):
    print(" ", f.name, f.stat().st_size // 1024, "KB")
