"""Step-14d — Semantic/JSCC-side thesis figures (READ-ONLY over results/).

  fig13_mse_vs_snr.png       reconstruction MSE vs SNR, 3 methods, 2 channels
  fig14_rate_distortion.png  rate-distortion: MSE vs bandwidth rho at 0/10/20 dB
  fig15_constellation.png    learned encoder symbol scatter (recon vs task)

Caveat shown on figures: trajectory MSE censors exited episodes (post-exit
su = s), so MSE curves are diagnostic-grade (see ARCHITECTURE.md section 5).

Run:  .venv/Scripts/python.exe tools/make_figures_semantic.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project.data.normalization import normalize
from project.evaluation.evaluate import Condition, generate_realizations
from project.models.reconstruction_jscc import ReconstructionDeepJSCC

FIG = Path("fyp_md_files") / "figures"
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

s = json.loads(Path("results/M2/step11_m2_summary.json").read_text())
p2 = {(r["method"], r["k"], r["channel"], r["snr_db"]): r
      for r in s["pooled_rows"]}


def mse(m, k, ch, snr):
    return p2[(m, k, ch, snr)]["mse_mean_mean_over_seeds"]


CAVEAT = ("trajectory MSE censors boundary-exited episodes "
          "(post-exit su = s) — diagnostic metric, see thesis section 5.4")

# ---- fig13: reconstruction MSE vs SNR --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True)
for ax, ch in zip(axes, CHANNELS):
    for m, lab, c, mk in METH:
        for k in KS2:
            ys = [mse(m, k, ch, x) for x in SNR2]
            ax.plot(SNR2, ys, marker=mk, ms=3.5, lw=1.3, color=c,
                    alpha=0.45 + 0.55 * KS2.index(k) / 4,
                    label=f"{lab} k={k}")
    ax.set_title(ch.upper())
    ax.set_xlabel("SNR (dB)")
    ax.set_xticks(SNR2)
    ax.set_yscale("log")
axes[0].set_ylabel("trajectory MSE (normalized state, /6)")
axes[0].legend(ncol=2, fontsize=6.8, loc="upper right")
fig.text(0.5, -0.02, CAVEAT, ha="center", fontsize=8, color="0.35")
fig.suptitle("Semantic reconstruction fidelity vs SNR — all methods, "
             "all bandwidths", y=1.04, fontsize=12)
fig.savefig(FIG / "fig13_mse_vs_snr.png")
plt.close(fig)

# ---- fig14: rate-distortion (MSE vs bandwidth rho) --------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
for ax, snr in zip(axes, (0.0, 10.0, 20.0)):
    for m, lab, c, mk in METH:
        ax.plot([k / 6 for k in KS2],
                [mse(m, k, "awgn", snr) for k in KS2],
                marker=mk, ms=5, lw=1.5, color=c, label=lab)
    ax.set_title(f"AWGN @ {snr:g} dB")
    ax.set_xlabel("bandwidth ratio ρ = k/6")
    ax.set_yscale("log")
    ax.set_xticks([k / 6 for k in KS2])
    ax.set_xticklabels([f"{k/6:g}\n(k={k})" for k in KS2], fontsize=8)
axes[0].set_ylabel("trajectory MSE (normalized state, /6)")
axes[0].legend(loc="upper right")
fig.text(0.5, -0.02, CAVEAT, ha="center", fontsize=8, color="0.35")
fig.suptitle("Rate-distortion view — semantic fidelity vs bandwidth "
             "(AWGN, pooled)", y=1.04, fontsize=12)
fig.savefig(FIG / "fig14_rate_distortion.png")
plt.close(fig)

# ---- fig15: learned encoder constellation -----------------------------------
def load(glob):
    d = sorted(Path("results/M2").glob(glob))[0]
    m = ReconstructionDeepJSCC(k=24)
    m.load_state_dict(torch.load(d / "final_ckpt.pt", weights_only=False,
                                 map_location="cpu")["model_state"])
    m.eval()
    return m


m_recon = load("M2_reconstruction_awgn_k24_seed42__*")
m_task = load("M2_task_awgn_k24_seed42__*")

cond = Condition(k=24, channel="awgn", bits_per_component=3, t_max=100,
                 episodes=500, seed=10042, snr_db=10.0)
real = generate_realizations(cond, episodes=500)
st = torch.cat([real["p0"], real["v0"], real["goal"]], dim=-1)
sbar = normalize(st)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharex=True, sharey=True)
for ax, model, lab, c in ((axes[0], m_recon, "recon encoder", C_RECON),
                          (axes[1], m_task, "task encoder", C_TASK)):
    with torch.no_grad():
        out = model.encoder(sbar)                     # [E, 2k] real
    z0 = (out[:, 0] + 1j * out[:, 1]).numpy()         # symbol 1
    z1 = (out[:, 2] + 1j * out[:, 3]).numpy()         # symbol 2
    ax.scatter(z0.real, z0.imag, s=6, alpha=0.35, color=c,
               label="symbol 1")
    ax.scatter(z1.real, z1.imag, s=6, alpha=0.35, color="0.35",
               marker="^", label="symbol 2")
    ax.set_title(f"{lab} (k=24): first 2 of 24 symbols, 500 states")
    ax.set_xlabel("in-phase")
    ax.set_ylabel("quadrature")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
fig.suptitle("Learned JSCC code structure — pre-channel encoder output "
             "@ 10 dB (semantic vs reconstruction objective)", y=1.02,
             fontsize=12)
fig.savefig(FIG / "fig15_constellation.png")
plt.close(fig)

print("figures written to", FIG)
for f in sorted(FIG.glob("fig1[3-5]*.png")):
    print(" ", f.name, f.stat().st_size // 1024, "KB")
