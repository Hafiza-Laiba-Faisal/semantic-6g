"""Step-14b — Thesis figures for M1 + M2 (READ-ONLY over results/).

Generates publication-grade PNGs into fyp_md_files/figures/:

  fig1_m1_sr_vs_snr.png        M1: recon vs task, k={1,2,3}, AWGN|Rayleigh
  fig2_m2_awgn.png             M2 AWGN: 3 methods, k={12,18,24,30,54}
  fig3_m2_rayleigh.png         M2 Rayleigh: 3 methods, k={12,18,24,30,54}
  fig4_bandwidth_crossover.png M2 AWGN: SR vs k at 0 dB and 20 dB
  fig5_full_bandwidth_axis.png M1+M2 combined: SR vs k at 0 dB (AWGN)

Run:  .venv/Scripts/python.exe tools/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
FIG = ROOT / "fyp_md_files" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SNR1 = [0.0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
SNR2 = [0.0, 5.0, 10.0, 15.0, 20.0]
KS1 = (1, 2, 3)
KS2 = (12, 18, 24, 30, 54)
CHANNELS = ("awgn", "rayleigh")
SEEDS = (42, 43, 44)

C_RECON, C_TASK, C_DIG = "#0072B2", "#D55E00", "#009E73"
K_SHADE = {1: "#74a9cf", 2: "#3690c0", 3: "#045a8d",
           12: "#74a9cf", 18: "#3690c0", 24: "#045a8d",
           30: "#034e7b", 54: "#023858"}
K_SHADE_T = {1: "#f4a582", 2: "#ef6548", 3: "#d7301f",
             12: "#f4a582", 18: "#ef6548", 24: "#d7301f",
             30: "#b30000", 54: "#7f0000"}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


def pooled(path):
    s = json.loads(Path(path).read_text())
    return {(r["method"], r["k"], r["channel"], r["snr_db"]): r
            for r in s["pooled_rows"]}


p1 = pooled("results/M1/step10_m1_summary.json")
p2 = pooled("results/M2/step11_m2_summary.json")

# M4 secondary reference (B=8 at k=54), from results/M4/step13_m4_summary.json
M4 = {"awgn":  [0.9736, 1.0, 1.0, 1.0, 1.0],
      "rayleigh": [0.7178, 0.9908, 0.9984, 0.9998, 1.0]}


def sr1(m, k, ch, s):
    return p1[(m, k, ch, s)]["success_rate_pooled"]


def sr2(m, k, ch, s):
    try:
        return p2[(m, k, ch, s)]["success_rate_pooled"]
    except KeyError:          # excluded conditions (see PROTOCOL_EXCEPTIONS.md)
        return float("nan")


# ---------------- Figure 1: M1 recon vs task ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, ch in zip(axes, CHANNELS):
    for k in KS1:
        ax.plot(SNR1, [sr1("reconstruction", k, ch, s) for s in SNR1],
                "-o", color=K_SHADE[k], ms=3.5,
                label=f"recon k={k} (ρ={k}/6)")
        ax.plot(SNR1, [sr1("task", k, ch, s) for s in SNR1],
                "--s", color=K_SHADE_T[k], ms=3.5,
                label=f"task k={k} (ρ={k}/6)")
    ax.set_title(f"M1 — {ch.upper()} (pooled over 3 seeds)")
    ax.set_xlabel("SNR (dB)")
    ax.set_xticks(SNR1)
    ax.set_ylim(0, 0.36)
axes[0].set_ylabel("Goal-reaching success rate")
axes[0].legend(ncol=2, loc="upper right")
fig.suptitle("M1 (low-k regime): reconstruction vs task-oriented DeepJSCC",
             y=1.04, fontsize=12)
fig.savefig(FIG / "fig1_m1_sr_vs_snr.png")
plt.close(fig)

# ---------------- Figures 2/3: M2 per-channel ---------------------------
for ch, fname in (("awgn", "fig2_m2_awgn.png"),
                  ("rayleigh", "fig3_m2_rayleigh.png")):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.1), sharey=True)
    for ax, k in zip(axes, KS2):
        ax.plot(SNR2, [sr2("reconstruction", k, ch, s) for s in SNR2],
                "-o", color=C_RECON, ms=4, label="recon")
        ax.plot(SNR2, [sr2("task", k, ch, s) for s in SNR2],
                "--s", color=C_TASK, ms=4, label="task")
        ax.plot(SNR2, [sr2("digital", k, ch, s) for s in SNR2],
                "-^", color=C_DIG, ms=4, label="digital")
        ax.set_title(f"k={k}  (ρ={k}/6)")
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks(SNR2)
        ax.set_ylim(0, 1.05)
        if k == 54:  # secondary B=8 reference
            ax.plot(SNR2, M4[ch], ":*", color="0.35", ms=7,
                    label="digital B=8 (secondary)")
        if ch == "awgn" and k in (30, 54):
            ax.text(0.5, 0.42, "EXCLUDED\n(2 protocol-strict failures)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8.5, color="0.35")
    axes[0].set_ylabel("Success rate")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"M2 matched-budget 3-way comparison — {ch.upper()} "
                 "(pooled over 3 seeds, 15000 episodes/condition)",
                 y=1.05, fontsize=12)
    fig.savefig(FIG / fname)
    plt.close(fig)

# ---------------- Figure 4: bandwidth crossover (AWGN) ------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, snr in zip(axes, (0.0, 20.0)):
    for m, c, mk, lbl in (("reconstruction", C_RECON, "o", "recon"),
                          ("task", C_TASK, "s", "task"),
                          ("digital", C_DIG, "^", "digital")):
        ax.plot(KS2, [sr2(m, k, "awgn", snr) for k in KS2],
                f"-{mk}", color=c, ms=6, label=lbl)
    ax.set_title(f"AWGN @ {snr:g} dB")
    ax.set_xlabel("k (channel uses per state)")
    ax.set_xticks(KS2)
    ax.set_ylim(0, 0.55)
    ax.axvspan(27, 58, color="0.94", zorder=0)
    ax.text(40, 0.47, "k=30, 54 excluded\n(2 cells)", ha="center",
            fontsize=8, color="0.35")
axes[0].set_ylabel("Success rate")
axes[0].annotate("digital overtakes recon\nbetween k=18 and k=24",
                 xy=(24, 0.305), xytext=(27, 0.12), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="0.3"))
axes[0].legend(loc="upper left")
fig.suptitle("M2 AWGN: success rate vs bandwidth (matched budgets, "
             "B = (k−6)/6)", y=1.04, fontsize=12)
fig.savefig(FIG / "fig4_bandwidth_crossover.png")
plt.close(fig)

# ---------------- Figure 5: full bandwidth axis (M1+M2, 0 dB) -----------
fig, ax = plt.subplots(figsize=(8.5, 3.8))
k_all = list(KS1) + list(KS2)
ax.plot(KS1, [sr1("reconstruction", k, "awgn", 0.0) for k in KS1],
        "-o", color=C_RECON, ms=6, label="recon (M1)")
ax.plot(KS1, [sr1("task", k, "awgn", 0.0) for k in KS1],
        "--s", color=C_TASK, ms=6, label="task (M1)")
ax.plot(KS2, [sr2("reconstruction", k, "awgn", 0.0) for k in KS2],
        "-o", color=C_RECON, ms=6, label="recon (M2)")
ax.plot(KS2, [sr2("task", k, "awgn", 0.0) for k in KS2],
        "--s", color=C_TASK, ms=6, label="task (M2)")
ax.plot(KS2, [sr2("digital", k, "awgn", 0.0) for k in KS2],
        "-^", color=C_DIG, ms=6, label="digital (M2)")
ax.axvline(7.5, color="0.6", lw=1, ls=":")
ax.axvspan(27, 58, color="0.94", zorder=0)
ax.text(40, 0.05, "k=30, 54 excluded\n(2 cells)", ha="center", fontsize=8,
        color="0.35")
ax.text(4.2, 0.33, "M1 regime\nρ ≤ 1/2", ha="center", fontsize=9,
        color="0.3")
ax.text(27, 0.33, "M2 regime\nρ ≥ 2", ha="center", fontsize=9, color="0.3")
ax.set_xticks(k_all)
ax.set_ylim(0, 0.4)
ax.set_xlabel("k (channel uses per 6-D state)")
ax.set_ylabel("Success rate @ 0 dB (AWGN)")
ax.legend(ncol=3, loc="upper right")
ax.set_title("Full bandwidth axis at 0 dB — M1 + M2 (AWGN, pooled)")
fig.savefig(FIG / "fig5_full_bandwidth_axis.png")
plt.close(fig)

print("figures written to", FIG)
for f in sorted(FIG.glob("*.png")):
    print(" ", f.name, f.stat().st_size // 1024, "KB")
