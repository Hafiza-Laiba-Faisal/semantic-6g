"""Figures for the A-10 (aerial), A-11 (polar) and A-12 (aerial training) blocks.

Reads existing artifacts only (read-only):
  results/AERIAL/step_aerial_summary.json
  results/POLAR/step_polar_summary_E1000.json
  results/AERIAL_TRAIN/step_aerial_training_summary.json

Writes:
  fyp_md_files/figures/fig_aerial_required_snr.png
  fyp_md_files/figures/fig_polar_vs_conv.png
  fyp_md_files/figures/fig_aerial_training.png

No training, no evaluation, no writes under results/.
Run:  .venv\\Scripts\\python.exe tools/make_figures_extensions.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AER = ROOT / "results" / "AERIAL" / "step_aerial_summary.json"
POL = ROOT / "results" / "POLAR" / "step_polar_summary_E1000.json"
FIG = ROOT / "fyp_md_files" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})

NAME = {"lam_ctl_heavy_k18": "lambda-shaped, k=18",
        "lam_ctl_heavy_k54": "lambda-shaped, k=54",
        "digital_k18": "digital (conv), k=18",
        "digital_k54": "digital (conv), k=54"}
COLOR = {"lam_ctl_heavy_k18": "#d62728", "lam_ctl_heavy_k54": "#ff7f0e",
         "digital_k18": "#2ca02c", "digital_k54": "#1f77b4"}


def aerial():
    d = json.loads(AER.read_text())
    methods = d["methods"]
    rows = [m for m in ("lam_ctl_heavy_k18", "lam_ctl_heavy_k54",
                        "digital_k18", "digital_k54") if m in methods]
    y = np.arange(len(rows))
    h = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    xmax = 130.0
    for target, off, hatch, col in (("0.90", -h / 2, "", "#4c72b0"),
                                    ("0.99", +h / 2, "///", "#dd8452")):
        for yi, m in zip(y, rows):
            v = methods[m]["required_tx_db"].get(target)
            if v is None:
                ax.text(43.0, yi + off, "not reachable at any SNR", va="center",
                        fontsize=6.5, color="#777", style="italic")
            else:
                ax.barh(yi + off, v - 42.0, left=42.0, height=h, color=col,
                        hatch=hatch, edgecolor="white", lw=0.6)
                ax.text(v + 1.0, yi + off, f"{v:.0f} dB", va="center",
                        fontsize=7, color="#333")
        ax.barh(0, 0, color=col, hatch=hatch, edgecolor="white",
                label=f"SR >= {target}")
    ax.set_yticks(y)
    ax.set_yticklabels([NAME[m] for m in rows])
    ax.set_xlim(42, xmax)
    ax.set_xlabel("required transmit SNR (dB, relative to a unit-gain benchmark)\n"
                  f"dotted line = mean path loss {d['aerial']['pl_mean_db']:.1f} dB")
    ax.axvline(d["aerial"]["pl_mean_db"], color="grey", ls=":", lw=1)
    ax.set_title("3GPP UMa-aerial, d2D=500 m, h=50 m, 2 GHz (A-10), E=1000")
    ax.legend(frameon=False, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), ncol=2)
    fig.tight_layout()
    out = FIG / "fig_aerial_required_snr.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def polar():
    d = json.loads(POL.read_text())
    rows = d["rows"]
    snrs = [float(x) for x in d["design"]["snr_grid_db"]]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), sharey=True)
    for ax, k in zip(axes, (18, 30, 54)):
        for ch, ls in (("awgn", "-"), ("rayleigh", ":")):
            for m, col, mk in (("polar", "#c44e52", "s"),
                               ("conv", "#1f77b4", "o")):
                ys = [next(r[m]["success_rate"] for r in rows
                           if r["k"] == k and r["channel"] == ch
                           and r["snr_db"] == s) for s in snrs]
                ax.plot(snrs, ys, ls, color=col, marker=mk, ms=3.2, lw=1.2,
                        label=f"{m} ({ch})", alpha=0.95 if ch == "awgn" else 0.65)
        ax.set_title(f"k = {k} (B = {d['design']['bits_by_k'][str(k)]})")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylim(0, 1.06)
        ax.set_xticks(snrs)
    axes[0].set_ylabel("success rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Polar (CRC-8, SC) vs convolutional baseline, E=1000 (A-11)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = FIG / "fig_polar_vs_conv.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def aerial_training():
    src = (ROOT / "results" / "AERIAL_TRAIN"
           / "step_aerial_training_summary.json")
    if not src.exists():
        print("skip aerial_training: summary not written yet")
        return
    d = json.loads(src.read_text())
    pl = d["channel"]["pl_mean_db"]
    style = {
        "frozen_rayleigh": ("trained on Rayleigh (A-10 baseline)",
                            "#1f77b4", "o", "-"),
        "aerial_train": ("trained on 3GPP, full dispersion (A-12)",
                         "#d62728", "s", "-"),
        "digital": ("digital (conv), B = M2_B_BY_K", "#2ca02c", "^", ":"),
    }
    # A-13 (if it ran): same design, training tail above the cap resampled
    src13 = (ROOT / "results" / "AERIAL_TRAIN_TRUNC"
             / "step_aerial_training_summary.json")
    d13 = json.loads(src13.read_text()) if src13.exists() else None
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    for ax, k in zip(axes, (18, 54)):
        for method, (label, col, mk, ls) in style.items():
            rows = d["curves"].get(f"{method}_k{k}")
            if not rows:
                continue
            ax.plot([r["nominal_snr_db"] for r in rows],
                    [r["success_rate"] for r in rows],
                    ls, color=col, marker=mk, ms=3.2, lw=1.3, label=label)
        if d13 is not None:
            rows = d13["curves"].get(f"aerial_train_k{k}")
            cap = d13.get("aerial_train_max_offset_db")
            if rows and cap is not None:
                ax.plot([r["nominal_snr_db"] for r in rows],
                        [r["success_rate"] for r in rows],
                        "-", color="#9467bd", marker="D", ms=3.2, lw=1.5,
                        label=f"trained on 3GPP, outage tail "
                              f">{cap:g} dB resampled (A-13)")
        for t in (0.9, 0.99):
            ax.axhline(t, color="grey", lw=0.6, ls=":")
        ax.set_title(f"k = {k}")
        ax.set_xlabel("nominal SNR (dB);  tx = nominal + "
                      f"{pl:.1f} dB")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("success rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, -0.06))
    title = "Training under the 3GPP UMa-aerial dispersion"
    if d13 is not None:
        title += " (A-12 vs A-13)"
    fig.suptitle(title + ", E=1000, paired gains", fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = FIG / "fig_aerial_training.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    aerial()
    polar()
    aerial_training()