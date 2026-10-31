"""Figures for the M3 / M4 blocks, the 3GPP UMa-aerial model, and the matrix.

Reads existing artifacts only (read-only):
  results/M3/m3_report.json
  results/M4/step13_m4_summary.json
  results/M2/step11_m2_summary.json   (for the 0 dB comparison)
  results/WAVE_C/step_wave_c_summary.json (lambda-heavy reference points)

Writes:
  fyp_md_files/figures/fig_m3_infeasibility.png
  fyp_md_files/figures/fig_m4_bandwidth_reference.png
  fyp_md_files/figures/fig_3gpp_channel_model.png
  fyp_md_files/figures/fig_experiment_matrix.png

No training, no evaluation, no writes under results/.
Run:  .venv\\Scripts\\python.exe tools/make_figures_m3_m4.py
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project.channels import aerial  # noqa: E402

FIG = ROOT / "fyp_md_files" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.28, "axes.spines.top": False, "axes.spines.right": False,
})


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def m3_infeasibility():
    d = _load(ROOT / "results" / "M3" / "m3_report.json")
    cells = d["cells"]
    ks = [c["k"] for c in cells]
    avail = [c["available_channel_uses"] for c in cells]
    req = [c["required_channel_uses"] for c in cells]

    fig, axes = plt.subplots(
        1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [1.0, 1.45]})

    ax = axes[0]
    x = np.arange(len(ks))
    w = 0.36
    ax.bar(x - w / 2, avail, w, label="available k", color="#4c72b0")
    ax.bar(x + w / 2, req, w, label="required 6B + 6", color="#c44e52")
    for xi, c in zip(x, cells):
        ax.text(xi + w / 2, 6.0, f"deficit\n{c['deficit']}", ha="center",
                va="center", fontsize=7.2, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k = {k}" for k in ks])
    ax.set_ylim(0, 14.5)
    ax.set_ylabel("channel uses per state packet")
    ax.set_title("(a) M3 cells: infeasible by design", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)

    ax = axes[1]
    all_k = [1, 2, 3, 12, 18, 24, 30, 54]
    feasible = [k >= 12 for k in all_k]
    b_by_k = {12: 1, 18: 2, 24: 3, 30: 4, 54: 8}
    xs = np.arange(len(all_k))
    for xi, k, ok in zip(xs, all_k, feasible):
        color = "#55a868" if ok else "#c44e52"
        marker = "o" if ok else "X"
        ax.scatter(xi, 0.0, s=150, color=color, marker=marker, zorder=3)
        if ok:
            ax.text(xi, 0.16, f"B = {b_by_k[k]}\n6B+6 = {6 * b_by_k[k] + 6}",
                    ha="center", va="bottom", fontsize=7.2, color="#2f6b3f")
        else:
            ax.text(xi, -0.16, "no packet\nfits", ha="center", va="top",
                    fontsize=7.2, color="#a11")
    ax.axvline(2.5, color="#777", ls="--", lw=1.0)
    ax.text(2.5, 0.62, "feasibility boundary", ha="center", fontsize=8,
            color="#555")
    ax.text(1.0, 0.82, "M3: structural\n(no measurement)", ha="center",
            fontsize=8, color="#a11")
    ax.text(5.5, 0.82, "M2: measured", ha="center", fontsize=8, color="#2f6b3f")
    ax.set_xlim(-0.7, len(all_k) - 0.3)
    ax.set_ylim(-0.55, 1.0)
    ax.set_yticks([])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"k = {k}" for k in all_k], rotation=0, fontsize=7.5)
    ax.set_title("(b) The digital packet exists only from k = 12", fontsize=9)
    ax.grid(False)

    fig.suptitle(
        "M3: the matched-budget digital baseline cannot exist at k = 1, 2, 3 "
        "(6B + 6 = 12 at B = 1)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG / "fig_m3_infeasibility.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def _m2_pooled_0db():
    """M2 pooled success at 0 dB for digital / reconstruction, per k."""
    d = _load(ROOT / "results" / "M2" / "step11_m2_summary.json")
    out = {}
    for r in d["pooled_rows"]:
        if r["channel"] == "awgn" and r["snr_db"] == 0.0:
            out[(r["method"], r["k"])] = r["success_rate_pooled"]
    return out


def m4_bandwidth_reference():
    d = _load(ROOT / "results" / "M4" / "step13_m4_summary.json")
    m2 = _m2_pooled_0db()

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    ax = axes[0]
    style = {"awgn": ("#1f77b4", "o", "AWGN"),
             "rayleigh": ("#d62728", "s", "Rayleigh")}
    for ch, (col, mk, label) in style.items():
        rows = [c for c in d["conditions"] if c["channel"] == ch]
        snr = [c["snr_db"] for c in rows]
        sr = [c["success_rate"] for c in rows]
        ax.plot(snr, sr, marker=mk, ms=4, lw=1.5, color=col,
                label=f"{label} (B = 8, k = 54)")
        v0 = sr[snr.index(0.0)]
        ax.annotate(f"{v0:.3f}", (0.0, v0), textcoords="offset points",
                    xytext=(-8, 8) if ch == "awgn" else (-8, -14),
                    ha="center", fontsize=7, color=col)
    ax.axhline(0.90, color="grey", ls=":", lw=0.9)
    ax.axhline(0.99, color="grey", ls=":", lw=0.9)
    ax.text(20.0, 0.905, "0.90", ha="right", va="bottom", fontsize=7,
            color="#555")
    ax.text(20.0, 0.995, "0.99", ha="right", va="bottom", fontsize=7,
            color="#555")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-1, 21)
    ax.set_title("(a) M4: B = 8 reference at k = 54, E = 5000", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    ax = axes[1]
    ks = [12, 18, 24, 30, 54]
    digital = [m2[("digital", k)] for k in ks]
    recon = [m2[("reconstruction", k)] for k in ks]
    lam = {18: 0.999, 54: 0.998}   # Wave-C, AWGN, 0 dB, E = 1000
    x = np.arange(len(ks))
    w = 0.26
    ax.bar(x - w, digital, w, label="digital (conv)", color="#1f77b4")
    ax.bar(x, recon, w, label="reconstruction", color="#8c8c8c")
    lam_vals = [lam.get(k, np.nan) for k in ks]
    ax.bar(x + w, lam_vals, w, label="lambda-heavy (Wave-C)", color="#c44e52")
    for xi, v in zip(x - w, digital):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8)
    for xi, v in zip(x + w, lam_vals):
        if not math.isnan(v):
            ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.8)
    ax.axhline(0.90, color="grey", ls=":", lw=1.0)
    ax.text(-0.45, 0.915, "0.90", ha="left", va="bottom", fontsize=7,
            color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k = {k}" for k in ks], fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("success rate at 0 dB (AWGN)")
    ax.set_title("(b) Success rate at 0 dB vs bandwidth", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=3)

    fig.suptitle(
        "M4: the separated B = 8 design reaches 0.974 at 0 dB only at k = 54; "
        "the lambda-shaped link reaches 0.999 at k = 18", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG / "fig_m4_bandwidth_reference.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def g3pp_channel_model():
    h = 50.0
    fc = 2.0
    d = np.logspace(1, math.log10(4000), 300)
    p_los = np.array([aerial.los_probability(x, h) for x in d])
    d3 = np.hypot(d, h)
    pl_los = np.array([aerial.los_path_loss_db(x, fc) for x in d3])
    pl_nlos = np.array([aerial.nlos_path_loss_db(x, h, fc) for x in d3])
    pl_mean = p_los * pl_los + (1 - p_los) * pl_nlos

    fig = plt.figure(figsize=(9.6, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.15])

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.plot([0, 1.0], [0, 0], color="#333", lw=1.2)
    ax.scatter([0.08], [0.05], s=55, color="#1f77b4", zorder=3)
    ax.text(0.08, -0.02, "ground\nstation", ha="center", va="top", fontsize=7)
    ax.scatter([0.78], [0.62], s=60, marker="^", color="#c44e52", zorder=3)
    ax.text(0.78, 0.67, "UAV", ha="center", fontsize=7.5)
    ax.plot([0.08, 0.78], [0.05, 0.62], color="#55a868", lw=1.4,
            label="LoS")
    ax.plot([0.08, 0.55, 0.78], [0.05, 0.02, 0.62], color="#d62728",
            lw=1.2, ls="--", label="NLoS (reflection)")
    ax.annotate("", xy=(0.78, 0.05), xytext=(0.08, 0.05),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=0.9))
    ax.text(0.43, 0.10, r"$d_{2D}$", ha="center", va="bottom", fontsize=8)
    ax.annotate("", xy=(0.80, 0.62), xytext=(0.80, 0.05),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=0.9))
    ax.text(0.82, 0.33, r"$h$", fontsize=8)
    ax.text(0.33, 0.39, r"$d_{3D}$", fontsize=8, rotation=36, color="#333")
    ax.text(0.50, 0.40, "LoS", fontsize=7, color="#2f6b3f")
    ax.text(0.31, 0.17, "NLoS\n(reflection)", fontsize=7, color="#a11")
    ax.text(0.5, 0.90, "P_LoS = 0.889 at\nd2D = 500 m, h = 50 m",
            ha="center", va="top", fontsize=7.2, color="#333")
    ax.set_title("(a) UMa-aerial geometry", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(d, p_los, color="#1f77b4", lw=1.6)
    ax.axvline(500, color="#777", ls=":", lw=1.0)
    ax.scatter([500], [aerial.los_probability(500, h)], color="#c44e52",
               zorder=3, s=30)
    ax.annotate("0.889", (500, aerial.los_probability(500, h)),
                textcoords="offset points", xytext=(6, -12), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("ground distance d2D (m)")
    ax.set_ylabel("LoS probability")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) TR 36.777 LoS probability", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(d, pl_los, color="#55a868", lw=1.5, label="LoS")
    ax.plot(d, pl_nlos, color="#d62728", lw=1.5, ls="--", label="NLoS")
    ax.plot(d, pl_mean, color="#1f77b4", lw=2.0, label="mean")
    ax.axvline(500, color="#777", ls=":", lw=1.0)
    ax.scatter([500], [aerial.expected_path_loss_db(500, h, fc)],
               color="#c44e52", zorder=3, s=30)
    ax.annotate("95.63 dB", (500, aerial.expected_path_loss_db(500, h, fc)),
                textcoords="offset points", xytext=(6, -14), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("ground distance d2D (m)")
    ax.set_ylabel("path loss (dB)")
    ax.set_title("(c) Path loss at fc = 2 GHz", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    fig.suptitle(
        "3GPP TR 36.777 / TR 38.901 UMa-aerial model used in A-10, A-12, A-13 "
        "(h = 50 m, fc = 2 GHz)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG / "fig_3gpp_channel_model.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def experiment_matrix():
    rows = [
        ("M1", "Task vs reconstruction at severe bandwidth",
         "k = 1, 2, 3; recon + task; AWGN, Rayleigh", "complete"),
        ("M2", "Matched-budget three-way comparison",
         "k = 12...54; digital + recon + task; AWGN, Rayleigh", "complete"),
        ("M3", "Digital infeasibility boundary",
         "k = 1, 2, 3; digital; structural", "complete"),
        ("M4", "Secondary B = 8 bandwidth reference",
         "k = 54; digital; AWGN, Rayleigh; E = 5000", "complete"),
        ("M5", "Task-loss removal, uncoded digital, cross-channel",
         "declared by the frozen protocol; no cell run", "deferred"),
        ("Wave-A", "Loss-component / lambda ablation",
         "k = 12; six arms; AWGN, Rayleigh; E = 1000", "complete*"),
        ("Wave-B", "CSI-estimation-error ablation",
         "all k; sigma = 0...0.4; 350 cells", "complete"),
        ("Wave-C", "lambda-shaping at other bandwidths",
         "k = 18, 54; lambda-heavy; AWGN, Rayleigh", "complete*"),
        ("A-9", "Capacity ablation (width 256)",
         "k = 18; frozen task arm; AWGN, Rayleigh", "complete"),
        ("A-10", "3GPP UMa-aerial link budget",
         "k = 18, 54; lambda-heavy + digital; E = 1000", "complete"),
        ("A-11", "Polar vs convolutional baseline",
         "all k; CRC-8 SC polar; AWGN, Rayleigh", "complete"),
        ("A-12", "Training under full 3GPP dispersion",
         "k = 18, 54; lambda-heavy; UMa-aerial", "negative"),
        ("A-13", "3GPP training, outage tail resampled",
         "k = 18, 54; lambda-heavy; UMa-aerial", "complete"),
        ("A-14", "Seed variance of the lambda arm at k = 24",
         "k = 24; seeds 45, 46, 47; AWGN, Rayleigh", "complete"),
        ("A-15", "Full 5G NR polar comparator",
         "all k; CRC-6/11, CA-SCL L = 8; AWGN, Rayleigh", "complete"),
    ]
    status_color = {
        "complete": "#dff0d8", "complete*": "#fcf8e3",
        "negative": "#f2dede", "deferred": "#e8e8e8",
    }
    status_text = {
        "complete": "complete", "complete*": "complete (documented gaps)",
        "negative": "complete (negative, kept)", "deferred": "deferred (not run)",
    }

    fig, ax = plt.subplots(figsize=(9.8, 5.3))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n = len(rows)
    top = 0.93
    bottom = 0.04
    row_h = (top - bottom) / n
    cols = [0.015, 0.115, 0.52, 0.805]
    headers = ["Block", "Question", "Design actually run", "Status"]
    for c, htxt in zip(cols, headers):
        ax.text(c, top + 0.015, htxt, fontsize=8, fontweight="bold",
                va="bottom", color="#222")
    ax.plot([0.0, 1.0], [top, top], color="#444", lw=1.0)
    for i, (block, question, design, status) in enumerate(rows):
        y = top - (i + 0.5) * row_h
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.0, y - row_h / 2), 1.0, row_h,
                                       facecolor="#fafafa", edgecolor="none",
                                       zorder=0))
        ax.text(cols[0], y, block, fontsize=7.8, va="center",
                fontweight="bold", color="#222")
        ax.text(cols[1], y, question, fontsize=7.2, va="center", color="#222")
        ax.text(cols[2], y, design, fontsize=6.7, va="center", color="#333")
        ax.add_patch(plt.Rectangle((cols[3] - 0.008, y - row_h * 0.32),
                                   0.198, row_h * 0.64,
                                   facecolor=status_color[status],
                                   edgecolor="#bbb", lw=0.5, zorder=1))
        ax.text(cols[3] + 0.091, y, status_text[status], fontsize=6.9,
                va="center", ha="center", color="#222", zorder=2)
    ax.plot([0.0, 1.0], [bottom, bottom], color="#444", lw=1.0)
    ax.text(0.0, bottom - 0.035,
            "M1-M4 complete; M5 deferred by the frozen protocol. "
            "Wave-A/B/C and A-9...A-15 are dated, additive amendments.",
            fontsize=7.0, color="#555", va="top")
    ax.set_title("Experiment matrix actually executed (M1-M5, Waves A-C, A-9...A-15)",
                 fontsize=9.5, pad=14)
    fig.tight_layout()
    out = FIG / "fig_experiment_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    m3_infeasibility()
    m4_bandwidth_reference()
    g3pp_channel_model()
    experiment_matrix()
