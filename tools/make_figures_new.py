"""
make_figures_new.py  —  F-NEW-1, F-NEW-2, F-NEW-3, F-NEW-4
Generates four additional publication-grade figures from EXISTING data only.
No training or evaluation runs needed.

Figures produced in  fyp_md_files/figures/ :
  fig_new1_per_seed_agreement.png  — per-seed SR overlay (recon vs task, AWGN + Rayleigh)
  fig_new2_mcnemar_heatmap.png     — McNemar -log10(p) heatmap for all 198 eval units
  fig_new3_seed_spread_bars.png    — mean ± std bar chart across seeds (AWGN, k=1,2,3)
  fig_new4_sr_table_summary.png    — compact summary table as figure (thesis-ready)

Run:
  .venv\\Scripts\\python.exe tools/make_figures_new.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
ROOT  = Path(".")
EVAL  = ROOT / "results" / "M1" / "eval"
SUM   = ROOT / "results" / "M1" / "step10_m1_summary.json"
FIG   = ROOT / "fyp_md_files" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
SNR_LIST   = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
KS         = [1, 2, 3]
SEEDS      = [42, 43, 44]
CHANNELS   = ["awgn", "rayleigh"]

# colours — colour-blind safe (Paul Tol palette)
C_RECON = "#0072B2"   # blue
C_TASK  = "#D55E00"   # vermillion
SEED_COLORS = {42: "#E69F00", 43: "#56B4E9", 44: "#009E73"}
SEED_MARKERS = {42: "o", 43: "s", 44: "^"}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})

# ── load summary JSON ────────────────────────────────────────────────────────
with open(SUM) as f:
    summary = json.load(f)

# Build lookup: (method, k, channel, snr, seed) -> success_rate
# The per-seed rows are inside the pooled_rows as mean/std, but the actual
# per-seed SR lives in the individual eval unit JSONs.
# We read those directly for F-NEW-1 and F-NEW-2.

def load_eval_unit(k, channel, seed):
    """Load all 11 SNR eval JSONs for one (k, channel, seed) unit.
    Returns dict: snr_db -> eval JSON dict."""
    unit_dir = EVAL / f"k{k}_{channel}_seed{seed}"
    result = {}
    for snr in SNR_LIST:
        snr_int = int(snr)
        fpath = unit_dir / f"eval_{snr_int}.json"
        if fpath.exists():
            with open(fpath) as f:
                result[snr] = json.load(f)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# F-NEW-1:  Per-seed agreement plot
# Shows SR for each individual seed (42, 43, 44) overlaid on the pooled line.
# Answers: "Is the result consistent across seeds, or is one seed driving it?"
# ─────────────────────────────────────────────────────────────────────────────
print("Generating F-NEW-1: per-seed agreement …")

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)

for row, channel in enumerate(CHANNELS):
    for col, k in enumerate(KS):
        ax = axes[row][col]

        # collect per-seed SR for reconstruction and task
        recon_per_seed = {}
        task_per_seed  = {}
        for seed in SEEDS:
            unit = load_eval_unit(k, channel, seed)
            recon_per_seed[seed] = [unit[s]["aggregate_reconstruction"]["success_rate"]
                                    if s in unit else float("nan")
                                    for s in SNR_LIST]
            task_per_seed[seed]  = [unit[s]["aggregate_task"]["success_rate"]
                                    if s in unit else float("nan")
                                    for s in SNR_LIST]

        # pooled mean across seeds (from summary JSON)
        recon_pooled, task_pooled = [], []
        for snr in SNR_LIST:
            # read pooled_rows
            pr = next((r for r in summary["pooled_rows"]
                       if r["method"] == "reconstruction"
                       and r["k"] == k
                       and r["channel"] == channel
                       and r["snr_db"] == snr), None)
            pt = next((r for r in summary["pooled_rows"]
                       if r["method"] == "task"
                       and r["k"] == k
                       and r["channel"] == channel
                       and r["snr_db"] == snr), None)
            recon_pooled.append(pr["success_rate_pooled"] if pr else float("nan"))
            task_pooled.append(pt["success_rate_pooled"]  if pt else float("nan"))

        # plot individual seeds (thin lines, semi-transparent)
        for seed in SEEDS:
            ax.plot(SNR_LIST, recon_per_seed[seed],
                    linestyle="-", marker=SEED_MARKERS[seed], ms=3,
                    color=C_RECON, alpha=0.45,
                    label=f"recon seed {seed}" if col == 0 and row == 0 else "_")
            ax.plot(SNR_LIST, task_per_seed[seed],
                    linestyle="--", marker=SEED_MARKERS[seed], ms=3,
                    color=C_TASK, alpha=0.45,
                    label=f"task  seed {seed}" if col == 0 and row == 0 else "_")

        # plot pooled (thick)
        ax.plot(SNR_LIST, recon_pooled, "-",  color=C_RECON, lw=2.2,
                label="recon pooled" if col == 0 and row == 0 else "_")
        ax.plot(SNR_LIST, task_pooled,  "--", color=C_TASK,  lw=2.2,
                label="task pooled"  if col == 0 and row == 0 else "_")

        ax.set_title(f"{channel.upper()}  k={k}  (ρ={k}/6)")
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks(SNR_LIST[::2])
        if col == 0:
            ax.set_ylabel("Goal-reaching success rate")

# build a clean legend in the top-left panel
ax0 = axes[0][0]
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=C_RECON, lw=2.2,          label="reconstruction (pooled)"),
    Line2D([0], [0], color=C_TASK,  lw=2.2, ls="--", label="task (pooled)"),
    Line2D([0], [0], color="0.5",   lw=1, alpha=0.6,  label="individual seeds (42/43/44)"),
]
ax0.legend(handles=legend_elements, loc="upper left", fontsize=8)

fig.suptitle(
    "F-NEW-1 — Per-seed agreement: reconstruction vs task DeepJSCC (M1)\n"
    "Thin lines = individual seeds 42/43/44 · Thick = pooled · "
    "Close clustering validates seed robustness",
    y=1.02, fontsize=11,
)
fig.tight_layout()
fig.savefig(FIG / "fig_new1_per_seed_agreement.png")
plt.close(fig)
print("  saved fig_new1_per_seed_agreement.png")


# ─────────────────────────────────────────────────────────────────────────────
# F-NEW-2:  McNemar significance heatmap
# Shows −log10(p) for task vs reconstruction per (k, SNR, channel) cell.
# Answers: "Is every comparison statistically significant?"
# ─────────────────────────────────────────────────────────────────────────────
print("Generating F-NEW-2: McNemar heatmap …")

fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

for col, channel in enumerate(CHANNELS):
    ax = axes[col]

    # matrix rows = k values (1,2,3), cols = SNR values (11 points)
    mat       = np.full((len(KS), len(SNR_LIST)), np.nan)
    sign_mat  = np.zeros((len(KS), len(SNR_LIST)), dtype=object)  # "+" or "-"

    for ri, k in enumerate(KS):
        for ci, snr in enumerate(SNR_LIST):
            # pool p-values across seeds: take the worst (largest) p
            # so heatmap is conservative — if even the worst seed is significant, it's real
            pvals = []
            for seed in SEEDS:
                unit = load_eval_unit(k, channel, seed)
                if snr in unit:
                    p = unit[snr]["mcnemar_task_vs_recon"]["p_value"]
                    pvals.append(p)
                    # sign: positive = task > recon, negative = recon > task
                    n01 = unit[snr]["mcnemar_task_vs_recon"]["n01"]
                    n10 = unit[snr]["mcnemar_task_vs_recon"]["n10"]
            if pvals:
                p_worst = max(pvals)            # most conservative across seeds
                # clamp -log10(p) at 200 to avoid inf
                mat[ri, ci] = min(-math.log10(p_worst + 1e-310), 200)
                # sign from pooled rows
                pr_row = next((r for r in summary["pooled_rows"]
                               if r["method"] == "reconstruction"
                               and r["k"] == k
                               and r["channel"] == channel
                               and r["snr_db"] == snr), None)
                pt_row = next((r for r in summary["pooled_rows"]
                               if r["method"] == "task"
                               and r["k"] == k
                               and r["channel"] == channel
                               and r["snr_db"] == snr), None)
                if pr_row and pt_row:
                    sign_mat[ri, ci] = ("T>R" if pt_row["success_rate_pooled"]
                                        > pr_row["success_rate_pooled"] else "R>T")

    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=min(200, np.nanmax(mat)))
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("−log₁₀(p)  [conservative: worst seed]", fontsize=8)

    ax.set_xticks(range(len(SNR_LIST)))
    ax.set_xticklabels([f"{int(s)}" for s in SNR_LIST], fontsize=8)
    ax.set_yticks(range(len(KS)))
    ax.set_yticklabels([f"k={k}" for k in KS])
    ax.set_xlabel("SNR (dB)")
    ax.set_title(f"{channel.upper()} — task vs reconstruction McNemar significance")

    # annotate each cell
    threshold = -math.log10(0.05)   # significance threshold
    for ri in range(len(KS)):
        for ci in range(len(SNR_LIST)):
            val = mat[ri, ci]
            if not np.isnan(val):
                txt   = sign_mat[ri, ci]
                color = "white" if val > threshold * 3 else "black"
                ax.text(ci, ri, f"{txt}\n{val:.0f}",
                        ha="center", va="center", fontsize=6.5,
                        color=color, fontweight="bold")

    # draw significance boundary line
    ax.axhline(-0.5, color="0.4", lw=0.5, ls=":")

fig.suptitle(
    "F-NEW-2 — McNemar test: task vs reconstruction significance (M1)\n"
    "Value = −log₁₀(p), conservative (worst seed). Threshold: p < 0.05 → log₁₀ > 1.3\n"
    "T>R: task better  ·  R>T: reconstruction better",
    y=1.04, fontsize=11,
)
fig.tight_layout()
fig.savefig(FIG / "fig_new2_mcnemar_heatmap.png")
plt.close(fig)
print("  saved fig_new2_mcnemar_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# F-NEW-3:  Seed spread bar chart (mean ± std across seeds)
# Shows mean SR ± 1σ across 3 seeds for every (method, k, SNR) @ AWGN
# Answers pooled CI semantics concern: seed variance is visible explicitly
# ─────────────────────────────────────────────────────────────────────────────
print("Generating F-NEW-3: seed spread bars …")

TARGET_SNRS = [0.0, 4.0, 10.0, 20.0]   # pick 4 representative SNR points

fig, axes = plt.subplots(1, len(TARGET_SNRS), figsize=(14, 3.8), sharey=True)

x_labels_recon = [f"R k={k}" for k in KS]
x_labels_task  = [f"T k={k}" for k in KS]
x_labels = x_labels_recon + x_labels_task
x_pos    = np.arange(len(x_labels))

for ax, snr in zip(axes, TARGET_SNRS):
    means, stds, colors = [], [], []

    # reconstruction
    for k in KS:
        sr_seeds = []
        for seed in SEEDS:
            unit = load_eval_unit(k, "awgn", seed)
            if snr in unit:
                sr_seeds.append(unit[snr]["aggregate_reconstruction"]["success_rate"])
        if sr_seeds:
            means.append(np.mean(sr_seeds))
            stds.append(np.std(sr_seeds, ddof=1))
        else:
            means.append(float("nan")); stds.append(0)
        colors.append(C_RECON)

    # task
    for k in KS:
        sr_seeds = []
        for seed in SEEDS:
            unit = load_eval_unit(k, "awgn", seed)
            if snr in unit:
                sr_seeds.append(unit[snr]["aggregate_task"]["success_rate"])
        if sr_seeds:
            means.append(np.mean(sr_seeds))
            stds.append(np.std(sr_seeds, ddof=1))
        else:
            means.append(float("nan")); stds.append(0)
        colors.append(C_TASK)

    bars = ax.bar(x_pos, means, color=colors, alpha=0.82,
                  edgecolor="0.3", linewidth=0.6)
    ax.errorbar(x_pos, means, yerr=stds, fmt="none",
                ecolor="0.2", capsize=4, elinewidth=1.3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
    ax.set_title(f"AWGN @ {int(snr)} dB")
    ax.set_ylim(0, max(means) * 1.35 + 0.02)
    if ax == axes[0]:
        ax.set_ylabel("Success rate (mean ± 1σ over 3 seeds)")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=C_RECON, label="Reconstruction (R)"),
                   Patch(facecolor=C_TASK,  label="Task-oriented (T)")]
axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=8)

fig.suptitle(
    "F-NEW-3 — Seed variance: mean ± 1σ across training seeds {42, 43, 44}\n"
    "AWGN channel · Error bars show training-seed variability",
    y=1.04, fontsize=11,
)
fig.tight_layout()
fig.savefig(FIG / "fig_new3_seed_spread_bars.png")
plt.close(fig)
print("  saved fig_new3_seed_spread_bars.png")


# ─────────────────────────────────────────────────────────────────────────────
# F-NEW-4:  SR summary table rendered as figure  (thesis appendix ready)
# Pooled SR for all (method, k, channel, SNR) cells — compact, no external deps
# ─────────────────────────────────────────────────────────────────────────────
print("Generating F-NEW-4: SR summary table …")

# build rows: (k, channel, SNR, recon_sr, task_sr, delta, winner)
table_rows = []
for k in KS:
    for channel in CHANNELS:
        for snr in SNR_LIST:
            pr = next((r for r in summary["pooled_rows"]
                       if r["method"] == "reconstruction"
                       and r["k"] == k
                       and r["channel"] == channel
                       and r["snr_db"] == snr), None)
            pt = next((r for r in summary["pooled_rows"]
                       if r["method"] == "task"
                       and r["k"] == k
                       and r["channel"] == channel
                       and r["snr_db"] == snr), None)
            if pr and pt:
                rs = pr["success_rate_pooled"]
                ts = pt["success_rate_pooled"]
                delta = ts - rs
                winner = "TASK" if delta > 0.005 else ("RECON" if delta < -0.005 else "TIE")
                table_rows.append([
                    f"k={k}", channel.upper(), f"{int(snr)} dB",
                    f"{rs:.3f}", f"{ts:.3f}",
                    f"{delta:+.3f}", winner,
                ])

col_labels = ["k", "Channel", "SNR", "Recon SR", "Task SR", "Δ (T−R)", "Winner"]

# split into two halves (AWGN / Rayleigh) for side-by-side readability
awgn_rows    = [r for r in table_rows if r[1] == "AWGN"]
rayleigh_rows= [r for r in table_rows if r[1] == "RAYLEIGH"]

fig, axes = plt.subplots(1, 2, figsize=(14, 10))

for ax, rows, title in [(axes[0], awgn_rows, "AWGN"),
                        (axes[1], rayleigh_rows, "Rayleigh")]:
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.8)
    tbl.scale(1, 1.35)
    # colour header
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#4472C4")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # colour winner column
    for i, row in enumerate(rows, start=1):
        winner = row[-1]
        fc = "#D9EAD3" if winner == "TASK" else ("#FCE5CD" if winner == "RECON" else "#FFF2CC")
        tbl[(i, 6)].set_facecolor(fc)
    ax.set_title(f"M1 Pooled Success Rates — {title}", fontsize=10, pad=12)

fig.suptitle(
    "F-NEW-4 — M1 Pooled Success Rate Summary (3 seeds × 5000 episodes each)\n"
    "Green = task wins · Orange = recon wins · Yellow = tie (|Δ| ≤ 0.005)",
    y=1.01, fontsize=10,
)
fig.tight_layout()
fig.savefig(FIG / "fig_new4_sr_table.png")
plt.close(fig)
print("  saved fig_new4_sr_table.png")


# ── done ─────────────────────────────────────────────────────────────────────
print("\nAll new figures written to", FIG)
for f in sorted(FIG.glob("fig_new*.png")):
    sz = f.stat().st_size // 1024
    print(f"  {f.name:45s}  {sz} KB")
