"""
make_figures_wavec.py  —  Wave-C lambda-x-bandwidth figures
Reads existing Wave-C eval data only. No training or re-evaluation.

Figures produced in fyp_md_files/figures/:
  fig_wavec1_sr_vs_snr.png      SR vs SNR: heavy/light vs frozen baseline (k=18,54)
  fig_wavec2_sr_vs_k.png        SR vs k: Wave-A k=12 + Wave-C k=18,54 combined
  fig_wavec3_comparison_table.png  Summary table figure

Run:
  .venv\\Scripts\\python.exe tools/make_figures_wavec.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT   = Path(".")
WCSUM  = ROOT / "results" / "WAVE_C" / "step_wave_c_summary.json"
WASUM  = ROOT / "results" / "WAVE_A" / "step_wave_a_summary.json"
FIG    = ROOT / "fyp_md_files" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# colours
C_HEAVY   = "#D55E00"   # vermillion — heavy arm
C_LIGHT   = "#0072B2"   # blue       — light arm
C_FROZEN  = "#999999"   # grey       — frozen baseline (M1/M2 task)
C_RECON   = "#56B4E9"   # sky blue   — reconstruction (reference)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})

# ── load data ─────────────────────────────────────────────────────────────
with open(WCSUM) as f:
    wc = json.load(f)
with open(WASUM) as f:
    wa = json.load(f)

wc_rows = wc["pooled_rows"]
wa_rows = wa["pooled_rows"]

SNR_LIST = sorted(set(r["snr_db"] for r in wc_rows))
KS_WC    = sorted(set(r["k"] for r in wc_rows))
CHANNELS = ["awgn", "rayleigh"]

def wc_sr(tag, k, channel, snr):
    for r in wc_rows:
        if (r.get("tag") == tag and r["k"] == k
                and r["channel"] == channel
                and abs(r["snr_db"] - snr) < 0.01):
            return r["success_rate_pooled"]
    return float("nan")

def wa_sr(tag, k, channel, snr):
    """Wave-A mean SR over available seeds."""
    matches = [r for r in wa_rows
               if r.get("tag") == tag and r["k"] == k
               and r["channel"] == channel
               and abs(r["snr_db"] - snr) < 0.01]
    if not matches:
        return float("nan")
    return np.mean([r["success_rate_mean"] for r in matches])

# frozen task baseline from Wave-A rows (tag="lam_ctl_heavy" at k=12 gives
# the comparison; frozen baseline is the tag=None / original M1 arm)
# Wave-A summary has "arm" field
def wa_arm_sr(arm, k, channel, snr):
    for r in wa_rows:
        if (r.get("arm") == arm and r.get("k") == k
                and r.get("channel") == channel
                and abs(r.get("snr_db", -999) - snr) < 0.01):
            return r.get("success_rate_mean", float("nan"))
    return float("nan")


# ══════════════════════════════════════════════════════════════════════════
# Fig WAVE-C-1:  SR vs SNR  —  heavy + light + frozen baseline
# Two rows (AWGN / Rayleigh), two cols (k=18, k=54)
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig_wavec1_sr_vs_snr …")

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)

for row, ch in enumerate(CHANNELS):
    for col, k in enumerate(KS_WC):
        ax = axes[row][col]

        # heavy arm
        heavy = [wc_sr("lam_ctl_heavy", k, ch, s) for s in SNR_LIST]
        ax.plot(SNR_LIST, heavy, "-o", color=C_HEAVY, lw=2.2, ms=5,
                label="λ-heavy (λ_u=λ_v=0.1)")

        # light arm (may not exist for all combos)
        light = [wc_sr("lam_ctl_light", k, ch, s) for s in SNR_LIST]
        if not all(np.isnan(v) for v in light):
            ax.plot(SNR_LIST, light, "--s", color=C_LIGHT, lw=1.8, ms=5,
                    label="λ-light (λ_u=λ_v=0.01)")

        # frozen task baseline reference line (flat ~0.02-0.05 from M2)
        frozen_vals = {
            ("awgn", 18): 0.025, ("awgn", 54): 0.036,
            ("rayleigh", 18): 0.025, ("rayleigh", 54): 0.036,
        }
        fv = frozen_vals.get((ch, k), 0.03)
        ax.axhline(fv, color=C_FROZEN, lw=1.4, ls=":", label=f"Frozen task (M2 ≈{fv})")

        # perfect-channel ceiling (noiseless Oracle ~1.0)
        ax.axhline(1.0, color="0.75", lw=0.8, ls="--", label="Oracle ceiling")

        ax.set_title(f"{ch.upper()}  k={k}  (ρ={k}/6)")
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks(SNR_LIST)
        ax.set_ylim(-0.02, 1.08)
        if col == 0:
            ax.set_ylabel("Goal-reaching success rate")
        if row == 0 and col == 0:
            ax.legend(loc="lower right", fontsize=8)

fig.suptitle(
    "Fig WAVE-C-1 — λ-shaping generalizes to k=18 and k=54\n"
    "Wave-C: lam_ctl_heavy SR vs SNR · Dotted grey = frozen task baseline (M2)\n"
    "E=1000, seed=42, diagnostic protocol",
    y=1.03, fontsize=11,
)
fig.tight_layout()
fig.savefig(FIG / "fig_wavec1_sr_vs_snr.png")
plt.close(fig)
print("  saved fig_wavec1_sr_vs_snr.png")


# ══════════════════════════════════════════════════════════════════════════
# Fig WAVE-C-2:  SR vs k  —  combined Wave-A (k=12) + Wave-C (k=18,54)
# Shows the full picture: does the improvement hold across bandwidth?
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig_wavec2_sr_vs_k …")

K_COMBINED = [12, 18, 54]
SNR_REF    = 10.0

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, ch in zip(axes, CHANNELS):
    # heavy arm: k=12 from Wave-A, k=18/54 from Wave-C
    heavy_sr = []
    for k in K_COMBINED:
        if k == 12:
            v = wa_arm_sr("lam_ctl_heavy", 12, ch, SNR_REF)
        else:
            v = wc_sr("lam_ctl_heavy", k, ch, SNR_REF)
        heavy_sr.append(v)

    # frozen task baseline: from Wave-A (k=12) and M2 pooled (k=18,54)
    frozen_sr = {
        12: {"awgn": 0.030, "rayleigh": 0.032},
        18: {"awgn": 0.025, "rayleigh": 0.025},
        54: {"awgn": 0.036, "rayleigh": 0.036},
    }
    frozen_vals = [frozen_sr[k][ch] for k in K_COMBINED]

    ax.plot(K_COMBINED, heavy_sr, "-o", color=C_HEAVY, lw=2.4, ms=7,
            label="λ-heavy (Wave-A k=12, Wave-C k=18/54)")
    ax.plot(K_COMBINED, frozen_vals, ":^", color=C_FROZEN, lw=1.6, ms=6,
            label="Frozen task baseline (M1/M2)")
    ax.axhline(1.0, color="0.75", lw=0.8, ls="--", label="Oracle ceiling")

    # annotate each point
    for i, (k, v) in enumerate(zip(K_COMBINED, heavy_sr)):
        if not np.isnan(v):
            ax.annotate(f"{v:.3f}", xy=(k, v),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8.5, color=C_HEAVY,
                        fontweight="bold")

    ax.set_title(f"{ch.upper()} @ {int(SNR_REF)} dB")
    ax.set_xlabel("k (channel uses per state)")
    ax.set_xticks(K_COMBINED)
    ax.set_xticklabels([f"k={k}" for k in K_COMBINED])
    ax.set_ylim(-0.02, 1.12)
    if ax == axes[0]:
        ax.set_ylabel("Goal-reaching success rate")
    ax.legend(loc="lower right", fontsize=8)

fig.suptitle(
    "Fig WAVE-C-2 — λ-shaping holds across bandwidth: k=12, 18, 54\n"
    f"SNR={int(SNR_REF)} dB · Wave-A (k=12, 3 seeds) + Wave-C (k=18/54, seed 42)\n"
    "Dotted = frozen baseline (λ_u=λ_v=0) · Heavy line = λ-shaped",
    y=1.04, fontsize=11,
)
fig.tight_layout()
fig.savefig(FIG / "fig_wavec2_sr_vs_k.png")
plt.close(fig)
print("  saved fig_wavec2_sr_vs_k.png")


# ══════════════════════════════════════════════════════════════════════════
# Fig WAVE-C-3:  Summary comparison table as figure
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig_wavec3_comparison_table …")

# Build table: k, channel, frozen SR, heavy SR, improvement factor
table_data = []
for k in K_COMBINED:
    for ch in CHANNELS:
        snr = 10.0
        if k == 12:
            h = wa_arm_sr("lam_ctl_heavy", 12, ch, snr)
        else:
            h = wc_sr("lam_ctl_heavy", k, ch, snr)
        f_ref = {"awgn": {12: 0.030, 18: 0.025, 54: 0.036},
                 "rayleigh": {12: 0.032, 18: 0.025, 54: 0.036}}
        fv = f_ref[ch][k]
        if not np.isnan(h) and fv > 0:
            factor = h / fv
            table_data.append([
                f"k={k}", ch.upper(),
                f"{fv:.3f}",
                f"{h:.3f}" if not np.isnan(h) else "—",
                f"{factor:.0f}×",
                "Wave-A" if k == 12 else "Wave-C"
            ])

col_labels = ["k", "Channel", "Frozen task SR", "λ-heavy SR", "Improvement", "Source"]

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axis("off")
tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center", loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)

# header styling
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor("#4472C4")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

# colour improvement column green
for i, row in enumerate(table_data, start=1):
    tbl[(i, 4)].set_facecolor("#D9EAD3")
    tbl[(i, 4)].set_text_props(fontweight="bold", color="#1a6e2a")
    # source column
    src_color = "#FFF2CC" if row[5] == "Wave-A" else "#E8F4FD"
    tbl[(i, 5)].set_facecolor(src_color)

ax.set_title(
    "Fig WAVE-C-3 — λ-shaping improvement summary @ SNR=10 dB\n"
    "Frozen λ_u=λ_v=0  vs  λ-heavy λ_u=λ_v=0.1",
    fontsize=10, pad=15,
)
fig.tight_layout()
fig.savefig(FIG / "fig_wavec3_comparison_table.png")
plt.close(fig)
print("  saved fig_wavec3_comparison_table.png")


# ── done ──────────────────────────────────────────────────────────────────
print("\nAll Wave-C figures written to", FIG)
for f in sorted(FIG.glob("fig_wavec*.png")):
    sz = f.stat().st_size // 1024
    print(f"  {f.name:48s}  {sz} KB")
