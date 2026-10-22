"""
make_figures_waveb.py  —  Wave-B CSI robustness figures
Generates two publication-grade figures from Wave-B eval JSONs.
No training or re-evaluation needed — reads existing data only.

Figures produced in  fyp_md_files/figures/ :
  fig_waveb1_sr_vs_k_csi_bands.png   — SR vs k with CSI error bands
  fig_waveb2_mse_vs_csi_sigma.png    — MSE vs sigma_csi per method

Run:
  .venv\\Scripts\\python.exe tools/make_figures_waveb.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────
ROOT  = Path(".")
WBSUM = ROOT / "results" / "WAVE_B" / "step_wave_b_summary.json"
WBEVAL= ROOT / "results" / "WAVE_B" / "eval"
FIG   = ROOT / "fyp_md_files" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────
KS       = [12, 18, 24, 30, 54]
SIGMAS   = [0.0, 0.05, 0.1, 0.2, 0.4]
SNR_LIST = [0.0, 5.0, 10.0, 15.0, 20.0]
SEEDS    = [42, 43, 44]
METHODS  = ["reconstruction", "task", "digital"]

# colour-blind safe palette
C_RECON  = "#0072B2"
C_TASK   = "#D55E00"
C_DIG    = "#009E73"
METHOD_COLOR = {"reconstruction": C_RECON, "task": C_TASK, "digital": C_DIG}
METHOD_LABEL = {"reconstruction": "Reconstruction JSCC",
                "task":           "Task-Oriented JSCC",
                "digital":        "Digital (matched budget)"}

# sigma display names
SIG_NAME = {0.0: "σ=0 (perfect CSI)", 0.05: "σ=0.05",
            0.1: "σ=0.1", 0.2: "σ=0.2", 0.4: "σ=0.4"}
SIG_ALPHA = {0.0: 1.0, 0.05: 0.55, 0.1: 0.55, 0.2: 0.75, 0.4: 1.0}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})

# ── load summary rows ──────────────────────────────────────────────────────
with open(WBSUM) as f:
    wb = json.load(f)

rows = wb["summary_rows"]   # list of dicts

def get_sr(method, k, snr, sigma):
    """Pool across seeds from summary_rows."""
    matches = [r for r in rows
               if r["method"] == method
               and r["k"] == k
               and r["snr_db"] == snr
               and abs(r["csi_error_sigma"] - sigma) < 1e-6]
    if not matches:
        return float("nan")
    # summary rows are already pooled across seeds
    return matches[0]["success_rate_pooled"]

def load_mse(k, sigma, seed):
    """Read mse_mean per (k, sigma, seed) averaged over SNR grid."""
    unit_dir = WBEVAL / f"k{k}_rayleigh_seed{seed}"
    sig_str  = {0.0: "csi0", 0.05: "csi0p05",
                0.1: "csi0p1", 0.2: "csi0p2", 0.4: "csi0p4"}[sigma]
    mses = []
    for snr in SNR_LIST:
        fpath = unit_dir / f"eval_{int(snr)}_{sig_str}.json"
        if fpath.exists():
            with open(fpath) as f:
                d = json.load(f)
            for m in METHODS:
                key = f"aggregate_{m}"
                if key in d:
                    mses.append((m, d[key]["mse_mean"]))
    # group by method
    result = {}
    for m in METHODS:
        vals = [v for (mm, v) in mses if mm == m]
        result[m] = np.mean(vals) if vals else float("nan")
    return result


# ══════════════════════════════════════════════════════════════════════════
# F-WAVE-B-1:  SR vs k  with CSI error bands
# Shows how success rate changes with bandwidth for σ=0, σ=0.2, σ=0.4
# One panel per method — easy to read, thesis-grade
# ══════════════════════════════════════════════════════════════════════════
print("Generating F-WAVE-B-1: SR vs k with CSI bands …")

SHOW_SIGMAS = [0.0, 0.2, 0.4]   # three representative levels
SNR_REF     = 10.0               # representative SNR

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)

for ax, method in zip(axes, METHODS):
    color = METHOD_COLOR[method]

    for sigma in SHOW_SIGMAS:
        sr_vals = [get_sr(method, k, SNR_REF, sigma) for k in KS]
        lw   = 2.4 if sigma in (0.0, 0.4) else 1.2
        ls   = "-"  if sigma == 0.0 else ("--" if sigma == 0.2 else ":")
        alph = SIG_ALPHA[sigma]
        label= SIG_NAME[sigma]
        ax.plot(KS, sr_vals, linestyle=ls, linewidth=lw,
                marker="o", ms=5, color=color, alpha=alph, label=label)

    # shade between sigma=0 and sigma=0.4
    sr_lo = [get_sr(method, k, SNR_REF, 0.0) for k in KS]
    sr_hi = [get_sr(method, k, SNR_REF, 0.4) for k in KS]
    ax.fill_between(KS, sr_lo, sr_hi, color=color, alpha=0.10,
                    label="σ=0–0.4 band")

    ax.set_title(METHOD_LABEL[method])
    ax.set_xlabel("k (channel uses per state)")
    ax.set_xticks(KS)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left" if method == "digital" else "upper right",
              fontsize=8)
    if ax == axes[0]:
        ax.set_ylabel(f"Success rate @ {int(SNR_REF)} dB SNR")

# add crossover annotation on digital panel
axes[2].axvline(x=21, color="0.4", lw=1.0, ls=":")
axes[2].text(21.5, 0.55, "approx.\ncrossover\n(k≈18–24)",
             fontsize=7.5, color="0.35", va="center")

fig.suptitle(
    "F-WAVE-B-1 — CSI error robustness: success rate vs bandwidth\n"
    f"Rayleigh fading, SNR={int(SNR_REF)} dB, E=5000 episodes/cell (pooled 3 seeds)\n"
    "Solid=σ=0 (perfect CSI) · Dashed=σ=0.2 · Dotted=σ=0.4 · Band=full range",
    y=1.04, fontsize=10,
)
fig.tight_layout()
fig.savefig(FIG / "fig_waveb1_sr_vs_k_csi_bands.png")
plt.close(fig)
print("  saved fig_waveb1_sr_vs_k_csi_bands.png")


# ══════════════════════════════════════════════════════════════════════════
# F-WAVE-B-2:  MSE vs sigma_csi  (per method, per k)
# Shows the ~18% consistent MSE penalty — the key Wave-B quantitative finding
# ══════════════════════════════════════════════════════════════════════════
print("Generating F-WAVE-B-2: MSE vs CSI sigma …")

fig, axes = plt.subplots(1, len(KS), figsize=(14, 3.8), sharey=False)

for ax, k in zip(axes, KS):
    for method in METHODS:
        color = METHOD_COLOR[method]
        mse_vals = []
        for sigma in SIGMAS:
            # average across seeds
            mse_seeds = []
            for seed in SEEDS:
                unit_dir = WBEVAL / f"k{k}_rayleigh_seed{seed}"
                if not unit_dir.exists():
                    continue
                sig_str = {0.0: "csi0", 0.05: "csi0p05",
                           0.1: "csi0p1", 0.2: "csi0p2",
                           0.4: "csi0p4"}[sigma]
                snr_mses = []
                for snr in SNR_LIST:
                    fpath = unit_dir / f"eval_{int(snr)}_{sig_str}.json"
                    if fpath.exists():
                        with open(fpath) as f:
                            d = json.load(f)
                        key = f"aggregate_{method}"
                        if key in d:
                            snr_mses.append(d[key]["mse_mean"])
                if snr_mses:
                    mse_seeds.append(np.mean(snr_mses))
            mse_vals.append(np.mean(mse_seeds) if mse_seeds else float("nan"))

        ax.plot(SIGMAS, mse_vals, "-o", color=color, ms=5, lw=1.8,
                label=METHOD_LABEL[method])

        # annotate % change from sigma=0 to sigma=0.4
        if not (np.isnan(mse_vals[0]) or np.isnan(mse_vals[-1])):
            pct = (mse_vals[-1] - mse_vals[0]) / mse_vals[0] * 100
            ax.annotate(f"{pct:+.0f}%",
                        xy=(0.4, mse_vals[-1]),
                        xytext=(0.32, mse_vals[-1] * 1.12),
                        fontsize=7, color=color,
                        arrowprops=dict(arrowstyle="->",
                                        color=color, lw=0.8))

    ax.set_title(f"k={k}  (ρ={k}/6)")
    ax.set_xlabel("CSI error σ")
    ax.set_xticks(SIGMAS)
    ax.set_xticklabels([str(s) for s in SIGMAS], fontsize=8)
    if ax == axes[0]:
        ax.set_ylabel("Trajectory MSE\n(mean over SNR grid, 3 seeds)")

axes[0].legend(loc="upper left", fontsize=7.5)

fig.suptitle(
    "F-WAVE-B-2 — MSE vs CSI estimation error σ (Rayleigh, all k values)\n"
    "Annotation = % change from σ=0 to σ=0.4 · ~18% consistent penalty across k",
    y=1.04, fontsize=10,
)
fig.tight_layout()
fig.savefig(FIG / "fig_waveb2_mse_vs_csi_sigma.png")
plt.close(fig)
print("  saved fig_waveb2_mse_vs_csi_sigma.png")


# ══════════════════════════════════════════════════════════════════════════
# F-WAVE-B-3:  SR vs SNR at fixed k=12, all sigma levels — one panel
# Shows that SR is flat across SNR AND across sigma for neural methods
# ══════════════════════════════════════════════════════════════════════════
print("Generating F-WAVE-B-3: SR vs SNR all sigma levels (k=12) …")

K_REF = 12
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)

for ax, method in zip(axes, METHODS):
    color = METHOD_COLOR[method]
    for sigma in SIGMAS:
        sr_vals = [get_sr(method, K_REF, snr, sigma) for snr in SNR_LIST]
        lw    = 2.2 if sigma in (0.0, 0.4) else 1.0
        ls    = "-"  if sigma == 0.0 else ("--" if sigma == 0.4 else "-")
        alph  = 0.5 + 0.5 * (sigma / 0.4)
        label = SIG_NAME[sigma]
        ax.plot(SNR_LIST, sr_vals, linestyle=ls, linewidth=lw,
                marker="o", ms=4, color=color, alpha=alph, label=label)

    ax.set_title(METHOD_LABEL[method])
    ax.set_xlabel("SNR (dB)")
    ax.set_xticks(SNR_LIST)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=7.5)
    if ax == axes[0]:
        ax.set_ylabel(f"Success rate (k={K_REF}, Rayleigh)")

fig.suptitle(
    f"F-WAVE-B-3 — SR vs SNR at k={K_REF}, all CSI error levels\n"
    "Neural methods: flat across both SNR and σ · "
    "Digital: mildly increases with σ (anomaly — see note)",
    y=1.04, fontsize=10,
)
fig.tight_layout()
fig.savefig(FIG / "fig_waveb3_sr_vs_snr_csi.png")
plt.close(fig)
print("  saved fig_waveb3_sr_vs_snr_csi.png")


# ── done ──────────────────────────────────────────────────────────────────
print("\nAll Wave-B figures written to", FIG)
for f in sorted(FIG.glob("fig_waveb*.png")):
    sz = f.stat().st_size // 1024
    print(f"  {f.name:48s}  {sz} KB")
