"""Wave-A / Wave-B figures generated from EXISTING artifacts only.

Reads (read-only):
  results/WAVE_B/eval/*/eval_*.json   (350 cells, 3 methods, sigma sweep)
  results/WAVE_A/eval/*/*.json        (85 slots, arm comparison, E=1000)

Writes:
  fyp_md_files/figures/fig_wave_b_sr_vs_k.png
  fyp_md_files/figures/fig_wave_b_sr_vs_snr.png
  fyp_md_files/figures/fig_wave_b_csi_sweep.png
  fyp_md_files/figures/fig_wave_a_arms.png

No training, no evaluation, no writes under results/.
Run:  .venv\\Scripts\\python.exe tools/make_figures_wave.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WB = ROOT / "results" / "WAVE_B" / "eval"
WA = ROOT / "results" / "WAVE_A" / "eval"
FIG = ROOT / "fyp_md_files" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

METHODS = ("reconstruction", "task", "digital")
LABEL = {"reconstruction": "Reconstruction JSCC",
         "task": "Task-oriented JSCC",
         "digital": "Digital baseline"}
COLOR = {"reconstruction": "#1f77b4", "task": "#d62728", "digital": "#2ca02c"}

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "legend.fontsize": 8,
})


def _load_wave_b():
    rows = []
    for p in sorted(WB.glob("*/*.json")):
        r = json.loads(p.read_text())
        if "aggregate_reconstruction" not in r:
            continue
        rows.append(r)
    return rows


def _load_wave_a():
    rows = []
    for p in sorted(WA.glob("*/*.json")):
        r = json.loads(p.read_text())
        if "_error" in r:
            continue
        for key, agg in r.items():
            if key.startswith("aggregate_"):
                rows.append({"tag": key[len("aggregate_"):],
                             "k": r["k"], "channel": r["channel"],
                             "snr_db": r["snr_db"], "agg": agg})
    return rows


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


# ---------------------------------------------------------------------------
# Fig 1: success rate vs bandwidth k
# ---------------------------------------------------------------------------

def fig_sr_vs_k(rows):
    ks = sorted({r["k"] for r in rows})
    per = defaultdict(list)                      # (method,k) -> [(seed, sr)]
    for r in rows:
        for m in METHODS:
            per[(m, r["k"], r["seed"])].append(
                r[f"aggregate_{m}"]["success_rate"])

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.26
    for i, m in enumerate(METHODS):
        means, spread = [], []
        for k in ks:
            seed_means = [_mean(per[(m, k, sd)]) for sd in
                          sorted({s for (mm, kk, s) in per
                                  if mm == m and kk == k})]
            means.append(_mean(seed_means))
            spread.append([min(seed_means), max(seed_means)])
        lo = [m - s[0] for m, s in zip(means, spread)]
        hi = [s[1] - m for m, s in zip(means, spread)]
        xs = [ks.index(k) + (i - 1) * w for k in ks]
        ax.bar(xs, means, width=w, label=LABEL[m], color=COLOR[m], alpha=0.9,
               yerr=[lo, hi], capsize=2,
               error_kw={"elinewidth": 0.8, "alpha": 0.7})
        for x, v, h in zip(xs, means, hi):
            ax.text(x, v + h + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.12)
    ax.set_title("Wave-B: success rate vs bandwidth (mean of seeds; bars = seed range)")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    p = FIG / "fig_wave_b_sr_vs_k.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Fig 2: success rate vs SNR for the low- and high-bandwidth extremes
# ---------------------------------------------------------------------------

def fig_sr_vs_snr(rows):
    ks = sorted({r["k"] for r in rows})
    picks = [ks[0], ks[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, k in zip(axes, picks):
        for m in METHODS:
            snrs = sorted({r["snr_db"] for r in rows if r["k"] == k})
            ys = []
            for s in snrs:
                v = [r[f"aggregate_{m}"]["success_rate"] for r in rows
                     if r["k"] == k and r["snr_db"] == s]
                ys.append(_mean(v))
            ax.plot(snrs, ys, marker="o", ms=3.5, lw=1.6, label=LABEL[m],
                    color=COLOR[m])
        ax.set_title(f"k = {k}")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("Success rate (sigma pooled)")
    axes[0].legend(loc="center right")
    fig.suptitle("Wave-B: success rate vs SNR at the bandwidth extremes", y=1.02)
    fig.tight_layout()
    p = FIG / "fig_wave_b_sr_vs_snr.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Fig 3: CSI-error sweep (success rate + reconstruction MSE)
# ---------------------------------------------------------------------------

def fig_csi_sweep(rows):
    sigmas = sorted({r["csi_error_sigma"] for r in rows})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for m in METHODS:
        ys = [_mean([r[f"aggregate_{m}"]["success_rate"] for r in rows
                     if r["csi_error_sigma"] == s]) for s in sigmas]
        ax1.plot(sigmas, ys, marker="o", ms=3.5, lw=1.6, label=LABEL[m],
                 color=COLOR[m])
    ax1.set_xlabel(r"CSI error $\sigma$")
    ax1.set_ylabel("Success rate")
    ax1.set_title("(a) Success rate is insensitive to CSI error")
    ax1.legend(loc="center right")

    ys = [_mean([r["aggregate_reconstruction"]["mse_mean"] for r in rows
                 if r["csi_error_sigma"] == s]) for s in sigmas]
    ax2.plot(sigmas, ys, marker="s", ms=3.5, lw=1.6, color=COLOR["reconstruction"])
    base = ys[0]
    ax2.set_xlabel(r"CSI error $\sigma$")
    ax2.set_ylabel("Reconstruction MSE")
    ax2.set_title(f"(b) Reconstruction MSE degrades (+{100 * (ys[-1] / base - 1):.0f}% at "
                  rf"$\sigma$={sigmas[-1]})")
    for x, y in zip(sigmas, ys):
        ax2.annotate(f"{100 * (y / base - 1):+.1f}%", (x, y), fontsize=6.5,
                     xytext=(0, 5), textcoords="offset points", ha="center")
    fig.suptitle("Wave-B: CSI-estimation-error sweep (all k, sigma=0 is the frozen path)",
                 y=1.02)
    fig.tight_layout()
    p = FIG / "fig_wave_b_csi_sweep.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Fig 4: Wave-A arm comparison (diagnostic grid, E=1000)
# ---------------------------------------------------------------------------

def fig_wave_a_arms(rows):
    arms = sorted({r["tag"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, ch in zip(axes, ("awgn", "rayleigh")):
        sub = [r for r in rows if r["channel"] == ch]
        ks = sorted({r["k"] for r in sub})
        w = 0.8 / max(len(arms), 1)
        for i, arm in enumerate(arms):
            means = []
            for k in ks:
                v = [r["agg"]["success_rate"] for r in sub
                     if r["tag"] == arm and r["k"] == k]
                means.append(_mean(v) if v else np.nan)
            xs = [ks.index(k) + (i - (len(arms) - 1) / 2) * w for k in ks]
            ax.bar(xs, means, width=w, label=arm)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_title(ch.upper())
        ax.set_xlabel("bandwidth k")
    axes[0].set_ylabel("Success rate (mean over SNR, seeds)")
    axes[0].set_ylim(0, 1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
               frameon=False)
    fig.suptitle("Wave-A: arm comparison (diagnostic grid, E=1000 — not 1:1 "
                 "comparable with frozen M1/M2)", y=1.01)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = FIG / "fig_wave_a_arms.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    wb = _load_wave_b()
    wa = _load_wave_a()
    print(f"read {len(wb)} Wave-B cells, {len(wa)} Wave-A arm-rows")
    if wb:
        for f in (fig_sr_vs_k, fig_sr_vs_snr, fig_csi_sweep):
            print("wrote", f(wb).relative_to(ROOT))
    if wa:
        print("wrote", fig_wave_a_arms(wa).relative_to(ROOT))


if __name__ == "__main__":
    main()
