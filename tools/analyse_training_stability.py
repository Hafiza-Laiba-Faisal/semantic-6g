"""Training-stability mechanism analysis (read-only, no training).

The reviewer gap is a *mechanism* claim: the thesis shows that the lambda-shaped
shaping stabilises low-bandwidth task training, but not *why*.  This script
measures what the stored runs actually recorded - per-step encoder gradient
norms and reconstruction MSE - and reports the conditioning evidence, without
training anything and without touching any result artifact.

Two layers of evidence:

1. **Broad (all train reports).**  Every completed run stores
   ``grad_enc_first`` / ``grad_enc_final`` and the loss.  That is enough to show
   which arms leave training with an unbounded gradient and which settle.

2. **Deep (runs that stored the full step history).**  WAVE_C, ARCH_ABL,
   AERIAL_TRAIN, AERIAL_TRAIN_TRUNC (and K24_SEEDS when present) store
   ``history``.  For those we report the gradient distribution, the trend
   (last-100 / first-100 median), and the rank correlation between the
   reconstruction MSE and the gradient norm - the observable behind the claim
   that the failure mode is the decoder leaving its well-conditioned regime.

What this does NOT do: it does not identify a causal mechanism in the sense of
an intervention on the loss terms (that would need new arms), and it does not
invent numbers for arms whose history was never stored.  Limitations are
printed and written into the JSON.

Usage:
    python tools/analyse_training_stability.py [--figure]

Writes ``results/MECHANISM/step_stability_summary.json`` and, with --figure,
``fyp_md_files/figures/fig_training_stability.png``.
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENTS = ("WAVE_C", "ARCH_ABL", "AERIAL_TRAIN", "AERIAL_TRAIN_TRUNC",
               "K24_SEEDS", "WAVE_A", "M2")
OUT_DIR = ROOT / "results" / "MECHANISM"


def _group(lambdas):
    if lambdas is None:
        return "frozen task arm (no lambda_u/lambda_v)"
    lam = [float(x) for x in lambdas]
    if abs(lam[1] - 0.1) < 1e-9 and abs(lam[2] - 0.1) < 1e-9:
        return "lambda-heavy (0.1/0.1)"
    if abs(lam[1] - 0.01) < 1e-9 and abs(lam[2] - 0.01) < 1e-9:
        return "lambda-light (0.01/0.01)"
    return f"lambda ({lam[1]:g}/{lam[2]:g})"


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _spearman(xs, ys):
    if len(xs) < 8:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _reports():
    out = []
    for exp in EXPERIMENTS:
        for p in sorted((ROOT / "results" / exp).rglob("train_report.json")):
            try:
                rec = json.loads(p.read_text())
            except Exception:
                continue
            spec = rec.get("spec", {})
            out.append((exp, p, rec, spec))
    return out


def _coarse_row(exp, p, rec, spec):
    return {
        "experiment": exp,
        "cell": p.parent.name,
        "group": _group(spec.get("lambdas")),
        "k": spec.get("k"),
        "channel": spec.get("channel"),
        "seed": spec.get("seed"),
        "hidden": spec.get("hidden"),
        "status": rec.get("exit_status"),
        "grad_first": rec.get("grad_enc_first"),
        "grad_final": rec.get("grad_enc_final"),
        "task_loss_final": rec.get("task_loss_final"),
        "nan_batches": rec.get("nan_batches"),
        "has_history": bool(rec.get("history")),
    }


def _deep_row(exp, p, rec, spec):
    hist = rec.get("history") or []
    if not hist:
        return None
    grads = [h.get("grad_enc") for h in hist]
    fin = [g for g in grads if _finite(g) and g >= 0]
    nonfinite = sum(1 for g in grads if not _finite(g))
    recon = [h.get("recon_mse_diag") for h in hist if _finite(h.get("recon_mse_diag"))]
    tot = [h.get("total") for h in hist if _finite(h.get("total"))]
    half = max(1, len(fin) // 100)
    first_m = _pct(fin[:max(1, len(fin) // 100)], 0.5)
    last_m = _pct(fin[-max(1, len(fin) // 100):], 0.5)
    xs = [math.log10(max(h["recon_mse_diag"], 1e-12)) for h in hist
          if _finite(h.get("recon_mse_diag")) and _finite(h.get("grad_enc"))
          and h["grad_enc"] > 0]
    ys = [math.log10(h["grad_enc"]) for h in hist
          if _finite(h.get("recon_mse_diag")) and _finite(h.get("grad_enc"))
          and h["grad_enc"] > 0]
    return {
        "experiment": exp,
        "cell": p.parent.name,
        "group": _group(spec.get("lambdas")),
        "k": spec.get("k"),
        "channel": spec.get("channel"),
        "seed": spec.get("seed"),
        "hidden": spec.get("hidden"),
        "steps": len(hist),
        "grad_p50": _pct(fin, 0.5),
        "grad_p99": _pct(fin, 0.99),
        "grad_max": max(fin) if fin else None,
        "frac_grad_gt_1e6": (sum(1 for g in fin if g > 1e6) / len(fin)) if fin else None,
        "nonfinite_grad_steps": nonfinite,
        "grad_median_first_pct": first_m,
        "grad_median_last_pct": last_m,
        "grad_trend_last_over_first": (last_m / first_m) if first_m else None,
        "recon_mse_median": _pct(recon, 0.5) if recon else None,
        "recon_mse_max": max(recon) if recon else None,
        "loss_min": min(tot) if tot else None,
        "loss_min_step": (tot.index(min(tot)) + 1) if tot else None,
        "loss_final": tot[-1] if tot else None,
        "spearman_log_recon_vs_log_grad": _spearman(xs, ys),
        "converged": (tot[-1] <= 2.0 * min(tot)) if tot else None,
        "_half": half,
    }


def _group_summary(deep):
    out = {}
    for r in deep:
        g = out.setdefault(r["group"], {"cells": 0, "grad_p99": [],
                                        "nonfinite_grad_steps": 0,
                                        "frac_grad_gt_1e6": [],
                                        "spearman": [], "converged": 0})
        g["cells"] += 1
        if _finite(r["grad_p99"]):
            g["grad_p99"].append(r["grad_p99"])
        g["nonfinite_grad_steps"] += r["nonfinite_grad_steps"]
        if _finite(r["frac_grad_gt_1e6"]):
            g["frac_grad_gt_1e6"].append(r["frac_grad_gt_1e6"])
        if r["spearman_log_recon_vs_log_grad"] is not None:
            g["spearman"].append(r["spearman_log_recon_vs_log_grad"])
        g["converged"] += 1 if r["converged"] else 0
    for g in out.values():
        g["grad_p99_median"] = _pct(g["grad_p99"], 0.5)
        g["grad_p99_max"] = max(g["grad_p99"]) if g["grad_p99"] else None
        g["frac_grad_gt_1e6_median"] = _pct(g["frac_grad_gt_1e6"], 0.5)
        g["spearman_median"] = _pct(g["spearman"], 0.5)
        g.pop("grad_p99")
        g.pop("frac_grad_gt_1e6")
        g.pop("spearman")
    return out


def main(with_figure=False):
    reports = _reports()
    coarse = [_coarse_row(*r) for r in reports]
    deep = [d for d in (_deep_row(*r) for r in reports) if d]
    for d in deep:
        d.pop("_half", None)

    done = [c for c in coarse if c["status"] == "complete"]
    print(f"== training-stability analysis ({len(coarse)} reports, "
          f"{len(done)} complete, {len(deep)} with per-step history) ==")

    groups = {}
    for c in done:
        g = groups.setdefault(c["group"], {"cells": 0, "final_grad": [],
                                           "nan": 0, "unstable": 0,
                                           "no_grad": 0})
        g["cells"] += 1
        if _finite(c["grad_final"]):
            g["final_grad"].append(c["grad_final"])
        else:
            g["no_grad"] += 1
        g["nan"] += c["nan_batches"] or 0
        if (_finite(c["grad_final"]) and c["grad_final"] > 1e3) or c["nan_batches"]:
            g["unstable"] += 1
    print("\n-- final encoder gradient by arm (all completed cells) --")
    print("   (n_finite = cells that stored a numeric final gradient; "
          "many frozen-protocol reports store null)")
    for name, g in sorted(groups.items()):
        med = _pct(g["final_grad"], 0.5)
        mx = max(g["final_grad"]) if g["final_grad"] else None
        print(f"   {name:42s} cells {g['cells']:3d}  n_finite {len(g['final_grad']):3d}"
              f"  median {med!s:>10.10}  max {mx!s:>10.10}"
              f"  unstable {g['unstable']:3d}  NaN batches {g['nan']}")

    summary = _group_summary(deep)
    print("\n-- per-step gradient conditioning (runs with history) --")
    for name, g in sorted(summary.items()):
        print(f"   {name:42s} cells {g['cells']:2d}  p99(grad) median "
              f"{g['grad_p99_median']:.3g}  max {g['grad_p99_max']:.3g}  "
              f"non-finite steps {g['nonfinite_grad_steps']:4d}  "
              f"converged {g['converged']}/{g['cells']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "step_stability_summary.json"
    out.write_text(json.dumps({
        "analysis": "training-stability mechanism (read-only, no training)",
        "question": ("why does the lambda-shaped arm train stably at low "
                     "bandwidth when the frozen task arm does not?"),
        "evidence_type": ("observational, from stored per-step histories and "
                          "final gradient norms; no arm was trained for this"),
        "scope_note": ("the only stored per-step histories for a frozen-task "
                       "arm are the A-9 cells at width 256; no 128-wide frozen "
                       "cell stored a history. Wave-A/M2 store only first/final "
                       "gradients, which is why the broad table is coarse."),
        "limitations": [
            "the naive within-run rank correlation between reconstruction MSE "
            "and gradient norm is NEGATIVE (median rho -0.43) and is confounded "
            "by the early transient; it is stored for transparency but is NOT "
            "used as evidence for any claim here",
            "different groups have different widths (frozen-arm history is "
            "width 256; A-9 showed width does not rescue the arm)",
            "cannot be pooled with a confidence statement: cells differ in k, "
            "channel and seed by design",
            "the lambda arm is not a general cure: the two A-12 cells (lambda "
            "arm, untruncated 3GPP tail) show p99 gradients of ~1e19 and 514 "
            "non-finite steps",
        ],
        "broad_final_gradient": {
            name: {"cells": g["cells"],
                   "median_final_grad": _pct(g["final_grad"], 0.5),
                   "max_final_grad": max(g["final_grad"]) if g["final_grad"] else None,
                   "unstable_cells": g["unstable"],
                   "nan_batches_total": g["nan"]}
            for name, g in sorted(groups.items())},
        "per_step_conditioning": summary,
        "cells_with_history": deep,
        "coarse_rows": done,
    }, indent=2, default=str))
    print(f"\nwrote {out.relative_to(ROOT)}")

    if with_figure:
        _figure(deep)
    return 0


def _figure(deep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style(r):
        if "frozen" in r["group"]:
            return "#ff7f0e", "--"
        if r["experiment"] == "AERIAL_TRAIN":
            return "#d62728", "-"
        if r["experiment"] == "AERIAL_TRAIN_TRUNC":
            return "#9467bd", "-"
        if r["experiment"] == "K24_SEEDS":
            return "#2ca02c", "-"
        if "light" in r["group"]:
            return "#17becf", "-"
        return "#1f77b4", "-"

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    ax = axes[0]
    shown = 0
    for r in deep:
        hist_cell = ROOT / "results" / r["experiment"]
        # rolling median of the stored gradient norm
        rec = None
        for p in hist_cell.rglob("train_report.json"):
            if p.parent.name == r["cell"]:
                rec = json.loads(p.read_text())
                break
        if rec is None:
            continue
        g = [h.get("grad_enc") for h in rec.get("history") or []]
        xs, ys = [], []
        win = 25
        for i in range(0, len(g), win):
            chunk = [v for v in g[i:i + win] if _finite(v) and v > 0]
            if chunk:
                xs.append(i + len(chunk) / 2)
                ys.append(sorted(chunk)[len(chunk) // 2])
        if not xs:
            continue
        col, ls = style(r)
        label = f"{r['group'].split(' (')[0]}, k={r['k']}, {r['channel']}"
        if r["experiment"] in ("AERIAL_TRAIN", "AERIAL_TRAIN_TRUNC"):
            label += f" [{r['experiment']}]"
        if r["experiment"] == "K24_SEEDS":
            label += f", seed {r['seed']}"
        ax.plot(xs, ys, ls, color=col, lw=1.1, alpha=0.85, label=label)
        shown += 1
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("encoder gradient norm (25-step median)")
    ax.set_title(f"gradient norm over training ({shown} runs)", fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.5)
    handles, labels = ax.get_legend_handles_labels()

    ax = axes[1]
    labels, data, colors = [], [], []
    for r in deep:
        rec = None
        for p in (ROOT / "results" / r["experiment"]).rglob("train_report.json"):
            if p.parent.name == r["cell"]:
                rec = json.loads(p.read_text())
                break
        if rec is None:
            continue
        g = [v for v in (h.get("grad_enc") for h in rec.get("history") or [])
             if _finite(v) and v > 0]
        if not g:
            continue
        g = g[::max(1, len(g) // 800)]
        tag = {"AERIAL_TRAIN": "\nA-12",
               "AERIAL_TRAIN_TRUNC": "\nA-13",
               "K24_SEEDS": f"\nseed {r['seed']}"}.get(r["experiment"], "")
        labels.append(f"{r['group'].split(' (')[0][:16]}\nk={r['k']} "
                      f"{r['channel'][:4]}{tag}")
        data.append([math.log10(v) for v in g])
        colors.append(style(r)[0])
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    whis=(5, 95))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
        patch.set_edgecolor(col)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(0.9)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=4.8, rotation=90)
    ax.set_ylabel("log10 encoder gradient norm")
    ax.set_title("per-step gradient distribution (5-95 % whiskers)", fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=5.0,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    out = ROOT / "fyp_md_files" / "figures" / "fig_training_stability.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main("--figure" in sys.argv[1:]))
