"""Step-14 — M1 analysis & reporting tables (READ-ONLY over results/M1).

No training, no evaluation, no modification of any historical JSON.
Reads results/M1/step10_m1_summary.json (pooled rows) and all 198
results/M1/eval/<unit>/eval_<snr>.json files, then:

  1. pooled success-rate tables (reconstruction vs task) with Wilson CIs
  2. SNR-threshold summary (first SNR reaching pooled SR >= 0.50 / 0.95)
  3. McNemar audit: corrected p = 2 x stored p (Step-10E defect fix);
     verifies n10 > n01 (reconstruction wins) in every paired cell
  4. paired-t summary on final distance and control effort
  5. per-seed agreement (in how many of the 66 conditions do all 3 seeds
     independently agree on the winner)

Outputs:
  - prints markdown tables to stdout
  - writes results/M1/m1_analysis_tables.json (NEW file; nothing overwritten)

Run:  .venv/Scripts/python.exe tools/analyze_m1.py
"""

from __future__ import annotations

import json
from pathlib import Path

M1 = Path("results/M1")
EVAL = M1 / "eval"
SNRS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
KS = (1, 2, 3)
CHANNELS = ("awgn", "rayleigh")
SEEDS = (42, 43, 44)


def load_units():
    units = []
    for k in KS:
        for ch in CHANNELS:
            for seed in SEEDS:
                for snr in SNRS:
                    p = EVAL / f"k{k}_{ch}_seed{seed}" / f"eval_{snr:g}.json"
                    if p.exists():
                        units.append((k, ch, seed, snr,
                                      json.loads(p.read_text())))
    return units


def f3(x):
    return f"{x:.3f}"


def main():
    summary = json.loads((M1 / "step10_m1_summary.json").read_text())
    pooled = {(r["method"], r["k"], r["channel"],
               r["snr_db"]): r for r in summary["pooled_rows"]}
    units = load_units()
    print(f"# M1 analysis - {len(units)}/198 eval units, "
          f"{len(pooled)}/132 pooled rows (66 conditions x 2 methods)\n")

    # ---- 1. pooled SR tables -------------------------------------------
    print("## Pooled success rates (3 seeds, 15000 episodes/condition)\n")
    for ch in CHANNELS:
        print(f"### {ch}\n")
        print("| k (rho) | metric | " +
              " | ".join(f"{s:g} dB" for s in SNRS) + " |")
        print("|---|---|" + "---|" * len(SNRS))
        for k in KS:
            for method, label in (("reconstruction", "recon"),
                                  ("task", "task")):
                vals = []
                for s in SNRS:
                    r = pooled[(method, k, ch, s)]
                    vals.append(f"{r['success_rate_pooled']:.3f}")
                print(f"| k={k} (rho={k}/6) | {label} SR | " +
                      " | ".join(vals) + " |")
        print()

    # ---- 2. SNR thresholds ---------------------------------------------
    print("## First SNR (dB) reaching pooled SR threshold\n")
    print("| k | channel | method | SR>=0.50 | SR>=0.95 |")
    print("|---|---|---|---|---|")
    for k in KS:
        for ch in CHANNELS:
            for method in ("reconstruction", "task"):
                s50 = s95 = "—"
                for s in SNRS:
                    sr = pooled[(method, k, ch, s)]["success_rate_pooled"]
                    if s50 == "—" and sr >= 0.50:
                        s50 = f"{s:g}"
                    if s95 == "—" and sr >= 0.95:
                        s95 = f"{s:g}"
                print(f"| {k} | {ch} | {method} | {s50} | {s95} |")
    print()

    # ---- 3. McNemar audit (corrected p = 2 x stored) --------------------
    print("## McNemar audit — task vs reconstruction, all paired cells\n")
    n_cells = n_recon_wins = n_sig = 0
    max_p_corr = 0.0
    max_p_cell = None
    n01_min, n01_max = 10**9, -1
    n10_min, n10_max = 10**9, -1
    n_seeds_disagree_dir = 0
    for k, ch, seed, snr, u in units:
        m = u["mcnemar_task_vs_recon"]
        n01, n10 = m["n01"], m["n10"]
        p_corr = 2.0 * m["p_value"]
        n_cells += 1
        if n10 > n01:
            n_recon_wins += 1
        if p_corr < 0.05:
            n_sig += 1
        if p_corr > max_p_corr:
            max_p_corr, max_p_cell = p_corr, f"k{k}_{ch}_s{seed}@{snr:g}dB"
        n01_min, n01_max = min(n01_min, n01), max(n01_max, n01)
        n10_min, n10_max = min(n10_min, n10), max(n10_max, n10)
    print(f"- cells: {n_cells}")
    print(f"- reconstruction wins (n10 > n01): {n_recon_wins}/{n_cells}")
    print(f"- corrected p < 0.05: {n_sig}/{n_cells}")
    print(f"- max corrected p = {max_p_corr:.3e} at {max_p_cell}")
    print(f"- n01 (task better) range: {n01_min}..{n01_max}; "
          f"n10 (recon better) range: {n10_min}..{n10_max}")
    print("- correction: stored p = 0.5*erfc(...) [one-sided]; "
          "corrected p = 2 x stored (two-sided chi2(1) survival); "
          "chi2 values unaffected")
    print()

    # ---- per-seed agreement (66 conditions) ------------------------------
    agree = 0
    for k in KS:
        for ch in CHANNELS:
            for s in SNRS:
                wins = 0
                for seed in SEEDS:
                    u = next((x for kk, cc, ss, nn, x in units
                              if (kk, cc, ss, nn) == (k, ch, seed, s)), None)
                    if u is None:
                        continue
                    sr_t = u["aggregate_task"]["success_rate"]
                    sr_r = u["aggregate_reconstruction"]["success_rate"]
                    if sr_r > sr_t:
                        wins += 1
                if wins == len(SEEDS):
                    agree += 1
    print(f"- per-seed agreement: reconstruction SR > task SR in all 3 "
          f"seeds for {agree}/66 conditions")
    print()

    # ---- 4. paired-t summaries ------------------------------------------
    print("## Paired-t (task - reconstruction) sign counts\n")
    for key, label in (
            ("paired_t_final_distance_task_vs_recon", "final distance"),
            ("paired_t_avg_distance_task_vs_recon", "avg distance"),
            ("paired_t_control_effort_task_vs_recon", "control effort")):
        pos = neg = nsig = 0
        tmin, tmax = 10**9, -10**9
        for k, ch, seed, snr, u in units:
            t = u[key]
            d = t["mean_diff"]
            pos += d > 0
            neg += d < 0
            nsig += t["p_value"] < 0.05
            tmin, tmax = min(tmin, t["t_stat"]), max(tmax, t["t_stat"])
        print(f"- {label}: task>recon in {pos} cells, task<recon in {neg}; "
              f"significant (p<0.05) in {nsig}/{n_cells}; "
              f"t range [{tmin:.1f}, {tmax:.1f}]")
    print()

    # ---- write tables JSON ----------------------------------------------
    out = {
        "STEP": "14", "purpose": "M1 analysis tables (read-only derivation)",
        "source": "results/M1/step10_m1_summary.json + 198 eval unit JSONs",
        "mcnemar_audit": {
            "cells": n_cells,
            "recon_wins": n_recon_wins,
            "significant_corrected_p_lt_0.05": n_sig,
            "max_corrected_p": max_p_corr,
            "max_corrected_p_cell": max_p_cell,
            "n01_range": [n01_min, n01_max],
            "n10_range": [n10_min, n10_max],
            "correction": "corrected p = 2 x stored p (Step-10E)"},
        "per_seed_agreement_conditions": agree,
        "of_conditions": len(KS) * len(CHANNELS) * len(SNRS),
    }
    dst = M1 / "m1_analysis_tables.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"-- wrote {dst} (new file; no historical JSON touched)")


if __name__ == "__main__":
    main()
