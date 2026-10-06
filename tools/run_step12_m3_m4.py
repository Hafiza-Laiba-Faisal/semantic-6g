"""Step 12/13 — M3 (digital-infeasibility display) + M4 (secondary digital
reference) — user-approved 2026-10-02.

M3 (frozen, structural display — audit 8.5): the matched-budget digital
baseline is INFEASIBLE at k in {1, 2, 3} because the smallest transmissible
packet needs 6B + 6 = 7 channel uses at B = 1. This step RECORDS the frozen
packet arithmetic for the 3 display cells; nothing is trained or evaluated
(no digital signal exists at these k by construction).

M4 (frozen, SECONDARY reference — audit 8.6/8.7): digital baseline at the
fixed secondary point B = 8, k = n_dig = 54 (6*8 + 6 = 54, exact fit), one
cell per channel (seed 42), evaluated on the M2 SNR grid {0,5,10,15,20} dB
with 5000 episodes, test seed 10042 — evaluation-ONLY (digital has no
training). M4 is explicitly NOT a primary cell and never carries the
matched-budget label.

No M1/M2 artifact is touched. No M5 path exists in this tool.

Subcommands (repository root):
    verify      frozen-protocol + workload hard gate
    run-m3      record the 3 structural infeasibility display cells
    run-m4      run the 10 evaluation-only secondary-reference conditions
    validate    completion checks
    finalize    write results/M3/m3_report.json (run-m3) and
                results/M4/step13_m4_summary.json (finalize)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from project import config
from project.evaluation.evaluate import Condition, evaluate_condition
from project.evaluation.statistics import wilson_ci_95
from project.experiments.config import (
    CHANNELS, M2_SNR_DB, M4_DIGITAL_B, M4_K, ExperimentConfig)

M3_ROOT = Path("results/M3")
M4_ROOT = Path("results/M4")
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# verify — hard gate
# ---------------------------------------------------------------------------

def cmd_verify():
    print("== frozen protocol verification (Step-12/13 gate) ==")
    check("m3_cells==k{1,2,3}_B1_dry_run",
          all(ExperimentConfig.m3(k, bits=1, run_mode="dry_run").expected_infeasible
              for k in (1, 2, 3)))
    check("m4_point==B8_k54_exact_fit",
          M4_DIGITAL_B == 8 and M4_K == 54
          and 6 * M4_DIGITAL_B + 6 == M4_K)
    check("m4_eval_grid==M2_grid", M2_SNR_DB == (0.0, 5.0, 10.0, 15.0, 20.0))
    check("m4_cells==2channels_x5snr_seed42", 2 * len(M2_SNR_DB) == 10)
    check("n_test_episodes==5000", config.N_TEST_EPISODES == 5000)
    check("test_seed==10042", config.TEST_SEED == 10042)
    check("digital_K7_rate_half_gray_qpsk",
          config.CONV_CONSTRAINT_LENGTH == 7 and config.CONV_RATE == 0.5
          and config.CONV_GENERATORS_OCTAL == (0o171, 0o133)
          and config.MODULATION_BITS_PER_SYMBOL == 2)
    check("M5==deferred+disabled", config.M5_STATUS == "deferred"
          and config.M5_EXECUTION_ENABLED is False)
    config.validate()
    print("config.validate() OK")
    print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    return 1 if FAILS else 0


# ---------------------------------------------------------------------------
# M3 — structural infeasibility display (no training, no eval)
# ---------------------------------------------------------------------------

def cmd_run_m3():
    print("== M3: digital infeasibility display cells (k in {1,2,3}) ==")
    rows = []
    for k in (1, 2, 3):
        cfg = ExperimentConfig.m3(k, bits=1, run_mode="dry_run")
        try:
            pkt = config.digital_packet(k, cfg.bits_per_component)
            feasible, note = True, pkt
        except config.InfeasibleDigitalConfig as e:
            feasible, note = False, str(e)
        row = {
            "run_id": cfg.run_id if hasattr(cfg, "run_id") else None,
            "experiment": "M3", "method": "digital",
            "k": k, "bits_per_component": cfg.bits_per_component,
            "required_channel_uses": 6 * cfg.bits_per_component + 6,
            "available_channel_uses": k,
            "deficit": (6 * cfg.bits_per_component + 6) - k,
            "expected_infeasible": cfg.expected_infeasible,
            "status": ("infeasible-by-design" if not feasible
                       else "UNEXPECTED-FEASIBLE"),
            "reason": note,
        }
        rows.append(row)
        tag = "PASS" if not feasible else "FAIL"
        print(f"  {tag} k={k}: needs {row['required_channel_uses']} uses, "
              f"has {k} -> deficit {row['deficit']} "
              f"({row['status']})")
    # contrast: the matched-budget M2 points must all be feasible
    contrast = {}
    for k in (12, 18, 24, 30, 54):
        b = {12: 1, 18: 2, 24: 3, 30: 4, 54: 8}[k]
        try:
            pkt = config.digital_packet(k, b)
            contrast[str(k)] = {"B": b, "feasible": True,
                                "n_info_bits": pkt["n_info_bits"]}
        except config.InfeasibleDigitalConfig:
            contrast[str(k)] = {"B": b, "feasible": False}
    check("all-three-m3-cells-infeasible",
          all(r["status"] == "infeasible-by-design" for r in rows))
    check("all-m2-digital-points-feasible",
          all(v["feasible"] for v in contrast.values()))
    report = {
        "STEP": "12", "experiment": "M3",
        "status": "complete" if not FAILS else "incomplete",
        "definition": ("Structural display: the matched-budget digital "
                       "baseline cannot exist at k in {1,2,3} because the "
                       "smallest packet needs 6B+6 = 7 channel uses at B=1. "
                       "No training or evaluation exists for these cells "
                       "BY DESIGN (audit 8.5); this is a feasibility "
                       "boundary, not a measured result."),
        "cells": rows,
        "m2_contrast_feasible_points": contrast,
        "interpretation_note": ("M3 is architectural infeasibility, NOT an "
                               "empirical performance result. The M3/M4 "
                               "distinction: M4's B=8 reference transmits "
                               "the same 54 channel uses as the k=54 neural "
                               "cells and is labeled SECONDARY."),
    }
    out = M3_ROOT / "m3_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out} (3 display cells)")
    return 1 if FAILS else 0


# ---------------------------------------------------------------------------
# M4 — secondary digital reference, evaluation-only (10 conditions)
# ---------------------------------------------------------------------------

def _m4_eval_path(channel, snr):
    return M4_ROOT / "eval" / channel / f"eval_{snr:g}.json"


def cmd_run_m4():
    print("== M4: secondary digital reference (B=8, k=54) — 10 conditions ==")
    n_new = 0
    total_wall = 0.0
    for channel in CHANNELS:
        for snr in M2_SNR_DB:
            out = _m4_eval_path(channel, snr)
            if out.exists():
                continue
            cond = Condition(k=M4_K, channel=channel, snr_db=snr,
                             bits_per_component=M4_DIGITAL_B,
                             t_max=config.T_MAX,
                             episodes=config.N_TEST_EPISODES,
                             seed=config.TEST_SEED)
            t0 = time.perf_counter()
            with torch.no_grad():
                cell = evaluate_condition(cond, methods=["digital"])
            wall = time.perf_counter() - t0
            total_wall += wall
            agg = cell["digital"]["aggregate"]
            rec = {
                "STEP": "13", "experiment": "M4",
                "channel": channel, "snr_db": snr,
                "k": M4_K, "bits_per_component": M4_DIGITAL_B,
                "test_seed": config.TEST_SEED,
                "episodes": config.N_TEST_EPISODES, "t_max": config.T_MAX,
                "secondary_reference": True,
                "wall_s": round(wall, 1),
                "aggregate_digital": agg,
                "wilson_digital": wilson_ci_95(
                    cell["digital"]["metrics"]["success"]),
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=2, default=str))
            n_new += 1
            print(f"  EVAL {channel} snr={snr:5.1f} "
                  f"SR digital={agg['success_rate']:.4f} ({wall:.1f}s)")
    done = sum(1 for ch in CHANNELS for snr in M2_SNR_DB
               if _m4_eval_path(ch, snr).exists())
    print(f"-- M4 eval finished: {n_new} new this session, "
          f"{done}/10 complete, eval_wall={total_wall:.1f}s")
    return 0 if done == 10 else 1


# ---------------------------------------------------------------------------
# validate / finalize
# ---------------------------------------------------------------------------

def cmd_validate():
    print("== M3/M4 completion validation ==")
    FAILS.clear()
    rep = M3_ROOT / "m3_report.json"
    check("m3-report-exists", rep.exists())
    if rep.exists():
        r = json.loads(rep.read_text())
        check("m3-three-infeasible-cells",
              len(r["cells"]) == 3
              and all(c["status"] == "infeasible-by-design"
                      for c in r["cells"]))
    missing = [f"{ch}@{snr:g}" for ch in CHANNELS for snr in M2_SNR_DB
               if not _m4_eval_path(ch, snr).exists()]
    check("m4-10-eval-units-complete", len(missing) == 0,
          f"missing={missing}")
    eps_ok = True
    for ch in CHANNELS:
        for snr in M2_SNR_DB:
            p = _m4_eval_path(ch, snr)
            if p.exists():
                rec = json.loads(p.read_text())
                if rec.get("episodes") != 5000 or rec.get("test_seed") != 10042:
                    eps_ok = False
                if not rec.get("secondary_reference"):
                    eps_ok = False
    check("m4-episodes==5000-test-seed==10042-secondary", eps_ok)
    check("m1-intact", len(list(Path("results/M1").glob("*__*/DONE.json"))) == 36)
    print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    return 1 if FAILS else 0


def cmd_finalize():
    print("== finalize: results/M4/step13_m4_summary.json ==")
    rows = []
    for ch in CHANNELS:
        for snr in M2_SNR_DB:
            p = _m4_eval_path(ch, snr)
            if not p.exists():
                continue
            rec = json.loads(p.read_text())
            agg = rec["aggregate_digital"]
            w = rec["wilson_digital"]
            rows.append({
                "channel": ch, "snr_db": snr, "k": M4_K,
                "bits_per_component": M4_DIGITAL_B,
                "success_rate": agg["success_rate"],
                "success_count": agg["success_count"],
                "episodes": agg["episodes"],
                "wilson_low": w["wilson_low"], "wilson_high": w["wilson_high"],
                "final_distance_mean": agg.get("final_distance_mean"),
                "avg_distance_mean": agg.get("avg_distance_mean"),
                "control_effort_mean": agg.get("control_effort_mean"),
            })
    summary = {
        "STEP": "13", "experiment": "M4",
        "status": "complete" if len(rows) == 10 else "incomplete",
        "definition": ("SECONDARY digital reference (never a primary cell): "
                       "B=8, k=n_dig=54 (6*8+6=54 exact fit), evaluation-"
                       "only, one cell per channel, M2 SNR grid, 5000 "
                       "episodes, test seed 10042."),
        "conditions": rows,
        "n_conditions": len(rows),
        "interpretation_note": ("Descriptive only. M4's B=8 reference uses "
                               "the same 54 channel uses as the k=54 neural "
                               "cells but is a DIFFERENT operating point "
                               "(secondary); it is never pooled with the "
                               "matched-budget M2 primary comparisons."),
    }
    out = M4_ROOT / "step13_m4_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out} ({len(rows)} conditions)")
    return 0 if len(rows) == 10 else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in (
            "verify", "run-m3", "run-m4", "validate", "finalize", "all"):
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "all":
        rc = cmd_verify() or cmd_run_m3() or cmd_run_m4() or cmd_validate() \
            or cmd_finalize()
        sys.exit(rc)
    sys.exit({"verify": cmd_verify, "run-m3": cmd_run_m3, "run-m4": cmd_run_m4,
              "validate": cmd_validate, "finalize": cmd_finalize}[cmd]())
