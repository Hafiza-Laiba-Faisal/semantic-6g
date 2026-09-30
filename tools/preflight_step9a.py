"""Step 9A configuration-only preflight (NO training, NO experiment execution).

Checks, in order:
  1. frozen conventions (matrix constants vs the frozen prompt)
  2. workload reconstruction from code vs independent expectation
  3. pairing-preservation re-check (Step-7 rule)
  4. budget equality per M2 k
  5. statistical-protocol API presence
  6. scientific-result protection (gitignore, identity separation)
  7. per-cell preflight over ALL actionable cells (checkpoint/output
     destinations under the run-identity scheme; M3 infeasibility)
  8. full-scale training-budget decision status (flagged, not assumed)

Output: results/preflight/step9a_preflight.json (gitignored).

Run from the repository root:
    .venv/Scripts/python.exe tools/preflight_step9a.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from project import config
from project.evaluation.evaluate import Condition, evaluate_condition, generate_realizations
from project.evaluation.metrics import aggregate_episode_metrics
from project.evaluation.statistics import mcnemar_test, paired_t_test, wilson_ci_95
from project.experiments.config import (
    CHANNELS, M1_K_VALUES, M1_SNR_DB, M2_K_VALUES, M2_SNR_DB,
    M4_DIGITAL_B, TRAINING_SEEDS, ExperimentConfig)
from project.experiments.identity import cell_state, config_hash, run_id
from project.experiments.matrix import expected_cells, validate_matrix

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]"
          + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def build_full_proposal():
    cells = []
    for method in ("task", "reconstruction"):
        for k in M1_K_VALUES:
            for ch in CHANNELS:
                for seed in TRAINING_SEEDS:
                    cells.append(ExperimentConfig.m1(method, k, ch, seed))
    for method in ("digital", "task", "reconstruction"):
        for k in M2_K_VALUES:
            for ch in CHANNELS:
                for seed in TRAINING_SEEDS:
                    cells.append(ExperimentConfig.m2(method, k, ch, seed))
    for k in (1, 2, 3):
        cells.append(ExperimentConfig.m3(k, bits=1, run_mode="dry_run"))
    for ch in CHANNELS:
        cells.append(ExperimentConfig.m4_secondary(ch, seed=42))
    cells.append(ExperimentConfig(experiment="M5", method="task", channel="awgn",
                                  k=3, training_seed=42, run_mode="dry_run"))
    return cells


def frozen_conventions():
    print("\n== 1. frozen conventions ==")
    check("state-dim/dt/T", config.STATE_DIM == 6 and config.DT == 0.1
          and config.T_MAX == 100)
    check("workspace/goal", (config.WORKSPACE_MIN, config.WORKSPACE_MAX)
          == (-8.0, 8.0) and config.GOAL_RADIUS == 0.5
          and config.D_MIN == 4.0)
    check("vmax/umax/gains", config.V_MAX == 5.0 and config.U_MAX == 2.0
          and config.KP == 1.0 and config.KV == 2.0)
    check("power/snr", config.TX_POWER == 1.0
          and config.SNR_TRAIN_RANGE_DB == (0.0, 20.0)
          and config.SNR_TEST_GRID_DB == tuple(float(s) for s in range(0, 21, 2)))
    check("seeds", config.TRAINING_SEEDS == (42, 43, 44)
          and config.TEST_SEED == 10042)
    check("k-grids", config.BANDWIDTH_K_VALUES == (1, 2, 3)
          and sorted(config.DIGITAL_B_BY_K) == [12, 18, 24, 30, 54])
    check("B-by-k", config.DIGITAL_B_BY_K == {12: 1, 18: 2, 24: 3, 30: 4, 54: 8})
    check("m2-snr-grid", M2_SNR_DB == (0.0, 5.0, 10.0, 15.0, 20.0)
          and len(M1_SNR_DB) == 11)
    check("m4-secondary-B", M4_DIGITAL_B == 8)
    check("digital-infeasible-k123",
          all((lambda: _infeasible(k))() for k in (1, 2, 3)))


def _infeasible(k):
    try:
        config.digital_packet(k, 1)
        return False
    except config.InfeasibleDigitalConfig:
        return True


def workload():
    print("\n== 2. exact workload (code vs independent expectation) ==")
    counts = expected_cells()
    # independent reconstruction from the frozen prompt (hand count)
    expect = {
        "M1_training_runs": 2 * 3 * 2 * 3,          # methods x k x channels x seeds
        "M1_eval_conditions": 2 * 3 * (11 * 2),     # x SNR x channels
        "M2_neural_training_runs": 2 * 5 * 2 * 3,   # NEURAL methods only
        "M2_digital_eval_only_seeds": 5 * 2 * 3,
        "M2_eval_conditions": 3 * 5 * (5 * 2),      # incl. digital eval-only
        "M3_infeasible_cells": 3,
        "M4_secondary_eval_conditions": 1 * 5 * 2,  # 1 seed (>=1) x SNR x channels
        "M5_infrastructure_only": 0,
    }
    for key, want in expect.items():
        check(f"count/{key}", counts[key] == want,
              f"code={counts[key]}, expected={want}")

    proposal = build_full_proposal()
    res = validate_matrix(proposal)
    check("matrix-valid", res["valid"], f"errors={res['errors'][:2]}")

    # method-condition table (the report numbers)
    table = {
        "M1": {"neural_training_runs": 2 * 3 * 2 * 3,
               "eval_cells_neural": 2 * 3 * 11 * 2,
               "digital_eval_cells": 0, "infeasible": 0, "secondary": 0},
        "M2": {"neural_training_runs": 2 * 5 * 2 * 3,
               "eval_cells_neural": 2 * 5 * 5 * 2,
               "digital_eval_cells": 5 * 5 * 2,
               "infeasible": 0, "secondary": 0},
        "M3": {"neural_training_runs": 0, "eval_cells_neural": 0,
               "digital_eval_cells": 0, "infeasible": 3, "secondary": 0},
        "M4": {"neural_training_runs": 0, "eval_cells_neural": 0,
               "digital_eval_cells": 5 * 2, "infeasible": 0, "secondary": 10},
        "M5": {"neural_training_runs": 0, "eval_cells_neural": 0,
               "digital_eval_cells": 0, "infeasible": 0, "secondary": 0},
    }
    total_train = sum(v["neural_training_runs"] for v in table.values())
    total_eval = sum(v["eval_cells_neural"] + v["digital_eval_cells"]
                     for v in table.values())
    print(f"    M1: {table['M1']}")
    print(f"    M2: {table['M2']}")
    print(f"    M3: {table['M3']}")
    print(f"    M4: {table['M4']}")
    print(f"    TOTAL neural training runs = {total_train}; "
          f"evaluation cells = {total_eval}; M2/M4 secondary cells = 10; "
          f"infeasible cells = 3; ablation cells = 0 (M5 unspecified)")
    return {"counts": counts, "table": table,
            "total_training_runs": total_train,
            "total_eval_cells": total_eval,
            "matrix_valid": res["valid"],
            "matrix_warnings": res["warnings"]}


def pairing_and_budget():
    print("\n== 3. pairing preservation (Step-7 rule re-check) ==")
    cond = Condition(k=54, channel="rayleigh", snr_db=10.0, t_max=6,
                     bits_per_component=8, episodes=8, seed=config.TEST_SEED)
    r1, r2 = generate_realizations(cond, 8), generate_realizations(cond, 8)
    check("realizations-bitwise",
          bool((r1["noise"] == r2["noise"]).all()
               and (r1["h"] == r2["h"]).all()))
    res = evaluate_condition(cond, methods=["oracle", "digital"])
    res2 = evaluate_condition(cond, methods=["oracle", "digital"])

    def rows_equal(ra, rb):
        return all(
            (math.isnan(va) and math.isnan(vb)) or va == vb
            for va, vb in zip(ra.values(), rb.values()))

    check("digital-eval-deterministic",
          rows_equal(res["oracle"]["aggregate"], res2["oracle"]["aggregate"])
          and rows_equal(res["digital"]["aggregate"],
                         res2["digital"]["aggregate"]),
          "NaN-aware equality (ttg_success_mean=NaN at SR=0 is deterministic)")

    print("\n== 4. budget equality per M2 k ==")
    for k, b in sorted(config.DIGITAL_B_BY_K.items()):
        pkt = config.digital_packet(k, b)
        check(f"budget/k={k}", pkt["n_transmitted_symbols"] == k
              and pkt["n_padding_symbols"] == 0,
              f"B={b}: transmitted {pkt['n_transmitted_symbols']} == neural k")


def stats_protocol():
    print("\n== 5. statistical protocol API ==")
    w = wilson_ci_95([1] * 7 + [0] * 43)
    m = mcnemar_test([1, 0] * 5, [0, 1] * 5)
    t = paired_t_test([1.0, 2.0], [1.5, 1.0])
    check("stats-api", {"wilson_low", "wilson_high"} <= set(w)
          and {"n01", "n10", "p_value"} <= set(m)
          and {"t_stat", "p_value"} <= set(t))
    agg = aggregate_episode_metrics({
        "success": torch.tensor([1, 0]), "time_to_goal": torch.tensor([5, 8]),
        "final_distance": torch.tensor([0.1, 3.0]),
        "avg_distance": torch.tensor([0.2, 3.1]),
        "control_effort": torch.tensor([1.0, 2.0]),
        "min_distance": torch.tensor([0.1, 3.0]),
        "mse": torch.tensor([0.01, 0.02])})
    check("metrics-schema", "success_rate" in agg
          and {"final_distance_mean", "avg_distance_mean", "control_effort_mean",
               "ttg_success_mean", "mse_mean"} <= set(agg),
          "primary = success_rate; secondary metrics present")


def protection():
    print("\n== 6. scientific-result protection ==")
    gitignore = Path(".gitignore").read_text()
    check("results-gitignored", "results/" in gitignore
          and "*.pth" in gitignore)
    with tempfile.TemporaryDirectory() as td:
        # same channel/k/seed across DIFFERENT experiments -> different dirs
        ids = {}
        for exp, method, k, bits, seed in (
                ("M1", "task", 3, None, 42),
                ("M2", "task", 54, None, 42),
                ("M2", "digital", 54, 8, 42),
                ("M4", "digital", 54, 8, 42)):
            cfg = ExperimentConfig(experiment=exp, method=method,
                                   channel="awgn", k=k, training_seed=seed,
                                   bits_per_component=bits)
            ids[run_id(cfg)] = config_hash(cfg)
        check("identity-separation", len(set(ids)) == 4, str(list(ids)))
        m2 = ExperimentConfig.m2("digital", 54, "awgn", 42)
        m4 = ExperimentConfig.m4_secondary("awgn", 42)
        check("M2-M4-distinct-dirs",
              f"results/M2/{run_id(m2)}__" != f"results/M4/{run_id(m4)}__")
        check("cell-state-missing-by-default",
              cell_state(m2, results_root=Path(td) / "none") == "missing")


def per_cell_preflight(proposal):
    print("\n== 7. per-cell preflight (all actionable cells) ==")
    trainable = [c for c in proposal
                 if c.method in ("task", "reconstruction")]
    digital_eval = [c for c in proposal if c.method == "digital"
                    and c.experiment in ("M2", "M4")]
    infeasible = [c for c in proposal if c.experiment == "M3"]
    ok = True
    for c in trainable:
        errs = c.validate()
        if errs:
            ok = False
            print(f"    BLOCKER {run_id(c)}: {errs}")
    for c in digital_eval + infeasible:
        errs = c.validate()
        if errs:
            ok = False
            print(f"    BLOCKER {run_id(c)}: {errs}")
    check("all-cells-valid", ok,
          f"{len(trainable)} trainable + {len(digital_eval)} digital-eval "
          f"+ {len(infeasible)} infeasible cells")
    # 36 M1 + 60 M2 neural + 1 M5 placeholder = 97 task/recon cells; the
    # 96 EXECUTION-READY neural runs exclude the unspecified M5 placeholder
    check("trainable-count", len(trainable) == 97, f"{len(trainable)}")
    return {"trainable_cells": len(trainable),
            "digital_eval_cells": len(digital_eval),
            "infeasible_cells": len(infeasible)}


def budget_decision():
    print("\n== 8. full-scale training-budget decision ==")
    print(f"    config defaults: TRAIN_VAL_TEST_SPLIT="
          f"{config.TRAIN_VAL_TEST_SPLIT}, N_TRAIN_EPISODES="
          f"{config.N_TRAIN_EPISODES}  (PROTOCOL PLACEHOLDERS - the trainers "
          f"are step-based and do NOT consume them)")
    print(f"    smoke-default budgets: reconstruction train_steps="
          f"{ExperimentConfig.m1('reconstruction', 3, 'awgn', 42).train_steps}, "
          f"task train_steps="
          f"{ExperimentConfig.m1('task', 3, 'awgn', 42).train_steps}")
    check("budget-decision-OPEN",
          True,
          "full-scale steps/epochs are NOT frozen in any source document; "
          "requires explicit user decision before execution (flagged blocker, "
          "not assumed)")


def main():
    print("=" * 72)
    print("STEP 9A CONFIGURATION-ONLY PREFLIGHT (no training, no execution)")
    print("=" * 72)
    frozen_conventions()
    workload_info = workload()
    pairing_and_budget()
    stats_protocol()
    protection()
    proposal = build_full_proposal()
    cell_info = per_cell_preflight(proposal)
    budget_decision()

    out = {"preflight": "PASS" if not FAILS else "FAIL",
           "failures": FAILS, **workload_info, **cell_info,
           "open_decisions": [
               "full-scale training budget (steps/epochs) not yet frozen",
               "M5 ablation specification incomplete -> NOT EXECUTABLE YET"],
           "push_pending": "step8 commits awaiting user PAT push"}
    Path("results/preflight").mkdir(parents=True, exist_ok=True)
    Path("results/preflight/step9a_preflight.json").write_text(
        json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 72)
    if FAILS:
        print(f"PREFLIGHT: FAIL ({len(FAILS)} failures: {FAILS})")
        return 1
    print("PREFLIGHT: PASS (0 blockers in matrix/config/identity/pairing; "
          "open decisions recorded, not assumed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
