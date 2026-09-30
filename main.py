"""Top-level entry point (Step 8: dry-run / smoke infrastructure only).

Usage:
    python main.py config                  # print the frozen configuration
    python main.py dry-run                 # configuration-only dry run of
                                           # representative M1/M2/M3/M4/M5 cells
                                           # (NO training, NO side effects)
    python main.py validate-matrix         # structural M1-M5 validation of the
                                           # full frozen proposal (no execution)
    python main.py smoke <exp> <method> <channel> <k> <seed>
                                           # one smoke cell, e.g.
                                           #   python main.py smoke M1 reconstruction awgn 3 42
                                           #   python main.py smoke M2 task rayleigh 54 42
    python main.py full ...                # REFUSED: needs explicit approval
"""

from __future__ import annotations

import json
import sys

from project import config
from project.experiments.config import (
    CHANNELS,
    M1_K_VALUES,
    M2_K_VALUES,
    TRAINING_SEEDS,
    ExperimentConfig,
)
from project.experiments.runner import dry_run, run_full, run_smoke, validate_matrix_proposal


def _matrix_proposal():
    """The frozen M1-M5 proposal as a cell list (validated, never executed).

    M1: task+reconstruction x k {1,2,3} x {AWGN,Rayleigh} x seeds {42,43,44}
    M2: digital+task+reconstruction x k {12,18,24,30,54} x channels x seeds
        (digital B = (k-6)/6 per audit 8.6)
    M3: digital infeasibility cells k {1,2,3} (expected infeasible)
    M4: SECONDARY digital reference B=8 (n_dig=54), one cell per channel
    M5: ablation infrastructure placeholder (no cells invented)
    """
    cells = []
    for method in ("task", "reconstruction"):
        for k in M1_K_VALUES:
            for ch in CHANNELS:
                for seed in TRAINING_SEEDS:
                    cells.append(ExperimentConfig.m1(
                        method, k, ch, seed, run_mode="dry_run"))
    for method in ("digital", "task", "reconstruction"):
        for k in M2_K_VALUES:
            for ch in CHANNELS:
                for seed in TRAINING_SEEDS:
                    cells.append(ExperimentConfig.m2(
                        method, k, ch, seed, run_mode="dry_run"))
    for k in (1, 2, 3):
        cells.append(ExperimentConfig.m3(k, bits=1, run_mode="dry_run"))
    for ch in CHANNELS:
        cells.append(ExperimentConfig.m4_secondary(ch, seed=42,
                                                   run_mode="dry_run"))
    cells.append(ExperimentConfig(experiment="M5", method="task",
                                  channel="awgn", k=3, training_seed=42,
                                  run_mode="dry_run"))
    return cells


def _representative_cells():
    """One compact representative cell per matrix row (for the dry run)."""
    return [
        ExperimentConfig.m1("reconstruction", 3, "awgn", 42, run_mode="dry_run"),
        ExperimentConfig.m1("task", 1, "rayleigh", 43, run_mode="dry_run"),
        ExperimentConfig.m2("digital", 24, "awgn", 42, run_mode="dry_run"),
        ExperimentConfig.m2("task", 54, "rayleigh", 44, run_mode="dry_run"),
        ExperimentConfig.m3(2, bits=1, run_mode="dry_run"),
        ExperimentConfig.m4_secondary("awgn", 42, run_mode="dry_run"),
        ExperimentConfig(experiment="M5", method="task", channel="awgn",
                         k=3, training_seed=42, run_mode="dry_run"),
    ]


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 0
    cmd = argv[1]

    if cmd == "config":
        print(config.describe())
        return 0

    if cmd == "dry-run":
        for cell in _representative_cells():
            out = dry_run(cell)
            s = out["summary"]
            feas = s["digital_feasibility"]
            if not out["valid"]:
                status = f"INVALID {out['errors']}"
            elif feas and feas["expected_infeasible"] and not feas["feasible"]:
                status = "EXPECTED-INFEASIBLE (M3, by design)"
            else:
                status = "VALID"
            pkt = s["digital_packet"]
            pkt_note = ""
            if pkt:
                if pkt.get("feasible") is False:
                    pkt_note = f"  [INFEASIBLE: {pkt['note'][:52]}]"
                else:
                    pkt_note = (f" 6B={pkt['n_src_bits']} info={pkt['n_info_bits']}"
                                f" payload={pkt['n_payload_symbols']}"
                                f" pad={pkt['n_padding_symbols']}")
            print(f"[{status}] {s['experiment']:3s} {s['method']:14s} "
                  f"{s['channel']:8s} k={s['k']:2d} rho={s['rho']:.3f} "
                  f"seed={s['training_seed']} "
                  f"SNR={s['evaluation_config']['eval_snr_grid_db'][:3]}..."
                  f"{pkt_note}")
        return 0

    if cmd == "validate-matrix":
        proposal = _matrix_proposal()
        result = validate_matrix_proposal(proposal)
        print(json.dumps({"n_cells": len(proposal),
                          "valid": result["valid"],
                          "errors": result["errors"],
                          "warnings": result["warnings"],
                          "expected_counts": result["counts"]}, indent=2))
        return 0 if result["valid"] else 1

    if cmd == "smoke":
        if len(argv) < 7:
            print(__doc__)
            return 1
        experiment, method, channel, k, seed = (
            argv[2].upper(), argv[3], argv[4], int(argv[5]), int(argv[6]))
        bits = None
        if method == "digital":
            bits = config.DIGITAL_B_BY_K.get(k)
            if bits is None:
                print(f"REFUSED: digital is infeasible at k={k} "
                      f"(6B + 6 <= k; audit 8.5). This is the M3 "
                      f"infeasibility cell - nothing to execute. "
                      f"Matched-budget k values: {sorted(config.DIGITAL_B_BY_K)}.")
                return 2
        # smoke-scale protocol (infrastructure validation only, NON-scientific)
        cfg = ExperimentConfig(
            experiment=experiment, method=method, channel=channel, k=k,
            training_seed=seed, bits_per_component=bits,
            train_steps=20, batch_episodes=64, batch_states=2048,
            eval_snr_grid_db=(5.0, 10.0), t_max=15, eval_snr_db=10.0,
            n_test_episodes=25, eval_episodes=64, run_mode="smoke")
        out = run_smoke(cfg)
        print(json.dumps({key: out[key]
                          for key in ("status", "non_scientific", "note")},
                         indent=2))
        print("train:", json.dumps(out["train"], default=str)[:600]
              if out["train"] else "none (digital/oracle)")
        print("eval rows:", json.dumps(out["eval"], default=str)[:1200])
        print("checkpoint:", out["checkpoint"])
        return 0

    if cmd == "full":
        run_full(None)   # records the structured refusal
        print("REFUSED: full-scale M1-M5 execution requires explicit user "
              "approval. Step 8 is infrastructure + smoke validation only.")
        return 2

    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
