"""Frozen experiment-matrix validation (M1-M5) - structural only.

This module VALIDATES the frozen matrix design (audit sections 8.5-8.7).
It never executes experiments. Detected problems:

* invalid k / invalid B / digital budget mismatch (6B + 6 <= k)
* wrong SNR grid per matrix row
* missing/wrong training seeds (must be exactly {42, 43, 44})
* channel outside {AWGN, Rayleigh}
* M2/M4 confusion: any cell labeling the B = 8 secondary reference as
  primary, or giving primary cells a digital budget other than B = (k-6)/6
* M3 cells that are unexpectedly FEASIBLE (they must stay infeasible)
* unsupported method/channel/experiment combinations

The budget arithmetic is the frozen ``config.digital_packet`` via
``config.digital_feasible`` - no duplicated mathematics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from project import config
from project.experiments.config import (
    CHANNELS,
    M1_K_VALUES,
    M1_SNR_DB,
    M2_B_BY_K,
    M2_K_VALUES,
    M2_SNR_DB,
    M4_DIGITAL_B,
    M4_K,
    TRAINING_SEEDS,
    ExperimentConfig,
    digital_feasible,
)


def matrix_digital_feasible(k: int, bits_per_component) -> Tuple[bool, str]:
    """Backward-compatible alias for the frozen budget check."""
    return digital_feasible(k, bits_per_component)


def expected_cells() -> Dict[str, int]:
    """Frozen cell counts per matrix row (structural expectation, audit 8.7).

    Training-run counts: a training run is per (method, k, channel, seed);
    the SNR grid enters at evaluation (one fixed SNR per condition), so it
    multiplies the EVALUATION conditions, not the training runs.
    """
    n_eval_m1 = len(M1_SNR_DB) * len(CHANNELS)
    n_eval_m2 = len(M2_SNR_DB) * len(CHANNELS)
    return {
        # 2 methods x 3 k x 2 channels x 3 seeds
        "M1_training_runs": 2 * len(M1_K_VALUES) * len(CHANNELS) * len(TRAINING_SEEDS),
        "M1_eval_conditions": 2 * len(M1_K_VALUES) * n_eval_m1,
        # 3 methods x 5 k x 2 channels x 3 seeds
        "M2_training_runs": 3 * len(M2_K_VALUES) * len(CHANNELS) * len(TRAINING_SEEDS),
        "M2_eval_conditions": 3 * len(M2_K_VALUES) * n_eval_m2,
        "M3_infeasible_cells": len((1, 2, 3)),          # k in {1, 2, 3}
        # SECONDARY digital reference: both channels x SNR grid (>= 1 seed)
        "M4_secondary_eval_conditions": 2 * len(M2_SNR_DB),
        # ablation infrastructure only; cells defined later per ablation
        "M5_infrastructure_only": 0,
    }


def validate_matrix(cells: List[ExperimentConfig]) -> Dict:
    """Validate a proposed cell list against the frozen M1-M5 design.

    Returns ``{"errors": [...], "warnings": [...], "counts": {...},
    "valid": bool}``. No experiment is executed.
    """
    errors: List[str] = []
    warnings: List[str] = []

    for i, cell in enumerate(cells):
        tag = (f"cell[{i}] {cell.experiment}/{cell.method}/"
               f"{cell.channel}/k={cell.k}")
        # generic config validation (run_mode, seeds, method set, digital
        # feasibility; M3 expected-infeasible handled inside validate())
        for err in cell.validate():
            errors.append(f"{tag}: {err}")

        if cell.experiment == "M1":
            if cell.method not in ("task", "reconstruction"):
                errors.append(f"{tag}: M1 is neural task-vs-reconstruction only")
            if cell.k not in M1_K_VALUES:
                errors.append(f"{tag}: M1 k must be in {M1_K_VALUES}")
            if any(s not in M1_SNR_DB for s in cell.eval_snr_grid_db):
                errors.append(f"{tag}: M1 SNR grid must be {M1_SNR_DB}")
            if cell.method == "digital":
                errors.append(f"{tag}: M2/M4 confusion - digital does not "
                              f"belong in M1 (primary neural ablation)")

        elif cell.experiment == "M2":
            if cell.method not in ("task", "reconstruction", "digital"):
                errors.append(f"{tag}: M2 methods are digital+task+reconstruction")
            if cell.k not in M2_K_VALUES:
                errors.append(f"{tag}: M2 k must be in {M2_K_VALUES}")
            if any(s not in M2_SNR_DB for s in cell.eval_snr_grid_db):
                errors.append(f"{tag}: M2 SNR grid must be {M2_SNR_DB}")
            if (cell.method == "digital"
                    and cell.bits_per_component != M2_B_BY_K.get(cell.k)):
                errors.append(
                    f"{tag}: M2 digital budget mismatch - expected "
                    f"B={M2_B_BY_K.get(cell.k)} for k={cell.k}, got "
                    f"{cell.bits_per_component}")
            # M2/M4 confusion: primary cells must NOT carry the secondary
            # budget (B = 8 belongs to M4 only; only k=54 coincides with it)
            if (cell.method == "digital"
                    and cell.bits_per_component == M4_DIGITAL_B
                    and cell.k != M4_K):
                errors.append(f"{tag}: M2/M4 confusion - B=8 is the M4 "
                              f"SECONDARY reference, not a primary budget")

        elif cell.experiment == "M3":
            if cell.method != "digital":
                errors.append(f"{tag}: M3 cells are digital-only")
            if cell.k not in (1, 2, 3):
                errors.append(f"{tag}: M3 cells are exactly k in {{1, 2, 3}}")
            feasible, reason = digital_feasible(cell.k, cell.bits_per_component)
            if feasible:
                errors.append(f"{tag}: M3 cell unexpectedly feasible - {reason}")
            # expected-infeasible is recorded as an explicit warning-level
            # note, never silently dropped
            if not feasible and cell.run_mode not in ("dry_run",):
                warnings.append(f"{tag}: M3 cell executes nothing - digital is "
                                f"infeasible here by design ({reason})")

        elif cell.experiment == "M4":
            if cell.method != "digital" or cell.bits_per_component != M4_DIGITAL_B:
                errors.append(f"{tag}: M4 is the SECONDARY digital reference "
                              f"with B={M4_DIGITAL_B} (n_dig={M4_K}) only")
            if cell.k != M4_K:
                errors.append(f"{tag}: M4 must use n_dig={M4_K} channel uses, "
                              f"got k={cell.k}")
            if any(s not in M2_SNR_DB for s in cell.eval_snr_grid_db):
                errors.append(f"{tag}: M4 SNR grid must be {M2_SNR_DB}")
            # unconditional SECONDARY-label reminder (audit 8.7: labeled in
            # every table/legend); reported once per cell
            warnings.append(f"{tag}: M4 must remain labeled SECONDARY in "
                            f"every table/legend (audit 8.7)")

        elif cell.experiment == "M5":
            warnings.append(f"{tag}: M5 is ablation infrastructure only - "
                            f"no ablation cells are invented in this step")

        else:
            errors.append(f"{tag}: unknown experiment {cell.experiment!r}")

    # cross-cell consistency: M1/M2 seeds must cover exactly the frozen set
    for row in ("M1", "M2"):
        seeds = sorted({c.training_seed for c in cells
                        if c.experiment == row})
        if seeds and seeds != sorted(TRAINING_SEEDS):
            errors.append(f"{row}: training seeds {seeds} must be exactly "
                          f"{sorted(TRAINING_SEEDS)}")
        chans = {c.channel for c in cells if c.experiment == row}
        if chans and chans != set(CHANNELS):
            errors.append(f"{row}: channels {sorted(chans)} must cover exactly "
                          f"{sorted(CHANNELS)}")

    return {
        "errors": errors,
        "warnings": warnings,
        "counts": expected_cells(),
        "valid": len(errors) == 0,
    }
