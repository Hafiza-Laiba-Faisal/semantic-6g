"""Experiment runner (Step 8: dry-run + smoke only).

Flow for one cell: validate -> (dry_run | smoke) via ExperimentConfig.run.
Full-scale execution is rejected here as a second line of defense after
the config-level guard. The M1-M5 proposal validator lives in matrix.py.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from project.experiments.config import ExperimentConfig
from project.experiments.matrix import validate_matrix


def validate_config(cfg: ExperimentConfig) -> Dict:
    """Validate one configuration and return the structured summary."""
    errors = cfg.validate()
    return {"valid": len(errors) == 0, "errors": errors,
            "summary": cfg.dry_run_summary()}


def dry_run(cfg: ExperimentConfig) -> Dict:
    """Configuration-only dry run: NO training, NO side effects.

    The summary describes the DRY-RUN variant of the cell (run_mode is
    reported as ``dry_run`` and ``training_will_occur`` is False), so the
    printed plan can never be mistaken for an executing configuration.
    """
    dry = replace(cfg, run_mode="dry_run")
    errors = dry.validate()
    return {"status": "dry_run",
            "valid": len(errors) == 0,
            "errors": errors,
            "summary": dry.dry_run_summary()}


def run_smoke(cfg: ExperimentConfig) -> Dict:
    """Validate then execute one cell in smoke mode (non-scientific)."""
    cfg.validate_or_raise()          # rejects run_mode='full'
    out = cfg.run()
    out["non_scientific"] = True
    out["note"] = ("Step-8 smoke validation of infrastructure only; "
                   "not an experimental result")
    return out


def run_full(cfg: ExperimentConfig) -> Dict:
    """Explicitly blocked: full-scale execution needs user approval."""
    return {"status": "blocked",
            "reason": ("full-scale M1-M5 execution is not approved; "
                       "Step 8 is infrastructure + smoke validation only")}


def validate_matrix_proposal(cells: List[ExperimentConfig]) -> Dict:
    """Validate a proposed M1-M5 cell list without executing it."""
    return validate_matrix(cells)
