"""Stable experiment/run identity (Step 9A recovery design, section 14).

One canonical identifier per experiment cell:

    run_id      = "<experiment>_<method>_<channel>_k<k>_seed<seed>"
    config_hash = sha256(sorted canonical config JSON), first 16 hex

Output/checkpoint layout is derived ONLY from these, so recovery logic is
pure filesystem semantics:

    completed run  -> DONE marker present                -> skip
    interrupted    -> RUNNING/FAILED marker, no DONE     -> resume
    missing        -> nothing on disk                    -> train

Markers (json) live next to the artifacts:
    <run_id>__<config_hash>/.json suffixes:
        .DONE  .RUNNING  .FAILED  .ckpt.meta.json
Scientific outputs live in per-cell result directories
(results/<experiment>/<run_id>__<hash>/) so different cells can never
overwrite each other (section 19).

The run_id vocabulary is exactly the ExperimentConfig vocabulary; the
config hash covers the full training + evaluation configuration so a
changed protocol changes the identity (stale outputs are never reused).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional

from project import config
from project.experiments.config import ExperimentConfig


def canonical_config(cfg: ExperimentConfig) -> Dict:
    """The full configuration that determines a run's identity."""
    return {
        "experiment": cfg.experiment,
        "method": cfg.method,
        "channel": cfg.channel,
        "k": cfg.k,
        "bits_per_component": cfg.bits_per_component,
        "training_seed": cfg.training_seed,
        "test_seed": cfg.test_seed,
        "train_steps": cfg.train_steps,
        "batch_episodes": cfg.batch_episodes,
        "batch_states": cfg.batch_states,
        "lr": cfg.lr if cfg.lr is not None else config.LEARNING_RATE,
        "train_snr_policy": cfg.train_snr_policy,
        "eval_snr_grid_db": list(cfg.eval_snr_grid_db),
        "t_max": cfg.t_max,
        "eval_snr_db": cfg.eval_snr_db,
        "n_train_episodes": cfg.n_train_episodes,
        "n_test_episodes": cfg.n_test_episodes,
        "eval_episodes": cfg.eval_episodes,
    }


def config_hash(cfg: ExperimentConfig) -> str:
    payload = json.dumps(canonical_config(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_id(cfg: ExperimentConfig) -> str:
    return (f"{cfg.experiment}_{cfg.method}_{cfg.channel}"
            f"_k{cfg.k}_seed{cfg.training_seed}")


def run_dir(cfg: ExperimentConfig, results_root: str = "results") -> Path:
    """results/<experiment>/<run_id>__<config_hash>/"""
    d = Path(results_root) / cfg.experiment / f"{run_id(cfg)}__{config_hash(cfg)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# lifecycle markers
# ---------------------------------------------------------------------------

def _marker_path(cfg: ExperimentConfig, kind: str, results_root: str) -> Path:
    return run_dir(cfg, results_root) / f"{kind}.json"


def mark_running(cfg: ExperimentConfig, results_root: str = "results") -> Path:
    p = _marker_path(cfg, "RUNNING", results_root)
    p.write_text(json.dumps({"run_id": run_id(cfg),
                             "config_hash": config_hash(cfg),
                             "started_utc": time.strftime(
                                 "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return p


def mark_done(cfg: ExperimentConfig, summary: Optional[Dict] = None,
              results_root: str = "results") -> Path:
    p = _marker_path(cfg, "DONE", results_root)
    p.write_text(json.dumps({"run_id": run_id(cfg),
                             "config_hash": config_hash(cfg),
                             "finished_utc": time.strftime(
                                 "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "summary": summary or {}}))
    return p


def mark_failed(cfg: ExperimentConfig, reason: str,
                results_root: str = "results") -> Path:
    p = _marker_path(cfg, "FAILED", results_root)
    p.write_text(json.dumps({"run_id": run_id(cfg),
                             "config_hash": config_hash(cfg),
                             "reason": str(reason)[:400]}))
    return p


def cell_state(cfg: ExperimentConfig, results_root: str = "results") -> str:
    """\"done\" | \"failed\" | \"running\" | \"missing\" from the markers only.

    Completion is decided by the DONE marker alone (a stale RUNNING file
    from a crashed process does not make a finished run look unfinished,
    and a DONE marker can never be mistaken for an unfinished one).
    """
    root = Path(results_root) / cfg.experiment
    for state in ("done", "failed", "running"):
        if (root / f"{run_id(cfg)}__{config_hash(cfg)}"
                / f"{state.upper()}.json").exists():
            return state
    return "missing"
