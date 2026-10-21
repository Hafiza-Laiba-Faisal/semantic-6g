"""Wave-A diagnostic ablation runner (post-freeze amendment; M5-expanded).

GUARDRAILS (read this first):
* The frozen Step-9C protocol and ALL M1-M4 artifacts are READ-ONLY.
  Every Wave-A cell lives under results/WAVE_A/<wave_run_id>__<wave_hash>/
  where the id AND hash embed the arm tag + non-frozen knobs, so Wave-A
  outputs can never collide with, overwrite, or reuse a frozen run.
  project/experiments/identity.py untouched; wave identity is local here.
* Ablation values (lambdas, state weights, extra seeds, extended steps,
  hidden width, fixed-SNR curriculum) are PASSED AS PARAMETERS;
  project/config.py and the ExperimentConfig dataclass are untouched.
* Wave-A evaluation uses a REDUCED 1000-episode paired grid and is a
  COMPARATIVE DIAGNOSTIC ONLY: no frozen M1/M2 number may be re-derived
  from it; every table/figure built from Wave-A data must caption
  "Wave-A diagnostic protocol: E=1000, non-frozen amendment 2026-10-12".

Arms (fyp_md_files/WAVE_A_DESIGN.md + W-A4 figure-diagnosis amendments):
  A1 lam_ctl_*   task + non-zero (lu, lv) lambda ablation    [supervisor: loss ablation]
  A2 wrecon_*    task-aware weighted reconstruction          [supervisor: objective ablation, ATROC-lite]
  A3 seedvar_*   extra seeds {45,46,47}                      [supervisor: instability variance]
  A4 long_*      extended 12000-step task training           [figure diagnosis: convergence]
  A4 fixsnr_*    fixed-SNR (10 dB) curriculum task training   [figure diagnosis: SNR-mixing isolation]
  A4 caps_*      capacity ablation (hidden 256)              [supervisor: model capacity]

Usage (other PC):
    python tools/run_wave_a.py verify      # hard gate: frozen infra assumptions
    python tools/run_wave_a.py plan        # full cell list + runtime estimate
    python tools/run_wave_a.py status      # done/failed/missing census
    python tools/run_wave_a.py train       # all runs, idempotent (SKIP done/failed)
    python tools/run_wave_a.py train awgn  # lane filter: awgn | rayleigh
    python tools/run_wave_a.py evaluate    # 1000-episode paired Wave-A evals
    python tools/run_wave_a.py finalize    # results/WAVE_A/step_wave_a_summary.json
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from project import config
from project.evaluation.evaluate import Condition, evaluate_condition
from project.evaluation.statistics import (mcnemar_test, paired_t_test,
                                           wilson_ci_95)
from project.experiments.config import ExperimentConfig
from project.models.reconstruction_jscc import ReconstructionDeepJSCC
from project.training.common import save_checkpoint
from project.training.train_reconstruction import train_reconstruction
from project.training.train_task_oriented import train_task_oriented

WAVE_A_ROOT = Path("results/WAVE_A")
CHANNELS = ("awgn", "rayleigh")
MATCH_K = (1, 12)             # one low-k (M1 regime) + one matched-budget (M2 regime)
MATCH_SNR_DB = (0.0, 5.0, 10.0, 15.0, 20.0)
FROZEN_SEEDS = (42, 43, 44)
EXTRA_SEEDS = (45, 46, 47)
EVAL_EPISODES = 1000          # reduced diagnostic grid (frozen protocol uses 5000)
LONG_STEPS = 4 * config.TASK_TRAIN_STEPS      # 12000
CAP_HIDDEN = 256
GENERIC_CELL_BUDGET_STEPS = 500               # ExperimentConfig-required placeholder
                                              # (real budget lives in wave spec)
FAILS = []
_OK = {"n": 0}
_BAD = {"n": 0}

INVALID_ARMS = {
    # Verified from the checkpoint state_dict on 2026-10-21: metadata claims
    # hidden=256 but every layer is 128-wide (encoder.backbone.0.weight is
    # (128, 6), max width 128 - identical to the 128-wide control arm), because
    # the encoder default width is bound at import time and the later patch has
    # no effect.  The arm is NOT a capacity ablation and must not be cited as
    # one.  See fyp_md_files/PROTOCOL_AMENDMENTS.md A-6.
    "caps_h256": ("trained artifact is 128-wide (metadata says 256); "
                  "not a capacity ablation - A-6"),
}

# --- documented-failure contract -------------------------------------------
# Two Wave-A training cells are frozen, documented FAILURES (both task cells
# exhausted the numerical headroom of the frozen config: "incomplete
# numerical state").  They were NOT retrained (retraining a failure because
# the result was bad would be survivorship bias) and their directories are
# evidence-frozen under C:\wa_evidence\WAVE_A_2026-10-20T1645Z\.
#
# Rule: a queue that ends with ONLY these declared cells failing is a
# COMPLETED stage, not a broken one.  It exits with DOCUMENTED_FAILURE_EXIT
# (4) so tools/run_all.py can mark the stage "accepted" instead of "failed"
# and does not burn a retry re-running a 12-hour deterministic retrain that
# cannot change the outcome.  ANY undeclared failure still exits 1.
DOCUMENTED_FAILURES = {
    # base wave_id (directory = <wave_id>__<hash>)
    "WAVE_A_long_12k_task_awgn_k12_seed42",
    "WAVE_A_seedvar_frozen_task_awgn_k12_seed45",
}
DOCUMENTED_FAILURE_EXIT = 4


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]" + (f"  ({detail})" if detail else ""))
    if ok:
        _OK["n"] += 1
    else:
        _BAD["n"] += 1
        FAILS.append(name)


# ---------------------------------------------------------------------------
# wave cell definition + tag-aware identity (local; identity.py untouched)
# ---------------------------------------------------------------------------

class WaveCell:
    """One Wave-A run: base frozen-protocol scalars + non-frozen overrides."""

    def __init__(self, arm, method, channel, k, seed, *, tag,
                 steps=None, lambdas=None, state_weights=None,
                 hidden=None, fixed_snr=None):
        self.arm = arm                  # "A1" | "A2" | "A3" | "A4"
        self.method = method            # "task" | "recon_w"
        self.channel = channel
        self.k = k
        self.seed = seed
        self.tag = tag
        self.steps = steps if steps is not None else (
            config.RECON_TRAIN_STEPS if method == "recon_w"
            else config.TASK_TRAIN_STEPS)
        self.lambdas = lambdas          # None or (ld, lu, lv, lT)
        self.state_weights = state_weights  # None or 6-tuple
        self.hidden = hidden            # None = frozen 128
        self.fixed_snr = fixed_snr      # None = frozen U(0,20) policy
        self.wave_id = (f"WAVE_A_{self.tag}_{self.method}_{self.channel}"
                        f"_k{k}_seed{seed}")

    def spec(self):
        """Canonical wave config (everything that defines identity)."""
        return {
            "experiment": "WAVE_A",
            "tag": self.tag,
            "arm": self.arm,
            "method": self.method,
            "channel": self.channel,
            "k": self.k,
            "seed": self.seed,
            "train_steps": self.steps,
            "batch_states": config.BATCH_STATES,
            "batch_episodes": config.BATCH_EPISODES,
            "t_max": config.TASK_TRAINING_T,
            "lr": config.LEARNING_RATE,
            "grad_clip": config.GRAD_CLIP_NORM,
            "train_snr_policy": ({"distribution": "fixed", "snr_db": self.fixed_snr}
                                 if self.fixed_snr is not None else
                                 {"distribution": "uniform",
                                  "range_db": list(config.SNR_TRAIN_RANGE_DB),
                                  "granularity": "per_batch"}),
            "lambdas": list(self.lambdas) if self.lambdas else None,
            "state_weights": (list(self.state_weights)
                              if self.state_weights else None),
            "hidden": self.hidden if self.hidden else list(config.HIDDEN_DIMS)[0],
            "eval_snr_grid_db": list(MATCH_SNR_DB),
            "eval_episodes": EVAL_EPISODES,
            "test_seed": config.TEST_SEED,
            "amendment": "WAVE_A-non-frozen-2026-10-12",
        }

    def wave_hash(self):
        payload = json.dumps(self.spec(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def run_dirname(self):
        return f"{self.wave_id}__{self.wave_hash()}"

    def dir(self):
        d = WAVE_A_ROOT / self.run_dirname()
        d.mkdir(parents=True, exist_ok=True)
        return d

    # marker files (same DONE > FAILED > RUNNING precedence as identity.py)
    def _mpath(self, kind):
        return self.dir() / f"{kind}.json"

    def mark_running(self):
        # RUNNING.json is a HISTORICAL START MARKER: it is written at the
        # start of every attempt and intentionally retained after completion
        # so the start time survives for provenance.  It is NOT a liveness
        # signal; state() resolves DONE > FAILED > RUNNING, and tooling must
        # never count a bare RUNNING marker as "active".
        self._mpath("RUNNING").write_text(json.dumps(
            {"wave_id": self.wave_id, "spec": self.spec(),
             "started_utc": _utc()}))

    def mark_done(self, summary):
        self._mpath("DONE").write_text(json.dumps(
            {"wave_id": self.wave_id, "finished_utc": _utc(),
             "summary": summary}))
        # Defensive: a DONE cell must never also carry a FAILED marker.
        stale = self._mpath("FAILED")
        if stale.exists():
            stale.unlink()

    def mark_failed(self, reason):
        # DONE is terminal: never overwrite a completed run with a failure
        # marker (the training queue skips DONE cells, so this is defensive).
        if self._mpath("DONE").exists():
            return
        self._mpath("FAILED").write_text(json.dumps(
            {"wave_id": self.wave_id, "reason": str(reason)[:400]}))

    def state(self):
        for st in ("done", "failed", "running"):
            if self._mpath(st.upper()).exists():
                return st
        return "missing"

    def load_ckpt(self):
        ckpt = self.dir() / "final_ckpt.pt"
        if not ckpt.exists():
            return None
        model = ReconstructionDeepJSCC(k=self.k)
        payload = torch.load(ckpt, weights_only=False)
        model.load_state_dict(payload["model_state"])
        return model


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# cell universe
# ---------------------------------------------------------------------------

LAMBDA_TAGS = {                     # arm A1: (ld, lu, lv, lT)
    "lam_ctl_light": (1.0, 0.01, 0.01, 2.0),
    "lam_ctl_heavy": (1.0, 0.10, 0.10, 2.0),
}
STATE_WEIGHT_TAGS = {               # arm A2: per-component [x, y, vx, vy, xg, yg]
    "wrecon_pos": (2.0, 2.0, 1.0, 1.0, 2.0, 2.0),
}


def wave_cells():
    cells = []
    # A1 - lambda ablation: task, match k x channels x frozen seeds
    for tag, lam in LAMBDA_TAGS.items():
        for k in MATCH_K:
            for ch in CHANNELS:
                for seed in FROZEN_SEEDS:
                    cells.append(WaveCell("A1", "task", ch, k, seed,
                                          tag=tag, lambdas=lam))
    # A2 - weighted reconstruction: match k x channels x frozen seeds
    for tag, w in STATE_WEIGHT_TAGS.items():
        for k in MATCH_K:
            for ch in CHANNELS:
                for seed in FROZEN_SEEDS:
                    cells.append(WaveCell("A2", "recon_w", ch, k, seed,
                                          tag=tag, state_weights=w))
    # A3 - seed variance: FROZEN protocol re-run at extra seeds {45,46,47}
    #      (frozen lambdas; only the seed is the ablation; task, k=12, both channels)
    for ch in CHANNELS:
        for seed in EXTRA_SEEDS:
            cells.append(WaveCell("A3", "task", ch, 12, seed, tag="seedvar_frozen"))
    # A4 - convergence/instability arms (k=12, awgn, seed 42 - the diagnosed cell)
    cells.append(WaveCell("A4", "task", "awgn", 12, 42, tag="long_12k",
                          steps=LONG_STEPS))
    cells.append(WaveCell("A4", "task", "awgn", 12, 42, tag="fixsnr_10db",
                          fixed_snr=10.0))
    cells.append(WaveCell("A4", "task", "awgn", 12, 42, tag="caps_h256",
                          hidden=CAP_HIDDEN))
    cells.append(WaveCell("A4", "task", "rayleigh", 12, 42, tag="long_12k",
                          steps=LONG_STEPS))
    return cells


def eval_units():
    """Per (k, channel, seed): paired set of Wave-A trained models.

    Derived from the ACTUAL declared cell universe (``wave_cells``) instead
    of a hand-written cross-product.  A1/A2 cover seeds 42-44 at both
    k in {1,12}; A3 covers extra seeds 45-47 at k=12 only.  The derived
    universe is therefore 18 units / 90 SNR slots (not 24 / 120), which
    keeps the evaluator and the finalize completeness denominator in
    lock-step with the declared experiment.  Failed cells stay in the
    universe; they simply produce no evaluation artifacts.
    """
    seen = {}
    for c in wave_cells():
        key = (c.k, c.channel, c.seed)
        if key in seen:
            continue
        seen[key] = {"unit": f"k{c.k}_{c.channel}_seed{c.seed}",
                     "k": c.k, "channel": c.channel, "seed": c.seed}
    return [seen[key] for key in sorted(seen)]


def _eval_path(unit, snr):
    return WAVE_A_ROOT / "eval" / unit["unit"] / f"eval_{snr:g}.json"


# ---------------------------------------------------------------------------
# governance: verify (hard gate)
# ---------------------------------------------------------------------------

def cmd_verify():
    print("== Wave-A governance gate (frozen infra assumptions) ==")
    check("frozen_9C_intact",
          (config.RECON_TRAIN_STEPS, config.TASK_TRAIN_STEPS)
          == (5000, 3000)
          and config.LAMBDA_DISTANCE == 1.0
          and config.LAMBDA_CONTROL == 0.0
          and config.LAMBDA_VELOCITY == 0.0
          and config.LAMBDA_TERMINAL == 2.0
          and config.TRAINING_SEEDS == (42, 43, 44)
          and config.TEST_SEED == 10042,
          "config.py defaults unchanged")
    check("frozen_M1_artifacts_present",
          len(list(Path("results/M1").glob("*__*/DONE.json"))) == 36,
          "36 training runs")
    try:
        from run_step11_m2 import CONDITIONAL_ACCEPT as _M2_CONDITIONAL_ACCEPT
    except ImportError:                      # standalone invocation fallback
        _M2_CONDITIONAL_ACCEPT = frozenset()
    _m2_done = {p.parent.name
                for p in list(Path("results/M2").glob("*__*/DONE.json"))}
    # run dirs may exist under multiple hash stamps; match prefix.
    _m2_cond = sum(1 for cid in _M2_CONDITIONAL_ACCEPT
                   if cid not in _m2_done
                   and (Path("results/M2") / cid).exists())
    check("frozen_M2_artifacts_present",
          len(_m2_done) + _m2_cond >= 58,
          f"{len(_m2_done)} done + {_m2_cond} conditional-accept "
          f">=58 of 60 training runs")
    # trainer amendment surface exists (additive, default-identity)
    import inspect
    sig_task = inspect.signature(train_task_oriented)
    sig_rec = inspect.signature(train_reconstruction)
    check("task_trainer_has_lambdas", "lambdas" in sig_task.parameters)
    check("recon_trainer_has_state_weights",
          "state_weights" in sig_rec.parameters)
    check("M2_round_never_touched_by_wave_roots",
          WAVE_A_ROOT != Path("results/M2") and WAVE_A_ROOT != Path("results/M1"),
          str(WAVE_A_ROOT))
    # no frozen run_id shape can appear under WAVE_A (prefix partition check)
    frozen_prefixes = ("M1_", "M2_", "M3_", "M4_", "M5_")
    check("wave_ids_cannot_collide_with_frozen",
          all(not c.wave_id.startswith(p) for c in wave_cells()
              for p in frozen_prefixes))
    cfg = ExperimentConfig.m1("reconstruction", 3, "awgn", 42, run_mode="dry_run")
    from project.experiments.identity import config_hash as frozen_hash
    check("frozen_identity_fn_untouched_fingerprint",
          isinstance(frozen_hash(cfg), str) and len(frozen_hash(cfg)) == 16)
    print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    return 1 if FAILS else 0


# ---------------------------------------------------------------------------
# plan / status
# ---------------------------------------------------------------------------

def _runtime_estimate_s(cell):
    # measured from frozen M2: task ~0.37 s/step, recon ~0.004 s/step (k=12 CPU)
    per_step = 0.37 if cell.method == "task" else 0.004
    return int(cell.steps * per_step)


def cmd_plan():
    cells = wave_cells()
    print("== Wave-A execution plan (deterministic order; NO execution) ==")
    total_s = 0
    for i, c in enumerate(cells, 1):
        est = _runtime_estimate_s(c)
        total_s += est
        print(f"[{i:2d}/{len(cells)}] {c.wave_id:44s} steps={c.steps:5d} "
              f"lam={c.lambdas} w={c.state_weights} h={c.hidden} "
              f"snr={c.fixed_snr} ~{est/60:.0f}min")
    units = eval_units()
    print(f"-- training: {len(cells)} runs; est wall ~{total_s/3600:.1f} h (CPU)")
    print(f"-- eval: {len(units)} units x {len(MATCH_SNR_DB)} SNRs x "
          f"{EVAL_EPISODES} episodes (Wave-A diagnostic grid, non-frozen)")
    return 0


def cmd_status():
    print("== Wave-A status ==")
    counts = {"done": 0, "failed": 0, "running": 0, "missing": 0}
    for c in wave_cells():
        st = c.state()
        counts[st] += 1
        print(f"{st:8s} {c.wave_id}")
    ev_done = sum(1 for u in eval_units() for snr in MATCH_SNR_DB
                  if _eval_path(u, snr).exists())
    n_eval_slots = len(eval_units()) * len(MATCH_SNR_DB)
    print(f"-- training: done={counts['done']} failed={counts['failed']} "
          f"running={counts['running']} missing={counts['missing']} "
          f"(of {len(wave_cells())})")
    print(f"-- eval: {ev_done}/{n_eval_slots} unit-SNR slots complete")
    return 0


# ---------------------------------------------------------------------------
# training queue
# ---------------------------------------------------------------------------

def _train_one(c: WaveCell):
    """Dispatch by method; frozen-path values when a knob is None.

    Caps arm: the frozen trainers build their model from
    ``config.HIDDEN_DIMS`` internally, so the width override temporarily
    patches that config tuple for the duration of one run and restores it
    in a finally block (never persisted; illustrated in the run report).
    """
    patched_hidden = c.hidden is not None and c.hidden != config.HIDDEN_DIMS[0]
    saved = None
    if patched_hidden:
        saved = tuple(config.HIDDEN_DIMS)
        config.HIDDEN_DIMS = (c.hidden, c.hidden)   # transient, restored below
    try:
        if c.method == "task":
            snr_range_override = ((c.fixed_snr, c.fixed_snr)
                                  if c.fixed_snr is not None
                                  else config.SNR_TRAIN_RANGE_DB)
            model, rep = train_task_oriented(
                k=c.k, channel=c.channel, seed=c.seed,
                steps=c.steps, batch_episodes=config.BATCH_EPISODES,
                t_max=config.TASK_TRAINING_T,
                snr_range=snr_range_override,
                lambdas=c.lambdas,                   # None = frozen weights
                verbose_every=0)
            g0, g1 = rep["grad_enc_final"], rep["grad_dec_final"]
            grads_finite = (math.isfinite(g0) and math.isfinite(g1)
                            and g0 > 0 and g1 > 0)
            summary = {"task_loss_initial": rep["train_total_initial"],
                       "task_loss_final": rep["train_total_final"],
                       "recon_mse_diag_initial": rep["recon_mse_diag_initial"],
                       "recon_mse_diag_final": rep["recon_mse_diag_final"],
                       "eval_after_success_rate":
                           rep["eval_after"]["success_rate"],
                       "lambdas": rep["lambdas"],
                       "lambdas_amendment": rep.get("lambdas_amendment",
                                                    False)}
        elif c.method == "recon_w":
            model, rep = train_reconstruction(
                k=c.k, channel=c.channel, seed=c.seed,
                steps=c.steps, batch_states=config.BATCH_STATES,
                state_weights=c.state_weights)       # None = frozen uniform
            g0 = rep["grad_norm_encoder_final"]
            g1 = rep["grad_norm_decoder_final"]
            grads_finite = (math.isfinite(g0) and math.isfinite(g1)
                            and g0 > 0 and g1 > 0)
            summary = {"loss_initial": rep["loss_initial"],
                       "loss_final": rep["loss_final"],
                       "loss_reduction_pct": rep["loss_reduction_pct"],
                       "state_weights": rep.get("state_weights"),
                       "state_weights_amendment":
                           rep.get("state_weights_amendment", False)}
        else:
            raise ValueError(f"unknown wave method {c.method!r}")
    finally:
        if patched_hidden:
            config.HIDDEN_DIMS = saved               # restore frozen default
    return model, rep, grads_finite, summary


def _should_skip(state, smoke_tag=None):
    """Documented idempotency contract for scientific cells.

    DONE (complete) and FAILED (documented failure) are terminal and are
    skipped.  A failed cell is retried ONLY after the operator explicitly
    removes its FAILED.json marker.  Smoke runs live under an isolated root
    and are never skipped.
    """
    return smoke_tag is None and state in ("done", "failed")


def _provenance_fields(rep, method):
    """Additive diagnostics persisted in train_report.json (schema v2).

    Historical reports written before this schema do NOT contain these
    keys; consumers must report them as "unavailable" rather than
    fabricating or back-filling values.
    """
    out = {"report_schema_version": 2}
    if "history" in rep:
        out["history"] = rep["history"]
    if method == "task":
        for key in ("grad_enc_first", "grad_enc_final",
                    "grad_dec_first", "grad_dec_final"):
            if key in rep:
                out[key] = rep[key]
    else:
        for key in ("grad_encoder_history", "grad_decoder_history"):
            if key in rep:
                out[key] = rep[key]
        if "grad_norm_encoder_final" in rep:
            out["grad_enc_final"] = rep["grad_norm_encoder_final"]
        if "grad_norm_decoder_final" in rep:
            out["grad_dec_final"] = rep["grad_norm_decoder_final"]
    return out


def cmd_train(lane=None):
    """Training queue; optional lane filter ("awgn" | "rayleigh").

    ``smoke:TAG`` lane = smoke validation of a specific arm: 100-step
    budget, ALL smoke cells isolated under results/WAVE_A_SMOKE (their own
    root; can never touch results/WAVE_A/); non-scientific.
    """
    smoke_tag = None
    if lane is not None and lane.startswith("smoke:"):
        smoke_tag, lane = lane[len("smoke:"):], None
    global WAVE_A_ROOT
    if smoke_tag is not None:
        WAVE_A_ROOT = Path("results/WAVE_A_SMOKE")   # isolated smoke root
    print("== Wave-A training queue (non-frozen amendment) =="
          + (f"  [lane: {lane or smoke_tag if smoke_tag else lane}]" if (lane or smoke_tag) else ""))
    if lane not in (None, "awgn", "rayleigh"):
        print(f"unknown lane {lane!r} (use awgn|rayleigh|smoke:TAG or none)")
        return 1
    if smoke_tag is not None:
        cells = [c for c in wave_cells() if c.tag == smoke_tag]
        if not cells:
            print(f"unknown smoke tag {smoke_tag!r}; "
                  f"available: {sorted({c.tag for c in wave_cells()})}")
            return 1
        for c in cells:
            c.steps = 100
    else:
        cells = [c for c in wave_cells()
                 if lane is None or c.channel == lane]
    total = len(cells)
    n_new = n_skip = n_skip_failed = 0
    total_wall = 0.0
    for i, c in enumerate(cells, 1):
        st = c.state()
        if _should_skip(st, smoke_tag):
            if st == "failed":
                print(f"[{i:2d}/{total}] SKIP {c.wave_id} "
                      f"(documented failure; delete FAILED.json to retry)",
                      flush=True)
                n_skip_failed += 1
            else:
                print(f"[{i:2d}/{total}] SKIP {c.wave_id} (already complete)",
                      flush=True)
            n_skip += 1
            continue
        print(f"[{i:2d}/{total}] TRAIN {c.wave_id} hash={c.wave_hash()} "
              f"steps={c.steps} arm={c.arm}", flush=True)
        d = c.dir()
        c.mark_running()
        t0 = time.perf_counter()
        try:
            model, rep, grads_finite, summary = _train_one(c)
        except Exception as e:
            c.mark_failed(f"{type(e).__name__}: {e}")
            (d / "train_report.json").write_text(json.dumps(
                {"wave_id": c.wave_id, "spec": c.spec(),
                 "exit_status": "failed",
                 "error": f"{type(e).__name__}: {e}"}, indent=2))
            print(f"[{i:2d}/{total}] FAILED {c.wave_id}: "
                  f"{type(e).__name__}: {e}", flush=True)
            FAILS.append(c.wave_id)
            continue
        wall = time.perf_counter() - t0
        total_wall += wall
        loss_key = ("task_loss_final" if c.method == "task"
                    else "loss_final")
        complete = (rep["nan_batches"] == 0 and grads_finite
                    and math.isfinite(summary[loss_key]))
        rec = {
            "wave_id": c.wave_id, "spec": c.spec(),
            "wall_s": round(wall, 1), "s_per_step": round(wall / c.steps, 4),
            "nan_batches": rep["nan_batches"], "grads_finite": grads_finite,
            "exit_status": "complete" if complete else "incomplete",
            "checkpoint": str(d / "final_ckpt.pt"),
            "optimizer_state_saved": False,
            "optimizer_note": ("trainer-internal optimizer not exposed "
                               "(documented Step-8 limitation); wave "
                               "recovery = deterministic RETRAIN by spec"),
            **summary,
            **_provenance_fields(rep, c.method),
        }
        save_checkpoint(
            d / "final_ckpt.pt", model, optimizer=None,
            step=c.steps, seed=c.seed,
            metadata={"wave_id": c.wave_id, "wave_hash": c.wave_hash(),
                      "arm": c.arm, "tag": c.tag, "method": c.method,
                      "channel": c.channel, "k": c.k, "training_seed": c.seed,
                      "train_steps": c.steps,
                      "hidden": c.hidden if c.hidden else 128,
                      "lambdas": list(c.lambdas) if c.lambdas else None,
                      "state_weights": (list(c.state_weights)
                                        if c.state_weights else None),
                      "fixed_snr": c.fixed_snr,
                      "non_frozen": True},)
        (d / "train_report.json").write_text(
            json.dumps(rec, indent=2, default=str))
        if complete:
            c.mark_done(summary={"wall_s": rec["wall_s"],
                                 loss_key: summary[loss_key]})
            print(f"[{i:2d}/{total}] DONE  {c.wave_id}  {rec['wall_s']}s  "
                  f"{loss_key}={summary[loss_key]:.4f}", flush=True)
            n_new += 1
        else:
            c.mark_failed("incomplete numerical state")
            print(f"[{i:2d}/{total}] INCOMPLETE {c.wave_id} "
                  f"(nan={rep['nan_batches']}, grads={grads_finite})",
                  flush=True)
            FAILS.append(c.wave_id)
    if smoke_tag is not None:
        import shutil
        shutil.rmtree(WAVE_A_ROOT, ignore_errors=True)
        print("-- smoke-only run; isolated smoke root deleted; no scientific use")
        return 0
    failed_documented = sum(1 for c in cells if c.state() == "failed")
    print(f"-- wave training finished: new={n_new} skipped={n_skip} "
          f"(failed skipped={n_skip_failed}) failed_this_run={len(FAILS)} "
          f"documented_failed={failed_documented} wall={total_wall:.1f}s",
          flush=True)
    # Failed cells remain failures in the stage exit code even when they are
    # intentionally skipped, so an automatic orchestrator retry cannot turn a
    # documented failure into a silent success.  A queue whose failures are
    # ALL declared/documented is a completed stage (exit 4 = "accepted");
    # anything undeclared keeps the hard-failure exit code 1.
    failed_ids = set(FAILS) | {c.wave_id for c in cells if c.state() == "failed"}
    undeclared = sorted(failed_ids - DOCUMENTED_FAILURES)
    if undeclared:
        print(f"-- UNDECLARED failure(s) -> exit 1: {undeclared}", flush=True)
        return 1
    if failed_ids:
        print(f"-- {len(failed_ids)} documented failure(s) only -> exit "
              f"{DOCUMENTED_FAILURE_EXIT} (accepted; not retried)", flush=True)
        return DOCUMENTED_FAILURE_EXIT
    return 0


# ---------------------------------------------------------------------------
# evaluation queue (reduced 1000-episode diagnostic grid; paired pre-gen)
# ---------------------------------------------------------------------------

def cmd_evaluate(lane=None):
    print("== Wave-A paired diagnostic evaluation (E=1000, NON-FROZEN) =="
          + (f"  [lane: {lane}]" if lane else ""), flush=True)
    if lane not in (None, "awgn", "rayleigh"):
        print(f"unknown lane {lane!r}")
        return 1
    cells = [c for c in wave_cells()
             if lane is None or c.channel == lane]
    cells_by_unit = {}
    for c in cells:
        if c.state() != "done":
            print(f"NOTE: {c.wave_id} not complete; skipped from eval",
                  flush=True)
            continue
        key = (c.k, c.channel, c.seed)
        cells_by_unit.setdefault(key, []).append(c)
    n_new = 0
    total_wall = 0.0
    for (k, ch, seed), arm_cells in sorted(cells_by_unit.items()):
        unit = {"unit": f"k{k}_{ch}_seed{seed}", "k": k, "channel": ch,
                "seed": seed}
        # NOTE: each arm must load ITS OWN checkpoint.  The previous form
        # used the stale outer loop variable ``c`` (the last cell of
        # ``wave_cells()``), so every unit silently evaluated the k=12
        # checkpoint of the final cell and crashed on the k=1 units with
        # "noise shape/kind mismatch: (1000, 1) ... vs z (1000, 12)".
        models = [(m.tag, m.method, m, m.load_ckpt()) for m in arm_cells]
        models = [x for x in models if x[3] is not None]
        if not models:
            continue
        for snr in MATCH_SNR_DB:
            out = _eval_path(unit, snr)
            if out.exists():
                continue
            cond = Condition(k=k, channel=ch, snr_db=snr, t_max=config.T_MAX,
                             bits_per_component=None,
                             episodes=EVAL_EPISODES, seed=config.TEST_SEED)
            t0 = time.perf_counter()
            aggregates = {}
            with torch.no_grad():
                for c_tag, c_method, _cell, model in models:
                    name = "task" if c_method == "task" else "reconstruction"
                    cell = evaluate_condition(
                        cond, methods=[name],
                        recon_model=(model if c_method == "recon_w" else None),
                        task_model=(model if c_method == "task" else None))
                    aggregates[c_tag] = cell[name]["aggregate"]
            wall = time.perf_counter() - t0
            total_wall += wall
            rec = {
                "STEP": "WAVE_A", "experiment": "WAVE_A",
                "unit": unit["unit"], "k": k, "channel": ch, "seed": seed,
                "snr_db": snr, "test_seed": config.TEST_SEED,
                "episodes": EVAL_EPISODES, "t_max": config.T_MAX,
                "paired_reference": ("frozen M1/M2 realizations not reused; "
                                     "wave evals are self-paired (same "
                                     "episodes within unit, across tags)"),
                "protocol_note": (f"Wave-A diagnostic protocol: E="
                                  f"{EVAL_EPISODES}, non-frozen amendment "
                                  f"2026-10-12; NOT comparable 1:1 to frozen "
                                  f"M1/M2 (E=5000)"),
                "non_frozen": True,
                "wall_s": round(wall, 1),
                **{f"aggregate_{tag}": agg for tag, agg in aggregates.items()},
            }
            # within-unit paired stats between the wave arm and the FROZEN
            #-protocol arm of the same (k, channel, seed) cannot be computed
            # here (frozen eval realizations differ); report SR per arm only.
            _export_stats(rec, aggregates)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=2, default=str))
            n_new += 1
            srs = {t: a["success_rate"] for t, a in aggregates.items()}
            print(f"  EVAL {unit['unit']} snr={snr:5.1f} "
                  + " ".join(f"{t}={v:.3f}" for t, v in srs.items())
                  + f" ({wall:.1f}s)", flush=True)
    print(f"-- wave eval finished: {n_new} new unit-SNR slots, "
          f"wall={total_wall:.1f}s", flush=True)
    return 0


def _export_stats(rec, aggregates):
    """Wilson CIs per tag; paired McNemar task-vs-frozen-arm is NOT computed
    here (frozen eval realizations were consumed elsewhere); wave evals are
    labeled self-paired."""
    for tag, agg in aggregates.items():
        rec[f"wilson_{tag}"] = wilson_ci_95(
            [1] * agg["success_count"] + [0] * agg["failure_count"])


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

def cmd_finalize():
    print("== finalize: results/WAVE_A/step_wave_a_summary.json ==")
    cells = wave_cells()
    train_rows = []
    for c in cells:
        p = c.dir() / "train_report.json"
        rec = json.loads(p.read_text()) if p.exists() else {"exit_status": "missing"}
        train_rows.append({"wave_id": c.wave_id, "arm": c.arm, "tag": c.tag,
                           "method": c.method, "channel": c.channel,
                           "k": c.k, "seed": c.seed,
                           "exit_status": rec.get("exit_status", "missing"),
                           "task_loss_final": rec.get("task_loss_final"),
                           "loss_final": rec.get("loss_final"),
                           "eval_after_success_rate":
                               rec.get("eval_after_success_rate"),
                           "history_available": "history" in rec,
                           "grad_health_available": (
                               "grad_enc_final" in rec
                               and "grad_dec_final" in rec)})
    n_slots = 0
    complete_units = 0
    pooled = {}
    units = eval_units()
    for unit in units:
        k, ch, seed = unit["k"], unit["channel"], unit["seed"]
        unit_slots = 0
        for snr in MATCH_SNR_DB:
            p = _eval_path(unit, snr)
            if not p.exists():
                continue
            n_slots += 1
            unit_slots += 1
            R = json.loads(p.read_text())
            for key, agg in R.items():
                if key.startswith("aggregate_"):
                    tag = key[len("aggregate_"):]
                    pooled.setdefault(
                        (tag, k, ch, snr), []).append(
                        {"seed": seed, **agg})
        if unit_slots == len(MATCH_SNR_DB):
            complete_units += 1
    pooled_rows = []
    for (tag, k, ch, snr), rows in sorted(pooled.items(),
                                          key=lambda x: (x[0][2], x[0][3],
                                                         x[0][0], x[0][1])):
        sc = sum(r["success_count"] for r in rows)
        tot = sum(r["episodes"] for r in rows)
        pooled_rows.append({
            "tag": tag, "k": k, "channel": ch, "snr_db": snr,
            "n_seeds": len(rows), "success_count_pooled": sc,
            "episodes_pooled": tot,
            "success_rate_pooled": sc / tot if tot else float("nan")})
    expected_slots = len(units) * len(MATCH_SNR_DB)
    train_done = sum(1 for r in train_rows if r["exit_status"] == "complete")
    train_failed = sum(1 for r in train_rows
                       if r["exit_status"] not in ("complete", "missing"))
    train_missing = sum(1 for r in train_rows
                        if r["exit_status"] == "missing")
    out = WAVE_A_ROOT / "step_wave_a_summary.json"
    out.write_text(json.dumps({
        "experiment": "WAVE_A",
        "status": ("complete"
                   if train_done == len(train_rows)
                   and n_slots == expected_slots
                   else "incomplete"),
        "amendment": "WAVE_A-non-frozen-2026-10-12",
        "protocol_note": (f"Wave-A diagnostic: E={EVAL_EPISODES} per unit; "
                          "NOT 1:1 comparable to frozen M1/M2 (E=5000)"),
        "train_rows": train_rows,
        "pooled_rows": pooled_rows,
        "invalid_arms": INVALID_ARMS,
        "documented_failures": sorted(DOCUMENTED_FAILURES),
        "train_done": train_done,
        "train_failed": train_failed,
        "train_missing": train_missing,
        "eval_units_expected": len(units),
        "eval_slots_expected": expected_slots,
        "eval_slots_complete": n_slots,
        "eval_units_complete": complete_units,
        "eval_units_unreachable": (f"{len(units) - complete_units} unit(s) not "
                                   "fully evaluated (documented failures)"),
        "provenance_note": ("train_report.json schema v2 persists loss "
                            "history and gradient-health diagnostics; older "
                            "reports lack those keys and are marked "
                            "history_available=false / "
                            "grad_health_available=false."),
    }, indent=2, default=str))
    print(f"wrote {out} ({len(train_rows)} train rows, "
          f"{len(pooled_rows)} pooled tag-rows, {n_slots}/{expected_slots} "
          f"eval slots, {complete_units}/{len(units)} complete units, train "
          f"done/failed/missing="
          f"{train_done}/{train_failed}/{train_missing})", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    dispatch = {"verify": cmd_verify, "plan": cmd_plan, "status": cmd_status,
                "train": cmd_train, "evaluate": cmd_evaluate,
                "finalize": cmd_finalize}
    if cmd not in dispatch:
        print(__doc__)
        sys.exit(1)
    lane = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd in ("verify", "plan", "status", "finalize"):
        sys.exit(dispatch[cmd]())
    sys.exit(dispatch[cmd](lane))
