"""Step 10 — Full M1 Execution (user-approved, 2026-10-01).

Executes the FROZEN M1 matrix exactly as specified (no frozen value is
touched, no digital cells, no M2/M3/M4/M5):

    training  : 2 methods x 3 k x 2 channels x 3 seeds = 36 neural runs
                (frozen 9C budgets: recon 5000x4096, task 3000x256 T=100)
    evaluation: paired task-vs-reconstruction per (k, channel, seed, SNR),
                SNR grid {0,2,...,20} dB, 5000 episodes, test seed 10042,
                via the frozen Step-7 harness: both same-seed models are
                handed to ONE evaluate_condition call, so test episodes,
                initial states, goals, noise and fading realizations are
                bitwise identical for both methods by construction.

Statistics (frozen statistics.py, no SciPy): Wilson 95% CI per method per
cell; McNemar on the paired per-episode successes; paired t on the paired
continuous metrics (final_distance, avg_distance, control_effort).
Descriptive only — the tool never ranks methods or picks winners.

Identity/recovery: the Step-8/9A identity system
(results/M1/<run_id>__<hash>/ with DONE/RUNNING/FAILED markers).
Completed runs SKIP; interrupted/failed runs are RETRAINED (deterministic
by seed+config — the frozen 9C policy; honest: mid-run bitwise
continuation is NOT claimed for the production trainers).

Subcommands (repository root):
    verify      frozen-protocol + workload hard gate (STOP on mismatch)
    plan        print the exact deterministic execution order (no side effects)
    status      per-cell state table + counts (no side effects)
    train       run the training queue (skip completed; retrain on retry)
    evaluate    run the paired-eval queue (skips completed units)
    validate    completion checks: counts/ids/keys + frozen-matrix validation
    checkint    per-run checkpoint integrity re-verification
    finalize    write results/M1/step10_m1_summary.json (execution facts)

Artifacts (all gitignored):
    results/M1/<run_id>__<hash>/{final_ckpt.pt, train_report.json, DONE.json}
    results/M1/eval/k<k>_<channel>_seed<seed>/eval_<snr>.json
    results/M1/step10_m1_summary.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from project import config
from project.evaluation.evaluate import Condition, evaluate_condition
from project.evaluation.statistics import mcnemar_test, paired_t_test, wilson_ci_95
from project.experiments.config import (
    CHANNELS, M1_K_VALUES, M1_SNR_DB, TRAINING_SEEDS, ExperimentConfig)
from project.experiments.identity import (
    cell_state, config_hash, mark_done, mark_failed, mark_running,
    run_dir, run_id)
from project.models.reconstruction_jscc import ReconstructionDeepJSCC
from project.training.common import save_checkpoint
from project.training.train_reconstruction import train_reconstruction
from project.training.train_task_oriented import train_task_oriented

M1_ROOT = Path("results/M1")
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def _m1_cfg(method, k, channel, seed):
    """The exact frozen M1 training cell (frozen values only)."""
    return ExperimentConfig(
        experiment="M1", method=method, channel=channel, k=k,
        training_seed=seed, test_seed=config.TEST_SEED,
        train_steps=(config.RECON_TRAIN_STEPS if method == "reconstruction"
                     else config.TASK_TRAIN_STEPS),
        batch_states=config.BATCH_STATES,
        batch_episodes=config.BATCH_EPISODES,
        t_max=config.TASK_TRAINING_T,
        lr=config.LEARNING_RATE,
        eval_snr_grid_db=M1_SNR_DB,
        eval_snr_db=10.0,
        n_train_episodes=config.N_TRAIN_EPISODES,
        n_test_episodes=config.N_TEST_EPISODES,
        eval_episodes=config.BATCH_EPISODES,
        run_mode="smoke")   # identity/vocabulary only; execution is THIS tool


def m1_cells():
    """36 cells in the frozen deterministic order."""
    cells = []
    for k in M1_K_VALUES:
        for channel in CHANNELS:
            for method in ("reconstruction", "task"):
                for seed in TRAINING_SEEDS:
                    cells.append(_m1_cfg(method, k, channel, seed))
    return cells


def eval_units():
    """198 paired evaluation units: (k, channel, seed) x 11 SNRs.

    Each unit pairs the SAME-SEED reconstruction and task models in one
    evaluate_condition call (paired episodes/noise/h by construction).
    """
    units = []
    for k in M1_K_VALUES:
        for channel in CHANNELS:
            for seed in TRAINING_SEEDS:
                units.append({
                    "unit": f"k{k}_{channel}_seed{seed}",
                    "k": k, "channel": channel, "seed": seed,
                    "recon": _m1_cfg("reconstruction", k, channel, seed),
                    "task": _m1_cfg("task", k, channel, seed),
                })
    return units


def _eval_path(unit, snr):
    return M1_ROOT / "eval" / unit["unit"] / f"eval_{snr:g}.json"


def _load_model(cfg):
    ckpt = run_dir(cfg, results_root="results") / "final_ckpt.pt"
    if not ckpt.exists():
        return None
    model = ReconstructionDeepJSCC(k=cfg.k)
    payload = torch.load(ckpt, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return model


# ---------------------------------------------------------------------------
# verify — hard gate on the frozen protocol and the workload
# ---------------------------------------------------------------------------

def cmd_verify():
    print("== frozen Step-9C protocol verification (Step-10 gate) ==")
    check("recon_steps==5000", config.RECON_TRAIN_STEPS == 5000)
    check("task_steps==3000", config.TASK_TRAIN_STEPS == 3000)
    check("task_training_T==100", config.TASK_TRAINING_T == 100 == config.T_MAX)
    check("batch_states==4096", config.BATCH_STATES == 4096)
    check("batch_episodes==256", config.BATCH_EPISODES == 256)
    check("lr==1e-3", config.LEARNING_RATE == 1e-3)
    check("grad_clip==1.0", config.GRAD_CLIP_NORM == 1.0)
    check("training_snr==U(0,20)/batch",
          config.SNR_TRAIN_RANGE_DB == (0.0, 20.0))
    check("seeds=={42,43,44}", config.TRAINING_SEEDS == (42, 43, 44))
    check("test_seed==10042", config.TEST_SEED == 10042)
    check("early_stopping==disabled", config.EARLY_STOPPING is False)
    check("M5==deferred+disabled", config.M5_STATUS == "deferred"
          and config.M5_EXECUTION_ENABLED is False)
    check("m1_eval_grid=={0,2,...,20}",
          M1_SNR_DB == tuple(float(s) for s in range(0, 21, 2))
          == config.SNR_TEST_GRID_DB)
    check("m1_k_values=={1,2,3}", M1_K_VALUES == (1, 2, 3))
    check("task_lambdas_frozen",
          (config.LAMBDA_DISTANCE, config.LAMBDA_CONTROL,
           config.LAMBDA_VELOCITY, config.LAMBDA_TERMINAL) == (1.0, 0.0, 0.0, 2.0))
    check("n_test_episodes==5000", config.N_TEST_EPISODES == 5000)
    check("goal_radius==0.5", config.GOAL_RADIUS == 0.5)
    check("horizon_T_max==100_dt==0.1",
          config.T_MAX == 100 and config.DT == 0.1)
    check("tool_has_no_M2_M3_M4_M5_paths", True,
          "this tool implements M1 only")
    config.validate()
    print("config.validate() OK")
    check("workload==36_training_runs", len(m1_cells()) == 36)
    check("workload==198_paired_eval_units",
          len(eval_units()) * len(M1_SNR_DB) == 198
          and len(m1_cells()) * len(M1_SNR_DB) == 396,
          "396 per-method rows == 198 paired units x 2 methods")
    check("pooled_eval_conditions==132",
          2 * 3 * 2 * len(M1_SNR_DB) == 132)
    print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    return 1 if FAILS else 0


# ---------------------------------------------------------------------------
# plan / status (no side effects)
# ---------------------------------------------------------------------------

def cmd_plan():
    print("== M1 execution plan (deterministic order; NO execution) ==")
    for i, cfg in enumerate(m1_cells(), 1):
        print(f"{i:2d}. {run_id(cfg):46s} steps={cfg.train_steps:5d} "
              f"batch={'states4096' if cfg.method == 'reconstruction' else 'episodes256_T100'}")
    n_units = len(eval_units()) * len(M1_SNR_DB)
    print(f"total: {len(m1_cells())} training runs; {n_units} paired eval "
          f"units (2 methods each, 11 SNRs x 5000 episodes, test seed "
          f"{config.TEST_SEED}); pooled conditions: 132")
    return 0


def cmd_status():
    print("== M1 status ==")
    counts = {"done": 0, "failed": 0, "running": 0, "missing": 0}
    for cfg in m1_cells():
        st = cell_state(cfg, results_root="results")
        counts[st] += 1
        print(f"{st:8s} {run_id(cfg)}  hash={config_hash(cfg)}")
    print(f"-- training: done={counts['done']} failed={counts['failed']} "
          f"running={counts['running']} missing={counts['missing']} (of 36)")
    ev_done = sum(1 for u in eval_units() for snr in M1_SNR_DB
                  if _eval_path(u, snr).exists())
    print(f"-- eval: {ev_done}/198 paired units complete")
    ok = counts["done"] == 36 and counts["failed"] == 0 and ev_done == 198
    print(f"-- M1 complete: {ok}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# training queue
# ---------------------------------------------------------------------------

def _train_one(cfg):
    if cfg.method == "reconstruction":
        model, rep = train_reconstruction(
            k=cfg.k, channel=cfg.channel, seed=cfg.training_seed,
            steps=cfg.train_steps, batch_states=cfg.batch_states)
        g0, g1 = rep["grad_norm_encoder_final"], rep["grad_norm_decoder_final"]
        grads_finite = (math.isfinite(g0) and math.isfinite(g1)
                        and g0 > 0 and g1 > 0)
        summary = {"loss_initial": rep["loss_initial"],
                   "loss_final": rep["loss_final"],
                   "loss_reduction_pct": rep["loss_reduction_pct"]}
    else:
        model, rep = train_task_oriented(
            k=cfg.k, channel=cfg.channel, seed=cfg.training_seed,
            steps=cfg.train_steps, batch_episodes=cfg.batch_episodes,
            t_max=cfg.t_max, eval_episodes=cfg.eval_episodes,
            verbose_every=0)
        g0, g1 = rep["grad_enc_final"], rep["grad_dec_final"]
        grads_finite = (math.isfinite(g0) and math.isfinite(g1)
                        and g0 > 0 and g1 > 0)
        summary = {"task_loss_initial": rep["train_total_initial"],
                   "task_loss_final": rep["train_total_final"],
                   "recon_mse_diag_initial": rep["recon_mse_diag_initial"],
                   "recon_mse_diag_final": rep["recon_mse_diag_final"],
                   "eval_after_success_rate": rep["eval_after"]["success_rate"],
                   "lambdas": rep["lambdas"]}
    return model, rep, grads_finite, summary


def cmd_train(lane=None):
    """Training queue. Optional lane filter ("awgn"|"rayleigh") allows two
    independent processes to share the matrix by channel (each cell still
    runs exactly once, in its own deterministic order; per-cell computation
    is unchanged — purely a wall-clock scheduling device)."""
    print("== M1 training queue (frozen 9C budgets) =="
          + (f"  [lane: {lane}]" if lane else ""))
    cells = [c for c in m1_cells()
             if lane is None or c.channel == lane]
    total = len(cells)
    n_trained = n_skipped = n_retrained = 0
    total_wall = 0.0
    for i, cfg in enumerate(cells, 1):
        rid, ch = run_id(cfg), config_hash(cfg)
        st = cell_state(cfg, results_root="results")
        if st == "done":
            print(f"[{i:2d}/{total}] SKIP {rid} (already complete)")
            n_skipped += 1
            continue
        if st == "failed":
            print(f"[{i:2d}/{total}] RETRAIN-after-FAILURE {rid} "
                  f"(frozen 9C policy: deterministic by seed+config)")
            n_retrained += 1
        elif st == "running":
            print(f"[{i:2d}/{total}] RETRAIN-after-INTERRUPTION {rid} "
                  f"(stale RUNNING marker; environmental; frozen 9C policy)")
            n_retrained += 1
        d = run_dir(cfg, results_root="results")
        mark_running(cfg, results_root="results")
        print(f"[{i:2d}/{total}] TRAIN {rid} hash={ch} steps={cfg.train_steps} "
              f"batch={'4096' if cfg.method == 'reconstruction' else '256xT100'}")
        t0 = time.perf_counter()
        try:
            model, rep, grads_finite, summary = _train_one(cfg)
        except Exception as e:              # structured failure, no silent retry
            mark_failed(cfg, f"{type(e).__name__}: {e}", results_root="results")
            (d / "train_report.json").write_text(json.dumps({
                "run_id": rid, "config_hash": ch, "exit_status": "failed",
                "error": f"{type(e).__name__}: {e}"}, indent=2))
            print(f"[{i:2d}/{total}] FAILED {rid}: {type(e).__name__}: {e}")
            FAILS.append(rid)
            continue
        wall = time.perf_counter() - t0
        total_wall += wall
        loss_key = ("loss_final" if cfg.method == "reconstruction"
                    else "task_loss_final")
        complete = (rep["nan_batches"] == 0 and grads_finite
                    and math.isfinite(summary[loss_key]))
        rec = {
            "STEP": "10", "experiment": "M1",
            "run_id": rid, "config_hash": ch,
            "method": cfg.method, "channel": cfg.channel, "k": cfg.k,
            "seed": cfg.training_seed,
            "protocol": {
                "train_steps": cfg.train_steps,
                "batch_states": cfg.batch_states,
                "batch_episodes": cfg.batch_episodes,
                "t_max": cfg.t_max, "lr": cfg.lr,
                "optimizer": "Adam", "grad_clip": config.GRAD_CLIP_NORM,
                "train_snr_policy": {"distribution": "uniform",
                                     "range_db": [0.0, 20.0],
                                     "granularity": "per_batch"},
                "early_stopping": False},
            "wall_s": round(wall, 1),
            "s_per_step": round(wall / cfg.train_steps, 4),
            "nan_batches": rep["nan_batches"],
            "grads_finite": grads_finite,
            "exit_status": "complete" if complete else "incomplete",
            "checkpoint": str(d / "final_ckpt.pt"),
            "optimizer_state_saved": False,
            "optimizer_note": ("trainer-internal optimizer not exposed "
                               "(documented Step-8 limitation); recovery is "
                               "deterministic RETRAIN by seed+config (9C)"),
            **summary,
        }
        save_checkpoint(
            d / "final_ckpt.pt", model, optimizer=None,
            step=cfg.train_steps, seed=cfg.training_seed,
            metadata={"run_id": rid, "config_hash": ch,
                      "experiment": "M1", "method": cfg.method,
                      "channel": cfg.channel, "k": cfg.k,
                      "training_seed": cfg.training_seed,
                      "train_steps": cfg.train_steps,
                      "batch_states": cfg.batch_states,
                      "batch_episodes": cfg.batch_episodes,
                      "t_max": cfg.t_max, "lr": cfg.lr,
                      "model_config": {"hidden_dims": list(config.HIDDEN_DIMS),
                                       "activation": "PReLU"},
                      "train_snr_policy": rec["protocol"]["train_snr_policy"]},
            history=rep["history"], capture_rng=True)
        (d / "train_report.json").write_text(
            json.dumps(rec, indent=2, default=str))
        n_trained += 1
        if complete:
            mark_done(cfg, summary={"wall_s": rec["wall_s"],
                                    loss_key: summary[loss_key]},
                      results_root="results")
            print(f"[{i:2d}/{total}] DONE  {rid}  {rec['wall_s']}s  "
                  f"{loss_key}={summary[loss_key]:.4f}")
        else:
            mark_failed(cfg, "incomplete numerical state",
                        results_root="results")
            FAILS.append(rid)
            print(f"[{i:2d}/{total}] INCOMPLETE {rid} "
                  f"(nan_batches={rep['nan_batches']}, "
                  f"grads_finite={grads_finite})")
    print(f"-- training queue finished: trained={n_trained} "
          f"skipped={n_skipped} retrained={n_retrained} "
          f"failed={len(FAILS)} train_wall={total_wall:.1f}s")
    return 1 if FAILS else 0


# ---------------------------------------------------------------------------
# paired evaluation queue
# ---------------------------------------------------------------------------

def _integrity_failures(cfg):
    """11-checkpoint integrity for one run; returns a list of failures."""
    bad = []
    d = run_dir(cfg, results_root="results")
    ckpt = d / "final_ckpt.pt"
    if not ckpt.exists():
        return ["ckpt-exists"]
    fresh = ReconstructionDeepJSCC(k=cfg.k)
    payload = torch.load(ckpt, weights_only=False)
    md = payload.get("metadata", {})
    if not md:
        bad.append("ckpt-metadata")
    if not (md.get("train_steps") == cfg.train_steps
            and md.get("t_max") == cfg.t_max
            and math.isclose(md.get("lr", -1), config.LEARNING_RATE)
            and md.get("batch_states") == cfg.batch_states
            and md.get("batch_episodes") == cfg.batch_episodes):
        bad.append("config-matches")
    if payload.get("seed") != cfg.training_seed:
        bad.append("seed-matches")
    if not (md.get("method") == cfg.method and md.get("channel") == cfg.channel
            and md.get("k") == cfg.k):
        bad.append("method-channel-k")
    if payload.get("step") != cfg.train_steps:
        bad.append("final-step==budget")
    if not all(bool(torch.isfinite(p).all()) for p in fresh.parameters()):
        bad.append("params-finite")
    if md.get("run_id") != run_id(cfg):
        bad.append("identity-matches")
    if md.get("config_hash") != config_hash(cfg):
        bad.append("hash-matches")
    rep_path = d / "train_report.json"
    if not (rep_path.exists()
            and json.loads(rep_path.read_text()).get("exit_status") == "complete"):
        bad.append("report-complete")
    return bad


def _paired_stats(cell):
    """Wilson CI per method + McNemar + paired t from one paired cell."""
    out = {}
    m_rec, m_task = cell["reconstruction"]["metrics"], cell["task"]["metrics"]
    for name, m in (("reconstruction", m_rec), ("task", m_task)):
        out[f"wilson_{name}"] = wilson_ci_95(m["success"])
    out["mcnemar_task_vs_recon"] = mcnemar_test(m_task["success"],
                                                m_rec["success"])
    for key in ("final_distance", "avg_distance", "control_effort"):
        out[f"paired_t_{key}_task_vs_recon"] = paired_t_test(
            m_task[key].to(torch.float64), m_rec[key].to(torch.float64))
    return out


def cmd_evaluate(lane=None):
    """Paired-eval queue; optional lane filter ("awgn"|"rayleigh") lets two
    independent processes share the units by channel (disjoint outputs)."""
    print("== M1 paired evaluation queue (Step-7 harness, 5000 episodes) =="
          + (f"  [lane: {lane}]" if lane else ""))
    print("-- per-run checkpoint integrity gate --")
    blocked = []
    for cfg in m1_cells():
        if lane is not None and cfg.channel != lane:
            continue
        if cell_state(cfg, results_root="results") != "done":
            blocked.append(f"{run_id(cfg)}: not complete")
        else:
            bad = _integrity_failures(cfg)
            if bad:
                blocked.append(f"{run_id(cfg)}: {bad}")
    if blocked:
        for b in blocked:
            print(f"REFUSED: {b}")
        return 2
    print(f"   integrity gate passed for lane="
          f"{lane or 'all'} ({len(m1_cells()) if lane is None else 18} runs)")
    n_new = 0
    total_wall = 0.0
    units = [u for u in eval_units()
             if lane is None or u["channel"] == lane]
    for u in units:
        m_recon = _load_model(u["recon"])
        m_task = _load_model(u["task"])
        cond_base = dict(k=u["k"], channel=u["channel"], t_max=config.T_MAX,
                         episodes=config.N_TEST_EPISODES,
                         seed=config.TEST_SEED)
        for snr in M1_SNR_DB:
            out = _eval_path(u, snr)
            if out.exists():
                continue
            cond = Condition(snr_db=snr, **cond_base)
            t0 = time.perf_counter()
            with torch.no_grad():
                cell = evaluate_condition(
                    cond, methods=["reconstruction", "task"],
                    recon_model=m_recon, task_model=m_task)
            wall = time.perf_counter() - t0
            total_wall += wall
            rec = {
                "STEP": "10", "experiment": "M1",
                "unit": u["unit"], "k": u["k"], "channel": u["channel"],
                "seed": u["seed"], "snr_db": snr,
                "test_seed": config.TEST_SEED,
                "episodes": config.N_TEST_EPISODES, "t_max": config.T_MAX,
                "paired": True, "wall_s": round(wall, 1),
                "aggregate_reconstruction": cell["reconstruction"]["aggregate"],
                "aggregate_task": cell["task"]["aggregate"],
                **_paired_stats(cell),
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=2, default=str))
            n_new += 1
            sr_r = rec["aggregate_reconstruction"]["success_rate"]
            sr_t = rec["aggregate_task"]["success_rate"]
            print(f"  EVAL {u['unit']} snr={snr:5.1f} "
                  f"SR recon={sr_r:.4f} task={sr_t:.4f} ({wall:.1f}s)")
        del m_recon, m_task
    print(f"-- evaluation queue finished: {n_new} new units this session, "
          f"eval_wall={total_wall:.1f}s total")
    return 0


# ---------------------------------------------------------------------------
# checkint / validate
# ---------------------------------------------------------------------------

def cmd_checkint():
    bad = {}
    for cfg in m1_cells():
        f = _integrity_failures(cfg)
        if f:
            bad[run_id(cfg)] = f
    for rid, f in bad.items():
        print(f"FAIL {rid}: {f}")
    print(f"== checkpoint integrity: {36 - len(bad)}/36 runs pass ==")
    return 1 if bad else 0


def cmd_validate():
    print("== M1 completion validation ==")
    FAILS.clear()
    cells = m1_cells()
    ids = [(run_id(c), config_hash(c)) for c in cells]
    check("36-unique-identities", len(set(ids)) == 36)
    done = sum(1 for c in cells
               if cell_state(c, results_root="results") == "done")
    check("36-training-runs-complete", done == 36, f"{done}/36")
    check("methods==reconstruction,task",
          sorted({c.method for c in cells}) == ["reconstruction", "task"])
    check("k=={1,2,3}", sorted({c.k for c in cells}) == [1, 2, 3])
    check("channels==awgn,rayleigh",
          sorted({c.channel for c in cells}) == ["awgn", "rayleigh"])
    check("seeds=={42,43,44}",
          sorted({c.training_seed for c in cells}) == [42, 43, 44])
    check("no-digital-cells", not any(c.method == "digital" for c in cells))
    check("no-m2-m3-m4-m5-cells",
          not any(c.experiment != "M1" for c in cells))
    # eval completeness: 198 paired units, 5000 episodes, test seed 10042
    missing = [f"{u['unit']}@{snr:g}" for u in eval_units()
               for snr in M1_SNR_DB if not _eval_path(u, snr).exists()]
    check("198-eval-units-complete", len(missing) == 0,
          f"missing={missing[:6]}{'...' if len(missing) > 6 else ''}")
    pooled, eps_ok, pair_ok = {}, True, True
    for u in eval_units():
        for snr in M1_SNR_DB:
            p = _eval_path(u, snr)
            if not p.exists():
                continue
            rec = json.loads(p.read_text())
            if rec.get("episodes") != 5000 or rec.get("test_seed") != 10042:
                eps_ok = False
            if not rec.get("paired"):
                pair_ok = False
            for method in ("reconstruction", "task"):
                key = (method, u["k"], u["channel"], snr)
                pooled.setdefault(key, []).append(rec[f"aggregate_{method}"])
    check("episodes==5000-test-seed==10042-everywhere", eps_ok)
    check("paired-true-everywhere", pair_ok)
    check("132-pooled-eval-conditions", len(pooled) == 132, f"{len(pooled)}")
    check("3-seeds-per-pooled-cell",
          all(len(v) == 3 for v in pooled.values()))
    # frozen matrix expectations + structural validation of the proposal
    from project.experiments.matrix import expected_cells, validate_matrix
    from main import _matrix_proposal
    ec = expected_cells()
    check("matrix-expected-M1-counts",
          ec["M1_training_runs"] == 36 and ec["M1_eval_conditions"] == 132)
    res = validate_matrix(_matrix_proposal())
    check("frozen-matrix-proposal-valid", res["valid"],
          f"warnings={len(res['warnings'])}")
    # every unit carries both methods' aggregates
    both = True
    for u in eval_units():
        for snr in M1_SNR_DB:
            p = _eval_path(u, snr)
            if not p.exists():
                continue
            rec = json.loads(p.read_text())
            if ("aggregate_reconstruction" not in rec
                    or "aggregate_task" not in rec):
                both = False
    check("both-methods-in-every-unit", both)
    print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    return 1 if FAILS else 0


# ---------------------------------------------------------------------------
# finalize — execution-facts summary (descriptive, no ranking)
# ---------------------------------------------------------------------------

def cmd_finalize():
    print("== finalize: results/M1/step10_m1_summary.json ==")
    cells = m1_cells()
    train_rows = []
    for cfg in cells:
        d = run_dir(cfg, results_root="results")
        rec = json.loads((d / "train_report.json").read_text())
        train_rows.append({k2: rec.get(k2) for k2 in
                           ("run_id", "config_hash", "method", "channel", "k",
                            "seed", "wall_s", "s_per_step", "nan_batches",
                            "grads_finite", "exit_status")})
    n_units = 0
    pooled = {}
    wilson = {}
    for u in eval_units():
        for snr in M1_SNR_DB:
            p = _eval_path(u, snr)
            if not p.exists():
                continue
            n_units += 1
            rec = json.loads(p.read_text())
            for method in ("reconstruction", "task"):
                key = (method, u["k"], u["channel"], snr)
                pooled.setdefault(key, []).append(
                    rec[f"aggregate_{method}"])
                w = rec[f"wilson_{method}"]
                wilson.setdefault(key, []).append(
                    (w["success_count"], w["total"]))
    pool_rows = []
    for (method, k, ch, snr) in sorted(pooled.keys(),
                                       key=lambda x: (x[1], x[2], x[0], x[3])):
        aggs = pooled[(method, k, ch, snr)]
        row = {"method": method, "k": k, "channel": ch, "snr_db": snr,
               "n_seeds": len(aggs)}
        # pooled success: sum of successes over the seed rows / totals
        sc = sum(a["success_count"] for a in aggs)
        tot = sum(a["episodes"] for a in aggs)
        row["success_count_pooled"] = sc
        row["episodes_pooled"] = tot
        row["success_rate_pooled"] = sc / tot if tot else float("nan")
        w = wilson[(method, k, ch, snr)]
        succ = ([1] * sum(c for c, _ in w) + [0] * sum(t - c for c, t in w))
        ci = wilson_ci_95(succ)
        row["wilson_low_pooled"] = ci["wilson_low"]
        row["wilson_high_pooled"] = ci["wilson_high"]
        for f in ("final_distance_mean", "avg_distance_mean",
                  "control_effort_mean", "min_distance_mean", "mse_mean",
                  "ttg_success_mean"):
            vals = [a[f] for a in aggs
                    if isinstance(a.get(f), (int, float))
                    and math.isfinite(a[f])]
            if vals:
                row[f"{f}_mean_over_seeds"] = sum(vals) / len(vals)
                if len(vals) > 1:
                    mu = sum(vals) / len(vals)
                    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
                    row[f"{f}_std_over_seeds"] = var ** 0.5
        pool_rows.append(row)
    total_train_s = sum(r["wall_s"] or 0 for r in train_rows)
    summary = {
        "STEP": "10", "experiment": "M1",
        "status": ("complete" if len(train_rows) == 36 and n_units == 198
                   else "incomplete"),
        "protocol_frozen_9C": {
            "recon_steps": config.RECON_TRAIN_STEPS,
            "task_steps": config.TASK_TRAIN_STEPS,
            "task_training_T": config.TASK_TRAINING_T,
            "batch_states": config.BATCH_STATES,
            "batch_episodes": config.BATCH_EPISODES,
            "optimizer": "Adam", "lr": config.LEARNING_RATE,
            "grad_clip": config.GRAD_CLIP_NORM,
            "train_snr_policy": "U(0,20) dB per batch",
            "training_seeds": list(config.TRAINING_SEEDS),
            "test_seed": config.TEST_SEED, "early_stopping": False,
            "eval_grid_db": list(M1_SNR_DB),
            "eval_episodes": config.N_TEST_EPISODES,
            "eval_horizon": {"T_max": config.T_MAX, "dt": config.DT},
            "goal_radius": config.GOAL_RADIUS,
            "m5_status": config.M5_STATUS},
        "training_runs": {"expected": 36, "completed": len(train_rows),
                          "failed": sum(1 for r in train_rows
                                        if r["exit_status"] != "complete")},
        "eval": {"paired_units_expected": 198, "paired_units_complete": n_units,
                 "pooled_conditions": len(pool_rows),
                 "episodes_per_unit_per_method": config.N_TEST_EPISODES},
        "total_train_wall_s": round(total_train_s, 1),
        "interpretation_note": ("Execution report only; no overall ranking "
                                "or scientific verdict is made in this step. "
                                "Pooled rows are descriptive (pooled success "
                                "over the 3 training-seed rows + mean/std of "
                                "per-seed aggregates); paired per-seed "
                                "McNemar/paired-t results live in the "
                                "eval unit JSONs."),
        "train_rows": train_rows,
        "pooled_rows": pool_rows,
    }
    out = M1_ROOT / "step10_m1_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out} ({len(train_rows)} train rows, "
          f"{len(pool_rows)} pooled rows, {n_units}/198 units)")
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    dispatch = {
        "verify": cmd_verify, "plan": cmd_plan, "status": cmd_status,
        "train": cmd_train, "evaluate": cmd_evaluate,
        "validate": cmd_validate, "checkint": cmd_checkint,
        "finalize": cmd_finalize,
    }
    if cmd not in dispatch:
        print(__doc__)
        sys.exit(1)
    if cmd in ("train", "evaluate"):
        lane = sys.argv[2] if len(sys.argv) > 2 else None
        if lane not in (None, "awgn", "rayleigh"):
            print(f"unknown lane {lane!r} (use awgn|rayleigh or none)")
            sys.exit(1)
        sys.exit(dispatch[cmd](lane))
    sys.exit(dispatch[cmd]())
