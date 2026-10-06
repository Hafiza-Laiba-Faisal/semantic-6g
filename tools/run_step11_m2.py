"""Step 11 — Full M2 Execution (user-approved, 2026-10-02).

Executes the FROZEN M2 matrix exactly as specified (no frozen value is
touched, no M1/M3/M4/M5 paths):

    training  : task + reconstruction x k {12,18,24,30,54} x 2 channels
                x 3 seeds = 60 neural runs (frozen 9C budgets: recon
                5000x4096, task 3000x256 T=100)
    evaluation: 3-method PAIRED evaluation (reconstruction, task, digital)
                per (k, channel, seed, SNR) in ONE evaluate_condition call,
                SNR grid {0,5,10,15,20} dB, 5000 episodes, test seed 10042
                -> 150 paired units x 3 methods. The digital baseline is
                evaluation-ONLY (30 seed combinations, matched budgets
                B={12:1,18:2,24:3,30:4,54:8}); it has no training run.

Pairing contract (frozen Step-7 harness): test episodes, initial states,
goals, noise and fading realizations are bitwise identical across the three
methods by construction.

Statistics (frozen statistics.py): Wilson 95% CI per method per unit;
McNemar on paired per-episode successes for the 3 pairs (task-vs-recon,
task-vs-digital, recon-vs-digital); paired t on final_distance,
avg_distance, control_effort for the same 3 pairs. Descriptive only —
the tool never ranks methods or picks winners.

Identity/recovery: the frozen identity system
(results/M2/<run_id>__<hash>/ with DONE/RUNNING/FAILED markers).
Completed runs SKIP; interrupted/failed runs are RETRAINED (deterministic
by seed+config — the frozen 9C policy). M1 results are never touched:
this tool only reads results/M1 to confirm it stays intact in validate().

Subcommands (repository root):
    verify      frozen-protocol + workload hard gate (STOP on mismatch)
    plan        print the exact deterministic execution order (no side effects)
    status      per-cell state table + counts (no side effects)
    train       run the training queue (skip completed; retrain on retry)
    evaluate    run the 3-method paired-eval queue (skips completed units)
    validate    completion checks + frozen-matrix validation + M1-intact check
    checkint    per-run checkpoint integrity re-verification (60 neural runs)
    finalize    write results/M2/step11_m2_summary.json (execution facts)

Lane filter ("awgn"|"rayleigh") on train/evaluate is purely a wall-clock
scheduling device: two independent processes share the matrix by channel;
each cell still runs exactly once in its own deterministic order.

Artifacts (all gitignored):
    results/M2/<run_id>__<hash>/{final_ckpt.pt, train_report.json, DONE.json}
    results/M2/eval/k<k>_<channel>_seed<seed>/eval_<snr>.json
    results/M2/step11_m2_summary.json
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
    CHANNELS, M2_B_BY_K, M2_K_VALUES, M2_SNR_DB, TRAINING_SEEDS,
    ExperimentConfig)
from project.experiments.identity import (
    cell_state, config_hash, mark_done, mark_failed, mark_running,
    run_dir, run_id)
from project.models.reconstruction_jscc import ReconstructionDeepJSCC
from project.training.common import save_checkpoint
from project.training.train_reconstruction import train_reconstruction
from project.training.train_task_oriented import train_task_oriented

M2_ROOT = Path("results/M2")
NEURAL_METHODS = ("reconstruction", "task")
ALL_METHODS = ("reconstruction", "task", "digital")
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# frozen cells
# ---------------------------------------------------------------------------

def _packet_feasible(k, b):
    """True iff config.digital_packet accepts (k, B) without raising."""
    try:
        config.digital_packet(k, b)
        return True
    except config.InfeasibleDigitalConfig:
        return False


def _m2_cfg(method, k, channel, seed):
    """The exact frozen M2 cell (frozen values only; nothing is set here)."""
    return ExperimentConfig(
        experiment="M2", method=method, channel=channel, k=k,
        training_seed=seed, test_seed=config.TEST_SEED,
        train_steps=(config.RECON_TRAIN_STEPS if method == "reconstruction"
                     else (config.TASK_TRAIN_STEPS if method == "task" else 0)),
        batch_states=config.BATCH_STATES,
        batch_episodes=config.BATCH_EPISODES,
        t_max=config.TASK_TRAINING_T,
        lr=config.LEARNING_RATE,
        bits_per_component=(M2_B_BY_K[k] if method == "digital" else None),
        eval_snr_grid_db=M2_SNR_DB,
        eval_snr_db=10.0,
        n_train_episodes=config.N_TRAIN_EPISODES,
        n_test_episodes=config.N_TEST_EPISODES,
        eval_episodes=config.BATCH_EPISODES,
        run_mode="smoke")   # identity/vocabulary only; execution is THIS tool


def m2_neural_cells():
    """60 neural training cells in the frozen deterministic order."""
    cells = []
    for k in M2_K_VALUES:
        for channel in CHANNELS:
            for method in NEURAL_METHODS:
                for seed in TRAINING_SEEDS:
                    cells.append(_m2_cfg(method, k, channel, seed))
    return cells


def m2_digital_cells():
    """30 digital evaluation-only seed combinations (no training)."""
    return [_m2_cfg("digital", k, channel, seed)
            for k in M2_K_VALUES for channel in CHANNELS
            for seed in TRAINING_SEEDS]


def eval_units():
    """30 paired evaluation units (k, channel, seed); each unit evaluates
    reconstruction + task + digital in ONE paired call per SNR."""
    units = []
    for k in M2_K_VALUES:
        for channel in CHANNELS:
            for seed in TRAINING_SEEDS:
                units.append({
                    "unit": f"k{k}_{channel}_seed{seed}",
                    "k": k, "channel": channel, "seed": seed,
                    "B": M2_B_BY_K[k],
                    "recon": _m2_cfg("reconstruction", k, channel, seed),
                    "task": _m2_cfg("task", k, channel, seed),
                    "digital": _m2_cfg("digital", k, channel, seed),
                })
    return units


def _eval_path(unit, snr):
    return M2_ROOT / "eval" / unit["unit"] / f"eval_{snr:g}.json"


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
    print("== frozen Step-9C protocol verification (Step-11/M2 gate) ==")
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
    check("n_test_episodes==5000", config.N_TEST_EPISODES == 5000)
    check("goal_radius==0.5", config.GOAL_RADIUS == 0.5)
    check("horizon_T_max==100_dt==0.1",
          config.T_MAX == 100 and config.DT == 0.1)
    check("task_lambdas_frozen",
          (config.LAMBDA_DISTANCE, config.LAMBDA_CONTROL,
           config.LAMBDA_VELOCITY, config.LAMBDA_TERMINAL) == (1.0, 0.0, 0.0, 2.0))
    # frozen M2 matrix
    check("m2_k_values=={12,18,24,30,54}", M2_K_VALUES == (12, 18, 24, 30, 54))
    check("m2_snr_grid=={0,5,10,15,20}", M2_SNR_DB == (0.0, 5.0, 10.0, 15.0, 20.0))
    check("m2_b_by_k=={12:1,18:2,24:3,30:4,54:8}",
          M2_B_BY_K == {12: 1, 18: 2, 24: 3, 30: 4, 54: 8}
          and M2_B_BY_K == dict(config.DIGITAL_B_BY_K))
    # frozen digital baseline constants
    check("digital_K7_rate_half",
          config.CONV_CONSTRAINT_LENGTH == 7 and config.CONV_RATE == 0.5)
    check("digital_generators_G1_first",
          config.CONV_GENERATORS_OCTAL == (0o171, 0o133))
    check("digital_gray_qpsk", config.MODULATION_BITS_PER_SYMBOL == 2)
    check("digital_tail_bits==6", config.CONV_TAIL_BITS == 6)
    check("digital_feasible_all_m2_k",
          all(_packet_feasible(k, M2_B_BY_K[k]) for k in M2_K_VALUES),
          "6B+6<=k holds for every frozen M2 k")
    config.validate()
    print("config.validate() OK")
    # workload
    neural, digital = m2_neural_cells(), m2_digital_cells()
    check("workload==60_neural_training_runs", len(neural) == 60)
    check("workload==30_digital_eval_only", len(digital) == 30)
    check("workload==150_paired_eval_units",
          len(eval_units()) * len(M2_SNR_DB) == 150
          and len(eval_units()) == 30)
    check("workload==150_pooled_conditions",
          3 * 5 * 2 * len(M2_SNR_DB) == 150)
    check("tool_has_no_M1_M3_M4_M5_paths", True,
          "this tool implements M2 only (M1 read-only intactness in validate)")
    print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    return 1 if FAILS else 0


# ---------------------------------------------------------------------------
# plan / status (no side effects)
# ---------------------------------------------------------------------------

def cmd_plan():
    print("== M2 execution plan (deterministic order; NO execution) ==")
    for i, cfg in enumerate(m2_neural_cells(), 1):
        print(f"{i:2d}. {run_id(cfg):48s} steps={cfg.train_steps:5d} "
              f"batch={'states4096' if cfg.method == 'reconstruction' else 'episodes256_T100'}")
    print(f"    + {len(m2_digital_cells())} digital evaluation-only cells "
          f"(no training; B={sorted(set(M2_B_BY_K.values()))})")
    print(f"total: {len(m2_neural_cells())} neural training runs; "
          f"{len(m2_digital_cells())} digital eval-only combos; "
          f"{len(eval_units()) * len(M2_SNR_DB)} paired 3-method eval units "
          f"(30 units x 5 SNRs x 5000 episodes, test seed {config.TEST_SEED}); "
          f"pooled conditions: 150")
    return 0


def cmd_status():
    print("== M2 status ==")
    counts = {"done": 0, "failed": 0, "running": 0, "missing": 0}
    for cfg in m2_neural_cells():
        st = cell_state(cfg, results_root="results")
        counts[st] += 1
        print(f"{st:8s} {run_id(cfg)}  hash={config_hash(cfg)}")
    print(f"-- training: done={counts['done']} failed={counts['failed']} "
          f"running={counts['running']} missing={counts['missing']} (of 60)")
    ev_done = sum(1 for u in eval_units() for snr in M2_SNR_DB
                  if _eval_path(u, snr).exists())
    print(f"-- eval: {ev_done}/150 paired units complete")
    ok = counts["done"] == 60 and counts["failed"] == 0 and ev_done == 150
    print(f"-- M2 complete: {ok}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# training queue (neural runs only)
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


def _split_lane(lane):
    """Lane spec -> (channel, k) filter. "awgn"/"rayleigh" filter by channel;
    "k12".."k54" filter by k (disjoint scheduling partitions for
    multi-machine runs); None = everything."""
    if lane is None:
        return None, None
    if lane.startswith("k"):
        return None, int(lane[1:])
    return lane, None


def cmd_train(lane=None):
    """Training queue; optional lane filter ("awgn"|"rayleigh"|"k12".."k54")."""
    f_channel, f_k = _split_lane(lane)
    print("== M2 training queue (frozen 9C budgets) =="
          + (f"  [lane: {lane}]" if lane else ""))
    cells = [c for c in m2_neural_cells()
             if (f_channel is None or c.channel == f_channel)
             and (f_k is None or c.k == f_k)]
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
        if not complete:
            print(f"[{i:2d}/{total}] NUMERICAL ALERT {rid}: "
                  f"nan_batches={rep['nan_batches']} "
                  f"grads_finite={grads_finite} "
                  f"loss_finite={math.isfinite(summary[loss_key])}")
        rec = {
            "STEP": "11", "experiment": "M2",
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
                      "experiment": "M2", "method": cfg.method,
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
# checkpoint integrity (neural runs only)
# ---------------------------------------------------------------------------

def _integrity_failures(cfg):
    """11-checkpoint integrity for one neural run; list of failures."""
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


# ---------------------------------------------------------------------------
# 3-method paired evaluation queue
# ---------------------------------------------------------------------------

def _paired_stats3(cell):
    """Wilson CI per method + 3 McNemar + 3x3 paired t from one paired unit."""
    out = {}
    for name in ALL_METHODS:
        out[f"wilson_{name}"] = wilson_ci_95(cell[name]["metrics"]["success"])
    for a, b in (("task", "reconstruction"), ("task", "digital"),
                 ("reconstruction", "digital")):
        out[f"mcnemar_{a}_vs_{b}"] = mcnemar_test(
            cell[a]["metrics"]["success"], cell[b]["metrics"]["success"])
        for key in ("final_distance", "avg_distance", "control_effort"):
            out[f"paired_t_{key}_{a}_vs_{b}"] = paired_t_test(
                cell[a]["metrics"][key].to(torch.float64),
                cell[b]["metrics"][key].to(torch.float64))
    return out


def cmd_evaluate(lane=None):
    """3-method paired-eval queue; lane filter ("awgn"|"rayleigh"|"k12".."k54")."""
    f_channel, f_k = _split_lane(lane)
    print("== M2 paired evaluation queue (Step-7 harness, 3 methods, "
          "5000 episodes) ==" + (f"  [lane: {lane}]" if lane else ""))
    print("-- per-run checkpoint integrity gate --")
    blocked = []
    neural = [c for c in m2_neural_cells()
              if (f_channel is None or c.channel == f_channel)
              and (f_k is None or c.k == f_k)]
    for cfg in neural:
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
          f"{lane or 'all'} ({len(neural)} runs)")
    n_new = 0
    total_wall = 0.0
    units = [u for u in eval_units()
             if (f_channel is None or u["channel"] == f_channel)
             and (f_k is None or u["k"] == f_k)]
    for u in units:
        m_recon = _load_model(u["recon"])
        m_task = _load_model(u["task"])
        cond_base = dict(k=u["k"], channel=u["channel"],
                         bits_per_component=u["B"],
                         t_max=config.T_MAX,
                         episodes=config.N_TEST_EPISODES,
                         seed=config.TEST_SEED)
        for snr in M2_SNR_DB:
            out = _eval_path(u, snr)
            if out.exists():
                continue
            cond = Condition(snr_db=snr, **cond_base)
            t0 = time.perf_counter()
            with torch.no_grad():
                cell = evaluate_condition(
                    cond, methods=list(ALL_METHODS),
                    recon_model=m_recon, task_model=m_task)
            wall = time.perf_counter() - t0
            total_wall += wall
            rec = {
                "STEP": "11", "experiment": "M2",
                "unit": u["unit"], "k": u["k"], "channel": u["channel"],
                "seed": u["seed"], "bits_per_component": u["B"],
                "snr_db": snr,
                "test_seed": config.TEST_SEED,
                "episodes": config.N_TEST_EPISODES, "t_max": config.T_MAX,
                "paired": True,
                "methods": list(ALL_METHODS),
                "wall_s": round(wall, 1),
                **{f"aggregate_{name}": cell[name]["aggregate"]
                   for name in ALL_METHODS},
                **_paired_stats3(cell),
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=2, default=str))
            n_new += 1
            sr = {name: rec[f"aggregate_{name}"]["success_rate"]
                  for name in ALL_METHODS}
            print(f"  EVAL {u['unit']} snr={snr:5.1f} "
                  f"SR recon={sr['reconstruction']:.4f} "
                  f"task={sr['task']:.4f} digital={sr['digital']:.4f} "
                  f"({wall:.1f}s)")
        del m_recon, m_task
    print(f"-- evaluation queue finished: {n_new} new units this session, "
          f"eval_wall={total_wall:.1f}s total")
    return 0


# ---------------------------------------------------------------------------
# checkint / validate
# ---------------------------------------------------------------------------

def cmd_checkint():
    bad = {}
    for cfg in m2_neural_cells():
        f = _integrity_failures(cfg)
        if f:
            bad[run_id(cfg)] = f
    for rid, f in bad.items():
        print(f"FAIL {rid}: {f}")
    print(f"== checkpoint integrity: {60 - len(bad)}/60 runs pass ==")
    return 1 if bad else 0


def cmd_validate():
    print("== M2 completion validation ==")
    FAILS.clear()
    cells = m2_neural_cells()
    ids = [(run_id(c), config_hash(c)) for c in cells]
    check("60-unique-identities", len(set(ids)) == 60)
    done = sum(1 for c in cells
               if cell_state(c, results_root="results") == "done")
    check("60-training-runs-complete", done == 60, f"{done}/60")
    check("methods==reconstruction,task",
          sorted({c.method for c in cells}) == ["reconstruction", "task"])
    check("k=={12,18,24,30,54}", sorted({c.k for c in cells})
          == [12, 18, 24, 30, 54])
    check("channels==awgn,rayleigh",
          sorted({c.channel for c in cells}) == ["awgn", "rayleigh"])
    check("seeds=={42,43,44}",
          sorted({c.training_seed for c in cells}) == [42, 43, 44])
    check("no-m1-m3-m4-m5-cells",
          not any(c.experiment != "M2" for c in cells))
    # eval completeness: 150 paired 3-method units, 5000 episodes, seed 10042
    missing = [f"{u['unit']}@{snr:g}" for u in eval_units()
               for snr in M2_SNR_DB if not _eval_path(u, snr).exists()]
    check("150-eval-units-complete", len(missing) == 0,
          f"missing={missing[:6]}{'...' if len(missing) > 6 else ''}")
    pooled, eps_ok, pair_ok, methods_ok = {}, True, True, True
    for u in eval_units():
        for snr in M2_SNR_DB:
            p = _eval_path(u, snr)
            if not p.exists():
                continue
            rec = json.loads(p.read_text())
            if rec.get("episodes") != 5000 or rec.get("test_seed") != 10042:
                eps_ok = False
            if not rec.get("paired"):
                pair_ok = False
            if sorted(rec.get("methods", [])) != sorted(ALL_METHODS):
                methods_ok = False
            for method in ALL_METHODS:
                key = (method, u["k"], u["channel"], snr)
                pooled.setdefault(key, []).append(rec[f"aggregate_{method}"])
    check("episodes==5000-test-seed==10042-everywhere", eps_ok)
    check("paired-true-everywhere", pair_ok)
    check("three-methods-in-every-unit", methods_ok)
    check("150-pooled-eval-conditions", len(pooled) == 150, f"{len(pooled)}")
    check("3-seeds-per-pooled-cell",
          all(len(v) == 3 for v in pooled.values()))
    # frozen matrix expectations + structural validation of the proposal
    from project.experiments.matrix import expected_cells, validate_matrix
    from main import _matrix_proposal
    ec = expected_cells()
    check("matrix-expected-M2-counts",
          ec["M2_neural_training_runs"] == 60
          and ec["M2_digital_eval_only_seeds"] == 30
          and ec["M2_eval_conditions"] == 150)
    res = validate_matrix(_matrix_proposal())
    check("frozen-matrix-proposal-valid", res["valid"],
          f"warnings={len(res['warnings'])}")
    # every unit carries all three methods' aggregates
    both = True
    for u in eval_units():
        for snr in M2_SNR_DB:
            p = _eval_path(u, snr)
            if not p.exists():
                continue
            rec = json.loads(p.read_text())
            if any(f"aggregate_{m}" not in rec for m in ALL_METHODS):
                both = False
    check("all-methods-in-every-unit", both)
    # M1 intactness (read-only): M1 artifacts must be untouched
    m1_done = len(list(Path("results/M1").glob("*__*/DONE.json")))
    m1_eval = len(list(Path("results/M1/eval").glob("*/eval_*.json")))
    m1_sum = Path("results/M1/step10_m1_summary.json")
    m1_status = (json.loads(m1_sum.read_text()).get("status")
                 if m1_sum.exists() else "missing")
    check("m1-intact-36-train-198-eval",
          m1_done == 36 and m1_eval == 198 and m1_status == "complete",
          f"train={m1_done}/36 eval={m1_eval}/198 summary={m1_status}")
    print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    return 1 if FAILS else 0


# ---------------------------------------------------------------------------
# finalize — execution-facts summary (descriptive, no ranking)
# ---------------------------------------------------------------------------

def cmd_finalize():
    print("== finalize: results/M2/step11_m2_summary.json ==")
    cells = m2_neural_cells()
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
        for snr in M2_SNR_DB:
            p = _eval_path(u, snr)
            if not p.exists():
                continue
            n_units += 1
            rec = json.loads(p.read_text())
            for method in ALL_METHODS:
                key = (method, u["k"], u["channel"], snr)
                pooled.setdefault(key, []).append(
                    rec[f"aggregate_{method}"])
                w = rec[f"wilson_{method}"]
                wilson.setdefault(key, []).append(
                    (w["success_count"], w["total"]))
    pool_rows = []
    for (method, k, ch, snr) in sorted(pooled.keys(),
                                       key=lambda x: (x[2], x[3], x[0], x[1])):
        aggs = pooled[(method, k, ch, snr)]
        row = {"method": method, "k": k, "channel": ch, "snr_db": snr,
               "n_seeds": len(aggs)}
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
        "STEP": "11", "experiment": "M2",
        "status": ("complete" if len(train_rows) == 60 and n_units == 150
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
            "m2_k_values": list(M2_K_VALUES),
            "m2_snr_grid_db": list(M2_SNR_DB),
            "m2_digital_b_by_k": {str(k): b for k, b in M2_B_BY_K.items()},
            "eval_episodes": config.N_TEST_EPISODES,
            "eval_horizon": {"T_max": config.T_MAX, "dt": config.DT},
            "goal_radius": config.GOAL_RADIUS,
            "m5_status": config.M5_STATUS},
        "training_runs": {"expected": 60, "completed": len(train_rows),
                          "failed": sum(1 for r in train_rows
                                        if r["exit_status"] != "complete")},
        "digital_baseline": {"evaluation_only": True,
                             "seed_combinations": 30,
                             "matched_budgets": {str(k): b
                                                 for k, b in M2_B_BY_K.items()}},
        "eval": {"paired_units_expected": 150, "paired_units_complete": n_units,
                 "pooled_conditions": len(pool_rows),
                 "episodes_per_unit_per_method": config.N_TEST_EPISODES,
                 "methods": list(ALL_METHODS)},
        "total_train_wall_s": round(total_train_s, 1),
        "interpretation_note": ("Execution report only; no overall ranking "
                               "or scientific verdict is made in this step. "
                               "Three-way comparison is empirical: M1 showed "
                               "task-oriented UNDERPERFORMED reconstruction in "
                               "the low-k regime, so no superiority is "
                               "presumed. Pooled rows are descriptive "
                               "(pooled success over the 3 training-seed rows "
                               "+ mean/std of per-seed aggregates); paired "
                               "per-seed McNemar/paired-t results live in "
                               "the eval unit JSONs."),
        "train_rows": train_rows,
        "pooled_rows": pool_rows,
    }
    out = M2_ROOT / "step11_m2_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out} ({len(train_rows)} train rows, "
          f"{len(pool_rows)} pooled rows, {n_units}/150 units)")
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
        valid = (lane is None or lane in ("awgn", "rayleigh")
                 or (lane.startswith("k") and lane[1:].isdigit()
                     and int(lane[1:]) in M2_K_VALUES))
        if not valid:
            print(f"unknown lane {lane!r} "
                  f"(use awgn|rayleigh|k12|k18|k24|k30|k54 or none)")
            sys.exit(1)
        sys.exit(dispatch[cmd](lane))
    sys.exit(dispatch[cmd]())
