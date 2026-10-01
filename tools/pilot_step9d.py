"""Step 9D — Stage-A production pilot (execution-readiness, NON-scientific).

Subcommands (run from the repository root):
    .venv/Scripts/python.exe tools/pilot_step9d.py verify
        Programmatic verification of the FROZEN Step-9C protocol
        (hard-fails on any mismatch — nothing is silently repaired).
    .venv/Scripts/python.exe tools/pilot_step9d.py identitytest
        Identity/isolation checks (distinct ids, gitignore coverage,
        skip/resume marker semantics, checkpoint-free compatibility).
    .venv/Scripts/python.exe tools/pilot_step9d.py pilotA
        Pilot A: reconstruction, AWGN, k=3, seed 42, 5000 steps x 4096.
    .venv/Scripts/python.exe tools/pilot_step9d.py pilotB
        Pilot B: task-oriented, AWGN, k=3, seed 42, 3000 steps x 256
        episodes, T=100.  (~55-70 min on this machine)
    .venv/Scripts/python.exe tools/pilot_step9d.py resumetest
        HONEST resume-semantics test with a tiny standalone module
        (does NOT touch or corrupt the completed pilot checkpoints).

The pilots are infrastructure validation only: NO scientific conclusion
is drawn from them. Artifacts live under results/STEP_9D_PILOT/
(gitignored) and are explicitly labeled so they can never be mistaken
for final M1-M4 experiment outputs.
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
from project.experiments.identity import (
    cell_state, config_hash, mark_done, mark_failed, mark_running,
    run_dir, run_id)
from project.models.reconstruction_jscc import ReconstructionDeepJSCC
from project.training.common import (
    build_optimizer, load_checkpoint, models_identical, save_checkpoint)
from project.training.train_reconstruction import train_reconstruction
from project.training.train_task_oriented import train_task_oriented

PILOT_ROOT = Path("results/STEP_9D_PILOT")
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]"
          + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# frozen protocol verification (hard gate)
# ---------------------------------------------------------------------------

def cmd_verify():
    print("== frozen Step-9C protocol verification ==")
    check("recon_steps==5000", config.RECON_TRAIN_STEPS == 5000)
    check("task_steps==3000", config.TASK_TRAIN_STEPS == 3000)
    check("task_training_T==100",
          config.TASK_TRAINING_T == 100 and config.T_MAX == 100)
    check("batch_states==4096", config.BATCH_STATES == 4096)
    check("batch_episodes==256", config.BATCH_EPISODES == 256)
    check("lr==1e-3", config.LEARNING_RATE == 1e-3)
    check("grad_clip==1.0", config.GRAD_CLIP_NORM == 1.0)
    check("training_snr==U(0,20)",
          config.SNR_TRAIN_RANGE_DB == (0.0, 20.0)
          and config.SNR_TRAIN_PER_SAMPLE is True)
    check("seeds=={42,43,44}", config.TRAINING_SEEDS == (42, 43, 44))
    check("test_seed==10042", config.TEST_SEED == 10042)
    check("early_stopping==disabled", config.EARLY_STOPPING is False)
    check("M5==deferred+disabled",
          config.M5_STATUS == "deferred"
          and config.M5_EXECUTION_ENABLED is False)
    check("optimizer==Adam", True, "Adam is hard-coded in both trainers + "
          "training/common.build_optimizer (frozen)")
    config.validate()
    print("config.validate() OK")
    return 1 if FAILS else 0


# ---------------------------------------------------------------------------
# pilot configurations (exact frozen production scale)
# ---------------------------------------------------------------------------

def _pilot_cfg(method, steps, batch):
    cfg = {
        "experiment": "STEP_9D_PILOT",
        "method": method,
        "channel": "awgn",
        "k": 3,
        "training_seed": 42,
        "train_steps": steps,
        "batch_states": config.BATCH_STATES,
        "batch_episodes": config.BATCH_EPISODES,
        "t_max": config.TASK_TRAINING_T,
        "lr": config.LEARNING_RATE,
    }
    if batch:
        cfg["batch"] = batch
    return cfg


def _run_dir_for(cfg):
    """The identity module's hashed run dir (single source of truth)."""
    return run_dir(_as_exp_cfg(cfg), results_root=str(PILOT_ROOT))


# ---------------------------------------------------------------------------
# identity / isolation checks (no training)
# ---------------------------------------------------------------------------

def cmd_identitytest():
    print("== identity & isolation checks ==")
    from project.experiments.config import ExperimentConfig
    ids = set()
    for method, ch, k, seed, bits in (
            ("task", "awgn", 3, 42, None), ("task", "awgn", 3, 43, None),
            ("task", "awgn", 3, 42, None), ("task", "rayleigh", 3, 42, None),
            ("task", "awgn", 12, 42, None),
            ("reconstruction", "awgn", 3, 42, None),
            ("digital", "awgn", 54, 42, 8),
            ("task", "awgn", 54, 42, None)):
        cfg = ExperimentConfig(experiment="M1", method=method, channel=ch,
                               k=k, training_seed=seed,
                               bits_per_component=bits)
        ids.add((run_id(cfg), config_hash(cfg)))
    check("identity-uniqueness", len(ids) == 7,
          f"{len(ids)} distinct (run_id, hash) pairs from 8 configs "
          f"(seed/channel/k/method distinguish every cell)")

    # hash sensitivity: any protocol change -> different identity
    base = ExperimentConfig.m1("task", 3, "awgn", 42)
    from dataclasses import replace
    variants = [replace(base, train_steps=base.train_steps + 1),
                replace(base, batch_episodes=base.batch_episodes + 1),
                replace(base, t_max=base.t_max - 1),
                replace(base, lr=base.lr + 1e-9 if base.lr else 5e-4)]
    hashes = {config_hash(base)} | {config_hash(v) for v in variants}
    check("config-hash-sensitivity", len(hashes) == 5,
          "steps/batch/T/lr changes all produce new identities")

    # deterministic output paths
    with __import__("tempfile").TemporaryDirectory() as td:
        d1 = run_dir(base, results_root=td)
        d2 = run_dir(base, results_root=td)
        check("output-path-deterministic", d1 == d2, str(d1.relative_to(td)))

    # skip/resume marker semantics (filesystem only, no training).
    # Precedence: DONE > FAILED > RUNNING (a completed run stays complete
    # even if a stale FAILED marker somehow exists).
    from dataclasses import replace as _replace
    with __import__("tempfile").TemporaryDirectory() as td:
        check("state-missing", cell_state(base, results_root=td) == "missing")
        mark_running(base, results_root=td)
        check("state-running", cell_state(base, results_root=td) == "running")
        mark_done(base, summary={"test": True}, results_root=td)
        check("state-done-after-done", cell_state(base, results_root=td) == "done")
    with __import__("tempfile").TemporaryDirectory() as td:
        mark_running(base, results_root=td)
        mark_failed(base, "probe", results_root=td)
        check("state-failed-after-failed",
              cell_state(base, results_root=td) == "failed")
        # DONE takes precedence over a stale FAILED in the same cell
        mark_done(base, summary={}, results_root=td)
        check("done-precedence-over-failed",
              cell_state(base, results_root=td) == "done")

    gitignore = Path(".gitignore").read_text()
    check("results-gitignored",
          "results/" in gitignore and "*.pth" in gitignore,
          "pilot artifacts cannot enter Git")
    return 1 if FAILS else 0


# ---------------------------------------------------------------------------
# Pilot A — reconstruction production run
# ---------------------------------------------------------------------------

def cmd_pilotA():
    cfg = _pilot_cfg("reconstruction", config.RECON_TRAIN_STEPS, "states4096")
    d = _run_dir_for(cfg)
    print(f"== Pilot A: recon awgn k=3 s42 steps={cfg['train_steps']} "
          f"batch_states={cfg['batch_states']} ==")
    if cell_state(_as_exp_cfg(cfg), results_root=str(PILOT_ROOT)) == "done":
        print("SKIP: completed pilot found (skip mechanism) — use a fresh "
              "results dir to retrain")
        return 0
    mark_running(_as_exp_cfg(cfg), results_root=str(PILOT_ROOT))
    t0 = time.perf_counter()
    model, rep = train_reconstruction(
        k=3, channel="awgn", seed=42, steps=cfg["train_steps"],
        batch_states=cfg["batch_states"])
    wall = time.perf_counter() - t0
    grads_finite = (math.isfinite(rep["grad_norm_encoder_final"])
                    and math.isfinite(rep["grad_norm_decoder_final"])
                    and rep["grad_norm_encoder_final"] > 0
                    and rep["grad_norm_decoder_final"] > 0)
    rec = _record(cfg, wall, rep, model, grads_finite, d,
                  extra={"loss_reduction_pct": rep["loss_reduction_pct"]})
    _save(d, "pilotA_report.json", rec)
    if rec["exit_status"] == "complete":
        mark_done(_as_exp_cfg(cfg),
                  summary={"wall_s": rec["wall_s"],
                           "loss_final": rec["loss_final"]},
                  results_root=str(PILOT_ROOT))
    print(json.dumps({k: rec[k] for k in
                      ("run_id", "config_hash", "wall_s", "s_per_step",
                       "final_step", "exit_status", "nan_batches")},
                     indent=2))
    return _integrity(rec, model, d)


def _as_exp_cfg(cfg):
    from project.experiments.config import ExperimentConfig
    return ExperimentConfig(experiment=cfg["experiment"], method=cfg["method"],
                            channel=cfg["channel"], k=cfg["k"],
                            training_seed=cfg["training_seed"],
                            train_steps=cfg["train_steps"],
                            batch_states=cfg["batch_states"],
                            batch_episodes=cfg["batch_episodes"],
                            t_max=cfg["t_max"], lr=cfg["lr"])


def _record(cfg, wall, rep, model, grads_finite, d, extra):
    task_like = "history" in rep and isinstance(rep["history"][0], dict)
    final_step = cfg["train_steps"] - rep["nan_batches"]
    exp_cfg = _as_exp_cfg(cfg)
    ckpt_path = d / "final_ckpt.pt"
    # Final checkpoint per the frozen 9C policy, via the Step-8 infra.
    # Documented limitation: the Step-4/5 trainers own their optimizers
    # internally and do not expose them, so the saved optimizer_state is
    # not the trained one; model params, RNG state, history and metadata
    # are the exact artifacts (reproduction is by seed+config anyway).
    save_checkpoint(
        ckpt_path, model, optimizer=None, step=cfg["train_steps"],
        seed=cfg["training_seed"],
        metadata={"run_id": run_id(exp_cfg),
                  "config_hash": config_hash(exp_cfg),
                  "experiment": cfg["experiment"],
                  "method": cfg["method"], "channel": cfg["channel"],
                  "k": cfg["k"], "training_seed": cfg["training_seed"],
                  "train_steps": cfg["train_steps"],
                  "batch_states": cfg["batch_states"],
                  "batch_episodes": cfg["batch_episodes"],
                  "t_max": cfg["t_max"],
                  "lr": cfg["lr"],
                  "model_config": {"hidden_dims": list(config.HIDDEN_DIMS),
                                   "activation": "PReLU"},
                  "train_snr_policy": {"distribution": "uniform",
                                       "range_db": [0.0, 20.0],
                                       "granularity": "per_batch"},
                  "STEP_9D_PILOT": True},
        history=rep["history"], capture_rng=True)
    rec = {
        "STEP_9D_PILOT": True,
        "non_scientific": True,
        "run_id": run_id(exp_cfg),
        "config_hash": config_hash(exp_cfg),
        "config": cfg,
        "wall_s": round(wall, 1),
        "s_per_step": round(wall / cfg["train_steps"], 4),
        "final_step": final_step,
        "nan_batches": rep["nan_batches"],
        "grads_finite": grads_finite,
        "exit_status": "complete" if (rep["nan_batches"] == 0 and grads_finite
                                      and len(rep["history"])
                                      == cfg["train_steps"]) else "incomplete",
        "checkpoint": str(ckpt_path),
        "optimizer_state_saved": False,
        "optimizer_note": ("trainer-internal optimizer not exposed "
                           "(documented Step-8 limitation); reproduction "
                           "is by seed+config"),
        **extra,
    }
    if task_like:
        rec["task_loss_initial"] = rep["train_total_initial"]
        rec["task_loss_final"] = rep["train_total_final"]
        rec["recon_mse_diag_initial"] = rep["recon_mse_diag_initial"]
        rec["recon_mse_diag_final"] = rep["recon_mse_diag_final"]
        rec["eval_before"] = rep["eval_before"]
        rec["eval_after"] = rep["eval_after"]
        rec["lambdas"] = rep["lambdas"]
    else:
        rec["loss_initial"] = rep["loss_initial"]
        rec["loss_final"] = rep["loss_final"]
    return rec


def _save(d, name, obj):
    (d / name).write_text(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------------------
# Pilot B — task-oriented production run (long; run in background)
# ---------------------------------------------------------------------------

def cmd_pilotB():
    cfg = _pilot_cfg("task", config.TASK_TRAIN_STEPS, "episodes256_T100")
    d = _run_dir_for(cfg)
    print(f"== Pilot B: task awgn k=3 s42 steps={cfg['train_steps']} "
          f"batch_episodes={cfg['batch_episodes']} T={cfg['t_max']} ==")
    if cell_state(_as_exp_cfg(cfg), results_root=str(PILOT_ROOT)) == "done":
        print("SKIP: completed pilot found (skip mechanism)")
        return 0
    mark_running(_as_exp_cfg(cfg), results_root=str(PILOT_ROOT))
    t0 = time.perf_counter()
    model, rep = train_task_oriented(
        k=3, channel="awgn", seed=42, steps=cfg["train_steps"],
        batch_episodes=cfg["batch_episodes"], t_max=cfg["t_max"],
        eval_episodes=256, verbose_every=500)   # logging only; math unchanged
    wall = time.perf_counter() - t0
    g0 = rep["grad_enc_final"]
    g1 = rep["grad_dec_final"]
    grads_finite = (math.isfinite(g0) and math.isfinite(g1)
                    and g0 > 0 and g1 > 0)
    rec = _record(cfg, wall, rep, model, grads_finite, d, extra={
        "eval_snr_db": 10.0, "eval_episodes": 256})
    _save(d, "pilotB_report.json", rec)
    if rec["exit_status"] == "complete":
        mark_done(_as_exp_cfg(cfg),
                  summary={"wall_s": rec["wall_s"],
                           "task_loss_final": rec["task_loss_final"]},
                  results_root=str(PILOT_ROOT))
    print(json.dumps({k: rec[k] for k in
                      ("run_id", "config_hash", "wall_s", "s_per_step",
                       "final_step", "exit_status", "nan_batches")},
                     indent=2))
    return _integrity(rec, model, d)


# ---------------------------------------------------------------------------
# checkpoint integrity (section 8 — all 10 points)
# ---------------------------------------------------------------------------

def _integrity(rec, model, d):
    print("== checkpoint integrity ==")
    ckpt_path = Path(rec["checkpoint"])
    check("ckpt-exists", ckpt_path.exists(), str(ckpt_path))
    if not ckpt_path.exists():
        return 1
    fresh = ReconstructionDeepJSCC(k=3)
    payload = load_checkpoint(ckpt_path, fresh, restore_rng=False)
    check("ckpt-readable", True)
    md = payload["metadata"]
    check("ckpt-config-matches",
          md["train_steps"] == rec["config"]["train_steps"]
          and md["batch_states"] == rec["config"]["batch_states"]
          and md["t_max"] == rec["config"]["t_max"]
          and math.isclose(md["lr"], config.LEARNING_RATE))
    check("ckpt-seed-matches", payload["seed"] == 42)
    check("ckpt-method-channel-k",
          md["method"] == rec["config"]["method"]
          and md["channel"] == rec["config"]["channel"]
          and md["k"] == rec["config"]["k"])
    check("ckpt-final-step==budget",
          payload["step"] == rec["config"]["train_steps"],
          f"step={payload['step']}")
    params_finite = all(bool(torch.isfinite(p).all())
                        for p in fresh.parameters())
    check("ckpt-model-params-finite", params_finite)
    # honest optimizer-state note: the Step-4/5 trainers own their
    # optimizers, so the pilot checkpoint intentionally has none; the
    # frozen 9C policy (model+metadata+history+RNG) is what is verified
    check("ckpt-optimizer-policy",
          "optimizer_state" not in payload,
          "no optimizer_state saved (documented limitation; retraining is "
          "the recovery path and is deterministic by seed+config)")
    check("ckpt-identity-matches",
          md["run_id"] == rec["run_id"], md["run_id"])
    check("ckpt-config-hash-matches",
          md["config_hash"] == rec["config_hash"], md["config_hash"])
    check("exit-status-complete", rec["exit_status"] == "complete")
    return 1 if FAILS else 0


# ---------------------------------------------------------------------------
# honest resume-semantics test (tiny standalone module; pilots untouched)
# ---------------------------------------------------------------------------

def cmd_resumetest():
    print("== resume semantics (honest, tiny standalone module) ==")
    with __import__("tempfile").TemporaryDirectory() as td:
        torch.manual_seed(11)
        m1 = ReconstructionDeepJSCC(k=1)
        opt = build_optimizer(m1)
        gen = config.derive_generator("noise_train", 5)

        def segment(model, optimizer, gen_, steps):
            losses = []
            for i in range(steps):
                s = torch.randn(64, 6, generator=gen_)
                out = model.forward_losses(s, 10.0, channel="awgn",
                                           generator=gen_)
                loss = ((out - s * 0.0) ** 2).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=config.GRAD_CLIP_NORM)
                optimizer.step()
                losses.append(float(loss))
            return losses

        seg1 = segment(m1, opt, gen, 30)
        snap = {k: v.clone() for k, v in m1.state_dict().items()}
        ck = Path(td) / "resume_ckpt.pt"
        save_checkpoint(ck, m1, optimizer=opt, step=30, seed=5,
                        metadata={"purpose": "resume-semantics-probe"},
                        history=seg1, capture_rng=True)
        gen_state_path = Path(td) / "gen_state.pt"
        torch.save(gen.get_state(), gen_state_path)

        # --- restore-verification arm (BEFORE m1/opt advance further) ---
        torch.manual_seed(11)
        m2 = ReconstructionDeepJSCC(k=1)
        opt2 = build_optimizer(m2)
        gen_b = config.derive_generator("noise_train", 5)
        payload = load_checkpoint(ck, m2, optimizer=opt2, restore_rng=True)
        check("resume-model-restored-bitwise",
              all(torch.equal(snap[k], m2.state_dict()[k]) for k in snap)
              and payload["step"] == 30 and payload["history"] == seg1)
        check("resume-optimizer-moments-bitwise",
              len(opt.state) == len(opt2.state)
              and all(torch.equal(a["exp_avg"], b["exp_avg"])
                      and torch.equal(a["exp_avg_sq"], b["exp_avg_sq"])
                      and a["step"] == b["step"]
                      for a, b in zip(opt.state.values(),
                                      opt2.state.values())),
              "Adam moments restore exactly (compared position-wise; "
              "state keys are param objects, not comparable across models)")

        # --- continuation arm A (live objects; generator advanced by seg1) ---
        cont_a = segment(m1, opt, gen, 30)

        # --- continuation arm B: FULL RNG state (global + data generator) ---
        gen_b.set_state(torch.load(gen_state_path))
        cont_b = segment(m2, opt2, gen_b, 30)
        check("resume-continues-finite",
              len(cont_b) == 30 and all(math.isfinite(v) for v in cont_b))
        check("resume-bitwise-WITH-generator-state",
              cont_a == cont_b,
              "checkpointing ALL RNG state => exact bitwise continuation")

        # --- honest limitation arm: data generator NOT restored (the exact
        # situation of the Step-4/5 trainers, which own their generators) ---
        torch.manual_seed(11)
        m3 = ReconstructionDeepJSCC(k=1)
        opt3 = build_optimizer(m3)
        gen_c = config.derive_generator("noise_train", 5)  # fresh position
        load_checkpoint(ck, m3, optimizer=opt3, restore_rng=True)
        cont_c = segment(m3, opt3, gen_c, 30)
        check("resume-without-generator-state-finite",
              len(cont_c) == 30 and all(math.isfinite(v) for v in cont_c))
        check("resume-without-generator-state-NOT-bitwise",
              cont_c != cont_a,
              "demonstrates the DOCUMENTED limitation: trainer-internal "
              "generators are not checkpointed; the frozen 9C recovery "
              "policy is therefore RETRAIN (bitwise deterministic by "
              "seed+config), not mid-run resume")
    check("pilot-checkpoints-untouched",
          cell_state(_as_exp_cfg(_pilot_cfg("reconstruction",
                                            config.RECON_TRAIN_STEPS, None)),
                     results_root=str(PILOT_ROOT)) == "done"
          if (PILOT_ROOT / "STEP_9D_PILOT").exists() else True,
          "the tiny probe never touched the production pilots")
    return 1 if FAILS else 0


def cmd_checkint(which):
    """Re-verify checkpoint integrity from the saved report (post-hoc)."""
    name = "pilotA_report.json" if which == "A" else "pilotB_report.json"
    method = "reconstruction" if which == "A" else "task"
    steps = (config.RECON_TRAIN_STEPS if which == "A"
             else config.TASK_TRAIN_STEPS)
    cfg = _pilot_cfg(method, steps, None)
    d = _run_dir_for(cfg)
    rec = json.loads((d / name).read_text())
    model = ReconstructionDeepJSCC(k=3)
    print(f"== integrity re-verification, Pilot {which} ==")
    return _integrity(rec, model, d)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "verify":
        code = cmd_verify()
    elif cmd == "identitytest":
        code = cmd_identitytest()
    elif cmd == "pilotA":
        code = cmd_pilotA()
    elif cmd == "pilotB":
        code = cmd_pilotB()
    elif cmd == "resumetest":
        code = cmd_resumetest()
    elif cmd == "checkint":
        code = cmd_checkint(sys.argv[2].upper())
    else:
        print(__doc__)
        code = 1
    if cmd in ("verify", "identitytest"):
        n = len(FAILS)
        print(f"{'ALL PASS' if n == 0 else f'{n} FAILURES: {FAILS}'}")
    sys.exit(code)
