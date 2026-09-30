"""Step 8 experiments tests (Gates C-L).

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_experiments

Gate C: experiment-config validation (frozen seeds, run modes, methods).
Gate D: frozen M1-M5 matrix validation (budgets, grids, seeds, M2/M4
        confusion, M3 expected infeasibility).
Gate E: digital budget/infeasibility (config-level + harness status).
Gate F: dry-run produces NO training side effects.
Gate G: smoke reconstruction training (runner -> train -> eval ->
        checkpoint round-trip).
Gate H: smoke task-oriented training.
Gate I: smoke AWGN + Rayleigh paths.
Gate J: evaluation-harness integration via ExperimentConfig conditions.
Gate K: deterministic rerun (bitwise histories, eval rows, model state,
        evaluation realizations).
Gate L: in-process core regression of Steps 0-7 (the FULL suites are also
        run as separate module invocations - see the checkpoint log).
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch

from project import config
from project.evaluation.evaluate import Condition, evaluate_condition, generate_realizations
from project.experiments.config import (
    CHANNELS,
    M1_K_VALUES,
    M1_SNR_DB,
    M2_K_VALUES,
    M2_SNR_DB,
    TRAINING_SEEDS,
    ExperimentConfig,
    digital_feasible,
)
from project.experiments.matrix import expected_cells, validate_matrix
from project.experiments.runner import dry_run, run_full, run_smoke
from project.models.reconstruction_jscc import ReconstructionDeepJSCC
from project.training.common import load_checkpoint, models_identical
from project.baselines.digital import DigitalBaseline

SUMMARY = []


def check(name, ok, detail=""):
    SUMMARY.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]" +
          (f"  ({detail})" if detail else ""))


def _smoke_recon_cfg(channel="awgn", k=2, seed=42, **kw):
    base = dict(train_steps=30, batch_states=512,
                eval_snr_grid_db=(5.0, 10.0), t_max=15,
                n_test_episodes=25, run_mode="smoke")
    base.update(kw)
    return ExperimentConfig(experiment="M1", method="reconstruction",
                            channel=channel, k=k, training_seed=seed, **base)


def _smoke_task_cfg(channel="awgn", k=2, seed=42, **kw):
    base = dict(train_steps=15, batch_episodes=32, t_max=12,
                eval_snr_grid_db=(5.0,), n_test_episodes=20,
                eval_episodes=32, eval_snr_db=10.0, run_mode="smoke")
    base.update(kw)
    return ExperimentConfig(experiment="M1", method="task",
                            channel=channel, k=k, training_seed=seed, **base)


# ---------------------------------------------------------------------------
# Gate C - configuration validation
# ---------------------------------------------------------------------------

def test_gate_c():
    ok_valid = ExperimentConfig.m1("task", 3, "awgn", 42).validate() == []
    check("C/valid-m1", ok_valid)
    ok_m2 = ExperimentConfig.m2("digital", 24, "rayleigh", 43).validate() == []
    check("C/valid-m2-digital", ok_m2)

    def errs(**kw):
        base = dict(experiment="M1", method="task", channel="awgn",
                    k=3, training_seed=42)
        base.update(kw)
        return ExperimentConfig(**base).validate()

    bad = ExperimentConfig(experiment="M9", method="task", channel="awgn",
                           k=3, training_seed=42).validate()
    check("C/unknown-experiment", any("unknown experiment" in e for e in bad))
    bad = ExperimentConfig(experiment="M1", method="quantum", channel="awgn",
                           k=3, training_seed=42).validate()
    check("C/unknown-method", any("unknown method" in e for e in bad))
    bad = errs(channel="wifi")
    check("C/unknown-channel", any("unknown channel" in e for e in bad))
    bad = errs(run_mode="hyper")
    check("C/unknown-run_mode", any("unknown run_mode" in e for e in bad))
    bad = errs(training_seed=7)
    check("C/bad-training-seed",
          any("not in" in e for e in bad), "7 not in (42, 43, 44)")
    bad = errs(test_seed=1)
    check("C/bad-test-seed", any("test seed" in e for e in bad),
          "must be 10042")
    bad = errs(k=0)
    check("C/bad-k", any("k must be" in e for e in bad))
    bad = errs(run_mode="full")
    check("C/full-blocked", any("BLOCKED" in e for e in bad),
          "full-scale execution refused at config level")
    bad = errs(method="digital")   # no bits_per_component
    check("C/digital-needs-B", any("requires bits_per_component" in e
                                   for e in bad))
    try:
        ExperimentConfig.m1("task", 3, "awgn", 42).validate_or_raise()
        ExperimentConfig(experiment="M1", method="task", channel="awgn",
                         k=3, training_seed=7).validate_or_raise()
        ok_raise = False
    except ValueError:
        ok_raise = True
    check("C/validate_or_raise", ok_raise)


# ---------------------------------------------------------------------------
# Gate D - frozen matrix validation
# ---------------------------------------------------------------------------

def _full_proposal():
    cells = []
    for method in ("task", "reconstruction"):
        for k in M1_K_VALUES:
            for ch in CHANNELS:
                for seed in TRAINING_SEEDS:
                    cells.append(ExperimentConfig.m1(method, k, ch, seed,
                                                     run_mode="dry_run"))
    for method in ("digital", "task", "reconstruction"):
        for k in M2_K_VALUES:
            for ch in CHANNELS:
                for seed in TRAINING_SEEDS:
                    cells.append(ExperimentConfig.m2(method, k, ch, seed,
                                                     run_mode="dry_run"))
    cells.append(ExperimentConfig.m3(1, bits=1, run_mode="dry_run"))
    cells.append(ExperimentConfig.m3(2, bits=1, run_mode="dry_run"))
    cells.append(ExperimentConfig.m3(3, bits=1, run_mode="dry_run"))
    for ch in CHANNELS:
        cells.append(ExperimentConfig.m4_secondary(ch, seed=42,
                                                   run_mode="dry_run"))
    cells.append(ExperimentConfig(experiment="M5", method="task",
                                  channel="awgn", k=3, training_seed=42,
                                  run_mode="dry_run"))
    return cells


def test_gate_d():
    res = validate_matrix(_full_proposal())
    check("D/full-frozen-proposal-valid", res["valid"],
          f"errors={res['errors'][:3]}; warnings={len(res['warnings'])}")

    counts = expected_cells()
    check("D/expected-counts",
          counts["M1_training_runs"] == 2 * 3 * 2 * 3
          and counts["M2_training_runs"] == 3 * 5 * 2 * 3
          and counts["M3_infeasible_cells"] == 3
          and counts["M4_secondary_eval_conditions"] == 10,
          str(counts))

    def first_error(cells):
        r = validate_matrix(cells)
        return r["errors"][0] if r["errors"] else None

    digital_in_m1 = ExperimentConfig(experiment="M1", method="digital",
                                     channel="awgn", k=12, training_seed=42,
                                     bits_per_component=1, run_mode="dry_run")
    r_m1dig = validate_matrix([digital_in_m1])
    check("D/digital-in-M1-rejected",
          r_m1dig["errors"] and any("M2/M4 confusion" in e
                                    for e in r_m1dig["errors"]),
          f"{len(r_m1dig['errors'])} errors")

    wrong_b = ExperimentConfig(experiment="M2", method="digital",
                               channel="awgn", k=24, training_seed=42,
                               bits_per_component=2, run_mode="dry_run")
    r_b = validate_matrix([wrong_b])
    check("D/wrong-M2-budget-rejected",
          any("budget mismatch" in e for e in r_b["errors"]),
          f"expected B=3 for k=24; {len(r_b['errors'])} errors reported")

    wrong_snr = ExperimentConfig.m2("task", 12, "awgn", 42, run_mode="dry_run")
    wrong_snr.eval_snr_grid_db = (0.0, 2.0)     # M1 grid inside an M2 cell
    check("D/wrong-M2-SNR-grid-rejected",
          "SNR grid" in (first_error([wrong_snr]) or ""))

    wrong_k = ExperimentConfig.m2("task", 3, "awgn", 42, run_mode="dry_run")
    check("D/wrong-M2-k-rejected",
          "k must be in" in (first_error([wrong_k]) or ""),
          "k=3 is not a matched-budget point")

    # missing seeds: M1 cells present only for seed 42
    one_seed = [ExperimentConfig.m1("task", 1, "awgn", 42, run_mode="dry_run")]
    check("D/missing-seeds-rejected",
          any("must be exactly" in e for e in validate_matrix(one_seed)["errors"]))

    # M2/M4 confusion: a PRIMARY cell carrying the B=8 secondary budget
    conf = ExperimentConfig(experiment="M2", method="digital", channel="awgn",
                            k=24, training_seed=42, bits_per_component=8,
                            run_mode="dry_run")
    r_conf = validate_matrix([conf])
    check("D/M2-M4-confusion-rejected",
          any("M2/M4 confusion" in e for e in r_conf["errors"]),
          f"{len(r_conf['errors'])} errors reported")

    # M4 mislabeled/mis-specified
    m4_bad = ExperimentConfig(experiment="M4", method="digital",
                              channel="awgn", k=54, training_seed=42,
                              bits_per_component=4, run_mode="dry_run")
    check("D/M4-wrong-budget-rejected",
          any("SECONDARY digital reference" in e
              for e in validate_matrix([m4_bad])["errors"]))
    m4 = ExperimentConfig.m4_secondary("awgn", 42, run_mode="dry_run")
    r4 = validate_matrix([m4])
    check("D/M4-secondary-warning-present",
          r4["valid"] and any("SECONDARY" in w for w in r4["warnings"]),
          "valid but always labeled SECONDARY")

    # feasible M3 cell is an error (must remain infeasible)
    m3_feasible = ExperimentConfig(experiment="M3", method="digital",
                                   channel="awgn", k=54, training_seed=42,
                                   bits_per_component=8, run_mode="dry_run")
    check("D/M3-unexpectedly-feasible-rejected",
          any("unexpectedly feasible" in e
              for e in validate_matrix([m3_feasible])["errors"]))


# ---------------------------------------------------------------------------
# Gate E - digital budget / infeasibility
# ---------------------------------------------------------------------------

def test_gate_e():
    for k, b in sorted(config.DIGITAL_B_BY_K.items()):
        feasible, note = digital_feasible(k, b)
        pkt = config.digital_packet(k, b)
        check(f"E/k={k},B={b}",
              feasible and pkt["n_padding_symbols"] == 0,
              f"exact fit (audit 8.6): {note[:48]}")
    for k in (1, 2, 3):
        feasible, note = digital_feasible(k, 1)
        check(f"E/k={k}-infeasible",
              (not feasible) and "12" in note,
              f"k_min=12 > k={k}: {note[:40]}")
    feasible_none, _ = digital_feasible(12, None)
    check("E/B-none-infeasible", not feasible_none)

    # M3 configs are VALID (expected infeasible) and summarize as such
    m3 = ExperimentConfig.m3(2, bits=1, run_mode="dry_run")
    s = m3.dry_run_summary()
    check("E/M3-config-valid-expected-infeasible",
          m3.validate() == []
          and s["digital_feasibility"]["feasible"] is False
          and s["digital_feasibility"]["expected_infeasible"] is True)

    # harness reports the structured infeasible status (never fake metrics)
    cond = Condition(k=3, channel="awgn", snr_db=10.0, t_max=4, episodes=4,
                     bits_per_component=1, seed=1234)
    res = evaluate_condition(cond, methods=["digital"],
                             digital=DigitalBaseline(k=54, bits_per_component=8))
    check("E/harness-infeasible-status",
          res["digital"].get("status") == "infeasible"
          and "reason" in res["digital"],
          str(res["digital"].get("reason", ""))[:48])


# ---------------------------------------------------------------------------
# Gate F - dry run: configuration only, no side effects
# ---------------------------------------------------------------------------

def test_gate_f():
    with tempfile.TemporaryDirectory() as td:
        cfg = _smoke_recon_cfg(checkpoint_path=str(Path(td) / "dry.pth"))
        out = dry_run(cfg)
        s = out["summary"]
        required = ("experiment", "method", "k", "channel_uses", "rho",
                    "training_seed", "test_seed", "training_config",
                    "evaluation_config", "digital_feasibility",
                    "checkpoint_path", "run_mode")
        check("F/dry-run-structured-summary",
              out["status"] == "dry_run" and out["valid"]
              and all(k in s for k in required),
              "all section-16 summary fields present")
        check("F/dry-run-no-training-flag",
              s["training_will_occur"] is False)
        check("F/dry-run-snr-policy-explicit",
              s["training_config"]["train_snr_policy"] == {
                  "distribution": "uniform", "range_db": [0.0, 20.0],
                  "granularity": "per_batch"},
              "U(0,20) dB per batch - frozen protocol, inspectable")
        leftovers = list(Path(td).iterdir())
        check("F/dry-run-no-side-effects", leftovers == [],
              f"nothing written: {leftovers}")
        cfg_run = cfg.run(dry_run=True)
        check("F/cfg.run-dry-mode",
              cfg_run["status"] == "dry_run" and list(Path(td).iterdir()) == [],
              "run(dry_run=True) also trains nothing")


# ---------------------------------------------------------------------------
# Gate G - smoke reconstruction training (runner end-to-end)
# ---------------------------------------------------------------------------

def test_gate_g():
    with tempfile.TemporaryDirectory() as td:
        cfg = _smoke_recon_cfg(
            checkpoint_path=str(Path(td) / "recon.pth"))
        out = run_smoke(cfg)
        tb = out["train"]
        ok_train = (out["status"] == "smoke_complete"
                    and tb["loss_final"] < tb["loss_initial"]
                    and tb["nan_batches"] == 0
                    and len(tb["history"]) == cfg.train_steps
                    and all(math.isfinite(v) for v in tb["history"]))
        check("G/recon-train-finite-decreasing", ok_train,
              f"loss {tb['loss_initial']:.4f} -> {tb['loss_final']:.4f} "
              f"({tb['loss_reduction_pct']:.1f}%)")
        ok_eval = ("5" in out["eval"] and "10" in out["eval"]
                   and "success_rate" in out["eval"]["10"]
                   and all(math.isfinite(float(out["eval"][snr]["success_rate"]))
                           for snr in out["eval"]))
        check("G/recon-eval-harness-rows", ok_eval,
              f"SNR keys {sorted(out['eval'])}")
        check("G/non-scientific-label",
              out["non_scientific"] is True
              and "not an experimental result" in out["note"])
        ckpt = Path(out["checkpoint"])
        model_loaded = ReconstructionDeepJSCC(k=cfg.k)
        payload = load_checkpoint(ckpt, model_loaded, restore_rng=False)
        ok_meta = (payload["metadata"]["method"] == "reconstruction"
                   and payload["metadata"]["k"] == cfg.k
                   and payload["metadata"]["channel"] == cfg.channel
                   and payload["seed"] == cfg.training_seed
                   and payload["step"] == cfg.train_steps
                   and payload["history"] == tb["history"])
        check("G/checkpoint-roundtrip-metadata", ok_meta)
        fresh = ReconstructionDeepJSCC(k=cfg.k)
        check("G/checkpoint-restores-different-model",
              not models_identical(fresh, model_loaded),
              "loaded state != fresh init (real weights restored)")


# ---------------------------------------------------------------------------
# Gate H - smoke task-oriented training
# ---------------------------------------------------------------------------

def test_gate_h():
    with tempfile.TemporaryDirectory() as td:
        cfg = _smoke_task_cfg(checkpoint_path=str(Path(td) / "task.pth"))
        out = run_smoke(cfg)
        tb = out["train"]
        ok_train = (out["status"] == "smoke_complete"
                    and tb["nan_batches"] == 0
                    and all(math.isfinite(v) for v in
                            (tb["train_total_initial"], tb["train_total_final"]))
                    and all(math.isfinite(v) for v in
                            (tb["eval_before"]["total"],
                             tb["eval_after"]["total"])))
        check("H/task-train-finite", ok_train,
              f"total {tb['train_total_initial']:.3f} -> "
              f"{tb['train_total_final']:.3f}")
        ok_lambdas = ("L_d" in tb["history"][0] and "L_T" in tb["history"][0]
                      and tb["lambdas"] == {"ld": config.LAMBDA_DISTANCE,
                                            "lu": config.LAMBDA_CONTROL,
                                            "lv": config.LAMBDA_VELOCITY,
                                            "lT": config.LAMBDA_TERMINAL})
        check("H/frozen-lambdas-visible", ok_lambdas,
              f"{tb['lambdas']} (lu=lv=0 diagnostics only)")
        ok_eval = ("5" in out["eval"] and "success_rate" in out["eval"]["5"]
                   and math.isfinite(float(out["eval"]["5"]["success_rate"])))
        check("H/task-eval-harness-row", ok_eval)
        model_loaded = ReconstructionDeepJSCC(k=cfg.k)
        payload = load_checkpoint(Path(out["checkpoint"]), model_loaded,
                                  restore_rng=False)
        check("H/task-checkpoint-roundtrip",
              payload["metadata"]["method"] == "task"
              and payload["history"] == tb["history"])


# ---------------------------------------------------------------------------
# Gate I - smoke AWGN + Rayleigh paths
# ---------------------------------------------------------------------------

def test_gate_i():
    with tempfile.TemporaryDirectory() as td:
        for channel in ("awgn", "rayleigh"):
            cfg = _smoke_recon_cfg(
                channel=channel, seed=43, train_steps=20,
                checkpoint_path=str(Path(td) / f"recon_{channel}.pth"))
            out = run_smoke(cfg)
            tb = out["train"]
            ok = (out["status"] == "smoke_complete"
                  and tb["nan_batches"] == 0
                  and all(math.isfinite(v) for v in tb["history"])
                  and all(math.isfinite(float(out["eval"][snr]["final_distance_mean"]))
                          for snr in out["eval"]))
            check(f"I/{channel}", ok,
                  f"recon k=2: loss {tb['loss_initial']:.4f} -> "
                  f"{tb['loss_final']:.4f}, eval rows {sorted(out['eval'])}")
        # task path on Rayleigh (block fading + ZF inside the rollout)
        cfg = _smoke_task_cfg(channel="rayleigh", seed=44, train_steps=8,
                              t_max=8, eval_episodes=24, n_test_episodes=16,
                              checkpoint_path=str(Path(td) / "task_ray.pth"))
        out = run_smoke(cfg)
        check("I/rayleigh-task",
              out["status"] == "smoke_complete"
              and out["train"]["nan_batches"] == 0
              and math.isfinite(float(out["eval"]["5"]["success_rate"])))


# ---------------------------------------------------------------------------
# Gate J - evaluation-harness integration
# ---------------------------------------------------------------------------

def test_gate_j():
    torch.manual_seed(5)
    cfg = _smoke_recon_cfg(seed=44, eval_snr_grid_db=(10.0,),
                           n_test_episodes=12, t_max=10)
    cond = cfg.to_eval_condition(10.0)
    model = ReconstructionDeepJSCC(k=cfg.k)
    res = evaluate_condition(cond, methods=["oracle", "reconstruction"],
                             recon_model=model)
    keys = ("success_count", "success_rate", "final_distance_mean",
            "ttg_success_mean")
    ok = all(all(k in res[m]["aggregate"] for k in keys)
             for m in ("oracle", "reconstruction"))
    check("J/harness-schema-via-config", ok,
          "Condition built from ExperimentConfig; aggregates present")
    real1 = generate_realizations(cond, cond.episodes)
    real2 = generate_realizations(cond, cond.episodes)
    check("J/paired-realizations-identical",
          bool((real1["noise"] == real2["noise"]).all()),
          "pre-generated realizations deterministic per (seed, SNR, k)")


# ---------------------------------------------------------------------------
# Gate K - deterministic rerun
# ---------------------------------------------------------------------------

def _dicts_equal(a, b):
    """NaN-aware equality for aggregate rows (NaN == NaN by position)."""
    if a.keys() != b.keys():
        return False
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, dict) and isinstance(vb, dict):
            if not _dicts_equal(va, vb):
                return False
        elif (isinstance(va, float) and isinstance(vb, float)
              and math.isnan(va) and math.isnan(vb)):
            continue
        elif va != vb:
            return False
    return True


def test_gate_k():
    with tempfile.TemporaryDirectory() as td:
        outs, models = [], []
        for i in (1, 2):
            cfg = _smoke_recon_cfg(
                checkpoint_path=str(Path(td) / f"rerun{i}.pth"))
            outs.append(run_smoke(cfg))
            m = ReconstructionDeepJSCC(k=2)
            load_checkpoint(Path(outs[-1]["checkpoint"]), m, restore_rng=False)
            models.append(m)
        same_hist = outs[0]["train"]["history"] == outs[1]["train"]["history"]
        check("K/loss-history-bitwise", same_hist,
              f"last losses {outs[0]['train']['loss_final']:.6f} vs "
              f"{outs[1]['train']['loss_final']:.6f}")
        check("K/model-bitwise", models_identical(models[0], models[1]))
        same_eval = all(
            _dicts_equal(outs[0]["eval"][snr], outs[1]["eval"][snr])
            for snr in outs[0]["eval"])
        check("K/eval-outputs-identical", same_eval,
              "aggregates equal across reruns (paired realizations)")
        cond = _smoke_recon_cfg().to_eval_condition(5.0)
        r1 = generate_realizations(cond, 25)
        r2 = generate_realizations(cond, 25)
        check("K/eval-realizations-bitwise",
              bool((r1["noise"] == r2["noise"]).all()))
        # task rerun determinism (shorter budget)
        t_outs = []
        for i in (1, 2):
            cfg = _smoke_task_cfg(
                checkpoint_path=str(Path(td) / f"task_rerun{i}.pth"))
            t_outs.append(run_smoke(cfg))
        same_task = (t_outs[0]["train"]["history"]
                     == t_outs[1]["train"]["history"]
                     and all(_dicts_equal(t_outs[0]["eval"][snr],
                                          t_outs[1]["eval"][snr])
                             for snr in t_outs[0]["eval"]))
        check("K/task-rerun-identical", same_task)


# ---------------------------------------------------------------------------
# Gate L - in-process core regression (Steps 0-7)
# ---------------------------------------------------------------------------

def test_gate_l():
    import tests.test_channels as tc
    tc.SUMMARY.clear()
    tc.test_conversion(); tc.test_power_normalization(); tc.test_awgn_variance()
    tc.test_awgn_snr(); tc.test_rayleigh_stats(); tc.test_block_fading()
    tc.test_equalization(); tc.test_repro_and_conventions()
    check("REG/Step-3", all(tc.SUMMARY), f"{sum(tc.SUMMARY)}/{len(tc.SUMMARY)}")

    import tests.test_step02 as ts
    fns = [ts.test_config, ts.test_normalization_roundtrip,
           ts.test_normalization_endpoints_and_clip, ts.test_dynamics_closed_form,
           ts.test_dynamics_autograd, ts.test_velocity_clip,
           ts.test_controller_formula, ts.test_controller_saturation_and_gradient,
           ts.test_env_success_semantics, ts.test_env_exit_freeze_and_determinism,
           ts.test_env_sampling_and_nans]
    ok = True
    for fn in fns:
        try:
            fn()
        except AssertionError:
            ok = False
    check("REG/Step 0-2", ok, "11/11")

    import tests.test_deepjscc as td
    td.SUMMARY.clear()
    td.test_channel_use_count(); td.test_decoder_input_order_parity()
    td.test_shapes(); td.test_power_through_model(); td.test_gradient_flow()
    check("REG/Step-4 structural", all(td.SUMMARY),
          f"{sum(td.SUMMARY)}/{len(td.SUMMARY)}")

    import tests.test_task_oriented as tt
    tt.SUMMARY.clear()
    tt.test_a_loss_algebra(); tt.test_b_rollout_dims(); tt.test_c_zero_error()
    tt.test_d_gradient_flow(); tt.test_e_controller_dynamics()
    check("REG/Step-5 core", all(tt.SUMMARY), f"{sum(tt.SUMMARY)}/{len(tt.SUMMARY)}")

    import tests.test_digital as tg
    tg.SUMMARY.clear()
    tg.test_gate_a(); tg.test_gate_c()
    check("REG/Step-6 core", all(tg.SUMMARY), f"{sum(tg.SUMMARY)}/{len(tg.SUMMARY)}")


if __name__ == "__main__":
    print("=" * 72)
    print("STEP 8 EXPERIMENTS TESTS (Gates C-L)")
    print("=" * 72)
    test_gate_c()
    test_gate_d()
    test_gate_e()
    test_gate_f()
    test_gate_g()
    test_gate_h()
    test_gate_i()
    test_gate_j()
    test_gate_k()
    test_gate_l()
    n_pass, n_all = sum(SUMMARY), len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)
