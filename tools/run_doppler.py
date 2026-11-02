"""A-19: Doppler / time-correlated Rayleigh fading, evaluation only.

Registered in fyp_md_files/PROTOCOL_AMENDMENTS.md (A-19) on 2026-11-02,
before any cell ran.  Nothing frozen is edited: this tool only *evaluates*
stored Wave-A (k = 12) and Wave-C (k = 54) lambda-heavy checkpoints plus the
frozen digital baseline on a correlated fading process, and writes under
results/DOPPLER/ only.

The sweep axis is the displacement per control period in wavelengths,
    delta = v * DT / lambda,
because v and f_c enter the Clarke process only through delta.

Usage
    python tools/run_doppler.py verify
    python tools/run_doppler.py evaluate [--limit N]
    python tools/run_doppler.py summary
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project import config                                   # noqa: E402
from project.baselines.digital import DigitalBaseline        # noqa: E402
from project.channels import doppler as dop                  # noqa: E402
from project.evaluation.evaluate import (                    # noqa: E402
    Condition, aggregate, evaluate_condition, generate_realizations,
    run_method,
)
from project.models.reconstruction_jscc import ReconstructionDeepJSCC  # noqa: E402
from project.training.common import save_checkpoint               # noqa: E402
from project.training.train_task_oriented import (                 # noqa: E402
    train_task_oriented,
)

DOPPLER_ROOT = ROOT / "results" / "DOPPLER"
EVAL_EPISODES = 1000
SNR_GRID = (0.0, 5.0, 10.0, 15.0, 20.0)
K_LIST = (12, 54)
AMENDMENT = "A-19"
CARRIER_DEFAULT_GHZ = 2.0
FADE_THRESHOLD_DB = -10.0

# --- A-19 declared follow-up arm (only if the decision rule triggers) --------
# The rule fired on the *digital* baseline at the most correlated point
# (delta = 0.033, k = 54: -0.028), so one training arm is required at that
# point.  It re-trains the lambda-heavy arm at k = 12 so the cost is bounded
# and the comparison is against the same arm as Wave-A.
TRAIN_POINT = "d0033_2g"
TRAIN_K = 12
TRAIN_SEED = 42
TRAIN_TAG = "a19_doppler"
TRAIN_LAMBDAS = (1.0, 0.10, 0.10, 2.0)      # Wave-A lam_ctl_heavy

# (id, carrier GHz, speed m/s).  ``None`` speed is the frozen i.i.d. control.
POINTS = (
    ("frozen_iid", 2.0, None),
    ("d0033_2g", 2.0, 0.05),
    ("d0067_2g", 2.0, 0.10),
    ("d0133_2g", 2.0, 0.20),
    ("d0333_2g", 2.0, 0.50),
    ("d0667_2g", 2.0, 1.00),
    ("d1333_2g", 2.0, 2.00),
    ("d3333_2g", 2.0, 5.00),
    ("d0117_07g", 0.7, 0.50),
    ("d0234_07g", 0.7, 1.00),
)

CHECKPOINT_GLOBS = {
    12: "results/WAVE_A/WAVE_A_lam_ctl_heavy_task_rayleigh_k12_seed42__*/final_ckpt.pt",
    54: "results/WAVE_C/WAVE_C_lam_ctl_heavy_task_rayleigh_k54_seed42__*/final_ckpt.pt",
}


# ---------------------------------------------------------------------------
# grid helpers
# ---------------------------------------------------------------------------

def point_spec(point_id):
    for pid, carrier, speed in POINTS:
        if pid == point_id:
            return {"id": pid, "carrier_ghz": carrier, "speed_mps": speed}
    raise KeyError(point_id)


def point_doppler_hz(carrier_ghz, speed_mps):
    if speed_mps is None:
        return None
    return dop.doppler_hz(speed_mps, carrier_ghz * 1e9)


def point_label(spec):
    if spec["speed_mps"] is None:
        return "frozen i.i.d."
    delta = dop.displacement_per_step(spec["speed_mps"],
                                      spec["carrier_ghz"] * 1e9)
    return (f"{spec['speed_mps']:g} m/s @ {spec['carrier_ghz']:g} GHz "
            f"(delta={delta:.3f} lambda)")


def eval_path(point_id, k, snr):
    return DOPPLER_ROOT / "eval" / point_id / f"k{k}_snr{snr:g}.json"


def summary_path():
    return DOPPLER_ROOT / "step_doppler_summary.json"


def checkpoint_path(k):
    hits = sorted(ROOT.glob(CHECKPOINT_GLOBS[k]))
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous checkpoint for k={k}: {hits}")
    return hits[0]


def train_dir():
    return (DOPPLER_ROOT / "train"
            / f"{TRAIN_TAG}_k{TRAIN_K}_rayleigh_seed{TRAIN_SEED}")


def trained_eval_path(snr):
    return DOPPLER_ROOT / "eval_trained" / TRAIN_POINT / f"k{TRAIN_K}_snr{snr:g}.json"


def load_model(k):
    path = checkpoint_path(k)
    model = ReconstructionDeepJSCC(k=k)
    payload = torch.load(path, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def make_condition(point_id, k, snr):
    spec = point_spec(point_id)
    fd = point_doppler_hz(spec["carrier_ghz"], spec["speed_mps"])
    return Condition(
        k=k, channel="rayleigh", snr_db=snr, t_max=config.T_MAX,
        bits_per_component=config.DIGITAL_B_BY_K[k],
        episodes=EVAL_EPISODES, seed=config.TEST_SEED,
        doppler_hz=fd, carrier_ghz=spec["carrier_ghz"],
    )


def fade_stats_for(point_id):
    spec = point_spec(point_id)
    fd = point_doppler_hz(spec["carrier_ghz"], spec["speed_mps"])
    gen = config.derive_generator("doppler_test", config.TEST_SEED)
    if fd is None:
        h = torch.complex(
            torch.randn(EVAL_EPISODES, config.T_MAX, generator=gen) * 0.5 ** 0.5,
            torch.randn(EVAL_EPISODES, config.T_MAX, generator=gen) * 0.5 ** 0.5,
        )
    else:
        h = dop.jakes_fading(EVAL_EPISODES, config.T_MAX, fd,
                             dt=config.DT, generator=gen)
    stats = dop.fade_statistics(h, FADE_THRESHOLD_DB)
    stats.update({"point": point_id, "doppler_hz": fd,
                  "carrier_ghz": spec["carrier_ghz"],
                  "speed_mps": spec["speed_mps"]})
    if fd is not None:
        stats["displacement_per_step"] = dop.displacement_per_step(
            spec["speed_mps"], spec["carrier_ghz"] * 1e9)
    else:
        stats["displacement_per_step"] = None
    return stats


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify():
    print("== A-19 pre-flight gates ==", flush=True)
    bad = 0

    def gate(name, ok, detail=""):
        nonlocal bad
        print(f"{'PASS' if ok else 'FAIL'}  [{name}]"
              + (f"  ({detail})" if detail else ""), flush=True)
        if not ok:
            bad += 1

    amendment = ROOT / "fyp_md_files" / "PROTOCOL_AMENDMENTS.md"
    text = amendment.read_text(encoding="utf-8")
    gate("amendment A-19 registered", "## A-19" in text)

    gate("grid size", len(POINTS) == 10, f"{len(POINTS)} points")
    gate("k grid", tuple(K_LIST) == (12, 54), str(K_LIST))
    gate("snr grid", tuple(SNR_GRID) == (0.0, 5.0, 10.0, 15.0, 20.0))

    for k in K_LIST:
        ck = checkpoint_path(k)
        gate(f"lambda-heavy checkpoint k={k}", ck is not None,
             str(ck.relative_to(ROOT)) if ck else "missing")

    # physics: E|h|^2 = 1 and lag-1 autocorrelation near J0(2*pi*fd*DT)
    g = torch.Generator().manual_seed(100)
    h = dop.jakes_fading(4000, 30, 10.0, dt=0.1, generator=g)
    power = (h.abs() ** 2).mean().item()
    gate("generator power ~ 1", abs(power - 1.0) < 0.03, f"{power:.4f}")
    r1 = (h[:, :-1] * h[:, 1:].conj()).mean().real.item() / power
    j0 = _j0(2.0 * math.pi * 10.0 * 0.1)
    gate("lag-1 autocorrelation ~ J0", abs(r1 - j0) < 0.06,
         f"empirical {r1:+.3f} vs J0 {j0:+.3f}")

    # the frozen i.i.d. path must be untouched by the new Condition fields
    from project.evaluation.evaluate import generate_realizations
    c = Condition(k=12, channel="rayleigh", snr_db=10.0, t_max=20,
                  episodes=32, seed=config.TEST_SEED)
    real = generate_realizations(c, 32)
    gen = config.derive_generator("noise_test", c.seed)
    sigma2 = config.TX_POWER * 10.0 ** (-10.0 / 10.0)
    shape = (32, 20, 12)
    noise = torch.complex(
        torch.randn(shape, generator=gen) * (sigma2 / 2.0) ** 0.5,
        torch.randn(shape, generator=gen) * (sigma2 / 2.0) ** 0.5)
    hh = torch.complex(
        torch.randn(32, 20, generator=gen) * 0.5 ** 0.5,
        torch.randn(32, 20, generator=gen) * 0.5 ** 0.5)
    gate("frozen noise stream unchanged", bool(torch.equal(real["noise"], noise)))
    gate("frozen fading stream unchanged", bool(torch.equal(real["h"], hh)))

    # every point is physically distinct in delta
    deltas = []
    for pid, carrier, speed in POINTS:
        if speed is None:
            continue
        deltas.append(round(dop.displacement_per_step(speed, carrier * 1e9), 4))
    gate("distinct displacement values", len(set(deltas)) == len(deltas),
         str(sorted(deltas)))

    # digital framing is feasible at both k
    for k in K_LIST:
        try:
            pkt = config.digital_packet(k, config.DIGITAL_B_BY_K[k])
            gate(f"digital packet feasible k={k}", pkt["k_min"] <= k,
                 f"B={config.DIGITAL_B_BY_K[k]}, needs {pkt['k_min']}")
        except config.InfeasibleDigitalConfig as exc:
            gate(f"digital packet feasible k={k}", False, str(exc))

    gate("output root isolated",
         str(DOPPLER_ROOT).startswith(str(ROOT / "results")),
         str(DOPPLER_ROOT.relative_to(ROOT)))

    # training plumbing for the declared follow-up arm
    from project.training.train_task_oriented import (       # noqa: E402
        closed_loop_rollout,
    )
    torch.manual_seed(0)
    probe = ReconstructionDeepJSCC(k=12)
    probe.eval()
    s0 = torch.rand(2, 6)
    span = torch.tensor(config.STATE_MAX) - torch.tensor(config.STATE_MIN)
    s0 = s0 * span + torch.tensor(config.STATE_MIN)
    base_a = closed_loop_rollout(
        probe, s0, t_max=2, snr_db=10.0, channel="rayleigh",
        generator=config.derive_generator("noise_test", 5))[0]
    base_b = closed_loop_rollout(
        probe, s0, t_max=2, snr_db=10.0, channel="rayleigh",
        generator=config.derive_generator("noise_test", 5))[0]
    gate("rollout deterministic without fading",
         bool(torch.equal(base_a, base_b)))
    fd_probe = point_doppler_hz(2.0, 0.05)
    probe_fading = dop.jakes_fading(
        2, 2, fd_probe, generator=config.derive_generator("doppler_test", 5))
    with_fading = closed_loop_rollout(
        probe, s0, t_max=2, snr_db=10.0, channel="rayleigh",
        generator=config.derive_generator("noise_test", 5),
        fading=probe_fading)[0]
    gate("supplied fading changes the rollout",
         not bool(torch.equal(base_a, with_fading)))

    print("== gates OK ==" if bad == 0 else f"== {bad} gate(s) FAILED ==",
          flush=True)
    return 0 if bad == 0 else 1


def _j0(x):
    if abs(x) < 12.0:
        term, total, k = 1.0, 1.0, 0
        while True:
            k += 1
            term *= -(x * x) / (4.0 * k * k)
            total += term
            if abs(term) < 1e-16:
                return total
    return math.sqrt(2.0 / (math.pi * abs(x))) * math.cos(abs(x) - math.pi / 4.0)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(limit=None):
    print("== A-19 Doppler sweep (evaluation only) ==", flush=True)
    n_new, total_wall = 0, 0.0
    models = {}

    for point_id, carrier, speed in POINTS:
        spec = point_spec(point_id)
        print(f"-- {point_id}: {point_label(spec)}", flush=True)
        for k in K_LIST:
            if k not in models:
                models[k] = load_model(k)
            for snr in SNR_GRID:
                if limit is not None and n_new >= limit:
                    print(f"-- limit {limit} reached", flush=True)
                    return 0
                out = eval_path(point_id, k, snr)
                if out.exists():
                    continue
                cond = make_condition(point_id, k, snr)
                digital = DigitalBaseline(k=k,
                                          bits_per_component=cond.bits_per_component)
                t0 = time.perf_counter()
                with torch.no_grad():
                    res = evaluate_condition(
                        cond, methods=["task", "digital"],
                        task_model=models[k], digital=digital)
                wall = time.perf_counter() - t0
                total_wall += wall
                rec = {
                    "experiment": "DOPPLER",
                    "amendment": AMENDMENT,
                    "point": point_id,
                    "carrier_ghz": carrier,
                    "speed_mps": speed,
                    "doppler_hz": cond.doppler_hz,
                    "displacement_per_step": (
                        None if speed is None else
                        dop.displacement_per_step(speed, carrier * 1e9)),
                    "k": k,
                    "channel": "rayleigh",
                    "snr_db": snr,
                    "bits_per_component": cond.bits_per_component,
                    "episodes": EVAL_EPISODES,
                    "test_seed": config.TEST_SEED,
                    "t_max": config.T_MAX,
                    "non_frozen": True,
                    "protocol_note": ("A-19 time-correlated fading; "
                                      "evaluation of stored checkpoints"),
                    "aggregate_task": res["task"]["aggregate"],
                    "aggregate_digital": res["digital"]["aggregate"],
                    "wall_s": round(wall, 1),
                }
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rec, indent=2, default=str))
                n_new += 1
                print(f"   k={k:>2} snr={snr:5.1f} "
                      f"lam={res['task']['aggregate']['success_rate']:.3f} "
                      f"dig={res['digital']['aggregate']['success_rate']:.3f} "
                      f"({wall:.1f}s)", flush=True)

    print(f"-- finished: {n_new} new slots, wall={total_wall:.1f}s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# declared follow-up arm: train at the most correlated point
# ---------------------------------------------------------------------------

def cmd_train():
    """One training cell at the worst measured Doppler point (A-19 rule)."""
    d = train_dir()
    done = d / "DONE.json"
    if done.exists():
        print(f"-- already trained: {d.relative_to(ROOT)}", flush=True)
        return 0

    spec = point_spec(TRAIN_POINT)
    fd = point_doppler_hz(spec["carrier_ghz"], spec["speed_mps"])
    print(f"== A-19 follow-up arm: k={TRAIN_K} Rayleigh, "
          f"{point_label(spec)}, fd={fd:.4f} Hz, {config.TASK_TRAIN_STEPS} "
          f"steps ==", flush=True)
    d.mkdir(parents=True, exist_ok=True)
    (d / "RUNNING.json").write_text(json.dumps({
        "cell": TRAIN_TAG, "point": TRAIN_POINT, "doppler_hz": fd,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))

    t0 = time.perf_counter()
    model, rep = train_task_oriented(
        k=TRAIN_K, channel="rayleigh", seed=TRAIN_SEED,
        steps=config.TASK_TRAIN_STEPS,
        batch_episodes=config.BATCH_EPISODES,
        t_max=config.TASK_TRAINING_T,
        snr_range=config.SNR_TRAIN_RANGE_DB,
        lambdas=TRAIN_LAMBDAS,
        doppler_hz=fd,
        verbose_every=0,
    )
    wall = time.perf_counter() - t0

    save_checkpoint(
        d / "final_ckpt.pt", model, optimizer=None,
        step=config.TASK_TRAIN_STEPS, seed=TRAIN_SEED,
        metadata={"cell": TRAIN_TAG, "amendment": AMENDMENT,
                  "channel": "rayleigh", "k": TRAIN_K,
                  "training_seed": TRAIN_SEED,
                  "train_steps": config.TASK_TRAIN_STEPS,
                  "lambdas": list(TRAIN_LAMBDAS),
                  "doppler_hz": fd, "point": TRAIN_POINT,
                  "non_frozen": True})

    g0, g1 = rep["grad_enc_final"], rep["grad_dec_final"]
    grads_finite = (math.isfinite(g0) and math.isfinite(g1)
                    and g0 > 0 and g1 > 0)
    report = {
        "cell": TRAIN_TAG, "amendment": AMENDMENT,
        "point": TRAIN_POINT, "doppler_hz": fd,
        "carrier_ghz": spec["carrier_ghz"], "speed_mps": spec["speed_mps"],
        "k": TRAIN_K, "channel": "rayleigh", "seed": TRAIN_SEED,
        "steps": config.TASK_TRAIN_STEPS,
        "lambdas": {"ld": TRAIN_LAMBDAS[0], "lu": TRAIN_LAMBDAS[1],
                    "lv": TRAIN_LAMBDAS[2], "lT": TRAIN_LAMBDAS[3]},
        "nan_batches": rep["nan_batches"], "grads_finite": grads_finite,
        "task_loss_initial": rep["train_total_initial"],
        "task_loss_final": rep["train_total_final"],
        "eval_after_success_rate": rep["eval_after"]["success_rate"],
        "wall_s": round(wall, 1), "checkpoint": str(d / "final_ckpt.pt"),
    }
    (d / "train_report.json").write_text(json.dumps(report, indent=2,
                                                    default=str))
    if rep["nan_batches"] == 0 and grads_finite:
        done.write_text(json.dumps({"cell": TRAIN_TAG, "finished": True,
                                    "wall_s": report["wall_s"]}))
        print(f"-- DONE in {wall / 60:.1f} min, loss "
              f"{report['task_loss_initial']:.1f} -> "
              f"{report['task_loss_final']:.1f}", flush=True)
        return 0
    (d / "FAILED.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"-- FAILED: nan_batches={rep['nan_batches']} "
          f"grads_finite={grads_finite}", flush=True)
    return 1


def load_trained_model():
    path = train_dir() / "final_ckpt.pt"
    if not path.exists():
        raise FileNotFoundError(f"no trained arm at {path}")
    model = ReconstructionDeepJSCC(k=TRAIN_K)
    payload = torch.load(path, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def cmd_evaluate_trained():
    print(f"== A-19 follow-up arm evaluation ({TRAIN_POINT}, "
          f"k={TRAIN_K}) ==", flush=True)
    model = load_trained_model()
    n_new = 0
    for snr in SNR_GRID:
        out = trained_eval_path(snr)
        if out.exists():
            continue
        cond = make_condition(TRAIN_POINT, TRAIN_K, snr)
        real = generate_realizations(cond, EVAL_EPISODES)
        t0 = time.perf_counter()
        with torch.no_grad():
            metrics = run_method(
                cond, real,
                estimate_fn=lambda s, n, hh: model.forward(
                    s, snr, channel="rayleigh", denormalize_output=True,
                    noise=n, h=hh))
        agg = aggregate(metrics)
        wall = time.perf_counter() - t0
        rec = {
            "experiment": "DOPPLER",
            "amendment": AMENDMENT,
            "arm": "a19_trained_correlated",
            "point": TRAIN_POINT,
            "doppler_hz": cond.doppler_hz,
            "k": TRAIN_K,
            "channel": "rayleigh",
            "snr_db": snr,
            "episodes": EVAL_EPISODES,
            "test_seed": config.TEST_SEED,
            "non_frozen": True,
            "protocol_note": ("A-19 follow-up: lambda-heavy retrained under "
                              "correlated fading at the worst measured point"),
            "aggregate_task": agg,
            "wall_s": round(wall, 1),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2, default=str))
        n_new += 1
        print(f"   snr={snr:5.1f} SR={agg['success_rate']:.3f} ({wall:.1f}s)",
              flush=True)
    print(f"-- {n_new} new slots", flush=True)
    return 0


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def cmd_summary():
    rows = []
    missing = 0
    for point_id, carrier, speed in POINTS:
        for k in K_LIST:
            for snr in SNR_GRID:
                p = eval_path(point_id, k, snr)
                if not p.exists():
                    missing += 1
                    continue
                d = json.loads(p.read_text(encoding="utf-8"))
                rows.append({
                    "point": point_id,
                    "carrier_ghz": carrier,
                    "speed_mps": speed,
                    "doppler_hz": d["doppler_hz"],
                    "displacement_per_step": d["displacement_per_step"],
                    "k": k,
                    "snr_db": snr,
                    "lam": d["aggregate_task"]["success_rate"],
                    "digital": d["aggregate_digital"]["success_rate"],
                })

    stats = {pid: fade_stats_for(pid) for pid, _, _ in POINTS}

    per_point = {}
    for row in rows:
        per_point.setdefault(row["point"], {}).setdefault(
            row["k"], {}).setdefault(row["snr_db"], {})
        per_point[row["point"]][row["k"]][row["snr_db"]] = row

    mean = {}
    for pid in per_point:
        mean[pid] = {}
        for k in per_point[pid]:
            vals_l = [v["lam"] for v in per_point[pid][k].values()]
            vals_d = [v["digital"] for v in per_point[pid][k].values()]
            mean[pid][k] = {"lam_mean": sum(vals_l) / len(vals_l),
                            "digital_mean": sum(vals_d) / len(vals_d)}

    control = mean.get("frozen_iid", {})
    trained = {}
    for snr in SNR_GRID:
        p = trained_eval_path(snr)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            trained[str(snr)] = d["aggregate_task"]["success_rate"]
    trained_mean = (sum(trained.values()) / len(trained)) if trained else None
    if trained_mean is not None:
        base = mean.get(TRAIN_POINT, {}).get(TRAIN_K, {}).get("lam_mean")
        trained = {"per_snr": trained, "mean": trained_mean,
                   "stored_arm_mean": base,
                   "delta_vs_stored_arm": (None if base is None
                                           else trained_mean - base)}

    deltas = {}
    for pid in mean:
        if pid == "frozen_iid":
            continue
        deltas[pid] = {}
        for k in mean[pid]:
            if k in control:
                deltas[pid][k] = {
                    "lam_delta_vs_iid": (mean[pid][k]["lam_mean"]
                                         - control[k]["lam_mean"]),
                    "digital_delta_vs_iid": (mean[pid][k]["digital_mean"]
                                             - control[k]["digital_mean"]),
                }

    summary = {
        "experiment": "DOPPLER",
        "amendment": AMENDMENT,
        "registered": "2026-11-02",
        "episodes": EVAL_EPISODES,
        "snr_grid": list(SNR_GRID),
        "k_list": list(K_LIST),
        "fade_threshold_db": FADE_THRESHOLD_DB,
        "expected_slots": len(POINTS) * len(K_LIST) * len(SNR_GRID),
        "completed_slots": len(rows),
        "missing_slots": missing,
        "status": "complete" if missing == 0 else "incomplete",
        "fade_statistics": stats,
        "mean_over_snr": mean,
        "delta_vs_iid": deltas,
        "trained_arm": trained or None,
        "decision_rule": ("|delta| > 0.02 at displacement <= 0.13 lambda "
                          "means the i.i.d. model is not adequate"),
        "rows": rows,
    }
    summary_path().parent.mkdir(parents=True, exist_ok=True)
    summary_path().write_text(json.dumps(summary, indent=2, default=str),
                              encoding="utf-8")

    print(f"slots: {len(rows)}/{summary['expected_slots']} "
          f"(missing {missing}) -> {summary['status']}", flush=True)
    print(f"\n{'point':<12} {'delta':>7} {'fade%':>6} {'run':>5} "
          f"{'k':>3} {'lam':>6} {'dig':>6} {'dlam':>7}", flush=True)
    for pid, carrier, speed in POINTS:
        s = stats[pid]
        dl = s["displacement_per_step"]
        for k in K_LIST:
            if pid not in mean or k not in mean[pid]:
                continue
            m = mean[pid][k]
            dd = deltas.get(pid, {}).get(k, {}).get("lam_delta_vs_iid")
            print(f"{pid:<12} {('-' if dl is None else f'{dl:7.3f}')} "
                  f"{100 * s['deep_fade_fraction']:6.1f} "
                  f"{s['mean_fade_run_steps']:5.2f} {k:>3} "
                  f"{m['lam_mean']:6.3f} {m['digital_mean']:6.3f} "
                  f"{('-' if dd is None else f'{dd:+7.3f}')}", flush=True)
    print(f"\nwrote {summary_path().relative_to(ROOT)}", flush=True)
    if trained_mean is not None:
        print(f"follow-up arm ({TRAIN_POINT}, k={TRAIN_K}): "
              f"trained mean {trained_mean:.3f} vs stored "
              f"{trained['stored_arm_mean']:.3f} "
              f"({trained['delta_vs_stored_arm']:+.3f})", flush=True)
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "verify":
        return cmd_verify()
    if cmd == "evaluate":
        limit = None
        for a in argv[1:]:
            if a.startswith("--limit="):
                limit = int(a.split("=", 1)[1])
        return cmd_evaluate(limit)
    if cmd == "train":
        return cmd_train()
    if cmd == "evaluate-trained":
        return cmd_evaluate_trained()
    if cmd == "summary":
        return cmd_summary()
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
