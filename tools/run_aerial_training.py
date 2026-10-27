"""3GPP UMa-aerial TRAINING run (A-12): does training UNDER the channel help?

The A-10 study evaluated the frozen (Rayleigh-trained) checkpoints on a
3GPP TR 36.777 / TR 38.901 UMa-aerial link and found a ~99 dB transmit-SNR
penalty.  A reviewer is entitled to object that the model was never trained
on that channel.  This runner trains the lambda-shaped arm *under* the 3GPP
dispersion and measures what changes.

Channel (A-12, ``channel="umaaerial"`` in the model/training path):

* one LoS/NLoS draw per EPISODE with the 3GPP P_LoS of the geometry,
* one log-normal shadow term per episode (TR 38.901 UMa: 4 dB LoS, 6 dB NLoS),
* the per-episode gain is constant over the episode (block fading),
* the gain is CENTRED on the geometry's mean path loss, so the training SNR
  axis keeps the frozen meaning (0-20 dB nominal, U(0,20) per batch) and the
  experiment isolates the dispersion, not an arbitrary link budget,
* perfect-CSI ZF on the true ``h*g``: an episode at ``-offset`` dB is exactly
  the frozen Rayleigh channel at ``SNR - offset`` (verified in
  ``tests/test_aerial_training.py``).

Random seeds, step budget, architecture (128), lambda weights, environment,
controller, evaluation episodes/seed, and SNR grid are the frozen ones.  The
ONLY change is the channel seen during training.

Design (pre-registered in PROTOCOL_AMENDMENTS.md A-12 before the first cell):

  arms      : lam_ctl_heavy (the arm that works; the frozen task arm is at
              ~0.02 success and cannot discriminate a channel effect)
  k         : 18, 54
  channel   : umaaerial (3GPP dispersion)
  seeds     : 42
  width     : 128 (frozen)
  steps     : config.TASK_TRAIN_STEPS (3000)
  cells     : 2
  eval      : E=1000, 8 nominal-SNR points (-15..20 dB), identical paired
              episodes/noise for every method and a single per-episode gain
              vector shared by all methods

Comparators (all evaluated on the SAME paired dispersion):
  * frozen_rayleigh : the Wave-C lambda-heavy checkpoint, trained WITHOUT
                      dispersion (this is the A-10 baseline);
  * aerial_train    : this run;
  * digital         : frozen K=7 convolutional baseline, B = M2_B_BY_K[k].

Evaluation note: under perfect-CSI ZF with gain ``g``, running the model at
``snr0`` with the pre-generated noise divided by ``g`` is EXACTLY the
UMa-aerial realization at ``snr0`` (block gain) - no new evaluation code path
and no interpolation.  Transfer: tx SNR = nominal + PL_mean.

Isolation: writes ONLY under ``results/AERIAL_TRAIN/``; M1/M2 Wave-A/B/C
artifacts are read (checkpoints/eval) and never modified.

Usage:
    python tools/run_aerial_training.py verify
    python tools/run_aerial_training.py smoke
    python tools/run_aerial_training.py train [--shard=i/N]
    python tools/run_aerial_training.py evaluate
    python tools/run_aerial_training.py finalize
    python tools/run_aerial_training.py status
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import tools.run_wave_a as rw  # noqa: E402
from project import config  # noqa: E402
from project.channels import aerial  # noqa: E402
from project.evaluation.evaluate import (Condition, aggregate,  # noqa: E402
                                         generate_realizations, run_method)
from project.evaluation.statistics import wilson_ci_95  # noqa: E402
from project.experiments.config import M2_B_BY_K  # noqa: E402

AERIAL_ROOT = ROOT / "results" / "AERIAL_TRAIN"
AMENDMENT = "AERIAL-TRAIN-3GPP-2026-10-26"
CHANNEL = "umaaerial"
KS = (18, 54)
SEEDS = (42,)
TAG = "lam_ctl_heavy"
# nominal (mean-path-loss-removed) SNR grid; the useful range shifts down
# because the LoS/NLoS split spreads the effective SNR by ~20 dB
GRID_NOMINAL = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
EVAL_EPISODES = rw.EVAL_EPISODES           # 1000, the wave protocol
TARGETS = (0.90, 0.99)
GEOMETRY = dict(aerial.TRAINING_GEOMETRY)

rw.WAVE_A_ROOT = AERIAL_ROOT
rw.WAVE_LABEL = "AerialTrain"
rw.WAVE_EXPERIMENT = "AERIAL_TRAIN"
rw.WAVE_AMENDMENT = AMENDMENT


class AerialCell(rw.WaveCell):
    """Wave-A cell trained on the A-12 3GPP dispersion channel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, hidden=None, **kwargs)
        self.wave_id = self.wave_id.replace("WAVE_A_", "AERIAL_")

    def spec(self):
        spec = super().spec()
        spec["experiment"] = "AERIAL_TRAIN"
        spec["amendment"] = AMENDMENT
        spec["aerial_channel"] = {
            "model": "3GPP TR 36.777 / TR 38.901 UMa-aerial (dispersion)",
            "geometry": dict(GEOMETRY),
            "reference_pl_db": "mean path loss of the geometry (centred)",
            "shadow_sigma_db": dict(aerial.SHADOW_SIGMA_DB),
            "time_correlation": "per-episode block gain (no Doppler)",
        }
        spec["eval_snr_grid_db"] = list(GRID_NOMINAL)
        spec["eval_episodes"] = EVAL_EPISODES
        spec["eval_convention"] = ("nominal SNR (mean PL removed); the "
                                   "per-episode gain vector is shared by all "
                                   "methods")
        return spec


def cells():
    out = []
    for k in KS:
        for seed in SEEDS:
            out.append(AerialCell("A5", "task", CHANNEL, k, seed, tag=TAG,
                                  lambdas=rw.LAMBDA_TAGS[TAG]))
    return out


def units():
    return [{"unit": f"k{c.k}_aerial_seed{c.seed}", "k": c.k,
             "channel": CHANNEL, "seed": c.seed} for c in cells()]


EXPECTED_CELLS = len(cells())
EXPECTED_UNITS = len(units())
EXPECTED_SLOTS = EXPECTED_UNITS * len(GRID_NOMINAL)


def _eval_path(method, k, snr):
    return AERIAL_ROOT / "eval" / f"{method}_k{k}" / f"eval_{snr:g}.json"


# ---------------------------------------------------------------------------
# pre-flight
# ---------------------------------------------------------------------------

def cmd_verify():
    print("== 3GPP aerial-training pre-flight (A-12) ==")
    bad = 0

    def gate(name, ok, detail=""):
        nonlocal bad
        print(f"{'PASS' if ok else 'FAIL'} [{name}]" + (f" {detail}" if detail else ""))
        if not ok:
            bad += 1

    gate("isolated root", AERIAL_ROOT.name == "AERIAL_TRAIN",
         str(AERIAL_ROOT.relative_to(ROOT)))
    gate("design", (EXPECTED_CELLS, EXPECTED_UNITS, EXPECTED_SLOTS) == (2, 2, 16),
         f"{EXPECTED_CELLS} cells / {EXPECTED_UNITS} units / {EXPECTED_SLOTS} slots")
    gate("channel", {c.channel for c in cells()} == {CHANNEL}, CHANNEL)
    gate("frozen width", {c.hidden for c in cells()} == {None},
         f"frozen {list(config.HIDDEN_DIMS)[0]}")
    gate("frozen step budget", {c.steps for c in cells()} == {config.TASK_TRAIN_STEPS},
         str(config.TASK_TRAIN_STEPS))
    gate("frozen lambdas", {c.lambdas for c in cells()} == {rw.LAMBDA_TAGS[TAG]},
         str(rw.LAMBDA_TAGS[TAG]))
    gate("frozen seeds", {c.seed for c in cells()} == set(SEEDS), str(SEEDS))
    gate("geometry pinned", GEOMETRY == {"d2d_m": 500.0, "h_m": 50.0,
                                         "fc_ghz": 2.0}, str(GEOMETRY))
    gate("shadow sigma per TR 38.901",
         aerial.SHADOW_SIGMA_DB == {"los": 4.0, "nlos": 6.0})
    gate("p_los reference", abs(aerial.los_probability(500.0, 50.0)
                                - 0.888748512) < 1e-8)
    off, _ = aerial.sample_aerial_offsets_db(
        4096, generator=torch.Generator().manual_seed(1))
    gate("centred channel (dB)", abs(float(off.mean())) < 0.3,
         f"mean offset {float(off.mean()):+.3f} dB")

    # the training path must actually accept the new channel string
    ok_channel = True
    try:
        import inspect

        from project.training.train_task_oriented import closed_loop_rollout
        src = inspect.getsource(closed_loop_rollout)
        ok_channel = CHANNEL in src
    except Exception as e:                     # pragma: no cover
        print(f"WARN {type(e).__name__}: {e}")
        ok_channel = False
    gate("training rollout accepts the channel", ok_channel)

    print(f"INFO [wave-a universe untouched] {len(rw.wave_cells())} ids")
    print("== gates OK ==" if bad == 0 else f"== {bad} gate(s) FAILED ==")
    return 0 if bad == 0 else 1


def cmd_smoke():
    c = cells()[0]
    c.steps = 20
    import time
    t0 = time.perf_counter()
    _m, rep, grads_finite, summary = rw._train_one(c)
    wall = time.perf_counter() - t0
    print(f"== smoke: {c.wave_id} 20 steps in {wall:.1f}s "
          f"({wall / 20:.2f} s/step) nan={rep['nan_batches']} "
          f"grads_finite={grads_finite} loss={summary.get('task_loss_final'):.3f}")
    print("   smoke wrote NOTHING (no markers, no checkpoints)")
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def _select(args):
    out = cells()
    shard = next((a.split("=", 1)[1] for a in args if a.startswith("--shard=")),
                 None)
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        out = [c for j, c in enumerate(out) if j % n == i]
    return out


def cmd_train(args=()):
    sel = _select(args)
    if len(sel) != len(cells()):
        print(f"== selection: {len(sel)} of {len(cells())} cells ==", flush=True)
        for c in sel:
            print(f"   {c.wave_id}", flush=True)
    rw.wave_cells = lambda: sel
    rw.DOCUMENTED_FAILURES = frozenset()
    return rw.cmd_train(None)


# ---------------------------------------------------------------------------
# evaluate: paired dispersion Monte Carlo
# ---------------------------------------------------------------------------

def _frozen_models():
    """Wave-C lambda-heavy Rayleigh checkpoints (the A-10 comparator)."""
    import tools.run_wave_c as wc

    saved = rw.WAVE_A_ROOT
    rw.WAVE_A_ROOT = wc.WAVE_C_ROOT           # wc cells live under Wave-C
    try:
        out = {}
        for c in wc.cells():
            if (c.tag == TAG and c.channel == "rayleigh" and c.seed in SEEDS
                    and c.k in KS):
                model = c.load_ckpt()
                if model is not None:
                    out[c.k] = model
        return out
    finally:
        rw.WAVE_A_ROOT = saved


def _trained_models():
    out = {}
    for c in cells():
        m = c.load_ckpt()
        if m is not None:
            out[c.k] = m
    return out


def _gain_vector():
    gen = config.derive_generator("aerial_eval", config.TEST_SEED)
    off_db, is_los = aerial.sample_aerial_offsets_db(EVAL_EPISODES,
                                                     generator=gen)
    return off_db, is_los


def _wilson(count, total):
    return wilson_ci_95([1] * int(count) + [0] * int(total - count))


def _curve(method, k, gains):
    off_db, is_los = gains
    gain = off_gain(off_db).unsqueeze(-1)         # [E, 1, 1] broadcast
    rows = []
    with torch.no_grad():
        for snr in GRID_NOMINAL:
            path = _eval_path(method, k, snr)
            if path.exists():
                rec = json.loads(path.read_text())
                if rec.get("experiment") == "AERIAL_TRAIN":
                    rows.append(rec)
                    print(f"   {method:16s} k={k:2d} nominal={snr:+5.1f} dB "
                          f"SR={rec['success_rate']:.4f} (cached)", flush=True)
                    continue
            cond = Condition(k=k, channel="rayleigh", snr_db=float(snr),
                             t_max=config.T_MAX, episodes=EVAL_EPISODES,
                             seed=config.TEST_SEED)
            real = generate_realizations(cond, EVAL_EPISODES)
            # exact UMa-aerial realization: dividing the pre-generated noise
            # by the per-episode gain is the block-gain channel with
            # perfect-CSI ZF on h*g (no approximation, no new code path)
            real["noise"] = real["noise"] / gain
            m = run_method(cond, real, estimate_fn=_estimator(method, k, snr))
            agg = aggregate(m)
            succ = m["success"].float()
            rec = {
                "experiment": "AERIAL_TRAIN", "amendment": AMENDMENT,
                "method": method, "k": k, "nominal_snr_db": float(snr),
                "episodes": EVAL_EPISODES, "test_seed": config.TEST_SEED,
                "success_rate": agg["success_rate"],
                "final_distance_mean": agg["final_distance_mean"],
                "control_effort_mean": agg["control_effort_mean"],
                "success_rate_los": float((succ * is_los.squeeze(-1)).sum()
                                          / max(int(is_los.sum()), 1)),
                "success_rate_nlos": float((succ * (~is_los).squeeze(-1)).sum()
                                           / max(int((~is_los).sum()), 1)),
                "wilson": _wilson(agg["success_count"], EVAL_EPISODES),
                "gain_mean_db": float(off_db.mean()),
                "gain_std_db": float(off_db.std()),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec, indent=2, default=str))
            rows.append(rec)
            print(f"   {method:16s} k={k:2d} nominal={snr:+5.1f} dB "
                  f"SR={rec['success_rate']:.4f}", flush=True)
    return rows


def off_gain(off_db):
    return 10.0 ** (-off_db / 20.0)


_EST = {}


def _estimator(method, k, snr):
    """Closure over (method, k, snr); the model is registered in ``_EST``."""
    return lambda s, noise_c, h_c: _EST[method][k].forward(
        s, snr, channel="rayleigh", noise=noise_c, h=h_c)


def cmd_evaluate():
    frozen = _frozen_models()
    rw.WAVE_A_ROOT = AERIAL_ROOT                 # restore our own root
    trained = _trained_models()
    gains = _gain_vector()

    _EST["frozen_rayleigh"] = frozen
    _EST["aerial_train"] = trained
    _EST["digital"] = {k: _digital_for(k) for k in KS}

    print(f"checkpoints: frozen={sorted(frozen)} trained={sorted(trained)}")
    print(f"gain vector: mean={float(gains[0].mean()):+.3f} dB "
          f"std={float(gains[0].std()):.3f} dB "
          f"P_LoS_emp={float(gains[1].float().mean()):.4f}", flush=True)
    for k in KS:
        for method in ("frozen_rayleigh", "aerial_train", "digital"):
            if k not in _EST[method]:
                print(f"NOTE: no {method} checkpoint for k={k}; skipped",
                      flush=True)
                continue
            print(f"-- {method} k={k} --", flush=True)
            _curve(method, k, gains)
    return 0


def _digital_for(k):
    from project.baselines.digital import DigitalBaseline
    return DigitalBaseline(k=k, bits_per_component=M2_B_BY_K[k])


# ---------------------------------------------------------------------------
# finalize / status
# ---------------------------------------------------------------------------

def _required(rows, target):
    xs = [r["nominal_snr_db"] for r in rows]
    ys = [r["success_rate"] for r in rows]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    if max(ys) < target:
        return None
    if ys[0] >= target:
        return float(xs[0])
    for i in range(1, len(xs)):
        if ys[i] >= target:
            y0, y1 = ys[i - 1], ys[i]
            if y1 == y0:
                return float(xs[i])
            f = (target - y0) / (y1 - y0)
            return float(xs[i - 1] + f * (xs[i] - xs[i - 1]))
    return None


def _state(c):
    d = AERIAL_ROOT / c.run_dirname()
    if not d.is_dir():
        return "missing"
    for kind, name in (("done", "DONE.json"), ("failed", "FAILED.json"),
                       ("running", "RUNNING.json")):
        if (d / name).exists():
            return kind
    return "missing"


def cmd_status():
    states = {}
    for c in cells():
        st = _state(c)
        states[st] = states.get(st, 0) + 1
    n_slots = sum(1 for k in KS
                  for m in ("frozen_rayleigh", "aerial_train", "digital")
                  for snr in GRID_NOMINAL
                  if _eval_path(m, k, snr).exists())
    print(f"== {AERIAL_ROOT.relative_to(ROOT)} ==")
    print(f"-- train: {states} of {EXPECTED_CELLS}")
    print(f"-- eval: {n_slots}/{3 * EXPECTED_SLOTS} method-unit-SNR slots")
    return 0


def cmd_finalize():
    AERIAL_ROOT.mkdir(parents=True, exist_ok=True)
    p_los = aerial.los_probability(GEOMETRY["d2d_m"], GEOMETRY["h_m"])
    pl_mean = aerial.expected_path_loss_db(GEOMETRY["d2d_m"], GEOMETRY["h_m"],
                                           GEOMETRY["fc_ghz"])

    train_rows = []
    for c in cells():
        p = c.dir() / "train_report.json"
        rec = json.loads(p.read_text()) if p.exists() else {"exit_status": "missing"}
        train_rows.append({"wave_id": c.wave_id, "tag": c.tag, "k": c.k,
                           "channel": c.channel, "seed": c.seed,
                           "exit_status": rec.get("exit_status", "missing"),
                           "task_loss_final": rec.get("task_loss_final"),
                           "nan_batches": rec.get("nan_batches"),
                           "grads_finite": rec.get("grads_finite"),
                           "wall_seconds": rec.get("wall_seconds"),
                           "eval_after_success_rate":
                               rec.get("eval_after_success_rate")})

    curves = {}
    required = {}
    for k in KS:
        for method in ("frozen_rayleigh", "aerial_train", "digital"):
            rows = []
            for snr in GRID_NOMINAL:
                path = _eval_path(method, k, snr)
                if path.exists():
                    rows.append(json.loads(path.read_text()))
            if not rows:
                continue
            rows.sort(key=lambda r: r["nominal_snr_db"])
            curves[f"{method}_k{k}"] = rows
            required[f"{method}_k{k}"] = {
                f"{t:.2f}": (None if (v := _required(rows, t)) is None
                             else round(v, 2)) for t in TARGETS}

    links = {}
    for name, req in required.items():
        links[name] = {t: (None if v is None else round(v + pl_mean, 2))
                       for t, v in req.items()}

    gaps = {}
    for k in KS:
        a = required.get(f"aerial_train_k{k}", {})
        f = required.get(f"frozen_rayleigh_k{k}", {})
        d = required.get(f"digital_k{k}", {})
        gaps[f"k{k}"] = {
            "aerial_train_minus_frozen_db":
                {t: (None if a.get(t) is None or f.get(t) is None
                     else round(a[t] - f[t], 2)) for t in a},
            "aerial_train_minus_digital_db":
                {t: (None if a.get(t) is None or d.get(t) is None
                     else round(a[t] - d[t], 2)) for t in a},
        }

    done = sum(1 for r in train_rows if r["exit_status"] == "complete")
    n_slots = sum(len(v) for v in curves.values())
    out = AERIAL_ROOT / "step_aerial_training_summary.json"
    out.write_text(json.dumps({
        "experiment": "AERIAL_TRAIN",
        "amendment": AMENDMENT,
        "status": ("complete" if done == EXPECTED_CELLS
                   and n_slots == 3 * EXPECTED_SLOTS else "incomplete"),
        "question": ("does training UNDER the 3GPP UMa-aerial dispersion "
                     "change the lambda arm's operating point?"),
        "design": {"tag": TAG, "lambdas": list(rw.LAMBDA_TAGS[TAG]),
                   "k": list(KS), "seeds": list(SEEDS), "channel": CHANNEL,
                   "hidden": list(config.HIDDEN_DIMS)[0],
                   "steps": config.TASK_TRAIN_STEPS,
                   "eval_grid_nominal_db": list(GRID_NOMINAL),
                   "eval_episodes": EVAL_EPISODES,
                   "test_seed": config.TEST_SEED},
        "channel": {"geometry": GEOMETRY, "p_los": p_los,
                    "pl_mean_db": pl_mean,
                    "shadow_sigma_db": dict(aerial.SHADOW_SIGMA_DB),
                    "formulas": aerial.CHANNEL_INFO["umaaerial"]},
        "train_rows": train_rows,
        "curves": curves,
        "required_nominal_db": required,
        "required_tx_db": links,
        "gaps": gaps,
        "protocol_note": ("nominal SNR = mean path loss removed; transmit "
                          "SNR = nominal + PL_mean. E=1000 per point, one "
                          "shared per-episode gain vector (paired)."),
        "train_done": done,
        "eval_slots": n_slots,
        "eval_slots_expected": 3 * EXPECTED_SLOTS,
    }, indent=2, default=str))
    print(f"wrote {out.relative_to(ROOT)} ({done}/{EXPECTED_CELLS} train, "
          f"{n_slots}/{3 * EXPECTED_SLOTS} eval slots)")
    for name in sorted(required):
        r = required[name]
        print(f"  {name:24s} 0.90: {r['0.90']} 0.99: {r['0.99']}")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "verify":
        return cmd_verify()
    if cmd == "smoke":
        return cmd_smoke()
    if cmd == "train":
        return cmd_train([a for a in rest if a.startswith("--")])
    if cmd == "evaluate":
        return cmd_evaluate()
    if cmd == "finalize":
        return cmd_finalize()
    if cmd == "status":
        return cmd_status()
    print(f"unknown command {cmd!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
