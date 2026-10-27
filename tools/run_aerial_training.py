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

A-13 variant (training episodes with gain offset > +10 dB resampled; the
evaluation dispersion is never truncated; artifacts go to
``results/AERIAL_TRAIN_TRUNC/``):

    python tools/run_aerial_training.py preflight --variant=a13
    python tools/run_aerial_training.py train --variant=a13 --shard=i/N
    python tools/run_aerial_training.py evaluate --variant=a13
    python tools/run_aerial_training.py finalize --variant=a13
"""

import json
import math
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

# A-12 = the pre-registered run (full dispersion during training).
# A-13 = identical design; training episodes with gain offset > +10 dB are
#        resampled (PROTOCOL_AMENDMENTS.md A-13, approved 2026-10-27).
#        Evaluation is NEVER truncated in either variant.
VARIANTS = {
    "a12": {"root": "AERIAL_TRAIN", "label": "AerialTrain",
            "experiment": "AERIAL_TRAIN",
            "amendment": "AERIAL-TRAIN-3GPP-2026-10-26",
            "max_offset_db": None, "suffix": ""},
    "a13": {"root": "AERIAL_TRAIN_TRUNC", "label": "AerialTrainTrunc",
            "experiment": "AERIAL_TRAIN_TRUNC",
            "amendment": "AERIAL-TRAIN-TRUNC-2026-10-27",
            "max_offset_db": 20.0, "suffix": "_trunc20",
            # pre-registered ladder (PROTOCOL_AMENDMENTS.md A-13): primary 20,
            # one retry at 15, then stop.  No other cap is allowed.
            "allowed_caps": (20.0, 15.0)},
}
VARIANT = "a12"
AERIAL_ROOT = ROOT / "results" / "AERIAL_TRAIN"
AMENDMENT = VARIANTS["a12"]["amendment"]
EXPERIMENT = VARIANTS["a12"]["experiment"]
TRUNCATE_DB = VARIANTS["a12"]["max_offset_db"]
SUFFIX = VARIANTS["a12"]["suffix"]
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

# A-12 identity (must never change: committed artifacts live in these dirs)
A12_K18_DIRNAME = ("AERIAL_lam_ctl_heavy_task_umaaerial_k18_seed42"
                   "__9c5d26c729af57a3")

# A-13 pre-flight gate (pre-registered in the amendment, before the first cell)
PREFLIGHT_STEPS = 300
PREFLIGHT_MAX_LOSS_AFTER = (50, 1e3)       # (step index, loss ceiling)
PREFLIGHT_LAST_MEAN_STEPS = 20
PREFLIGHT_LAST_MEAN_MAX = 50.0


def _configure(variant, cap=None):
    global VARIANT, AERIAL_ROOT, AMENDMENT, EXPERIMENT, TRUNCATE_DB, SUFFIX
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; "
                         f"choose from {sorted(VARIANTS)}")
    v = VARIANTS[variant]
    VARIANT = variant
    AERIAL_ROOT = ROOT / "results" / v["root"]
    AMENDMENT = v["amendment"]
    EXPERIMENT = v["experiment"]
    TRUNCATE_DB = v["max_offset_db"]
    SUFFIX = v["suffix"]
    if cap is not None:
        allowed = v.get("allowed_caps")
        if allowed is None:
            raise SystemExit(f"variant {variant!r} takes no --cap")
        if float(cap) not in allowed:
            raise SystemExit(
                f"cap {cap} dB is not pre-registered for {variant} "
                f"(allowed: {list(allowed)})")
        TRUNCATE_DB = float(cap)
        SUFFIX = f"_trunc{TRUNCATE_DB:g}"
    rw.WAVE_A_ROOT = AERIAL_ROOT
    rw.WAVE_LABEL = v["label"]
    rw.WAVE_EXPERIMENT = v["experiment"]
    rw.WAVE_AMENDMENT = AMENDMENT


_configure("a12")


class AerialCell(rw.WaveCell):
    """Wave-A cell trained on the A-12/A-13 3GPP dispersion channel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, hidden=None, **kwargs)
        self.wave_id = self.wave_id.replace("WAVE_A_", "AERIAL_") + SUFFIX
        self.aerial_max_offset_db = TRUNCATE_DB

    def spec(self):
        spec = super().spec()
        spec["experiment"] = EXPERIMENT
        spec["amendment"] = AMENDMENT
        spec["aerial_channel"] = {
            "model": "3GPP TR 36.777 / TR 38.901 UMa-aerial (dispersion)",
            "geometry": dict(GEOMETRY),
            "reference_pl_db": "mean path loss of the geometry (centred)",
            "shadow_sigma_db": dict(aerial.SHADOW_SIGMA_DB),
            "time_correlation": "per-episode block gain (no Doppler)",
        }
        if TRUNCATE_DB is not None:
            spec["aerial_train_max_offset_db"] = TRUNCATE_DB
            spec["aerial_train_tail"] = (
                "training episodes with gain offset above the cap are "
                "resampled; evaluation always keeps the full dispersion")
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
    print(f"== 3GPP aerial-training pre-flight ({VARIANT}"
          + (f", cap {TRUNCATE_DB:g} dB" if TRUNCATE_DB is not None else "")
          + ") ==")
    bad = 0

    def gate(name, ok, detail=""):
        nonlocal bad
        print(f"{'PASS' if ok else 'FAIL'} [{name}]" + (f" {detail}" if detail else ""))
        if not ok:
            bad += 1

    gate("isolated root",
         AERIAL_ROOT == ROOT / "results" / VARIANTS[VARIANT]["root"],
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

    # A-12 identity must never move; A-13 must be a distinct cell
    if VARIANT == "a12":
        gate("A-12 identity unchanged",
             cells()[0].run_dirname() == A12_K18_DIRNAME,
             cells()[0].run_dirname())
    else:
        gate("A-13 identity distinct from A-12",
             cells()[0].run_dirname() != A12_K18_DIRNAME,
             cells()[0].run_dirname())

    if TRUNCATE_DB is None:
        gate("training tail policy", True, "full dispersion (A-12 design)")
    else:
        capped, _ = aerial.sample_aerial_offsets_db(
            4096, generator=torch.Generator().manual_seed(3),
            max_offset_db=TRUNCATE_DB)
        raw, _ = aerial.sample_aerial_offsets_db(
            4096, generator=torch.Generator().manual_seed(3))
        eval_off, _ = _gain_vector()
        gate("training tail capped", float(capped.max()) <= TRUNCATE_DB,
             f"max {float(capped.max()):+.2f} dB <= {TRUNCATE_DB:+.1f} dB")
        gate("sampler keeps its tail by default",
             float(raw.max()) > TRUNCATE_DB, f"max {float(raw.max()):+.2f} dB")
        gate("evaluation keeps the full dispersion",
             float(eval_off.max()) > TRUNCATE_DB,
             f"eval max offset {float(eval_off.max()):+.2f} dB")

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


def cmd_preflight():
    """A-13 pre-registered gate: 300 steps, k=18, no markers written."""
    c = cells()[0]
    c.steps = PREFLIGHT_STEPS
    import math as _math
    import time
    t0 = time.perf_counter()
    _m, rep, _gf, summary = rw._train_one(c)
    wall = time.perf_counter() - t0
    hist = rep["history"]
    tot = [h["total"] for h in hist]
    nonfinite = sum(1 for h in hist
                    if not (_math.isfinite(h["grad_enc"])
                            and _math.isfinite(h["grad_dec"])))
    after = tot[PREFLIGHT_MAX_LOSS_AFTER[0]:]
    last_mean = sum(tot[-PREFLIGHT_LAST_MEAN_STEPS:]) / min(
        PREFLIGHT_LAST_MEAN_STEPS, len(tot))
    gates = {
        "no non-finite gradients": nonfinite == 0,
        f"no loss > {PREFLIGHT_MAX_LOSS_AFTER[1]:g} after step "
        f"{PREFLIGHT_MAX_LOSS_AFTER[0]}": max(after) <= PREFLIGHT_MAX_LOSS_AFTER[1],
        f"mean of last {PREFLIGHT_LAST_MEAN_STEPS} <= "
        f"{PREFLIGHT_LAST_MEAN_MAX:g}": last_mean <= PREFLIGHT_LAST_MEAN_MAX,
    }
    print(f"== pre-flight ({VARIANT}): {c.wave_id} {PREFLIGHT_STEPS} steps "
          f"in {wall:.0f}s ({wall / PREFLIGHT_STEPS:.2f} s/step) ==")
    for name, ok in gates.items():
        print(f"{'PASS' if ok else 'FAIL'} [{name}]")
    print(f"   loss: min {min(tot):.2f} (step {tot.index(min(tot)) + 1}), "
          f"max after step {PREFLIGHT_MAX_LOSS_AFTER[0]} {max(after):.1f}, "
          f"last-{PREFLIGHT_LAST_MEAN_STEPS} mean {last_mean:.2f}, "
          f"non-finite grad steps {nonfinite}, nan losses {rep['nan_batches']}")
    print(f"   eval_after (full dispersion, diagnostic) "
          f"{summary.get('eval_after_success_rate'):.4f}")
    ok = all(gates.values())
    print("== pre-flight PASS: full run may start ==" if ok else
          "== pre-flight FAIL: do NOT spend the full run ==")
    print("   pre-flight wrote NOTHING (no markers, no checkpoints)")
    return 0 if ok else 1


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
                if rec.get("experiment") == EXPERIMENT:
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
                "experiment": EXPERIMENT, "amendment": AMENDMENT,
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


def _train_diagnostics(c, rec):
    """Mechanical collapse check on a completed training history.

    ``diverged`` is true if any step produced a non-finite gradient, or if the
    final loss is more than 5x the best loss the run ever reached.  It is a
    description of the saved report, not a judgement about the channel.
    """
    hist = rec.get("history") or []
    if not hist:
        return {"wave_id": c.wave_id, "history": 0, "diverged": None}
    tot = [h["total"] for h in hist]
    nonfinite = sum(
        1 for h in hist
        if not (math.isfinite(h.get("grad_enc", float("nan")))
                and math.isfinite(h.get("grad_dec", float("nan")))))
    lo = min(tot)
    step = tot.index(lo) + 1
    return {
        "wave_id": c.wave_id,
        "history": len(hist),
        "loss_min": round(lo, 3),
        "loss_min_step": step,
        "loss_final": round(tot[-1], 3),
        "frac_loss_gt_1e3": round(sum(1 for t in tot if t > 1e3) / len(tot), 4),
        "nonfinite_grad_steps": nonfinite,
        "diverged": bool(nonfinite) or tot[-1] > 5.0 * lo,
    }


def cmd_finalize():
    AERIAL_ROOT.mkdir(parents=True, exist_ok=True)
    p_los = aerial.los_probability(GEOMETRY["d2d_m"], GEOMETRY["h_m"])
    pl_mean = aerial.expected_path_loss_db(GEOMETRY["d2d_m"], GEOMETRY["h_m"],
                                           GEOMETRY["fc_ghz"])

    train_rows = []
    diagnostics = []
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
        diagnostics.append(_train_diagnostics(c, rec))
    diverged = [d["wave_id"] for d in diagnostics if d.get("diverged")]

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
    best_aerial = max((r["success_rate"] for name, rows in curves.items()
                       if name.startswith("aerial_train") for r in rows),
                      default=None)
    best_frozen = max((r["success_rate"] for name, rows in curves.items()
                       if name.startswith("frozen_rayleigh") for r in rows),
                      default=None)
    if done != EXPECTED_CELLS or n_slots != 3 * EXPECTED_SLOTS:
        outcome = "incomplete - every declared slot must exist before reading"
    elif diverged:
        outcome = ("negative - training diverged in "
                   + ", ".join(diverged)
                   + " (see 'train_diagnostics'); no cell was retrained")
    elif best_aerial is not None and best_frozen is not None \
            and best_aerial < 0.5 * best_frozen:
        outcome = ("negative - the trained arm stays far below the "
                   "Rayleigh-trained comparator")
    else:
        outcome = ("informative - the trained arm is competitive with the "
                   "Rayleigh-trained comparator on the paired dispersion")
    out.write_text(json.dumps({
        "experiment": EXPERIMENT,
        "variant": VARIANT,
        "amendment": AMENDMENT,
        "status": ("complete" if done == EXPECTED_CELLS
                   and n_slots == 3 * EXPECTED_SLOTS else "incomplete"),
        "status_note": ("'complete' = every declared cell was trained and every "
                        "evaluation slot written; it is NOT a statement that "
                        "the run succeeded. See 'outcome'."),
        "outcome": outcome,
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
        "train_diagnostics": diagnostics,
        "train_diverged": diverged,
        "best_success_rate": {"trained": best_aerial,
                              "frozen_rayleigh": best_frozen},
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
    # --variant=a12|a13 is global (selects root, amendment and tail policy);
    # it is stripped before the command sees its own flags.
    variant = next((a.split("=", 1)[1] for a in argv
                    if a.startswith("--variant=")), None)
    cap = next((a.split("=", 1)[1] for a in argv
                if a.startswith("--cap=")), None)
    if variant is not None or cap is not None:
        _configure(variant or VARIANT, cap=cap)
        argv = [a for a in argv
                if not (a.startswith("--variant=") or a.startswith("--cap="))]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "verify":
        return cmd_verify()
    if cmd == "smoke":
        return cmd_smoke()
    if cmd == "preflight":
        return cmd_preflight()
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
