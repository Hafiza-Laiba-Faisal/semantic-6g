"""Seed variance of the lambda arm at k = 24 (A-14).

Every k = 24 number in the thesis is currently a single seed, and k = 24 is
where the task-oriented and classical links cross over.  This runner asks the
narrow question a reviewer is entitled to ask: *is that one number an artefact
of one initialisation?*

Design (pre-registered in PROTOCOL_AMENDMENTS.md A-14, before the first cell):

  arm      : lam_ctl_heavy  (ld=1.0, lu=0.1, lv=0.1, lT=2.0) - Wave-C tag
  k        : 24 (the crossover condition)
  seeds    : 45, 46, 47  (the Wave-A A3 extra-seed set, reused)
  channels : awgn, rayleigh
  cells    : 6
  steps    : config.TASK_TRAIN_STEPS (3000, frozen budget)
  width    : frozen 128
  eval     : E=1000, 5-SNR grid -> 6 units / 30 slots

Why a separate runner: the Wave-A/B/C universes are already declared
(18 units / 90 slots, 350 cells, 32 cells).  Adding cells to any of them
would silently change a published denominator.  This runner has its own root,
`results/K24_SEEDS/`, and never writes anywhere else.

Statistics are reported per seed and as mean + range across the three seeds.
The range is NOT a confidence interval: n = 3.  Nothing is pooled across seeds
to manufacture a tighter interval, and no seed may be dropped.

Usage:
    python tools/run_k24_seeds.py verify
    python tools/run_k24_seeds.py smoke
    python tools/run_k24_seeds.py train [--shard=i/N]
    python tools/run_k24_seeds.py evaluate
    python tools/run_k24_seeds.py finalize
    python tools/run_k24_seeds.py status
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

K24_ROOT = ROOT / "results" / "K24_SEEDS"
AMENDMENT = "K24-SEEDS-2026-10-27"

# Captured before the Wave-A root is redirected (collision gate needs them).
WAVE_A_DIR = Path(rw.WAVE_A_ROOT)
WAVE_C_DIR = ROOT / "results" / "WAVE_C"

K = 24
SEEDS = (45, 46, 47)
CHANNELS = ("awgn", "rayleigh")
TAG = "lam_ctl_heavy"
LAMBDAS = rw.LAMBDA_TAGS[TAG]
SNR_GRID = rw.MATCH_SNR_DB

rw.WAVE_A_ROOT = K24_ROOT
rw.WAVE_LABEL = "K24"
rw.WAVE_EXPERIMENT = "K24_SEEDS"
rw.WAVE_AMENDMENT = AMENDMENT


class K24Cell(rw.WaveCell):
    """Wave-A cell with a K24 identity; frozen width and lambda arm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wave_id = self.wave_id.replace("WAVE_A_", "K24_")

    def spec(self):
        spec = super().spec()
        spec["experiment"] = "K24_SEEDS"
        spec["amendment"] = AMENDMENT
        spec["statistics_note"] = (
            "3 seeds at one k; report per-seed values and mean/range, not a "
            "confidence interval")
        return spec


def cells():
    return [K24Cell("A2", "task", ch, K, seed, tag=TAG, lambdas=LAMBDAS)
            for ch in CHANNELS for seed in SEEDS]


def units():
    seen = {}
    for c in cells():
        key = (c.k, c.channel, c.seed)
        seen.setdefault(key, {"unit": f"k{c.k}_{c.channel}_seed{c.seed}",
                              "k": c.k, "channel": c.channel,
                              "seed": c.seed})
    return [seen[key] for key in sorted(seen)]


EXPECTED_CELLS = len(cells())
EXPECTED_UNITS = len(units())
EXPECTED_SLOTS = EXPECTED_UNITS * len(SNR_GRID)


# ---------------------------------------------------------------------------
# verify / smoke
# ---------------------------------------------------------------------------

def cmd_verify():
    print("== k=24 seed-variance pre-flight (A-14) ==")
    bad = 0

    def gate(name, ok, detail=""):
        nonlocal bad
        print(f"{'PASS' if ok else 'FAIL'} [{name}]"
              + (f" {detail}" if detail else ""))
        if not ok:
            bad += 1

    cs = cells()
    gate("isolated root", K24_ROOT.name == "K24_SEEDS",
         str(K24_ROOT.relative_to(ROOT)))
    gate("design", (EXPECTED_CELLS, EXPECTED_UNITS, EXPECTED_SLOTS)
         == (6, 6, 30),
         f"{EXPECTED_CELLS} cells / {EXPECTED_UNITS} units / "
         f"{EXPECTED_SLOTS} slots")
    gate("k", {c.k for c in cs} == {K}, str(K))
    gate("seeds", {c.seed for c in cs} == set(SEEDS), str(SEEDS))
    gate("channels", {c.channel for c in cs} == set(CHANNELS), str(CHANNELS))
    gate("frozen lambdas", {c.lambdas for c in cs} == {LAMBDAS}, str(LAMBDAS))
    gate("frozen width", {c.hidden for c in cs} == {None},
         f"frozen {config.HIDDEN_DIMS[0]}")
    gate("frozen step budget", {c.steps for c in cs}
         == {config.TASK_TRAIN_STEPS}, str(config.TASK_TRAIN_STEPS))
    gate("eval grid", list(SNR_GRID) == [0.0, 5.0, 10.0, 15.0, 20.0],
         str(SNR_GRID))
    rel = {c.run_dirname() for c in cs}
    gate("distinct cell identities", len(rel) == EXPECTED_CELLS,
         f"{len(rel)} unique")
    clash = [r for c in cs for r in (WAVE_A_DIR / c.run_dirname(),
                                     WAVE_C_DIR / c.run_dirname())
             if r.exists()]
    gate("no collision with WAVE_A/WAVE_C dirs", not clash,
         f"{len(clash)} collisions")
    other = ROOT / "results" / "M2"
    gate("no collision with frozen M2 dirs",
         not any((other / c.run_dirname()).exists() for c in cs))
    print("== gates OK ==" if not bad else f"== {bad} gate(s) FAILED ==")
    return 0 if not bad else 1


def cmd_smoke():
    """20 training steps on one cell; writes NOTHING (no markers/checkpoints)."""
    c = cells()[0]
    c.steps = 20
    model, rep, grads_finite, summary = rw._train_one(c)
    n = len(rep["history"])
    finite = all(math.isfinite(h["total"]) for h in rep["history"])
    print(f"== smoke: {c.wave_id} {n} steps ==")
    print(f"   loss {rep['history'][0]['total']:.1f} -> "
          f"{rep['history'][-1]['total']:.1f}, all finite={finite}, "
          f"grads_finite={grads_finite}, nan_batches={rep['nan_batches']}")
    print("   smoke wrote NOTHING (no markers, no checkpoints)")
    return 0 if (n == 20 and finite and grads_finite and not rep["nan_batches"]) else 1


# ---------------------------------------------------------------------------
# train / evaluate
# ---------------------------------------------------------------------------

def _select(args):
    out = cells()
    shard = next((a.split("=", 1)[1] for a in args
                  if a.startswith("--shard=")), None)
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


def cmd_evaluate(args=()):
    rw.wave_cells = lambda: cells()
    return rw.cmd_evaluate(None)


# ---------------------------------------------------------------------------
# status / finalize
# ---------------------------------------------------------------------------

def _state(c):
    d = K24_ROOT / c.run_dirname()
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
    n_slots = sum(1 for u in units() for s in SNR_GRID
                  if rw._eval_path(u, s).exists())
    print(f"== {K24_ROOT.relative_to(ROOT)} ==")
    print(f"-- train: {states} of {EXPECTED_CELLS}")
    print(f"-- eval: {n_slots}/{EXPECTED_SLOTS} unit-SNR slots")
    return 0


def cmd_finalize():
    K24_ROOT.mkdir(parents=True, exist_ok=True)
    train_rows = []
    per_seed = {}
    for c in cells():
        p = c.dir() / "train_report.json"
        rec = json.loads(p.read_text()) if p.exists() else {
            "exit_status": "missing"}
        train_rows.append({
            "wave_id": c.wave_id, "channel": c.channel, "seed": c.seed,
            "k": c.k, "exit_status": rec.get("exit_status", "missing"),
            "task_loss_final": rec.get("task_loss_final"),
            "nan_batches": rec.get("nan_batches"),
            "grads_finite": rec.get("grads_finite"),
            "wall_seconds": rec.get("wall_seconds")})
        rows = []
        for snr in SNR_GRID:
            path = rw._eval_path(
                {"unit": f"k{c.k}_{c.channel}_seed{c.seed}", "k": c.k,
                 "channel": c.channel, "seed": c.seed}, snr)
            if not path.exists():
                continue
            agg = json.loads(path.read_text())["aggregate_" + c.tag]
            rows.append({"snr_db": snr,
                         "success_rate": agg["success_rate"],
                         "success_count": agg["success_count"],
                         "episodes": agg["episodes"],
                         "final_distance_mean": agg["final_distance_mean"],
                         "control_effort_mean": agg["control_effort_mean"]})
        per_seed[f"{c.channel}_seed{c.seed}"] = rows

    # mean + range across seeds, per channel and SNR (n = 3, not a CI)
    spread = []
    for ch in CHANNELS:
        for snr in SNR_GRID:
            vals = []
            for seed in SEEDS:
                for r in per_seed.get(f"{ch}_seed{seed}", []):
                    if r["snr_db"] == snr:
                        vals.append(r["success_rate"])
            if len(vals) == len(SEEDS):
                spread.append({"channel": ch, "snr_db": snr, "n_seeds": len(vals),
                               "mean": sum(vals) / len(vals),
                               "min": min(vals), "max": max(vals),
                               "range": max(vals) - min(vals),
                               "per_seed": vals})

    n_slots = sum(1 for u in units() for s in SNR_GRID
                  if rw._eval_path(u, s).exists())
    done = sum(1 for r in train_rows if r["exit_status"] == "complete")
    out = K24_ROOT / "step_k24_summary.json"
    out.write_text(json.dumps({
        "experiment": "K24_SEEDS",
        "amendment": AMENDMENT,
        "status": ("complete" if done == EXPECTED_CELLS
                   and n_slots == EXPECTED_SLOTS else "incomplete"),
        "question": ("is the k=24 lambda-arm number an artefact of one "
                     "initialisation?"),
        "design": {"arm": TAG, "lambdas": list(LAMBDAS), "k": K,
                   "seeds": list(SEEDS), "channels": list(CHANNELS),
                   "steps": config.TASK_TRAIN_STEPS, "width":
                   config.HIDDEN_DIMS[0], "eval_snr_grid_db": list(SNR_GRID),
                   "eval_episodes": rw.EVAL_EPISODES},
        "protocol_note": (f"E={rw.EVAL_EPISODES} Wave-A diagnostic grid "
                          "(frozen M1/M2 protocol uses E=5000)"),
        "statistics_note": ("n=3 seeds; per-seed values are reported and the "
                            "range across seeds is reported instead of a "
                            "confidence interval; no value is pooled across "
                            "seeds"),
        "train_rows": train_rows,
        "per_seed": per_seed,
        "spread_across_seeds": spread,
        "train_done": done,
        "eval_slots_complete": n_slots,
        "eval_slots_expected": EXPECTED_SLOTS,
    }, indent=2, default=str))
    print(f"wrote {out.relative_to(ROOT)} ({done}/{EXPECTED_CELLS} train, "
          f"{n_slots}/{EXPECTED_SLOTS} eval slots)")
    for row in spread:
        if row["snr_db"] in (0.0, 10.0, 20.0):
            print(f"  {row['channel']:8s} {row['snr_db']:+5.0f} dB  "
                  f"mean {row['mean']:.4f}  range {row['min']:.4f}"
                  f"-{row['max']:.4f}")
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
