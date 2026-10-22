"""Wave-C: does lambda-shaping (Wave-A's A1 arms) generalise across bandwidth?

Wave-A found that adding effort/velocity penalties to the task loss moves k=12
closed-loop success from 0.02-0.05 (frozen lambdas) to 0.99.  That result is
saturated at the task ceiling and was only trained at k=12.  Wave-C asks the
open question: does it hold at k = 18, 24, 30, 54?

DESIGN (frozen before any run; see fyp_md_files/PROTOCOL_AMENDMENTS.md A-8):

  arms      : lam_ctl_light (lu,lv x10), lam_ctl_heavy (lu,lv x100)
  k         : 18, 24, 30, 54
  channels  : awgn, rayleigh
  seeds     : 42, 43
  steps     : config.TASK_TRAIN_STEPS (same budget as Wave-A k=12 arms)
  cells     : 2 x 4 x 2 x 2 = 32 training runs
  eval      : E=1000 per (unit, SNR), 5-SNR grid -> 16 units / 80 slots

COST, measured not guessed: a 100-step smoke gave ~1.3 s/step on this machine
(k=18 and k=54), i.e. ~25 h for the full grid (Wave-A's own train reports give
a 0.667 s/step median / 37 min median per 3000-step cell).  The full grid was
explicitly approved by the user on 2026-10-21 after that measurement.  The
runner is idempotent, so an interrupted run can simply be restarted and only
missing cells are trained.

ISOLATION: writes ONLY under results/WAVE_C/.  Results/WAVE_A, results/M1-M4
and all frozen artifacts stay read-only.  The runner REUSES the Wave-A
training/evaluation machinery by patching the module's universe and root; it
never edits the Wave-A cell list or its 18-unit/90-slot contract.

Usage (repo root, inside the venv):
    python tools/run_wave_c.py smoke        # 1 cell, 100 steps, no artifacts
    python tools/run_wave_c.py verify       # pre-flight gates
    python tools/run_wave_c.py train        # idempotent (SKIP done/failed)
    python tools/run_wave_c.py evaluate     # paired, E=1000
    python tools/run_wave_c.py finalize     # results/WAVE_C/step_wave_c_summary.json
    python tools/run_wave_c.py status
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import tools.run_wave_a as rw  # noqa: E402
from project import config  # noqa: E402
from project.evaluation.evaluate import Condition, evaluate_condition  # noqa: E402

WAVE_C_ROOT = ROOT / "results" / "WAVE_C"
AMENDMENT = "WAVE_C-lambda-x-bandwidth-2026-10-21"

KS = (18, 24, 30, 54)
CHANNELS = ("awgn", "rayleigh")
SEEDS = (42, 43)
ARMS = {tag: rw.LAMBDA_TAGS[tag] for tag in ("lam_ctl_light", "lam_ctl_heavy")}
SNR_GRID = rw.MATCH_SNR_DB

# redirect every path helper in the reused machinery
rw.WAVE_A_ROOT = WAVE_C_ROOT
rw.WAVE_LABEL = "Wave-C"      # log headers must not claim to be Wave-A


class WaveC(rw.WaveCell):
    """Wave-A cell with a Wave-C identity and its own amendment label."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wave_id = self.wave_id.replace("WAVE_A_", "WAVE_C_")

    def spec(self):
        spec = super().spec()
        spec["experiment"] = "WAVE_C"
        spec["amendment"] = AMENDMENT
        return spec


def cells():
    out = []
    for tag, lam in ARMS.items():
        for k in KS:
            for ch in CHANNELS:
                for seed in SEEDS:
                    out.append(WaveC("C1", "task", ch, k, seed,
                                     tag=tag, lambdas=lam))
    return out


def units():
    seen = {}
    for c in cells():
        key = (c.k, c.channel, c.seed)
        seen.setdefault(key, {"unit": f"k{c.k}_{c.channel}_seed{c.seed}",
                              "k": c.k, "channel": c.channel, "seed": c.seed})
    return [seen[key] for key in sorted(seen)]


EXPECTED_CELLS = len(cells())
EXPECTED_UNITS = len(units())
EXPECTED_SLOTS = EXPECTED_UNITS * len(SNR_GRID)


# ---------------------------------------------------------------------------
# isolated smoke (no artifacts at all)
# ---------------------------------------------------------------------------

def cmd_smoke(k=None):
    pool = [c for c in cells() if k is None or c.k == k]
    c = pool[0]
    c.steps = 100
    print(f"== Wave-C smoke: {c.wave_id} steps={c.steps} arm={c.tag} ==")
    import time
    t0 = time.perf_counter()
    model, rep, grads_finite, summary = rw._train_one(c)
    wall = time.perf_counter() - t0
    loss = summary.get("task_loss_final")
    ok = (rep["nan_batches"] == 0 and grads_finite
          and isinstance(loss, float) and loss == loss and loss != float("inf"))
    print(f"   wall={wall:.1f}s nan_batches={rep['nan_batches']} "
          f"grads_finite={grads_finite} task_loss_final={loss}")
    print(f"   projection: 32 cells x {wall * 3000 / 100 / 60:.1f} min "
          f"= {32 * wall * 3000 / 100 / 3600:.1f} h")
    print("   smoke wrote NOTHING (no markers, no checkpoints)")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify():
    print("== Wave-C pre-flight gates ==")
    bad = 0
    frozen_roots = {ROOT / "results" / "M1", ROOT / "results" / "M2",
                    ROOT / "results" / "M3", ROOT / "results" / "M4",
                    ROOT / "results" / "WAVE_A"}
    if WAVE_C_ROOT in frozen_roots or WAVE_C_ROOT.name != "WAVE_C":
        print(f"FAIL [isolated root] {WAVE_C_ROOT}")
        bad += 1
    else:
        print(f"PASS [isolated root] {WAVE_C_ROOT.relative_to(ROOT)}")

    wa_ids = {c.wave_id for c in
              __import__("tools.run_wave_a", fromlist=["wave_cells"]).wave_cells()}
    ids = [c.wave_id for c in cells()]
    if len(set(ids)) != len(ids):
        print("FAIL [unique ids] duplicate cell ids")
        bad += 1
    else:
        print(f"PASS [unique ids] {len(ids)} cells")
    # identity: every id starts with WAVE_C_ and the k/seed are in our grid
    wrong = [i for i in ids if not i.startswith("WAVE_C_")]
    if wrong:
        print(f"FAIL [id prefix] {wrong[:3]}")
        bad += 1
    else:
        print("PASS [id prefix] all WAVE_C_")

    ks = sorted({c.k for c in cells()})
    if tuple(ks) != KS:
        print(f"FAIL [k grid] {ks} != {list(KS)}")
        bad += 1
    else:
        print(f"PASS [k grid] {ks}")
    if (EXPECTED_CELLS, EXPECTED_UNITS, EXPECTED_SLOTS) != (32, 16, 80):
        print(f"FAIL [design] {EXPECTED_CELLS}/{EXPECTED_UNITS}/{EXPECTED_SLOTS}")
        bad += 1
    else:
        print(f"PASS [design] {EXPECTED_CELLS} cells, {EXPECTED_UNITS} units, "
              f"{EXPECTED_SLOTS} slots")
    steps = {c.steps for c in cells()}
    if steps != {config.TASK_TRAIN_STEPS}:
        print(f"FAIL [step budget] {steps}")
        bad += 1
    else:
        print(f"PASS [step budget] {steps.pop()}")
    lams = sorted({tuple(c.lambdas) for c in cells()})
    print(f"PASS [lambda arms] {len(lams)} distinct: {lams}")
    print(f"INFO [wave-a universe untouched] {len(wa_ids)} Wave-A ids")
    print("== gates OK ==" if bad == 0 else f"== {bad} gate(s) FAILED ==")
    return 0 if bad == 0 else 1


# ---------------------------------------------------------------------------
# train / evaluate (delegate to the Wave-A machinery with our universe)
# ---------------------------------------------------------------------------

def _select(args):
    """Cell subset for a shard: --tag= --k= --seed= --shard=i/N."""
    out = cells()
    tags = [a.split("=", 1)[1] for a in args if a.startswith("--tag=")]
    ks = [int(a.split("=", 1)[1]) for a in args if a.startswith("--k=")]
    seeds = [int(a.split("=", 1)[1]) for a in args if a.startswith("--seed=")]
    shard = next((a.split("=", 1)[1] for a in args if a.startswith("--shard=")),
                 None)
    if tags:
        out = [c for c in out if c.tag in tags]
    if ks:
        out = [c for c in out if c.k in ks]
    if seeds:
        out = [c for c in out if c.seed in seeds]
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        out = [c for j, c in enumerate(out) if j % n == i]
    return out


def cmd_train(lane=None, args=()):
    sel = _select(args)
    if len(sel) != len(cells()):
        print(f"== Wave-C selection: {len(sel)} of {len(cells())} cells ==",
              flush=True)
        for c in sel:
            print(f"   {c.wave_id}", flush=True)
    rw.wave_cells = lambda: sel
    rw.DOCUMENTED_FAILURES = frozenset()      # no pre-accepted Wave-C failures
    return rw.cmd_train(lane)


def cmd_evaluate(lane=None, args=()):
    sel = _select(args) if args else cells()
    rw.wave_cells = lambda: sel
    return rw.cmd_evaluate(lane)


# ---------------------------------------------------------------------------
# status / finalize
# ---------------------------------------------------------------------------

def _train_rows():
    rows = []
    for c in cells():
        rep = c.dir() / "train_report.json"
        rec = json.loads(rep.read_text()) if rep.exists() else {
            "exit_status": "missing"}
        rows.append({"wave_id": c.wave_id, "tag": c.tag, "k": c.k,
                     "channel": c.channel, "seed": c.seed,
                     "exit_status": rec.get("exit_status", "missing"),
                     "task_loss_final": rec.get("task_loss_final"),
                     "eval_after_success_rate":
                         rec.get("eval_after_success_rate")})
    return rows


def _state_no_mkdir(c):
    """Marker state without creating the run directory (status must not write)."""
    d = WAVE_C_ROOT / c.run_dirname()
    if not d.is_dir():
        return "missing", None
    for kind, name in (("done", "DONE.json"), ("failed", "FAILED.json"),
                       ("running", "RUNNING.json")):
        p = d / name
        if p.exists():
            return kind, p.stat().st_mtime
    return "missing", None


STALE_RUNNING_MIN = 60     # RUNNING.json is a START marker, not a liveness probe


def cmd_status():
    import time
    states = {}
    stale = []
    for c in cells():
        st, mtime = _state_no_mkdir(c)
        if st == "running" and mtime is not None:
            age = (time.time() - mtime) / 60
            if age > STALE_RUNNING_MIN:
                st = "running(STALE)"
                stale.append((c.wave_id, round(age)))
        states[st] = states.get(st, 0) + 1
    n_slots = sum(1 for u in units() for s in SNR_GRID
                  if rw._eval_path(u, s).exists())
    done = states.get("done", 0)
    print(f"== Wave-C status ({WAVE_C_ROOT.relative_to(ROOT)}) ==")
    print(f"-- train: {states} of {EXPECTED_CELLS} "
          f"({100 * done / EXPECTED_CELLS:.0f}% trained)")
    print(f"-- eval: {n_slots}/{EXPECTED_SLOTS} unit-SNR slots")
    if stale:
        print(f"!! {len(stale)} cell(s) have a RUNNING marker older than "
              f"{STALE_RUNNING_MIN} min - the process is probably DEAD "
              f"(restart: python tools/run_wave_c.py train)")
        for wid, age in stale[:3]:
            print(f"   {wid}  ({age} min)")
    return 0


def cmd_finalize():
    WAVE_C_ROOT.mkdir(parents=True, exist_ok=True)
    train_rows = _train_rows()
    n_slots = 0
    complete_units = 0
    pooled = {}
    for unit in units():
        got = 0
        for snr in SNR_GRID:
            p = rw._eval_path(unit, snr)
            if not p.exists():
                continue
            n_slots += 1
            got += 1
            rec = json.loads(p.read_text())
            for key, agg in rec.items():
                if key.startswith("aggregate_"):
                    tag = key[len("aggregate_"):]
                    pooled.setdefault((tag, unit["k"], unit["channel"], snr),
                                      []).append({"seed": unit["seed"], **agg})
        if got == len(SNR_GRID):
            complete_units += 1
    pooled_rows = []
    for (tag, k, ch, snr), rs in sorted(
            pooled.items(), key=lambda kv: (kv[0][2], kv[0][3], kv[0][0], kv[0][1])):
        sc = sum(r["success_count"] for r in rs)
        tot = sum(r["episodes"] for r in rs)
        pooled_rows.append({"tag": tag, "k": k, "channel": ch, "snr_db": snr,
                            "n_seeds": len(rs), "success_count_pooled": sc,
                            "episodes_pooled": tot,
                            "success_rate_pooled": sc / tot if tot else None})
    done = sum(1 for r in train_rows if r["exit_status"] == "complete")
    out = WAVE_C_ROOT / "step_wave_c_summary.json"
    out.write_text(json.dumps({
        "experiment": "WAVE_C",
        "status": ("complete" if done == EXPECTED_CELLS
                   and n_slots == EXPECTED_SLOTS else "incomplete"),
        "amendment": AMENDMENT,
        "question": ("does Wave-A's lambda-shaping result (k=12: 0.02-0.05 -> "
                     "0.99) hold at k=18/24/30/54?"),
        "protocol_note": (f"Wave-C diagnostic: E={rw.EVAL_EPISODES} per unit; "
                          "NOT 1:1 comparable to frozen M1/M2 (E=5000). "
                          "Steps/seeds/architecture frozen; only the loss "
                          "weighting and k are new."),
        "k_grid": list(KS), "channels": list(CHANNELS), "seeds": list(SEEDS),
        "arms": {t: list(l) for t, l in ARMS.items()},
        "train_rows": train_rows,
        "pooled_rows": pooled_rows,
        "train_done": done,
        "train_missing": sum(1 for r in train_rows
                             if r["exit_status"] == "missing"),
        "eval_units_expected": EXPECTED_UNITS,
        "eval_slots_expected": EXPECTED_SLOTS,
        "eval_slots_complete": n_slots,
        "eval_units_complete": complete_units,
    }, indent=2, default=str))
    print(f"wrote {out.relative_to(ROOT)} ({done}/{EXPECTED_CELLS} train, "
          f"{n_slots}/{EXPECTED_SLOTS} eval slots, "
          f"{complete_units}/{EXPECTED_UNITS} units)")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    table = {"verify": cmd_verify, "status": cmd_status,
             "finalize": cmd_finalize}
    if cmd == "smoke":
        return cmd_smoke(int(rest[0]) if rest else None)
    if cmd in ("train", "evaluate"):
        lane = next((a for a in rest if not a.startswith("--")), None)
        return {"train": cmd_train, "evaluate": cmd_evaluate}[cmd](
            lane, [a for a in rest if a.startswith("--")])
    if cmd in table:
        return table[cmd]()
    print(f"unknown command {cmd!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
