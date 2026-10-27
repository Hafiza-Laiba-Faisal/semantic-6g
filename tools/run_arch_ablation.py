"""Architecture-capacity ablation (A-9): does a wider MLP rescue the failure?

The frozen task-oriented arm at k=18 reaches ~0.02 success and its loss
oscillates.  A reviewer is entitled to ask whether the failure is a capacity
artefact: the frozen MLP is 6-128-128-2k.  This runner trains the same task
loss at **hidden width 256** and compares it against the lambda-shaped arm at
the same width.

Why a separate runner: the frozen M1/M2 matrix and Wave-A/B/C universes are
already declared (18 units / 90 slots, 350 cells, 32 cells).  Adding cells to
any of them would silently change a published denominator.  This runner has
its own root, `results/ARCH_ABL/`, and never writes anywhere else.

Design (pre-registered in PROTOCOL_AMENDMENTS.md A-9, before the first cell):

  arms      : frozen task loss (lambda_u = lambda_v = 0)
  hidden    : 256 (the frozen width is 128)
  k         : 18 (the condition where the frozen arm fails and training is
              unstable; k=54 is saturated for the digital baseline, so it
              cannot discriminate capacity)
  channels  : awgn, rayleigh
  seeds     : 42
  steps     : config.TASK_TRAIN_STEPS (3000, same budget as every other arm)
  cells     : 2
  eval      : E=1000 per unit-SNR, 5-SNR grid -> 2 units / 10 slots

SCOPE REDUCTION (recorded in A-9, taken at 12:25 on 2026-10-26 with no cell
completed and no metric in existence): the pre-registered design was 8 cells
(2 arms x 2 channels x 2 seeds).  A 256-wide cell costs roughly 3-4x the wall
clock of a frozen 128-wide cell on this machine, so the run was cut to the
2 cells that answer the headline question - does width 256 rescue the frozen
task arm, on each channel.  The lambda arm at 256 and both seed-43 cells stay
declared but unrun; they can be added later without changing anything that is
reported.  Do not describe the ablation as 2x2x2.

Important history: the Wave-A arm `caps_h256` *claimed* width 256 but every
layer was 128-wide, because the model constructors captured
``config.HIDDEN_DIMS`` as a default argument at import time.  That arm is
declared INVALID and is not retrained.  The constructor bug is fixed in
project/models/{deepjscc_encoder,deepjscc_decoder,reconstruction_jscc}.py and
this runner verifies the actual weight shape before training.

Usage:
    python tools/run_arch_ablation.py verify
    python tools/run_arch_ablation.py smoke
    python tools/run_arch_ablation.py train [--shard=i/N]
    python tools/run_arch_ablation.py evaluate
    python tools/run_arch_ablation.py finalize
    python tools/run_arch_ablation.py status
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import tools.run_wave_a as rw  # noqa: E402
from project import config  # noqa: E402
from project.evaluation.evaluate import Condition  # noqa: E402
from project.models.reconstruction_jscc import ReconstructionDeepJSCC  # noqa: E402

ARCH_ROOT = ROOT / "results" / "ARCH_ABL"
AMENDMENT = "ARCH_CAPACITY-2026-10-26"
WIDTH = 256
K = 18
CHANNELS = ("awgn", "rayleigh")
SEEDS = (42,)
ARMS = {
    # tag -> lambdas (None = frozen task loss, i.e. lambda_u = lambda_v = 0)
    "frozen_task": None,
}
DECLARED_UNRUN = {
    "lam_ctl_heavy@256": "lambda arm at width 256 (declared, not run)",
    "seed43@256": "seed 43 for both channels (declared, not run)",
}
SNR_GRID = rw.MATCH_SNR_DB

rw.WAVE_A_ROOT = ARCH_ROOT
rw.WAVE_LABEL = "Arch"
rw.WAVE_EXPERIMENT = "ARCH_ABL"
rw.WAVE_AMENDMENT = AMENDMENT


class ArchCell(rw.WaveCell):
    """Wave-A cell with an ARCH identity and an explicit width."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, hidden=WIDTH, **kwargs)
        self.wave_id = self.wave_id.replace("WAVE_A_", "ARCH_")

    def spec(self):
        spec = super().spec()
        spec["experiment"] = "ARCH_ABL"
        spec["amendment"] = AMENDMENT
        spec["hidden"] = WIDTH
        return spec

    def load_ckpt(self):
        """Load with the REAL width; the base loader assumes the frozen 128."""
        ckpt = self.dir() / "final_ckpt.pt"
        if not ckpt.exists():
            return None
        model = ReconstructionDeepJSCC(k=self.k, hidden_dims=(WIDTH, WIDTH))
        payload = torch.load(ckpt, weights_only=False)
        model.load_state_dict(payload["model_state"])
        return model


def cells():
    out = []
    for tag, lam in ARMS.items():
        for ch in CHANNELS:
            for seed in SEEDS:
                out.append(ArchCell("A4", "task", ch, K, seed, tag=tag,
                                    lambdas=lam))
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
# smoke: the width MUST be 256 for real, not just in metadata
# ---------------------------------------------------------------------------

def cmd_smoke():
    c = cells()[0]
    model = ReconstructionDeepJSCC(k=c.k, hidden_dims=(WIDTH, WIDTH))
    w = tuple(model.encoder.backbone[0].weight.shape)
    print(f"== smoke: {w} (expect ({WIDTH}, {config.STATE_DIM})) ==")
    if w != (WIDTH, config.STATE_DIM):
        print("FAIL width")
        return 1
    c.steps = 50
    import time
    t0 = time.perf_counter()
    _model, rep, grads_finite, summary = rw._train_one(c)
    wall = time.perf_counter() - t0
    print(f"   {c.wave_id} 50 steps in {wall:.1f}s "
          f"nan={rep['nan_batches']} grads_finite={grads_finite} "
          f"loss={summary.get('task_loss_final')}")
    print("   smoke wrote NOTHING (no markers, no checkpoints)")
    return 0


def cmd_verify():
    print("== capacity-ablation pre-flight ==")
    bad = 0
    if ARCH_ROOT.name != "ARCH_ABL":
        print(f"FAIL [root] {ARCH_ROOT}")
        bad += 1
    else:
        print(f"PASS [isolated root] {ARCH_ROOT.relative_to(ROOT)}")
    if (EXPECTED_CELLS, EXPECTED_UNITS, EXPECTED_SLOTS) != (2, 2, 10):
        print(f"FAIL [design] {EXPECTED_CELLS}/{EXPECTED_UNITS}/{EXPECTED_SLOTS}")
        bad += 1
    else:
        print(f"PASS [design] {EXPECTED_CELLS} cells, {EXPECTED_UNITS} units, "
              f"{EXPECTED_SLOTS} slots (declared-unrun: {sorted(DECLARED_UNRUN)})")
    widths = {c.hidden for c in cells()}
    print(f"PASS [width] {sorted(widths)} (frozen is "
          f"{list(config.HIDDEN_DIMS)[0]})")
    if widths != {WIDTH}:
        bad += 1
    steps = {c.steps for c in cells()}
    if steps != {config.TASK_TRAIN_STEPS}:
        print(f"FAIL [step budget] {steps}")
        bad += 1
    else:
        print(f"PASS [step budget] {steps.pop()}")
    model = ReconstructionDeepJSCC(k=K, hidden_dims=(WIDTH, WIDTH))
    shape = tuple(model.encoder.backbone[0].weight.shape)
    if shape != (WIDTH, config.STATE_DIM):
        print(f"FAIL [real width] {shape}")
        bad += 1
    else:
        print(f"PASS [real width] encoder.backbone.0.weight={shape}")
    wa = len(rw.wave_cells())
    print(f"INFO [wave-a universe untouched] {wa} ids")
    print("== gates OK ==" if bad == 0 else f"== {bad} gate(s) FAILED ==")
    return 0 if bad == 0 else 1


# ---------------------------------------------------------------------------
# train / evaluate
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


def cmd_evaluate(args=()):
    rw.wave_cells = lambda: cells()
    return rw.cmd_evaluate(None)


# ---------------------------------------------------------------------------
# finalize / status
# ---------------------------------------------------------------------------

def _state(c):
    d = ARCH_ROOT / c.run_dirname()
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
    done = states.get("done", 0)
    print(f"== {ARCH_ROOT.relative_to(ROOT)} ==")
    print(f"-- train: {states} of {EXPECTED_CELLS} "
          f"({100 * done / EXPECTED_CELLS:.0f}%)")
    print(f"-- eval: {n_slots}/{EXPECTED_SLOTS} unit-SNR slots")
    return 0


def cmd_finalize():
    ARCH_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cells():
        p = c.dir() / "train_report.json"
        rec = json.loads(p.read_text()) if p.exists() else {"exit_status":
                                                            "missing"}
        rows.append({"wave_id": c.wave_id, "tag": c.tag or "frozen_task",
                     "k": c.k, "channel": c.channel, "seed": c.seed,
                     "hidden": c.hidden,
                     "exit_status": rec.get("exit_status", "missing"),
                     "task_loss_final": rec.get("task_loss_final"),
                     "nan_batches": rec.get("nan_batches"),
                     "grads_finite": rec.get("grads_finite"),
                     "wall_seconds": rec.get("wall_seconds")})
    pooled = {}
    n_slots = 0
    complete_units = 0
    for u in units():
        got = 0
        for snr in SNR_GRID:
            p = rw._eval_path(u, snr)
            if not p.exists():
                continue
            n_slots += 1
            got += 1
            rec = json.loads(p.read_text())
            for key, agg in rec.items():
                if key.startswith("aggregate_"):
                    tag = key[len("aggregate_"):]
                    pooled.setdefault((tag, u["k"], u["channel"], snr),
                                      []).append(agg)
        if got == len(SNR_GRID):
            complete_units += 1
    prow = []
    for (tag, k, ch, snr), aggs in sorted(
            pooled.items(), key=lambda kv: (kv[0][2], kv[0][3], kv[0][0])):
        sc = sum(a["success_count"] for a in aggs)
        tot = sum(a["episodes"] for a in aggs)
        prow.append({"tag": tag, "k": k, "channel": ch, "snr_db": snr,
                     "n_seeds": len(aggs), "success_rate_pooled": sc / tot,
                     "final_distance_mean": sum(a["final_distance_mean"]
                                                for a in aggs) / len(aggs),
                     "control_effort_mean": sum(a["control_effort_mean"]
                                                for a in aggs) / len(aggs)})
    done = sum(1 for r in rows if r["exit_status"] == "complete")
    out = ARCH_ROOT / "step_arch_summary.json"
    out.write_text(json.dumps({
        "experiment": "ARCH_ABL",
        "status": ("complete" if done == EXPECTED_CELLS
                   and n_slots == EXPECTED_SLOTS else "incomplete"),
        "amendment": AMENDMENT,
        "question": ("is the frozen-arm failure a capacity artefact? "
                     "hidden 256 vs the frozen 128 at k=18"),
        "design": {"width": WIDTH, "k": K, "arms": {
            t: (None if l is None else list(l)) for t, l in ARMS.items()},
            "channels": list(CHANNELS), "seeds": list(SEEDS),
            "steps": config.TASK_TRAIN_STEPS},
        "protocol_note": (f"E={rw.EVAL_EPISODES} diagnostic grid; the frozen "
                          "M1/M2 protocol uses E=5000"),
        "train_rows": rows,
        "pooled_rows": prow,
        "declared_but_unrun": DECLARED_UNRUN,
        "scope_note": ("pre-registered 2x2x2=8 cells; reduced to the frozen "
                       "arm x 2 channels x seed 42 at 12:25 UTC on "
                       "2026-10-26 (no cell completed, no metric existed)"),
        "train_done": done,
        "eval_slots_complete": n_slots,
        "eval_slots_expected": EXPECTED_SLOTS,
        "eval_units_complete": complete_units,
        "capacity_bug_note": ("Wave-A cap_h256 was invalid because the model "
                              "constructors bound config.HIDDEN_DIMS at import "
                              "time; this run verifies the real weight shape "
                              "before training."),
    }, indent=2, default=str))
    print(f"wrote {out.relative_to(ROOT)} ({done}/{EXPECTED_CELLS} train, "
          f"{n_slots}/{EXPECTED_SLOTS} eval slots)")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "smoke":
        return cmd_smoke()
    if cmd == "verify":
        return cmd_verify()
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
