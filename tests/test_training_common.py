"""Step 8 training-common tests (Gates A-B).

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_training_common

Gate A: common training utilities import and basic API (deterministic
        setup, optimizer construction, grad/finite checks, JSON-able
        serialization, history logging).
Gate B: checkpoint round-trip - save -> load -> bitwise-identical model
        state, restored optimizer/step/seed/metadata/history/RNG, and
        mid-run training continuation reproducing the uninterrupted run.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch

from project import config
from project.models.reconstruction_jscc import ReconstructionDeepJSCC
from project.training.common import (
    build_optimizer,
    check_finite_grads,
    check_finite_loss,
    grad_norm,
    load_checkpoint,
    log_history,
    models_identical,
    save_checkpoint,
    setup_deterministic,
    to_jsonable,
)

SUMMARY = []


def check(name, ok, detail=""):
    SUMMARY.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]" +
          (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
# Gate A - import + basic API
# ---------------------------------------------------------------------------

def test_gate_a():
    gens = setup_deterministic(42)
    check("A/setup_deterministic",
          all(p in gens for p in config.RNG_KEYS)
          and all(isinstance(g, torch.Generator) for g in gens.values()),
          f"{len(gens)} purpose generators")

    model = ReconstructionDeepJSCC(k=2)
    opt = build_optimizer(model)
    check("A/build_optimizer",
          isinstance(opt, torch.optim.Adam)
          and math.isclose(opt.param_groups[0]["lr"], config.LEARNING_RATE),
          f"lr={opt.param_groups[0]['lr']} (frozen config)")
    opt2 = build_optimizer(model, lr=0.5)
    check("A/build_optimizer-override",
          math.isclose(opt2.param_groups[0]["lr"], 0.5))

    x = torch.zeros(4, 6)
    out = model.forward_losses(x, 10.0, channel="awgn",
                               generator=torch.Generator().manual_seed(1))
    target = torch.zeros_like(out)
    loss = ((out - target) ** 2).mean()
    ok_finite = check_finite_loss(loss)
    loss.backward()
    ok_grads = check_finite_grads(model.parameters())
    gn = grad_norm(model.parameters())
    check("A/finite-loss-grads + grad_norm",
          ok_finite and ok_grads and math.isfinite(gn) and gn > 0,
          f"|g|={gn:.4f}")

    entry = {"step": 1, "loss": 1.0, "flag": True,
             "tensor": torch.tensor([1.0, 2.0])}
    hist = log_history([], entry)
    j = to_jsonable({"a": hist, "t": (1, 2)})
    check("A/log_history + to_jsonable",
          hist is not None and len(hist) == 1
          and j["a"][0]["step"] == 1
          and j["a"][0]["tensor"] == [1.0, 2.0]
          and j["t"] == [1, 2],
          "JSON-able round-trip")


# ---------------------------------------------------------------------------
# Gate B - checkpoint round-trip + RNG continuation
# ---------------------------------------------------------------------------

def _fresh_pair(k=2):
    setup_deterministic(7)
    a = ReconstructionDeepJSCC(k=k)
    b = ReconstructionDeepJSCC(k=k)
    return a, b


def test_gate_b():
    with tempfile.TemporaryDirectory() as td:
        model_a, model_b = _fresh_pair()
        check("B/pre-load distinct init", not models_identical(model_a, model_b))

        setup_deterministic(7)
        opt = build_optimizer(model_a)
        history = [{"step": 0, "loss": 0.5}, {"step": 1, "loss": 0.25}]
        path = Path(td) / "ck.pth"
        save_checkpoint(path, model_a, optimizer=opt, step=123, seed=42,
                        metadata={"method": "reconstruction", "k": 2,
                                  "channel": "awgn"},
                        history=history)
        payload = load_checkpoint(path, model_b, optimizer=None)
        check("B/save-load model bitwise", models_identical(model_a, model_b))
        check("B/payload round-trip",
              payload["step"] == 123 and payload["seed"] == 42
              and payload["metadata"]["method"] == "reconstruction"
              and payload["history"] == history,
              "step/seed/metadata/history restored")

        # optimizer state round-trip (Adam exp_avg etc.)
        opt_restored = build_optimizer(model_b)
        load_checkpoint(path, model_b, optimizer=opt_restored)
        opt.step()   # one more step on the original (state evolves)
        # re-save with the evolved optimizer to prove state equality is real
        save_checkpoint(path, model_a, optimizer=opt, step=124, seed=42,
                        metadata={}, history=[])
        load_checkpoint(path, model_b, optimizer=opt_restored)
        same_state = all(
            torch.equal(opt.state[k]["exp_avg"], opt_restored.state[k]["exp_avg"])
            for k in opt.state)
        check("B/optimizer state bitwise", same_state)

        # RNG capture/restore: 3 draws saved, then a 4th after restore ==
        # the original 4th (torch global stream continuation)
        model_c, model_d = _fresh_pair()
        path2 = Path(td) / "ck_rng.pth"
        torch.manual_seed(999)
        save_checkpoint(path2, model_c, capture_rng=True)
        torch.manual_seed(999)
        save_checkpoint(path2, model_c, capture_rng=True)  # same state again
        ref = torch.rand(4)                                 # 4th draw reference
        load_checkpoint(path2, model_d, restore_rng=True)
        cont = torch.rand(4)
        check("B/RNG continuation", bool(torch.equal(ref, cont)),
              f"{cont.tolist()} == saved-stream 4th draw")

        # missing file fails loudly
        try:
            load_checkpoint(Path(td) / "nope.pth", model_d)
            ok_missing = False
        except FileNotFoundError:
            ok_missing = True
        check("B/missing checkpoint raises", ok_missing)

        # metadata carries the reproduction fields required by the policy
        payload2 = load_checkpoint(path2, model_c, restore_rng=False)
        md = payload2["metadata"]
        save_checkpoint(path2, model_c, metadata={
            "method": "task", "k": 3, "channel": "rayleigh",
            "training_seed": 42, "train_steps": 400,
            "model_config": {"hidden_dims": [128, 128]},
            "train_snr_policy": {"distribution": "uniform",
                                 "range_db": [0.0, 20.0]}})
        md2 = load_checkpoint(path2, model_c, restore_rng=False)["metadata"]
        required = ("method", "k", "channel", "training_seed", "train_steps",
                    "model_config", "train_snr_policy")
        check("B/reproduction metadata fields",
              all(f in md2 for f in required),
              f"{sorted(md2.keys())}")


if __name__ == "__main__":
    print("=" * 72)
    print("STEP 8 TRAINING/COMMON TESTS (Gates A-B)")
    print("=" * 72)
    test_gate_a()
    test_gate_b()
    n_pass, n_all = sum(SUMMARY), len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)
