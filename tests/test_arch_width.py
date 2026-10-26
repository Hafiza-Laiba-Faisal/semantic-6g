"""Capacity-ablation plumbing tests (A-9).

Regression guard for the bug that invalidated the Wave-A ``caps_h256`` arm:
the model constructors used ``hidden_dims: tuple = config.HIDDEN_DIMS``, a
default evaluated at IMPORT time, so patching the config later (the way the
ablation runner does it) silently produced 128-wide layers while the metadata
said 256.

These tests assert the fix and the pre-registered ablation design.  They do
not train and do not touch ``results/``.

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_arch_width
"""

from __future__ import annotations

import sys

SUMMARY = []


def check(name, ok, detail=""):
    SUMMARY.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]"
          + (f"  ({detail})" if detail else ""))


def test_explicit_width():
    from project import config
    from project.models.reconstruction_jscc import ReconstructionDeepJSCC

    m = ReconstructionDeepJSCC(k=18, hidden_dims=(256, 256))
    enc = tuple(m.encoder.backbone[0].weight.shape)
    check("explicit hidden_dims=256 builds a 256-wide encoder",
          enc == (256, config.STATE_DIM), str(enc))
    # final layer must still emit 2k reals
    out = tuple(m.encoder.backbone[-1].weight.shape)
    check("encoder output is still 2k real values", out == (2 * 18, 256), str(out))


def test_config_patch_is_honoured():
    from project import config
    from project.models.reconstruction_jscc import ReconstructionDeepJSCC

    old = config.HIDDEN_DIMS
    try:
        config.HIDDEN_DIMS = (256, 256)
        m = ReconstructionDeepJSCC(k=18)
        enc = tuple(m.encoder.backbone[0].weight.shape)
        check("runtime config patch now reaches the model (was the bug)",
              enc == (256, config.STATE_DIM), str(enc))
    finally:
        config.HIDDEN_DIMS = old

    m = ReconstructionDeepJSCC(k=18)
    enc = tuple(m.encoder.backbone[0].weight.shape)
    check("frozen default is still 128", enc == (128, config.STATE_DIM), str(enc))


def test_decoder_matches_encoder_width():
    from project import config
    from project.models.reconstruction_jscc import ReconstructionDeepJSCC

    m = ReconstructionDeepJSCC(k=18, hidden_dims=(256, 256))
    enc_in = m.encoder.backbone[-1].weight.shape[0]      # 2k real outputs
    dec_in = m.decoder.backbone[0].weight.shape[1]       # decoder consumes 2k
    check("encoder/decoder agree on 2k", enc_in == dec_in == 36,
          f"{enc_in}/{dec_in}")
    dec = tuple(m.decoder.backbone[0].weight.shape)
    check("decoder is 256-wide too", dec == (256, 36), str(dec))


def test_ablation_design():
    import tools.run_wave_a as rw

    saved = (rw.WAVE_A_ROOT, rw.WAVE_LABEL, rw.WAVE_EXPERIMENT,
             rw.WAVE_AMENDMENT)
    try:
        import tools.run_arch_ablation as arch
    finally:
        (rw.WAVE_A_ROOT, rw.WAVE_LABEL, rw.WAVE_EXPERIMENT,
         rw.WAVE_AMENDMENT) = saved

    cells = arch.cells()
    check("2 cells declared (reduced scope)", len(cells) == 2, str(len(cells)))
    check("2 units declared", len(arch.units()) == 2, str(len(arch.units())))
    check("lambda arm at 256 is declared-unrun",
          "lam_ctl_heavy@256" in arch.DECLARED_UNRUN,
          str(sorted(arch.DECLARED_UNRUN)))
    check("root is isolated", arch.ARCH_ROOT.name == "ARCH_ABL",
          str(arch.ARCH_ROOT))
    check("all cells are 256-wide", {c.hidden for c in cells} == {256},
          str({c.hidden for c in cells}))
    check("k=18 only", {c.k for c in cells} == {18}, str({c.k for c in cells}))
    check("one arm (frozen task)", {c.tag for c in cells} == {"frozen_task"},
          str({c.tag for c in cells}))
    check("seed 42 only", {c.seed for c in cells} == {42},
          str({c.seed for c in cells}))
    from project.models.reconstruction_jscc import ReconstructionDeepJSCC
    shape = tuple(ReconstructionDeepJSCC(
        k=18, hidden_dims=(arch.WIDTH, arch.WIDTH)).encoder
        .backbone[0].weight.shape)
    check("ablation width is real, not metadata", shape == (256, 6), str(shape))
    check("Wave-A universe untouched", len(rw.wave_cells()) == 46,
          str(len(rw.wave_cells())))


if __name__ == "__main__":
    print("=" * 72)
    print("CAPACITY-ABLATION PLUMBING TESTS (no training)")
    print("=" * 72)
    test_explicit_width()
    test_config_patch_is_honoured()
    test_decoder_matches_encoder_width()
    test_ablation_design()
    n_pass, n_all = sum(SUMMARY), len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)
