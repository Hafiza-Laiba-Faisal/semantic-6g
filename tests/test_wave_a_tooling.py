"""Wave-A safe-repair regression tests (2026-10-20).

Covers the *tooling* fixes only; no training or evaluation is launched.
Every test uses a temporary directory, so the live ``results/WAVE_A``
tree is never written to.

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_wave_a_tooling
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SUMMARY = []


def check(name, ok, detail=""):
    SUMMARY.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]"
          + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
# A. evaluation universe
# ---------------------------------------------------------------------------

def test_eval_universe():
    from tools.run_wave_a import MATCH_SNR_DB, eval_units, wave_cells

    units = eval_units()
    keys = {(u["k"], u["channel"], u["seed"]) for u in units}
    expected = {(k, ch, s)
                for k in (1, 12) for ch in ("awgn", "rayleigh")
                for s in (42, 43, 44)}
    expected |= {(12, ch, s)
                 for ch in ("awgn", "rayleigh") for s in (45, 46, 47)}

    check("eval_units count == 18", len(units) == 18, str(len(units)))
    check("eval slots == 90",
          len(units) * len(MATCH_SNR_DB) == 90,
          str(len(units) * len(MATCH_SNR_DB)))
    check("eval_units == declared cell universe", keys == expected,
          f"missing={sorted(expected - keys)} extra={sorted(keys - expected)}")
    check("no k=1 extra-seed units",
          not any(k == 1 and s in (45, 46, 47) for k, _, s in keys))
    check("every declared cell maps to a unit",
          all((c.k, c.channel, c.seed) in keys for c in wave_cells()))
    check("unit ids are unique", len({u["unit"] for u in units}) == len(units))


# ---------------------------------------------------------------------------
# B. failure/retry contract
# ---------------------------------------------------------------------------

def test_retry_contract():
    from tools.run_wave_a import _should_skip

    check("DONE is skipped", _should_skip("done") is True)
    check("FAILED is skipped", _should_skip("failed") is True)
    check("RUNNING is retrained", _should_skip("running") is False)
    check("MISSING is trained", _should_skip("missing") is False)
    check("smoke cells are never skipped",
          _should_skip("done", "smoke:lam") is False)
    check("smoke failed cells are never skipped",
          _should_skip("failed", "smoke:lam") is False)


def test_marker_exclusivity():
    import tools.run_wave_a as rw

    old_root = rw.WAVE_A_ROOT
    try:
        with tempfile.TemporaryDirectory() as td:
            rw.WAVE_A_ROOT = Path(td)
            c = rw.wave_cells()[0]

            c.mark_failed("boom")
            check("failed state after mark_failed", c.state() == "failed")
            check("FAILED marker written", (c.dir() / "FAILED.json").exists())

            c.mark_done({"x": 1})
            check("done state after mark_done", c.state() == "done")
            check("mark_done removes stale FAILED",
                  not (c.dir() / "FAILED.json").exists())
            check("DONE marker written", (c.dir() / "DONE.json").exists())

            c.mark_failed("late error")
            check("mark_failed cannot overwrite DONE", c.state() == "done")
            check("late failure marker not created",
                  not (c.dir() / "FAILED.json").exists())
    finally:
        rw.WAVE_A_ROOT = old_root


# ---------------------------------------------------------------------------
# C. provenance persistence
# ---------------------------------------------------------------------------

def test_provenance_fields():
    from tools.run_wave_a import _provenance_fields

    task_rep = {
        "history": [{"total": 1.0}, {"total": 0.5}],
        "grad_enc_first": 1.0, "grad_enc_final": 2.0,
        "grad_dec_first": 3.0, "grad_dec_final": 4.0,
    }
    pf = _provenance_fields(task_rep, "task")
    check("schema v2 marker", pf.get("report_schema_version") == 2)
    check("task history persisted", pf.get("history") == task_rep["history"])
    check("task gradient health persisted",
          pf.get("grad_enc_final") == 2.0 and pf.get("grad_dec_final") == 4.0)

    recon_rep = {
        "history": [1.0, 0.5],
        "grad_encoder_history": [1.0, 2.0],
        "grad_decoder_history": [3.0, 4.0],
        "grad_norm_encoder_final": 2.0,
        "grad_norm_decoder_final": 4.0,
    }
    pf2 = _provenance_fields(recon_rep, "recon_w")
    check("recon history persisted", pf2.get("history") == [1.0, 0.5])
    check("recon gradient health mapped",
          pf2.get("grad_enc_final") == 2.0
          and pf2.get("grad_dec_final") == 4.0)

    pf3 = _provenance_fields({}, "task")
    check("absent history is not fabricated", "history" not in pf3)
    check("absent gradients are not fabricated",
          "grad_enc_final" not in pf3 and "grad_dec_final" not in pf3)


# ---------------------------------------------------------------------------
# D. finalize completeness denominator
# ---------------------------------------------------------------------------

def test_finalize_denominator():
    import tools.run_wave_a as rw

    old_root = rw.WAVE_A_ROOT
    try:
        with tempfile.TemporaryDirectory() as td:
            rw.WAVE_A_ROOT = Path(td)
            cells = rw.wave_cells()
            for c in cells:
                d = c.dir()
                (d / "train_report.json").write_text(json.dumps(
                    {"wave_id": c.wave_id, "exit_status": "complete"}))
                (d / "DONE.json").write_text("{}")
            unit = rw.eval_units()[0]
            for snr in rw.MATCH_SNR_DB:
                p = rw._eval_path(unit, snr)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(
                    {"aggregate_task": {"success_count": 1,
                                        "failure_count": 0,
                                        "episodes": 1}}))

            rw.cmd_finalize()
            summ = json.loads(
                (Path(td) / "step_wave_a_summary.json").read_text())

            check("finalize expects 18 units",
                  summ.get("eval_units_expected") == 18,
                  str(summ.get("eval_units_expected")))
            check("finalize expects 90 slots",
                  summ.get("eval_slots_expected") == 90,
                  str(summ.get("eval_slots_expected")))
            check("finalize is incomplete while eval slots are missing",
                  summ.get("status") == "incomplete")
            check("finalize reports all train rows done",
                  summ.get("train_done") == len(cells),
                  str(summ.get("train_done")))
            check("finalize counts the 5 completed eval slots",
                  summ.get("eval_slots_complete") == 5,
                  str(summ.get("eval_slots_complete")))
            check("finalize counts complete UNITS, not slots",
                  summ.get("eval_units_complete") == 1,
                  str(summ.get("eval_units_complete")))
            check("historical reports marked history-unavailable",
                  all(r["history_available"] is False
                      for r in summ.get("train_rows", [])))
    finally:
        rw.WAVE_A_ROOT = old_root


# ---------------------------------------------------------------------------
# E. evaluate: every arm must load ITS OWN checkpoint
# ---------------------------------------------------------------------------

def _sd_equal(a, b):
    import torch

    if set(a) != set(b):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def test_evaluate_loads_each_arm_checkpoint():
    """Regression for the evaluate loader bug (2026-10-20).

    ``cmd_evaluate`` used the stale outer loop variable ``c`` instead of the
    per-arm cell, so EVERY arm of a unit loaded the checkpoint of the last
    cell returned by ``wave_cells()``.  For k=1 units that meant a k=12
    checkpoint was fed k=1 noise, raising
    ``ValueError: noise shape/kind mismatch: (1000, 1) ... vs z (1000, 12)``.
    """
    import torch

    import tools.run_wave_a as rw
    from project.models.reconstruction_jscc import ReconstructionDeepJSCC

    old_root = rw.WAVE_A_ROOT
    old_cells = rw.wave_cells
    old_eval = rw.evaluate_condition
    captured = []
    try:
        with tempfile.TemporaryDirectory() as td:
            rw.WAVE_A_ROOT = Path(td)
            # two DISTINCT task arms inside the same (k, channel, seed) unit
            cells = [
                rw.WaveCell("A1", "task", "awgn", 12, 42, tag="t_alpha"),
                rw.WaveCell("A1", "task", "awgn", 12, 42, tag="t_beta"),
            ]
            reference = {}
            for i, c in enumerate(cells):
                d = c.dir()
                torch.manual_seed(1000 + i)
                model = ReconstructionDeepJSCC(k=c.k)
                torch.save({"model_state": model.state_dict()},
                           d / "final_ckpt.pt")
                (d / "DONE.json").write_text("{}")
                reference[c.tag] = model.state_dict()
            rw.wave_cells = lambda: cells

            def fake_evaluate_condition(cond, methods, recon_model=None,
                                        task_model=None):
                m = recon_model if recon_model is not None else task_model
                captured.append(m.state_dict())
                return {methods[0]: {"aggregate": {
                    "success_rate": 0.0, "success_count": 0,
                    "failure_count": 1000}}}

            rw.evaluate_condition = fake_evaluate_condition
            rc = rw.cmd_evaluate("awgn")

            check("evaluate returns 0 on an isolated unit", rc == 0, str(rc))
            check("evaluate ran every arm at every SNR",
                  len(captured) == len(cells) * len(rw.MATCH_SNR_DB),
                  str(len(captured)))

            per_tag = {tag: sum(1 for sd in captured if _sd_equal(sd, ref))
                       for tag, ref in reference.items()}
            check("each arm loads its own checkpoint",
                  per_tag == {"t_alpha": len(rw.MATCH_SNR_DB),
                              "t_beta": len(rw.MATCH_SNR_DB)},
                  str(per_tag))
            check("arm checkpoints are not silently identical",
                  not _sd_equal(reference["t_alpha"], reference["t_beta"]))

            unit = rw.eval_units()[0]
            out = rw._eval_path(
                {"unit": "k12_awgn_seed42", "k": 12,
                 "channel": "awgn", "seed": 42}, rw.MATCH_SNR_DB[0])
            check("evaluate writes one artifact per unit-SNR slot",
                  out.exists())
            rec = json.loads(out.read_text())
            check("artifact carries both arm aggregates",
                  {"aggregate_t_alpha", "aggregate_t_beta"} <= set(rec),
                  str(sorted(k for k in rec if k.startswith("aggregate_"))))
    finally:
        rw.WAVE_A_ROOT = old_root
        rw.wave_cells = old_cells
        rw.evaluate_condition = old_eval

    # the loader must build the model for the CELL's own k (root cause of the
    # k=1 vs k=12 noise-shape crash)
    old_root = rw.WAVE_A_ROOT
    try:
        with tempfile.TemporaryDirectory() as td:
            rw.WAVE_A_ROOT = Path(td)
            cell = rw.WaveCell("A1", "task", "awgn", 1, 42, tag="k1_loader")
            d = cell.dir()
            torch.manual_seed(7)
            torch.save({"model_state": ReconstructionDeepJSCC(k=1).state_dict()},
                       d / "final_ckpt.pt")
            loaded = cell.load_ckpt()
            z = loaded.encode(torch.zeros(4, 6))
            check("loader rebuilds the cell's own k (k=1 -> z [4,1])",
                  tuple(z.shape) == (4, 1), str(tuple(z.shape)))
    finally:
        rw.WAVE_A_ROOT = old_root


# ---------------------------------------------------------------------------
# F. M2 gap accounting: approved failures are EXCLUDED, never retrained
# ---------------------------------------------------------------------------

def test_m2_gap_accounting():
    """The M2 conditional-accept list is one source of truth for Wave-B."""
    import tools.run_wave_b as wb

    check("Wave-B excludes exactly the approved Rayleigh failure",
          wb.EXCLUDED_UNITS == {(54, 43): "M2_task_rayleigh_k54_seed43"},
          str(wb.EXCLUDED_UNITS))
    check("Wave-B coverage contract is 14/15 units",
          (wb.UNITS_COVERED, wb.UNITS_TOTAL) == (14, 15),
          f"{wb.UNITS_COVERED}/{wb.UNITS_TOTAL}")
    cells = list(wb.cells())
    check("Wave-B cell universe shrinks 375 -> 350",
          (len(cells), wb.EXPECTED_CELLS, wb.CELLS_TOTAL) == (350, 350, 375),
          f"{len(cells)} cells")
    check("excluded unit never appears in the run universe",
          all((k, seed) != (54, 43) for k, _snr, seed, _sig in cells))
    check("sigma0 gate expectation tracks coverage (70 not 75)",
          wb.EXPECTED_GATES == 70, str(wb.EXPECTED_GATES))

    import tools.build_dashboard as bd
    scope = bd._wave_b_scope()
    check("dashboard Wave-B denominator matches the runner",
          scope["expected_cells"] == wb.EXPECTED_CELLS and
          scope["covered_units"] == wb.UNITS_COVERED, str(scope))
    check("dashboard counts approved failures as accounted",
          len(bd._m2_conditional_accept(set())) == 5,
          str(len(bd._m2_conditional_accept(set()))))

    # regression: the Wave-A table must render once eval JSONs exist
    fake = {("t_alpha", 12, "awgn", 5.0): {"sc": 3, "tot": 10, "n": 3},
            ("t_alpha", 12, "rayleigh", 5.0): {"sc": 1, "tot": 10, "n": 3},
            ("t_beta", 1, "awgn", 0.0): {"sc": 0, "tot": 10, "n": 1}}
    try:
        table = bd._wave_table(fake)
        rendered = "<table" in table and "t_alpha" in table and "t_beta" in table
        detail = f"{len(table)} chars"
    except Exception as e:                      # noqa: BLE001
        rendered, detail = False, repr(e)
    check("Wave-A table renders a non-empty pooled dict", rendered, detail)


# ---------------------------------------------------------------------------
# G. run_all stall detector (no infinite relaunch loop)
# ---------------------------------------------------------------------------

def test_run_all_stall_detector():
    import tools.run_all as ra

    a = {"m1_train": {"status": "ok", "exit": 0},
         "wave_a_train": {"status": "failed", "exit": 1}}
    b = {"m1_train": {"status": "ok", "exit": 0},
         "wave_a_train": {"status": "failed", "exit": 1}}
    c = {"m1_train": {"status": "ok", "exit": 0},
         "wave_a_train": {"status": "ok", "exit": 0}}
    check("identical outcomes fingerprint identically",
          ra._progress_fingerprint(a) == ra._progress_fingerprint(b))
    check("a changed outcome changes the fingerprint",
          ra._progress_fingerprint(a) != ra._progress_fingerprint(c))
    check("stall exit code is distinct from a real failure",
          ra.NO_PROGRESS_EXIT == 3)
    check("documented-failure exit is not a stage failure",
          (ra._stage_status(0), ra._stage_status(4), ra._stage_status(1))
          == ("ok", "accepted", "failed"))


def test_documented_failure_contract():
    """The declared 'accepted' failures must be exactly the FAILED evidence."""
    import tools.run_wave_a as rw

    root = Path(__file__).resolve().parents[1] / "results" / "WAVE_A"
    on_disk = {p.name.split("__")[0] for p in root.iterdir()
               if p.is_dir() and (p / "FAILED.json").exists()}
    check("DOCUMENTED_FAILURES matches the FAILED evidence tree",
          on_disk == set(rw.DOCUMENTED_FAILURES),
          f"disk={sorted(on_disk)} declared={sorted(rw.DOCUMENTED_FAILURES)}")
    check("documented-failure exit code is 4 (distinct from 0 and 1)",
          rw.DOCUMENTED_FAILURE_EXIT == 4)


# ---------------------------------------------------------------------------
# H. dashboard census + units
# ---------------------------------------------------------------------------

def test_dashboard_census_and_units():
    import tools.build_dashboard as bd
    from tools.run_wave_a import wave_cells

    old_root = bd.ROOT
    try:
        with tempfile.TemporaryDirectory() as td:
            bd.ROOT = Path(td)
            cells = wave_cells()

            d0 = bd.ROOT / "WAVE_A" / cells[0].run_dirname()
            d0.mkdir(parents=True)
            (d0 / "DONE.json").write_text("{}")
            (d0 / "FAILED.json").write_text("{}")      # contradiction

            d1 = bd.ROOT / "WAVE_A" / cells[1].run_dirname()
            d1.mkdir(parents=True)
            (d1 / "FAILED.json").write_text("{}")
            (d1 / "RUNNING.json").write_text("{}")

            d2 = bd.ROOT / "WAVE_A" / cells[2].run_dirname()
            d2.mkdir(parents=True)
            (d2 / "RUNNING.json").write_text("{}")

            counts = bd._wave_census_readonly()
            check("dashboard census done precedence",
                  counts["done"] == 1, str(counts))
            check("dashboard census failed precedence",
                  counts["failed"] == 1, str(counts))
            check("dashboard census running",
                  counts["running"] == 1, str(counts))
            check("dashboard census missing",
                  counts["missing"] == len(cells) - 3, str(counts))
            check("dashboard total == declared universe",
                  sum(counts.values()) == len(cells), str(counts))

            check("wall minutes are not labelled as hours",
                  bd._wall_min_label(2537.4) == "2537.4 min (42.3 h)",
                  bd._wall_min_label(2537.4))
    finally:
        bd.ROOT = old_root


if __name__ == "__main__":
    print("=" * 72)
    print("WAVE-A SAFE-REPAIR TOOLING TESTS (no training / no evaluation)")
    print("=" * 72)
    test_eval_universe()
    test_retry_contract()
    test_marker_exclusivity()
    test_provenance_fields()
    test_finalize_denominator()
    test_evaluate_loads_each_arm_checkpoint()
    test_m2_gap_accounting()
    test_run_all_stall_detector()
    test_documented_failure_contract()
    test_dashboard_census_and_units()
    n_pass, n_all = sum(SUMMARY), len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)
