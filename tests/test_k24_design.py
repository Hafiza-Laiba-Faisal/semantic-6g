"""A-14 design lock: six k = 24 cells, three seeds, frozen everything else.

These checks pin the pre-registered design of `tools/run_k24_seeds.py` so a
later edit cannot silently change the number of cells, the seeds, the arm, the
width, the step budget or the evaluation grid.  They do not train and do not
touch `results/`.

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_k24_design
"""

from __future__ import annotations

import sys
from pathlib import Path

SUMMARY = []
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check(name, ok, detail=""):
    SUMMARY.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]"
          + (f"  ({detail})" if detail else ""))


def _load():
    import tools.run_wave_a as rw
    saved = rw.WAVE_A_ROOT
    import tools.run_k24_seeds as k24
    rw.WAVE_A_ROOT = saved
    return rw, k24


def test_design():
    from project import config

    rw, k24 = _load()
    cells = k24.cells()

    check("6 cells", len(cells) == 6, str(len(cells)))
    check("6 units", len(k24.units()) == 6, str(len(k24.units())))
    check("30 eval slots", k24.EXPECTED_SLOTS == 30, str(k24.EXPECTED_SLOTS))
    check("k = 24 everywhere", {c.k for c in cells} == {24},
          str(sorted({c.k for c in cells})))
    check("seeds 45/46/47", {c.seed for c in cells} == {45, 46, 47},
          str(sorted({c.seed for c in cells})))
    check("channels awgn + rayleigh",
          {c.channel for c in cells} == {"awgn", "rayleigh"},
          str(sorted({c.channel for c in cells})))
    check("arm is lam_ctl_heavy",
          {c.tag for c in cells} == {"lam_ctl_heavy"},
          str(sorted({c.tag for c in cells})))
    check("lambdas are the frozen Wave-C heavy arm",
          {c.lambdas for c in cells} == {(1.0, 0.1, 0.1, 2.0)},
          str(sorted({c.lambdas for c in cells})))
    check("width is the frozen default (None)",
          {c.hidden for c in cells} == {None},
          str(config.HIDDEN_DIMS))
    check("step budget is the frozen 3000",
          {c.steps for c in cells} == {config.TASK_TRAIN_STEPS},
          str(config.TASK_TRAIN_STEPS))
    check("evaluation grid is the 5-SNR frozen grid",
          list(k24.SNR_GRID) == [0.0, 5.0, 10.0, 15.0, 20.0],
          str(k24.SNR_GRID))
    check("evaluation uses E=1000", rw.EVAL_EPISODES == 1000,
          str(rw.EVAL_EPISODES))


def test_identity():
    _rw, k24 = _load()
    cells = k24.cells()
    dirs = [c.run_dirname() for c in cells]
    check("cell directories are unique", len(set(dirs)) == 6, str(len(set(dirs))))
    check("ids carry the K24 prefix",
          all(c.wave_id.startswith("K24_") for c in cells),
          cells[0].wave_id)
    check("no leftover WAVE_A_ id prefix",
          all("WAVE_A_" not in c.wave_id for c in cells))
    check("own root results/K24_SEEDS",
          k24.K24_ROOT == ROOT / "results" / "K24_SEEDS",
          str(k24.K24_ROOT.relative_to(ROOT)))
    spec = cells[0].spec()
    check("spec records the amendment",
          spec.get("amendment") == "K24-SEEDS-2026-10-27",
          str(spec.get("amendment")))
    check("spec warns about the statistics",
          "not a confidence interval" in (spec.get("statistics_note") or ""),
          str(spec.get("statistics_note")))


def test_preregistered():
    doc = (ROOT / "fyp_md_files" / "PROTOCOL_AMENDMENTS.md").read_text(
        encoding="utf-8")
    check("A-14 section exists", "## A-14" in doc)
    a14 = doc.split("## A-14", 1)[1].split("\n## ", 1)[0] if "## A-14" in doc else ""
    check("A-14 records the seeded design", "45, 46, 47" in a14,
          "seeds 45, 46, 47 present in the A-14 section")
    check("A-14 fixes k = 24", "k | 24" in a14 or "k = 24" in a14)
    check("A-14 is dated", "2026-10-27" in a14)
    accounting = (ROOT / "fyp_md_files" / "GAP_ANALYSIS_RESPONSE.md").read_text(
        encoding="utf-8")
    check("gap doc points at A-14", "A-14" in accounting)


def main():
    test_design()
    test_identity()
    test_preregistered()
    ok = all(SUMMARY)
    print(f"\n{sum(SUMMARY)}/{len(SUMMARY)} checks passed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
