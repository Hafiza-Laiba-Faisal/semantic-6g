"""Build project_all_in_one.py from the frozen repository sources.

This script performs only mechanical, behavior-preserving surgery:
  * removes `from __future__ ...` lines (re-added once at the top)
  * removes all `project.*` imports (paren-aware, any indentation)
  * strips `if __name__ == "__main__":` blocks from flat modules
  * wraps tools/ scripts into factory functions, converting their
    __main__ dispatch into a nested `_dispatch()` (behavior preserved)
  * emits the `config` module-style namespace (keeps the original
    `from project import config` / `config.X` attribute contract)
  * emits `dynamics_step` and `EvalCondition` aliases exactly as the
    original import aliases
  * audits duplicate top-level names across flat modules

No algorithm, protocol, hyperparameter, loss, channel convention or
statistic is modified.  The original project/ and tools/ files remain
the source of truth.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (section, [ (source_path, mode) ])  mode: "flat" | "factory"
SECTIONS = [
    ("1  FROZEN CONFIGURATION", [
        ("project/config.py", "flat+configns"),
    ]),
    ("2  STATE / NORMALIZATION", [
        ("project/data/normalization.py", "flat"),
    ]),
    ("3  CHANNELS (complex utilities, power normalization, AWGN, Rayleigh + ZF)", [
        ("project/channels/complex_utils.py", "flat"),
        ("project/channels/awgn.py", "flat"),
        ("project/channels/rayleigh.py", "flat"),
    ]),
    ("4  DEEPJSCC MODELS (task-oriented AND reconstruction-oriented share this "
     "exact architecture: ReconstructionDeepJSCC container)", [
        ("project/models/deepjscc_encoder.py", "flat"),
        ("project/models/deepjscc_decoder.py", "flat"),
        ("project/models/reconstruction_jscc.py", "flat"),
    ]),
    ("5  CONVENTIONAL DIGITAL BASELINE (quantizer, conv K=7 Rc=1/2, soft Viterbi, QPSK Gray)", [
        ("project/baselines/digital.py", "flat"),
    ]),
    ("6  UAV NAVIGATION (dynamics, controller, environment)", [
        ("project/uav/dynamics.py", "flat"),
        ("project/uav/controller.py", "flat"),
        ("project/uav/environment.py", "flat"),
    ]),
    ("7  LOSS FUNCTIONS (task-oriented + reconstruction)", [
        ("project/losses/task_loss.py", "flat"),
        ("project/losses/reconstruction_loss.py", "flat"),
    ]),
    ("8  TRAINING (common utilities, task-oriented, reconstruction)", [
        ("project/training/common.py", "flat"),
        ("project/training/train_task_oriented.py", "flat"),
        ("project/training/train_reconstruction.py", "flat"),
    ]),
    ("9  EVALUATION + STATISTICAL TESTS (paired harness, metrics, Wilson CI, "
     "corrected McNemar, paired t)", [
        ("project/evaluation/metrics.py", "flat"),
        ("project/evaluation/statistics.py", "flat"),
        ("project/evaluation/evaluate.py", "flat"),
    ]),
    ("10 FROZEN EXPERIMENT MATRICES (M1-M4 grids, seeds, SNRs, k values, "
     "digital budgets, identity/recovery, matrix validator)", [
        ("project/experiments/config.py", "flat"),
        ("project/experiments/identity.py", "flat"),
        ("project/experiments/matrix.py", "flat"),
    ]),
    ("11 M1 EXPERIMENT DRIVER (recon vs task, k in {1,2,3})", [
        ("tools/run_step10_m1.py", "factory:_run_step10_m1_main"),
    ]),
    ("12 M2 EXPERIMENT DRIVER (3-way matched budgets, k in {12,18,24,30,54})", [
        ("tools/run_step11_m2.py", "factory:_run_step11_m2_main"),
    ]),
    ("13 M3 + M4 EXPERIMENT DRIVERS (M3: structural digital feasibility at "
     "low k; M4: SECONDARY digital reference B=8 at k=54)", [
        ("tools/run_step12_m3_m4.py", "factory:_run_step12_m3_m4_main"),
    ]),
    ("14 FIGURE/TABLE GENERATION (thesis figures fig1-fig15; read-only over results/)", [
        ("tools/make_figures.py", "factoryonly:_figures_step14b"),
        ("tools/make_figures_extra.py", "factoryonly:_figures_step14c"),
        ("tools/make_figures_semantic.py", "factoryonly:_figures_step14d"),
    ]),
]

RE_FUTURE = re.compile(r"^\s*from\s+__future__\s+import\b")
RE_PROJ = re.compile(r"^\s*from\s+project(\.[\w.]+)?\s+import\b")
RE_PROJ_IMP = re.compile(r"^\s*import\s+project(\.[\w.]+)?\s*$")
RE_MAIN = re.compile(r"^(\s*)if\s+__name__\s*==\s*['\"]__main__['\"]\s*:")


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n")


def strip_lines(text: str):
    """Remove __future__ and project.* imports (paren-balance aware, any indent)."""
    out = []
    skipping = False
    bal = 0
    for line in text.split("\n"):
        if skipping:
            bal += line.count("(") - line.count(")")
            if bal <= 0:
                skipping = False
            continue
        if RE_FUTURE.match(line) or RE_PROJ.match(line) or RE_PROJ_IMP.match(line):
            bal = line.count("(") - line.count(")")
            if bal > 0:
                skipping = True
            continue
        out.append(line)
    return out


def remove_main_block(lines):
    """Drop `if __name__ == "__main__":` block (module level)."""
    out, i = [], 0
    while i < len(lines):
        m = RE_MAIN.match(lines[i])
        if m and m.group(1) == "":
            indent = m.group(1)
            i += 1
            while i < len(lines):
                s = lines[i]
                if s.strip() == "":
                    i += 1
                    continue
                cur = s[:len(s) - len(s.lstrip())]
                if len(cur) <= len(indent):
                    break
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def indent(lines, n=4):
    pad = " " * n
    return [pad + s if s.strip() else s for s in lines]


def factory(path: str, name: str, has_main: bool):
    """Wrap a tools/ script body into `def name():` returning its dispatch."""
    text = read(path)
    doc = ast.get_docstring(ast.parse(text)) or ""
    lines = strip_lines(text)
    lines = remove_main_block(lines) if not has_main else lines
    body = indent(lines)
    header = [f"def {name}():",
              f'    __doc__ = {doc!r}']
    if has_main:
        # convert the __main__ block into a nested _dispatch()
        conv = []
        for ln in lines:
            m = RE_MAIN.match(ln)
            conv.append(RE_MAIN.sub(lambda m: m.group(1) + "def _dispatch():", ln)
                        if m else ln)
        body = indent(conv)
        footer = ["    return _dispatch", f"{name} = {name}()"]
    else:
        footer = []
    return header + body + footer


def config_namespace(src: str) -> list:
    """Emit `config = SimpleNamespace(...)` over config.py's top-level names."""
    tree = ast.parse(src)
    names = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
                elif isinstance(t, ast.Tuple):
                    names += [e.id for e in t.elts if isinstance(e, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append(node.name)
    lines = [
        "",
        "# Module-style namespace preserving the original",
        "# `from project import config` / `config.X` attribute contract",
        "# used verbatim by the executor drivers and evaluation harness.",
        "config = SimpleNamespace(**{",
    ]
    for n in sorted(set(names)):
        lines.append(f'    "{n}": {n},')
    lines += ["})", ""]
    return lines


def top_level_names(src: str):
    tree = ast.parse(src)
    names = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names[t.id] = ast.dump(node.value)[:80]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names[node.target.id] = ast.dump(node.value or node.annotation)[:80]
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names[node.name] = "<def>"
    return names


def build():
    out = []
    header = '''"""Consolidated reproducibility source for the FYP
"Task-Oriented Deep Joint Source-Channel Coding for UAV Navigation under
Noisy and Fading Wireless Channels".

This file is a consolidated reproducibility source for the FYP.
It consolidates the paper-relevant implementation from the frozen
project repository.

It is NOT the canonical development source.
The original project/ and tools/ files remain the source of truth.

This file does not contain or regenerate historical experiment results.
(results/M1..M4 artifacts are read-only inputs; nothing here overwrites
them.  Default execution runs ONLY the minimal deterministic smoke test.)

Usage:
    python project_all_in_one.py                 # smoke test (default)
    python project_all_in_one.py --smoke-test    # same
    python project_all_in_one.py --m1 status     # original M1 executor dispatch
    python project_all_in_one.py --m1 train awgn # M1 training lane (heavy!)
    python project_all_in_one.py --m2 status     # original M2 executor dispatch
    python project_all_in_one.py --m34 validate  # original M3+M4 dispatch
    python project_all_in_one.py --figures       # regenerate fig1-fig15 PNGs

Provenance: every section banner cites the original source file.  The
repository's own docstrings preserve the [FROZEN]/[LIT]/[PROJ] distinction
(literature-derived vs standard equation vs project-specific design choice
vs experimental parameter); these comments are reproduced verbatim and no
new provenance claims are added.

SECTION INDEX
    sec1  Frozen configuration               (project/config.py)
    sec2  State / normalization              (project/data/normalization.py)
    sec3  Channels: complex utils, AWGN, Rayleigh+ZF  (project/channels/*.py)
    sec4  DeepJSCC models (encoder/decoder/ReconstructionDeepJSCC; shared by
          the task-oriented and reconstruction-oriented variants)
                                              (project/models/*.py)
    sec5  Conventional digital baseline      (project/baselines/digital.py)
    sec6  UAV navigation: dynamics, controller, environment  (project/uav/*.py)
    sec7  Loss functions                     (project/losses/*.py)
    sec8  Training                           (project/training/*.py)
    sec9  Evaluation + statistics            (project/evaluation/*.py)
    sec10 Frozen experiment matrices          (project/experiments/*.py)
    sec11 M1 experiment driver                (tools/run_step10_m1.py)
    sec12 M2 experiment driver                (tools/run_step11_m2.py)
    sec13 M3 + M4 experiment drivers          (tools/run_step12_m3_m4.py)
    sec14 Figure/table generation             (tools/make_figures*.py)
    sec15 Main / smoke test                   (this build)

M1-M4 are EXPERIMENT CONFIGURATIONS over the same underlying
implementation -- not separate algorithms.
"""'''
    out.append(header)
    out.append("from __future__ import annotations")
    out.append("from types import SimpleNamespace")
    out.append("import sys")
    out.append("")
    seen = {}
    for title, files in SECTIONS:
        out.append("")
        out.append("# " + "=" * 60)
        out.append(f"# SECTION {title}")
        out.append("# " + "=" * 60)
        for path, mode in files:
            src = read(path)
            out.append(f"# ---- Source: {path} ----")
            if mode == "flat+configns":
                lines = remove_main_block(strip_lines(src))
                out += lines
                out += config_namespace(src)
                for n, sig in top_level_names(src).items():
                    seen.setdefault(n, []).append((path, sig))
            elif mode == "flat":
                lines = remove_main_block(strip_lines(src))
                out += lines
                for n, sig in top_level_names(src).items():
                    seen.setdefault(n, []).append((path, sig))
            elif mode.startswith("factory:"):
                name = mode.split(":", 1)[1]
                out += factory(path, name, has_main=True)
            elif mode.startswith("factoryonly:"):
                name = mode.split(":", 1)[1]
                out += factory(path, name, has_main=False)
        if title.startswith("6 "):
            out.append("")
            out.append("# Alias preserving the original "
                       "`from project.uav.dynamics import step as dynamics_step`")
            out.append("dynamics_step = step")
        if title.startswith("9 "):
            out.append("")
            out.append("# Alias preserving the original "
                       "`from project.evaluation.evaluate import Condition as "
                       "EvalCondition` (project/experiments/config.py)")
            out.append("EvalCondition = Condition")
    out.append("")
    out.append("# " + "=" * 60)
    out.append("# SECTION 15 MAIN / SMOKE TEST")
    out.append("# " + "=" * 60)
    out.append(read("tools/_aio_smoke.py"))
    out.append(MAIN_CLI.strip())
    text = "\n".join(x for x in out if x is not None) + "\n"
    (ROOT / "project_all_in_one.py").write_text(text, encoding="utf-8")
    dups = {n: v for n, v in seen.items() if len(v) > 1}
    print(f"wrote project_all_in_one.py ({len(text.splitlines())} lines)")
    if dups:
        print("DUPLICATE TOP-LEVEL NAMES (flat modules):")
        for n, v in sorted(dups.items()):
            sigs = {s for _, s in v}
            flag = "SAME" if len(sigs) == 1 else "DIFFERENT!!"
            print(f"  {n}: {[p for p, _ in v]} -> {flag}")
    else:
        print("no duplicate top-level names")


MAIN_CLI = '''

def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Consolidated FYP source; default = smoke test only")
    ap.add_argument("--smoke-test", action="store_true",
                    help="(default) minimal deterministic verification")
    ap.add_argument("--m1", nargs="+", metavar=("CMD", "LANE"),
                    help="original M1 executor dispatch, e.g. --m1 status")
    ap.add_argument("--m2", nargs="+", metavar=("CMD", "LANE"),
                    help="original M2 executor dispatch, e.g. --m2 status")
    ap.add_argument("--m34", nargs="+", metavar="CMD",
                    help="original M3+M4 dispatch, e.g. --m34 validate")
    ap.add_argument("--figures", action="store_true",
                    help="regenerate fig1-fig15 from results/ JSONs")
    args, extra = ap.parse_known_args(argv)
    if args.m1:
        sys.argv = ["run_step10_m1.py"] + args.m1 + extra
        return _run_step10_m1_main()
    if args.m2:
        sys.argv = ["run_step11_m2.py"] + args.m2 + extra
        return _run_step11_m2_main()
    if args.m34:
        sys.argv = ["run_step12_m3_m4.py"] + args.m34 + extra
        return _run_step12_m3_m4_main()
    if args.figures:
        _figures_step14b()
        _figures_step14c()
        _figures_step14d()
        return 0
    return _smoke_test()


if __name__ == "__main__":
    sys.exit(_main())
'''

if __name__ == "__main__":
    build()
