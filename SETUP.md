# Setup Guide

Task-Oriented DeepJSCC for UAV Navigation — simulation-only, Python + PyTorch.

## Requirements

| Component | Frozen (thesis results) | Notes |
|---|---|---|
| OS | Windows 10/11 (tested on Win10) | Any OS works in principle; paths in docs are Windows |
| Python | **3.9.7** | Thesis artifacts were produced on this exact version |
| PyTorch | **2.4.1 + CPU** | `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu` |
| NumPy | (bundled/pinned) | |
| Matplotlib | **3.9.4** | figures only |
| Hardware | CPU-only, ~8 GB RAM, ~1 GB disk | 12 logical cores recommended (6 torch threads/lane) |

> **Reproducibility warning:** the frozen 9C results (M1–M4 artifacts in
> `results/`) are bitwise-tied to Python 3.9.7 + torch 2.4.1+CPU. A newer
> Python (3.10–3.14, torch ≥ 2.9) runs the code fine but is **not**
> guaranteed to reproduce bitwise-identical numbers. Use the pinned
> environment to re-derive thesis numbers.

## Install (pinned environment)

```bat
git clone https://github.com/Hafiza-Laiba-Faisal/semantic-6g.git
cd semantic-6g

:: any Python 3.9.x will do; the frozen runs used 3.9.7
py -3.9 -m venv .venv
.venv\Scripts\activate

pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib==3.9.4
```

## Verify the installation

```bat
.venv\Scripts\python.exe -m tests.test_channels          :: channel math gates
.venv\Scripts\python.exe -m tests.test_digital           :: framing/modem gates
.venv\Scripts\python.exe -m tests.test_evaluation        :: metrics/statistics gates
.venv\Scripts\python.exe -m tests.test_mcnemar_corrected :: corrected McNemar (5/5)
.venv\Scripts\python.exe -m tests.test_experiments       :: frozen matrix gates
```

All gates print `PASS` lines and a `SUMMARY: n/n checks passed` footer.

## Quick tour (smoke runs, no artifacts harmed)

```bat
:: smoke train + eval at the proposal's bandwidth ratios (k = 1, 2, 3)
.venv\Scripts\python.exe main.py smoke M1 task awgn 1 42
.venv\Scripts\python.exe main.py smoke M2 task rayleigh 54 42
```

## Experiment executors (frozen 9C protocol, resumable)

```bat
:: M1 — task vs reconstruction, k {1,2,3}, 11 SNRs
.venv\Scripts\python.exe tools\run_step10_m1.py status
.venv\Scripts\python.exe tools\run_step10_m1.py train all

:: M2 — matched-budget 3-way comparison, k {12,18,24,30,54}
.venv\Scripts\python.exe tools\run_step11_m2.py status
.venv\Scripts\python.exe tools\run_step11_m2.py train k30
.venv\Scripts\python.exe tools\run_step11_m2.py evaluate k30
.venv\Scripts\python.exe tools\run_step11_m2.py validate
.venv\Scripts\python.exe tools\run_step11_m2.py finalize

:: M3 + M4 — structural display + secondary digital reference
.venv\Scripts\python.exe tools\run_step12_m3_m4.py
```

Every run is identity-addressed: `results/<EXP>/<run_id>__<config_sha256>/`
with `DONE/RUNNING/FAILED` marker JSONs. Completed runs SKIP on re-invocation;
interrupted/failed runs retrain deterministically (frozen 9C policy).

## Analysis & figures

```bat
.venv\Scripts\python.exe tools\analyze_m1.py     :: tables -> results/M1/m1_analysis_tables.json
.venv\Scripts\python.exe tools\make_figures.py   :: 5 thesis PNGs -> fyp_md_files/figures/
.venv\Scripts\python.exe diagnostics\matched_mse_diagnostic.py --snr_db 0.0
```

## Repository layout

```
project/            core library (see ARCHITECTURE.md)
tools/              experiment executors + watcher + analysis
tests/              per-layer gate tests (channels, digital, eval, matrix)
diagnostics/        read-only diagnostic scripts
results/            experiment artifacts (committed as recovery backup)
fyp_md_files/       reports, status docs, figures
```

## Troubleshooting

- **`torch.load` fails on a checkpoint** → you are on a different torch
  version than the frozen one; reinstall `torch==2.4.1+cpu`.
- **Eval gate prints `REFUSED: <run>: not complete`** → a cell in that lane
  is not `done`; check `results/M2/PROTOCOL_EXCEPTIONS.md` for the two
  documented exceptions.
- **Slow runs** → the executors size torch threads per lane; don't launch
  more than ~4 lanes on a 12-core machine.
