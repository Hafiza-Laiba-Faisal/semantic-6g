# Local machine setup notes (this Windows machine)

Machine setup done on **2026-10-14** for the frozen thesis stack in [SETUP.md](SETUP.md).

## What was installed

| Component | Version | Where |
|---|---|---|
| Python | 3.9.13 (per-user, via winget `Python.Python.3.9`) | `%LOCALAPPDATA%\Programs\Python\Python39` |
| venv | pinned CPU environment | **`C:\Users\LENOVO\venvs\sem6g`** |
| torch | **2.4.1+cpu** (from the PyTorch CPU index) | venv site-packages |
| numpy | 1.26.4 | venv site-packages |
| matplotlib | 3.9.4 | venv site-packages |

## Why the venv is NOT inside the repo

This checkout lives at a very long path
(`...freebuff-desktop\no-folder\threads\<128-hex>\semantic-6g`).
Torch ships ~8,500 files under `Lib\site-packages\torch\include\ATen\ops\...`,
and from inside the repo the full paths exceeded Windows' **260-char MAX_PATH**
limit. The first install silently produced a broken torch
(6,712 of 10,644 files missing, `OSError: No such file or directory` during
extraction). Putting the venv at the short path above fixes it:
all 12,540 torch files extract cleanly.

## How to run anything

```bat
:: in cmd.exe, from the repo root
activate.bat                       :: activates the venv + prints versions
python main.py dry-run             :: config-only sanity, no training
python main.py smoke M1 task awgn 1 42
python main.py smoke M2 task rayleigh 54 42
python -m tests.test_channels      :: (likewise test_digital, test_evaluation,
                                     ::  test_mcnemar_corrected, test_experiments)
tools\run_step10_m1.py status      :: experiment executors
```

In PowerShell, `activate.bat` is not directly usable; use
`cmd /k "...\activate.bat"` or call the interpreter by full path:
`C:\Users\LENOVO\venvs\sem6g\Scripts\python.exe main.py dry-run`.

## Verification results (2026-10-14)

- Gate tests: **162/162 checks passed** (channels 43/43, digital 33/33,
  evaluation 18/18, mcnemar 5/5, experiments 63/63)
- `main.py dry-run`: all M1–M5 cells valid
- Smoke runs `M1 task awgn k=1 seed=42` and `M2 task rayleigh k=54 seed=42`:
  trained + evaluated, checkpoints written to `results/checkpoints/`

Smoke artifacts are non-scientific (as the repo states); frozen `results/`
artifacts were not touched.
