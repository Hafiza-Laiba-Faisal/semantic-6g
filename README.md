# semantic-6g

**Task-Oriented Deep Joint Source-Channel Coding for UAV Navigation under Noisy and Fading Wireless Channels — with a 3GPP UMa-Aerial Link-Budget and Dispersion Extension**

A matched-budget, reproducible comparison of three ways to send a UAV's 6-D kinematic state
over a wireless link:

| Arm | What it sends | How it is trained |
|---|---|---|
| **Task-oriented DeepJSCC** | learned `6 → 2k` symbols | end-to-end through the closed-loop UAV plant, on navigation success |
| **Reconstruction DeepJSCC** | learned `6 → 2k` symbols | on state reconstruction MSE |
| **Digital baseline** | quantize → channel code → QPSK | not trained (conventional) |

All three share the same episodes, the same noise and fading realizations, the same SNR, and
the same number of channel uses. The digital packet invariant `6B + 6 ≤ k` is enforced, never
relaxed.

## Headline result

- **Bandwidth crossover:** neural JSCC wins at `k ≤ 18`; the digital link wins at `k ≥ 24`.
- **The frozen task loss fails** at low bandwidth (0.025 vs 0.237 for reconstruction at k = 18),
  and it is **not** a capacity, training-budget, or metric artefact.
- **λ-shaping fixes it:** adding control and velocity penalties (`λ_u = λ_v = 0.1`) takes the
  same architecture from **0.025 to 0.999**, because the encoder gradient improves by roughly
  **fourteen orders of magnitude**.
- **3GPP UMa-aerial:** the shaped policy needs **87–91 dB** transmit SNR for 0.90 success;
  the digital link needs **108 dB** at k = 54 and never reaches 0.90 at k = 18.
- **Robustness:** time-correlated (Clarke/Jakes) fading moves the neural arms by **≤ 0.003**
  (A-19), and a **5G NR LDPC BG2** comparator does not change the crossover (A-20).

## Repository map

```
project/            the library (importable package)
  config.py           single frozen configuration + runtime assertions
  channels/           AWGN, Rayleigh, 3GPP UMa-aerial (LoS/NLoS, path loss, shadowing)
  models/             DeepJSCC encoder/decoder
  training/           reconstruction and task-oriented (closed-loop) trainers
  evaluation/         paired evaluation harness, metrics, McNemar/Wilson statistics
  baselines/          convolutional, polar, 5G NR polar, and 5G NR LDPC BG2 digital links
  losses/             task and reconstruction losses
  uav/                double-integrator dynamics, controller, environment
tools/              one runner/builder per experiment or document
tests/              17 protocol/design test modules + noiseless oracle gate
results/            per-experiment artifacts (JSON, reports); see results/README.md
fyp_md_files/       thesis source, protocol amendments, results notes; see its INDEX.md
artifacts/          generated bundles and console logs (not source)
.github/workflows/  CI: matrix validation + all tests + Docker build
```

## Quick start

```powershell
# 1. environment (Python 3.9, CPU torch)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. validate the frozen experiment matrix (no compute)
.\.venv\Scripts\python.exe main.py validate-matrix

# 3. run the test suite (111 tests, ~2 min on CPU)
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q

# 4. rebuild the thesis / figures documents
.\.venv\Scripts\python.exe tools\build_thesis_docx.py --md fyp_md_files\THESIS_MASTER_DOCUMENT.md --out THESIS_MASTER_DOCUMENT.docx --header "Task-Oriented DeepJSCC for UAV Navigation"
```

`python main.py full` is intentionally **refused**: the full matrix runs through the dedicated
runners in `tools/`, each of which is idempotent and writes only under its own `results/`
subdirectory.

## Where to read first

| Question | File |
|---|---|
| What is the project, in plain language? | `SUBMISSION_KIT/ONE_PAGER.md` (the older narrative is archived at `ARCHIVE/SUPERVISOR_BRIEFING_PLAIN.md`) |
| Full thesis source | `fyp_md_files/THESIS_MASTER_DOCUMENT.md` |
| Every protocol deviation, dated | `fyp_md_files/PROTOCOL_AMENDMENTS.md` |
| Independent IEEE-style audit + remaining work | `ARCHIVE/IEEE_EVALUATION_AND_REMAINING_WORK.md` |
| Pipeline status and frozen decisions | `ARCHIVE/PROJECT_STATUS.md` |
| Index of all 46 documents | `fyp_md_files/INDEX.md` |

## Ground rules

- Do not modify frozen mathematical conventions without a dated protocol amendment.
- Do not invent or retype numerical values; every number comes from a stored artifact.
- Negative results are kept, not retrained away (A-9 capacity, A-12 full-dispersion training, A-18 M5a cell).
- Experiments are run only under a dated amendment registered before the first cell (most recently A-19 Doppler and A-20 LDPC).

## Licence / status

Simulation-only, CPU-only research code. No hardware, no flight test, no fabricated data.
