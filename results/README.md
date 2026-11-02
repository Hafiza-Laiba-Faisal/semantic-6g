# results/ — artifact map

Every number quoted in the thesis comes from a JSON file under this directory.
Directories are additive: a new experiment gets a new root, and no frozen
artifact is ever overwritten.

| Directory | What it holds | Frozen? | Amendment |
|---|---|---|---|
| `M1/` | Low-bandwidth study, k = 1, 2, 3, 3 seeds, E = 5000 | **Yes (Step 9C)** | — |
| `M2/` | Matched-budget three-way comparison, k = 12–54 | **Yes (Step 9C)** | A-1 (conditional accept) |
| `M3/` | Digital infeasibility display (k = 1, 2, 3) | Yes | — |
| `M4/` | B = 8 reference at k = 54, E = 5000 | Yes (secondary) | — |
| `M5/` | M5a λ-term removal, M5b uncoded QPSK, M5c cross-channel | No | A-16, A-18 |
| `STEP_9D_PILOT/` | Production-scale pilots, explicitly non-scientific | No | — |
| `WAVE_A/` | λ and objective ablations, k = 1/12 | No | A-2, A-3 |
| `WAVE_B/` | CSI estimation-error ablation, 350 cells | No | A-4, A-5 |
| `WAVE_C/` | λ-shaping across bandwidth, k = 18–54 | No | A-8, A-8b |
| `ARCH_ABL/` | Capacity ablation, width 256 (negative result) | No | A-9 |
| `K24_SEEDS/` | Seed variance at k = 24 | No | A-14 |
| `AERIAL/` | 3GPP UMa-aerial link budget | No | A-10 |
| `AERIAL_GEOM/` | 3GPP multi-geometry sweep, 12 geometries | No | A-17 |
| `AERIAL_TRAIN/` | 3GPP dispersion training (diverges — kept negative) | No | A-12 |
| `AERIAL_TRAIN_TRUNC/` | 3GPP dispersion, outage tail resampled | No | A-13 |
| `POLAR/` | CRC-8 polar vs convolutional comparator | No | A-11 |
| `POLAR5G/` | Full 5G NR polar + CA-SCL comparator | No | A-15 |
| `DOPPLER/` | Clarke/Jakes time-correlated Rayleigh sweep + follow-up arm | No | A-19 |
| `LDPC/` | 5G NR LDPC BG2 comparator (no rate-matching chain) | No | A-20 |
| `MECHANISM/` | Gradient-conditioning analysis of stored runs | No | analysis only |
| `RUN_ALL/` | Console logs of the unattended full runs | No | — |
| `benchmark/`, `preflight/` | Timing and readiness evidence (non-scientific) | No | — |
| `checkpoints/`, `*.pth` | Local model weights (gitignored) | — | — |
| `_archive/` | Superseded artifact snapshots | — | — |
| `_evidence/` | Byte-level evidence snapshots referenced by amendments | — | — |

## Conventions

- One run = one directory named `<experiment>_<arm>_<method>_<channel>_k<k>_seed<seed>__<hash>`.
- Marker files: `DONE.json`, `FAILED.json`, `RUNNING.json`. `RUNNING.json` is a start
  marker, not a liveness probe.
- Every run writes `train_report.json`; evaluation writes one JSON per (unit, SNR).
- `results/M5/`, `results/AERIAL_GEOM/`, `results/DOPPLER/` and `results/LDPC/`
  exist and are authoritative for their own amendments (A-16, A-17, A-19, A-20).
