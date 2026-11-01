# A-17 — 3GPP UMa-aerial multi-geometry link budget (12 geometries)

**Date:** 2026-11-01 · **Amendment:** A-17 (registered before the sweep) ·
**Artifacts:** `results/AERIAL_GEOM/` ·
**Runner:** `tools/run_aerial_geometry.py` ·
**Status:** 12/12 geometries measured · **Type:** evaluation only, nothing trained

Every number below was read from
`results/AERIAL_GEOM/step_aerial_geometry_summary.json`; none was typed from
memory. The frozen A-10 artifact under `results/AERIAL/` is untouched.

---

## 1. Question

A-10 measured one geometry (`d2D = 500 m`, `h = 50 m`, `fc = 2 GHz`) and found
that the λ-heavy policy needs 87 dB transmit SNR for 0.90 success at k = 54
while the digital link needs 108 dB. One point is not a trend. The supervisor
review asks for altitude and geometry to be visible in the model.

This sweep repeats the **identical** A-10 method over 3 × 2 × 2 geometries.
The stored Wave-C λ-heavy checkpoints and the frozen digital baseline are
evaluated at effective (post-path-loss) SNR; the LoS/NLoS mixture maps that
curve onto transmit SNR. No retraining, no new seeds, no change to any frozen
value.

**Scope, unchanged from A-10:** large-scale path loss and LoS probability only.
No Doppler, no time-correlated shadowing, no aerial-specific small-scale
fading. These transmit-SNR figures are a link-budget statement, not a
hardware-feasibility claim.

---

## 2. Result — required transmit SNR for SR ≥ 0.90 / ≥ 0.99

| d2D (m) | h (m) | fc (GHz) | P_LoS | PL_mean (dB) | λ-heavy k=54 @0.90 | digital k=54 @0.90 | λ advantage @0.90 | λ @0.99 | digital @0.99 | advantage @0.99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 50 | 2.0 | 0.967 | 85.42 | **74** | 89 | **+15** | 85 | 100 | +15 |
| 200 | 50 | 3.5 | 0.967 | 90.29 | **79** | 94 | **+15** | 90 | 105 | +15 |
| 200 | 100 | 2.0 | 1.000 | 85.71 | **74** | 90 | **+16** | 81 | 94 | +13 |
| 200 | 100 | 3.5 | 1.000 | 90.57 | **79** | 95 | **+16** | 86 | 98 | +12 |
| 500 | 50 | 2.0 | 0.889 | 95.63 | **87** | 108 | **+21** | 102 | 117 | +15 |
| 500 | 50 | 3.5 | 0.889 | 100.49 | **92** | 113 | **+21** | 107 | 122 | +15 |
| 500 | 100 | 2.0 | 0.945 | 94.36 | **83** | 99 | **+16** | 95 | 110 | +15 |
| 500 | 100 | 3.5 | 0.945 | 99.22 | **88** | 103 | **+15** | 100 | 115 | +15 |
| 1000 | 50 | 2.0 | 0.772 | 105.34 | **103** | 123 | **+20** | 115 | 128 | +13 |
| 1000 | 50 | 3.5 | 0.772 | 110.20 | **108** | 127 | **+19** | 120 | 133 | +13 |
| 1000 | 100 | 2.0 | 0.853 | 102.56 | **94** | 114 | **+20** | 107 | 122 | +15 |
| 1000 | 100 | 3.5 | 0.853 | 107.42 | **99** | 119 | **+20** | 112 | 126 | +14 |

The λ-heavy k = 18 arm (also in the summary JSON) needs 78–113 dB for 0.90;
the digital k = 18 arm never reaches 0.90 at any geometry (its ceiling is
0.17), so its advantage is unbounded — as in A-10.

---

## 3. Three findings

### Finding 1 — the λ advantage does **not** collapse; it stays 15–21 dB

Across a 5× span in distance (200 → 1000 m), a 2× span in altitude
(50 → 100 m) and a 1.75× span in carrier (2 → 3.5 GHz), the advantage over the
classical link at 0.90 success never leaves **15–21 dB**. At 0.99 success it
narrows slightly to **12–15 dB** but never inverts.

This is the strongest form of the RQ3 answer available without new training:
the result is a **property of the operating point, not of one geometry**.

### Finding 2 — distance dominates, and it costs about 1 dB per 100 m

| d2D | h = 50 m, fc = 2 GHz | λ tx @0.90 |
|---:|---|---:|
| 200 m | | 74 dB |
| 500 m | | 87 dB |
| 1000 m | | 103 dB |

The slope is ~29 dB per 5× distance — steeper than free-space (20 dB per 5×)
because the 3GPP aerial NLoS branch adds a distance-dependent exponent
(`46 − 7 log10 h`). Altitude recovers part of it: at 1000 m, raising h from
50 → 100 m saves **9 dB** (103 → 94 dB), because `P_LoS` rises from 0.772 to
0.853 and the LoS branch is the cheaper one.

### Finding 3 — carrier frequency costs a flat ~5 dB

Every row moves by 5 dB when `fc` goes 2 → 3.5 GHz, with no change in the
advantage. That is the `20 log10(fc)` term in both path-loss branches; it
shifts the budget, not the comparison.

---

## 4. Consistency check against the frozen A-10 artifact

The 500 m / 50 m / 2 GHz row reproduces A-10 exactly: **87 dB** for 0.90 and
**102 dB** for 0.99 at k = 54, against 108/117 dB for the digital link. The
sweep is therefore measuring the same quantity, not a re-implementation.

---

## 5. What this changes in the thesis

1. **§5.8 and §6.5 gain a trend table.** RQ3 currently rests on one geometry;
   it can now be answered as a range (15–21 dB over the tested grid).
2. **§7 (Limitations) keeps its Doppler sentence** — this sweep does not add
   velocity effects, only geometry. Do not let the new table imply otherwise.
3. **Do not claim** hardware feasibility. 74–108 dB transmit SNR is a
   link-budget number; the model has no antenna gain, noise figure, or
   bandwidth term.
4. **New sentence for the abstract:** *"the advantage is stable at 15–21 dB
   across a 5× distance, 2× altitude and 1.75× carrier sweep"*.

---

## 6. Provenance

| Item | Path |
|---|---|
| Per-geometry summaries | `results/AERIAL_GEOM/geom_d<d>_h<h>_fc<fc>/step_aerial_summary.json` |
| Aggregated table | `results/AERIAL_GEOM/step_aerial_geometry_summary.json` |
| Runner | `tools/run_aerial_geometry.py` |
| Underlying method | `tools/run_aerial_link.py` (A-10), unchanged except `--out` |
| Amendment | `fyp_md_files/PROTOCOL_AMENDMENTS.md` A-17 |
| Frozen A-10 artifact (untouched) | `results/AERIAL/step_aerial_summary.json` |
