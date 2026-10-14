# Research-standards audit — does this project meet the field's reporting standards?

Scope: `project/evaluation/*`, `project/config.py`, `project/baselines/*`,
`tools/run_step1*.py`, `tools/run_wave_a.py`, `tools/run_wave_b.py`,
frozen eval JSONs (`results/<EXP>/eval/**`). Literature comparison sources:
Bourtsoulatze et al. 2019 (arXiv:1809.01733), Kurka & Gündüz DeepJSCC-ff
(arXiv:1911.11174), Pillet et al. on 5G NR polar (arXiv:1907.00784),
NeurIPS reproducibility checklist.

---

## 1. Verdict

**Yes — statistically this project is above the published bar for deep-JSCC
evaluation, and pre-registration discipline is close to NeurIPS-checklist
level. Three gaps remain that a reviewer/supervisor could target: the polar
baseline uses SC (not CRC-aided SCL, the accepted 5G NR reference decoder),
the Rayleigh assumptions (perfect CSI + ZF + packet-granularity i.i.d.
fading) are implemented cleanly but never stated as idealized disclosures in
one place, and the pooled Wilson CI slightly over-counts independence
because all three training seeds share `TEST_SEED = 10042`.**

## 2. Where the project exceeds common practice

| Dimension | Field norm (published) | This project |
|---|---|---|
| Test volume | Bourtsoulatze: 10k CIFAR images ×10 transmissions per point, **no CI, no seeds** | **5,000 paired episodes** per (k, channel, SNR, seed) × 3 seeds, E=5000 frozen |
| Uncertainty | none in the JSCC papers | **Wilson 95% CI** per cell (`wilson_*` fields in every eval JSON) |
| Significance | none | **McNemar** (paired binary) + **paired t** (continuous), pairing preserved end-to-end |
| Seeds | rarely reported | 3 training seeds (42/43/44), pooled with per-cell counts |
| Baseline fairness | separate coding + capacity bound | **matched-budget digital** (6B+6 → exactly fits k, asserted at import), explicit `infeasible` status at k∈{1,2,3} instead of silent drops, oracle ceiling |
| Protocol discipline | loose | **frozen config with provenance tags**, fixed-step budgets (no early-stop), final-checkpoint-only, verify gates that halt the queue, M5 deferred as "spec incomplete" rather than improvised |
| Task metrics | control-cost / stability norms of wireless-NCS work | goal-reaching SR, final/avg/min distance, control effort, TTG, exited flag — success defined once, in `metrics.py` |
| RNG hygiene | ad-hoc | purpose-separated streams (`RNG_KEYS`), `derive_generator`, `validate()` asserts, deterministic CPU runs |
| Implementation gates | uncommon | bitwise-verified polar pipeline (clean roundtrip exact, GF(2) inversion, unit-vector convention gate) |

The paired-evaluation contract in `evaluate.py`
(pre-generated episodes + channel realizations handed to every method, shared
controller/dynamics outside any method) is *stronger* than what the cited
deep-JSCC papers do — reviewers should be shown this explicitly.

## 3. Gaps against standards (ranked)

### 3.1 Polar baseline: SC vs CRC-aided SCL (highest risk)
`project/baselines/digital_polar.py` implements CRC-8-aided **SC** decoding.
5G NR practice/research (Pillet et al.) shows finite-length error-rate
performance depends materially on the list size of **CRC-aided SCL**; SC is
the low-complexity ablation, not the practical reference. Since the headline
claim is reconstruction/task > digital at matched budget, a weak digital
decoder is the first objection a referee raises. Watch correlates: Wave-B
BER curves at low SNR will differ from published polar curves. Remedies:
(a) add SCL (L=8, e.g. sorting-free log-domain path metric or per-candidate
CRC) — eval-only, so cost is a one-time implementation; or (b) keep SC but
label every table "SC (low-complexity), not the 5G NR SCL reference" and add
one L=8 SCL point at the operating SNRs to show the gap.

### 3.2 Rayleigh assumption disclosure
`evaluate.py` generates **h[E,T]** — one gain per control step, i.i.d.
across steps, shared across the k symbols of that packet (packet-granularity
block fading), decoded with perfect CSI + ZF. Each choice is defensible —
Deep-JSCC itself uses block fading (one gain per transmission), and
Receiver-CSI-coherent reception is a standard idealization — but the
combination must be **stated in the thesis as idealized**: (i) perfect
receiver CSI, no channel estimation error; (ii) ZF's noise-enhancement
choice vs matched-filter/MMSE; (iii) fading coherence equal to exactly one
control packet, faster/slower than a real UAV channel at real Doppler;
(iv) no CSI at transmitter (true here). A short "channel model and its
idealisations" table satisfies this.

### 3.3 Pooled CI semantics (all 3 seeds share TEST_SEED=10042)
Pooling 3×5000 episodes into the Wilson interval assumes 15,000 independent
binomial trials, but the three seed rows are the *same 5,000 initial states
and noise realizations* re-evaluated under independently trained networks.
The interval is therefore mildly optimistic. Both fixes are cheap: report
per-seed mean ± std across seeds next to the pooled CI (and say the pooling
is over *training replicas*, not fresh episodes), and rely on the
within-seed McNemar tests for significance (already the stronger, correctly
paired test). Note this is Common-Random-Numbers, which is a *good* design —
it just changes what the CI may claim.

### 3.4 Determinism guard is implicit (minor)
"Runs are deterministic by seed+config" relies on CPU-only execution and
fixed budgets, but nothing enforces it — no `torch.use_deterministic_algorithms(True)`
and no cudnn guard. On this CPU setup it is fine. If the second PC (or
`run_all.py`) ever picks up CUDA, checkpoint-identity hashing could change
across runs and DONE-marker idempotence would silently weaken. Add two guard lines:
`torch.use_deterministic_algorithms(True)` when cuda is available, and log
`torch.backends.cudnn.deterministic/benchmark` into each train report.

### 3.5 Multiplicity (minor, disclosure only)
Dozens of paired tests per summary with no multiplicity framing. At the p
magnitudes observed (1e-174, 1e-89…) this is cosmetic, but the thesis should
either note Holm/Bonferroni or state pairs are pre-specified (they are —
frozen matrix), which is the cleaner defense.

## 4. Things checked and found correct

- Wilson closed form (`statistics.py`) matches the standard formula, stable
  edge handling; McNemar continuity correction (|d|−1)²/(n01+n10), and the
  n01==n10 tie case handled explicitly — textbook-correct.
- Paired t returns per-episode differences only; dof and two-sided p via the
  continued-fraction incomplete beta — numerically adequate at these tails.
- `digital_packet` arithmetic asserts zero padding at every matched k
  (`validate()` fails loudly otherwise) — the baseline can't be run past a
  mismatched budget silently.
- Eval JSONs are self-describing with `paired: true`, `test_seed`, per-method
  aggregates — meets a reproducibility-audit reading of the artifacts.
- Metrics definition located in one file (`metrics.py`), applied identically
  to all methods.

## 5. Recommended action list

1. Add CRC-aided **SCL (L=8)** to `digital_polar.py` and one comparison row
   per frozen cell family at the same conditions (eval-only; ~1 day).
2. Add the "channel model & idealizations" disclosure block to the thesis
   (and mirror it in the dashboard note).
3. From `results/`, generate per-seed mean±std tables alongside pooled CIs.
4. Add the two-line determinism guard for CUDA.
5. One multiplicity sentence in the statistics section.
