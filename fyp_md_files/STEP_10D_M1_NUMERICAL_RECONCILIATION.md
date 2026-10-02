# Step 10D - Independent M1 Numerical Reconciliation

**Date:** 2026-10-02  
**Scope:** Read-only audit of M1 evaluation JSONs, M1 training reports, and
the evaluation/training source paths. No training, evaluation, M2-M5 work,
or experiment-artifact changes were performed. This report and the project
status are the only documentation edits.

## Verdict

**STEP 10D BLOCKED - numerical discrepancy remains unresolved; M2 NOT STARTED**

The pooled success and continuous-metric summaries reproduce. The stored
McNemar p-values are half the correct two-sided chi-square survival
probabilities, and the large train-time versus post-training evaluation MSE
gap cannot be quantitatively reconciled from the stored JSON summaries. M2
remains unstarted.

## Evidence and Coverage

Read `results/M1/eval/`: 198 evaluation JSONs, 18 `(k, channel, training seed)`
units times 11 SNRs. Every record has 5,000 episodes, test seed 10042, and
`paired=true`, for 990,000 episode rows per method across the full matrix.
Each of the 132 method x k x channel x SNR pooled conditions has all three
training seeds and 15,000 episodes. The JSONs store aggregate values and
discordance counts, not per-episode outcome or MSE vectors.

There are 18 task M1 `train_report.json` files. No Step 10C report file was
present in `fyp_md_files`; the Step 10C claims supplied in the request were
checked against the underlying reports and code.

## Verified Numerical Results

### Pooled success and Wilson intervals

For every pooled cell, the recomputation was `sum(success_count) /
sum(episodes)` across seeds; all denominators are 15,000. The 132 pooled
percentages match the Step 10B tables to their displayed 0.1 percentage-point
precision. Examples at AWGN, k=1, 0 dB:

| Method | Pooled successes / episodes | Rate | Wilson 95% CI |
|---|---:|---:|---:|
| reconstruction | 2,085 / 15,000 | 13.9000% | 13.3556%-14.4629% |
| task | 262 / 15,000 | 1.7467% | 1.5490%-1.9690% |

These reproduce the displayed Step 10B intervals [13.36%, 14.46%] and
[1.55%, 1.97%]. The implementation is the Wilson score formula in
[`statistics.py`](../project/evaluation/statistics.py#L43).

### Paired McNemar results

The executor calls `mcnemar_test(task_success, reconstruction_success)`;
therefore, with A=task and B=reconstruction:

- `n01 = #(task=1, reconstruction=0)`: discordant task wins.
- `n10 = #(task=0, reconstruction=1)`: discordant task losses.

In all 198 JSON cells, `n10 - n01` equals reconstruction successes minus task
successes; `n10 > n01` and reconstruction has the larger success count in
198/198. Mean discordant pairs are 1,181.894 per cell; maximum is 1,674.
For pooled AWGN, k=1, 0 dB across seeds, `n01=216`, `n10=2,039`, and 2,255
pairs are discordant. The definitions and call order are in
[`statistics.py`](../project/evaluation/statistics.py#L83) and
[`run_step10_m1.py`](../tools/run_step10_m1.py#L406).

The continuity-corrected statistic
`chi2 = (abs(n01-n10)-1)^2 / (n01+n10)` matches all JSON cells. However,
[`statistics.py`](../project/evaluation/statistics.py#L107) computes
`0.5 * erfc(sqrt(chi2/2))`; its own docstring at line 89 specifies the
chi-square(1) survival probability `erfc(sqrt(chi2/2))`. The stored and
Step 10B p-value extrema (`7.1771e-304` to `5.1859e-27`) are consequently
half the correct two-sided chi-square p-values (`1.4354e-303` to
`1.0372e-26`). All corrected values remain far below 0.05, so the significance
and direction claims stand; the reported numerical p-values do not.

Pairing is supported by the harness: it pre-generates one condition's episode
states, noise, and (for Rayleigh) fading and passes the same realizations to
both methods. Step 10 invokes both same-seed models in the same
`evaluate_condition` call. See [`evaluate.py`](../project/evaluation/evaluate.py)
and [`run_step10_m1.py`](../tools/run_step10_m1.py#L457). The JSONs do not
contain episode-level success masks, so the discordance counts can be checked
against the stored counts and success-count difference, but cannot be
independently regenerated from per-episode outcomes.

### Continuous metrics and aggregation

All 198 cells have 5,000 episodes. The reported grand means are reproduced by
the arithmetic mean of the 198 cell means per method, which is equivalent to
pooling every episode with equal weight:

| Metric | reconstruction | task |
|---|---:|---:|
| Final distance | 3.787605 | 9.565264 |
| Average distance | 4.536429 | 9.154987 |
| Control effort | 691.162116 | 770.247189 |

The definitions are per episode: final distance is at T; average distance is
the mean of distances at t=1..T; control effort is the sum of squared control
norms over t=0..T-1. The JSON's cell-level aggregates are then equally
weighted because the episode counts are identical. Definitions and
aggregation are in [`metrics.py`](../project/evaluation/metrics.py#L31) and
[`metrics.py`](../project/evaluation/metrics.py#L98). Step 10B's rounded
values (3.79/9.57, 4.54/9.15, 691.2/770.2) agree.

### SNR trends

Spearman correlations were recomputed from the exact pooled counts at the 11
SNR values, using average ranks for ties:

| Method | Channel | k=1 | k=2 | k=3 |
|---|---|---:|---:|---:|
| reconstruction | AWGN | -0.03636 | -0.83636 | -0.94545 |
| reconstruction | Rayleigh | +0.87273 | -0.47273 | -0.85455 |
| task | AWGN | -0.36530 | -0.82727 | -0.64841 |
| task | Rayleigh | +0.84545 | +0.97495 | +0.80909 |

The Step 10B correlation summaries reproduce to their shown precision. Peak
locations need tie-aware wording: reconstruction/Rayleigh/k=3 ties at 2 and
4 dB; task/Rayleigh/k=2 ties at 18 and 20 dB. Step 10B's summarized task
Rayleigh peak range ending at 18 dB omits the 20 dB tie.

These are descriptive correlations across related conditions, not 11
independent experimental replications. The fixed test seed and deterministic
generation make the episode set and base noise/fading draws common across
SNRs; the SNR values are repeated measurements on the same trained models.
No inferential interpretation of rho or independent-sample standard error is
licensed by this grid.

## Train-Time Diagnostic Versus Evaluation MSE

### Definitions verified in source

- Training samples one scalar SNR per batch from U(0,20) dB; the 256 episodes
  in that batch share it. After the rollout, the detached diagnostic calls
  `forward_losses(s0, ...)` once on the batch's initial six-feature states
  and computes the squared error against `normalize(s0)`, summed over the
  six features, averaged over episodes, then divided by six. It is a
  normalized per-feature MSE for one batch, not the optimized task loss and
  not a trajectory average. This diagnostic is measured before that step's
  optimizer update. See [`train_task_oriented.py`](../project/training/train_task_oriented.py#L164)
  and [`train_task_oriented.py`](../project/training/train_task_oriented.py#L177).
- `forward_losses` returns normalized decoder output; normalization uses
  the frozen component-wise state bounds. See
  [`reconstruction_jscc.py`](../project/models/reconstruction_jscc.py) and
  [`normalization.py`](../project/data/normalization.py).
- Evaluation MSE sums squared normalized errors over six features and T
  recorded states, divides by `6*T`, then averages per-episode values across
  the 5,000 episodes. Each evaluation cell has one fixed SNR, unlike the
  training batch's random SNR. See [`metrics.py`](../project/evaluation/metrics.py#L83)
  and [`metrics.py`](../project/evaluation/metrics.py#L118).
- There is a measurement-window mismatch in the comments: `metrics.py`
  describes estimates at t=1..T, but `evaluate.run_method` records the state
  before each action inside `for t in range(T)`, corresponding to t=0..T-1.
  The implementation records the initial state and omits the final state.
- A second, material evaluation behavior: after workspace exit, `active` is
  false and the estimator is not called for that episode. `s_hat` defaults to
  the true `s`; therefore its later recorded MSE is exactly zero. The
  evaluation aggregate includes these zero-error suffixes. The train-time
  diagnostic has no such inactive-episode bypass. See
  [`evaluate.py`](../project/evaluation/evaluate.py#L165) and the active-index
  handling immediately below it.

Both values are normalized per-feature squared error, but their sampling,
SNR, time window, and evaluation behavior differ. They must not be read as
the same estimand.

### Direct report checks

Across the 18 task `train_report.json` files, `recon_mse_diag_initial` ranges
from 0.229233 to 0.257373, while the **final endpoint fields** range from
10.914890 to 230.272324. Thus the Step 10C range 10.9-230.3 matches the
reports' final endpoints. The files contain only initial/final scalar
diagnostics and task losses; they do not contain the per-step history array.
The runner saves that array in the checkpoint via
[`run_step10_m1.py`](../tools/run_step10_m1.py#L327) and
[`run_step10_m1.py`](../tools/run_step10_m1.py#L341), not in `train_report`.
No checkpoint history was loaded for this audit.

For AWGN, k=1, the three report rows are:

| Seed | Task loss initial -> final | `recon_mse_diag` initial -> final | Train report eval success |
|---:|---:|---:|---:|
| 42 | 4283.084961 -> 4096.872070 | 0.253165 -> 54.733109 | 2.34375% |
| 43 | 5258.258301 -> 720.775330 | 0.239644 -> 10.914890 | 0.78125% |
| 44 | 4773.522461 -> 366.984619 | 0.230479 -> 37.480534 | 1.171875% |

Seed 42's task-loss values are therefore **4283.084961 initially and
4096.872070 finally**. The `eval_after_success_rate` is a separate 256-episode,
10 dB training-report diagnostic, not the 5,000-episode Step 10 evaluation.
The seed-42 report is
[`train_report.json`](../results/M1/M1_task_awgn_k1_seed42__7a41dabe6895102d/train_report.json);
seed 43 and 44 reports are in their corresponding seed-specific M1 run
directories.

For task evaluation MSE, the k=1/AWGN/seed-42/0 dB JSON is 0.375277. Across
all 198 task evaluation cells, cell means range from 0.332265 to 0.856790;
their grand mean is 0.569319. Therefore 0.38 is a valid individual-cell
example, not the grand mean across M1.

### Plausible explanations and unresolved items

The single initial-state training snapshot at a randomly selected SNR and the
100-step fixed-SNR evaluation average are different samples and windows.
Feedback changes the state distribution, and the evaluation's zero-error
post-exit suffix can materially lower its reported MSE. These are plausible
contributors; the JSONs do not provide enough data to attribute the observed
gap or quantify each contributor. A matched-window, same-SNR comparison with
per-step active masks and episode-level errors is needed before claiming the
gap is resolved.

The full per-step training histories and episode-level evaluation outcomes
are not present in `train_report.json` or the evaluation JSONs. The checkpoint
history path exists in the runner, but checkpoint deserialization was not
available in the installed Python interpreters (PyTorch is absent); no package
was installed. The exact Step 10C document was also not present in the
workspace, so only the values quoted in the request could be cross-checked.

## Verified Versus Unresolved

- **Verified numerical results:** pooled success denominators/rates; selected
  Wilson intervals; stored McNemar discordance counts and chi-square
  statistics; corrected two-sided p-value range; pooled continuous metrics;
  Spearman correlations; AWGN k=1 report endpoints; final train-diagnostic
  range across 18 task reports.
- **Verified implementation facts:** paired harness inputs; n01/n10 argument
  order; half-survival McNemar p-value implementation; normalized MSE formulas;
  random per-batch training SNR versus fixed evaluation SNR; evaluation
  time-window mismatch; true-state bypass and zero-MSE suffix after exit.
- **Plausible interpretations:** differing state/SNR/time samples and
  post-exit passthrough contribute to the training/evaluation MSE gap.
- **Unresolved:** exact contribution of post-exit passthrough; the full
  per-step training trajectory from checkpoint history; a matched-condition
  MSE comparison; exact Step 10C prose/transcription because its source file
  is absent.

No source, configuration, checkpoint, or results files were modified. The
only edits were this audit report and `PROJECT_STATUS.md`. No M2-M5 directory
or run was started; **M2 NOT STARTED**.