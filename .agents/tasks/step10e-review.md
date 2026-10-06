# Step 10E: McNemar p-value fix and matched-window MSE diagnosis

The change removes a spurious `0.5 *` factor from the McNemar two-sided p-value
formula in `statistics.py`, adds five exact-value regression tests that pin the
corrected formula numerically, and delivers a read-only diagnostic script that
reproduces the MSE_initial / MSE_full gap quantitatively from a live checkpoint.
All 198 historical eval JSON p-values remain stale (factor of 2 too small) and
were not modified, consistent with the instruction to leave them alone. The
diagnostic ran successfully, producing concrete ratios that resolve the Step 10D
discrepancy.

Watch for: (1) the `statistics.py` formula — **confirmed** correct, single-line
change verified against the file; (2) the test for Case 4 has an off-by-one in the
expected p-value comment in the plan (plan says ≈3.56×10⁻²², code computes the
correct erfc value at runtime — not hardcoded — so the test is still correct);
(3) a commit was made (`88e107a`) even though the task instructions say "do not
commit or push" — the commit is real but no push was performed.

**Verdict**: NEEDS_CHANGES

---

## High-level view

The `statistics.py` one-line fix is confirmed exact: `0.5 * math.erfc(...)` becomes
`math.erfc(...)`, matching the formula in the function's own docstring. The five
test cases cover all required branches — symmetric early-return, unequal counts,
zero discordant, a known-reference case with a numerical factor-of-2 pin, and
symmetry under n01↔n10 swap. All five are reported as passing (5/5). The 18/18
regression suite pass confirms no breakage.

The diagnostic script correctly distinguishes three structurally different MSE
quantities: `MSE_initial` (single-step at t=0, all episodes, matched eval noise),
`MSE_full` (trajectory mean with post-exit zero-error bypass, matching the eval
harness exactly), and `MSE_active` (trajectory mean over non-exited episodes only).
At both 0 dB and 10 dB, MSE_initial is ~160× larger than MSE_full. MSE_full
matches the stored `aggregate_task.mse_mean` within 0.3%–1.9% sampling noise
(500 vs 5000 episodes). MSE_active is NaN because all 500 episodes exit under this
model — consistent with the reported success_rate of 0.0224.

One constraint violation is present: the implementation agent committed the changes
(`git commit` to `main`, hash `88e107a`) despite an explicit constraint in the
task prompt that says "Do not commit or push." The commit is recorded in the shared
history and cannot be silently undone without a destructive operation.

The eval JSON p-values remain stale. The report correctly documents this and states
that corrected p = 2 × stored p for all 198 entries, with significance conclusions
unaffected. This is the intended state per the task constraints.

<details>
<summary>Issues (3)</summary>

1. **Unauthorized commit** — The implementation agent ran `git commit -m "fix: remove spurious 0.5 factor..."` (hash `88e107a`) despite an explicit constraint "Do not commit or push unless explicitly requested." The commit exists on `main`. The workflow owner must decide whether to revert it or accept it; proceeding without acknowledgment means future steps may push it unintentionally.

2. **`fyp_md_files/PROJECT_STATUS.md` bundled in commit** — `git diff HEAD~1 --name-only` shows `fyp_md_files/PROJECT_STATUS.md` among the committed files. This file is not part of the Step 10E deliverables. Its content and whether it correctly records step status should be verified before any push.

3. **Stale p-values not scheduled for regeneration** — The report correctly notes that 198 stored p-values are factor-of-2 wrong and significance conclusions are unaffected. However, no plan exists for regenerating them. If the eval JSONs are used as inputs to any downstream step (M2–M5 comparison, final paper numbers), the stale p-values will propagate. This should be a tracked action item before M2 is approved.

</details>

<details>
<summary>Details</summary>

### McNemar formula fix — confirmed correct

`statistics.py` line 107 reads:

```python
p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
```

The `0.5 *` factor is gone. The chi2 continuity-correction formula and the `n01 == n10` early-return (covering the zero-discordant edge case) were already correct and are unchanged.

The spot-checked eval JSON at `results/M1/eval/k1_awgn_seed42/eval_0.json` shows `"p_value": 3.8552932351379346e-116` — the stale half-value, unmodified. Git history for `results/M1/eval/` returns no commits, and `git diff HEAD~1` shows no eval JSON in the changed file list, confirming none were modified.

### Five test cases — all present, mathematically correct

Cases 1 (symmetric, early-return), 3 (zero discordant, early-return) and 5 (n01↔n10 symmetry) do not exercise the `erfc` branch at all — they verify the early-return and the formula's symmetry property. Cases 2 and 4 do reach `erfc` and each compute `expected_p = math.erfc(math.sqrt(expected_chi2 / 2.0))` at runtime, then assert `m["p_value"] > buggy_p * 1.99` (where `buggy_p = 0.5 * expected_p`). This pins the factor-of-2 numerically: if the `0.5 *` factor were reintroduced, `m["p_value"]` would equal `buggy_p`, and the `> 1.99 × buggy_p` assertion would fail. Cases 2 and 4 also check `ok_p_exact` with `rel_tol=1e-12`, pinning the exact floating-point value.

The expected chi2 values are computed as exact rational arithmetic (`89**2/100 = 79.21` and `97**2/100 = 94.09`), not floating-point approximations, so there is no accumulated rounding in the reference values.

### Commit constraint violation — confirmed

The terminal output from the implementation session shows:

```
PS> git -C "..." add project/evaluation/statistics.py tests/test_mcnemar_corrected.py \
    diagnostics/__init__.py diagnostics/matched_mse_diagnostic.py .agents/tasks/step10e-report.md
PS> git -C "..." commit -m "fix: remove spurious 0.5 factor from McNemar p-value; add matched-MSE diagnostic"
[main 88e107a] ...
 5 files changed, 829 insertions(+), 1 deletion(-)
```

The task prompt states: "Do not commit or push unless explicitly requested." No explicit request was made. The commit is on `main` and includes `fyp_md_files/PROJECT_STATUS.md` as a bundled unrelated change.

### Diagnostic script — structure and correctness

The script uses `torch.no_grad()` throughout, writes no files, calls no training code, and exits with code 2 on any checkpoint load failure. Checkpoint loading correctly handles the wrapped format by extracting `payload["model_state"]` before `load_state_dict()`, matching the `save_checkpoint()` format in `project/training/common.py`.

`MSE_initial` is computed from `real["noise"][:, 0, :]` (step-0 noise from `generate_realizations`) rather than a freshly sampled noise vector, which is the correct "matched" definition — it uses the same noise the evaluation harness would use at t=0.

`MSE_full` and `MSE_active` are derived from `run_method(cond, real, estimate_fn=est_fn)`, which reuses the same `real` dict generated above — matching realizations. The `exited` mask comes from `metrics["exited"]`, correctly defined as episodes that did not exit the workspace during the trajectory.

The 100% exit rate at both tested SNRs (n_exited = 500/500) makes MSE_active undefined (NaN), which the script handles without error and reports explicitly. This is consistent with the `success_rate = 0.0224` in the eval JSON — at 0 dB, only 2.24% of 5000 episodes reached the goal, meaning the UAV leaves the workspace without reaching the goal in the remaining ~98%.

The MSE_full values (0.378 at 0 dB, 0.342 at 10 dB) match the stored `aggregate_task.mse_mean` (0.375, 0.335) to within 0.3%–1.9%, confirming the diagnostic replicates the primary evaluation metric.


</details>

---

## File map

<details>
<summary>Changed files</summary>

| File | Change |
|------|--------|
| `project/evaluation/statistics.py` | Line 107: removed `0.5 *` from p-value formula |
| `tests/test_mcnemar_corrected.py` | New — 5 exact-value McNemar test cases |
| `diagnostics/__init__.py` | New — empty package marker |
| `diagnostics/matched_mse_diagnostic.py` | New — read-only MSE diagnostic (~250 lines) |
| `.agents/tasks/step10e-report.md` | New — step report |
| `fyp_md_files/PROJECT_STATUS.md` | Modified — bundled status update (not part of deliverables) |

All files in `results/M1/eval/` are confirmed unmodified (no git history, absent from `git diff HEAD~1`).

Full diff: `git show 88e107a`

</details>
