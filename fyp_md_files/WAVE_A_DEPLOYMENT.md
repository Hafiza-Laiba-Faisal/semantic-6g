# Wave-A other-PC deployment guide (long training runs)

## What this does
Runs the 46 Wave-A diagnostic ablation runs (~12.4h CPU total) + reduced-episode
Wave-A evaluations on a separate PC, isolated from frozen M1/M2, and produces a
`dashboard.html` updatable with one command at any time.

## Steps (on the other PC)

1. **Clone repo** (or copy the working tree):
   ```
   git clone <your-repo-url> semantic-6g
   cd semantic-6g
   ```

2. **Set up Python venv** (see SETUP.md):
   ```
   python -m venv .venv
   .venv/Scripts/pip install torch numpy matplotlib     # Windows
   # .venv/bin/pip install torch numpy matplotlib       # Linux/Posix
   ```

3. **Governance gate (MUST pass first)**:
   ```
   .venv/Scripts/python.exe tools/run_wave_a.py verify
   ```
   Expect `ALL PASS`.

4. **Sanity smoke (optional, 1-2 min)**:
   ```
   .venv/Scripts/python.exe tools/run_wave_a.py train smoke:fixsnr_10db
   .venv/Scripts/python.exe tools/run_wave_a.py train smoke:wrecon_pos
   ```
   Runs 100-step versions of one A4 + one A2 arm into an isolated root,
   non-scientific, auto-tidy.

5. **Full deterministic plan preview** (no training):
   ```
   .venv/Scripts/python.exe tools/run_wave_a.py plan
   ```

6. **Start the training queue** (idempotent: DONE.json markers make repeat
   invocations SKIP completed cells, so kill & resume is safe):
   ```
   .venv/Scripts/python.exe tools/run_wave_a.py train           # all runs (~12.4h)
   # lane filters (multi-machine option):
   #   .venv/Scripts/python.exe tools/run_wave_a.py train awgn
   #   .venv/Scripts/python.exe tools/run_wave_a.py train rayleigh
   ```

7. **Evaluate (reduced 1000-episode diagnostic grid)**:
   ```
   .venv/Scripts/python.exe tools/run_wave_a.py evaluate        # ~2.5h
   # lane filters: `evaluate awgn`, `evaluate rayleigh`
   ```

8. **Summarize + update the dashboard** (one command each, any time):
   ```
   .venv/Scripts/python.exe tools/run_wave_a.py finalize
   .venv/Scripts/python.exe tools/build_dashboard.py --open
   ```

## Multi-machine partitioning (optional speed-up)
- PC-1: `python tools/run_wave_a.py train awgn && python tools/run_wave_a.py evaluate awgn`
- PC-2: `python tools/run_wave_a.py train rayleigh && python tools/run_wave_a.py evaluate rayleigh`
- Mid-run check: `python tools/run_wave_a.py status`

## Non-negotiable guardrails
- NEVER call `tools/run_step10_m1.py` or `tools/run_step11_m2.py` on the
  Wave-A side — Wave-A never rewrites M1/M2 paths.
- Every Wave-A result carries `amendment: "WAVE_A-non-frozen-2026-10-12"`;
  every table/figure must caption E=1000 diagnostic-only (not 1:1 with
  frozen E=5000 numbers).
- Commit back: `git add results/WAVE_A dashboard.html && git commit -m
  "Wave-A ablation results"` then push as usual.

## What to report back
1. `tools/run_wave_a.py status` completion counts.
2. `finalize` → `step_wave_a_summary.json` exit-status census.
3. `dashboard.html` — pooled SR per arm tag; the λ-tags vs frozen-arm SR
   comparison is the headline deliverable for the diagnosis chapter,
   alongside the long-vs-frozen loss-envelope plots from W-A4
   (long_* run histories after completion).
