# artifacts/ — generated, not source

These files are build outputs. They are committed once for the record and then
regenerated on demand; nothing here is imported by `project/`, `tools/`, or
`tests/`.

| File | Produced by | Notes |
|---|---|---|
| `project_all_in_one.py` | `tools/build_all_in_one.py` | Single-file flattening of the package, for readers who want one file. **This is not the project layout** — the real code is `project/`. |
| `project_story.html` | `tools/build_dashboard.py` (story mode) | Narrative dashboard |
| `thesis_dashboard.html` | `tools/build_dashboard.py` (thesis mode) | Progress dashboard |
| `run_all_output.txt` | `tools/run_all.py` console capture | Full unattended run log |
| `test_eval_output.txt` | test/eval console capture | Historical console output |

Live dashboards and logs are regenerated at the repository root and are
gitignored. If you are reviewing the code, start at `README.md` and `project/`.
