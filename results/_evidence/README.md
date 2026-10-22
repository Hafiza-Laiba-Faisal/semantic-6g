# Evidence bundle (archived 2026-10-23)

This directory exists so the thesis record does **not** depend on any file
outside the repository.

## `wa_evidence_2026-10-23.zip`

A byte-for-byte zip of `C:\wa_evidence\` as it stood on 2026-10-23 (22 files,
~0.6 MB uncompressed). It contains the out-of-repo evidence that the results
chapters reference:

| path inside the zip | what it is |
|---|---|
| `WAVE_A_2026-10-20T1645Z/` | SHA256 + checkpoint evidence for the two Wave-A cells that failed and were **not** retrained (survivorship-bias protection) |
| `WAVE_A_PRE_EVAL_2026-10-21/` | pre-evaluation snapshot of the Wave-A failed-cell artifacts |
| `PRE_C3_FINALIZE_2026-10-21/` | state of `results/` before the C3 `caps_h256` invalid-arm declaration |
| `PRE_RUNALL_REFRESH_2026-10-21/` | state before the `run_all` refresh |

These files were never tracked by git (they lived outside the repo root) and
are otherwise unrecoverable if the local machine copy is deleted.

## `.pth` checkpoints

`results/checkpoints/M1_task_awgn_k1_seed42.pth` and
`results/checkpoints/M2_task_rayleigh_k54_seed42.pth` are legacy `.pth`
binaries that `.gitignore` excludes. They were force-added on 2026-10-23 so a
fresh clone still has them. Their identical-content `.pt` successors already
exist under `results/M1/.../final_ckpt.pt` and
`results/M2/.../final_ckpt.pt`.

## Restoring the project from GitHub alone

```bat
git clone https://github.com/Hafiza-Laiba-Faisal/semantic-6g.git
cd semantic-6g
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python tools/run_wave_c.py status
python tools/run_wave_c.py verify
```

`.venv/` is the only large thing not in git (1.3 GB, reproducible). Everything
else — 299 `.pt` checkpoints, all results, the archive, the documents and the
figures — is in the repository.

To unpack the evidence bundle:

```bat
powershell -Command "Expand-Archive results\_evidence\wa_evidence_2026-10-23.zip -DestinationPath C:\wa_evidence"
```
