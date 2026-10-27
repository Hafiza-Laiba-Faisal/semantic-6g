# Cancelled starts (no results)

Each directory here contains only the `RUNNING.json` start marker written when
a 256-wide training cell began. Nothing was scored: no checkpoint, no
`train_report.json`, no evaluation slot.

Reason: the capacity ablation was pre-registered as 8 cells (A-9) and reduced
to 2 cells (frozen task arm, awgn + rayleigh, seed 42) at 12:25 UTC on
2026-10-26 because a 256-wide cell costs roughly 3-4x a frozen 128-wide cell
and the machine has 4 physical cores. The decision was taken before any cell
completed, so no result influenced it.

`..._rayleigh_k18_seed42__..._attempt1` is a killed start that was restarted
cleanly; the shipped cell is the live directory with the same name minus the
`__attempt1` suffix.
