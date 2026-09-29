"""Step 4 reconstruction smoke training (item 13).

Runs the frozen-objective smoke train on AWGN and Rayleigh, prints the
required report fields, checks meaningful loss reduction, finiteness and
bitwise reproducibility.

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_reconstruction_smoke
"""

from __future__ import annotations

import math
import sys

import torch

from project.training.train_reconstruction import train_reconstruction

SUMMARY = []


def check(name, ok, detail=""):
    SUMMARY.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]" +
          (f"  ({detail})" if detail else ""))


def report_line(tag, rep):
    print(f"\n--- {tag} smoke train report (item 13) ---")
    print(f"  seed                        = {rep['seed']}")
    print(f"  k (complex channel uses)    = {rep['k']}")
    print(f"  SNR sampling range (dB)     = U({rep['snr_range_db'][0]}, "
          f"{rep['snr_range_db'][1]})")
    print(f"  steps x batch               = {rep['steps']} x "
          f"{rep['batch_states']}  (Adam lr={rep['lr']})")
    print(f"  initial reconstruction loss = {rep['loss_initial']:.6f}")
    print(f"  final reconstruction loss   = {rep['loss_final']:.6f}")
    print(f"  loss reduction              = {rep['loss_reduction_pct']:.1f}%")
    print(f"  encoder grad norm (first)   = {rep['grad_norm_encoder_initial']:.4e}")
    print(f"  encoder grad norm (final)   = {rep['grad_norm_encoder_final']:.4e}")
    print(f"  decoder grad norm (first)   = {rep['grad_norm_decoder_initial']:.4e}")
    print(f"  decoder grad norm (final)   = {rep['grad_norm_decoder_final']:.4e}")
    print(f"  NaN/Inf batches             = {rep['nan_batches']}")


def main():
    torch.manual_seed(0)
    min_reduction = {"awgn": 40.0, "rayleigh": 20.0}

    for channel in ("awgn", "rayleigh"):
        _, rep = train_reconstruction(channel=channel, k=3, seed=42,
                                      steps=400, batch_states=2048)
        report_line(channel.upper(), rep)
        ok_finite = all(math.isfinite(v) for v in
                        (rep["loss_initial"], rep["loss_final"],
                         rep["grad_norm_encoder_final"],
                         rep["grad_norm_decoder_final"]))
        check(f"13/{channel} finite losses & grads", ok_finite)
        check(f"13/{channel} no NaN batches", rep["nan_batches"] == 0,
              f"{rep['nan_batches']} NaN batches")
        check(f"13/{channel} meaningful loss reduction",
              rep["loss_reduction_pct"] >= min_reduction[channel],
              f"{rep['loss_reduction_pct']:.1f}% >= "
              f"{min_reduction[channel]}%")
        check(f"13/{channel} grads alive at end",
              rep["grad_norm_encoder_final"] > 0 and
              rep["grad_norm_decoder_final"] > 0)

        # bitwise reproducibility on a short identical budget
        _, rep_a = train_reconstruction(channel=channel, k=3, seed=42,
                                        steps=60, batch_states=512)
        _, rep_b = train_reconstruction(channel=channel, k=3, seed=42,
                                        steps=60, batch_states=512)
        check(f"13/{channel} reproducible (bitwise, 60-step budget)",
              rep_a["history"] == rep_b["history"])

    n_pass, n_all = sum(SUMMARY), len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"RECONSTRUCTION SMOKE SUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)


if __name__ == "__main__":
    main()
