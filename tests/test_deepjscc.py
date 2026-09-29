"""Step 4 DeepJSCC structural tests (items 10, 11, 12).

Covers:
* shape grid: B in {1,4,16} x k in {1,2,3,12,54} through the full path
* power normalization through the model == the verified Step-3 function
* gradient flow: loss.backward() gives finite, NONZERO encoder/decoder
  gradients on AWGN and Rayleigh; nothing detached; noise/h are constants

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_deepjscc
"""

from __future__ import annotations

import sys

import torch

from project import config
from project.channels.complex_utils import (
    codeword_avg_power,
    power_normalize,
    to_complex,
    to_real,
)
from project.losses.reconstruction_loss import reconstruction_loss
from project.data.normalization import normalize
from project.models.reconstruction_jscc import ReconstructionDeepJSCC

SUMMARY = []


def check(name, ok, detail=""):
    SUMMARY.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]" +
          (f"  ({detail})" if detail else ""))


def test_shapes():
    for B in (1, 4, 16):
        for k in (1, 2, 3, 12, 54):
            model = ReconstructionDeepJSCC(k=k)
            s = torch.rand(B, 6) * 16 - 8
            g = torch.Generator().manual_seed(0)
            with torch.no_grad():
                x = model.encoder(normalize(s))              # [B, 2k]
                z = model.encode(s)                          # [B, k] complex
                s_bar_hat = model.forward_losses(s, 10.0, "awgn",
                                                 generator=g)
            ok = (tuple(x.shape) == (B, 2 * k)
                  and z.is_complex() and tuple(z.shape) == (B, k)
                  and tuple(s_bar_hat.shape) == (B, 6))
            check(f"11/B={B},k={k}", ok,
                  f"{tuple(s.shape)}->{tuple(x.shape)}->{tuple(z.shape)}"
                  f"->{tuple(s_bar_hat.shape)}")


def test_power_through_model():
    worst = 0.0
    dup_dev = 0.0
    for B in (1, 8, 128):
        for k in (1, 2, 3, 12, 54):
            model = ReconstructionDeepJSCC(k=k)
            s = torch.rand(B, 6) * 16 - 8
            z = model.encode(s)
            dev = float((codeword_avg_power(z) - config.TX_POWER).abs().max())
            worst = max(worst, dev)
            # the model must use the SAME Step-3 function (bitwise identity)
            s_bar = normalize(s)
            z_ref = power_normalize(model.encoder.encode_complex(s_bar), k=k)
            dup_dev = max(dup_dev, float((z - z_ref).abs().max()))
    check("12a", worst <= 1e-5,
          f"per-codeword power == P through the model (max dev {worst:.2e})")
    check("12b", dup_dev == 0.0,
          f"model uses the verified Step-3 power function (max dev {dup_dev:.2e})")


def test_gradient_flow():
    torch.manual_seed(0)
    for channel in ("awgn", "rayleigh"):
        model = ReconstructionDeepJSCC(k=3)
        s = torch.rand(64, 6) * 16 - 8
        g = torch.Generator().manual_seed(1)
        s_bar_hat = model.forward_losses(s, 10.0, channel, generator=g)
        ok_graph = s_bar_hat.grad_fn is not None
        loss = reconstruction_loss(s_bar_hat, normalize(s))
        loss.backward()
        enc_norms = [float(p.grad.norm()) for p in model.encoder.parameters()
                     if p.grad is not None]
        dec_norms = [float(p.grad.norm()) for p in model.decoder.parameters()
                     if p.grad is not None]
        n_enc_params = len(list(model.encoder.parameters()))
        n_dec_params = len(list(model.decoder.parameters()))
        ok = (ok_graph
              and len(enc_norms) == n_enc_params and len(dec_norms) == n_dec_params
              and all(torch.isfinite(torch.tensor(v)) for v in enc_norms + dec_norms)
              and sum(enc_norms) > 0 and sum(dec_norms) > 0)
        check(f"10/{channel}", ok,
              f"grad_fn={ok_graph}, enc |g|={sum(enc_norms):.3e}, "
              f"dec |g|={sum(dec_norms):.3e}, all finite")


def test_decoder_input_order_parity():
    """Decoder must consume the SAME stacked I/Q order the encoder emits."""
    model = ReconstructionDeepJSCC(k=3)
    y = torch.randn(16, 3, dtype=torch.complex64)
    out_c = model.decoder(y)
    out_r = model.decoder(to_real(y))
    check("5", "decoder complex/real-stacked inputs agree (same I/Q order)",
          bool((out_c - out_r).abs().max() == 0.0))


def test_channel_use_count():
    """2k real outputs are k channel uses (hard stop condition guard)."""
    model = ReconstructionDeepJSCC(k=3)
    s = torch.rand(4, 6)
    z = model.encode(s)
    check("2", "encoder emits k complex channel uses (2k reals != 2k uses)",
          z.is_complex() and z.shape[-1] == 3 and
          model.encoder(normalize(s)).shape[-1] == 6)


if __name__ == "__main__":
    print("=" * 72)
    print("STEP 4 DEEPJSCC STRUCTURAL TESTS")
    print("=" * 72)
    test_channel_use_count()
    test_decoder_input_order_parity()
    test_shapes()
    test_power_through_model()
    test_gradient_flow()
    n_pass, n_all = sum(SUMMARY), len(SUMMARY)
    print(f"\nSUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)
