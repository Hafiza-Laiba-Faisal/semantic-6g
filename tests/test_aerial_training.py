"""3GPP UMa-aerial TRAINING-channel tests (A-12).

Scope: the per-episode dispersion channel added to ``project/channels/aerial.py``
and the ``channel="umaaerial"`` branch of the model / training rollout.

What is verified
----------------
A. sampling: determinism, empirical P_LoS, centred mean, per-branch shadow
   standard deviation, absolute-reference convention, gain/offset identity;
B. model: finite output, exact equivalence with Rayleigh at gain 1, gradient
   flow, stronger path loss => worse reconstruction, frozen channels ignore the
   new argument, unknown channels still rejected;
C. rollout: the closed-loop training path runs under the new channel with a
   finite loss and finite gradients, and still rejects unknown channels.

No training, no evaluation, no writes outside ``tempfile``.

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_aerial_training
"""

from __future__ import annotations

import math
import sys

SUMMARY = []


def check(name, ok, detail=""):
    SUMMARY.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]"
          + (f"  ({detail})" if detail else ""))


def close(a, b, tol=1e-3):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# A. sampling
# ---------------------------------------------------------------------------

def test_determinism():
    import torch
    from project.channels import aerial

    g1 = aerial.sample_aerial_gain(8, generator=torch.Generator().manual_seed(7))
    g2 = aerial.sample_aerial_gain(8, generator=torch.Generator().manual_seed(7))
    g3 = aerial.sample_aerial_gain(8, generator=torch.Generator().manual_seed(8))
    check("gain sampling is deterministic", bool(torch.equal(g1, g2)))
    check("a different seed changes the draws", not bool(torch.equal(g1, g3)))
    check("gain has shape [B, 1]", tuple(g1.shape) == (8, 1), str(tuple(g1.shape)))
    check("gain is positive and finite",
          bool(torch.isfinite(g1).all()) and bool((g1 > 0).all()))


def test_statistics():
    import torch
    from project.channels import aerial

    n = 20000
    off, is_los = aerial.sample_aerial_offsets_db(
        n, generator=torch.Generator().manual_seed(11))
    p_los_emp = float(is_los.float().mean())
    p_los = aerial.los_probability(500.0, 50.0)
    check("empirical P_LoS matches the 3GPP formula",
          close(p_los_emp, p_los, 0.02), f"{p_los_emp:.4f} vs {p_los:.4f}")

    # centred on the mean path loss => zero mean offset (dB)
    check("dispersion offset is centred on PL_mean",
          close(float(off.mean()), 0.0, 0.15), f"{float(off.mean()):+.4f} dB")

    d3d = math.hypot(500.0, 50.0)
    pl_los = aerial.los_path_loss_db(d3d, 2.0)
    pl_nlos = aerial.nlos_path_loss_db(d3d, 50.0, 2.0)
    pl_mean = p_los * pl_los + (1 - p_los) * pl_nlos
    shadow_los = (off[is_los] - (pl_los - pl_mean)).std().item()
    shadow_nlos = (off[~is_los] - (pl_nlos - pl_mean)).std().item()
    check("LoS shadowing std matches TR 38.901 (4 dB)",
          close(shadow_los, aerial.SHADOW_SIGMA_DB["los"], 0.25),
          f"{shadow_los:.3f} dB")
    check("NLoS shadowing std matches TR 38.901 (6 dB)",
          close(shadow_nlos, aerial.SHADOW_SIGMA_DB["nlos"], 0.25),
          f"{shadow_nlos:.3f} dB")

    # a numeric reference keeps the absolute loss instead of centring
    off_abs, _ = aerial.sample_aerial_offsets_db(
        n, generator=torch.Generator().manual_seed(11), reference_pl_db=0.0)
    check("absolute convention reproduces PL_mean",
          close(float(off_abs.mean()), pl_mean, 0.2),
          f"{float(off_abs.mean()):.2f} vs {pl_mean:.2f} dB")

    g = aerial.sample_aerial_gain(
        5, generator=torch.Generator().manual_seed(3))
    off5, _ = aerial.sample_aerial_offsets_db(
        5, generator=torch.Generator().manual_seed(3))
    check("gain == 10^(-offset/20)",
          bool(torch.allclose(g, 10.0 ** (-off5 / 20.0), atol=1e-12)))


# ---------------------------------------------------------------------------
# B. model
# ---------------------------------------------------------------------------

def _model(k=6):
    from project.models.reconstruction_jscc import ReconstructionDeepJSCC
    return ReconstructionDeepJSCC(k=k)


def _paired_inputs(batch=16, k=6, seed=5):
    import torch
    from project.channels.rayleigh import sample_fading

    torch.manual_seed(seed)
    s = torch.randn(batch, 6) * 0.5
    gen = torch.Generator().manual_seed(seed + 1)
    h = sample_fading(batch, generator=gen)
    return s, h


def test_gain_one_equals_rayleigh():
    import torch
    from project.channels.complex_utils import (
        snr_db_to_sigma2, sample_complex_gaussian)

    model = _model().eval()
    s, h = _paired_inputs()
    gen = torch.Generator().manual_seed(99)
    noise = sample_complex_gaussian(
        torch.Size((s.shape[0], 6)), snr_db_to_sigma2(10.0),
        generator=gen, complex_out=True)
    with torch.no_grad():
        base = model.forward(s, 10.0, channel="rayleigh", noise=noise, h=h)
        aer = model.forward(s, 10.0, channel="umaaerial", noise=noise, h=h,
                            aerial_gain=torch.ones(s.shape[0], 1))
    check("gain 1 is exactly the Rayleigh channel",
          bool(torch.allclose(base, aer, atol=0.0, rtol=0.0)),
          f"max|diff|={float((base - aer).abs().max()):.3e}")


def test_gain_degrades_reconstruction():
    import torch

    model = _model().eval()
    s, h = _paired_inputs(batch=64, seed=13)
    from project.data.normalization import normalize
    target = normalize(s)
    gen = torch.Generator().manual_seed(17)
    from project.channels.complex_utils import (
        snr_db_to_sigma2, sample_complex_gaussian)
    noise = sample_complex_gaussian(
        torch.Size((s.shape[0], 6)), snr_db_to_sigma2(10.0),
        generator=gen, complex_out=True)

    def mse(gain):
        with torch.no_grad():
            out = model.forward(s, 10.0, channel="umaaerial", noise=noise,
                                h=h, aerial_gain=torch.full((s.shape[0], 1), gain))
        return float(((out - target) ** 2).sum(dim=-1).mean() / 6.0)

    clean, faded = mse(1.0), mse(0.1)
    check("a -20 dB episode gain degrades reconstruction", faded > clean * 1.5,
          f"MSE {clean:.4f} -> {faded:.4f}")


def test_gradients_and_frozen_channels():
    import torch

    model = _model()
    s = torch.randn(8, 6) * 0.5
    out = model.forward_losses(s, 10.0, channel="umaaerial",
                               generator=torch.Generator().manual_seed(2))
    check("umaaerial output is finite", bool(torch.isfinite(out).all()))
    out.sum().backward()
    g_enc = sum(float(p.grad.norm()) for p in model.encoder.parameters()
                if p.grad is not None)
    g_dec = sum(float(p.grad.norm()) for p in model.decoder.parameters()
                if p.grad is not None)
    check("gradients reach encoder and decoder", g_enc > 0 and g_dec > 0,
          f"{g_enc:.2e}/{g_dec:.2e}")

    model.zero_grad(set_to_none=True)
    a = model.forward(s, 5.0, channel="awgn",
                      generator=torch.Generator().manual_seed(4))
    b = model.forward(s, 5.0, channel="awgn",
                      generator=torch.Generator().manual_seed(4),
                      aerial_gain=torch.full((8, 1), 0.01))
    check("awgn ignores aerial_gain", bool(torch.equal(a, b)))

    c = model.forward(s, 5.0, channel="rayleigh",
                      generator=torch.Generator().manual_seed(4))
    d = model.forward(s, 5.0, channel="rayleigh",
                      generator=torch.Generator().manual_seed(4),
                      aerial_gain=torch.full((8, 1), 0.01))
    check("rayleigh ignores aerial_gain", bool(torch.equal(c, d)))

    raised = False
    try:
        model.forward(s, 5.0, channel="bogus")
    except ValueError:
        raised = True
    check("unknown channel still rejected", raised)


def test_forward_losses_passthrough():
    import torch
    from project.data.normalization import normalize

    model = _model().eval()
    s = torch.randn(4, 6) * 0.5
    gen = torch.Generator().manual_seed(21)
    gain = torch.full((4, 1), 0.5)
    with torch.no_grad():
        direct = model.forward(s, 3.0, channel="umaaerial", generator=gen,
                               aerial_gain=gain, denormalize_output=False)
    gen = torch.Generator().manual_seed(21)
    with torch.no_grad():
        via = model.forward_losses(s, 3.0, channel="umaaerial",
                                   generator=gen, aerial_gain=gain)
    check("forward_losses forwards aerial_gain",
          bool(torch.equal(direct, via)))
    check("forward_losses returns the normalized estimate",
          tuple(via.shape) == (4, 6) and bool(torch.isfinite(via).all())
          and not bool(torch.equal(via, normalize(s))))


# ---------------------------------------------------------------------------
# C. training rollout
# ---------------------------------------------------------------------------

def test_closed_loop_rollout():
    import torch
    from project.losses.task_loss import task_loss
    from project.training.train_task_oriented import closed_loop_rollout

    model = _model(k=6)
    torch.manual_seed(31)
    s0 = torch.randn(6, 6) * 0.4
    gen = torch.Generator().manual_seed(32)
    pos, ctl, vel = closed_loop_rollout(model, s0, t_max=3, snr_db=10.0,
                                        channel="umaaerial", generator=gen)
    out = task_loss(pos, ctl, vel, s0[:, 4:6])
    loss = out["total"]
    check("umaaerial rollout is finite", bool(torch.isfinite(loss).all()),
          f"loss={float(loss):.4f}")
    loss.backward()
    grads = [float(p.grad.norm()) for p in model.parameters()
             if p.grad is not None and p.grad.norm() > 0]
    check("umaaerial rollout gradients are finite and non-zero", len(grads) > 0)

    gen = torch.Generator().manual_seed(32)
    p1, _, _ = closed_loop_rollout(model, s0, t_max=3, snr_db=10.0,
                                   channel="umaaerial", generator=gen)
    model.zero_grad(set_to_none=True)
    gen = torch.Generator().manual_seed(32)
    p2, _, _ = closed_loop_rollout(model, s0, t_max=3, snr_db=10.0,
                                   channel="umaaerial", generator=gen)
    check("umaaerial rollout is deterministic", bool(torch.equal(p1, p2)))

    raised = False
    try:
        closed_loop_rollout(model, s0, t_max=1, channel="bogus")
    except ValueError:
        raised = True
    check("rollout still rejects unknown channels", raised)


def test_train_entry_rejects_unknown():
    from project.training.train_task_oriented import train_task_oriented

    raised = False
    try:
        train_task_oriented(k=6, channel="bogus", steps=1)
    except ValueError:
        raised = True
    check("train_task_oriented still rejects unknown channels", raised)


def test_eval_convention_matches_the_channel():
    """The A-12 evaluation divides the noise by g; the channel branch scales
    the signal by g and equalizes with h*g.  Both must be the same channel."""
    import torch
    from project.channels.complex_utils import (
        snr_db_to_sigma2, sample_complex_gaussian)

    model = _model().eval()
    s, h = _paired_inputs(batch=32, seed=23)
    gen = torch.Generator().manual_seed(24)
    noise = sample_complex_gaussian(
        torch.Size((s.shape[0], 6)), snr_db_to_sigma2(7.5),
        generator=gen, complex_out=True)
    g = torch.exp(torch.linspace(-0.8, 0.8, s.shape[0])).unsqueeze(-1)
    with torch.no_grad():
        channel_branch = model.forward(s, 7.5, channel="umaaerial", noise=noise,
                                       h=h, aerial_gain=g)
        eval_path = model.forward(s, 7.5, channel="rayleigh", noise=noise / g,
                                  h=h)
    diff = float((channel_branch - eval_path).abs().max())
    check("eval noise/g equals the umaaerial branch", diff < 1e-4,
          f"max|diff|={diff:.2e}")


def test_truncated_tail():
    """A-13: opt-in resampling of the outage tail; everything else unchanged."""
    import torch
    from project.channels import aerial

    cap = 20.0
    big = 20000
    # default path is untouched: identical stream to the A-12 sampler
    a = aerial.sample_aerial_offsets_db(
        big, generator=torch.Generator().manual_seed(31))
    b = aerial.sample_aerial_offsets_db(
        big, generator=torch.Generator().manual_seed(31), max_offset_db=None)
    check("max_offset_db=None is bit-identical to A-12",
          bool(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])))

    g = torch.Generator().manual_seed(31)
    capped, capped_los = aerial.sample_aerial_offsets_db(
        big, generator=g, max_offset_db=cap)
    check("truncated sampler never exceeds the cap",
          float(capped.max()) <= cap, f"max={float(capped.max()):+.2f} dB")
    check("truncated sampler keeps the training batch size",
          tuple(capped.shape) == (big, 1) and tuple(capped_los.shape) == (big, 1),
          str(tuple(capped.shape)))
    frac = float((a[0] > cap).float().mean())
    check("the resampled tail is the outage tail (3-5 % of episodes)",
          0.03 < frac < 0.05, f"P(offset>{cap:g} dB)={frac:.4f}")
    # and the cap must NOT delete the NLoS branch (that was the 10 dB mistake)
    _, truncated_los = aerial.sample_aerial_offsets_db(
        big, generator=torch.Generator().manual_seed(35), max_offset_db=cap)
    nlos_kept = float((~truncated_los.squeeze(-1)).float().mean())
    check("truncation keeps NLoS episodes in the training batch",
          nlos_kept > 0.05, f"NLoS share of batch {100 * nlos_kept:.1f} %")
    # the truncated sample must be a fair draw from the conditional law: its
    # mean must sit below the unconditional mean
    check("truncation lowers the mean loss offset",
          float(capped.mean()) < float(a[0].mean()),
          f"{float(capped.mean()):+.3f} vs {float(a[0].mean()):+.3f} dB")
    # determinism of the rejection loop
    c2, _ = aerial.sample_aerial_offsets_db(
        big, generator=torch.Generator().manual_seed(31), max_offset_db=cap)
    check("truncated sampling is deterministic", bool(torch.equal(capped, c2)))

    # gain = 10^(-offset/20) must stay >= 10^(-cap/20) i.e. never deeper
    gain = aerial.sample_aerial_gain(
        4096, generator=torch.Generator().manual_seed(32), max_offset_db=cap)
    check("truncated gain floor respected",
          float(gain.min()) >= 10.0 ** (-(cap + 5.0) / 20.0),
          f"min gain {float(gain.min()):.4f}")

    # the rollout takes the flag and stays deterministic
    from project.training.train_task_oriented import closed_loop_rollout
    model = _model().eval()
    s, _ = _paired_inputs(batch=8, k=6, seed=33)
    outs = []
    for _ in range(2):
        with torch.no_grad():
            outs.append(closed_loop_rollout(
                model, s, t_max=3, snr_db=10.0, channel="umaaerial",
                generator=torch.Generator().manual_seed(34),
                aerial_max_offset_db=cap)[0])
    check("rollout accepts the truncation flag and is deterministic",
          bool(torch.equal(outs[0], outs[1])))


if __name__ == "__main__":
    print("=" * 72)
    print("3GPP UMa-AERIAL TRAINING-CHANNEL TESTS (A-12 / A-13)")
    print("=" * 72)
    test_determinism()
    test_statistics()
    test_gain_one_equals_rayleigh()
    test_gain_degrades_reconstruction()
    test_eval_convention_matches_the_channel()
    test_gradients_and_frozen_channels()
    test_forward_losses_passthrough()
    test_closed_loop_rollout()
    test_train_entry_rejects_unknown()
    test_truncated_tail()
    n_pass, n_all = sum(SUMMARY), len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)
