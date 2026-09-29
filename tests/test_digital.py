"""Step 6 digital baseline tests (Gates A-E, frozen audit section 8).

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_digital
"""

from __future__ import annotations

import math
import sys

import torch

from project import config
from project.baselines.digital import (
    DigitalBaseline,
    conv_encode,
    dequantize,
    pack_bits,
    qpsk_llrs,
    qpsk_modulate,
    quantize,
    unpack_bits,
    viterbi_decode,
)
from project.channels.rayleigh import sample_fading
from project.data.normalization import denormalize, normalize

SUMMARY = []


def check(name, ok, detail=""):
    SUMMARY.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]" +
          (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
# Gate A - quantizer
# ---------------------------------------------------------------------------

def test_gate_a():
    torch.manual_seed(0)
    for B in (1, 2, 3, 4, 8):
        delta = config.quantizer_step(B)
        ok_delta = math.isclose(delta, 2.0 / (2 ** B - 1), rel_tol=1e-12)
        s = torch.rand(4096, 6) * 2.4 - 1.2            # 10% out of range
        q, s_hat = quantize(s, B)
        in_range = (s >= -1) & (s <= 1)
        err = (s_hat - s.clamp(-1, 1)).abs()[in_range]
        max_err = float(err.max()) if err.numel() else 0.0
        # endpoints exact, clipping at boundaries
        q_e, s_e = quantize(torch.tensor([[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]]), B)
        ep_ok = bool((s_e[0, [0, 2, 4]] == -1).all()
                     and (s_e[0, [1, 3, 5]] == 1).all())
        q_clip, s_clip = quantize(torch.tensor([[4.0, -4.0, 0.3, 0.9, -0.7, 0.0]]), B)
        clip_ok = bool(s_clip[0, 0] == 1.0 and s_clip[0, 1] == -1.0)
        # round-trip dequantize(quantize(x)) == reconstruction
        rt = float((dequantize(q, B) - s_hat).abs().max())
        check(f"A/B={B}", ok_delta and max_err <= delta / 2 + 1e-9
              and ep_ok and clip_ok and rt == 0.0,
              f"delta={delta:.4f}, max_err={max_err:.5f} <= delta/2, "
              f"endpoints+clip+roundtrip OK")


# ---------------------------------------------------------------------------
# Gate B - uncoded Gray-QPSK reference vs theory
# ---------------------------------------------------------------------------

def test_gate_b():
    print("      theory axis: SNR = P/sigma^2 per complex symbol = Es/N0;")
    print("      equivalently Q(sqrt(2*Eb/N0)) with Eb = Es/2 (supplementary label)")
    torch.manual_seed(1)
    n_syms = 4_000_000
    for snr_db in (0.0, 6.0, 10.0):
        sigma2 = 10 ** (-snr_db / 10.0)
        bits = torch.randint(0, 2, (1, 2 * n_syms))
        z = qpsk_modulate(bits)
        g = torch.Generator().manual_seed(123)
        from project.channels.complex_utils import sample_complex_gaussian
        n = sample_complex_gaussian(z.shape, sigma2, generator=g)
        y = z + n
        llrs = qpsk_llrs(y, torch.tensor(sigma2))
        decision = (llrs.sign() < 0).long()         # LLR<0 -> decide bit 1
        ber = float((decision != bits).float().mean())
        # Gray-QPSK: each label bit rides one axis; I/Q are independent
        # BPSKs with antipodal distance 2/sqrt(2) = sqrt(2) Es=1:
        # BER = Q(sqrt(Es/N0)) = Q(sqrt(SNR)) = 0.5*erfc(sqrt(SNR/2)).
        # Equivalently Q(sqrt(2*Eb/N0)) with Eb = Es/2 (supplementary label).
        theory = 0.5 * math.erfc(math.sqrt(0.5 * 10 ** (snr_db / 10.0)))
        tol = 4.0 * math.sqrt(theory * (1 - theory) / (n_syms * 2))
        check(f"B/awgn snr={snr_db:.0f}dB", abs(ber - theory) <= max(tol, 1e-6),
              f"ber={ber:.6f}, Q-theory={theory:.6f}, tol={max(tol, 1e-6):.6f} "
              f"(N={n_syms * 2} bits)")
    # exact log-MAP == closed form on clean points (documentation check)
    z = qpsk_modulate(torch.tensor([[0, 1]]))
    llr = qpsk_llrs(z, torch.tensor(0.5))
    check("B/llr-convention", bool((llr[0, 0] > 0).all() and (llr[0, 1] < 0).all()),
          f"bits (0,1) -> LLRs {[round(float(v), 2) for v in llr[0]]} "
          "(LLR>0 means bit 0)")


# ---------------------------------------------------------------------------
# Gate C - convolutional code + soft Viterbi
# ---------------------------------------------------------------------------

def test_gate_c():
    # 1. hand-computed reference vector (G1 first, zero-tail termination)
    # input u=1 then the 6 tail zeros; register walk from state 0 with
    # c1 = u^m3^m4^m5^m6, c2 = u^m1^m3^m4^m6 (m1 = most recent):
    #   t0 s=0,u=1 -> (1,1)   t1 s=1,u=0 -> (0,1)   t2 s=2,u=0 -> (0,0)
    #   t3 s=4,u=0 -> (1,1)   t4 s=8,u=0 -> (1,1)   t5 s=16,u=0 -> (1,0)
    #   t6 s=32,u=0 -> (1,1), final state 0
    # equivalently (1+D3+D4+D5+D6) and (1+D+D3+D4+D6) applied to [1,0^6].
    info_hand = torch.tensor([[1, 0, 0, 0, 0, 0, 0]])
    expected = torch.tensor([[1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]])
    coded_hand = conv_encode(info_hand)
    check("C/hand-vector", torch.equal(coded_hand, expected),
          f"got {coded_hand[0].tolist()}")
    # 2. zero-tail: final state must be 0 for random sequences (assert inside)
    torch.manual_seed(2)
    info = torch.randint(0, 2, (64, 114))
    info = torch.cat([info, torch.zeros(64, 6, dtype=torch.long)], dim=1)
    coded = conv_encode(info)
    check("C/zero-tail", coded.shape == (64, 240))
    # 3. noiseless chain: exact LLR + Viterbi = zero bit errors (incl. tail)
    z = qpsk_modulate(coded)
    llrs = qpsk_llrs(z, torch.tensor(0.7))       # any sigma2, no noise
    dec = viterbi_decode(llrs)
    n_err = int((dec != info).sum())
    check("C/noiseless-exact", n_err == 0, f"{n_err} bit errors in {info.numel()}")
    # 4. corrupted symbol decodes correctly (single strong distortion)
    z_bad = z.clone()
    z_bad[0, 5] = -z_bad[0, 5]                   # flip one QPSK symbol
    dec_bad = viterbi_decode(qpsk_llrs(z_bad, torch.tensor(0.25)))
    n_bad = int((dec_bad != info).sum())
    check("C/corruption-recovery", n_bad <= 6,
          f"{n_bad} residual bit errors after 1-symbol hit (Viterbi limits spread)")
    # 5. bit ordering: pack/unpack round-trip, component-major MSB-first
    q = torch.randint(0, 256, (32, 6)).float()
    rt = unpack_bits(pack_bits(q, 8), 8)
    check("C/pack-order", bool((rt == q).all()),
          "component-major MSB-first round-trip exact")
    # explicit MSB check: q=1 -> [0,0,0,0,0,0,0,1]
    bits1 = pack_bits(torch.tensor([[1.0]]), 8)
    check("C/msb-first", bool((bits1[0, :7] == 0).all() and bits1[0, 7] == 1),
          f"level 1 -> {bits1[0].tolist()}")


# ---------------------------------------------------------------------------
# Gate D - Rayleigh path
# ---------------------------------------------------------------------------

def test_gate_d():
    torch.manual_seed(3)
    B_cfg = 8
    base = DigitalBaseline(k=54, bits_per_component=B_cfg)
    s = torch.rand(256, 6) * 16 - 8
    sym = base.encode_state(s)
    check("D/shapes", sym.shape[-1] == 54 and torch.is_complex(sym),
          f"{tuple(sym.shape)}")
    snr_db = 12.0
    sigma2 = 10 ** (-snr_db / 10.0)
    g1 = torch.Generator().manual_seed(9)
    g2 = torch.Generator().manual_seed(9)
    y1, h1 = None, None
    from project.channels.rayleigh import rayleigh_channel
    y1, h1 = rayleigh_channel(sym, snr_db, generator=g1, return_h=True)
    y2, h2 = rayleigh_channel(sym, snr_db, generator=g2, return_h=True)
    check("D/reproducible", bool((y1 == y2).all()) and bool((h1 == h2).all()))
    # block fading: h constant across the k symbols
    y0 = h1 * sym                                  # noiseless fading
    r0 = y0[:, 0] / sym[:, 0]
    r53 = y0[:, 53] / sym[:, 53]
    check("D/block-fading", float((r0 - r53).abs().max()) < 1e-4)
    # CSI-scaled LLR: finite under the guard, tiny |h| included
    h_tiny = h1.clone()
    h_tiny[0, 0] = 1e-6 + 0j
    s_hat = base.decode_symbols(y1, "rayleigh", snr_db, h=h_tiny)
    check("D/guard-finite", bool(torch.isfinite(s_hat).all()))
    # perfect-CSI Rayleigh decode end-to-end works
    s_hat2 = base.decode_symbols(y1, "rayleigh", snr_db, h=h1)
    check("D/decode-ok", s_hat2.shape == (256, 6)
          and bool(torch.isfinite(s_hat2).all()))


# ---------------------------------------------------------------------------
# Gate E - full digital navigation chain
# ---------------------------------------------------------------------------

def test_gate_e():
    torch.manual_seed(4)
    base = DigitalBaseline(k=54, bits_per_component=8)
    s = torch.rand(512, 6) * 16 - 8
    # noiseless == quantizer-only distortion
    s_hat = base.forward(s, snr_db=10.0, channel="awgn", noiseless=True)
    q, s_bar_ref = quantize(normalize(s), 8)
    dev = float((s_hat - denormalize(s_bar_ref)).abs().max())
    check("E/noiseless==quantizer", dev == 0.0,
          f"max dev vs quantizer-only reconstruction: {dev:.2e}")
    # framed packet accounting (framing arithmetic verification)
    pkt = base.pkt
    ok = (pkt["n_src_bits"] == 48 and pkt["n_info_bits"] == 54
          and pkt["n_coded_bits"] == 108 and pkt["n_payload_symbols"] == 54
          and pkt["n_padding_symbols"] == 0 and pkt["n_transmitted_symbols"] == 54)
    check("E/framing-arithmetic", ok,
          f"6B=48, info=54, coded=108, payload syms=54, pad=0, k=54")
    # every feasible k point executes end-to-end with a feasible B
    for k, b in config.DIGITAL_B_BY_K.items():
        bl = DigitalBaseline(k=k, bits_per_component=b)
        out = bl.forward(s[:64], snr_db=10.0, channel="awgn", noiseless=True)
        assert out.shape == (64, 6)
    check("E/all-feasible-k", True,
          f"k in {sorted(config.DIGITAL_B_BY_K)} execute; "
          f"packet == {len(config.DIGITAL_B_BY_K)} configs valid")
    # AWGN + Rayleigh closed loop (finite outputs, in-range)
    out_a = base.forward(s, snr_db=10.0, channel="awgn",
                         generator=torch.Generator().manual_seed(5))
    out_r = base.forward(s, snr_db=12.0, channel="rayleigh",
                         generator=torch.Generator().manual_seed(6))
    check("E/awgn+rayleigh", bool(torch.isfinite(out_a).all())
          and bool(torch.isfinite(out_r).all())
          and float(out_a.abs().max()) <= 8.0 + 1e-6,
          f"AWGN/rayleigh outputs finite and within the workspace range")
    # closed-loop navigation into the FROZEN controller/dynamics (sanity)
    from project.uav.controller import pd_controller
    from project.uav.dynamics import step as dyn_step
    p0 = torch.tensor([[0.0, 0.0]]); v0 = torch.zeros(1, 2)
    goal = torch.tensor([[6.0, 4.0]])
    p, v = p0, v0
    for _ in range(60):
        s_now = torch.cat([p, v, goal], dim=-1)
        s_hat = base.forward(s_now, snr_db=14.0, channel="awgn",
                             generator=torch.Generator().manual_seed(7))
        u = pd_controller(s_hat)
        p, v = dyn_step(p, v, u)
    d_final = float((p - goal).norm())
    check("E/closed-loop-navigation", d_final <= 0.8,
          f"digital feedback loop (B=8, k=54, 14 dB): final distance "
          f"{d_final:.3f} m after 60 steps")


# ---------------------------------------------------------------------------
# infeasibility (k in {1,2,3}) + framing asserts
# ---------------------------------------------------------------------------

def test_infeasibility():
    for k in (1, 2, 3):
        try:
            DigitalBaseline(k=k, bits_per_component=1)
            check(f"infeasible/k={k}", False, "should have raised")
        except config.InfeasibleDigitalConfig as e:
            check(f"infeasible/k={k}", "payload needs 12" in str(e),
                  str(e)[:60])
    # B=8 at k=3 also infeasible, never relaxed
    try:
        DigitalBaseline(k=3, bits_per_component=8)
        check("infeasible/k=3,B=8", False, "should have raised")
    except config.InfeasibleDigitalConfig:
        check("infeasible/k=3,B=8", True)


# ---------------------------------------------------------------------------
# regression (in-process): Steps 0-2, 3, 4, 5
# ---------------------------------------------------------------------------

def test_regression():
    import tests.test_channels as tc
    tc.SUMMARY.clear()
    tc.test_conversion(); tc.test_power_normalization(); tc.test_awgn_variance()
    tc.test_awgn_snr(); tc.test_rayleigh_stats(); tc.test_block_fading()
    tc.test_equalization(); tc.test_repro_and_conventions()
    check("REG/Step-3", all(tc.SUMMARY), f"{sum(tc.SUMMARY)}/{len(tc.SUMMARY)}")

    import tests.test_step02 as ts
    fns = [ts.test_config, ts.test_normalization_roundtrip,
           ts.test_normalization_endpoints_and_clip, ts.test_dynamics_closed_form,
           ts.test_dynamics_autograd, ts.test_velocity_clip,
           ts.test_controller_formula, ts.test_controller_saturation_and_gradient,
           ts.test_env_success_semantics, ts.test_env_exit_freeze_and_determinism,
           ts.test_env_sampling_and_nans]
    ok = all((lambda f: (f() or True))(fn) is not None for fn in fns)
    try:
        for fn in fns:
            fn()
        check("REG/Step 0-2", True, "11/11")
    except AssertionError:
        check("REG/Step 0-2", False)

    import tests.test_deepjscc as td
    td.SUMMARY.clear()
    td.test_channel_use_count(); td.test_decoder_input_order_parity()
    td.test_shapes(); td.test_power_through_model(); td.test_gradient_flow()
    check("REG/Step-4 structural", all(td.SUMMARY),
          f"{sum(td.SUMMARY)}/{len(td.SUMMARY)}")

    import tests.test_task_oriented as tt
    tt.SUMMARY.clear()
    tt.test_a_loss_algebra(); tt.test_b_rollout_dims(); tt.test_c_zero_error()
    tt.test_d_gradient_flow(); tt.test_e_controller_dynamics()
    check("REG/Step-5 core", all(tt.SUMMARY),
          f"{sum(tt.SUMMARY)}/{len(tt.SUMMARY)} (smoke trains run separately)")


if __name__ == "__main__":
    print("=" * 72)
    print("STEP 6 DIGITAL BASELINE TESTS (Gates A-E)")
    print("=" * 72)
    test_gate_a()
    test_gate_b()
    test_gate_c()
    test_gate_d()
    test_gate_e()
    test_infeasibility()
    test_regression()
    n_pass, n_all = sum(SUMMARY), len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)
