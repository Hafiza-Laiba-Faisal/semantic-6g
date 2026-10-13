"""One-shot digital decode check: does CSI error change raw BER as expected?

Draws random physical states from the frozen test-episode stream, transmits
the frozen digital packet through the frozen Rayleigh channel, decodes with
perfect CSI vs h_hat = h + eps, and reports bit error rate on the quantized
payload. No closed loop: isolates the receiver-side effect.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from project import config
from project.baselines.digital import (DigitalBaseline, dequantize, qpsk_llrs,
                                       pack_bits, quantize, unpack_bits,
                                       viterbi_decode)
from project.channels.rayleigh import equalize_zf, rayleigh_channel
from project.data.normalization import normalize
from project.uav.environment import NavigationEnv


def ber_at(k, snr, sigma, episodes=4000):
    B = config.DIGITAL_B_BY_K[k]
    base = DigitalBaseline(k=k, bits_per_component=B)
    env = NavigationEnv()
    g = config.derive_generator("episodes", config.TEST_SEED)
    p0, v0, goal = env.sample_starts(episodes, generator=g)
    g_noise = config.derive_generator("noise_test", config.TEST_SEED)
    sigma2 = config.TX_POWER * (10.0 ** (-snr / 10.0))
    n = torch.complex(
        torch.randn(episodes, k, generator=g_noise) * (sigma2 / 2.0) ** 0.5,
        torch.randn(episodes, k, generator=g_noise) * (sigma2 / 2.0) ** 0.5)
    h = torch.complex(torch.randn(episodes, 1, generator=g_noise) * 0.5 ** 0.5,
                      torch.randn(episodes, 1, generator=g_noise) * 0.5 ** 0.5)
    # CSI error stream (Wave-B convention, dedicated generator)
    g_eps = torch.Generator()
    g_eps.manual_seed(config.TEST_SEED + 1)
    s = sigma / (2.0 ** 0.5)
    eps = torch.complex(torch.randn(episodes, 1, generator=g_eps) * s,
                        torch.randn(episodes, 1, generator=g_eps) * s)
    h_hat = h + eps

    states = torch.cat([p0, v0, goal], dim=-1)
    s_bar = normalize(states)
    q_ref, _ = quantize(s_bar, B)
    bits_ref = pack_bits(q_ref, B)

    sym = base.encode_state(states)                      # complex [B, k]
    y, _ = rayleigh_channel(sym, snr, noise=n, h=h, return_h=True)
    n_pay = base.pkt["n_payload_symbols"]
    y_pay = y[..., :n_pay]
    y_eq = equalize_zf(y_pay, h_hat)
    hh = (h_hat.real ** 2 + h_hat.imag ** 2).clamp_min(config.EPS_EQUALIZER)
    llrs = qpsk_llrs(y_eq, sigma2 / hh)
    info = viterbi_decode(llrs)[:, : -base.n_tail]
    bits_hat = pack_bits(unpack_bits(info, B), B)
    ber = (bits_hat != bits_ref).float().mean().item()
    return ber


if __name__ == "__main__":
    for k, snr in ((30, 15.0), (54, 0.0), (54, 15.0), (12, 10.0)):
        row = []
        for sigma in (0.0, 0.1, 0.4):
            row.append(f"sig{sigma:g}={ber_at(k, snr, sigma):.5f}")
        print(f"k{k} snr{snr:g}: " + "  ".join(row), flush=True)
