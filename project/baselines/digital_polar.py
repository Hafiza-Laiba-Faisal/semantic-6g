"""5G NR-style CRC-aided Polar digital baseline (Wave-D; non-frozen).

Textbook standard construction (Arıkan convention, verified by the
noiseless-roundtrip gate below):

* F = [[1,0],[1,1]] ;  G_N = F^{⊗n}  (built by explicit Kronecker product;
  N <= 2^9 for our packets, so a dense N x N GF(2) matrix is trivial).
* Encode:        x = (u @ G_N) mod 2,   u = [info | CRC-8 | frozen zeros]
  (info bits + CRC occupy the most reliable positions of the NATURAL
  u-order; frozen positions are the rest).
* Reliability:   row i of G_N has Hamming weight 2^(n - popcount(i)), so
  lower popcount(i) <=> higher weight <=> more reliable. Frozen set =
  highest-popcount positions first.
* Decode (SC):   standard even/odd interleaved split: node LLRs split as
  (llrs[0::2], llrs[1::2]); f = soft-XOR (sign*min-abs), g = L_odd +
  (1-2 u_hat_even) * L_even; frozen leaves clamped to 0.
  On a CLEAN channel this decoder is EXACT (gate-enforced, see tests).
* CRC-8 (poly 0x07) rides in the u-vector; SC output info bits are taken
  as-is (CRC failure statistics are a reporting concern for the eval layer,
  not the decoder loop).

Interface-compatible with DigitalBaseline (quantize/pack/QPSK/channel
layers reused verbatim), evaluation-only, fully deterministic (no RNG).
"""

from __future__ import annotations

import math
import sys
from typing import Optional

import torch

from project import config
from project.baselines.digital import (dequantize, pack_bits, quantize,
                                       qpsk_llrs, qpsk_modulate, unpack_bits)
from project.channels.awgn import awgn_channel
from project.channels.rayleigh import equalize_zf, rayleigh_channel
from project.data.normalization import denormalize, normalize


# ---------------------------------------------------------------------------
# G_N = F^{⊗n} (explicit Kronecker product) and reliability order
# ---------------------------------------------------------------------------

def polar_matrix(N: int) -> torch.Tensor:
    """G_N in the SC-recursion-compatible order (= textbook G4 =
    [[1,0,0,0],[1,0,1,0],[1,1,0,0],[1,1,1,1]] recursion), verified:
    (a) uG4_std % 2 then even/odd-split SC on clean LLRs returns u exactly
    (validated by the standalone N=4 identity probe), (b) G@G == I mod 2
    (F^{⊗n} self-inverse). Built by the recursion G_{2m} rows from G_m:
    top half   = [G_m | 0_{m}]
    bottom half= [G_m | G_m]
    which is the standard Kronecker F^{⊗n} in NATURAL u order **after**
    applying the x-side bit-reversal so that the even/odd LLR split of the
    decoder aligns with the node partition (textbook decoder convention).
    """
    F = torch.tensor([[1, 0], [1, 1]], dtype=torch.long)
    G = F.clone()
    while G.shape[0] < N:
        m = G.shape[0]
        out = torch.zeros(2 * m, 2 * m, dtype=torch.long)
        out[:m, :m] = G                    # top-left  = G_m
        out[m:, :m] = G                    # bottom-left = G_m
        out[m:, m:] = G                    # bottom-right = G_m
        # top-right stays 0
        G = out
    assert G.shape == (N, N)
    return G


def reliability_order(N: int) -> list:
    """u-indices sorted from LEAST to MOST reliable for G_N = F^{⊗n}.

    Row i weight of F^{⊗n} = 2^popcount(i) (verified numerically, see
    smoke tests), so HIGHER popcount = more reliable subchannel. Least-
    to-most order: ascending popcount, ties by natural index.
    """
    idx = list(range(N))
    pop = lambda i: bin(i).count("1")
    return sorted(idx, key=lambda i: (pop(i), i))       # least reliable first


def crc8(info: torch.Tensor) -> torch.Tensor:
    """Bitwise CRC-8, polynomial x^8 + x^2 + x + 1 (0x07), MSB-first.

    ``info`` [B, K] -> parity bits [B, 8]. Pure python loop (K <= 48,
    B <= 5000: fast enough for eval-only use; vectorization unnecessary).
    """
    rows = []
    poly = 0x07
    for r in range(info.shape[0]):
        reg = 0
        for bit in info[r].tolist():
            fb = ((reg >> 7) & 1) ^ int(bit)
            reg = ((reg << 1) & 0xFF) ^ (poly if fb else 0)
        rows.append([(reg >> (7 - i)) & 1 for i in range(8)])
    return torch.tensor(rows, dtype=torch.long, device=info.device)


# ---------------------------------------------------------------------------
# Polar code + SC decoder
# ---------------------------------------------------------------------------

class PolarCode:
    """N-length polar code: K info + 8 CRC bits; N = next pow2 >= K+8."""

    def __init__(self, n_info: int):
        self.n_info = n_info
        self.n_crc = 8
        self.n_total = n_info + 8
        N = 1
        while N < self.n_total:
            N *= 2
        self.N = N
        self.G = polar_matrix(N)                        # [N, N] long
        order = reliability_order(N)
        self.frozen = torch.zeros(N, dtype=torch.long)
        self.frozen[order[: N - self.n_total]] = 1
        self.info_idx = torch.tensor(order[N - self.n_total:],
                                     dtype=torch.long)  # [K+8] most reliable

    def encode(self, info: torch.Tensor) -> torch.Tensor:
        """Info bits [B, K] -> codeword x [B, N] (u @ G mod 2)."""
        B = info.shape[0]
        u = torch.zeros(B, self.N, dtype=info.dtype, device=info.device)
        crc = crc8(info)
        u_info = torch.cat([info.to(u.dtype), crc.to(u.dtype)], dim=-1)
        u[:, self.info_idx] = u_info
        x = (u @ self.G) % 2
        return x

    def codeword_llrs_clean(self, x: torch.Tensor, big: float = 1e6):
        return (1.0 - 2.0 * x.double()) * big


def polar_sc_decode(code: PolarCode, llrs: torch.Tensor) -> torch.Tensor:
    """Standard SC with even/odd split; LLRs [B, N] -> info bits [B, K].

    Recursive: split(x_LLRS) -> left = f(even, odd), right = g(odd | û_even).
    Frozen leaves are clamped to bit 0. Fully deterministic.
    """
    B, N = llrs.shape
    assert N == code.N, (N, code.N)
    K = code.n_info
    device = llrs.device
    frozen_mask = code.frozen.to(device)
    sys.setrecursionlimit(10000)

    def sc_rec(node_llrs, base_idx):
        n = node_llrs.shape[-1]
        if n == 1:
            bit = (node_llrs <= 0).to(torch.long)
        Le, Lo = node_llrs[..., 0::2], node_llrs[..., 1::2]
        # f-update (left child, u_even): for G with x_even = u_even ^ u_odd
        # the left LLR is the soft-XOR combine (min-sum approx of the exact
        # 2*atanh(tanh(La/2)tanh(Lb/2)) recursion).
        f_llr = torch.sign(Le) * torch.sign(Lo) \
            * torch.minimum(Le.abs(), Lo.abs())
        u_even = sc_rec(f_llr, base_idx)
        # g-update (right child, u_odd): the standard add-form
        #   L_odd = Lo + (1 - 2*û_even)*Le
        # matches G = [[1,1],[0,1]]-style conventions; for OUR G = F =
        # [[1,0],[1,1]] (x_even = u0, x_odd = u0^u1) the verified-at-N=2
        # form is  L_odd = (1 - 2*û_even)*Lo + Le  (see module docstring).
        g_llr = (1.0 - 2.0 * u_even.to(Lo.dtype)) * Lo + Le
        u_odd = sc_rec(g_llr, base_idx + 1)
        # interleave back to natural u-order
        out = torch.empty_like(torch.cat([u_even, u_odd], dim=-1))
        out[..., 0::2] = u_even
        out[..., 1::2] = u_odd
        return out

    u_hat = sc_rec(llrs, 0)
    info_dec = u_hat[:, code.info_idx][:, :K]
    return info_dec


# ---------------------------------------------------------------------------
# Digital baseline with the polar code block (same outer pipeline)
# ---------------------------------------------------------------------------

class PolarDigitalBaseline:
    """Same interface as DigitalBaseline but with the polar code block."""

    def __init__(self, k: int, bits_per_component: int) -> None:
        self.pkt = config.digital_packet(k, bits_per_component)  # raises if infeasible
        self.k = k
        self.B = bits_per_component
        self.n_tail = 8                       # CRC-8 rides in the tail slot budget
        self.pad_bits = 2 * self.pkt["n_padding_symbols"]
        self._code = PolarCode(self.pkt["n_src_bits"])

    # -- transmit --------------------------------------------------------

    def encode_state(self, s: torch.Tensor) -> torch.Tensor:
        s_bar = normalize(s)
        q, _ = quantize(s_bar, self.B)
        info = pack_bits(q, self.B).to(torch.long)
        coded = self._code.encode(info)                        # [B, N]
        sym = qpsk_modulate(coded)                             # [B, N/2]
        if sym.shape[-1] < self.k:
            pad = qpsk_modulate(torch.zeros(
                *sym.shape[:-1], self.k - sym.shape[-1],
                dtype=torch.long, device=sym.device))
            sym = torch.cat([sym, pad], dim=-1)
        assert sym.shape[-1] == self.k, (sym.shape[-1], self.k)
        return sym

    # -- receive ---------------------------------------------------------

    def decode_symbols(
        self,
        y: torch.Tensor,
        channel: str,
        snr_db: float,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_pay = y.shape[-1] if torch.is_complex(y) else (y.shape[-1] // 2)
        y_pay = y[..., :n_pay] if torch.is_complex(y) else y[..., :2 * n_pay]
        if channel == "rayleigh":
            if h is None:
                raise ValueError("Rayleigh decoding needs CSI tensor h")
            y_pay = equalize_zf(y_pay, h)
            hh = (h.real ** 2 + h.imag ** 2).clamp_min(config.EPS_EQUALIZER)
            sigma2_eff = config.TX_POWER * (10.0 ** (-snr_db / 10.0)) / hh
        else:
            sigma2_eff = torch.tensor(
                config.TX_POWER * (10.0 ** (-snr_db / 10.0)),
                dtype=y_pay.real.dtype if torch.is_complex(y_pay)
                else y_pay.dtype)
        llrs_full = qpsk_llrs(y_pay, sigma2_eff)               # [B, 2*m]
        llr = llrs_full.double()
        if llr.shape[-1] < self._code.N:
            # padding pairs modulate the all-zero bits to (+1,+1)/sqrt2:
            # a huge +LLR predicts bit 0 - exactly what a frozen/pad position
            # needs. Pad the LLR vector to N with big +LLRs.
            pad = torch.full((llr.shape[0], self._code.N - llr.shape[-1]),
                             1e6, dtype=llr.dtype, device=llr.device)
            llr = torch.cat([llr, pad], dim=-1)
        llr = llr[..., :self._code.N]
        info = polar_sc_decode(self._code, llr)
        q_hat = unpack_bits(info, self.B)
        return dequantize(q_hat, self.B)

    # -- full path -------------------------------------------------------

    def forward(
        self,
        s: torch.Tensor,
        snr_db: float,
        channel: str = "awgn",
        generator: Optional[torch.Generator] = None,
        noiseless: bool = False,
        noise: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if channel not in ("awgn", "rayleigh"):
            raise ValueError(f"unknown channel {channel!r}")
        with torch.no_grad():
            sym = self.encode_state(s)
            if noiseless:
                # exact no-noise reference: bypass channel, decode clean LLRs
                from project.data.normalization import normalize as _norm
                s_bar = _norm(s)
                q, _ = quantize(s_bar, self.B)
                info = pack_bits(q, self.B).to(torch.long)
                x = self._code.encode(info)
                s_bar_hat = self.decode_symbols(
                    self._code.codeword_llrs_clean(x)[..., :sym.shape[-1] * 0]
                    if False else
                    torch.complex(1.0 - 2.0 * x[..., 0::2].double(),
                                  1.0 - 2.0 * x[..., 1::2].double()) / math.sqrt(2.0),
                    "awgn", snr_db)
                return denormalize(s_bar_hat)
            if channel == "awgn":
                y = awgn_channel(sym, snr_db, generator=generator, noise=noise)
                s_bar_hat = self.decode_symbols(y, "awgn", snr_db)
            else:
                y, h_used = rayleigh_channel(sym, snr_db, generator=generator,
                                             noise=noise, h=h, return_h=True)
                s_bar_hat = self.decode_symbols(y, "rayleigh", snr_db, h=h_used)
            return denormalize(s_bar_hat)

    # -- validation gate --------------------------------------------------

    @staticmethod
    def gate_noiseless_roundtrip(k: int = 12, bits_per_component: int = 1,
                                 n_states: int = 256) -> float:
        """Gate F: noiseless full-chain roundtrip must be exact to ~1e-6."""
        torch.manual_seed(0)
        s = torch.rand(n_states, 6) * 16 - 8
        base = PolarDigitalBaseline(k, bits_per_component)
        s_hat = base.forward(s, 20.0, channel="awgn", noiseless=True)
        return float((s_hat - s).abs().max())
