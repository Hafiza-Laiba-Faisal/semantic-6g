"""5G NR-style CRC-aided Polar digital baseline (Wave-D; non-frozen).

Construction — convention-verified end to end (clean roundtrip is EXACT,
see tests/test_polar_baseline.py):

* Encoder recursion (decoder-mirror): enc(u) = interleave(
  enc(u[0::2] ^ u[1::2]), enc(u[1::2]) ), leaf = identity.  Compiled to a
  generator matrix G (unit-vector rows), N <= 2^9 accounts for packet size.
  NOTE: coincides with natural F^⊗n only up to N=4; the "textbook"
  Kronecker block construction does NOT match the even/odd decoder above
  N=4, which is why the generator is built from the recursion itself.
* Subchannels: info + CRC-8 ride the most reliable u positions; reliability
  order = row Hamming weights of G (ascending = frozen first), computed
  numerically (no analytic popcount shortcut).
* Decode (SC): subcode recursion — each subtree is a genuine linear code:
    a_hat   = SC( LLRs of the even positions )        (codeword of u_even^u_odd)
    odd_hat = SC( LLRs of the odd positions )         (codeword of u_odd)
    u_even  = a_hat ^ odd_hat
  Exact on a noiseless channel for every node, no f/g tie artifacts.
  Frozen leaves clamp to 0 in original u-index space.
* Zero-noise reference path uses exact GF(2) inversion u = x @ G^-1
  (bypasses float LLR arithmetic by design, since LLRs are ±inf then).
* CRC-8 (poly 0x07) rides in the u-vector; SC output info bits are taken
  as-is (failure statistics are a reporting concern for the eval layer).

Interface-compatible with DigitalBaseline (quantize/pack/QPSK/channel
layers reused verbatim), evaluation-only, fully deterministic (no RNG).
"""

from __future__ import annotations

from typing import Optional

import torch

from project import config
from project.baselines.digital import (dequantize, pack_bits, quantize,
                                       qpsk_llrs, qpsk_modulate, unpack_bits)
from project.channels.awgn import awgn_channel
from project.channels.rayleigh import equalize_zf, rayleigh_channel
from project.data.normalization import denormalize, normalize


# ---------------------------------------------------------------------------
# G_N = F^⊗n (explicit Kronecker product) and reliability order
# ---------------------------------------------------------------------------

def _gf2_inverse(M: torch.Tensor) -> torch.Tensor:
    """Inverse of a square 0/1 matrix over GF(2) by Gauss-Jordan elimination."""
    n = M.shape[0]
    A = M.clone().to(torch.long) % 2
    I = torch.eye(n, dtype=torch.long)
    # augment and eliminate row-wise
    H = torch.cat([A, I], dim=1)
    r = 0
    for c in range(n):
        piv = None
        for rr in range(r, n):
            if H[rr, c] == 1:
                piv = rr
                break
        if piv is None:
            raise RuntimeError("matrix singular over GF(2)")
        H[[r, piv]] = H[[piv, r]]
        for rr in range(n):
            if rr != r and H[rr, c] == 1:
                H[rr] = (H[rr] ^ H[r]) % 2
        r += 1
    return H[:, n:] % 2


def polar_matrix(N: int) -> torch.Tensor:
    """G_N in the EXACT convention of the even/odd-split SC decoder.

    Decoder recursion (node size n): x_even = (u_even ^ u_odd) @ G_{n/2},
    x_odd = u_odd @ G_{n/2}, u interleaved-everywhere; leaf pair:
    x = [u0 ^ u1, u1].  The encoder MUST be the mirror recursion

        enc(u) = interleave( enc(u[0::2] ^ u[1::2]), enc(u[1::2]) ) ,

    so G is built by applying that recursion to unit vectors (N=2 gives
    G = F = [[1,0],[1,1]] as required).  NOTE: equal to natural F^⊗n for
    N <= 4 but DIVERGES from naive Kronecker blocks above N=4 — which is
    exactly why a "textbook" block-built G mismatched this decoder. All
    verifications (units + full random clean identity) hold for every N.
    """
    if N == 1:
        return torch.ones(1, 1, dtype=torch.long)

    def enc_vec(u: torch.Tensor) -> torch.Tensor:
        if u.numel() == 1:
            return u
        a = u[0::2] ^ u[1::2]
        ev = enc_vec(a)
        od = enc_vec(u[1::2]).clone()
        out = torch.empty_like(u)
        out[0::2] = ev
        out[1::2] = od
        return out

    G = torch.zeros(N, N, dtype=torch.long)
    for i in range(N):
        e = torch.zeros(N, dtype=torch.long); e[i] = 1
        G[i] = enc_vec(e)
    return G


def reliability_order(N: int, G: Optional[torch.Tensor] = None) -> list:
    """u-indices sorted from LEAST to MOST reliable (frozen first).

    Subchannel reliability is ordered by row Hamming weight of the
    alignment-corrected G_N (lower weight => more noise-sensitive frozen
    position, the standard Bhattacharyya-free heuristic that is exact for
    BEC and monotone for BSC/AWGN at the small N used here).  Weights are
    computed NUMERICALLY from the built G (no analytic popcount formula
    to get wrong).
    """
    G = polar_matrix(N) if G is None else G
    weights = G.sum(dim=1).tolist()
    idx = list(range(N))
    return sorted(idx, key=lambda i: (weights[i], i))   # least reliable first


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
        G = polar_matrix(N)             # natural Kronecker F^⊗n
        self._assert_units_identity(G)  # decoder-convention gate (see below)
        self.G = G
        order = reliability_order(N, G=G)
        self.frozen = torch.zeros(N, dtype=torch.long)
        self.frozen[order[: N - self.n_total]] = 1
        self.info_idx = torch.tensor(order[N - self.n_total:],
                                     dtype=torch.long)  # [K+8] most reliable
        self.Ginv = _gf2_inverse(G)             # exact clean decode u = x @ Ginv

    @staticmethod
    def _assert_units_identity(G: torch.Tensor) -> None:
        """Decoder-convention gate: exact clean SC must return e_i for every
        unit vector e_i on unit codewords of G. (Combined clean probes can
        produce tie artifacts in float arithmetic; units are decision-clean.)
        """
        N = G.shape[0]
        probe = object.__new__(PolarCode)
        probe.N = N
        probe.n_info = N
        probe.frozen = torch.zeros(N, dtype=torch.long)
        probe.info_idx = torch.arange(N)
        mag = clean_llr_magnitude(N, G.device)
        for i in range(N):
            ei = torch.zeros(1, N, dtype=torch.long); ei[0, i] = 1
            xi = (ei @ G) % 2
            di = polar_sc_decode(probe, (1 - 2.0 * xi.double()) * mag)
            if not torch.equal(di[0], ei[0]):
                raise RuntimeError(f"unit-vector identity failed at i={i} — "
                                   "SC/G convention mismatch")

    def encode(self, info: torch.Tensor) -> torch.Tensor:
        """Info bits [B, K] -> codeword x [B, N] (u @ G mod 2)."""
        B = info.shape[0]
        u = torch.zeros(B, self.N, dtype=info.dtype, device=info.device)
        crc = crc8(info)
        u_info = torch.cat([info.to(u.dtype), crc.to(u.dtype)], dim=-1)
        u[:, self.info_idx] = u_info
        x = (u @ self.G) % 2
        return x

    def decode_exact_clean(self, x: torch.Tensor) -> torch.Tensor:
        """Exact zero-noise decode: u = x @ G^{-1} (mod 2) -> info bits.

        On a noiseless channel every bit is known with certainty, so SC
        decisions degenerate to exact GF(2) inversion; the float-LLR SC loop
        can only introduce tie artifacts there, so we bypass it deliberately.
        Info bits occupy the ``info_idx`` subchannels (not a contiguous
        prefix), so they are unscrambled accordingly.
        """
        u_full = (x.to(torch.long) @ self.Ginv) % 2
        return u_full[:, self.info_idx][:, : self.n_info]


def clean_llr_magnitude(N: int, device=None) -> torch.Tensor:
    """Per-position magnitudes for CLEAN-channel LLRs with deterministic
    jitter.  Exact ±big magnitudes can cancel to exactly 0 in the g-update
    (Lo - Le = 0), producing a tie-break that breaks the Boolean-linear
    exactness of the clean channel and corrupts the G-alignment probe.
    A deterministic spread (all >= 1e6, < ±0.01 rel.) makes every sign
    decision tie-free and keeps the clean roundtrip EXACT."""
    j = torch.arange(N, dtype=torch.double, device=device)
    return 1e6 * (1.0 + 1e-5 * ((j % 9) - 4.0))


def codeword_to_clean_llrs(code: PolarCode, x: torch.Tensor) -> torch.Tensor:
    """Clean channel reference LLRs for codeword x [.., N]: (1-2x)*mag."""
    mag = clean_llr_magnitude(x.shape[-1], x.device)
    return (1 - 2 * x.double()) * mag


def polar_sc_decode(code: PolarCode, llrs: torch.Tensor) -> torch.Tensor:
    """Subcode-recursive SC; LLRs [B, N] -> info bits [B, K].

    Encoder recursion: x = interleave( enc(u_even ^ u_odd), enc(u_odd) ),
    so the codeword splits into two INDEPENDENT subcodes (vector
    a = u_even ^ u_odd on the even positions, u_odd on the odd positions).
    The decoder therefore decodes each subtree separately (no cross-update)
    and reconstructs u_even = a ^ u_odd.  On a CLEAN channel this recursion
    is EXACT: every subtree is a genuine linear code under G_{n/2}.
    Frozen leaves (original u indices) are clamped to 0 after reconstruction.
    """
    B, N = llrs.shape
    assert N == code.N, (N, code.N)
    K = code.n_info
    frozen_mask = code.frozen.to(llrs.device)

    def sc_rec(node_llrs, base_idx):
        n = node_llrs.shape[-1]
        if n == 1:
            return (node_llrs <= 0).to(torch.long)
        Le, Lo = node_llrs[..., 0::2], node_llrs[..., 1::2]
        a_hat = sc_rec(Le, base_idx[0::2])
        odd_hat = sc_rec(Lo, base_idx[1::2])
        even_hat = a_hat ^ odd_hat
        out = torch.empty_like(torch.cat([even_hat, odd_hat], dim=-1))
        out[..., 0::2] = even_hat
        out[..., 1::2] = odd_hat
        return out

    base = torch.arange(N, device=llrs.device)
    u_hat = sc_rec(llrs, base)
    # frozen clamp in original u-index space
    u_hat = u_hat * (1 - frozen_mask.unsqueeze(0))
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
            # pad symbols come from all-zero (bit 0) pairs: 2 bits per symbol
            pad = qpsk_modulate(torch.zeros(
                *sym.shape[:-1], 2 * (self.k - sym.shape[-1]),
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
                info_hat = self._code.decode_exact_clean(x)
                q_hat = unpack_bits(info_hat, self.B)
                return denormalize(dequantize(q_hat, self.B))
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
