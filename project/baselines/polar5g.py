"""5G NR polar chain (TS 38.212) as a digital baseline -- rate matched, CA-SCL.

Why this file exists
--------------------
The frozen digital comparator (``digital.py``) is a rate-1/2 K=7 convolutional
code; the A-11 comparator (``digital_polar.py``) is a polar code that is
"5G-style" only in spirit: its own reliability order, no rate matching, SC
decoding only.  A reviewer's fair question is what a *specified* classical
link does at the same payload and the same number of channel uses.  For
payloads of 6..48 bits, 5G NR itself uses polar coding (LDPC starts at much
larger blocks), so the honest comparator is the 5G NR polar chain:

    payload A -> CRC (6 or 11) -> K = A + L -> polar N = 2^n (5.3.1)
      -> sub-block interleaver (5.4.1.1) -> bit selection, i.e. puncturing or
      repetition to E bits (5.4.1.2) -> QPSK -> channel -> LLR
      -> repetition combining / erasure fill -> CA-SCL (list L) -> payload

Budget matching.  The frozen matrix uses ``k`` complex channel uses per state
and ``B`` bits per component, with the conv packet rule ``6B + 6 == k``.  This
chain uses exactly the same payload ``A = 6B`` and exactly the same number of
coded bits ``E = 2k`` (QPSK, 2 bits per symbol).  So payload, channel uses and
modulation are identical; only the code and the decoder differ.

Spec features implemented: the 1024-entry reliability sequence (Table
5.3.1.2-1), mother-code length selection (5.3.1), CRC-6 for A <= 19 and
CRC-11 for A >= 20 (5.1/5.2.1), the 32-column sub-block interleaver (Table
5.4.1.1-1), puncturing *and* repetition bit selection (5.4.1.2), and
CRC-aided list decoding.  Features deliberately *not* implemented, with the
reason each cannot move the BLER on our channels:

* input bit interleaver (5.3.1.1, IIL) -- a fixed permutation of the K bits;
  it changes which *payload* bit sits on which subchannel, not the code;
* channel interleaver (5.4.1.3, IBIL) -- a fixed permutation of the E bits
  before modulation; it only re-maps burst errors, and our channels are AWGN
  and block-fading Rayleigh (no bursts);
* PC (parity-check) bits -- used only for downlink DCI, not UCI;
* segmentation -- not needed for A <= 48.

The A = 6 cell (k = 12, B = 1) is below the 5G polar minimum (A >= 12); the
same chain is run there and that cell is labelled a non-spec extension.

Decoding is CA-SCL with list size L = 8 and the exact (tanh) LLR recursion;
the path metric is the standard log-domain metric.  If no path passes CRC the
best-metric path is returned (the decoder must always emit a state; the SR
metric counts that as a normal trial, so no favourable selection happens).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch

from project import config
from project.baselines.digital import (dequantize, pack_bits, qpsk_llrs,
                                       qpsk_modulate, quantize, unpack_bits)
from project.baselines.polar5g_tables import (CRC_POLY, SUB_BLOCK_PATTERN,
                                              reliability_order)
from project.channels.awgn import awgn_channel
from project.channels.rayleigh import equalize_zf, rayleigh_channel
from project.data.normalization import denormalize, normalize

LIST_SIZE = 8


# ---------------------------------------------------------------------------
# CRC
# ---------------------------------------------------------------------------

def crc_len_for(a_bits: int) -> int:
    """TS 38.212 uplink rule: CRC-6 for A <= 19, CRC-11 for A >= 20."""
    return 6 if a_bits <= 19 else 11


def _crc_remainder(bits: np.ndarray, crc_len: int) -> np.ndarray:
    """Remainder of ``bits`` (M, K) times D^crc_len, modulo g(D).

    MSB-first bit-serial division; no reflection, no final XOR (the 5G
    generator is used exactly as tabulated in TS 38.212 5.1).
    """
    poly, deg = CRC_POLY[crc_len]
    assert deg == crc_len
    taps = np.array([(poly >> (deg - 1 - j)) & 1 for j in range(deg)],
                    dtype=np.int8)
    reg = np.zeros((bits.shape[0], deg), dtype=np.int8)
    for b in range(bits.shape[1]):
        fb = bits[:, b] ^ reg[:, 0]
        reg = np.concatenate([reg[:, 1:], np.zeros((reg.shape[0], 1),
                                                   dtype=np.int8)], axis=1)
        reg ^= fb[:, None] * taps
    return reg


def crc_attach(a_bits: np.ndarray, crc_len: int) -> np.ndarray:
    """``a_bits`` (M, A) -> (M, A + crc_len) with the CRC appended."""
    rem = _crc_remainder(a_bits, crc_len)
    return np.concatenate([a_bits, rem], axis=1).astype(np.int8)


def crc_passes(k_bits: np.ndarray, crc_len: int) -> np.ndarray:
    """True where the (M, K) word, payload + CRC, is a valid codeword."""
    return ~_crc_remainder(k_bits, crc_len).any(axis=1)


# ---------------------------------------------------------------------------
# Polar transform, length selection, rate matching
# ---------------------------------------------------------------------------

def mother_length(k_bits: int, e_bits: int, n_max: int = 10,
                  n_min: int = 5) -> int:
    """TS 38.212 5.3.1: n = max(min(n1, n2, n_max), n_min), N = 2^n."""
    if e_bits < 1:
        raise ValueError("E must be positive")
    base = math.ceil(math.log2(e_bits))
    if (e_bits <= (9.0 / 8.0) * 2 ** (base - 1)
            and k_bits / e_bits < 9.0 / 16.0):
        n1 = base - 1
    else:
        n1 = base
    n2 = math.ceil(math.log2(k_bits / (1.0 / 8.0)))
    return max(min(n1, n2, n_max), n_min)


def polar_transform(u: np.ndarray) -> np.ndarray:
    """d = u G_N with G_N = F^{(x)n}, F = [[1,0],[1,1]] (5.3.1.2).

    Butterfly: stage blocks from the largest half down to 1; in each block the
    first half is XORed with the second half.  Verified against a Kronecker
    product of F in tests/test_polar5g.py.
    """
    x = u.copy()
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError("N must be a power of two")
    half = n >> 1
    while half >= 1:
        for start in range(0, n, 2 * half):
            x[..., start:start + half] ^= x[..., start + half:start + 2 * half]
        half >>= 1
    return x


def sub_block_perm(n: int) -> np.ndarray:
    """TS 38.212 5.4.1.1 index map: y_i = d_{perm[i]}."""
    if n % 32:
        raise ValueError("N must be a multiple of 32")
    block = n // 32
    i = np.arange(n)
    col = (i * 32) // n
    row = i % block
    return np.asarray(SUB_BLOCK_PATTERN, dtype=np.int64)[col] * block + row


def rate_match_mode(k_bits: int, n: int, e_bits: int) -> str:
    """TS 38.212 5.4.1.2 mode: repetition, puncturing or shortening."""
    if e_bits >= n:
        return "repetition"
    if k_bits / e_bits <= 7.0 / 16.0:
        return "puncturing"
    return "shortening"


def rate_match_indices(n: int, e_bits: int, mode: str) -> np.ndarray:
    """Indices into the sub-block-interleaved word y for the E output bits."""
    if mode == "repetition":
        return np.arange(e_bits) % n
    if mode == "puncturing":
        return np.arange(e_bits) + (n - e_bits)
    if mode == "shortening":
        return np.arange(e_bits)
    raise ValueError(f"unknown rate-matching mode {mode!r}")


def pre_frozen_positions(k_bits: int, n: int, e_bits: int) -> np.ndarray:
    """Subchannels frozen *by the rate-matching scheme* (38.212 5.3.1.2).

    These are the positions whose code bits are not transmitted (puncturing)
    or are forced to zero (shortening); they may never carry information.
    """
    mode = rate_match_mode(k_bits, n, e_bits)
    if mode == "repetition":
        return np.empty(0, dtype=np.int64)
    if mode == "puncturing":
        n_int = 32 * math.ceil((n - e_bits) / 32)
        pattern = sub_block_perm(n_int)
        pre = list(int(x) for x in pattern[:n - e_bits])
        if e_bits >= 3 * n / 4:
            t = int(math.ceil(3.0 / 4.0 * n - e_bits / 2.0)) - 1
        else:
            t = int(math.ceil(9.0 / 16.0 * n - e_bits / 4.0)) - 1
        pre.extend(range(max(0, t)))
        return np.array(sorted(set(pre)), dtype=np.int64)
    pattern = sub_block_perm(n)
    return np.array([int(x) for x in pattern[e_bits:]], dtype=np.int64)


# ---------------------------------------------------------------------------
# The code block: encode + CA-SCL decode
# ---------------------------------------------------------------------------

def _f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact f: 2 atanh(tanh(a/2) tanh(b/2)), numerically stable."""
    return (np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
            + np.log1p(np.exp(-np.abs(a + b)))
            - np.log1p(np.exp(-np.abs(a - b))))


def _g(a: np.ndarray, b: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Exact g: b + (1 - 2u) a with u in {0, 1}."""
    return b + (1.0 - 2.0 * u) * a


class Polar5GCode:
    """One polar code block: K = A + CRC bits, E rate-matched bits."""

    def __init__(self, a_bits: int, e_bits: int, list_size: int = LIST_SIZE,
                 crc_len: Optional[int] = None) -> None:
        self.a_bits = a_bits
        self.e_bits = e_bits
        self.crc_len = crc_len_for(a_bits) if crc_len is None else crc_len
        self.k_bits = a_bits + self.crc_len
        if self.k_bits > e_bits:
            raise ValueError(f"K={self.k_bits} exceeds E={e_bits}")
        self.n = mother_length(self.k_bits, e_bits)
        self.N = 1 << self.n
        if self.N < 32:
            raise ValueError("N below the 5G minimum")
        self.mode = rate_match_mode(self.k_bits, self.N, e_bits)
        self.pre_frozen = pre_frozen_positions(self.k_bits, self.N, e_bits)
        # information set: the K most reliable subchannels that are not
        # frozen by the rate-matching scheme (38.212 5.3.1.2)
        blocked = set(int(i) for i in self.pre_frozen)
        allowed = [int(q) for q in reliability_order(self.N)
                   if int(q) not in blocked]
        if len(allowed) < self.k_bits:
            raise ValueError("not enough unfrozen subchannels")
        self.info_pos = np.array(sorted(allowed[-self.k_bits:]),
                                 dtype=np.int64)
        self.perm = sub_block_perm(self.N)
        self.sel = rate_match_indices(self.N, e_bits, self.mode)
        self.repeated = self.mode == "repetition"
        self.list_size = int(list_size)
        if self.list_size < 1:
            raise ValueError("list size must be >= 1")

    # -- transmit --------------------------------------------------------

    def encode(self, payload: np.ndarray) -> np.ndarray:
        """``payload`` (M, A) bits -> rate-matched codeword bits (M, E)."""
        payload = np.asarray(payload, dtype=np.int8)
        if payload.ndim != 2 or payload.shape[1] != self.a_bits:
            raise ValueError(f"payload must be (M, {self.a_bits})")
        k_bits = crc_attach(payload, self.crc_len)
        u = np.zeros((payload.shape[0], self.N), dtype=np.int8)
        u[:, self.info_pos] = k_bits
        d = polar_transform(u)
        y = d[:, self.perm]
        return y[:, self.sel]

    # -- receive ---------------------------------------------------------

    def _llrs_to_mother(self, llr_e: np.ndarray) -> np.ndarray:
        """E LLRs -> N LLRs (sum repeated copies, erase punctured ones)."""
        m = llr_e.shape[0]
        y = np.zeros((m, self.N), dtype=np.float64)
        if self.mode == "repetition":
            np.add.at(y, (slice(None), self.sel), llr_e)
        elif self.mode == "puncturing":
            y[:, self.sel] = llr_e
        else:  # shortening: the dropped code bits are known zeros
            y[:, :self.e_bits] = llr_e
            y[:, self.e_bits:] = 1e6
        # undo the sub-block interleaver: y_i = d_{perm[i]}
        out = np.zeros((m, self.N), dtype=np.float64)
        out[:, self.perm] = y
        return out

    def decode(self, llr_e: np.ndarray, chunk: int = 512) -> np.ndarray:
        """``llr_e`` (M, E), LLR = log P(0)/P(1) -> payload (M, A).

        The SCL engine is batched over codewords, so a whole evaluation cell
        is a handful of vectorised passes instead of one Python recursion per
        episode.  ``chunk`` bounds the peak memory of the per-level state.
        """
        llr_n = self._llrs_to_mother(np.asarray(llr_e, dtype=np.float64))
        out = np.empty((llr_n.shape[0], self.a_bits), dtype=np.int8)
        for start in range(0, llr_n.shape[0], chunk):
            stop = min(start + chunk, llr_n.shape[0])
            out[start:stop] = _scl_decode(llr_n[start:stop], self.info_pos,
                                          self.N, self.k_bits, self.crc_len,
                                          self.list_size)
        return out

    # -- diagnostics -----------------------------------------------------

    def params(self) -> dict:
        return {"a_bits": self.a_bits, "crc_len": self.crc_len,
                "k_bits": self.k_bits, "e_bits": self.e_bits,
                "n": self.n, "N": self.N, "rate": self.k_bits / self.e_bits,
                "mode": self.mode,
                "n_pre_frozen": int(self.pre_frozen.size),
                "list_size": self.list_size}


def _scl_decode(llr: np.ndarray, info_pos: np.ndarray, n_len: int,
                k_bits: int, crc_len: int, list_size: int) -> np.ndarray:
    """Batched CRC-aided SCL.

    ``llr`` (M, N) -> payload bits (M, A).  The list is grown the canonical
    way: it starts with the single empty path and doubles at every
    information bit until it reaches L, so every stored path has a distinct
    information-bit history.  (Seeding L identical copies instead would make
    all paths stay identical and silently reduce SCL to SC.)  At each split
    the 2L candidates are ranked by the exact path metric and the surviving
    paths are copied with a permutation applied to all in-flight state (the
    Tal-Vardy copy-path operation).
    """
    m_batch, n = llr.shape
    if n != n_len:
        raise ValueError("LLR length does not match N")
    levels = int(round(math.log2(n)))
    if 1 << levels != n:
        raise ValueError("N must be a power of two")
    L = int(list_size)
    bidx = np.arange(m_batch)[:, None]

    alpha: List[Optional[np.ndarray]] = [None] * (levels + 1)
    beta: List[Optional[np.ndarray]] = [None] * (levels + 1)
    left: List[Optional[np.ndarray]] = [None] * (levels + 1)

    alpha[levels] = llr[:, None, :]            # one path
    pm = np.zeros((m_batch, 1), dtype=np.float64)
    u = np.zeros((m_batch, 1, n), dtype=np.int8)
    active = [1]                               # mutable: list width

    frozen_set = set(range(n)) - set(int(i) for i in info_pos)
    counter = [0]  # depth-first leaf index, left to right

    def rec(level: int) -> None:
        nonlocal u, pm
        width = 1 << level
        if width == 1:
            i = counter[0]
            counter[0] = i + 1
            if i in frozen_set:
                new_bit = np.zeros((m_batch, active[0]), dtype=np.int8)
                u[:, :, i] = new_bit
                beta[0] = new_bit[..., None]
                return
            a = alpha[0][..., 0]               # (M, active)
            c0 = np.logaddexp(0.0, -a)
            c1 = np.logaddexp(0.0, a)
            cand = np.stack([pm + c0, pm + c1],
                            axis=-1).reshape(m_batch, 2 * active[0])
            take = min(2 * active[0], L)
            order = np.argsort(cand, axis=1, kind="stable")[:, :take]
            pm = np.take_along_axis(cand, order, axis=1)
            new_bit = (order & 1).astype(np.int8)
            for d in range(1, levels + 1):
                if alpha[d] is not None:
                    alpha[d] = np.repeat(alpha[d], 2, axis=1)[bidx, order]
            for d in range(0, levels):
                if beta[d] is not None:
                    beta[d] = np.repeat(beta[d], 2, axis=1)[bidx, order]
            for d in range(1, levels + 1):
                if left[d] is not None:
                    left[d] = np.repeat(left[d], 2, axis=1)[bidx, order]
            u = np.repeat(u, 2, axis=1)[bidx, order]
            u[:, :, i] = new_bit
            beta[0] = new_bit[..., None]
            active[0] = take
            return

        half = width >> 1
        a = alpha[level]
        alpha[level - 1] = _f(a[..., :half], a[..., half:])
        rec(level - 1)
        left[level] = beta[level - 1].copy()
        a = alpha[level]
        alpha[level - 1] = _g(a[..., :half], a[..., half:], left[level])
        rec(level - 1)
        right = beta[level - 1]
        beta[level] = np.concatenate([left[level] ^ right, right], axis=-1)

    rec(levels)

    words = u[:, :, info_pos]                       # (M, active, K)
    flat = words.reshape(m_batch * active[0], k_bits)
    ok = crc_passes(flat, crc_len).reshape(m_batch, active[0])
    any_ok = ok.any(axis=1)
    score = np.where(ok, pm, np.inf)
    choice = np.argmin(score, axis=1)
    choice = np.where(any_ok, choice, np.argmin(pm, axis=1))
    picked = flat.reshape(m_batch, active[0], k_bits)[np.arange(m_batch),
                                                      choice]
    return picked[:, :k_bits - crc_len].astype(np.int8)


# ---------------------------------------------------------------------------
# Digital baseline with the 5G polar chain (same outer pipeline)
# ---------------------------------------------------------------------------

class Polar5GBaseline:
    """Same interface as ``DigitalBaseline`` / ``PolarDigitalBaseline``."""

    def __init__(self, k: int, bits_per_component: int,
                 list_size: int = LIST_SIZE) -> None:
        self.pkt = config.digital_packet(k, bits_per_component)
        self.k = k
        self.B = bits_per_component
        self.a_bits = self.pkt["n_src_bits"]           # 6B payload
        self.e_bits = 2 * k                            # QPSK, E = 2k
        self.code = Polar5GCode(self.a_bits, self.e_bits, list_size=list_size)

    def encode_state(self, s: torch.Tensor) -> torch.Tensor:
        s_bar = normalize(s)
        q, _ = quantize(s_bar, self.B)
        info = pack_bits(q, self.B).numpy()            # (M, A)
        e = self.code.encode(info)                     # (M, E)
        bits = torch.from_numpy(e.astype(np.int64))
        sym = qpsk_modulate(bits)                      # (M, E/2 = k)
        assert sym.shape[-1] == self.k, (sym.shape[-1], self.k)
        return sym

    def decode_symbols(self, y: torch.Tensor, channel: str, snr_db: float,
                       h: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        llr = qpsk_llrs(y_pay, sigma2_eff).double().numpy()
        info = self.code.decode(llr)
        q_hat = unpack_bits(torch.from_numpy(info.astype(np.int64)), self.B)
        return dequantize(q_hat, self.B)

    def forward(self, s: torch.Tensor, snr_db: float, channel: str = "awgn",
                generator: Optional[torch.Generator] = None,
                noiseless: bool = False,
                noise: Optional[torch.Tensor] = None,
                h: Optional[torch.Tensor] = None) -> torch.Tensor:
        if channel not in ("awgn", "rayleigh"):
            raise ValueError(f"unknown channel {channel!r}")
        with torch.no_grad():
            if noiseless:
                s_bar = normalize(s)
                q, _ = quantize(s_bar, self.B)
                info = self.code.encode(pack_bits(q, self.B).numpy())
                # exact LLRs of the transmitted bits, then decode
                llr = np.where(info == 0, 1e6, -1e6).astype(np.float64)
                dec = self.code.decode(llr)
                q_hat = unpack_bits(torch.from_numpy(dec.astype(np.int64)),
                                    self.B)
                return denormalize(dequantize(q_hat, self.B))
            sym = self.encode_state(s)
            if channel == "awgn":
                y = awgn_channel(sym, snr_db, generator=generator, noise=noise)
                s_bar_hat = self.decode_symbols(y, "awgn", snr_db)
            else:
                y, h_used = rayleigh_channel(sym, snr_db, generator=generator,
                                             noise=noise, h=h, return_h=True)
                s_bar_hat = self.decode_symbols(y, "rayleigh", snr_db, h=h_used)
            return denormalize(s_bar_hat)

    @staticmethod
    def gate_noiseless_roundtrip(k: int = 12, bits_per_component: int = 1,
                                 n_states: int = 64) -> float:
        torch.manual_seed(0)
        s = torch.rand(n_states, 6) * 16 - 8
        base = Polar5GBaseline(k, bits_per_component)
        s_hat = base.forward(s, 20.0, channel="awgn", noiseless=True)
        return float((s_hat - s).abs().max())
