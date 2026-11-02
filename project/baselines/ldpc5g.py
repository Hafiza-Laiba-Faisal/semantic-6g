"""5G NR LDPC, base graph 2 (3GPP TS 38.212), A-20.

WHAT THIS IS
------------
A real LDPC comparator for the frozen digital framing, built on the base graph
that 5G NR actually specifies:

* base graph **2**, 42 x 52, shift coefficients from TS 38.212 Table 5.3.2-3,
  vendored verbatim in ``ldpc5g_bg2.csv`` (copied from the py3gpp data file,
  which reproduces the ETSI table);
* the eight lifting sets of Table 5.3.2-1, ``Kb = 10`` for K <= 640,
  ``Z = min{z in the sets : 10z >= A}``, ``K = 10Z``, ``N = 52Z``;
* quasi-cyclic lifting ``P_ij = I(c_ij)`` with the shift applied as a cyclic
  row permutation;
* systematic encoding by a GF(2) solve of the parity block,
  ``H_p c_p^T = H_s c_s^T``;
* filler bits (the ``K - A`` systematic positions when ``K < Kb*Z``) are
  NULL: they are **not transmitted**, exactly as in 38.212 rate matching;
* offset min-sum belief propagation, 15 iterations, offset 0.5.

WHAT THIS IS NOT
----------------
The 38.212 rate-matching chain (circular buffer, bit interleaving, redundancy
version), the transport-block CRC and the LDPC base-graph *selection* are not
implemented.  Transmission takes the first ``E`` non-NULL mother bits
(systematic payload then parity).  Every table and figure that reports this
arm says "5G NR LDPC BG2 without the 38.212 rate-matching chain"; it is never
called "full 5G NR LDPC".  This is the same honesty level as A-11, which is
"CRC-8 polar with SC decoding and no rate matching".

CONVENTIONS
-----------
LLRs follow ``project.baselines.digital.qpsk_llrs``: ``LLR = log P(0)/P(1)``,
so a positive LLR favours bit 0.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from project import config
from project.baselines.digital import (dequantize, pack_bits, qpsk_llrs,
                                       qpsk_modulate, quantize, unpack_bits)
from project.channels.awgn import awgn_channel
from project.channels.rayleigh import equalize_zf, rayleigh_channel
from project.data.normalization import denormalize, normalize

_CSV = Path(__file__).with_name("ldpc5g_bg2.csv")

# TS 38.212 Table 5.3.2-1: the eight lifting sets.  ``i_ls`` is the index used
# in the base-graph table; ``Z`` must be a member of set ``i_ls``.
LIFTING_SETS = (
    (2, 4, 8, 16, 32, 64, 128, 256),
    (3, 6, 12, 24, 48, 96, 192, 384),
    (5, 10, 20, 40, 80, 160, 320),
    (7, 14, 28, 56, 112, 224),
    (9, 18, 36, 72, 144, 288),
    (11, 22, 44, 88, 176, 352),
    (13, 26, 52, 104, 208),
    (15, 30, 60, 120, 240),
)

BG2_ROWS = 42
BG2_COLS = 52
KB = 10                    # BG2 systematic columns for K <= 640


def _load_base_graph(path: Path = _CSV):
    """Parse the vendored CSV into ``{(row, col): (c_0 .. c_7)}``."""
    entries = {}
    row = None
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        fields = [f.strip() for f in line.split(";")]
        if len(fields) < 10:
            raise ValueError(f"malformed base-graph line: {line!r}")
        if fields[0]:
            if not fields[0].isdigit():
                continue                      # header rows
            row = int(fields[0])
        if row is None or not fields[1].isdigit():
            continue
        col = int(fields[1])
        shifts = tuple(int(v) for v in fields[2:10])
        entries[(row, col)] = shifts
    if len(entries) != 197:
        raise ValueError(f"BG2 should have 197 non-zero entries, got "
                         f"{len(entries)}")
    return entries


BASE_GRAPH = _load_base_graph()


def lifting_size(k_bits: int) -> tuple[int, int]:
    """Smallest valid ``(Z, i_ls)`` with ``KB * Z >= k_bits``.

    TS 38.212 picks the smallest lifting size in Table 5.3.2-1 that satisfies
    ``Kb * Z >= K``, not the first set that happens to contain one.
    """
    best = None
    for i_ls, zset in enumerate(LIFTING_SETS):
        for z in zset:
            if KB * z >= k_bits and (best is None or z < best[0]):
                best = (z, i_ls)
    if best is None:
        raise ValueError(f"A = {k_bits} is too large for base graph 2")
    return best


def _row_bit(row: Sequence[int]) -> int:
    """Pack a 0/1 sequence into an int (index 0 = most significant)."""
    value = 0
    for bit in row:
        value = (value << 1) | (1 if bit else 0)
    return value


class _GF2:
    """Tiny GF(2) linear algebra on packed ints."""

    @staticmethod
    def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve ``a X = b`` over GF(2); ``a`` must be square and invertible.

        ``a``: ``[n, n]`` 0/1, ``b``: ``[n, m]`` 0/1.  Returns ``X`` ``[n, m]``.
        """
        n = a.shape[0]
        m = b.shape[1]
        width = n + m
        rows = []
        for i in range(n):
            packed = 0
            for j in range(n):
                if a[i, j]:
                    packed |= 1 << (width - 1 - j)
            for j in range(m):
                if b[i, j]:
                    packed |= 1 << (m - 1 - j)
            rows.append(packed)

        for col in range(n):
            pivot = None
            bit = 1 << (width - 1 - col)
            for r in range(col, n):
                if rows[r] & bit:
                    pivot = r
                    break
            if pivot is None:
                raise ValueError("parity block is singular")
            rows[col], rows[pivot] = rows[pivot], rows[col]
            for r in range(n):
                if r != col and (rows[r] & bit):
                    rows[r] ^= rows[col]

        out = np.zeros((n, m), dtype=np.uint8)
        for i in range(n):
            low = rows[i] & ((1 << m) - 1)
            for j in range(m):
                if low & (1 << (m - 1 - j)):
                    out[i, j] = 1
        return out


class LDPC5GCode:
    """Base-graph-2 LDPC for a fixed ``(A, E)`` packet pair."""

    def __init__(self, a_bits: int, e_bits: int, max_iter: int = 15,
                 offset: float = 0.5) -> None:
        if a_bits < 1:
            raise ValueError("a_bits must be positive")
        self.a_bits = int(a_bits)
        self.e_bits = int(e_bits)
        self.max_iter = int(max_iter)
        self.offset = float(offset)

        self.z, self.i_ls = lifting_size(self.a_bits)
        self.k_bits = KB * self.z
        self.n_bits = BG2_COLS * self.z
        self.n_filler = self.k_bits - self.a_bits
        if self.n_filler < 0:
            raise ValueError("lifting produced K < A")

        self._build_encoder()
        self._build_decoder()

    # -- construction ------------------------------------------------------
    def _shift(self, row: int, col: int) -> Optional[int]:
        shifts = BASE_GRAPH.get((row, col))
        if shifts is None:
            return None
        return shifts[self.i_ls]

    def _build_encoder(self):
        par = BG2_COLS - KB            # 42 parity columns
        h_s = np.zeros((BG2_ROWS * self.z, KB * self.z), dtype=np.uint8)
        h_p = np.zeros((BG2_ROWS * self.z, par * self.z), dtype=np.uint8)
        for (i, j), _ in BASE_GRAPH.items():
            c = self._shift(i, j)
            for z in range(self.z):
                r = i * self.z + z
                if j < KB:
                    h_s[r, j * self.z + (z + c) % self.z] = 1
                else:
                    h_p[r, (j - KB) * self.z + (z + c) % self.z] = 1
        self.h_s = h_s
        self.h_p = h_p
        self.g = _GF2.solve(h_p, h_s.astype(np.uint8))     # [42Z, 10Z]

    def _build_decoder(self):
        edges = []
        for (i, j), _ in BASE_GRAPH.items():
            c = self._shift(i, j)
            for z in range(self.z):
                check = i * self.z + z
                var = j * self.z + (z + c) % self.z
                edges.append((check, var))
        edges.sort()
        self.edges = edges
        n_edges = len(edges)
        self.n_edges = n_edges
        self.edge_check = np.array([e[0] for e in edges], dtype=np.int64)
        self.edge_var = np.array([e[1] for e in edges], dtype=np.int64)

        n_checks = BG2_ROWS * self.z
        counts = np.bincount(self.edge_check, minlength=n_checks)
        max_deg = int(counts.max())
        self.max_deg = max_deg
        self.check_edges = np.full((n_checks, max_deg), n_edges, dtype=np.int64)
        self.check_used = np.zeros((n_checks, max_deg), dtype=bool)
        ptr = np.zeros(n_checks, dtype=np.int64)
        for e, c in enumerate(self.edge_check):
            self.check_edges[c, ptr[c]] = e
            self.check_used[c, ptr[c]] = True
            ptr[c] += 1

        n_vars = self.n_bits
        vcounts = np.bincount(self.edge_var, minlength=n_vars)
        vmax = int(vcounts.max()) if vcounts.size else 1
        self.var_edges = np.full((n_vars, vmax), n_edges, dtype=np.int64)
        self.var_used = np.zeros((n_vars, vmax), dtype=bool)
        vptr = np.zeros(n_vars, dtype=np.int64)
        for e, v in enumerate(self.edge_var):
            self.var_edges[v, vptr[v]] = e
            self.var_used[v, vptr[v]] = True
            vptr[v] += 1

        # transmitted positions: everything except the leading filler bits
        self.tx_positions = np.arange(self.n_filler, self.n_bits)
        if self.tx_positions.size < self.e_bits:
            raise ValueError("packet needs more bits than the mother code has")

    # -- encoding ----------------------------------------------------------
    def full_codeword(self, payload: np.ndarray) -> np.ndarray:
        """Mother codeword ``[M, N]`` (payload + filler + parity)."""
        payload = np.asarray(payload, dtype=np.uint8)
        if payload.ndim == 1:
            payload = payload[None, :]
        if payload.shape[1] != self.a_bits:
            raise ValueError(f"payload width {payload.shape[1]} != "
                             f"{self.a_bits}")
        m = payload.shape[0]
        c_sys = np.zeros((m, self.k_bits), dtype=np.uint8)
        c_sys[:, self.n_filler:] = payload
        parity = (c_sys @ self.g.T) & 1
        return np.concatenate([c_sys, parity], axis=1)

    def encode(self, payload: np.ndarray) -> np.ndarray:
        """``payload [M, A]`` 0/1 -> transmitted bits ``[M, E]``."""
        codeword = self.full_codeword(payload)
        return codeword[:, self.tx_positions[:self.e_bits]]

    def parity_ok(self, codeword: np.ndarray) -> bool:
        """Full mother-codeword parity check (``H c^T = 0``)."""
        c = np.asarray(codeword, dtype=np.uint8)
        h = np.concatenate([self.h_s, self.h_p], axis=1)
        return not np.any((h @ c) & 1)

    # -- decoding ----------------------------------------------------------
    def decode(self, llr: np.ndarray, chunk: int = 256) -> np.ndarray:
        """``llr [M, E]`` -> payload bits ``[M, A]``."""
        llr = np.asarray(llr, dtype=np.float32)
        if llr.ndim == 1:
            llr = llr[None, :]
        if llr.shape[1] != self.e_bits:
            raise ValueError(f"llr width {llr.shape[1]} != {self.e_bits}")
        m = llr.shape[0]
        out = np.zeros((m, self.a_bits), dtype=np.uint8)
        for s in range(0, m, chunk):
            e = min(m, s + chunk)
            out[s:e] = self._decode_chunk(llr[s:e])
        return out

    def _decode_chunk(self, llr_e: np.ndarray) -> np.ndarray:
        m = llr_e.shape[0]
        full = np.zeros((m, self.n_bits), dtype=np.float32)
        full[:, :self.n_filler] = 1e6          # filler bits are known zeros
        full[:, self.tx_positions[:self.e_bits]] = llr_e
        post = full.copy()

        ce = self.check_edges
        used = self.check_used
        n_edges = self.n_edges
        c2v = np.zeros((m, n_edges), dtype=np.float32)
        big = np.float32(1e6)

        for iteration in range(self.max_iter):
            v2c = post[:, self.edge_var] - c2v
            # unused slots carry the sentinel index n_edges; pad one zero
            v2c_pad = np.concatenate(
                [v2c, np.zeros((v2c.shape[0], 1), dtype=v2c.dtype)], axis=1)
            # gather per check, padded: [M, C, D]
            g = np.take(v2c_pad, ce, axis=1)
            mag = np.abs(g)
            mag = np.where(used[None, :, :], mag, big)
            signs = np.sign(g)
            signs[signs == 0] = 1.0
            sign_prod = np.prod(signs, axis=2, keepdims=True)

            idx = np.argmin(mag, axis=2)
            min1 = np.take_along_axis(mag, idx[:, :, None], axis=2)
            tmp = mag.copy()
            np.put_along_axis(tmp, idx[:, :, None], big, axis=2)
            min2 = tmp.min(axis=2, keepdims=True)
            # the edge that IS the minimum must use the second minimum
            min_val = np.where(
                np.arange(self.max_deg)[None, None, :] == idx[:, :, None],
                min2, min1)

            msg = sign_prod * signs * np.maximum(min_val - self.offset, 0.0)
            msg = np.where(used[None, :, :], msg, 0.0)
            c2v = np.take(msg.reshape(m, -1), self._check_flat, axis=1)

            post = full + self._sum_c2v(c2v)
            hard = (post < 0).astype(np.uint8)
            if iteration % 2 == 1 and self._syndrome_ok(hard):
                break

        return hard[:, self.n_filler:self.k_bits]

    def _syndrome_ok(self, hard: np.ndarray) -> bool:
        """All check nodes satisfied?  Unused slots gather a zero column."""
        pad = np.zeros((hard.shape[0], self.n_edges - self.n_bits + 1),
                       dtype=np.uint8)
        padded = np.concatenate([hard, pad], axis=1)
        gathered = np.take(padded, self.check_edges, axis=1)
        return not np.any(gathered.sum(axis=2) & 1)

    # helpers built lazily in _build_decoder's first use
    @property
    def _check_flat(self):
        flat = getattr(self, "_flat_cache", None)
        if flat is None:
            flat = np.ravel_multi_index(
                (self.edge_check, self._check_slot), self.check_edges.shape)
            self._flat_cache = flat
        return flat

    @property
    def _check_slot(self):
        slot = getattr(self, "_slot_cache", None)
        if slot is None:
            slot = np.zeros(self.n_edges, dtype=np.int64)
            ptr = np.zeros(self.check_edges.shape[0], dtype=np.int64)
            for e, c in enumerate(self.edge_check):
                slot[e] = ptr[c]
                ptr[c] += 1
            self._slot_cache = slot
        return slot

    def _sum_c2v(self, c2v: np.ndarray) -> np.ndarray:
        out = np.zeros((c2v.shape[0], self.n_bits), dtype=np.float32)
        padded = np.concatenate(
            [c2v, np.zeros((c2v.shape[0], 1), dtype=c2v.dtype)], axis=1)
        for slot in range(self.var_edges.shape[1]):
            idx = self.var_edges[:, slot]
            mask = self.var_used[:, slot]
            if not mask.any():
                continue
            contrib = padded[:, idx]
            contrib = np.where(mask[None, :], contrib, 0.0)
            out += contrib
        return out

    def _all_checks_ok(self, hard: np.ndarray) -> bool:
        return self._syndrome_ok(hard)

    def params(self) -> dict:
        return {
            "standard": "3GPP TS 38.212",
            "base_graph": 2,
            "a_bits": self.a_bits,
            "e_bits": self.e_bits,
            "z": self.z,
            "i_ls": self.i_ls,
            "k_bits": self.k_bits,
            "n_bits": self.n_bits,
            "n_filler": self.n_filler,
            "n_edges": self.n_edges,
            "max_iter": self.max_iter,
            "offset": self.offset,
            "rate_matching_chain": False,
            "crc": False,
            "selection": "first E non-NULL mother bits",
        }


# ---------------------------------------------------------------------------
# baseline wrapper - same interface as DigitalBaseline / Polar5GBaseline
# ---------------------------------------------------------------------------

class LDPC5GBaseline:
    """5G NR LDPC (BG2) at the frozen packet size, A = 6B, E = 2k.

    Drop-in replacement for ``DigitalBaseline`` / ``Polar5GBaseline``: it
    consumes the same normalised state, produces the same ``k`` QPSK symbols
    and returns a denormalised estimate.  Only the channel code differs.
    """

    def __init__(self, k: int, bits_per_component: int,
                 max_iter: int = 15, offset: float = 0.5) -> None:
        self.pkt = config.digital_packet(k, bits_per_component)
        self.k = k
        self.B = bits_per_component
        self.a_bits = self.pkt["n_src_bits"]      # 6B payload
        self.e_bits = 2 * k                       # QPSK, E = 2k
        self.code = LDPC5GCode(self.a_bits, self.e_bits,
                               max_iter=max_iter, offset=offset)

    def encode_state(self, s: torch.Tensor) -> torch.Tensor:
        s_bar = normalize(s)
        q, _ = quantize(s_bar, self.B)
        info = pack_bits(q, self.B).numpy()
        e = self.code.encode(info)
        bits = torch.from_numpy(e.astype(np.int64))
        sym = qpsk_modulate(bits)
        if sym.shape[-1] != self.k:
            raise AssertionError((sym.shape[-1], self.k))
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
                s_bar_hat = self.decode_symbols(y, "rayleigh", snr_db,
                                                h=h_used)
            return denormalize(s_bar_hat)

    @staticmethod
    def gate_noiseless_roundtrip(k: int = 12, bits_per_component: int = 1,
                                 n_states: int = 64) -> float:
        torch.manual_seed(0)
        s = torch.rand(n_states, 6) * 16 - 8
        base = LDPC5GBaseline(k, bits_per_component)
        s_hat = base.forward(s, 20.0, channel="awgn", noiseless=True)
        return float((s_hat - s).abs().max())
