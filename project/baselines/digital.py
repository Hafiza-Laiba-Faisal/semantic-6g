"""Conventional separated digital baseline (audit section 8, frozen A1 + A11).

Chain (one state transmission = one control step):

    s -> normalize -> uniform scalar quantizer (B bits/component)
      -> bit packing (component-major, MSB-first per component)
      -> + 6 zero tail bits (K - 1, zero-tail termination)
      -> convolutional encoder K = 7, R_c = 1/2, G1 = (171)_8 first
      -> Gray-QPSK (unit-average-power constellation)
      -> padding symbols (zero bits) to exactly k complex channel uses
      -> SHARED Step-3 channel (AWGN or block Rayleigh)
      -> [Rayleigh: SHARED Step-3 ZF equalization]
      -> exact log-MAP soft LLRs (AWGN: sigma^2; Rayleigh: sigma^2/|h|^2
         per codeword with the Step-3 guard, per-symbol perfect-CSI scaling)
      -> soft-decision Viterbi (traceback from the known zero-tail state 0)
      -> drop 6 tail bits -> unpack -> dequantize -> denormalize -> s_hat

Frozen conventions and their exact definitions:

* Quantizer (endpoint-inclusive uniform, source-document "Quantizer
  mathematics"): Delta = 2 / (2^B - 1) on the normalized range [-1, 1],
  q = clamp(round((clip(s,-1,1) + 1)/Delta), 0, 2^B - 1), s_hat = -1 + Delta*q.
  Maximum granular error <= Delta/2; endpoints reproduce exactly.
* Bit order: component-major, MSB-first per component (fixed project
  convention, audit 8.1).
* Convolutional code (polynomial convention; D^0 = current input u,
  register m1..m6 with m1 the most recent previous input):
      G1 = (171)_8 = 1111001_2 = 1 + D^3 + D^4 + D^5 + D^6
           -> c1 = u ^ m3 ^ m4 ^ m5 ^ m6
      G2 = (133)_8 = 1011011_2 = 1 + D + D^3 + D^4 + D^6
           -> c2 = u ^ m1 ^ m3 ^ m4 ^ m6
  Output ordering: (c1, c2) per input bit - G1 FIRST (fixed convention).
* Gray-QPSK label mapping (unit average power): the coded bit pair
  (b0, b1) maps to  z = ((1 - 2*b0) + j*(1 - 2*b1)) / sqrt(2),
  i.e. b0 is the I-axis bit, b1 the Q-axis bit; adjacent constellation
  points differ in exactly one bit (Gray).
* Exact log-MAP LLR sign convention: LLR = log[ P(bit = 0 | y) / P(bit = 1 | y) ],
  so the Viterbi/decision rule is  bit = 0 iff LLR > 0.  For the frozen
  Gray-QPSK mapping the two label bits separate across the I/Q axes, so
  the exact log-MAP marginalization reduces - EXACTLY, with no Max-Log
  approximation - to the linear form
      LLR_b0 = 4*sqrt(2)*Re(y_eq)/sigma_eff^2 ,
      LLR_b1 = 4*sqrt(2)*Im(y_eq)/sigma_eff^2 .
  AWGN: sigma_eff^2 = sigma^2 (per complex symbol). Rayleigh: with the
  frozen ZF equalizer y_eq = y h* / max(|h|^2, eps_eq) the per-symbol
  effective complex noise variance is sigma_eff^2 = sigma^2 / max(|h|^2,
  eps_eq) (per-codeword scalar, perfect CSI, the Step-3 guard reused).
  Theory axis: this is SNR = P/sigma^2 per complex symbol; equivalently
  the bit error rate is Q(sqrt(2 Eb/N0)) with Eb = Es/2, N0 = sigma^2 -
  the same curve expressed in the supplementary digital label.
* Padding: filler QPSK symbols mapped from zero bits, added AFTER coding,
  transmitted inside the k-symbol packet at the same power, and DROPPED
  before LLR extraction (never reaches Viterbi/dequantizer). Tail bits
  are overhead and never counted as payload.
* Budget: 6B + 6 <= k is enforced by config.digital_packet; violations
  raise InfeasibleDigitalConfig (k in {1, 2, 3} is INFEASIBLE by design).

No channel, normalization, controller, or dynamics logic is duplicated
here: the Step-3 channel functions and the frozen Step 0-2 modules are
reused verbatim. This module contains no learnable parameters.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import math

import torch

from project import config
from project.channels.awgn import awgn_channel
from project.channels.complex_utils import check_codeword
from project.channels.rayleigh import equalize_zf, rayleigh_channel
from project.data.normalization import denormalize, normalize


# ---------------------------------------------------------------------------
# Gate A - quantizer
# ---------------------------------------------------------------------------

def quantize(s_bar: torch.Tensor, bits_per_component: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Frozen endpoint-inclusive uniform quantizer on the normalized state.

    Returns ``(q, s_hat_bar)`` with integer levels in [0, 2^B - 1] and the
    reconstructed values on [-1, 1].
    """
    delta = config.quantizer_step(bits_per_component)
    levels = 2 ** bits_per_component
    clipped = torch.clamp(s_bar, min=-1.0, max=1.0)
    q = torch.clamp(torch.round((clipped + 1.0) / delta), 0, levels - 1)
    return q, -1.0 + delta * q


def dequantize(q: torch.Tensor, bits_per_component: int) -> torch.Tensor:
    """Inverse map of :func:`quantize` for recovered levels."""
    delta = config.quantizer_step(bits_per_component)
    return -1.0 + delta * q.to(torch.get_default_dtype())


# ---------------------------------------------------------------------------
# bit packing / unpacking (component-major, MSB-first)
# ---------------------------------------------------------------------------

def pack_bits(q: torch.Tensor, bits_per_component: int) -> torch.Tensor:
    """Integer levels ``[..., 6]`` -> bits ``[..., 6B]``.

    Component-major: component j occupies bits [j*B, (j+1)*B); within a
    component, MSB first.
    """
    B = bits_per_component
    powers = torch.pow(2, torch.arange(B - 1, -1, -1, device=q.device,
                                       dtype=q.dtype))       # MSB first
    bits = ((q.unsqueeze(-1) / powers).floor() % 2).to(torch.long)
    return bits.flatten(start_dim=-2)


def unpack_bits(bits: torch.Tensor, bits_per_component: int) -> torch.Tensor:
    """Inverse of :func:`pack_bits`; returns levels ``[..., 6]``."""
    B = bits_per_component
    shaped = bits.reshape(*bits.shape[:-1], 6, B)
    powers = torch.pow(2, torch.arange(B - 1, -1, -1, device=bits.device,
                                       dtype=bits.dtype))
    return (shaped * powers).sum(dim=-1)


# ---------------------------------------------------------------------------
# Gate C - convolutional encoder (K = 7, R = 1/2, G1 first)
# ---------------------------------------------------------------------------

def _conv_tables(device: torch.device) -> Dict[str, torch.Tensor]:
    """State tables for the frozen code.

    State bit layout: state = sum_{i=1..6} m_i * 2^(i-1), m1 = most recent
    previous input. next_state = (m1..m6 << 1 | u) keeping 6 bits.
    """
    states = torch.arange(64, device=device)
    m1 = (states >> 0) & 1
    m2 = (states >> 1) & 1
    m3 = (states >> 2) & 1
    m4 = (states >> 3) & 1
    m5 = (states >> 4) & 1
    m6 = (states >> 5) & 1
    u0 = torch.zeros_like(states)
    u1 = torch.ones_like(states)
    # G1 = 1 + D^3 + D^4 + D^5 + D^6 ; G2 = 1 + D + D^3 + D^4 + D^6
    out = torch.stack(
        [torch.stack([u0 ^ m3 ^ m4 ^ m5 ^ m6, u0 ^ m1 ^ m3 ^ m4 ^ m6], dim=-1),
         torch.stack([u1 ^ m3 ^ m4 ^ m5 ^ m6, u1 ^ m1 ^ m3 ^ m4 ^ m6], dim=-1)],
        dim=1)                                       # [64, 2(u), 2(c)]
    nxt = ((states * 2) % 64).unsqueeze(1).expand(64, 2) \
        | torch.stack([u0, u1], dim=1)                # [64, 2]
    return {"out": out, "next": nxt}


def conv_encode(info_bits: torch.Tensor) -> torch.Tensor:
    """Information bits ``[B, L]`` (tail already appended) -> coded bits
    ``[B, 2L]`` with (c1, c2) per input bit, starting and ending at state 0.
    """
    tables = _conv_tables(info_bits.device)
    B, L = info_bits.shape
    out_states = tables["out"]                        # [64, 2, 2]
    nxt = tables["next"]                              # [64, 2]
    state = torch.zeros(B, dtype=torch.long, device=info_bits.device)
    coded = info_bits.new_zeros(B, 2 * L)
    for t in range(L):
        u = info_bits[:, t]
        pair = out_states[state, u]                   # [B, 2] -> (c1, c2)
        coded[:, 2 * t] = pair[:, 0]
        coded[:, 2 * t + 1] = pair[:, 1]
        state = nxt[state, u]
    assert bool((state == 0).all()), "zero tail must drive the register to 0"
    return coded


# ---------------------------------------------------------------------------
# Gate B - Gray-QPSK modem with exact log-MAP LLRs
# ---------------------------------------------------------------------------

def qpsk_modulate(bits: torch.Tensor) -> torch.Tensor:
    """Coded bit pairs ``[B, N]`` (even N) -> complex symbols ``[B, N/2]``.

    (b0, b1) -> ((1 - 2 b0) + j (1 - 2 b1)) / sqrt(2); unit average power.
    """
    if bits.shape[-1] % 2 != 0:
        raise ValueError("QPSK needs an even number of bits")
    b0 = bits[..., 0::2].to(torch.get_default_dtype())
    b1 = bits[..., 1::2].to(torch.get_default_dtype())
    return (torch.complex(1.0 - 2.0 * b0, 1.0 - 2.0 * b1)
            / math.sqrt(2.0))


def qpsk_llrs(
    y_eq: torch.Tensor,
    sigma2_eff: torch.Tensor,
) -> torch.Tensor:
    """Exact log-MAP LLRs for the frozen Gray-QPSK mapping.

    ``y_eq``: equalized complex symbols ``[B, n]``.
    ``sigma2_eff``: effective complex noise variance per codeword,
    broadcastable to ``y_eq`` (AWGN: sigma^2; Rayleigh: sigma^2/|h|^2).

    Returns LLRs ``[B, 2n]`` ordered as (b0(sym1), b1(sym1), b0(sym2), ...),
    with LLR = log[P(0)/P(1)] > 0 meaning bit 0 is more likely.
    """
    if torch.is_complex(y_eq):
        yi, yq = y_eq.real, y_eq.imag
    else:
        k = y_eq.shape[-1] // 2
        yi, yq = y_eq[..., :k], y_eq[..., k:]
    s2 = sigma2_eff.to(yi.dtype)
    llr_i = (4.0 * math.sqrt(2.0)) * yi / s2
    llr_q = (4.0 * math.sqrt(2.0)) * yq / s2
    return torch.stack([llr_i, llr_q], dim=-1).flatten(start_dim=-2)


# ---------------------------------------------------------------------------
# Gate C - soft-decision Viterbi (zero tail, traceback from state 0)
# ---------------------------------------------------------------------------

def viterbi_decode(llrs: torch.Tensor) -> torch.Tensor:
    """Soft-decision Viterbi for the frozen code.

    ``llrs``: ``[B, 2L]`` in (c1, c2) order matching the encoder output.
    Branch metric for codeword bit c given LLR l: +l if c = 0, -l if c = 1
    (i.e. l * (1 - 2c)).

    ACS (textbook butterfly): next state j = (2s + u) mod 64 merges exactly
    two previous states, both with the SAME input u = j & 1:
        prev_a = j >> 1,   prev_b = prev_a + 32.
    Each next state therefore sees exactly two candidates; the winner and
    its previous state are stored. (Note: a scatter over prev->next is
    INVALID here because the map is 2-to-1 and scatter semantics for
    duplicate indices are undefined.)

    Traceback starts from state 0 (guaranteed by the zero tail); the
    information bit at step t is the LSB of the next state (= the common
    input u of both merged branches). Returns the L recovered information
    bits ``[B, L]``.
    """
    tables = _viterbi_tables(llrs.device)
    out = tables["out"]                               # [64, 2(u), 2(c)]
    B, twoL = llrs.shape
    L = twoL // 2
    device = llrs.device
    j_all = torch.arange(64, device=device)
    prev_a = j_all >> 1                               # [64] per next state
    prev_b = prev_a + 32                              # [64]
    u_all = j_all & 1                                 # [64] shared input
    sign_a1 = 1.0 - 2.0 * out[prev_a, u_all, 0].to(llrs.dtype)   # [64]
    sign_a2 = 1.0 - 2.0 * out[prev_a, u_all, 1].to(llrs.dtype)
    sign_b1 = 1.0 - 2.0 * out[prev_b, u_all, 0].to(llrs.dtype)
    sign_b2 = 1.0 - 2.0 * out[prev_b, u_all, 1].to(llrs.dtype)

    metrics = torch.full((B, 64), -1e30, device=device, dtype=llrs.dtype)
    metrics[:, 0] = 0.0                               # start state 0
    hist = []                                         # per-step winning prev
    for t in range(L):
        l1 = llrs[:, 2 * t].unsqueeze(1)              # [B, 1]
        l2 = llrs[:, 2 * t + 1].unsqueeze(1)          # [B, 1]
        bm_a = l1 * sign_a1.unsqueeze(0) + l2 * sign_a2.unsqueeze(0)  # [B, 64]
        bm_b = l1 * sign_b1.unsqueeze(0) + l2 * sign_b2.unsqueeze(0)
        c_a = metrics.gather(1, prev_a.unsqueeze(0).expand(B, 64)) + bm_a
        c_b = metrics.gather(1, prev_b.unsqueeze(0).expand(B, 64)) + bm_b
        take_b = c_b > c_a
        metrics = torch.where(take_b, c_b, c_a)
        hist.append(torch.where(take_b,
                                prev_b.unsqueeze(0).expand(B, 64),
                                prev_a.unsqueeze(0).expand(B, 64)))
    # traceback from state 0; bit(t) = LSB of the next state
    bits = llrs.new_zeros(B, L, dtype=torch.long)
    state = torch.zeros(B, dtype=torch.long, device=device)
    for t in range(L - 1, -1, -1):
        bits[:, t] = state & 1
        state = hist[t].gather(1, state.unsqueeze(1)).squeeze(1)
    return bits


def _viterbi_tables(device: torch.device) -> Dict[str, torch.Tensor]:
    t = _conv_tables(device)
    return {"next": t["next"], "out": t["out"]}


# ---------------------------------------------------------------------------
# Gate E - full digital baseline
# ---------------------------------------------------------------------------

class DigitalBaseline:
    """Separated quantize-code-modulate chain with the frozen framing.

    Not differentiable and not trainable by design (standard practice for
    the conventional baseline); evaluation-only.
    """

    def __init__(self, k: int, bits_per_component: int) -> None:
        self.pkt = config.digital_packet(k, bits_per_component)  # raises if infeasible
        self.k = k
        self.B = bits_per_component
        self.n_tail = config.CONV_TAIL_BITS
        self.pad_bits = 2 * self.pkt["n_padding_symbols"]        # zero-bit filler pairs

    # -- transmit side --------------------------------------------------

    def encode_state(self, s: torch.Tensor) -> torch.Tensor:
        """Physical state -> k complex channel symbols (payload + padding)."""
        s_bar = normalize(s)
        q, _ = quantize(s_bar, self.B)
        info = torch.cat([pack_bits(q, self.B).to(torch.long),
                          torch.zeros(*q.shape[:-1], self.n_tail,
                                      dtype=torch.long, device=q.device)], dim=-1)
        coded = conv_encode(info)                                # [B, 12B+12]
        sym = qpsk_modulate(coded)                               # [B, 6B+6]
        if self.pad_bits:
            pad = qpsk_modulate(torch.zeros(*sym.shape[:-1], self.pad_bits,
                                            dtype=torch.long,
                                            device=sym.device))
            sym = torch.cat([sym, pad], dim=-1)
        assert sym.shape[-1] == self.k
        return sym

    # -- receive side ---------------------------------------------------

    def decode_symbols(
        self,
        y: torch.Tensor,
        channel: str,
        snr_db: float,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Channel output -> dequantized NORMALIZED state estimate."""
        check_codeword(y)
        n_pay = self.pkt["n_payload_symbols"]
        if torch.is_complex(y):
            y_pay = y[..., :n_pay]
        else:                                                    # stacked real
            k = y.shape[-1] // 2
            y_pay = y[..., : 2 * n_pay]
        if channel == "rayleigh":
            if h is None:
                raise ValueError("Rayleigh decoding needs the CSI tensor h")
            y_pay = equalize_zf(y_pay, h)
        sigma2 = config.TX_POWER * (10.0 ** (-snr_db / 10.0))
        if channel == "rayleigh":
            hh = (h.real ** 2 + h.imag ** 2).clamp_min(config.EPS_EQUALIZER)
            sigma2_eff = sigma2 / hh                             # [B, 1], per-codeword CSI
        else:
            sigma2_eff = torch.tensor(sigma2, dtype=y_pay.real.dtype
                                      if torch.is_complex(y_pay) else y_pay.dtype)
        llrs = qpsk_llrs(y_pay, sigma2_eff)
        info = viterbi_decode(llrs)
        info = info[:, :-self.n_tail]                            # drop tail
        q_hat = unpack_bits(info, self.B)
        return dequantize(q_hat, self.B)

    # -- full path ------------------------------------------------------

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
        """Physical state -> estimated physical state through the full chain.

        ``noiseless=True`` bypasses the channel (validation reference for
        Gate E: the estimate then equals the quantizer reconstruction).
        ``noise``/``h`` accept pre-generated realizations for paired
        evaluation (used verbatim; no sampling). The frozen framing,
        budget and Step-3 channel conventions are untouched.
        """
        if channel not in ("awgn", "rayleigh"):
            raise ValueError(f"unknown channel {channel!r}")
        with torch.no_grad():
            sym = self.encode_state(s)
            if noiseless:
                # validation reference: any sigma2 gives exact decisions on
                # an unnoised symbol sequence
                s_bar_hat = self.decode_symbols(sym, "awgn", snr_db)
                return denormalize(s_bar_hat)
            if channel == "awgn":
                y = awgn_channel(sym, snr_db, generator=generator, noise=noise)
                s_bar_hat = self.decode_symbols(y, "awgn", snr_db)
            else:
                y, h_used = rayleigh_channel(sym, snr_db, generator=generator,
                                             noise=noise, h=h, return_h=True)
                s_bar_hat = self.decode_symbols(y, "rayleigh", snr_db, h=h_used)
            return denormalize(s_bar_hat)
