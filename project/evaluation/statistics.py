"""Evaluation statistics (audit 2.8; protocol doc sections V/W).

Implemented exactly: mean, std, success count/total, Wilson 95% CI,
McNemar's test (paired binary outcomes) and the paired t-test (paired
continuous metrics). All functions are dependency-free (no SciPy) and
machine-readable with clearly named fields. They never rank methods or
pick winners; they only compute the requested quantities.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def mean_std(x) -> Dict[str, float]:
    """Mean and (sample) standard deviation of a 1-D sequence."""
    t = torch.as_tensor(list(x) if not torch.is_tensor(x) else x,
                        dtype=torch.float64).reshape(-1)
    if t.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=True)) if t.numel() > 1 else 0.0,
        "n": int(t.numel()),
    }


def success_count(success) -> Dict[str, float]:
    """Success count / total (n = 0 safe)."""
    t = torch.as_tensor(list(success) if not torch.is_tensor(success) else success)
    t = t.reshape(-1).to(torch.uint8)
    n = int(t.numel())
    k = int(t.sum())
    return {"success_count": k, "total": n,
            "success_rate": (k / n) if n else float("nan")}


def wilson_ci_95(success) -> Dict[str, float]:
    """Wilson score 95% interval for a binomial proportion.

    Uses the exact observed count and total; n = 0 returns NaNs; the
    all-success and zero-success cases yield one-sided-touching intervals
    (never a degenerate [p, p] unless the Normal approximation truly
    collapses). Numerically stable closed form:

        center = (p + z^2/(2n)) / (1 + z^2/n)
        half   = z/(1 + z^2/n) * sqrt(p(1-p)/n + z^2/(4n^2))
    """
    c = success_count(success)
    k, n = c["success_count"], c["total"]
    if n == 0:
        return {"wilson_low": float("nan"), "wilson_high": float("nan"),
                "p_hat": float("nan"), "success_count": 0, "total": 0}
    z = Z_95
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(max(p * (1.0 - p) / n
                                       + z * z / (4.0 * n * n), 0.0))
    return {"wilson_low": max(center - half, 0.0),
            "wilson_high": min(center + half, 1.0),
            "p_hat": p, "success_count": k, "total": n}


def _binom_sf_approx(k: int, n: int, p: float) -> float:
    """Upper-tail binomial probability P(X >= k) via the normal
    approximation with continuity correction (no SciPy dependency;
    accurate enough at the tail probabilities the test uses)."""
    if k <= 0:
        return 1.0
    var = n * p * (1.0 - p)
    if var <= 0:
        return 1.0 if k > n * p else 0.0
    z = (k - 0.5 - n * p) / math.sqrt(var)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def mcnemar_test(success_a, success_b) -> Dict[str, float]:
    """Exact-form McNemar test on PAIRED binary outcomes.

    Only discordant pairs matter: n01 = #(A=1, B=0), n10 = #(A=0, B=1).
    Statistic chi2 = (|n01 - n10| - 1)^2 / (n01 + n10) with 1 dof
    (continuity correction), p from the survival function of chi2(1)
    = erfc(sqrt(chi2/2)). Perfectly symmetric discordance (n01 == n10)
    is the known edge case where the correction would produce a spurious
    chi2 = (n01+n10)/2; it is handled explicitly as chi2 = 0, p = 1.
    Preserves pairing by construction.
    """
    a = torch.as_tensor(list(success_a) if not torch.is_tensor(success_a) else success_a
                        ).reshape(-1).to(torch.uint8)
    b = torch.as_tensor(list(success_b) if not torch.is_tensor(success_b) else success_b
                        ).reshape(-1).to(torch.uint8)
    if a.shape != b.shape:
        raise ValueError(f"paired outcomes must align: {a.shape} vs {b.shape}")
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n_pairs = int(a.numel())
    if n01 == n10:
        return {"n01": n01, "n10": n10, "n_pairs": n_pairs,
                "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return {"n01": n01, "n10": n10, "n_pairs": n_pairs,
            "chi2": float(chi2), "p_value": float(p)}


def paired_t_test(x_a, x_b) -> Dict[str, float]:
    """Paired t-test on paired continuous metrics (x_a - x_b differences).

    Preserves pairing: only the per-episode differences enter. Returns the
    mean difference, t statistic, degrees of freedom and the two-sided
    p-value from the t-distribution (computed with the regularized
    incomplete beta function, no SciPy).
    """
    a = torch.as_tensor(list(x_a) if not torch.is_tensor(x_a) else x_a,
                        dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(list(x_b) if not torch.is_tensor(x_b) else x_b,
                        dtype=torch.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"paired samples must align: {a.shape} vs {b.shape}")
    n = a.numel()
    if n < 2:
        return {"mean_diff": float("nan"), "t_stat": float("nan"),
                "dof": int(n - 1), "p_value": float("nan"), "n": int(n)}
    d = a - b
    md = float(d.mean())
    sd = float(d.std(unbiased=True))
    if sd == 0.0:
        t_stat = float("inf") if md != 0 else 0.0
        p = 0.0 if md != 0 else 1.0
    else:
        t_stat = md / (sd / math.sqrt(n))
        p = _t_sf_two_sided(abs(t_stat), n - 1)
    return {"mean_diff": md, "t_stat": float(t_stat), "dof": int(n - 1),
            "p_value": float(p), "n": int(n)}


def _betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via the continued fraction
    (Lentz), standard numerical recipe; adequate for p-value tails."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-12):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _t_sf_two_sided(t: float, dof: int) -> float:
    """Two-sided p-value P(|T| > t) for Student-t with dof >= 1."""
    if dof <= 0:
        return float("nan")
    x = dof / (dof + t * t)
    return _betainc_reg(dof / 2.0, 0.5, x)
