"""Exact-value tests for the corrected McNemar implementation (Step 10E).

The bug fixed in Step 10E: statistics.py previously had
    p = 0.5 * math.erfc(math.sqrt(chi2 / 2))
instead of the correct two-sided chi-square(1) survival probability:
    p = math.erfc(math.sqrt(chi2 / 2))

These tests pin the exact formula numerically, catching any future
reintroduction of the factor-of-2 error and verifying all edge cases.

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_mcnemar_corrected
"""

from __future__ import annotations

import math
import sys

from project.evaluation.statistics import mcnemar_test

SUMMARY = []


def check(name: str, ok: bool, detail: str = "") -> None:
    SUMMARY.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]" +
          (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
# Case 1 — symmetric discordance: n01 == n10 == 10
# The n01 == n10 early-return branch fires; must return chi2=0.0, p=1.0
# regardless of the corrected formula.
# ---------------------------------------------------------------------------

def test_mcnemar_symmetric() -> None:
    # Build sequences that give exactly n01=10, n10=10, with 20 concordant (1,1)
    #   group A: 10×(1,0) pairs  -> n01 += 10
    #   group B: 10×(0,1) pairs  -> n10 += 10
    #   group C: 20×(1,1) pairs  -> concordant
    a = [1] * 10 + [0] * 10 + [1] * 20
    b = [0] * 10 + [1] * 10 + [1] * 20
    m = mcnemar_test(a, b)

    ok_counts = (m["n01"] == 10 and m["n10"] == 10)
    ok_chi2 = (m["chi2"] == 0.0)
    ok_p = math.isclose(m["p_value"], 1.0, abs_tol=1e-12)
    ok = ok_counts and ok_chi2 and ok_p
    check("mcnemar_symmetric",
          ok,
          f"n01={m['n01']}, n10={m['n10']}, chi2={m['chi2']}, p={m['p_value']}")


# ---------------------------------------------------------------------------
# Case 2 — unequal discordance: n01=5, n10=95
# chi2 = (|5-95| - 1)^2 / (5+95) = (90-1)^2 / 100 = 89^2 / 100 = 7921/100
# p = erfc(sqrt(7921/200)) = erfc(sqrt(39.605))  ≈ very small positive
# Also verifies that the corrected p is TWICE what the buggy formula gives.
# ---------------------------------------------------------------------------

def test_mcnemar_unequal() -> None:
    # Build sequences: n01=5 (A=1,B=0), n10=95 (A=0,B=1), concordant=100
    a = [1] * 5 + [0] * 95 + [1] * 100
    b = [0] * 5 + [1] * 95 + [1] * 100
    m = mcnemar_test(a, b)

    expected_chi2 = 89 ** 2 / 100   # 79.21 exactly (rational arithmetic)
    # Correct two-sided p-value
    expected_p = math.erfc(math.sqrt(expected_chi2 / 2.0))
    # Buggy one-sided value (factor 0.5 applied)
    buggy_p = 0.5 * expected_p

    ok_counts = (m["n01"] == 5 and m["n10"] == 95)
    ok_chi2 = math.isclose(m["chi2"], expected_chi2, abs_tol=1e-9)
    ok_p_small = (m["p_value"] < 1e-10)
    # The corrected p is strictly larger than the buggy p (factor 2 check)
    ok_factor2 = (m["p_value"] > buggy_p * 1.99)
    # And matches the correct formula to full precision
    ok_p_exact = math.isclose(m["p_value"], expected_p, rel_tol=1e-12)

    ok = ok_counts and ok_chi2 and ok_p_small and ok_factor2 and ok_p_exact
    check("mcnemar_unequal",
          ok,
          f"chi2={m['chi2']:.4f} (expected {expected_chi2:.4f}), "
          f"p={m['p_value']:.3e} (expected {expected_p:.3e}), "
          f"factor2_ok={ok_factor2}")


# ---------------------------------------------------------------------------
# Case 3 — zero discordant pairs: n01=0, n10=0
# No discordant pairs: the early-return fires -> chi2=0.0, p=1.0
# ---------------------------------------------------------------------------

def test_mcnemar_zero_discordant() -> None:
    # All concordant pairs: (1,1) and (0,0)
    a = [1, 1, 0, 0]
    b = [1, 1, 0, 0]
    m = mcnemar_test(a, b)

    ok_counts = (m["n01"] == 0 and m["n10"] == 0)
    ok_chi2 = (m["chi2"] == 0.0)
    ok_p = math.isclose(m["p_value"], 1.0, abs_tol=1e-12)
    ok = ok_counts and ok_chi2 and ok_p
    check("mcnemar_zero_discordant",
          ok,
          f"n01={m['n01']}, n10={m['n10']}, chi2={m['chi2']}, p={m['p_value']}")


# ---------------------------------------------------------------------------
# Case 4 — known reference case: n01=1, n10=99
# chi2 = (|1-99| - 1)^2 / (1+99) = 97^2 / 100 = 9409/100 = 94.09
# p = erfc(sqrt(94.09/2)) = erfc(sqrt(47.045))  ≈ extremely small
# The test additionally verifies p > 2 * buggy_p * 0.99, pinning the
# factor-of-2 numerically.
# ---------------------------------------------------------------------------

def test_mcnemar_reference() -> None:
    # Build sequences: n01=1, n10=99, 100 concordant
    a = [1] * 1 + [0] * 99 + [1] * 100
    b = [0] * 1 + [1] * 99 + [1] * 100
    m = mcnemar_test(a, b)

    expected_chi2 = 97 ** 2 / 100   # 94.09 exactly
    expected_p = math.erfc(math.sqrt(expected_chi2 / 2.0))
    buggy_p = 0.5 * expected_p      # what the old formula would give

    ok_counts = (m["n01"] == 1 and m["n10"] == 99)
    ok_chi2 = math.isclose(m["chi2"], expected_chi2, abs_tol=1e-9)
    ok_p_tiny = (m["p_value"] < 1e-15)
    # Corrected p is strictly > 1.99 × the buggy (one-sided) value:
    ok_factor2 = (m["p_value"] > buggy_p * 1.99)
    ok_p_exact = math.isclose(m["p_value"], expected_p, rel_tol=1e-12)

    ok = ok_counts and ok_chi2 and ok_p_tiny and ok_factor2 and ok_p_exact
    check("mcnemar_reference",
          ok,
          f"chi2={m['chi2']:.4f} (expected {expected_chi2:.4f}), "
          f"p={m['p_value']:.3e} (expected {expected_p:.3e}), "
          f"factor2_ok={ok_factor2}")


# ---------------------------------------------------------------------------
# Case 5 — symmetry: swapping n01 ↔ n10 produces identical chi2 and p
# chi2 depends only on |n01 - n10|; swapping leaves the statistic unchanged.
# ---------------------------------------------------------------------------

def test_mcnemar_symmetry() -> None:
    # Forward: n01=30, n10=5
    a1 = [1] * 30 + [0] * 5 + [0] * 65
    b1 = [0] * 30 + [1] * 5 + [0] * 65
    # Swapped: n01=5, n10=30
    a2 = [0] * 30 + [1] * 5 + [0] * 65
    b2 = [1] * 30 + [0] * 5 + [0] * 65

    m1 = mcnemar_test(a1, b1)
    m2 = mcnemar_test(a2, b2)

    ok_swap = (m1["n01"] == m2["n10"] and m1["n10"] == m2["n01"])
    ok_chi2 = math.isclose(m1["chi2"], m2["chi2"], abs_tol=1e-12)
    ok_p = math.isclose(m1["p_value"], m2["p_value"], abs_tol=1e-12)

    ok = ok_swap and ok_chi2 and ok_p
    check("mcnemar_symmetry",
          ok,
          f"m1: n01={m1['n01']}, n10={m1['n10']}, chi2={m1['chi2']:.6f}, p={m1['p_value']:.3e} | "
          f"m2: n01={m2['n01']}, n10={m2['n10']}, chi2={m2['chi2']:.6f}, p={m2['p_value']:.3e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("STEP 10E: McNemar corrected formula tests (5 cases)")
    print("=" * 72)
    test_mcnemar_symmetric()
    test_mcnemar_unequal()
    test_mcnemar_zero_discordant()
    test_mcnemar_reference()
    test_mcnemar_symmetry()
    n_pass = sum(SUMMARY)
    n_all = len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)
