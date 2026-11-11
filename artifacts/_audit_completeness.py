"""Final completeness audit for the A-19 / A-20 block.

Reads only stored artifacts, the Markdown sources and the built docx.
Prints PASS/FAIL for each invariant and a list of stale strings.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fails = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  (" + detail + ")") if detail else ""))
    if not ok:
        fails.append(name)


# ---------- artifacts ----------
dop = json.loads((ROOT / "results/DOPPLER/step_doppler_summary.json").read_text(encoding="utf-8"))
ldpc = json.loads((ROOT / "results/LDPC/step_ldpc_summary.json").read_text(encoding="utf-8"))

check("DOPPLER summary complete 100/100",
      dop["status"] == "complete" and dop["completed_slots"] == 100 == dop["expected_slots"],
      "%s %d/%d" % (dop["status"], dop["completed_slots"], dop["expected_slots"]))
check("DOPPLER follow-up arm recorded",
      "trained_arm" in dop and abs(dop["trained_arm"]["mean"] - 0.993) < 1e-9,
      "mean %.3f vs stored %.4f (delta %+.4f)" % (
          dop["trained_arm"]["mean"], dop["trained_arm"]["stored_arm_mean"],
          dop["trained_arm"]["delta_vs_stored_arm"]))
check("LDPC summary complete 50/50",
      ldpc["status"] == "complete" and ldpc["completed_slots"] == 50 == ldpc["expected_slots"],
      "%s %d/%d" % (ldpc["status"], ldpc["completed_slots"], ldpc["expected_slots"]))

n_dop_files = len(list((ROOT / "results/DOPPLER").rglob("*.json")))
n_ldpc_files = len(list((ROOT / "results/LDPC").rglob("*.json")))
check("DOPPLER eval slots on disk", n_dop_files >= 106, "%d json files" % n_dop_files)
check("LDPC eval slots on disk", n_ldpc_files >= 51, "%d json files" % n_ldpc_files)

# ---------- protocol ledger ----------
pa = (ROOT / "fyp_md_files/PROTOCOL_AMENDMENTS.md").read_text(encoding="utf-8")
check("A-19 outcome section", "### A-19 outcome" in pa)
check("A-20 outcome section", "### A-20 outcome" in pa)
check("accounting: DOPPLER 100 / 100", "100 / 100" in pa)
check("accounting: follow-up 6 / 6", "6 / 6" in pa)
check("accounting: LDPC 50 / 50", "50 / 50" in pa)
check("A-20 deviations disclosed",
      "Payload length" in pa and "Table citation" in pa and "Table 5.3.2-3" in pa)
check("no 'in progress' left in ledger", "in progress at time of writing" not in pa)

# ---------- thesis source ----------
th = (ROOT / "fyp_md_files/THESIS_MASTER_DOCUMENT.md").read_text(encoding="utf-8")
for h in ("## 5.13 A-19", "## 5.14 A-20", "## 5.15 Summary of Findings",
          "## 6.6 Two Robustness Axes", "## 6.7 Threats to Validity"):
    check("heading present: " + h, th.count(h) == 1)

for cap in ("**Table 5.10 — A-17", "**Table 5.11 — A-15", "**Table 5.15 — A-19",
            "**Table 5.16 — A-19", "**Table 5.17 — A-20"):
    check("caption present: " + cap, th.count(cap) == 1)

check("Table 5.10 appears before 5.11 (document order)",
      th.index("**Table 5.10 — A-17") < th.index("**Table 5.11 — A-15"))
check("no placeholder marker", "PLACEHOLDER" not in th)
check("no stale '99 passing'", "99 passing" not in th)
check("no stale A-9…A-15 figure caption", "and A-9…A-15." not in th)
check("tests count is 111", "111 passing tests" in th)

# Table 5.17 numbers must equal the artifact means
means = ldpc["mean_over_snr"]
block = th[th.index("**Table 5.17"):th.index("## 5.15 Summary of Findings")]
for k in (12, 18, 24, 30, 54):
    for ch, col in (("awgn", "LDPC AWGN"), ("rayleigh", "LDPC Rayleigh")):
        val = "%.4f" % means["k%d_%s" % (k, ch)]
        check("Table 5.17 k=%d %s = artifact %s" % (k, ch, val), val in block)

check("summary item 10 is Doppler", "10. **Doppler is not a threat" in th)
check("summary item 11 is LDPC", "11. **The code family does not change" in th)

# ---------- results notes ----------
for note in ("RESULTS_DOPPLER_2026-11-02.md", "RESULTS_LDPC_2026-11-02.md"):
    p = ROOT / "fyp_md_files" / note
    t = p.read_text(encoding="utf-8")
    check("results note exists: " + note, p.is_file() and len(t) > 2000)
    check("no placeholder in " + note,
          "filled in when" not in t and "<!-- results tables" not in t
          and "<!-- filled from" not in t)

# ---------- docs ----------
idx = (ROOT / "fyp_md_files/INDEX.md").read_text(encoding="utf-8")
check("INDEX points to A-1…A-20", "A-1 … A-20" in idx)
check("INDEX lists both results notes",
      "RESULTS_DOPPLER_2026-11-02.md" in idx and "RESULTS_LDPC_2026-11-02.md" in idx)
check("INDEX says 24 numbered tables", "24 numbered tables" in idx)

ps = (ROOT / "ARCHIVE/PROJECT_STATUS.md").read_text(encoding="utf-8")
check("PROJECT_STATUS has the 2026-11-02 block",
      "## 2026-11-02 — Doppler (A-19) and 5G NR LDPC (A-20) closed" in ps)

brief = (ROOT / "ARCHIVE/SUPERVISOR_BRIEFING_PLAIN.md").read_text(encoding="utf-8")
check("briefing has A-19 and A-20 bullets",
      "**A-19 — Doppler" in brief and "**A-20 — 5G NR LDPC" in brief)
check("briefing no 'LDPC not done'", "LDPC not done" not in brief)

res_readme = (ROOT / "results/README.md").read_text(encoding="utf-8")
check("results/README lists DOPPLER and LDPC",
      "`DOPPLER/`" in res_readme and "`LDPC/`" in res_readme)
check("results/README no 'M5 does not exist'", "does **not** exist" not in res_readme)

# ---------- docx ----------
try:
    import docx
    d = docx.Document(str(ROOT / "THESIS_MASTER_DOCUMENT.docx"))
    check("docx table count = 28", len(d.tables) == 28, str(len(d.tables)))
    check("docx paragraph count = 523", len(d.paragraphs) == 523, str(len(d.paragraphs)))
    heads = [p.text for p in d.paragraphs if p.style.name.startswith("Heading")]
    check("docx has 5.13/5.14/5.15 headings",
          any(h.startswith("5.13") for h in heads)
          and any(h.startswith("5.14") for h in heads)
          and any(h.startswith("5.15") for h in heads))
    txt = "\n".join(p.text for p in d.paragraphs)
    check("docx contains the LDPC gap sentence", "0.017" in txt)
    check("docx contains the Doppler retraining number", "0.993" in txt)
except Exception as exc:  # noqa: BLE001
    check("docx readable", False, repr(exc))

# ---------- stale-string sweep over the authoritative docs ----------
stale = ("M5 is deferred", "no results/M5 exists", "LDPC not done",
         "99 passing tests", "contributions (99 tests)")
srcs = ["README.md", "fyp_md_files/THESIS_MASTER_DOCUMENT.md",
        "ARCHIVE/PROJECT_STATUS.md", "ARCHIVE/SUPERVISOR_BRIEFING_PLAIN.md",
        "fyp_md_files/INDEX.md", "results/README.md"]
for s in srcs:
    t = (ROOT / s).read_text(encoding="utf-8")
    hits = [w for w in stale if w in t]
    check("no stale strings in " + s, not hits, ", ".join(hits))

print()
print("FAILURES:", len(fails))
for f in fails:
    print("  -", f)
sys.exit(1 if fails else 0)
