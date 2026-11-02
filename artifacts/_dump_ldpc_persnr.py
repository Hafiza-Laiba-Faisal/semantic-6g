"""Dump A-20 per-SNR tables (LDPC from its summary, conv from the A-11 run)."""
import json

ldpc = json.load(open("results/LDPC/step_ldpc_summary.json", encoding="utf-8"))
polar = json.load(open("results/POLAR/step_polar_summary_E1000.json",
                       encoding="utf-8"))

snrs = ldpc["snr_grid"]
idx = {(r["k"], r["channel"], r["snr_db"]): r["success_rate"]
       for r in ldpc["rows"]}
cidx = {(r["k"], r["channel"], r["snr_db"]): r["conv"]["success_rate"]
        for r in polar["rows"]}

for name, table in (("LDPC", idx), ("Conv", cidx)):
    print("### %s" % name)
    print()
    print("| k | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | Mean |")
    print("|---|---|---|---|---|---|---|---|")
    for k in ldpc["k_list"]:
        for ch in ("awgn", "rayleigh"):
            vals = [table[(k, ch, s)] for s in snrs]
            print("| %d | %s | %s | %.4f |" % (
                k, ch, " | ".join("%.3f" % v for v in vals),
                sum(vals) / len(vals)))
    print()
