"""Merge the A-20 LDPC summary with the A-11 conv column (same episodes).

Prints a markdown table with no hand-typed numbers.  The conv column comes from
``results/POLAR/step_polar_summary_E1000.json``, which re-ran the frozen
convolutional baseline at E = 1000 on the same TEST_SEED and the same episode
stream as A-20.
"""
import json

LDPC = "results/LDPC/step_ldpc_summary.json"
POLAR = "results/POLAR/step_polar_summary_E1000.json"

ldpc = json.load(open(LDPC, encoding="utf-8"))
polar = json.load(open(POLAR, encoding="utf-8"))

assert ldpc["episodes"] == polar["design"]["episodes"] == 1000
assert ldpc["snr_grid"] == polar["design"]["snr_grid_db"]
assert [str(k) for k in ldpc["k_list"]] == [str(k) for k in polar["design"]["k_grid"]]

ldpc_mean = ldpc["mean_over_snr"]
conv = {}
for row in polar["rows"]:
    conv.setdefault((row["k"], row["channel"]), []).append(
        row["conv"]["success_rate"])
conv_mean = {"k%s_%s" % key: sum(v) / len(v) for key, v in conv.items()}

print("status %s %d/%d" % (ldpc["status"], ldpc["completed_slots"],
                           ldpc["expected_slots"]))
print()
print("| k | B | A bits | Z | N | Channel | LDPC | Conv | Delta |")
print("|---:|---:|---:|---:|---:|---|---:|---:|---:|")
from project import config
from project.baselines.ldpc5g import LDPC5GCode


for k in ldpc["k_list"]:
    B = config.DIGITAL_B_BY_K[k]
    code = LDPC5GCode(6 * B, 2 * k)
    for ch in ("awgn", "rayleigh"):
        key = "k%d_%s" % (k, ch)
        a = ldpc_mean[key]
        c = conv_mean[key]
        print("| %d | %d | %d | %d | %d | %s | %.4f | %.4f | %+.4f |"
              % (k, B, 6 * B, code.z, code.n_bits, ch, a, c, a - c))

print()
for k in ldpc["k_list"]:
    print("k=%d  awgn %.4f  rayleigh %.4f" % (
        k, ldpc_mean["k%d_awgn" % k], ldpc_mean["k%d_rayleigh" % k]))
