"""Dump the LDPC summary as a markdown table (no numbers typed by hand)."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'results/LDPC/step_ldpc_summary.json'
d = json.load(open(path, encoding='utf-8'))
print('status', d['status'], d['completed_slots'], '/', d['expected_slots'])
idx = {(r['k'], r['channel'], r['snr_db']): r['success_rate'] for r in d['rows']}
snrs = d['snr_grid']
for k in d['k_list']:
    for ch in ('awgn', 'rayleigh'):
        vals = [idx.get((k, ch, s)) for s in snrs]
        if any(v is None for v in vals):
            print('%3d %-9s incomplete' % (k, ch))
            continue
        print('%3d %-9s %s  mean %.4f' % (
            k, ch, ' '.join('%.4f' % v for v in vals),
            sum(vals) / len(vals)))
