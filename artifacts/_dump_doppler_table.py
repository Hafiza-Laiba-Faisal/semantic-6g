import json

d = json.load(open('results/DOPPLER/step_doppler_summary.json'))
rows = d['rows']
idx = {(r['point'], r['k'], r['snr_db']): r for r in rows}
order = ['frozen_iid', 'd0033_2g', 'd0067_2g', 'd0133_2g', 'd0333_2g',
         'd0667_2g', 'd1333_2g', 'd3333_2g', 'd0117_07g', 'd0234_07g']
for pid in order:
    st = d['fade_statistics'][pid]
    dl = st['displacement_per_step']
    dls = 'n/a' if dl is None else '%.3f' % dl
    for k in (12, 54):
        lam = ' '.join('%.3f' % idx[(pid, k, s)]['lam'] for s in (0, 5, 10, 15, 20))
        dig = ' '.join('%.3f' % idx[(pid, k, s)]['digital'] for s in (0, 5, 10, 15, 20))
        print('%s|%d|%s|%.2f|%s|%s' % (pid, k, dls,
                                       st['mean_fade_run_steps'], lam, dig))
