#!/usr/bin/env python3
"""
calib_followup_summary.py -- summary of run_calib_followup.sh:
  1. sampling noise of the injection calibration: the ideal calibration
     repeated with independent draws of the injected galaxies (seeds), compared
     with Table 1;
  2. empirical-PA scenarios (EMP) against the reference calibration;
  3. PSF bootstrap with many realisations.
Usage: python3 calib_followup_summary.py <calib_tests dir> <table1_disks_F444W.csv>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

R, T1 = sys.argv[1], sys.argv[2]
t1 = pd.read_csv(T1).set_index('z')
out = []

seeds = sorted(glob.glob(os.path.join(R, 'seeds', 'seed_*', 'qpa_calibration.csv')))
ref = os.path.join(R, 'qpa', 'qpa_calibration.csv')
files = seeds + ([ref] if os.path.exists(ref) else [])
if files:
    cal = pd.concat([pd.read_csv(f).query("variant == 'ideal'").assign(run=os.path.basename(os.path.dirname(f)))
                     for f in files])
    out.append(f'1. Ideal calibration with {cal.run.nunique()} independent draws of the injected galaxies\n')
    out.append('  z    rec1 mean+-sd  (paper)    rec2 mean+-sd  (paper)    A1_final mean+-sd (paper)   A2_final mean+-sd (paper)')
    chi = {k: 0.0 for k in ('A1_rec', 'A2_rec', 'A1_final_conv', 'A2_final_conv')}
    pmap = {'A1_rec': 'A1_recovery', 'A2_rec': 'A2_recovery',
            'A1_final_conv': 'A1_final_conv', 'A2_final_conv': 'A2_final_conv'}
    for z, g in cal.groupby('zbin'):
        s = []
        for k in chi:
            m, sd, p = g[k].mean(), g[k].std(), t1.loc[z, pmap[k]]
            chi[k] += ((p - m) / sd) ** 2 if sd > 0 else 0
            s.append(f'{m:.3f}+-{sd:.3f} ({p:.3f})')
        out.append(f'  {z:4.2f}  ' + '   '.join(s))
    out.append('  chi2 of the paper values about the mean of the draws (9 bins, using the scatter '
               'of the draws): ' + ', '.join(f'{k} {v:.1f}' for k, v in chi.items()))
    # --- percentile of Table 1 among the draws, and an empirical null for chi2:
    # each draw in turn is compared with the mean/sd of the others (leave-one-out),
    # which accounts for the small number of draws and for correlations between bins
    piv = {k: cal.pivot_table(index='run', columns='zbin', values=k) for k in chi}
    zs = list(piv['A1_rec'].columns)
    out.append('\n  Table 1 within the distribution of the draws (percentile; 68% / 95% range):')
    for k in chi:
        P = piv[k]
        row = []
        for z in zs:
            v = P[z].dropna().values
            p = t1.loc[z, pmap[k]]
            lo68, hi68, lo95, hi95 = np.percentile(v, [16, 84, 2.5, 97.5])
            row.append(f'{z}: {100 * np.mean(v < p):.0f}% [{lo68:.3f},{hi68:.3f}|{lo95:.3f},{hi95:.3f}]')
        out.append(f'  {k}: ' + '; '.join(row))
    out.append('\n  chi2 of Table 1 against the empirical (leave-one-out) null distribution:')
    for k in chi:
        P = piv[k].dropna()
        n = len(P)
        if n < 5:
            out.append(f'  {k}: too few draws ({n}) for a leave-one-out null')
            continue
        null = []
        for i in range(n):
            rest = P.drop(P.index[i])
            null.append(float((((P.iloc[i] - rest.mean()) / rest.std()) ** 2).sum()))
        tab = np.array([t1.loc[z, pmap[k]] for z in P.columns])
        c_tab = float((((tab - P.mean().values) / P.std().values) ** 2).sum())
        null = np.array(null)
        out.append(f'  {k:14s}: Table 1 chi2 = {c_tab:5.1f}; null (n={n}) median {np.median(null):5.1f}, '
                   f'84% {np.percentile(null, 84):5.1f}, 95% {np.percentile(null, 95):5.1f}; '
                   f'fraction of draws with larger chi2 = {np.mean(null >= c_tab):.2f}')
    gr = []
    for r_, g in cal.groupby('run'):
        g = g.set_index('zbin')
        gr.append([g.loc[[4.5, 5.5], f'A{m}_final_conv'].mean() / g.loc[0.75, f'A{m}_final_conv'] for m in (1, 2)])
    gr = np.array(gr)
    g1, g2 = (t1.loc[[4.5, 5.5], f'A{m}_final_conv'].mean() / t1.loc[0.75, f'A{m}_final_conv'] for m in (1, 2))
    out.append(f'\n  growth over the draws: A1 {gr[:, 0].mean():.3f} +- {gr[:, 0].std(ddof=1):.3f} '
               f'[95%: {np.percentile(gr[:, 0], 2.5):.3f}-{np.percentile(gr[:, 0], 97.5):.3f}; '
               f'Table 1 at percentile {100 * np.mean(gr[:, 0] < g1):.0f}], '
               f'A2 {gr[:, 1].mean():.3f} +- {gr[:, 1].std(ddof=1):.3f} '
               f'[95%: {np.percentile(gr[:, 1], 2.5):.3f}-{np.percentile(gr[:, 1], 97.5):.3f}; '
               f'Table 1 at percentile {100 * np.mean(gr[:, 1] < g2):.0f}]  (Table 1: '
               f'{t1.loc[[4.5, 5.5], "A1_final_conv"].mean() / t1.loc[0.75, "A1_final_conv"]:.3f}, '
               f'{t1.loc[[4.5, 5.5], "A2_final_conv"].mean() / t1.loc[0.75, "A2_final_conv"]:.3f})\n')

g = os.path.join(R, 'qpa_emp', 'qpa_growth.csv')
if os.path.exists(g):
    gg = pd.read_csv(g)
    out.append('2. Deprojection errors incl. empirical PA differences (same models as the reference)\n')
    out.append(gg.round(4).to_string(index=False))
    e = os.path.join(R, 'qpa_emp', 'qpa_sn_errors.csv')
    if os.path.exists(e):
        out.append('\n  applied errors (robust sigma) per bin:\n' + pd.read_csv(e).round(3).to_string(index=False))
    out.append('')

for d in ('psf100', 'psf'):
    g = os.path.join(R, d, 'psf_growth.csv')
    if os.path.exists(g):
        gg = pd.read_csv(g)
        b = gg[gg.variant.str.startswith('boot_')]
        out.append(f'3. PSF bootstrap ({d}, {len(b)} realisations)\n')
        for k in ('A1_growth', 'A2_growth', 'R12_lo', 'R12_hi'):
            out.append(f'  {k:9s}: nominal {gg.set_index("variant").loc["nominal", k]:.4f}, '
                       f'bootstrap {b[k].mean():.4f} +- {b[k].std():.4f} '
                       f'(16-84%: {b[k].quantile(.16):.4f}-{b[k].quantile(.84):.4f})')
        s = os.path.join(R, d, 'psf_bootstrap_scatter.csv')
        if os.path.exists(s):
            out.append('  per-bin scatter:\n' + pd.read_csv(s).to_string(index=False, float_format='%.5f'))
        break

txt = '\n'.join(out)
print(txt)
open(os.path.join(R, 'followup_summary.txt'), 'w').write(txt + '\n')
