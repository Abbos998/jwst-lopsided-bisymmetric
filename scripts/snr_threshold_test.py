#!/usr/bin/env python3
"""
snr_threshold_test.py -- lowest aperture-S/N threshold at which the field groups
(deep = CEERS + GOODS, PRIMER-COSMOS, PRIMER-UDS) give the same corrected
amplitudes at high redshift. Uses the per-galaxy S/N calibration of
depth_closure.py (must be in the same directory). NumPy/pandas only.

Criterion, fixed before looking at the result: a threshold passes if, in both
z~3.75 and z>4 and for both A1 and A2,
   (i)  every pair of field groups agrees within 2 sigma, and
   (ii) chi^2 of the three groups about their weighted mean is < 5.99 (2 d.o.f., 95%).
The adopted threshold is the lowest one that passes, provided all higher ones
pass as well.

Usage:
  python3 snr_threshold_test.py --galaxies ../data/intermediate/corrected_disk_f444w.csv \
      --inj ../data/intermediate/inj_f444w_m/injection_per_object.csv \
      --sample $DATA/dja_real.csv --snrcat $DATA/dja_snr_f444w.csv \
      --out $ROOT/referee_tests/threshold
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth_closure as dc

THR = (0, 20, 30, 40, 50, 75, 100)
RANGES = (('z~3.75', (3.75,)), ('z>4', dc.HI))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxies', required=True)
    ap.add_argument('--inj', required=True)
    ap.add_argument('--sample', required=True)
    ap.add_argument('--snrcat', required=True)
    ap.add_argument('--centering', default='catalog')
    ap.add_argument('--field_col', default='field')
    ap.add_argument('--mass_col')
    ap.add_argument('--snr_col')
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--nboot', type=int, default=500)
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.set_option('display.width', 250)
    L = []

    g = pd.read_csv(a.galaxies).dropna(subset=['A1_V6', 'A2_V6'])
    cat = pd.read_csv(a.sample).rename(columns={a.idx_col: 'idx'})
    mcol = a.mass_col or next((c for c in dc.MASS if c in cat.columns), None)
    sn = pd.read_csv(a.snrcat).rename(columns={a.idx_col: 'idx'})
    scol = a.snr_col or next((c for c in ('snr', 'snr_ap', 'snr_aper', 'SNR') if c in sn.columns), None)
    g = g.merge(cat[['idx', a.field_col, mcol]].rename(columns={a.field_col: 'field', mcol: 'logM'}), on='idx', how='left')
    g = g.merge(sn[['idx', scol]].rename(columns={scol: 'snr'}).drop_duplicates('idx'), on='idx', how='left')
    g['zb'] = pd.cut(g.z, dc.EDGES, labels=dc.ZC, right=False).astype(float)
    g = g[g.zb.notna() & g.snr.notna() & (g.snr > 0)].copy()
    g['grp'] = g.field.map(dc.group)
    g = dc.apply(g, dc.calibration(pd.read_csv(a.inj), a.centering))

    # sample size per bin
    n = pd.DataFrame({f'>={t}': [int((np.isclose(g.zb, zc) & (g.snr >= t)).sum()) for zc in dc.ZC] for t in THR}, index=dc.ZC)
    n.index.name = 'z'
    L += ['=== 1 galaxies per redshift bin above each S/N threshold', n.to_string()]

    det, verdict = [], []
    for t in THR:
        u = g[g.snr >= t]
        ok_all, worst, chimax = True, 0.0, 0.0
        for name, zs in RANGES:
            s = u[u.zb.isin(zs)]
            vals = {}
            for gname, w in s.groupby('grp'):
                r = {'S/N >=': t, 'bin': name, 'group': gname, 'N': len(w), 'logM_med': w.logM.median(), 'snr_med': w.snr.median()}
                for k in (1, 2):
                    r[f'A{k}'], r[f'A{k}_err'] = dc.med_err(w[f'C{k}'], rng, a.nboot)
                det.append(r)
                vals[gname] = r
            for k in (1, 2):
                x = np.array([v[f'A{k}'] for v in vals.values()], float)
                e = np.array([v[f'A{k}_err'] for v in vals.values()], float)
                good = np.isfinite(x) & np.isfinite(e) & (e > 0)
                if good.sum() < 3:
                    ok_all = False
                    continue
                x, e = x[good], e[good]
                wt = 1 / e ** 2
                chi2 = float(np.sum(wt * (x - np.sum(wt * x) / wt.sum()) ** 2))
                pair = max(abs(x[i] - x[j]) / np.hypot(e[i], e[j]) for i in range(3) for j in range(i))
                worst, chimax = max(worst, pair), max(chimax, chi2)
                ok_all &= (pair < 2) and (chi2 < 5.99)
        gr = dc.growth(u, rng, a.nboot)
        verdict.append({'S/N >=': t, 'N_0.75': gr['N_0.75'], 'N_z>4': gr['N_z>4'], 'max pair diff [sigma]': worst,
                        'max chi2 (2 dof)': chimax, 'passes': 'yes' if ok_all else 'no',
                        'A1_growth': gr['A1_growth'], 'A1_err': gr['A1_growth_err'],
                        'A2_growth': gr['A2_growth'], 'A2_err': gr['A2_growth_err']})
    D, V = pd.DataFrame(det), pd.DataFrame(verdict)
    D.to_csv(f'{a.out}_groups.csv', index=False)
    V.to_csv(f'{a.out}_verdict.csv', index=False)
    L += ['', '=== 2 corrected medians of the field groups above each threshold (conventional normalisation)',
          D.round(3).to_string(index=False)]
    L += ['', '=== 3 verdict (criterion in the header of this script) and growth z~0.75 -> z>4 of the whole sample',
          V.round(3).to_string(index=False)]
    p = V['passes'].eq('yes').values
    best = next((t for i, t in enumerate(V['S/N >=']) if p[i:].all()), None)
    L += ['', f'ADOPTED THRESHOLD: S/N >= {best}' if best is not None else
          'NO THRESHOLD passes together with all higher ones; send the output back']

    txt = '\n'.join(L)
    print(txt)
    with open(f'{a.out}_summary.txt', 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
