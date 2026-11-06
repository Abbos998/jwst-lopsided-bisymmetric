#!/usr/bin/env python3
"""
median_recovery_test.py — does the bin-median correction of Eq. (2) recover the
median of a DISTRIBUTION of true amplitudes?

Each injection run gives every galaxy of a bin the same amplitude (x0.5, x1 or
x1.5 the observed one). Mixing the injected galaxies of the three runs with
different weights produces populations whose true amplitudes span a factor 3
with different shapes (skewed low, uniform, skewed high). For each mixture the
Table 1 chain is applied exactly as to the data -- median of the measured
amplitudes, floor from the zero scenario and recovery fraction of the NOMINAL
run -- and compared with the median of the true amplitudes (measured with the
same estimator on the noiseless, unconvolved models). The same is done for the
median per-galaxy A1/A2 with the ratio bias b of the nominal run.

Usage:  python3 median_recovery_test.py --inter ../data/intermediate --out ../data/median_recovery_test.csv
"""
import argparse

import numpy as np
import pandas as pd

MIXES = {'skewed low (60/25/15)': (0.60, 0.25, 0.15), 'uniform (1/3 each)': (1 / 3, 1 / 3, 1 / 3),
         'skewed high (15/25/60)': (0.15, 0.25, 0.60)}
RUNS = ('inj_f444w_scale0.5', 'inj_f444w_m', 'inj_f444w_scale1.5')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inter', default='../data/intermediate')
    ap.add_argument('--centering', default='catalog')
    ap.add_argument('--out', default='../data/median_recovery_test.csv')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    obj = {r: pd.read_csv(f'{a.inter}/{r}/injection_per_object.csv') for r in RUNS}
    obj = {r: o[o.centering == a.centering] for r, o in obj.items()}
    nom = obj['inj_f444w_m']
    rows = []
    for zc in sorted(nom.zbin.unique()):
        n = nom[nom.zbin == zc]
        z0, mt = n[n.scenario == 'zero'], n[n.scenario == 'matched']
        cal = {}
        for k in ('A1', 'A2'):
            f, m, t = z0[k].median(), mt[k].median(), mt[f'{k}_true'].median()
            cal[k] = (f, np.sqrt(max(m * m - f * f, 0)) / t)
        b = mt.R12.median() / (mt.A1_true / mt.A2_true).median()
        groups = [obj[r][(obj[r].zbin == zc) & (obj[r].scenario == 'matched')] for r in RUNS]
        size = min(len(g) for g in groups)
        for name, w in MIXES.items():
            mix = pd.concat([g.sample(n=int(round(wi * size * 3)), replace=True,
                                      random_state=int(rng.integers(1 << 31)))
                             for g, wi in zip(groups, w)])
            row = {'z': zc, 'mixture': name, 'N': len(mix)}
            for k in ('A1', 'A2'):
                f, rec = cal[k]
                m = mix[k].median()
                row[f'{k}_true_median'] = mix[f'{k}_true'].median()
                row[f'{k}_recovered'] = np.sqrt(max(m * m - f * f, 0)) / rec
                row[f'{k}_rel_error'] = row[f'{k}_recovered'] / row[f'{k}_true_median'] - 1
            row['R12_true_median'] = (mix.A1_true / mix.A2_true).median()
            row['R12_recovered'] = mix.R12.median() / b
            row['R12_rel_error'] = row['R12_recovered'] / row['R12_true_median'] - 1
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(a.out, index=False)
    pd.set_option('display.width', 200)
    print(t[['z', 'mixture', 'A1_rel_error', 'A2_rel_error', 'R12_rel_error']].round(3).to_string(index=False))
    print("\nrelative error of the recovered median (recovered/true - 1), per mixture:")
    print(t.groupby('mixture')[['A1_rel_error', 'A2_rel_error', 'R12_rel_error']]
          .agg(lambda x: f"{x.mean():+.3f} (range {x.min():+.3f} .. {x.max():+.3f})").to_string())


if __name__ == '__main__':
    main()
