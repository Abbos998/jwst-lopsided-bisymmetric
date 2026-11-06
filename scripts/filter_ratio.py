#!/usr/bin/env python3
"""
filter_ratio.py — wavelength dependence of the amplitudes for the SAME galaxies:
median per-galaxy ratio A_m(F277W) / A_m(F444W) per redshift bin (no model
correction; F277W and F444W have similar PSFs and S/N). Quoted in Sect. 4.2.

Usage:  python3 filter_ratio.py --ref ~/Downloads/pasp_v3/corrected_disk_f444w.csv \
            --other ~/Downloads/pasp_v3/corrected_disk_f277w.csv --out ../data/filter_ratio_F277W_F444W.csv
"""
import argparse

import numpy as np
import pandas as pd

Z_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
Z_CEN = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--other', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--nboot', type=int, default=500)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    cols = ['idx', 'z', 'A1_V6', 'A2_V6']
    d = pd.read_csv(a.ref, usecols=cols).merge(pd.read_csv(a.other, usecols=cols), on=['idx', 'z'],
                                                suffixes=('_ref', '_oth'))
    d['zb'] = pd.cut(d.z, Z_EDGES, labels=Z_CEN).astype(float)
    rows = []
    for zc in Z_CEN:
        s = d[np.isclose(d.zb, zc)]
        r = {'z': zc}
        for k in ('A1', 'A2'):
            x = (s[f'{k}_V6_oth'] / s[f'{k}_V6_ref']).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            r[f'N_{k}'] = len(x)
            r[f'{k}_ratio'] = np.median(x) if len(x) else np.nan
            r[f'{k}_ratio_err'] = (np.std(np.median(rng.choice(x, (a.nboot, len(x))), axis=1))
                                   if len(x) > 4 else np.nan)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(a.out, index=False)
    print(t.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
