#!/usr/bin/env python3
"""
injection_sample_match.py — are the injected galaxies representative of the
observed sample? Per redshift bin: median and 16-84 percentiles of the disk
radius r_d [cutout pix] and axis ratio q of the observed galaxies with a valid
measurement and of the injected galaxies (nominal run), and the medians of the
Sersic index n and aperture S/N used for the injections (drawn from the same
catalogue). Also the injected (matched) amplitudes vs the observed corrected
medians.

Usage:  python3 injection_sample_match.py --inter ../data/intermediate --table ../data/table1_disks_F444W.csv \
            --out ../data/injection_sample_match.csv
"""
import argparse

import numpy as np
import pandas as pd

Z_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
Z_CEN = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50]


def q3(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.percentile(x, [16, 50, 84]) if len(x) else (np.nan,) * 3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inter', default='../data/intermediate')
    ap.add_argument('--table', default='../data/table1_disks_F444W.csv')
    ap.add_argument('--out', default='../data/injection_sample_match.csv')
    ap.add_argument('--rref', type=float, default=0.04,
                    help='arcsec/pix of the catalogue disk radius (DJA: 0.04)')
    ap.add_argument('--pixscale', type=float, default=0.05, help='arcsec/pix of the cutouts')
    a = ap.parse_args()
    obs = pd.read_csv(f'{a.inter}/corrected_disk_f444w.csv')
    obs = obs[obs.A1_V6.notna()].copy()
    obs['disk_r'] = obs.disk_r * a.rref / a.pixscale      # catalogue -> cutout pixels
    obs['zb'] = pd.cut(obs.z, Z_EDGES, labels=Z_CEN).astype(float)
    inj = pd.read_csv(f'{a.inter}/inj_f444w_m/injection_per_object.csv')
    # injected galaxies with a valid measurement (same >= 50 pixel rule as the data)
    inj = inj[(inj.scenario == 'matched') & (inj.centering == 'catalog') & inj.A1.notna()]
    t1 = pd.read_csv(a.table).set_index('z')
    rows = []
    for zc in Z_CEN:
        o, i = obs[np.isclose(obs.zb, zc)], inj[np.isclose(inj.zbin, zc)]
        r = {'z': zc, 'N_obs': len(o), 'N_inj': len(i)}
        for lab, x in (('rd_obs', o.disk_r), ('rd_inj', i.re_pix), ('q_obs', o.q), ('q_inj', i.q)):
            r[f'{lab}_p16'], r[f'{lab}_med'], r[f'{lab}_p84'] = q3(x)
        r['n_inj_med'], r['snr_inj_med'] = i.n.median(), i.snr.median()
        for k in ('A1', 'A2'):
            r[f'{k}_inj_true_conv'] = 2 * i[f'{k}_true'].median()
            r[f'{k}_table1_conv'] = t1.loc[zc, f'{k}_final_conv']
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(a.out, index=False)
    pd.set_option('display.width', 220)
    show = t[['z', 'N_obs', 'N_inj', 'rd_obs_med', 'rd_inj_med', 'rd_obs_p16', 'rd_inj_p16', 'rd_obs_p84',
              'rd_inj_p84', 'q_obs_med', 'q_inj_med', 'n_inj_med', 'snr_inj_med',
              'A1_inj_true_conv', 'A1_table1_conv', 'A2_inj_true_conv', 'A2_table1_conv']]
    print(show.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
