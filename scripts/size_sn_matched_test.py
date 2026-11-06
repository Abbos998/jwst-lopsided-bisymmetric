#!/usr/bin/env python3
"""
size_sn_matched_test.py -- does the growth of A1 and A2 survive in a
subsample with the SAME distribution window of resolution (r_d/FWHM) and
aperture S/N in every redshift bin?

1. The valid galaxies of the paper (corrected_disk_f444w.csv, A1_V6/A2_V6)
   are joined to the injection catalogue (dja_snr_f444w.csv) on idx.
2. Resolution: a common window in x = r_d/FWHM, the overlap of the central
   ranges (default 10th-90th percentiles) of all bins with N >= 20 (or
   --x_range). S/N: a common minimum S/N in every bin (default the median S/N
   of the highest-redshift bin), because the S/N distributions of low- and
   high-redshift bins barely overlap (or --snr_range). --match selects size,
   snr or both.
3. For the galaxies inside the window, the bin medians of the measured
   amplitudes are recomputed, and a dedicated injection-recovery calibration
   is run on galaxies drawn from the same window (same code, PSF, noise ACF
   and estimator as the paper; 'ideal' calibration of injection_qpa_test.py).
4. The full sample is calibrated in the same run with the same seed, as a
   reference.
Output: N per bin, corrected amplitudes, growth factors and mean A1/A2 for
the matched subsample and the reference. Bins with N < 20 are flagged and
should not be over-interpreted.

Run from the package scripts/ directory (see run_selection_tests.sh).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_core as cc
import injection_qpa_test as iq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cc.add_common_args(ap)
    ap.add_argument('--galaxies', required=True, help='data/intermediate/corrected_disk_f444w.csv')
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--fwhm_arcsec', type=float, default=0.156, help='F444W PSF FWHM')
    ap.add_argument('--pct', type=float, nargs=2, default=[10, 90])
    ap.add_argument('--x_range', type=float, nargs=2, help='r_d/FWHM window (overrides --pct)')
    ap.add_argument('--snr_range', type=float, nargs=2, help='S/N window (overrides --pct)')
    ap.add_argument('--min_n', type=int, default=20)
    ap.add_argument('--match', choices=['size', 'snr', 'both'], default='both',
                    help='match the resolution r_d/FWHM, the S/N, or both')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inj = cc.load_module(os.path.join(a.scripts, 'injection_test.py'), 'inj')

    cat = cc.load_catalog(a, inj)
    raw = pd.read_csv(a.catalog)
    if a.idx_col not in raw.columns:
        sys.exit(f'--idx_col {a.idx_col} not in {a.catalog}; columns: {list(raw.columns)[:40]}')
    cat['idx'] = raw[a.idx_col].values[cat.row.values].astype(int)
    gal = pd.read_csv(a.galaxies).dropna(subset=['A1_V6', 'A2_V6'])
    m = gal[[a.idx_col, 'z', 'A1_V6', 'A2_V6']].rename(columns={a.idx_col: 'idx', 'z': 'z_meas'}) \
        .merge(cat, on='idx', how='inner')
    dz = np.abs(m.z_meas - m.z)
    print(f'{len(gal)} valid galaxies, {len(cat)} in the injection catalogue, {len(m)} joined on idx '
          f'(median |dz| between the two files {np.median(dz):.1e})')
    if len(m) < 0.9 * len(gal):
        print('  WARNING: fewer than 90% of the valid galaxies were joined; check --idx_col')
    fwhm_pix = a.fwhm_arcsec / a.pixscale
    m['x'] = m.re_pix / fwhm_pix
    m['zbin'] = cc.Z_CEN[np.clip(np.digitize(m.z_meas, cc.Z_EDGES) - 1, 0, len(cc.Z_CEN) - 1)]

    # --- common window
    big = [z for z, g in m.groupby('zbin') if len(g) >= a.min_n]
    # resolution: common window of r_d/FWHM (overlap of the central ranges of all bins)
    xlo, xhi = m.x.min(), m.x.max()
    if a.match in ('size', 'both'):
        if a.x_range:
            xlo, xhi = a.x_range
        else:
            xlo = max(np.percentile(m[m.zbin == z].x, a.pct[0]) for z in big)
            xhi = min(np.percentile(m[m.zbin == z].x, a.pct[1]) for z in big)
            if xlo >= xhi:
                print(f'  r_d/FWHM ranges do not overlap at the {a.pct} percentiles; using the overlap of the full ranges')
                xlo = max(m[m.zbin == z].x.min() for z in big)
                xhi = min(m[m.zbin == z].x.max() for z in big)
    # S/N: the S/N distributions of low- and high-redshift bins barely overlap (median
    # ~500 at z~0.75 vs ~50 at z~4.5), so a common window would keep only the faint tail
    # of the low-z bins. Instead a common minimum S/N is applied to every bin (default:
    # the median S/N of the highest-redshift bin with N >= min_n), which removes the
    # noisiest galaxies that dominate the high-redshift bins.
    slo, shi = m.snr.min(), np.inf
    if a.match in ('snr', 'both'):
        if a.snr_range:
            slo, shi = a.snr_range
        else:
            slo = np.median(m[m.zbin == max(big)].snr)
    w = m[(m.x >= xlo) & (m.x <= xhi) & (m.snr >= slo) & (m.snr <= shi)]
    print(f'match={a.match}; r_d/FWHM {xlo:.2f}-{xhi:.2f}, S/N >= {slo:.0f}' + (f' and <= {shi:.0f}' if np.isfinite(shi) else '') + f'; {len(w)} galaxies')

    # --- matched "Table 1" (measured medians of the subsample; injected amplitudes as Table 1)
    t1 = pd.read_csv(a.table1).set_index('z')
    rows = []
    for zc in cc.Z_CEN:
        g = w[w.zbin == zc]
        full = m[m.zbin == zc]
        r = dict(z=zc, N=len(g), N_full=len(full),
                 x_med=g.x.median() if len(g) else np.nan, x_med_full=full.x.median(),
                 snr_med=g.snr.median() if len(g) else np.nan, snr_med_full=full.snr.median(),
                 A1_meas_pipe=g.A1_V6.median() if len(g) else np.nan,
                 A2_meas_pipe=g.A2_V6.median() if len(g) else np.nan,
                 A1A2_obs=(g.A1_V6 / g.A2_V6).median() if len(g) else np.nan,
                 A1_final_conv=t1.loc[zc, 'A1_final_conv'], A2_final_conv=t1.loc[zc, 'A2_final_conv'])
        rows.append(r)
    MT = pd.DataFrame(rows)
    mt_path = os.path.join(a.out, 'matched_table.csv')
    MT.to_csv(mt_path, index=False)
    print(MT[['z', 'N', 'N_full', 'x_med', 'x_med_full', 'snr_med', 'snr_med_full']].round(2).to_string(index=False))

    # --- injection calibrations: matched window and full sample, same seed
    psf = cc.psf_from_fits(a.psf, a.psf_pixscale, a.pixscale, inj)
    acf = np.load(a.noise_acf) if a.noise_acf else None
    base = dict(scripts=os.path.abspath(a.scripts), am_func=a.am_func, psf=psf, acf=acf,
                npix=int(round(a.cutout_arcsec / a.pixscale)), aper=a.snr_aper_arcsec / a.pixscale,
                sigma_cat=a.sigma_cat, eps=cc.matched_eps(a.table1), cat_err=False,
                emp=None, emp_qedges=None, variants={'ideal'})
    cal = {}
    for name, sample, table in (('matched', cat[cat.idx.isin(w.idx)], mt_path),
                                ('full', cat[cat.idx.isin(m.idx)], a.table1)):
        items = cc.draw_sample(sample.reset_index(drop=True), a.n_per_bin, a.seed)
        print(f'{name}: {len(items)} injected models')
        rows = pd.DataFrame(cc.run_parallel(iq.worker, items, base, a.workers))
        rows['variant'] = name
        c = cc.calibrate(rows, table)
        cal[name] = c
        c.to_csv(os.path.join(a.out, f'calibration_{name}.csv'), index=False)

    # --- results
    out = []
    for name, c in cal.items():
        c = c.set_index('zbin')
        n = MT.set_index('z')['N'] if name == 'matched' else MT.set_index('z')['N_full']
        ok = [z for z in c.index if n.loc[z] >= a.min_n]
        hi = [z for z in (4.5, 5.5) if z in ok]
        r = dict(sample=name, N_total=int(n.sum()), bins_used_high=','.join(str(z) for z in hi))
        for mm in (1, 2):
            lo_val = c.loc[0.75, f'A{mm}_final_conv']
            ok_lo = 0.75 in ok and np.isfinite(lo_val) and lo_val > 0
            r[f'A{mm}_growth'] = c.loc[hi, f'A{mm}_final_conv'].mean() / lo_val if (hi and ok_lo) else np.nan
            if not ok_lo:
                print(f'  {name}: A{mm} at z~0.75 is not usable (N or amplitude at the noise floor); growth not computed')
        r['R12_lo'] = c.loc[[z for z in ok if z < 3.5], 'R12_corr'].mean()
        r['R12_hi'] = c.loc[[z for z in ok if z > 3.5], 'R12_corr'].mean()
        out.append(r)
    G = pd.DataFrame(out)
    G.to_csv(os.path.join(a.out, 'growth_matched_vs_full.csv'), index=False)
    per = pd.DataFrame({'z': cc.Z_CEN, 'N_matched': MT.N, 'N_full': MT.N_full,
                        'A1_matched': cal['matched'].set_index('zbin').A1_final_conv.values,
                        'A1_full': cal['full'].set_index('zbin').A1_final_conv.values,
                        'A2_matched': cal['matched'].set_index('zbin').A2_final_conv.values,
                        'A2_full': cal['full'].set_index('zbin').A2_final_conv.values})
    per['flag'] = np.where(per.N_matched < a.min_n, f'N<{a.min_n}', '')
    per.to_csv(os.path.join(a.out, 'perbin_matched_vs_full.csv'), index=False)
    txt = (f'Match: {a.match}. r_d/FWHM {xlo:.2f}-{xhi:.2f}, S/N >= {slo:.0f}\n\n'
           + per.round(3).to_string(index=False) + '\n\n' + G.round(3).to_string(index=False) +
           f'\n\nBins with N < {a.min_n} are excluded from the growth factors and means; the matched growth '
           f'should be compared with the full-sample row (same seed), not with Table 1.')
    print('\n' + txt)
    open(os.path.join(a.out, 'size_sn_summary.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
