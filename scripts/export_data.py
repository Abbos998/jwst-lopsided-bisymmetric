#!/usr/bin/env python3
"""
export_data.py — write the supplementary tables (data/*.csv) directly from the
pipeline output files, with the full calculation chain and explicit units, so
every number in the paper can be traced and recomputed (verify_table1.py).

Normalisation suffixes in the column names:
  _pipe : pipeline normalisation |sum I e^{im phi}| / sum I
  _conv : conventional normalisation 2|<I e^{im phi}>|/<I> = 2 x _pipe
Ratios (A1/A2) and recovery fractions are normalisation-free.

Inputs (relative to --root; a missing input skips that table):
  table1_disk_final.csv           make_table1.py with matched injections
  table1_ell_v7.csv               make_table1.py, early types
  corrected_disk_f444w.csv        run_corrected_am.py per-galaxy (Table 2 / Fig. 3 steps)
  inj_f444w_m/injection_summary.csv injection_test.py (zero + matched)
  restframe_disk_m.csv            restframe_test.py
  sfr_split_snr100.csv            fix_sfr.py
  psf/F444W_stars.csv, psf/*_empirical_psf.fits  star_psf_astrometry.py

Usage:
    python3 export_data.py --root ~/Downloads/pasp_v3 --out ../data
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


STEP_KEYS = ['V0 original', '+ WCS centre and field offset', '+ background', '+ deprojection',
             '+ masking', '+ 0.5-1.5 r_d annulus', '+ Sersic PA (adopted)']
Z_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
Z_CEN = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50]


def load(root, name):
    p = os.path.join(root, name)
    if not os.path.exists(p):
        print(f"  skip: {p} not found")
        return None
    return pd.read_csv(p)


def psf_fwhm(path):
    from astropy.io import fits
    with fits.open(path) as h:
        psf, hdr = np.asarray(h[0].data, float), h[0].header
    c = psf.shape[0] // 2
    y, x = np.indices(psf.shape)
    r2 = ((x - c) ** 2 + (y - c) ** 2).ravel()
    core = (r2 < 9) & (psf.ravel() > 0)
    slope = np.polyfit(r2[core], np.log(psf.ravel()[core]), 1)[0]
    return 2 * 1.1774 * np.sqrt(-1 / (2 * slope)) * hdr.get('PIXSCALE', np.nan), hdr.get('NSTARS', np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R, O = args.root, args.out

    t = load(R, 'table1_disk_final.csv')
    if t is not None:
        out = pd.DataFrame({'z': t.z, 'N_valid': t.N, 'N_total': t.get('N_total', np.nan)})
        for k in ('A1', 'A2'):
            out[f'{k}_meas_pipe'] = t[k]
            out[f'{k}_meas_err_pipe'] = t[f'{k}_err']
            out[f'{k}_floor_pipe'] = t[f'{k}_floor']
            out[f'{k}_floorcorr_pipe'] = t[f'{k}_corr']
            out[f'{k}_recovery'] = t[f'{k}_rec']
            out[f'{k}_recovery_err'] = t.get(f'{k}_rec_err', np.nan)
            out[f'{k}_final_conv'] = t[f'{k}_final_std']
            # total = sqrt(stat_calib^2 + syst^2); the two parts are given separately
            out[f'{k}_final_err_conv'] = t.get(f'{k}_final_err', np.nan)
            out[f'{k}_final_err_stat_conv'] = t.get(f'{k}_final_err_stat', np.nan)
            out[f'{k}_final_syst_conv'] = t.get(f'{k}_final_syst', np.nan)
        out['A3_floorcorr_conv'] = t.A3_corr_std
        out['A1A2_obs'] = t.R12
        out['A1A2_obs_err'] = t.R12_err
        out['A1A2_bias'] = t.R12_bias_matched
        out['A1A2_corrected'] = t.R12_debiased_matched
        out['A1A2_corrected_err'] = t.get('R12_final_err', np.nan)
        out['A1A2_corrected_err_stat'] = t.get('R12_final_err_stat', np.nan)
        out['A1A2_corrected_syst'] = t.get('R12_final_syst', np.nan)
        out['syst_runs'] = t.get('syst_runs', '')
        out['calib_err_included'] = t.get('calib_err', False)
        out.to_csv(f'{O}/table1_disks_F444W.csv', index=False)
        print('  wrote table1_disks_F444W.csv')

    e = load(R, 'table1_ell_v7.csv')
    if e is not None:
        pd.DataFrame({'z': e.z, 'N_valid': e.N, 'A1_floorcorr_conv': e.A1_corr_std,
                      'A1_floorcorr_err_conv': e.A1_corr_std_err, 'A2_floorcorr_conv': e.A2_corr_std,
                      'A2_floorcorr_err_conv': e.A2_corr_std_err, 'A1A2_obs': e.R12,
                      'A1A2_corrected_approx': e.get('R12_debiased', np.nan)}) \
            .to_csv(f'{O}/early_types_F444W.csv', index=False)
        print('  wrote early_types_F444W.csv')

    # Table 2 / Fig. 3: median amplitudes after each pipeline step (V0-V6), computed from
    # the per-galaxy measurements of the adopted run (the same file as Table 1, so the
    # last step equals A_meas of Table 1). The per-field centre offsets are part of V1.
    d = load(R, 'corrected_disk_f444w.csv')
    if d is not None:
        rng = np.random.default_rng(0)
        zb = pd.cut(d.z, Z_EDGES, labels=Z_CEN).astype(float)
        rows = []
        for v, label in zip(('V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'), STEP_KEYS):
            if f'A1_{v}' not in d:
                continue
            for zc in Z_CEN:
                g = d[np.isclose(zb, zc)]
                row = {'step': label, 'z': zc, 'N': int(g[f'A1_{v}'].notna().sum())}
                for k, x in (('A1', g[f'A1_{v}']), ('A2', g[f'A2_{v}']), ('A1A2', g[f'A1_{v}'] / g[f'A2_{v}'])):
                    x = x.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
                    unit = '' if k == 'A1A2' else '_pipe'
                    row[f'{k}{unit}'] = np.median(x) if len(x) else np.nan
                    row[f'{k}_err{unit}'] = (np.std(np.median(rng.choice(x, (500, len(x))), axis=1))
                                             if len(x) > 4 else np.nan)
                rows.append(row)
        pd.DataFrame(rows).to_csv(f'{O}/correction_steps_V0_V6.csv', index=False)
        print('  wrote correction_steps_V0_V6.csv')

    inj = load(R, 'inj_f444w_m/injection_summary.csv')
    if inj is not None:
        z0 = inj[(inj.scenario == 'zero') & (inj.centering == 'catalog')].set_index('zbin')
        mt = inj[(inj.scenario == 'matched') & (inj.centering == 'catalog')].set_index('zbin')
        out = pd.DataFrame({'z': z0.index})
        for k in ('A1', 'A2'):
            f = z0[f'{k}_med'].values
            out[f'{k}_floor_pipe'] = f
            out[f'{k}_floor_conv'] = 2 * f
            if len(mt):
                m = mt[f'{k}_med'].reindex(z0.index).values
                tr = mt[f'{k}_true_med'].reindex(z0.index).values
                out[f'{k}_matched_meas_pipe'] = m
                out[f'{k}_matched_true_pipe'] = tr
                out[f'{k}_recovery'] = np.sqrt(np.clip(m ** 2 - f ** 2, 0, None)) / tr
        out.to_csv(f'{O}/injection_calibration_F444W.csv', index=False)
        print('  wrote injection_calibration_F444W.csv')

    r = load(R, 'restframe_disk_m.csv')
    if r is not None:
        r = r.rename(columns={c: c.replace('R12', 'A1A2') for c in r.columns})
        # restframe_test.py labels '0.70 um' -> 'rest 0.7 um' (as in the manuscript)
        r['set'] = [s if not s.endswith(' um') or s.startswith('rest') else
                    f"rest {float(s.split()[0]):.1f} um" for s in r['set']]
        r = r.rename(columns={'A1': 'A1_conv', 'A1_err': 'A1_err_conv', 'A2': 'A2_conv',
                              'A2_err': 'A2_err_conv', 'A3': 'A3_conv', 'A3_err': 'A3_err_conv'})
        r.to_csv(f'{O}/restframe_test.csv', index=False)
        print('  wrote restframe_test.csv')

    f = load(R, 'sfr_split_snr100.csv')
    if f is not None:
        f.rename(columns={'A1_SF': 'A1_SF_conv', 'A1_SF_err': 'A1_SF_err_conv', 'A1_Q': 'A1_Q_conv',
                          'A1_Q_err': 'A1_Q_err_conv', 'ratio': 'A1_SF_over_Q',
                          'ratio_err': 'A1_SF_over_Q_err'}).to_csv(f'{O}/sfr_split_snr100.csv', index=False)
        print('  wrote sfr_split_snr100.csv')

    st = load(R, 'psf/F444W_stars.csv')
    if st is not None:
        rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
        g = st.groupby('field').agg(N_stars=('dx', 'size'), dx_median=('dx', 'median'),
                                     dy_median=('dy', 'median'), dx_sigma=('dx', rs),
                                     dy_sigma=('dy', rs)).reset_index()
        g.to_csv(f'{O}/star_astrometry_F444W.csv', index=False)
        print('  wrote star_astrometry_F444W.csv')

    psfs = sorted(glob.glob(os.path.join(R, 'psf', '*_empirical_psf.fits')))
    if psfs:
        rows = []
        for p in psfs:
            fw, ns = psf_fwhm(p)
            rows.append({'filter': os.path.basename(p).split('_')[0], 'fwhm_arcsec': round(fw, 4),
                         'stars_used': ns})
        pd.DataFrame(rows).to_csv(f'{O}/empirical_psf.csv', index=False)
        print('  wrote empirical_psf.csv')


if __name__ == '__main__':
    main()
