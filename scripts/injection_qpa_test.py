#!/usr/bin/env python3
"""
injection_qpa_test.py -- effect of errors in the catalogue axis ratio q and
position angle PA (used for deprojection) on A1, A2 and on Table 1.

The injected galaxies are built and projected with the TRUE (q, PA), as in
injection_test.py. Every image (zero and matched scenarios) is then measured
with the true (q, PA) -- the 'ideal' calibration of the paper -- and with
perturbed (q + dq, PA + dPA). Because the same image is measured, the
difference isolates the deprojection error. For each error scenario the
noise floor, recovered fraction and ratio bias are recomputed, and Table 1
is re-derived from the observed medians: this is the calibration one would
have obtained had the simulations included deprojection errors.

Error scenarios (the same draw is used for the zero and matched images of
a galaxy, as a catalogue error belongs to the galaxy):
  ideal       no error (reproduces the paper's calibration; checked)
  T1, T2, T3  fixed Gaussian (sigma_q, sigma_PA) = (0.02, 2), (0.05, 5), (0.10, 10 deg)
  SN          noise-induced error measured on each injected image: q and PA
              from flux-weighted second moments within 2 r_d of the noisy
              image minus those of the noiseless convolved image
  SNx2        SN doubled (allowance for model-mismatch errors of real fits)
  SN_q, SN_pa SN applied to q only / to PA only
  CAT, CATx2  Gaussian errors with the catalogue uncertainties (only with
              --q_err_col/--pa_err_col or --err_fits; see below)
  EMP         PA error drawn from the observed differences between the
              catalogue Sersic PA and the moments PA of the real galaxies in the
              same z bin and q tercile (--emp_dpa); q error as SN. The full
              difference is an upper bound: it also contains the moments-PA
              error and isophote twists between the annulus and the whole galaxy.
  EMP_r2      EMP / sqrt(2) (both PA estimates contribute equally)
  EMP_x2q     EMP with twice the noise-induced q error

Run from the package scripts/ directory, e.g.
  python3 injection_qpa_test.py --scripts . --catalog $DATA/dja_snr_f444w.csv \
     --table1 ../data/table1_disks_F444W.csv --psf $ROOT/psf/F444W_empirical_psf.fits \
     --noise_acf $ROOT/noise_acf_f444w.npy --pixscale 0.05 --out $ROOT/qpa_f444w
Catalogue uncertainties: either columns of --catalog (--q_err_col, --pa_err_col),
or columns of the DJA morpho FITS (--err_fits $M --idx_col idx --q_err_col ...
--pa_err_col ...), rows indexed by the catalogue's idx. List candidates with
  python3 injection_qpa_test.py --list_err_columns $M
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_core as cc

FIXED = {'T1': (0.02, 2.0), 'T2': (0.05, 5.0), 'T3': (0.10, 10.0)}


def moments_qpa(img, x0, y0, rmax):
    ny, nx = img.shape
    y, x = np.mgrid[:ny, :nx]
    dx, dy = x - x0, y - y0
    sel = dx * dx + dy * dy <= rmax * rmax
    w = np.clip(img[sel], 0, None)
    if w.sum() <= 0:
        return np.nan, np.nan
    dx, dy = dx[sel], dy[sel]
    qxx, qyy, qxy = [(w * v).sum() / w.sum() for v in (dx * dx, dy * dy, dx * dy)]
    t = np.sqrt(((qxx - qyy) / 2) ** 2 + qxy ** 2)
    l1, l2 = (qxx + qyy) / 2 + t, (qxx + qyy) / 2 - t
    return np.sqrt(max(l2, 1e-12) / l1), 0.5 * np.degrees(np.arctan2(2 * qxy, qxx - qyy))


def worker(items, cfg):
    inj = cc.load_module(os.path.join(cfg['scripts'], 'injection_test.py'), 'inj')
    measure = cc.get_measure(cfg['scripts'], cfg['am_func'])
    psf, acf, npix, aper = cfg['psf'], cfg['acf'], cfg['npix'], cfg['aper']
    rows = []
    for g in items:
        rng = np.random.default_rng(g['seed_err'])
        eps_m = cfg['eps'][round(g['zbin'], 2)]
        imgs = {}
        for scen, eps, seed in (('zero', (0., 0., 0.), g['seed_zero']),
                                ('matched', eps_m, g['seed_match'])):
            clean, x0, y0 = cc.build_clean(g, eps, npix, inj)
            img = cc.observe(clean, psf, g, x0, y0, seed, aper, acf, inj)
            truth = measure(clean, x0, y0, g['q'], g['pa'], g['re_pix'])
            imgs[scen] = (img, clean, x0, y0, truth)
        # noise-induced q, PA errors, from the matched image of this galaxy
        img, clean, x0, y0, _ = imgs['matched']
        conv = inj.fftconvolve(clean, psf, mode='same')
        qc, pc = moments_qpa(conv, x0, y0, 2 * g['re_pix'])
        qn, pn = moments_qpa(img, x0, y0, 2 * g['re_pix'])
        dq_sn, dpa_sn = qn - qc, ((pn - pc + 90) % 180) - 90
        errs = {'ideal': (0., 0.)}
        for k, (sq, sp) in FIXED.items():
            errs[k] = (rng.normal(0, sq), rng.normal(0, sp))
        errs.update(SN=(dq_sn, dpa_sn), SNx2=(2 * dq_sn, 2 * dpa_sn),
                    SN_q=(dq_sn, 0.), SN_pa=(0., dpa_sn))
        if cfg['cat_err']:
            sq, sp = g.get('q_err', np.nan), g.get('pa_err', np.nan)
            if np.isfinite(sq) and np.isfinite(sp):
                e1, e2 = rng.normal(), rng.normal()
                errs['CAT'] = (sq * e1, sp * e2)
                errs['CATx2'] = (2 * sq * e1, 2 * sp * e2)
        if cfg['emp'] is not None:
            # PA error drawn from the observed (Sersic - moments) PA differences
            # of galaxies in the same redshift bin and axis-ratio tercile
            cell = cfg['emp'].get((round(g['zbin'], 2), int(np.digitize(g['q'], cfg['emp_qedges']))))
            if cell is not None and len(cell):
                dp = float(cell[np.random.default_rng(g['seed_err'] + 7).integers(len(cell))])
                errs['EMP'] = (dq_sn, dp)
                errs['EMP_r2'] = (dq_sn, dp / np.sqrt(2))
                errs['EMP_x2q'] = (2 * dq_sn, dp)
        if cfg['variants']:
            errs = {k: v for k, v in errs.items() if k in cfg['variants']}
        cx = x0 + cfg['sigma_cat'] * g['cdx']
        cy = y0 + cfg['sigma_cat'] * g['cdy']
        for scen, (img, clean, x0_, y0_, truth) in imgs.items():
            for v, (dq, dp) in errs.items():
                qm = float(np.clip(g['q'] + dq, 0.3, 1.0))
                a1, a2, a3 = measure(img, cx, cy, qm, g['pa'] + dp, g['re_pix'])
                rows.append(dict(variant=v, scenario=scen, zbin=g['zbin'], q=g['q'],
                                 snr=g['snr'], re_pix=g['re_pix'], dq=dq, dpa=dp,
                                 A1=a1, A2=a2, A1_true=truth[0], A2_true=truth[1]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    if len(sys.argv) == 3 and sys.argv[1] == '--list_err_columns':
        from astropy.io import fits
        names = fits.open(sys.argv[2])[1].columns.names
        for k in ('err', 'axrat', 'axis', 'angle', 'theta', 'q_', 'pa'):
            print(k, [n for n in names if k in n.lower()][:30])
        return
    cc.add_common_args(ap)
    ap.add_argument('--q_err_col')
    ap.add_argument('--pa_err_col', help='PA uncertainty column (degrees; radians are converted '
                    'if all values < 3.2)')
    ap.add_argument('--err_fits', help='FITS with the error columns, rows indexed by --idx_col')
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--emp_dpa', help='CSV zbin,q,dpa of observed (Sersic - moments) PA '
                    'differences (empirical_dpa_f444w.csv): adds scenarios EMP (full '
                    'difference as PA error), EMP_r2 (difference/sqrt2) and EMP_x2q '
                    '(EMP with twice the noise-induced q error)')
    ap.add_argument('--variants', help='comma list of scenarios to run (default: all)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inj = cc.load_module(os.path.join(a.scripts, 'injection_test.py'), 'inj')
    d = cc.load_catalog(a, inj)
    raw = pd.read_csv(a.catalog)
    cat_err = False
    if a.q_err_col and a.pa_err_col:
        if a.err_fits:
            from astropy.io import fits
            t = fits.open(a.err_fits)[1].data
            if a.idx_col not in raw.columns:
                sys.exit(f'--idx_col {a.idx_col} not in the catalogue')
            idx = raw[a.idx_col].values[d.row.values].astype(int)
            d['q_err'] = np.asarray(t[a.q_err_col], float)[idx]
            d['pa_err'] = np.asarray(t[a.pa_err_col], float)[idx]
        else:
            d['q_err'] = pd.to_numeric(raw[a.q_err_col], errors='coerce').values[d.row.values]
            d['pa_err'] = pd.to_numeric(raw[a.pa_err_col], errors='coerce').values[d.row.values]
        if np.nanmax(d.pa_err) < 3.2:
            d['pa_err'] = np.degrees(d.pa_err)
        cat_err = True
        print(f"catalogue errors: median sigma_q={np.nanmedian(d.q_err):.3f}, "
              f"sigma_PA={np.nanmedian(d.pa_err):.1f} deg")
    items = cc.draw_sample(d, a.n_per_bin, a.seed)
    acf = np.load(a.noise_acf) if a.noise_acf else None
    emp, qedges = None, None
    if a.emp_dpa:
        e = pd.read_csv(a.emp_dpa).dropna()
        qedges = np.quantile(e.q, [1 / 3, 2 / 3])
        e['qc'] = np.digitize(e.q, qedges)
        emp = {(round(float(z), 2), int(c)): g.dpa.values for (z, c), g in e.groupby(['zbin', 'qc'])}
        rs = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
        print('empirical dPA robust sigma per bin [deg]: ' +
              ', '.join(f'{z}: {rs(g.dpa):.1f}' for z, g in e.groupby('zbin')))
    cfg = dict(scripts=os.path.abspath(a.scripts), am_func=a.am_func,
               psf=cc.psf_from_fits(a.psf, a.psf_pixscale, a.pixscale, inj), acf=acf,
               npix=int(round(a.cutout_arcsec / a.pixscale)), aper=a.snr_aper_arcsec / a.pixscale,
               sigma_cat=a.sigma_cat, eps=cc.matched_eps(a.table1), cat_err=cat_err,
               emp=emp, emp_qedges=qedges,
               variants=set(a.variants.split(',')) if a.variants else None)
    print(f'{len(d)} galaxies after cuts; {len(items)} injected models; {a.workers} workers')
    rows = pd.DataFrame(cc.run_parallel(worker, items, cfg, a.workers))
    rows.to_csv(os.path.join(a.out, 'qpa_per_object.csv'), index=False)

    cal = cc.calibrate(rows, a.table1)
    cal.to_csv(os.path.join(a.out, 'qpa_calibration.csv'), index=False)
    gr = cc.growth(cal)
    order = ['ideal', 'T1', 'T2', 'T3', 'SN', 'SNx2', 'SN_q', 'SN_pa', 'EMP_r2', 'EMP',
             'EMP_x2q', 'CAT', 'CATx2']
    gr['o'] = gr.variant.map({k: i for i, k in enumerate(order)})
    gr = gr.sort_values('o').drop(columns='o')
    gr.to_csv(os.path.join(a.out, 'qpa_growth.csv'), index=False)

    # sizes of the noise-induced errors per bin
    rsig = lambda v: 1.4826 * np.median(np.abs(v))
    mt = rows[rows.scenario == 'matched']
    sn = pd.DataFrame({f'{k}_{c}': mt[mt.variant == k].groupby('zbin')[c].agg(rsig)
                       for k in ('SN', 'EMP') if (mt.variant == k).any() for c in ('dq', 'dpa')})
    sn.to_csv(os.path.join(a.out, 'qpa_sn_errors.csv'))
    pd.set_option('display.precision', 4)
    print('\nCheck: ideal calibration reproduced here vs Table 1 (differences = sampling noise)')
    print(cc.compare_to_paper(cal, a.table1, 'ideal'))
    print('\nApplied q / PA errors per bin (robust sigma):\n' + sn.round(3).to_string())
    print('\nGrowth factors and mean corrected A1/A2 with each calibration:')
    print(gr.round(4).to_string(index=False))
    print(f'\nOutputs in {a.out}')


if __name__ == '__main__':
    main()
