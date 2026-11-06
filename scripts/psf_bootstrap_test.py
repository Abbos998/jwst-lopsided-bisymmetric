#!/usr/bin/env python3
"""
psf_bootstrap_test.py -- propagation of the empirical-PSF uncertainty into
the noise floor, the recovered fractions and the corrected amplitudes of
Table 1.

PSF variants (all written to <out>/psf_variants/*.fits with a PIXSCALE
header and loaded exactly as the nominal PSF):
  nominal     the adopted $ROOT/psf/F444W_empirical_psf.fits
  restack     median of the re-extracted star stamps (must equal nominal;
              checks that the stamps are rebuilt as in star_psf_astrometry.py)
  boot_k      median stack of a bootstrap resample of the stars (k = 1..B):
              statistical uncertainty of the PSF, including its wings and
              shape
  fwhm_m5/p5  nominal PSF radially rescaled by 0.95 / 1.05
  azavg       azimuthal average of the nominal PSF (same radial profile, no
              ellipticity or spikes): sensitivity to PSF asymmetry, which
              can add or remove m=2 power
  field_<f>   stack of the stars of one field (fields with >= --field_min
              stars): field-to-field PSF variation

The star stamps are rebuilt from the cutouts written by
star_psf_astrometry.py ($ROOT/psf/stars_f444w/s_{i:06d}.fits) and its table
of kept stars ($ROOT/psf/F444W_stars.csv: i, wx, wy, dx, dy, field), with the
same sky subtraction, sub-pixel Fourier shift and normalisation.

Every variant is evaluated on identical injected models and noise fields
(common random numbers; see calib_core.py), with the adopted estimator and
catalogue centring. For each variant the noise floor, recovered fraction and
ratio bias are recomputed and Table 1 is re-derived from the observed
medians. The scatter over boot_k is the PSF contribution to the uncertainty
of each corrected value.

Run from the package scripts/ directory, e.g.
  python3 psf_bootstrap_test.py --scripts . --catalog $DATA/dja_snr_f444w.csv \
     --table1 ../data/table1_disks_F444W.csv --psf $ROOT/psf/F444W_empirical_psf.fits \
     --stars_csv $ROOT/psf/F444W_stars.csv --stars_dir $ROOT/psf/stars_f444w \
     --noise_acf $ROOT/noise_acf_f444w.npy --pixscale 0.05 --n_boot 20 --out $ROOT/psfboot_f444w
Without --stars_csv only nominal, fwhm_m5/p5 and azavg are run.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_core as cc


def fourier_shift(img, dx, dy):          # as in star_psf_astrometry.py
    ny, nx = img.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (kx * dx + ky * dy))))


def rebuild_stamps(stars_csv, stars_dir, stamp, pattern='s_{:06d}.fits'):
    from astropy.io import fits
    t = pd.read_csv(stars_csv)
    h = stamp // 2
    st, fld = [], []
    for r in t.itertuples():
        p = os.path.join(stars_dir, pattern.format(int(r.i)))
        if not os.path.exists(p):
            continue
        with fits.open(p) as cut:
            hdu = cut[0] if cut[0].data is not None else cut[1]
            img = np.nan_to_num(np.asarray(hdu.data, float))
        ny, nx = img.shape
        yy, xx = np.indices(img.shape)
        sky = img[np.hypot(xx - r.wx, yy - r.wy) > 0.35 * min(nx, ny)]
        img = img - np.median(sky)
        cx, cy = r.wx + r.dx, r.wy + r.dy
        ix, iy = int(round(cx)), int(round(cy))
        if not (h <= ix < nx - h and h <= iy < ny - h):
            continue
        s = fourier_shift(img[iy - h:iy + h + 1, ix - h:ix + h + 1], -(cx - ix), -(cy - iy))
        if s.sum() <= 0:
            continue
        st.append(s / s.sum())
        fld.append(str(getattr(r, 'field', 'all')))
    return np.array(st), np.array(fld)


def stack(st):
    p = np.clip(np.median(st, axis=0), 0, None)
    return p / p.sum()


def rescale(psf, s):
    """psf(r) -> psf(r/s) about the centre (bilinear), renormalised."""
    n = psf.shape[0]
    c = (n - 1) / 2
    y, x = np.indices(psf.shape, dtype=float)
    xs, ys = (x - c) / s + c, (y - c) / s + c
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    out = np.zeros_like(psf)
    for ddx, ddy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                        (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + ddx, y0 + ddy
        ok = (xi >= 0) & (xi < n) & (yi >= 0) & (yi < n)
        out[ok] += w[ok] * psf[yi[ok], xi[ok]]
    out = np.clip(out, 0, None)
    return out / out.sum()


def azavg(psf):
    n = psf.shape[0]
    c = (n - 1) / 2
    y, x = np.indices(psf.shape)
    r = np.hypot(x - c, y - c)
    rb = np.round(r * 2).astype(int)
    prof = np.bincount(rb.ravel(), psf.ravel()) / np.maximum(np.bincount(rb.ravel()), 1)
    out = prof[rb]
    return out / out.sum()


def psf_props(psf):
    n = psf.shape[0]
    h = (n - 1) / 2
    y, x = np.indices(psf.shape)
    r2 = ((x - h) ** 2 + (y - h) ** 2).ravel()
    core = (r2 < 9) & (psf.ravel() > 0)
    slope = np.polyfit(r2[core], np.log(psf.ravel()[core]), 1)[0]
    fwhm = 2 * 1.1774 * np.sqrt(-1 / (2 * slope))
    sel = r2.reshape(psf.shape) < 36
    w, dx, dy = psf[sel], (x - h)[sel], (y - h)[sel]
    qxx, qyy, qxy = [(w * v).sum() / w.sum() for v in (dx * dx, dy * dy, dx * dy)]
    e = np.hypot(qxx - qyy, 2 * qxy) / (qxx + qyy)
    ring = (np.sqrt(r2.reshape(psf.shape)) > 0.5 * fwhm) & (np.sqrt(r2.reshape(psf.shape)) < 1.5 * fwhm)
    phi = np.arctan2(y - h, x - h)[ring]
    I = psf[ring]
    a2 = np.hypot((I * np.cos(2 * phi)).sum(), (I * np.sin(2 * phi)).sum()) / I.sum()
    return fwhm, e, a2


def worker(items, cfg):
    inj = cc.load_module(os.path.join(cfg['scripts'], 'injection_test.py'), 'inj')
    measure = cc.get_measure(cfg['scripts'], cfg['am_func'])
    npix, aper, acf = cfg['npix'], cfg['aper'], cfg['acf']
    rows = []
    for g in items:
        eps_m = cfg['eps'][round(g['zbin'], 2)]
        cx_off, cy_off = cfg['sigma_cat'] * g['cdx'], cfg['sigma_cat'] * g['cdy']
        for scen, eps, seed in (('zero', (0., 0., 0.), g['seed_zero']),
                                ('matched', eps_m, g['seed_match'])):
            clean, x0, y0 = cc.build_clean(g, eps, npix, inj)
            truth = measure(clean, x0, y0, g['q'], g['pa'], g['re_pix'])
            for v, psf in cfg['psfs'].items():
                img = cc.observe(clean, psf, g, x0, y0, seed, aper, acf, inj)
                a1, a2, _ = measure(img, x0 + cx_off, y0 + cy_off, g['q'], g['pa'], g['re_pix'])
                rows.append(dict(variant=v, scenario=scen, zbin=g['zbin'],
                                 A1=a1, A2=a2, A1_true=truth[0], A2_true=truth[1]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    cc.add_common_args(ap)
    ap.add_argument('--stars_csv', help='$ROOT/psf/F444W_stars.csv from star_psf_astrometry.py')
    ap.add_argument('--stars_dir', help='$ROOT/psf/stars_f444w')
    ap.add_argument('--stars_pattern', default='s_{:06d}.fits')
    ap.add_argument('--n_boot', type=int, default=20)
    ap.add_argument('--field_min', type=int, default=15)
    ap.add_argument('--fwhm_scale', type=float, default=0.05)
    a = ap.parse_args()
    from astropy.io import fits
    os.makedirs(os.path.join(a.out, 'psf_variants'), exist_ok=True)
    inj = cc.load_module(os.path.join(a.scripts, 'injection_test.py'), 'inj')
    nom = np.asarray(fits.getdata(a.psf), float)
    ps = a.psf_pixscale or fits.getheader(a.psf).get('PIXSCALE', a.pixscale)
    nom = np.clip(nom, 0, None)
    nom /= nom.sum()
    var = {'nominal': nom,
           f'fwhm_m{int(100 * a.fwhm_scale)}': rescale(nom, 1 - a.fwhm_scale),
           f'fwhm_p{int(100 * a.fwhm_scale)}': rescale(nom, 1 + a.fwhm_scale),
           'azavg': azavg(nom)}
    if a.stars_csv:
        st, fld = rebuild_stamps(a.stars_csv, a.stars_dir, nom.shape[0], a.stars_pattern)
        if len(st) < 10:
            sys.exit(f'only {len(st)} star stamps rebuilt; check --stars_dir/--stars_pattern')
        re_ = stack(st)
        dev = np.abs(re_ - nom).max() / nom.max()
        print(f'{len(st)} star stamps rebuilt; restack vs nominal: max |diff| = {dev:.3%} of peak')
        if dev > 0.02:
            print('  WARNING: restacked PSF differs from the nominal one by >2% of the peak; '
                  'check that --stars_csv lists only the kept stars and matches the PSF file')
        var['restack'] = re_
        rng = np.random.default_rng(a.seed + 1)
        for k in range(1, a.n_boot + 1):
            var[f'boot_{k:02d}'] = stack(st[rng.integers(0, len(st), len(st))])
        for f in np.unique(fld):
            if (fld == f).sum() >= a.field_min:
                var[f'field_{f.split("-")[0]}'] = stack(st[fld == f])
    props = []
    for v, p in var.items():
        fits.PrimaryHDU(p.astype(np.float32), fits.Header({'PIXSCALE': ps})).writeto(
            os.path.join(a.out, 'psf_variants', f'{v}.fits'), overwrite=True)
        fw, e, a2 = psf_props(p)
        props.append(dict(variant=v, fwhm_pix=fw, fwhm_arcsec=fw * ps, ellipticity=e, A2_ring=a2))
    props = pd.DataFrame(props)
    props.to_csv(os.path.join(a.out, 'psf_variant_properties.csv'), index=False)
    print(props.round(4).to_string(index=False))

    d = cc.load_catalog(a, inj)
    items = cc.draw_sample(d, a.n_per_bin, a.seed)
    psfs = {v: inj.load_psf(os.path.join(a.out, 'psf_variants', f'{v}.fits'), ps, a.pixscale)
            for v in var}
    cfg = dict(scripts=os.path.abspath(a.scripts), am_func=a.am_func, psfs=psfs,
               acf=np.load(a.noise_acf) if a.noise_acf else None,
               npix=int(round(a.cutout_arcsec / a.pixscale)), aper=a.snr_aper_arcsec / a.pixscale,
               sigma_cat=a.sigma_cat, eps=cc.matched_eps(a.table1))
    print(f'\n{len(items)} injected models x {len(psfs)} PSF variants; {a.workers} workers')
    rows = pd.DataFrame(cc.run_parallel(worker, items, cfg, a.workers, chunk=10))
    rows.to_csv(os.path.join(a.out, 'psf_per_object.csv'), index=False)
    cal = cc.calibrate(rows, a.table1)
    cal.to_csv(os.path.join(a.out, 'psf_calibration.csv'), index=False)
    gr = cc.growth(cal)
    gr.to_csv(os.path.join(a.out, 'psf_growth.csv'), index=False)

    t1 = pd.read_csv(a.table1).set_index('z')
    boot = cal[cal.variant.str.startswith('boot_')]
    summ = []
    for zb, g in cal.groupby('zbin'):
        n = g.set_index('variant')
        r = {'zbin': zb}
        for m in (1, 2):
            k = f'A{m}_final_conv'
            r[f'A{m}_nominal'] = n.loc['nominal', k]
            if len(boot):
                r[f'A{m}_sigma_psfboot'] = boot[boot.zbin == zb][k].std()
            for v in n.index:
                if v.startswith(('fwhm', 'azavg', 'field')):
                    r[f'A{m}_{v}'] = n.loc[v, k]
            r[f'A{m}_sigma_tot_paper'] = t1.loc[zb, f'A{m}_final_err_conv']
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(a.out, 'psf_summary.csv'), index=False)

    print('\nCheck: nominal calibration reproduced here vs Table 1 (differences = sampling noise)')
    print(cc.compare_to_paper(cal, a.table1, 'nominal'))
    print('\nCorrected amplitudes per PSF variant (conventional) and PSF scatter vs sigma_tot:')
    print(summ.round(3).to_string(index=False))
    print('\nGrowth factors and mean corrected A1/A2 per variant:')
    gb = gr[gr.variant.str.startswith('boot_')]
    print(gr[~gr.variant.str.startswith('boot_')].round(3).to_string(index=False))
    if len(gb):
        print(f'bootstrap ({len(gb)}): A1 growth {gb.A1_growth.mean():.4f} +- {gb.A1_growth.std():.4f}, '
              f'A2 growth {gb.A2_growth.mean():.4f} +- {gb.A2_growth.std():.4f}, '
              f'R12(z<3.5) {gb.R12_lo.mean():.4f} +- {gb.R12_lo.std():.4f}, '
              f'R12(z>3.5) {gb.R12_hi.mean():.4f} +- {gb.R12_hi.std():.4f}')
        bs = cal[cal.variant.str.startswith('boot_')].groupby('zbin').agg(
            A1_sd=('A1_final_conv', 'std'), A2_sd=('A2_final_conv', 'std'),
            R12_sd=('R12_corr', 'std'), A1_rec_sd=('A1_rec', 'std'), A2_rec_sd=('A2_rec', 'std'))
        bs.to_csv(os.path.join(a.out, 'psf_bootstrap_scatter.csv'))
        print('bootstrap scatter per bin (full precision):\n' + bs.to_string(float_format='%.5f'))
    print(f'\nOutputs in {a.out}')


if __name__ == '__main__':
    main()
