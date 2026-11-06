#!/usr/bin/env python3
"""
star_psf_astrometry.py — empirical PSF and catalogue centring error from
point sources.

1. Star candidates are taken from the DJA morpho-phot catalogue: the most
   compact objects (smallest --radius_col) with S/N between --snr_min and
   --snr_max (bright enough, not saturated).
2. Their cutouts are downloaded (fetch_cutouts.fetch) in each filter.
3. Each star is centred with an iterative Gaussian-weighted centroid
   (unbiased for a point source). Its offset from the WCS position of the
   catalogue RA/Dec is the catalogue centring error for a perfectly
   symmetric object -> sigma_cat for injection_test.py.
4. Objects whose size or asymmetry is far from the median are rejected
   (galaxies, doubles); the rest are shifted to a common centre (Fourier
   shift), normalised and median-stacked -> empirical PSF FITS.

Usage:
    python3 star_psf_astrometry.py \
        --morpho /scratch/pulsating_disk/dja_analysis/full-good_morpho-phot.fits.gz \
        --filters F444W,F277W,F150W --outdir ../psf/ --n_cand 300
Run with --list_columns first to see which columns exist.
"""

import argparse
import os
import re
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from fetch_cutouts import fetch


def gauss_centroid(img, x0, y0, sigma, maxit=30):
    yy, xx = np.indices(img.shape)
    for _ in range(maxit):
        w = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2)) * np.clip(img, 0, None)
        if w.sum() <= 0:
            return np.nan, np.nan
        nx0, ny0 = (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()
        if np.hypot(nx0 - x0, ny0 - y0) < 1e-4:
            return nx0, ny0
        x0, y0 = nx0, ny0
    return x0, y0


def fourier_shift(img, dx, dy):
    ny, nx = img.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (kx * dx + ky * dy))))


def find_col(names, patterns):
    for p in patterns:
        for n in names:
            if re.fullmatch(p, n, flags=re.I):
                return n
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--morpho', required=True)
    ap.add_argument('--filters', default='F444W')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--list_columns', action='store_true')
    ap.add_argument('--radius_col', help='size column in HDU1 (default: auto, flux_radius)')
    ap.add_argument('--flux_col', help='flux column for S/N (default: auto, f444w_tot_*)')
    ap.add_argument('--err_col', help='error column for S/N (default: auto, e444w_tot_*)')
    ap.add_argument('--snr_min', type=float, default=100)
    ap.add_argument('--snr_max', type=float, default=3000)
    ap.add_argument('--n_cand', type=int, default=300)
    ap.add_argument('--size', type=float, default=3.0, help='cutout size [arcsec]')
    ap.add_argument('--stamp', type=int, default=41, help='PSF stamp size [pix], odd')
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()

    with fits.open(args.morpho) as h:
        t = h[1].data
        names = t.columns.names
        if args.list_columns:
            for key in ('radius', 'star', 'class', 'f444w', 'e444w', 'flag', 'mag'):
                print(key, [n for n in names if key in n.lower()][:25])
            return
        rcol = args.radius_col or find_col(names, [r'flux_radius', r'flux_radius_\d+', r'r50'])
        fcol = args.flux_col or find_col(names, [r'f444w_tot_\d', r'f444w_corr_\d', r'f444w_flux.*'])
        ecol = args.err_col or find_col(names, [r'e444w_tot_\d', r'e444w_corr_\d', r'e444w.*'])
        if not (rcol and fcol and ecol):
            sys.exit(f"Columns not found (radius={rcol}, flux={fcol}, err={ecol}); "
                     "run with --list_columns and pass --radius_col/--flux_col/--err_col.")
        rad = np.asarray(t[rcol], float)
        snr = np.asarray(t[fcol], float) / np.asarray(t[ecol], float)
        ra, dec = np.asarray(t['ra'], float), np.asarray(t['dec'], float)
        field = np.asarray(t['field']).astype(str) if 'field' in names else np.full(len(ra), 'all')
    print(f"Using radius={rcol}, S/N={fcol}/{ecol}")

    ok = np.isfinite(rad) & (rad > 0) & (snr > args.snr_min) & (snr < args.snr_max)
    order = np.argsort(np.where(ok, rad, np.inf))[:args.n_cand]
    print(f"{ok.sum()} objects pass S/N; taking the {len(order)} most compact "
          f"({rcol} {rad[order].min():.2f}..{rad[order].max():.2f}; "
          f"sample median {np.median(rad[ok]):.2f})")

    os.makedirs(args.outdir, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    for filt in [f.strip().upper() for f in args.filters.split(',')]:
        cdir = os.path.join(args.outdir, f'stars_{filt.lower()}')
        os.makedirs(cdir, exist_ok=True)
        paths = {i: os.path.join(cdir, f's_{i:06d}.fits') for i in order}
        with ThreadPoolExecutor(args.threads) as ex:
            list(ex.map(lambda i: fetch(ra[i], dec[i], args.size, filt, paths[i]), order))

        rows, stamps = [], []
        h = args.stamp // 2
        for i in order:
            p = paths[i]
            if not os.path.exists(p):
                continue
            with fits.open(p) as cut:
                hdu = cut[0] if cut[0].data is not None else cut[1]
                img, hdr = np.nan_to_num(np.asarray(hdu.data, float)), hdu.header
            ny, nx = img.shape
            wx, wy = [float(v) for v in WCS(hdr).celestial.all_world2pix(ra[i], dec[i], 0)]
            yy, xx = np.indices(img.shape)
            rr = np.hypot(xx - wx, yy - wy)
            sky = img[rr > 0.35 * min(nx, ny)]
            img = img - np.median(sky)
            cx, cy = gauss_centroid(img, wx, wy, 1.5)
            if not np.isfinite(cx) or np.hypot(cx - wx, cy - wy) > 3:
                continue
            ix, iy = int(round(cx)), int(round(cy))
            if not (h <= ix < nx - h and h <= iy < ny - h):
                continue
            st = img[iy - h:iy + h + 1, ix - h:ix + h + 1]
            st = fourier_shift(st, -(cx - ix), -(cy - iy))
            flux = st.sum()
            if flux <= 0:
                continue
            st = st / flux
            sy, sx = np.indices(st.shape)
            m2 = (st * ((sx - h) ** 2 + (sy - h) ** 2)).sum()
            asym = np.abs(st - st[::-1, ::-1]).sum() / 2
            rows.append((i, cx - wx, cy - wy, m2, asym, wx, wy, nx, ny))
            stamps.append(st)
        if len(rows) < 10:
            print(f"{filt}: only {len(rows)} usable stars, skipped")
            continue
        rows = np.array(rows)
        m2, asym = rows[:, 3], rows[:, 4]
        keep = (np.abs(m2 - np.median(m2)) < 3 * 1.4826 * np.median(np.abs(m2 - np.median(m2)))) & \
               (asym < np.median(asym) + 3 * 1.4826 * np.median(np.abs(asym - np.median(asym))))
        psf = np.median(np.array(stamps)[keep], axis=0)
        psf = np.clip(psf, 0, None)
        psf /= psf.sum()
        ps = np.sqrt(abs(np.linalg.det(WCS(hdr).celestial.pixel_scale_matrix))) * 3600
        out = os.path.join(args.outdir, f'{filt}_empirical_psf.fits')
        hdr_out = fits.Header()
        hdr_out['PIXSCALE'] = (ps, 'arcsec/pix')
        hdr_out['NSTARS'] = int(keep.sum())
        fits.PrimaryHDU(psf, hdr_out).writeto(out, overwrite=True)
        # FWHM from a Gaussian fit to the core (ln I linear in r^2, r < 3 pix)
        sy, sx = np.indices(psf.shape)
        r2 = ((sx - h) ** 2 + (sy - h) ** 2).ravel()
        core = (r2 < 9) & (psf.ravel() > 0)
        slope = np.polyfit(r2[core], np.log(psf.ravel()[core]), 1)[0]
        half = 1.1774 * np.sqrt(-1 / (2 * slope))   # HWHM [pix]
        dx, dy = rows[keep, 1], rows[keep, 2]
        rs = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
        import pandas as pd
        tab = pd.DataFrame(rows[keep], columns=['i', 'dx', 'dy', 'm2', 'asym', 'wx', 'wy',
                                                'nx', 'ny'])
        tab['i'] = tab.i.astype(int)
        tab['field'] = field[tab.i.values]
        tab['ra'], tab['dec'] = ra[tab.i.values], dec[tab.i.values]
        tab['fx'] = tab.wx - np.floor(tab.wx)        # sub-pixel phase of the WCS position
        tab['fy'] = tab.wy - np.floor(tab.wy)
        tab['wx_rel'] = tab.wx - tab.nx / 2
        tab['wy_rel'] = tab.wy - tab.ny / 2
        tab.to_csv(os.path.join(args.outdir, f'{filt}_stars.csv'), index=False)
        print(f"  per field: mean dx, dy (sigma) [pix]:")
        for f_, g_ in tab.groupby('field'):
            print(f"    {f_[:30]:30s} N={len(g_):3d}  dx {g_.dx.mean():+.2f} ({rs(g_.dx):.2f})"
                  f"  dy {g_.dy.mean():+.2f} ({rs(g_.dy):.2f})")
        print("  median dx vs sub-pixel phase of WCS x (0-.25-.5-.75-1): " + ' '.join(
            f"{tab[(tab.fx >= a) & (tab.fx < a + .25)].dx.median():+.2f}" for a in (0, .25, .5, .75)))
        print("  median dx vs WCS x - nx/2 (<-.25, -.25..0, 0...25, >.25): " + ' '.join(
            f"{tab[m_].dx.median():+.2f}" for m_ in (tab.wx_rel < -.25,
                                                      (tab.wx_rel >= -.25) & (tab.wx_rel < 0),
                                                      (tab.wx_rel >= 0) & (tab.wx_rel < .25),
                                                      tab.wx_rel >= .25)))
        # per-field centre offsets (star centroid - WCS); fields with <5 stars get 0
        offs = tab.groupby('field').agg(N=('dx', 'size'), dx=('dx', 'median'), dy=('dy', 'median'))
        offs.loc[offs.N < 5, ['dx', 'dy']] = 0.0
        offs.reset_index().to_csv(os.path.join(args.outdir, f'{filt}_field_offsets.csv'), index=False)
        rx = tab.dx - tab.field.map(offs.dx)
        ry = tab.dy - tab.field.map(offs.dy)
        print(f"  after per-field offset correction: robust sigma ({rs(rx):.3f}, {rs(ry):.3f}) "
              f"-> sigma_cat ~ {np.mean([rs(rx), rs(ry)]):.2f} pix  "
              f"[offsets: {os.path.join(args.outdir, filt + '_field_offsets.csv')}]")
        h_, e_ = np.histogram(tab.dx, bins=np.arange(-1, 1.01, 0.1))
        print("  dx histogram (-1..1, step 0.1): " + ' '.join(str(v) for v in h_))
        print(f"\n{filt}: {keep.sum()} stars kept of {len(rows)} usable "
              f"({len(order)} candidates)")
        print(f"  PSF: {out}  pixscale {ps:.4f}\"  FWHM ~ {2 * half * ps:.3f}\"")
        print(f"  star centroid - WCS(RA,Dec) [pix]: mean ({dx.mean():+.3f}, {dy.mean():+.3f}), "
              f"robust sigma ({rs(dx):.3f}, {rs(dy):.3f})")
        print(f"  -> sigma_cat ~ {np.mean([rs(dx), rs(dy)]):.2f} pix per axis "
              f"(use --sigma_cat in injection_test.py, --psf {out} --psf_pixscale {ps:.4f})")


if __name__ == '__main__':
    main()
