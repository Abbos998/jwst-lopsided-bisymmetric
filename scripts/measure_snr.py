#!/usr/bin/env python3
"""
measure_snr.py — per-galaxy S/N from the F444W cutouts, defined exactly as in
injection_test.py, so the injection test can be run at each galaxy's real S/N.

    S/N_ap   = F_ap / (sigma_pix * sqrt(N_ap))      aperture radius --aper_arcsec
    S/N_ring = same, inside the 0.5–1.5 R_e ring used for A_m

sigma_pix is the robust (MAD) rms of background pixels: r > max(3 R_e, 0.35 L)
from the centre, after iterative 3-sigma clipping to drop neighbours.
NOTE: drizzled mosaics have correlated noise; sigma_pix is the per-pixel rms
(what the white-noise injection reproduces), not the true aperture error.

Usage:
    python3 measure_snr.py \
        --catalog /tmp/dja_real.csv \
        --cutouts /scratch/pulsating_disk/dja_analysis/cutouts_full/ \
        --output /tmp/dja_real_snr.csv

Then:
    python3 injection_test.py --catalog /tmp/dja_real_snr.csv \
        --re_col disk_r_pix --re_unit pix --n_col n_sersic --snr_col snr_ap \
        --pixscale <printed> --cutout_arcsec <printed> --output ../injection_out_real/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

Z_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]


def pixel_scale(header):
    """arcsec/pix from CD or CDELT keywords."""
    if 'CD1_1' in header:
        cd = np.array([[header.get('CD1_1', 0), header.get('CD1_2', 0)],
                       [header.get('CD2_1', 0), header.get('CD2_2', 0)]])
        return float(np.sqrt(abs(np.linalg.det(cd))) * 3600)
    if 'CDELT1' in header:
        pc = np.array([[header.get('PC1_1', 1), header.get('PC1_2', 0)],
                       [header.get('PC2_1', 0), header.get('PC2_2', 1)]])
        return float(np.sqrt(abs(header['CDELT1'] * header['CDELT2'] * np.linalg.det(pc))) * 3600)
    return np.nan


def clipped_stats(x, nsig=3.0, maxit=10):
    x = x[np.isfinite(x)]
    for _ in range(maxit):
        med = np.median(x)
        mad = 1.4826 * np.median(np.abs(x - med))
        keep = np.abs(x - med) < nsig * mad
        if keep.all() or keep.sum() < 20:
            break
        x = x[keep]
    return np.median(x), 1.4826 * np.median(np.abs(x - np.median(x))), len(x)


def measure_one(path, re_pix, aper_arcsec, default_pixscale):
    with fits.open(path) as h:
        hdu = next((x for x in h if x.data is not None and np.ndim(x.data) == 2), None)
        if hdu is None:
            return None
        img = np.asarray(hdu.data, float)
        ps = pixel_scale(hdu.header)
    if not np.isfinite(ps) or ps <= 0:
        ps = default_pixscale
    ny, nx = img.shape
    y, x = np.mgrid[:ny, :nx]
    x0, y0 = (nx - 1) / 2, (ny - 1) / 2
    r = np.hypot(x - x0, y - y0)

    bg_sel = r > max(3 * re_pix, 0.35 * min(nx, ny))
    if bg_sel.sum() < 50:
        bg_sel = r > 0.4 * min(nx, ny)
    bg, sig, nbg = clipped_stats(img[bg_sel])
    if not np.isfinite(sig) or sig <= 0:
        return None
    im = img - bg

    r_ap = aper_arcsec / ps
    ap = (r <= r_ap) & np.isfinite(im)
    ring = (r >= 0.5 * re_pix) & (r < 1.5 * re_pix) & np.isfinite(im)
    f_ap, f_ring = im[ap].sum(), im[ring].sum()
    return dict(pixscale=ps, npix_side=nx, sigma_pix=sig, n_bg=nbg,
                aper_pix=r_ap, flux_ap=f_ap,
                snr_ap=f_ap / (sig * np.sqrt(ap.sum())),
                snr_ring=f_ring / (sig * np.sqrt(max(ring.sum(), 1))),
                peak_snr=np.nanmax(im[r <= max(2.0, 0.5 * re_pix)]) / sig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--cutouts', required=True)
    ap.add_argument('--output', required=True, help='catalog + S/N columns (CSV)')
    ap.add_argument('--pattern', default='g_{:06d}.fits', help='cutout file name from idx')
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--re_col', default='disk_r_pix')
    ap.add_argument('--aper_arcsec', type=float, default=0.25,
                    help='aperture RADIUS; must match --snr_aper_arcsec of injection_test.py')
    ap.add_argument('--default_pixscale', type=float, default=0.04,
                    help='used only when the header has no CD/CDELT')
    ap.add_argument('--all', action='store_true', help='measure all rows (default: q>0.7, n<2.5)')
    args = ap.parse_args()

    cat = pd.read_csv(args.catalog)
    sel = cat if args.all else cat[(cat.q > 0.7) & (cat.n_sersic < 2.5)]
    print(f"Measuring {len(sel)} cutouts ...")

    rows, missing = [], 0
    for i, g in enumerate(sel.itertuples(index=False)):
        idx = int(getattr(g, args.idx_col))
        path = os.path.join(args.cutouts, args.pattern.format(idx))
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            m = measure_one(path, float(getattr(g, args.re_col)), args.aper_arcsec,
                            args.default_pixscale)
        except Exception as e:  # corrupt file etc.
            print(f"  {path}: {e}")
            m = None
        if m:
            m[args.idx_col] = idx
            rows.append(m)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(sel)}")
    if not rows:
        sys.exit(f"No cutouts measured (missing: {missing}). Check --cutouts / --pattern.")

    snr = pd.DataFrame(rows)
    out = cat.merge(snr, on=args.idx_col, how='left', validate='one_to_one')
    out.to_csv(args.output, index=False)

    print(f"\nMeasured: {len(snr)}   missing files: {missing}")
    ps = snr.pixscale.round(4).value_counts()
    print("Pixel scale [arcsec/pix]:", dict(ps))
    sz = snr.npix_side.value_counts()
    print("Cutout side [pix]:", dict(sz))
    print("Cutout side [arcsec]:", sorted(set((snr.npix_side * snr.pixscale).round(2))))
    print(f"Aperture radius = {args.aper_arcsec}\" = {snr.aper_pix.median():.2f} pix")

    m = out[out.snr_ap.notna()].copy()
    m['zb'] = pd.cut(m.z, Z_EDGES)
    print("\nPer-bin medians:")
    print(m.groupby('zb', observed=True)[['disk_r_pix', 'snr_ap', 'snr_ring', 'peak_snr']]
          .median().round(1).to_string())
    print(f"\nS/N_ap < 10: {(m.snr_ap < 10).sum()} of {len(m)}")

    ps_med = snr.pixscale.median()
    side = (snr.npix_side * snr.pixscale).median()
    print(f"\nSaved {args.output}\nNext:\n  python3 injection_test.py --catalog {args.output} "
          f"--re_col disk_r_pix --re_unit pix --n_col n_sersic --snr_col snr_ap "
          f"--pixscale {ps_med:.4f} --cutout_arcsec {side:.2f} "
          f"--n_per_bin 300 --output ../injection_out_real/")


if __name__ == '__main__':
    main()
