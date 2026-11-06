#!/usr/bin/env python3
"""
measure_centering.py — empirical catalog-centroid error (sigma_cat) from the
cutouts, to feed --sigma_cat of injection_test.py.

For every cutout we find the "symmetric centre": the sub-pixel position where
the m=1 Fourier coefficient in the 0.5–1.5 R_e ring vanishes (Newton iteration
on the complex coefficient c1 = <I e^{i phi}> / <I>, circular rings). We then
report its offset from

  * the geometric cutout centre  (nx-1)/2, (ny-1)/2   -> 'geo'
  * the cutout centre with the other convention nx/2  -> printed mean shows it
  * the WCS pixel of the catalog RA/Dec (if --ra_col/--dec_col given) -> 'wcs'

Run it on the EARLY-TYPE cutouts (nearly symmetric light): the scatter of the
offsets is then an upper limit on sigma_cat, and a non-zero MEAN offset reveals
a centre-convention bug (e.g. +0.5 pix). On the disk cutouts the offset mixes
centring error with real lopsidedness, so use it only for comparison.

Usage:
    python3 measure_centering.py \
        --catalog /scratch/pulsating_disk/dja_analysis/full_ell_A1_results.csv \
        --cutouts /scratch/pulsating_disk/dja_analysis/cutouts_ell_full/ \
        --pattern e_{:06d}.fits --output /tmp/centering_ell.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

Z_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]


def c1_coeff(img, x0, y0, re_pix, oversample=5):
    """Complex m=1 coefficient <I e^{i phi}>/<I> in 0.5–1.5 R_e (circular)."""
    ny, nx = img.shape
    s = 1.0 / oversample
    xs = (np.arange(nx * oversample) + 0.5) * s - 0.5
    ys = (np.arange(ny * oversample) + 0.5) * s - 0.5
    X, Y = np.meshgrid(xs - x0, ys - y0)
    r = np.hypot(X, Y)
    sel = (r >= 0.5 * re_pix) & (r < 1.5 * re_pix)
    I = np.repeat(np.repeat(img, oversample, 0), oversample, 1)[sel]
    a0 = I.mean()
    if not np.isfinite(a0) or a0 <= 0:
        return np.nan
    return np.mean(I * (X[sel] + 1j * Y[sel]) / r[sel]) / a0


def symmetric_centre(img, x0, y0, re_pix, h=0.1, maxit=8, maxshift=3.0):
    """Newton iteration for c1(x, y) = 0 starting at (x0, y0)."""
    x, y = x0, y0
    for _ in range(maxit):
        c = c1_coeff(img, x, y, re_pix)
        cx = c1_coeff(img, x + h, y, re_pix)
        cy = c1_coeff(img, x, y + h, re_pix)
        if not np.all(np.isfinite([c, cx, cy])):
            return np.nan, np.nan, np.nan
        J = np.array([[(cx - c).real, (cy - c).real],
                      [(cx - c).imag, (cy - c).imag]]) / h
        try:
            step = -np.linalg.solve(J, [c.real, c.imag])
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        n = np.hypot(*step)
        if n > 1.0:
            step /= n
        x, y = x + step[0], y + step[1]
        if np.hypot(x - x0, y - y0) > maxshift:
            return np.nan, np.nan, np.nan
        if n < 0.005:
            break
    return x, y, abs(c1_coeff(img, x, y, re_pix))


def load_cutout(path):
    with fits.open(path) as h:
        hdu = next((x for x in h if x.data is not None and np.ndim(x.data) == 2), None)
        if hdu is None:
            return None, None
        return np.asarray(hdu.data, float), hdu.header


def robust_sigma(x):
    x = np.asarray(x)[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--cutouts', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--pattern', default='e_{:06d}.fits')
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--re_col', default='disk_r_pix', help='R_e in cutout pixels')
    ap.add_argument('--z_col', default='z')
    ap.add_argument('--ra_col', default='ra')
    ap.add_argument('--dec_col', default='dec')
    ap.add_argument('--max_n', type=int, default=3000, help='random subsample size')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    cat = pd.read_csv(args.catalog)
    for c in (args.idx_col, args.re_col, args.z_col):
        if c not in cat.columns:
            print("Catalog columns:", list(cat.columns))
            sys.exit(f"Column '{c}' not found; pass --idx_col/--re_col/--z_col.")
    if len(cat) > args.max_n:
        cat = cat.sample(args.max_n, random_state=args.seed)
    use_wcs = args.ra_col in cat.columns and args.dec_col in cat.columns
    if use_wcs:
        from astropy.wcs import WCS

    rows, missing = [], 0
    for i, g in enumerate(cat.itertuples(index=False)):
        idx = int(getattr(g, args.idx_col))
        re_pix = float(getattr(g, args.re_col))
        path = os.path.join(args.cutouts, args.pattern.format(idx))
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            img, hdr = load_cutout(path)
        except Exception as e:
            print(f"  {path}: {e}")
            continue
        if img is None or not np.isfinite(re_pix) or re_pix <= 0:
            continue
        ny, nx = img.shape
        gx, gy = (nx - 1) / 2, (ny - 1) / 2
        # background from the outer region, then crop for speed
        yy, xx = np.mgrid[:ny, :nx]
        rr = np.hypot(xx - gx, yy - gy)
        outer = img[(rr > max(3 * re_pix, 0.35 * min(nx, ny))) & np.isfinite(img)]
        img = img - (np.median(outer) if outer.size > 50 else 0.0)
        half = int(np.ceil(1.5 * re_pix + 5))
        x_lo, y_lo = max(int(gx) - half, 0), max(int(gy) - half, 0)
        crop = np.nan_to_num(img[y_lo:int(gy) + half + 2, x_lo:int(gx) + half + 2])

        sx, sy, a1_sym = symmetric_centre(crop, gx - x_lo, gy - y_lo, re_pix)
        a1_geo = abs(c1_coeff(crop, gx - x_lo, gy - y_lo, re_pix))
        row = dict(idx=idx, z=getattr(g, args.z_col), re_pix=re_pix, nx=nx, ny=ny,
                   dx_geo=sx + x_lo - gx, dy_geo=sy + y_lo - gy,
                   A1_geo=a1_geo, A1_sym=a1_sym)
        if use_wcs:
            try:
                wx, wy = WCS(hdr).celestial.all_world2pix(
                    getattr(g, args.ra_col), getattr(g, args.dec_col), 0)
                row.update(wcs_minus_geo_x=float(wx) - gx, wcs_minus_geo_y=float(wy) - gy,
                           dx_wcs=sx + x_lo - float(wx), dy_wcs=sy + y_lo - float(wy))
            except Exception:
                pass
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(cat)}")

    if not rows:
        sys.exit(f"Nothing measured (missing files: {missing}).")
    res = pd.DataFrame(rows)
    res.to_csv(args.output, index=False)
    ok = res.dropna(subset=['dx_geo', 'dy_geo'])
    print(f"\nMeasured {len(ok)} / {len(res)} (missing files {missing}, "
          f"Newton failures {len(res) - len(ok)})")

    def report(df, kx, ky, label):
        print(f"\n--- symmetric centre minus {label} [pix] ---")
        print(f"  mean  dx={df[kx].mean():+.3f}  dy={df[ky].mean():+.3f}   "
              f"(a +0.5 here means a centre-convention bug)")
        print(f"  median dx={df[kx].median():+.3f}  dy={df[ky].median():+.3f}")
        print(f"  robust sigma per axis: sx={robust_sigma(df[kx]):.3f}  "
              f"sy={robust_sigma(df[ky]):.3f}")
        print(f"  median |offset| = {np.median(np.hypot(df[kx], df[ky])):.3f}")

    report(ok, 'dx_geo', 'dy_geo', 'geometric centre (nx-1)/2')
    if 'dx_wcs' in ok:
        w = ok.dropna(subset=['dx_wcs'])
        report(w, 'dx_wcs', 'dy_wcs', 'WCS(RA,Dec)')
        print(f"  WCS(RA,Dec) minus (nx-1)/2: median dx={w.wcs_minus_geo_x.median():+.3f} "
              f"dy={w.wcs_minus_geo_y.median():+.3f}")

    ok = ok.copy()
    ok['zb'] = pd.cut(ok.z, Z_EDGES)
    g = ok.groupby('zb', observed=True)
    tab = pd.DataFrame({
        'N': g.size(),
        're_pix': g.re_pix.median(),
        'mean_dx': g.dx_geo.mean(), 'mean_dy': g.dy_geo.mean(),
        'sigma_x': g.dx_geo.apply(robust_sigma), 'sigma_y': g.dy_geo.apply(robust_sigma),
        'A1_geo': g.A1_geo.median(), 'A1_sym': g.A1_sym.median()})
    print("\nPer z-bin (offsets vs geometric centre):")
    print(tab.round(3).to_string())
    sig = np.nanmedian([robust_sigma(ok.dx_geo), robust_sigma(ok.dy_geo)])
    print(f"\n=> sigma_cat estimate (per axis) ~ {sig:.2f} pix "
          "(upper limit if run on early types: includes their intrinsic asymmetry)")
    print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
