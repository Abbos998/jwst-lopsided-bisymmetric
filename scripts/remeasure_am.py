#!/usr/bin/env python3
"""
remeasure_am.py — re-measure A1, A2, A3 on the real cutouts with two centres:

  grid : (nx//2, ny//2), what the current pipeline uses. Cutouts are
         pixel-aligned, so the source sits at a random sub-pixel position and
         this centre is off by U(-0.5, 0.5) pix per axis.
  wcs  : sub-pixel WCS position of the catalog RA/Dec (all_world2pix).

Built-in estimator of injection_test.py (0.5–1.5 R_e, sub-pixel sampling,
deprojection with catalog q), or the pipeline's own via --am_func
am_wrapper.py:measure_am. The catalog has no PA, so PA is taken from the
flux-weighted second moments inside 2 R_e around the WCS centre.

If the A1 (and A1/A2) rise with redshift survives with the 'wcs' centre, it is
not a centring artefact. Also printed: A1 x R_e(pix), which is constant across
z if A1 is dominated by a fixed-pixel centring error (A1 ~ k delta / R_e).

Usage (disks):
    python3 remeasure_am.py --catalog /tmp/dja_real.csv \
        --cutouts /scratch/pulsating_disk/dja_analysis/cutouts_full/ \
        --pattern g_{:06d}.fits --output ../remeasure_disk.csv
Early types:
    python3 remeasure_am.py \
        --catalog /scratch/pulsating_disk/dja_analysis/full_ell_A1_results.csv \
        --cutouts /scratch/pulsating_disk/dja_analysis/cutouts_ell_full/ \
        --pattern e_{:06d}.fits --q_col axratio --n_col sersic_n --no_cuts \
        --output ../remeasure_ell.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from injection_test import (Z_CEN, Z_EDGES, OBS_A1, OBS_A2, OBS_A3, OBS_R12,
                            measure_am_builtin, load_user_am)


def moments_pa(img, x0, y0, rmax):
    """Major-axis angle [deg, from +x, counter-clockwise] from second moments."""
    ny, nx = img.shape
    y, x = np.mgrid[:ny, :nx]
    sel = (np.hypot(x - x0, y - y0) <= rmax) & (img > 0)
    w = img[sel]
    if w.sum() <= 0:
        return 0.0
    dx, dy = x[sel] - x0, y[sel] - y0
    mxx, myy, mxy = (w * dx * dx).sum(), (w * dy * dy).sum(), (w * dx * dy).sum()
    return float(np.degrees(0.5 * np.arctan2(2 * mxy, mxx - myy)))


def load(path):
    with fits.open(path) as h:
        hdu = next((x for x in h if x.data is not None and np.ndim(x.data) == 2), None)
        if hdu is None:
            return None, None
        return np.asarray(hdu.data, float), hdu.header


def boot_med(x, rng, nb=1000):
    x = np.asarray(x)[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan
    return np.median(x), np.std(np.median(rng.choice(x, (nb, len(x))), axis=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--cutouts', required=True)
    ap.add_argument('--pattern', default='g_{:06d}.fits')
    ap.add_argument('--output', required=True)
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--z_col', default='z')
    ap.add_argument('--re_col', default='disk_r_pix')
    ap.add_argument('--q_col', default='q')
    ap.add_argument('--n_col', default='n_sersic')
    ap.add_argument('--ra_col', default='ra')
    ap.add_argument('--dec_col', default='dec')
    ap.add_argument('--no_cuts', action='store_true', help='skip q>0.7, n<2.5')
    ap.add_argument('--max_n', type=int, default=0, help='random subsample (0 = all)')
    ap.add_argument('--oversample', type=int, default=5)
    ap.add_argument('--am_func', help="estimator to use instead of the built-in one, "
                    "e.g. am_wrapper.py:measure_am (the pipeline's extract_am)")
    args = ap.parse_args()

    cat = pd.read_csv(args.catalog)
    need = [args.idx_col, args.z_col, args.re_col, args.q_col, args.ra_col, args.dec_col]
    if any(c not in cat.columns for c in need):
        print("Catalog columns:", list(cat.columns))
        sys.exit(f"Missing one of {need}; pass the matching --*_col options.")
    if not args.no_cuts:
        cat = cat[(cat[args.q_col] > 0.7) & (cat[args.n_col] < 2.5)]
    if args.max_n and len(cat) > args.max_n:
        cat = cat.sample(args.max_n, random_state=1)
    if args.am_func:
        measure = load_user_am(args.am_func)
    else:
        def measure(img, x0, y0, q, pa, re):
            return measure_am_builtin(img, x0, y0, q, pa, re, oversample=args.oversample)
    print(f"Re-measuring {len(cat)} galaxies with "
          f"{args.am_func or 'built-in estimator'} ...")

    rows, bad = [], 0
    for i, g in enumerate(cat.itertuples(index=False)):
        idx = int(getattr(g, args.idx_col))
        re_pix = float(getattr(g, args.re_col))
        q = float(getattr(g, args.q_col))
        path = os.path.join(args.cutouts, args.pattern.format(idx))
        try:
            img, hdr = load(path)
            wx, wy = WCS(hdr).celestial.all_world2pix(
                getattr(g, args.ra_col), getattr(g, args.dec_col), 0)
        except Exception:
            bad += 1
            continue
        if img is None or not (np.isfinite(re_pix) and re_pix > 0 and np.isfinite(q)):
            bad += 1
            continue
        ny, nx = img.shape
        wx, wy = float(wx), float(wy)
        gx, gy = nx // 2, ny // 2
        if np.hypot(wx - gx, wy - gy) > 3:   # WCS far from the cutout centre: skip
            bad += 1
            continue
        # background (clipped median of the outer region), then crop for speed
        yy, xx = np.mgrid[:ny, :nx]
        rr = np.hypot(xx - gx, yy - gy)
        outer = img[(rr > max(3 * re_pix, 0.35 * min(nx, ny))) & np.isfinite(img)]
        if outer.size > 50:
            med = np.median(outer)
            mad = 1.4826 * np.median(np.abs(outer - med))
            img = img - np.median(outer[np.abs(outer - med) < 3 * mad])
        half = int(np.ceil(2 * re_pix + 6))
        x0c, y0c = max(gx - half, 0), max(gy - half, 0)
        crop = np.nan_to_num(img[y0c:gy + half + 1, x0c:gx + half + 1])
        pa = moments_pa(crop, wx - x0c, wy - y0c, 2 * re_pix)

        row = dict(idx=idx, z=getattr(g, args.z_col), re_pix=re_pix, q=q, pa=pa,
                   dx_grid=gx - wx, dy_grid=gy - wy,
                   A1_cat=getattr(g, 'A1', np.nan), A2_cat=getattr(g, 'A2', np.nan))
        for name, (cx, cy) in (('grid', (gx, gy)), ('wcs', (wx, wy))):
            a1, a2, a3 = measure(crop, cx - x0c, cy - y0c, q, pa, re_pix)
            row.update({f'A1_{name}': a1, f'A2_{name}': a2, f'A3_{name}': a3})
        rows.append(row)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(cat)}")

    res = pd.DataFrame(rows)
    for c in ('grid', 'wcs'):
        res[f'R12_{c}'] = res[f'A1_{c}'] / res[f'A2_{c}']
    # A2 here uses a moments PA (catalog has none), which can lock onto a bar
    # and under-estimate A2. A2 is insensitive to centring, so the robust
    # ratio uses the pipeline's own catalog A2: A1_wcs / A2_cat.
    res['R12_mix'] = res.A1_wcs / res.A2_cat
    res.to_csv(args.output, index=False)
    print(f"\nMeasured {len(res)}  (skipped {bad})")
    print(f"|grid - wcs| offset: median {np.median(np.hypot(res.dx_grid, res.dy_grid)):.2f} pix")

    rng = np.random.default_rng(0)
    res['zb'] = pd.cut(res.z, Z_EDGES, labels=Z_CEN)
    print("\nPer z-bin medians (bootstrap errors). 'cat' = A1 already in the catalog.")
    hdr_ = (f"{'z':>5} {'N':>5} {'Re':>5} | {'A1_cat':>6} {'A1_grid':>12} {'A1_wcs':>12} | "
            f"{'A2_cat':>6} {'A3_wcs':>6} | {'R12_cat':>7} {'A1w/A2cat':>12} | "
            f"{'A1g*Re':>6} {'A1w*Re':>6}")
    print(hdr_)
    out = []
    for zc in Z_CEN:
        s = res[res.zb == zc]
        if s.empty:
            continue
        a1g, a1ge = boot_med(s.A1_grid, rng)
        a1w, a1we = boot_med(s.A1_wcs, rng)
        rg, _ = boot_med(s.R12_grid, rng)
        rw, rwe = boot_med(s.R12_mix, rng)
        rc = (s.A1_cat / s.A2_cat).median()
        re_m = s.re_pix.median()
        print(f"{zc:5.2f} {len(s):5d} {re_m:5.2f} | {s.A1_cat.median():6.3f} "
              f"{a1g:6.3f}±{a1ge:.3f} {a1w:6.3f}±{a1we:.3f} | {s.A2_cat.median():6.3f} "
              f"{s.A3_wcs.median():6.3f} | {rc:7.2f} {rw:6.2f}±{rwe:.2f} | "
              f"{a1g * re_m:6.2f} {a1w * re_m:6.2f}")
        out.append(dict(z=zc, N=len(s), re_pix=re_m, A1_grid=a1g, A1_grid_err=a1ge,
                        A1_wcs=a1w, A1_wcs_err=a1we, A2_cat=s.A2_cat.median(),
                        A3_wcs=s.A3_wcs.median(), R12_grid=rg, R12_cat=rc,
                        R12_mix=rw, R12_mix_err=rwe))
    summ = pd.DataFrame(out)
    summ.to_csv(args.output.replace('.csv', '_summary.csv'), index=False)

    # figure: grid vs wcs vs paper Table 1
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    axs[0].plot(Z_CEN, OBS_A1, 'ks-', label='Paper Table 1')
    axs[0].errorbar(summ.z, summ.A1_grid, summ.A1_grid_err, fmt='o--', capsize=3,
                    label='Re-measured, centre nx//2')
    axs[0].errorbar(summ.z, summ.A1_wcs, summ.A1_wcs_err, fmt='D-', capsize=3,
                    label='Re-measured, WCS sub-pixel centre')
    axs[0].set(ylabel='median $A_1$', title='(a) $A_1$')
    axs[1].plot(Z_CEN, OBS_R12, 'ks-', label='Paper Table 1')
    axs[1].plot(summ.z, summ.R12_cat, 'o--', label='catalog A1/A2 (this sample)')
    axs[1].errorbar(summ.z, summ.R12_mix, summ.R12_mix_err, fmt='D-', capsize=3,
                    label='$A_1^{\\rm WCS}/A_2^{\\rm cat}$')
    axs[1].set(ylabel='median $(A_1/A_2)$', title='(b) $A_1/A_2$')
    axs[2].plot(summ.z, summ.A1_grid * summ.re_pix, 'o--', label='nx//2')
    axs[2].plot(summ.z, summ.A1_wcs * summ.re_pix, 'D-', label='WCS')
    axs[2].plot(Z_CEN, OBS_A1 * np.interp(Z_CEN, summ.z, summ.re_pix), 'ks-',
                label='Paper Table 1')
    axs[2].set(ylabel='$A_1 \\times R_e$ [pix]',
               title='(c) constant if A1 is a fixed-pixel centring error')
    for a in axs:
        a.set_xlabel('Redshift $z$')
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    png = args.output.replace('.csv', '.png')
    plt.savefig(png, dpi=180)
    print(f"\nSaved {args.output}, {args.output.replace('.csv', '_summary.csv')}, {png}")


if __name__ == '__main__':
    main()
