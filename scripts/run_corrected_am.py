#!/usr/bin/env python3
"""
run_corrected_am.py — re-run the A_m measurement of run_full_parallel.py on
the existing cutouts, adding one fix at a time so each fix's effect on
Table 1 can be quoted separately.

  V0  original measure_Am, verbatim (centre W/2,H/2; circular annulus
      max(2, 0.3 r_d) .. min(N/2-2, 2.5 r_d); only pixels > 0; no background
      subtraction; no deprojection; no masking). Must reproduce the catalog.
  V1  V0 + sub-pixel WCS centre of the catalog RA/Dec
  V2  V1 + background subtraction (clipped median), negative pixels kept
  V3  V2 + deprojection with catalog q and a PA from flux-weighted moments
  V4  V3 + neighbouring sources masked: separate 3-sigma segments, plus
      pixels > 3x (and > 5 sigma) brighter than their 180-deg counterpart
      (blended neighbours, as in CAS asymmetry); dilated by 1 pix
  V5  V4 with the annulus stated in the paper: 0.5–1.5 r_d
      (r_d multiplied by --r_scale, e.g. 0.8 if DISK_RADIUS_pix is in
      0.04"/pix units while the cutouts are 0.05"/pix)
  V6  V5 with the catalog Sersic PA instead of the moments PA (--pa_fits);
      the PA convention is chosen automatically by agreement with the
      moments PA (the moments PA can lock onto a bar and bias A2 low)

All variants keep the original normalisation A_m = |sum I e^{im phi}| / sum I
(half the conventional 2|<I e^{im phi}>|/<I>); multiply by 2 for literature
comparisons. A1/A2 is unaffected.

Usage (disks):
    python3 run_corrected_am.py --catalog /tmp/dja_real.csv \
        --cutouts /scratch/pulsating_disk/dja_analysis/cutouts_full/ \
        --pattern g_{:06d}.fits --output ../corrected_disk.csv
Early types:
    python3 run_corrected_am.py \
        --catalog /scratch/pulsating_disk/dja_analysis/full_ell_A1_results.csv \
        --cutouts /scratch/pulsating_disk/dja_analysis/cutouts_ell_full/ \
        --pattern e_{:06d}.fits --q_col axratio --n_col sersic_n --no_cuts \
        --output ../corrected_ell.csv
"""

import argparse
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

Z_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
Z_CEN = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50]
VARIANTS = ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# candidate conventions for the catalog PA (theta in deg) -> our PA
PA_CONVENTIONS = {'theta': lambda t: t, '-theta': lambda t: -t,
                  '90+theta': lambda t: 90 + t, '90-theta': lambda t: 90 - t}
LABELS = {
    'V0': 'original (W/2 centre)',
    'V1': '+ WCS sub-pixel centre',
    'V2': '+ background, keep <0 pixels',
    'V3': '+ deprojection (q, PA)',
    'V4': '+ neighbour mask',
    'V5': '+ 0.5-1.5 r_d annulus',
    'V6': '+ catalog (Sersic) PA',
}


def measure_Am(img, disk_r):
    """Verbatim copy of run_full_parallel.py."""
    H, W = img.shape; cx, cy = W/2, H/2
    y, x = np.indices(img.shape)
    r = np.sqrt((x-cx)**2 + (y-cy)**2); phi = np.arctan2(y-cy, x-cx)
    r_min = max(2.0, 0.3*disk_r)
    r_max = min(min(H,W)/2 - 2, 2.5*disk_r)
    mask = (r >= r_min) & (r <= r_max) & np.isfinite(img) & (img > 0)
    if mask.sum() < 50: return None, None, None, 0
    I = img[mask]; p = phi[mask]
    A0 = np.sum(I)
    if A0 <= 0: return None, None, None, 0
    def Am(m):
        c = np.sum(I*np.cos(m*p)); s = np.sum(I*np.sin(m*p))
        return np.sqrt(c*c+s*s)/A0
    return Am(1), Am(2), Am(3), int(mask.sum())


def am_general(img, cx, cy, r_min, r_max, q=1.0, pa_deg=0.0, positive_only=False,
               bad=None):
    """Same estimator as measure_Am, with free centre, deprojection and mask."""
    y, x = np.indices(img.shape)
    dx, dy = x - cx, y - cy
    t = np.deg2rad(pa_deg)
    xp = dx * np.cos(t) + dy * np.sin(t)
    yp = (-dx * np.sin(t) + dy * np.cos(t)) / max(q, 0.1)
    r, phi = np.hypot(xp, yp), np.arctan2(yp, xp)
    sel = (r >= r_min) & (r <= r_max) & np.isfinite(img)
    if positive_only:
        sel &= img > 0
    if bad is not None:
        sel &= ~bad
    if sel.sum() < 50:
        return np.nan, np.nan, np.nan
    I, p = img[sel], phi[sel]
    A0 = I.sum()
    if A0 <= 0:
        return np.nan, np.nan, np.nan
    return tuple(np.hypot((I * np.cos(m * p)).sum(), (I * np.sin(m * p)).sum()) / A0
                 for m in (1, 2, 3))


def background(img, cx, cy, r_excl):
    ny, nx = img.shape
    y, x = np.indices(img.shape)
    v = img[(np.hypot(x - cx, y - cy) > max(r_excl, 0.35 * min(nx, ny))) & np.isfinite(img)]
    if v.size < 50:
        return 0.0, np.nan
    for _ in range(10):
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        keep = np.abs(v - med) < 3 * mad
        if keep.all() or keep.sum() < 30:
            break
        v = v[keep]
    return float(np.median(v)), float(1.4826 * np.median(np.abs(v - np.median(v))))


def neighbour_mask(img, sigma, cx, cy, nsig=3.0):
    """Boolean mask of >nsig*sigma connected components not containing the
    centre pixel (4-connectivity), dilated by one pixel."""
    det = np.isfinite(img) & (img > nsig * sigma)
    ny, nx = det.shape
    lab = np.zeros(det.shape, int)
    cur = 0
    for sy, sx in zip(*np.nonzero(det)):
        if lab[sy, sx]:
            continue
        cur += 1
        lab[sy, sx] = cur
        dq = deque([(sy, sx)])
        while dq:
            yy, xx = dq.popleft()
            for ny_, nx_ in ((yy + 1, xx), (yy - 1, xx), (yy, xx + 1), (yy, xx - 1)):
                if 0 <= ny_ < ny and 0 <= nx_ < nx and det[ny_, nx_] and not lab[ny_, nx_]:
                    lab[ny_, nx_] = cur
                    dq.append((ny_, nx_))
    ic, jc = int(round(cy)), int(round(cx))
    own = lab[ic, jc] if 0 <= ic < ny and 0 <= jc < nx else 0
    bad = (lab > 0) & (lab != own)
    grown = bad.copy()
    grown[1:, :] |= bad[:-1, :]; grown[:-1, :] |= bad[1:, :]
    grown[:, 1:] |= bad[:, :-1]; grown[:, :-1] |= bad[:, 1:]
    return grown


def rot180(img, cx, cy):
    """Image rotated by 180 deg about (cx, cy), bilinear, NaN outside."""
    ny, nx = img.shape
    y, x = np.indices(img.shape)
    xs, ys = 2 * cx - x, 2 * cy - y
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    out = np.full(img.shape, np.nan)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < nx - 1) & (y0 < ny - 1)
    a, b = x0[ok], y0[ok]
    out[ok] = (img[b, a] * (1 - fx[ok]) * (1 - fy[ok]) + img[b, a + 1] * fx[ok] * (1 - fy[ok])
               + img[b + 1, a] * (1 - fx[ok]) * fy[ok] + img[b + 1, a + 1] * fx[ok] * fy[ok])
    return out


def symmetric_mask(img, sigma, cx, cy, factor=3.0, nsig=5.0):
    """Pixels much brighter than their 180-deg counterpart (> factor x and
    > nsig sigma above it): neighbours/contaminants blended with the galaxy.
    A genuine m=1 mode with A1 ~ 0.3 (conventional) gives contrasts ~1.5-2,
    so it is not removed. Dilated by one pixel."""
    rot = rot180(img, cx, cy)
    bad = np.isfinite(rot) & (img > factor * np.clip(rot, 0, None)) & \
        (img - rot > nsig * sigma)
    grown = bad.copy()
    grown[1:, :] |= bad[:-1, :]; grown[:-1, :] |= bad[1:, :]
    grown[:, 1:] |= bad[:, :-1]; grown[:, :-1] |= bad[:, 1:]
    return grown


def moments_pa(img, cx, cy, rmax, bad=None):
    y, x = np.indices(img.shape)
    sel = (np.hypot(x - cx, y - cy) <= rmax) & np.isfinite(img) & (img > 0)
    if bad is not None:
        sel &= ~bad
    w = img[sel]
    if w.sum() <= 0:
        return 0.0
    dx, dy = x[sel] - cx, y[sel] - cy
    mxx, myy, mxy = (w * dx * dx).sum(), (w * dy * dy).sum(), (w * dx * dy).sum()
    return float(np.degrees(0.5 * np.arctan2(2 * mxy, mxx - myy)))


def process(img, hdr, ra, dec, disk_r, q, r_scale, cat_pa=np.nan, r_ref_scale=None,
            centre_offset=(0.0, 0.0)):
    out = {}
    a1, a2, a3, _ = measure_Am(img, disk_r)
    out['V0'] = (a1, a2, a3) if a1 is not None else (np.nan,) * 3

    H, W = img.shape
    w = WCS(hdr).celestial
    if r_ref_scale:
        # disk_r is in pixels of r_ref_scale arcsec; convert to this cutout's pixels
        ps = np.sqrt(abs(np.linalg.det(w.pixel_scale_matrix))) * 3600
        disk_r = disk_r * r_ref_scale / ps
    wx, wy = w.all_world2pix(ra, dec, 0)
    # per-field astrometric offset (star centroid - WCS), see star_psf_astrometry.py
    wx, wy = float(wx) + centre_offset[0], float(wy) + centre_offset[1]
    if not (np.isfinite(wx) and np.isfinite(wy)) or np.hypot(wx - W / 2, wy - H / 2) > 3:
        return out, np.nan, np.nan, np.nan
    r_min = max(2.0, 0.3 * disk_r)
    r_max = min(min(H, W) / 2 - 2, 2.5 * disk_r)

    out['V1'] = am_general(img, wx, wy, r_min, r_max, positive_only=True)
    bg, sig = background(img, wx, wy, 3 * r_max)
    im = img - bg
    out['V2'] = am_general(im, wx, wy, r_min, r_max)
    pa = moments_pa(im, wx, wy, r_max)
    out['V3'] = am_general(im, wx, wy, r_min, r_max, q, pa)
    bad = None
    if np.isfinite(sig) and sig > 0:
        bad = neighbour_mask(im, sig, wx, wy) | symmetric_mask(im, sig, wx, wy)
    pa_m = moments_pa(im, wx, wy, r_max, bad)
    out['V4'] = am_general(im, wx, wy, r_min, r_max, q, pa_m, bad=bad)
    rd = disk_r * r_scale
    out['V5'] = am_general(im, wx, wy, max(1.0, 0.5 * rd), 1.5 * rd, q, pa_m, bad=bad)
    if np.isfinite(cat_pa):
        for name, f in PA_CONVENTIONS.items():
            out[f'V6[{name}]'] = am_general(im, wx, wy, max(1.0, 0.5 * rd), 1.5 * rd,
                                            q, f(cat_pa), bad=bad)
    frac_masked = np.nan
    if bad is not None:
        yy, xx = np.indices(img.shape)
        frac_masked = float(bad[np.hypot(xx - wx, yy - wy) <= r_max].mean())
    return out, pa_m, frac_masked, wx - W / 2


def boot(x, rng, nb=1000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
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
    ap.add_argument('--r_col', default='disk_r_pix')
    ap.add_argument('--q_col', default='q')
    ap.add_argument('--n_col', default='n_sersic')
    ap.add_argument('--ra_col', default='ra')
    ap.add_argument('--dec_col', default='dec')
    ap.add_argument('--r_scale', type=float, default=1.0,
                    help='factor applied to disk_r in V5 only (0.8 converts 0.04"->0.05" pix)')
    ap.add_argument('--no_cuts', action='store_true', help='skip q>0.7, n<2.5')
    ap.add_argument('--max_n', type=int, default=0)
    ap.add_argument('--r_ref_scale', type=float,
                    help='arcsec/pix in which disk_r is effectively expressed; converts it to '
                         "each cutout's pixel scale (V1-V6). Use 0.05 for every filter so "
                         'F444W is unchanged and other filters use the same physical ring.')
    ap.add_argument('--centre_offsets', help='<FILTER>_field_offsets.csv from '
                    'star_psf_astrometry.py: per-field correction of the WCS centre')
    ap.add_argument('--field_col', default='field')
    ap.add_argument('--pa_fits', help='DJA morpho-phot FITS whose rows are indexed by idx '
                    '(e.g. full-good_morpho-phot.fits.gz); enables V6')
    ap.add_argument('--pa_hdu', type=int, default=2)
    ap.add_argument('--pa_col', default='ANGLE')
    args = ap.parse_args()

    offsets = {}
    if args.centre_offsets:
        o = pd.read_csv(args.centre_offsets)
        offsets = {str(r.field): (float(r.dx), float(r.dy)) for r in o.itertuples()}
        print("Centre offsets [pix]: " + ', '.join(f"{k}: ({v[0]:+.2f},{v[1]:+.2f})"
                                                   for k, v in offsets.items()))
    pa_arr = None
    if args.pa_fits:
        with fits.open(args.pa_fits) as h:
            pa_arr = np.asarray(h[args.pa_hdu].data[args.pa_col], float)
        if np.nanmax(np.abs(pa_arr)) <= 2 * np.pi + 0.01:
            pa_arr = np.degrees(pa_arr)
            print(f"{args.pa_col} looks like radians -> converted to degrees")
        print(f"Catalog PA: {args.pa_col} from HDU {args.pa_hdu}, "
              f"range {np.nanmin(pa_arr):.1f}..{np.nanmax(pa_arr):.1f} deg")

    cat = pd.read_csv(args.catalog)
    need = [args.idx_col, args.z_col, args.r_col, args.q_col, args.ra_col, args.dec_col]
    if any(c not in cat.columns for c in need):
        print("Catalog columns:", list(cat.columns))
        sys.exit(f"Missing one of {need}; pass the matching --*_col options.")
    if not args.no_cuts:
        cat = cat[(cat[args.q_col] > 0.7) & (cat[args.n_col] < 2.5)]
    if args.max_n and len(cat) > args.max_n:
        cat = cat.sample(args.max_n, random_state=1)
    print(f"Processing {len(cat)} galaxies ...")

    rows, bad_files = [], 0
    for i, g in enumerate(cat.itertuples(index=False)):
        idx = int(getattr(g, args.idx_col))
        path = os.path.join(args.cutouts, args.pattern.format(idx))
        try:
            with fits.open(path) as cut:
                hdu = cut[0] if cut[0].data is not None else cut[1]
                img, hdr = np.asarray(hdu.data, float), hdu.header
            cat_pa = (float(pa_arr[idx]) if pa_arr is not None and idx < len(pa_arr)
                      else np.nan)
            res, pa, fmask, wdx = process(img, hdr, getattr(g, args.ra_col),
                                          getattr(g, args.dec_col),
                                          float(getattr(g, args.r_col)),
                                          float(getattr(g, args.q_col)), args.r_scale,
                                          cat_pa, args.r_ref_scale,
                                          offsets.get(str(getattr(g, args.field_col, '')),
                                                      (0.0, 0.0)))
        except Exception:
            bad_files += 1
            continue
        row = dict(idx=idx, z=getattr(g, args.z_col), disk_r=getattr(g, args.r_col),
                   q=getattr(g, args.q_col), pa=pa, frac_masked=fmask, wcs_dx=wdx,
                   A1_cat=getattr(g, 'A1', np.nan), A2_cat=getattr(g, 'A2', np.nan),
                   A3_cat=getattr(g, 'A3', np.nan), cat_pa=cat_pa)
        for v in VARIANTS + [f'V6[{c}]' for c in PA_CONVENTIONS]:
            a = res.get(v, (np.nan,) * 3)
            row.update({f'A1_{v}': a[0], f'A2_{v}': a[1], f'A3_{v}': a[2]})
        rows.append(row)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(cat)}")

    res = pd.DataFrame(rows)
    if pa_arr is not None:
        # pick the PA convention that best matches the moments PA, judged on
        # the more elongated objects where the moments PA is well defined
        el = res[(res.q < 0.85) & res.cat_pa.notna() & res.pa.notna()]
        el = el if len(el) > 50 else res[res.cat_pa.notna() & res.pa.notna()]
        print("\nCatalog-PA convention vs moments PA (median |dPA|, deg):")
        best, best_d = None, np.inf
        for name, f in PA_CONVENTIONS.items():
            d = np.median(np.abs((f(el.cat_pa) - el.pa + 90) % 180 - 90))
            print(f"  {name:9s}: {d:5.1f}")
            if d < best_d:
                best, best_d = name, d
        print(f"  -> using '{best}' (random PAs would give ~45 deg)")
        for k in ('A1', 'A2', 'A3'):
            res[f'{k}_V6'] = res[f'{k}_V6[{best}]']
    for v in VARIANTS:
        res[f'R12_{v}'] = res[f'A1_{v}'] / res[f'A2_{v}']
    res.to_csv(args.output, index=False)
    print(f"\nProcessed {len(res)} (unreadable/failed: {bad_files})")

    ok = res.dropna(subset=['A1_V0', 'A1_cat'])
    if len(ok):
        print(f"Reproduction check V0 vs catalog: median |A1_V0 - A1_cat| = "
              f"{np.median(np.abs(ok.A1_V0 - ok.A1_cat)):.5f}, "
              f"exact (<1e-4): {(np.abs(ok.A1_V0 - ok.A1_cat) < 1e-4).mean() * 100:.1f}%")
    print(f"Median fraction of annulus masked (V4): {res.frac_masked.median():.3f}")

    rng = np.random.default_rng(0)
    res['zb'] = pd.cut(res.z, Z_EDGES, labels=Z_CEN)
    summ = []
    for zc in Z_CEN:
        s = res[res.zb == zc]
        if s.empty:
            continue
        row = dict(z=zc, N=len(s), disk_r=s.disk_r.median())
        for v in VARIANTS:
            for k in ('A1', 'A2', 'A3', 'R12'):
                row[f'{k}_{v}'], row[f'{k}_{v}_err'] = boot(s[f'{k}_{v}'], rng)
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(args.output.replace('.csv', '_summary.csv'), index=False)

    for k in ('A1', 'A2', 'R12'):
        name = {'A1': 'median A1', 'A2': 'median A2', 'R12': 'median(A1/A2)'}[k]
        print(f"\n=== {name} (original normalisation) ===")
        print(f"{'variant':32s}" + ''.join(f"{z:>7.2f}" for z in summ.z))
        for v in VARIANTS:
            print(f"{v + ' ' + LABELS[v]:32s}" + ''.join(f"{x:7.3f}" for x in summ[f'{k}_{v}']))
    print("\nA1 rise (z>4 bins vs z=0.75):")
    for v in VARIANTS:
        a = summ[f'A1_{v}'].values
        print(f"  {v}: x{np.nanmean(a[-2:]) / a[0]:.2f}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, ax in zip(('A1', 'A2', 'R12'), axs):
        for v in VARIANTS:
            ax.errorbar(summ.z + (VARIANTS.index(v) - 2.5) * 0.03, summ[f'{k}_{v}'],
                        summ[f'{k}_{v}_err'], fmt='o-', ms=4, capsize=2,
                        label=f'{v}: {LABELS[v]}')
        ax.set(xlabel='Redshift $z$',
               ylabel={'A1': 'median $A_1$', 'A2': 'median $A_2$',
                       'R12': 'median $(A_1/A_2)$'}[k])
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    plt.tight_layout()
    png = args.output.replace('.csv', '.png')
    plt.savefig(png, dpi=180)
    print(f"\nSaved {args.output}, {args.output.replace('.csv', '_summary.csv')}, {png}")


if __name__ == '__main__':
    main()
