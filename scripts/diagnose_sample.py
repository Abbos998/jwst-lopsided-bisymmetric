#!/usr/bin/env python3
"""
diagnose_sample.py — two checks requested in review:

1. Sample accounting: why galaxies of the face-on sample have no valid V6
   measurement (missing/unreadable cutout, WCS position > 3 pix from the
   cutout centre, fewer than 50 unmasked pixels in the 0.5-1.5 r_d annulus,
   missing catalogue PA), and whether the excluded galaxies differ in z or r_d.

2. Definition of r_d (DISK_RADIUS_pix of the DJA bulge+disk table):
   * column units / comments from the FITS header;
   * pixel scale implied by DISK_RADIUS_deg / DISK_RADIUS_pix;
   * ratio to the single-Sersic RADIUS for nearly exponential galaxies
     (0.8 < n < 1.2): ~1 if DISK_RADIUS is a half-light radius,
     ~1/1.678 = 0.60 if it is an exponential scale length.

Usage:
    python3 diagnose_sample.py --objects ../corrected_disk_f444w.csv \
        --sample ~/Downloads/pasp_v3/data/dja_real.csv \
        --morpho /scratch/pulsating_disk/dja_analysis/full-good_morpho-phot.fits.gz
"""

import argparse

import numpy as np
import pandas as pd
from astropy.io import fits

Z_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]


def header_info(hdu, col):
    names = hdu.columns.names
    if col not in names:
        return 'not present'
    i = names.index(col) + 1
    h = hdu.header
    return (f"unit={h.get(f'TUNIT{i}', '-')!r}  comment={h.comments[f'TTYPE{i}']!r}  "
            f"desc={h.get(f'TCOMM{i}', h.get(f'TDESC{i}', '-'))!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--objects', required=True, help='run_corrected_am.py per-object CSV')
    ap.add_argument('--sample', required=True, help='dja_real.csv')
    ap.add_argument('--morpho', required=True)
    ap.add_argument('--variant', default='V6')
    args = ap.parse_args()
    v = args.variant

    # ---------------------------------------------------------------- 1
    smp = pd.read_csv(args.sample)
    face = smp[(smp.q > 0.7) & (smp.n_sersic < 2.5)].copy()
    obj = pd.read_csv(args.objects)
    d = face.merge(obj.drop(columns=[c for c in ('z', 'q') if c in obj]), on='idx', how='left',
                   indicator=True)

    def reason(r):
        if r['_merge'] == 'left_only':
            return 'cutout missing or unreadable'
        if np.isfinite(r[f'A1_{v}']) and np.isfinite(r[f'A2_{v}']):
            return 'valid'
        if not np.isfinite(r.get('A1_V1', np.nan)):
            return 'WCS position >3 pix from cutout centre'
        if v == 'V6' and np.isfinite(r.get('A1_V5', np.nan)) and not np.isfinite(r.get('cat_pa', 0)):
            return 'no catalogue position angle'
        if not np.isfinite(r.get('A1_V5', np.nan)):
            return '<50 unmasked pixels in 0.5-1.5 r_d annulus'
        return 'other (masked/negative flux)'

    d['reason'] = d.apply(reason, axis=1)
    print(f"Face-on sample: {len(face)}")
    print(d.reason.value_counts().to_string())
    bad = d[d.reason != 'valid']
    good = d[d.reason == 'valid']
    print(f"\nExcluded {len(bad)}: median z {bad.z.median():.2f} (valid {good.z.median():.2f}), "
          f"median disk_r_pix {bad.disk_r_pix.median():.2f} (valid {good.disk_r_pix.median():.2f})")
    zb = pd.cut(d.z, Z_EDGES)
    frac = d.assign(bad=d.reason != 'valid').groupby(zb, observed=True).bad.agg(['sum', 'size'])
    frac['percent'] = (100 * frac['sum'] / frac['size']).round(1)
    print("\nExcluded per redshift bin:")
    print(frac.rename(columns={'sum': 'excluded', 'size': 'total'}).to_string())

    # ---------------------------------------------------------------- 2
    with fits.open(args.morpho) as h:
        bd, se = h[3], h[2]
        print("\nColumn metadata:")
        for k, hdu, col in ((3, bd, 'DISK_RADIUS_pix'), (3, bd, 'DISK_RADIUS_deg'), (2, se, 'RADIUS'),
                            (2, se, 'SERSIC'), (2, se, 'ANGLE')):
            print(f"  HDU{k} {col:16s} {header_info(hdu, col)}")
        rpix = np.asarray(bd.data['DISK_RADIUS_pix'], float)
        rdeg = np.asarray(bd.data['DISK_RADIUS_deg'], float) if 'DISK_RADIUS_deg' in bd.columns.names else None
        rs = np.asarray(se.data['RADIUS'], float)
        ns = np.asarray(se.data['SERSIC'], float)
    idx = face.idx.values
    if rdeg is not None:
        ps = rdeg[idx] * 3600 / rpix[idx]
        ps = ps[np.isfinite(ps) & (ps > 0)]
        print(f"\nPixel scale implied by DISK_RADIUS_deg/DISK_RADIUS_pix: median {np.median(ps):.4f}\"/pix "
              f"(16-84%: {np.percentile(ps, 16):.4f}-{np.percentile(ps, 84):.4f})")
        print("  -> the cutouts are 0.05\"/pix; if this is 0.04, r_d in cutout pixels is "
              "1.25x too large (use --r_ref_scale 0.04 in run_corrected_am.py).")
        rd_arcsec = rdeg[idx] * 3600
    else:
        rd_arcsec = None
    sel = (ns[idx] > 0.8) & (ns[idx] < 1.2) & np.isfinite(rs[idx]) & (rs[idx] > 0)
    # the DJA single-Sersic RADIUS is in degrees (median ~3e-5 deg = 0.1"); compare like with like
    for unit, conv in (('degrees (DJA convention; compared with DISK_RADIUS_deg)',
                        rd_arcsec / 3600 if rd_arcsec is not None else None),
                       ('arcsec', rd_arcsec)):
        if conv is None:
            continue
        ratio = conv[sel] / rs[idx][sel]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        print(f"DISK_RADIUS / Sersic RADIUS for 0.8<n<1.2 ({sel.sum()} galaxies), assuming RADIUS in "
              f"{unit}: median {np.median(ratio):.3f}")
    print("  -> ~1.0: DISK_RADIUS behaves like a half-light radius; ~0.60: like an exponential "
          "scale length (R_e = 1.678 h); >1 is expected for a disk half-light radius when a "
          "compact bulge makes the total R_e smaller.")


if __name__ == '__main__':
    main()
