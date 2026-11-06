#!/usr/bin/env python3
"""
fetch_cutouts.py — download grizli-cutout FITS thumbnails in other filters
for the same galaxies (same URL and size rule as run_full_parallel.py), so
the A_m pipeline can be run at other rest-frame wavelengths.

Each file is checked after download; cutouts without data (no coverage in
that filter: all NaN/zero) are deleted and listed in <outdir>/missing.txt.

Usage:
    python3 fetch_cutouts.py --catalog /tmp/dja_real.csv \
        --filters F150W,F277W --outroot /scratch/pulsating_disk/dja_analysis/ \
        --prefix g --threads 8
    -> cutouts_f150w/g_000123.fits, cutouts_f277w/g_000123.fits, ...
"""

import argparse
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy.io import fits

URL = ("https://grizli-cutout.herokuapp.com/thumb?ra={ra}&dec={dec}&size={size}"
       "&filters={filt}-CLEAR&output=fits")


def cutout_size(disk_r):
    """Same rule as run_full_parallel.py [arcsec]."""
    return max(3.0, min(10.0, disk_r * 0.04 * 10))


def valid(path):
    try:
        with fits.open(path) as h:
            hdu = h[0] if h[0].data is not None else h[1]
            d = np.asarray(hdu.data, float)
    except Exception:
        return False
    good = np.isfinite(d) & (d != 0)
    return d.ndim == 2 and good.mean() > 0.5 and np.nanstd(d[good]) > 0


def fetch(ra, dec, size, filt, path, retries=3):
    if os.path.exists(path) and valid(path):
        return 'cached'
    for k in range(retries):
        try:
            urllib.request.urlretrieve(URL.format(ra=ra, dec=dec, size=size, filt=filt), path)
            if valid(path):
                return 'ok'
            os.remove(path)
            return 'empty'
        except Exception:
            time.sleep(2 ** k)
    if os.path.exists(path):
        os.remove(path)
    return 'failed'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--filters', required=True, help='comma list, e.g. F150W,F277W')
    ap.add_argument('--outroot', required=True)
    ap.add_argument('--prefix', default='g', help="file prefix ('g' disks, 'e' early types)")
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--r_col', default='disk_r_pix')
    ap.add_argument('--size', type=float, help='fixed size [arcsec] instead of the pipeline rule')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--max_n', type=int, default=0)
    args = ap.parse_args()

    cat = pd.read_csv(args.catalog)
    if args.max_n:
        cat = cat.sample(min(args.max_n, len(cat)), random_state=1)
    for filt in [f.strip().upper() for f in args.filters.split(',')]:
        outdir = os.path.join(args.outroot, f'cutouts_{filt.lower()}')
        os.makedirs(outdir, exist_ok=True)
        jobs, stats, missing = {}, {}, []
        with ThreadPoolExecutor(args.threads) as ex:
            for g in cat.itertuples(index=False):
                idx = int(getattr(g, args.idx_col))
                size = args.size or cutout_size(float(getattr(g, args.r_col)))
                path = os.path.join(outdir, f'{args.prefix}_{idx:06d}.fits')
                jobs[ex.submit(fetch, g.ra, g.dec, size, filt, path)] = idx
            for i, fut in enumerate(as_completed(jobs)):
                st = fut.result()
                stats[st] = stats.get(st, 0) + 1
                if st in ('empty', 'failed'):
                    missing.append(jobs[fut])
                if (i + 1) % 500 == 0:
                    print(f"  {filt}: {i + 1}/{len(jobs)} {stats}")
        np.savetxt(os.path.join(outdir, 'missing.txt'), np.array(missing, int), fmt='%d')
        print(f"{filt}: {stats} -> {outdir}")


if __name__ == '__main__':
    main()
