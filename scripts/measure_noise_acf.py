#!/usr/bin/env python3
"""
measure_noise_acf.py — pixel-noise autocorrelation function (ACF) of the
cutouts, for correlated-noise injection (injection_test.py --noise_acf).

Drizzled/resampled mosaics have noise correlated between neighbouring
pixels, so white noise with the same per-pixel rms underestimates the noise
in ring sums (and hence the A_m floor). For each cutout we mask the galaxy
(r < 3 r_max) and all >3-sigma sources (dilated by 3 pix), and compute the
masked ACF with FFTs:
    ACF(tau) = IFFT|FFT(m x)|^2 / IFFT|FFT(m)|^2,
normalised to ACF(0) = 1 and truncated to (2h+1)^2. The ACFs of all cutouts
are averaged (weighted by the number of sky pixel pairs).

Usage:
    python3 measure_noise_acf.py --catalog /tmp/dja_real.csv \
        --cutouts /scratch/pulsating_disk/dja_analysis/cutouts_full/ \
        --pattern g_{:06d}.fits --output ../noise_acf_f444w.npy
"""

import argparse
import os

import numpy as np
import pandas as pd
from astropy.io import fits


def clipped(v):
    v = v[np.isfinite(v)]
    for _ in range(10):
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        keep = np.abs(v - med) < 3 * mad
        if keep.all() or keep.sum() < 30:
            break
        v = v[keep]
    return np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))


def dilate(m, k):
    out = m.copy()
    for _ in range(k):
        g = out.copy()
        g[1:, :] |= out[:-1, :]; g[:-1, :] |= out[1:, :]
        g[:, 1:] |= out[:, :-1]; g[:, :-1] |= out[:, 1:]
        out = g
    return out


def masked_acf(img, mask_sky, h):
    """ACF over sky pixels (mask_sky True = use). Returns (acf, npairs)."""
    x = np.where(mask_sky, img, 0.0)
    m = mask_sky.astype(float)
    ny, nx = img.shape
    s = (2 * ny, 2 * nx)                     # zero-pad: no wrap-around
    fx, fm = np.fft.rfft2(x, s), np.fft.rfft2(m, s)
    num = np.fft.irfft2(fx * np.conj(fx), s)
    den = np.fft.irfft2(fm * np.conj(fm), s)
    num = np.roll(np.roll(num, h, 0), h, 1)[:2 * h + 1, :2 * h + 1]
    den = np.roll(np.roll(den, h, 0), h, 1)[:2 * h + 1, :2 * h + 1]
    with np.errstate(all='ignore'):
        acf = num / den
    return acf / acf[h, h], den


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--cutouts', required=True)
    ap.add_argument('--pattern', default='g_{:06d}.fits')
    ap.add_argument('--output', required=True, help='.npy file for the ACF')
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--r_col', default='disk_r_pix')
    ap.add_argument('--field_col', default='field')
    ap.add_argument('--n', type=int, default=600, help='number of cutouts to use')
    ap.add_argument('--h', type=int, default=4, help='ACF half-size [pix]')
    args = ap.parse_args()

    cat = pd.read_csv(args.catalog)
    cat = cat.sample(min(args.n, len(cat)), random_state=3)
    h = args.h
    tot, wsum = np.zeros((2 * h + 1, 2 * h + 1)), np.zeros((2 * h + 1, 2 * h + 1))
    per_field = {}
    used = 0
    for g in cat.itertuples(index=False):
        path = os.path.join(args.cutouts, args.pattern.format(int(getattr(g, args.idx_col))))
        try:
            with fits.open(path) as cut:
                hdu = cut[0] if cut[0].data is not None else cut[1]
                img = np.asarray(hdu.data, float)
        except Exception:
            continue
        ny, nx = img.shape
        yy, xx = np.indices(img.shape)
        r_max = 2.5 * float(getattr(g, args.r_col))
        rr = np.hypot(xx - nx / 2, yy - ny / 2)
        bg, sig = clipped(img[(rr > 3 * r_max) & np.isfinite(img)])
        if not np.isfinite(sig) or sig <= 0:
            continue
        im = img - bg
        sources = dilate(np.isfinite(im) & (im > 3 * sig), 3)
        sky = np.isfinite(im) & ~sources & (rr > 3 * r_max)
        if sky.sum() < 500:
            continue
        acf, den = masked_acf(np.nan_to_num(im), sky, h)
        if not np.all(np.isfinite(acf)):
            continue
        tot += acf * den
        wsum += den
        used += 1
        f = getattr(g, args.field_col, 'all')
        per_field.setdefault(f, []).append(acf)

    if not used:
        raise SystemExit("No usable cutouts.")
    acf = tot / wsum
    np.save(args.output, acf)
    print(f"ACF from {used} cutouts, saved to {args.output}")
    print("Central 5x5 of the ACF (ACF(0)=1):")
    print(np.array2string(acf[h - 2:h + 3, h - 2:h + 3], precision=3, suppress_small=True))
    print(f"Noise variance in an aperture is ~{acf.sum():.2f}x the white-noise value "
          f"(sum of ACF over {2 * h + 1}x{2 * h + 1}).")
    print("\nACF(1,0) and ACF(0,1) per field:")
    for f, lst in sorted(per_field.items()):
        a = np.median(lst, axis=0)
        print(f"  {str(f):28s} N={len(lst):4d}  x:{a[h, h + 1]:.3f}  y:{a[h + 1, h]:.3f}")


if __name__ == '__main__':
    main()
