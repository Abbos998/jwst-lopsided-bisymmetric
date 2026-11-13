#!/usr/bin/env python3
"""
real_sky_injection.py -- the decisive test of Sect. 3.4: is the low-S/N excess a
noise floor that the synthetic-noise injections underestimate?

It repeats the injection--recovery calibration of injection_test.py but, instead
of adding synthetic (white or ACF) noise, pastes each PSF-convolved model onto a
REAL blank-sky patch cut from the observed cutouts of the same field. Real sky
carries faint undetected sources, background structure and the true pixel
correlation, which smooth Sersic + synthetic noise cannot. Everything else --
the models, the PSF, the centring, the amplitude estimator (am_wrapper.measure_v6,
which does no background subtraction) and the S/N definition -- is identical to
injection_test.py (imported from it), so the only difference is the noise source.

The stored cutouts are sized to each galaxy (3--10 arcsec), so a galaxy fills a
roughly fixed fraction of its own cutout and a large fixed blank window rarely
exists. The measurement only reads the 0.5--1.5 r_d annulus, so for each injected
galaxy we cut a blank window just large enough for its annulus (and the PSF
wings) from the field's cutouts, masking the sources first. The window is
background-subtracted, so adding the model leaves the real sky noise, faint
sources and residual background in place.

Outputs, per redshift bin and per S/N tercile at z>4:
  - real-sky noise floor vs the synthetic floor (Table 1 / Table 3),
  - real-sky recovered fraction vs the synthetic one,
  - corrected amplitudes and growth factors obtained when the real-sky
    calibration is applied to the observed medians (A_meas of Table 1),
    compared with the synthetic calibration.
If the real-sky floor is higher at low S/N and the real-sky growth factor drops
toward the deep-field value, the excess is an underestimated noise floor.

Usage (run from the package scripts/ directory):
  python3 real_sky_injection.py \
      --catalog $DATA/dja_snr_f444w.csv --cutouts $D/cutouts_full/ --pattern g_{:06d}.fits \
      --table1 ../data/table1_disks_F444W.csv --sample $DATA/dja_real.csv \
      --psf ../psf/F444W_empirical_psf.fits --psf_pixscale 0.04 --pixscale 0.05 \
      --re_col disk_r_pix --re_unit pix --re_ref_scale 0.04 --n_col n_sersic --snr_col snr_ap \
      --am_func am_wrapper.py:measure_v6 --n_per_bin 300 --out $ROOT/referee_tests/realsky
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import injection_test as inj

HI = (4.5, 5.5)
FIELD = ['field', 'FIELD', 'file', 'root', 'survey']


def clipped(v, nit=5, k=3.0):
    v = v[np.isfinite(v)]
    for _ in range(nit):
        m, s = np.median(v), np.std(v)
        v2 = v[np.abs(v - m) < k * s]
        if len(v2) < 10 or len(v2) == len(v):
            v = v2
            break
        v = v2
    return (np.median(v), np.std(v)) if len(v) else (0.0, 0.0)


def dilate(mask, k=2):
    out = mask.copy()
    for _ in range(k):
        out[1:] |= out[:-1].copy(); out[:-1] |= out[1:].copy()
        out[:, 1:] |= out[:, :-1].copy(); out[:, :-1] |= out[:, 1:].copy()
    return out


def prep_cutout(path, nsig=2.5):
    """Return (bg-subtracted image, integral image of the sky mask, sig) or None."""
    try:
        with fits.open(path, memmap=False) as h:
            img = next(hd.data for hd in h if hd.data is not None and np.ndim(hd.data) == 2).astype(float)
    except Exception:
        return None
    bg, sig = clipped(img.ravel())
    if not np.isfinite(sig) or sig <= 0:
        return None
    src = dilate((np.nan_to_num(img) - bg > nsig * sig) | ~np.isfinite(img), 2)
    sky = (~src).astype(np.float64)
    ii = np.zeros((sky.shape[0] + 1, sky.shape[1] + 1))
    ii[1:, 1:] = np.cumsum(np.cumsum(sky, 0), 1)
    return img - bg, ii, sig


def find_blank(cuts, need, rng, thr=0.97, tries=300):
    """A need x need background-subtracted real-sky patch (sky fraction >= thr)."""
    if not cuts:
        return None
    target = thr * need * need
    for _ in range(tries):
        im, ii, _ = cuts[rng.integers(len(cuts))]
        ny, nx = im.shape
        if ny < need or nx < need:
            continue
        y = int(rng.integers(0, ny - need + 1))
        x = int(rng.integers(0, nx - need + 1))
        tot = ii[y + need, x + need] - ii[y, x + need] - ii[y + need, x] + ii[y, x]
        if tot >= target:
            return im[y:y + need, x:x + need].copy()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--cutouts', required=True)
    ap.add_argument('--pattern', default='g_{:06d}.fits')
    ap.add_argument('--table1', required=True)
    ap.add_argument('--sample', required=True)
    ap.add_argument('--psf')
    ap.add_argument('--psf_pixscale', type=float)
    ap.add_argument('--psf_fwhm', type=float, default=0.145)
    ap.add_argument('--pixscale', type=float, default=0.05)
    ap.add_argument('--snr_aper_arcsec', type=float, default=0.25)
    ap.add_argument('--re_unit', choices=['arcsec', 'pix'], default='pix')
    ap.add_argument('--re_ref_scale', type=float, default=0.04)
    ap.add_argument('--ring_scale', type=float, default=1.0)
    ap.add_argument('--sigma_cat', type=float, default=0.10)
    ap.add_argument('--centering', default='catalog')
    ap.add_argument('--am_func', default='am_wrapper.py:measure_v6')
    ap.add_argument('--matched_col', default='A{m}_meas_pipe',
                    help='per-bin injected amplitudes; default = observed A_meas of Table 1')
    ap.add_argument('--n_per_bin', type=int, default=300)
    ap.add_argument('--max_cutouts_per_field', type=int, default=500)
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--field_col')
    ap.add_argument('--snr_col')
    ap.add_argument('--re_col')
    ap.add_argument('--n_col')
    ap.add_argument('--q_col')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.set_option('display.width', 250)

    measure = inj.load_user_am(a.am_func)
    psf = inj.load_psf(a.psf, a.psf_pixscale, a.pixscale) if a.psf else inj.gaussian_psf(a.psf_fwhm / a.pixscale)
    psf_pad = psf.shape[0]
    aper = a.snr_aper_arcsec / a.pixscale

    # ---- targets
    df = pd.read_csv(a.catalog)
    cols = inj.resolve_columns(df, {k: getattr(a, f'{k}_col', None) for k in inj.COL_CANDIDATES})
    for need in ('re', 'n', 'q', 'snr'):
        if need not in cols:
            sys.exit(f"missing column for '{need}'; have {list(df.columns)}; pass --{need}_col")
    d = pd.DataFrame({k: pd.to_numeric(df[cols[k]], errors='coerce') for k in cols})
    d['idx'] = df[a.idx_col].values
    if 'z' not in d:
        sys.exit("no redshift column in the catalogue")
    d['re_pix'] = d['re'] * (a.re_ref_scale / a.pixscale) if a.re_unit == 'pix' else d['re'] / a.pixscale
    if 'pa' not in d:
        d['pa'] = rng.uniform(0, 180, len(d))
    cat = pd.read_csv(a.sample).rename(columns={a.idx_col: 'idx'})
    fcol = a.field_col or next((c for c in FIELD if c in cat.columns), None)
    if fcol is None:
        sys.exit(f"no field column in {a.sample}")
    d = d.merge(cat[['idx', fcol]].rename(columns={fcol: 'field'}), on='idx', how='left')
    d['grp'] = d.field.map(lambda f: 'PRIMER-UDS' if 'uds' in str(f).lower()
                           else 'PRIMER-COSMOS' if 'cosmos' in str(f).lower() else 'deep')
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=['z', 're_pix', 'n', 'q', 'snr', 'field'])
    d = d[(d.q > 0.7) & (d.n > 0.3) & (d.n < 2.5) & (d.snr > 10) &
          (d.z >= inj.Z_EDGES[0]) & (d.z < inj.Z_EDGES[-1]) & (d.re_pix > 0)]
    print(f"{len(d)} target galaxies; fields {sorted(d.field.unique())}")

    # ---- injected amplitudes per bin
    t = pd.read_csv(a.table1).set_index('z')
    inj_amp = {}
    for zc in inj.Z_CEN:
        amp = []
        for m in (1, 2, 3):
            col = a.matched_col.format(m=m)
            amp.append(float(t.loc[zc, col]) if (zc in t.index and col in t.columns) else 0.0)
        inj_amp[round(zc, 2)] = tuple(amp)

    # ---- prepare real cutouts per field (bg-subtracted + sky integral image)
    cutcache = {}
    for fld in sorted(d.field.unique()):
        ids = d[d.field == fld].idx.astype(int).tolist()
        rng.shuffle(ids)
        prepared = []
        for i in ids:
            p = os.path.join(a.cutouts, a.pattern.format(i))
            if not os.path.exists(p):
                continue
            c = prep_cutout(p)
            if c is not None:
                prepared.append(c)
            if len(prepared) >= a.max_cutouts_per_field:
                break
        cutcache[fld] = prepared
        print(f"  field {fld}: prepared {len(prepared)} real cutouts for sky")
    allcuts = [c for v in cutcache.values() for c in v]
    if not allcuts:
        sys.exit("No usable cutouts: check --cutouts/--pattern.")

    def need_pix(re_pix):
        return int(round(2 * 1.6 * re_pix + psf_pad)) + 6   # annulus to 1.5 r_d + PSF wings + pad

    def a_meas_aper(img, cx, cy):
        y, x = np.mgrid[:img.shape[0], :img.shape[1]]
        apmask = (x - cx) ** 2 + (y - cy) ** 2 <= aper ** 2
        return img[apmask].sum(), int(apmask.sum())

    # ---- injection loop
    rows = []
    skipped = 0
    for ib, zc in enumerate(inj.Z_CEN):
        lo, hi = inj.Z_EDGES[ib], inj.Z_EDGES[ib + 1]
        sub = d[(d.z >= lo) & (d.z < hi)]
        if sub.empty:
            continue
        draw = sub.sample(n=a.n_per_bin, replace=len(sub) < a.n_per_bin,
                          random_state=int(rng.integers(1 << 31)))
        amp = inj_amp[round(zc, 2)]
        done = 0
        for g in draw.itertuples():
            npix_g = need_pix(g.re_pix)
            patch = find_blank(cutcache.get(g.field, []), npix_g, rng)
            if patch is None:
                patch = find_blank(allcuts, npix_g, rng)
            if patch is None:
                skipped += 1
                continue
            npix_g = patch.shape[0]
            n = float(np.clip(g.n, 0.3, 2.5))
            x0 = (npix_g - 1) / 2 + rng.uniform(-0.5, 0.5)
            y0 = (npix_g - 1) / 2 + rng.uniform(-0.5, 0.5)
            phases = rng.uniform(0, 2 * np.pi, 3)
            _, sbg = clipped(patch.ravel())
            if sbg <= 0:
                continue
            for scen, eps in (('zero', (0.0, 0.0, 0.0)), ('matched', amp)):
                clean = inj.make_model(npix_g, x0, y0, g.re_pix, n, g.q, g.pa, eps, phases)
                t1, t2, t3 = measure(clean, x0, y0, g.q, g.pa, g.re_pix * a.ring_scale)
                obs = inj.fftconvolve(clean, psf, mode='same')
                f_model, nap = a_meas_aper(obs, x0, y0)
                if f_model <= 0:
                    continue
                scale = g.snr * sbg * np.sqrt(nap) / f_model
                img = scale * obs + patch
                cx = x0 + (rng.normal(0, a.sigma_cat) if a.centering == 'catalog' else 0.0)
                cy = y0 + (rng.normal(0, a.sigma_cat) if a.centering == 'catalog' else 0.0)
                a1, a2, a3 = measure(img, cx, cy, g.q, g.pa, g.re_pix * a.ring_scale)
                rows.append(dict(zbin=zc, snr=g.snr, grp=g.grp, scenario=scen,
                                 A1_true=t1, A2_true=t2, A1=a1, A2=a2))
            done += 1
        print(f"  z={zc:.2f}: {done}/{len(draw)} injected onto real sky")
    if skipped:
        print(f"  ({skipped} galaxies skipped: no blank sky patch large enough in their field)")

    r = pd.DataFrame(rows)
    if r.empty:
        sys.exit("No successful injections.")
    r.to_csv(f'{a.out}_per_object.csv', index=False)

    # ---- real-sky floor and recovered fraction per bin (conventional = 2x pipeline)
    def med(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return np.median(x) if len(x) else np.nan
    L = [f"{len(d)} targets; real cutouts for sky: {len(allcuts)}; PSF {'file' if a.psf else 'Gaussian'}; "
         f"injected amplitudes = {a.matched_col}; {skipped} galaxies skipped."]
    out = []
    for zc in inj.Z_CEN:
        z0 = r[(np.isclose(r.zbin, zc)) & (r.scenario == 'zero')]
        mt = r[(np.isclose(r.zbin, zc)) & (r.scenario == 'matched')]
        row = {'z': zc, 'N': len(z0)}
        for k in (1, 2):
            fl_pipe = med(z0[f'A{k}'])
            mm = med(mt[f'A{k}']); tt = med(mt[f'A{k}_true'])
            frec = np.sqrt(max(mm ** 2 - fl_pipe ** 2, 0)) / tt if tt else np.nan
            row[f'floor{k}_real'] = 2 * fl_pipe
            row[f'floor{k}_syn'] = 2 * t.loc[zc, f'A{k}_floor_pipe'] if zc in t.index else np.nan
            row[f'frec{k}_real'] = frec
            row[f'frec{k}_syn'] = t.loc[zc, f'A{k}_recovery'] if zc in t.index else np.nan
            ameas = t.loc[zc, f'A{k}_meas_pipe'] if zc in t.index else np.nan
            row[f'A{k}_syn'] = t.loc[zc, f'A{k}_final_conv'] if zc in t.index else np.nan
            row[f'A{k}_real'] = (2 * np.sqrt(max(ameas ** 2 - fl_pipe ** 2, 0)) / frec
                                 if frec and frec > 0 and np.isfinite(ameas) else np.nan)
        out.append(row)
    O = pd.DataFrame(out)
    O.to_csv(f'{a.out}_bins.csv', index=False)
    L += ['', '=== real-sky vs synthetic noise floor and recovered fraction (conventional)',
          O[['z', 'N', 'floor1_syn', 'floor1_real', 'frec1_syn', 'frec1_real',
             'floor2_syn', 'floor2_real', 'frec2_syn', 'frec2_real']].round(3).to_string(index=False)]
    L += ['', '=== corrected amplitudes with each calibration (same observed medians)',
          O[['z', 'A1_syn', 'A1_real', 'A2_syn', 'A2_real']].round(3).to_string(index=False)]

    def grow(col):
        loO = O[np.isclose(O.z, 0.75)][col].values[0]
        hiO = np.nanmean([O[np.isclose(O.z, z)][col].values[0] for z in HI])
        return hiO / loO if loO and loO > 0 else np.nan
    L += ['', '=== growth factors z~0.75 -> z>4',
          f"  A1: synthetic {grow('A1_syn'):.2f}   real-sky {grow('A1_real'):.2f}",
          f"  A2: synthetic {grow('A2_syn'):.2f}   real-sky {grow('A2_real'):.2f}",
          '  (if the real-sky floor is higher at high z and the real-sky growth is lower,',
          '   the excess is a noise floor the synthetic injections underestimate.)']

    hi = r[r.zbin.isin(HI)].copy()
    if len(hi):
        hi['terc'] = pd.qcut(hi.snr.rank(method='first'), 3, labels=['low', 'mid', 'high'])
        rowsT = []
        for tc in ('low', 'mid', 'high'):
            z0 = hi[(hi.terc == tc) & (hi.scenario == 'zero')]
            rowsT.append({'tercile': tc, 'snr_med': med(hi[hi.terc == tc].snr),
                          'floor1_real': 2 * med(z0.A1), 'floor2_real': 2 * med(z0.A2)})
        L += ['', '=== real-sky floor by S/N tercile at z>4 (conventional)',
              pd.DataFrame(rowsT).round(3).to_string(index=False)]

    txt = '\n'.join(L)
    print(txt)
    with open(f'{a.out}_summary.txt', 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
