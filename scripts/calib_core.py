#!/usr/bin/env python3
"""
calib_core.py -- shared machinery for the calibration-systematics tests
(injection_qpa_test.py, psf_bootstrap_test.py).

It reuses the paper's own code: model disks, PSF loading, correlated noise
and catalogue handling from injection_test.py, and the adopted estimator
(am_wrapper.py:measure_v6, pipeline normalisation). The galaxy sample is
drawn exactly as in injection_test.py (n_per_bin per redshift bin, with
replacement when a bin has fewer galaxies), and every random quantity of a
model (centre, phases, PA, centring error, noise field) is fixed by a
per-galaxy seed, so that different calibration variants are evaluated on
identical models and noise ("common random numbers"). Differences between
variants are then not diluted by sampling noise.

Calibration terms follow make_table1.py:
    floor_m = median A_m (zero scenario)
    rec_m   = sqrt(median A_m(matched)^2 - floor_m^2) / median A_m,true(matched)
    A_final = 2 sqrt(A_meas^2 - floor^2) / rec           (conventional)
    b       = median(A1/A2)(matched) / (median A1_true / median A2_true)
with A_meas, the observed medians, taken from table1_disks_F444W.csv.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

Z_EDGES = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
Z_CEN = np.array([0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50])

_MODS = {}


def load_module(path, name):
    if name in _MODS:
        return _MODS[name]
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _MODS[name] = m
    return m


def get_measure(scripts_dir, am_func):
    """am_func 'file.py:function' (relative to scripts_dir) or 'builtin'."""
    inj = load_module(os.path.join(scripts_dir, 'injection_test.py'), 'inj')
    if am_func == 'builtin':
        return lambda img, x0, y0, q, pa, re: inj.measure_am_builtin(img, x0, y0, q, pa, re)
    path, func = am_func.rsplit(':', 1)
    if not os.path.isabs(path):
        path = os.path.join(scripts_dir, path)
    return inj.load_user_am(f'{path}:{func}')


def add_common_args(ap):
    a = ap.add_argument
    a('--scripts', required=True, help='package scripts/ directory (injection_test.py, am_wrapper.py)')
    a('--catalog', required=True, help='face-on disk catalogue, e.g. $DATA/dja_snr_f444w.csv')
    a('--table1', required=True, help='table1_disks_F444W.csv (A?_meas_pipe, A?_final_conv)')
    a('--out', required=True)
    a('--psf', required=True, help='empirical PSF FITS, e.g. $ROOT/psf/F444W_empirical_psf.fits')
    a('--psf_pixscale', type=float, help='arcsec/pix of the PSF (default: PIXSCALE header)')
    a('--noise_acf', help='noise ACF .npy, e.g. $ROOT/noise_acf_f444w.npy')
    a('--am_func', default='am_wrapper.py:measure_v6')
    a('--pixscale', type=float, default=0.05)
    a('--cutout_arcsec', type=float, default=5.0)
    a('--n_per_bin', type=int, default=300)
    a('--sigma_cat', type=float, default=0.10)
    a('--snr_aper_arcsec', type=float, default=0.25)
    a('--re_col', default='disk_r_pix')
    a('--re_unit', choices=['arcsec', 'pix'], default='pix')
    a('--re_ref_scale', type=float, default=0.04)
    a('--n_col', default='n_sersic')
    a('--snr_col', default='snr_ap')
    a('--q_col')
    a('--z_col')
    a('--pa_col')
    a('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a('--seed', type=int, default=2026)


def load_catalog(args, inj):
    df = pd.read_csv(args.catalog) if not args.catalog.endswith(('.fits', '.fits.gz')) else \
        __import__('astropy.table', fromlist=['Table']).Table.read(args.catalog).to_pandas()
    over = {'re': args.re_col, 'n': args.n_col, 'snr': args.snr_col,
            'q': args.q_col, 'z': args.z_col, 'pa': args.pa_col}
    cols = inj.resolve_columns(df, over)
    miss = [k for k in ('z', 're', 'n', 'q', 'snr') if k not in cols]
    if miss:
        sys.exit(f'catalogue columns not found for {miss}; columns are {list(df.columns)}')
    d = pd.DataFrame({k: pd.to_numeric(df[c], errors='coerce') for k, c in cols.items()})
    d['row'] = np.arange(len(df))          # row in the input catalogue
    if args.re_unit == 'arcsec':
        d['re_pix'] = d['re'] / args.pixscale
    else:
        d['re_pix'] = d['re'] * (args.re_ref_scale / args.pixscale if args.re_ref_scale else 1.0)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=['z', 're_pix', 'n', 'q', 'snr'])
    d = d[(d.q > 0.7) & (d.n > 0.3) & (d.n < 2.5) & (d.snr > 10)]
    d = d[(d.z >= Z_EDGES[0]) & (d.z < Z_EDGES[-1]) & (d.re_pix > 0)]
    return d.reset_index(drop=True)


def draw_sample(d, n_per_bin, seed):
    """One record per injected galaxy, with all random quantities fixed."""
    rng = np.random.default_rng(seed)
    out = []
    for ib, (lo, hi) in enumerate(zip(Z_EDGES[:-1], Z_EDGES[1:])):
        sub = d[(d.z >= lo) & (d.z < hi)]
        if sub.empty:
            continue
        draw = sub.sample(n=n_per_bin, replace=len(sub) < n_per_bin,
                          random_state=int(rng.integers(1 << 31)))
        for g in draw.to_dict('records'):
            g.update(zbin=float(Z_CEN[ib]), n=float(np.clip(g['n'], 0.3, 2.5)),
                     ux=rng.uniform(-.5, .5), uy=rng.uniform(-.5, .5),
                     phases=rng.uniform(0, 2 * np.pi, 3),
                     pa=float(g['pa']) if np.isfinite(g.get('pa', np.nan)) else rng.uniform(0, 180),
                     cdx=rng.normal(), cdy=rng.normal(),
                     seed_zero=int(rng.integers(1 << 31)), seed_match=int(rng.integers(1 << 31)),
                     seed_err=int(rng.integers(1 << 31)))
            out.append(g)
    return out


def matched_eps(table1):
    t = pd.read_csv(table1)
    return {round(float(r.z), 2): (float(r.A1_final_conv), float(r.A2_final_conv), 0.0)
            for r in t.itertuples()}


def psf_from_fits(path, psf_pixscale, pixscale, inj):
    from astropy.io import fits
    ps = psf_pixscale or fits.getheader(path).get('PIXSCALE', pixscale)
    return inj.load_psf(path, ps, pixscale)


def build_clean(g, eps, npix, inj):
    x0 = (npix - 1) / 2 + g['ux']
    y0 = (npix - 1) / 2 + g['uy']
    clean = inj.make_model(npix, x0, y0, g['re_pix'], g['n'], g['q'], g['pa'], eps, g['phases'])
    return clean, x0, y0


def observe(clean, psf, g, x0, y0, seed, aper, acf, inj):
    obs = inj.fftconvolve(clean, psf, mode='same')
    img, _ = inj.add_noise(obs, g['snr'], aper, x0, y0, np.random.default_rng(seed), acf)
    return img


def calibrate(rows, table1, label_col='variant'):
    """rows: DataFrame with zbin, scenario (zero/matched), variant, A1, A2,
    A1_true, A2_true. Returns per-variant, per-bin calibration and the
    corrected values of Table 1 obtained with it."""
    t1 = pd.read_csv(table1).set_index('z')
    out = []
    for (v, zb), g in rows.groupby([label_col, 'zbin']):
        z0, mt = g[g.scenario == 'zero'], g[g.scenario == 'matched']
        r = {label_col: v, 'zbin': zb, 'N': len(mt)}
        for m in (1, 2):
            f = np.nanmedian(z0[f'A{m}'])
            mm = np.nanmedian(mt[f'A{m}'])
            tt = np.nanmedian(mt[f'A{m}_true'])
            rec = np.sqrt(max(mm * mm - f * f, 0)) / tt
            am = t1.loc[zb, f'A{m}_meas_pipe']
            r[f'A{m}_floor_pipe'] = f
            r[f'A{m}_rec'] = rec
            r[f'A{m}_final_conv'] = 2 * np.sqrt(max(am * am - f * f, 0)) / rec
        ratio = (mt.A1 / mt.A2).replace([np.inf, -np.inf], np.nan)
        b = np.nanmedian(ratio) / (np.nanmedian(mt.A1_true) / np.nanmedian(mt.A2_true))
        r['R12_bias'] = b
        r['R12_corr'] = t1.loc[zb, 'A1A2_obs'] / b
        out.append(r)
    return pd.DataFrame(out)


def growth(cal, label_col='variant'):
    rows = []
    for v, g in cal.groupby(label_col):
        g = g.set_index('zbin')
        hi = [z for z in (4.5, 5.5) if z in g.index]
        r = {label_col: v}
        for m in (1, 2):
            r[f'A{m}_growth'] = g.loc[hi, f'A{m}_final_conv'].mean() / g.loc[0.75, f'A{m}_final_conv']
        r['R12_lo'] = g.loc[g.index < 3.5, 'R12_corr'].mean()
        r['R12_hi'] = g.loc[g.index > 3.5, 'R12_corr'].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def run_parallel(worker, items, cfg, workers, chunk=20):
    """Run worker(chunk_of_items, cfg) -> list of dicts, in processes."""
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    res = []
    if workers <= 1:
        for k, c in enumerate(chunks):
            res += worker(c, cfg)
            print(f'  {k + 1}/{len(chunks)} chunks', flush=True)
        return res
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(worker, c, cfg) for c in chunks]
        for k, f in enumerate(as_completed(futs)):
            res += f.result()
            if (k + 1) % max(1, len(chunks) // 20) == 0 or k + 1 == len(chunks):
                print(f'  {k + 1}/{len(chunks)} chunks', flush=True)
    return res


def compare_to_paper(cal, table1, variant):
    """Check that the reproduced nominal calibration matches Table 1."""
    t1 = pd.read_csv(table1).set_index('z')
    c = cal[cal.iloc[:, 0] == variant].set_index('zbin')
    lines = ['  z    rec1(here/paper)   rec2(here/paper)   A1_final(here/paper)   A2_final(here/paper)']
    for z in c.index:
        lines.append(f'  {z:4.2f}  {c.loc[z, "A1_rec"]:.3f}/{t1.loc[z, "A1_recovery"]:.3f}'
                     f'       {c.loc[z, "A2_rec"]:.3f}/{t1.loc[z, "A2_recovery"]:.3f}'
                     f'       {c.loc[z, "A1_final_conv"]:.3f}/{t1.loc[z, "A1_final_conv"]:.3f}'
                     f'          {c.loc[z, "A2_final_conv"]:.3f}/{t1.loc[z, "A2_final_conv"]:.3f}')
    return '\n'.join(lines)
