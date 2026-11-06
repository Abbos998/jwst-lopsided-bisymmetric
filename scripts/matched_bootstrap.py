#!/usr/bin/env python3
"""
matched_bootstrap.py -- uncertainties of the growth factors of the resolution-
and S/N-matched subsamples (Table B1), by a joint bootstrap of the observed
galaxies and of the injected galaxies of their calibration.

The windows are rebuilt exactly as in size_sn_matched_test.py (same defaults),
the injection runs are repeated with the same seed (the per-galaxy random
numbers are fixed, so the point estimates must reproduce size_sn_summary.txt)
and their per-galaxy rows are saved (--reuse_rows skips the injections next
time). In every bootstrap resample the observed galaxies of each bin and the
zero-amplitude and matched injections of that bin are resampled with
replacement, the floor, recovered fraction and ratio bias are recomputed and
the bin medians corrected with Eq. (1) of the paper.

Output per sample (full, size, snr, both): N per bin, median log M per bin,
growth factors with bootstrap s.d. and 16-84 / 2.5-97.5 percentiles, the
fraction of resamples with growth <= 1, and the mean corrected A1/A2 below and
above z = 3.5.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_core as cc
import injection_qpa_test as iq

HI = (4.5, 5.5)


def windows(m, big, pct, min_snr_bin):
    xlo = max(np.percentile(m[m.zbin == z].x, pct[0]) for z in big)
    xhi = min(np.percentile(m[m.zbin == z].x, pct[1]) for z in big)
    if xlo >= xhi:
        xlo = max(m[m.zbin == z].x.min() for z in big)
        xhi = min(m[m.zbin == z].x.max() for z in big)
    slo = np.median(m[m.zbin == min_snr_bin].snr)
    return xlo, xhi, slo


def terms(R):
    z0, mt = R[R.scenario == 'zero'], R[R.scenario == 'matched']
    t = {}
    for k in (1, 2):
        f = np.nanmedian(z0[f'A{k}'].values)
        mm = np.nanmedian(mt[f'A{k}'].values)
        tt = np.nanmedian(mt[f'A{k}_true'].values)
        t[f'f{k}'] = f
        t[f'r{k}'] = np.sqrt(max(mm * mm - f * f, 0)) / tt
    ratio = (mt.A1 / mt.A2).replace([np.inf, -np.inf], np.nan).values
    t['b'] = np.nanmedian(ratio) / (np.nanmedian(mt.A1_true.values) / np.nanmedian(mt.A2_true.values))
    return t


def corrected(a1, a2, t):
    out = {}
    for k, a in ((1, a1), (2, a2)):
        med = np.median(a)
        out[f'A{k}'] = 2 * np.sqrt(max(med * med - t[f'f{k}'] ** 2, 0)) / t[f'r{k}'] if t[f'r{k}'] > 0 else np.nan
    out['R12'] = np.median(a1 / a2) / t['b']
    return out


def summary_from(per, ok):
    hi = [z for z in HI if z in ok]
    g = {}
    for k in (1, 2):
        lo = per[0.75][f'A{k}'] if 0.75 in ok else np.nan
        g[f'A{k}_growth'] = np.mean([per[z][f'A{k}'] for z in hi]) / lo if (hi and lo > 0) else np.nan
    g['R12_lo'] = np.mean([per[z]['R12'] for z in ok if z < 3.5])
    g['R12_hi'] = np.mean([per[z]['R12'] for z in ok if z > 3.5])
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cc.add_common_args(ap)
    ap.add_argument('--galaxies', required=True)
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--mass_col', default='logM')
    ap.add_argument('--fwhm_arcsec', type=float, default=0.156)
    ap.add_argument('--pct', type=float, nargs=2, default=[10, 90])
    ap.add_argument('--min_n', type=int, default=20)
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--reuse_rows', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inj = cc.load_module(os.path.join(a.scripts, 'injection_test.py'), 'inj')

    cat = cc.load_catalog(a, inj)
    raw = pd.read_csv(a.catalog)
    cat['idx'] = raw[a.idx_col].values[cat.row.values].astype(int)
    cat['logM'] = pd.to_numeric(raw[a.mass_col], errors='coerce').values[cat.row.values] \
        if a.mass_col in raw.columns else np.nan
    gal = pd.read_csv(a.galaxies).dropna(subset=['A1_V6', 'A2_V6'])
    m = gal[[a.idx_col, 'z', 'A1_V6', 'A2_V6']].rename(columns={a.idx_col: 'idx', 'z': 'z_meas'}) \
        .merge(cat, on='idx', how='inner')
    m['x'] = m.re_pix / (a.fwhm_arcsec / a.pixscale)
    m['zbin'] = cc.Z_CEN[np.clip(np.digitize(m.z_meas, cc.Z_EDGES) - 1, 0, len(cc.Z_CEN) - 1)]
    big = [z for z, g in m.groupby('zbin') if len(g) >= a.min_n]
    xlo, xhi, slo = windows(m, big, a.pct, max(big))
    sel = {'full': np.ones(len(m), bool),
           'size': ((m.x >= xlo) & (m.x <= xhi)).values,
           'snr': (m.snr >= slo).values}
    sel['both'] = sel['size'] & sel['snr']
    print(f'{len(m)} galaxies joined; r_d/FWHM {xlo:.2f}-{xhi:.2f}; S/N >= {slo:.0f}')

    t1 = pd.read_csv(a.table1)
    psf = cc.psf_from_fits(a.psf, a.psf_pixscale, a.pixscale, inj)
    acf = np.load(a.noise_acf) if a.noise_acf else None
    base = dict(scripts=os.path.abspath(a.scripts), am_func=a.am_func, psf=psf, acf=acf,
                npix=int(round(a.cutout_arcsec / a.pixscale)), aper=a.snr_aper_arcsec / a.pixscale,
                sigma_cat=a.sigma_cat, eps=cc.matched_eps(a.table1), cat_err=False,
                emp=None, emp_qedges=None, variants={'ideal'})

    rng = np.random.default_rng(a.seed + 7)
    res, perbin_out = [], []
    for name in ('full', 'size', 'snr', 'both'):
        w = m[sel[name]]
        rp = os.path.join(a.out, f'rows_{name}.csv')
        if a.reuse_rows and os.path.exists(rp):
            rows = pd.read_csv(rp)
        else:
            items = cc.draw_sample(cat[cat.idx.isin(w.idx)].reset_index(drop=True), a.n_per_bin, a.seed)
            print(f'{name}: {len(w)} galaxies, {len(items)} injected models')
            rows = pd.DataFrame(cc.run_parallel(iq.worker, items, base, a.workers))
            rows.to_csv(rp, index=False)
        ok = [z for z in cc.Z_CEN if (w.zbin == z).sum() >= a.min_n]
        obs = {z: (w[w.zbin == z].A1_V6.values, w[w.zbin == z].A2_V6.values) for z in ok}
        inj_ix = {z: {s: np.flatnonzero((rows.zbin.values == z) & (rows.scenario.values == s))
                      for s in ('zero', 'matched')} for z in ok}

        def estimate(boot):
            per = {}
            for z in ok:
                a1, a2 = obs[z]
                ix = [inj_ix[z]['zero'], inj_ix[z]['matched']]
                if boot:
                    j = rng.integers(0, len(a1), len(a1))
                    a1, a2 = a1[j], a2[j]
                    ix = [i[rng.integers(0, len(i), len(i))] for i in ix]
                per[z] = corrected(a1, a2, terms(rows.iloc[np.concatenate(ix)]))
            return per

        per0 = estimate(False)
        g0 = summary_from(per0, ok)
        B = pd.DataFrame([summary_from(estimate(True), ok) for _ in range(a.n_boot)])
        r = dict(sample=name, N_lo=int((w.zbin == 0.75).sum()),
                 N_hi=int(w.zbin.isin(HI).sum()), N_45=int((w.zbin == 4.5).sum()), N_55=int((w.zbin == 5.5).sum()),
                 logM_lo=w[w.zbin == 0.75].logM.median(), logM_hi=w[w.zbin.isin(HI)].logM.median())
        for k in ('A1_growth', 'A2_growth', 'R12_lo', 'R12_hi'):
            v = B[k].dropna().values
            r[k] = g0[k]
            r[f'{k}_sd'] = np.std(v, ddof=1)
            r[f'{k}_p16'], r[f'{k}_p84'] = np.percentile(v, [16, 84])
            r[f'{k}_p2.5'], r[f'{k}_p97.5'] = np.percentile(v, [2.5, 97.5])
        for k in ('A1_growth', 'A2_growth'):
            r[f'P({k}<=1)'] = np.mean(B[k].dropna().values <= 1)
        res.append(r)
        for z in cc.Z_CEN:
            ww = w[w.zbin == z]
            perbin_out.append(dict(sample=name, zbin=z, N=len(ww), logM_med=ww.logM.median(),
                                   snr_med=ww.snr.median(), x_med=ww.x.median(),
                                   A1=per0[z]['A1'] if z in per0 else np.nan,
                                   A2=per0[z]['A2'] if z in per0 else np.nan))
    R = pd.DataFrame(res)
    P = pd.DataFrame(perbin_out)
    R.to_csv(os.path.join(a.out, 'matched_bootstrap_growth.csv'), index=False)
    P.to_csv(os.path.join(a.out, 'matched_bootstrap_perbin.csv'), index=False)
    pd.set_option('display.width', 250)
    lines = [f'r_d/FWHM {xlo:.2f}-{xhi:.2f}; S/N >= {slo:.0f}; joint bootstrap ({a.n_boot} resamples) of observed '
             f'and injected galaxies; bins with N < {a.min_n} excluded', '',
             'Per bin:', P.pivot(index='zbin', columns='sample', values='N').to_string(), '',
             'Median log M per bin:', P.pivot(index='zbin', columns='sample', values='logM_med').round(2).to_string(), '']
    for _, r in R.iterrows():
        lines.append(f"{r['sample']:5s} N(z~0.75)={r.N_lo:5d}  N(z>4)={r.N_hi:4d} ({r.N_45}+{r.N_55})  "
                     f"logM {r.logM_lo:.2f} -> {r.logM_hi:.2f}")
        for k in ('A1_growth', 'A2_growth', 'R12_lo', 'R12_hi'):
            extra = f"   P(<=1) = {r[f'P({k}<=1)']:.3f}" if 'growth' in k else ''
            lines.append(f"    {k:9s} {r[k]:.3f} +- {r[k + '_sd']:.3f}   16-84%: {r[k + '_p16']:.3f}-{r[k + '_p84']:.3f}"
                         f"   2.5-97.5%: {r[k + '_p2.5']:.3f}-{r[k + '_p97.5']:.3f}{extra}")
    lines += ['', 'Point estimates must agree with size_sn_summary.txt (same seed); the full sample is the '
              'reference of the same run.']
    txt = '\n'.join(lines)
    print('\n' + txt)
    open(os.path.join(a.out, 'matched_bootstrap_summary.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
