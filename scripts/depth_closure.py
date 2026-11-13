#!/usr/bin/env python3
"""
depth_closure.py -- closure test of the noise calibration across field depth.

The same kind of galaxy (redshift, stellar mass) observed in a deep and in a
shallow field must give the same corrected amplitude. Here every galaxy is
corrected with a noise floor that depends on its own aperture S/N,

    log A_floor = a + b log(S/N)     fitted per redshift bin to the
                                     zero-amplitude injections,

and with the recovered fraction of its S/N tercile (matched injections):
    A_corr = 2 sqrt(max(A^2 - A_floor(S/N)^2, 0)) / f_rec .
NumPy/pandas only, no images, no new injections.

 1. floor fit per bin and the fraction of galaxies below their own floor
 2. stellar mass and S/N of the field groups at z>4
 3. raw and corrected medians of the field groups in bins of stellar mass
    (z~3.75 and z>4): the closure test
 4. growth z~0.75 -> z>4 with the per-galaxy calibration, per field group

Usage:
  python3 depth_closure.py --galaxies ../data/intermediate/corrected_disk_f444w.csv \
      --inj ../data/intermediate/inj_f444w_m/injection_per_object.csv \
      --sample $DATA/dja_real.csv --snrcat $DATA/dja_snr_f444w.csv \
      --out $ROOT/referee_tests/closure
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter('ignore', RuntimeWarning)
EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
ZC = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50]
HI = (4.5, 5.5)
MASS = ['logM', 'log_mass', 'lmass', 'mass_log', 'logmstar', 'log_mstar']
MBINS = [(9.0, 9.5), (9.5, 10.0), (10.0, 11.5)]


def group(f):
    f = str(f).lower()
    return 'PRIMER-UDS' if 'uds' in f else 'PRIMER-COSMOS' if 'cosmos' in f else 'deep (CEERS+GOODS)'


def calibration(inj, centering):
    """Per bin: floor(S/N) power law for m = 1, 2; recovered fraction per S/N tercile."""
    cal = {}
    q = inj[inj.centering == centering]
    for zc in ZC:
        s = q[np.isclose(q.zbin, zc)]
        z0, mt = s[s.scenario == 'zero'], s[s.scenario == 'matched']
        c = {'edges': np.quantile(z0.snr, [1 / 3, 2 / 3]), 'snr_rng': (z0.snr.min(), z0.snr.max())}
        for k in (1, 2):
            ok = (z0[f'A{k}'] > 0) & (z0.snr > 0)
            b, a = np.polyfit(np.log10(z0.snr[ok]), np.log10(z0[f'A{k}'][ok]), 1)
            # the fit gives the mean of log A; shift it to the median of A
            res = np.log10(z0[f'A{k}'][ok]) - (a + b * np.log10(z0.snr[ok]))
            c[f'a{k}'], c[f'b{k}'] = a + np.median(res), b
            it = np.digitize(mt.snr, c['edges'])
            rec = []
            for j in range(3):
                u = mt[it == j]
                f = 10 ** (c[f'a{k}'] + c[f'b{k}'] * np.log10(np.nanmedian(u.snr)))
                m = np.nanmedian(u[f'A{k}'])
                rec.append(np.sqrt(max(m * m - f * f, 0)) / np.nanmedian(u[f'A{k}_true']))
            c[f'rec{k}'] = np.array(rec)
        cal[zc] = c
    return cal


def apply(g, cal):
    g = g.copy()
    for k in (1, 2):
        g[f'F{k}'] = np.nan
        g[f'C{k}'] = np.nan
    for zc in ZC:
        m = np.isclose(g.zb, zc) & g.snr.notna() & (g.snr > 0)
        c = cal[zc]
        snr = np.clip(g.loc[m, 'snr'], *c['snr_rng'])            # no extrapolation of the fit
        it = np.digitize(g.loc[m, 'snr'], c['edges'])
        for k in (1, 2):
            f = 10 ** (c[f'a{k}'] + c[f'b{k}'] * np.log10(snr))
            a = g.loc[m, f'A{k}_V6']
            g.loc[m, f'F{k}'] = f
            g.loc[m, f'C{k}'] = 2 * np.sqrt(np.clip(a * a - f * f, 0, None)) / c[f'rec{k}'][it]
    return g


def med_err(x, rng, nb=400):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan
    return float(np.median(x)), float(np.std(np.median(x[rng.integers(0, len(x), (nb, len(x)))], axis=1)))


def growth(d, rng, nb):
    def one(s):
        r = {}
        for k in (1, 2):
            lo = s[np.isclose(s.zb, 0.75)][f'C{k}'].median()
            hi = np.nanmean([s[np.isclose(s.zb, z)][f'C{k}'].median() for z in HI])
            r[f'A{k}_lo'], r[f'A{k}_hi'], r[f'A{k}_growth'] = lo, hi, hi / lo if lo > 0 else np.nan
        return r
    r = one(d)
    parts = [d[np.isclose(d.zb, z)] for z in (0.75,) + HI]
    bs = [one(pd.concat([p.iloc[rng.integers(0, len(p), len(p))] for p in parts if len(p)])) for _ in range(nb)]
    for key in list(r):
        r[f'{key}_err'] = np.nanstd([b[key] for b in bs])
    return {'N_0.75': int(np.isclose(d.zb, 0.75).sum()), 'N_z>4': int(d.zb.isin(HI).sum()), **r}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxies', required=True)
    ap.add_argument('--inj', required=True, help='inj_f444w_m/injection_per_object.csv')
    ap.add_argument('--sample', required=True)
    ap.add_argument('--snrcat', required=True)
    ap.add_argument('--centering', default='catalog')
    ap.add_argument('--field_col', default='field')
    ap.add_argument('--mass_col')
    ap.add_argument('--snr_col')
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--nboot', type=int, default=500)
    ap.add_argument('--seed', type=int, default=9)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.set_option('display.width', 250)
    L = []

    g = pd.read_csv(a.galaxies).dropna(subset=['A1_V6', 'A2_V6'])
    cat = pd.read_csv(a.sample).rename(columns={a.idx_col: 'idx'})
    mcol = a.mass_col or next((c for c in MASS if c in cat.columns), None)
    if mcol is None or a.field_col not in cat.columns:
        raise SystemExit(f'mass or field column not found in {a.sample}: {list(cat.columns)}')
    sn = pd.read_csv(a.snrcat).rename(columns={a.idx_col: 'idx'})
    scol = a.snr_col or next((c for c in ('snr', 'snr_ap', 'snr_aper', 'SNR') if c in sn.columns), None)
    g = g.merge(cat[['idx', a.field_col, mcol]].rename(columns={a.field_col: 'field', mcol: 'logM'}), on='idx', how='left')
    g = g.merge(sn[['idx', scol]].rename(columns={scol: 'snr'}).drop_duplicates('idx'), on='idx', how='left')
    g['zb'] = pd.cut(g.z, EDGES, labels=ZC, right=False).astype(float)
    g = g[g.zb.notna() & g.snr.notna() & (g.snr > 0)].copy()
    g['grp'] = g.field.map(group)
    cal = calibration(pd.read_csv(a.inj), a.centering)
    g = apply(g, cal)
    L.append(f'{len(g)} valid galaxies with S/N; injections: centring "{a.centering}"')

    # 1 -------------------------------------------------------------------
    rows = []
    for zc in ZC:
        s, c = g[np.isclose(g.zb, zc)], cal[zc]
        r = {'z': zc, 'N': len(s), 'inj S/N min': c['snr_rng'][0], 'inj S/N max': c['snr_rng'][1],
             'obs S/N p5': s.snr.quantile(.05), 'obs outside inj range': float(((s.snr < c['snr_rng'][0]) | (s.snr > c['snr_rng'][1])).mean())}
        for k in (1, 2):
            r[f'slope{k}'] = c[f'b{k}']
            r[f'floor{k}(S/N=20)'] = 10 ** (c[f'a{k}'] + c[f'b{k}'] * np.log10(20))
            r[f'floor{k}(S/N=200)'] = 10 ** (c[f'a{k}'] + c[f'b{k}'] * np.log10(200))
            r[f'rec{k} low/mid/high'] = '/'.join(f'{x:.2f}' for x in c[f'rec{k}'])
            r[f'below floor {k}'] = float((s[f'A{k}_V6'] <= s[f'F{k}']).mean())
        rows.append(r)
    L += ['', '=== 1 floor(S/N) fit per bin (pipeline normalisation) and fraction of galaxies at or below their own floor',
          pd.DataFrame(rows).round(3).to_string(index=False)]

    # 2 -------------------------------------------------------------------
    h = g[g.zb.isin(HI)]
    L += ['', '=== 2 field groups at z>4', h.groupby('grp').agg(N=('snr', 'size'), logM_med=('logM', 'median'),
          snr_p16=('snr', lambda v: v.quantile(.16)), snr_med=('snr', 'median'), snr_p84=('snr', lambda v: v.quantile(.84)),
          r_d_med=('disk_r', 'median')).round(2).to_string()]

    # 3 -------------------------------------------------------------------
    rows = []
    for name, zs in (('z~0.75', (0.75,)), ('z~3.75', (3.75,)), ('z>4', HI)):
        s = g[g.zb.isin(zs)]
        for lo, hi in [(8.0, 12.5)] + MBINS:
            for gname, u in s[(s.logM >= lo) & (s.logM < hi)].groupby('grp'):
                r = {'bin': name, 'logM': 'all' if lo == 8.0 else f'{lo}-{hi}', 'group': gname, 'N': len(u),
                     'logM_med': u.logM.median(), 'snr_med': u.snr.median()}
                for k in (1, 2):
                    r[f'A{k}_raw'] = u[f'A{k}_V6'].median()
                    r[f'A{k}_corr'], r[f'A{k}_corr_err'] = med_err(u[f'C{k}'], rng)
                rows.append(r)
    C = pd.DataFrame(rows)
    C.to_csv(f'{a.out}_mass_bins.csv', index=False)
    L += ['', '=== 3 closure: field groups at fixed redshift and stellar mass '
          '(raw: pipeline normalisation; corr: per-galaxy S/N calibration, conventional)', C.round(3).to_string(index=False)]

    # 4 -------------------------------------------------------------------
    rows = [{'sample': 'all fields', **growth(g, rng, a.nboot)}]
    rows += [{'sample': n, **growth(s, rng, a.nboot)} for n, s in g.groupby('grp')]
    for thr in (30, 50, 100):
        rows.append({'sample': f'all, S/N >= {thr}', **growth(g[g.snr >= thr], rng, a.nboot)})
    G = pd.DataFrame(rows)
    G.to_csv(f'{a.out}_growth.csv', index=False)
    L += ['', '=== 4 growth with the per-galaxy S/N calibration (medians of the corrected values)', G.round(3).to_string(index=False)]

    txt = '\n'.join(L)
    print(txt)
    with open(f'{a.out}_summary.txt', 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
