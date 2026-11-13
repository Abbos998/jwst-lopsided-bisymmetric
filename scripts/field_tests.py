#!/usr/bin/env python3
"""
field_tests.py -- is the field-to-field difference of the growth of A1, A2 a
depth (S/N) effect? No images, no injections, NumPy/pandas only. Corrected
values use the bin-level calibration of Table 1 (Eq. 2), so they are approximate
for subsets.

 A. per field and redshift bin: N, median aperture S/N, median r_d, raw medians
 B. corrected A1, A2 at z~0.75 and z>4 and their growth per field group
    (deep = CEERS + GOODS; PRIMER-COSMOS; PRIMER-UDS; all without UDS), and a
    chi^2 test of the homogeneity of the seven fields (in log growth)
 C. raw amplitudes of the field groups inside the same S/N tercile of a bin:
    if the groups agree at fixed S/N, the field difference is a depth effect
 D. Spearman correlation of the raw amplitudes with S/N in every bin
 E. growth of the field groups above a common S/N threshold

Usage:
  python3 field_tests.py --galaxies ../data/intermediate/corrected_disk_f444w.csv \
      --table1 ../data/table1_disks_F444W.csv --sample $DATA/dja_real.csv \
      --snrcat $DATA/dja_snr_f444w.csv --out $ROOT/referee_tests/fields
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
ZC = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50]
HI = (4.5, 5.5)
warnings.simplefilter("ignore", RuntimeWarning)


def group(f):
    f = str(f).lower()
    return 'PRIMER-UDS' if 'uds' in f else 'PRIMER-COSMOS' if 'cosmos' in f else 'deep (CEERS+GOODS)'


def final(med, fl, rec):
    return 2 * np.sqrt(max(med * med - fl * fl, 0)) / rec


def amps(d, t):
    """Corrected A1, A2 at z~0.75 and z>4 (mean of the two bins) and the growth."""
    r = {'N_0.75': int(np.isclose(d.zb, 0.75).sum()), 'N_z>4': int(d.zb.isin(HI).sum())}
    for k in (1, 2):
        v = {}
        for zc in (0.75,) + HI:
            s = d[np.isclose(d.zb, zc)]
            v[zc] = (final(np.median(s[f'A{k}_V6']), t.loc[zc, f'A{k}_floor_pipe'], t.loc[zc, f'A{k}_recovery'])
                     if len(s) >= 5 else np.nan)
        hi = [v[z] for z in HI if np.isfinite(v[z])]
        r[f'A{k}_lo'] = v[0.75]
        r[f'A{k}_hi'] = np.mean(hi) if hi else np.nan
        r[f'A{k}_growth'] = r[f'A{k}_hi'] / r[f'A{k}_lo'] if r[f'A{k}_lo'] > 0 else np.nan
    return r


def amps_boot(d, t, rng, nb):
    r = amps(d, t)
    parts = [d[np.isclose(d.zb, zc)] for zc in (0.75,) + HI]
    bs = [amps(pd.concat([p.iloc[rng.integers(0, len(p), len(p))] for p in parts if len(p)]), t) for _ in range(nb)]
    for k in (1, 2):
        for q in ('lo', 'hi', 'growth'):
            r[f'A{k}_{q}_err'] = np.nanstd([b[f'A{k}_{q}'] for b in bs])
        lg = np.log(np.array([b[f'A{k}_growth'] for b in bs], float))
        r[f'_lg{k}'] = np.nanstd(lg[np.isfinite(lg)]) if np.isfinite(lg).sum() > nb / 2 else np.nan
    return r


def spearman(x, y):
    rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


def show(title, rows, L, cols=None):
    df = pd.DataFrame(rows)
    df = df[[c for c in df.columns if not c.startswith('_')]]
    L += ['', f'=== {title}', (df[cols] if cols else df).round(3).to_string(index=False)]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxies', required=True)
    ap.add_argument('--table1', required=True)
    ap.add_argument('--sample', required=True)
    ap.add_argument('--snrcat', required=True)
    ap.add_argument('--field_col', default='field')
    ap.add_argument('--snr_col')
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--nboot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.set_option('display.width', 250)
    L = []

    t = pd.read_csv(a.table1).set_index('z')
    g = pd.read_csv(a.galaxies).dropna(subset=['A1_V6', 'A2_V6'])
    cat = pd.read_csv(a.sample).rename(columns={a.idx_col: 'idx'})
    if a.field_col not in cat.columns:
        raise SystemExit(f"no column '{a.field_col}' in {a.sample}: {list(cat.columns)}")
    sn = pd.read_csv(a.snrcat).rename(columns={a.idx_col: 'idx'})
    scol = a.snr_col or next((c for c in ('snr', 'snr_ap', 'snr_aper', 'SNR') if c in sn.columns), None)
    if scol is None:
        raise SystemExit(f'no S/N column in {a.snrcat}: {list(sn.columns)}; pass --snr_col')
    g = g.merge(cat[['idx', a.field_col]].rename(columns={a.field_col: 'field'}), on='idx', how='left')
    g = g.merge(sn[['idx', scol]].rename(columns={scol: 'snr'}).drop_duplicates('idx'), on='idx', how='left')
    g['zb'] = pd.cut(g.z, EDGES, labels=ZC, right=False).astype(float)
    g = g[g.zb.notna()].copy()
    g['grp'] = g.field.map(group)
    L.append(f"{len(g)} valid galaxies; {int(g.snr.notna().sum())} with S/N (column '{scol}'); "
             f"groups: {g.grp.value_counts().to_dict()}")

    # A -------------------------------------------------------------------
    rows = []
    for (f, zc), s in g.groupby(['field', 'zb']):
        rows.append({'field': f, 'z': zc, 'N': len(s), 'snr_med': s.snr.median(), 'r_d_med': s.disk_r.median(),
                     'A1_raw': s.A1_V6.median(), 'A2_raw': s.A2_V6.median()})
    A = pd.DataFrame(rows)
    A.to_csv(f'{a.out}_per_field_bin.csv', index=False)
    L += ['', '=== A median aperture S/N per field and bin', A.pivot(index='z', columns='field', values='snr_med').round(0).to_string()]
    L += ['', 'median r_d [pix] per field and bin', A.pivot(index='z', columns='field', values='r_d_med').round(2).to_string()]
    L += ['', 'raw median A1 (pipeline normalisation)', A.pivot(index='z', columns='field', values='A1_raw').round(3).to_string()]
    L += ['', 'raw median A2 (pipeline normalisation)', A.pivot(index='z', columns='field', values='A2_raw').round(3).to_string()]

    # B -------------------------------------------------------------------
    per_field = [{'sample': f, **amps_boot(s, t, rng, a.nboot)} for f, s in g.groupby('field')]
    rows = [{'sample': 'all fields', **amps_boot(g, t, rng, a.nboot)}]
    rows += [{'sample': n, **amps_boot(s, t, rng, a.nboot)} for n, s in g.groupby('grp')]
    rows.append({'sample': 'all without PRIMER-UDS', **amps_boot(g[g.grp != 'PRIMER-UDS'], t, rng, a.nboot)})
    cols = ['sample', 'N_0.75', 'N_z>4'] + [f'A{k}_{q}{e}' for k in (1, 2) for q in ('lo', 'hi', 'growth') for e in ('', '_err')]
    show('B corrected amplitudes (conventional) at z~0.75 (lo) and z>4 (hi), and growth', rows + per_field, L, cols)
    for k in (1, 2):
        x = np.log(np.array([r[f'A{k}_growth'] for r in per_field], float))
        e = np.array([r[f'_lg{k}'] for r in per_field], float)
        ok = np.isfinite(x) & np.isfinite(e) & (e > 0)
        w = 1 / e[ok] ** 2
        m = np.sum(w * x[ok]) / w.sum()
        chi2 = float(np.sum(w * (x[ok] - m) ** 2))
        L.append(f'homogeneity of the fields, A{k} growth: weighted mean {np.exp(m):.2f}, '
                 f'chi2 = {chi2:.1f} for {ok.sum() - 1} d.o.f. (log growth, bootstrap errors)')
        for q in ('lo', 'hi'):
            x = np.array([r[f'A{k}_{q}'] for r in per_field], float)
            e = np.array([r[f'A{k}_{q}_err'] for r in per_field], float)
            ok = np.isfinite(x) & (e > 0)
            w = 1 / e[ok] ** 2
            m = np.sum(w * x[ok]) / w.sum()
            L.append(f'   A{k} {q}: weighted mean {m:.3f}, chi2 = {np.sum(w * (x[ok] - m) ** 2):.1f} for {ok.sum() - 1} d.o.f.')

    # C -------------------------------------------------------------------
    s_all = g[g.snr.notna()].copy()
    rows = []
    for name, zs in (('z~0.75', (0.75,)), ('z~3.75', (3.75,)), ('z>4', HI)):
        s = s_all[s_all.zb.isin(zs)].copy()
        if len(s) < 30:
            continue
        s['terc'] = s.groupby('zb').snr.transform(lambda v: pd.qcut(v.rank(method='first'), 3, labels=['low', 'mid', 'high'])).astype(str)
        for tc in ('low', 'mid', 'high'):
            for gname, u in s[s.terc == tc].groupby('grp'):
                rows.append({'bin': name, 'S/N tercile': tc, 'group': gname, 'N': len(u), 'snr_med': u.snr.median(),
                             'A1_raw': u.A1_V6.median(), 'A2_raw': u.A2_V6.median(),
                             'A1_err': 1.2533 * u.A1_V6.std() / np.sqrt(len(u)) if len(u) > 2 else np.nan,
                             'A2_err': 1.2533 * u.A2_V6.std() / np.sqrt(len(u)) if len(u) > 2 else np.nan})
    show('C raw medians of the field groups inside the same S/N tercile (errors: 1.25 sigma/sqrt N)', rows, L)
    pd.DataFrame(rows).to_csv(f'{a.out}_snr_terciles.csv', index=False)

    # D -------------------------------------------------------------------
    rows = []
    for zc in ZC:
        s = s_all[np.isclose(s_all.zb, zc)]
        rows.append({'z': zc, 'N': len(s), 'snr_p16': s.snr.quantile(.16), 'snr_med': s.snr.median(), 'snr_p84': s.snr.quantile(.84),
                     'rho(A1,S/N)': spearman(s.A1_V6, s.snr), 'rho(A2,S/N)': spearman(s.A2_V6, s.snr),
                     'floor_A1': t.loc[zc, 'A1_floor_pipe'], 'floor_A2': t.loc[zc, 'A2_floor_pipe']})
    show('D Spearman correlation of the raw amplitudes with S/N', rows, L)

    # E -------------------------------------------------------------------
    rows = []
    for thr in (0, 30, 50, 75, 100):
        u = s_all[s_all.snr >= thr]
        for name, v in [('all', u)] + list(u.groupby('grp')):
            r = amps_boot(v, t, rng, max(a.nboot // 4, 50))
            rows.append({'S/N >=': thr, 'sample': name, 'N_0.75': r['N_0.75'], 'N_z>4': r['N_z>4'],
                         'A1_growth': r['A1_growth'], 'A1_err': r['A1_growth_err'],
                         'A2_growth': r['A2_growth'], 'A2_err': r['A2_growth_err']})
    show('E growth above a common S/N threshold (bin-level calibration of Table 1: the floor is too high for '
         'the galaxies kept, so the absolute values are lower limits; compare the groups)', rows, L)

    txt = '\n'.join(L)
    print(txt)
    with open(f'{a.out}_summary.txt', 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
