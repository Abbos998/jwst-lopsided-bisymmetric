#!/usr/bin/env python3
"""
referee_tests.py -- five checks on the per-galaxy measurements that need no new
injection runs. Every corrected value uses the floor, recovered fraction and
ratio bias of the whole redshift bin (Table 1, Eq. 2), exactly as mass_test.py,
so the corrections of the subsets are approximate.

 1. fields      per-field growth and leave-one-field-out growth of A1, A2
 2. excluded    galaxies without a valid measurement vs the measured ones:
                medians of log M*, n, q, r_d per redshift bin (+ KS p-values)
 3. sersic      growth for narrower Sersic-index cuts
 4. snrjoin     the valid galaxies that are missing from the S/N catalogue used
                by the tests of Table B1: where they are and what removing them
                does to the growth factors
 5. specz       outlier fraction of the photometric redshifts among the
                galaxies with a spectroscopic redshift

Usage:
  python3 referee_tests.py --galaxies ../data/intermediate/corrected_disk_f444w.csv \
      --table1 ../data/table1_disks_F444W.csv --sample $DATA/dja_real.csv \
      --snrcat $DATA/dja_snr_f444w.csv --morpho $D/full-good_morpho-phot.fits.gz \
      --out $ROOT/referee_tests/referee
Columns are auto-detected; override with --field_col, --mass_col, --n_col,
--zspec_col, --zphot_col. If the catalogue has no field column the field is taken from the
sky position (five regions; the two PRIMER fields are then not split).
"""
import argparse
import os

import numpy as np
import pandas as pd

EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
ZC = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50]
HI = (4.5, 5.5)
MASS = ['logM', 'log_mass', 'lmass', 'mass_log', 'logmstar', 'log_mstar']
NSER = ['n_sersic', 'n', 'sersic', 'SERSIC']
FIELD = ['field', 'FIELD', 'file', 'root', 'file_phot', 'file_zout', 'survey']
SPEC = ['z_spec', 'zspec', 'z_spec_best', 'spec_z']
PHOT = ['z_phot', 'zphot', 'z_ml', 'z500', 'z50', 'z_peak']
SKY = {'ceers': (214.9, 52.9), 'gdn': (189.2, 62.2), 'gds': (53.1, -27.8),
       'primer-cosmos': (150.1, 2.3), 'primer-uds': (34.4, -5.2)}


def pick(cols, cands, forced=None):
    if forced:
        if forced not in cols:
            raise SystemExit(f"column '{forced}' not found; available: {list(cols)[:80]}")
        return forced
    return next((c for c in cands if c in cols), None)


def final(med, fl, rec):
    return 2 * np.sqrt(max(med * med - fl * fl, 0)) / rec


def ks_p(x, y):
    """Two-sample Kolmogorov-Smirnov p-value (asymptotic; no SciPy needed)."""
    x, y = np.sort(x), np.sort(y)
    al = np.concatenate([x, y])
    d = np.max(np.abs(np.searchsorted(x, al, side='right') / len(x)
                      - np.searchsorted(y, al, side='right') / len(y)))
    en = np.sqrt(len(x) * len(y) / (len(x) + len(y)))
    lam = (en + 0.12 + 0.11 / en) * d
    j = np.arange(1, 101)
    return float(np.clip(2 * np.sum((-1.0) ** (j - 1) * np.exp(-2 * j * j * lam * lam)), 0, 1))


def bins(d, t):
    """Corrected A1, A2 per bin (Table 1 calibration); NaN below 5 galaxies."""
    out = {}
    for zc in ZC:
        s = d[np.isclose(d.zb, zc)]
        r = {'N': len(s)}
        for k in (1, 2):
            r[f'A{k}'] = (final(np.median(s[f'A{k}_V6']), t.loc[zc, f'A{k}_floor_pipe'],
                                t.loc[zc, f'A{k}_recovery']) if len(s) >= 5 else np.nan)
        out[zc] = r
    return out


def growth(d, t):
    b = bins(d, t)
    g = {'N_0.75': b[0.75]['N'], 'N_z>4': sum(b[z]['N'] for z in HI)}
    for k in (1, 2):
        hi = [b[z][f'A{k}'] for z in HI if np.isfinite(b[z][f'A{k}'])]
        lo = b[0.75][f'A{k}']
        g[f'A{k}_growth'] = np.mean(hi) / lo if (hi and np.isfinite(lo) and lo > 0) else np.nan
    return g


def growth_boot(d, t, rng, nb):
    """Growth factors with bootstrap s.d. (galaxies resampled within each bin)."""
    g = growth(d, t)
    parts = {zc: d[np.isclose(d.zb, zc)] for zc in (0.75,) + HI}
    bs = []
    for _ in range(nb):
        r = pd.concat([p.iloc[rng.integers(0, len(p), len(p))] for p in parts.values() if len(p)])
        bs.append(growth(r, t))
    for k in (1, 2):
        v = np.array([b[f'A{k}_growth'] for b in bs], float)
        g[f'A{k}_err'] = np.nanstd(v) if np.isfinite(v).sum() > nb / 2 else np.nan
    return g


def show(title, rows, lines):
    df = pd.DataFrame(rows)
    lines += ['', f'=== {title}', df.round(3).to_string(index=False)]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxies', required=True)
    ap.add_argument('--table1', required=True)
    ap.add_argument('--sample', required=True, help='dja_real.csv (idx, q, Sersic n, log M*, ra, dec)')
    ap.add_argument('--snrcat', help='dja_snr_f444w.csv (test 4)')
    ap.add_argument('--morpho', help='DJA morpho-phot FITS, rows indexed by idx (field, z_spec)')
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--field_col')
    ap.add_argument('--mass_col')
    ap.add_argument('--n_col')
    ap.add_argument('--zspec_col')
    ap.add_argument('--zphot_col', help='photometric redshift (default: auto-detected, else the adopted z)')
    ap.add_argument('--outlier', type=float, default=0.15, help='|dz|/(1+z_spec) of a catastrophic outlier')
    ap.add_argument('--nboot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    L = []

    t = pd.read_csv(a.table1).set_index('z')
    allg = pd.read_csv(a.galaxies)
    cat = pd.read_csv(a.sample).rename(columns={a.idx_col: 'idx'})
    mcol, ncol = pick(cat.columns, MASS, a.mass_col), pick(cat.columns, NSER, a.n_col)
    keep = {c: n for c, n in ((mcol, 'logM'), (ncol, 'n_ser')) if c}
    extra = [c for c in ('ra', 'dec') if c in cat.columns and c not in allg.columns]
    fcol = pick(cat.columns, FIELD, a.field_col if a.field_col in cat.columns else None)
    if fcol:
        keep[fcol] = 'field'
    allg = allg.merge(cat[['idx'] + list(keep) + extra].rename(columns=keep), on='idx', how='left')

    fits_cols = {}
    if a.morpho:
        from astropy.io import fits
        with fits.open(a.morpho, memmap=True) as h:
            dat = h[1].data
            names = dat.columns.names
            idx = allg.idx.astype(int).values
            if 'field' not in allg:
                fc = pick(names, FIELD, a.field_col)
                if fc:
                    allg['field'] = np.char.strip(np.asarray(dat[fc]).astype(str))[idx]
            sc = pick(names, SPEC, a.zspec_col)
            if sc:
                fits_cols['zspec'] = np.asarray(dat[sc], float)[idx]
            pc = pick(names, PHOT, a.zphot_col)
            if pc:
                fits_cols['zphot'] = np.asarray(dat[pc], float)[idx]
            for c in ('ra', 'dec'):
                if c not in allg and c in names:
                    allg[c] = np.asarray(dat[c], float)[idx]
    if 'field' not in allg:
        if not {'ra', 'dec'} <= set(allg.columns):
            raise SystemExit('no field column and no ra/dec: pass --field_col (and --morpho)')
        nm = list(SKY)
        dist = np.array([np.hypot((allg.ra - r) * np.cos(np.radians(dd)), allg.dec - dd) for r, dd in SKY.values()])
        allg['field'] = np.array(nm)[dist.argmin(0)]
        L.append('NOTE: no field column found; fields assigned from the sky position (5 regions).')

    allg['zb'] = pd.cut(allg.z, EDGES, labels=ZC, right=False).astype(float)
    allg['valid'] = allg.A1_V6.notna() & allg.A2_V6.notna()
    g = allg[allg.valid & allg.zb.notna()].copy()
    ref = growth_boot(g, t, rng, a.nboot)
    dev = max(abs(bins(g, t)[zc][f'A{k}'] - t.loc[zc, f'A{k}_final_conv']) for zc in ZC for k in (1, 2))
    L += [f'{len(allg)} face-on galaxies, {len(g)} with a valid measurement; Table 1 reproduced to {dev:.1e}',
          f"columns: mass '{mcol}', Sersic '{ncol}', field '{fcol or 'FITS / sky position'}'"]
    if dev > 1e-3:
        L.append('WARNING: the bin medians do not reproduce Table 1; check the inputs')

    # 1 ---------------------------------------------------------------- fields
    rows = [{'sample': 'all fields', **ref}]
    for f in sorted(g.field.dropna().unique()):
        rows.append({'sample': f'only {f}', **growth_boot(g[g.field == f], t, rng, a.nboot)})
    for f in sorted(g.field.dropna().unique()):
        rows.append({'sample': f'without {f}', **growth_boot(g[g.field != f], t, rng, a.nboot)})
    show('1 fields: growth z~0.75 -> z>4 (bootstrap s.d.)', rows, L).to_csv(f'{a.out}_fields.csv', index=False)
    mix = pd.crosstab(g.zb, g.field)
    L += ['', 'galaxies per bin and field:', mix.to_string()]

    # 2 -------------------------------------------------------------- excluded
    props = [c for c in ('logM', 'n_ser', 'q', 'disk_r') if c in allg]
    rows = []
    for zc in ZC:
        s = allg[np.isclose(allg.zb, zc)]
        v, x = s[s.valid], s[~s.valid]
        r = {'z': zc, 'N_valid': len(v), 'N_excl': len(x), 'excl_frac': len(x) / max(len(s), 1)}
        for p in props:
            r[f'{p}_valid'], r[f'{p}_excl'] = v[p].median(), x[p].median()
            ok = v[p].notna().sum() > 5 and x[p].notna().sum() > 5
            r[f'{p}_KSp'] = ks_p(v[p].dropna().values, x[p].dropna().values) if ok else np.nan
        rows.append(r)
    show('2 excluded vs measured galaxies (medians; KS p-value)', rows, L).to_csv(f'{a.out}_excluded.csv', index=False)

    # 3 ---------------------------------------------------------------- Sersic
    if 'n_ser' in g:
        rows = [{'cut': 'adopted (0.3<n<2.5)', **ref}]
        for lo, hi in ((0.3, 1.5), (0.5, 1.5), (0.8, 1.2), (1.5, 2.5)):
            rows.append({'cut': f'{lo}<n<{hi}', **growth_boot(g[(g.n_ser > lo) & (g.n_ser < hi)], t, rng, a.nboot)})
        show('3 Sersic-index cuts', rows, L).to_csv(f'{a.out}_sersic.csv', index=False)

    # 4 --------------------------------------------------------------- S/N join
    if a.snrcat:
        sn = pd.read_csv(a.snrcat).rename(columns={a.idx_col: 'idx'})
        scol = next((c for c in ('snr', 'snr_ap') if c in sn.columns), None)
        ids = set(sn.idx if scol is None else sn.loc[np.isfinite(sn[scol]), 'idx'])
        g['joined'] = g.idx.isin(ids)
        rows = []
        for zc in ZC:
            s = g[np.isclose(g.zb, zc)]
            j, m = s[s.joined], s[~s.joined]
            rows.append({'z': zc, 'N_joined': len(j), 'N_missing': len(m),
                         'A1_joined': j.A1_V6.median(), 'A1_missing': m.A1_V6.median(),
                         'A2_joined': j.A2_V6.median(), 'A2_missing': m.A2_V6.median(),
                         'r_d_joined': j.disk_r.median(), 'r_d_missing': m.disk_r.median()})
        show(f'4 valid galaxies missing from the S/N catalogue ({(~g.joined).sum()} of {len(g)}); '
             'medians, pipeline normalisation', rows, L).to_csv(f'{a.out}_snrjoin.csv', index=False)
        rows = [{'sample': 'all valid', **ref},
                {'sample': 'joined only', **growth_boot(g[g.joined], t, rng, a.nboot)}]
        show('4b growth with and without the missing galaxies (Table 1 calibration)', rows, L)

    # 5 ---------------------------------------------------------------- spec-z
    if 'zspec' in fits_cols:
        zs = pd.Series(fits_cols['zspec'], index=allg.index)[g.index]
        ok = np.isfinite(zs) & (zs > 0)
        zp = pd.Series(fits_cols['zphot'], index=allg.index)[g.index] if 'zphot' in fits_cols else g.z
        ok &= np.isfinite(zp)
        dz = (zp[ok] - zs[ok]) / (1 + zs[ok])
        if (np.abs(dz) < 1e-6).mean() > 0.9:
            L += ['', '=== 5 spec-z: the adopted z equals z_spec for these galaxies, so the photometric '
                  'outlier fraction cannot be measured from this column; needs the z_phot column.']
        else:
            rows = []
            for name, sel in (('all', dz.index), ('z_spec<3.5', dz.index[zs[ok] < 3.5]), ('z_spec>3.5', dz.index[zs[ok] >= 3.5])):
                d = dz[sel]
                n_out = int((np.abs(d) > a.outlier).sum())
                rows.append({'sample': name, 'N_spec': len(d), 'N_outlier': n_out,
                             'outlier_frac': n_out / max(len(d), 1),
                             'sigma_NMAD': 1.4826 * np.median(np.abs(d - np.median(d))) if len(d) else np.nan})
            show(f'5 photometric vs spectroscopic redshift (outlier: |dz|/(1+z) > {a.outlier})', rows, L)
    else:
        L += ['', '=== 5 spec-z: no spectroscopic-redshift column found (pass --morpho / --zspec_col)']

    txt = '\n'.join(L)
    print(txt)
    with open(f'{a.out}_summary.txt', 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
