#!/usr/bin/env python3
"""
mass_test.py — is the redshift growth of A1, A2 a change of the sample's stellar
mass distribution? Three tests on the per-galaxy measurements (variant V6):

 (a) mass-matched: in every redshift bin the galaxies are re-weighted so that the
     stellar-mass distribution equals that of the whole sample (0.25 dex bins);
     weighted medians are corrected with the floor and recovery fraction of the
     bin (Eq. 2), exactly as Table 1;
 (b) narrow mass slices: the same, with no weights, in fixed slices of log M*;
 (c) joint regression: medians of cells (z bin x 0.5 dex in log M*, N >= 20),
     corrected with Eq. (2), fitted by weighted least squares
         log10 A = a + b log10(1+z) + c (log M* - 10)
     with and without the mass term; b measures the redshift growth at fixed mass.

Floor and recovery fractions are those of the whole bin; they depend on size and
S/N, which correlate weakly with mass, so the corrections of the mass-selected
subsets are approximate (stated in the output).

Usage:
  python3 mass_test.py --objects ../data/intermediate/corrected_disk_f444w.csv \
      --catalog ~/Downloads/pasp_v3/data/dja_real.csv \
      --table ../data/table1_disks_F444W.csv --out ../data/mass_test
"""
import argparse

import numpy as np
import pandas as pd

Z_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
Z_CEN = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50]
MASS_COLS = ['logM', 'log_mass', 'lmass', 'mass_log', 'logmstar', 'log_mstar']


def wmedian(x, w):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(np.interp(0.5 * c[-1], c, x))


def final(med, fl, rec):
    """Eq. (2): conventional normalisation."""
    return 2 * np.sqrt(max(med * med - fl * fl, 0)) / rec


def corrected(s, t, zc, w, rng, nb):
    """Final A1, A2 and A1/A2 (+ bootstrap errors) of the galaxies s of bin zc."""
    r = t.loc[zc]
    out = {}
    x = {k: s[f'{k}_V6'].to_numpy(float) for k in ('A1', 'A2')}
    x['R12'] = x['A1'] / x['A2']
    ok = np.isfinite(x['A1']) & np.isfinite(x['A2']) & np.isfinite(x['R12'])
    x = {k: v[ok] for k, v in x.items()}
    w = w[ok]
    if len(w) < 20:
        return None

    def est(idx):
        e = {}
        for k in ('A1', 'A2'):
            e[k] = final(wmedian(x[k][idx], w[idx]), r[f'{k}_floor_pipe'], r[f'{k}_recovery'])
        e['R12'] = wmedian(x['R12'][idx], w[idx]) / r['A1A2_bias']
        return e

    base = est(np.arange(len(w)))
    bs = [est(rng.integers(0, len(w), len(w))) for _ in range(nb)]
    for k in base:
        out[k] = base[k]
        out[f'{k}_err'] = float(np.std([b[k] for b in bs]))
    out['N'] = int(len(w))
    out['N_eff'] = float(w.sum() ** 2 / (w * w).sum())
    return out


def growth(tab, k):
    lo = tab[np.isclose(tab.z, 0.75)][k]
    hi = tab[tab.z > 4][k]
    return float(hi.mean() / lo.iloc[0]) if len(lo) and len(hi) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--objects', required=True)
    ap.add_argument('--catalog', required=True, help='CSV with idx and log stellar mass')
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--mass_col')
    ap.add_argument('--table', default='../data/table1_disks_F444W.csv')
    ap.add_argument('--slices', default='9.0,9.5,10.0,10.5,11.5')
    ap.add_argument('--nboot', type=int, default=300)
    ap.add_argument('--out', default='../data/mass_test')
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    d = pd.read_csv(a.objects)
    cat = pd.read_csv(a.catalog)
    mcol = a.mass_col or next((c for c in MASS_COLS if c in cat.columns), None)
    if mcol is None:
        raise SystemExit(f"no stellar-mass column in {a.catalog}: {list(cat.columns)}")
    d = d.merge(cat[[a.idx_col, mcol]].rename(columns={a.idx_col: 'idx', mcol: 'logM'}),
                on='idx', how='left')
    d = d[d.A1_V6.notna() & d.logM.notna()]
    d['zb'] = pd.cut(d.z, Z_EDGES, labels=Z_CEN).astype(float)
    t = pd.read_csv(a.table).set_index('z')
    print(f"galaxies with a valid measurement and a mass: {len(d)} (mass column '{mcol}')\n")

    # distribution of mass per bin
    dist = d.groupby('zb').logM.describe(percentiles=[.16, .5, .84])[['count', '16%', '50%', '84%']]
    print("log M* per redshift bin (16/50/84 percentiles):")
    print(dist.round(2).to_string(), '\n')

    rows = []
    # Table 1 as reference (unweighted, same function)
    for zc in Z_CEN:
        s = d[np.isclose(d.zb, zc)]
        r = corrected(s, t, zc, np.ones(len(s)), rng, a.nboot)
        if r:
            rows.append({'test': 'all (Table 1 sample)', 'z': zc, **r})
    # (a) mass-matched to the pooled distribution
    edges = np.arange(8.75, 11.76, 0.25)
    ref = np.histogram(d.logM, edges)[0].astype(float)
    ref /= ref.sum()
    for zc in Z_CEN:
        s = d[np.isclose(d.zb, zc)]
        h = np.histogram(s.logM, edges)[0].astype(float)
        h /= max(h.sum(), 1)
        ib = np.clip(np.digitize(s.logM, edges) - 1, 0, len(ref) - 1)
        w = np.where(h[ib] > 0, ref[ib] / np.maximum(h[ib], 1e-12), 0.0)
        w = np.minimum(w, 10 * np.median(w[w > 0])) if (w > 0).any() else w
        missing = ref[h == 0].sum()
        r = corrected(s[w > 0], t, zc, w[w > 0], rng, a.nboot)
        if r:
            rows.append({'test': 'mass-matched (pooled)', 'z': zc, 'ref_mass_not_covered': missing, **r})
    # (b) slices
    sl = [float(x) for x in a.slices.split(',')]
    for lo, hi in zip(sl[:-1], sl[1:]):
        for zc in Z_CEN:
            s = d[np.isclose(d.zb, zc) & (d.logM >= lo) & (d.logM < hi)]
            r = corrected(s, t, zc, np.ones(len(s)), rng, a.nboot)
            if r:
                rows.append({'test': f'{lo:.1f}<logM<{hi:.1f}', 'z': zc, **r})
    res = pd.DataFrame(rows)
    res.to_csv(f'{a.out}.csv', index=False)

    pd.set_option('display.width', 200)
    print("Final amplitudes (Eq. 2, conventional) per test:")
    for test, g in res.groupby('test', sort=False):
        print(f"\n[{test}]")
        print(g[['z', 'N', 'N_eff', 'A1', 'A1_err', 'A2', 'A2_err', 'R12', 'R12_err']].round(3).to_string(index=False))
        print(f"  growth z~0.75 -> mean(z>4): A1 x{growth(g, 'A1'):.2f}, A2 x{growth(g, 'A2'):.2f}")

    # (c) regression on cell medians
    cells = []
    mb = np.arange(9.0, 11.51, 0.5)
    for zc in Z_CEN:
        for lo, hi in zip(mb[:-1], mb[1:]):
            s = d[np.isclose(d.zb, zc) & (d.logM >= lo) & (d.logM < hi)]
            r = corrected(s, t, zc, np.ones(len(s)), rng, 100) if len(s) >= 20 else None
            if r:
                cells.append({'z': zc, 'logM': s.logM.median(), **r})
    c = pd.DataFrame(cells)
    c.to_csv(f'{a.out}_cells.csv', index=False)
    fits = []
    for k in ('A1', 'A2', 'R12'):
        ok = c[k] > 0
        y = np.log10(c.loc[ok, k])
        sig = (c.loc[ok, f'{k}_err'] / c.loc[ok, k] / np.log(10)).clip(lower=0.01)
        X1 = np.column_stack([np.ones(ok.sum()), np.log10(1 + c.loc[ok, 'z'])])
        X2 = np.column_stack([X1, c.loc[ok, 'logM'] - 10])
        for name, X in (('z only', X1), ('z + mass', X2)):
            W = 1 / sig ** 2
            cov = np.linalg.inv(X.T @ (X * W.to_numpy()[:, None]))
            beta = cov @ (X.T @ (W * y))
            fits.append({'quantity': k, 'model': name, 'b_log1pz': beta[1],
                         'b_err': np.sqrt(cov[1, 1]),
                         'c_logM': beta[2] if X.shape[1] > 2 else np.nan,
                         'c_err': np.sqrt(cov[2, 2]) if X.shape[1] > 2 else np.nan,
                         'cells': int(ok.sum())})
    f = pd.DataFrame(fits)
    f.to_csv(f'{a.out}_regression.csv', index=False)
    print("\n(c) log10 A = a + b log10(1+z) [+ c (logM-10)], cell medians (N>=20):")
    print(f.round(3).to_string(index=False))
    print("\nNote: floors and recovery fractions are those of the whole bin (approximate for "
          "mass-selected subsets).")


if __name__ == '__main__':
    main()
