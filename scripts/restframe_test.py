#!/usr/bin/env python3
"""
restframe_test.py — A_m evolution at FIXED rest-frame wavelength.

F444W samples ~2.5 um at z~0.75 but ~0.7 um at z~5, so a trend measured in
one filter mixes structural evolution with the change of rest-frame tracer.
Here every galaxy is measured (run_corrected_am.py, variant V6) in several
NIRCam filters; for a target rest wavelength lambda0 each galaxy uses the
filter whose lambda_pivot/(1+z) is closest to lambda0 (within --tol dex).

Per (z-bin, filter) the median A_m is corrected with that filter's own
injection run (same PSF, pixel scale, noise ACF and S/N):
    floor-corrected   A' = sqrt(A^2 - floor^2)              (ZERO scenario)
    recovery-corrected A'' = A' / rec,  rec = recovered/true (MATCHED if
                                               present, floor-corrected)
and the filters contributing to a bin are combined with N weights. Floor and
recovery use the same centring (--centering, default catalog) and scenario as
make_table1.py, so the "F444W only" row reproduces Table 1.
A1/A2 here is the ratio of the corrected median amplitudes.
All values in the conventional normalisation (x2).

Also printed: the per-galaxy ratio A_m(F_short)/A_m(F444W) at fixed z, i.e.
how much the amplitudes depend on wavelength alone.

Usage:
    python3 restframe_test.py \
        --objects F444W=../corrected_disk.csv,F277W=../corrected_disk_f277w.csv,F150W=../corrected_disk_f150w.csv \
        --injections F444W=../injection_v6/injection_summary.csv,F277W=../inj_f277w/injection_summary.csv,F150W=../inj_f150w/injection_summary.csv \
        --targets 0.8,1.1 --out ../restframe_disk
"""

import argparse

import numpy as np
import pandas as pd

Z_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
Z_CEN = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50]
PIVOT = {'F090W': 0.902, 'F115W': 1.154, 'F150W': 1.501, 'F200W': 1.990,
         'F277W': 2.786, 'F356W': 3.563, 'F410M': 4.092, 'F444W': 4.421}


def parse_map(s):
    return {k.strip().upper(): v.strip() for k, v in (x.split('=', 1) for x in s.split(','))}


def boot(x, rng, nb=1000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan, len(x)
    return np.median(x), np.std(np.median(rng.choice(x, (nb, len(x))), axis=1)), len(x)


def corrections(inj_path, centering='catalog'):
    """Per z-bin floor and recovery factor for A1, A2, A3 from an injection run.
    Floor and recovery use the same centring scheme and scenarios as Table 1
    (make_table1.py), so that the single-filter F444W row reproduces Table 1."""
    inj = pd.read_csv(inj_path)
    zero = inj[(inj.scenario == 'zero') & (inj.centering == centering)].set_index('zbin')
    scen = 'matched' if (inj.scenario == 'matched').any() else 'constant'
    cons = inj[(inj.scenario == scen) & (inj.centering == centering)].set_index('zbin')
    if zero.empty or cons.empty:
        raise SystemExit(f"{inj_path}: no zero/{scen} rows with centering={centering}")
    out = {}
    for zc in Z_CEN:
        if zc not in zero.index:
            continue
        row = {}
        for k in ('A1', 'A2', 'A3'):
            f = zero.loc[zc, f'{k}_med']
            rec = np.nan
            truth = f'{k}_true_med'
            if zc in cons.index and truth in cons.columns and cons.loc[zc, truth] > 0:
                m = cons.loc[zc, f'{k}_med']
                rec = np.sqrt(max(m * m - f * f, 0)) / cons.loc[zc, truth]
            row[k] = (f, rec)
        out[zc] = row
    return out


def corrected(med, err, f, rec):
    if not np.isfinite(med) or med <= f:
        return np.nan, np.nan
    c = np.sqrt(med * med - f * f)
    ce = err * med / c
    if np.isfinite(rec) and rec > 0:
        c, ce = c / rec, ce / rec
    return 2 * c, 2 * ce


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--objects', required=True, help='FILTER=per-object CSV,...')
    ap.add_argument('--injections', help='FILTER=injection_summary.csv,... (same filters)')
    ap.add_argument('--targets', default='0.8,1.1', help='rest-frame wavelengths [um]')
    ap.add_argument('--tol', type=float, default=0.12, help='max |log(lambda_rest/lambda0)| [dex]')
    ap.add_argument('--variant', default='V6')
    ap.add_argument('--ref', default='F444W', help='single-filter reference')
    ap.add_argument('--centering', default='catalog',
                    help='injection centring scheme for floor and recovery (as Table 1)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    objs = parse_map(args.objects)
    injs = parse_map(args.injections) if args.injections else {}
    unknown = [f for f in objs if f not in PIVOT]
    if unknown:
        raise SystemExit(f"Unknown filters {unknown}; known: {list(PIVOT)}")
    corr = {f: corrections(p, args.centering) for f, p in injs.items()}
    v = args.variant
    rng = np.random.default_rng(0)

    tabs = {}
    for f, path in objs.items():
        t = pd.read_csv(path)[['idx', 'z', f'A1_{v}', f'A2_{v}', f'A3_{v}']]
        tabs[f] = t.rename(columns={f'A1_{v}': f'A1_{f}', f'A2_{v}': f'A2_{f}',
                                    f'A3_{v}': f'A3_{f}'})
    d = None
    for f, t in tabs.items():
        d = t if d is None else d.merge(t, on=['idx', 'z'], how='outer')
    d['zb'] = pd.cut(d.z, Z_EDGES, labels=Z_CEN)
    print("Galaxies with a measurement per filter: " +
          ', '.join(f"{f}: {d[f'A1_{f}'].notna().sum()}" for f in objs))

    def bin_values(sel_filter, label):
        """sel_filter: Series of chosen filter per galaxy (NaN = none)."""
        rows = []
        for zc in Z_CEN:
            row = dict(z=zc, set=label)
            parts = {k: [] for k in ('A1', 'A2', 'A3')}
            used = []
            for f in objs:
                s = d[(d.zb == zc) & (sel_filter == f)]
                if s.empty:
                    continue
                used.append(f'{f}:{len(s)}')
                for k in ('A1', 'A2', 'A3'):
                    m, e, n = boot(s[f'{k}_{f}'], rng)
                    fl, rec = corr.get(f, {}).get(zc, {}).get(k, (0.0, np.nan))
                    c, ce = corrected(m, e, fl, rec)
                    if np.isfinite(c):
                        parts[k].append((c, ce, n))
            row['filters'] = ' '.join(used)
            for k in ('A1', 'A2', 'A3'):
                if parts[k]:
                    w = np.array([p[2] for p in parts[k]], float)
                    row[k] = np.sum(w * [p[0] for p in parts[k]]) / w.sum()
                    row[f'{k}_err'] = np.sqrt(np.sum((w * [p[1] for p in parts[k]]) ** 2)) / w.sum()
                else:
                    row[k], row[f'{k}_err'] = np.nan, np.nan
            row['R12'] = row['A1'] / row['A2']
            row['R12_err'] = row['R12'] * np.hypot(row['A1_err'] / row['A1'],
                                                   row['A2_err'] / row['A2'])
            rows.append(row)
        return pd.DataFrame(rows)

    results = []
    # single filter reference (F444W at all z)
    # a galaxy counts if any amplitude is valid; each A_m median then uses its own valid
    # values, exactly as make_table1.py, so this row reproduces Table 1
    has = {f: d[[f'A1_{f}', f'A2_{f}']].notna().any(axis=1) for f in objs}
    ref = pd.Series(np.where(has[args.ref], args.ref, None), index=d.index)
    r = bin_values(ref, f'{args.ref} only')
    r['lambda_rest'] = PIVOT[args.ref] / (1 + r.z)
    results.append(r)
    # fixed rest-frame wavelength
    for lam in [float(x) for x in args.targets.split(',')]:
        best = pd.Series([None] * len(d), index=d.index, dtype=object)
        bestd = pd.Series(np.inf, index=d.index)
        for f in objs:
            dist = np.abs(np.log10(PIVOT[f] / (1 + d.z) / lam))
            ok = has[f] & (dist <= args.tol) & (dist < bestd)
            best[ok], bestd[ok] = f, dist[ok]
        r = bin_values(best, f'{lam:.2f} um')
        r['lambda_rest'] = lam
        results.append(r)
    res = pd.concat(results, ignore_index=True)
    res.to_csv(f'{args.out}.csv', index=False)

    pd.set_option('display.width', 220)
    for label, g in res.groupby('set', sort=False):
        print(f"\n=== {label} (corrected, conventional normalisation) ===")
        print(g[['z', 'filters', 'A1', 'A1_err', 'A2', 'A2_err', 'R12', 'R12_err']]
              .round(3).to_string(index=False))
        ok = g.dropna(subset=['A1'])
        if len(ok) >= 2:
            print(f"  A1: x{ok.A1.iloc[-1] / ok.A1.iloc[0]:.2f}  A2: x{ok.A2.iloc[-1] / ok.A2.iloc[0]:.2f}"
                  f"  between z={ok.z.iloc[0]} and z={ok.z.iloc[-1]}")

    # wavelength dependence at fixed z
    print(f"\n=== Same galaxies, other filter / {args.ref} (median per-galaxy ratio) ===")
    for f in objs:
        if f == args.ref:
            continue
        line = []
        for zc in Z_CEN:
            s = d[(d.zb == zc)]
            rr = (s[f'A1_{f}'] / s[f'A1_{args.ref}']).replace([np.inf, -np.inf], np.nan).dropna()
            line.append(f"{rr.median():5.2f}" if len(rr) >= 10 else '   - ')
        print(f"  A1 {f}/{args.ref}: " + ' '.join(line))
        line = []
        for zc in Z_CEN:
            s = d[(d.zb == zc)]
            rr = (s[f'A2_{f}'] / s[f'A2_{args.ref}']).replace([np.inf, -np.inf], np.nan).dropna()
            line.append(f"{rr.median():5.2f}" if len(rr) >= 10 else '   - ')
        print(f"  A2 {f}/{args.ref}: " + ' '.join(line))
    print("  (z bins: " + ' '.join(f"{z:5.2f}" for z in Z_CEN) + ")")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for label, g in res.groupby('set', sort=False):
        ls = '--' if 'only' in label else '-'
        for k, ax in zip(('A1', 'A2', 'R12'), axs):
            ax.errorbar(g.z, g[k], g[f'{k}_err'], fmt='o' + ls, capsize=3, label=label)
    for k, ax in zip(('A1', 'A2', 'R12'), axs):
        ax.set(xlabel='Redshift $z$', ylabel={'A1': '$A_1$', 'A2': '$A_2$',
                                             'R12': '$A_1/A_2$'}[k])
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{args.out}.png', dpi=180)
    print(f"\nSaved {args.out}.csv, {args.out}.png")


if __name__ == '__main__':
    main()
