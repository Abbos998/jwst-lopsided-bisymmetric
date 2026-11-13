#!/usr/bin/env python3
"""
aggregate_realsky.py -- combine several seeds of real_sky_injection.py into
mean +/- standard deviation. Reads <prefix>_s*_bins.csv and <prefix>_s*_tercile.csv.

Usage:
  python3 aggregate_realsky.py --prefix $ROOT/referee_tests/realsky --out $ROOT/referee_tests/realsky_seeds
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HI = (4.5, 5.5)


def ms(series):
    v = np.asarray(series, float)
    v = v[np.isfinite(v)]
    return (np.mean(v), np.std(v)) if len(v) else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True, help='e.g. .../realsky  (reads .../realsky_s*_bins.csv)')
    ap.add_argument('--synthetic', help='synthetic injection per_object.csv (e.g. inj_f444w_m/injection_per_object.csv) '
                    'to compare the floor in the SAME S/N bands at z>4')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    binf = sorted(glob.glob(f'{a.prefix}_s*_bins.csv'))
    tercf = sorted(glob.glob(f'{a.prefix}_s*_tercile.csv'))
    if not binf:
        raise SystemExit(f'no files match {a.prefix}_s*_bins.csv')
    bins = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(binf)], ignore_index=True)
    K = bins.seed.nunique()
    L = [f'{K} seeds: {", ".join(os.path.basename(f) for f in binf)}']

    # per-bin mean +/- sd
    cols = ['floor1_real', 'floor2_real', 'frec1_real', 'frec2_real', 'A1_real', 'A2_real']
    rows = []
    for z, g in bins.groupby('z'):
        r = {'z': z, 'floor1_syn': g.floor1_syn.iloc[0], 'floor2_syn': g.floor2_syn.iloc[0]}
        for c in cols:
            m, s = ms(g[c])
            r[c] = f'{m:.3f}+-{s:.3f}'
        rows.append(r)
    perbin = pd.DataFrame(rows)
    L += ['', '=== per-bin real-sky floor, recovered fraction and corrected amplitude (mean +- sd over seeds)',
          perbin.to_string(index=False)]

    # growth per seed -> mean +/- sd
    def grow(df, col):
        lo = df[np.isclose(df.z, 0.75)][col]
        hi = df[df.z.isin(HI)][col]
        return (hi.mean() / lo.values[0]) if len(lo) and lo.values[0] > 0 else np.nan
    g1 = [grow(bins[bins.seed == s], 'A1_real') for s in range(K)]
    g2 = [grow(bins[bins.seed == s], 'A2_real') for s in range(K)]
    syn1 = grow(bins[bins.seed == 0], 'A1_syn'); syn2 = grow(bins[bins.seed == 0], 'A2_syn')
    L += ['', '=== growth factors z~0.75 -> z>4',
          f'  A1: synthetic {syn1:.2f};  real-sky {np.nanmean(g1):.2f} +- {np.nanstd(g1):.2f}',
          f'  A2: synthetic {syn2:.2f};  real-sky {np.nanmean(g2):.2f} +- {np.nanstd(g2):.2f}']

    # floor ratio real/synthetic, mean over seeds, per bin
    bins['rt1'] = bins.floor1_real / bins.floor1_syn
    bins['rt2'] = bins.floor2_real / bins.floor2_syn
    rows = []
    for z, g in bins.groupby('z'):
        m1, s1 = ms(g.rt1); m2, s2 = ms(g.rt2)
        rows.append({'z': z, 'floor1_real/syn': f'{m1:.2f}+-{s1:.2f}', 'floor2_real/syn': f'{m2:.2f}+-{s2:.2f}'})
    L += ['', '=== real-sky / synthetic floor ratio (mean +- sd over seeds)',
          pd.DataFrame(rows).to_string(index=False)]

    # tercile floors
    if tercf:
        terc = pd.concat([pd.read_csv(f) for f in tercf], ignore_index=True)
        rows = []
        for tc in ('low', 'mid', 'high'):
            g = terc[terc.tercile == tc]
            m1, s1 = ms(g.floor1_real); m2, s2 = ms(g.floor2_real); sn, _ = ms(g.snr_med)
            rows.append({'tercile': tc, 'snr_med': round(sn, 0),
                         'floor1_real': f'{m1:.3f}+-{s1:.3f}', 'floor2_real': f'{m2:.3f}+-{s2:.3f}'})
        L += ['', '=== real-sky floor by S/N tercile at z>4 (mean +- sd over seeds; conventional)',
              pd.DataFrame(rows).to_string(index=False),
              f'  (synthetic floor at z~4.5: A1 {bins[np.isclose(bins.z,4.5)].floor1_syn.iloc[0]:.3f}, '
              f'A2 {bins[np.isclose(bins.z,4.5)].floor2_syn.iloc[0]:.3f})']

    # same-S/N-band floor comparison: real-sky (per-object, all seeds) vs synthetic
    if a.synthetic and os.path.exists(a.synthetic):
        rsf = sorted(glob.glob(f'{a.prefix}_s*_per_object.csv'))
        if rsf:
            real = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(rsf)], ignore_index=True)
            real = real[(real.scenario == 'zero') & real.zbin.isin(HI)]
            syn = pd.read_csv(a.synthetic)
            syn = syn[syn.scenario == 'zero']
            if 'centering' in syn.columns:
                syn = syn[syn.centering == 'catalog']
            syn = syn[syn.zbin.isin(HI)] if 'zbin' in syn.columns else syn
            # common S/N band edges from the union of the two z>4 samples
            allsnr = np.concatenate([real.snr.values, syn.snr.values])
            e = np.quantile(allsnr[np.isfinite(allsnr)], [1 / 3, 2 / 3])
            band = lambda s: np.where(s < e[0], 'low', np.where(s < e[1], 'mid', 'high'))
            real['band'] = band(real.snr.values)
            syn['band'] = band(syn.snr.values)
            rows = []
            for b in ('low', 'mid', 'high'):
                rb, sb = real[real.band == b], syn[syn.band == b]
                r = {'S/N band': b, 'S/N range': f'{sb.snr.min():.0f}-{sb.snr.max():.0f}' if len(sb) else '-'}
                for k in (1, 2):
                    # real-sky: mean +/- sd of the per-seed median floor (conventional)
                    perseed = [2 * np.nanmedian(rb[rb.seed == s][f'A{k}']) for s in rb.seed.unique()]
                    msr, ssr = ms(perseed)
                    syn_fl = 2 * np.nanmedian(sb[f'A{k}']) if len(sb) else np.nan
                    r[f'floor{k}_syn'] = round(syn_fl, 3)
                    r[f'floor{k}_real'] = f'{msr:.3f}+-{ssr:.3f}'
                    r[f'ratio{k}'] = round(msr / syn_fl, 2) if syn_fl and syn_fl > 0 else np.nan
                rows.append(r)
            L += ['', '=== floor at z>4 in the SAME S/N bands: real-sky vs synthetic (conventional)',
                  pd.DataFrame(rows).to_string(index=False),
                  '  reading: ratio ~constant across bands => the floor underestimate is uniform in S/N,',
                  '  so the low-S/N excess is not mainly a sky noise floor; ratio rising toward low S/N',
                  '  => the real sky does add a low-S/N floor the synthetic injections miss.']
        else:
            L += ['', f'(no {a.prefix}_s*_per_object.csv found for the same-S/N-band comparison)']

    txt = '\n'.join(L)
    print(txt)
    with open(f'{a.out}_summary.txt', 'w') as f:
        f.write(txt + '\n')
    perbin.to_csv(f'{a.out}_perbin.csv', index=False)


if __name__ == '__main__':
    main()
