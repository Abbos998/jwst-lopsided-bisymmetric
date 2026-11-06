#!/usr/bin/env python3
"""
photoz_mc_test.py -- effect of photometric-redshift uncertainties on the
binned amplitudes, growth factors and A1/A2 (Monte Carlo).

For each realisation every galaxy receives a new redshift drawn from its
photometric-redshift uncertainty (split normal from the catalogue 16th/50th/
84th percentiles; galaxies with a spectroscopic redshift keep it), the
galaxies are re-binned, the bin medians of the measured amplitudes are
recomputed and corrected with the calibration of the bin they now fall in
(noise floor, recovered fraction, ratio bias of Table 1). Optionally a
fraction of catastrophic outliers is reassigned to a uniform redshift.

Without perturbation the script must reproduce Table 1 exactly (checked).

Inputs
  --galaxies  data/intermediate/corrected_disk_f444w.csv (idx, z, A1_V6, A2_V6)
  --table1    data/table1_disks_F444W.csv
  --zcat      DJA morpho-phot FITS (rows indexed by idx) with redshift percentiles
  --z_lo_col / --z_hi_col   16th / 84th percentile columns (auto-detected)
  --zspec_col optional spectroscopic-redshift column (auto-detected)
List candidate columns with:  photoz_mc_test.py --list_z_columns <FITS>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

EDGES = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
ZC = np.array([0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50])
LO_HI = [('z160', 'z840'), ('z16', 'z84'), ('z_phot_l68', 'z_phot_u68'),
         ('zphot_l68', 'zphot_u68'), ('z_l68', 'z_u68'), ('l68', 'u68'),
         ('z_phot_16', 'z_phot_84'), ('zphot_16', 'zphot_84')]
SPEC = ['z_spec', 'zspec', 'z_spec_best', 'spec_z']


def read_fits_cols(path, cols=None):
    from astropy.io import fits
    with fits.open(path, memmap=True) as h:
        d = h[1].data
        names = d.columns.names
        if cols is None:
            return names
        return {c: np.asarray(d[c], float) for c in cols}


def binned(z, a1, a2, t1):
    """Corrected per-bin values as in make_table1 (left-closed bins)."""
    ib = np.digitize(z, EDGES) - 1
    out = []
    for k, zc in enumerate(ZC):
        sel = ib == k
        n = int(sel.sum())
        r = dict(zbin=zc, N=n)
        if n < 3:
            out.append(r)
            continue
        for m, a in ((1, a1), (2, a2)):
            med = np.median(a[sel])
            f = t1.loc[zc, f'A{m}_floor_pipe']
            rec = t1.loc[zc, f'A{m}_recovery']
            r[f'A{m}_meas'] = med
            r[f'A{m}_final'] = 2 * np.sqrt(max(med * med - f * f, 0)) / rec
        r['R12_obs'] = np.median(a1[sel] / a2[sel])
        r['R12_corr'] = r['R12_obs'] / t1.loc[zc, 'A1A2_bias']
        out.append(r)
    return pd.DataFrame(out).set_index('zbin')


def summary(b):
    hi = [z for z in (4.5, 5.5) if z in b.index and np.isfinite(b.loc[z].get('A1_final', np.nan))]
    g = {f'A{m}_growth': b.loc[hi, f'A{m}_final'].mean() / b.loc[0.75, f'A{m}_final'] for m in (1, 2)}
    g['R12_lo'] = b.loc[b.index < 3.5, 'R12_corr'].mean()
    g['R12_hi'] = b.loc[b.index > 3.5, 'R12_corr'].mean()
    return g


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--list_z_columns':
        names = read_fits_cols(sys.argv[2])
        print([n for n in names if n.lower().startswith('z') or '68' in n or 'spec' in n.lower()][:80])
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxies', required=True)
    ap.add_argument('--table1', required=True)
    ap.add_argument('--zcat', required=True)
    ap.add_argument('--z_lo_col')
    ap.add_argument('--z_hi_col')
    ap.add_argument('--zspec_col')
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--n_mc', type=int, default=500)
    ap.add_argument('--outlier_frac', type=float, default=0.05,
                    help='fraction of catastrophic outliers in the second scenario')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    g = pd.read_csv(a.galaxies).dropna(subset=['A1_V6', 'A2_V6'])
    t1 = pd.read_csv(a.table1).set_index('z')
    z0 = g.z.values
    a1, a2 = g.A1_V6.values, g.A2_V6.values

    # --- check: unperturbed redshifts reproduce Table 1
    b0 = binned(z0, a1, a2, t1)
    dev = max(abs(b0.loc[zc, f'A{m}_final'] - t1.loc[zc, f'A{m}_final_conv']) for zc in ZC for m in (1, 2))
    print(f'{len(g)} galaxies; unperturbed binning reproduces Table 1 to {dev:.1e} '
          f'(N per bin {list(b0.N)})')
    if dev > 1e-3:
        sys.exit('STOP: the unperturbed calculation does not reproduce Table 1')
    nominal = summary(b0)

    # --- redshift uncertainties from the catalogue
    names = read_fits_cols(a.zcat)
    lo, hi = a.z_lo_col, a.z_hi_col
    if not (lo and hi):
        for c1, c2 in LO_HI:
            if c1 in names and c2 in names:
                lo, hi = c1, c2
                break
    if not (lo and hi):
        sys.exit(f'no 16/84 percentile columns found; pass --z_lo_col/--z_hi_col. '
                 f'Candidates: {[n for n in names if "z" in n.lower()][:60]}')
    spec = a.zspec_col or next((c for c in SPEC if c in names), None)
    cols = read_fits_cols(a.zcat, [lo, hi] + ([spec] if spec else []))
    idx = g[a.idx_col].astype(int).values
    zlo, zhi = cols[lo][idx], cols[hi][idx]
    sig_lo = np.clip(z0 - zlo, 0, None)
    sig_hi = np.clip(zhi - z0, 0, None)
    bad = ~np.isfinite(sig_lo) | ~np.isfinite(sig_hi) | (zlo <= 0) | (zhi <= 0)
    sig_lo[bad] = sig_hi[bad] = 0.0
    is_spec = np.zeros(len(g), bool)
    if spec:
        zs = cols[spec][idx]
        is_spec = np.isfinite(zs) & (zs > 0)
        sig_lo[is_spec] = sig_hi[is_spec] = 0.0
    frac_in = np.mean((zlo <= z0) & (z0 <= zhi))
    rel = (zhi - zlo) / 2 / (1 + z0)
    print(f'columns {lo}/{hi}' + (f', spec-z column {spec} ({is_spec.sum()} galaxies kept fixed)' if spec else '')
          + f'; adopted z inside [{lo},{hi}] for {100 * frac_in:.0f}% of galaxies; '
          f'median sigma_z/(1+z) = {np.nanmedian(rel[~bad]):.3f}; {bad.sum()} galaxies without percentiles kept fixed')

    rng = np.random.default_rng(a.seed)
    rows, perbin = [], []
    for scen, fout in (('split-normal', 0.0), (f'split-normal + {100 * a.outlier_frac:.0f}% outliers', a.outlier_frac)):
        for k in range(a.n_mc):
            u = rng.standard_normal(len(z0))
            z = z0 + np.where(u < 0, u * sig_lo, u * sig_hi)
            if fout > 0:
                o = (rng.random(len(z0)) < fout) & ~is_spec
                z[o] = rng.uniform(EDGES[0], EDGES[-1], o.sum())
            b = binned(z, a1, a2, t1)
            s = summary(b)
            s.update(scenario=scen, mc=k)
            rows.append(s)
            bb = b.reset_index()
            bb['scenario'] = scen
            perbin.append(bb)
    R = pd.DataFrame(rows)
    P = pd.concat(perbin)
    R.to_csv(os.path.join(a.out, 'photoz_mc_realisations.csv'), index=False)
    pb = P.groupby(['scenario', 'zbin']).agg(N_mean=('N', 'mean'), N_sd=('N', 'std'),
                                              A1_mean=('A1_final', 'mean'), A1_sd=('A1_final', 'std'),
                                              A2_mean=('A2_final', 'mean'), A2_sd=('A2_final', 'std'))
    pb.to_csv(os.path.join(a.out, 'photoz_mc_perbin.csv'))
    lines = [f'Nominal (Table 1, same definitions): A1 growth {nominal["A1_growth"]:.3f}, '
             f'A2 growth {nominal["A2_growth"]:.3f}, <A1/A2> z<3.5 {nominal["R12_lo"]:.3f}, z>3.5 {nominal["R12_hi"]:.3f}',
             '(<A1/A2> are simple means of the corrected bins, as in the robustness table)', '']
    for scen, r in R.groupby('scenario', sort=False):
        lines.append(f'{scen} ({len(r)} realisations):')
        for k in ('A1_growth', 'A2_growth', 'R12_lo', 'R12_hi'):
            v = r[k].values
            lines.append(f'  {k:9s} {np.mean(v):.3f} +- {np.std(v, ddof=1):.3f}  '
                         f'[2.5-97.5%: {np.percentile(v, 2.5):.3f}-{np.percentile(v, 97.5):.3f}]')
    lines += ['', 'Per-bin mean +- s.d. over realisations:', pb.round(4).to_string()]
    txt = '\n'.join(lines)
    print(txt)
    open(os.path.join(a.out, 'photoz_mc_summary.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
