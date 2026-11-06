#!/usr/bin/env python3
"""
fix_sfr.py — star-forming vs quiescent A_m with SFRs matched by ROW INDEX.

The original Fig. 9 merged the CEERS-only SFR catalogue on `id` alone, so
~88% of the sample received another galaxy's SFR. Here SFR and mass are read
from the same multi-field catalogue the A_m pipeline used
(full-good_morpho-phot.fits.gz, HDU 1) at row `idx`, so the match is exact.
The match is verified by comparing the catalogue mass with logM in the
sample file.

Main sequence: Speagle et al. (2014), eq. 28, Chabrier IMF:
    log SFR_MS = (0.84 - 0.026 t) log M* - (6.51 - 0.11 t),  t = age [Gyr]
(flat LCDM, H0=70, Om=0.3, as in the paper).

Usage:
    python3 fix_sfr.py \
        --morpho /scratch/pulsating_disk/dja_analysis/full-good_morpho-phot.fits.gz \
        --objects ../corrected_disk.csv --sample /tmp/dja_real.csv \
        --injection ../injection_v6/injection_summary.csv --out ../sfr_split
"""

import argparse
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

Z_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
Z_CEN = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50]


def age_gyr(z, h0=70.0, om=0.3):
    """Age of a flat LCDM universe at redshift z [Gyr] (numerical integral)."""
    z = np.atleast_1d(np.asarray(z, float))
    a = np.linspace(1e-6, 1.0, 20001)
    integrand = 1.0 / (a * np.sqrt(om / a ** 3 + (1 - om)))
    cum = np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(a))])
    hubble_time = 977.792 / h0  # Gyr
    return np.interp(1.0 / (1.0 + z), a, cum) * hubble_time


def speagle_ms(logm, z):
    t = age_gyr(z)
    return (0.84 - 0.026 * t) * logm - (6.51 - 0.11 * t)


def boot(x, rng, nb=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan, len(x)
    return np.median(x), np.std(np.median(rng.choice(x, (nb, len(x))), axis=1)), len(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--morpho', required=True)
    ap.add_argument('--objects', required=True, help='per-object CSV of run_corrected_am.py')
    ap.add_argument('--sample', required=True, help='sample CSV with idx, logM (e.g. dja_real.csv)')
    ap.add_argument('--injection', help='injection_summary.csv of the same estimator (floors)')
    ap.add_argument('--variant', default='V6')
    ap.add_argument('--sfr_col', default='sfr')
    ap.add_argument('--mass_col', default='mass')
    ap.add_argument('--sf_cut', type=float, default=-0.3, help='SF: dMS > sf_cut [dex]')
    ap.add_argument('--q_cut', type=float, default=-1.0, help='Q: dMS < q_cut [dex]')
    ap.add_argument('--min_n', type=int, default=15, help='bins with fewer objects are flagged')
    ap.add_argument('--snr_csv', help='measure_snr.py output (adds snr_ap per galaxy)')
    ap.add_argument('--snr_min', type=float, default=0.0,
                    help='keep only galaxies with snr_ap above this (needs --snr_csv)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with fits.open(args.morpho) as h:
        cols = h[1].columns.names
        if args.sfr_col not in cols or args.mass_col not in cols:
            print("HDU1 columns with sfr/mass:",
                  [c for c in cols if 'sfr' in c.lower() or 'mass' in c.lower()])
            sys.exit("Pass --sfr_col / --mass_col.")
        sfr_all = np.asarray(h[1].data[args.sfr_col], float)
        mass_all = np.asarray(h[1].data[args.mass_col], float)

    obj = pd.read_csv(args.objects)
    smp = pd.read_csv(args.sample)[['idx', 'logM']]
    d = obj.merge(smp, on='idx', how='left', validate='one_to_one')
    d['sfr'] = sfr_all[d.idx.values]
    with np.errstate(all='ignore'):
        d['logM_cat'] = np.log10(mass_all[d.idx.values])
    dm = (d.logM - d.logM_cat).abs()
    print(f"Index-match check: median |logM(sample) - logM(catalogue[idx])| = "
          f"{dm.median():.4f} dex, >0.01 dex: {(dm > 0.01).mean() * 100:.1f}%")
    if dm.median() > 0.01:
        sys.exit("Masses do not match: the catalogue rows are not aligned with idx.")

    good = np.isfinite(d.sfr) & (d.sfr > 0) & np.isfinite(d.logM)
    print(f"Valid SFR: {good.sum()} / {len(d)}  (SFR <= 0 or NaN dropped, not clipped)")
    d = d[good].copy()
    d['logSFR'] = np.log10(d.sfr)
    d['dMS'] = d.logSFR - speagle_ms(d.logM.values, d.z.values)
    d['cls'] = np.where(d.dMS > args.sf_cut, 'SF', np.where(d.dMS < args.q_cut, 'Q', 'GV'))
    if args.snr_csv:
        snr = pd.read_csv(args.snr_csv)[['idx', 'snr_ap']]
        d = d.merge(snr, on='idx', how='left')
        zb = pd.cut(d.z, Z_EDGES, labels=Z_CEN)
        print("\nMedian aperture S/N by class (SF / GV / Q) per z bin:")
        for zc in Z_CEN:
            s_ = d[zb == zc]
            print(f"  z={zc:4.2f}: " + ' / '.join(
                f"{s_[s_.cls == c].snr_ap.median():6.0f}" for c in ('SF', 'GV', 'Q')))
        if args.snr_min > 0:
            n0 = len(d)
            d = d[d.snr_ap > args.snr_min].copy()
            print(f"S/N > {args.snr_min:g}: kept {len(d)} of {n0}")
    d.to_csv(f'{args.out}_objects.csv', index=False)

    floors = None
    if args.injection:
        inj = pd.read_csv(args.injection)
        floors = inj[(inj.scenario == 'zero') & (inj.centering == 'catalog')].set_index('zbin')

    v = args.variant
    d['zb'] = pd.cut(d.z, Z_EDGES, labels=Z_CEN)
    rng = np.random.default_rng(0)
    rows = []
    for zc in Z_CEN:
        row = dict(z=zc)
        for c in ('SF', 'Q'):
            s = d[(d.zb == zc) & (d.cls == c)]
            m, e, n = boot(s[f'A1_{v}'], rng)
            f = floors.loc[zc, 'A1_med'] if floors is not None and zc in floors.index else 0.0
            if np.isfinite(m) and m > f:
                mc = np.sqrt(m * m - f * f)
                ec = e * m / mc
            else:
                mc, ec = np.nan, np.nan
            row.update({f'N_{c}': n, f'A1_{c}': 2 * mc, f'A1_{c}_err': 2 * ec,
                        f'dMS_{c}': s.dMS.median()})
        row['ratio'] = row['A1_SF'] / row['A1_Q']
        row['ratio_err'] = row['ratio'] * np.hypot(row['A1_SF_err'] / row['A1_SF'],
                                                   row['A1_Q_err'] / row['A1_Q'])
        row['flag'] = 'low N' if min(row['N_SF'], row['N_Q']) < args.min_n else ''
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(f'{args.out}.csv', index=False)

    print(f"\nClass fractions (all z): " +
          ', '.join(f"{c}: {(d.cls == c).mean() * 100:.0f}%" for c in ('SF', 'GV', 'Q')))
    print(f"\nA1 ({v}, floor-corrected, conventional normalisation) by class:")
    print(t.round(3).to_string(index=False))
    ok = t[(t.flag == '') & np.isfinite(t.ratio)]
    if len(ok):
        w = 1 / ok.ratio_err ** 2
        mean = np.sum(w * ok.ratio) / w.sum()
        err = 1 / np.sqrt(w.sum())
        chi2 = np.sum(w * (ok.ratio - 1) ** 2)
        print(f"\nWeighted mean A1(SF)/A1(Q) over {len(ok)} well-populated bins: "
              f"{mean:.3f} +- {err:.3f};  chi2 vs 1 = {chi2:.1f} for {len(ok)} dof")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for c, mk in (('SF', 'o'), ('Q', 's')):
        axs[0].errorbar(t.z, t[f'A1_{c}'], t[f'A1_{c}_err'], fmt=mk + '-', capsize=3,
                        label={'SF': 'star-forming', 'Q': 'quiescent'}[c])
    axs[0].set(xlabel='Redshift $z$', ylabel='median $A_1$ (corrected)')
    good_t = t[t.flag == '']
    axs[1].errorbar(good_t.z, good_t.ratio, good_t.ratio_err, fmt='D', capsize=3)
    low = t[t.flag != '']
    axs[1].errorbar(low.z, low.ratio, low.ratio_err, fmt='D', mfc='none', capsize=3,
                    label=f'N < {args.min_n}')
    axs[1].axhline(1, color='gray', ls='--')
    axs[1].set(xlabel='Redshift $z$', ylabel='$A_1^{\\rm SF}/A_1^{\\rm Q}$')
    for a in axs:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{args.out}.png', dpi=180)
    print(f"\nSaved {args.out}.csv, {args.out}.png")


if __name__ == '__main__':
    main()
