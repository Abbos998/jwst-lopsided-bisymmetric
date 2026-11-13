#!/usr/bin/env python3
"""
diagnose_excess.py -- why do low-S/N galaxies in the shallow fields have higher
corrected amplitudes than the injection calibration predicts (Sect. 3.4)?
No images and no new injections; uses only the per-galaxy measurements, the
bin-level calibration of Table 1, the parent catalogue (positions, mass, field)
and the aperture-S/N catalogue. NumPy/pandas only.

Each galaxy is corrected with the bin-level floor and recovered fraction of its
redshift bin (Eq. 2). If the calibration were complete, the corrected amplitude
would not depend on S/N at fixed redshift and mass. The tests ask what the
residual dependence tracks:

 A. residual S/N dependence at fixed mass (pooled z>3.5 and z>4):
    log A_corr ~ 1 + (logM-10) + log10(S/N), and whether the S/N coefficient
    shrinks when crowding proxies (masked-pixel fraction, bright-neighbour
    density) are added -- a mediation test for unmasked neighbours
 B. A1 vs A2 excess by S/N tercile at z>4: if A2 is inflated more than A1 at low
    S/N, part of the excess is real m=2 (clumpy) structure rather than a pure
    noise floor, which is nearly equal for the two modes
 C. field residual: after the S/N and mass trend is removed, how much field
    dependence is left (is it S/N, or the field itself?)

Caveat printed in the output: the neighbour density is built from the parent
catalogue, so it probes bright neighbours only; faint undetected sources, the
leading floor suspect, are not in any catalogue and need the real-sky injection
test.

Usage:
  python3 diagnose_excess.py --galaxies ../data/intermediate/corrected_disk_f444w.csv \
      --table1 ../data/table1_disks_F444W.csv --sample $DATA/dja_real.csv \
      --snrcat $DATA/dja_snr_f444w.csv --out $ROOT/referee_tests/excess
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
FIELD = ['field', 'FIELD', 'file', 'root', 'survey']


def group(f):
    f = str(f).lower()
    return 'PRIMER-UDS' if 'uds' in f else 'PRIMER-COSMOS' if 'cosmos' in f else 'deep (CEERS+GOODS)'


def final(meas, fl, rec):
    return 2 * np.sqrt(np.clip(meas * meas - fl * fl, 0, None)) / rec


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    rx = pd.Series(x[m]).rank().values
    ry = pd.Series(y[m]).rank().values
    return float(np.corrcoef(rx, ry)[0, 1]) if m.sum() > 5 else np.nan


def ols(y, X, names):
    """Weighted-free OLS; returns dict name->(coef, se)."""
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[m], X[m]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return {n: (beta[i], np.sqrt(cov[i, i])) for i, n in enumerate(names)}, int(m.sum())


def neighbour_counts(ra, dec, ra_all, dec_all, radii_arcsec):
    """Count parent-catalogue sources within each radius (minus the galaxy itself).
    NumPy only: the parent catalogue is hashed into sky cells of the largest
    radius, and each galaxy is compared with the sources in its 3x3 cell block."""
    out = {r: np.zeros(len(ra), int) for r in radii_arcsec}
    rmax = max(radii_arcsec) / 3600.0
    cd = np.cos(np.radians(np.median(dec_all)))
    rad2 = {r: (r / 3600.0) ** 2 for r in radii_arcsec}
    # cell index of every catalogue source
    cx_all = np.floor(ra_all * cd / rmax).astype(np.int64)
    cy_all = np.floor(dec_all / rmax).astype(np.int64)
    cells = {}
    for i, (cx, cy) in enumerate(zip(cx_all, cy_all)):
        cells.setdefault((cx, cy), []).append(i)
    cells = {k: np.asarray(v) for k, v in cells.items()}
    gx = np.floor(ra * cd / rmax).astype(np.int64)
    gy = np.floor(dec / rmax).astype(np.int64)
    for i in range(len(ra)):
        idx = [cells[(gx[i] + dx, gy[i] + dy)] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               if (gx[i] + dx, gy[i] + dy) in cells]
        if not idx:
            continue
        j = np.concatenate(idx)
        d2 = ((ra_all[j] - ra[i]) * cd) ** 2 + (dec_all[j] - dec[i]) ** 2
        for r in radii_arcsec:
            out[r][i] = int((d2 < rad2[r]).sum()) - 1   # drop the galaxy itself
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxies', required=True)
    ap.add_argument('--table1', required=True)
    ap.add_argument('--sample', required=True, help='parent dja_real.csv (idx, ra, dec, logM, field)')
    ap.add_argument('--snrcat', required=True)
    ap.add_argument('--field_col')
    ap.add_argument('--mass_col')
    ap.add_argument('--snr_col')
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.set_option('display.width', 250)
    L = []

    t = pd.read_csv(a.table1).set_index('z')
    g = pd.read_csv(a.galaxies).dropna(subset=['A1_V6', 'A2_V6'])
    cat = pd.read_csv(a.sample).rename(columns={a.idx_col: 'idx'})
    mcol = a.mass_col or next((c for c in MASS if c in cat.columns), None)
    fcol = a.field_col or next((c for c in FIELD if c in cat.columns), None)
    if mcol is None or fcol is None or not {'ra', 'dec'} <= set(cat.columns):
        raise SystemExit(f'need mass, field and ra/dec in {a.sample}; have {list(cat.columns)}')
    sn = pd.read_csv(a.snrcat).rename(columns={a.idx_col: 'idx'})
    scol = a.snr_col or next((c for c in ('snr', 'snr_ap', 'snr_aper', 'SNR') if c in sn.columns), None)

    ra_all, dec_all = cat.ra.values, cat.dec.values           # full parent catalogue
    g = g.merge(cat[['idx', 'ra', 'dec', mcol, fcol]].rename(columns={mcol: 'logM', fcol: 'field'}), on='idx', how='left')
    g = g.merge(sn[['idx', scol]].rename(columns={scol: 'snr'}).drop_duplicates('idx'), on='idx', how='left')
    g['zb'] = pd.cut(g.z, EDGES, labels=ZC, right=False).astype(float)
    g = g[g.zb.notna() & g.snr.notna() & (g.snr > 0) & g.logM.notna()].copy()
    g['grp'] = g.field.map(group)

    # bin-level corrected amplitudes (conventional), per galaxy
    for k in (1, 2):
        g[f'C{k}'] = np.nan
    for zc in ZC:
        m = np.isclose(g.zb, zc)
        for k in (1, 2):
            g.loc[m, f'C{k}'] = final(g.loc[m, f'A{k}_V6'], t.loc[zc, f'A{k}_floor_pipe'], t.loc[zc, f'A{k}_recovery'])

    nb = neighbour_counts(g.ra.values, g.dec.values, ra_all, dec_all, (1.5, 3.0))
    g['nbr15'], g['nbr30'] = nb[1.5], nb[3.0]
    L.append(f'{len(g)} valid galaxies with S/N, mass, field and position; S/N column "{scol}".')
    L.append('NOTE: neighbour density is from the parent catalogue (bright neighbours only); '
             'faint undetected sources need the real-sky injection test.')

    # A -------------------------------------------------- residual S/N dependence
    for name, sel in (('z>3.5', g.zb >= 3.75), ('z>4', g.zb.isin(HI))):
        s = g[sel & (g.C1 > 0) & (g.C2 > 0)].copy()
        if len(s) < 40:
            L += ['', f'=== A ({name}): too few galaxies ({len(s)})']
            continue
        s['lsnr'] = np.log10(s.snr)
        s['dm'] = s.logM - 10
        s['fm'] = s.frac_masked - s.frac_masked.median()
        s['lnb'] = np.log10(1 + s.nbr30)
        L += ['', f'=== A residual S/N dependence at fixed mass ({name}, {len(s)} galaxies)']
        for k in (1, 2):
            y = np.log10(s[f'C{k}'].values)
            base, n = ols(y, np.column_stack([np.ones(len(s)), s.dm, s.lsnr]), ['const', 'logM', 'logSN'])
            full, _ = ols(y, np.column_stack([np.ones(len(s)), s.dm, s.lsnr, s.fm, s.lnb]),
                          ['const', 'logM', 'logSN', 'fmask', 'lnbr'])
            rho = spearman(s[f'C{k}'].values, s.snr.values)
            L.append(f'  A{k}: d logA/d logS/N = {base["logSN"][0]:+.3f}+-{base["logSN"][1]:.3f} '
                     f'(mass only); with crowding added {full["logSN"][0]:+.3f}+-{full["logSN"][1]:.3f}; '
                     f'Spearman(A,S/N)={rho:+.2f}')
            L.append(f'       crowding coeffs: frac_masked {full["fmask"][0]:+.3f}+-{full["fmask"][1]:.3f}, '
                     f'log(1+nbr3") {full["lnbr"][0]:+.3f}+-{full["lnbr"][1]:.3f}')
        L.append('  reading: if |d logA/d logS-N| drops toward 0 when crowding is added, '
                 'unmasked bright neighbours explain part of the excess; if it is unchanged, they do not.')

    # B ------------------------------------------------- A1 vs A2 excess by tercile
    L += ['', '=== B A1 vs A2 excess by S/N tercile (z>4; corrected, conventional)']
    s = g[g.zb.isin(HI)].copy()
    s['terc'] = pd.qcut(s.snr.rank(method='first'), 3, labels=['low', 'mid', 'high'])
    rows = []
    for tc in ('low', 'mid', 'high'):
        u = s[s.terc == tc]
        rows.append({'tercile': tc, 'N': len(u), 'snr_med': u.snr.median(), 'logM_med': u.logM.median(),
                     'A1_corr': u.C1.median(), 'A2_corr': u.C2.median()})
    B = pd.DataFrame(rows)
    L.append(B.round(3).to_string(index=False))
    lo, hi = B[B.tercile == 'low'].iloc[0], B[B.tercile == 'high'].iloc[0]
    r1, r2 = lo.A1_corr / hi.A1_corr, lo.A2_corr / hi.A2_corr
    L.append(f'  low/high tercile ratio: A1 {r1:.2f}, A2 {r2:.2f}  '
             f'(A2>A1 favours real m=2 structure; A1=A2 favours a noise floor)')
    B.to_csv(f'{a.out}_tercile.csv', index=False)

    # C ---------------------------------------------------------- field residual
    s = g[g.zb.isin(HI) & (g.C1 > 0) & (g.C2 > 0)].copy()
    if len(s) >= 40:
        s['dm'] = s.logM - 10
        s['lsnr'] = np.log10(s.snr)
        L += ['', '=== C field dependence of the residual at z>4 (after S/N and mass)']
        for k in (1, 2):
            y = np.log10(s[f'C{k}'].values)
            pred = np.column_stack([np.ones(len(s)), s.dm, s.lsnr])
            beta, *_ = np.linalg.lstsq(pred, y, rcond=None)
            s[f'res{k}'] = y - pred @ beta
        rows = []
        for gr, u in s.groupby('grp'):
            rows.append({'group': gr, 'N': len(u), 'resid_logA1': u.res1.mean(), 'resid_logA2': u.res2.mean()})
        C = pd.DataFrame(rows)
        L.append(C.round(3).to_string(index=False))
        L.append('  reading: residuals near 0 in all groups -> the field adds nothing beyond S/N and mass '
                 '(the excess is an S/N effect); a group offset -> a field-specific effect remains.')

    txt = '\n'.join(L)
    print(txt)
    with open(f'{a.out}_summary.txt', 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
