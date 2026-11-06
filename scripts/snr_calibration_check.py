#!/usr/bin/env python3
"""
snr_calibration_check.py -- why do the high-redshift amplitudes drop when only
high-S/N galaxies are kept?  Two explanations are tested:

 (a) the bin-level noise-floor/recovery calibration fails for the low-S/N
     galaxies that dominate the high-redshift bins (then the published
     high-z amplitudes would be inflated by noise);
 (b) at high redshift a S/N cut selects the brightest, most massive galaxies,
     which have lower amplitudes at any redshift (A decreases with mass at
     fixed z in the mass test).

Part 1 -- closure test on the injections: within each redshift bin the
injected galaxies are split into terciles of S/N (edges from the observed
galaxies of the bin). The bin-level correction (floor and recovered
fraction from all injections of the bin, as in the paper) is applied to the
median measured amplitude of each tercile and compared with the injected
amplitude. Ratios ~1 in every tercile mean that the calibration does not
depend on S/N; a ratio > 1 in the lowest tercile would support (a).

Part 2 -- observed galaxies by S/N tercile: each tercile of each bin is
corrected with the floor and recovered fraction of the injected galaxies in
the same tercile (S/N-specific calibration), with bootstrap uncertainties,
together with the median stellar mass and size of the tercile. If the high-S/N
tercile has lower amplitudes AND higher masses while the closure test is
fine, (b) is the explanation.

Run from the package scripts/ directory (see run_snr_check.sh).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_core as cc
import injection_qpa_test as iq

MASS_COLS = ['logM', 'log_mass', 'lmass', 'mass_log', 'logmstar', 'log_mstar']


def corr(med, floor, rec):
    return 2 * np.sqrt(np.maximum(med * med - floor * floor, 0)) / rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cc.add_common_args(ap)
    ap.add_argument('--galaxies', required=True, help='data/intermediate/corrected_disk_f444w.csv')
    ap.add_argument('--idx_col', default='idx')
    ap.add_argument('--mass_csv', help='CSV with idx and log stellar mass (default: --catalog)')
    ap.add_argument('--mass_col')
    ap.add_argument('--nboot', type=int, default=500)
    ap.add_argument('--reuse_rows', action='store_true', help='reuse snr_injection_rows.csv of a previous run')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inj = cc.load_module(os.path.join(a.scripts, 'injection_test.py'), 'inj')
    rng = np.random.default_rng(a.seed + 99)

    cat = cc.load_catalog(a, inj)
    raw = pd.read_csv(a.catalog)
    cat['idx'] = raw[a.idx_col].values[cat.row.values].astype(int)
    mraw = pd.read_csv(a.mass_csv) if a.mass_csv else raw
    mcol = a.mass_col or next((c for c in MASS_COLS if c in mraw.columns), None)
    gal = pd.read_csv(a.galaxies).dropna(subset=['A1_V6', 'A2_V6'])
    m = gal[[a.idx_col, 'z', 'A1_V6', 'A2_V6']].rename(columns={a.idx_col: 'idx', 'z': 'z_meas'}) \
        .merge(cat[['idx', 'snr', 're_pix']], on='idx', how='inner')
    if mcol:
        mm = mraw[[a.idx_col, mcol]].rename(columns={a.idx_col: 'idx', mcol: 'logM'})
        m = m.merge(mm, on='idx', how='left')
        print(f'stellar mass column: {mcol}')
    else:
        m['logM'] = np.nan
        print('no stellar-mass column found (pass --mass_csv/--mass_col); masses not reported')
    m['zbin'] = cc.Z_CEN[np.clip(np.digitize(m.z_meas, cc.Z_EDGES) - 1, 0, len(cc.Z_CEN) - 1)]
    print(f'{len(m)} observed galaxies joined')

    # tercile edges of S/N from the observed galaxies of each bin
    edges = {zc: np.percentile(g.snr, [100 / 3, 200 / 3]) for zc, g in m.groupby('zbin')}
    m['terc'] = [int(np.digitize(s, edges[z])) for s, z in zip(m.snr, m.zbin)]

    # injections drawn from the observed galaxies of each bin (as in the paper)
    psf = cc.psf_from_fits(a.psf, a.psf_pixscale, a.pixscale, inj)
    cfg = dict(scripts=os.path.abspath(a.scripts), am_func=a.am_func, psf=psf,
               acf=np.load(a.noise_acf) if a.noise_acf else None,
               npix=int(round(a.cutout_arcsec / a.pixscale)), aper=a.snr_aper_arcsec / a.pixscale,
               sigma_cat=a.sigma_cat, eps=cc.matched_eps(a.table1), cat_err=False,
               emp=None, emp_qedges=None, variants={'ideal'})
    rows_path = os.path.join(a.out, 'snr_injection_rows.csv')
    if a.reuse_rows and os.path.exists(rows_path):
        rows = pd.read_csv(rows_path)
        print(f'reusing {len(rows)} injection measurements from {rows_path}')
    else:
        items = cc.draw_sample(cat[cat.idx.isin(m.idx)].reset_index(drop=True), a.n_per_bin, a.seed)
        print(f'{len(items)} injected models')
        rows = pd.DataFrame(cc.run_parallel(iq.worker, items, cfg, a.workers))
        rows.to_csv(rows_path, index=False)
    # injected galaxies that fail the measurement (e.g. < 50 unmasked pixels) give NaN,
    # exactly as in the paper's calibration, which uses nan-medians
    nbad = int(rows[['A1', 'A2']].isna().any(axis=1).sum())
    print(f'{nbad} of {len(rows)} injection measurements are NaN (failed measurement) and are ignored')
    rows = rows.dropna(subset=['A1', 'A2', 'A1_true', 'A2_true'])
    rows['terc'] = [int(np.digitize(s, edges[z])) for s, z in zip(rows.snr, rows.zbin)]
    eps = cc.matched_eps(a.table1)

    out1, out2 = [], []
    for zc in cc.Z_CEN:
        R = rows[rows.zbin == zc]
        z0, mt = R[R.scenario == 'zero'], R[R.scenario == 'matched']
        if mt.empty:
            continue
        cal_bin = {}
        for k in (1, 2):
            f = np.median(z0[f'A{k}'])
            rec = np.sqrt(max(np.median(mt[f'A{k}']) ** 2 - f * f, 0)) / np.median(mt[f'A{k}_true'])
            cal_bin[k] = (f, rec)
        G = m[m.zbin == zc]
        for t in range(3):
            zt, mtt = z0[z0.terc == t], mt[mt.terc == t]
            Gt = G[G.terc == t]
            r1 = dict(zbin=zc, tercile=t, N_inj=len(mtt), snr_med_inj=mtt.snr.median())
            r2 = dict(zbin=zc, tercile=t, N=len(Gt), snr_med=Gt.snr.median(),
                      logM_med=Gt.logM.median(), x_med=(Gt.re_pix / (0.156 / a.pixscale)).median())
            for k in (1, 2):
                f, rec = cal_bin[k]
                true_conv = eps[round(zc, 2)][k - 1]
                if len(mtt) >= 10:
                    r1[f'A{k}_closure_binlevel'] = corr(np.median(mtt[f'A{k}']), f, rec) / true_conv
                    ft = np.median(zt[f'A{k}']) if len(zt) >= 10 else f
                    rect = np.sqrt(max(np.median(mtt[f'A{k}']) ** 2 - ft * ft, 0)) / np.median(mtt[f'A{k}_true'])
                    r1[f'A{k}_floor_terc'] = ft
                    r1[f'A{k}_floor_bin'] = f
                    r1[f'A{k}_rec_terc'] = rect
                    r1[f'A{k}_rec_bin'] = rec
                else:
                    ft, rect = f, rec
                if len(Gt) >= 5:
                    v = Gt[f'A{k}_V6'].values
                    r2[f'A{k}_terc_cal'] = corr(np.median(v), ft, rect)
                    bs = [corr(np.median(v[rng.integers(0, len(v), len(v))]), ft, rect) for _ in range(a.nboot)]
                    r2[f'A{k}_terc_cal_err'] = np.std(bs)
                    r2[f'A{k}_bin_cal'] = corr(np.median(v), f, rec)
            out1.append(r1)
            out2.append(r2)
    T1, T2 = pd.DataFrame(out1), pd.DataFrame(out2)
    T1.to_csv(os.path.join(a.out, 'closure_by_snr_tercile.csv'), index=False)
    T2.to_csv(os.path.join(a.out, 'observed_by_snr_tercile.csv'), index=False)
    # ---------------- PART 3: S/N-stratified calibration applied galaxy by galaxy
    # floor, recovered fraction and ratio bias per (bin, tercile); each observed galaxy is
    # corrected with those of its own tercile, and the bin value is the median over galaxies
    # (with bin-level terms instead, this reproduces the paper's bin-level correction).
    def terms(R):
        z0, mt = R[R.scenario == 'zero'], R[R.scenario == 'matched']
        t = {}
        for k in (1, 2):
            f = np.median(z0[f'A{k}'])
            rec = np.sqrt(max(np.median(mt[f'A{k}']) ** 2 - f * f, 0)) / np.median(mt[f'A{k}_true'])
            t[k] = (f, rec)
        t['b'] = np.median(mt.A1 / mt.A2) / (np.median(mt.A1_true) / np.median(mt.A2_true))
        return t

    cal_bin, cal_t = {}, {}
    for zc in cc.Z_CEN:
        R = rows[rows.zbin == zc]
        cal_bin[zc] = terms(R)
        for t in range(3):
            Rt = R[R.terc == t]
            ok = (Rt.scenario == 'zero').sum() >= 10 and (Rt.scenario == 'matched').sum() >= 10
            ct = terms(Rt) if ok else None
            if ct is None or not all(np.isfinite(ct[k][1]) and ct[k][1] > 0 for k in (1, 2)) \
                    or not np.isfinite(ct['b']):
                if ok:
                    print(f'  z={zc} tercile {t}: tercile calibration undefined (recovered fraction <= 0); '
                          f'bin-level terms used')
                ct = cal_bin[zc]
            cal_t[(zc, t)] = ct

    def bin_values(G, stratified):
        out = {}
        for zc, g in G.groupby('zbin'):
            vals = {1: [], 2: [], 'r': []}
            for t, gt in g.groupby('terc'):
                c = cal_t[(zc, t)] if stratified else cal_bin[zc]
                for k in (1, 2):
                    vals[k].append(corr(gt[f'A{k}_V6'].values, *c[k]))
                vals['r'].append((gt.A1_V6 / gt.A2_V6).values / c['b'])
            out[zc] = {k: np.median(np.concatenate(v)) for k, v in vals.items()}
        return out

    def growth_of(bv):
        hi = [z for z in (4.5, 5.5) if z in bv]
        g = {f'A{k}_growth': np.mean([bv[z][k] for z in hi]) / bv[0.75][k] for k in (1, 2)}
        g['R12_lo'] = np.mean([bv[z]['r'] for z in bv if z < 3.5])
        g['R12_hi'] = np.mean([bv[z]['r'] for z in bv if z > 3.5])
        return g

    res3, per3 = [], []
    for name, strat in (('bin-level (paper method)', False), ('S/N-stratified', True)):
        bv = growth_of(bin_values(m, strat))
        boots = []
        bin_idx = [np.flatnonzero(m.zbin.values == zc) for zc in cc.Z_CEN]
        for _ in range(a.nboot):
            sel = np.concatenate([ix[rng.integers(0, len(ix), len(ix))] for ix in bin_idx if len(ix)])
            boots.append(growth_of(bin_values(m.iloc[sel], strat)))
        B = pd.DataFrame(boots)
        r = {'calibration': name}
        for k in ('A1_growth', 'A2_growth', 'R12_lo', 'R12_hi'):
            r[k] = bv[k]
            r[k + '_err'] = B[k].std()
        res3.append(r)
        for zc, v in bin_values(m, strat).items():
            per3.append(dict(calibration=name, zbin=zc, A1=v[1], A2=v[2], R12_corr=v['r']))
    T3 = pd.DataFrame(res3)
    P3 = pd.DataFrame(per3).pivot(index='zbin', columns='calibration')
    T3.to_csv(os.path.join(a.out, 'stratified_growth.csv'), index=False)
    P3.to_csv(os.path.join(a.out, 'stratified_perbin.csv'))

    pd.set_option('display.width', 220)
    txt3 = ['', 'PART 3 - S/N-stratified calibration applied galaxy by galaxy (bootstrap over observed galaxies; '
            f'{a.nboot} resamples; the injection sampling adds about 0.07 to the growth factors)',
            P3.round(3).to_string(), '', T3.round(3).to_string(index=False), '',
            'R12_lo / R12_hi are simple means of the corrected bins below / above z = 3.5.']
    txt = ['PART 1 - closure: bin-level correction applied to each S/N tercile of the injections '
           '(corrected / injected; 1 = perfect)',
           T1[['zbin', 'tercile', 'N_inj', 'snr_med_inj', 'A1_closure_binlevel', 'A2_closure_binlevel']]
           .round(3).to_string(index=False), '',
           'PART 2 - observed galaxies by S/N tercile, each corrected with the calibration of its own tercile',
           T2[['zbin', 'tercile', 'N', 'snr_med', 'logM_med', 'x_med', 'A1_terc_cal', 'A1_terc_cal_err',
               'A2_terc_cal', 'A2_terc_cal_err', 'A1_bin_cal', 'A2_bin_cal']].round(3).to_string(index=False), '',
           'Reading: closure ~1 in all terciles -> the bin-level calibration is not S/N dependent, and lower '
           'amplitudes in the high-S/N tercile reflect the galaxies themselves (compare logM_med). '
           'Closure > 1 in the low-S/N tercile -> the noise correction is incomplete there. '
           'A value of 0.000 in Part 2 means that the median of the tercile lies at or below the noise floor '
           'of that tercile (amplitude not measurable at that S/N), not that the amplitude is zero.'] + txt3
    txt = '\n'.join(txt)
    print('\n' + txt)
    open(os.path.join(a.out, 'snr_check_summary.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
