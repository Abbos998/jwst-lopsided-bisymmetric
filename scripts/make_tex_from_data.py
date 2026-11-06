#!/usr/bin/env python3
"""
make_tex_from_data.py — write every data-derived table and number of the
manuscript from the supplementary tables in data/, so that the PDF, the CSV
files and the code cannot drift apart.

Inputs (data/):
  table1_disks_F444W.csv          Table 1 and its calculation chain
  injection_calibration_F444W.csv noise floors and recovered fractions
  correction_steps_V0_V6.csv      Table 2
  restframe_test.csv              Table 3
  early_types_F444W.csv, sfr_split_snr100.csv   numbers quoted in the text

Outputs (tex/):
  table1.tex        Table 1 (values, total errors)
  table_chain.tex   Table 4, Appendix A (calculation chain)
  table_robust.tex  Table 6, Appendix B (robustness tests, from data/review7_*.csv, mass_test*.csv)
  table_errors.tex  Table 5, Appendix A (error budget: stat+calibration, systematics, total)
  table_steps.tex   Table 2
  table_restframe.tex Table 3
  numbers.tex       \newcommand macros for every number quoted in the text

Missing uncertainties are printed as a red "?" in the PDF, so a manuscript
built before the final run (build_all.sh) is visibly incomplete.

Consistency checks (printed; with --submission any failure exits with 1):
  * Table 2, adopted pipeline = A_meas of Table 1 / Table 4
  * Table 3, "F444W only" = Table 1 (same galaxies and calibration)
  * Table 1 recomputed from its chain (Eq. 2)

Usage:  python3 make_tex_from_data.py --data ../data --tex ../tex [--submission]
"""

import argparse
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

PEND = r'\pending'
ZLO, ZHI = 0.75, 4.5            # bins of Tables 2 and 3
STEP_LABELS = [                 # data/correction_steps_V0_V6.csv -> Table 2
    ('V0 original', r'V0 original pipeline'),
    ('+ WCS centre and field offset', r'V1 + WCS centre, field offset'),
    ('+ background', r'V2 + background, all pixels'),
    ('+ deprojection', r'V3 + deprojection'),
    ('+ masking', r'V4 + neighbour masking'),
    ('+ 0.5-1.5 r_d annulus', r'V5 + $0.5$--$1.5\,r_d$ annulus'),
    ('+ Sersic PA (adopted)', r"V6 + S\'ersic position angle (adopted)"),
]
RF_LABELS = [('F444W only', 'F444W only'),
             ('rest 0.7 um', r'$\lambda_{\rm rest}=0.7\,\mu$m'),
             ('rest 1.0 um', r'$\lambda_{\rm rest}=1.0\,\mu$m')]


def f(x, nd=3):
    """Round half up (as in the hand-made tables), not to the nearest binary float."""
    if not np.isfinite(x):
        return '--'
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def pm(v, e, nd=3):
    if not np.isfinite(v):
        return '--'
    return f"${f(v, nd)}\\pm" + (f"{f(e, nd)}$" if np.isfinite(e) else f"${PEND}")


def rng(a, nd=2):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return f"{f(a.min(), nd)}--{f(a.max(), nd)}"


def thousands(n):
    return f"{n:,}".replace(',', '{,}')


def row_at(t, z):
    r = t[np.isclose(t.z, z)]
    if len(r) != 1:
        raise SystemExit(f"bin z={z} not found exactly once")
    return r.iloc[0]


def final_amp(m, fl, rec):
    return 2 * np.sqrt(max(m * m - fl * fl, 0)) / rec


class Checks:
    def __init__(self):
        self.failed = []

    def __call__(self, name, ok, detail):
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            self.failed.append(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='../data')
    ap.add_argument('--tex', default='../tex')
    ap.add_argument('--submission', action='store_true')
    ap.add_argument('--core', action='store_true',
                    help='check only Table 1 and its chain (Tables 1, 4, 5); the robustness tests '
                         '(Appendix B) and the F277W/F444W ratios may be missing')
    a = ap.parse_args()
    rd = lambda n: pd.read_csv(os.path.join(a.data, n))
    t = rd('table1_disks_F444W.csv')
    cal = rd('injection_calibration_F444W.csv')
    steps = rd('correction_steps_V0_V6.csv')
    rf = rd('restframe_test.csv')
    et = rd('early_types_F444W.csv')
    sfr = rd('sfr_split_snr100.csv')
    vals = t[['A1_meas_pipe', 'A1_final_conv']].to_numpy(float).ravel()
    rounded = np.allclose(vals, np.round(vals, 3), atol=1e-9)
    tol = 1.5e-3 if rounded else 1e-6       # rounding of 3-decimal inputs
    chk = Checks()
    W = lambda name, lines: open(os.path.join(a.tex, name), 'w').write('\n'.join(lines) + '\n')

    # ---------------------------------------------------------------- Table 1
    L = [r"\begin{tabular}{rrcccc}", r"\toprule",
         r"$z$ & $N$ & $A_1$ & $A_2$ & $A_1/A_2$ (obs.) & $A_1/A_2$ (corr.)\\", r"\midrule"]
    for r in t.itertuples():
        L.append(f"{r.z:.2f} & {int(r.N_valid)} & {pm(r.A1_final_conv, r.A1_final_err_conv)} & "
                 f"{pm(r.A2_final_conv, r.A2_final_err_conv)} & {f(r.A1A2_obs, 2)} & "
                 f"{pm(r.A1A2_corrected, r.A1A2_corrected_err, 2)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    W('table1.tex', L)

    # ------------------------------------------------------- Table 4 (chain)
    L = [r"\begin{tabular}{r|cccc|cccc|ccc}", r"\toprule",
         r" & \multicolumn{4}{c|}{$m=1$} & \multicolumn{4}{c|}{$m=2$} & "
         r"\multicolumn{3}{c}{$A_1/A_2$}\\",
         r"$z$ & $A_{\rm meas}$ & $A_{\rm floor}$ & $f_{\rm rec}$ & $A_{\rm final}$ & "
         r"$A_{\rm meas}$ & $A_{\rm floor}$ & $f_{\rm rec}$ & $A_{\rm final}$ & obs. & $b$ & corr.\\",
         r"\midrule"]
    worst = 0.0
    for r in t.itertuples():
        row = [f(r.z, 2)]
        for k in ('A1', 'A2'):
            m, fl, rec, fin = (getattr(r, f'{k}_meas_pipe'), getattr(r, f'{k}_floor_pipe'),
                               getattr(r, f'{k}_recovery'), getattr(r, f'{k}_final_conv'))
            worst = max(worst, abs(final_amp(m, fl, rec) - fin))
            row += [f(m), f(fl), f(rec), f(fin)]
        worst = max(worst, abs(r.A1A2_obs / r.A1A2_bias - r.A1A2_corrected))
        row += [f(r.A1A2_obs), f(r.A1A2_bias), f(r.A1A2_corrected)]
        L.append(' & '.join(row) + r'\\')
    L += [r"\bottomrule", r"\end{tabular}"]
    W('table_chain.tex', L)

    # ------------------------------------------------- Table 5 (error budget)
    L = [r"\begin{tabular}{r|ccc|ccc|ccc}", r"\toprule",
         r" & \multicolumn{3}{c|}{$A_1$} & \multicolumn{3}{c|}{$A_2$} & \multicolumn{3}{c}{$A_1/A_2$}\\",
         r"$z$ & $\sigma_{\rm stat+cal}$ & $\sigma_{\rm syst}$ & $\sigma_{\rm tot}$ & "
         r"$\sigma_{\rm stat+cal}$ & $\sigma_{\rm syst}$ & $\sigma_{\rm tot}$ & "
         r"$\sigma_{\rm stat+cal}$ & $\sigma_{\rm syst}$ & $\sigma_{\rm tot}$\\", r"\midrule"]
    budget = [('A1_final_err_stat_conv', 'A1_final_syst_conv', 'A1_final_err_conv'),
              ('A2_final_err_stat_conv', 'A2_final_syst_conv', 'A2_final_err_conv'),
              ('A1A2_corrected_err_stat', 'A1A2_corrected_syst', 'A1A2_corrected_err')]
    have = all(c in t for b_ in budget for c in b_)
    dq = 0.0
    for r in t.itertuples():
        row = [f(r.z, 2)]
        for st, sy, tot in budget:
            v = [getattr(r, c, np.nan) if have else np.nan for c in (st, sy, tot)]
            if have:
                dq = max(dq, abs(v[2] - np.hypot(v[0], v[1])))
            row += [f(x) if np.isfinite(x) else PEND for x in v]
        L.append(' & '.join(row) + r'\\')
    L += [r"\bottomrule", r"\end{tabular}"]
    W('table_errors.tex', L)
    chk('Table 1 errors = sqrt(stat_cal^2 + syst^2) (Table 5)', have and dq <= (1e-3 if rounded else 1e-9),
        f"max deviation {dq:.1e}" if have else "stat/syst columns missing (rerun export_data.py)")
    print("Consistency checks" + (" (inputs rounded to 3 decimals)" if rounded else ""))
    chk('Table 1 = Eq. (2) applied to Table 4', worst <= (2.5e-3 if rounded else 1e-6),
        f"max |recomputed - stored| = {worst:.4f} (strict check: verify_table1.py)")

    # ---------------------------------------------------------------- Table 2
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{$A_1$} & \multicolumn{2}{c}{$A_2$} & \multicolumn{2}{c}{$A_1/A_2$}\\",
         r"Step & $z\sim0.75$ & $z\sim4.5$ & $z\sim0.75$ & $z\sim4.5$ & $z\sim0.75$ & $z\sim4.5$\\",
         r"\midrule"]
    for key, lab in STEP_LABELS:
        s = steps[steps.step == key]
        lo, hi = s[np.isclose(s.z, ZLO)].iloc[0], s[np.isclose(s.z, ZHI)].iloc[0]
        L.append(f"{lab} & {f(lo.A1_pipe)} & {f(hi.A1_pipe)} & {f(lo.A2_pipe)} & {f(hi.A2_pipe)} & "
                 f"{f(lo.A1A2, 2)} & {f(hi.A1A2, 2)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    W('table_steps.tex', L)
    fin = steps[steps.step == STEP_LABELS[-1][0]]
    d = max(abs(row_at(fin, z)[c] - row_at(t, z)[k]) for z in t.z
            for c, k in (('A1_pipe', 'A1_meas_pipe'), ('A2_pipe', 'A2_meas_pipe'), ('A1A2', 'A1A2_obs')))
    chk('Table 2 adopted step = A_meas of Table 1', d <= (5e-4 if rounded else 1e-6),
        f"max difference {d:.4f} over all bins")

    # ---------------------------------------------------------------- Table 3
    L = [r"\begin{tabular}{lcccccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{$N$} & \multicolumn{3}{c}{$A_1$} & \multicolumn{3}{c}{$A_2$}\\",
         r" & $z\sim0.75$ & $z\sim4.5$ & $z\sim0.75$ & $z\sim4.5$ & growth & $z\sim0.75$ & $z\sim4.5$ & growth\\",
         r"\midrule"]
    growth = {}

    def nfilt(row):
        """N and filters used, from the 'filters' column ('F444W:241 F277W:8')."""
        parts = [p_.split(':') for p_ in str(row.get('filters', '')).split() if ':' in p_]
        n = sum(int(c) for _, c in parts)
        fl = '+'.join(fn.replace('F', '').replace('W', '') for fn, _ in parts)
        return f"{n} ({fl})" if parts else '--'

    for key, lab in RF_LABELS:
        s = rf[rf.set == key]
        lo, hi = row_at(s, ZLO), row_at(s, ZHI)
        g1, g2 = hi.A1_conv / lo.A1_conv, hi.A2_conv / lo.A2_conv
        growth[key] = (g1, g2)
        L.append(f"{lab} & {nfilt(lo)} & {nfilt(hi)} & {f(lo.A1_conv)} & {f(hi.A1_conv)} & {g1:.1f} & "
                 f"{f(lo.A2_conv)} & {f(hi.A2_conv)} & {g2:.1f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    W('table_restframe.tex', L)
    s = rf[rf.set == 'F444W only']
    d = max(abs(row_at(s, z)[c] - row_at(t, z)[k]) for z in t.z
            for c, k in (('A1_conv', 'A1_final_conv'), ('A2_conv', 'A2_final_conv')))
    chk('Table 3 "F444W only" = Table 1', d <= tol,
        f"max difference {d:.3f} (rerun restframe_test.py with the Table 1 calibration if it fails)")

    # ------------------------------------------------------- numbers in text
    N = {}
    hi2 = t[t.z > 4]
    lo = row_at(t, ZLO)
    for k, K in (('A1', 'One'), ('A2', 'Two')):
        low, high = lo[f'{k}_final_conv'], hi2[f'{k}_final_conv'].mean()
        N[f'A{K}Low'] = f(low)
        N[f'A{K}LowR'] = f(low, 2)
        N[f'A{K}High'] = f(high, 2)
        N[f'A{K}Growth'] = f(high / low, 1)
        fc = hi2[f'{k}_floorcorr_pipe'].mean() / lo[f'{k}_floorcorr_pipe']
        N[f'A{K}GrowthFloor'] = f(fc, 2)
    N['Nvalid'] = thousands(int(t.N_valid.sum()))
    N['RcorrLow'] = f(lo.A1A2_corrected, 2)
    N['RcorrLowErr'] = f(lo.A1A2_corrected_err, 2) if np.isfinite(lo.A1A2_corrected_err) else PEND
    plat = t[t.z >= 1.75].A1A2_corrected
    N['RcorrPlat'] = f(plat.mean(), 2)
    N['RcorrPlatSc'] = f(plat.std(ddof=1), 2)
    N['RobsLow'] = f(lo.A1A2_obs, 2)
    N['RobsHighRange'] = rng(t[t.z >= 1.75].A1A2_obs, 1)
    c = cal.set_index('z')
    N['FloorConvLow'], N['FloorConvHigh'] = f(c.loc[ZLO, 'A1_floor_conv'], 2), f(c.loc[ZHI, 'A1_floor_conv'], 2)
    N['FloorPipeLow'], N['FloorPipeHigh'] = f(c.loc[ZLO, 'A1_floor_pipe']), f(c.loc[ZHI, 'A1_floor_pipe'])
    N['RecOneRange'], N['RecTwoRange'] = rng(t.A1_recovery), rng(t.A2_recovery)
    N['BiasRange'] = rng(t.A1A2_bias)
    for tag, k, z in (('ExA', 'A1', ZLO), ('ExB', 'A2', ZHI)):
        r = row_at(t, z)
        m, fl, rec = r[f'{k}_meas_pipe'], r[f'{k}_floor_pipe'], r[f'{k}_recovery']
        N[f'{tag}meas'], N[f'{tag}floor'], N[f'{tag}rec'] = f(m), f(fl), f(rec)
        N[f'{tag}result'] = f(final_amp(round(m, 3), round(fl, 3), round(rec, 3)))
        N[f'{tag}table'] = f(r[f'{k}_final_conv'])
        if k == 'A2':
            N[f'{tag}inv'] = f(1 / rec, 1)
    mid = t[(t.z > 1.5) & (t.z < 3)].A1_final_conv
    N['AOneMidRange'] = rng(mid)
    N['AOneIncr'] = f"{round((row_at(t, 3.75).A1_final_conv / row_at(t, 1.25).A1_final_conv - 1) * 10) * 10:.0f}"
    e_lo, e_hi = row_at(et, ZLO), et[et.z > 4]
    N['ETgrowthOne'] = f(e_hi.A1_floorcorr_conv.mean() / e_lo.A1_floorcorr_conv, 1)
    N['ETgrowthTwo'] = f(e_hi.A2_floorcorr_conv.mean() / e_lo.A2_floorcorr_conv, 1)
    N['ETratio'] = f(et.A1A2_corrected_approx.mean(), 2)
    N['ETratioR'] = f(et.A1A2_corrected_approx.mean(), 1)
    s = sfr[(sfr.z < 3) & sfr.A1_SF_over_Q.notna() & sfr.A1_SF_over_Q_err.gt(0)]
    w = 1 / s.A1_SF_over_Q_err ** 2
    N['SFQ'] = f((w * s.A1_SF_over_Q).sum() / w.sum(), 2)
    N['SFQerr'] = f(1 / np.sqrt(w.sum()), 2)
    N['RFgrowthOne'] = f(growth['rest 1.0 um'][0], 1)
    N['RFgrowthTwo'] = f(growth['rest 1.0 um'][1], 1)
    rr = rf[(rf.set != 'F444W only') & (rf.z >= 1.25)].A1A2
    N['RFratioRange'] = rng(rr)
    # robustness tests computed from the package (median_recovery_test.py,
    # injection_sample_match.py); pending if the files are missing
    mr_p = os.path.join(a.data, 'median_recovery_test.csv')
    im_p = os.path.join(a.data, 'injection_sample_match.csv')
    pct = lambda x: f"{abs(100 * x):.0f}"
    if os.path.exists(mr_p):
        mr = pd.read_csv(mr_p).groupby('mixture')[['A1_rel_error', 'A2_rel_error', 'R12_rel_error']].mean()
        lo_, un_, hi_ = (mr[mr.index.str.startswith(p_)].iloc[0] for p_ in ('skewed low', 'uniform', 'skewed high'))
        N['MedLowAone'], N['MedLowAtwo'] = pct(lo_.A1_rel_error), pct(lo_.A2_rel_error)
        N['MedHighAone'], N['MedHighAtwo'] = pct(hi_.A1_rel_error), pct(hi_.A2_rel_error)
        N['MedUni'] = pct(max(abs(un_.A1_rel_error), abs(un_.A2_rel_error)))
        N['MedRatio'] = pct(mr.R12_rel_error.abs().max())
    else:
        for k in ('MedLowAone', 'MedLowAtwo', 'MedHighAone', 'MedHighAtwo', 'MedUni', 'MedRatio'):
            N[k] = PEND
    if os.path.exists(im_p):
        im = pd.read_csv(im_p)
        N['InjRdDev'] = pct((im.rd_inj_med / im.rd_obs_med - 1).abs().max())
        N['InjQDev'] = f((im.q_inj_med - im.q_obs_med).abs().max(), 2)
    else:
        N['InjRdDev'], N['InjQDev'] = PEND, PEND
    # effects of the pipeline steps quoted in Sect. 3.2 (Table 2 / Fig. 3)
    stp = {k: g.set_index('z') for k, g in steps.groupby('step')}
    v = [stp[k] for k, _ in STEP_LABELS]
    lo4 = v[0].index < 4
    d_wcs = 1 - v[1].A1_pipe / v[0].A1_pipe
    N['StepWcs'] = f"{f(100 * d_wcs[lo4].min(), 0)}--{f(100 * d_wcs[lo4].max(), 0)}"
    N['StepDeproj'] = pct(1 - v[3].loc[ZLO, 'A2_pipe'] / v[2].loc[ZLO, 'A2_pipe'])
    d_mask = 1 - v[4].A1_pipe / v[3].A1_pipe
    N['StepMaskLo'], N['StepMaskHi'] = pct(d_mask[lo4].mean()), pct(d_mask[~lo4].mean())
    N['StepVzeroR'] = f(v[0][v[0].index > 3].A1A2.mean(), 1)
    # size relative to the PSF at z~4.5 (measured galaxies, cutout pixels of 0.05")
    im_p2, ps_p = os.path.join(a.data, 'injection_sample_match.csv'), os.path.join(a.data, 'empirical_psf.csv')
    if os.path.exists(im_p2) and os.path.exists(ps_p):
        rd_hi = pd.read_csv(im_p2).set_index('z').loc[ZHI, 'rd_obs_med']
        fwhm = pd.read_csv(ps_p).set_index('filter').loc['F444W', 'fwhm_arcsec'] / 0.05
        N['RdHiPix'], N['FwhmPix'], N['RdFwhm'] = f(rd_hi, 1), f(fwhm, 1), f(rd_hi / fwhm, 1)
    else:
        N['RdHiPix'] = N['FwhmPix'] = N['RdFwhm'] = PEND
    # sample accounting
    ntot = int(t.N_total.sum()) if t.N_total.notna().all() else None
    if ntot:
        N['Ntotal'] = thousands(ntot)
        N['Nexcl'] = thousands(ntot - int(t.N_valid.sum()))
        N['NexclPct'] = f(100 * (ntot - t.N_valid.sum()) / ntot, 1)
        fr = 100 * (1 - t.N_valid / t.N_total)
        N['NexclRange'] = f"{f(fr.min(), 0)}--{f(fr.max(), 0)}"
    else:
        N['Ntotal'] = N['Nexcl'] = N['NexclPct'] = N['NexclRange'] = PEND
    # corrected A1/A2: weighted mean, chi2 against a constant, slope with log(1+z)
    ok = t.A1A2_corrected_err.notna() & (t.A1A2_corrected_err > 0)
    if ok.all():
        w = 1 / t.A1A2_corrected_err ** 2
        y = t.A1A2_corrected
        mean = float((w * y).sum() / w.sum())
        X = np.column_stack([np.ones(len(t)), np.log10(1 + t.z)])
        cov = np.linalg.inv(X.T @ (X * w.to_numpy()[:, None]))
        beta = cov @ (X.T @ (w * y))
        N['RatioMean'], N['RatioMeanErr'] = f(mean, 2), f(1 / np.sqrt(w.sum()), 2)
        N['RatioChi'], N['RatioDof'] = f(float(((y - mean) ** 2 * w).sum()), 1), str(len(t) - 1)
        N['RatioSlope'], N['RatioSlopeErr'] = f(beta[1], 2), f(np.sqrt(cov[1, 1]), 2)
        for tag, sel in (('Lo', t.z < 3.5), ('Hi', t.z > 3.5)):
            ws = w[sel]
            N[f'Ratio{tag}'] = f(float((ws * y[sel]).sum() / ws.sum()), 2)
            N[f'Ratio{tag}Err'] = f(1 / np.sqrt(ws.sum()), 2)
    else:
        for k in ('RatioMean', 'RatioMeanErr', 'RatioChi', 'RatioDof', 'RatioSlope', 'RatioSlopeErr',
                  'RatioLo', 'RatioLoErr', 'RatioHi', 'RatioHiErr'):
            N[k] = PEND
    # same galaxies in F277W and F444W (filter_ratio.py)
    fr_p = os.path.join(a.data, 'filter_ratio_F277W_F444W.csv')
    if os.path.exists(fr_p):
        frt = pd.read_csv(fr_p)
        N['FRoneRange'], N['FRtwoRange'] = rng(frt.A1_ratio), rng(frt.A2_ratio)
        dev = max((frt.A1_ratio - 1).abs().max(), (frt.A2_ratio - 1).abs().max())
        N['FRmaxDev'] = pct(dev)
        # scale the change over the factor 1.6 between the filters to the factor 3.2 of F444W
        N['FRexpect'] = pct(dev * np.log(3.2) / np.log(1.6))
        sc_ = np.log(3.2) / np.log(1.6)
        N['FRexpectOne'] = pct((frt.A1_ratio - 1).abs().max() * sc_)
        N['FRexpectTwo'] = pct((frt.A2_ratio - 1).abs().max() * sc_)
    else:
        N['FRoneRange'] = N['FRtwoRange'] = N['FRmaxDev'] = N['FRexpect'] = PEND
        N['FRexpectOne'] = N['FRexpectTwo'] = PEND
    # ------------------------------------------- Appendix B: robustness tests
    def gr(tab, col):
        lo_ = tab[np.isclose(tab.z, ZLO)][col]
        hi_ = tab[tab.z > 4][col]
        return float(hi_.mean() / lo_.iloc[0]) if len(lo_) and len(hi_) else np.nan

    def rload(n):
        p_ = os.path.join(a.data, n)
        return pd.read_csv(p_) if os.path.exists(p_) else None

    rob = [('Nominal (Table~\\ref{tab:main})', t.rename(columns={'A1_final_conv': 'a1', 'A2_final_conv': 'a2',
                                                                 'A1A2_corrected': 'r'}))]
    sc = rload('review7_table1_disk_sc.csv')
    if sc is not None:
        rob.append(('Self-consistent calibration, 2nd iteration',
                    sc.rename(columns={'A1_final_std': 'a1', 'A2_final_std': 'a2', 'R12_debiased_matched': 'r'})))
    cl = rload('review7_clumps.csv')
    if cl is not None:
        for lab_, g in cl.groupby('model'):
            nc, fc = lab_.replace('clumps', '').split('_f')
            g = g.merge(t[['z', 'A1A2_corrected']], on='z')
            g['r'] = g.A1A2_corrected * g.b_smooth / g.b_clumpy
            rob.append((f'Clumpy calibration ({nc} clumps, {100 * float(fc):.0f}\\% of flux)',
                        g.rename(columns={'A1_final_with_clumpy_cal': 'a1', 'A2_final_with_clumpy_cal': 'a2'})))
    an = rload('review7_annulus.csv')
    if an is not None:
        for sc_, g in an.groupby('ring_scale'):
            lo_r, hi_r = 0.5 * sc_, 1.5 * sc_
            rob.append((f'Annulus {lo_r:.2f}--{hi_r:.2f}\\,$r_d$', g.rename(columns={'A1': 'a1', 'A2': 'a2', 'R12': 'r'})))
    mt_ = rload('mass_test.csv')
    if mt_ is not None:
        for lab_, g in mt_.groupby('test', sort=False):
            if lab_.startswith('all'):
                continue
            nice = ('Mass-matched to the pooled $M_\\star$ distribution' if lab_.startswith('mass')
                    else '$' + lab_.replace('logM', '\\log M_\\star') + '$')
            rob.append((nice, g.rename(columns={'A1': 'a1', 'A2': 'a2', 'R12': 'r'})))
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Test & $N_{z>4}$ & $A_1$ growth & $A_2$ growth & $\langle A_1/A_2\rangle_{z<3.5}$ & "
         r"$\langle A_1/A_2\rangle_{z>3.5}$\\", r"\midrule"]
    for lab_, g in rob:
        lo_m, hi_m = g[g.z < 3.5].r.mean(), g[g.z > 3.5].r.mean()
        ncol = next((c for c in ('N', 'N_valid') if c in g), None)
        nhi = str(int(g[g.z > 4][ncol].sum())) if ncol else '--'
        L.append(f"{lab_} & {nhi} & {f(gr(g, 'a1'), 1)} & {f(gr(g, 'a2'), 1)} & {f(lo_m, 2)} & {f(hi_m, 2)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    W('table_robust.tex', L)
    # the robustness files must come from the same run as Table 1 (no stale results)
    stale = []
    if sc is not None:
        scs_ = rload('review7_selfconsistent.csv')
        if scs_ is not None and np.nanmax(np.abs(scs_.A1_table1.to_numpy() - t.A1_final_conv.to_numpy())) > 1e-6:
            stale.append('review7_selfconsistent.csv')
    if mt_ is not None:
        nall = mt_[mt_.test.str.startswith('all')].N.to_numpy()
        if len(nall) != len(t) or (nall != t.N_valid.to_numpy()).any():
            stale.append('mass_test.csv')
    if an is not None and not np.allclose(an.groupby('z').N_nominal.first().to_numpy(), t.N_valid.to_numpy()):
        stale.append('review7_annulus.csv')
    present = [x for x in (sc, cl, an, mt_) if x is not None]
    chk('robustness tests (Appendix B) from the same run as Table 1',
        (bool(present) or a.core) and not stale,
        'stale: ' + ', '.join(stale) if stale else ('ok' if present else 'not available (run run_review7.sh)'))
    names = dict(rob)
    for key, tag in (('Self-consistent calibration, 2nd iteration', 'Sc'),):
        pass
    if an is not None:
        for sc_, g in an.groupby('ring_scale'):
            tag = 'AnnSmall' if sc_ < 1 else 'AnnLarge'
            N[f'{tag}One'], N[f'{tag}Two'] = f(gr(g, 'A1'), 1), f(gr(g, 'A2'), 1)
            N[f'{tag}R'] = f(g.R12.mean(), 2)
    else:
        for tag in ('AnnSmall', 'AnnLarge'):
            N[f'{tag}One'] = N[f'{tag}Two'] = N[f'{tag}R'] = PEND
    if cl is not None:
        gs = [(gr(g, 'A1_final_with_clumpy_cal'), gr(g, 'A2_final_with_clumpy_cal')) for _, g in cl.groupby('model')]
        span = lambda v: f(min(v), 1) if f(min(v), 1) == f(max(v), 1) else f"{f(min(v), 1)}--{f(max(v), 1)}"
        N['ClumpOne'] = span([x[0] for x in gs])
        N['ClumpTwo'] = span([x[1] for x in gs])
    else:
        N['ClumpOne'] = N['ClumpTwo'] = PEND
    sp = rload('review7_spread.csv')
    if sp is not None:
        m_ = sp.groupby('spread')[['A1_rel_error', 'A2_rel_error', 'R12_rel_error']].mean().abs().max().max()
        N['SpreadMax'] = pct(m_)
    else:
        N['SpreadMax'] = PEND
    scs = rload('review7_selfconsistent.csv')
    N['ScShift'] = f(scs.filter(like='shift_sigma').abs().max().max(), 1) if scs is not None else PEND
    if mt_ is not None:
        g = mt_[mt_.test.str.startswith('mass')]
        N['MassOne'], N['MassTwo'] = f(gr(g, 'A1'), 1), f(gr(g, 'A2'), 1)
    else:
        N['MassOne'] = N['MassTwo'] = PEND
    rg = rload('mass_test_regression.csv')
    if rg is not None:
        for q_, tag in (('A1', 'One'), ('A2', 'Two'), ('R12', 'R')):
            r0 = rg[(rg.quantity == q_) & (rg.model == 'z only')].iloc[0]
            r1 = rg[(rg.quantity == q_) & (rg.model == 'z + mass')].iloc[0]
            N[f'RegB{tag}'], N[f'RegBm{tag}'] = f(r0.b_log1pz, 2), f(r1.b_log1pz, 2)
            N[f'RegBmErr{tag}'], N[f'RegC{tag}'] = f(r1.b_err, 2), f(r1.c_logM, 2)
            N[f'RegCErr{tag}'] = f(r1.c_err, 2)
    else:
        for tag in ('One', 'Two', 'R'):
            for k in ('RegB', 'RegBm', 'RegBmErr', 'RegC', 'RegCErr'):
                N[f'{k}{tag}'] = PEND
    L = ["% generated by make_tex_from_data.py from data/*.csv -- do not edit"]
    L += [f"\\newcommand{{\\num{k}}}{{{v}}}" for k, v in N.items()]
    W('numbers.tex', L)
    print(f"wrote table1.tex, table_chain.tex, table_errors.tex, table_steps.tex, table_restframe.tex, "
          f"numbers.tex ({len(N)} numbers) in {a.tex}")

    optional = ('FR', 'Ann', 'Clump', 'Spread', 'ScShift', 'Mass', 'Reg')
    pending = [k for k, v in N.items() if v == PEND and not (a.core and k.startswith(optional))] + \
        [c for c in ('A1_final_err_conv', 'A2_final_err_conv', 'A1A2_corrected_err') if t[c].isna().any()]
    chk('uncertainties present', not pending, ', '.join(pending) if pending else 'all')
    if chk.failed:
        print(f"{len(chk.failed)} check(s) failed: " + '; '.join(chk.failed))
        if a.submission:
            sys.exit(1)
    else:
        print("All manuscript consistency checks passed.")


if __name__ == '__main__':
    main()
