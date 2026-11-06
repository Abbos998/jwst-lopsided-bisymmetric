#!/usr/bin/env python3
"""
make_table1.py — corrected Table 1 from run_corrected_am.py (per-object CSV)
and the injection floors of injection_test.py (same estimator).

Per redshift bin:
  * median A1, A2, A3 and median(A1/A2) with bootstrap errors (variant --variant)
  * noise/PSF floor from the injection ZERO scenario (--centering scheme)
  * floor-corrected medians  A_m,corr = sqrt(A_m^2 - floor_m^2)
  * the same in the conventional normalisation (x2; the pipeline uses
    |sum I e^{im phi}| / sum I, half of 2|<I e^{im phi}>|/<I>)
  * the A1/A2 that a NO-EVOLUTION population would show (injection CONSTANT
    scenario), normalised to the first bin: subtract its rise from the
    observed ratio to see the evolution beyond the resolution/noise bias.

  * with a 'matched' injection (observed amplitudes injected): recovery
    fractions and the final amplitudes A_final = 2 sqrt(A^2 - floor^2) / rec,
    with errors from a joint bootstrap of the observed galaxies AND of the
    injected galaxies (calibration uncertainty), plus optional systematics
    from other injection runs (--syst label=summary.csv,...), e.g. other
    amplitudes (--matched_scale) or radial profiles (--pert_profile).
  * N_total (galaxies in the bin) and N (galaxies with a valid measurement).

The four systematics runs (scale0.5, scale1.5, rising, falling) and the
injection_per_object.csv of the main run are required; without them the
script stops (--exploratory relaxes this for tests only). The labels of the
systematics runs used are stored in the column syst_runs.

Writes <out>.csv and <out>.tex (plain tabular).

Usage:
    python3 make_table1.py --objects ../corrected_disk.csv \
        --injection ../injection_v6/injection_summary.csv --out ../table1_disk
"""

import argparse

import numpy as np
import pandas as pd

Z_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
Z_CEN = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50]
REQUIRED_SYST = ['scale0.5', 'scale1.5', 'rising', 'falling']


def boot(x, rng, nb=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan, len(x)
    return np.median(x), np.std(np.median(rng.choice(x, (nb, len(x))), axis=1)), len(x)


def corr(m, e, f):
    """sqrt(m^2 - f^2) and its propagated error (floor error neglected)."""
    if not (np.isfinite(m) and np.isfinite(f)) or m <= f:
        return np.nan, np.nan
    c = np.sqrt(m * m - f * f)
    return c, e * m / c


def joint_bootstrap(s, v, inj_obj, zc, centering, rng, nb):
    """Errors of the final values: resample the observed galaxies of the bin and,
    independently, the injected galaxies of the ZERO and MATCHED runs."""
    out = {}
    obs = {k: np.asarray(s[f'{k}_{v}'], float) for k in ('A1', 'A2')}
    obs = {k: x[np.isfinite(x)] for k, x in obs.items()}
    r12 = np.asarray(s[f'A1_{v}'] / s[f'A2_{v}'], float)
    r12 = r12[np.isfinite(r12)]
    if min(len(obs['A1']), len(obs['A2']), len(r12)) < 5:
        return out

    def med_boot(x):
        return np.median(rng.choice(x, (nb, len(x))), axis=1)

    ob = {k: med_boot(x) for k, x in obs.items()}
    ob_r = med_boot(r12)
    if inj_obj is None:
        return out
    q = inj_obj[(inj_obj.zbin.round(2) == round(zc, 2)) & (inj_obj.centering == centering)]
    z0, mt = q[q.scenario == 'zero'], q[q.scenario == 'matched']
    if len(z0) < 5 or len(mt) < 5:
        return out
    cal = {}
    for k in ('A1', 'A2'):
        f = med_boot(z0[k].dropna().values)
        m = med_boot(mt[k].dropna().values)
        t = med_boot(mt[f'{k}_true'].dropna().values)
        rec = np.sqrt(np.clip(m * m - f * f, 0, None)) / t
        cal[k] = (f, rec)
        fin = 2 * np.sqrt(np.clip(ob[k] ** 2 - f ** 2, 0, None)) / rec
        out[f'{k}_rec_err'] = np.std(rec)
        out[f'{k}_final_err_stat'] = np.std(fin)
    rr = (mt.A1 / mt.A2).replace([np.inf, -np.inf], np.nan).dropna().values
    tt = (mt.A1_true / mt.A2_true).replace([np.inf, -np.inf], np.nan).dropna().values
    bias = med_boot(rr) / med_boot(tt)
    out['R12_bias_matched_err'] = np.std(bias)
    out['R12_final_err_stat'] = np.std(ob_r / bias)
    return out


def central_values(row, summ, zc, centering):
    """Final A1, A2 (conventional) and debiased A1/A2 with another calibration."""
    z0 = summ[(summ.scenario == 'zero') & (summ.centering == centering)].set_index('zbin')
    mt = summ[(summ.scenario == 'matched') & (summ.centering == centering)].set_index('zbin')
    out = {'A1': np.nan, 'A2': np.nan, 'R12': np.nan}
    if zc not in mt.index:
        return out
    for k in ('A1', 'A2'):
        f = z0.loc[zc, f'{k}_med'] if zc in z0.index else row[f'{k}_floor']
        m, t = mt.loc[zc, f'{k}_med'], mt.loc[zc, f'{k}_true_med']
        rec = np.sqrt(max(m * m - f * f, 0)) / t if t > 0 else np.nan
        c = np.sqrt(max(row[k] ** 2 - f * f, 0))
        out[k] = 2 * c / rec if rec > 0 else np.nan
    tr = mt.loc[zc, 'A1_true_med'] / mt.loc[zc, 'A2_true_med']
    out['R12'] = row['R12'] / (mt.loc[zc, 'R12_med'] / tr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--objects', required=True, help='per-object CSV of run_corrected_am.py')
    ap.add_argument('--injection', required=True, help='injection_summary.csv (same estimator)')
    ap.add_argument('--variant', default='V6')
    ap.add_argument('--centering', default='catalog', help='injection centring scheme for the floor')
    ap.add_argument('--out', required=True, help='output prefix')
    ap.add_argument('--syst', help='label=injection_summary.csv,... alternative matched '
                    'calibrations; their spread is added as a systematic error')
    ap.add_argument('--nboot', type=int, default=1000)
    ap.add_argument('--require_syst', default=','.join(REQUIRED_SYST),
                    help='systematics runs that must all be given with --syst')
    ap.add_argument('--exploratory', action='store_true',
                    help='allow missing systematics / injection_per_object.csv '
                         '(never for the published table)')
    args = ap.parse_args()
    import os
    inj_obj_path = os.path.join(os.path.dirname(args.injection) or '.', 'injection_per_object.csv')
    inj_obj = pd.read_csv(inj_obj_path) if os.path.exists(inj_obj_path) else None
    syst = {}
    if args.syst:
        for item in args.syst.split(','):
            lab, path = item.split('=', 1)
            syst[lab.strip()] = pd.read_csv(path.strip())
    required = [x for x in args.require_syst.split(',') if x]
    problems = [f"systematics run '{x}' missing" for x in required if x not in syst]
    if inj_obj is None:
        problems.append(f"{inj_obj_path} missing (calibration errors)")
    if problems:
        msg = '; '.join(problems)
        if not args.exploratory:
            raise SystemExit(f"ERROR: {msg}. All of them are required for Table 1 "
                             "(use --exploratory only for tests).")
        print(f"WARNING (exploratory): {msg}")

    d = pd.read_csv(args.objects)
    inj = pd.read_csv(args.injection)
    zero = inj[(inj.scenario == 'zero') & (inj.centering == args.centering)].set_index('zbin')
    cons = inj[(inj.scenario == 'constant') & (inj.centering == args.centering)].set_index('zbin')
    mat = inj[(inj.scenario == 'matched') & (inj.centering == args.centering)].set_index('zbin')
    if zero.empty:
        raise SystemExit(f"No ZERO/{args.centering} rows in {args.injection}")

    v = args.variant
    d['R12'] = d[f'A1_{v}'] / d[f'A2_{v}']
    d['zb'] = pd.cut(d.z, Z_EDGES, labels=Z_CEN)
    rng = np.random.default_rng(0)
    r0 = cons.loc[Z_CEN[0], 'R12_med'] if Z_CEN[0] in cons.index else np.nan

    rows = []
    for zc in Z_CEN:
        s = d[d.zb == zc]
        if s.empty:
            continue
        row = dict(z=zc, N_total=len(s))
        for k in ('A1', 'A2', 'A3'):
            m, e, n = boot(s[f'{k}_{v}'], rng)
            f = zero.loc[zc, f'{k}_med'] if zc in zero.index else np.nan
            c, ce = corr(m, e, f)
            row.update({k: m, f'{k}_err': e, f'{k}_floor': f, f'{k}_corr': c,
                        f'{k}_corr_err': ce, f'{k}_corr_std': 2 * c,
                        f'{k}_corr_std_err': 2 * ce})
            row['N'] = n if k == 'A1' else row['N']
        row['R12'], row['R12_err'], _ = boot(s.R12, rng)
        row['R12_corr'] = row['A1_corr'] / row['A2_corr']
        row['R12_corr_err'] = row['R12_corr'] * np.hypot(row['A1_corr_err'] / row['A1_corr'],
                                                         row['A2_corr_err'] / row['A2_corr'])
        if zc in mat.index:
            # recovery of the injected (observed) amplitudes and of their ratio
            for k in ('A1', 'A2'):
                m_, f_, t_ = mat.loc[zc, f'{k}_med'], row[f'{k}_floor'], mat.loc[zc, f'{k}_true_med']
                rec = np.sqrt(max(m_ * m_ - f_ * f_, 0)) / t_ if t_ > 0 else np.nan
                row[f'{k}_rec'] = rec
                row[f'{k}_final_std'] = 2 * row[f'{k}_corr'] / rec if rec > 0 else np.nan
            tr = mat.loc[zc, 'A1_true_med'] / mat.loc[zc, 'A2_true_med']
            row['R12_bias_matched'] = mat.loc[zc, 'R12_med'] / tr
            row['R12_debiased_matched'] = row['R12'] / row['R12_bias_matched']
            # joint bootstrap: observed galaxies x injected galaxies
            fin = joint_bootstrap(s, v, inj_obj, zc, args.centering, rng, args.nboot)
            for key, val in fin.items():
                row[key] = val
            # systematics: final values with alternative matched calibrations
            devs = {'A1': [], 'A2': [], 'R12': []}
            for lab, sm in syst.items():
                alt = central_values(row, sm, zc, args.centering)
                for k in devs:
                    if np.isfinite(alt[k]):
                        devs[k].append(abs(alt[k] - {'A1': row['A1_final_std'],
                                                     'A2': row['A2_final_std'],
                                                     'R12': row['R12_debiased_matched']}[k]))
                        row[f'{k}_final_{lab}'] = alt[k]
            for k in devs:
                # NaN (not 0) without --syst runs, so a missing systematic is visible
                row[f'{k}_final_syst'] = max(devs[k]) if devs[k] else np.nan
            for k, stat in (('A1', 'A1_final_err_stat'), ('A2', 'A2_final_err_stat'),
                            ('R12', 'R12_final_err_stat')):
                row[f'{k}_final_err'] = np.hypot(row.get(stat, np.nan),
                                                 np.nan_to_num(row[f'{k}_final_syst']))
        rb = cons.loc[zc, 'R12_med'] if zc in cons.index else np.nan
        row['R12_bias'] = rb / r0 if np.isfinite(r0) else np.nan
        row['R12_debiased'] = row['R12'] / row['R12_bias']
        row['syst_runs'] = ';'.join(sorted(syst))
        row['calib_err'] = inj_obj is not None
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(f'{args.out}.csv', index=False)

    pd.set_option('display.width', 200)
    cols = ['z', 'N', 'A1', 'A1_floor', 'A1_corr', 'A2', 'A2_floor', 'A2_corr',
            'R12', 'R12_corr', 'R12_bias', 'R12_debiased']
    if 'R12_bias_matched' in t:
        cols += ['A1_rec', 'A2_rec', 'R12_bias_matched', 'R12_debiased_matched']
    show = t[[c for c in cols if c in t]]
    print(f"Variant {v}, floors from injection ZERO/{args.centering} (original normalisation)\n")
    print(show.round(3).to_string(index=False))
    a1 = t.A1_corr.values
    with np.errstate(all='ignore'):
        print(f"\nA1_corr rise, mean(z>4) / z=0.75: x{np.nanmean(a1[-2:]) / a1[0]:.2f}")
    a2 = t.A2_corr.values
    print(f"A2_corr rise, mean(z>4) / z=0.75: x{np.nanmean(a2[-2:]) / a2[0]:.2f}")
    if 'R12_debiased_matched' in t:
        v_ = t.R12_debiased_matched.dropna()
        print(f"R12_debiased_matched: mean {v_.mean():.3f}, scatter {v_.std():.3f}; "
              f"A?_rec = recovered/true amplitude for injections at the observed amplitudes")
        if 'A1_final_std' in t:
            print("PSF/noise-recovery-corrected amplitudes (conventional): A1 " +
                  ' '.join(f"{x:.3f}" for x in t.A1_final_std) + " | A2 " +
                  ' '.join(f"{x:.3f}" for x in t.A2_final_std))
        if 'A1_final_err' in t:
            print("\nFinal values, total errors (statistical + calibration [+ systematics]):")
            show2 = pd.DataFrame({
                'z': t.z, 'N': t.N, 'N_total': t.N_total,
                'A1': t.A1_final_std.round(3), 'A1_err': t.A1_final_err.round(3),
                'A1_syst': t.A1_final_syst.round(3),
                'A2': t.A2_final_std.round(3), 'A2_err': t.A2_final_err.round(3),
                'A2_syst': t.A2_final_syst.round(3),
                'A1/A2': t.R12_debiased_matched.round(3), 'err': t.R12_final_err.round(3),
                'syst': t.R12_final_syst.round(3)})
            print(show2.to_string(index=False))
    print(f"N valid / total over bins: {int(t.N.sum())} / {int(t.N_total.sum())} "
          f"({int(t.N_total.sum() - t.N.sum())} without a valid measurement)")
    print("R12_bias = A1/A2 rise a NO-EVOLUTION population shows (injection CONSTANT),"
          " relative to z=0.75;\nR12_debiased = observed median(A1/A2) / R12_bias.")

    def pm(v, e, nd=3):
        return f"${v:.{nd}f}\\pm{e:.{nd}f}$" if np.isfinite(v) and np.isfinite(e) else "--"

    def one(v):
        return f"${v:.3f}$" if np.isfinite(v) else "--"

    if 'A1_final_std' in t:
        # final table: floor- and recovery-corrected amplitudes, debiased ratio
        with open(f'{args.out}.tex', 'w') as fh:
            fh.write("\\begin{tabular}{rrccccc}\n\\hline\n")
            fh.write("$z$ & $N$ & $A_1$ & $A_2$ & $A_3^{\\rm floor}$ & "
                     "$A_1/A_2$ (obs.) & $A_1/A_2$ (corr.) \\\\\n\\hline\n")
            has_tot = 'A1_final_err' in t
            for r in t.itertuples():
                if has_tot:
                    e1, e2, er = r.A1_final_err, r.A2_final_err, r.R12_final_err
                else:
                    e1 = 2 * r.A1_corr_err / r.A1_rec if r.A1_rec > 0 else np.nan
                    e2 = 2 * r.A2_corr_err / r.A2_rec if r.A2_rec > 0 else np.nan
                    er = r.R12_err / r.R12_bias_matched if r.R12_bias_matched > 0 else np.nan
                fh.write(f"{r.z:.2f} & {r.N} & {pm(r.A1_final_std, e1)} & {pm(r.A2_final_std, e2)} & "
                         f"{pm(r.A3_corr_std, r.A3_corr_std_err)} & {pm(r.R12, r.R12_err, 2)} & "
                         f"{pm(r.R12_debiased_matched, er, 2)} \\\\\n")
            fh.write("\\hline\n\\end{tabular}\n")
            fh.write("% A1, A2: noise-floor and PSF/noise-recovery corrected (injections at the "
                     "observed amplitudes), conventional normalisation; errors: joint bootstrap of "
                     "observed and injected galaxies plus calibration systematics (if --syst).\n"
                     "% A3: floor-corrected only. A1/A2 (obs.): median per-galaxy ratio; "
                     "(corr.): divided by the ratio bias of the matched injections.\n"
                     "% N: galaxies with a valid measurement.\n")
        print(f"\nSaved {args.out}.csv and {args.out}.tex (final, recovery-corrected)")
        return

    with open(f'{args.out}.tex', 'w') as fh:
        fh.write("\\begin{tabular}{rrcccccc}\n\\hline\n")
        fh.write("$z$ & $N$ & $A_1$ & $A_2$ & $A_3$ & $A_1/A_2$ & "
                 "$A_1^{\\rm corr}$ & $A_2^{\\rm corr}$ \\\\\n\\hline\n")
        for r in t.itertuples():
            fh.write(f"{r.z:.2f} & {r.N} & {pm(r.A1_corr_std, r.A1_corr_std_err)} & "
                     f"{pm(r.A2_corr_std, r.A2_corr_std_err)} & "
                     f"{pm(r.A3_corr_std, r.A3_corr_std_err)} & {pm(r.R12, r.R12_err, 2)} & "
                     f"{one(r.A1_corr)} & {one(r.A2_corr)} \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")
        fh.write("% A_m columns 3-5: floor-corrected, conventional normalisation "
                 "(2|<I e^{im phi}>|/<I>).\n% Last two columns: floor-corrected, "
                 "pipeline normalisation (|sum I e^{im phi}|/sum I).\n")
    print(f"\nSaved {args.out}.csv and {args.out}.tex")


if __name__ == '__main__':
    main()
