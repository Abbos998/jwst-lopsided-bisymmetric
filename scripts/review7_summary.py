#!/usr/bin/env python3
"""
review7_summary.py — collect the robustness tests requested in the 7th review
(run by run_review7.sh) into data/review7_*.csv and a short report:

  1. self-consistent calibration: injections at the FINAL Table 1 amplitudes
     (nominal + 4 systematics runs) -> Table 1 recomputed; change per bin in
     units of the total error
  2. clumpy galaxies: recovery fractions with Gaussian clumps vs smooth models,
     and the implied change of the final amplitudes
  3. distribution of amplitudes: galaxies with log-normal scatter of their
     amplitudes; median recovered by the Table 1 chain vs true median
  4. radius of the annulus: Table 1 recomputed with the annulus scaled by 0.6
     and 1.68 (scale length <-> half-light radius), with matching injections
  (mass: see mass_test.py; r_d definition: diagnose_sample.py)

Usage:  python3 review7_summary.py --root ~/Downloads/pasp_v3 --pkg ..
"""
import argparse
import os

import numpy as np
import pandas as pd


def load(p):
    return pd.read_csv(p) if os.path.exists(p) else None


def cal(summ, centering='catalog'):
    """floor and recovery per bin (pipeline normalisation) from an injection summary."""
    s = summ[summ.centering == centering]
    z0 = s[s.scenario == 'zero'].set_index('zbin')
    mt = s[s.scenario == 'matched'].set_index('zbin')
    out = {}
    for zc in mt.index:
        r = {}
        for k in ('A1', 'A2'):
            f, m, t = z0.loc[zc, f'{k}_med'], mt.loc[zc, f'{k}_med'], mt.loc[zc, f'{k}_true_med']
            r[k] = (f, np.sqrt(max(m * m - f * f, 0)) / t)
        r['b'] = mt.loc[zc, 'R12_med'] / (mt.loc[zc, 'A1_true_med'] / mt.loc[zc, 'A2_true_med'])
        out[zc] = r
    return out


def growth(t, col):
    return t[t.z > 4][col].mean() / t[np.isclose(t.z, 0.75)][col].iloc[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--pkg', default='..')
    a = ap.parse_args()
    R, D = a.root, os.path.join(a.pkg, 'data')
    rep = ["# Robustness tests (7th review)"]
    t1 = load(f'{R}/table1_disk_final.csv')
    if t1 is None:
        raise SystemExit("table1_disk_final.csv missing")

    # 1. self-consistent calibration
    sc = load(f'{R}/table1_disk_sc.csv')
    if sc is not None:
        rows = []
        for (_, r0), (_, r1) in zip(t1.iterrows(), sc.iterrows()):
            row = {'z': r0.z}
            for k, kk in (('A1', 'A1_final_std'), ('A2', 'A2_final_std'), ('R12', 'R12_debiased_matched')):
                e = {'A1': 'A1_final_err', 'A2': 'A2_final_err', 'R12': 'R12_final_err'}[k]
                row[f'{k}_table1'], row[f'{k}_selfcons'] = r0[kk], r1[kk]
                row[f'{k}_selfcons_err'] = r1[e]
                row[f'{k}_shift_sigma'] = (r1[kk] - r0[kk]) / r0[e]
            rows.append(row)
        x = pd.DataFrame(rows)
        x.to_csv(f'{D}/review7_selfconsistent.csv', index=False)
        rep += ["\n## 1. Self-consistent calibration (injection at the final amplitudes)\n",
                x.round(3).to_string(index=False), "",
                f"Growth z~0.75 -> z>4: Table 1 A1 x{growth(t1, 'A1_final_std'):.2f}, "
                f"A2 x{growth(t1, 'A2_final_std'):.2f}; self-consistent A1 x{growth(sc, 'A1_final_std'):.2f}, "
                f"A2 x{growth(sc, 'A2_final_std'):.2f}. "
                f"Largest shift: {np.nanmax(np.abs(x.filter(like='shift_sigma').to_numpy())):.2f} sigma.\n"]
    ref_summ = load(f'{R}/inj_f444w_sc_m/injection_summary.csv')
    if ref_summ is None:
        ref_summ = load(f'{R}/inj_f444w_m/injection_summary.csv')
    ref = cal(ref_summ)
    base = sc if sc is not None else t1

    # 2. clumps
    rows = []
    for lab in ('clumps4_f0.2', 'clumps8_f0.3'):
        s = load(f'{R}/inj_f444w_{lab}/injection_summary.csv')
        if s is None:
            continue
        c = cal(s)
        for zc in sorted(c):
            row = {'model': lab, 'z': zc}
            for k in ('A1', 'A2'):
                row[f'rec_{k}_smooth'], row[f'rec_{k}_clumpy'] = ref[zc][k][1], c[zc][k][1]
                fin = base.set_index('z').loc[zc, f'{k}_final_std']
                row[f'{k}_final_with_clumpy_cal'] = fin * ref[zc][k][1] / c[zc][k][1]
            row['b_smooth'], row['b_clumpy'] = ref[zc]['b'], c[zc]['b']
            rows.append(row)
    if rows:
        x = pd.DataFrame(rows)
        x.to_csv(f'{D}/review7_clumps.csv', index=False)
        rep += ["\n## 2. Clumpy models (recovery and ratio bias vs smooth models)\n",
                x.round(3).to_string(index=False), ""]
        for lab, g in x.groupby('model'):
            rep.append(f"{lab}: growth with clumpy calibration A1 x{growth(g, 'A1_final_with_clumpy_cal'):.2f}, "
                       f"A2 x{growth(g, 'A2_final_with_clumpy_cal'):.2f}")
        rep.append("")

    # 3. spread of amplitudes
    rows = []
    for lab in ('spread0.15', 'spread0.30'):
        o = load(f'{R}/inj_f444w_{lab}/injection_per_object.csv')
        if o is None:
            continue
        o = o[(o.scenario == 'matched') & (o.centering == 'catalog')]
        for zc, g in o.groupby('zbin'):
            row = {'spread': lab, 'z': zc, 'N': len(g)}
            for k in ('A1', 'A2'):
                f, rec = ref[zc][k]
                m = g[k].median()
                row[f'{k}_true_median'] = 2 * g[f'{k}_true'].median()
                row[f'{k}_recovered'] = 2 * np.sqrt(max(m * m - f * f, 0)) / rec
                row[f'{k}_rel_error'] = row[f'{k}_recovered'] / row[f'{k}_true_median'] - 1
            rt = (g.A1_true / g.A2_true).median()
            row['R12_rel_error'] = g.R12.median() / ref[zc]['b'] / rt - 1
            rows.append(row)
    if rows:
        x = pd.DataFrame(rows)
        x.to_csv(f'{D}/review7_spread.csv', index=False)
        rep += ["\n## 3. Galaxy-to-galaxy spread of amplitudes: recovered vs true median\n",
                x[['spread', 'z', 'A1_rel_error', 'A2_rel_error', 'R12_rel_error']].round(3).to_string(index=False),
                "", x.groupby('spread')[['A1_rel_error', 'A2_rel_error', 'R12_rel_error']].mean().round(3).to_string(), ""]

    # 4. annulus radius
    rows = []
    for s in ('0.6', '1.68'):
        t = load(f'{R}/table1_disk_r{s}.csv')
        if t is None:
            continue
        for (_, r0), (_, r1) in zip(base.iterrows(), t.iterrows()):
            rows.append({'ring_scale': float(s), 'z': r0.z, 'N_nominal': r0.N, 'N': r1.N,
                         'A1_nominal': r0.A1_final_std, 'A1': r1.A1_final_std,
                         'A2_nominal': r0.A2_final_std, 'A2': r1.A2_final_std,
                         'R12_nominal': r0.R12_debiased_matched, 'R12': r1.R12_debiased_matched})
    if rows:
        x = pd.DataFrame(rows)
        x.to_csv(f'{D}/review7_annulus.csv', index=False)
        rep += ["\n## 4. Annulus radius (0.5-1.5 r_d scaled by 0.6 and 1.68)\n",
                x.round(3).to_string(index=False), ""]
        for s, g in x.groupby('ring_scale'):
            rep.append(f"ring x{s}: growth A1 x{growth(g, 'A1'):.2f} (nominal x{growth(g, 'A1_nominal'):.2f}), "
                       f"A2 x{growth(g, 'A2'):.2f} (nominal x{growth(g, 'A2_nominal'):.2f}); "
                       f"A1/A2 at z~0.75 {g.R12.iloc[0]:.2f}, mean z>=1.75 {g[g.z >= 1.75].R12.mean():.2f}")
        rep.append("")

    # mass test
    for f in ('mass_test.csv', 'mass_test_regression.csv'):
        if os.path.exists(f'{D}/{f}'):
            rep.append(f"Mass test: see data/{f}")
    out = os.path.join(a.pkg, 'logs', 'review7_summary.md')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
