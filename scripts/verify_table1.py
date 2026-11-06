#!/usr/bin/env python3
"""
verify_table1.py — recompute Table 1 from data/table1_disks_F444W.csv alone.

For each redshift bin, from columns of the same table and in the same
(pipeline) normalisation:
    A_floorcorr = sqrt(A_meas^2 - A_floor^2)
    A_final     = 2 * A_floorcorr / recovery          (conventional norm.)
    A1/A2_corr  = A1/A2_obs / A1/A2_bias
and compare with the stored final values.

The comparison is strict and depends on the precision of the inputs:
  * full precision (as written by build_all.sh): the recomputed value must
    agree with the stored one to a relative 1e-6;
  * inputs rounded to 3 decimals (transcribed tables): every input is only
    known to +-0.0005, so the recomputed value is an interval obtained by
    moving each input to the ends of its rounding range (A_final grows with
    A_meas and falls with A_floor and recovery). The stored value, itself
    rounded, must lie inside the interval widened by 0.0005.
The largest deviation is reported in both cases.

--submission also requires the statistical and systematic errors, N_total,
the calibration bootstrap and all four systematics runs (scale0.5, scale1.5,
rising, falling), as recorded by make_table1.py in syst_runs.

Note: the floors in injection_calibration_F444W.csv are given in both
normalisations (_pipe, _conv); mixing _conv floors with _pipe amplitudes
gives wrong results.

Usage:  python3 verify_table1.py ../data/table1_disks_F444W.csv [--submission]
"""

import sys

import numpy as np
import pandas as pd

H = 5e-4          # half of the last digit of a value rounded to 3 decimals
REL = 1e-6        # tolerance for full-precision inputs
REQUIRED_SYST = ['scale0.5', 'scale1.5', 'rising', 'falling']


def is_rounded(t, cols):
    x = t[cols].to_numpy(float).ravel()
    x = x[np.isfinite(x)]
    return np.allclose(x, np.round(x, 3), rtol=0, atol=1e-9)


def final_amp(m, f, rec):
    return 2 * np.sqrt(np.maximum(m * m - f * f, 0)) / rec


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    submission = '--submission' in sys.argv
    path = args[0] if args else '../data/table1_disks_F444W.csv'
    t = pd.read_csv(path)
    inputs = [f'{k}_{c}' for k in ('A1', 'A2') for c in ('meas_pipe', 'floor_pipe', 'recovery')] + \
             ['A1A2_obs', 'A1A2_bias']
    rounded = is_rounded(t, inputs)
    print(f"input precision: {'rounded to 3 decimals -> interval check (+-0.0005 per input)' if rounded else 'full -> relative tolerance 1e-6'}\n")
    print(f"{'z':>5} | {'A1 recomputed':>19} {'stored':>7} | {'A2 recomputed':>19} {'stored':>7} | "
          f"{'A1/A2 recomputed':>19} {'stored':>7}")
    bad, worst = 0, 0.0
    for r in t.itertuples():
        cells, flag = [], ''
        for k in ('A1', 'A2', 'R'):
            if k == 'R':
                o, b, s = r.A1A2_obs, r.A1A2_bias, r.A1A2_corrected
                val = o / b
                lo, hi = (o - H) / (b + H), (o + H) / (b - H)
            else:
                m, f, rec = (getattr(r, f'{k}_meas_pipe'), getattr(r, f'{k}_floor_pipe'),
                             getattr(r, f'{k}_recovery'))
                s = getattr(r, f'{k}_final_conv')
                val = final_amp(m, f, rec)
                lo, hi = final_amp(m - H, f + H, rec + H), final_amp(m + H, f - H, rec - H)
            worst = max(worst, abs(val - s))
            ok = (lo - H <= s <= hi + H) if rounded else abs(val - s) <= REL * max(abs(s), 1e-12)
            if not ok:
                flag = '  <-- MISMATCH'
                bad += 1
            cells.append(f"{val:.4f} [{lo:.3f},{hi:.3f}] {s:7.3f}" if rounded else
                         f"{val:19.6f} {s:7.4f}")
        print(f"{r.z:5.2f} | " + ' | '.join(cells) + flag)
    print(f"\nlargest |recomputed - stored| = {worst:.4f}")
    # total errors = quadrature sum of the statistical+calibration and systematic parts
    parts = [('A1_final_err_conv', 'A1_final_err_stat_conv', 'A1_final_syst_conv'),
             ('A2_final_err_conv', 'A2_final_err_stat_conv', 'A2_final_syst_conv'),
             ('A1A2_corrected_err', 'A1A2_corrected_err_stat', 'A1A2_corrected_syst')]
    if all(c in t for p in parts for c in p):
        dq = max(float(np.nanmax(np.abs(t[tot] - np.hypot(t[st], t[sy])))) for tot, st, sy in parts)
        okq = dq <= (1e-3 if rounded else 1e-9)
        bad += 0 if okq else 1
        print(f"total error = sqrt(stat_calib^2 + syst^2): max deviation {dq:.2e} "
              f"({'ok' if okq else 'MISMATCH'})")

    n_valid = int(t.N_valid.sum())
    has_ntot = 'N_total' in t and t.N_total.notna().all()
    print(f"N valid = {n_valid}" + (f", N total = {int(t.N_total.sum())} "
          f"({int(t.N_total.sum()) - n_valid} without a valid measurement)" if has_ntot
          else ", N total not recorded"))
    err_cols = ['A1_final_err_conv', 'A1_final_err_stat_conv', 'A1_final_syst_conv',
                'A2_final_err_conv', 'A2_final_err_stat_conv', 'A2_final_syst_conv',
                'A1A2_corrected_err', 'A1A2_corrected_err_stat', 'A1A2_corrected_syst']
    missing = [c for c in err_cols if c not in t or t[c].isna().any()]
    if missing:
        print("errors missing: " + ', '.join(missing))
    runs = set()
    if 'syst_runs' in t and t.syst_runs.notna().all():
        runs = set.intersection(*(set(str(x).split(';')) for x in t.syst_runs))
    lacking = [x for x in REQUIRED_SYST if x not in runs]
    if lacking:
        missing.append('systematics runs ' + ','.join(lacking))
        print("systematics runs missing: " + ', '.join(lacking))
    if not ('calib_err_included' in t and t.calib_err_included.astype(str).eq('True').all()):
        missing.append('calibration bootstrap')
        print("calibration (injection) bootstrap not included")
    print("All values reproduced." if bad == 0 else f"{bad} mismatches.")
    if submission:
        ready = bad == 0 and not rounded and has_ntot and not missing
        print("SUBMISSION CHECK: " + ("passed" if ready else
              "FAILED (needs full-precision values, errors and N_total from build_all.sh)"))
        sys.exit(0 if ready else 1)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
