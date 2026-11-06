#!/usr/bin/env python3
"""make_synthetic_root.py OUTDIR — synthetic pipeline outputs with the file layout
build_all.sh expects (3 filters, main + 4 systematics injection runs). Used only by
tests/test_gates.sh to exercise the build and its checks; no physical meaning."""
import os
import sys

import numpy as np
import pandas as pd

R = sys.argv[1]
rng = np.random.default_rng(1)
ZC = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.5, 5.5]
ED = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]
NINJ = 100                      # injected galaxies per (bin, scenario, centring)
n = 2000
z = rng.uniform(0.5, 6, n)


def objects(scale):
    d = pd.DataFrame({'idx': np.arange(n), 'z': z, 'disk_r': 9 - z, 'q': 0.8})
    for k, s in (('A1', .05), ('A2', .06), ('A3', .03)):
        d[f'{k}_V6'] = np.abs(rng.normal(s * scale * (1 + z / 3), .02))
    d.loc[::40, 'A1_V6'] = np.nan
    for i, v in enumerate(('V0', 'V1', 'V2', 'V3', 'V4', 'V5')):   # earlier pipeline steps
        for k in ('A1', 'A2', 'A3'):
            d[f'{k}_{v}'] = d[f'{k}_V6'] * (1.5 - 0.08 * i)
    return d


def boot_med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.median(x)


for filt, sc in (('f444w', 1.0), ('f277w', 1.03), ('f150w', 1.3)):
    d = objects(sc)
    d.to_csv(f'{R}/corrected_disk_{filt}.csv', index=False)
    if filt == 'f444w':
        zb = pd.cut(d.z, ED, labels=ZC)
        rows = []
        for zc in ZC:
            s = d[zb == zc]
            r = {'z': zc, 'N': int(s.A1_V6.notna().sum())}
            for k in ('A1', 'A2'):
                r[f'{k}_V6'], r[f'{k}_V6_err'] = boot_med(s[f'{k}_V6']), 0.001
            r['R12_V6'], r['R12_V6_err'] = boot_med(s.A1_V6 / s.A2_V6), 0.01
            rows.append(r)
        pd.DataFrame(rows).to_csv(f'{R}/corrected_disk_f444w_summary.csv', index=False)


def injection(path, amp):
    os.makedirs(path, exist_ok=True)
    rows, obj = [], []
    for zc in ZC:
        for sc in ('zero', 'matched'):
            for c in ('catalog', 'true'):
                o = pd.DataFrame({'zbin': [zc] * NINJ, 'scenario': sc, 'centering': c,
                                  're_pix': 9 - zc, 'q': 0.8, 'n': 1.0, 'snr': 100.0})
                for k, t in (('A1', .06), ('A2', .07), ('A3', .03)):
                    tr = np.full(NINJ, t * amp if sc == 'matched' else 0.0)
                    o[f'{k}_true'] = tr
                    o[k] = np.abs(tr * 0.85 + rng.normal(0, .015 + zc * .004, NINJ))
                o['R12'] = o.A1 / o.A2
                obj.append(o)
                r = {'zbin': zc, 'scenario': sc, 'centering': c}
                for k in ('A1', 'A2', 'A3'):
                    r[f'{k}_med'], r[f'{k}_true_med'] = o[k].median(), o[f'{k}_true'].median()
                r['R12_med'] = (o.A1 / o.A2).median()
                rows.append(r)
    pd.DataFrame(rows).to_csv(f'{path}/injection_summary.csv', index=False)
    pd.concat(obj).to_csv(f'{path}/injection_per_object.csv', index=False)


for d, amp in (('inj_f444w_m', 1), ('inj_f277w_m', 1), ('inj_f150w_m', 1),
               ('inj_f444w_scale0.5', .5), ('inj_f444w_scale1.5', 1.5),
               ('inj_f444w_rising', 1.05), ('inj_f444w_falling', .95)):
    injection(f'{R}/{d}', amp)
print(f"synthetic outputs in {R}")
