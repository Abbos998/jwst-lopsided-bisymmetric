#!/usr/bin/env python3
"""
plot_fig3_steps.py — Fig. 3: median A1, A2 and A1/A2 after each pipeline step (V0-V6),
computed directly from the per-galaxy measurements of the adopted run.

Usage:
  python3 plot_fig3_steps.py ~/Downloads/pasp_v3/corrected_disk_f444w.csv Fig3_steps
  -> Fig3_steps.pdf and Fig3_steps.png (300 dpi)
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Z_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
Z_CEN = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50]
STEPS = [('V0', 'V0 original'),
         ('V1', 'V1 + WCS centre, field offset'),
         ('V2', 'V2 + background'),
         ('V3', 'V3 + deprojection'),
         ('V4', 'V4 + masking'),
         ('V5', 'V5 + $0.5$–$1.5\\,r_d$ annulus'),
         ('V6', 'V6 + Sérsic PA (adopted)')]

src = sys.argv[1] if len(sys.argv) > 1 else 'corrected_disk_f444w.csv'
out = sys.argv[2] if len(sys.argv) > 2 else 'Fig3_steps'
d = pd.read_csv(src)
zb = pd.cut(d.z, Z_EDGES, labels=Z_CEN).astype(float)


def med(x):
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    return x.median() if len(x) else np.nan


# width 13 in and tight_layout(w_pad) keep the y-labels of panels 2 and 3 out of the panel on their left
fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
cmap = plt.get_cmap('viridis')
for i, (v, label) in enumerate(STEPS):
    if f'A1_{v}' not in d:
        continue
    vals = {'A1': [], 'A2': [], 'R': []}
    for zc in Z_CEN:
        g = d[np.isclose(zb, zc)]
        vals['A1'].append(med(g[f'A1_{v}']))
        vals['A2'].append(med(g[f'A2_{v}']))
        vals['R'].append(med(g[f'A1_{v}'] / g[f'A2_{v}']))
    last = i == len(STEPS) - 1
    lw = 2.4 if i in (0, len(STEPS) - 1) else 1.1
    for ax, k in zip(axs, ('A1', 'A2', 'R')):
        ax.plot(Z_CEN, vals[k], 'o-', ms=3.5 if last else 3, lw=lw, color=cmap(i / (len(STEPS) - 1)),
                label=label, zorder=3 if last else 2)
for ax, k in zip(axs, ('$A_1$', '$A_2$', '$A_1/A_2$')):
    ax.set(xlabel='Redshift $z$', ylabel=f'median {k} (pipeline norm.)')
axs[0].legend(fontsize=7.5, loc='upper left', frameon=False)
fig.tight_layout(w_pad=2.5)
for ext in ('pdf', 'png'):
    fig.savefig(f'{out}.{ext}', dpi=300)
print(f"wrote {out}.pdf, {out}.png")
