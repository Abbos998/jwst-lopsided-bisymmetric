#!/usr/bin/env python3
"""
make_paper_figures.py — all figures of the revised manuscript from the
pipeline outputs (run from the scripts/ directory; outputs live in --root).

  fig_amplitudes.pdf   A1, A2 vs z: final (floor + recovery corrected),
                       floor-only (lower limits), early types
  fig_ratio.pdf        A1/A2 vs z: observed, corrected; early types
  fig_steps.pdf        effect of each pipeline correction V0..V6
  fig_restframe.pdf    A1, A2 at fixed rest-frame wavelength vs F444W only
  fig_injection.pdf    noise floor and recovery fraction vs z
  fig_sfr.pdf          A1 of star-forming vs quiescent disks
  fig_astrometry.pdf   star centroid - catalogue position per field

Inputs (defaults relative to --root = ..):
  table1_disk_final.csv, table1_ell_v7.csv, corrected_disk_f444w_summary.csv,
  restframe_disk_m.csv, inj_f444w_m/injection_summary.csv,
  sfr_split_snr100.csv, psf/F444W_stars.csv
A missing input skips that figure.

Usage:
    python3 make_paper_figures.py --root .. --out ../paper/figs
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'legend.fontsize': 8,
                     'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'savefig.bbox': 'tight'})
C_DISK, C_ELL, C_GREY = '#1f5aa6', '#c0392b', '#7f7f7f'


def load(root, name):
    p = os.path.join(root, name)
    if not os.path.exists(p):
        print(f"  skip: {p} not found")
        return None
    return pd.read_csv(p)


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf/png")


def final_errors(t):
    """Total errors from make_table1.py (statistical + calibration [+ syst]);
    older tables without them: statistical only."""
    if 'A1_final_err' in t:
        return t.A1_final_err, t.A2_final_err
    return 2 * t.A1_corr_err / t.A1_rec, 2 * t.A2_corr_err / t.A2_rec


def fig_amplitudes(root, out):
    t = load(root, 'table1_disk_final.csv')
    if t is None:
        return
    e = load(root, 'table1_ell_v7.csv')
    e1, e2 = final_errors(t)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))   # own y-range: A2 > A1 at high z
    for ax, k, err in ((axs[0], 'A1', e1), (axs[1], 'A2', e2)):
        ax.errorbar(t.z, t[f'{k}_final_std'], err, fmt='o-', color=C_DISK, capsize=2,
                    label='disks, corrected')
        ax.errorbar(t.z, t[f'{k}_corr_std'], t[f'{k}_corr_std_err'], fmt='o--', mfc='none',
                    color=C_DISK, capsize=2, alpha=0.7, label='disks, floor only (lower limit)')
        if e is not None:
            ax.errorbar(e.z + 0.05, e[f'{k}_corr_std'], e[f'{k}_corr_std_err'], fmt='s--',
                        mfc='none', color=C_ELL, capsize=2, alpha=0.8,
                        label='early types, floor only')
        ax.set(xlabel='Redshift $z$', ylabel=f'median $A_{k[1]}$')
        ax.set_ylim(0, ax.get_ylim()[1])      # after all data are drawn, so nothing is clipped
        ax.set_title({'A1': '(a) $m=1$', 'A2': '(b) $m=2$'}[k], fontsize=10)
    axs[0].legend(loc='upper left')
    save(fig, out, 'fig_amplitudes')


def fig_ratio(root, out):
    t = load(root, 'table1_disk_final.csv')
    if t is None:
        return
    e = load(root, 'table1_ell_v7.csv')
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.errorbar(t.z, t.R12, t.R12_err, fmt='o--', mfc='none', color=C_DISK, capsize=2,
                label='disks, observed')
    er = t.R12_final_err if 'R12_final_err' in t else t.R12_err / t.R12_bias_matched
    ax.errorbar(t.z, t.R12_debiased_matched, er, fmt='o-', color=C_DISK, capsize=2,
                label='disks, corrected')
    if e is not None and 'R12_debiased' in e:
        ax.errorbar(e.z + 0.05, e.R12_debiased, e.R12_err / e.R12_bias, fmt='s-', color=C_ELL,
                    capsize=2, label='early types, corrected')
    ax.axhline(1, color=C_GREY, ls=':', lw=1)
    ax.set(xlabel='Redshift $z$', ylabel='median $A_1/A_2$', ylim=(0.3, 1.4))
    ax.legend(loc='lower right')
    save(fig, out, 'fig_ratio')


def fig_steps(root, out, data=None):
    """Median A1, A2, A1/A2 after each pipeline step, from data/correction_steps_V0_V6.csv
    (the same table as Table 2; the last step is the adopted pipeline)."""
    p = os.path.join(data, 'correction_steps_V0_V6.csv') if data else None
    if not p or not os.path.exists(p):
        print(f"  skip fig_steps: {p} not found")
        return
    s = pd.read_csv(p)
    steps = list(dict.fromkeys(s.step))
    labels = {'V0 original': 'V0 original',
              '+ WCS centre and field offset': 'V1 + WCS centre, field offset',
              '+ background': 'V2 + background',
              '+ deprojection': 'V3 + deprojection',
              '+ masking': 'V4 + masking',
              '+ 0.5-1.5 r_d annulus': 'V5 + $0.5$–$1.5\\,r_d$ annulus',
              '+ Sersic PA (adopted)': 'V6 + Sérsic PA (adopted)'}
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
    cmap = plt.get_cmap('viridis')
    for i, st in enumerate(steps):
        g = s[s.step == st].sort_values('z')
        last = i == len(steps) - 1
        col = cmap(i / max(len(steps) - 1, 1))
        lw = 2.4 if i in (0, len(steps) - 1) else 1.1
        for ax, k in zip(axs, ('A1_pipe', 'A2_pipe', 'A1A2')):
            ax.plot(g.z, g[k], 'o-', ms=3.5 if last else 3, lw=lw, color=col,
                    label=labels.get(st, st), zorder=3 if last else 2)
    for ax, k in zip(axs, ('$A_1$', '$A_2$', '$A_1/A_2$')):
        ax.set(xlabel='Redshift $z$', ylabel=f'median {k} (pipeline norm.)')
    axs[0].legend(fontsize=7.5, loc='upper left', frameon=False)
    fig.tight_layout(w_pad=2.5)
    save(fig, out, 'fig_steps')


def fig_restframe(root, out):
    r = load(root, 'restframe_disk_m.csv')
    if r is None:
        return
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    styles = {}
    for i, (label, g) in enumerate(r.groupby('set', sort=False)):
        st = ('o--', C_GREY) if 'only' in label else (['s-', 'D-'][i % 2], [C_DISK, C_ELL][i % 2])
        styles[label] = st
        for ax, k in zip(axs, ('A1', 'A2', 'R12')):
            ax.errorbar(g.z + 0.04 * i, g[k], g[f'{k}_err'], fmt=st[0], color=st[1], capsize=2,
                        label=label.replace('um', r'$\mu$m'))
    for ax, k in zip(axs, ('$A_1$', '$A_2$', '$A_1/A_2$')):
        ax.set(xlabel='Redshift $z$', ylabel=k)
    axs[0].legend()
    save(fig, out, 'fig_restframe')


def fig_injection(root, out):
    s = load(root, 'inj_f444w_m/injection_summary.csv')
    if s is None:
        return
    z0 = s[(s.scenario == 'zero') & (s.centering == 'catalog')]
    m = s[(s.scenario == 'matched') & (s.centering == 'catalog')]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, mk in (('A1', 'o'), ('A2', 's')):
        axs[0].plot(z0.zbin, 2 * z0[f'{k}_med'], mk + '-', label=f'$A_{k[1]}$ floor')
        if len(m):
            f = z0.set_index('zbin')[f'{k}_med'].reindex(m.zbin).values
            rec = np.sqrt(np.clip(m[f'{k}_med'] ** 2 - f ** 2, 0, None)) / m[f'{k}_true_med']
            axs[1].plot(m.zbin, rec, mk + '-', label=f'$A_{k[1]}$')
    axs[0].set(xlabel='Redshift $z$', ylabel='noise floor (conventional)')
    axs[1].set(xlabel='Redshift $z$', ylabel='recovered / injected', ylim=(0, 1.2))
    axs[1].axhline(1, color=C_GREY, ls=':')
    for a in axs:
        a.legend()
    save(fig, out, 'fig_injection')


def fig_sfr(root, out):
    t = load(root, 'sfr_split_snr100.csv')
    if t is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[0].errorbar(t.z, t.A1_SF, t.A1_SF_err, fmt='o-', color=C_DISK, capsize=2,
                    label='star-forming')
    axs[0].errorbar(t.z + 0.05, t.A1_Q, t.A1_Q_err, fmt='s-', color=C_ELL, capsize=2,
                    label='quiescent')
    ok = t.flag.fillna('') == ''
    axs[1].errorbar(t.z[ok], t.ratio[ok], t.ratio_err[ok], fmt='D', color='k', capsize=2)
    axs[1].errorbar(t.z[~ok], t.ratio[~ok], t.ratio_err[~ok], fmt='D', mfc='none', color='k',
                    capsize=2, label='$N<15$')
    axs[1].axhline(1, color=C_GREY, ls=':')
    axs[0].set(xlabel='Redshift $z$', ylabel='median $A_1$')
    axs[1].set(xlabel='Redshift $z$', ylabel='$A_1^{\\rm SF}/A_1^{\\rm Q}$')
    for a in axs:
        a.legend()
    save(fig, out, 'fig_sfr')


def fig_astrometry(root, out):
    s = load(root, 'psf/F444W_stars.csv')
    if s is None:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for f, g in s.groupby('field'):
        ax.scatter(g.dx, g.dy, s=8, alpha=0.7, label=f'{f.split("-grizli")[0]} ({len(g)})')
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.axvline(0, color=C_GREY, lw=0.8)
    ax.set(xlabel='star centroid $-$ catalogue position, $x$ [pix]',
           ylabel='$y$ [pix]', xlim=(-1, 1), ylim=(-1, 1), aspect='equal')
    ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.02, 1))
    save(fig, out, 'fig_astrometry')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='..', help='directory with the pipeline outputs')
    ap.add_argument('--out', default='../paper/figs')
    ap.add_argument('--data', help='package data/ directory (Fig. 3 is drawn from '
                    'correction_steps_V0_V6.csv, the same table as Table 2)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in (fig_amplitudes, fig_ratio, fig_restframe, fig_injection, fig_sfr, fig_astrometry):
        f(args.root, args.out)
    fig_steps(args.root, args.out, args.data)


if __name__ == '__main__':
    main()
