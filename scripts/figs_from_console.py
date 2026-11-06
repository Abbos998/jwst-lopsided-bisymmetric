#!/usr/bin/env python3
"""
figs_from_console.py — the manuscript figures drawn from the numbers printed
by the pipeline runs (console outputs of make_table1.py, run_corrected_am.py,
restframe_test.py, fix_sfr.py and star_psf_astrometry.py). It needs no data
files, so main.pdf can be built anywhere.

For the final figures with exact per-bin bootstrap errors, run instead
    scripts/make_paper_figures.py --root <outputs> --out paper/figs
The only approximation here: Table 1 / Fig. 1 error bars use the relative
bootstrap errors of the 'F444W only' run of restframe_test.py (same galaxies,
same bootstrap), because make_table1.py does not print them.

Usage:  python3 figs_from_console.py   (writes ../figures/ and ../tex/Fig*.pdf|png)
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'legend.fontsize': 8,
                     'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'savefig.bbox': 'tight'})
C_DISK, C_ELL, C_GREY = '#1f5aa6', '#c0392b', '#7f7f7f'
Z = np.array([0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50])
A = np.array

# --- make_table1.py, table1_disk_final (F444W, matched injections) ---------
D_A1_FINAL = A([0.097, 0.120, 0.154, 0.142, 0.175, 0.179, 0.200, 0.273, 0.287])
D_A2_FINAL = A([0.134, 0.155, 0.171, 0.177, 0.189, 0.199, 0.244, 0.395, 0.316])
D_A1_FLOOR = 2 * A([0.044, 0.051, 0.064, 0.062, 0.069, 0.071, 0.077, 0.108, 0.116])  # A1_corr
D_A2_FLOOR = 2 * A([0.051, 0.054, 0.056, 0.058, 0.060, 0.061, 0.079, 0.108, 0.095])
D_R12_OBS = A([0.852, 0.914, 1.005, 1.053, 1.135, 1.057, 1.044, 1.087, 1.106])
D_R12_COR = A([0.743, 0.796, 0.867, 0.929, 0.987, 0.902, 0.926, 0.871, 0.901])
INJ_A1_FLOOR = 2 * A([0.015, 0.017, 0.021, 0.022, 0.027, 0.029, 0.032, 0.042, 0.039])
INJ_A2_FLOOR = 2 * A([0.015, 0.019, 0.021, 0.026, 0.027, 0.030, 0.031, 0.044, 0.041])
INJ_A1_REC = A([0.902, 0.841, 0.826, 0.875, 0.787, 0.792, 0.773, 0.786, 0.807])
INJ_A2_REC = A([0.763, 0.699, 0.660, 0.658, 0.634, 0.616, 0.648, 0.546, 0.602])
# --- make_table1.py, table1_ell_v7 (early types, floor only) ----------------
E_A1_FLOOR = 2 * A([0.037, 0.040, 0.049, 0.059, 0.064, 0.067, 0.079, 0.100, 0.114])
E_A2_FLOOR = 2 * A([0.073, 0.089, 0.101, 0.102, 0.135, 0.136, 0.138, 0.163, 0.155])
E_R12_COR = A([0.502, 0.448, 0.467, 0.506, 0.422, 0.437, 0.507, 0.455, 0.587])

# --- restframe_test.py, restframe_disk_m (value, error) ---------------------
RF = {
    'F444W only': dict(
        A1=A([.104, .122, .160, .152, .174, .187, .202, .283, .305]),
        A1e=A([.003, .003, .004, .005, .009, .009, .012, .018, .037]),
        A2=A([.132, .151, .167, .173, .191, .203, .239, .385, .336]),
        A2e=A([.004, .006, .005, .010, .010, .015, .016, .037, .034]),
        R12=A([.787, .809, .958, .876, .914, .921, .846, .734, .907]),
        R12e=A([.032, .036, .037, .059, .065, .082, .076, .084, .143])),
    r'$\lambda_{\rm rest}=0.7\,\mu$m': dict(
        A1=A([.112, .143, .189, .168, .159, .188, .186, .279, .305]),
        A1e=A([.004, .004, .009, .006, .008, .011, .013, .019, .036]),
        A2=A([.148, .190, .242, .169, .192, .235, .270, .386, .336]),
        A2e=A([.005, .006, .010, .006, .013, .013, .029, .035, .034]),
        R12=A([.755, .752, .780, .992, .830, .801, .686, .723, .907]),
        R12e=A([.036, .031, .049, .051, .070, .065, .088, .083, .141])),
    r'$\lambda_{\rm rest}=1.0\,\mu$m': dict(
        A1=A([.106, .121, .140, .167, .175, .187, .202, .279, np.nan]),
        A1e=A([.003, .005, .004, .006, .009, .010, .013, .023, np.nan]),
        A2=A([.145, .144, .170, .169, .191, .203, .239, .381, np.nan]),
        A2e=A([.005, .007, .005, .007, .010, .014, .017, .046, np.nan]),
        R12=A([.731, .843, .825, .989, .915, .921, .846, .732, np.nan]),
        R12e=A([.034, .055, .033, .052, .069, .081, .080, .108, np.nan])),
}
# relative bootstrap errors of the F444W-only run, applied to Table 1 values
REL_A1 = RF['F444W only']['A1e'] / RF['F444W only']['A1']
REL_A2 = RF['F444W only']['A2e'] / RF['F444W only']['A2']
REL_R12 = RF['F444W only']['R12e'] / RF['F444W only']['R12']

# --- run_corrected_am.py (F444W, --pa_fits), pipeline normalisation ---------
STEPS = {
    'V0 original': ([.078, .090, .103, .101, .109, .121, .128, .152, .128],
                    [.117, .113, .116, .115, .116, .114, .134, .127, .114],
                    [.749, .874, .991, 1.028, 1.037, 1.113, 1.108, 1.115, 1.090]),
    '+ WCS centre': ([.054, .062, .073, .070, .078, .080, .104, .143, .112],
                     [.119, .117, .118, .120, .125, .114, .140, .137, .121],
                     [.534, .603, .762, .686, .760, .836, .895, 1.069, .992]),
    '+ background': ([.057, .066, .080, .080, .089, .099, .125, .228, .171],
                     [.126, .123, .129, .132, .142, .136, .173, .202, .169],
                     [.531, .605, .770, .704, .773, .842, .943, 1.087, 1.001]),
    '+ deprojection': ([.059, .066, .080, .077, .091, .099, .119, .213, .163],
                       [.063, .064, .072, .080, .082, .086, .102, .148, .101],
                       [1.013, 1.070, 1.175, 1.079, 1.287, 1.195, 1.294, 1.325, 1.395]),
    '+ masking': ([.046, .053, .063, .064, .066, .071, .086, .121, .123],
                  [.061, .062, .066, .073, .072, .077, .088, .116, .099],
                  [.757, .828, .906, .879, .995, .860, .997, 1.045, 1.184]),
    r'+ $0.5$–$1.5\,r_d$': ([.051, .059, .069, .068, .074, .079, .084, .122, .116],
                            [.072, .074, .077, .084, .082, .089, .104, .140, .103],
                            [.745, .802, .853, .768, .890, .895, .836, .826, .983]),
    '+ Sérsic PA': ([.052, .058, .071, .070, .073, .078, .088, .119, .117],
                    [.055, .059, .060, .065, .068, .071, .087, .116, .108],
                    [.901, 1.016, 1.065, 1.027, 1.130, 1.091, 1.059, 1.070, 1.086]),
    '+ field offsets (final)': ([.046, .053, .067, .066, .074, .076, .084, .115, .122],
                                [.053, .057, .060, .064, .066, .068, .085, .116, .104],
                                [.852, .914, 1.005, 1.053, 1.135, 1.057, 1.044, 1.087, 1.106]),
}

# --- fix_sfr.py, sfr_split_snr100 (conventional normalisation) -------------
SFR = dict(
    N_SF=A([531, 472, 524, 316, 165, 110, 80, 39, 29]),
    A1_SF=A([.101, .109, .126, .113, .120, .135, .150, .122, .190]),
    A1_SFe=A([.005, .005, .006, .008, .008, .017, .015, .013, .023]),
    N_Q=A([111, 60, 68, 41, 25, 12, 2, 9, 0]),
    A1_Q=A([.078, .109, .142, .088, .157, .122, np.nan, .117, np.nan]),
    A1_Qe=A([.010, .011, .017, .020, .018, .045, np.nan, .044, np.nan]),
    ratio=A([1.294, 1.005, .888, 1.287, .768, 1.105, np.nan, 1.045, np.nan]),
    ratioe=A([.174, .115, .111, .303, .101, .427, np.nan, .409, np.nan]))

# --- star_psf_astrometry.py (F444W): field, N, mean dx (sig), mean dy (sig) --
STARS = [('CEERS', 16, .04, .12, .02, .15), ('GOODS-N', 6, .03, .17, -.10, .05),
         ('GOODS-S', 50, .11, .19, .06, .19), ('PRIMER-COSMOS-E', 55, .38, .11, -.08, .13),
         ('PRIMER-COSMOS-W', 55, .37, .10, -.04, .13), ('PRIMER-UDS-N', 28, .01, .06, .01, .06),
         ('PRIMER-UDS-S', 36, -.03, .05, -.02, .05)]


TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tex')


def save(fig, name):
    # figures/ for separate upload, tex/ for compiling the manuscript
    for d in (OUT, TEX):
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f'{name}.pdf'))
        fig.savefig(os.path.join(d, f'{name}.png'), dpi=300)
    plt.close(fig)
    print('wrote', name)


def fig_amplitudes():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, fin, rel, flo, ell, lab in (
            (axs[0], D_A1_FINAL, REL_A1, D_A1_FLOOR, E_A1_FLOOR, '$A_1$'),
            (axs[1], D_A2_FINAL, REL_A2, D_A2_FLOOR, E_A2_FLOOR, '$A_2$')):
        ax.errorbar(Z, fin, fin * rel, fmt='o-', color=C_DISK, capsize=2, label='disks, corrected')
        ax.errorbar(Z, flo, flo * rel, fmt='o--', mfc='white', color=C_DISK, capsize=2,
                    alpha=0.8, label='disks, floor only (lower limit)')
        ax.plot(Z + 0.05, ell, 's--', mfc='white', color=C_ELL, alpha=0.9,
                label='early types, floor only')
        ax.set(xlabel='Redshift $z$', ylabel=f'median {lab}', ylim=(0, 0.45))
    axs[0].set_title('(a) $m=1$', fontsize=10)
    axs[1].set_title('(b) $m=2$', fontsize=10)
    axs[0].legend(loc='upper left')
    save(fig, 'Fig1_amplitudes')


def fig_ratio():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.errorbar(Z, D_R12_OBS, D_R12_OBS * REL_R12, fmt='o--', mfc='white', color=C_DISK,
                capsize=2, label='disks, observed')
    ax.errorbar(Z, D_R12_COR, D_R12_COR * REL_R12, fmt='o-', color=C_DISK, capsize=2,
                label='disks, corrected')
    ax.plot(Z + 0.05, E_R12_COR, 's-', color=C_ELL, label='early types, corrected')
    ax.axhline(1, color=C_GREY, ls=':', lw=1)
    ax.set(xlabel='Redshift $z$', ylabel='median $A_1/A_2$', ylim=(0.3, 1.45))
    ax.legend(loc='upper left', ncol=2, fontsize=7)
    save(fig, 'Fig2_ratio')


def fig_steps():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    cmap = plt.get_cmap('viridis')
    n = len(STEPS)
    for i, (lab, vals) in enumerate(STEPS.items()):
        lw = 2.2 if i in (0, n - 1) else 1.0
        for ax, v in zip(axs, vals):
            ax.plot(Z, v, 'o-', ms=3, lw=lw, color=cmap(i / (n - 1)), label=lab)
    for ax, k in zip(axs, ('$A_1$', '$A_2$', '$A_1/A_2$')):
        ax.set(xlabel='Redshift $z$', ylabel=f'median {k} (pipeline norm.)')
    axs[0].legend(fontsize=7)
    save(fig, 'Fig3_steps')


def fig_restframe():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    styles = [('o--', C_GREY), ('s-', C_DISK), ('D-', C_ELL)]
    for i, (lab, d) in enumerate(RF.items()):
        fmt, col = styles[i]
        for ax, k in zip(axs, ('A1', 'A2', 'R12')):
            ax.errorbar(Z + 0.04 * i, d[k], d[k + 'e'], fmt=fmt, color=col, capsize=2,
                        mfc='white' if i == 0 else col, label=lab)
    for ax, k in zip(axs, ('$A_1$', '$A_2$', '$A_1/A_2$')):
        ax.set(xlabel='Redshift $z$', ylabel=k)
    axs[0].legend()
    save(fig, 'Fig6_restframe')


def fig_injection():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[0].plot(Z, INJ_A1_FLOOR, 'o-', color=C_DISK, label='$A_1$')
    axs[0].plot(Z, INJ_A2_FLOOR, 's-', color=C_ELL, label='$A_2$')
    axs[1].plot(Z, INJ_A1_REC, 'o-', color=C_DISK, label='$A_1$')
    axs[1].plot(Z, INJ_A2_REC, 's-', color=C_ELL, label='$A_2$')
    axs[1].axhline(1, color=C_GREY, ls=':')
    axs[0].set(xlabel='Redshift $z$', ylabel='noise floor', ylim=(0, None))
    axs[1].set(xlabel='Redshift $z$', ylabel='recovered / injected', ylim=(0, 1.1))
    for a in axs:
        a.legend()
    save(fig, 'Fig5_injection')


def fig_sfr():
    s = SFR
    ok = (s['N_SF'] >= 15) & (s['N_Q'] >= 15)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[0].errorbar(Z, s['A1_SF'], s['A1_SFe'], fmt='o-', color=C_DISK, capsize=2,
                    label='star-forming')
    axs[0].errorbar(Z + 0.05, s['A1_Q'], s['A1_Qe'], fmt='s-', color=C_ELL, capsize=2,
                    label='quiescent')
    axs[1].errorbar(Z[ok], s['ratio'][ok], s['ratioe'][ok], fmt='D', color='k', capsize=2,
                    label='$N\\geq15$ per class')
    axs[1].errorbar(Z[~ok], s['ratio'][~ok], s['ratioe'][~ok], fmt='D', mfc='white', color='k',
                    capsize=2, label='$N<15$')
    axs[1].axhline(1, color=C_GREY, ls=':')
    axs[1].axhline(0.94, color=C_DISK, ls='--', lw=1, label='weighted mean 0.94')
    axs[0].set(xlabel='Redshift $z$', ylabel='median $A_1$')
    axs[1].set(xlabel='Redshift $z$', ylabel='$A_1^{\\rm SF}/A_1^{\\rm Q}$', ylim=(0.2, 1.9))
    for a in axs:
        a.legend()
    save(fig, 'Fig7_sfr')


def fig_astrometry():
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for i, (f, n, dx, sx, dy, sy) in enumerate(STARS):
        col = C_ELL if 'COSMOS' in f else plt.get_cmap('tab10')(i)
        ax.errorbar(dx, dy, xerr=sx, yerr=sy, fmt='o', color=col, capsize=2,
                    label=f'{f} ({n})')
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.axvline(0, color=C_GREY, lw=0.8)
    ax.set(xlabel=r'star centroid $-$ catalogue, $\Delta x$ [pix]', ylabel=r'$\Delta y$ [pix]',
           xlim=(-0.4, 0.7), ylim=(-0.4, 0.4))
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1))
    save(fig, 'Fig4_astrometry')


if __name__ == '__main__':
    for f in (fig_amplitudes, fig_ratio, fig_steps, fig_restframe, fig_injection, fig_sfr,
              fig_astrometry):
        f()
