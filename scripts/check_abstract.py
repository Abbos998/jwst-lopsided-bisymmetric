#!/usr/bin/env python3
"""
check_abstract.py — compare every number quoted in the Abstract and the Summary
of the compiled PDF with the final tables in data/, independently of the
macros in tex/numbers.tex (values are recomputed here from the CSV files and
searched for in the text extracted from the PDF with pdftotext).

Usage:  python3 check_abstract.py --pdf ../jwst_pasp_revised.pdf --data ../data
"""

import argparse
import re
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd


def r(x, nd):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def section(text, start, end):
    # case-insensitive, so that both article ("Abstract") and AASTeX ("ABSTRACT") layouts work
    low = text.lower()
    i = low.find(start.lower())
    j = low.find(end.lower(), i + 1)
    if i < 0 or j < 0:
        sys.exit(f"cannot find the section between '{start}' and '{end}' in the PDF text")
    return re.sub(r'\s+', ' ', text[i:j])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pdf', default='../jwst_pasp_revised.pdf')
    ap.add_argument('--data', default='../data')
    a = ap.parse_args()
    # three extractions: -layout (one-column article), reading order (two-column) and
    # -raw (content-stream order, robust when a section continues in the other column);
    # bare line numbers (AASTeX linenumbers) are dropped
    texts = []
    for opt in (['-layout'], [], ['-raw']):
        t_ = subprocess.run(['pdftotext', *opt, a.pdf, '-'], capture_output=True, text=True,
                            check=True).stdout
        texts.append(re.sub(r'(?m)^\s*\d+\s*$', '', t_))
    abstract = summary = None
    for t_ in texts:
        try:
            ab = section(t_, 'Abstract', 'Introduction')
        except SystemExit:
            continue
        try:
            su = section(t_, 'Summary', 'Data and code availability')
        except SystemExit:
            su = ''      # MNRAS layout: no such heading; the blocks below are used
        # two-column AASTeX: the "Data and code availability" heading can sit beside the
        # Summary in the other column, which truncates the block above; the block up to
        # the "Software:" line (end of the back matter) contains the whole Summary
        for end in ('Software:', 'Acknowledgements', 'Data availability'):
            try:
                su += ' || ' + section(t_, 'Summary', end)
            except SystemExit:
                pass
        abstract, summary = (abstract or '') + ' || ' + ab, (summary or '') + ' || ' + su
    if abstract is None:
        sys.exit("cannot find the Abstract and Summary in the PDF text")

    t = pd.read_csv(f'{a.data}/table1_disks_F444W.csv')
    rf = pd.read_csv(f'{a.data}/restframe_test.csv')
    et = pd.read_csv(f'{a.data}/early_types_F444W.csv')
    lo, hi = t[t.z == 0.75].iloc[0], t[t.z > 4]
    rf1 = rf[rf.set == 'rest 1.0 um'].set_index('z')
    def wmean(sel):
        w = 1 / t[sel].A1A2_corrected_err ** 2
        return (w * t[sel].A1A2_corrected).sum() / w.sum(), 1 / w.sum() ** 0.5
    rlo, rlo_e = wmean(t.z < 3.5)
    rhi, rhi_e = wmean(t.z > 3.5)
    a1lo, a1hi = lo.A1_final_conv, hi.A1_final_conv.mean()
    a2lo, a2hi = lo.A2_final_conv, hi.A2_final_conv.mean()
    n = int(t.N_valid.sum())
    checks = [   # (where, quantity, value from data/, text that must appear)
        ('Abstract', 'N valid galaxies (Table 1, sum)', f"{n:,}", f"{n:,}"),
        ('Abstract', 'A1, z~0.75 (Table 1)', r(a1lo, 3), f"from {r(a1lo, 2)}"),
        ('Abstract', 'A1, z>4 (mean of z~4.5, 5.5)', r(a1hi, 3), f"to {r(a1hi, 2)}"),
        ('Abstract', 'A2, z~0.75 (Table 1)', r(a2lo, 3), f"from {r(a2lo, 2)}"),
        ('Abstract', 'A2, z>4', r(a2hi, 3), f"to {r(a2hi, 2)}"),
        ('Abstract', 'A1 growth at 1.0 um (Table 3)', r(rf1.loc[4.5, 'A1_conv'] / rf1.loc[0.75, 'A1_conv'], 2),
         f"≈ {r(rf1.loc[4.5, 'A1_conv'] / rf1.loc[0.75, 'A1_conv'], 1)}"),
        ('Abstract', 'A2 growth at 1.0 um (Table 3)', r(rf1.loc[4.5, 'A2_conv'] / rf1.loc[0.75, 'A2_conv'], 2),
         f"≈ {r(rf1.loc[4.5, 'A2_conv'] / rf1.loc[0.75, 'A2_conv'], 1)}"),
        ('Abstract', 'A1/A2 corr., weighted mean z<3.5', f"{r(rlo, 3)} ± {r(rlo_e, 3)}",
         f"{r(rlo, 2)} ± {r(rlo_e, 2)}"),
        ('Abstract', 'A1/A2 corr., weighted mean z>3.5', f"{r(rhi, 3)} ± {r(rhi_e, 3)}",
         f"{r(rhi, 2)} ± {r(rhi_e, 2)}"),
        ('Abstract', 'early types A1/A2 (mean)', r(et.A1A2_corrected_approx.mean(), 3),
         f"≈ {r(et.A1A2_corrected_approx.mean(), 1)}"),
        ('Summary', 'A1 growth z~0.75 -> z>4', r(a1hi / a1lo, 2), f"factors {r(a1hi / a1lo, 1)}"),
        ('Summary', 'A2 growth z~0.75 -> z>4', r(a2hi / a2lo, 2), f"and {r(a2hi / a2lo, 1)}"),
        ('Summary', 'A1/A2 corr., weighted mean z<3.5', r(rlo, 3), f"{r(rlo, 2)} ± {r(rlo_e, 2)}"),
        ('Summary', 'A1/A2 corr., weighted mean z>3.5', r(rhi, 3), f"{r(rhi, 2)} ± {r(rhi_e, 2)}"),
        ('Summary', 'early types A1/A2', r(et.A1A2_corrected_approx.mean(), 3),
         f"≈ {r(et.A1A2_corrected_approx.mean(), 1)}"),
    ]
    bad = 0
    print(f"{'where':9} {'quantity':38} {'data/':>15}  {'expected in PDF':22} result")
    for where, q, val, want in checks:
        blk = abstract if where == 'Abstract' else summary
        ok = want.replace(' ', '') in blk.replace(' ', '')
        bad += not ok
        print(f"{where:9} {q:38} {val:>15}  {want:22} {'ok' if ok else 'NOT FOUND'}")
    print("\nAll Abstract and Summary numbers agree with the final tables." if not bad
          else f"\n{bad} number(s) not found: the text and the tables disagree.")
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
