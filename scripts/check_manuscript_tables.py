#!/usr/bin/env python3
"""
check_manuscript_tables.py — the AASTeX manuscript (main.tex) types its tables as
deluxetables by hand; compare every data row with the table generated from data/
by make_tex_from_data.py (tex/table*.tex). Any difference -> exit 1.

Usage:  python3 check_manuscript_tables.py --main ../../manuscript/main.tex --tex ../tex
"""
import argparse
import re
import sys

GEN = {'tab:main': 'table1', 'tab:restframe': 'table_restframe', 'tab:chain': 'table_chain',
       'tab:errors': 'table_errors', 'tab:steps': 'table_steps', 'tab:robust': 'table_robust'}


def norm(row):
    row = re.sub(r'\\\\\s*$', '', row.strip())
    row = row.replace(r'\Mstar', r'M_\star').replace('$', '').replace(r'\pm', '±')
    return re.sub(r'\s+', '', row)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--main', default='../../manuscript/main.tex')
    ap.add_argument('--tex', default='../tex')
    a = ap.parse_args()
    s = open(a.main).read()
    blocks = re.findall(r'\\begin\{deluxetable\*?\}.*?\\label\{([^}]*)\}.*?\\startdata(.*?)\\enddata', s, re.S)
    # MNRAS version: table/table* + tabular; the data rows lie between the second
    # and the third \hline (header, rows, closing rule)
    for t in re.findall(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}', s, re.S):
        lab = re.search(r'\\label\{([^}]*)\}', t)
        tab = re.search(r'\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}', t, re.S)
        if lab and tab:
            parts = tab.group(1).split('\\hline')
            if len(parts) >= 4:
                blocks.append((lab.group(1), parts[2]))
    bad = 0
    seen = set()
    for label, body in blocks:
        if label not in GEN:
            continue
        seen.add(label)
        rows = [norm(x) for x in body.strip().split('\n') if x.strip() and not x.strip().startswith('\\')]
        g = open(f'{a.tex}/{GEN[label]}.tex').read()
        grows = [norm(x) for x in g.split(r'\midrule')[1].split(r'\bottomrule')[0].strip().split('\n') if x.strip()]
        diff = [(x, y) for x, y in zip(rows, grows) if x != y]
        ok = not diff and len(rows) == len(grows)
        bad += not ok
        print(f"  [{'ok' if ok else 'FAIL'}] {label}: {len(rows)} rows in main.tex, {len(grows)} generated"
              + ('' if ok else f", {len(diff)} differ"))
        for x, y in diff[:5]:
            print(f"      main.tex : {x}\n      generated: {y}")
    for label in sorted(set(GEN) - seen):
        print(f"  [FAIL] {label}: not found in {a.main}")
        bad += 1
    print("All manuscript tables agree with data/." if not bad else f"{bad} table(s) differ from data/.")
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
