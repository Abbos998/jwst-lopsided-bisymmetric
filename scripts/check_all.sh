#!/usr/bin/env bash
# check_all.sh — run every check of the package and keep the output in logs/:
#   verify_table1.log      Table 1 recomputed from its calculation chain (strict)
#   manuscript_check.log   Tables 1-4 and the 46 numbers of the text regenerated from data/
#                          and compared with tex/ (must be identical)
#   reproduce_table1.log   Table 1, errors and Table 4 regenerated from data/intermediate/
#   check_abstract.log     numbers of the Abstract and Summary vs the final tables
#   test_gates.log         24 failure/tampering cases on synthetic data (~1 min)
#   environment.txt        versions of the tools used
# Needs no images and no pipeline outputs. Run from scripts/:  bash check_all.sh
set -uo pipefail
PKG=$(cd "$(dirname "$0")/.." && pwd); L="$PKG/logs"; mkdir -p "$L"
cd "$PKG/scripts"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
status=0
run() {   # run <log> <command...>
    local log=$1; shift
    { echo "# $(date -u '+%Y-%m-%d %H:%M UTC')  $*"; "$@"; } > "$L/$log" 2>&1
    local rc=$?
    printf '  %-24s %s\n' "$log" "$([ $rc -eq 0 ] && echo passed || echo "FAILED (exit $rc)")"
    [ $rc -eq 0 ] || status=1
}
compare_tex() {
    python3 median_recovery_test.py --inter ../data/intermediate --out "$W/mr.csv" > /dev/null || return 1
    python3 injection_sample_match.py --out "$W/im.csv" > /dev/null || return 1
    cmp -s "$W/mr.csv" ../data/median_recovery_test.csv && cmp -s "$W/im.csv" ../data/injection_sample_match.csv \
        && echo "median_recovery_test.csv, injection_sample_match.csv identical to the regenerated files" \
        || { echo "robustness CSVs DIFFER from the regenerated files"; return 1; }
    python3 make_tex_from_data.py --data ../data --tex "$W" --submission || return 1
    for f in numbers.tex table1.tex table_chain.tex table_errors.tex table_steps.tex table_restframe.tex; do
        if cmp -s "$W/$f" "../tex/$f"; then echo "tex/$f identical to the regenerated file"
        else echo "tex/$f DIFFERS from the regenerated file"; return 1; fi
    done
}
{
    echo "date: $(date -u '+%Y-%m-%d %H:%M UTC')"; uname -sr
    python3 -c "import sys, numpy, pandas; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__)"
    command -v pdftotext > /dev/null && pdftotext -v 2>&1 | head -1
    command -v pdflatex > /dev/null && pdflatex --version | head -1
} > "$L/environment.txt" 2>&1
echo "Checks (logs in logs/):"
run verify_table1.log python3 verify_table1.py ../data/table1_disks_F444W.csv --submission
run manuscript_check.log compare_tex
run reproduce_table1.log bash reproduce_table1.sh
# the manuscript: the article-class build (../jwst_pasp_revised.pdf) or the AASTeX submission
# (../../manuscript/), whose hand-typed deluxetables are compared with data/ as well
PDF=$(ls ../jwst_pasp_revised.pdf ../../manuscript/jwst_*.pdf 2>/dev/null | head -1 || true)
if [ -f ../../manuscript/main.tex ]; then
    run manuscript_tables.log python3 check_manuscript_tables.py --main ../../manuscript/main.tex --tex ../tex
fi
if command -v pdftotext > /dev/null && [ -n "$PDF" ]; then
    run check_abstract.log python3 check_abstract.py --pdf "$PDF" --data ../data
else
    echo "  check_abstract.log       skipped (no manuscript PDF in this directory)"
fi
run test_gates.log bash ../tests/test_gates.sh
echo; [ $status -eq 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED (see logs/)"
exit $status
