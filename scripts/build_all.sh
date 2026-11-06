#!/usr/bin/env bash
# build_all.sh — regenerate every number, table and figure of the paper from the
# pipeline output files, then compile the manuscript. One source of truth:
#
#   outputs in $ROOT  ->  make_table1.py  ->  table1_disk_final.{csv,tex}
#                     ->  restframe_test.py (same calibration as Table 1)
#                     ->  export_data.py  ->  data/*.csv, data/intermediate/
#                     ->  verify_table1.py --submission
#                     ->  make_tex_from_data.py --submission  ->  tex/numbers.tex,
#                         tex/table1.tex, table_steps.tex, table_restframe.tex, table_chain.tex
#                     ->  make_paper_figures.py  ->  figures/, tex/Fig*.pdf|png
#                     ->  pdflatex + bibtex  ->  jwst_pasp_revised.pdf
#
# Required in $ROOT (the script stops if any file is missing):
#   corrected_disk_f444w.csv, corrected_disk_f277w.csv, corrected_disk_f150w.csv
#   inj_f444w_m/, inj_f277w_m/, inj_f150w_m/            (zero + matched)
#   inj_f444w_{scale0.5,scale1.5,rising,falling}/       (calibration systematics)
#   injection_summary.csv in each, plus injection_per_object.csv in the inj_f444w_* runs
#
# Run from the package's scripts/ directory:   ROOT=~/Downloads/pasp_v3 bash build_all.sh
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
PKG=$(cd "$(dirname "$0")/.." && pwd)
cd "$PKG/scripts"

SYST_LABELS="scale0.5 scale1.5 rising falling"
NBOOT=${NBOOT:-1000}          # bootstrap resamplings (reproduce_table1.sh must use the same)
sha256() { if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
need=(corrected_disk_f444w.csv corrected_disk_f277w.csv corrected_disk_f150w.csv)
for d in inj_f444w_m $(printf 'inj_f444w_%s ' $SYST_LABELS); do
    need+=("$d/injection_summary.csv" "$d/injection_per_object.csv")
done
need+=(inj_f277w_m/injection_summary.csv inj_f150w_m/injection_summary.csv)   # rest-frame test
miss=0
for f in "${need[@]}"; do
    [ -f "$ROOT/$f" ] || { echo "missing $ROOT/$f"; miss=1; }
done
[ $miss -eq 0 ] || { echo "STOP: required inputs missing (see README_SUBMISSION_READY.md)"; exit 1; }

SYST=""
for lab in $SYST_LABELS; do SYST="$SYST,$lab=$ROOT/inj_f444w_$lab/injection_summary.csv"; done
echo "=== 1/7 Table 1 (systematics: $SYST_LABELS)"
python3 make_table1.py --objects "$ROOT/corrected_disk_f444w.csv" \
    --injection "$ROOT/inj_f444w_m/injection_summary.csv" --syst "${SYST#,}" \
    --nboot "$NBOOT" --out "$ROOT/table1_disk_final" | sed -n '/Final values/,/without a valid/p'

echo "=== 2/7 rest-frame test (calibration as Table 1)"
python3 restframe_test.py \
    --objects F444W=$ROOT/corrected_disk_f444w.csv,F277W=$ROOT/corrected_disk_f277w.csv,F150W=$ROOT/corrected_disk_f150w.csv \
    --injections F444W=$ROOT/inj_f444w_m/injection_summary.csv,F277W=$ROOT/inj_f277w_m/injection_summary.csv,F150W=$ROOT/inj_f150w_m/injection_summary.csv \
    --targets 0.7,1.0 --centering catalog --out "$ROOT/restframe_disk_m" > /dev/null

echo "=== 3/7 data/*.csv and data/intermediate/ (inputs of make_table1.py)"
python3 export_data.py --root "$ROOT" --out "$PKG/data"
I="$PKG/data/intermediate"
find "$I" -mindepth 1 ! -name README.md -delete 2>/dev/null || true
mkdir -p "$I"
cp "$ROOT/corrected_disk_f444w.csv" "$I/"
for d in inj_f444w_m $(printf 'inj_f444w_%s ' $SYST_LABELS); do
    mkdir -p "$I/$d"
    cp "$ROOT/$d/injection_summary.csv" "$ROOT/$d/injection_per_object.csv" "$I/$d/"
done
(cd "$I" && sha256 $(find . -name '*.csv' | sort) > SHA256SUMS)
# robustness tables computed from data/intermediate/ (quoted in Sect. 3.3)
python3 median_recovery_test.py --inter "$I" --out "$PKG/data/median_recovery_test.csv" > /dev/null
python3 injection_sample_match.py --inter "$I" --table "$PKG/data/table1_disks_F444W.csv" \
    --out "$PKG/data/injection_sample_match.csv" > /dev/null

echo "=== 4/7 verify Table 1 from data/ (strict)"
python3 verify_table1.py "$PKG/data/table1_disks_F444W.csv" --submission | tail -4

echo "=== 5/7 manuscript tables and numbers from data/"
# full checks need the robustness tests of run_review7.sh; before they exist (first build)
# or in the synthetic tests use MANUSCRIPT_CHECKS=core (Table 1 and its chain only)
python3 make_tex_from_data.py --data "$PKG/data" --tex "$PKG/tex" --submission \
    $([ "${MANUSCRIPT_CHECKS:-full}" = core ] && echo --core)

if [ "${SKIP_FIGURES_FOR_TESTS:-0}" = 1 ]; then
    echo "TEST MODE: figures and PDF skipped -- not a submission build"; exit 0
fi
echo "=== 6/7 figures"
python3 make_paper_figures.py --root "$ROOT" --data "$PKG/data" --out "$ROOT/figs_tmp" > /dev/null
i=1
for n in amplitudes ratio steps astrometry injection restframe sfr; do
    for e in pdf png; do
        src="$ROOT/figs_tmp/fig_$n.$e"
        [ -f "$src" ] || { echo "STOP: fig_$n.$e not produced"; exit 1; }
        cp "$src" "$PKG/figures/Fig${i}_$n.$e"
        cp "$src" "$PKG/tex/Fig${i}_$n.$e"
    done
    i=$((i + 1))
done

echo "=== 7/7 manuscript"
command -v pdflatex > /dev/null && command -v bibtex > /dev/null || {
    echo "STOP: pdflatex/bibtex not installed. Tables, numbers and figures are ready in $PKG/tex."
    echo "  Install:  sudo apt install texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended lmodern"
    echo "  then rerun bash build_all.sh (fast, injection runs are not repeated)"; exit 3; }
cd "$PKG/tex"
pdflatex -interaction=nonstopmode jwst_pasp_revised.tex > /dev/null || true
bibtex jwst_pasp_revised > /dev/null || true
pdflatex -interaction=nonstopmode jwst_pasp_revised.tex > /dev/null || true
pdflatex -interaction=nonstopmode jwst_pasp_revised.tex > /dev/null || true
nerr=$(grep -c '^!' jwst_pasp_revised.log || true)
echo "LaTeX errors: $nerr"
[ "$nerr" -eq 0 ] || exit 1
cp jwst_pasp_revised.pdf "$PKG/"
rm -f *.aux *.blg *.log *.out
command -v pdftotext > /dev/null && python3 "$PKG/scripts/check_abstract.py" \
    --pdf "$PKG/jwst_pasp_revised.pdf" --data "$PKG/data" | tail -1
echo "Done: $PKG/jwst_pasp_revised.pdf"
echo "SUBMISSION CHECK: passed (now run bash check_all.sh)"
