#!/usr/bin/env bash
# run_referee_tests.sh -- field, exclusion, Sersic, S/N-join and spec-z checks
# (referee_tests.py). No images and no injections are needed; takes about a minute.
#   ROOT=~/Downloads/pasp_v3 bash run_referee_tests.sh 2>&1 | tee ../logs/referee_tests.log
# Optional env: D (DJA directory), DATA, FIELD_COL, ZSPEC_COL, ZPHOT_COL, MASS_COL.
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
D=${D:-/scratch/pulsating_disk/dja_analysis}
DATA=${DATA:-$ROOT/data}
M=$D/full-good_morpho-phot.fits.gz
cd "$(dirname "$0")"
OUT=$ROOT/referee_tests; mkdir -p "$OUT"
ARGS=()
[ -e "$DATA/dja_snr_f444w.csv" ] && ARGS+=(--snrcat "$DATA/dja_snr_f444w.csv") || echo "no S/N catalogue: test 4 skipped"
[ -e "$M" ] && ARGS+=(--morpho "$M") || echo "no morpho FITS: field from sky position, test 5 skipped"
[ -n "${FIELD_COL:-}" ] && ARGS+=(--field_col "$FIELD_COL")
[ -n "${ZSPEC_COL:-}" ] && ARGS+=(--zspec_col "$ZSPEC_COL")
[ -n "${ZPHOT_COL:-}" ] && ARGS+=(--zphot_col "$ZPHOT_COL")
[ -n "${MASS_COL:-}" ] && ARGS+=(--mass_col "$MASS_COL")
python3 referee_tests.py --galaxies ../data/intermediate/corrected_disk_f444w.csv \
    --table1 ../data/table1_disks_F444W.csv --sample "$DATA/dja_real.csv" \
    ${ARGS[@]+"${ARGS[@]}"} --out "$OUT/referee"
echo
echo "FINISHED. Send back: $OUT/referee_summary.txt"
