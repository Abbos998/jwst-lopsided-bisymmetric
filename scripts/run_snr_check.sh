#!/usr/bin/env bash
# run_snr_check.sh -- is the drop of the high-z amplitudes in high-S/N subsamples a
# calibration failure at low S/N, or a selection of massive galaxies?
#   ROOT=~/Downloads/pasp_v3 bash run_snr_check.sh 2>&1 | tee ../logs/snr_check.log
# Optional env: DATA, NPB (600: ~200 injected galaxies per S/N tercile), RREF, WORKERS,
#   MASS_CSV / MASS_COL if the stellar mass is not a column of dja_snr_f444w.csv.
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
DATA=${DATA:-$ROOT/data}
NPB=${NPB:-600}
RREF=${RREF:-0.04}
WORKERS=${WORKERS:-$(python3 -c "import os; print(max(1,(os.cpu_count() or 2)-1))")}
cd "$(dirname "$0")"
CAT=$DATA/dja_snr_f444w.csv
for f in snr_calibration_check.py calib_core.py injection_qpa_test.py ../data/table1_disks_F444W.csv \
         ../data/intermediate/corrected_disk_f444w.csv "$CAT" "$ROOT/psf/F444W_empirical_psf.fits" \
         "$ROOT/noise_acf_f444w.npy"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
PIX=$(python3 -c "import pandas as pd; print(round(pd.read_csv('$CAT').pixscale.median(), 4))" 2>/dev/null || echo 0.05)
MARGS=()
[ -n "${MASS_CSV:-}" ] && MARGS+=(--mass_csv "$MASS_CSV")
[ -n "${MASS_COL:-}" ] && MARGS+=(--mass_col "$MASS_COL")
python3 -c "import pandas as pd; print('catalogue columns:', list(pd.read_csv('$CAT', nrows=1).columns))"
python3 snr_calibration_check.py --scripts . --catalog "$CAT" \
    --galaxies ../data/intermediate/corrected_disk_f444w.csv --table1 ../data/table1_disks_F444W.csv \
    --psf "$ROOT/psf/F444W_empirical_psf.fits" --noise_acf "$ROOT/noise_acf_f444w.npy" \
    --pixscale "$PIX" --re_ref_scale "$RREF" --n_per_bin "$NPB" --workers "$WORKERS" \
    ${MARGS[@]+"${MARGS[@]}"} ${REUSE:+--reuse_rows} --out "$ROOT/selection_tests/snr_check" | grep -v ' chunks$'
echo "FINISHED. Send back: $ROOT/selection_tests/snr_check/snr_check_summary.txt"
