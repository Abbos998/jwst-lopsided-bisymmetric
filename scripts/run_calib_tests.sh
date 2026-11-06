#!/usr/bin/env bash
# run_calib_tests.sh -- calibration-systematics tests requested in review:
#   1. deprojection errors (q, PA)      -> injection_qpa_test.py
#   2. empirical-PSF uncertainty        -> psf_bootstrap_test.py
# Both reuse injection_test.py and am_wrapper.py:measure_v6 with the same
# PSF, noise ACF, catalogue and 300 galaxies per bin as the paper.
#
# Copy calib_core.py, injection_qpa_test.py, psf_bootstrap_test.py and this
# file into the package's scripts/ directory, then (~10-40 min):
#   cd <package>/supplementary/scripts
#   ROOT=~/Downloads/pasp_v3 bash run_calib_tests.sh 2>&1 | tee ../logs/calib_tests.log
# Optional env: D, DATA (as in run_rd_fix.sh), NPB (300), NBOOT (20),
#   WORKERS (cpu-1), RREF (0.04), Q_ERR_COL / PA_ERR_COL (catalogue
#   uncertainties in the DJA morpho FITS; list candidates with
#   python3 injection_qpa_test.py --list_err_columns $M), IDX_COL (idx).
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
D=${D:-/scratch/pulsating_disk/dja_analysis}
M=$D/full-good_morpho-phot.fits.gz
DATA=${DATA:-$ROOT/data}
NPB=${NPB:-300}
NBOOT=${NBOOT:-20}
RREF=${RREF:-0.04}
WORKERS=${WORKERS:-$(python3 -c "import os; print(max(1,(os.cpu_count() or 2)-1))")}
cd "$(dirname "$0")"
T1=../data/table1_disks_F444W.csv
CAT=$DATA/dja_snr_f444w.csv
PSF=$ROOT/psf/F444W_empirical_psf.fits
ACF=$ROOT/noise_acf_f444w.npy
STARS=$ROOT/psf/F444W_stars.csv
SDIR=$ROOT/psf/stars_f444w
OUT=$ROOT/calib_tests; mkdir -p "$OUT"

echo "=== 0 inputs"
miss=0
for f in calib_core.py injection_qpa_test.py psf_bootstrap_test.py injection_test.py am_wrapper.py \
         "$T1" "$CAT" "$PSF" "$ACF"; do
    [ -e "$f" ] || { echo "  missing: $f"; miss=1; }
done
[ $miss -eq 0 ] || { echo "STOP: set ROOT=... / DATA=... if the files are elsewhere"; exit 1; }
STARARGS=()
if [ -e "$STARS" ] && [ -d "$SDIR" ]; then
    STARARGS=(--stars_csv "$STARS" --stars_dir "$SDIR" --n_boot "$NBOOT")
    echo "  stars: $STARS ($(($(wc -l < "$STARS") - 1)) kept), cutouts in $SDIR"
else
    echo "  NOTE: $STARS or $SDIR not found -> PSF test without bootstrap/field variants"
    echo "        (re-create them with star_psf_astrometry.py --filters F444W --outdir $ROOT/psf)"
fi
PIX=$(python3 -c "import pandas as pd; print(round(pd.read_csv('$CAT').pixscale.median(), 4))" 2>/dev/null || echo 0.05)
echo "  cutout pixscale $PIX, workers $WORKERS, $NPB galaxies per bin"
COMMON=(--scripts . --catalog "$CAT" --table1 "$T1" --psf "$PSF" --noise_acf "$ACF"
        --pixscale "$PIX" --re_ref_scale "$RREF" --n_per_bin "$NPB" --workers "$WORKERS")

ERRARGS=()
if [ -n "${Q_ERR_COL:-}" ] && [ -n "${PA_ERR_COL:-}" ]; then
    ERRARGS=(--err_fits "$M" --idx_col "${IDX_COL:-idx}" --q_err_col "$Q_ERR_COL" --pa_err_col "$PA_ERR_COL")
    echo "  catalogue q/PA errors: $Q_ERR_COL, $PA_ERR_COL from $M"
else
    echo "  catalogue q/PA error columns not given: running fixed and noise-scaled scenarios only"
    [ -e "$M" ] && python3 injection_qpa_test.py --list_err_columns "$M" | sed 's/^/    /' || true
fi

echo "=== 1 q/PA deprojection-error test"
python3 injection_qpa_test.py "${COMMON[@]}" ${ERRARGS[@]+"${ERRARGS[@]}"} --out "$OUT/qpa"

echo "=== 2 PSF uncertainty test"
python3 psf_bootstrap_test.py "${COMMON[@]}" ${STARARGS[@]+"${STARARGS[@]}"} --out "$OUT/psf"

cp ../logs/calib_tests.log "$OUT/" 2>/dev/null || true
echo
echo "FINISHED. Send back:  cd $ROOT && zip -r calib_tests.zip calib_tests -x 'calib_tests/*/psf_per_object.csv'"
