#!/usr/bin/env bash
# run_calib_followup.sh -- follow-up of run_calib_tests.sh (second review):
#   1. deprojection test with PA errors drawn from the observed differences
#      between the catalogue Sersic PA and the moments PA (EMP scenarios);
#      same models and random numbers as calib_tests/qpa, so directly comparable
#   2. sampling noise of the calibration: ideal calibration repeated with
#      NSEED independent draws of the injected galaxies
#   3. PSF bootstrap with NBOOT (default 100) realisations
#   4. summary -> calib_tests/followup_summary.txt
# Run from the package scripts/ directory, after run_calib_tests.sh:
#   ROOT=~/Downloads/pasp_v3 bash run_calib_followup.sh 2>&1 | tee ../logs/calib_followup.log
# Optional env: NSEED (5), NBOOT (100), SKIP_PSF=1, WORKERS, DATA, RREF, NPB.
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
DATA=${DATA:-$ROOT/data}
NPB=${NPB:-300}
NSEED=${NSEED:-5}
NBOOT=${NBOOT:-100}
RREF=${RREF:-0.04}
WORKERS=${WORKERS:-$(python3 -c "import os; print(max(1,(os.cpu_count() or 2)-1))")}
cd "$(dirname "$0")"
T1=../data/table1_disks_F444W.csv
CAT=$DATA/dja_snr_f444w.csv
PSF=$ROOT/psf/F444W_empirical_psf.fits
ACF=$ROOT/noise_acf_f444w.npy
EMP=empirical_dpa_f444w.csv
OUT=$ROOT/calib_tests; mkdir -p "$OUT"
for f in calib_core.py injection_qpa_test.py psf_bootstrap_test.py calib_followup_summary.py \
         "$EMP" "$T1" "$CAT" "$PSF" "$ACF"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
PIX=$(python3 -c "import pandas as pd; print(round(pd.read_csv('$CAT').pixscale.median(), 4))" 2>/dev/null || echo 0.05)
COMMON=(--scripts . --catalog "$CAT" --table1 "$T1" --psf "$PSF" --noise_acf "$ACF"
        --pixscale "$PIX" --re_ref_scale "$RREF" --n_per_bin "$NPB" --workers "$WORKERS")

echo "=== 1 empirical PA errors (EMP), same models as calib_tests/qpa"
python3 injection_qpa_test.py "${COMMON[@]}" --emp_dpa "$EMP" \
    --variants ideal,SN,SNx2,EMP_r2,EMP,EMP_x2q --out "$OUT/qpa_emp"

echo "=== 2 sampling noise of the calibration ($NSEED independent draws)"
for s in $(seq 1 "$NSEED"); do
    python3 injection_qpa_test.py "${COMMON[@]}" --variants ideal --seed $((1000 + s)) \
        --out "$OUT/seeds/seed_$s" > "$OUT/seeds_$s.log" 2>&1
    echo "  draw $s done"
done

if [ "${SKIP_PSF:-0}" != 1 ]; then
    echo "=== 3 PSF bootstrap with $NBOOT realisations"
    python3 psf_bootstrap_test.py "${COMMON[@]}" --stars_csv "$ROOT/psf/F444W_stars.csv" \
        --stars_dir "$ROOT/psf/stars_f444w" --n_boot "$NBOOT" --out "$OUT/psf100"
fi

echo "=== 4 summary"
python3 calib_followup_summary.py "$OUT" "$T1"
echo
echo "FINISHED. Send back:  cd $ROOT && zip -r calib_followup.zip calib_tests -x '*per_object.csv' -x '*/psf_variants/*'"
