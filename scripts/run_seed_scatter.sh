#!/usr/bin/env bash
# run_seed_scatter.sh -- sampling noise of the injection calibration with many
# independent draws of the injected galaxies (ideal calibration only).
# Continues calib_tests/seeds/: draws that already exist are kept, so with
# NSEED=25 only draws 6..25 are computed after run_calib_followup.sh.
# Then writes calib_tests/followup_summary.txt (percentiles of Table 1 among the
# draws, 68/95% ranges, chi2 against a leave-one-out null).
#   ROOT=~/Downloads/pasp_v3 NSEED=25 bash run_seed_scatter.sh 2>&1 | tee ../logs/seed_scatter.log
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
DATA=${DATA:-$ROOT/data}
NPB=${NPB:-300}
NSEED=${NSEED:-25}
RREF=${RREF:-0.04}
WORKERS=${WORKERS:-$(python3 -c "import os; print(max(1,(os.cpu_count() or 2)-1))")}
cd "$(dirname "$0")"
T1=../data/table1_disks_F444W.csv
CAT=$DATA/dja_snr_f444w.csv
OUT=$ROOT/calib_tests
PIX=$(python3 -c "import pandas as pd; print(round(pd.read_csv('$CAT').pixscale.median(), 4))" 2>/dev/null || echo 0.05)
for s in $(seq 1 "$NSEED"); do
    if [ -e "$OUT/seeds/seed_$s/qpa_calibration.csv" ]; then echo "  draw $s exists"; continue; fi
    python3 injection_qpa_test.py --scripts . --catalog "$CAT" --table1 "$T1" \
        --psf "$ROOT/psf/F444W_empirical_psf.fits" --noise_acf "$ROOT/noise_acf_f444w.npy" \
        --pixscale "$PIX" --re_ref_scale "$RREF" --n_per_bin "$NPB" --workers "$WORKERS" \
        --variants ideal --seed $((1000 + s)) --out "$OUT/seeds/seed_$s" > "$OUT/seeds_$s.log" 2>&1
    echo "  draw $s done"
done
python3 calib_followup_summary.py "$OUT" "$T1"
echo
echo "FINISHED. Send back: $OUT/followup_summary.txt"
