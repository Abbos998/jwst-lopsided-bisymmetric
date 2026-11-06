#!/usr/bin/env bash
# run_matched_bootstrap.sh -- bootstrap uncertainties of the resolution- and
# S/N-matched growth factors (Table B1), joint over observed and injected galaxies.
#   ROOT=~/Downloads/pasp_v3 bash run_matched_bootstrap.sh 2>&1 | tee ../logs/matched_bootstrap.log
# Optional env: DATA, NPB (300, as in run_selection_tests.sh), NBOOT (1000), RREF, WORKERS,
#   REUSE=1 (reuse the saved injection rows; only the bootstrap is repeated).
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
DATA=${DATA:-$ROOT/data}
NPB=${NPB:-300}
NBOOT=${NBOOT:-1000}
RREF=${RREF:-0.04}
WORKERS=${WORKERS:-$(python3 -c "import os; print(max(1,(os.cpu_count() or 2)-1))")}
cd "$(dirname "$0")"
CAT=$DATA/dja_snr_f444w.csv
for f in matched_bootstrap.py calib_core.py injection_qpa_test.py ../data/table1_disks_F444W.csv \
         ../data/intermediate/corrected_disk_f444w.csv "$CAT" "$ROOT/psf/F444W_empirical_psf.fits" \
         "$ROOT/noise_acf_f444w.npy"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
PIX=$(python3 -c "import pandas as pd; print(round(pd.read_csv('$CAT').pixscale.median(), 4))" 2>/dev/null || echo 0.05)
python3 matched_bootstrap.py --scripts . --catalog "$CAT" \
    --galaxies ../data/intermediate/corrected_disk_f444w.csv --table1 ../data/table1_disks_F444W.csv \
    --psf "$ROOT/psf/F444W_empirical_psf.fits" --noise_acf "$ROOT/noise_acf_f444w.npy" \
    --pixscale "$PIX" --re_ref_scale "$RREF" --n_per_bin "$NPB" --n_boot "$NBOOT" --workers "$WORKERS" \
    ${REUSE:+--reuse_rows} --out "$ROOT/selection_tests/matched_bootstrap" | grep -v ' chunks$'
echo
echo "Check: the point estimates must equal those of logs/selection_tests.log"
echo "(full 2.091/2.317, size 1.721/3.004, snr 1.634/1.407, both 1.144/2.565)."
echo "FINISHED. Send back: $ROOT/selection_tests/matched_bootstrap/matched_bootstrap_summary.txt"
