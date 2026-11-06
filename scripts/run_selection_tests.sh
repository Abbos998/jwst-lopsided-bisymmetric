#!/usr/bin/env bash
# run_selection_tests.sh -- tests requested in the fourth review:
#   1. photometric-redshift Monte Carlo        -> photoz_mc_test.py
#   2. resolution- and S/N-matched subsamples   -> size_sn_matched_test.py
#      (three variants: size only, S/N only, both), each with its own
#      injection-recovery calibration and a full-sample reference (same seed)
# Run from the package scripts/ directory:
#   ROOT=~/Downloads/pasp_v3 bash run_selection_tests.sh 2>&1 | tee ../logs/selection_tests.log
# Optional env: D (DJA directory), DATA, NPB (300), NMC (500), RREF (0.04), WORKERS,
#   Z_LO_COL / Z_HI_COL / ZSPEC_COL (redshift-percentile columns, auto-detected).
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
D=${D:-/scratch/pulsating_disk/dja_analysis}
M=$D/full-good_morpho-phot.fits.gz
DATA=${DATA:-$ROOT/data}
NPB=${NPB:-300}
NMC=${NMC:-500}
RREF=${RREF:-0.04}
WORKERS=${WORKERS:-$(python3 -c "import os; print(max(1,(os.cpu_count() or 2)-1))")}
cd "$(dirname "$0")"
T1=../data/table1_disks_F444W.csv
GAL=../data/intermediate/corrected_disk_f444w.csv
CAT=$DATA/dja_snr_f444w.csv
PSF=$ROOT/psf/F444W_empirical_psf.fits
ACF=$ROOT/noise_acf_f444w.npy
OUT=$ROOT/selection_tests; mkdir -p "$OUT"
miss=0
for f in calib_core.py injection_qpa_test.py photoz_mc_test.py size_sn_matched_test.py \
         "$T1" "$GAL" "$CAT" "$PSF" "$ACF" "$M"; do
    [ -e "$f" ] || { echo "missing: $f"; miss=1; }
done
[ $miss -eq 0 ] || { echo "STOP: set ROOT / DATA / D"; exit 1; }
PIX=$(python3 -c "import pandas as pd; print(round(pd.read_csv('$CAT').pixscale.median(), 4))" 2>/dev/null || echo 0.05)

echo "=== 1 photometric-redshift Monte Carlo"
ZARGS=()
[ -n "${Z_LO_COL:-}" ] && ZARGS+=(--z_lo_col "$Z_LO_COL" --z_hi_col "$Z_HI_COL")
[ -n "${ZSPEC_COL:-}" ] && ZARGS+=(--zspec_col "$ZSPEC_COL")
python3 photoz_mc_test.py --galaxies "$GAL" --table1 "$T1" --zcat "$M" --n_mc "$NMC" \
    ${ZARGS[@]+"${ZARGS[@]}"} --out "$OUT/photoz" || {
    echo "  photo-z test failed; candidate columns:"; python3 photoz_mc_test.py --list_z_columns "$M"; }

echo "=== 2 resolution- and S/N-matched subsamples"
for MATCH in size snr both; do
    echo "--- match = $MATCH"
    python3 size_sn_matched_test.py --scripts . --catalog "$CAT" --galaxies "$GAL" --table1 "$T1" \
        --psf "$PSF" --noise_acf "$ACF" --pixscale "$PIX" --re_ref_scale "$RREF" \
        --n_per_bin "$NPB" --workers "$WORKERS" --match "$MATCH" --out "$OUT/matched_$MATCH" \
        | grep -v ' chunks$'
done
echo
echo "FINISHED. Send back: cd $ROOT && zip -r selection_tests.zip selection_tests -x '*per_object.csv'"
