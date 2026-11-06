#!/usr/bin/env bash
# run_final.sh — compute everything the manuscript still lacks (the red "?":
# uncertainties of Table 1, N_total, calibration systematics) and rebuild the
# package, in one command:
#
#   1. main F444W injection run, with injection_per_object.csv   (~10 min)
#   2. the four systematics runs, in parallel                     (~10-15 min)
#   3. F277W / F150W injection runs, only if they lack the catalog-centring
#      rows that the rest-frame test now needs                    (~10 min each)
#   4. build_all.sh       (Table 1-4, numbers, figures, PDF, strict checks)
#   5. reproduce_table1.sh (independent check from the package alone)
#
# Run from the package's scripts/ directory:
#   cd ~/Downloads/pasp_v3/pasp_SUBMISSION/scripts
#   ROOT=~/Downloads/pasp_v3 bash run_final.sh 2>&1 | tee ../run_final.log
#
# Optional env: DATA (default $ROOT/data), NPB (galaxies per z bin, default 300),
#               FORCE=1 to rerun injection runs that already have all outputs.
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
DATA=${DATA:-$ROOT/data}
NPB=${NPB:-300}
RREF=${RREF:-0.04}   # arcsec/pix of DISK_RADIUS_pix (DJA: DISK_RADIUS_deg/DISK_RADIUS_pix = 0.04")
FORCE=${FORCE:-0}
cd "$(dirname "$0")"
LOG=$ROOT/run_final_logs; mkdir -p "$LOG"

echo "=== 0/5 inputs"
need=("$ROOT/corrected_disk_f444w.csv" "$ROOT/corrected_disk_f444w_summary.csv"
      "$ROOT/corrected_disk_f277w.csv" "$ROOT/corrected_disk_f150w.csv"
      "$ROOT/table1_disk_v7.csv" "$ROOT/table1_ell_v7.csv" "$ROOT/sfr_split_snr100.csv"
      "$ROOT/psf/F444W_stars.csv")
for f in f444w f277w f150w; do
    F=$(echo $f | tr a-z A-Z)
    need+=("$DATA/dja_snr_$f.csv" "$ROOT/noise_acf_$f.npy" "$ROOT/psf/${F}_empirical_psf.fits")
done
miss=0
for f in "${need[@]}"; do [ -f "$f" ] || { echo "  missing: $f"; miss=1; }; done
[ $miss -eq 0 ] || { echo "STOP: produce the missing files first (docs/PIPELINE_README.md)"; exit 1; }
echo "  all inputs present"

# inj <filter> <outdir> [extra injection_test.py options]
inj() {
    local f=$1 out=$2; shift 2
    local F; F=$(echo "$f" | tr a-z A-Z)
    python3 injection_test.py --catalog "$DATA/dja_snr_$f.csv" --re_col disk_r_pix --re_unit pix \
        --re_ref_scale "$RREF" --n_col n_sersic --snr_col snr_ap --pixscale 0.05 --cutout_arcsec 5.0 \
        --n_per_bin "$NPB" --centering true,catalog --sigma_cat 0.10 \
        --psf "$ROOT/psf/${F}_empirical_psf.fits" --psf_pixscale 0.05 \
        --noise_acf "$ROOT/noise_acf_$f.npy" --am_func am_wrapper.py:measure_v6 \
        --scenarios zero --matched_table "$ROOT/table1_disk_v7.csv" "$@" --output "$out/"
}
complete() {   # complete <dir>: summary + per-object present and catalog/matched rows exist
    [ -f "$1/injection_summary.csv" ] && [ -f "$1/injection_per_object.csv" ] && python3 - "$1" <<'PY'
import sys, pandas as pd
s = pd.read_csv(f"{sys.argv[1]}/injection_summary.csv")
ok = all(((s.scenario == sc) & (s.centering == 'catalog')).any() for sc in ('zero', 'matched'))
sys.exit(0 if ok else 1)
PY
}

echo "=== 1/5 main F444W injection run"
if [ "$FORCE" = 1 ] || ! complete "$ROOT/inj_f444w_m"; then
    inj f444w "$ROOT/inj_f444w_m" > "$LOG/inj_f444w_m.log" 2>&1
    complete "$ROOT/inj_f444w_m" || { echo "STOP: see $LOG/inj_f444w_m.log"; exit 1; }
    echo "  done"
else
    echo "  already complete (FORCE=1 to rerun)"
fi

echo "=== 2/5 systematics runs (parallel; logs in $LOG)"
pids=()
for X in "scale0.5:--matched_scale 0.5" "scale1.5:--matched_scale 1.5" \
         "rising:--pert_profile rising" "falling:--pert_profile falling"; do
    L=${X%%:*}; O=${X#*:}
    if [ "$FORCE" = 1 ] || ! complete "$ROOT/inj_f444w_$L"; then
        # shellcheck disable=SC2086
        inj f444w "$ROOT/inj_f444w_$L" $O > "$LOG/inj_f444w_$L.log" 2>&1 &
        pids+=($!); echo "  started $L"
    else
        echo "  $L already complete"
    fi
done
for p in "${pids[@]}"; do wait "$p" || { echo "STOP: a systematics run failed (logs in $LOG)"; exit 1; }; done
for L in scale0.5 scale1.5 rising falling; do
    complete "$ROOT/inj_f444w_$L" || { echo "STOP: $L incomplete, see $LOG/inj_f444w_$L.log"; exit 1; }
done
echo "  all four complete"

echo "=== 3/5 F277W / F150W runs for the rest-frame test"
for f in f277w f150w; do
    if [ "$FORCE" = 1 ] || ! complete "$ROOT/inj_${f}_m"; then
        echo "  running $f"
        inj "$f" "$ROOT/inj_${f}_m" > "$LOG/inj_${f}_m.log" 2>&1
        complete "$ROOT/inj_${f}_m" || { echo "STOP: see $LOG/inj_${f}_m.log"; exit 1; }
    else
        echo "  $f already complete"
    fi
done

echo "=== 4/5 build_all.sh"
ROOT=$ROOT bash build_all.sh

echo "=== 5/5 reproduce_table1.sh"
bash check_all.sh

echo
echo "FINISHED. Open ../jwst_pasp_revised.pdf: there must be no red '?' left."
echo "Check Abstract / Table 3 rest-frame numbers (recomputed with the Table 1 calibration)."
