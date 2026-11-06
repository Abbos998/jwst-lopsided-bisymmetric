#!/usr/bin/env bash
# run_review7.sh — robustness tests requested in the 7th review, on the author's
# computer (needs the cutouts, catalogues, PSFs and the outputs of run_final.sh).
#
#   1. r_d definition (diagnose_sample.py)                          ~1 min
#   2. stellar mass: mass-matched, mass slices, regression           ~1 min
#   3. self-consistent calibration: injections at the FINAL Table 1
#      amplitudes (nominal + 4 systematics), Table 1 recomputed      ~15 min (parallel)
#   4. clumpy models (2 settings) and amplitude spread (2 settings)  ~15 min (parallel)
#   5. annulus x0.6 and x1.68: measurement + injections + Table 1    ~30-60 min
#   6. review7_summary.py -> logs/review7_summary.md, data/review7_*.csv
#
# Run from the package's scripts/ directory:
#   cd ~/Downloads/pasp_v3/pasp_SUBMISSION/scripts
#   ROOT=~/Downloads/pasp_v3 bash run_review7.sh 2>&1 | tee ../logs/author_machine_run_review7.log
# Optional env: D (DJA directory, default /scratch/pulsating_disk/dja_analysis),
#   DATA (default $ROOT/data), NPB (default 300), FORCE=1, SKIP_ANNULUS=1.
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
D=${D:-/scratch/pulsating_disk/dja_analysis}
M=$D/full-good_morpho-phot.fits.gz
DATA=${DATA:-$ROOT/data}
NPB=${NPB:-300}
RREF=${RREF:-0.04}   # arcsec/pix of DISK_RADIUS_pix (DJA: DISK_RADIUS_deg/DISK_RADIUS_pix = 0.04")
FORCE=${FORCE:-0}
cd "$(dirname "$0")"
PKG=$(cd .. && pwd)
LOG=$ROOT/review7_logs; mkdir -p "$LOG" "$PKG/logs"

echo "=== 0 inputs"
need=("$ROOT/table1_disk_final.csv" "$ROOT/corrected_disk_f444w.csv" "$DATA/dja_real.csv"
      "$DATA/dja_snr_f444w.csv" "$ROOT/noise_acf_f444w.npy" "$ROOT/psf/F444W_empirical_psf.fits"
      "$ROOT/psf/F444W_field_offsets.csv" "$M")
[ "${SKIP_ANNULUS:-0}" = 1 ] || need+=("$D/cutouts_full")
miss=0
for f in "${need[@]}"; do [ -e "$f" ] || { echo "  missing: $f"; miss=1; }; done
[ $miss -eq 0 ] || { echo "STOP: set D=... or DATA=... if the files are elsewhere"; exit 1; }
echo "  all inputs present"

inj() {   # inj <outdir> [options]: F444W injection at the FINAL Table 1 amplitudes
    local out=$1; shift
    if [ "$FORCE" != 1 ] && [ -f "$out/injection_per_object.csv" ]; then echo "  $(basename "$out") exists"; return 0; fi
    python3 injection_test.py --catalog "$DATA/dja_snr_f444w.csv" --re_col disk_r_pix --re_unit pix \
        --re_ref_scale "$RREF" --n_col n_sersic --snr_col snr_ap --pixscale 0.05 --cutout_arcsec 5.0 \
        --n_per_bin "$NPB" --centering true,catalog --sigma_cat 0.10 \
        --psf "$ROOT/psf/F444W_empirical_psf.fits" --psf_pixscale 0.05 \
        --noise_acf "$ROOT/noise_acf_f444w.npy" --am_func am_wrapper.py:measure_v6 \
        --scenarios zero --matched_table "$ROOT/table1_disk_final.csv" --matched_col final_std \
        "$@" --output "$out/" > "$LOG/$(basename "$out").log" 2>&1
}
waitall() { local p; for p in "$@"; do wait "$p" || { echo "STOP: a run failed (logs in $LOG)"; exit 1; }; done; }

echo "=== 1 r_d definition"
python3 diagnose_sample.py --objects "$ROOT/corrected_disk_f444w.csv" --sample "$DATA/dja_real.csv" \
    --morpho "$M" > "$PKG/logs/review7_diagnose_sample.log" 2>&1 \
    || echo "  diagnose_sample.py failed (see logs/review7_diagnose_sample.log); continuing"
tail -25 "$PKG/logs/review7_diagnose_sample.log"

echo "=== 1b F277W/F444W ratio of the same galaxies"
python3 filter_ratio.py --ref "$ROOT/corrected_disk_f444w.csv" --other "$ROOT/corrected_disk_f277w.csv" \
    --out "$PKG/data/filter_ratio_F277W_F444W.csv" | tee "$PKG/logs/review7_filter_ratio.log"

echo "=== 2 stellar mass"
python3 mass_test.py --objects "$ROOT/corrected_disk_f444w.csv" --catalog "$DATA/dja_real.csv" \
    --table "$PKG/data/table1_disks_F444W.csv" --out "$PKG/data/mass_test" \
    > "$PKG/logs/review7_mass_test.log" 2>&1
sed -n '/growth z/p;/(c)/,/Note/p' "$PKG/logs/review7_mass_test.log"

echo "=== 3 + 4 injections (parallel; logs in $LOG)"
pids=()
inj "$ROOT/inj_f444w_sc_m" & pids+=($!)
inj "$ROOT/inj_f444w_sc_scale0.5" --matched_scale 0.5 & pids+=($!)
inj "$ROOT/inj_f444w_sc_scale1.5" --matched_scale 1.5 & pids+=($!)
inj "$ROOT/inj_f444w_sc_rising" --pert_profile rising & pids+=($!)
inj "$ROOT/inj_f444w_sc_falling" --pert_profile falling & pids+=($!)
waitall "${pids[@]}"; pids=()
inj "$ROOT/inj_f444w_clumps4_f0.2" --clumps 4 --clump_frac 0.2 & pids+=($!)
inj "$ROOT/inj_f444w_clumps8_f0.3" --clumps 8 --clump_frac 0.3 & pids+=($!)
inj "$ROOT/inj_f444w_spread0.15" --matched_spread 0.15 & pids+=($!)
inj "$ROOT/inj_f444w_spread0.30" --matched_spread 0.30 & pids+=($!)
waitall "${pids[@]}"
SYST=""
for L in scale0.5 scale1.5 rising falling; do SYST="$SYST,$L=$ROOT/inj_f444w_sc_$L/injection_summary.csv"; done
python3 make_table1.py --objects "$ROOT/corrected_disk_f444w.csv" \
    --injection "$ROOT/inj_f444w_sc_m/injection_summary.csv" --syst "${SYST#,}" \
    --out "$ROOT/table1_disk_sc" | sed -n '/Final values/,/without a valid/p'

if [ "${SKIP_ANNULUS:-0}" != 1 ]; then
    echo "=== 5 annulus x0.6 and x1.68 (measurement + injections)"
    for S in 0.6 1.68; do
        if [ "$FORCE" = 1 ] || [ ! -f "$ROOT/corrected_disk_f444w_r$S.csv" ]; then
            python3 run_corrected_am.py --catalog "$DATA/dja_real.csv" --cutouts "$D/cutouts_full/" \
                --pattern 'g_{:06d}.fits' --pa_fits "$M" --r_ref_scale "$RREF" --r_scale "$S" \
                --centre_offsets "$ROOT/psf/F444W_field_offsets.csv" \
                --output "$ROOT/corrected_disk_f444w_r$S.csv" > "$LOG/measure_r$S.log" 2>&1 &
            pids+=($!)
        fi
        inj "$ROOT/inj_f444w_r$S" --ring_scale "$S" & pids+=($!)
    done
    waitall "${pids[@]}"
    for S in 0.6 1.68; do
        python3 make_table1.py --objects "$ROOT/corrected_disk_f444w_r$S.csv" \
            --injection "$ROOT/inj_f444w_r$S/injection_summary.csv" --exploratory \
            --out "$ROOT/table1_disk_r$S" > "$LOG/table1_r$S.log" 2>&1
        echo "  ring x$S done"
    done
fi

echo "=== 6 summary"
python3 review7_summary.py --root "$ROOT" --pkg "$PKG"
for f in table1_disk_sc.csv table1_disk_r0.6.csv table1_disk_r1.68.csv; do
    [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$PKG/data/review7_$f"
done
echo
echo "FINISHED. Send back:  cd $ROOT && zip -r review7_results.zip pasp_SUBMISSION/data pasp_SUBMISSION/logs"
