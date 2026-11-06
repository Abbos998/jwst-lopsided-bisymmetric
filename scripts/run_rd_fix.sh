#!/usr/bin/env bash
# run_rd_fix.sh — redo the whole analysis with the correct pixel scale of the DJA disk
# radius. DISK_RADIUS_deg/DISK_RADIUS_pix = 0.040"/pix for every galaxy, while the
# previous runs assumed 0.05"/pix (--r_ref_scale 0.05): the annulus was 0.63-1.88 r_d
# instead of 0.5-1.5 r_d, and the injected models were 25% too large.
#
#   0. back up the previous outputs to $ROOT/old_rref0.05/
#   1. A_m of all disks in F444W, F277W, F150W and of the early types, r_ref_scale 0.04
#   2. injections (r_ref_scale 0.04), two iterations of the self-consistent calibration:
#        it0: matched amplitudes = previous final Table 1
#        it1: matched amplitudes = Table 1 of it0       (F444W nominal + 4 systematics)
#      plus F444W zero/constant (early types), F277W and F150W (rest-frame test)
#   3. early-type table, SFR split
#   4. build_all.sh (tables, numbers, figures, PDF, checks) and check_all.sh
#   5. run_review7.sh (robustness tests at the new scale)
#
# Run from the package's scripts/ directory (~3-4 h, mostly parallel):
#   cd ~/Downloads/pasp_v3/pasp_SUBMISSION/scripts
#   ROOT=~/Downloads/pasp_v3 bash run_rd_fix.sh 2>&1 | tee ../logs/author_machine_run_rd_fix.log
# Optional env: D (DJA directory), DATA (default $ROOT/data), NPB (300), RREF (0.04),
#               SKIP_REVIEW7=1 (skip step 5).
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
D=${D:-/scratch/pulsating_disk/dja_analysis}
M=$D/full-good_morpho-phot.fits.gz
DATA=${DATA:-$ROOT/data}
NPB=${NPB:-300}
RREF=${RREF:-0.04}
export RREF
cd "$(dirname "$0")"
PKG=$(cd .. && pwd)
LOG=$ROOT/rd_fix_logs; mkdir -p "$LOG" "$PKG/logs"
OFF=$ROOT/psf/F444W_field_offsets.csv

echo "=== 0 inputs and backup"
need=("$M" "$DATA/dja_real.csv" "$D/full_ell_A1_results.csv" "$D/cutouts_full" "$D/cutouts_f277w"
      "$D/cutouts_f150w" "$D/cutouts_ell_full" "$OFF" "$ROOT/table1_disk_final.csv")
for f in f444w f277w f150w; do
    F=$(echo $f | tr a-z A-Z)
    need+=("$DATA/dja_snr_$f.csv" "$ROOT/noise_acf_$f.npy" "$ROOT/psf/${F}_empirical_psf.fits")
done
miss=0
for f in "${need[@]}"; do [ -e "$f" ] || { echo "  missing: $f"; miss=1; }; done
[ $miss -eq 0 ] || { echo "STOP: set D=... or DATA=... if the files are elsewhere"; exit 1; }
B=$ROOT/old_rref0.05
if [ ! -d "$B" ]; then
    mkdir -p "$B"
    for f in "$ROOT"/corrected_*.csv "$ROOT"/table1_*.csv "$ROOT"/restframe_disk_m.csv "$ROOT"/sfr_split_snr100.csv; do
        [ -f "$f" ] && cp "$f" "$B/"
    done
    for d in "$ROOT"/inj_*; do [ -d "$d" ] && cp -r "$d" "$B/"; done
    echo "  previous outputs copied to $B"
else
    echo "  backup $B already exists (kept)"
fi
cp "$B/table1_disk_final.csv" "$ROOT/table1_start.csv"     # amplitudes for iteration 0

waitall() { local p; for p in "$@"; do wait "$p" || { echo "STOP: a run failed (logs in $LOG)"; exit 1; }; done; }
pixscale() { python3 -c "import pandas as pd; print(round(pd.read_csv('$DATA/dja_snr_$1.csv').pixscale.median(), 4))" 2>/dev/null || echo 0.05; }
psfscale() { python3 -c "from astropy.io import fits; print(fits.getheader('$ROOT/psf/$1_empirical_psf.fits').get('PIXSCALE', 0.05))"; }
inj() {   # inj <filter> <outdir> <matched_table|-> [options]
    local f=$1 out=$2 mt=$3; shift 3
    local F; F=$(echo "$f" | tr a-z A-Z)
    local extra=(--scenarios zero)
    [ "$mt" = "-" ] && extra=(--scenarios zero,constant) || extra+=(--matched_table "$mt" --matched_col final_std)
    rm -rf "$out"
    python3 injection_test.py --catalog "$DATA/dja_snr_$f.csv" --re_col disk_r_pix --re_unit pix \
        --re_ref_scale "$RREF" --n_col n_sersic --snr_col snr_ap --pixscale "$(pixscale "$f")" \
        --cutout_arcsec 5.0 --n_per_bin "$NPB" --centering true,catalog --sigma_cat 0.10 \
        --psf "$ROOT/psf/${F}_empirical_psf.fits" --psf_pixscale "$(psfscale "$F")" \
        --noise_acf "$ROOT/noise_acf_$f.npy" --am_func am_wrapper.py:measure_v6 \
        "${extra[@]}" "$@" --output "$out/" > "$LOG/$(basename "$out").log" 2>&1
}
table1() {   # table1 <out> : F444W Table 1 from inj_f444w_m and the four systematics runs
    local S=""
    for L in scale0.5 scale1.5 rising falling; do S="$S,$L=$ROOT/inj_f444w_$L/injection_summary.csv"; done
    python3 make_table1.py --objects "$ROOT/corrected_disk_f444w.csv" \
        --injection "$ROOT/inj_f444w_m/injection_summary.csv" --syst "${S#,}" --out "$1" \
        | sed -n '/Final values/,/without a valid/p'
}
f444w_runs() {   # f444w_runs <matched_table>: nominal + 4 systematics, in parallel
    local pids=()
    inj f444w "$ROOT/inj_f444w_m" "$1" & pids+=($!)
    inj f444w "$ROOT/inj_f444w_scale0.5" "$1" --matched_scale 0.5 & pids+=($!)
    inj f444w "$ROOT/inj_f444w_scale1.5" "$1" --matched_scale 1.5 & pids+=($!)
    inj f444w "$ROOT/inj_f444w_rising" "$1" --pert_profile rising & pids+=($!)
    inj f444w "$ROOT/inj_f444w_falling" "$1" --pert_profile falling & pids+=($!)
    waitall "${pids[@]}"
}

echo "=== 1 measurements with r_ref_scale $RREF (parallel; logs in $LOG)"
pids=()
for f in f444w f277w f150w; do
    CUT=$D/cutouts_$f; [ $f = f444w ] && CUT=$D/cutouts_full
    python3 run_corrected_am.py --catalog "$DATA/dja_real.csv" --cutouts "$CUT/" --pattern 'g_{:06d}.fits' \
        --pa_fits "$M" --r_ref_scale "$RREF" --centre_offsets "$OFF" \
        --output "$ROOT/corrected_disk_$f.csv" > "$LOG/measure_$f.log" 2>&1 &
    pids+=($!)
done
python3 run_corrected_am.py --catalog "$D/full_ell_A1_results.csv" --cutouts "$D/cutouts_ell_full/" \
    --pattern 'e_{:06d}.fits' --q_col axratio --n_col sersic_n --no_cuts --pa_fits "$M" \
    --r_ref_scale "$RREF" --centre_offsets "$OFF" --output "$ROOT/corrected_ell_v7.csv" \
    > "$LOG/measure_ell.log" 2>&1 &
pids+=($!)
waitall "${pids[@]}"
python3 - "$ROOT" <<'PY'
import sys, pandas as pd
R = sys.argv[1]
for f in ('f444w', 'f277w', 'f150w'):
    d = pd.read_csv(f'{R}/corrected_disk_{f}.csv')
    print(f"  {f}: {d.A1_V6.notna().sum()} valid of {len(d)}")
e = pd.read_csv(f'{R}/corrected_ell_v7.csv'); print(f"  early types: {e.A1_V6.notna().sum()} valid of {len(e)}")
PY

echo "=== 2a injections, iteration 0 (matched = previous Table 1)"
pids=()
inj f444w "$ROOT/inj_f444w" - & pids+=($!)
inj f277w "$ROOT/inj_f277w_m" "$ROOT/table1_start.csv" & pids+=($!)
inj f150w "$ROOT/inj_f150w_m" "$ROOT/table1_start.csv" & pids+=($!)
f444w_runs "$ROOT/table1_start.csv"
waitall "${pids[@]}"
table1 "$ROOT/table1_it0"

echo "=== 2b iteration 1 (matched = Table 1 of iteration 0)"
f444w_runs "$ROOT/table1_it0.csv"
table1 "$ROOT/table1_disk_final"
python3 - "$ROOT" <<'PY'
import sys, numpy as np, pandas as pd
R = sys.argv[1]
a, b = pd.read_csv(f'{R}/table1_it0.csv'), pd.read_csv(f'{R}/table1_disk_final.csv')
for k, e in (('A1_final_std', 'A1_final_err'), ('A2_final_std', 'A2_final_err'),
             ('R12_debiased_matched', 'R12_final_err')):
    s = (b[k] - a[k]) / b[e]
    print(f"  convergence {k}: max |it1 - it0| = {np.nanmax(np.abs(s)):.2f} sigma")
PY

echo "=== 3 early types, SFR split"
python3 make_table1.py --objects "$ROOT/corrected_ell_v7.csv" --injection "$ROOT/inj_f444w/injection_summary.csv" \
    --exploratory --out "$ROOT/table1_ell_v7" > "$LOG/table1_ell.log" 2>&1
python3 fix_sfr.py --morpho "$M" --objects "$ROOT/corrected_disk_f444w.csv" --sample "$DATA/dja_real.csv" \
    --injection "$ROOT/inj_f444w_m/injection_summary.csv" --snr_csv "$DATA/dja_snr_f444w.csv" \
    --snr_min 100 --out "$ROOT/sfr_split_snr100" > "$LOG/sfr.log" 2>&1
tail -3 "$LOG/sfr.log"

echo "=== 4 build_all.sh + check_all.sh"
rc=0; ROOT=$ROOT MANUSCRIPT_CHECKS=core bash build_all.sh || rc=$?
[ $rc -eq 0 ] || [ $rc -eq 3 ] || { echo "STOP: build_all.sh failed ($rc)"; exit 1; }
[ $rc -eq 3 ] && echo "  (no LaTeX here: the PDF is compiled later; continuing)"
bash check_all.sh || echo "  check_all.sh reported a failure (see logs/); continuing"

if [ "${SKIP_REVIEW7:-0}" != 1 ]; then
    echo "=== 5 robustness tests at the new scale"
    rm -rf "$ROOT"/inj_f444w_sc_* "$ROOT"/inj_f444w_clumps* "$ROOT"/inj_f444w_spread* "$ROOT"/inj_f444w_r0.6 \
        "$ROOT"/inj_f444w_r1.68 "$ROOT"/corrected_disk_f444w_r0.6.csv "$ROOT"/corrected_disk_f444w_r1.68.csv
    ROOT=$ROOT D=$D DATA=$DATA NPB=$NPB bash run_review7.sh
fi
echo
echo "FINISHED. Send back:  cd $ROOT && zip -r rd_fix_results.zip pasp_SUBMISSION -x 'pasp_SUBMISSION/previous_version/*'"
