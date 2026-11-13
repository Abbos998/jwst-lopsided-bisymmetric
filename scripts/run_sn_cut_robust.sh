#!/usr/bin/env bash
# run_sn_cut_robust.sh -- step 4b: rerun the robustness tests of the paper for the
# S/N-limited sample and finish the build (figures, PDF). Works only inside $NEW
# (default ~/Downloads/pasp_v4_sn40) and its copy of the package; $ROOT is read-only.
#
#   1. links and the annulus measurements (x0.6, x1.68) restricted to the sample
#   2. run_review7.sh      self-consistent calibration, clumps, spread, annuli, mass
#   3. build_all.sh        tables, numbers, figures, PDF (full checks)
#   4. run_calib_tests.sh, run_calib_followup.sh, run_seed_scatter.sh   (Tables 6, 7)
#   5. run_selection_tests.sh, run_matched_bootstrap.sh, run_snr_check.sh (Table B1)
# A failing stage is reported and the script continues with the next one.
#
# Run from the ORIGINAL package's scripts/ directory (1-3 hours):
#   ROOT=~/Downloads/pasp_v3 bash run_sn_cut_robust.sh 2>&1 | tee ../logs/run_sn_cut_robust.log
# Optional env: THR (40), NEW, D, STAGES (default "review7 build calib selection")
set -uo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
D=${D:-/scratch/pulsating_disk/dja_analysis}
THR=${THR:-40}
NEW=${NEW:-$HOME/Downloads/pasp_v4_sn$THR}
STAGES=${STAGES:-review7 build calib selection}
SRC=$(cd "$(dirname "$0")/.." && pwd)
PKG=$NEW/pasp_SUBMISSION
LOG=$NEW/logs
[ -d "$PKG/scripts" ] && [ -f "$NEW/table1_disk_final.csv" ] || { echo "STOP: run run_sn_cut.sh and run_sn_cut_build.sh first"; exit 1; }
cp "$SRC"/scripts/*.py "$SRC"/scripts/*.sh "$PKG/scripts/"

echo "=== 1 links and annulus measurements"
ln -sf "$ROOT/data/dja_real.csv" "$NEW/data/dja_real.csv"
for f in F444W_field_offsets.csv F444W_stars.csv stars_f444w; do
    [ -e "$ROOT/psf/$f" ] && ln -sfn "$ROOT/psf/$f" "$NEW/psf/$f" || echo "  note: $ROOT/psf/$f not found"
done
python3 - "$ROOT" "$NEW" <<'PY'
import os
import sys
import pandas as pd
R, N = sys.argv[1], sys.argv[2]
keep = set(pd.read_csv(f'{N}/corrected_disk_f444w.csv', usecols=['idx']).idx)
for s in ('0.6', '1.68'):
    p = f'{R}/corrected_disk_f444w_r{s}.csv'
    if not os.path.exists(p):
        print(f'  {p} not found: run_review7.sh will measure this annulus again (all S/N!)')
        continue
    d = pd.read_csv(p)
    k = d[d.idx.isin(keep)]
    k.to_csv(f'{N}/corrected_disk_f444w_r{s}.csv', index=False)
    print(f'  annulus x{s}: {len(k)} of {len(d)} galaxies, {int(k.A1_V6.notna().sum())} valid')
PY

stage() {   # stage <name> <log> <command...>
    local name=$1 log=$2; shift 2
    echo "--- $name (log: $LOG/$log)"
    local t0=$SECONDS rc=0
    (cd "$PKG/scripts" && ROOT=$NEW DATA=$NEW/data D=$D "$@") > "$LOG/$log" 2>&1 || rc=$?
    echo "    exit code $rc after $(( (SECONDS - t0) / 60 )) min"
    [ $rc -eq 0 ] || { echo "    last lines:"; tail -6 "$LOG/$log" | sed 's/^/      /'; }
}

for S in $STAGES; do
    case $S in
    review7)
        echo "=== 2 run_review7.sh"
        stage review7 robust_review7.log bash run_review7.sh ;;
    build)
        echo "=== 3 build_all.sh"
        stage build robust_build_all.log bash build_all.sh
        grep -E '\[FAIL\]|LaTeX errors|check\(s\) failed|^Done' "$LOG/robust_build_all.log" | sed 's/^/    /' || true ;;
    calib)
        echo "=== 4 calibration tests (Tables 6, 7)"
        stage calib_tests robust_calib_tests.log bash run_calib_tests.sh
        stage calib_followup robust_calib_followup.log bash run_calib_followup.sh
        stage seed_scatter robust_seed_scatter.log bash run_seed_scatter.sh ;;
    selection)
        echo "=== 5 selection tests (Table B1)"
        stage selection_tests robust_selection_tests.log bash run_selection_tests.sh
        stage matched_bootstrap robust_matched_bootstrap.log bash run_matched_bootstrap.sh
        stage snr_check robust_snr_check.log bash run_snr_check.sh ;;
    *) echo "unknown stage $S" ;;
    esac
done
echo
echo "FINISHED. Send back (attach the file to the chat):"
echo "  cd $NEW && zip -r sn_cut_all.zip logs table1_*.csv sfr_split_snr100.csv restframe_disk_m.csv fields_sn${THR}_summary.txt calib_tests selection_tests pasp_SUBMISSION/data pasp_SUBMISSION/tex pasp_SUBMISSION/logs pasp_SUBMISSION/figures -x '*.png' -x '*per_object.csv' -x '*/psf_variants/*' -x '*_rows.csv'"
