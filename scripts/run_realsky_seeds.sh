#!/usr/bin/env bash
# run_realsky_seeds.sh -- repeat real_sky_injection.py over several seeds and
# aggregate to mean +/- sd (error bars on the real-sky floor and growth).
#
#   ROOT=~/Downloads/pasp_v3 bash run_realsky_seeds.sh 2>&1 | tee ../logs/realsky_seeds.log
# Optional env: K (seeds, default 6), NPB (n_per_bin, 300), D (DJA dir), DATA, PSF.
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
DATA=${DATA:-$ROOT/data}
D=${D:-/scratch/pulsating_disk/dja_analysis}
K=${K:-6}
NPB=${NPB:-300}
PSF=${PSF:-$ROOT/psf/F444W_empirical_psf.fits}
cd "$(dirname "$0")"
OUT=$ROOT/referee_tests; mkdir -p "$OUT"

for s in $(seq 1 "$K"); do
    echo "=== seed $s/$K"
    python3 real_sky_injection.py \
        --catalog "$DATA/dja_snr_f444w.csv" \
        --cutouts "$D/cutouts_full/" --pattern 'g_{:06d}.fits' \
        --table1 ../data/table1_disks_F444W.csv \
        --sample "$DATA/dja_real.csv" \
        --psf "$PSF" --psf_pixscale 0.04 --pixscale 0.05 \
        --re_col disk_r_pix --re_unit pix --re_ref_scale 0.04 --n_col n_sersic --snr_col snr_ap \
        --am_func am_wrapper.py:measure_v6 --n_per_bin "$NPB" --seed "$s" \
        --out "$OUT/realsky_s${s}" > "$OUT/realsky_s${s}.log" 2>&1 \
        || { echo "  seed $s failed, see $OUT/realsky_s${s}.log"; tail -5 "$OUT/realsky_s${s}.log"; exit 1; }
    echo "  done"
done

echo "=== aggregating $K seeds"
python3 aggregate_realsky.py --prefix "$OUT/realsky" --out "$OUT/realsky_seeds"
echo
echo "FINISHED. Send back: $OUT/realsky_seeds_summary.txt"
