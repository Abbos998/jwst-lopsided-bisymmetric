#!/usr/bin/env bash
# run_filter.sh FILTER — full V6 pipeline + calibration for one NIRCam filter.
#   1. A_m (V6) on that filter's cutouts           -> ../corrected_disk_<f>.csv
#   2. per-galaxy S/N and pixel scale               -> /tmp/dja_snr_<f>.csv
#   3. noise autocorrelation                        -> ../noise_acf_<f>.npy
#   4. injection floors/recovery with that filter's empirical PSF, pixel
#      scale, noise ACF and S/N                     -> ../inj_<f>/
# Needs: cutouts in $D/cutouts_<f>/ (fetch_cutouts.py; F444W uses cutouts_full)
#        ../psf/<FILTER>_empirical_psf.fits (star_psf_astrometry.py)
# Uses the per-field centre offsets ../psf/F444W_field_offsets.csv (the same in
# all filters: an astrometric offset of the catalogue, measured on stars).
# Optional env: SIGMA_CAT (default 0.1), NPB (galaxies per z bin, default 300),
#               DATA (default ~/Downloads/pasp_v3/data: dja_real.csv, S/N tables)
set -euo pipefail
F=$(echo "$1" | tr a-z A-Z); f=$(echo "$F" | tr A-Z a-z)
D=/scratch/pulsating_disk/dja_analysis
M=$D/full-good_morpho-phot.fits.gz
CUT=$D/cutouts_$f
[ "$F" = "F444W" ] && CUT=$D/cutouts_full
SIG=${SIGMA_CAT:-0.1}; NPB=${NPB:-300}
RREF=${RREF:-0.04}   # arcsec/pix of DISK_RADIUS_pix (DJA: DISK_RADIUS_deg/DISK_RADIUS_pix = 0.04")
DATA=${DATA:-$HOME/Downloads/pasp_v3/data}   # persistent (not /tmp)
CAT=$DATA/dja_real.csv
[ -f "$CAT" ] || { echo "No catalogue $CAT"; exit 1; }
PSF=../psf/${F}_empirical_psf.fits
[ -d "$CUT" ] || { echo "No cutouts in $CUT"; exit 1; }
[ -f "$PSF" ] || { echo "No PSF $PSF (run star_psf_astrometry.py)"; exit 1; }

echo "=== [$F] 1/4 V6 measurement"
python3 run_corrected_am.py --catalog "$CAT" --cutouts "$CUT/" --pattern 'g_{:06d}.fits' \
    --pa_fits "$M" --r_ref_scale "$RREF" --centre_offsets ../psf/F444W_field_offsets.csv \
    --output "../corrected_disk_$f.csv" 2>&1 \
    | grep -v Warn | grep -E "Processed|convention|using|V6|rise" || true

echo "=== [$F] 2/4 S/N"
python3 measure_snr.py --catalog "$CAT" --cutouts "$CUT/" --output "$DATA/dja_snr_$f.csv" 2>&1 \
    | grep -v Warn | sed -n '/Pixel scale/,/S\/N_ap < 10/p'

echo "=== [$F] 3/4 noise ACF"
python3 measure_noise_acf.py --catalog "$CAT" --cutouts "$CUT/" --output "../noise_acf_$f.npy" 2>&1 \
    | grep -v Warn | head -9

PS=$(python3 -c "import pandas as pd; print(round(pd.read_csv('$DATA/dja_snr_$f.csv').pixscale.median(), 4))")
PSFPS=$(python3 -c "from astropy.io import fits; print(fits.getheader('$PSF')['PIXSCALE'])")
echo "=== [$F] 4/4 injection (pixscale $PS, PSF pixscale $PSFPS, sigma_cat $SIG)"
python3 injection_test.py --catalog "$DATA/dja_snr_$f.csv" --re_col disk_r_pix --re_unit pix \
    --re_ref_scale "$RREF" --n_col n_sersic --snr_col snr_ap --pixscale "$PS" --cutout_arcsec 5.0 \
    --n_per_bin "$NPB" --centering true,catalog --sigma_cat "$SIG" --psf "$PSF" --psf_pixscale "$PSFPS" \
    --noise_acf "../noise_acf_$f.npy" --am_func am_wrapper.py:measure_v6 --output "../inj_$f/" 2>&1 \
    | grep -v Warn | sed -n '/CONSTANT/,/observed/p'
echo "=== [$F] done: ../corrected_disk_$f.csv  ../inj_$f/injection_summary.csv"
