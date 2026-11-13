#!/usr/bin/env bash
# run_sn_cut.sh -- recalibrate the measurement for the S/N-limited sample
# (aperture S/N in F444W >= THR, default 40; threshold from snr_threshold_test.py).
#
# Nothing in $ROOT or in the package is changed: all outputs go to a new
# directory $NEW (default ~/Downloads/pasp_v4_sn40).
#
#   0. $NEW: the S/N catalogues and per-galaxy measurements of the three filters
#      restricted to the galaxies with F444W aperture S/N >= THR; PSFs and noise
#      autocorrelations are linked from $ROOT
#   1. iteration 0: injections (nominal + 4 systematics) at the amplitudes of the
#      previous Table 1  ->  table1_it0
#   2. iteration 1: injections at the amplitudes of table1_it0 (F444W nominal +
#      4 systematics, F277W, F150W)  ->  table1_disk_final, convergence check
#   3. growth factors of the new Table 1
#
# Run from the package's scripts/ directory (about 30-60 min):
#   ROOT=~/Downloads/pasp_v3 bash run_sn_cut.sh 2>&1 | tee ../logs/run_sn_cut.log
# Optional env: THR (40), NEW, DATA ($ROOT/data), NPB (300), RREF (0.04), NBOOT (1000)
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
DATA=${DATA:-$ROOT/data}
THR=${THR:-40}
NEW=${NEW:-$HOME/Downloads/pasp_v4_sn$THR}
NPB=${NPB:-300}
RREF=${RREF:-0.04}
NBOOT=${NBOOT:-1000}
cd "$(dirname "$0")"
LOG=$NEW/logs
mkdir -p "$NEW/data" "$LOG"

echo "=== 0 inputs (S/N >= $THR) -> $NEW"
need=("$ROOT/table1_disk_final.csv")
for f in f444w f277w f150w; do
    F=$(echo $f | tr a-z A-Z)
    need+=("$ROOT/corrected_disk_$f.csv" "$DATA/dja_snr_$f.csv" "$ROOT/noise_acf_$f.npy" "$ROOT/psf/${F}_empirical_psf.fits")
done
miss=0
for f in "${need[@]}"; do [ -e "$f" ] || { echo "  missing: $f"; miss=1; }; done
[ $miss -eq 0 ] || { echo "STOP: set ROOT / DATA"; exit 1; }
python3 - "$ROOT" "$DATA" "$NEW" "$THR" <<'PY'
import sys
import numpy as np
import pandas as pd
R, D, N, thr = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
s = pd.read_csv(f'{D}/dja_snr_f444w.csv')
if 'idx' not in s or 'snr_ap' not in s:
    sys.exit(f'STOP: idx / snr_ap not in dja_snr_f444w.csv: {list(s.columns)}')
keep = set(s.loc[np.isfinite(s.snr_ap) & (s.snr_ap >= thr), 'idx'])
print(f'  F444W catalogue: {len(keep)} of {len(s)} galaxies with S/N >= {thr:g}')
ED = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]
for f in ('f444w', 'f277w', 'f150w'):
    c = pd.read_csv(f'{D}/dja_snr_{f}.csv')
    if 'idx' not in c:
        sys.exit(f'STOP: no idx column in dja_snr_{f}.csv')
    c[c.idx.isin(keep)].to_csv(f'{N}/data/dja_snr_{f}.csv', index=False)
    o = pd.read_csv(f'{R}/corrected_disk_{f}.csv')
    k = o[o.idx.isin(keep)]
    k.to_csv(f'{N}/corrected_disk_{f}.csv', index=False)
    v = k.dropna(subset=['A1_V6', 'A2_V6'])
    n = np.histogram(v.z, ED)[0]
    print(f'  {f}: catalogue {int(c.idx.isin(keep).sum())}, measured galaxies {len(k)} of {len(o)}, '
          f'valid {len(v)}; valid per bin {list(n)}')
PY
mkdir -p "$NEW/psf"
for f in f444w f277w f150w; do
    F=$(echo $f | tr a-z A-Z)
    ln -sf "$ROOT/noise_acf_$f.npy" "$NEW/noise_acf_$f.npy"
    ln -sf "$ROOT/psf/${F}_empirical_psf.fits" "$NEW/psf/${F}_empirical_psf.fits"
done
cp "$ROOT/table1_disk_final.csv" "$NEW/table1_start.csv"

waitall() { local p; for p in "$@"; do wait "$p" || { echo "STOP: a run failed (logs in $LOG)"; exit 1; }; done; }
pixscale() { python3 -c "import pandas as pd; print(round(pd.read_csv('$NEW/data/dja_snr_$1.csv').pixscale.median(), 4))" 2>/dev/null || echo 0.05; }
psfscale() { python3 -c "from astropy.io import fits; print(fits.getheader('$NEW/psf/$1_empirical_psf.fits').get('PIXSCALE', 0.05))"; }
inj() {   # inj <filter> <outdir> <matched_table> [options]
    local f=$1 out=$2 mt=$3; shift 3
    local F; F=$(echo "$f" | tr a-z A-Z)
    rm -rf "$out"
    python3 injection_test.py --catalog "$NEW/data/dja_snr_$f.csv" --re_col disk_r_pix --re_unit pix \
        --re_ref_scale "$RREF" --n_col n_sersic --snr_col snr_ap --pixscale "$(pixscale "$f")" \
        --cutout_arcsec 5.0 --n_per_bin "$NPB" --centering true,catalog --sigma_cat 0.10 \
        --psf "$NEW/psf/${F}_empirical_psf.fits" --psf_pixscale "$(psfscale "$F")" \
        --noise_acf "$NEW/noise_acf_$f.npy" --am_func am_wrapper.py:measure_v6 \
        --scenarios zero --matched_table "$mt" --matched_col final_std "$@" --output "$out/" \
        > "$LOG/$(basename "$out").log" 2>&1
}
table1() {
    local S=""
    for L in scale0.5 scale1.5 rising falling; do S="$S,$L=$NEW/inj_f444w_$L/injection_summary.csv"; done
    python3 make_table1.py --objects "$NEW/corrected_disk_f444w.csv" \
        --injection "$NEW/inj_f444w_m/injection_summary.csv" --syst "${S#,}" --nboot "$NBOOT" --out "$1" \
        | sed -n '/Final values/,/without a valid/p'
}
f444w_runs() {
    local pids=()
    inj f444w "$NEW/inj_f444w_m" "$1" & pids+=($!)
    inj f444w "$NEW/inj_f444w_scale0.5" "$1" --matched_scale 0.5 & pids+=($!)
    inj f444w "$NEW/inj_f444w_scale1.5" "$1" --matched_scale 1.5 & pids+=($!)
    inj f444w "$NEW/inj_f444w_rising" "$1" --pert_profile rising & pids+=($!)
    inj f444w "$NEW/inj_f444w_falling" "$1" --pert_profile falling & pids+=($!)
    waitall "${pids[@]}"
}

echo "=== 1 iteration 0 (matched = previous Table 1; logs in $LOG)"
f444w_runs "$NEW/table1_start.csv"
table1 "$NEW/table1_it0"

echo "=== 2 iteration 1 (matched = Table 1 of iteration 0) + F277W, F150W"
pids=()
inj f277w "$NEW/inj_f277w_m" "$NEW/table1_it0.csv" & pids+=($!)
inj f150w "$NEW/inj_f150w_m" "$NEW/table1_it0.csv" & pids+=($!)
f444w_runs "$NEW/table1_it0.csv"
waitall "${pids[@]}"
table1 "$NEW/table1_disk_final"

echo "=== 3 convergence and growth factors"
python3 - "$NEW" "$ROOT" <<'PY'
import sys
import numpy as np
import pandas as pd
N, R = sys.argv[1], sys.argv[2]
a, b = pd.read_csv(f'{N}/table1_it0.csv'), pd.read_csv(f'{N}/table1_disk_final.csv')
for k, e in (('A1_final_std', 'A1_final_err'), ('A2_final_std', 'A2_final_err')):
    print(f"  convergence {k}: max |it1 - it0| = {np.nanmax(np.abs((b[k] - a[k]) / b[e])):.2f} sigma")


def growth(t, label):
    t = t.set_index(t.z.round(2))
    out = []
    for k in (1, 2):
        lo, elo = t.loc[0.75, f'A{k}_final_std'], t.loc[0.75, f'A{k}_final_err']
        hi = t.loc[[4.5, 5.5], f'A{k}_final_std'].mean()
        ehi = np.hypot(*t.loc[[4.5, 5.5], f'A{k}_final_err']) / 2
        g = hi / lo
        out.append(f"A{k}: {lo:.3f} -> {hi:.3f}, growth {g:.2f} +- {g * np.hypot(elo / lo, ehi / hi):.2f}")
    print(f"  {label}: " + ';  '.join(out))


growth(pd.read_csv(f'{R}/table1_disk_final.csv'), 'previous (all S/N) ')
growth(b, 'S/N-limited sample  ')
PY
echo
echo "FINISHED. Send back:"
echo "  cd $NEW && zip -r sn_cut_results.zip table1_it0.csv table1_disk_final.csv inj_*/injection_summary.csv inj_f444w_m/injection_per_object.csv logs"
