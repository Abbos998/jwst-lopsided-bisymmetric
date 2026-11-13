#!/usr/bin/env bash
# run_sn_cut_build.sh -- step 4: rebuild tables, numbers and figures for the
# S/N-limited sample produced by run_sn_cut.sh. Everything is written to $NEW
# (default ~/Downloads/pasp_v4_sn40), including a copy of the package
# ($NEW/pasp_SUBMISSION); $ROOT and the original package are not changed.
#
#   1. per-bin step summaries (Table 2 / Fig. 1) of the S/N-limited sample
#   2. early types: aperture S/N from their cutouts, same S/N cut, floors from
#      a zero+constant injection run of the S/N-limited sample
#   3. star-forming / quiescent split (S/N >= 100) with the new calibration
#   4. copy of the package and build_all.sh with ROOT=$NEW
#   5. field closure with the new Table 1 and slope of log A vs log(1+z)
#
# Run from the package's scripts/ directory:
#   ROOT=~/Downloads/pasp_v3 bash run_sn_cut_build.sh 2>&1 | tee ../logs/run_sn_cut_build.log
# Optional env: THR (40), NEW, DATA, D (DJA directory with cutouts_ell_full and the
#   morpho-phot FITS), NPB (300), RREF (0.04)
set -euo pipefail
ROOT=${ROOT:-$HOME/Downloads/pasp_v3}
DATA=${DATA:-$ROOT/data}
D=${D:-/scratch/pulsating_disk/dja_analysis}
THR=${THR:-40}
NEW=${NEW:-$HOME/Downloads/pasp_v4_sn$THR}
NPB=${NPB:-300}
RREF=${RREF:-0.04}
M=$D/full-good_morpho-phot.fits.gz
SRC=$(cd "$(dirname "$0")/.." && pwd)
cd "$SRC/scripts"
LOG=$NEW/logs; mkdir -p "$LOG"

echo "=== 0 inputs"
need=("$NEW/table1_disk_final.csv" "$NEW/inj_f444w_m/injection_summary.csv" "$NEW/inj_f277w_m/injection_summary.csv"
      "$NEW/inj_f150w_m/injection_summary.csv" "$NEW/corrected_disk_f444w.csv" "$NEW/data/dja_snr_f444w.csv"
      "$ROOT/corrected_ell_v7.csv" "$ROOT/psf/F444W_stars.csv" "$DATA/dja_real.csv" "$M" "$D/cutouts_ell_full")
miss=0
for f in "${need[@]}"; do [ -e "$f" ] || { echo "  missing: $f"; miss=1; }; done
[ $miss -eq 0 ] || { echo "STOP: run run_sn_cut.sh first, or set ROOT / NEW / D / DATA"; exit 1; }
ln -sf "$ROOT/psf/F444W_stars.csv" "$NEW/psf/F444W_stars.csv"

echo "=== 1 step summaries of the S/N-limited sample"
python3 - "$NEW" <<'PY'
import sys
import numpy as np
import pandas as pd
N = sys.argv[1]
ZE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
ZC = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50]


def boot(x, rng, nb=1000):          # as in run_corrected_am.py
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan
    return np.median(x), np.std(np.median(rng.choice(x, (nb, len(x))), axis=1))


for f in ('f444w', 'f277w', 'f150w'):
    res = pd.read_csv(f'{N}/corrected_disk_{f}.csv')
    variants = [c[3:] for c in res.columns if c.startswith('A1_V') and '[' not in c]
    rng = np.random.default_rng(0)
    res['zb'] = pd.cut(res.z, ZE, labels=ZC)
    rows = []
    for zc in ZC:
        s = res[res.zb == zc]
        if s.empty:
            continue
        row = dict(z=zc, N=len(s), disk_r=s.disk_r.median())
        for v in variants:
            for k in ('A1', 'A2', 'A3', 'R12'):
                row[f'{k}_{v}'], row[f'{k}_{v}_err'] = boot(s[f'{k}_{v}'], rng)
        rows.append(row)
    pd.DataFrame(rows).to_csv(f'{N}/corrected_disk_{f}_summary.csv', index=False)
    print(f'  {f}: {len(rows)} bins, variants {variants}')
PY

echo "=== 2 early types (log: $LOG/ell_snr.log)"
python3 measure_snr.py --catalog "$ROOT/corrected_ell_v7.csv" --cutouts "$D/cutouts_ell_full/" \
    --pattern 'e_{:06d}.fits' --all --re_col disk_r --output "$NEW/data/ell_snr_f444w.csv" > "$LOG/ell_snr.log" 2>&1 || true   # its final printout needs disk columns
[ -s "$NEW/data/ell_snr_f444w.csv" ] || { echo "STOP: S/N of the early types failed, see $LOG/ell_snr.log"; exit 1; }
python3 - "$NEW" "$THR" "$ROOT" <<'PY'
import sys
import numpy as np
import pandas as pd
N, thr = sys.argv[1], float(sys.argv[2])
cols = list(pd.read_csv(f'{sys.argv[3]}/corrected_ell_v7.csv', nrows=0).columns)
e = pd.read_csv(f'{N}/data/ell_snr_f444w.csv')
if 'snr_ap' not in e:
    sys.exit(f'STOP: no snr_ap column in ell_snr_f444w.csv: {list(e.columns)[-8:]}')
v = e.A1_V6.notna()
k = e[np.isfinite(e.snr_ap) & (e.snr_ap >= thr)]
k[cols].to_csv(f'{N}/corrected_ell_v7.csv', index=False)
print(f'  early types: {int(v.sum())} valid of {len(e)}; with S/N: {int(e.snr_ap.notna().sum())}; '
      f'S/N >= {thr:g}: {len(k)} ({int(k.A1_V6.notna().sum())} valid); median S/N of the valid ones {e.snr_ap[v].median():.0f}')
PY
PIX=$(python3 -c "import pandas as pd; print(round(pd.read_csv('$NEW/data/dja_snr_f444w.csv').pixscale.median(), 4))" 2>/dev/null || echo 0.05)
PSC=$(python3 -c "from astropy.io import fits; print(fits.getheader('$NEW/psf/F444W_empirical_psf.fits').get('PIXSCALE', 0.05))")
rm -rf "$NEW/inj_f444w"
python3 injection_test.py --catalog "$NEW/data/dja_snr_f444w.csv" --re_col disk_r_pix --re_unit pix \
    --re_ref_scale "$RREF" --n_col n_sersic --snr_col snr_ap --pixscale "$PIX" \
    --cutout_arcsec 5.0 --n_per_bin "$NPB" --centering true,catalog --sigma_cat 0.10 \
    --psf "$NEW/psf/F444W_empirical_psf.fits" --psf_pixscale "$PSC" \
    --noise_acf "$NEW/noise_acf_f444w.npy" --am_func am_wrapper.py:measure_v6 \
    --scenarios zero,constant --output "$NEW/inj_f444w/" > "$LOG/inj_f444w.log" 2>&1 \
    || { echo "STOP: see $LOG/inj_f444w.log"; exit 1; }
python3 make_table1.py --objects "$NEW/corrected_ell_v7.csv" --injection "$NEW/inj_f444w/injection_summary.csv" \
    --exploratory --out "$NEW/table1_ell_v7" > "$LOG/table1_ell.log" 2>&1 \
    || { echo "STOP: see $LOG/table1_ell.log"; exit 1; }
tail -12 "$LOG/table1_ell.log"

echo "=== 3 star-forming / quiescent split"
python3 fix_sfr.py --morpho "$M" --objects "$NEW/corrected_disk_f444w.csv" --sample "$DATA/dja_real.csv" \
    --injection "$NEW/inj_f444w_m/injection_summary.csv" --snr_csv "$NEW/data/dja_snr_f444w.csv" \
    --snr_min 100 --out "$NEW/sfr_split_snr100" > "$LOG/sfr.log" 2>&1 \
    || { echo "STOP: see $LOG/sfr.log"; exit 1; }
tail -3 "$LOG/sfr.log"

echo "=== 4 package copy and build_all.sh (log: $LOG/build_all.log)"
PKG=$NEW/pasp_SUBMISSION
if [ ! -d "$PKG" ]; then cp -r "$SRC" "$PKG"; echo "  package copied to $PKG"; else cp "$SRC"/scripts/*.py "$SRC"/scripts/*.sh "$PKG/scripts/"; echo "  $PKG exists (scripts refreshed)"; fi
rc=0
(cd "$PKG/scripts" && ROOT=$NEW MANUSCRIPT_CHECKS=core bash build_all.sh) > "$LOG/build_all.log" 2>&1 || rc=$?
grep -E '^===|LaTeX errors|STOP|Error|error:|Traceback|DIFFER|Done' "$LOG/build_all.log" | tail -25 || true
echo "  build_all.sh exit code: $rc (0 = complete; 3 = no LaTeX; otherwise see the log -- the text of the"
echo "  manuscript still has the old numbers, so the abstract check is expected to fail at this stage)"

echo "=== 5 field closure with the new Table 1, and the slope of the growth"
T=$PKG/data/table1_disks_F444W.csv
if [ -f "$T" ]; then
    python3 field_tests.py --galaxies "$NEW/corrected_disk_f444w.csv" --table1 "$T" --sample "$DATA/dja_real.csv" \
        --snrcat "$NEW/data/dja_snr_f444w.csv" --out "$NEW/fields_sn$THR" | sed -n '/=== B/,/=== C/p' | grep -v '=== C'
else
    echo "  $T not written by build_all.sh: field test skipped"
fi
python3 - "$NEW" <<'PY'
import sys
import numpy as np
import pandas as pd
t = pd.read_csv(f'{sys.argv[1]}/table1_disk_final.csv')
x = np.log10(1 + t.z.values)
for k in (1, 2):
    for lab, e in (('total errors', f'A{k}_final_err'), ('statistical + calibration only', f'A{k}_final_err_stat')):
        if e not in t:
            continue
        a, s = t[f'A{k}_final_std'].values, t[e].values
        y, w = np.log10(a), (a * np.log(10) / s) ** 2
        X = np.vstack([np.ones_like(x), x]).T
        C = np.linalg.inv(X.T @ (w[:, None] * X))
        b = C @ X.T @ (w * y)
        print(f'  A{k}: d log A / d log(1+z) = {b[1]:.3f} +- {np.sqrt(C[1, 1]):.3f} ({lab}); '
              f'chi2 = {np.sum(w * (y - X @ b) ** 2):.1f} for {len(x) - 2} d.o.f.')
PY
echo
echo "FINISHED. Send back:"
echo "  cd $NEW && zip -r sn_cut_build.zip logs table1_*.csv sfr_split_snr100.csv restframe_disk_m.csv fields_sn${THR}_summary.txt pasp_SUBMISSION/data pasp_SUBMISSION/tex -x '*.png'"
