#!/usr/bin/env bash
# reproduce_table1.sh — independent check from the package alone (no images, no
# original working directory). From data/intermediate/ it regenerates
#   Table 1 (values, statistical + calibration errors, all four systematics, N_total)
#   Table 4 (Appendix A, calculation chain)
# and compares them with the shipped data/table1_disks_F444W.csv, tex/table1.tex
# and tex/table_chain.tex. Any difference, missing input or failed check -> exit 1.
#
#   data/intermediate/corrected_disk_f444w.csv      per-galaxy A_m (run_corrected_am.py)
#   data/intermediate/inj_f444w_*/injection_*.csv   injection runs (injection_test.py)
#
# Run from scripts/:  bash reproduce_table1.sh
set -euo pipefail
PKG=$(cd "$(dirname "$0")/.." && pwd); I="$PKG/data/intermediate"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cd "$PKG/scripts"
SYST_LABELS="scale0.5 scale1.5 rising falling"
NBOOT=${NBOOT:-1000}          # must equal the value used by build_all.sh
sha256() { if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

need=(corrected_disk_f444w.csv SHA256SUMS)
for d in inj_f444w_m $(printf 'inj_f444w_%s ' $SYST_LABELS); do
    need+=("$d/injection_summary.csv" "$d/injection_per_object.csv")
done
for f in "${need[@]}"; do
    [ -f "$I/$f" ] || { echo "FAILED: data/intermediate/$f missing (run build_all.sh first)"; exit 1; }
done
(cd "$I" && sha256 -c --status SHA256SUMS) || { echo "FAILED: checksums of data/intermediate/"; exit 1; }
echo "1/4 checksums OK"

SYST=""
for lab in $SYST_LABELS; do SYST="$SYST,$lab=$I/inj_f444w_$lab/injection_summary.csv"; done
python3 make_table1.py --objects "$I/corrected_disk_f444w.csv" \
    --injection "$I/inj_f444w_m/injection_summary.csv" --syst "${SYST#,}" \
    --nboot "$NBOOT" --out "$W/table1_disk_final" > /dev/null
echo "2/4 Table 1 regenerated (systematics: $SYST_LABELS)"

# rebuild data/ in a scratch copy; the other tables are taken from the package
mkdir -p "$W/root" "$W/data" "$W/tex"
cp "$W/table1_disk_final.csv" "$W/root/"
python3 export_data.py --root "$W/root" --out "$W/data" > /dev/null
for f in "$PKG"/data/*.csv; do
    [ "$(basename "$f")" = table1_disks_F444W.csv ] || cp "$f" "$W/data/"
done
python3 verify_table1.py "$W/data/table1_disks_F444W.csv" --submission | tail -1
python3 make_tex_from_data.py --data "$W/data" --tex "$W/tex" --submission --core > /dev/null
echo "3/4 Table 1 and Table 4 rebuilt, checks passed"

python3 - "$W/data/table1_disks_F444W.csv" "$PKG/data/table1_disks_F444W.csv" <<'PY'
import sys, numpy as np, pandas as pd
a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
if list(a.columns) != list(b.columns) or len(a) != len(b):
    sys.exit("FAILED: table structure differs from the shipped table")
num = [c for c in b.columns if pd.api.types.is_numeric_dtype(b[c])]
d = max(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))) for c in num)
nan_mismatch = [c for c in num if not a[c].isna().equals(b[c].isna())]
txt = [c for c in b.columns if c not in num and not a[c].astype(str).equals(b[c].astype(str))]
print(f"4/4 data/table1_disks_F444W.csv: max difference {d:.2e} over {len(num)} columns")
if d > 1e-9 or nan_mismatch or txt:
    sys.exit(f"FAILED: differs from the shipped table {nan_mismatch + txt}")
PY
for f in table1.tex table_chain.tex table_errors.tex; do
    cmp -s "$W/tex/$f" "$PKG/tex/$f" || { echo "FAILED: tex/$f differs from the regenerated one"; exit 1; }
    echo "    tex/$f identical"
done
echo "SUBMISSION CHECK: passed (Table 1, its errors and Table 4 reproduced from the package)"
