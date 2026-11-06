#!/usr/bin/env bash
# test_gates.sh — check that the build and the reproduction stop on every missing
# or inconsistent input, and pass on a complete one. Runs on synthetic data in a
# temporary copy of the package (the package itself is not modified).
#
# Run from the package root or anywhere:   bash tests/test_gates.sh
# Runtime ~1 min (each case prints its elapsed time). Needs bash, python3 with numpy and
# pandas, and sha256sum or shasum. Only synthetic data are used: the shipped results and
# the manuscript are never touched.
set -uo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
export NBOOT=100 MANUSCRIPT_CHECKS=core   # synthetic data: no robustness tests              # fewer bootstrap draws: the checks, not the values, are tested
START=$(date +%s)
python3 -c "import numpy, pandas" 2>/dev/null || { echo "python3 with numpy and pandas required"; exit 2; }
command -v sha256sum > /dev/null || command -v shasum > /dev/null || { echo "sha256sum or shasum required"; exit 2; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pass=0; fail=0
ok()  { echo "  PASS  $1"; pass=$((pass + 1)); }
bad() { echo "  FAIL  $1"; fail=$((fail + 1)); }
expect() {  # expect <0|1> <name> <command...>: 0 = must succeed, 1 = must fail
    local want=$1 name=$2; shift 2
    if "$@" > "$T/log" 2>&1; then got=0; else got=1; fi
    local el=$(( $(date +%s) - START ))
    if [ "$got" = "$want" ]; then ok "$name  [${el}s]"; else bad "$name  [${el}s]"; tail -5 "$T/log" | sed 's/^/        /'; fi
}
fresh() {   # fresh package copy + synthetic outputs
    rm -rf "$T/pkg" "$T/root"; mkdir -p "$T/root"
    cp -r "$SRC" "$T/pkg"
    # results of the real data that the synthetic build does not regenerate
    rm -f "$T/pkg"/data/review7_*.csv "$T/pkg"/data/mass_test*.csv "$T/pkg"/data/filter_ratio_*.csv
    python3 "$SRC/tests/make_synthetic_root.py" "$T/root" > /dev/null
}
build() { (cd "$T/pkg/scripts" && ROOT="$T/root" SKIP_FIGURES_FOR_TESTS=1 bash build_all.sh); }
repro() { (cd "$T/pkg/scripts" && bash reproduce_table1.sh); }
S="$T/pkg/scripts"; D="$T/pkg/data"

echo "== complete inputs"
fresh
expect 0 "build_all.sh passes with all inputs" build
expect 0 "reproduce_table1.sh reproduces Table 1, errors and Table 4" repro
expect 0 "verify_table1.py --submission passes" python3 "$S/verify_table1.py" "$D/table1_disks_F444W.csv" --submission
expect 0 "Table 3 F444W row equals Table 1 (same calibration)" \
    python3 "$S/make_tex_from_data.py" --data "$D" --tex "$T/pkg/tex" --submission --core

echo "== missing inputs stop the build"
for lab in scale0.5 scale1.5 rising falling; do
    fresh; rm "$T/root/inj_f444w_$lab/injection_summary.csv"
    expect 1 "build stops without systematics run $lab" build
done
fresh; rm "$T/root/inj_f444w_m/injection_per_object.csv"
expect 1 "build stops without injection_per_object.csv" build
fresh; rm "$T/root/corrected_disk_f277w.csv"
expect 1 "build stops without the F277W measurements (rest-frame test)" build

echo "== make_table1.py alone"
fresh
expect 1 "make_table1.py refuses to run without --syst" python3 "$S/make_table1.py" \
    --objects "$T/root/corrected_disk_f444w.csv" --injection "$T/root/inj_f444w_m/injection_summary.csv" --out "$T/x"
expect 1 "make_table1.py refuses with 3 of 4 systematics runs" python3 "$S/make_table1.py" \
    --objects "$T/root/corrected_disk_f444w.csv" --injection "$T/root/inj_f444w_m/injection_summary.csv" \
    --syst "scale0.5=$T/root/inj_f444w_scale0.5/injection_summary.csv,scale1.5=$T/root/inj_f444w_scale1.5/injection_summary.csv,rising=$T/root/inj_f444w_rising/injection_summary.csv" --out "$T/x"

echo "== tampering after a complete build is detected"
fresh; build > /dev/null 2>&1
for lab in scale0.5 scale1.5 rising falling; do
    cp -r "$T/pkg" "$T/pkg0"
    rm "$D/intermediate/inj_f444w_$lab/injection_summary.csv"
    expect 1 "reproduce fails without intermediate run $lab" repro
    rm -rf "$T/pkg"; mv "$T/pkg0" "$T/pkg"
done
cp -r "$T/pkg" "$T/pkg0"
sed -i '2s/,0\./,1./' "$D/intermediate/corrected_disk_f444w.csv"
expect 1 "reproduce fails if a per-galaxy value is changed (checksum)" repro
rm -rf "$T/pkg"; cp -r "$T/pkg0" "$T/pkg"
python3 - "$D/table1_disks_F444W.csv" <<'PY'
import sys, pandas as pd
t = pd.read_csv(sys.argv[1]); t.loc[0, 'A1_final_conv'] += 0.002; t.to_csv(sys.argv[1], index=False)
PY
expect 1 "verify fails if a Table 1 value is changed by 0.002" python3 "$S/verify_table1.py" "$D/table1_disks_F444W.csv"
expect 1 "reproduce fails if a Table 1 value is changed by 0.002" repro
rm -rf "$T/pkg"; cp -r "$T/pkg0" "$T/pkg"
python3 - "$D/table1_disks_F444W.csv" <<'PY'
import sys, pandas as pd
t = pd.read_csv(sys.argv[1]); t.loc[0, 'A1A2_corrected_err'] = float('nan'); t.to_csv(sys.argv[1], index=False)
PY
expect 1 "verify --submission fails if one error is missing" python3 "$S/verify_table1.py" "$D/table1_disks_F444W.csv" --submission
rm -rf "$T/pkg"; cp -r "$T/pkg0" "$T/pkg"
python3 - "$D/table1_disks_F444W.csv" <<'PY'
import sys, pandas as pd
t = pd.read_csv(sys.argv[1]); t.loc[0, 'A2_final_syst_conv'] *= 2; t.to_csv(sys.argv[1], index=False)
PY
expect 1 "verify fails if total error != sqrt(stat^2 + syst^2)" python3 "$S/verify_table1.py" "$D/table1_disks_F444W.csv"
rm -rf "$T/pkg"; cp -r "$T/pkg0" "$T/pkg"
python3 - "$D/table1_disks_F444W.csv" <<'PY'
import sys, pandas as pd
t = pd.read_csv(sys.argv[1]); t['syst_runs'] = 'rising;scale0.5;scale1.5'; t.to_csv(sys.argv[1], index=False)
PY
expect 1 "verify --submission fails if a systematics run is not recorded" python3 "$S/verify_table1.py" "$D/table1_disks_F444W.csv" --submission
rm -rf "$T/pkg"; cp -r "$T/pkg0" "$T/pkg"
python3 - "$D/restframe_test.csv" <<'PY'
import sys, pandas as pd
t = pd.read_csv(sys.argv[1]); i = t.index[(t.set == 'F444W only') & (t.z == 0.75)][0]
t.loc[i, 'A1_conv'] += 0.01; t.to_csv(sys.argv[1], index=False)
PY
expect 1 "manuscript check fails if Table 3 F444W row differs from Table 1" \
    python3 "$S/make_tex_from_data.py" --data "$D" --tex "$T/pkg/tex" --submission --core
rm -rf "$T/pkg"; cp -r "$T/pkg0" "$T/pkg"
sed -i '6s/[0-9]\.[0-9][0-9][0-9]/9.999/' "$T/pkg/tex/table_chain.tex"
expect 1 "reproduce fails if Appendix A table is edited by hand" repro
rm -rf "$T/pkg"; cp -r "$T/pkg0" "$T/pkg"
expect 0 "untouched build still reproduces" repro

echo; echo "$pass passed, $fail failed  (total $(( $(date +%s) - START )) s)"
[ "$fail" -eq 0 ]
