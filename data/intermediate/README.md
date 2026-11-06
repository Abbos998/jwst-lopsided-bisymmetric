# data/intermediate/ — inputs of Table 1 (level ii)

`scripts/build_all.sh` fills this directory automatically from the pipeline
output files:

| File | Content |
|---|---|
| corrected_disk_f444w.csv | per-galaxy A1, A2, A3 for every pipeline step (V0–V6), z, flags (run_corrected_am.py) |
| inj_f444w_m/injection_summary.csv, injection_per_object.csv | main calibration: zero + matched scenarios (injection_test.py) |
| inj_f444w_{scale0.5,scale1.5,rising,falling}/… | calibration systematics |
| SHA256SUMS | checksums |

With these files, `scripts/reproduce_table1.sh` regenerates Table 1, its
uncertainties, N_total and Table 4 without the images, and compares the
result with `data/table1_disks_F444W.csv` (the difference must be 0).

**v2.5:** filled by build_all.sh on the author's computer; checksums in
SHA256SUMS. `scripts/reproduce_table1.sh` reproduces the shipped table
exactly (logs/reproduce_table1.log).
