# data/ — derived per-bin quantities

Redshift bins 0.5-1, 1-1.5, ..., 3.5-4, 4-5, 5-6 (`z` = bin centre).
All amplitudes are medians over the galaxies of a bin.

**Normalisation is part of every column name:**
- `_pipe` — pipeline normalisation |Σ I e^{imφ}| / Σ I
- `_conv` — conventional normalisation 2|⟨I e^{imφ}⟩|/⟨I⟩ = 2 × `_pipe`
- ratios (A1/A2) and recovery fractions do not depend on it.

Do not combine `_conv` floors with `_pipe` amplitudes.

| File | Content | Produced by |
|---|---|---|
| table1_disks_F444W.csv | Table 1 with its full calculation chain per bin: measured median, noise floor, floor-corrected value, recovery fraction, final value; A1/A2 observed, bias, corrected | make_table1.py → export_data.py |
| injection_calibration_F444W.csv | noise floors (`_pipe` and `_conv`) and recovery fractions (Fig. 5) | injection_test.py |
| correction_steps_V0_V6.csv | Table 2 / Fig. 3, pipeline normalisation | run_corrected_am.py |
| restframe_test.csv | Table 3 / Fig. 6 | restframe_test.py |
| sfr_split_snr100.csv | Fig. 7 | fix_sfr.py |
| early_types_F444W.csv | early-type galaxies (floor-corrected; ratio correction approximate) | make_table1.py |
| star_astrometry_F444W.csv | Fig. 4: star centroid − catalogue position per field [pix] | star_psf_astrometry.py |
| empirical_psf.csv | FWHM of the stacked empirical PSFs | star_psf_astrometry.py |

**Uncertainties in table1_disks_F444W.csv:** `*_final_err_conv` and
`A1A2_corrected_err` are TOTAL errors, sqrt(stat_cal^2 + syst^2), as quoted in
Table 1. Their parts: `*_final_err_stat_conv` / `A1A2_corrected_err_stat`
(joint bootstrap of observed and injected galaxies) and `*_final_syst_conv` /
`A1A2_corrected_syst` (largest change among the four alternative
calibrations). Table 5 of the paper lists them; `verify_table1.py` checks the
quadrature sum.

**Robustness tables:** `median_recovery_test.csv` (median correction for a
spread of amplitudes), `injection_sample_match.csv` (injected vs observed
r_d, q). `mass_test*.csv` (stellar mass), `review7_*.csv` (self-consistent calibration,
clumps, amplitude spread, annulus radius), `filter_ratio_F277W_F444W.csv` (same
galaxies in two filters): written by `scripts/run_review7.sh`; Table 6 of the paper.

**Check (level i):** `python3 scripts/verify_table1.py data/table1_disks_F444W.csv`
recomputes A_final = 2·sqrt(A_meas² − A_floor²)/recovery and the corrected
A1/A2 from this table alone.
- Full-precision values (written by build_all.sh): agreement to a relative 1e-6.
- Values rounded to 3 decimals: each input is known only to ±0.0005, so the
  check propagates this interval and the stored value must lie inside it.
  The largest deviation is printed.
- `--submission` also requires the statistical errors, the systematics and
  N_total; it fails if any of them is missing.

**Check (level ii):** `scripts/reproduce_table1.sh` regenerates the table
from `data/intermediate/` (per-galaxy measurements and injection runs) and
compares it with the shipped table.

**Manuscript:** `scripts/make_tex_from_data.py` writes Tables 1–4 and every
number quoted in the text (`tex/numbers.tex`) from these files, and checks
that Table 2 (adopted step) and Table 3 (F444W only) agree with Table 1.

**Status (v2.8, final; disk radius in 0.04" pixels):** all files were written by `scripts/build_all.sh`
from the pipeline outputs, at full precision, with all uncertainties, the
four systematics runs and N_total. `data/intermediate/` holds the
per-galaxy measurements and injection runs (with SHA256SUMS). The checks
and their output are in `logs/` (`scripts/check_all.sh`).
