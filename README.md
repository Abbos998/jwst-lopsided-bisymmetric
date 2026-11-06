# Lopsided and bisymmetric structure in JWST galaxies at 0.5 < z < 6

Code and derived data for

> Omonov A. U., Nuritdinov S. N., Mirtadjieva K. T., *Lopsided and bisymmetric
> structure grow together in JWST galaxies at 0.5 < z < 6*, submitted to MNRAS.

The pipeline measures the lopsided (A1) and bisymmetric (A2) Fourier amplitudes of
face-on, low-Sérsic-index galaxies in NIRCam imaging from the
[Dawn JWST Archive](https://dawn-cph.github.io/dja/) (DJA), corrects the
measurement biases (pixel-grid centring, astrometric offsets, deprojection,
neighbours), and calibrates the noise floor and PSF suppression with
injection–recovery simulations that use empirical PSFs and the measured noise
autocorrelation.

## Repository layout

| Path | Content |
|---|---|
| `scripts/` | full pipeline, calibration and robustness tests, table/figure generation, checks |
| `data/` | per-bin tables (Table 1 and its calculation chain), robustness-test outputs |
| `data/intermediate/` | per-galaxy amplitudes and per-galaxy injection results (inputs of Table 1) |
| `tex/` | LaTeX tables and numbers generated from `data/` |
| `logs/` | outputs of all checks and tests |
| `tests/` | failure/tampering tests on synthetic data |

Requirements: Python ≥ 3.10 with numpy, pandas, astropy and matplotlib
(`pip install -r requirements.txt`). No SciPy is needed. `pdftotext` is optional.

## Three levels of verification

1. **Per-bin tables** – Table 1 follows from `data/table1_disks_F444W.csv` with
   Eq. (1) of the paper:
   `A_final = 2 sqrt(A_meas^2 - A_floor^2) / f_rec`, `(A1/A2)_corr = (A1/A2)_obs / b`.
2. **Per-galaxy data, no images** (about one minute):
   ```bash
   cd scripts && bash check_all.sh
   ```
   Regenerates Table 1, its uncertainties, the calculation chain and the numbers
   of the text from `data/intermediate/`, compares them with `data/` and `tex/`,
   and runs the synthetic failure tests. Results are written to `logs/`.
3. **Full pipeline from the public DJA data** – see below.

## Full pipeline

All commands run in `scripts/`. Set the location of the DJA catalogue and
cutouts and of the output directory:

```bash
export D=/path/to/dja_analysis            # DJA catalogue and cutouts
export ROOT=/path/to/output               # pipeline outputs
M=$D/full-good_morpho-phot.fits.gz
```

| Step | Script |
|---|---|
| empirical PSFs and field centring offsets from stars | `star_psf_astrometry.py` |
| cutouts in other filters | `fetch_cutouts.py` |
| amplitudes V0 → V6 (all corrections) | `run_corrected_am.py` |
| aperture S/N, noise autocorrelation | `measure_snr.py`, `measure_noise_acf.py` |
| injection–recovery calibration | `injection_test.py` (estimator in `am_wrapper.py`) |
| one filter end to end | `run_filter.sh <FILTER>` |
| Table 1, rest-frame test, SFR split | `make_table1.py`, `restframe_test.py`, `fix_sfr.py` |
| all tables, `data/`, `tex/` and figures | `run_final.sh` → `build_all.sh` |

## Robustness and selection tests

| Test (paper) | Runner |
|---|---|
| annuli, clumpy models, mass (Table 4) | `run_review7.sh`, `mass_test.py` |
| deprojection errors and PSF model (Table 6) | `run_calib_tests.sh`, `run_calib_followup.sh` |
| sampling of the injections (Table 7) | `run_seed_scatter.sh` |
| photometric redshifts, resolution/S/N matching (Table B1) | `run_selection_tests.sh` |
| bootstrap of the matched subsamples (Table B1) | `run_matched_bootstrap.sh` |
| S/N dependence of the calibration (Table B1) | `run_snr_check.sh` |

Each runner documents its inputs and optional environment variables in its header.
Default paths in the runners refer to the authors' machine and are overridden
with the environment variables `ROOT`, `DATA` and `D`.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

MIT, see `LICENSE`.
