#!/usr/bin/env python3
"""
injection_test.py — A_m noise/resolution/centering floor for the A1/A2 paper.

Question this answers:
    Could the observed rise of A1 (and A1/A2) with redshift be produced by the
    measurement itself (PSF, pixel sampling, surface-brightness dimming / S/N,
    centering), rather than by real lopsidedness?

Method:
    For each redshift bin we draw N galaxies from the *real* face-on disk
    catalog (their own R_e, Sersic n, q, PA, S/N), build a model disk with a
    KNOWN azimuthal structure in the disk plane, project it with q/PA, convolve
    with the F444W PSF, add noise to the galaxy's S/N, and re-measure A_m with
    the same 0.5–1.5 R_e ring method as the paper (Section 3.1).

    Two injection scenarios are run for every galaxy:
      zero      : A1_in = 0,          A2_in = 0     -> pure measurement floor
      constant  : A1_in = A1_LOW,     A2_in = A2_LOW (the z~0.75 values, Table 1)
                  -> if intrinsic structure did NOT evolve, would we still
                     measure A1/A2 rising from 0.74 to ~1.1?

    And several centering schemes:
      true      : exact model centre (ideal)
      catalog   : true centre + Gaussian scatter sigma_cat (default 0.1 pix)
      iterative : flux-weighted iterative centroid on the noisy image
      peak      : brightest pixel + 3x3 parabolic sub-pixel refinement

Usage (example):
    python3 injection_test.py \
        --catalog /tmp/dja_real.csv \
        --psf F444W_psf.fits --psf_pixscale 0.04 \
        --pixscale 0.04 --n_per_bin 300 --output ../injection_out/

    # Use YOUR OWN A_m function instead of the built-in one:
    python3 injection_test.py ... --am_func fourier_pipeline.py:measure_am
    #   expected signature: f(image, x0, y0, q, pa_deg, re_pix) -> (A1, A2, A3)
    #   or a dict with keys 'A1', 'A2', 'A3'.

Interpretation:
    * null-scenario A1 floor flat with z, and << observed A1 -> rise is real.
    * null floor rising with z and comparable to observed A1  -> rise is
      (partly) a measurement artefact; correct with
      A1_corr = sqrt(A1_obs^2 - A1_floor^2).
    * constant-scenario A1/A2 rising toward ~1.1 at z>3        -> the headline
      plateau can be reproduced with NO intrinsic evolution.
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# No SciPy on purpose: user environments often have a SciPy built for a
# different NumPy major version, which fails at import.


def fftconvolve(a, k, mode='same'):
    """numpy-only equivalent of scipy.signal.fftconvolve(a, k, mode='same')."""
    s = (a.shape[0] + k.shape[0] - 1, a.shape[1] + k.shape[1] - 1)
    full = np.fft.irfft2(np.fft.rfft2(a, s) * np.fft.rfft2(k, s), s)
    y0, x0 = (k.shape[0] - 1) // 2, (k.shape[1] - 1) // 2
    return full[y0:y0 + a.shape[0], x0:x0 + a.shape[1]]


def sersic_bn(n):
    """b_n with Gamma(2n, b_n) = Gamma(2n)/2: Ciotti & Bertin (1999) for
    n > 0.36, MacArthur et al. (2003) polynomial below."""
    if n > 0.36:
        return (2 * n - 1 / 3 + 4 / (405 * n) + 46 / (25515 * n ** 2)
                + 131 / (1148175 * n ** 3) - 2194697 / (30690717750 * n ** 4))
    return 0.01945 - 0.8902 * n + 10.95 * n ** 2 - 19.67 * n ** 3 + 13.43 * n ** 4


def resample(img, factor):
    """Separable linear resampling by `factor` (numpy only)."""
    ny, nx = img.shape
    my, mx = int(round(ny * factor)), int(round(nx * factor))
    yo = (np.arange(my) + 0.5) / factor - 0.5
    xo = (np.arange(mx) + 0.5) / factor - 0.5
    tmp = np.array([np.interp(xo, np.arange(nx), row) for row in img])
    return np.array([np.interp(yo, np.arange(ny), col) for col in tmp.T]).T

# Paper binning and Table 1 values (face-on disks), used for the overlay.
Z_EDGES = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
Z_CEN = np.array([0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50])
OBS_A1 = np.array([0.078, 0.090, 0.103, 0.101, 0.109, 0.121, 0.128, 0.152, 0.126])
OBS_A2 = np.array([0.117, 0.113, 0.116, 0.115, 0.116, 0.114, 0.134, 0.127, 0.114])
OBS_A3 = np.array([0.029, 0.035, 0.041, 0.040, 0.053, 0.061, 0.071, 0.080, 0.069])
OBS_R12 = np.array([0.74, 0.87, 0.98, 1.02, 1.03, 1.10, 1.11, 1.12, 1.09])
A1_LOW, A2_LOW, A3_LOW = OBS_A1[0], OBS_A2[0], OBS_A3[0]

COL_CANDIDATES = {
    'z':   ['z', 'z_best', 'zbest', 'z_phot', 'redshift', 'z_spec_or_phot'],
    're':  ['re', 're_pix', 're_arcsec', 'r_e', 'reff', 'r_eff', 'sersic_re',
            're_f444w', 'r50', 'flux_radius'],
    'n':   ['n', 'sersic_n', 'n_sersic', 'n_f444w'],
    'q':   ['q', 'axis_ratio', 'ba', 'b_a', 'q_f444w'],
    'pa':  ['pa', 'pa_deg', 'theta', 'theta_deg', 'pa_f444w'],
    'snr': ['snr', 'snr_f444w', 'sn_f444w', 'sn', 'snr444'],
}


# ----------------------------------------------------------------------------
# A_m measurement (paper Section 3.1)
# ----------------------------------------------------------------------------
def deproject(nx, ny, x0, y0, q, pa_deg, oversample=1):
    """Disk-plane radius and azimuth for every pixel (or sub-pixel).

    PA is measured from the +x axis, counter-clockwise, in degrees. The same
    convention is used to build the models, so the test is self-consistent;
    only --am_func users need to make sure their own convention matches.
    """
    s = 1.0 / oversample
    xs = (np.arange(nx * oversample) + 0.5) * s - 0.5
    ys = (np.arange(ny * oversample) + 0.5) * s - 0.5
    X, Y = np.meshgrid(xs, ys)
    dx, dy = X - x0, Y - y0
    t = np.deg2rad(pa_deg)
    xp = dx * np.cos(t) + dy * np.sin(t)
    yp = (-dx * np.sin(t) + dy * np.cos(t)) / max(q, 1e-3)
    return np.hypot(xp, yp), np.arctan2(yp, xp)


def measure_am_builtin(img, x0, y0, q, pa_deg, re_pix, rmin=0.5, rmax=1.5,
                       ring_width=1.0, oversample=5):
    """Global A1, A2, A3 over rmin*Re < r < rmax*Re.

    Per ring: a0 = <I>, a_m = 2<I cos(m phi)>, b_m = 2<I sin(m phi)> (unweighted
    average over the ring pixels), A_m(r) = sqrt(a_m^2 + b_m^2) / a0.
    Global value: flux-weighted mean of A_m(r) over the rings.

    Each pixel is split into oversample x oversample sub-pixels before the
    polar binning. With oversample=1 (whole pixels) a perfectly symmetric,
    noiseless disk gives A1 ~ 0.20 at Re = 2 pix and ~0.12 at Re = 5 pix
    (pixel-grid aliasing); oversample >= 5 brings this below ~0.03.
    """
    ny, nx = img.shape
    r, phi = deproject(nx, ny, x0, y0, q, pa_deg, oversample)
    img = np.repeat(np.repeat(img, oversample, axis=0), oversample, axis=1)
    r_in, r_out = rmin * re_pix, rmax * re_pix
    nring = max(1, int(np.ceil((r_out - r_in) / ring_width)))
    edges = np.linspace(r_in, r_out, nring + 1)
    amps, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (r >= lo) & (r < hi) & np.isfinite(img)
        if sel.sum() < 6:
            continue
        I, p = img[sel], phi[sel]
        a0 = I.mean()
        if a0 <= 0:
            continue
        am = [np.hypot(2 * np.mean(I * np.cos(m * p)),
                       2 * np.mean(I * np.sin(m * p))) / a0 for m in (1, 2, 3)]
        amps.append(am)
        weights.append(I.sum())
    if not amps:
        return np.nan, np.nan, np.nan
    amps, weights = np.array(amps), np.array(weights)
    return tuple(np.sum(amps * weights[:, None], axis=0) / weights.sum())


def load_user_am(spec):
    path, func = spec.rsplit(':', 1)
    mod_spec = importlib.util.spec_from_file_location('user_am', path)
    mod = importlib.util.module_from_spec(mod_spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    mod_spec.loader.exec_module(mod)
    f = getattr(mod, func)

    def wrapped(img, x0, y0, q, pa_deg, re_pix):
        out = f(img, x0, y0, q, pa_deg, re_pix)
        if isinstance(out, dict):
            return out['A1'], out['A2'], out['A3']
        return tuple(out[:3])
    return wrapped


# ----------------------------------------------------------------------------
# Model galaxies
# ----------------------------------------------------------------------------
def radial_weight(r, re_pix, profile):
    """Radial shape of the perturbation amplitude, normalised to 1 at r = R_e:
    constant, rising (prop. to r) or falling (prop. to 1/r, capped at 2)."""
    if profile == 'rising':
        return r / re_pix
    if profile == 'falling':
        return np.minimum(re_pix / np.maximum(r, 1e-3), 2.0)
    return 1.0


def make_model(npix, x0, y0, re_pix, n, q, pa_deg, eps, phases, oversample=5,
               profile='constant', clumps=None):
    """Sersic disk with I(r)(1 + sum eps_m w(r) cos(m(phi - phi_m))) in the
    disk plane, projected with q/PA, integrated over pixels (oversampled).
    clumps: optional (frac, list of (r/re, phi, sigma/re)) -- Gaussian clumps in
    the disk plane carrying a fraction frac of the total flux."""
    r, phi = deproject(npix, npix, x0, y0, q, pa_deg, oversample)
    bn = sersic_bn(n)
    I = np.exp(-bn * ((r / re_pix) ** (1.0 / n) - 1.0))
    pert = np.ones_like(I)
    w = radial_weight(r, re_pix, profile)
    for m, (e, ph) in enumerate(zip(eps, phases), start=1):
        if e:
            pert += e * w * np.cos(m * (phi - ph))
    I *= np.clip(pert, 0, None)
    if clumps:
        frac, cl = clumps
        I /= I.sum()
        xd, yd = r * np.cos(phi), r * np.sin(phi)
        C = np.zeros_like(I)
        for rc, pc, sc in cl:
            s_ = max(sc * re_pix, 0.5)
            C += np.exp(-((xd - rc * re_pix * np.cos(pc)) ** 2 +
                          (yd - rc * re_pix * np.sin(pc)) ** 2) / (2 * s_ * s_))
        if C.sum() > 0:
            I = (1 - frac) * I + frac * C / C.sum()
    I = I.reshape(npix, oversample, npix, oversample).mean(axis=(1, 3))
    return I / I.sum()


def gaussian_psf(fwhm_pix, size=None):
    sig = fwhm_pix / 2.3548
    size = size or int(2 * np.ceil(4 * sig) + 1)
    c = (size - 1) / 2
    y, x = np.mgrid[:size, :size]
    k = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sig ** 2))
    return k / k.sum()


def load_psf(path, psf_pixscale, pixscale):
    from astropy.io import fits
    with fits.open(path) as h:
        hdu = next(x for x in h if x.data is not None and x.data.ndim == 2)
        psf = np.asarray(hdu.data, float)
    if psf_pixscale and abs(psf_pixscale - pixscale) > 1e-6:
        psf = resample(psf, psf_pixscale / pixscale)
    psf = np.clip(psf, 0, None)
    # odd size, centred on the peak
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    h_ = min(cy, cx, psf.shape[0] - 1 - cy, psf.shape[1] - 1 - cx, 40)
    psf = psf[cy - h_:cy + h_ + 1, cx - h_:cx + h_ + 1]
    return psf / psf.sum()


def correlated_noise(shape, acf, rng):
    """Unit-variance Gaussian noise with the given autocorrelation function
    (odd-sized, centred, ACF(0) = 1), generated in Fourier space."""
    ny, nx = shape
    h = acf.shape[0] // 2
    k = np.zeros(shape)
    k[:2 * h + 1, :2 * h + 1] = acf
    k = np.roll(np.roll(k, -h, 0), -h, 1)
    p = np.clip(np.real(np.fft.fft2(k)), 0, None)
    n = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=shape)) * np.sqrt(p)))
    return n / np.sqrt(p.mean())


def add_noise(model, snr, aper_r_pix, x0, y0, rng, acf=None):
    """Background-limited Gaussian noise such that flux / (sigma_pix sqrt(N))
    in a circular aperture equals the catalog S/N (the definition used by
    measure_snr.py). With `acf` the noise has the measured pixel-to-pixel
    correlation and the same per-pixel rms."""
    ny, nx = model.shape
    y, x = np.mgrid[:ny, :nx]
    ap = (x - x0) ** 2 + (y - y0) ** 2 <= aper_r_pix ** 2
    sigma = model[ap].sum() / (snr * np.sqrt(ap.sum()))
    noise = rng.normal(size=model.shape) if acf is None else correlated_noise(model.shape, acf, rng)
    return model + sigma * noise, sigma


# ----------------------------------------------------------------------------
# Centering schemes
# ----------------------------------------------------------------------------
def centre_iterative(img, x0, y0, r_ap, tol=0.05, maxit=20):
    ny, nx = img.shape
    y, x = np.mgrid[:ny, :nx]
    for _ in range(maxit):
        ap = (x - x0) ** 2 + (y - y0) ** 2 <= r_ap ** 2
        w = np.clip(img[ap], 0, None)
        if w.sum() <= 0:
            break
        nx0, ny0 = (w * x[ap]).sum() / w.sum(), (w * y[ap]).sum() / w.sum()
        done = np.hypot(nx0 - x0, ny0 - y0) < tol
        x0, y0 = nx0, ny0
        if done:
            break
    return x0, y0


def _c1(img, x0, y0, re_pix, oversample=5):
    """Complex m=1 coefficient <I e^{i phi}>/<I> in 0.5–1.5 R_e (circular)."""
    ny, nx = img.shape
    s = 1.0 / oversample
    X, Y = np.meshgrid((np.arange(nx * oversample) + 0.5) * s - 0.5 - x0,
                       (np.arange(ny * oversample) + 0.5) * s - 0.5 - y0)
    r = np.hypot(X, Y)
    sel = (r >= 0.5 * re_pix) & (r < 1.5 * re_pix)
    I = np.repeat(np.repeat(img, oversample, 0), oversample, 1)[sel]
    a0 = I.mean()
    if not np.isfinite(a0) or a0 <= 0:
        return np.nan
    return np.mean(I * (X[sel] + 1j * Y[sel]) / r[sel]) / a0


def centre_symmetric(img, x0, y0, re_pix, h=0.1, maxit=8):
    """Position where the ring m=1 coefficient vanishes (same algorithm as
    measure_centering.py). Falls back to the start position on failure."""
    x, y = x0, y0
    for _ in range(maxit):
        c, cx, cy = (_c1(img, x, y, re_pix), _c1(img, x + h, y, re_pix),
                     _c1(img, x, y + h, re_pix))
        if not np.all(np.isfinite([c, cx, cy])):
            return x0, y0
        J = np.array([[(cx - c).real, (cy - c).real],
                      [(cx - c).imag, (cy - c).imag]]) / h
        try:
            step = -np.linalg.solve(J, [c.real, c.imag])
        except np.linalg.LinAlgError:
            return x0, y0
        n = np.hypot(*step)
        if n > 1.0:
            step /= n
        x, y = x + step[0], y + step[1]
        if np.hypot(x - x0, y - y0) > 3.0:
            return x0, y0
        if n < 0.005:
            break
    return x, y


def centre_peak(img, x0, y0, r_search):
    ny, nx = img.shape
    y, x = np.mgrid[:ny, :nx]
    m = np.where((x - x0) ** 2 + (y - y0) ** 2 <= r_search ** 2, img, -np.inf)
    py, px = np.unravel_index(np.argmax(m), m.shape)
    fx, fy = float(px), float(py)
    if 1 <= px < nx - 1 and 1 <= py < ny - 1:
        def sub(a, b, c):
            d = a - 2 * b + c
            return 0.5 * (a - c) / d if d != 0 else 0.0
        fx += np.clip(sub(img[py, px - 1], img[py, px], img[py, px + 1]), -0.5, 0.5)
        fy += np.clip(sub(img[py - 1, px], img[py, px], img[py + 1, px]), -0.5, 0.5)
    return fx, fy


# ----------------------------------------------------------------------------
# Catalog handling
# ----------------------------------------------------------------------------
def resolve_columns(df, overrides):
    lower = {c.lower(): c for c in df.columns}
    cols = {}
    for key, cands in COL_CANDIDATES.items():
        if overrides.get(key):
            if overrides[key] not in df.columns:
                sys.exit(f"--{key}_col '{overrides[key]}' not in catalog")
            cols[key] = overrides[key]
            continue
        for c in cands:
            if c.lower() in lower:
                cols[key] = lower[c.lower()]
                break
    return cols


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalog', required=True, help='face-on disk catalog (CSV or FITS)')
    ap.add_argument('--output', required=True)
    ap.add_argument('--n_per_bin', type=int, default=300)
    ap.add_argument('--pixscale', type=float, default=0.04, help='arcsec/pix of the cutouts')
    ap.add_argument('--cutout_arcsec', type=float, default=2.0)
    ap.add_argument('--psf', help='F444W PSF FITS (recommended; else Gaussian)')
    ap.add_argument('--psf_pixscale', type=float, help='arcsec/pix of the PSF file')
    ap.add_argument('--psf_fwhm', type=float, default=0.145, help='Gaussian fallback FWHM [arcsec]')
    ap.add_argument('--matched_table', help="make_table1.py CSV: adds a 'matched' scenario "
                    'injecting the observed corrected amplitudes (A?_corr_std) of each z bin')
    ap.add_argument('--matched_scale', type=float, default=1.0,
                    help='multiply the matched amplitudes (systematics test, e.g. 0.5 or 1.5)')
    ap.add_argument('--pert_profile', choices=['constant', 'rising', 'falling'],
                    default='constant', help='radial shape of the injected perturbations')
    ap.add_argument('--matched_col', default='corr_std',
                    help="column A?_<matched_col> of --matched_table to inject (corr_std: floor-"
                         "corrected; final_std: final Table 1 values, for a self-consistent "
                         "calibration)")
    ap.add_argument('--matched_spread', type=float, default=0.0,
                    help='log-normal scatter [dex] of the matched amplitudes from galaxy to '
                         'galaxy (independent for each m); 0 = same amplitude for all')
    ap.add_argument('--clumps', type=int, default=0,
                    help='number of Gaussian clumps added to every non-zero-scenario model')
    ap.add_argument('--clump_frac', type=float, default=0.2,
                    help='fraction of the total flux in the clumps')
    ap.add_argument('--clump_sigma', type=float, default=0.15,
                    help='clump Gaussian sigma in units of r_d (min 0.5 pix)')
    ap.add_argument('--ring_scale', type=float, default=1.0,
                    help='scale of the measurement annulus relative to r_d (both the true and '
                         'the observed measurement; as run_corrected_am.py --r_scale)')
    ap.add_argument('--scenarios', default='zero,constant',
                    help="comma list of scenarios to run: zero, constant, matched")
    ap.add_argument('--noise_acf', help='.npy noise autocorrelation from measure_noise_acf.py '
                    '(correlated noise; default: white noise)')
    ap.add_argument('--re_unit', choices=['arcsec', 'pix'], default='arcsec')
    ap.add_argument('--re_ref_scale', type=float,
                    help="with --re_unit pix: arcsec/pix in which R_e is expressed; converted to "
                         "--pixscale (e.g. 0.05 when simulating a filter with other pixels)")
    ap.add_argument('--snr_aper_arcsec', type=float, default=0.25,
                    help='radius of the aperture the catalog S/N refers to')
    ap.add_argument('--snr_default', type=float, default=None,
                    help='use this S/N when the catalog has no S/N column')
    ap.add_argument('--sigma_cat', type=float, default=0.1, help='catalog centroid scatter [pix]')
    ap.add_argument('--centering', default='true,catalog,iterative,peak',
                    help='comma list of: true, catalog, iterative, peak, grid, symmetric')
    ap.add_argument('--grid_offset', type=float, default=0.0,
                    help="constant offset added to the 'grid' centre (e.g. -0.5 for a "
                         "(nx-1)/2 vs nx/2 convention mismatch)")
    ap.add_argument('--no_cuts', action='store_true', help='skip q>0.7, 0.3<n<2.5 cuts')
    ap.add_argument('--am_func', help='path/to/file.py:function to replace built-in A_m')
    ap.add_argument('--am_oversample', type=int, default=5,
                    help='sub-pixel sampling of the built-in estimator (1 = naive whole pixels)')
    ap.add_argument('--seed', type=int, default=42)
    for k in COL_CANDIDATES:
        ap.add_argument(f'--{k}_col')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    noise_acf = np.load(args.noise_acf) if args.noise_acf else None
    if noise_acf is not None:
        print(f"Correlated noise: ACF {noise_acf.shape}, sum = {noise_acf.sum():.2f}")
    if args.am_func:
        measure = load_user_am(args.am_func)
    else:
        def measure(img, x0, y0, q, pa, re):
            return measure_am_builtin(img, x0, y0, q, pa, re, oversample=args.am_oversample)
    schemes = [s.strip() for s in args.centering.split(',') if s.strip()]

    # ---- catalog
    if args.catalog.endswith(('.fits', '.fits.gz', '.fit')):
        from astropy.table import Table
        df = Table.read(args.catalog).to_pandas()
    else:
        df = pd.read_csv(args.catalog)
    cols = resolve_columns(df, {k: getattr(args, f'{k}_col') for k in COL_CANDIDATES})
    missing = [k for k in ('z', 're', 'n', 'q') if k not in cols]
    if missing:
        print(f"Catalog columns: {list(df.columns)}")
        sys.exit(f"Cannot find columns for {missing}; pass --<name>_col explicitly.")
    if 'snr' not in cols and args.snr_default is None:
        print(f"Catalog columns: {list(df.columns)}")
        sys.exit("No S/N column found; pass --snr_col or --snr_default.")
    print("Using columns:", cols)

    d = pd.DataFrame({k: pd.to_numeric(df[c], errors='coerce') for k, c in cols.items()})
    if 'pa' not in d:
        d['pa'] = rng.uniform(0, 180, len(d))
    if 'snr' not in d:
        d['snr'] = args.snr_default
    if args.re_unit == 'arcsec':
        d['re_pix'] = d['re'] / args.pixscale
    else:
        d['re_pix'] = d['re'] * (args.re_ref_scale / args.pixscale if args.re_ref_scale else 1.0)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=['z', 're_pix', 'n', 'q', 'snr'])
    if not args.no_cuts:
        d = d[(d.q > 0.7) & (d.n > 0.3) & (d.n < 2.5) & (d.snr > 10)]
    d = d[(d.z >= Z_EDGES[0]) & (d.z < Z_EDGES[-1]) & (d.re_pix > 0)]
    print(f"Galaxies after cuts: {len(d)}")
    print("Median R_e [pix] per bin:",
          [round(float(np.median(d.re_pix[(d.z >= lo) & (d.z < hi)])), 2)
           if ((d.z >= lo) & (d.z < hi)).any() else None
           for lo, hi in zip(Z_EDGES[:-1], Z_EDGES[1:])])

    # ---- PSF
    if args.psf:
        psf = load_psf(args.psf, args.psf_pixscale, args.pixscale)
        print(f"PSF loaded: {psf.shape}")
    else:
        psf = gaussian_psf(args.psf_fwhm / args.pixscale)
        print(f"WARNING: Gaussian PSF FWHM={args.psf_fwhm}\" (no wings). "
              "Use --psf with a real F444W PSF for the paper.")

    npix = int(round(args.cutout_arcsec / args.pixscale))
    aper = args.snr_aper_arcsec / args.pixscale
    all_scen = {'zero': (0.0, 0.0, 0.0), 'constant': (A1_LOW, A2_LOW, A3_LOW)}
    want = [x.strip() for x in args.scenarios.split(',') if x.strip()]
    matched = {}
    if args.matched_table:
        mt = pd.read_csv(args.matched_table)
        for r in mt.itertuples():
            amps = tuple(getattr(r, f'A{m}_{args.matched_col}', np.nan) for m in (1, 2, 3))
            matched[round(float(r.z), 2)] = tuple(0.0 if not np.isfinite(a) else float(a)
                                                  for a in amps)
        if 'matched' not in want:
            want.append('matched')
        print("Matched amplitudes (A1, A2, A3) per z: " +
              ', '.join(f"{k}: ({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})" for k, v in matched.items()))
    scenarios = {k: v for k, v in all_scen.items() if k in want}

    rows = []
    for ib, (lo, hi) in enumerate(zip(Z_EDGES[:-1], Z_EDGES[1:])):
        sub = d[(d.z >= lo) & (d.z < hi)]
        if sub.empty:
            continue
        draw = sub.sample(n=args.n_per_bin, replace=len(sub) < args.n_per_bin,
                          random_state=int(rng.integers(1 << 31)))
        for g in draw.itertuples():
            n = float(np.clip(g.n, 0.3, 2.5))
            x0 = (npix - 1) / 2 + rng.uniform(-0.5, 0.5)
            y0 = (npix - 1) / 2 + rng.uniform(-0.5, 0.5)
            phases = rng.uniform(0, 2 * np.pi, 3)
            bin_scen = dict(scenarios)
            if 'matched' in want and round(Z_CEN[ib], 2) in matched:
                bin_scen['matched'] = tuple(args.matched_scale * a
                                            for a in matched[round(Z_CEN[ib], 2)])
            ring = g.re_pix * args.ring_scale
            for scen, eps in bin_scen.items():
                if scen == 'matched' and args.matched_spread > 0:
                    eps = tuple(e * 10 ** rng.normal(0, args.matched_spread) for e in eps)
                cl = None
                if args.clumps and scen != 'zero':
                    cl = (args.clump_frac,
                          [(rng.uniform(0.3, 1.8), rng.uniform(0, 2 * np.pi),
                            args.clump_sigma) for _ in range(args.clumps)])
                # truth: same estimator on the noiseless, unconvolved model
                clean = make_model(npix, x0, y0, g.re_pix, n, g.q, g.pa, eps, phases,
                                   profile=args.pert_profile, clumps=cl)
                t1, t2, t3 = measure(clean, x0, y0, g.q, g.pa, ring)
                obs = fftconvolve(clean, psf, mode='same')
                img, sig = add_noise(obs, g.snr, aper, x0, y0, rng, noise_acf)
                for sc in schemes:
                    if sc == 'true':
                        cx, cy = x0, y0
                    elif sc == 'catalog':
                        cx, cy = x0 + rng.normal(0, args.sigma_cat), y0 + rng.normal(0, args.sigma_cat)
                    elif sc == 'iterative':
                        cx, cy = centre_iterative(img, x0, y0, max(g.re_pix, 2.0))
                    elif sc == 'peak':
                        cx, cy = centre_peak(img, x0, y0, max(g.re_pix, 2.0))
                    elif sc == 'grid':
                        # centre snapped to the pixel grid (+ optional constant
                        # offset), as when a pixel-aligned cutout's
                        # geometric centre is used
                        cx = np.round(x0) + args.grid_offset
                        cy = np.round(y0) + args.grid_offset
                    elif sc == 'symmetric':
                        cx, cy = centre_symmetric(img, x0, y0, g.re_pix)
                    else:
                        sys.exit(f"unknown centering '{sc}'")
                    a1, a2, a3 = measure(img, cx, cy, g.q, g.pa, ring)
                    rows.append(dict(zbin=Z_CEN[ib], z=g.z, re_pix=g.re_pix, n=n,
                                     q=g.q, snr=g.snr, scenario=scen, centering=sc,
                                     dcen=np.hypot(cx - x0, cy - y0),
                                     dx=cx - x0, dy=cy - y0,
                                     A1_true=t1, A2_true=t2, A3_true=t3,
                                     A1=a1, A2=a2, A3=a3))
        print(f"  z={Z_CEN[ib]:.2f}: {len(draw)} galaxies done")

    res = pd.DataFrame(rows)
    res['R12'] = res.A1 / res.A2
    res.to_csv(os.path.join(args.output, 'injection_per_object.csv'), index=False)

    # ---- per-bin summary with bootstrap errors on the medians
    def boot(x, nb=500):
        x = x[np.isfinite(x)]
        if len(x) < 5:
            return np.nan, np.nan
        bs = np.median(rng.choice(x, (nb, len(x))), axis=1)
        return np.median(x), np.std(bs)

    summ = []
    for (scen, sc, zb), g in res.groupby(['scenario', 'centering', 'zbin']):
        row = dict(scenario=scen, centering=sc, zbin=zb, N=len(g),
                   re_pix_med=g.re_pix.median(), snr_med=g.snr.median(),
                   A1_true_med=g.A1_true.median(), A2_true_med=g.A2_true.median())
        for k in ('A1', 'A2', 'A3', 'R12'):
            row[f'{k}_med'], row[f'{k}_err'] = boot(g[k].values)
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(args.output, 'injection_summary.csv'), index=False)

    # ---- console verdict (null floor vs observed, ideal + catalog centering)
    print("\n=== NULL test (A1_in = 0): measurement floor vs observed A1 ===")
    for sc in schemes:
        s = summ[(summ.scenario == 'zero') & (summ.centering == sc)].set_index('zbin')
        print(f"\n[{sc}]  z    Re_pix  A1_floor   A1_obs   A1_corr=sqrt(obs^2-floor^2)")
        for zc, a1o in zip(Z_CEN, OBS_A1):
            if zc not in s.index:
                continue
            f_ = s.loc[zc, 'A1_med']
            corr = np.sqrt(max(a1o ** 2 - f_ ** 2, 0))
            print(f"      {zc:4.2f}  {s.loc[zc,'re_pix_med']:5.2f}  "
                  f"{f_:.3f}±{s.loc[zc,'A1_err']:.3f}  {a1o:.3f}   {corr:.3f}")
    print("\n=== CONSTANT test (intrinsic A1/A2 fixed at z~0.75 value "
          f"{A1_LOW/A2_LOW:.2f}) ===")
    for sc in schemes:
        s = summ[(summ.scenario == 'constant') & (summ.centering == sc)].set_index('zbin')
        vals = ' '.join(f"{s.loc[z_,'R12_med']:.2f}" if z_ in s.index else '  - '
                        for z_ in Z_CEN)
        print(f"[{sc:9s}] measured A1/A2: {vals}")
    print(f"[observed ] A1/A2         : {' '.join(f'{v:.2f}' for v in OBS_R12)}")
    print("\n=== Centre error per scheme (ZERO scenario): robust sigma per axis [pix] ===")
    for sc in schemes:
        z0 = res[(res.scenario == 'zero') & (res.centering == sc)]
        vals = []
        for z_ in Z_CEN:
            gz = z0[z0.zbin == z_]
            s_ = [1.4826 * np.median(np.abs(v - np.median(v))) for v in (gz.dx, gz.dy)]
            vals.append(f"{np.mean(s_):.2f}" if len(gz) else '  - ')
        print(f"[{sc:9s}] sigma_xy      : {' '.join(vals)}")
    print("\n=== ZERO test: A1/A2 of pure noise (noise-dominated limit) ===")
    for sc in schemes:
        s = summ[(summ.scenario == 'zero') & (summ.centering == sc)].set_index('zbin')
        vals = ' '.join(f"{s.loc[z_,'R12_med']:.2f}" if z_ in s.index else '  - '
                        for z_ in Z_CEN)
        print(f"[{sc:9s}] A1/A2 floor   : {vals}")

    # ---- figure
    colors = {'true': '#1f77b4', 'catalog': '#2ca02c', 'iterative': '#d62728', 'peak': '#ff7f0e'}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axs[0]
    ax.plot(Z_CEN, OBS_A1, 'ks-', label='Observed $A_1$ (Table 1)')
    for sc in schemes:
        s = summ[(summ.scenario == 'zero') & (summ.centering == sc)]
        ax.errorbar(s.zbin, s.A1_med, s.A1_err, fmt='o--', color=colors.get(sc),
                    capsize=3, label=f'Floor, {sc} centre')
    ax.set(xlabel='Redshift $z$', ylabel='median $A_1$',
           title='(a) Null injection: $A_1^{\\rm in}=0$')
    ax.legend(fontsize=8)

    ax = axs[1]
    ax.plot(Z_CEN, OBS_R12, 'ks-', label='Observed')
    ax.axhline(A1_LOW / A2_LOW, color='gray', ls=':', label='Injected (constant)')
    for sc in schemes:
        s = summ[(summ.scenario == 'constant') & (summ.centering == sc)]
        ax.errorbar(s.zbin, s.R12_med, s.R12_err, fmt='o--', color=colors.get(sc),
                    capsize=3, label=f'Recovered, {sc}')
    ax.set(xlabel='Redshift $z$', ylabel='median $(A_1/A_2)$',
           title='(b) No-evolution injection')
    ax.legend(fontsize=8)

    ax = axs[2]
    sel = res[(res.scenario == 'zero') & (res.centering == schemes[0])]
    sc_ = ax.scatter(sel.re_pix, sel.A1, c=np.log10(sel.snr), s=4, cmap='viridis', alpha=0.6)
    bins = np.logspace(np.log10(max(sel.re_pix.min(), 0.5)), np.log10(sel.re_pix.max()), 12)
    idx = np.digitize(sel.re_pix, bins)
    med = [(np.median(sel.re_pix[idx == i]), np.median(sel.A1[idx == i]))
           for i in range(1, len(bins)) if (idx == i).sum() > 10]
    if med:
        ax.plot(*zip(*med), 'r-', lw=2, label='running median')
    ax.axvline(args.psf_fwhm / args.pixscale, color='k', ls=':', label='PSF FWHM')
    ax.set(xscale='log', xlabel='$R_e$ [pix]', ylabel='$A_1$ (null)',
           title=f'(c) Floor vs size ({schemes[0]} centre)')
    plt.colorbar(sc_, ax=ax, label='log S/N')
    ax.legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(args.output, f'FigX_injection_test.{ext}'), dpi=200)
    print(f"\nSaved: {args.output}/injection_summary.csv, injection_per_object.csv, "
          "FigX_injection_test.pdf/png")


if __name__ == '__main__':
    main()
