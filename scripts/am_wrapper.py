"""
am_wrapper.py — the pipeline's extract_am (copied verbatim from
RUN_ALL_TESTS.py, lines 103-137) wrapped for injection_test.py --am_func.

    measure_am      : deprojects with the galaxy's TRUE PA (ideal pipeline)
    measure_am_pa0  : deprojects with PA = 0, i.e. what extract_am does when
                      only pa_off_deg = 0 is passed (no catalog PA)
    measure_v5      : the corrected V5 estimator of run_corrected_am.py
    measure_v6      : V5 with the galaxy's own PA (= V6 with catalog PA)

Usage:
    python3 injection_test.py ... --am_func am_wrapper.py:measure_am
    python3 injection_test.py ... --am_func am_wrapper.py:measure_am_pa0

injection_test.py builds its models with the same PA convention as
extract_am (x_rot = dx cos PA + dy sin PA), so no conversion is needed.
"""

import numpy as np


def extract_am(image, cx_off, cy_off, pa_off_deg, q, Re_pix, n_phi=90):
    """Compute A_m with center and PA offsets."""
    ny, nx = image.shape
    cy = ny // 2 + cy_off
    cx = nx // 2 + cx_off
    yy, xx = np.indices(image.shape)
    xd = xx - cx; yd = yy - cy
    pa = np.deg2rad(pa_off_deg)
    x_rot = xd * np.cos(-pa) - yd * np.sin(-pa)
    y_rot = xd * np.sin(-pa) + yd * np.cos(-pa)
    y_dep = y_rot / max(q, 0.1)
    r = np.sqrt(x_rot**2 + y_dep**2)
    phi = np.arctan2(y_dep, x_rot) % (2 * np.pi)
    mask_r = (r >= 0.5 * Re_pix) & (r <= 1.5 * Re_pix)
    phi_bins = np.linspace(0, 2*np.pi, n_phi + 1)
    sigma = np.zeros(n_phi)
    for i in range(n_phi):
        mask = mask_r & (phi >= phi_bins[i]) & (phi < phi_bins[i+1])
        pix = image[mask]
        pix = pix[np.isfinite(pix) & (pix > 0)]
        sigma[i] = np.median(pix) if len(pix) > 0 else np.nan
    valid = ~np.isnan(sigma)
    if valid.sum() < 10:
        return None
    pc = 0.5 * (phi_bins[1:] + phi_bins[:-1])[valid]
    sv = sigma[valid]
    a0 = np.mean(sv)
    result = {}
    for m in range(1, 4):
        a_m = (2/len(pc)) * np.sum(sv * np.cos(m*pc))
        b_m = (2/len(pc)) * np.sum(sv * np.sin(m*pc))
        result[f'A{m}'] = np.sqrt(a_m**2 + b_m**2) / a0
    return result


def _call(img, x0, y0, q, pa_deg, re_pix):
    ny, nx = img.shape
    res = extract_am(img, x0 - nx // 2, y0 - ny // 2, pa_deg, q, re_pix)
    if res is None:
        return np.nan, np.nan, np.nan
    return res['A1'], res['A2'], res['A3']


def measure_am(img, x0, y0, q, pa_deg, re_pix):
    return _call(img, x0, y0, q, pa_deg, re_pix)


def measure_am_pa0(img, x0, y0, q, pa_deg, re_pix):
    return _call(img, x0, y0, q, 0.0, re_pix)


def measure_v5(img, x0, y0, q, pa_deg, re_pix):
    """V5 estimator of run_corrected_am.py (sub-pixel centre, deprojection with
    q and a moments PA, 0.5-1.5 r_d annulus, original normalisation). The
    injected images have zero background and no neighbours, so no background
    subtraction or masking is needed here."""
    from run_corrected_am import am_general, moments_pa
    pa = moments_pa(img, x0, y0, 2.5 * re_pix)
    return am_general(img, x0, y0, max(1.0, 0.5 * re_pix), 1.5 * re_pix, q, pa)


def measure_v6(img, x0, y0, q, pa_deg, re_pix):
    """V6 estimator: as V5 but with the galaxy's own PA (in the injection the
    true PA stands in for the catalog Sersic PA)."""
    from run_corrected_am import am_general
    return am_general(img, x0, y0, max(1.0, 0.5 * re_pix), 1.5 * re_pix, q, pa_deg)
