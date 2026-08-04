"""
Uncertainty on the weighted-median V_circ(r) of a dwarf galaxy, driven by
observational uncertainty in logMstar.

For a given dwarf, the weighted median V_circ(r) across SatGen halos depends on
the observed logMstar through the Fattahi18-style SHMR weights. Because the
observed logMstar carries an (asymmetric) uncertainty, the median V_circ has a
corresponding uncertainty that is *not* the weighted posterior spread of V_circ
across halos. This module Monte-Carlos over the observed logMstar and returns
the 16/50/84 of the resulting median distribution at each radius.

Migrated from SatGen_Dwarf/python/vcirc_median_uncertainty.py verbatim: no
hardcoded paths, no import changes -- `from util import quantile` resolves the
same way it does for every other migrated script that imports from python/.
"""

import copy
import numpy as np

from util import quantile


def _sample_twosided(rng, mean, lowerr, higherr, size):
    """Draw `size` samples from the two-sided Gaussian used by Jdata.Dwarf."""
    p_low = lowerr / (lowerr + higherr)
    u = rng.uniform(size=size)
    z = np.abs(rng.standard_normal(size=size))
    below = u < p_low
    out = np.where(below, mean - z * lowerr, mean + z * higherr)
    return out


def vcirc_median_uncertainty(
    dwarf,
    weights_obj,
    logMstar_pred,
    vcirc_profile,
    r_grid,
    extra_mask=None,
    n_mc=200,
    narrow_sigma=0.1,
    quantiles=(0.16, 0.5, 0.84),
    mask_kwargs=None,
    seed=None,
    return_samples=False,
):
    """Uncertainty on the weighted-median V_circ(r) from observational sigma_logMstar.

    Parameters
    ----------
    dwarf : Jdata.Dwarf
        Target dwarf. Used for `logMstar`, `logMstar_err`, and passed to
        `weights_obj.get_mask`.
    weights_obj : halo_weights.ResampleMstar
        Already constructed for the desired SatGen version.
    logMstar_pred : np.ndarray, shape (N_halos,)
        Per-halo z=0 predicted logMstar (caller's SHMR choice).
    vcirc_profile : np.ndarray, shape (N_halos, N_r)
        Per-halo V_circ profile in km/s.
    r_grid : np.ndarray, shape (N_r,)
        Radii.
    extra_mask : np.ndarray(bool), shape (N_halos,), optional
        Additional halo-level selection (e.g. MW-host, LMC removal).
    n_mc : int
        Number of Monte-Carlo draws of the observed logMstar.
    narrow_sigma : float
        Width (dex) of the inner conditioning PDF. Each MC iteration treats the
        drawn logMstar as the "true" value with this width.
    quantiles : sequence of float
        Quantiles of the MC median distribution to report.
    mask_kwargs : dict, optional
        Passed through to `weights_obj.get_mask(dwarf, **mask_kwargs)`.
    seed : int, optional
    return_samples : bool
        If True, include the full `(n_mc, N_r)` array of per-draw medians.

    Returns
    -------
    dict with keys
        r_grid           : (N_r,)
        median_vcirc     : (N_r,) median across MC draws
        median_vcirc_err : (N_r, len(quantiles)) MC quantiles on the median
        mu_samples       : (n_mc,) drawn central logMstar values
        [mc_medians]     : (n_mc, N_r) if return_samples
    """
    rng = np.random.default_rng(seed)
    mask_kwargs = {} if mask_kwargs is None else dict(mask_kwargs)

    mask = weights_obj.get_mask(dwarf, **mask_kwargs)
    if extra_mask is not None:
        mask = mask & np.asarray(extra_mask, dtype=bool)

    lm = logMstar_pred[mask]
    vc = vcirc_profile[mask]

    kde = weights_obj._kde1D(lm, **weights_obj.kde_kwargs)
    norm = kde(lm)

    lowerr, higherr = dwarf.logMstar_err
    mu_samples = _sample_twosided(
        rng, dwarf.logMstar, lowerr, higherr, size=n_mc
    )

    n_r = len(r_grid)
    mc_medians = np.empty((n_mc, n_r))
    narrow_err = np.array([narrow_sigma, narrow_sigma])

    mock = copy.copy(dwarf)
    mock.logMstar_err = narrow_err
    for k, mu_k in enumerate(mu_samples):
        mock.logMstar = mu_k
        w = mock.logMstar_pdf(lm) / norm
        for j in range(n_r):
            mc_medians[k, j] = quantile(vc[:, j], weights=w, quantiles=0.5)

    q = np.asarray(quantiles)
    median_vcirc = np.median(mc_medians, axis=0)
    median_vcirc_err = np.empty((n_r, len(q)))
    for j in range(n_r):
        median_vcirc_err[j] = np.quantile(mc_medians[:, j], q)

    out = {
        'r_grid': np.asarray(r_grid),
        'median_vcirc': median_vcirc,
        'median_vcirc_err': median_vcirc_err,
        'mu_samples': mu_samples,
    }
    if return_samples:
        out['mc_medians'] = mc_medians
    return out
