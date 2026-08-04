"""Histogram + KDE contour computation, shared by save_contours.py and any
plot script that consumes its output.

Migrated from SatGen_Dwarf/python/save_contours.py (save_contour,
_compute_contour_level). Split out of that script rather than imported from
it: upstream parses sys.argv at module scope (dsph_idx/version/redshift), so
`from save_contours import save_contour` crashes on import outside a direct
CLI invocation. This module has no argv parsing and no path handling of its
own -- `save_dir` is a plain string prefix supplied by the caller, exactly as
upstream -- so it imports cleanly anywhere.

The two functions below compute the contours identically to upstream: grid
construction, KDE/histogram settings, and the contour-level definition are
unchanged. Persistence is not upstream's: `save_contour` writes through
`provenance.savez` rather than a raw `np.savez`, so this module does import
`provenance` (and, transitively, `config`) -- checked for a cycle: neither
module imports contour_io, so there is none.
"""
import numpy as np
from KDEpy import FFTKDE as KDE

import provenance


def save_contour(
    xdata, ydata, weights, name, save_dir, record, level=0.68,
    bins=30, kde_bw=0.01, kde_kernel='epa', n_kde=1024
):
    """
    Save both histogram-based and KDE-based contour data into the same file.

    Parameters
    ----------
    xdata, ydata : array-like
        Data coordinates.
    weights : array-like
        Weights for each point.
    name : str
        File name (without extension).
    save_dir : str
        Directory path to save the .npz file.
    record : dict
        Provenance record (from `provenance.stamp`) to attach to the product.
        Written via `provenance.savez`, which stamps the file in the same
        call that writes it (in-band key plus sidecar) rather than leaving a
        window where fresh bytes sit unstamped or, worse, beside a *previous*
        run's still-valid sidecar -- and which creates `save_dir` if it does
        not already exist, so the caller does not need to pre-create it.
    level : float or array-like of float
        Contour confidence level (e.g., 0.68 for 1-sigma). `save_contours.py`
        also calls this with a 3-element list (`[0.68, 0.95, 0.995]`) for its
        unweighted contours; `_compute_contour_level`'s
        `np.searchsorted(cumsum, level)` broadcasts over that, so `level`
        array-like in, `contour_hist`/`contour_kde` array-like out, is a
        real, exercised path, not just a scalar convenience.
    bins : int
        Number of bins for histogram.
    kde_bw : float
        Bandwidth for KDE.
    kde_kernel : str
        Kernel for KDE.
    n_kde : int
        Grid size per axis for KDE (total points = n_kde^2).
    """

    # -------------------------
    # Histogram method
    # -------------------------
    x_bins = np.linspace(min(xdata) - 1e-10, max(xdata), bins)
    y_bins = np.linspace(min(ydata) - 1e-10, max(ydata), bins)

    hist_hist, xedges, yedges = np.histogram2d(
        xdata, ydata, bins=[x_bins, y_bins], weights=weights, density=True
    )

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    X_hist, Y_hist = np.meshgrid(xcenters, ycenters)

    contour_hist = _compute_contour_level(hist_hist, level)

    # -------------------------
    # KDE method
    # -------------------------
    grid, height = KDE(bw=kde_bw, kernel=kde_kernel).fit(
        np.c_[xdata, ydata], weights=weights
    ).evaluate(n_kde)

    X_kde = np.unique(grid[:, 0])
    Y_kde = np.unique(grid[:, 1])
    hist_kde = height.reshape(n_kde, n_kde)

    contour_kde = _compute_contour_level(hist_kde, level)

    # -------------------------
    # Save both, stamped
    # -------------------------
    provenance.savez(
        f"{save_dir}{name}", record,
        # Histogram data
        X_hist=X_hist, Y_hist=Y_hist, hist_hist=hist_hist, contour_hist=contour_hist,
        # KDE data
        X_kde=X_kde, Y_kde=Y_kde, hist_kde=hist_kde, contour_kde=contour_kde
    )


def _compute_contour_level(density_grid, level):
    """Helper to compute contour threshold for a given confidence level."""
    flat = density_grid.flatten()
    sorted_vals = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_vals)
    cumsum /= cumsum[-1]
    return sorted_vals[np.searchsorted(cumsum, level)]
