"""Weighted-quantile and counting helpers, lifted verbatim from
../SatGen_Dwarf/jupyter/PaperPlots.ipynb, where they were pasted identically
into ~13 cells (70, 73, 76, 84, 96, 101, 113, 120, 138, 145, 154, 155, 162,
166, 167). All copies of each function were byte-identical except:

- `N_greater`: cell 70 gave `xax_grid` a default value
  (`np.linspace(0,11, num=100)`); the other 7 copies (73, 76, 96, 113, 138,
  145, 166) required it as a positional arg. Every call site in the notebook
  passes `xax_grid` explicitly, so the default is never exercised either way;
  the majority (no-default) form is kept here.
- `N_lesser`: only one copy exists (cell 84), with a default `xax_grid`; kept
  as-is.

`quantile` in the notebook is byte-identical, in every one of those copies, to
the `quantile` already defined in `python/util.py` (and imported from there by
Jdata.py). Rather than pasting a second implementation, this module imports
util's version and re-exports it, so `from weighted_stats import quantile`
keeps working for callers migrated from the notebook.
"""

import numpy as np

from util import quantile

__all__ = ['N_greater', 'N_lesser', 'has_greater', 'quantile', 'quantile_sorted']


def N_greater(xax_vals, xax_grid):
    return xax_grid, np.array([np.count_nonzero(xax_vals > x) for x in xax_grid])


def N_lesser(xax_vals, xax_grid=np.linspace(0, 11, num=100)):
    return xax_grid, np.array([np.count_nonzero(xax_vals < x) for x in xax_grid])


def has_greater(xax_vals, xax_grid):
    # note the > 0 part
    return xax_grid, np.array([np.count_nonzero(xax_vals > x) for x in xax_grid]) > 0


def quantile_sorted(values, sorted_idcs, weights=None, quantiles=0.5):
    """Upstream-faithful precondition, not enforced here: `sorted_idcs` must
    be the sort order of `values` AFTER NaN-filtering (or `values` must be
    NaN-free to begin with). This filters `values`/`weights` by ~isnan but
    then indexes the FILTERED arrays with `sorted_idcs` -- both notebook call
    sites compute `sorted_idcs` from the UNFILTERED array, which is only safe
    because those call sites pass NaN-free input. Passing `sorted_idcs` from
    an unfiltered array that DOES contain NaNs raises IndexError here."""
    # https://stackoverflow.com/a/73905572
    m = ~np.isnan(values)
    if not np.any(m):
        return np.full(np.shape(quantiles), np.nan)
    v = values[m]
    if weights is None:
        weights = np.ones_like(values)
    wgt = weights[m]
    # i = np.argsort(v)
    i = sorted_idcs
    c = np.cumsum(wgt[i])
    q = np.clip(np.searchsorted(c, quantiles * c[-1]), 0, len(c) - 2)
    return np.where(c[q]/c[-1] == quantiles, 0.5 * (v[i[q]] + v[i[q+1]]), v[i[q]])
