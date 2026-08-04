"""contour_io.py's persistence layer (save_contour) writes through
provenance.savez and its contour computation imports KDEpy at module scope
-- KDEpy lives in the J_calc environment, so importing the module at all
needs it, even though `_compute_contour_level` itself has no such dependency.
See tests/test_dwarf_categories.py for the same importorskip pattern.
"""

import numpy as np
import pytest

contour_io = pytest.importorskip('contour_io', reason='needs the J_calc environment')

pytestmark = pytest.mark.needs_satgen


def test_compute_contour_level_matches_manual_cumulative_threshold():
    # flat, sorted desc: [4, 3, 2, 1]; cumsum: [4, 7, 9, 10]; normalized:
    # [0.4, 0.7, 0.9, 1.0]. searchsorted(0.68) lands at index 1 -> 3.0.
    grid = np.array([1.0, 2.0, 3.0, 4.0])
    level = contour_io._compute_contour_level(grid, 0.68)
    assert level == pytest.approx(3.0)


def test_scalar_level_returns_a_scalar():
    grid = np.array([[4.0, 3.0], [2.0, 1.0]])
    level = contour_io._compute_contour_level(grid, 0.5)
    assert np.shape(level) == ()


def test_list_level_returns_a_3_element_array_not_a_scalar():
    """save_contours.py's '_unweighted' calls pass level=[0.68, 0.95, 0.995]
    -- a 3-element list -- into a parameter documented (and used everywhere
    else in this repo) as a single scalar confidence level.
    np.searchsorted broadcasts over the array `level` instead of raising, so
    the on-disk '_unweighted' contour arrays are shape-(3,), unlike every
    weighted product's scalar. Verified against on-disk products: this is
    upstream-intentional, not a bug to "fix" here -- pin the asymmetry."""
    grid = np.array([[4.0, 3.0], [2.0, 1.0]])
    level = contour_io._compute_contour_level(grid, [0.68, 0.95, 0.995])
    assert np.shape(level) == (3,)


def test_higher_level_selects_a_lower_or_equal_threshold():
    """A higher confidence level covers more probability mass, so the
    threshold density it selects (the highest still enclosing that mass)
    must be lower or equal."""
    grid = np.array([1.0, 2.0, 3.0, 4.0])
    low_level = contour_io._compute_contour_level(grid, 0.5)
    high_level = contour_io._compute_contour_level(grid, 0.95)
    assert high_level <= low_level


def test_list_level_matches_elementwise_scalar_calls():
    grid = np.array([1.0, 2.0, 3.0, 4.0])
    levels = [0.4, 0.68, 0.9]
    combined = contour_io._compute_contour_level(grid, levels)
    separate = [contour_io._compute_contour_level(grid, l) for l in levels]
    assert np.array_equal(combined, separate)
