"""weighted_stats.py imports `quantile` from util.py, which imports vegas --
so needs the J_calc environment. See conftest.py's `needs_satgen` marker.
"""

import numpy as np
import pytest

pytest.importorskip('vegas', reason='needs the J_calc environment')

import weighted_stats as ws

pytestmark = pytest.mark.needs_satgen


def test_weighted_median_matches_np_percentile_on_unit_weights():
    values = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    expected = np.percentile(values, 50)
    got = ws.quantile(values, weights=np.ones_like(values), quantiles=0.5)
    assert got == pytest.approx(expected)


def test_quantile_q0_edge_returns_minimum():
    values = np.arange(1, 11, dtype=float)
    assert ws.quantile(values, quantiles=0.0) == pytest.approx(values.min())


def test_quantile_q1_edge_hits_the_len_c_minus_2_clip():
    """np.clip(np.searchsorted(...), 0, len(c) - 2) is a live off-by-one
    candidate: with unit weights and quantiles=1.0, `c` is strictly
    increasing (1..n) so searchsorted lands on index n-1 (the true max), but
    the clip pulls it back to n-2 -- one short of the max. Pin this exact,
    surprising behavior (returns the second-largest value, not the max) so a
    future "fix" to the clip bound is caught as a behavior change, not
    silently accepted."""
    values = np.arange(1, 11, dtype=float)
    got = ws.quantile(values, quantiles=1.0)
    assert got == pytest.approx(9.0)
    assert got != pytest.approx(values.max())


def test_quantile_all_nan_returns_nan():
    values = np.full(5, np.nan)
    result = ws.quantile(values, quantiles=np.array([0.16, 0.5, 0.84]))
    assert np.all(np.isnan(result))
    assert np.shape(result) == (3,)


def test_quantile_is_the_same_object_as_util_quantile():
    """weighted_stats does not redefine quantile -- it re-exports util's."""
    import util
    assert ws.quantile is util.quantile


def test_quantile_sorted_raises_on_nan_input_with_unfiltered_sort_order():
    """Pins quantile_sorted's documented precondition: `sorted_idcs` computed
    from the unfiltered array is out of range once NaNs are dropped
    internally, so this raises IndexError rather than silently misbehaving.
    Both notebook call sites avoid this by only ever passing NaN-free input;
    this is a known, tested property, not a latent surprise."""
    values = np.array([3.0, np.nan, 1.0, 4.0])
    weights = np.ones_like(values)
    sorted_idcs = np.argsort(values)
    with pytest.raises(IndexError):
        ws.quantile_sorted(values, sorted_idcs, weights=weights, quantiles=0.5)


def test_quantile_sorted_matches_quantile_with_precomputed_order():
    rng = np.random.default_rng(0)
    values = rng.normal(size=50)
    weights = rng.uniform(0.5, 2.0, size=50)
    sorted_idcs = np.argsort(values)
    q = np.array([0.16, 0.5, 0.84])
    a = ws.quantile(values, weights=weights, quantiles=q)
    b = ws.quantile_sorted(values, sorted_idcs, weights=weights, quantiles=q)
    assert a == pytest.approx(b)


def test_N_greater_counts_strictly_above():
    xax_vals = np.array([1.0, 2.0, 3.0, 4.0])
    grid = np.array([0.0, 2.0, 4.0])
    grid_out, counts = ws.N_greater(xax_vals, grid)
    assert np.array_equal(grid_out, grid)
    assert np.array_equal(counts, [4, 2, 0])


def test_N_lesser_counts_strictly_below():
    xax_vals = np.array([1.0, 2.0, 3.0, 4.0])
    grid = np.array([0.0, 2.0, 4.0])
    grid_out, counts = ws.N_lesser(xax_vals, grid)
    assert np.array_equal(grid_out, grid)
    assert np.array_equal(counts, [0, 1, 3])


def test_has_greater_is_N_greater_thresholded_at_zero():
    xax_vals = np.array([1.0, 2.0, 3.0])
    grid = np.array([0.0, 5.0])
    _, has = ws.has_greater(xax_vals, grid)
    assert np.array_equal(has, [True, False])
