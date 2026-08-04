"""Tests for contour_plot.py's pure geometry helper.

Only `position_on_contour` is covered: it is the one function on this
module's function list that takes plain arrays in and plain tuples out. The
label-shifting/sci-fmt helpers all require a live matplotlib `contours`
object (from a real `plt.contour` call) to exercise meaningfully, and are
left untested here.

The `if y1 == y0: return (x0, y_val)` branch inside `position_on_contour` is
deliberately NOT targeted: it is unreachable given the guard above it
(`sign_change` requires `sign(d0)*sign(d1) <= 0`; `y0 == y1` forces
`d0 == d1`, so the only way to satisfy the guard is `d0 == d1 == 0`, which
`both_zero` already excludes). It is verbatim upstream code; this is a
documented finding, not something to fake coverage for.
"""

import numpy as np

import contour_plot as cp


def test_position_on_contour_finds_linear_crossing():
    seg = np.array([[0.0, 0.0], [1.0, 10.0]])
    x, y = cp.position_on_contour([seg], y_val=5.0)
    assert x == 5.0 / 10.0
    assert y == 5.0


def test_position_on_contour_returns_none_when_no_segment_crosses():
    seg = np.array([[0.0, 0.0], [1.0, 3.0]])
    assert cp.position_on_contour([seg], y_val=5.0) is None


def test_position_on_contour_skips_non_crossing_segments_in_order():
    seg_no_cross = np.array([[0.0, 0.0], [1.0, 3.0]])
    seg_cross = np.array([[2.0, 0.0], [3.0, 10.0]])
    x, y = cp.position_on_contour([seg_no_cross, seg_cross], y_val=5.0)
    assert x == 2.0 + 5.0 / 10.0
    assert y == 5.0


def test_position_on_contour_skips_segments_shorter_than_two_points():
    seg_too_short = np.array([[0.0, 5.0]])
    seg_cross = np.array([[1.0, 0.0], [2.0, 10.0]])
    x, y = cp.position_on_contour([seg_too_short, seg_cross], y_val=5.0)
    assert x == 1.0 + 5.0 / 10.0
    assert y == 5.0
