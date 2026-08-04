"""Tests for fermi_plot_helpers.py's pure numeric functions.

`find_thermal_crossing` and `ul_from_stack` are covered here -- both take
and return plain arrays/scalars. The Handler* legend classes,
`shade_bounding_box_to_zero`, and `plot_ellipse_from_csv` all need a live
matplotlib Axes/Figure or a CSV on disk to exercise meaningfully, and
`positive_preference` is a simple threshold check not called out by this
task, so none of those are covered here.
"""

import numpy as np
import pytest

import fermi_plot_helpers as fph


def test_find_thermal_crossing_finds_crossing_between_grid_points():
    mass_vec = np.array([10.0, 20.0, 40.0, 80.0])
    line_values = np.array([3e-25, 1.5e-25, 0.75e-25, 0.375e-25])
    sigma_thermal = np.array([[10.0, 1e-25], [80.0, 1e-25]])

    cross_mass, cross_sigmav, excl_factor = fph.find_thermal_crossing(
        mass_vec, line_values, sigma_thermal, min_mass=10.0, eval_mass=40.0)

    assert cross_mass == pytest.approx(31.74802103936399)
    assert cross_sigmav == pytest.approx(1e-25)
    assert excl_factor == pytest.approx(0.75)


def test_find_thermal_crossing_respects_min_mass_cut():
    """Raising min_mass above the point where line_values is still above
    the thermal curve should push the found crossing to a different (later)
    mass than with no cut, since the search only looks above min_mass."""
    mass_vec = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    line_values = np.array([3e-25, 1.5e-25, 0.75e-25, 0.375e-25, 3e-25])
    sigma_thermal = np.array([[10.0, 1e-25], [160.0, 1e-25]])

    cross_mass_full, _, _ = fph.find_thermal_crossing(
        mass_vec, line_values, sigma_thermal, min_mass=10.0, eval_mass=40.0)
    cross_mass_cut, _, _ = fph.find_thermal_crossing(
        mass_vec, line_values, sigma_thermal, min_mass=50.0, eval_mass=100.0)

    assert cross_mass_full == pytest.approx(31.74802103936399)
    assert cross_mass_cut is not None
    assert cross_mass_cut > 50.0
    assert cross_mass_cut != pytest.approx(cross_mass_full)


def test_find_thermal_crossing_no_crossing_returns_none_mass_and_sigmav():
    mass_vec = np.array([10.0, 20.0, 40.0])
    line_values = np.array([3e-25, 2e-25, 1.5e-25])  # stays above thermal throughout
    sigma_thermal = np.array([[10.0, 1e-26], [40.0, 1e-26]])

    cross_mass, cross_sigmav, excl_factor = fph.find_thermal_crossing(
        mass_vec, line_values, sigma_thermal, min_mass=10.0, eval_mass=20.0)

    assert cross_mass is None
    assert cross_sigmav is None
    assert excl_factor is not None


def test_find_thermal_crossing_eval_mass_out_of_range_returns_none_excl_factor():
    mass_vec = np.array([10.0, 20.0, 40.0, 80.0])
    line_values = np.array([3e-25, 1.5e-25, 0.75e-25, 0.375e-25])
    sigma_thermal = np.array([[10.0, 1e-25], [80.0, 1e-25]])

    _, _, excl_factor = fph.find_thermal_crossing(
        mass_vec, line_values, sigma_thermal, min_mass=10.0, eval_mass=200.0)

    assert excl_factor is None


def test_ul_from_stack_finds_downward_crossing_past_the_peak():
    sigmav_vec = np.linspace(0.0, 10.0, 1001)
    TS_row = 10.0 - 0.1 * (sigmav_vec - 5.0)**2  # peak TS=10 at sigmav=5

    ul = fph.ul_from_stack(TS_row, sigmav_vec, coeff=1.0)

    assert ul == pytest.approx(5.0 + 10.0**0.5, abs=1e-2)


def test_ul_from_stack_out_of_range_crossing_is_nan():
    sigmav_vec = np.linspace(0.0, 10.0, 1001)
    TS_row = 10.0 - 0.1 * (sigmav_vec - 5.0)**2  # peak TS=10 at sigmav=5

    ul = fph.ul_from_stack(TS_row, sigmav_vec, coeff=2.71)  # crossing beyond sigmav_vec.max()

    assert np.isnan(ul)
