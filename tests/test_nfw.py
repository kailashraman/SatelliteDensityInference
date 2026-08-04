"""nfw.py round-trip tests.

rho150_predicted(log_c) (the closure inside nfw_parameters_from_rho150_mpeak)
rises then falls with c for fixed Mpeak, so for a target rho150 close enough
to the achievable maximum there are two brackets with a sign change --
`nfw_parameters_from_rho150_mpeak` returns one dict per bracket via two
separate brentq calls. Both branches are exercised below by picking rho150
just under rho150_max for a fixed Mpeak (found empirically: 0.999x the max
yields 2 solutions; smaller fractions give only the rising-branch solution).
"""

import numpy as np
import pytest

import nfw


def test_single_solution_round_trips_rho150():
    rho150 = 1e8
    Mpeak = 1e10
    sols = nfw.nfw_parameters_from_rho150_mpeak(rho150, Mpeak)
    assert len(sols) == 1
    s = sols[0]
    assert not np.isnan(s['c'])
    recovered = nfw.nfw_density(0.15, s['rho_s'], s['r_s'])
    assert recovered == pytest.approx(rho150, rel=1e-6)


def test_both_brentq_branches_round_trip_rho150():
    """Pick rho150 near (but below) the achievable max for this Mpeak, which
    is where nfw_parameters_from_rho150_mpeak's sign-change search finds two
    brackets -- one on each side of the rho150(c) peak -- and hits brentq
    twice."""
    Mpeak = 1e8
    # rho150_max for this Mpeak, from the unconstrained/never-achievable probe.
    probe = nfw.nfw_parameters_from_rho150_mpeak(1e-10, Mpeak)
    rho_max = probe[0]['rho150_max']
    rho150 = 0.999 * rho_max

    sols = nfw.nfw_parameters_from_rho150_mpeak(rho150, Mpeak)
    assert len(sols) == 2, "expected both brentq branches to be hit"

    concentrations = [s['c'] for s in sols]
    assert concentrations[0] != pytest.approx(concentrations[1])

    for s in sols:
        assert not np.isnan(s['c'])
        recovered = nfw.nfw_density(0.15, s['rho_s'], s['r_s'])
        assert recovered == pytest.approx(rho150, rel=1e-5)


def test_rho150_above_max_returns_nan_result():
    Mpeak = 1e7
    probe = nfw.nfw_parameters_from_rho150_mpeak(1e-10, Mpeak)
    rho_max = probe[0]['rho150_max']
    sols = nfw.nfw_parameters_from_rho150_mpeak(rho_max * 10, Mpeak)
    assert len(sols) == 1
    assert np.isnan(sols[0]['c'])


def test_nfw_mass_and_density_are_consistent_with_notebook_example():
    """Regression pin from the notebook's own worked example (cell 196)."""
    params = nfw.nfw_parameters_from_rho150_mpeak(1e8, 1e10)[0]
    rho_check = nfw.nfw_density(0.15, params['rho_s'], params['r_s'])
    M_check = nfw.nfw_mass(params['r_vir'], params['rho_s'], params['r_s'])
    assert rho_check / 1e8 == pytest.approx(1.0, rel=1e-6)
    assert M_check / 1e10 == pytest.approx(1.0, rel=1e-6)


def test_rho150_predicted_direct_at_z0_matches_nested_rho150_predicted():
    """At z=0, rho150_predicted_direct derives r_vir from Mpeak with the same
    Delta_vir/rho_crit formula as the nested `rho150_predicted` closure in
    nfw_parameters_from_rho150_mpeak, and evaluates the same NFW density at
    0.15 kpc -- so feeding it the (c, Mpeak) solution that closure found for
    a target rho150 reproduces that rho150 exactly."""
    rho150 = 1e8
    Mpeak = 1e10
    sol = nfw.nfw_parameters_from_rho150_mpeak(rho150, Mpeak, z=0)[0]
    direct = nfw.rho150_predicted_direct(sol['c'], Mpeak, z=0)
    assert direct == pytest.approx(rho150, rel=1e-6)


def test_rho150_predicted_direct_default_z_disagrees_with_z0_pin_the_ratio():
    """rho150_predicted_direct's default is z=5, not z=0 like every
    neighbouring routine in this module -- upstream's choice, reproduced
    as-is (see nfw.py's docstring). This does NOT endorse that default; it
    pins the ~19.65x discrepancy it produces at this test point so a future
    change to the default (e.g. "fixing" it to z=0) is caught as a behavior
    change rather than silently passing because nothing else calls this
    function."""
    rho150 = 1e8
    Mpeak = 1e10
    sol = nfw.nfw_parameters_from_rho150_mpeak(rho150, Mpeak, z=0)[0]
    direct_z0 = nfw.rho150_predicted_direct(sol['c'], Mpeak, z=0)
    direct_default = nfw.rho150_predicted_direct(sol['c'], Mpeak)
    assert direct_default != pytest.approx(direct_z0)
    assert direct_default / direct_z0 == pytest.approx(19.65297616718624, rel=1e-9)
