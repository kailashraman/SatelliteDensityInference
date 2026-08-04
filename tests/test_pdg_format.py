"""pdg_format.py tests.

The `fmt_pdg` cases below are taken directly from asymmetric-error numbers
already printed in drafts_temp/temp_part1.tex (read-only; not edited here),
which follow the same PDG significant-figure convention this module
implements: $\\beta=0.18^{+0.12}_{-0.12}$, $\\alpha_{100} = 7.0^{+0.6}_{-0.5}$,
$\\sigma_{\\rm int} = 0.11^{+0.04}_{-0.03}$, $24^{+9}_{-8}$ (LMC-analog
satellite count), and $3.7^{+1.4}_{-1.0}$ (UMa3/U1 velocity dispersion).
These are prose numbers, not necessarily produced by this exact function, but
they exercise the documented rounding rule (leading digit 1 or 2 keeps 2 sig
figs) across several exponents and are a useful independent check.
"""

import numpy as np
import pytest

import pdg_format as pdg


@pytest.mark.parametrize('val, err_lo, err_hi, expected', [
    (0.18, 0.12, 0.12, r'$0.18^{+0.12}_{-0.12}$'),
    (7.0, 0.5, 0.6, r'$7.0^{+0.6}_{-0.5}$'),
    (0.11, 0.03, 0.04, r'$0.11^{+0.04}_{-0.03}$'),
    (24, 8, 9, r'$24^{+9}_{-8}$'),
    (3.7, 1.0, 1.4, r'$3.7^{+1.4}_{-1.0}$'),
])
def test_fmt_pdg_matches_published_table_rows(val, err_lo, err_hi, expected):
    assert pdg.fmt_pdg(val, err_lo, err_hi) == expected


@pytest.mark.parametrize('err, expected', [
    (0.12, 2),   # leading digit 1 -> keep 2 sig figs
    (0.25, 2),   # leading digit 2 -> keep 2 sig figs
    (3.0, 0),    # leading digit 3, just past the leading<=2 boundary -> 1 sig fig (ndec=0 here)
    (0.35, 1),   # leading digit 3, same boundary at a different exponent -> 1 sig fig
    (0.6, 1),    # leading digit 6 -> 1 sig fig
    (9.0, 0),    # leading digit 9, exponent 0 -> round to units
    (150, -1),   # leading digit 1, exponent 2 -> 2 sig figs -> round to tens
])
def test_pdg_decimals_leading_digit_rule(err, expected):
    assert pdg.pdg_decimals(err) == expected


def test_pdg_decimals_non_finite_or_nonpositive_falls_back_to_2():
    assert pdg.pdg_decimals(0.0) == 2
    assert pdg.pdg_decimals(-1.0) == 2
    assert pdg.pdg_decimals(np.nan) == 2


def test_fmt_pdg_large_error_rounds_to_power_of_ten():
    assert pdg.fmt_pdg(1234, 150, 150) == r'$1230^{+150}_{-150}$'


def test_to_log10_J_autodetects_linear_vs_log_input():
    assert pdg.to_log10_J(1e18) == pytest.approx(18.0)
    assert pdg.to_log10_J(18.0) == pytest.approx(18.0)


def test_to_log10_J_nonfinite_or_zero_is_nan():
    assert np.isnan(pdg.to_log10_J(0))
    assert np.isnan(pdg.to_log10_J(np.nan))


def test_fmt_pdg_logJ_matches_fmt_pdg_on_log_values():
    q16, q50, q84 = 17.5, 18.0, 18.6
    expected = pdg.fmt_pdg(q50, q50 - q16, q84 - q50)
    assert pdg.fmt_pdg_logJ(q16, q50, q84) == expected


def test_fmt_pdg_logJ_nonfinite_median_returns_cdots():
    assert pdg.fmt_pdg_logJ(np.nan, np.nan, np.nan) == r'$\cdots$'
