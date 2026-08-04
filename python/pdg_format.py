"""PDG-style significant-figure formatting for LaTeX table rows, lifted from
../SatGen_Dwarf/jupyter/PaperPlots.ipynb cells 105 and 106.

`pdg_decimals` and `fmt_pdg` are defined in both cells. They are NOT
byte-identical: cell 105 has the original, verbosely-docstringed
implementation; cell 106 rewrites the same two functions as a terser,
one-line-body version with shorter docstrings. Both were checked
statement-by-statement and compute the same thing (`pdg_decimals`: same
`if leading <= 2: ndec = -exponent + 1 else ndec = -exponent` logic, just
written as a conditional expression in cell 106; `fmt_pdg`: identical
branches, condensed). Cell 106's terser forms are kept here since that cell
is the later, more complete one -- it is also the only place `to_log10_J`
and `fmt_pdg_logJ` are defined at all.
"""

import numpy as np
import math


def pdg_decimals(err):
    """Number of decimal places for an error following PDG rules (can be negative)."""
    if err <= 0 or not np.isfinite(err):
        return 2  # fallback
    exponent = math.floor(math.log10(err))
    leading = int(math.floor(err / 10**exponent))
    return (-exponent + 1) if leading <= 2 else -exponent


def fmt_pdg(val, err_lo, err_hi):
    """Format  $val^{+err_hi}_{-err_lo}$  using PDG sig-fig rules."""
    max_err = max(abs(err_lo), abs(err_hi))
    ndec = pdg_decimals(max_err)
    if ndec >= 0:
        return (
            f"${val:.{ndec}f}"
            f"^{{+{abs(err_hi):.{ndec}f}}}"
            f"_{{-{abs(err_lo):.{ndec}f}}}$"
        )
    factor = 10**(-ndec)
    v_r  = int(factor * round(val / factor))
    eh_r = int(factor * round(abs(err_hi) / factor))
    el_r = int(factor * round(abs(err_lo) / factor))
    return f"${v_r}^{{+{eh_r}}}_{{-{el_r}}}$"


def to_log10_J(q):
    """Return log10(J), auto-detecting linear vs already-log input via a
    single threshold: values with |q| < 100 are assumed already log10(J)
    and returned as-is; everything else is assumed linear J and log10'd."""
    q = float(q)
    if not np.isfinite(q) or q == 0:
        return np.nan
    return q if abs(q) < 100 else math.log10(q)


def fmt_pdg_logJ(q16, q50, q84):
    """Format log10(J) with asymmetric errors (16/50/84 quantiles, linear or log)."""
    l16, l50, l84 = to_log10_J(q16), to_log10_J(q50), to_log10_J(q84)
    if not np.isfinite(l50):
        return r'$\cdots$'
    return fmt_pdg(l50, l50 - l16, l84 - l50)
