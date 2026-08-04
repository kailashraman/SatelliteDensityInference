"""NFW halo-parameter helpers, lifted from
../SatGen_Dwarf/jupyter/PaperPlots.ipynb cells 196 and 197.

Scope decision (flagged, not silently made): the function list for this
module names `f_nfw` and `rho150_predicted` as if they were standalone,
importable functions, but in the notebook neither is -- both are defined
*nested inside* `nfw_parameters_from_rho150_mpeak` (cell 196):

- `f_nfw(c)` is a pure function of its argument (no closure over anything in
  the enclosing scope), so it is hoisted to module scope here verbatim --
  its body is byte-for-byte unchanged, only its indentation moved. The
  nested `def f_nfw` inside `nfw_parameters_from_rho150_mpeak` is removed and
  replaced by a call to this module-level version, which is a pure
  deduplication with no numeric effect.
- `rho150_predicted(log_c)` is NOT pure -- it closes over `r_vir` and `Mpeak`
  from `nfw_parameters_from_rho150_mpeak`'s enclosing scope. Hoisting it to
  module scope would require changing its signature (passing `r_vir` and
  `Mpeak` explicitly), which is a structural change beyond verbatim lifting.
  It is left nested, exactly as in the notebook, so
  `nfw_parameters_from_rho150_mpeak` is otherwise untouched.

Cell 197 does not define a function named `rho150_predicted` either; the only
top-level, non-closure function there is `rho150_predicted_direct(c, Mpeak,
h=0.7, Omega_m=0.3, z=5)` -- a self-contained recomputation of the same
quantity (it derives its own r_vir from Mpeak rather than taking one) used
only by that cell's diagnostic plot. Note its `z=5` default differs from the
`z=0` default used everywhere else in this module; that is upstream's choice,
reproduced as-is. It is included here under its notebook name,
`rho150_predicted_direct`, since that is the function actually present at
module scope -- not renamed to match the function list's `rho150_predicted`.

Which of `nfw_parameters_from_rho150_mpeak`'s and `rho150_predicted_direct`'s
keyword arguments actually reach the computation, upstream-faithful and left
as-is (verified, not changed): `h` never reaches it -- `rho_crit_z` is
always called as `rho_crit_z(z)`, so its own `h` default (0.7) is what is
used regardless of the `h` passed in here, and `h=1.0` vs `h=0.7` gives
byte-identical `r_vir`. `Omega_m` IS live for `Delta_vir` (it sets `x` via
`omega_m(z, omega_m0=Omega_m)`), but not for `rho_crit`, since `rho_crit_z`
takes no `Omega_m` argument and always uses its own default (0.3). Varying
`h` here explores nothing; varying `Omega_m` only moves `Delta_vir`, not
`rho_crit`.
"""

import numpy as np
from scipy.optimize import brentq
import warnings


def E2(z, Omega_m=0.3):
    return Omega_m * (1+z)**3 + (1 - Omega_m)

def rho_crit_z(z, h=0.7, Omega_m=0.3):
    rho_crit_0 = 277.5 * h**2  # M_sun/kpc^3
    return rho_crit_0 * E2(z, Omega_m)

def omega_m(z, omega_m0=0.3, omega_lambda0=0.7, omega_r0=0.0):
    E2 = omega_r0 * (1 + z)**4 + omega_m0 * (1 + z)**3 + omega_lambda0
    return omega_m0 * (1 + z)**3 / E2

def f_nfw(c):
    return np.log1p(c) - c / (1 + c)

def nfw_parameters_from_rho150_mpeak(rho150, Mpeak, z=0, h=0.7, Omega_m=0.3,
                                      log_c_bounds=(-1, 3), verbose=False):
    # rho_crit = 277.5 * h**2
    rho_crit = rho_crit_z(z)
    # print(rho_crit)
    x = omega_m(z, omega_m0=Omega_m) - 1
    Delta_vir = 18 * np.pi**2 + 82 * x - 39 * x**2

    nan_result = {
        'rho_s': np.nan, 'r_s': np.nan, 'c': np.nan,
        'r_vir': np.nan, 'M_vir': np.nan, 'Delta_vir': Delta_vir,
        'rho150_max': np.nan,
    }

    r_vir = (3 * Mpeak / (4 * np.pi * Delta_vir * rho_crit)) ** (1 / 3)

    def rho150_predicted(log_c):
        c = 10**log_c
        r_s = r_vir / c
        rho_s = Mpeak / (4 * np.pi * r_s**3 * f_nfw(c))
        x_150 = 0.15 / r_s
        return rho_s / (x_150 * (1 + x_150)**2)

    def equation_in_log_c(log_c):
        return np.log10(rho150_predicted(log_c)) - np.log10(rho150)

    log_c_min, log_c_max = log_c_bounds
    log_c_dense = np.linspace(log_c_min, log_c_max, 2000)
    rho_curve = np.array([rho150_predicted(lc) for lc in log_c_dense])
    rho_max_achievable = np.max(rho_curve)

    nan_result['rho150_max'] = rho_max_achievable

    if rho150 > rho_max_achievable:
        if verbose:
            warnings.warn(
                f"No physical NFW solution for rho150={rho150:.3e}, Mpeak={Mpeak:.3e}. "
                f"Max achievable rho(150pc) = {rho_max_achievable:.3e}.",
                stacklevel=2,
            )
        return [nan_result,]

    residuals = np.log10(rho_curve) - np.log10(rho150)
    sign_changes = np.where(np.diff(np.sign(residuals)))[0]

    if len(sign_changes) == 0:
        if verbose:
            warnings.warn(
                f"Unexpected: no sign change found for rho150={rho150:.3e}, "
                f"Mpeak={Mpeak:.3e}.",
                stacklevel=2,
            )
        return [nan_result,]

    solutions = []
    for idx in sign_changes:
        bracket = (log_c_dense[idx], log_c_dense[idx + 1])
        try:
            log_c_sol = brentq(equation_in_log_c, bracket[0], bracket[1],
                               xtol=1e-12, rtol=1e-12, maxiter=300)
            c = 10**log_c_sol
            r_s = r_vir / c
            rho_s = Mpeak / (4 * np.pi * r_s**3 * f_nfw(c))
            solutions.append({
                'rho_s': rho_s,
                'r_s': r_s,
                'c': c,
                'r_vir': r_vir,
                'M_vir': Mpeak,
                'Delta_vir': Delta_vir,
                'rho150_max': rho_max_achievable,
            })
        except Exception as e:
            if verbose:
                warnings.warn(f"brentq failed at bracket near log_c={log_c_dense[idx]:.3f}: {e}",
                              stacklevel=2)

    return solutions  # list of 1 or 2 dicts; 2 when rho150 < rho_max_achievable


def nfw_density(r, rho_s, r_s):
    x = r / r_s
    return rho_s / (x * (1 + x)**2)


def nfw_mass(r, rho_s, r_s):
    x = r / r_s
    return 4 * np.pi * rho_s * r_s**3 * (np.log1p(x) - x / (1 + x))


def rho150_predicted_direct(c, Mpeak, h=0.7, Omega_m=0.3, z=5):
    rho_crit = rho_crit_z(z)
    x = omega_m(z, omega_m0=Omega_m) - 1
    Delta_vir = 18 * np.pi**2 + 82 * x - 39 * x**2
    r_vir = (3 * Mpeak / (4 * np.pi * Delta_vir * rho_crit)) ** (1/3)
    r_s = r_vir / c
    f_c = np.log1p(c) - c / (1 + c)
    rho_s = Mpeak / (4 * np.pi * r_s**3 * f_c)
    x_150 = 0.15 / r_s
    return rho_s / (x_150 * (1 + x_150)**2)
