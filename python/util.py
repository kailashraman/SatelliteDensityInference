import os
import numpy as np
import argparse

import vegas

# Dropped on migration: a sys.path.append into an absolute path in another
# user's home directory, followed by `import Jdata as obs`. The name `obs`
# was never referenced here, and no module in this repository may read from
# outside it. Removing the import also breaks a needless Jdata <-> util cycle.

from astropy import units as u
from astropy import constants as const

import scipy.interpolate as interp


# ================================================================== #

def integrator(fnc, interval):
    @vegas.lbatchintegrand
    def integrand(x):
        if len(x.shape) > 1:
            xx = []
            for i in range(x.shape[1]):
                xx.append(x[:,i])
        else:
            xx = x[:,0]
        return fnc(xx)
    integ = vegas.Integrator(interval)
    integ(integrand, nitn = 10, neval = 1e4)
    result = integ(integrand, nitn = 10, neval = 1e4)
    return result.mean


def jfactor(rho, L, theta, d):
    R = np.abs(L - d)
    z = L * theta

    return rho(np.sqrt(R**2 + z**2)) ** 2 * 2 * np.pi * np.sin(theta)
    # return rho(R, z) ** 2 * 2 * np.pi * np.sin(theta)

# ================================================================== #

def quantile(values, weights=None, quantiles=0.5):
    # https://stackoverflow.com/a/73905572
    m = ~np.isnan(values)
    if not np.any(m):
        return np.full(np.shape(quantiles), np.nan)
    v = values[m]
    if weights is None:
        weights = np.ones_like(values)
    wgt = weights[m]
    i = np.argsort(v)
    c = np.cumsum(wgt[i])
    q = np.clip(np.searchsorted(c, quantiles * c[-1]), 0, len(c) - 2)
    return np.where(c[q]/c[-1] == quantiles, 0.5 * (v[i[q]] + v[i[q+1]]), v[i[q]])
