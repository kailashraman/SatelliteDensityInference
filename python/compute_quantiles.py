# script to compute and save observable quantiles
#
# Migrated from SatGen_Dwarf/python/compute_quantiles.py. Path edits (save_dir,
# the J-factor path, weights_dir, mhalf_folder -> config.*) plus one
# correctness fix:
#
# In the Symphony/MWest branch, upstream reads
#     Mhalf = 10**(sats['logMhalf'][dwarf][()])
# `sats` is never defined or imported anywhere in this module or in anything
# it star-imports (confirmed by grepping the source tree) -- this is a
# NameError, and the branch cannot have run as written. `halo_weights.py`
# reads the same quantity for these two versions (ResampleMstar.from_logMhalf)
# as `self.h5['logMhalf'][name][mask]`, i.e. off the open h5 handle. Here
# there is no mask (this branch wants the full per-halo array, matching how
# Mhalf is loaded in the non-Symphony/MWest branch below), so the corrected
# read is `weights.h5['logMhalf'][dwarf][()]`. Because the unfixed line raises
# before any quantile is computed, any published Symphony/MWest quantiles
# must predate this state of the source -- they cannot have been produced by
# the script as found here.
#
# This fix corrects the branch's *shape* -- it no longer raises NameError --
# it does not make it runnable here: the local Symphony/MWest h5 files
# (data/additional/{SymphonyMilkyWay,MWest}.h5, rebuilt by make_h5.py sims)
# carry no `logMhalf` group at all, so `weights.h5['logMhalf'][dwarf]` still
# raises, now a KeyError instead of a NameError. See docs/migration-notes.md,
# "SymphonyMilkyWay_old.h5 mixed two simulation suites": the rebuild dropped
# that group deliberately, because no paper figure needs it.
#
# The `os.environ["PATH"] += ...texlive...` line from the source is dropped,
# matching compute_weights.py: this script never imports matplotlib or renders
# any text, so there is nothing here for a LaTeX-capable PATH to affect. The
# local `quantile()` function (identical to `util.quantile`) is pre-existing
# and left untouched: it was not in scope for this port. `quantile()`
# duplicates `util.quantile`; a later step is expected to extract shared
# weighted-statistics helpers.
import h5py
import numpy as np
import logging
from scipy import stats
from scipy import interpolate as interp
from scipy.special import erf
from KDEpy import FFTKDE as KDE

import sys
import Jdata as obs
from SatGen.profiles import Green, Dekel, NFW

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord

import os

import pandas as pd

# import stellar mass halo mass relations
from halo_weights import *
import config
import provenance

# set up colossus cosmology to match SatGen
from SatGen import profiles as pf
from colossus.cosmology import cosmology as co
from colossus.halo import profile_nfw

co.addCosmology('SatGen', flat=True, H0=pf.cfg.h*100, Om0=pf.cfg.Om,
                Ob0=pf.cfg.Ob, sigma8=pf.cfg.s8, ns=pf.cfg.ns)
col_co = co.setCosmology('SatGen')

###-----------------------###

dwarf_names = obs.dwarf_names
dsph_idx = int(sys.argv[1])
version = sys.argv[2]
# if version == 'Geha':
#     dwarf_names = obs.geha_names
#     use_geha = True
# else:
#     use_geha = False
dwarf = dwarf_names[dsph_idx]
print(dwarf)

save_dir = config.PAPER_QUANTILES_DIR / 'galactocentric' / version

# if version == 'Geha':
#     sim_version = 'Diemer'
# else:
#     sim_version = version

if version == 'lmc' or version == 'lmc_50':
    lmc_selection = True
else:
    lmc_selection = False

if version == 'lmc' or version == 'mhalf_scatter' or version == 'lmc_50' or version == 'lvdb' or version == 'mcdaniel':
    sim_version = 'Diemer'
else:
    sim_version = version

J_factor_file = config.PAPER_JS_DIR / sim_version / dwarf / 'halo_Js.npz'
has_J = os.path.exists(J_factor_file)
if has_J:
    J_arr = np.load(J_factor_file)
    green_Js = J_arr['green_Js']
    theta95 = J_arr['theta95']        # 95% J-factor containment angle (radians)
else:
    print('No J-factor file at ' + str(J_factor_file) + ' -- skipping J quantiles')

print(dwarf)
weights_dir = config.WEIGHTS_DIR / version
SHMR_names = ['Behroozi13', 'Moster13', 'RodriguezPuebla17', 'Fattahi18', 'Moster18', 'Behroozi19', 'Munshi21', 'Danieli23_const', 'Danieli23_grow', 'Kim24']
weights_file = np.load(weights_dir / (dwarf + '.npz'))
weights_list = []
weights_list.append(weights_file['mhalf_weights'])
if not version in ['Symphony', 'MWest']:
    for j, SHMR in enumerate(SHMR_names):
        weights_list.append(weights_file['mstar_weights'][j])
    for j, SHMR in enumerate(SHMR_names):
        weights_list.append(weights_file['joint_weights'][j])
    if version == 'mhalf_scatter':
        weights_list.append(weights_file['mdyn_errani_weights'])

print('opened weights file')

# a J-factor file from an older/smaller run of this version would misalign halo
# indices against the weights, silently poisoning every J quantile
if has_J and len(green_Js) != len(weights_list[0]):
    raise ValueError(f'{J_factor_file} has {len(green_Js)} halos but the weights have '
                     f'{len(weights_list[0])} -- rerun Jdwarf.py for version {sim_version}')

mhalf_folder = config.MHALF_DIR

provenance_inputs = [config.h5_path(version), weights_dir / (dwarf + '.npz')]
if not version in ['Symphony', 'MWest']:
    provenance_inputs.append(config.ADDITIONAL_DIR / f'{sim_version}_rho150.npz')
    provenance_inputs.append(mhalf_folder / sim_version / f'{dwarf}.npz')
if has_J:
    provenance_inputs.append(J_factor_file)
# Checked immediately after the version dispatch (and the small npz reads
# above), before the heavy per-halo catalog read below -- a bad input is
# otherwise found only after full walltime.
provenance.assert_single_version(provenance_inputs, expected=version)

# load sim data
with ResampleMstar(version = sim_version, mass_floor=8, cut='distance') as weights:
    if not version in ['Symphony', 'MWest']:
        rperi = weights.h5['rperi'][()]
        z = weights.z_in
        logMpeak = weights.logMpeak
        vmax_infall = weights.h5['tpeak_v_max'][()]
        vmax = weights.h5['v_max'][()]
        rmax_infall = weights.h5['tpeak_r_max'][()]
        rmax = weights.h5['r_max'][()]

        Greens = weights.h5['Green_params'][()]
        Mvir = weights.h5['virial_mass'][()]

        concentration = Greens[:,1]
        rho150 = np.load(config.ADDITIONAL_DIR / f'{sim_version}_rho150.npz')['rho150']
        Mhalf = 10**(np.load(mhalf_folder / sim_version / f'{dwarf}.npz')['logMhalf'])
        test_dwarf = obs.Dwarf(dwarf)
        dispersion = np.sqrt(const.G * Mhalf*u.Msun / (test_dwarf.rhalf.to(u.kpc) * 3)).to(u.km / u.s).value

    else:
        logMpeak = weights.logMpeak
        Mvir = weights.h5['mvir'][()]
        rho150 = weights.h5['rho150'][()]
        rmax = weights.h5['rmax'][()]
        vmax = weights.h5['vmax'][()]
        cV = weights.h5['cV'][()]
        Mhalf = 10**(weights.h5['logMhalf'][dwarf][()])
        test_dwarf = obs.Dwarf(dwarf)
        dispersion = np.sqrt(const.G * Mhalf*u.Msun / (test_dwarf.rhalf.to(u.kpc) * 3)).to(u.km / u.s).value

# a catalog-derived array from a different-sized run of this version would
# misalign halo indices against the weights, silently poisoning every quantile
if len(rho150) != len(weights_list[0]):
    raise ValueError(f'{sim_version} catalog (rho150) has {len(rho150)} halos but the '
                     f'weights in {weights_dir / (dwarf + ".npz")} have '
                     f'{len(weights_list[0])} -- rerun compute_weights.py for version {version}')

# Mhalf comes from a separate per-dwarf file (mhalf/<version>/<dwarf>.npz, or
# the h5's own logMhalf group for Symphony/MWest) rather than the shared
# catalog arrays above, so it is the most likely of the lot to drift out of
# sync with the weights.
if len(Mhalf) != len(weights_list[0]):
    raise ValueError(f'Mhalf for {dwarf} ({sim_version}) has {len(Mhalf)} halos but the '
                     f'weights in {weights_dir / (dwarf + ".npz")} have '
                     f'{len(weights_list[0])} -- rerun compute_mhalf.py for version {sim_version}')

print('Read sim data')

# np.savez(f'../data/additional/{version}_rho30', rho30=rho30)
# np.savez(f'../data/additional/{version}_M30', M30=M30)
# np.savez(f'../data/additional/{version}_rho150', rho150=rho150)

if not version in ['Symphony', 'MWest']:
    # physical concentration parameter
    cV_infall = 2 * (vmax_infall * u.kpc / u.Gyr / (rmax_infall * u.kpc * (col_co.H0 * u.km / u.s / u.Mpc).to(1 / u.Gyr))) ** 2
    # rho150
    rs_infall = (0.46241029979236 * rmax_infall * u.kpc).to(u.km)
    rhos_infall = 1.7212585601570 * (((vmax_infall * u.kpc / u.Gyr).to(u.m / u.s) / (rmax_infall * u.kpc).to(u.m)) **2 / const.G)
    r_infall = (150 * u.pc).to(u.km)
    rho150_infall = rhos_infall / (r_infall / rs_infall * (1 + r_infall / rs_infall) ** 2) * const.c ** 2
    rho150_infall = rho150_infall.to(u.GeV / u.cm ** 3).value

    # physical concentration parameter
    cV = 2 * (vmax * u.kpc / u.Gyr / (rmax * u.kpc * (col_co.H0 * u.km / u.s / u.Mpc).to(1 / u.Gyr))) ** 2

###-----------------------###

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

###-----------------------###

n_lists = len(weights_list)

rho150_infall_quantiles = np.zeros((3, n_lists))
cV_infall_quantiles = np.zeros((3, n_lists))

rho150_quantiles = np.zeros((3, n_lists))
cV_quantiles = np.zeros((3, n_lists))

J_quantiles = np.zeros((3, n_lists))
theta95_quantiles = np.zeros((3, n_lists))

Mvir_quantiles = np.zeros((3, n_lists))
Mpeak_quantiles = np.zeros((3, n_lists))
Mloss_quantiles = np.zeros((3, n_lists))

rmax_quantiles = np.zeros((3, n_lists))
vmax_quantiles = np.zeros((3, n_lists))

Mhalf_quantiles = np.zeros((3, n_lists))

dispersion_quantiles = np.zeros((3, n_lists))

for j in range(n_lists):

    if not version in ['Symphony', 'MWest']:
        rho150_infall_quantiles[:,j] = quantile(rho150_infall, weights_list[j], quantiles=np.array([0.16, 0.5, 0.84]))
        cV_infall_quantiles[:,j] = quantile(cV_infall, weights_list[j], quantiles=np.array([0.16, 0.5, 0.84]))


    rho150_quantiles[:,j] = quantile(rho150, weights_list[j], quantiles=np.array([0.16, 0.5, 0.84]))
    cV_quantiles[:,j] = quantile(cV, weights_list[j], quantiles=np.array([0.16, 0.5, 0.84]))

    if has_J:
        J_quantiles[:,j] = quantile(green_Js, weights_list[j], quantiles=np.array([0.16, 0.5, 0.84]))
        theta95_quantiles[:,j] = quantile(theta95, weights_list[j], quantiles=np.array([0.16, 0.5, 0.84]))

    Mvir_quantiles[:,j] = quantile(Mvir, weights_list[j], quantiles=np.array([0.16, 0.5, 0.84]))
    Mpeak_quantiles[:,j] = quantile(10**logMpeak, weights_list[j], quantiles=np.array([0.16, 0.5, 0.84]))

    Mloss_quantiles[:,j] = quantile(Mvir/10**logMpeak, weights_list[j], quantiles=np.array([0.16, 0.5, 0.84]))


    rmax_quantiles[:,j] = quantile(rmax, weights_list[j], quantiles=np.array([0.16, 0.5, 0.84]))
    vmax_quantiles[:,j] = quantile(vmax, weights_list[j], quantiles=np.array([0.16, 0.5, 0.84]))
    Mhalf_quantiles[:,j] = quantile(Mhalf, weights_list[j], quantiles=np.array([0.16, 0.5, 0.84]))

    dispersion_quantiles[:,j] = quantile(dispersion, weights_list[j], quantiles=np.array([0.16, 0.5, 0.84]))



record = provenance.stamp('python/compute_quantiles.py', version=version, argv=sys.argv[1:],
                          inputs=provenance_inputs, dwarf=dwarf)
if has_J:
    provenance.savez(save_dir / dwarf, record, rho150_infall_quantiles=rho150_infall_quantiles, cV_infall_quantiles=cV_infall_quantiles, rho150_quantiles=rho150_quantiles, cV_quantiles=cV_quantiles, J_quantiles=J_quantiles, theta95_quantiles=theta95_quantiles, Mvir_quantiles=Mvir_quantiles, Mpeak_quantiles=Mpeak_quantiles, Mloss_quantiles=Mloss_quantiles, rmax_quantiles=rmax_quantiles, vmax_quantiles=vmax_quantiles, Mhalf_quantiles=Mhalf_quantiles, dispersion_quantiles=dispersion_quantiles)
else:
    provenance.savez(save_dir / dwarf, record, rho150_infall_quantiles=rho150_infall_quantiles, cV_infall_quantiles=cV_infall_quantiles, rho150_quantiles=rho150_quantiles, cV_quantiles=cV_quantiles, Mvir_quantiles=Mvir_quantiles, Mpeak_quantiles=Mpeak_quantiles, Mloss_quantiles=Mloss_quantiles, rmax_quantiles=rmax_quantiles, vmax_quantiles=vmax_quantiles, Mhalf_quantiles=Mhalf_quantiles, dispersion_quantiles=dispersion_quantiles)
