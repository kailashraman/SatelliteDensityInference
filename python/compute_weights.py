# Script to compute and save halo weights
#
# Migrated from SatGen_Dwarf/python/compute_weights.py. Path/import/provenance
# edits only -- the version-dispatch logic (mass_floor, lmc_selection,
# sim_version, logMhalf_scatter, lmc_max_distance, use_kd, use_mcdaniel) and
# the SHMR list/order are unchanged; they set every downstream number.
#
# The `os.environ["PATH"] += ...texlive...` line from the source is dropped:
# this script never imports matplotlib or renders any text, so there is
# nothing here for a LaTeX-capable PATH to affect.
#
# RNG seeding: halo_weights.py draws each SHMR's intrinsic scatter
# (mstar_weights, joint_weights) from the global NumPy RNG, with no seed
# anywhere in that module (see docs/migration-notes.md, "SHMR weights are one
# unseeded realization"). This script now seeds that global RNG once, right
# after the version dispatch and before any draw, from
# zlib.crc32(f'{version}:{dsph_idx}') -- deterministic, and distinct per
# (version, dwarf) so different array tasks do not share a stream. `hash()` is
# not used: Python randomizes string hashes per process by default
# (PYTHONHASHSEED), which would make the seed itself irreproducible.
# mhalf_weights does not draw from the RNG at all and is unaffected.
#
# This makes *future* runs of this script reproducible; it cannot recover a
# draw made before the seed existed. Products in weights_gc/ that predate this
# change are a different, unrecoverable realization of mstar_weights and
# joint_weights -- see scripts/stamp_migrated.py's STOCHASTIC_CAVEAT, mirrored
# below in this script's own provenance record.
#
# Seeding on (version, dsph_idx) rather than on version alone is a deliberate
# modelling choice, not an implementation detail: each of the 43 dwarfs draws
# its own independent realization of the SHMR scatter over the same
# catalog-wide logMpeak. Seeding on version alone would instead share one
# realization across all 43 dwarfs for that version. Per-dwarf is the
# faithful choice here -- it reproduces the structure of the unseeded
# upstream, where 43 independent array tasks (one per dsph_idx) each drew
# fresh from the process-global RNG, rather than sharing a stream. This
# matters for any population-level statement that combines per-dwarf
# posteriors (e.g. an aggregate across all 43 dwarfs for one version): those
# posteriors are independent draws, not correlated through a shared
# realization, and a reader should not assume they share SHMR-scatter noise.
import h5py
import numpy as np
import logging
import zlib
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

dsph_idx = int(sys.argv[1])
version = sys.argv[2]

dwarf_names = obs.dwarf_names
if version == 'lvdb':
    use_kd = False
    use_mcdaniel = False
elif version == 'mcdaniel':
    use_kd = True
    use_mcdaniel = True
else:
    use_kd = True
    use_mcdaniel = False

save_dir = str(config.WEIGHTS_DIR / version) + '/'

dwarf = dwarf_names[dsph_idx]
print(dwarf)

###-----------------------###
if version == 'cut_m9':
    mass_floor = 9
elif version == 'cut_m8p5':
    mass_floor = 8.5
elif version == 'mass_floor_7':
    mass_floor = 7
else:
    mass_floor = 8

if version == 'lmc' or version == 'lmc_50':
    lmc_selection = True
else:
    lmc_selection = False

if version == 'lmc' or version == 'mhalf_scatter' or version == 'lmc_50' or version == 'lvdb' or version == 'mcdaniel':
    sim_version = 'Diemer'
else:
    sim_version = version

if version == 'mhalf_scatter':
    logMhalf_scatter = 0.1
else:
    logMhalf_scatter = 0.0

if version == 'lmc_50':
    lmc_max_distance = 50
else:
    lmc_max_distance = 100

# if version == 'Geha':
#     sim_version = 'Diemer'
# else:
#     sim_version = version

provenance_inputs = [config.h5_path(version)]
if not version in ['Symphony', 'MWest']:
    provenance_inputs.append(config.ADDITIONAL_DIR / f'{sim_version}_Mvir200.npz')
    provenance_inputs.append(config.MHALF_DIR / sim_version / f'{dwarf}.npz')
# Checked immediately after the version dispatch, before any heavy computation
# or RNG draw -- a bad input is otherwise found only after full walltime (and,
# for this script, after the RNG below has already been consumed).
provenance.assert_single_version(provenance_inputs, expected=version)

# See the module-docstring note on RNG seeding.
np.random.seed(zlib.crc32(f'{version}:{dsph_idx}'.encode()) & 0xffffffff)

if not version in ['Symphony', 'MWest']:
    logMpeak200 = np.log10(np.load(config.ADDITIONAL_DIR / f'{sim_version}_Mvir200.npz')['Mvir200'])
    with ResampleMstar(version = sim_version, mass_floor=mass_floor) as weights:
        logMpeak, z = weights.logMpeak, weights.z_in
        vpeak = weights.h5['tpeak_v_max'][()]

        Behroozi13 = weights.evolved_Mstar(logMstar_B13(logMpeak, z))
        Moster13 = weights.evolved_Mstar(logMstar_Moster13(logMpeak, z))
        RodriguezPuebla17 = weights.evolved_Mstar(logMstar_RP17(logMpeak, z))
        Fattahi18 = weights.evolved_Mstar(logMstar_Fattahi18(vpeak))
        Moster18 = weights.evolved_Mstar(logMstar_Moster18(logMpeak, z))
        Behroozi19 = weights.evolved_Mstar(logMstar_Behroozi19(logMpeak, z))
        Munshi21 = weights.evolved_Mstar(logMstar_Munshi21(logMpeak))
        Danieli23_const = weights.evolved_Mstar(logMstar_Danieli23(logMpeak, const=True))
        Danieli23_grow = weights.evolved_Mstar(logMstar_Danieli23(logMpeak, const=False))
        Kim24 = weights.evolved_Mstar(logMstar_Kim24(logMpeak200))

    ###-----------------------###

    logMstars = [Behroozi13, Moster13, RodriguezPuebla17, Fattahi18, Moster18, Behroozi19, Munshi21, Danieli23_const, Danieli23_grow, Kim24]
    SHMR_names = ['Behroozi13', 'Moster13', 'RodriguezPuebla17', 'Fattahi18', 'Moster18', 'Behroozi19', 'Munshi21', 'Danieli23_const', 'Danieli23_grow', 'Kim24']

    mstar_weights_list = []
    joint_weights_list = []

    for i in range(len(SHMR_names)):
        print(SHMR_names[i])
        with ResampleMstar(version = sim_version, mass_floor=mass_floor, cut='distance') as weights:
            mstar_weights_list.append(weights.from_logMstar_only(logMstars[i], dwarf, lmc_selection=lmc_selection, lmc_max_distance=lmc_max_distance, use_kd=use_kd, use_mcdaniel=use_mcdaniel))
            joint_weights_list.append(weights.from_logMstar(logMstars[i], dwarf, lmc_selection=lmc_selection, lmc_max_distance=lmc_max_distance, logMhalf_scatter=logMhalf_scatter, use_kd=use_kd, use_mcdaniel=use_mcdaniel))

    mstar_weights = np.array(mstar_weights_list)
    joint_weights = np.array(joint_weights_list)

with ResampleMstar(version = sim_version, mass_floor=mass_floor, cut='distance') as weights:
    mhalf_weights = weights.from_logMhalf(dwarf, lmc_selection=lmc_selection, lmc_max_distance=lmc_max_distance, logMhalf_scatter=logMhalf_scatter, use_kd=use_kd, use_mcdaniel=use_mcdaniel)
    if version == 'mhalf_scatter':
        mdyn_errani_weights = weights.from_logMdyn_errani(dwarf, lmc_selection=lmc_selection, lmc_max_distance=lmc_max_distance, use_kd=use_kd, use_mcdaniel=use_mcdaniel)

stochastic_kwargs = {}
if not version in ['Symphony', 'MWest']:
    # mstar_weights/joint_weights below draw from the RNG seeded above;
    # mhalf_weights (and, for mhalf_scatter, mdyn_errani_weights) do not.
    stochastic_kwargs = dict(
        stochastic_arrays=('mstar_weights', 'joint_weights'),
        stochastic_caveat=(
            'mstar_weights and joint_weights are draws from the global NumPy '
            'RNG (SHMR scatter in halo_weights.py), seeded above from '
            'zlib.crc32(f"{version}:{dsph_idx}") -- reproducible by rerunning '
            'this script with the same version and dsph_idx, but not '
            'recoverable for any product written before this seeding was '
            'added. mhalf_weights does not draw from the RNG and is '
            'unaffected.'))

# Name the infall -> z=0 stellar-mass convention in the record. Products made
# under the old constant +1.2 dex offset are numerically indistinguishable from
# these without it -- that is how a tree carrying two conventions (mass_floor_7
# offset, the other eight not) survived unnoticed and reached panel 6 of
# multipanel_systematics.pdf. Migrated products carry no such field, so absence
# means "unknown, predates the fix", not "agrees".
mstar_evolution = ('per_halo_tidal' if not version in ['Symphony', 'MWest']
                   else 'not_applicable')

record = provenance.stamp('python/compute_weights.py', version=version, argv=sys.argv[1:],
                          inputs=provenance_inputs, dwarf=dwarf,
                          mstar_evolution=mstar_evolution, **stochastic_kwargs)
if version == 'mhalf_scatter':
    provenance.savez(save_dir + dwarf, record, mstar_weights=mstar_weights, joint_weights=joint_weights, mhalf_weights=mhalf_weights, mdyn_errani_weights=mdyn_errani_weights)
elif version == 'Symphony' or version == 'MWest':
    provenance.savez(save_dir + dwarf, record, mhalf_weights=mhalf_weights)
else:
    provenance.savez(save_dir + dwarf, record, mstar_weights=mstar_weights, joint_weights=joint_weights, mhalf_weights=mhalf_weights)
