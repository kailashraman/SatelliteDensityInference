"""Save per-dwarf V_circ-r_max (and rho150-Mpeak) contours, both in the
galactocentric weighted-catalog sense and the DwarfJeansAnalysis prior-ladder
sense.

Migrated from SatGen_Dwarf/python/save_contours.py. save_contour and
_compute_contour_level now live in contour_io.py (see that module's
docstring for why); this is the entry point, taking its parameters as CLI
args exactly as upstream did, in the same order:

    python save_contours.py <dsph_idx> <version> <redshift>

Path/import/provenance edits only -- the grid construction, KDE/histogram
settings, contour levels, the weights indices used, and the redshift
handling are all character-identical to upstream.

Dropped from upstream: the matplotlib/PdfPages rcParams preamble (this
script never plots), `sys.path.append('../python/')` and the commented-out
J_test appends, and the texlive PATH mutation -- this script computes and
saves contours, it does not render text, so nothing here needs a
LaTeX-capable PATH. Also dropped: the module-level `logging.basicConfig()`
call and the `data_log.info(dwarf_names)` line it fed, a bare
`print(col_co.h)` left over from debugging the cosmology setup, and
`from halo_weights import *`, replaced by the explicit
`from halo_weights import ResampleMstar` above -- the only name from that
module this script actually uses, so the star import's namespace pollution
is gone.

The four hardcoded absolute SatGen_Dwarf paths become config-derived:
    weights_dir            -> config.WEIGHTS_DIR / version
    save_dir                -> config.PAPER_CONTOURS_DIR / 'galactocentric' / redshift / version
    rho150_mpeak_save_dir    -> config.PAPER_CONTOURS_DIR / 'rho150_mpeak' / version
    jeans_base                -> config.PAPER_CONTOURS_DIR
config.WEIGHTS_DIR and config.PAPER_CONTOURS_DIR already existed (used by
compute_weights.py and named in config.py respectively); no new constant was
added for these four.

`jeans_priors` is kept as upstream's literal 7-entry list rather than
config.DJA_PRIORS, which additionally lists 'jeffreys' -- a prior this script
never iterated upstream. Reusing DJA_PRIORS here would compute and save a
jeffreys contour tree with no upstream counterpart, which is a numerics/scope
change, not a path substitution.

Like upstream, this script does not resolve SatGen version aliases (no
`sim_version` step): `version` is used directly for ResampleMstar, the
weights/save directories, and the rho150 file. config.h5_path still resolves
aliases internally for the provenance fingerprint, but halo_weights.get_h5
(called by ResampleMstar) deliberately does not (see its docstring) -- so, as
upstream, this script only works for versions that are direct H5_REGISTRY
keys (e.g. Diemer, Zhao, mass_floor_7, cut_m9, cut_m8p5), not aliases like
'lmc'. This is inherited behavior, not introduced here.

Every output is written by contour_io.save_contour, which itself writes
through provenance.savez -- so each product is stamped in the same call that
writes it, not afterward with provenance.stamp_existing. A record is built
once per group (galactocentric, rho150_mpeak, or a given Jeans_<prior>) and
handed to every save_contour call in that group, including the group's
'_unweighted' member, so no product in a group is left unstamped and none is
stamped before it exists. The script writes THREE output trees:
galactocentric/<redshift>/<version> (V_circ-r_max, weighted by
mhalf/RP17/F18/M18/K24 and their joint variants, plus an unweighted
multi-level contour), Jeans_<prior>/ under the same root (one per entry in
jeans_priors, keyed only by dwarf -- no version/redshift in that path,
matching upstream), and rho150_mpeak/<version> (rho150-Mpeak, weighted by
mhalf/RP17/F18/M18/K24 and an unweighted multi-level contour).

The Jeans_<prior> stamp differs from the galactocentric/rho150_mpeak ones:
the Jeans contour is read entirely from DwarfJeansAnalysis's
<lvdb_key>/<prior>/derived.npz, not from any SatGen product, so its record
carries `version=None`, empty SatGen `inputs`, and `dja_prior`/`dja_files`
naming the actual derived.npz read (via Dwarf.jeans_dja_path) -- not the h5/
weights/rho150 inputs the galactocentric and rho150_mpeak records carry and
this branch never touches.

One thing that looks like a bug, left as upstream: the '_unweighted' calls
below pass `level=[0.68, 0.95, 0.995]` -- a list -- into save_contour, whose
`level` parameter is documented and used elsewhere as a single scalar
confidence level. `_compute_contour_level`'s `np.searchsorted(cumsum, level)`
still runs (searchsorted broadcasts over an array `level`), so this does not
raise, but `contour_hist`/`contour_kde` for the unweighted output are 3-element
arrays where every weighted call in this script produces a scalar. Not fixed
here; pinned by tests/test_contour_io.py.
"""
import sys
import os

import numpy as np
from astropy import units as u

import Jdata as obs
from halo_weights import ResampleMstar
import config
import provenance
from contour_io import save_contour

# set up colossus cosmology to match SatGen
from SatGen import profiles as pf
from colossus.cosmology import cosmology as co

co.addCosmology('SatGen', flat=True, H0=pf.cfg.h*100, Om0=pf.cfg.Om,
                Ob0=pf.cfg.Ob, sigma8=pf.cfg.s8, ns=pf.cfg.ns)
col_co = co.setCosmology('SatGen')

# ====================================================================================== #

dwarf_names = obs.dwarf_names

dsph_idx = int(sys.argv[1])
version = sys.argv[2]
redshift = sys.argv[3]

# Validated FIRST, before any directory is built from it: the only branch
# that actually reads `redshift` (below, choosing xvals/yvals) is an
# if/elif over exactly these two values with no else, so an unrecognized
# value used to fall through all the way to a NameError on an undefined
# xvals/yvals deep inside the ResampleMstar block. Catching it here, before
# save_dir is constructed and created, stops a typo from silently minting a
# 'galactocentric/<typo>/<version>' tree.
if redshift not in ('z0', 'infall'):
    raise ValueError(
        f"unknown redshift {redshift!r}; must be one of ('z0', 'infall')")

weights_dir = str(config.WEIGHTS_DIR / version) + '/'
save_dir = str(config.PAPER_CONTOURS_DIR / 'galactocentric' / redshift / version) + '/'
rho150_mpeak_save_dir = str(config.PAPER_CONTOURS_DIR / 'rho150_mpeak' / version) + '/'
jeans_priors = ['loguniform', 'satgen', 'satgen_box',
                'satgen_shmr_danieli23_const', 'satgen_shmr_fattahi18',
                'satgen_shmr_kim24', 'satgen_shmr_moster18']
jeans_base = str(config.PAPER_CONTOURS_DIR) + '/'
jeans_save_dirs = {p: jeans_base + 'Jeans_' + p + '/' for p in jeans_priors}
for d in jeans_save_dirs.values():
    os.makedirs(d, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(rho150_mpeak_save_dir, exist_ok=True)

dwarf = dwarf_names[dsph_idx]
print(dwarf)
weights_path = weights_dir + dwarf + '.npz'
weights_file = np.load(weights_path)
this_dwarf = obs.Dwarf(dwarf)

if version == 'cut_m9':
    mass_floor = 9
elif version == 'cut_m8p5':
    mass_floor = 8.5
elif version == 'mass_floor_7':
    mass_floor = 7
else:
    mass_floor = 8

classicals = ['Draco', 'Fornax', 'Sculptor', 'Sextans', 'Ursa Minor', 'Carina', 'Leo I', 'Leo II', 'Canes Venatici I', 'Sagittarius', 'SMC', 'LMC']
if dwarf in classicals:
    kde_bw = 0.025
else:
    kde_bw = 0.05

rho150_path = config.ADDITIONAL_DIR / f'{version}_rho150.npz'

provenance_inputs = [config.h5_path(version), weights_path, rho150_path]
provenance.assert_single_version(provenance_inputs, expected=version)

with ResampleMstar(version = version, mass_floor=mass_floor, cut='distance') as weights:
    rperi = weights.h5['rperi'][()]
    z = weights.z_in
    logMpeak = weights.logMpeak
    vmax_infall = weights.h5['tpeak_v_max'][()]
    vmax = weights.h5['v_max'][()]

    rmax = weights.h5['r_max'][()]
    rmax_infall = weights.h5['tpeak_r_max'][()]
    surviving = weights.h5['surviving'][()]

    if redshift == 'z0':
        xvals = (vmax * u.kpc/u.Gyr).to(u.km/u.s).value
        yvals = rmax
    elif redshift == 'infall':
        xvals = (vmax_infall * u.kpc/u.Gyr).to(u.km/u.s).value
        yvals = rmax_infall

    mask = weights.get_mask(dwarf)

    logrho150 = logrho150 = np.log10(np.load(rho150_path)['rho150'])

    mhalf_weights = weights_file['mhalf_weights']
    RP17_weights = weights_file['mstar_weights'][2]
    joint_weights_RP17 = weights_file['joint_weights'][2]
    F18_weights = weights_file['mstar_weights'][3]
    joint_weights_F18 = weights_file['joint_weights'][3]
    M18_weights = weights_file['mstar_weights'][4]
    joint_weights_M18 = weights_file['joint_weights'][4]
    K24_weights = weights_file['mstar_weights'][9]
    joint_weights_K24 = weights_file['joint_weights'][9]

    # One record for the whole galactocentric group -- every file in it
    # (including '_unweighted', written later once `mask` is applied) shares
    # the same dwarf/version/redshift, so it is built once and handed to
    # save_contour at each write instead of stamped afterward in a batch.
    galactocentric_record = provenance.stamp(
        'python/save_contours.py', version=version, argv=sys.argv[1:],
        inputs=provenance_inputs, dwarf=dwarf, redshift=redshift)

    save_contour(np.log10(xvals), np.log10(yvals), mhalf_weights, dwarf + '_mhalf', save_dir, galactocentric_record, level=0.68, kde_bw=kde_bw)
    save_contour(np.log10(xvals), np.log10(yvals), RP17_weights, dwarf + '_RP17', save_dir, galactocentric_record, level=0.68, kde_bw=kde_bw)
    save_contour(np.log10(xvals), np.log10(yvals), F18_weights, dwarf + '_F18', save_dir, galactocentric_record, level=0.68, kde_bw=kde_bw)
    save_contour(np.log10(xvals), np.log10(yvals), M18_weights, dwarf + '_M18', save_dir, galactocentric_record, level=0.68, kde_bw=kde_bw)
    save_contour(np.log10(xvals), np.log10(yvals), K24_weights, dwarf + '_K24', save_dir, galactocentric_record, level=0.68, kde_bw=kde_bw)

    save_contour(np.log10(xvals), np.log10(yvals), joint_weights_RP17, dwarf + '_joint_RP17', save_dir, galactocentric_record, level=0.68, kde_bw=kde_bw)
    save_contour(np.log10(xvals), np.log10(yvals), joint_weights_F18, dwarf + '_joint_F18', save_dir, galactocentric_record, level=0.68, kde_bw=kde_bw)
    save_contour(np.log10(xvals), np.log10(yvals), joint_weights_M18, dwarf + '_joint_M18', save_dir, galactocentric_record, level=0.68, kde_bw=kde_bw)
    save_contour(np.log10(xvals), np.log10(yvals), joint_weights_K24, dwarf + '_joint_K24', save_dir, galactocentric_record, level=0.68, kde_bw=kde_bw)

    for prior, jeans_save_dir in jeans_save_dirs.items():
        this_dwarf.get_Jeans_results(dja_prior=prior)
        jeans_rmax = this_dwarf.jeans_rmax
        jeans_vmax = this_dwarf.jeans_vmax
        jeans_weights = this_dwarf.jeans_weights
        if len(jeans_weights) == 0:
            print(f'Jeans {prior}: no posterior samples for {dwarf}, skipping')
            continue
        # The Jeans contour comes entirely from DwarfJeansAnalysis's
        # derived.npz for this dwarf/prior -- it reads none of
        # provenance_inputs (the SatGen h5/weights/rho150), so those are not
        # named as inputs here, and version is None rather than the SatGen
        # `version` this script otherwise threads through. dja_files
        # fingerprints the derived.npz Dwarf.get_Jeans_results actually
        # loaded (exposed via jeans_dja_path), not just the prior/directory.
        jeans_record = provenance.stamp(
            'python/save_contours.py', version=None, argv=sys.argv[1:],
            inputs=(), dwarf=dwarf, dja_prior=prior,
            dja_files=(this_dwarf.jeans_dja_path,) if this_dwarf.jeans_dja_path else ())
        save_contour(np.log10(jeans_vmax), np.log10(jeans_rmax), jeans_weights, dwarf + '_jeans', jeans_save_dir, jeans_record, level=0.68, kde_bw=kde_bw)

    if redshift == 'z0':
        xvals = (vmax * u.kpc/u.Gyr).to(u.km/u.s).value[mask]
        yvals = rmax[mask]
    elif redshift == 'infall':
        xvals = (vmax_infall * u.kpc/u.Gyr).to(u.km/u.s).value[mask]
        yvals = rmax_infall[mask]
    save_contour(np.log10(xvals), np.log10(yvals), np.ones_like(xvals), dwarf + '_unweighted', save_dir, galactocentric_record, level=[0.68, 0.95, 0.995], kde_bw=kde_bw)


    if dwarf in classicals:
        kde_bw = 0.05
    else:
        kde_bw = 0.1

    # One record for the rho150_mpeak group, same reasoning as
    # galactocentric_record above (no redshift field here -- this tree is
    # not keyed by it).
    rho150_mpeak_record = provenance.stamp(
        'python/save_contours.py', version=version, argv=sys.argv[1:],
        inputs=provenance_inputs, dwarf=dwarf)

    xvals = 10**logrho150
    yvals = 10**logMpeak
    save_contour(np.log10(xvals), np.log10(yvals), mhalf_weights, dwarf + '_mhalf', rho150_mpeak_save_dir, rho150_mpeak_record, level=0.68, kde_bw=kde_bw)
    save_contour(np.log10(xvals), np.log10(yvals), RP17_weights, dwarf + '_RP17', rho150_mpeak_save_dir, rho150_mpeak_record, level=0.68, kde_bw=kde_bw)
    save_contour(np.log10(xvals), np.log10(yvals), F18_weights, dwarf + '_F18', rho150_mpeak_save_dir, rho150_mpeak_record, level=0.68, kde_bw=kde_bw)
    save_contour(np.log10(xvals), np.log10(yvals), M18_weights, dwarf + '_M18', rho150_mpeak_save_dir, rho150_mpeak_record, level=0.68, kde_bw=kde_bw)
    save_contour(np.log10(xvals), np.log10(yvals), K24_weights, dwarf + '_K24', rho150_mpeak_save_dir, rho150_mpeak_record, level=0.68, kde_bw=kde_bw)

    kde_bw = 0.1

    xvals = 10**logrho150[mask]
    yvals = 10**logMpeak[mask]
    save_contour(np.log10(xvals), np.log10(yvals), np.ones_like(xvals), dwarf + '_unweighted', rho150_mpeak_save_dir, rho150_mpeak_record, level=[0.68, 0.95, 0.995], kde_bw=kde_bw)
