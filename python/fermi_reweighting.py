"""Convert per-dwarf Fermi-LAT SED likelihoods into DM (mass, sigmav) TS grids
for a set of J-factor "variants" (mhalf/F18/K24 SatGen weights, DJA Jeans
under several priors, McDaniel literature values, ...).

Migrated from SatGen_Dwarf/python/fermi_reweighting.py. Path/import/provenance
edits only -- the TS computation (convert_sed/convert_sed_batch, imported
unchanged from fermi_funcs.py), the mass_vec/sigmav_vec grids, the
WEIGHT_LIST-equivalent idcs/weight_names arrays and their ORDER, the
satgen_priors list, the F18/K24 Monte Carlo draw (including the RNG seeding
convention: default_rng(abs(hash(name)) % (2**32))), and every J-factor
extraction are all character-identical to upstream.

Usage, unchanged from upstream:
    python fermi_reweighting.py <dsph_idx>
    python fermi_reweighting.py <dsph_idx> --variants name1,name2,...
    python fermi_reweighting.py <dsph_idx> --legacy-version {legacy,update}

--legacy-version keeps upstream's default ('legacy'): the drafts cite
Circiello:2026inp and the production launcher (fermi_reweighting_update.sh)
always passes --legacy-version update explicitly, but the ported module's own
default is left untouched -- CLI defaults are a behavioural choice this
migration does not silently flip. Which SED tree and which results tree a run
used is therefore always explicit in its argv, never implied by a default
buried in a path string.

Dropped from upstream: `sys.path.append`-equivalent path setup (none was
present; imports already resolve via python/ on the path, as with every other
migrated script) and the unused top-level imports upstream carried alongside
`from fermi_funcs import *` -- matplotlib.pyplot, pandas, copy, scipy.interpolate,
iminuit.Minuit. None of the five is referenced directly in this file (they are
used inside fermi_funcs.py itself, which this module already imports via
`from fermi_funcs import *`); upstream carried them unused too, same treatment
as the dropped astropy imports in massProfile.py.

Path/provenance changes:
  - sed_root, fermi_names and the two hardcoded absolute Fermi-LAT SED paths
    are resolved via fermi_funcs.dSphs_sed_path / dSphs_sed_path_update
    (already config-derived; migrated in step 1) instead of upstream's
    relative '../fermi_legacy(_update)/dSphs/SEDs/' strings.
  - save_dir is config.FERMI_REWEIGHTING_DIR / name (legacy) or
    config.FERMI_REWEIGHTING_UPDATE_DIR / name (update) instead of upstream's
    relative '../results/fermi_reweighting(_update)/<dwarf>/'.
    config.FERMI_REWEIGHTING_DIR already existed; FERMI_REWEIGHTING_UPDATE_DIR
    is new (see config.py) since nothing already named the update results
    tree.
  - Every '../results/paper_quantiles/galactocentric/<version>/<name>.npz'
    read is config.PAPER_QUANTILES_DIR / 'galactocentric' / version / f'{name}.npz'
    (config.PAPER_QUANTILES_DIR already existed).

Every output .npz is stamped with provenance, following massProfile.py's
provenance.savez pattern rather than save_contours.py's raw-np.savez +
stamp_existing (this script never wrote via a shared low-level helper the way
save_contour does, so there is no reason to route around provenance.savez
here). Each output is stamped only against the input(s) that actually built
it, not against every quantiles file this script may have loaded for other
variants:
  - mhalf/F18/K24 and their MC variants, plus Jeans_mhalf_err (its sigmaJ is
    the mhalf error): SatGen version 'Diemer' (upstream's hardcoded `version`
    variable), input = the Diemer quantiles file.
  - mhalf_scatter/mdyn_errani: SatGen version 'mhalf_scatter' (an alias of
    Diemer -- see config.SIM_VERSION_ALIASES), input = the mhalf_scatter
    quantiles file.
  - mhalf_mcdaniel: SatGen version 'mcdaniel' (also a Diemer alias), input =
    the mcdaniel quantiles file.
  - Jeans, Jeans_satgen*: no SatGen h5 input at all (Jeans_J_50/err here come
    entirely from the DwarfJeansAnalysis posterior under dja_prior, or the
    LVDB-derived logJ_disp fallback for Jeans); stamped with dja_prior and no
    `version`.
  - test_jeans_J, mcdaniel_jeans: no input files at all (hardcoded/tabulated
    literature constants); stamped with no `inputs` and no `version`.
"""
import os
import sys

import numpy as np
from astropy.io import fits

from fermi_funcs import *
import Jdata as obs
import config
import provenance

# ====================================================================================== #

ALL_VARIANTS = (
    # Loguniform DJA Jeans (with fallback to logJ_disp for missing posteriors)
    'Jeans', 'Jeans_mhalf_err',
    # DJA Jeans under SatGen-informed priors (skip silently if missing)
    'Jeans_satgen', 'Jeans_satgen_box',
    'Jeans_satgen_shmr_fattahi18',
    'Jeans_satgen_shmr_kim24',
    'Jeans_satgen_shmr_danieli23_const',
    'Jeans_satgen_shmr_moster18',
    # SatGen inference J-factors
    'mhalf', 'F18', 'K24',
    # Segue 1 hardcoded test J
    'test_jeans_J',
    # F18 / K24 Monte Carlo over J-factor PDF (expensive: N_MC=1000)
    'F18_MC', 'F18_MC_mstar_err',
    'K24_MC', 'K24_MC_mstar_err',
    # Literature McDaniel Jeans J-factor table (no fallback)
    'mcdaniel_jeans',
    # mhalf_scatter-version J-factors (same base sim as Diemer, 0.1 dex Mhalf
    # scatter): standard mhalf weights and Errani dynamical-mass weights
    'mhalf_scatter', 'mdyn_errani',

)

# mhalf weighted with McDaniel 2023 sigma_los. Excluded from ALL_VARIANTS (and
# so from a default, no-flag run): results/paper_quantiles/galactocentric/
# holds nine versions and 'mcdaniel' is not among them -- its weights (17 GB)
# were deliberately not migrated, and there is no in-repo producer for the
# quantiles this variant reads. See docs/migration-notes.md, "Products removed
# as stale or orphaned rather than migrated". Reachable only via an explicit
# --variants mhalf_mcdaniel, which fails below with a message naming this.
MCDANIEL_VARIANT = 'mhalf_mcdaniel'

# mhalf_mcdaniel is reachable only via an explicit --variants request, not the
# default run -- see MCDANIEL_VARIANT above.
KNOWN_VARIANTS = ALL_VARIANTS + (MCDANIEL_VARIANT,)

dsph_idx = int(sys.argv[1])
rest = sys.argv[2:]
if '--variants' in rest:
    vi = rest.index('--variants')
    if vi + 1 >= len(rest):
        raise ValueError(f'--variants requires a comma-list; choose from {KNOWN_VARIANTS}')
    variants = tuple(v.strip() for v in rest[vi + 1].split(','))
    unknown = [v for v in variants if v not in KNOWN_VARIANTS]
    if unknown:
        raise ValueError(f'unknown variants: {unknown}; choose from {KNOWN_VARIANTS}')
else:
    variants = ALL_VARIANTS

if '--legacy-version' in rest:
    lv = rest.index('--legacy-version')
    if lv + 1 >= len(rest):
        raise ValueError("--legacy-version requires 'legacy' or 'update'")
    legacy_version = rest[lv + 1]
    if legacy_version not in ('legacy', 'update'):
        raise ValueError(f"unknown --legacy-version {legacy_version!r}; choose 'legacy' or 'update'")
else:
    legacy_version = 'legacy'

dwarf_names = obs.dwarf_names
if legacy_version == 'update':
    sed_root = dSphs_sed_path_update
    fermi_names = obs.fermi_names_update
    results_dir = config.FERMI_REWEIGHTING_UPDATE_DIR
else:
    sed_root = dSphs_sed_path
    fermi_names = obs.fermi_names
    results_dir = config.FERMI_REWEIGHTING_DIR

name = dwarf_names[dsph_idx]
srcname = fermi_names[dsph_idx]

# A run-scoped token, matching vcirc_scatter_300pc.py's stamp fields exactly.
# Each task here writes up to 22 products, each independently self-consistent
# and with no shared job identity otherwise -- None for a direct shell
# invocation, recorded faithfully as null rather than coerced to a string.
array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')

save_dir = str(results_dir / name) + '/'

# Check if save directory exists, if not, create it
if not os.path.exists(save_dir):
    print('No pre-existing path, creating directory.')
    os.makedirs(save_dir, exist_ok=True)
else:
    print('Path already exists')


# Load Fermi flux limits
if srcname is None:
    raise ValueError(f"No Fermi source name found for dwarf index {dsph_idx} ({name})")

sed_path = f'{sed_root}{srcname}_sed.fits'
sed = fits.open(sed_path)[1].data

# Shared-input gating: load only what selected variants need.
needs_quantiles = any(v in variants for v in (
    'mhalf', 'F18', 'K24', 'Jeans_mhalf_err',
    'F18_MC', 'F18_MC_mstar_err', 'K24_MC', 'K24_MC_mstar_err',
))
needs_loguniform_jeans = any(v in variants for v in ('Jeans', 'Jeans_mhalf_err'))
needs_satgen_priors = any(v.startswith('Jeans_satgen') for v in variants)
needs_mc = any(v in variants for v in ('F18_MC', 'F18_MC_mstar_err', 'K24_MC', 'K24_MC_mstar_err'))
needs_mcdaniel = 'mhalf_mcdaniel' in variants
needs_scatter = any(v in variants for v in ('mhalf_scatter', 'mdyn_errani'))
needs_dwarf = needs_loguniform_jeans or needs_satgen_priors

# get J factors
version = 'Diemer'
if needs_quantiles:
    quantiles_path = config.PAPER_QUANTILES_DIR / 'galactocentric' / version / f'{name}.npz'
    provenance.assert_single_version([quantiles_path], expected=version)
    quantiles_file = np.load(quantiles_path)
    # assert_single_version's alias check only fires when the raw
    # satgen_version differs from the resolved sim_version, so a plain
    # Diemer-stamped file dropped into another slot can pass it silently. The
    # column count is a check the stamp cannot give: Diemer quantiles carry
    # mhalf (1) + mstar_weights (10 SHMRs) + joint_weights (10 SHMRs) = 21
    # weight columns, with no mdyn_errani column.
    n_cols = quantiles_file['J_quantiles'].shape[1]
    if n_cols != 21:
        raise ValueError(
            f'{quantiles_path} has {n_cols} J_quantiles weight columns, '
            'expected 21 for a Diemer-version quantiles file (mhalf + 10 '
            'mstar_weights SHMRs + 10 joint_weights SHMRs); this file may be '
            'mislabeled version')
    logJquantiles = np.log10(quantiles_file['J_quantiles'])
    logJ_16, logJ_50, logJ_84 = logJquantiles[0], logJquantiles[1], logJquantiles[2]
    logJ_pos_err = logJ_84 - logJ_50
    logJ_neg_err = logJ_50 - logJ_16
    logJ_err = np.average([logJ_pos_err, logJ_neg_err], axis=0)
    diemer_record = provenance.stamp(
        'python/fermi_reweighting.py', version=version, argv=sys.argv[1:],
        inputs=[quantiles_path], dwarf=name, legacy_version=legacy_version,
        array_job_id=array_job_id, array_task_id=array_task_id)

idcs = np.array([0, 4, 10])
weight_names = np.array(['mhalf', 'F18', 'K24'])

# mhalf_scatter version J-factors (separate quantiles file; same base Diemer
# sim). Column 0 = mhalf weights, column 21 = mdyn_errani weights.
if needs_scatter:
    scatter_path = config.PAPER_QUANTILES_DIR / 'galactocentric' / 'mhalf_scatter' / f'{name}.npz'
    provenance.assert_single_version([scatter_path], expected='mhalf_scatter')
    scatter_file = np.load(scatter_path)
    # Same rationale as the Diemer check above: mhalf_scatter quantiles carry
    # one additional column (mdyn_errani, index 21) beyond Diemer's 21, for 22
    # total. A Diemer file dropped into this slot has only 21 columns and
    # sc_idcs=[0, 21] below would IndexError on it rather than silently
    # mislabel a Diemer number as mhalf_scatter.
    n_cols_sc = scatter_file['J_quantiles'].shape[1]
    if n_cols_sc != 22:
        raise ValueError(
            f'{scatter_path} has {n_cols_sc} J_quantiles weight columns, '
            'expected 22 for an mhalf_scatter-version quantiles file (the '
            '21 Diemer columns plus mdyn_errani at column 21); this file may '
            'be mislabeled version')
    logJq_sc = np.log10(scatter_file['J_quantiles'])
    logJ_50_sc = logJq_sc[1]
    logJ_err_sc = np.average([logJq_sc[2] - logJq_sc[1],
                              logJq_sc[1] - logJq_sc[0]], axis=0)
    scatter_record = provenance.stamp(
        'python/fermi_reweighting.py', version='mhalf_scatter', argv=sys.argv[1:],
        inputs=[scatter_path], dwarf=name, legacy_version=legacy_version,
        array_job_id=array_job_id, array_task_id=array_task_id)

if needs_dwarf:
    dwarf = obs.Dwarf(name)

if needs_loguniform_jeans:
    dwarf.get_Jeans_results()
    jeans_q = dwarf.Jeans_J_quantiles  # linear J at 16/50/84
    if np.all(np.isfinite(jeans_q)):
        Jeans_J_50 = np.log10(jeans_q[1])
        Jeans_J_err = np.average([np.log10(jeans_q[2]) - np.log10(jeans_q[1]),
                                  np.log10(jeans_q[1]) - np.log10(jeans_q[0])])
        jeans_source = 'dja_posterior'
        jeans_sigma_source = 'dja_posterior'
    else:
        # Fall back to the scaling-relation (logJ_disp) J-factor for dwarfs with
        # no DwarfJeansAnalysis posterior (e.g. Fornax, Sgr, SMC, LMC). Tucana V
        # does have a posterior -- it is not in this fallback set.
        Jeans_J_50 = dwarf.logJ_disp
        Jeans_J_err = 0.60
        # Jeans_J_err = np.average([dwarf.logJ_disp_err[1],dwarf.logJ_disp_err[0]])
        jeans_source = 'logJ_disp_scaling_relation'
        jeans_sigma_source = 'logJ_disp_scaling_relation_fixed_0.60dex'

mass_vec =  np.logspace(0,4,40)
# sv ceiling 1e-20 (was 1e-22): at high m_chi and small-J MC draws the
# data prefers sv > 1e-22, which previously pegged ~46% of MC draws at
# the upper grid edge and collapsed the per-mass TS_max 16/84 band.
sigmav_vec = np.logspace(-30,-20,70)

# To convert the SED likelihood into DM space, we need to provide all the relevant dm parameters,
# and the mass and cross ection range we want to scan over
# EWcorr = 'Yes', indexDMchannel = 13 are the relevant terms for the DM flux properties (bb)
# if sigmaJ is provided, the J factor prior will be aplied

if 'Jeans' in variants:
    TS_array = convert_sed(sed, Jfactor = pow(10,Jeans_J_50), sigmaJ = Jeans_J_err,
                           EWcorr = 'Yes', indexDMchannel = 13,
                           mass_vec=mass_vec, sigmav_vec=sigmav_vec)
    # No SatGen h5 dependency here -- Jeans_J_50/err come entirely from the
    # DJA loguniform posterior (or the logJ_disp scaling-relation fallback,
    # itself LVDB-derived, not SatGen-derived). dja_prior is only stamped when
    # the DJA posterior actually fired -- stamping it on the scaling-relation
    # fallback would falsely claim a DJA snapshot for a number that has none.
    record = provenance.stamp(
        'python/fermi_reweighting.py', argv=sys.argv[1:],
        inputs=[], dwarf=name, legacy_version=legacy_version,
        dja_prior=(config.DJA_PRIOR if jeans_source == 'dja_posterior' else None),
        jeans_source=jeans_source, jeans_sigma_source=jeans_sigma_source,
        array_job_id=array_job_id, array_task_id=array_task_id)
    provenance.savez(f'{save_dir}Jeans', record,
                     mass_vec=mass_vec, sigmav_vec=sigmav_vec, TS_array=TS_array)

if 'Jeans_mhalf_err' in variants:
    TS_array = convert_sed(sed, Jfactor = pow(10,Jeans_J_50), sigmaJ = logJ_err[0],
                           EWcorr = 'Yes', indexDMchannel = 13,
                           mass_vec=mass_vec, sigmav_vec=sigmav_vec)
    record = provenance.stamp(
        'python/fermi_reweighting.py', version=version, argv=sys.argv[1:],
        inputs=[quantiles_path], dwarf=name, legacy_version=legacy_version,
        dja_prior=(config.DJA_PRIOR if jeans_source == 'dja_posterior' else None),
        jeans_source=jeans_source, jeans_sigma_source='mhalf_error',
        array_job_id=array_job_id, array_task_id=array_task_id)
    provenance.savez(f'{save_dir}Jeans_mhalf_err', record,
                     mass_vec=mass_vec, sigmav_vec=sigmav_vec, TS_array=TS_array)

# DJA Jeans posteriors under SatGen-informed priors. Skip the dwarf/prior
# combinations where no posterior_samples.npz exists rather than falling back
# to the scaling-relation J — the point of these outputs is the satgen prior.
satgen_priors = [
    ('satgen',                          'Jeans_satgen'),
    ('satgen_box',                      'Jeans_satgen_box'),
    ('satgen_shmr_fattahi18',           'Jeans_satgen_shmr_fattahi18'),
    ('satgen_shmr_kim24',               'Jeans_satgen_shmr_kim24'),
    ('satgen_shmr_danieli23_const',     'Jeans_satgen_shmr_danieli23_const'),
    ('satgen_shmr_moster18',            'Jeans_satgen_shmr_moster18'),
]
for prior, outname in satgen_priors:
    if outname not in variants:
        continue
    dwarf.get_Jeans_results(dja_prior=prior)
    q = dwarf.Jeans_J_quantiles
    if not np.all(np.isfinite(q)):
        print(f'Skipping {outname}: no posterior for prior={prior}')
        continue
    J50 = np.log10(q[1])
    Jerr = np.average([np.log10(q[2]) - np.log10(q[1]),
                       np.log10(q[1]) - np.log10(q[0])])
    TS_array = convert_sed(sed, Jfactor=pow(10, J50), sigmaJ=Jerr,
                           EWcorr='Yes', indexDMchannel=13,
                           mass_vec=mass_vec, sigmav_vec=sigmav_vec)
    record = provenance.stamp(
        'python/fermi_reweighting.py', argv=sys.argv[1:],
        inputs=[], dwarf=name, legacy_version=legacy_version,
        dja_prior=prior,
        array_job_id=array_job_id, array_task_id=array_task_id)
    provenance.savez(f'{save_dir}{outname}', record,
                     mass_vec=mass_vec, sigmav_vec=sigmav_vec, TS_array=TS_array)

for i, idx in enumerate(idcs):
    wname = str(weight_names[i])
    if wname not in variants:
        continue
    TS_array = convert_sed(sed, Jfactor = pow(10,logJ_50[idx]), sigmaJ = logJ_err[idx],
                           EWcorr = 'Yes', indexDMchannel = 13,
                           mass_vec=mass_vec, sigmav_vec=sigmav_vec)
    provenance.savez(f'{save_dir}{wname}', diemer_record,
                     mass_vec=mass_vec, sigmav_vec=sigmav_vec, TS_array=TS_array)

if needs_scatter:
    sc_idcs = np.array([0, 21])
    sc_weight_names = np.array(['mhalf_scatter', 'mdyn_errani'])
    for i, idx in enumerate(sc_idcs):
        wname = str(sc_weight_names[i])
        if wname not in variants:
            continue
        TS_array = convert_sed(sed, Jfactor = pow(10,logJ_50_sc[idx]), sigmaJ = logJ_err_sc[idx],
                               EWcorr = 'Yes', indexDMchannel = 13,
                               mass_vec=mass_vec, sigmav_vec=sigmav_vec)
        provenance.savez(f'{save_dir}{wname}', scatter_record,
                         mass_vec=mass_vec, sigmav_vec=sigmav_vec, TS_array=TS_array)

if 'test_jeans_J' in variants and name == 'Segue 1':
    test_segue1_J = 19.54
    test_segue1_J_err = 0.45
    TS_array = convert_sed(sed, Jfactor = pow(10,test_segue1_J), sigmaJ = test_segue1_J_err,
                           EWcorr = 'Yes', indexDMchannel = 13,
                           mass_vec=mass_vec, sigmav_vec=sigmav_vec)
    record = provenance.stamp(
        'python/fermi_reweighting.py', argv=sys.argv[1:],
        inputs=[], dwarf=name, legacy_version=legacy_version,
        array_job_id=array_job_id, array_task_id=array_task_id)
    provenance.savez(f'{save_dir}test_jeans_J', record,
                     mass_vec=mass_vec, sigmav_vec=sigmav_vec, TS_array=TS_array)

# ============================================================
# Monte Carlo sampling over J-factor PDF for F18 and K24 weights
# ============================================================
if needs_mc:
    N_MC = 1000
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    mc_idcs = [4, 10]
    mc_weight_names = ['F18', 'K24']
    sigmaJ_fixed = logJ_err[0]  # mhalf error

    for w_i, idx in enumerate(mc_idcs):
        wname = mc_weight_names[w_i]
        do_central = f'{wname}_MC' in variants
        do_mstar_err = f'{wname}_MC_mstar_err' in variants
        # Always draw from rng to keep the F18->K24 RNG stream identical to
        # pre-refactor behavior, even when this weight's output is deselected.
        u = rng.standard_normal(N_MC)
        sampled_logJs = np.where(
            u >= 0,
            logJ_50[idx] + u * logJ_pos_err[idx],
            logJ_50[idx] + u * logJ_neg_err[idx],
        )
        if not (do_central or do_mstar_err):
            continue
        print(f'Running MC for {wname}')

        # Batch the 1000 draws through a single call that shares per-(mass,sigmav)
        # precompute and uses a tabulated profile-likelihood minimum instead of
        # a per-draw Minuit.migrad(). Minuit also silently converges to local
        # minima on a few % of cells in the strong-signal regime; the tabulated
        # version finds the true global minimum.
        if do_central:
            TS_arrays = convert_sed_batch(
                sed,
                Jfactors=np.power(10.0, sampled_logJs),
                sigmaJ=sigmaJ_fixed,
                EWcorr='Yes', indexDMchannel=13,
                mass_vec=mass_vec, sigmav_vec=sigmav_vec,
            )

            provenance.savez(f'{save_dir}{wname}_MC', diemer_record,
                             mass_vec=mass_vec, sigmav_vec=sigmav_vec,
                             TS_arrays=TS_arrays,
                             sampled_logJs=sampled_logJs)

        if do_mstar_err:
            TS_arrays = convert_sed_batch(
                sed,
                Jfactors=np.power(10.0, sampled_logJs),
                sigmaJ=logJ_err[idx],
                EWcorr='Yes', indexDMchannel=13,
                mass_vec=mass_vec, sigmav_vec=sigmav_vec,
            )

            provenance.savez(f'{save_dir}{wname}_MC_mstar_err', diemer_record,
                             mass_vec=mass_vec, sigmav_vec=sigmav_vec,
                             TS_arrays=TS_arrays,
                             sampled_logJs=sampled_logJs)

# ============================================================
# mhalf-weighted TS array using McDaniel 2023 sigma_los
# ============================================================
if needs_mcdaniel:
    mcdaniel_path = config.PAPER_QUANTILES_DIR / 'galactocentric' / 'mcdaniel' / f'{name}.npz'
    if not mcdaniel_path.exists():
        raise FileNotFoundError(
            f'{mcdaniel_path} does not exist: paper_quantiles/galactocentric/mcdaniel '
            "was deliberately not migrated (its 17 GB of weights were excluded; "
            'no in-repo producer) -- see docs/migration-notes.md, "Products '
            'removed as stale or orphaned rather than migrated". '
            f'{MCDANIEL_VARIANT!r} cannot be built in this repository.')
    provenance.assert_single_version([mcdaniel_path], expected='mcdaniel')
    mcd_quantiles = np.load(mcdaniel_path)
    mcd_logJ = np.log10(mcd_quantiles['J_quantiles'])
    mcd_logJ_50 = mcd_logJ[1, 0]
    mcd_logJ_err = np.average([mcd_logJ[2, 0] - mcd_logJ[1, 0],
                               mcd_logJ[1, 0] - mcd_logJ[0, 0]])

    TS_array = convert_sed(sed, Jfactor=10**mcd_logJ_50, sigmaJ=mcd_logJ_err,
                           EWcorr='Yes', indexDMchannel=13,
                           mass_vec=mass_vec, sigmav_vec=sigmav_vec)
    record = provenance.stamp(
        'python/fermi_reweighting.py', version='mcdaniel', argv=sys.argv[1:],
        inputs=[mcdaniel_path], dwarf=name, legacy_version=legacy_version,
        array_job_id=array_job_id, array_task_id=array_task_id)
    provenance.savez(f'{save_dir}mhalf_mcdaniel', record,
                     mass_vec=mass_vec, sigmav_vec=sigmav_vec, TS_array=TS_array)

# ============================================================
# Literature McDaniel Jeans J-factor (logJ, err read directly from
# obs.mcdaniel_Jeans_log10J_values; no fallback for missing dwarfs)
# ============================================================
if 'mcdaniel_jeans' in variants:
    mcj_logJ, mcj_logJ_err = obs.mcdaniel_Jeans_log10J_values[dsph_idx]
    if not (np.isfinite(mcj_logJ) and np.isfinite(mcj_logJ_err)):
        print(f'Skipping mcdaniel_jeans: no McDaniel Jeans J for {name}')
    else:
        TS_array = convert_sed(sed, Jfactor=10**mcj_logJ, sigmaJ=mcj_logJ_err,
                               EWcorr='Yes', indexDMchannel=13,
                               mass_vec=mass_vec, sigmav_vec=sigmav_vec)
        record = provenance.stamp(
            'python/fermi_reweighting.py', argv=sys.argv[1:],
            inputs=[], dwarf=name, legacy_version=legacy_version,
            array_job_id=array_job_id, array_task_id=array_task_id)
        provenance.savez(f'{save_dir}mcdaniel_jeans', record,
                         mass_vec=mass_vec, sigmav_vec=sigmav_vec, TS_array=TS_array)
