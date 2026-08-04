"""Null-distribution TS grids from Fermi-LAT blank-field SEDs, for expected
(Brazilian-band) limits: mirrors the mhalf/Jeans/F18/satgen_shmr_fattahi18
single-J branches of fermi_reweighting.py but loops the call over the 1008
blank-field SEDs instead of the dwarf's own SED.

Migrated from SatGen_Dwarf/python/fermi_expected_limits.py. Path/import/
provenance edits only -- the TS computation (convert_sed, imported unchanged
from fermi_funcs.py), the mass_vec/sigmav_vec grids (matching
fermi_reweighting.py's), the per-variant J-factor extraction, the F18 RNG
seeding convention (default_rng(abs(hash(name)) % (2**32)), one draw per
field), and the timing/looping/print structure are all character-identical to
upstream.

Usage, unchanged from upstream:
    python fermi_expected_limits.py <dsph_idx>
    python fermi_expected_limits.py <dsph_idx> --timing-test [N]
    python fermi_expected_limits.py <dsph_idx> --variants mhalf,Jeans,F18,satgen_shmr_fattahi18

No --legacy-version flag: unlike fermi_reweighting.py, this script never reads
the dwarf's own SED (legacy vs update), only the blank-field null-distribution
SEDs under data/fermi_legacy/blank_fields/, which upstream noted have no
_update counterpart (see fermi_funcs.py's blank_TS_prof_path, migrated in step
1 -- it stays on the legacy tree for the same reason). Adding a flag with
nothing behind it to select would be a feature this migration was not asked
for; results/fermi_expected_limits/ (no _update suffix) matches that this
stage does not vary by legacy-version.

Path/provenance changes:
  - blank_dir was a hardcoded absolute path outside the repository, pointing
    at SatGen_Dwarf's own data/fermi_legacy/blank_fields/SEDs;
    it is now fermi_funcs.blank_field_sed_path, which
    fermi_funcs.py builds from config.DATA_DIR the same way as its
    'fermi_legacy' siblings (blank_TS_prof_path etc.) -- that module already
    owns this path layout and is covered by tests/test_fermi_paths.py, so the
    constant belongs there rather than being rebuilt here by string join.
  - Every '../results/paper_quantiles/galactocentric/<version>/<name>.npz'
    read is config.PAPER_QUANTILES_DIR / 'galactocentric' / version / f'{name}.npz'
    (config.PAPER_QUANTILES_DIR already existed).
  - save_dir is config.FERMI_EXPECTED_LIMITS_DIR / name instead of upstream's
    relative '../results/fermi_expected_limits/<dwarf>/'
    (config.FERMI_EXPECTED_LIMITS_DIR already existed).

Every output .npz is stamped with provenance (provenance.savez, as in
massProfile.py), against only the input(s) that built it -- mhalf/F18 against
the Diemer quantiles file (SatGen version 'Diemer'), Jeans/satgen_shmr_fattahi18
against no SatGen h5 (their J-factor comes from the DJA posterior, tagged with
dja_prior; see the equivalent branches in fermi_reweighting.py). Skipped
entirely under --timing-test, matching upstream, which writes nothing in that
mode.
"""
import os
import sys
import glob
import re
import time

import numpy as np
from astropy.io import fits

from fermi_funcs import convert_sed, blank_field_sed_path
import Jdata as obs
import config
import provenance

# ====================================================================================== #

ALL_VARIANTS = ('mhalf', 'Jeans', 'F18', 'satgen_shmr_fattahi18')

dsph_idx = int(sys.argv[1])
rest = sys.argv[2:]
timing_test = '--timing-test' in rest
if timing_test:
    ti = rest.index('--timing-test')
    timing_N = int(rest[ti + 1]) if (ti + 1 < len(rest) and not rest[ti + 1].startswith('--')) else 20
else:
    timing_N = 20
if '--variants' in rest:
    vi = rest.index('--variants')
    if vi + 1 >= len(rest):
        raise ValueError(f'--variants requires a comma-list argument; choose from {ALL_VARIANTS}')
    variants = tuple(v.strip() for v in rest[vi + 1].split(','))
    unknown = [v for v in variants if v not in ALL_VARIANTS]
    if unknown:
        raise ValueError(f'unknown variants: {unknown}; choose from {ALL_VARIANTS}')
else:
    variants = ALL_VARIANTS

name = obs.dwarf_names[dsph_idx]

# A run-scoped token, matching vcirc_scatter_300pc.py's stamp fields exactly.
# None for a direct shell invocation, recorded faithfully as null rather than
# coerced to a string.
array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')

# This script has no --legacy-version flag (see the module docstring: it only
# ever reads the legacy blank-field null-distribution SEDs), so every product
# it writes is stamped against the same tree. Naming it explicitly means a
# future _update blank-field set cannot be confused with this one retroactively.
SED_TREE = 'legacy'

save_dir = str(config.FERMI_EXPECTED_LIMITS_DIR / name) + '/'
if not timing_test:
    os.makedirs(save_dir, exist_ok=True)

version = 'Diemer'
dwarf = obs.Dwarf(name)

needs_quantiles = any(v in variants for v in ('mhalf', 'F18'))
if needs_quantiles:
    # mhalf / F18 J-factor — same extraction as fermi_reweighting.py.
    quantiles_path = config.PAPER_QUANTILES_DIR / 'galactocentric' / version / f'{name}.npz'
    provenance.assert_single_version([quantiles_path], expected=version)
    quantiles_file = np.load(quantiles_path)
    # assert_single_version's alias check only fires when the raw
    # satgen_version differs from the resolved sim_version, so a plain
    # differently-labeled file can pass it silently. Diemer quantiles carry
    # mhalf (1) + mstar_weights (10 SHMRs) + joint_weights (10 SHMRs) = 21
    # weight columns; see fermi_reweighting.py's matching check.
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
    logJ_50_mhalf = logJ_50[0]
    logJ_err_mhalf = logJ_err[0]
    # F18 J-factor — index 4 in the J_quantiles table.
    logJ_50_F18 = logJ_50[4]
    logJ_pos_F18 = logJ_pos_err[4]
    logJ_neg_F18 = logJ_neg_err[4]
    sigmaJ_F18 = logJ_err[0]   # mhalf err, matches fermi_reweighting.py
    diemer_record = provenance.stamp(
        'python/fermi_expected_limits.py', version=version, argv=sys.argv[1:],
        inputs=[quantiles_path], dwarf=name, sed_tree=SED_TREE,
        array_job_id=array_job_id, array_task_id=array_task_id)

if 'Jeans' in variants:
    # Jeans/loguniform J-factor — fermi_reweighting.py's needs_loguniform_jeans branch.
    dwarf.get_Jeans_results()
    jeans_q = dwarf.Jeans_J_quantiles
    if np.all(np.isfinite(jeans_q)):
        Jeans_J_50 = np.log10(jeans_q[1])
        Jeans_J_err = np.average([
            np.log10(jeans_q[2]) - np.log10(jeans_q[1]),
            np.log10(jeans_q[1]) - np.log10(jeans_q[0]),
        ])
        jeans_source = 'dja_posterior'
        jeans_sigma_source = 'dja_posterior'
    else:
        # Fall back to the scaling-relation (logJ_disp) J-factor for dwarfs
        # with no DwarfJeansAnalysis posterior (e.g. Fornax, Sgr, SMC, LMC).
        Jeans_J_50 = dwarf.logJ_disp
        Jeans_J_err = 0.60
        jeans_source = 'logJ_disp_scaling_relation'
        jeans_sigma_source = 'logJ_disp_scaling_relation_fixed_0.60dex'
    # dja_prior is only stamped when the DJA posterior actually fired --
    # stamping it on the scaling-relation fallback would falsely claim a DJA
    # snapshot for a number that has none.
    jeans_record = provenance.stamp(
        'python/fermi_expected_limits.py', argv=sys.argv[1:],
        inputs=[], dwarf=name,
        dja_prior=(config.DJA_PRIOR if jeans_source == 'dja_posterior' else None),
        jeans_source=jeans_source, jeans_sigma_source=jeans_sigma_source,
        sed_tree=SED_TREE, array_job_id=array_job_id, array_task_id=array_task_id)

run_satgen_shmr = 'satgen_shmr_fattahi18' in variants
if run_satgen_shmr:
    # SatGen-SHMR-Fattahi18 Jeans posterior — mirrors fermi_reweighting.py's satgen_priors loop.
    dwarf.get_Jeans_results(dja_prior='satgen_shmr_fattahi18')
    satgen_shmr_q = dwarf.Jeans_J_quantiles
    if np.all(np.isfinite(satgen_shmr_q)):
        satgen_shmr_J50 = np.log10(satgen_shmr_q[1])
        satgen_shmr_Jerr = np.average([
            np.log10(satgen_shmr_q[2]) - np.log10(satgen_shmr_q[1]),
            np.log10(satgen_shmr_q[1]) - np.log10(satgen_shmr_q[0]),
        ])
        satgen_shmr_record = provenance.stamp(
            'python/fermi_expected_limits.py', argv=sys.argv[1:],
            inputs=[], dwarf=name, dja_prior='satgen_shmr_fattahi18',
            sed_tree=SED_TREE, array_job_id=array_job_id, array_task_id=array_task_id)
    else:
        print(f'[{name}] no posterior for satgen_shmr_fattahi18 — skipping this variant')
        run_satgen_shmr = False

# Match fermi_reweighting.py's grid.
mass_vec = np.logspace(0, 4, 40)
sigmav_vec = np.logspace(-30, -20, 70)

# Enumerate blank-field SEDs.
blank_dir = blank_field_sed_path.rstrip('/')
field_paths = sorted(glob.glob(f'{blank_dir}/field_*_sed.fits'))
if timing_test:
    field_paths = field_paths[:timing_N]
N_fields = len(field_paths)
if N_fields == 0:
    raise FileNotFoundError(f'No blank-field SEDs found under {blank_dir}')

print(f'[{name}] N_fields={N_fields}  variants={list(variants)}')
if 'mhalf' in variants:
    print(f'[{name}] mhalf logJ_50={logJ_50_mhalf:.3f}  sigmaJ={logJ_err_mhalf:.3f}')
if 'Jeans' in variants:
    print(f'[{name}] Jeans logJ_50={Jeans_J_50:.3f}  sigmaJ={Jeans_J_err:.3f}')
if 'F18' in variants:
    print(f'[{name}] F18   logJ_50={logJ_50_F18:.3f}  pos={logJ_pos_F18:.3f}  neg={logJ_neg_F18:.3f}  sigmaJ={sigmaJ_F18:.3f}')
if run_satgen_shmr:
    print(f'[{name}] satgen_shmr_fattahi18 logJ_50={satgen_shmr_J50:.3f}  sigmaJ={satgen_shmr_Jerr:.3f}')

if 'F18' in variants:
    # Pre-draw F18 logJs (one per field) using the same seed convention as
    # fermi_reweighting.py.
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    u = rng.standard_normal(N_fields)
    sampled_logJs_F18 = np.where(
        u >= 0,
        logJ_50_F18 + u * logJ_pos_F18,
        logJ_50_F18 + u * logJ_neg_F18,
    )

ts_shape = (N_fields, len(mass_vec), len(sigmav_vec))
TS_arrays_mhalf = np.empty(ts_shape) if 'mhalf' in variants else None
TS_arrays_Jeans = np.empty(ts_shape) if 'Jeans' in variants else None
TS_arrays_F18 = np.empty(ts_shape) if 'F18' in variants else None
TS_arrays_satgen_shmr = np.empty(ts_shape) if run_satgen_shmr else None
field_ids = np.empty(N_fields, dtype=int)

t0 = time.time()
for i, path in enumerate(field_paths):
    t_call = time.time()
    with fits.open(path) as h:
        sed = h[1].data.copy()
    if 'mhalf' in variants:
        TS_arrays_mhalf[i] = convert_sed(
            sed,
            Jfactor=10 ** logJ_50_mhalf, sigmaJ=logJ_err_mhalf,
            EWcorr='Yes', indexDMchannel=13,
            mass_vec=mass_vec, sigmav_vec=sigmav_vec,
        )
    if 'Jeans' in variants:
        TS_arrays_Jeans[i] = convert_sed(
            sed,
            Jfactor=10 ** Jeans_J_50, sigmaJ=Jeans_J_err,
            EWcorr='Yes', indexDMchannel=13,
            mass_vec=mass_vec, sigmav_vec=sigmav_vec,
        )
    if 'F18' in variants:
        TS_arrays_F18[i] = convert_sed(
            sed,
            Jfactor=10 ** sampled_logJs_F18[i], sigmaJ=sigmaJ_F18,
            EWcorr='Yes', indexDMchannel=13,
            mass_vec=mass_vec, sigmav_vec=sigmav_vec,
        )
    if run_satgen_shmr:
        TS_arrays_satgen_shmr[i] = convert_sed(
            sed,
            Jfactor=10 ** satgen_shmr_J50, sigmaJ=satgen_shmr_Jerr,
            EWcorr='Yes', indexDMchannel=13,
            mass_vec=mass_vec, sigmav_vec=sigmav_vec,
        )
    field_ids[i] = int(re.search(r'field_(\d+)_sed', os.path.basename(path)).group(1))
    if i < 3 or (i + 1) % 50 == 0:
        dt = time.time() - t_call
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (N_fields - i - 1)
        print(f'  [{i+1}/{N_fields}] field {field_ids[i]}  call={dt:.2f}s  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m')

total = time.time() - t0
print(f'[{name}] done in {total/60:.2f} min  ({total/N_fields:.2f} s/field)')

if timing_test:
    print('[timing-test] no output written')
    sys.exit(0)

if 'mhalf' in variants:
    provenance.savez(
        f'{save_dir}mhalf_blank_fields', diemer_record,
        mass_vec=mass_vec, sigmav_vec=sigmav_vec,
        TS_arrays=TS_arrays_mhalf, field_ids=field_ids,
        logJ_50=logJ_50_mhalf, logJ_err=logJ_err_mhalf,
    )
if 'Jeans' in variants:
    provenance.savez(
        f'{save_dir}Jeans_blank_fields', jeans_record,
        mass_vec=mass_vec, sigmav_vec=sigmav_vec,
        TS_arrays=TS_arrays_Jeans, field_ids=field_ids,
        logJ_50=Jeans_J_50, logJ_err=Jeans_J_err,
    )
if 'F18' in variants:
    provenance.savez(
        f'{save_dir}F18_MC_blank_fields', diemer_record,
        mass_vec=mass_vec, sigmav_vec=sigmav_vec,
        TS_arrays=TS_arrays_F18, field_ids=field_ids,
        sampled_logJs=sampled_logJs_F18,
        sigmaJ=sigmaJ_F18,
        logJ_50_F18=logJ_50_F18,
        logJ_pos_F18=logJ_pos_F18, logJ_neg_F18=logJ_neg_F18,
    )
if run_satgen_shmr:
    provenance.savez(
        f'{save_dir}satgen_shmr_fattahi18_blank_fields', satgen_shmr_record,
        mass_vec=mass_vec, sigmav_vec=sigmav_vec,
        TS_arrays=TS_arrays_satgen_shmr, field_ids=field_ids,
        logJ_50=satgen_shmr_J50, logJ_err=satgen_shmr_Jerr,
    )
