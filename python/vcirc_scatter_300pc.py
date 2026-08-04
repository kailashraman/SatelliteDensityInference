"""
Dex scatter on the weighted-median V_circ at r = 300 pc for one (dwarf, SHMR) pair.

Migrated from SatGen_Dwarf/python/vcirc_scatter_300pc.py. The three hardcoded
absolute SatGen_Dwarf paths (BASE, MASSPROF, MVIR200) are replaced with
config-derived paths (VCIRC_SCATTER_DIR/<version>,
PAPER_MASSPROFILE_DIR/<version>/massProfile.npz,
ADDITIONAL_DIR/<version>_Mvir200.npz). The numerics are otherwise unchanged:
N_MC, NARROW_SIGMA, R_TARGET_KPC, SEED, the SHMR_NAMES list and its order (the
task decomposition below depends on that order), the SHMR dispatch chain, and
the CSV field order/format specifiers are all as upstream. N_MC, NARROW_SIGMA,
R_TARGET_KPC, SEED and SHMR_NAMES live in vcirc_scatter_params.py, shared with
concat_vcirc_scatter_300pc.py, which cannot import this module directly (see
that module's docstring).

One deliberate behavioural change: this script now seeds the global NumPy RNG
(see the comment at the seed call). The source it was ported from did not, so
its products -- and the migrated ones here -- are one unrepeatable realization.
That is not a theoretical concern: rerunning the migrated script twice gives
v_med differing by ~0.1% and sigma_dex by ~13%, and neither run reproduces the
migrated reference, which additionally predates source changes made after those
reference CSVs were written. The migrated vcirc_scatter_300pc tree therefore
cannot serve as a parity gate for this stage; it was regenerated rather than
verified against.

Array task decomposition:
    dwarf_idx = task_id // len(SHMR_NAMES)
    shmr_idx  = task_id %  len(SHMR_NAMES)

Writes one CSV data line to:
    results/vcirc_scatter_300pc/<version>/<SHMR>/per_dwarf/row_<dwarf_idx:03d>.csv

Concatenate after the array finishes with scripts/concat_vcirc_scatter_300pc.sh.

Usage: python vcirc_scatter_300pc.py <task_id> <version>
"""
import os
import re
import sys
import zlib

import numpy as np

import Jdata as obs
import halo_weights
from vcirc_median_uncertainty import vcirc_median_uncertainty
import config
import provenance
from vcirc_scatter_params import (N_MC, NARROW_SIGMA, R_TARGET_KPC, SEED,
                                  SHMR_NAMES)


TASK_ID      = int(sys.argv[1])
VERSION      = sys.argv[2]

N_SHMR    = len(SHMR_NAMES)
dwarf_idx = TASK_ID // N_SHMR
shmr_idx  = TASK_ID %  N_SHMR

# Parsed from the launcher's own --array bound, rather than transcribed as a
# second literal: a copy here can drift from the one SLURM actually parses,
# and an assert against a stale copy passes on every dispatched task even
# after the dwarf list grows -- the tail tasks are simply never dispatched,
# and nothing in this file's own logic would catch that.
LAUNCHER_PATH = config.REPO_ROOT / 'scripts' / 'vcirc_scatter_300pc.sh'
LAUNCHER_SOURCE = LAUNCHER_PATH.read_text()

# SLURM's own --array grammar is more permissive than the previous regex
# tolerated: a `%N` throttle suffix (`0-429%20`) is a routine, non-semantic
# edit to this bound -- the kind of change someone makes without touching the
# task count at all -- but a `:N` step suffix (`0-429:2`) IS semantic: it
# halves the number of tasks SLURM actually dispatches (215 of them, every
# other index) while LAUNCHER_N_TASKS below would still read 430 and the
# equality assert would pass on every dispatched task, silently starving the
# odd-indexed SHMRs of any rows at all. The two are captured separately below
# so the step can be asserted against rather than discarded alongside the
# throttle. Variable whitespace after #SBATCH is legal, and a trailing
# comment is legal. SLURM itself takes the LAST #SBATCH directive when one is
# repeated, so `.search()` (first match) can silently read a stale,
# overridden value; and SLURM only recognises #SBATCH lines in the header --
# the contiguous run of blank/comment lines before the script body -- so
# scanning the whole file risks matching an unrelated #-prefixed line (e.g.
# inside a heredoc or a later comment) that SLURM never sees at all.
_header_lines = []
for _line in LAUNCHER_SOURCE.splitlines():
    if _line.strip() == '' or _line.lstrip().startswith('#'):
        _header_lines.append(_line)
    else:
        break
_HEADER = '\n'.join(_header_lines)

_ARRAY_BOUND_RE = re.compile(
    r'^#SBATCH[ \t]+--array=0-(\d+)(?::(\d+))?(?:%\d+)?[ \t]*(?:#.*)?$',
    re.MULTILINE)
_array_matches = list(_ARRAY_BOUND_RE.finditer(_HEADER))
assert _array_matches, (
    'could not find a "#SBATCH --array=0-<N>" line (optionally with a :M '
    f'step or %M throttle suffix) in the header of {LAUNCHER_PATH}; cannot '
    'derive the declared task count')
_array_step = _array_matches[-1].group(2)
assert _array_step is None or _array_step == '1', (
    f'{LAUNCHER_PATH} declares --array with a step of {_array_step} '
    '(e.g. "0-429:2"); a step changes WHICH task ids SLURM actually '
    'dispatches (every Nth one), not just how many -- the dwarf_idx/shmr_idx '
    'decomposition below assumes every task id in range is dispatched, so a '
    'step is not supported here')
LAUNCHER_N_TASKS = int(_array_matches[-1].group(1)) + 1

# The array bound in scripts/vcirc_scatter_300pc.sh is parsed above, not
# data-derived from dwarf_names. An inequality here (TASK_ID < n_tasks) only
# catches len(obs.dwarf_names) SHRINKING -- if it GROWS, every task id
# already in range stays in range and the new tail dwarfs are silently never
# computed. Asserting equality against the launcher's declared task count
# catches either direction, at task 0.
n_tasks = len(obs.dwarf_names) * N_SHMR
assert n_tasks == LAUNCHER_N_TASKS, (
    f'{len(obs.dwarf_names)} dwarfs x {N_SHMR} SHMRs = {n_tasks} tasks, but '
    f'the launcher declares {LAUNCHER_N_TASKS} (scripts/vcirc_scatter_300pc.sh '
    f'--array); update the --array bound to 0-{n_tasks - 1} and this constant')
assert TASK_ID < n_tasks, (
    f'TASK_ID={TASK_ID} is out of range for {len(obs.dwarf_names)} dwarfs x '
    f'{N_SHMR} SHMRs = {n_tasks} tasks; the launcher --array bound must be '
    f'0-{n_tasks - 1}')
assert TASK_ID >= 0, (
    f'TASK_ID={TASK_ID} is negative; a negative id passes the `< n_tasks` '
    'range check above via Python\'s negative-index wraparound in '
    'dwarf_idx/shmr_idx and would write a bogus row_-01.csv-style path')

name      = obs.dwarf_names[dwarf_idx]
shmr_name = SHMR_NAMES[shmr_idx]

OUT_DIR  = config.VCIRC_SCATTER_DIR / VERSION / shmr_name / 'per_dwarf'
OUT_PATH = OUT_DIR / f'row_{dwarf_idx:03d}.csv'
MASSPROF = config.PAPER_MASSPROFILE_DIR / VERSION / 'massProfile.npz'
MVIR200  = config.ADDITIONAL_DIR / f'{VERSION}_Mvir200.npz'

os.makedirs(OUT_DIR, exist_ok=True)

provenance_inputs = [config.h5_path(VERSION), MASSPROF]
if shmr_name == 'Kim24':
    provenance_inputs.append(MVIR200)
provenance.assert_single_version(provenance_inputs, expected=VERSION)

# Seed the LEGACY GLOBAL RNG, which is a different generator from the SEED=0
# above. SEED feeds a local np.random.default_rng inside
# vcirc_median_uncertainty, covering only the logMstar MC draws; every SHMR in
# halo_weights.py separately ends in an unseeded np.random.normal over all
# ~2.4M halos. So this stage advertised a seed while remaining irreducibly
# non-deterministic -- two runs of the migrated script differ by ~0.1% in
# v_med and ~13% in sigma_dex, which is enough to defeat any parity gate.
#
# Same treatment and rationale as compute_weights.py: deterministic, and
# distinct per (version, dwarf, SHMR) so array tasks do not share a stream.
# This is a semantic change relative to upstream -- it fixes one realization
# where upstream had none -- and it cannot recover the realization behind the
# migrated products, which were drawn before any seed existed.
np.random.seed(
    zlib.crc32(f'{VERSION}:{dwarf_idx}:{shmr_name}'.encode()) & 0xffffffff)

weights_obj = halo_weights.ResampleMstar(version=VERSION)
logMpeak    = weights_obj.logMpeak
z_in        = weights_obj.z_in
vpeak       = weights_obj.h5['tpeak_v_max'][()]

if shmr_name == 'Behroozi13':
    logMstar_infall = halo_weights.logMstar_B13(logMpeak, z_in)
elif shmr_name == 'Moster13':
    logMstar_infall = halo_weights.logMstar_Moster13(logMpeak, z_in)
elif shmr_name == 'RodriguezPuebla17':
    logMstar_infall = halo_weights.logMstar_RP17(logMpeak, z_in)
elif shmr_name == 'Fattahi18':
    logMstar_infall = halo_weights.logMstar_Fattahi18(vpeak)
elif shmr_name == 'Moster18':
    logMstar_infall = halo_weights.logMstar_Moster18(logMpeak, z_in)
elif shmr_name == 'Behroozi19':
    logMstar_infall = halo_weights.logMstar_Behroozi19(logMpeak, z_in)
elif shmr_name == 'Munshi21':
    logMstar_infall = halo_weights.logMstar_Munshi21(logMpeak)
elif shmr_name == 'Danieli23_const':
    logMstar_infall = halo_weights.logMstar_Danieli23(logMpeak, const=True)
elif shmr_name == 'Danieli23_grow':
    logMstar_infall = halo_weights.logMstar_Danieli23(logMpeak, const=False)
elif shmr_name == 'Kim24':
    logMpeak200     = np.log10(np.load(MVIR200)['Mvir200'])
    logMstar_infall = halo_weights.logMstar_Kim24(logMpeak200)
else:
    raise ValueError(f'unknown SHMR: {shmr_name}')

logMstar_pred = weights_obj.evolved_Mstar(logMstar_infall)

arr           = np.load(MASSPROF)
vcirc_profile = arr['vcircProfile']
r_grid        = arr['r_grid']

# coverage_verified is concat_massProfile.py's record of whether the
# halo_slice cross-check ran end to end (True) or the fragment ordering was
# never independently confirmed (False/missing). Warn rather than raise: this
# script cannot re-derive fragment ordering from massProfile.npz alone, so it
# only surfaces the assurance level of its input.
#
# provenance.read() can return None; today that is unreachable because
# assert_single_version above already rejects an unstamped MASSPROF, but that
# is an ordering dependency between two lines, not a guarantee this line can
# rely on -- handle None explicitly rather than trust it stays that way.
massprof_record = provenance.read(MASSPROF)
massprofile_coverage_verified = (
    massprof_record.get('coverage_verified') if massprof_record is not None else None)
if massprofile_coverage_verified is not True:
    print(f'WARNING: {MASSPROF} has coverage_verified={massprofile_coverage_verified!r} '
          '-- its fragment ordering was not verified', flush=True)

dwarf = obs.Dwarf(name)
out = vcirc_median_uncertainty(
    dwarf, weights_obj, logMstar_pred, vcirc_profile, r_grid,
    extra_mask=None,
    n_mc=N_MC, narrow_sigma=NARROW_SIGMA,
    # The 0.16 here is the 16th-percentile QUANTILE (paired with 0.84 for the
    # V16/V84 columns below), not NARROW_SIGMA -- the two are numerically
    # equal by coincidence, not by relation, so do not de-duplicate them.
    quantiles=(0.16, 0.5, 0.84), seed=SEED,
)

v_med = float(np.interp(R_TARGET_KPC, out['r_grid'], out['median_vcirc']))
v_lo  = float(np.interp(R_TARGET_KPC, out['r_grid'], out['median_vcirc_err'][:, 0]))
v_hi  = float(np.interp(R_TARGET_KPC, out['r_grid'], out['median_vcirc_err'][:, 2]))
sigma_dex = (np.log10(v_hi) - np.log10(v_lo)) / 2.0
lo, hi = dwarf.logMstar_err

# The sidecar is removed immediately before the row file is truncated, not
# after it is written: stamp_existing() runs last (below), so a task killed
# between the write and the stamp would otherwise leave FRESH row data
# sitting beside the PREVIOUS run's sidecar -- which the concat step's
# content checks are slot-invariant to (dwarf/shmr/version are all functions
# of this row's slot) and so cannot detect. Removing the sidecar first means
# that same kill instead leaves fresh, unstamped data, which concat's
# missing-sidecar check does catch.
sidecar_path = OUT_PATH.with_name(OUT_PATH.name + provenance.SIDECAR_SUFFIX)
sidecar_path.unlink(missing_ok=True)
with open(OUT_PATH, 'w') as f:
    f.write(
        f'{name},{dwarf.logMstar:.4f},{lo:.4f},{hi:.4f},'
        f'{v_med:.4f},{v_lo:.4f},{v_hi:.4f},{sigma_dex:.5f}\n'
    )

# This stage calls ResampleMstar.evolved_Mstar, so its products carry an
# infall -> z=0 stellar-mass convention and must name it. Without the field its
# output is indistinguishable from a product made under the old constant
# +1.2 dex rule -- which is exactly what every migrated vcirc product here is,
# and why the whole tree has to be regenerated rather than trusted.
record = provenance.stamp('python/vcirc_scatter_300pc.py', version=VERSION,
                          argv=sys.argv[1:], inputs=provenance_inputs,
                          dwarf=name, shmr=shmr_name,
                          mstar_evolution='per_halo_tidal',
                          massprofile_coverage_verified=massprofile_coverage_verified,
                          # A run token, so a stale survivor of a future rerun
                          # is distinguishable from a fresh row: the other
                          # sidecar fields (mstar_evolution, satgen_version,
                          # dwarf, shmr) are all functions of this row's slot
                          # plus a constant, so a stale row with the same
                          # slot passes every one of them. None for a direct
                          # shell invocation -- recorded faithfully as null,
                          # not coerced to a string.
                          array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                          array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                          stochastic_arrays=('v_med', 'v_lo', 'v_hi',
                                             'sigma_dex'),
                          stochastic_caveat=(
                              'Every value here except name/logMstar and its '
                              'errors depends on two RNG streams: the SHMR '
                              'intrinsic scatter drawn from the global NumPy '
                              'RNG in halo_weights.py, seeded by this script '
                              'from zlib.crc32(f"{version}:{dwarf_idx}:'
                              '{shmr}"), and the logMstar MC draw from a local '
                              'default_rng(SEED=0) in '
                              'vcirc_median_uncertainty. Reproducible by '
                              'rerunning with the same version, dwarf and '
                              'SHMR; NOT recoverable for products written '
                              'before this seeding was added, which are a '
                              'different realization.'))
provenance.stamp_existing(OUT_PATH, record)

print(
    f'[task={TASK_ID} dwarf={dwarf_idx:02d} shmr={shmr_idx}] '
    f'{name:25s} x {shmr_name:18s} '
    f'V_med={v_med:7.3f} km/s   sigma_dex={sigma_dex:.4f} dex  -> {OUT_PATH}',
    flush=True,
)
