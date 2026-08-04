"""Concatenate per-group massProfile-<g>.npz fragments into one massProfile.npz.

Upstream (SatGen_Dwarf) never had this step: massProfile.npz was concatenated
by hand, which is a reproducibility gap -- this script closes it. Modeled on
concat_Js.py: for one SatGen version it globs
results/paper_massProfile/<version>/massProfile/massProfile-<g>.npz,
concatenates massProfile, mpeakProfile, vcircProfile, r_grid in ascending
numeric group order, and writes results/paper_massProfile/<version>/massProfile.npz.

REFUSES to write if: the group numbers are not contiguous 0..N-1; any fragment
in that range is missing; r_grid differs between fragments; or a fragment's
row count disagrees with the chunk size derived from fragment 0 (a missing
trailing group, or a rerun after CHUNK_SIZE changed, would otherwise leave a
short or misaligned array that loads and plots fine).

Coverage is derived from the data, not from provenance: the chunk size C is
read from fragment 0's row count, and every group g is placed at
`[min(g*C, n_halos), min((g+1)*C, n_halos))`. For correctly-sized fragments
these bounds are arithmetically identical to a running offset, so this is a
per-group ROW-COUNT check -- it catches a fragment written under a different
CHUNK_SIZE, a truncated fragment, or a missing tail -- but it does NOT catch
two same-size fragments whose filenames were swapped: both schemes place
same-size fragments identically, so a transposed pair passes the row-count
check and lands silently misaligned. The only thing that catches that case is
the `halo_slice` cross-check below, and only where a fragment's `halo_slice`
provenance field IS present (most are not: scripts/stamp_migrated.py, which
stamped 63 of 64 Diemer fragments, never writes it) and was itself written by
python/massProfile.py from the slice it actually processed -- a `halo_slice`
merely derived from the group index would carry no independent information
and verify nothing. Where present, it is cross-checked against the derived
bounds, and any disagreement is a hard failure. The output stamp's
`coverage_verified` field records whether every fragment carried a matching
`halo_slice`; when it is False, transposition/misordering of same-size
fragments has NOT been ruled out.

Memory: these arrays are large (~5.9 GB for the Diemer result). Building a
Python list of all fragments and np.concatenate-ing at the end would hold both
the list of fragments AND the concatenated output in memory at once, roughly
doubling peak usage. Instead the catalog length is read from the h5 up front
(the same way massProfile.py reads it), the output arrays are preallocated
once from that length and the first fragment's r_grid/dtype, and each fragment
is loaded, copied into its slice, and dropped before the next one is read.

That bounds the *load* side, but not the write: provenance.savez compresses,
and np.savez_compressed buffers on top of the live arrays, so measured MaxRSS
for Diemer is 16.0 GB against 5.9 GB of output. Size the launcher from that
measurement, not from the array arithmetic.

Usage:
    python concat_massProfile.py <version>
e.g. python concat_massProfile.py Diemer
"""
import sys, os, glob

import numpy as np

import config
import provenance
from halo_weights import get_h5

version = sys.argv[1]

gdir = config.PAPER_MASSPROFILE_DIR / version / 'massProfile'
files = glob.glob(str(gdir / 'massProfile-*.npz'))
if not files:
    print(f'{gdir}: no per-group files, nothing to concatenate')
    sys.exit(1)

# map group index -> file (index parsed from 'massProfile-<g>.npz')
idx_file = {int(os.path.basename(f)[len('massProfile-'):-len('.npz')]): f for f in files}
gmax = max(idx_file)
missing = [g for g in range(gmax + 1) if g not in idx_file]
if missing:
    print(f'{version}: MISSING groups {missing} (have 0..{gmax}); '
          f'refusing to write -- resubmit those groups first')
    sys.exit(1)

fragment_files = [idx_file[g] for g in range(gmax + 1)]

# Every other consumer in this repo guards its inputs; a concat that merges 64
# fragments without checking they agree on sim_version is the "silent SatGen
# version mixing" class. Concretely reachable here: this tree can hold a mix of
# locally regenerated and migrated fragments.
provenance.assert_single_version(fragment_files, expected=version)

with get_h5(version) as f:
    n_halos = f['virial_mass'].shape[0]

DATA_KEYS = ('massProfile', 'mpeakProfile', 'vcircProfile')

with np.load(fragment_files[0]) as z0:
    r_grid_ref = z0['r_grid']
    n_r = r_grid_ref.shape[0]
    out_dtype = z0['massProfile'].dtype
    # Chunk size derived from fragment 0 itself, not from a provenance field
    # -- stamp-independent, so it applies uniformly whether or not a fragment
    # was ever stamped with halo_slice.
    C = z0['massProfile'].shape[0]

out = {k: np.empty((n_halos, n_r), dtype=out_dtype) for k in DATA_KEYS}

coverage_verified = True
stop = 0
for g in range(gmax + 1):
    start = min(g * C, n_halos)
    stop = min((g + 1) * C, n_halos)

    record = provenance.read(idx_file[g])
    halo_slice = record.get('halo_slice') if record is not None else None
    if halo_slice is None:
        coverage_verified = False
    elif tuple(halo_slice) != (start, stop):
        print(f'{version} group {g}: halo_slice {tuple(halo_slice)} does not '
              f'match the derived bounds ({start}, {stop}); refusing to write')
        sys.exit(1)

    with np.load(fragment_files[g]) as arr:
        if not np.array_equal(arr['r_grid'], r_grid_ref):
            raise ValueError(f'{version} group {g}: r_grid differs from group 0 '
                             f'({fragment_files[g]})')
        n_frag = arr['massProfile'].shape[0]
        if n_frag != stop - start:
            print(f'{version} group {g}: has {n_frag} rows, expected '
                  f'{stop - start} (derived bounds [{start}, {stop}) from a '
                  f'chunk size of {C} at group 0); refusing to write')
            sys.exit(1)
        for k in DATA_KEYS:
            # Assigning into a preallocated array casts silently, where
            # np.concatenate would promote: a float32 fragment would be
            # accepted and quietly lose precision against float64 neighbours.
            if arr[k].dtype != out_dtype:
                raise ValueError(
                    f'{version} group {g}: {k} has dtype {arr[k].dtype}, '
                    f'expected {out_dtype} (from group 0)')
            # Derived bounds here are equivalent to a running offset for a
            # correctly-sized fragment -- this only enforces the per-group
            # row count (catches a wrong CHUNK_SIZE, truncation, or a missing
            # tail). It does NOT catch a transposed pair of same-size
            # fragments; only the halo_slice cross-check above does that, and
            # only when coverage_verified is True.
            out[k][start:stop] = arr[k]

if stop != n_halos:
    print(f'{version}: groups cover up to halo {stop}, but the catalog has '
          f'{n_halos}; refusing to write')
    sys.exit(1)

record = provenance.stamp('python/concat_massProfile.py', version=version,
                          argv=sys.argv[1:], inputs=fragment_files,
                          n_groups=gmax + 1, n_halos=int(stop),
                          coverage_verified=coverage_verified)
provenance.savez(config.PAPER_MASSPROFILE_DIR / version / 'massProfile',
                 record, r_grid=r_grid_ref, **out)
print(f'{version}: {gmax + 1} groups -> {stop} halos  '
      f'keys={list(DATA_KEYS) + ["r_grid"]}  coverage_verified={coverage_verified}')
