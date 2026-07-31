"""Concatenate per-group halo_Js-<g>.npz files into a single halo_Js.npz per dwarf.

Replaces the manual merge cell in jupyter/JFactorPlots.ipynb. For each dwarf it
globs results/paper_Js/<version>/<dwarf>/halo_Js/halo_Js-*.npz, concatenates
every stored array (green_Js, theta95, ...) in numeric group order, and writes
results/paper_Js/<version>/<dwarf>/halo_Js.npz.

It is chunk-size agnostic (reads group indices from filenames) and REFUSES to
write a dwarf whose groups are not a contiguous 0..max set -- a missing middle
group would silently shift every downstream halo index out of alignment.

Usage:
    python concat_Js.py <tree> [dwarf]
e.g. python concat_Js.py Diemer
     python concat_Js.py Diemer "Antlia II"
     python concat_Js.py halo_position Diemer     # Jhalopos output

<tree> is normally a SatGen version, but for the Jhalopos products it is the
literal 'halo_position' and the second argument names the version. The catalog
to check coverage against is therefore read from the fragments' own stamps, not
from <tree>.
"""
import sys, os, glob
import numpy as np

import config
import provenance

BASE = str(config.PAPER_JS_DIR)

version = sys.argv[1]
only = sys.argv[2] if len(sys.argv) > 2 else None
version_dir = os.path.join(BASE, version)

if only:
    dwarfs = [only]
else:
    dwarfs = sorted(d for d in os.listdir(version_dir)
                    if os.path.isdir(os.path.join(version_dir, d, 'halo_Js')))

def catalog_version_of(fragment_files):
    """SatGen version these fragments came from, per their own stamps.

    Resolved from the fragments rather than the CLI argument, so it is correct
    for paper_Js/halo_position/<version>/, where the first path component is
    not a SatGen version. Falls back to the CLI argument for legacy fragments.
    """
    for path in fragment_files:
        record = provenance.read(path)
        if record and record.get('sim_version'):
            return record['sim_version']
    # Legacy fragments carry no stamp. Fall back to the CLI argument only if it
    # is a real version -- for the halo_position tree it is not, and inventing
    # one would be rejected by provenance.stamp as a typo.
    return version if version in config.H5_REGISTRY else None


def catalog_length(catalog_version):
    """Halo count of that catalog, or None if it is not available."""
    import h5py

    if catalog_version is None:
        print('catalog version unknown (unstamped fragments); '
              'checking group contiguity only')
        return None
    try:
        with h5py.File(config.h5_path(catalog_version), 'r') as handle:
            return handle['virial_mass'].shape[0]
    except Exception as exc:                   # noqa: BLE001 - advisory only
        print(f'catalog for {catalog_version!r} unavailable ({exc}); '
              f'checking group contiguity only')
        return None

had_error = False
for dwarf in dwarfs:
    gdir = os.path.join(version_dir, dwarf, 'halo_Js')
    files = glob.glob(os.path.join(gdir, 'halo_Js-*.npz'))
    if not files:
        print(f'{dwarf}: no per-group files, skipping')
        if only:                      # an explicitly-named dwarf with no files is an error
            had_error = True
        continue

    # map group index -> file (index parsed from 'halo_Js-<g>.npz')
    idx_file = {int(os.path.basename(f)[len('halo_Js-'):-len('.npz')]): f for f in files}
    gmax = max(idx_file)
    missing = [g for g in range(gmax + 1) if g not in idx_file]
    if missing:
        print(f'{dwarf}: MISSING groups {missing} (have 0..{gmax}); '
              f'refusing to write -- resubmit those groups first')
        had_error = True
        continue

    fragment_files = [idx_file[g] for g in range(gmax + 1)]
    catalog_version = catalog_version_of(fragment_files)
    n_halos = catalog_length(catalog_version)

    # Contiguous labels 0..gmax do not imply the groups COVER the catalog: a
    # missing trailing group, or a rerun after GROUPS changed, leaves a short
    # or misaligned array that loads and plots fine. Each fragment records the
    # halo slice it computed, so check the slices tile [0, n_halos) exactly.
    if n_halos is not None:
        slices = []
        for g in range(gmax + 1):
            record = provenance.read(idx_file[g])
            if record is None or record.get('halo_slice') is None:
                slices = None       # legacy fragment, predates the stamp
                break
            slices.append(tuple(record['halo_slice']))
        if slices is None:
            print(f'{dwarf}: fragments predate slice stamping; '
                  f'coverage not verified')
        else:
            covered, ok = 0, True
            for start, stop in sorted(slices):
                if start != covered:
                    ok = False
                    break
                covered = stop
            if not ok or covered != n_halos:
                print(f'{dwarf}: groups do not tile the catalog '
                      f'(covered {covered} of {n_halos}); refusing to write')
                had_error = True
                continue

    with np.load(idx_file[0]) as z0:
        # Skip the provenance stamp: it is a 0-d string, not a per-halo array,
        # so concatenating it would corrupt the output and lose the chain.
        keys = [k for k in z0.keys() if k != provenance.PROVENANCE_KEY]
    out = {k: [] for k in keys}
    for g in range(gmax + 1):
        with np.load(idx_file[g]) as arr:
            # Compare on the data keys only; the provenance stamp is excluded
            # from `keys` above and is per-fragment, not per-halo.
            arr_keys = {k for k in arr.keys() if k != provenance.PROVENANCE_KEY}
            if arr_keys != set(keys):          # guard against key-set drift across groups
                raise ValueError(f'{dwarf} group {g}: keys {sorted(arr_keys)} '
                                 f'!= group 0 keys {sorted(keys)}')
            for k in keys:
                out[k].append(arr[k])
    out = {k: np.concatenate(v) for k, v in out.items()}

    n = len(out[keys[0]])
    # Always-available coverage check: works for legacy fragments too, which
    # carry no slice stamp. Catches a dropped trailing group, which contiguity
    # alone cannot.
    if n_halos is not None and n != n_halos:
        print(f'{dwarf}: concatenation has {n} halos but the {catalog_version} catalog '
              f'has {n_halos}; refusing to write')
        had_error = True
        continue

    record = provenance.stamp('python/concat_Js.py', version=catalog_version,
                              argv=sys.argv[1:], tree=version,
                              inputs=[idx_file[g] for g in range(gmax + 1)],
                              dwarf=dwarf, n_groups=gmax + 1, n_halos=int(n))
    provenance.savez(os.path.join(version_dir, dwarf, 'halo_Js'), record, **out)
    print(f'{dwarf}: {gmax + 1} groups -> {n} halos  keys={keys}')

if had_error:
    sys.exit(1)
