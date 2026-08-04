#!/usr/bin/env python
"""Report derived products that are unstamped or that mix SatGen versions.

    python scripts/check_provenance.py [subdir ...]

With no argument, walks all of results/. Exit status is 1 if anything is
unstamped, so this can gate a commit.
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python'))

import config
import provenance

# Per-task fragments written by an array job, e.g.
# results/paper_Js/<version>/<dwarf>/halo_Js/halo_Js-7.npz. They are inputs to
# an aggregation step, not products in their own right, and the aggregate's
# stamp records each of them by path. Counted and reported, not required to
# carry their own sidecar -- there can be tens of thousands.
SHARD_PARENTS = {'halo_Js', 'massProfile'}


def is_shard(path):
    return path.parent.name in SHARD_PARENTS and path.stem != path.parent.name


def is_leftover_temp(path):
    """A dotfile mkstemp() orphan from a killed provenance.savez()/sidecar
    write (SIGKILL/OOM and default SIGTERM cannot run the cleanup
    `except BaseException` in provenance.py) -- evidence of a killed run,
    reported so a person sees it. Never deleted here."""
    return path.name.startswith('.') and '.tmp' in path.name


def main(argv):
    # data/additional is included by default: the h5 catalogs are the root of
    # the DAG, and an unstamped one is invisible in every product below it.
    roots = ([Path(a) for a in argv[1:]]
             or [config.RESULTS_DIR, config.ADDITIONAL_DIR])

    unstamped = []
    by_version = defaultdict(list)
    migrated = []
    unreadable = []
    shards = 0
    leftover_temps = []

    for root in roots:
        if not root.exists():
            print(f'skipping {root} (does not exist)')
            continue
        # Unlike glob.glob() (what the concat scripts use to find fragments),
        # pathlib's rglob('*.npz') does NOT skip dotfiles -- so a leftover
        # temp (named with a leading dot, see provenance.py) still shows up
        # in the products walk below and must be filtered out explicitly,
        # rather than falling out of the pattern the way it would for
        # glob.glob().
        leftover_temps.extend(sorted(
            p for p in root.rglob('.*') if p.is_file() and is_leftover_temp(p)))
        products = sorted(p for pattern in ('*.npz', '*.h5')
                          for p in root.rglob(pattern))
        for path in products:
            if path.name.endswith(provenance.SIDECAR_SUFFIX):
                continue
            if is_leftover_temp(path):
                continue
            if is_shard(path):
                shards += 1
                continue
            try:
                record = provenance.read(path)
            except Exception as exc:
                # A truncated .npz is exactly what a cancelled or OOM-killed
                # array task leaves behind, and it is the thing this sweep most
                # needs to report. Crashing on the first one instead hides
                # every product after it in the walk -- the sweep stops being
                # an inventory and becomes an assertion about file zero.
                unreadable.append((path, exc))
                continue
            if record is None:
                unstamped.append(path)
                continue
            by_version[record.get('sim_version')].append(path)
            if record.get('migrated_from'):
                migrated.append(path)

    total = sum(len(v) for v in by_version.values()) + len(unstamped)
    print(f'{total} product(s) under {", ".join(str(r) for r in roots)}')

    for version, paths in sorted(by_version.items(), key=lambda kv: str(kv[0])):
        print(f'  {version or "<no version>"}: {len(paths)}')
    if migrated:
        print(f'  ({len(migrated)} copied in from SatGen_Dwarf, not recomputed here)')
    if shards:
        print(f'  ({shards} array-task fragments, covered by their aggregate)')

    if unreadable:
        print(f'\n{len(unreadable)} unreadable product(s) -- truncated or '
              'corrupt, typically a cancelled or OOM-killed array task:')
        for path, exc in unreadable[:20]:
            print(f'  {path}: {type(exc).__name__}: {exc}')
        if len(unreadable) > 20:
            print(f'  ... and {len(unreadable) - 20} more')

    if unstamped:
        print(f'\n{len(unstamped)} unstamped product(s):')
        for path in unstamped[:20]:
            print(f'  {path}')
        if len(unstamped) > 20:
            print(f'  ... and {len(unstamped) - 20} more')

    if leftover_temps:
        print(f'\n{len(leftover_temps)} leftover temp file(s) -- orphaned '
              'mkstemp() output from a killed run (SIGKILL/OOM or default '
              'SIGTERM skip the cleanup handler); safe to inspect, not '
              'deleted by this sweep:')
        for path in leftover_temps[:20]:
            print(f'  {path}')
        if len(leftover_temps) > 20:
            print(f'  ... and {len(leftover_temps) - 20} more')

    if unstamped or unreadable:
        return 1

    print('\nall products stamped')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
