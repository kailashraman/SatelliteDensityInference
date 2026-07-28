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


def main(argv):
    # data/additional is included by default: the h5 catalogs are the root of
    # the DAG, and an unstamped one is invisible in every product below it.
    roots = ([Path(a) for a in argv[1:]]
             or [config.RESULTS_DIR, config.ADDITIONAL_DIR])

    unstamped = []
    by_version = defaultdict(list)
    migrated = []

    for root in roots:
        if not root.exists():
            print(f'skipping {root} (does not exist)')
            continue
        products = sorted(p for pattern in ('*.npz', '*.h5')
                          for p in root.rglob(pattern))
        for path in products:
            if path.name.endswith(provenance.SIDECAR_SUFFIX):
                continue
            record = provenance.read(path)
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

    if unstamped:
        print(f'\n{len(unstamped)} unstamped product(s):')
        for path in unstamped[:20]:
            print(f'  {path}')
        if len(unstamped) > 20:
            print(f'  ... and {len(unstamped) - 20} more')
        return 1

    print('\nall products stamped')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
