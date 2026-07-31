#!/usr/bin/env python
"""Stamp intermediates copied in from SatGen_Dwarf.

    python scripts/stamp_migrated.py paper_Js [--verified "Diemer/Antlia II"]

Exists as a tracked script rather than an ad-hoc snippet because the stamps it
writes are the provenance record: a sidecar produced by something not in the
repository is a claim nobody can re-derive.

Two rules it enforces, both learned the hard way:

* The producer recorded is the script that actually produces that product, not
  the one that produces its neighbours. `paper_Js/<version>/<dwarf>/halo_Js.npz`
  comes from Jdwarf.py + concat_Js.py; `paper_Js/halo_position/halo_Js.npz`
  comes from Jhalopos.py, which is NOT migrated -- it evaluates J at each
  halo's own position rather than at a dwarf's, and its values differ.
* `verified` names exactly which products were diffed against upstream. A
  blanket "reproduced bit-exactly" across a bulk copy is an assertion about
  files nobody checked.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python'))

import config
import provenance

UPSTREAM = Path('/global/home/users/kraman/pscratch/SatGen_Dwarf')

# Products whose producer is not python/Jdwarf.py + python/concat_Js.py.
SPECIAL_PRODUCERS = {
    'halo_position': (
        'python/Jhalopos.py (NOT MIGRATED): J-factors of SatGen halos at their '
        'own heliocentric positions, in 5000-halo chunks. Not reproducible in '
        'this repository; see docs/migration-notes.md'),
    'Diemer_backup': (
        'python/Jdwarf.py + python/concat_Js.py, from an earlier run that wrote '
        'only green_Js and theta95. The current Jdwarf.py writes full_Js too, '
        'so it cannot reproduce these files key-for-key'),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('tree', choices=['paper_Js'])
    parser.add_argument('--verified', action='append', default=[],
                        help='relative path of a product diffed against '
                             'upstream, e.g. "Diemer/Antlia II"')
    args = parser.parse_args(argv)

    root = config.PAPER_JS_DIR
    verified = set(args.verified)
    counts = {'stamped': 0, 'verified': 0}

    for product in sorted(root.rglob('halo_Js.npz')):
        rel = product.relative_to(root)
        top = rel.parts[0]
        dwarf_key = str(Path(*rel.parts[:-1]))

        version = 'Diemer' if top in SPECIAL_PRODUCERS else top
        producer = SPECIAL_PRODUCERS.get(
            top, 'python/Jdwarf.py + python/concat_Js.py')

        is_verified = dwarf_key in verified
        note = ('copied from SatGen_Dwarf; '
                + ('diffed against upstream: all arrays bit-exact'
                   if is_verified else 'not individually verified'))

        record = provenance.stamp(
            'scripts/stamp_migrated.py', version=version,
            migrated_from=str(UPSTREAM / 'results' / args.tree / rel),
            produced_by=producer, verified=is_verified, note=note)
        provenance.stamp_existing(product, record)
        counts['stamped'] += 1
        counts['verified'] += int(is_verified)

    print(f"stamped {counts['stamped']} products "
          f"({counts['verified']} verified against upstream)")
    if counts['verified'] == 0:
        print('WARNING: nothing was verified; every stamp says so')
    return 0


if __name__ == '__main__':
    sys.exit(main())
