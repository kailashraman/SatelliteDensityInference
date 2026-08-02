"""Regression test for the sticky-`verified` behaviour of stamp_migrated.py.

A bulk re-stamp must never downgrade an existing `verified: true` sidecar to
`false`: this stamper never performs the upstream diff itself, so a lost
`verified` flag cannot be reconstructed by rerunning it. This was discovered
the hard way -- an early bulk re-stamp of results/paper_Js silently downgraded
the one genuinely verified product (`Diemer/Antlia II`) -- see
docs/migration-notes.md.

Requires the J_calc environment: scripts/stamp_migrated.py imports Jdata,
which needs astropy. See conftest.py's `needs_satgen` marker.

A bare `import stamp_migrated` at module scope would raise ImportError before
the `needs_satgen` marker gets a chance to skip -- markers only take effect
after collection, so a transitive ImportError here aborts the whole test
session instead of skipping this file. `pytest.importorskip` turns that
ImportError into a skip, matching the pattern in test_jdata_module.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

import config
import provenance

pytestmark = pytest.mark.needs_satgen

sys.path.insert(0, str(config.REPO_ROOT / 'scripts'))
sm = pytest.importorskip('stamp_migrated', reason='needs the J_calc environment')


def test_verification_status_carries_forward_prior_verified(tmp_path):
    """Fails against the pre-fix code, which recomputed `is_verified` from
    this run's --verified list alone and had no way to see a prior true."""
    product = tmp_path / 'halo_Js.npz'
    record = provenance.stamp('scripts/stamp_migrated.py', version='Diemer',
                              migrated_from='/old/tree/halo_Js.npz',
                              verified=True,
                              note='copied from SatGen_Dwarf; diffed against '
                                   'upstream: all arrays bit-exact')
    provenance.stamp_existing(product, record)

    # Re-stamp WITHOUT re-verifying (is_verified=False from this run's own
    # --verified list) -- the prior true must survive, note included.
    is_verified, note, _ = sm._verification_status(product, False)
    assert is_verified is True
    assert note == 'copied from SatGen_Dwarf; diffed against upstream: all arrays bit-exact'


def test_verification_status_explicit_reverify_overrides_note(tmp_path):
    """This run's own --verified is an explicit re-verification: it writes
    the fresh default note, not whatever custom note a prior pass left."""
    product = tmp_path / 'halo_Js.npz'
    record = provenance.stamp('scripts/stamp_migrated.py', version='Diemer',
                              migrated_from='/old/tree/halo_Js.npz',
                              verified=True, note='some earlier custom note')
    provenance.stamp_existing(product, record)

    is_verified, note, _ = sm._verification_status(product, True)
    assert is_verified is True
    assert note == 'copied from SatGen_Dwarf; diffed against upstream: all arrays bit-exact'


def test_verification_status_defaults_to_unverified_when_nothing_prior(tmp_path):
    product = tmp_path / 'halo_Js.npz'
    is_verified, note, fp = sm._verification_status(product, False)
    assert is_verified is False
    assert note == 'copied from SatGen_Dwarf; not individually verified'
    assert fp is None


def test_verification_status_downgrades_when_product_bytes_change(tmp_path):
    """Fails against the pre-fix code, which carried `verified: true` forward
    with no tie to the product's own bytes: replacing the file underneath a
    verified flag would survive re-stamping, re-creating -- one layer up --
    exactly the false-verification failure this stamper exists to prevent."""
    product = tmp_path / 'halo_Js.npz'
    np.savez(product, some_array=np.arange(5))

    is_verified, note, fp = sm._verification_status(product, True)
    assert is_verified is True
    record = provenance.stamp('scripts/stamp_migrated.py', version='Diemer',
                              migrated_from='/old/tree/halo_Js.npz',
                              verified=True, note=note, verified_fingerprint=fp)
    provenance.stamp_existing(product, record)

    # The product's bytes change underneath the sidecar (a re-copy, a botched
    # edit, ...); re-stamping WITHOUT re-verifying must not carry the flag.
    np.savez(product, some_array=np.arange(500))
    is_verified, note, fp = sm._verification_status(product, False)
    assert is_verified is False
    assert note == 'copied from SatGen_Dwarf; not individually verified'


def test_verification_status_treats_missing_note_as_malformed(tmp_path):
    """A `verified: true` sidecar with no note names no statement of what was
    checked -- not carrying it forward, rather than propagating an
    unsubstantiated claim."""
    product = tmp_path / 'halo_Js.npz'
    record = provenance.stamp('scripts/stamp_migrated.py', version='Diemer',
                              migrated_from='/old/tree/halo_Js.npz',
                              verified=True, note=None)
    provenance.stamp_existing(product, record)

    is_verified, note, fp = sm._verification_status(product, False)
    assert is_verified is False
    assert note == 'copied from SatGen_Dwarf; not individually verified'
