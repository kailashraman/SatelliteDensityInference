"""concat_Js guards.

Jdwarf writes one fragment per (dwarf, halo group); every downstream consumer
reads the concatenation. The fragments are positional slices of the halo
catalog, so a missing middle group does not shorten the array harmlessly -- it
shifts every halo after the gap out of alignment with the h5, silently.
"""

import runpy
import sys

import numpy as np
import pytest

import config
import provenance


def _fragment(directory, group, n, start, version=None):
    """A stand-in Jdwarf fragment.

    `version` is left unset by default: provenance.stamp rejects a version that
    is not in the registry, deliberately, so a fixture must not invent one. The
    directory name is what concat_Js keys on.
    """
    directory.mkdir(parents=True, exist_ok=True)
    record = provenance.stamp('python/Jdwarf.py', version=version,
                              group=group, halo_slice=[start, start + n])
    return provenance.savez(
        directory / f'halo_Js-{group}', record,
        green_Js=np.arange(start, start + n, dtype=float),
        theta95=np.arange(start, start + n, dtype=float) + 0.5,
        full_Js=np.arange(start, start + n, dtype=float) * 2)


def _run(tmp_path, *args):
    """Invoke concat_Js as a script with PAPER_JS_DIR pointed at tmp_path.

    The version named in `args` is registered for the duration as a catalog
    that exists but was not migrated. That is what makes a throwaway version
    usable: config.h5_path then raises a 'not usable' KeyError, which
    concat_Js treats as "catalog unavailable, check contiguity only" and
    provenance.stamp tolerates -- whereas an unregistered version is rejected
    outright, deliberately, as a typo.
    """
    import config as cfg
    version = args[0]
    original = cfg.PAPER_JS_DIR
    registered = version in cfg.H5_REGISTRY
    cfg.PAPER_JS_DIR = tmp_path
    if not registered:
        cfg.H5_REGISTRY[version] = None
        cfg.NOT_MIGRATED[version] = 'test fixture'
    argv = sys.argv
    sys.argv = ['concat_Js.py', *args]
    try:
        runpy.run_path(str(config.REPO_ROOT / 'python' / 'concat_Js.py'),
                       run_name='__main__')
    finally:
        cfg.PAPER_JS_DIR = original
        sys.argv = argv
        if not registered:
            cfg.H5_REGISTRY.pop(version, None)
            cfg.NOT_MIGRATED.pop(version, None)


def test_concatenates_groups_in_numeric_order(tmp_path):
    """Filename order is lexicographic, so halo_Js-10 sorts before halo_Js-2."""
    gdir = tmp_path / 'synthetic' / 'Draco' / 'halo_Js'
    for group in range(12):
        _fragment(gdir, group, n=5, start=group * 5)

    _run(tmp_path, 'synthetic', 'Draco')

    with np.load(tmp_path / 'synthetic' / 'Draco' / 'halo_Js.npz') as data:
        # All three arrays: a constant fixture column could not detect
        # misordering at all.
        assert np.array_equal(data['green_Js'], np.arange(60, dtype=float))
        assert np.array_equal(data['theta95'], np.arange(60, dtype=float) + 0.5)
        assert np.array_equal(data['full_Js'], np.arange(60, dtype=float) * 2)


def test_refuses_a_missing_middle_group(tmp_path):
    """The important guard: a gap misaligns every halo after it."""
    gdir = tmp_path / 'synthetic' / 'Draco' / 'halo_Js'
    for group in (0, 1, 3):
        _fragment(gdir, group, n=5, start=group * 5)

    with pytest.raises(SystemExit) as excinfo:
        _run(tmp_path, 'synthetic', 'Draco')
    assert excinfo.value.code == 1
    assert not (tmp_path / 'synthetic' / 'Draco' / 'halo_Js.npz').exists(), \
        'wrote a misaligned concatenation instead of refusing'


def test_rejects_key_set_drift_between_groups(tmp_path):
    gdir = tmp_path / 'synthetic' / 'Draco' / 'halo_Js'
    _fragment(gdir, 0, n=5, start=0)
    provenance.savez(gdir / 'halo_Js-1',
                     provenance.stamp('python/Jdwarf.py'),
                     green_Js=np.arange(5.))          # missing theta95/full_Js

    with pytest.raises(ValueError, match='keys'):
        _run(tmp_path, 'synthetic', 'Draco')


def test_provenance_stamp_is_not_concatenated(tmp_path):
    """The stamp is a 0-d string; concatenating it would corrupt the output
    and destroy the chain back to the h5."""
    gdir = tmp_path / 'synthetic' / 'Draco' / 'halo_Js'
    for group in range(3):
        _fragment(gdir, group, n=4, start=group * 4)

    _run(tmp_path, 'synthetic', 'Draco')

    out = tmp_path / 'synthetic' / 'Draco' / 'halo_Js.npz'
    with np.load(out) as data:
        assert sorted(k for k in data.files if k != provenance.PROVENANCE_KEY) \
            == ['full_Js', 'green_Js', 'theta95']
        assert data['green_Js'].shape == (12,)

    record = provenance.read(out)
    assert record['script'] == 'python/concat_Js.py'
    assert record['n_groups'] == 3 and record['n_halos'] == 12
    assert len(record['inputs']) == 3
    assert record['inputs'][0]['provenance']['script'] == 'python/Jdwarf.py'


def test_named_dwarf_with_no_fragments_is_an_error(tmp_path):
    (tmp_path / 'synthetic' / 'Draco' / 'halo_Js').mkdir(parents=True)
    with pytest.raises(SystemExit) as excinfo:
        _run(tmp_path, 'synthetic', 'Draco')
    assert excinfo.value.code == 1


@pytest.mark.needs_data
def test_published_concatenations_match_their_catalog_lengths():
    """Every halo_Js.npz must have one entry per halo in its catalog.

    This is what a dropped or duplicated group would break, and it is
    invisible in any figure.
    """
    import h5py

    expected = {}
    for version in ('Diemer', 'Diemer_0p12', 'Zhao', 'm2e12', 'mass_floor_7'):
        path = config.PAPER_JS_DIR / version
        if not path.is_dir():
            continue
        with h5py.File(config.h5_path(version), 'r') as f:
            expected[version] = f['virial_mass'].shape[0]

    assert expected, 'no paper_Js versions migrated'
    for version, n_halos in expected.items():
        found = 0
        for dwarf_dir in sorted((config.PAPER_JS_DIR / version).iterdir()):
            product = dwarf_dir / 'halo_Js.npz'
            if not product.exists():
                continue
            found += 1
            with np.load(product) as data:
                assert data['green_Js'].shape[0] == n_halos, \
                    f'{version}/{dwarf_dir.name}: {data["green_Js"].shape[0]} != {n_halos}'
        # Without this the test passes if every product vanished.
        assert found == 43, f'{version}: found {found} products, expected 43'


# --------------------------------------------------------------------------
# Jdwarf's partition arithmetic. Not imported from Jdwarf.py: that module runs
# its whole pipeline at import (it is a script, not a library), so the formula
# is restated here and pinned against the real catalog.
# --------------------------------------------------------------------------

def _chunk_slices(n, groups):
    """The slicing in python/Jdwarf.py: ceil division, last group truncated."""
    chunk = -(-n // groups)
    return [(g * chunk, min((g + 1) * chunk, n)) for g in range(groups)]


@pytest.mark.parametrize('n', [1, 19, 20, 21, 100, 2432159, 4142592])
def test_group_slices_tile_the_catalog_exactly(n):
    """Every halo computed once: no gap, no overlap, nothing past the end.

    A gap here does not shorten the result harmlessly -- it shifts every halo
    after it out of alignment with the h5, and no figure would show it.
    """
    groups = 20
    slices = _chunk_slices(n, groups)

    covered = 0
    for start, stop in slices:
        # numpy reads start > stop as an empty slice, which is what the
        # trailing groups are when n < groups. Normalise before checking.
        stop = max(start, stop)
        if stop == start:
            continue
        assert start == covered, f'gap or overlap at {start} (covered {covered})'
        covered = stop
    assert covered == n, f'covered {covered} of {n}'
    assert all(stop <= n for _, stop in slices)


def test_group_slices_match_the_published_diemer_partition():
    """19 full groups plus a short tail, summing to the catalog."""
    slices = _chunk_slices(2432159, 20)
    widths = [stop - start for start, stop in slices]
    assert widths[:19] == [121608] * 19
    assert widths[19] == 121607
    assert sum(widths) == 2432159


@pytest.mark.needs_data
def test_coverage_guard_refuses_fragments_that_do_not_tile(tmp_path):
    """Contiguous labels 0..max do not prove the groups cover the catalog.

    Uses the real 'Diemer' version, so concat_Js compares against the actual
    catalog length and must refuse this 60-halo stand-in.
    """
    gdir = tmp_path / 'Diemer' / 'Draco' / 'halo_Js'
    for group in range(12):
        _fragment(gdir, group, n=5, start=group * 5, version='Diemer')

    with pytest.raises(SystemExit) as excinfo:
        _run(tmp_path, 'Diemer', 'Draco')
    assert excinfo.value.code == 1
    assert not (tmp_path / 'Diemer' / 'Draco' / 'halo_Js.npz').exists()
