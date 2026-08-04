"""concat_massProfile guards.

massProfile.py writes one fragment per halo group; concat_massProfile.py
merges them into massProfile.npz, the input vcirc_scatter_300pc.py reads. The
fragments are positional slices of the halo catalog (chunk size derived from
fragment 0's row count), so a fragment with the wrong row count, or one whose
halo_slice provenance disagrees with where its filename places it, does not
shorten the output harmlessly -- it misaligns every halo from that point on.
"""

import runpy
import sys

import h5py
import numpy as np
import pytest

import config
import provenance

N_R = 4
R_GRID = np.arange(N_R, dtype=float)
VERSION = 'synthetic'


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """Register a throwaway SatGen version backed by a real (tiny) h5 file.

    concat_massProfile.py's coverage check is mandatory -- unlike concat_Js.py
    it has no "catalog unavailable" fallback -- so the fixture must provide a
    real h5 with a `virial_mass` dataset, not merely register the version.
    """
    monkeypatch.setattr(config, 'ADDITIONAL_DIR', tmp_path)
    monkeypatch.setitem(config.H5_REGISTRY, VERSION, f'{VERSION}.h5')

    def _make(n_halos):
        with h5py.File(tmp_path / f'{VERSION}.h5', 'w') as f:
            f.create_dataset('virial_mass', data=np.ones(n_halos))
        out_dir = tmp_path / 'paper_massProfile'
        monkeypatch.setattr(config, 'PAPER_MASSPROFILE_DIR', out_dir)
        return out_dir

    return _make


def _fragment(directory, group, n, start, halo_slice=None, dtype=np.float64,
             r_grid=None):
    """A stand-in massProfile-<g>.npz fragment.

    Row i holds the constant value `start + i` across every column, so a row
    landing anywhere but its derived slice is detectable from the array alone.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rows = np.arange(start, start + n, dtype=dtype)
    massProfile = np.tile(rows[:, None], (1, N_R))
    extra = {} if halo_slice is None else {'halo_slice': list(halo_slice)}
    record = provenance.stamp('python/massProfile.py', version=VERSION, **extra)
    return provenance.savez(
        directory / f'massProfile-{group}', record,
        r_grid=(r_grid if r_grid is not None else R_GRID).astype(dtype),
        massProfile=massProfile,
        mpeakProfile=massProfile + 1000,
        vcircProfile=massProfile + 2000)


def _run():
    argv = sys.argv
    sys.argv = ['concat_massProfile.py', VERSION]
    try:
        runpy.run_path(str(config.REPO_ROOT / 'python' / 'concat_massProfile.py'),
                       run_name='__main__')
    finally:
        sys.argv = argv


def _out_path(out_dir):
    return out_dir / VERSION / 'massProfile.npz'


def test_happy_path_places_rows_at_derived_bounds(catalog):
    out_dir = catalog(15)
    gdir = out_dir / VERSION / 'massProfile'
    for group in range(3):
        _fragment(gdir, group, n=5, start=group * 5)

    _run()

    with np.load(_out_path(out_dir)) as data:
        assert np.array_equal(data['massProfile'][:, 0], np.arange(15, dtype=float))
        assert np.array_equal(data['mpeakProfile'][:, 0], np.arange(15, dtype=float) + 1000)
        assert np.array_equal(data['vcircProfile'][:, 0], np.arange(15, dtype=float) + 2000)

    record = provenance.read(_out_path(out_dir))
    assert record['n_groups'] == 3 and record['n_halos'] == 15
    assert record['coverage_verified'] is False, \
        'no fragment carried halo_slice -- must not claim it was verified'


def test_coverage_verified_true_when_halo_slice_present_and_correct(catalog):
    out_dir = catalog(15)
    gdir = out_dir / VERSION / 'massProfile'
    for group in range(3):
        start = group * 5
        _fragment(gdir, group, n=5, start=start, halo_slice=(start, start + 5))

    _run()

    record = provenance.read(_out_path(out_dir))
    assert record['coverage_verified'] is True


def test_refuses_a_missing_middle_group(catalog):
    out_dir = catalog(15)
    gdir = out_dir / VERSION / 'massProfile'
    for group in (0, 2):
        _fragment(gdir, group, n=5, start=group * 5)

    with pytest.raises(SystemExit) as excinfo:
        _run()
    assert excinfo.value.code == 1
    assert not _out_path(out_dir).exists(), \
        'wrote a misaligned concatenation instead of refusing'


def test_refuses_a_fragment_whose_row_count_disagrees_with_derived_chunking(catalog):
    """Fragment 0 sets the chunk size C=5; a same-total-length but
    differently-chunked group 1 must not simply be absorbed by a running
    offset -- it must be checked against its OWN derived [start, stop)."""
    out_dir = catalog(15)
    gdir = out_dir / VERSION / 'massProfile'
    _fragment(gdir, 0, n=5, start=0)
    _fragment(gdir, 1, n=7, start=5)          # should have been 5 rows, not 7
    _fragment(gdir, 2, n=3, start=12)

    with pytest.raises(SystemExit) as excinfo:
        _run()
    assert excinfo.value.code == 1
    assert not _out_path(out_dir).exists()


def test_refuses_mismatched_dtype(catalog):
    out_dir = catalog(10)
    gdir = out_dir / VERSION / 'massProfile'
    _fragment(gdir, 0, n=5, start=0, dtype=np.float64)
    _fragment(gdir, 1, n=5, start=5, dtype=np.float32)

    with pytest.raises(ValueError, match='dtype'):
        _run()
    assert not _out_path(out_dir).exists()


def test_refuses_mismatched_r_grid(catalog):
    out_dir = catalog(10)
    gdir = out_dir / VERSION / 'massProfile'
    _fragment(gdir, 0, n=5, start=0)
    _fragment(gdir, 1, n=5, start=5, r_grid=R_GRID * 2)

    with pytest.raises(ValueError, match='r_grid'):
        _run()
    assert not _out_path(out_dir).exists()


def test_refuses_halo_slice_disagreeing_with_derived_bounds(catalog):
    """Perturbs the STAMP, not the data: a fragment's own halo_slice stamp
    says it covers different halos than its filename's group index places it
    at, and the fragment DATA is still at its correctly-derived rows. This is
    the cross-check catching a wrong/stale halo_slice field -- it does not by
    itself demonstrate that a transposed *data* swap between two same-size
    fragments is caught (see
    test_transposed_same_size_fragment_data_is_not_detected below for that
    gap, which this check does NOT cover when halo_slice is absent)."""
    out_dir = catalog(15)
    gdir = out_dir / VERSION / 'massProfile'
    _fragment(gdir, 0, n=5, start=0, halo_slice=(0, 5))
    _fragment(gdir, 1, n=5, start=5, halo_slice=(10, 15))   # wrong -- should be (5, 10)
    _fragment(gdir, 2, n=5, start=10, halo_slice=(10, 15))

    with pytest.raises(SystemExit) as excinfo:
        _run()
    assert excinfo.value.code == 1
    assert not _out_path(out_dir).exists()


def test_transposed_same_size_fragment_data_is_not_detected(catalog):
    """Known gap, not a passing guarantee: when no fragment carries
    halo_slice, concat_massProfile.py has no way to detect that two same-size
    fragments' DATA was swapped between filenames -- the derived-bounds
    row-count check passes identically either way (a running offset and the
    derived bounds agree for correctly-sized fragments), and there is no
    other check when halo_slice is absent. This test pins that gap so it is
    visible in the suite rather than documented only in a comment. It is
    expected to keep passing (i.e. the swap stays silently undetected) unless
    a same-size ordering check is added elsewhere."""
    out_dir = catalog(10)
    gdir = out_dir / VERSION / 'massProfile'
    # Group 0's fragment holds what should be group 1's rows (start=5), and
    # group 1's fragment holds what should be group 0's rows (start=0) -- a
    # stand-in for two same-size files whose names were swapped.
    _fragment(gdir, 0, n=5, start=5)
    _fragment(gdir, 1, n=5, start=0)

    _run()

    with np.load(_out_path(out_dir)) as data:
        # Honest current behaviour: rows land at the derived [0,5)/[5,10)
        # bounds regardless of which file actually held which content, so
        # the output is silently misaligned instead of being refused.
        assert np.array_equal(
            data['massProfile'][:, 0],
            np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], dtype=float))

    record = provenance.read(_out_path(out_dir))
    assert record['coverage_verified'] is False
