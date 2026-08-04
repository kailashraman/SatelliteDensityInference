"""concat_vcirc_scatter_300pc guards.

vcirc_scatter_300pc.py writes one CSV row per (dwarf, SHMR); this script
merges the rows for each SHMR into one scatter_300pc.txt, the file a plot
would read. It must refuse to write anything -- for any SHMR -- if any part
of the tree fails validation, and it must stamp the file it does write with
a provenance sidecar, which the bash predecessor never did.
"""

import runpy
import sys

import numpy as np
import pytest

import config
import provenance

pytest.importorskip('Jdata', reason='needs the J_calc environment')
import Jdata as obs  # noqa: E402  (import after importorskip, same as Jdata itself elsewhere)
import vcirc_scatter_params as vsp  # noqa: E402

pytestmark = pytest.mark.needs_satgen

SCRIPT_PATH = config.REPO_ROOT / 'python' / 'concat_vcirc_scatter_300pc.py'
VERSION = 'synthetic_concat_test'
SHMR_A = 'ShmrA'
SHMR_B = 'ShmrB'
DWARF_NAMES = ['DwarfAlpha', 'DwarfBeta', 'DwarfGamma']


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A throwaway version/SHMR/dwarf combination, isolated from the real tree.

    No h5 file needs to exist on disk -- provenance.stamp() only requires the
    version to be a registered H5_REGISTRY key with a non-None filename to
    fingerprint (fingerprint() itself tolerates a missing file).
    """
    base = tmp_path / 'vcirc_scatter_300pc' / VERSION
    monkeypatch.setattr(config, 'VCIRC_SCATTER_DIR', tmp_path / 'vcirc_scatter_300pc')
    monkeypatch.setattr(config, 'ADDITIONAL_DIR', tmp_path)
    monkeypatch.setitem(config.H5_REGISTRY, VERSION, f'{VERSION}.h5')
    monkeypatch.setattr(vsp, 'SHMR_NAMES', [SHMR_A, SHMR_B])
    monkeypatch.setattr(obs, 'dwarf_names', np.array(DWARF_NAMES))
    return base


def _row(base, shmr, dwarf_idx, dwarf_name, write_sidecar=True,
         omit_array_job_id_field=False, **stamp_overrides):
    per_dwarf = base / shmr / 'per_dwarf'
    per_dwarf.mkdir(parents=True, exist_ok=True)
    row_path = per_dwarf / f'row_{dwarf_idx:03d}.csv'
    with open(row_path, 'w') as f:
        f.write(f'{dwarf_name},1.0000,2.0000,3.0000,4.0000,5.0000,6.0000,7.00000\n')
    if write_sidecar:
        # By default the sidecar carries an EXPLICIT array_job_id=None (a
        # direct shell invocation), not the field's absence. Rows that
        # predate the field entirely (omit_array_job_id_field=True) never
        # pass array_job_id/array_task_id to stamp() at all, so the key is
        # genuinely missing from the record -- the case
        # record.get('array_job_id') on its own cannot distinguish from an
        # explicit null.
        stamp_kwargs = {}
        if not omit_array_job_id_field:
            stamp_kwargs['array_job_id'] = None
            stamp_kwargs['array_task_id'] = None
        record = provenance.stamp('python/vcirc_scatter_300pc.py', version=VERSION,
                                  dwarf=dwarf_name, shmr=shmr,
                                  mstar_evolution='per_halo_tidal',
                                  **stamp_kwargs)
        record.update(stamp_overrides)
        provenance.stamp_existing(row_path, record)
    return row_path


def _fill_shmr(base, shmr, dwarf_names=DWARF_NAMES, start=0, **row_kwargs):
    for offset, name in enumerate(dwarf_names):
        _row(base, shmr, start + offset, name, **row_kwargs)


def _row_raw(base, shmr, filename, dwarf_name):
    """Write a per_dwarf CSV row under an arbitrary FILENAME, bypassing the
    dwarf_idx -> row_%03d.csv naming _row() enforces, and without a sidecar.
    For exercising the row-index parsing/contiguity checks directly -- those
    fire before the sidecar-presence check, so no sidecar is needed here."""
    per_dwarf = base / shmr / 'per_dwarf'
    per_dwarf.mkdir(parents=True, exist_ok=True)
    row_path = per_dwarf / filename
    with open(row_path, 'w') as f:
        f.write(f'{dwarf_name},1.0000,2.0000,3.0000,4.0000,5.0000,6.0000,7.00000\n')
    return row_path


def _out_path(base, shmr):
    return base / shmr / 'scatter_300pc.txt'


def _run():
    argv = sys.argv
    sys.argv = ['concat_vcirc_scatter_300pc.py', VERSION]
    try:
        runpy.run_path(str(SCRIPT_PATH), run_name='__main__')
    finally:
        sys.argv = argv


def test_happy_path_byte_exact_output_and_sidecar(tree):
    _fill_shmr(tree, SHMR_A, array_job_id='12345', array_task_id='0')
    _fill_shmr(tree, SHMR_B)

    _run()

    expected_header = (
        f'# vcirc scatter at r={vsp.R_TARGET_KPC} kpc  '
        '(sigma_dex = (log10(V84)-log10(V16))/2)\n'
        f'# version={VERSION} shmr={SHMR_A} n_mc={vsp.N_MC} '
        f'narrow_sigma={vsp.NARROW_SIGMA} local_mc_seed={vsp.SEED} '
        'global_rng=per-task crc32(version:dwarf_idx:shmr)\n'
        'dwarf_name,logMstar,logMstar_err_lo,logMstar_err_hi,V_median_kms,'
        'V_16_kms,V_84_kms,sigma_dex\n'
    )
    expected_rows = ''.join(
        f'{name},1.0000,2.0000,3.0000,4.0000,5.0000,6.0000,7.00000\n'
        for name in DWARF_NAMES)
    assert _out_path(tree, SHMR_A).read_text() == expected_header + expected_rows

    record = provenance.read(_out_path(tree, SHMR_A))
    assert record is not None, 'no provenance sidecar written'
    assert record['n_rows'] == len(DWARF_NAMES)
    assert record['source_array_job_ids'] == ['12345']
    assert record['n_rows_missing_array_job_id'] == 0
    assert record['n_rows_null_array_job_id'] == 0

    # SHMR_B's rows (via the default _row()) carry an EXPLICIT
    # array_job_id=None (a direct shell invocation), not the field's
    # absence -- see test_rows_missing_array_job_id_field_entirely below for
    # that case. Either way this must be recorded as non-fatal.
    record_b = provenance.read(_out_path(tree, SHMR_B))
    assert record_b['source_array_job_ids'] == []
    assert record_b['n_rows_missing_array_job_id'] == 0
    assert record_b['n_rows_null_array_job_id'] == len(DWARF_NAMES)


def test_rows_missing_array_job_id_field_entirely_are_counted_separately(tree):
    """A row written before this field existed lacks the key entirely; a row
    from a direct shell invocation carries it as an explicit null. Both
    collapse to None under a plain `.get('array_job_id')`, which is the bug
    these two counters exist to make visible again."""
    _fill_shmr(tree, SHMR_A, omit_array_job_id_field=True)
    _fill_shmr(tree, SHMR_B)

    _run()

    record_a = provenance.read(_out_path(tree, SHMR_A))
    assert record_a['n_rows_missing_array_job_id'] == len(DWARF_NAMES)
    assert record_a['n_rows_null_array_job_id'] == 0

    record_b = provenance.read(_out_path(tree, SHMR_B))
    assert record_b['n_rows_missing_array_job_id'] == 0
    assert record_b['n_rows_null_array_job_id'] == len(DWARF_NAMES)


def test_refuses_missing_shmr_directory(tree):
    _fill_shmr(tree, SHMR_A)
    # SHMR_B directory never created at all.

    with pytest.raises(SystemExit) as excinfo:
        _run()
    assert excinfo.value.code == 1
    assert not _out_path(tree, SHMR_A).exists(), \
        'wrote SHMR_A before confirming SHMR_B exists'


def test_refuses_unexpected_extra_shmr_directory(tree):
    _fill_shmr(tree, SHMR_A)
    _fill_shmr(tree, SHMR_B)
    _fill_shmr(tree, 'StaleLeftoverShmr')  # not in vsp.SHMR_NAMES

    with pytest.raises(SystemExit) as excinfo:
        _run()
    assert excinfo.value.code == 1
    assert not _out_path(tree, SHMR_A).exists()
    assert not _out_path(tree, SHMR_B).exists()


def test_refuses_wrong_row_count(tree):
    _fill_shmr(tree, SHMR_A)
    _fill_shmr(tree, SHMR_B, dwarf_names=DWARF_NAMES[:-1])  # one row short

    with pytest.raises(SystemExit) as excinfo:
        _run()
    assert excinfo.value.code == 1
    assert not _out_path(tree, SHMR_A).exists()
    assert not _out_path(tree, SHMR_B).exists()


def test_refuses_stray_row_index(tree, capsys):
    """Row 2's slot is replaced by an out-of-range row_099.csv: the per_dwarf
    FILE COUNT is still n_dwarfs (test_refuses_wrong_row_count's check would
    pass this tree), but the index set is {0, 1, 99} instead of {0, 1, 2} --
    only the contiguity check catches it, and before that check existed this
    tree passed every check and silently dropped a dwarf from the aggregate."""
    _fill_shmr(tree, SHMR_A)
    _row(tree, SHMR_B, 0, DWARF_NAMES[0])
    _row(tree, SHMR_B, 1, DWARF_NAMES[1])
    _row_raw(tree, SHMR_B, 'row_099.csv', DWARF_NAMES[2])

    with pytest.raises(SystemExit) as excinfo:
        _run()
    assert excinfo.value.code == 1
    assert not _out_path(tree, SHMR_A).exists()
    assert not _out_path(tree, SHMR_B).exists()

    stderr = capsys.readouterr().err
    assert 'missing [2], stray [99], duplicated []' in stderr


def test_refuses_duplicate_row_index(tree, capsys):
    """ROW_INDEX_RE matches any digit run, not just the zero-padded 3-digit
    form _row() writes -- row_001.csv and row_01.csv both parse to index 1.
    Together with row_000.csv the per_dwarf FILE COUNT still equals
    n_dwarfs, but index 1 is duplicated and index 2 never appears."""
    _fill_shmr(tree, SHMR_A)
    _row_raw(tree, SHMR_B, 'row_000.csv', DWARF_NAMES[0])
    _row_raw(tree, SHMR_B, 'row_001.csv', DWARF_NAMES[1])
    _row_raw(tree, SHMR_B, 'row_01.csv', DWARF_NAMES[1])

    with pytest.raises(SystemExit) as excinfo:
        _run()
    assert excinfo.value.code == 1
    assert not _out_path(tree, SHMR_A).exists()
    assert not _out_path(tree, SHMR_B).exists()

    stderr = capsys.readouterr().err
    assert 'missing [2], stray [], duplicated [1]' in stderr


def test_refuses_row_whose_dwarf_and_sidecar_agree_but_disagree_with_catalog(tree, capsys):
    """The CSV's first field and the sidecar's 'dwarf' field agree with each
    other (both are set from the same wrong name via _row()'s dwarf_name
    argument) -- the pre-existing test_refuses_sidecar_with_wrong_dwarf only
    exercises a sidecar that disagrees with an otherwise-correct CSV. This
    covers the CSV's own disagreement with Jdata.dwarf_names[idx], which is
    checked first and fires even when the sidecar is self-consistent with it."""
    _fill_shmr(tree, SHMR_A)
    _row(tree, SHMR_B, 0, DWARF_NAMES[0])
    _row(tree, SHMR_B, 1, 'SomeOtherDwarf')  # CSV+sidecar agree, wrong for slot 1
    _row(tree, SHMR_B, 2, DWARF_NAMES[2])

    with pytest.raises(SystemExit) as excinfo:
        _run()
    assert excinfo.value.code == 1
    assert not _out_path(tree, SHMR_A).exists()
    assert not _out_path(tree, SHMR_B).exists()

    stderr = capsys.readouterr().err
    assert (f"row 1 is for dwarf 'SomeOtherDwarf', but Jdata.dwarf_names[1] "
            f"is {DWARF_NAMES[1]!r}") in stderr


def test_refuses_missing_sidecar(tree):
    _fill_shmr(tree, SHMR_A)
    _row(tree, SHMR_B, 0, DWARF_NAMES[0], write_sidecar=False)
    _fill_shmr(tree, SHMR_B, dwarf_names=DWARF_NAMES[1:], start=1)

    with pytest.raises(SystemExit) as excinfo:
        _run()
    assert excinfo.value.code == 1
    assert not _out_path(tree, SHMR_A).exists()


def test_refuses_sidecar_with_wrong_dwarf(tree):
    _fill_shmr(tree, SHMR_A)
    _row(tree, SHMR_B, 0, DWARF_NAMES[0], dwarf='SomeOtherDwarf')
    _fill_shmr(tree, SHMR_B, dwarf_names=DWARF_NAMES[1:], start=1)

    with pytest.raises(SystemExit) as excinfo:
        _run()
    assert excinfo.value.code == 1
    assert not _out_path(tree, SHMR_A).exists()


def test_refuses_sidecar_with_wrong_mstar_evolution(tree):
    _fill_shmr(tree, SHMR_A)
    _row(tree, SHMR_B, 0, DWARF_NAMES[0], mstar_evolution='legacy_constant_offset')
    _fill_shmr(tree, SHMR_B, dwarf_names=DWARF_NAMES[1:], start=1)

    with pytest.raises(SystemExit) as excinfo:
        _run()
    assert excinfo.value.code == 1
    assert not _out_path(tree, SHMR_A).exists()


def test_refusal_leaves_existing_aggregates_unchanged_and_warns(tree, capsys):
    """A phase-1 refusal must not touch any EXISTING scatter_300pc.txt (a
    previous run's output), and must say so loudly and per-file -- otherwise
    a refusal silently leaves a complete, plausible, outdated tree on disk
    with nothing on screen to flag it."""
    _fill_shmr(tree, SHMR_A)
    _fill_shmr(tree, SHMR_B, dwarf_names=DWARF_NAMES[:-1])  # one row short -> refuses

    sentinel_a = 'SENTINEL CONTENT A -- previous run\n'
    sentinel_b = 'SENTINEL CONTENT B -- previous run\n'
    out_a, out_b = _out_path(tree, SHMR_A), _out_path(tree, SHMR_B)
    out_a.write_text(sentinel_a)
    out_b.write_text(sentinel_b)

    with pytest.raises(SystemExit) as excinfo:
        _run()
    assert excinfo.value.code == 1

    assert out_a.read_text() == sentinel_a, 'refusal touched an existing aggregate'
    assert out_b.read_text() == sentinel_b, 'refusal touched an existing aggregate'

    stderr = capsys.readouterr().err
    assert 'NOT refreshed' in stderr
    assert str(out_a) in stderr, 'stale warning did not name every existing aggregate'
    assert str(out_b) in stderr, 'stale warning did not name every existing aggregate'


def test_phase2_failure_leaves_no_aggregate_bytes_changed(tree, monkeypatch):
    """SHMR_A and SHMR_B both validate cleanly (phase 1 passes), but the
    write of SHMR_B's temporary file in phase 2 fails partway through (a
    stand-in for a full disk or a kill). Destinations are only replaced
    in-place after EVERY SHMR's temp file has been built successfully, so
    neither aggregate -- not even SHMR_A's, whose own temp file finished
    building without error -- may be touched."""
    _fill_shmr(tree, SHMR_A)
    _fill_shmr(tree, SHMR_B)

    sentinel_a = 'SENTINEL CONTENT A -- previous run\n'
    sentinel_b = 'SENTINEL CONTENT B -- previous run\n'
    out_a, out_b = _out_path(tree, SHMR_A), _out_path(tree, SHMR_B)
    out_a.write_text(sentinel_a)
    out_b.write_text(sentinel_b)

    real_open = open

    def _flaky_open(file, mode='r', *args, **kwargs):
        if mode == 'w' and f'{SHMR_B}' in str(file) and 'scatter_300pc.txt.tmp' in str(file):
            raise OSError('synthetic disk-full failure')
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr('builtins.open', _flaky_open)

    with pytest.raises(OSError, match='synthetic disk-full failure'):
        _run()

    assert out_a.read_text() == sentinel_a, \
        'a phase-2 failure on a later SHMR changed an earlier aggregate'
    assert out_b.read_text() == sentinel_b, \
        'a phase-2 failure changed the very aggregate that failed to write'
