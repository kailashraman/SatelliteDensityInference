"""Provenance round-trip and the version-mixing guard."""

import json

import numpy as np
import pytest

import provenance


def _write(tmp_path, name, version, **arrays):
    record = provenance.stamp('python/fake.py', version=version, argv=['0', version])
    return provenance.savez(tmp_path / name, record, **(arrays or {'x': np.arange(3)}))


def test_savez_roundtrip_preserves_arrays_and_stamp(tmp_path):
    path = _write(tmp_path, 'a.npz', 'Diemer', x=np.arange(5), y=np.ones(3))

    with np.load(path) as data:
        assert np.array_equal(data['x'], np.arange(5))
        assert np.array_equal(data['y'], np.ones(3))
        assert provenance.PROVENANCE_KEY in data.files

    record = provenance.read(path)
    assert record['satgen_version'] == 'Diemer'
    assert record['sim_version'] == 'Diemer'
    assert record['script'] == 'python/fake.py'
    assert record['lvdb_version'] == 'v1.0.5'
    assert record['written_utc'].endswith('+00:00')


def test_sidecar_matches_inband(tmp_path):
    path = _write(tmp_path, 'a.npz', 'Diemer')
    sidecar = path.with_name(path.name + provenance.SIDECAR_SUFFIX)
    assert sidecar.exists()

    with open(sidecar) as handle:
        from_sidecar = json.load(handle)
    with np.load(path) as data:
        from_inband = json.loads(str(data[provenance.PROVENANCE_KEY]))
    assert from_sidecar == from_inband


def test_provenance_key_is_reserved(tmp_path):
    record = provenance.stamp('python/fake.py')
    with pytest.raises(ValueError, match='reserved'):
        provenance.savez(tmp_path / 'a.npz', record,
                         **{provenance.PROVENANCE_KEY: np.arange(3)})


def test_savez_without_npz_suffix_keeps_sidecar_adjacent(tmp_path):
    """np.savez_compressed appends .npz; the sidecar must follow it there."""
    record = provenance.stamp('python/fake.py', version='Diemer')
    path = provenance.savez(tmp_path / 'noext', record, x=np.arange(3))

    assert path.name == 'noext.npz'
    assert path.exists()
    assert (tmp_path / 'noext.npz.prov.json').exists()
    assert not (tmp_path / 'noext.prov.json').exists()
    assert provenance.read(path)['satgen_version'] == 'Diemer'


def test_stamp_rejects_unknown_version():
    """A typo'd version must not yield a stamp that merely looks complete."""
    with pytest.raises(KeyError, match='Diemr'):
        provenance.stamp('python/fake.py', version='Diemr')


def test_stamp_allows_deliberately_unmigrated_version():
    """'not migrated' is a different case from 'mistyped'."""
    record = provenance.stamp('python/fake.py', version='Geha')
    assert record['sim_version'] == 'Geha'
    assert record['h5_file'] is None


def test_stamp_records_h5_fingerprint_when_catalog_present(tmp_path, monkeypatch):
    catalog = tmp_path / 'm12res8_10k_Diemer+scatter_sim.h5'
    catalog.write_bytes(b'x' * 128)
    monkeypatch.setattr(provenance.config, 'ADDITIONAL_DIR', tmp_path)

    record = provenance.stamp('python/fake.py', version='Diemer')
    assert record['h5_file'] == 'm12res8_10k_Diemer+scatter_sim.h5'
    assert record['h5_fingerprint']['size'] == 128


def test_dja_snapshot_fingerprints_files_not_directory(tmp_path):
    """A directory mtime does not change when a chain is overwritten in place,
    which is exactly how a DwarfJeansAnalysis re-run updates it."""
    posterior = tmp_path / 'posterior_samples.npz'
    np.savez(posterior, chain=np.arange(4))

    record = provenance.stamp('python/plot_fake.py', dja_prior='satgen',
                              dja_files=[posterior])
    snapshot = record['dja_snapshot']
    assert snapshot['prior'] == 'satgen'
    assert len(snapshot['files']) == 1
    assert snapshot['files'][0]['size'] == posterior.stat().st_size
    assert snapshot['files'][0]['path'] == str(posterior)


def test_input_provenance_is_one_hop(tmp_path):
    """Full recursion would copy the shared h5 subtree once per leaf on the
    ~39-dwarf aggregation figures."""
    a = _write(tmp_path, 'a.npz', 'Diemer')
    mid = provenance.savez(
        tmp_path / 'b.npz',
        provenance.stamp('python/mid.py', version='Diemer', inputs=[a]),
        y=np.arange(2))
    leaf = provenance.savez(
        tmp_path / 'c.npz',
        provenance.stamp('python/leaf.py', version='Diemer', inputs=[mid]),
        z=np.arange(2))

    record = provenance.read(leaf)
    upstream = record['inputs'][0]['provenance']
    assert upstream['script'] == 'python/mid.py'
    assert 'inputs' not in upstream        # not recursed
    assert upstream['n_inputs'] == 1       # but its fan-in is still recorded


def test_git_state_keeps_commit_if_status_fails(monkeypatch):
    def fake_git(*args):
        return 'abc123\n' if args[0] == 'rev-parse' else None
    monkeypatch.setattr(provenance, '_git', fake_git)

    commit, dirty = provenance._git_state()
    assert commit == 'abc123'
    assert dirty is None


def test_read_returns_none_for_unstamped(tmp_path):
    path = tmp_path / 'plain.npz'
    np.savez(path, x=np.arange(3))
    assert provenance.read(path) is None


def test_inputs_embed_upstream_provenance(tmp_path):
    """A leaf product must name the script and version of each input."""
    upstream = _write(tmp_path, 'up.npz', 'Diemer')
    record = provenance.stamp('python/downstream.py', version='Diemer',
                              inputs=[upstream])
    downstream = provenance.savez(tmp_path / 'down.npz', record, z=np.arange(2))

    chain = provenance.read(downstream)
    assert len(chain['inputs']) == 1
    assert chain['inputs'][0]['provenance']['script'] == 'python/fake.py'
    assert chain['inputs'][0]['provenance']['sim_version'] == 'Diemer'
    assert chain['inputs'][0]['size'] > 0


def test_assert_single_version_accepts_consistent(tmp_path):
    paths = [_write(tmp_path, f'{i}.npz', 'Diemer') for i in range(3)]
    assert provenance.assert_single_version(paths) == 'Diemer'


def test_assert_single_version_catches_mixing(tmp_path):
    """The failure this whole module exists to prevent."""
    paths = [_write(tmp_path, 'a.npz', 'Diemer'),
             _write(tmp_path, 'b.npz', 'Zhao')]
    with pytest.raises(ValueError, match='mix SatGen versions'):
        provenance.assert_single_version(paths)


def test_assert_single_version_resolves_aliases(tmp_path):
    """lmc reweights the Diemer catalog, so mixing them is not a conflict."""
    paths = [_write(tmp_path, 'a.npz', 'Diemer'),
             _write(tmp_path, 'b.npz', 'lmc')]
    assert provenance.assert_single_version(paths) == 'Diemer'


def test_assert_single_version_rejects_unstamped(tmp_path):
    """Unknown version is not the same as known-consistent."""
    good = _write(tmp_path, 'a.npz', 'Diemer')
    plain = tmp_path / 'plain.npz'
    np.savez(plain, x=np.arange(3))
    with pytest.raises(ValueError, match='unstamped'):
        provenance.assert_single_version([good, plain])


def test_assert_single_version_checks_expected(tmp_path):
    paths = [_write(tmp_path, 'a.npz', 'Diemer')]
    with pytest.raises(ValueError, match='expected'):
        provenance.assert_single_version(paths, expected='Zhao')


def test_assert_single_version_catches_mismatched_alias(tmp_path):
    """lmc and lmc_50 both resolve to the Diemer catalog, so grouping by the
    resolved sim_version alone cannot tell them apart -- lmc weights placed in
    the lmc_50 slot must still raise. Fails against the pre-fix code, which
    only compared the resolved catalog name."""
    paths = [_write(tmp_path, 'a.npz', 'lmc')]
    with pytest.raises(ValueError, match="built as SatGen version 'lmc'"):
        provenance.assert_single_version(paths, expected='lmc_50')


def test_assert_single_version_rejects_empty_when_expecting():
    """A glob that matched nothing must not read as success."""
    with pytest.raises(ValueError, match='no inputs'):
        provenance.assert_single_version([], expected='Diemer')
    assert provenance.assert_single_version([]) is None


def test_assert_single_version_catches_alias_mixing_with_no_expected(tmp_path):
    """lmc and lmc_50 both resolve to sim_version='Diemer', so the primary
    check (grouped by resolved sim_version) sees one version and passes them
    -- a bare call with no `expected` must still catch the mix via the raw
    satgen_version. This is the gap `test_assert_single_version_catches_
    mismatched_alias` above does not cover: that test only exercises the
    `expected`-supplied branch, via a single-file `paths` list. Fails against
    the pre-fix code, where the raw-satgen_version comparison sat entirely
    inside `if expected is not None`."""
    paths = [_write(tmp_path, 'a.npz', 'lmc'),
             _write(tmp_path, 'b.npz', 'lmc_50')]
    with pytest.raises(ValueError, match='mix aliased SatGen variants'):
        provenance.assert_single_version(paths)


def test_stamp_existing_for_files_we_cannot_rewrite(tmp_path):
    """Copied-in intermediates must not look like products we computed."""
    path = tmp_path / 'catalog.h5'
    path.write_bytes(b'not really hdf5')
    record = provenance.stamp('scripts/migrate.sh', version='Diemer',
                              migrated_from='/old/tree/catalog.h5')
    provenance.stamp_existing(path, record)

    back = provenance.read(path)
    assert back['migrated_from'] == '/old/tree/catalog.h5'
    assert back['sim_version'] == 'Diemer'


def test_stamp_existing_refuses_over_disagreeing_inband_record(tmp_path):
    """savez() writes both an in-band _provenance key and a sidecar. Calling
    stamp_existing() over that product -- as a migration sweep would, since
    it never writes in-band -- would overlay a possibly-false sidecar (e.g.
    'copied from SatGen_Dwarf') on top of a truthful in-band record of a
    genuine local run. Fails against the pre-fix code, which wrote the
    sidecar unconditionally."""
    path = _write(tmp_path, 'local.npz', 'Diemer')  # in-band + sidecar, via savez
    migrated_record = provenance.stamp('scripts/stamp_migrated.py', version='Diemer',
                                       migrated_from='/old/tree/local.npz')
    with pytest.raises(ValueError, match='already carries an in-band'):
        provenance.stamp_existing(path, migrated_record)


def test_stamp_existing_allows_products_with_no_inband_record(tmp_path):
    """The normal case: an .h5 or a migrated .npz that only ever gets a
    sidecar (never went through savez) is not blocked by the guard above."""
    path = tmp_path / 'catalog.h5'
    path.write_bytes(b'not really hdf5')
    record = provenance.stamp('scripts/migrate.sh', version='Diemer',
                              migrated_from='/old/tree/catalog.h5')
    provenance.stamp_existing(path, record)  # must not raise
    assert provenance.read(path)['migrated_from'] == '/old/tree/catalog.h5'


def test_figure_manifest(tmp_path):
    upstream = _write(tmp_path, 'q.npz', 'Diemer')
    figure = tmp_path / 'fig.pdf'
    figure.write_bytes(b'%PDF-1.4')
    provenance.figure_manifest(figure, 'python/plot_fake.py', [upstream],
                              version='Diemer', dja_prior='loguniform')

    record = provenance.read(figure)
    assert record['figure'] == str(figure)
    assert record['dja_snapshot']['prior'] == 'loguniform'
    assert record['inputs'][0]['provenance']['sim_version'] == 'Diemer'
