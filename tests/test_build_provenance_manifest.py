"""Regression test for build_provenance_manifest's raw-source-data rule.

Bug this pins: sidecars record literal absolute input paths (see
docs/provenance.md, "Absolute paths are retained deliberately"). After the
repository moved to a new root, `_is_raw_source_data` compared those old paths
against the new config.DATA_DIR, so committed data/ inputs (literature CSVs,
dwarf_all.csv, Symphony tables) were reported as "unknown provenance -- a
real gap": 47 false gaps across 8 figures. Fails on the pre-fix builder, which
returns False for an old-root path to a file that exists under data/ here.

The negative cases pin the guard against the opposite failure: re-anchoring
any path that merely contains a `data` directory onto a same-named local file,
which would hide real gaps.
"""
import importlib.util
from pathlib import Path

import pytest

import config

_SPEC = importlib.util.spec_from_file_location(
    'build_provenance_manifest',
    config.REPO_ROOT / 'scripts' / 'build_provenance_manifest.py')
bpm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bpm)

OLD_ROOT = Path('/nonexistent/old_root/SatelliteDensityInference')


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A checkout at tmp_path/SatelliteDensityInference with a few data files."""
    root = tmp_path / 'SatelliteDensityInference'
    data = root / 'data'
    (data / 'literature').mkdir(parents=True)
    (data / 'literature' / 'limit.csv').write_text('x\n')
    (data / 'additional').mkdir()
    (data / 'additional' / 'catalog.h5').write_text('')
    monkeypatch.setattr(config, 'REPO_ROOT', root)
    monkeypatch.setattr(config, 'DATA_DIR', data)
    monkeypatch.setattr(config, 'ADDITIONAL_DIR', data / 'additional')
    return root


LIMIT_SIZE = len('x\n')  # byte size of data/literature/limit.csv in fake_repo


def raw(path, root=OLD_ROOT, size=LIMIT_SIZE):
    return bpm._is_raw_source_data(str(path), root, size)


def test_old_root_data_input_is_raw_source_data(fake_repo):
    assert raw(OLD_ROOT / 'data/literature/limit.csv')


def test_current_root_data_input_is_raw_source_data(fake_repo):
    assert raw(fake_repo / 'data/literature/limit.csv', root=fake_repo)
    assert raw(fake_repo / 'data/literature/limit.csv', root=None)


def test_old_root_additional_input_is_still_a_gap(fake_repo):
    # data/additional/ holds derived products, which must carry a stamp.
    assert not raw(OLD_ROOT / 'data/additional/catalog.h5')


def test_old_root_missing_file_is_not_vouched_for(fake_repo):
    assert not raw(OLD_ROOT / 'data/literature/gone.csv')


def test_results_input_is_not_raw_source_data(fake_repo):
    assert not raw(OLD_ROOT / 'results/foo/bar.npz')


@pytest.mark.parametrize('size', [None, LIMIT_SIZE + 1])
def test_unknown_or_mismatched_size_is_not_rebased(fake_repo, size):
    """The root comes from the audited record itself; the size is the
    independent check that the local file is the recorded one."""
    assert not raw(OLD_ROOT / 'data/literature/limit.csv', size=size)


def test_embedded_nul_is_not_raw(fake_repo):
    assert not raw(f'{OLD_ROOT}/data/literature/lim\0it.csv')


def test_without_recorded_root_old_paths_are_not_rebased(fake_repo):
    assert not raw(OLD_ROOT / 'data/literature/limit.csv', root=None)


@pytest.mark.parametrize('path', [
    '/nonexistent/SatGen_Dwarf/data/literature/limit.csv',               # upstream tree
    f'{OLD_ROOT}/results/foo/data/literature/limit.csv',                 # data/ nested in results/
    f'{OLD_ROOT}/results/SatelliteDensityInference/data/literature/limit.csv',  # repo-named dir in results/
    '/nonexistent/DwarfJeansAnalysis/results/data/literature/limit.csv',  # sibling repo
    '/some/other/SatelliteDensityInference/data/literature/limit.csv',   # same-name clone elsewhere
    '/anywhere/data/literature/limit.csv',                               # no repo root before data/
    f'{OLD_ROOT}/data/literature/../additional/catalog.h5',              # .. into additional/
    f'{OLD_ROOT}/data/../results/x.npz',                                 # .. out of data/
])
def test_foreign_data_paths_are_not_rebased(fake_repo, path):
    """A same-named local file must not vouch for a different file elsewhere."""
    assert not raw(path)


@pytest.mark.parametrize('path', ['data/literature/limit.csv',
                                  'SatelliteDensityInference/data/literature/limit.csv'])
def test_relative_paths_are_never_vouched_for(fake_repo, monkeypatch, path):
    monkeypatch.chdir(fake_repo.parent if path.startswith('Sat') else fake_repo)
    assert not raw(path)


def test_symlink_out_of_data_is_not_raw(fake_repo, tmp_path):
    outside = tmp_path / 'outside.csv'
    outside.write_text('y\n')
    (fake_repo / 'data/literature/link.csv').symlink_to(outside)
    assert not raw(fake_repo / 'data/literature/link.csv', root=fake_repo)
    assert not raw(OLD_ROOT / 'data/literature/link.csv')


def test_recorded_root_parsed_from_figure_path():
    rec = {'figure': f'{OLD_ROOT}/plots/fermi/x.pdf'}
    assert bpm._recorded_root(rec, 'fermi/x.pdf') == OLD_ROOT
    assert bpm._recorded_root(rec, 'other/x.pdf') is None
    assert bpm._recorded_root({}, 'fermi/x.pdf') is None
    table = {'figure': f'{OLD_ROOT}/plots/tables/t.tex'}
    assert bpm._recorded_root(table, 'tables/t.tex') == OLD_ROOT
    nested = {'figure': f'{OLD_ROOT}/plots/supplementary/sub/x.pdf'}
    assert bpm._recorded_root(nested, 'supplementary/sub/x.pdf') == OLD_ROOT


def test_build_manifest_files_old_root_data_input_as_not_applicable(fake_repo, monkeypatch):
    """End to end through build_manifest, whose signature predates the fix, so
    this fails on the pre-fix builder with an assertion rather than a
    signature error: there the input landed in inputs_with_unknown_provenance."""
    import json
    plots = fake_repo / 'plots'
    (plots / 'fam').mkdir(parents=True)
    (plots / 'fam' / 'fig.pdf').write_bytes(b'%PDF')
    record = {
        'figure': f'{OLD_ROOT}/plots/fam/fig.pdf',
        'inputs': [
            {'path': f'{OLD_ROOT}/data/literature/limit.csv', 'size': LIMIT_SIZE,
             'provenance': None},
            {'path': f'{OLD_ROOT}/results/foo/bar.npz', 'size': 1, 'provenance': None},
        ],
    }
    sidecar = plots / 'fam' / ('fig.pdf' + bpm.provenance.SIDECAR_SUFFIX)
    sidecar.write_text(json.dumps(record))
    monkeypatch.setattr(config, 'PLOTS_DIR', plots)

    report = bpm.build_manifest()['report']
    assert report['inputs_not_applicable'] == {
        'fam/fig.pdf': [f'{OLD_ROOT}/data/literature/limit.csv']}
    assert report['inputs_with_unknown_provenance'] == {
        'fam/fig.pdf': [f'{OLD_ROOT}/results/foo/bar.npz']}


def test_build_manifest_foreign_root_sidecar_with_wrong_size_stays_a_gap(fake_repo, monkeypatch):
    """A copied/hand-edited sidecar naming a foreign root is self-consistent,
    so only the size check stops it vouching for the same-named local file."""
    import json
    plots = fake_repo / 'plots'
    (plots / 'fam').mkdir(parents=True)
    (plots / 'fam' / 'fig.pdf').write_bytes(b'%PDF')
    record = {'figure': '/anywhere/plots/fam/fig.pdf',
              'inputs': [{'path': '/anywhere/data/literature/limit.csv',
                          'size': 999, 'provenance': None}]}
    (plots / 'fam' / ('fig.pdf' + bpm.provenance.SIDECAR_SUFFIX)).write_text(json.dumps(record))
    monkeypatch.setattr(config, 'PLOTS_DIR', plots)
    report = bpm.build_manifest()['report']
    assert report['inputs_with_unknown_provenance'] == {
        'fam/fig.pdf': ['/anywhere/data/literature/limit.csv']}
