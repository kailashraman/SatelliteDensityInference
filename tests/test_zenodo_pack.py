"""zenodo_pack: members/sha256, determinism, remap, coverage check. No real data."""
import hashlib
import importlib.util
import json
import os
import subprocess
import tarfile

import pytest

import config

_SPEC = importlib.util.spec_from_file_location(
    'zenodo_pack', config.REPO_ROOT / 'scripts' / 'zenodo_pack.py')
zp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(zp)

_REAL_TRACKED, _REAL_MODIFIED = zp.tracked_files, zp.modified_tracked
_REAL_DIRTY = zp.dirty_build_files

OLD = '/old/place/SatelliteDensityInference'
OLD_DJA = '/old/DwarfJeansAnalysis/results/production'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(path, data=b'x'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    repo, dja = tmp_path / 'repo', tmp_path / 'dja'
    write(repo / 'data/satgen_catalogs/cat.h5', b'CAT')
    write(repo / 'data/satgen_catalogs/cat.h5.prov.json', b'{}')
    write(repo / 'data/halo_weights/Diemer/Foo Bar.npz', b'W')
    write(repo / 'data/literature/tracked.csv', b'T')
    write(repo / 'results/prod/r.npz', b'R')
    write(repo / 'results/prod/r.npz.prov.json', b'{}')
    write(dja / 'foo_1/loguniform/derived.npz', b'D')
    manifest = {'inputs': [
        {'path': f'{OLD}/data/additional/cat.h5'},            # remapped
        {'path': f'{OLD}/data/additional/weights_gc/Diemer/Foo Bar.npz'},
        {'path': f'{OLD}/data/literature/tracked.csv'},
        {'path': f'{OLD}/results/prod/r.npz'}],
        'dja_snapshot': [{'files': [{'path': f'{OLD_DJA}/foo_1/loguniform/derived.npz'}]}]}
    write(repo / 'plots/fam/a.pdf.prov.json', json.dumps(manifest).encode())
    spec = [
        {'name': 'cat.tar', 'compression': 'none', 'tiers': ['figures', 'inputs'],
         'include': ['data/satgen_catalogs/**']},
        {'name': 'w.tar.gz', 'compression': 'gzip', 'tiers': ['figures'],
         'include': ['data/halo_weights/Diemer/**']},
        {'name': 'res.tar', 'compression': 'none', 'tiers': ['figures'],
         'from_figure_manifests': {'source': 'repo', 'under': ['results/']}},
        {'name': 'dja.tar.gz', 'compression': 'gzip', 'tiers': ['figures'],
         'from_figure_manifests': {'source': 'dja', 'arc_prefix': 'data/dja_results/'}},
    ]
    spec_path = write(tmp_path / 'spec.json', json.dumps({'archives': spec}).encode())
    monkeypatch.setattr(zp, 'tracked_files', lambda root: {'data/literature/tracked.csv'})
    monkeypatch.setattr(zp, 'modified_tracked', lambda root, paths: set())
    monkeypatch.setattr(zp, 'git_state', lambda p: ('deadbeef', False))
    monkeypatch.setattr(zp, 'dirty_build_files', lambda paths: [])
    readme = write(tmp_path / 'README_DATA.md', b'# readme\n')
    return dict(repo=repo, dja=dja, spec=spec_path, tmp=tmp_path, spec_list=spec,
                readme=readme)


def go(w, out, **kw):
    kw.setdefault('readme_path', w['readme'])
    return zp.run(out, spec_path=w['spec'], repo_root=w['repo'], dja_dir=w['dja'],
                  names=['a.pdf'], **kw)


def edit_manifest(w, fn):
    man = w['repo'] / 'plots/fam/a.pdf.prov.json'
    d = json.loads(man.read_text())
    fn(d)
    man.write_text(json.dumps(d))


def add_spec_archive(w, entry):
    spec = json.loads(w['spec'].read_text())
    spec['archives'].append(entry)
    w['spec'].write_text(json.dumps(spec))


def test_remap_old_paths():
    r, d = '/new/repo', '/new/dja'
    assert zp.remap(f'{OLD}/data/additional/weights_gc_published/Diemer/x.npz', r, d) == \
        ('repo', 'data/halo_weights_published/Diemer/x.npz')
    assert zp.remap(f'{OLD}/data/additional/weights_gc/Diemer/x.npz', r, d) == \
        ('repo', 'data/halo_weights/Diemer/x.npz')
    assert zp.remap(f'{OLD}/data/additional/mhalf/Diemer/x.npz', r, d) == \
        ('repo', 'data/mhalf/Diemer/x.npz')
    assert zp.remap(f'{OLD}/data/additional/Diemer_rho150.npz', r, d) == \
        ('repo', 'data/satgen_catalogs/Diemer_rho150.npz')
    assert zp.remap(f'{OLD_DJA}/a_1/loguniform/derived.npz', r, d) == \
        ('dja', 'a_1/loguniform/derived.npz')
    assert zp.remap('/new/dja/a_1/p/derived.npz', r, d) == ('dja', 'a_1/p/derived.npz')
    assert zp.remap('/elsewhere/file', r, d)[0] == 'other'


def test_build_members_and_sha(world):
    out = world['tmp'] / 'out'
    assert go(world, out) == 0
    man = json.loads((out / 'MANIFEST.json').read_text())
    assert man['sdi_git_commit'] == 'deadbeef'
    with tarfile.open(out / 'cat.tar') as t:
        assert t.getnames() == ['data/satgen_catalogs/cat.h5',
                                'data/satgen_catalogs/cat.h5.prov.json']
        m = t.getmembers()[0]
        assert (m.uid, m.gid, m.uname, m.mtime) == (0, 0, '', 0)
    with tarfile.open(out / 'res.tar') as t:      # sidecar auto-included
        assert t.getnames() == ['results/prod/r.npz', 'results/prod/r.npz.prov.json']
    with tarfile.open(out / 'dja.tar.gz') as t:
        assert t.getnames() == ['data/dja_results/foo_1/loguniform/derived.npz']
    mem = {(m['archive'], m['path']): m for m in man['members']}
    assert mem[('cat.tar', 'data/satgen_catalogs/cat.h5')]['sha256'] == sha(b'CAT')
    assert mem[('dja.tar.gz', 'data/dja_results/foo_1/loguniform/derived.npz')]['size'] == 1
    for name, rec in man['archives'].items():
        data = (out / name).read_bytes()
        assert rec['sha256'] == sha(data)
        assert rec['md5'] == hashlib.md5(data).hexdigest()
        assert rec['size'] == len(data)
    assert man['archives']['w.tar.gz']['tiers'] == ['figures']


def test_rebuild_byte_identical(world):
    a, b = world['tmp'] / 'a', world['tmp'] / 'b'
    go(world, a)
    for p in world['repo'].rglob('*'):       # touch: mtimes must not leak in
        if p.is_file():
            p.touch()
    go(world, b)
    for name in ['cat.tar', 'w.tar.gz', 'res.tar', 'dja.tar.gz']:
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_dry_run_writes_nothing(world):
    out = world['tmp'] / 'nothere'
    assert go(world, out, dry_run=True) == 0
    assert not out.exists()


def test_coverage_fails_on_uncovered_input(world, capsys):
    w = world
    man = w['repo'] / 'plots/fam/a.pdf.prov.json'
    d = json.loads(man.read_text())
    write(w['repo'] / 'data/mhalf/Diemer/q.npz', b'Q')    # untracked, no archive
    d['inputs'].append({'path': f'{OLD}/data/additional/mhalf/Diemer/q.npz'})
    man.write_text(json.dumps(d))
    assert go(w, w['tmp'] / 'o', dry_run=True) == 1
    assert 'data/mhalf/Diemer/q.npz' in capsys.readouterr().out
    assert go(w, w['tmp'] / 'o') == 1 and not (w['tmp'] / 'o').exists()


def test_coverage_requires_figures_tier(world):
    spec = json.loads(world['spec'].read_text())
    for a in spec['archives']:
        if a['name'] == 'w.tar.gz':
            a['tiers'] = ['inputs']
    world['spec'].write_text(json.dumps(spec))
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1


def test_missing_source_fails(world):
    (world['dja'] / 'foo_1/loguniform/derived.npz').unlink()
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1


def test_paper_artifact_without_manifest_fails(world):
    with pytest.raises(RuntimeError, match='no figure manifest'):
        zp.paper_manifests(world['repo'], names=['a.pdf', 'b.pdf'])


def test_brace_expansion():
    assert zp.expand_braces('d/{Draco,Segue 1}.*') == ['d/Draco.*', 'd/Segue 1.*']


def test_only_builds_subset(world):
    out = world['tmp'] / 'o'
    go(world, out)
    before = (out / 'cat.tar').read_bytes()
    go(world, out, only=['res.tar'])
    man = json.loads((out / 'MANIFEST.json').read_text())
    assert set(man['archives']) == {'cat.tar', 'w.tar.gz', 'res.tar', 'dja.tar.gz'}
    assert (out / 'cat.tar').read_bytes() == before
    assert sum(m['archive'] == 'res.tar' for m in man['members']) == 2


# ------------------------------------------------------------ review fixes

def test_readme_required_even_in_dry_run(world, capsys):
    world['readme'].unlink()
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1
    assert 'README_DATA.md' in capsys.readouterr().out
    assert go(world, world['tmp'] / 'o') == 1


def test_readme_copied_and_recorded(world):
    out = world['tmp'] / 'o'
    assert go(world, out) == 0
    data = (out / 'README_DATA.md').read_bytes()
    assert data == b'# readme\n'
    r = json.loads((out / 'MANIFEST.json').read_text())['readme']
    assert r['sha256'] == sha(data) and r['md5'] == hashlib.md5(data).hexdigest()


def test_manifest_is_compact(world):
    out = world['tmp'] / 'o'
    go(world, out)
    text = (out / 'MANIFEST.json').read_text()
    assert '\n' not in text and ', ' not in text and '": ' not in text


def test_size_mismatch_fails(world, capsys):
    edit_manifest(world, lambda d: d['inputs'][0].update(size=999))   # cat.h5 is 3 bytes
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1
    assert 'size' in capsys.readouterr().out


def test_size_match_passes(world):
    edit_manifest(world, lambda d: d['inputs'][0].update(size=3))
    assert go(world, world['tmp'] / 'o', dry_run=True) == 0


def test_dja_recorded_size_checked(world):
    edit_manifest(world, lambda d: d['dja_snapshot'][0]['files'][0].update(size=5))
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1


def test_h5_fingerprint_checked_like_inputs(world, capsys):
    def add(d):
        d['h5_fingerprint'] = {'path': f'{OLD}/data/additional/mhalf/Diemer/q.npz',
                               'size': 1, 'mtime': 0}
    write(world['repo'] / 'data/mhalf/Diemer/q.npz', b'Q')    # untracked, uncovered
    edit_manifest(world, add)
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1
    assert 'data/mhalf/Diemer/q.npz' in capsys.readouterr().out


def test_h5_fingerprint_size_checked(world):
    edit_manifest(world, lambda d: d.update(h5_fingerprint={
        'path': f'{OLD}/data/additional/cat.h5', 'size': 12345, 'mtime': 0}))
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1


def test_tracked_input_missing_on_disk_fails(world, capsys):
    (world['repo'] / 'data/literature/tracked.csv').unlink()
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1
    assert 'tracked.csv' in capsys.readouterr().out


def _git(repo, *a):
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t',
                    '-c', 'commit.gpgsign=false', *a], check=True, capture_output=True)


def test_tracked_input_modified_vs_head_fails(world, monkeypatch, capsys):
    monkeypatch.setattr(zp, 'tracked_files', _REAL_TRACKED)
    monkeypatch.setattr(zp, 'modified_tracked', _REAL_MODIFIED)
    repo = world['repo']
    _git(repo, 'init', '-q')
    _git(repo, 'add', 'data/literature/tracked.csv')
    _git(repo, 'commit', '-q', '-m', 'x')
    assert go(world, world['tmp'] / 'o', dry_run=True) == 0
    (repo / 'data/literature/tracked.csv').write_bytes(b'edited')
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1
    assert 'modified vs HEAD' in capsys.readouterr().out


def test_per_alternative_empty_match():
    files, empty = zp.resolve_globs(['d/{Draco,Nope}.*'], '/nonexistent')
    assert empty == ['d/Draco.*', 'd/Nope.*']


def test_per_alternative_empty_fails_run(world, capsys):
    write(world['repo'] / 'data/fermi_legacy/Draco.txt')
    add_spec_archive(world, {'name': 'f.tar', 'compression': 'none', 'tiers': ['inputs'],
                             'include': ['data/fermi_legacy/{Draco,Hydrus I}.*']})
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1
    out = capsys.readouterr().out
    assert 'Hydrus I' in out and 'Draco' not in out.split('FAIL')[1]


def test_dirty_checkout_refused(world, monkeypatch, capsys):
    monkeypatch.setattr(zp, 'git_state', lambda p: ('deadbeef', True))
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1
    assert 'modified/staged' in capsys.readouterr().out
    assert go(world, world['tmp'] / 'o') == 1 and not (world['tmp'] / 'o').exists()
    assert go(world, world['tmp'] / 'o', allow_dirty=True) == 0
    man = json.loads((world['tmp'] / 'o/MANIFEST.json').read_text())
    assert man['archives']['cat.tar']['sdi_git_dirty'] is True
    assert man['archives']['cat.tar']['sdi_git_commit'] == 'deadbeef'


def test_per_archive_stamps(world):
    out = world['tmp'] / 'o'
    go(world, out)
    rec = json.loads((out / 'MANIFEST.json').read_text())['archives']['cat.tar']
    assert rec['sdi_git_commit'] == 'deadbeef' and rec['sdi_git_dirty'] is False
    assert rec['built_utc']


def test_only_keeps_other_stamps_and_stamps_rebuilt(world, monkeypatch):
    out = world['tmp'] / 'o'
    go(world, out)
    monkeypatch.setattr(zp, 'git_state', lambda p: ('cafe1234', False))
    go(world, out, only=['res.tar'])
    a = json.loads((out / 'MANIFEST.json').read_text())['archives']
    assert a['cat.tar']['sdi_git_commit'] == 'deadbeef'
    assert a['res.tar']['sdi_git_commit'] == 'cafe1234'


def test_build_unlinks_stale_manifest_and_no_part_files(world, monkeypatch):
    out = world['tmp'] / 'o'
    go(world, out)
    assert (out / 'MANIFEST.json').exists()

    def boom(*a, **k):
        raise RuntimeError('boom')
    monkeypatch.setattr(zp, 'build_archive', boom)
    with pytest.raises(RuntimeError):
        go(world, out)
    assert not (out / 'MANIFEST.json').exists()


def test_failed_archive_leaves_no_partial_file(world, tmp_path):
    src = write(tmp_path / 'f.bin', b'abc')
    dest = tmp_path / 'o' / 'x.tar'
    dest.parent.mkdir()
    with pytest.raises(ValueError):
        zp.build_archive(dest, 'bogus', {'f.bin': src})
    assert list(dest.parent.iterdir()) == []
    zp.build_archive(dest, 'none', {'f.bin': src})
    assert [p.name for p in dest.parent.iterdir()] == ['x.tar']


def test_file_changing_while_packed_fails(world, tmp_path, monkeypatch):
    src = write(tmp_path / 'f.bin', b'abc')

    class Touching(zp.HashingReader):
        def read(self, size=-1):
            b = super().read(size)
            os.utime(src, ns=(1, 1))
            return b
    monkeypatch.setattr(zp, 'HashingReader', Touching)
    with pytest.raises(RuntimeError, match='changed while packing'):
        zp.build_archive(tmp_path / 'x.tar', 'none', {'f.bin': src})
    assert not (tmp_path / 'x.tar').exists() and not (tmp_path / 'x.tar.part').exists()


@pytest.mark.parametrize('bad', ['/abs/f', '../f', 'a/../f', 'a//f'])
def test_unsafe_arcnames_rejected(tmp_path, bad):
    src = write(tmp_path / 'f.bin', b'abc')
    with pytest.raises(ValueError, match='unsafe'):
        zp.build_archive(tmp_path / 'x.tar', 'none', {bad: src})


def test_arcnames_normalised(tmp_path):
    src = write(tmp_path / 'f.bin', b'abc')
    _, mem = zp.build_archive(tmp_path / 'x.tar', 'none', {'./a/./f': src})
    assert [m['path'] for m in mem] == ['a/f']
    with pytest.raises(ValueError, match='duplicate'):
        zp.build_archive(tmp_path / 'y.tar', 'none', {'./a/f': src, 'a/f': src})


def test_unsafe_arcname_fails_run(world, capsys):
    add_spec_archive(world, {'name': 'bad.tar', 'compression': 'none', 'tiers': ['inputs'],
                             'include': ['data/satgen_catalogs/../satgen_catalogs/*.h5']})
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1
    assert 'unsafe archive member name' in capsys.readouterr().out


def test_directory_input_expanded(world):
    d = world['dja'] / 'foo_1/loguniform'
    write(d / 'extra.npz', b'E')
    edit_manifest(world, lambda x: x['inputs'].append(
        {'path': f'{OLD_DJA}/foo_1/loguniform', 'size': 4096}))   # a directory
    out = world['tmp'] / 'o'
    assert go(world, out) == 0
    with tarfile.open(out / 'dja.tar.gz') as t:
        assert 'data/dja_results/foo_1/loguniform/extra.npz' in t.getnames()


def test_other_kind_input_fails(world, capsys):
    edit_manifest(world, lambda d: d['inputs'].append({'path': '/elsewhere/file.dat'}))
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1
    assert '/elsewhere/file.dat' in capsys.readouterr().out


def test_sidecar_recorded_as_input(world):
    edit_manifest(world, lambda d: d['inputs'].append(
        {'path': f'{OLD}/results/prod/r.npz.prov.json'}))
    out = world['tmp'] / 'o'
    assert go(world, out) == 0
    with tarfile.open(out / 'res.tar') as t:
        assert t.getnames().count('results/prod/r.npz.prov.json') == 1
        assert 'results/prod/r.npz.prov.json.prov.json' not in t.getnames()


def test_untracked_build_defining_file_refused(world, monkeypatch, capsys):
    monkeypatch.setattr(zp, 'dirty_build_files', _REAL_DIRTY)
    d = world['tmp'] / 'bd'
    for n in ('spec.json', 'meta.json', 'script.py'):
        write(d / n)
    write(d / 'README_DATA.md')
    monkeypatch.setattr(zp, 'METADATA_PATH', d / 'meta.json')
    monkeypatch.setattr(zp, '__file__', str(d / 'script.py'))
    world['spec'] = d / 'spec.json'
    world['spec'].write_text(json.dumps({'archives': world['spec_list']}))
    world['readme'] = d / 'README_DATA.md'
    _git(d, 'init', '-q')
    _git(d, 'add', '.')
    _git(d, 'commit', '-q', '-m', 'x')
    assert go(world, world['tmp'] / 'o', dry_run=True) == 0
    (d / 'README_DATA.md').write_bytes(b'edited')           # differs from HEAD
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1
    assert 'build-defining' in capsys.readouterr().out
    _git(d, 'add', '.')
    _git(d, 'commit', '-q', '-m', 'y')
    write(d / 'meta.json', b'new')
    _git(d, 'rm', '-q', '--cached', 'meta.json')            # untracked
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1
    assert go(world, world['tmp'] / 'o', allow_dirty=True) == 0
    man = json.loads((world['tmp'] / 'o/MANIFEST.json').read_text())
    assert man['archives']['cat.tar']['sdi_git_dirty'] is True


def _during_build(monkeypatch, action):
    real = zp.build_archive

    def build(*a, **k):
        r = real(*a, **k)
        action()
        return r
    monkeypatch.setattr(zp, 'build_archive', build)


@pytest.mark.parametrize('only', [None, ['res.tar']])
@pytest.mark.parametrize('what', ['head_moved', 'dirty', 'build_file_dirty', 'dja_moved'])
def test_git_state_change_during_build_fails_and_leaves_nothing(world, monkeypatch,
                                                               capsys, what, only):
    out = world['tmp'] / 'o'
    go(world, out)                                      # a previous build

    def change():
        if what == 'head_moved':
            monkeypatch.setattr(zp, 'git_state', lambda p: ('newcommit', False))
        elif what == 'dirty':
            monkeypatch.setattr(zp, 'git_state', lambda p: ('deadbeef', True))
        elif what == 'dja_moved':
            monkeypatch.setattr(zp, 'git_state',
                                lambda p: ('deadbeef' if p == world['repo'] else 'other', False))
        else:
            monkeypatch.setattr(zp, 'dirty_build_files', lambda paths: [str(paths[0])])
    _during_build(monkeypatch, change)
    assert go(world, out, only=only) == 1
    assert 'during the build' in capsys.readouterr().out
    assert not (out / 'MANIFEST.json').exists()         # no manifest claiming the old commit
    assert not (out / zp.README_NAME).exists()
    built = {'res.tar'} if only else {'cat.tar', 'w.tar.gz', 'res.tar', 'dja.tar.gz'}
    assert not any((out / n).exists() for n in built)


def test_readme_shipped_is_the_one_read_at_the_start(world, monkeypatch):
    out = world['tmp'] / 'o'
    _during_build(monkeypatch, lambda: world['readme'].write_bytes(b'# newer readme\n'))
    assert go(world, out) == 0
    data = (out / 'README_DATA.md').read_bytes()
    assert data == b'# readme\n'
    r = json.loads((out / 'MANIFEST.json').read_text())['readme']
    assert r['sha256'] == sha(data) and r['size'] == len(data)


def test_only_drops_records_for_removed_archives(world):
    out = world['tmp'] / 'o'
    add_spec_archive(world, {'name': 'gone.tar', 'compression': 'none',
                             'tiers': ['inputs'], 'include': ['data/satgen_catalogs/*.h5']})
    go(world, out)
    spec = json.loads(world['spec'].read_text())
    spec['archives'] = [a for a in spec['archives'] if a['name'] != 'gone.tar']
    world['spec'].write_text(json.dumps(spec))
    go(world, out, only=['res.tar'])
    man = json.loads((out / 'MANIFEST.json').read_text())
    assert 'gone.tar' not in man['archives']
    assert all(m['archive'] != 'gone.tar' for m in man['members'])


def test_glob_includes_hidden_but_not_ipynb_checkpoints(tmp_path):
    write(tmp_path / 'd/.hidden.txt')
    write(tmp_path / 'd/sub/.dot/f.txt')
    write(tmp_path / 'd/vis.txt')
    write(tmp_path / 'd/.ipynb_checkpoints/c.txt')
    write(tmp_path / 'd/sub/.ipynb_checkpoints/c.txt')
    files, empty = zp.resolve_globs(['d/**'], tmp_path)
    assert set(files) == {'d/.hidden.txt', 'd/sub/.dot/f.txt', 'd/vis.txt'}
    assert not empty


# ------------------------------------------------------- --only consistency

def test_only_without_manifest_refused(world, capsys):
    out = world['tmp'] / 'o'
    assert go(world, out, only=['res.tar']) == 1
    assert 'full pack first' in capsys.readouterr().out
    assert not (out / 'MANIFEST.json').exists() and not (out / 'res.tar').exists()


def test_only_missing_kept_archive_refused(world, capsys):
    out = world['tmp'] / 'o'
    go(world, out)
    (out / 'cat.tar').unlink()
    assert go(world, out, only=['res.tar']) == 1
    assert 'cat.tar missing' in capsys.readouterr().out
    assert (out / 'MANIFEST.json').exists()             # refused before touching it


def test_only_kept_archive_size_mismatch_refused(world, capsys):
    out = world['tmp'] / 'o'
    go(world, out)
    with open(out / 'w.tar.gz', 'ab') as f:
        f.write(b'junk')
    assert go(world, out, only=['res.tar']) == 1
    assert 'w.tar.gz: on-disk size' in capsys.readouterr().out


def test_only_manifest_must_cover_every_spec_archive(world, capsys):
    out = world['tmp'] / 'o'
    go(world, out)
    edit = json.loads((out / 'MANIFEST.json').read_text())
    del edit['archives']['cat.tar']
    (out / 'MANIFEST.json').write_text(json.dumps(edit))
    assert go(world, out, only=['res.tar']) == 1
    assert 'no manifest record' in capsys.readouterr().out
    assert not (out / 'MANIFEST.json').exists()


def test_only_notes_kept_archive_from_older_commit(world, monkeypatch, capsys):
    out = world['tmp'] / 'o'
    go(world, out)
    capsys.readouterr()
    monkeypatch.setattr(zp, 'git_state', lambda p: ('cafe1234', False))
    assert go(world, out, only=['res.tar']) == 0
    text = capsys.readouterr().out
    assert 'NOTE' in text and 'cat.tar' in text and 'deadbeef' in text and 'cafe1234' in text
    assert 'res.tar was built' not in text


def test_gitignored_build_file_counts_as_dirty(tmp_path):
    d = tmp_path / 'g'
    write(d / 'a.json')
    write(d / 'b.json')
    _git(d, 'init', '-q')
    _git(d, 'add', '.')
    _git(d, 'commit', '-q', '-m', 'x')
    assert _REAL_DIRTY([d / 'a.json', d / 'b.json']) == []
    _git(d, 'rm', '-q', '--cached', 'b.json')
    write(d / '.gitignore', b'b.json\n')
    _git(d, 'add', '.gitignore')
    _git(d, 'commit', '-q', '-m', 'y')
    assert _REAL_DIRTY([d / 'a.json', d / 'b.json']) == [str(d / 'b.json')]
