"""zenodo_fetch: checksum verification, path traversal, overwrite policy. Offline."""
import hashlib
import importlib.util
import io
import json
import tarfile

import pytest

import config

_S = importlib.util.spec_from_file_location(
    'zenodo_fetch', config.REPO_ROOT / 'scripts' / 'zenodo_fetch.py')
zf = importlib.util.module_from_spec(_S)
_S.loader.exec_module(zf)


def make_tar(path, members, mode='w'):
    with tarfile.open(path, mode) as t:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))


def digests(path):
    b = path.read_bytes()
    return hashlib.sha256(b).hexdigest(), hashlib.md5(b).hexdigest()


def make_record(d, members=None, tiers=('figures',), name='a.tar'):
    d.mkdir()
    members = members or {'data/x/f.txt': b'hello'}
    make_tar(d / name, members)
    sha, md5 = digests(d / name)
    man = {'archives': {name: {'size': 1, 'sha256': sha, 'md5': md5, 'tiers': list(tiers)}},
           'members': [{'archive': name, 'path': k, 'size': len(v),
                        'sha256': hashlib.sha256(v).hexdigest()} for k, v in members.items()]}
    (d / 'MANIFEST.json').write_text(json.dumps(man))
    return man


def test_fetch_extracts(tmp_path, capsys):
    make_record(tmp_path / 'rec')
    zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest')
    assert (tmp_path / 'dest/data/x/f.txt').read_bytes() == b'hello'


def test_dja_export_hint(tmp_path, capsys):
    make_record(tmp_path / 'rec', {'data/dja_results/k/p/derived.npz': b'1'})
    zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest')
    assert f"export SDI_DJA_RESULTS_DIR={tmp_path / 'dest/data/dja_results'}" in capsys.readouterr().out


def test_tier_selection(tmp_path):
    d = tmp_path / 'rec'
    make_record(d, tiers=('inputs',))
    with pytest.raises(zf.FetchError, match='no archive in tier'):
        zf.fetch(files_from=d, tier='figures', dest=tmp_path / 'dest')


def test_manifest_checksum_mismatch(tmp_path):
    d = tmp_path / 'rec'
    man = make_record(d)
    man['archives']['a.tar']['sha256'] = '0' * 64
    (d / 'MANIFEST.json').write_text(json.dumps(man))
    with pytest.raises(zf.FetchError, match='MANIFEST'):
        zf.fetch(files_from=d, dest=tmp_path / 'dest')
    assert not (tmp_path / 'dest/data').exists()


class FakeResp:
    def __init__(self, js=None, content=b''):
        self._js, self.content = js, content

    def raise_for_status(self):
        pass

    def json(self):
        return self._js

    def iter_content(self, n):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class FakeSession:
    def __init__(self, record, md5_override=None):
        self.record, self.md5_override = record, md5_override

    def get(self, url, **kw):
        if url.endswith('/api/records/42'):
            files = []
            for p in self.record.iterdir():
                md5 = digests(p)[1]
                if self.md5_override and p.name == 'a.tar':
                    md5 = self.md5_override
                files.append({'key': p.name, 'checksum': f'md5:{md5}',
                              'links': {'self': f'https://z/files/{p.name}'}})
            return FakeResp({'files': files})
        return FakeResp(content=(self.record / url.rsplit('/', 1)[1]).read_bytes())


def test_zenodo_md5_mismatch(tmp_path):
    make_record(tmp_path / 'rec')
    s = FakeSession(tmp_path / 'rec', md5_override='1' * 32)
    with pytest.raises(zf.FetchError, match='Zenodo checksum'):
        zf.fetch('42', 'https://zenodo.org', dest=tmp_path / 'dest', session=s)


def test_fetch_via_api(tmp_path):
    make_record(tmp_path / 'rec')
    zf.fetch('42', 'https://zenodo.org', dest=tmp_path / 'dest',
             session=FakeSession(tmp_path / 'rec'))
    assert (tmp_path / 'dest/data/x/f.txt').exists()


@pytest.mark.parametrize('bad', ['../evil.txt', '/abs/evil.txt', 'a/../../evil.txt'])
def test_path_traversal_rejected(tmp_path, bad):
    d = tmp_path / 'rec'
    make_record(d, {bad: b'x'})
    with pytest.raises(zf.FetchError, match='unsafe path'):
        zf.fetch(files_from=d, dest=tmp_path / 'dest')
    assert not (tmp_path / 'evil.txt').exists()
    assert not (tmp_path / 'dest').exists() or not any((tmp_path / 'dest').rglob('*.txt'))


def test_refuses_differing_overwrite(tmp_path):
    make_record(tmp_path / 'rec')
    target = tmp_path / 'dest/data/x/f.txt'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'local edit')
    with pytest.raises(zf.FetchError, match='differ'):
        zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest')
    assert target.read_bytes() == b'local edit'
    zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest', force=True)
    assert target.read_bytes() == b'hello'


def test_identical_existing_file_ok(tmp_path):
    make_record(tmp_path / 'rec')
    target = tmp_path / 'dest/data/x/f.txt'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'hello')
    zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest')


def test_tmp_dir_under_dest_not_tmpdir(tmp_path, monkeypatch):
    make_record(tmp_path / 'rec')
    elsewhere = tmp_path / 'sys_tmp'
    elsewhere.mkdir()
    monkeypatch.setattr(zf.tempfile, 'tempdir', str(elsewhere))
    seen = []
    real = zf.download

    def spy(url, dest, session):
        seen.append(dest)
        real(url, dest, session)
    monkeypatch.setattr(zf, 'download', spy)
    zf.fetch('42', 'https://zenodo.org', dest=tmp_path / 'dest',
             session=FakeSession(tmp_path / 'rec'))
    assert seen and all(tmp_path / 'dest' in p.parents for p in seen)
    assert list(elsewhere.iterdir()) == []
    assert not list((tmp_path / 'dest').glob('.zenodo_fetch_*'))      # cleaned up


def test_tmp_option(tmp_path):
    make_record(tmp_path / 'rec')
    seen = []
    zf_download = zf.download

    def spy(url, dest, session):
        seen.append(dest)
        zf_download(url, dest, session)
    zf.download = spy
    try:
        zf.fetch('42', 'https://zenodo.org', dest=tmp_path / 'dest',
                 session=FakeSession(tmp_path / 'rec'), tmp=tmp_path / 'scr')
    finally:
        zf.download = zf_download
    assert seen and all(tmp_path / 'scr' in p.parents for p in seen)


class NoMd5Session(FakeSession):
    def __init__(self, record, drop):
        super().__init__(record)
        self.drop = drop

    def get(self, url, **kw):
        r = super().get(url, **kw)
        if url.endswith('/api/records/42'):
            for f in r._js['files']:
                if f['key'] == self.drop:
                    f['checksum'] = ''
        return r


@pytest.mark.parametrize('drop', ['MANIFEST.json', 'a.tar'])
def test_missing_record_checksum_fails(tmp_path, drop):
    make_record(tmp_path / 'rec')
    with pytest.raises(zf.FetchError, match='no checksum'):
        zf.fetch('42', 'https://zenodo.org', dest=tmp_path / 'dest',
                 session=NoMd5Session(tmp_path / 'rec', drop))
    assert not (tmp_path / 'dest/data').exists()


def test_manifest_md5_mismatch_vs_record_fails(tmp_path):
    make_record(tmp_path / 'rec')

    class S(FakeSession):
        def get(self, url, **kw):
            r = super().get(url, **kw)
            if url.endswith('/api/records/42'):
                for f in r._js['files']:
                    if f['key'] == 'MANIFEST.json':
                        f['checksum'] = 'md5:' + '2' * 32
            return r
    with pytest.raises(zf.FetchError, match='MANIFEST.json: md5'):
        zf.fetch('42', 'https://zenodo.org', dest=tmp_path / 'dest',
                 session=S(tmp_path / 'rec'))


def test_conflict_check_covers_all_archives_before_extracting(tmp_path):
    d = tmp_path / 'rec'
    d.mkdir()
    members = {'a.tar': {'data/one.txt': b'1'}, 'b.tar': {'data/two.txt': b'2'}}
    man = {'archives': {}, 'members': []}
    for name, mem in members.items():
        make_tar(d / name, mem)
        sha, md5 = digests(d / name)
        man['archives'][name] = {'size': 1, 'sha256': sha, 'md5': md5, 'tiers': ['figures']}
        man['members'] += [{'archive': name, 'path': k, 'size': 1,
                            'sha256': hashlib.sha256(v).hexdigest()} for k, v in mem.items()]
    (d / 'MANIFEST.json').write_text(json.dumps(man))
    dest = tmp_path / 'dest'
    (dest / 'data').mkdir(parents=True)
    (dest / 'data/two.txt').write_bytes(b'local edit')      # conflicts with b.tar only
    with pytest.raises(zf.FetchError, match='differ'):
        zf.fetch(files_from=d, dest=dest)
    assert not (dest / 'data/one.txt').exists()             # a.tar not extracted either
    assert (dest / 'data/two.txt').read_bytes() == b'local edit'


def test_requires_tarfile_data_filter(tmp_path, monkeypatch):
    make_record(tmp_path / 'rec')
    monkeypatch.delattr(zf.tarfile, 'data_filter')
    with pytest.raises(zf.FetchError, match='data_filter'):
        zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest')
    assert not (tmp_path / 'dest').exists()


def test_corrupt_archive_is_fetch_error(tmp_path, capsys):
    d = tmp_path / 'rec'
    make_record(d)
    (d / 'a.tar').write_bytes(b'not a tar' * 100)
    sha, md5 = digests(d / 'a.tar')
    man = json.loads((d / 'MANIFEST.json').read_text())
    man['archives']['a.tar'].update(sha256=sha, md5=md5)
    (d / 'MANIFEST.json').write_text(json.dumps(man))
    with pytest.raises(zf.FetchError, match='cannot extract'):
        zf.fetch(files_from=d, dest=tmp_path / 'dest')
    assert zf.main(['--files-from', str(d), '--dest', str(tmp_path / 'dest2')]) == 1
    assert 'ERROR:' in capsys.readouterr().err


def test_extracted_mtime_is_extraction_time(tmp_path):
    import time
    make_record(tmp_path / 'rec')
    t0 = time.time()
    zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest')
    assert (tmp_path / 'dest/data/x/f.txt').stat().st_mtime >= t0 - 1
