"""zenodo_upload: mocked session; never publishes, resumes, token required."""
import hashlib
import importlib.util
import json

import pytest
import requests

import config

_S = importlib.util.spec_from_file_location(
    'zenodo_upload', config.REPO_ROOT / 'scripts' / 'zenodo_upload.py')
zu = importlib.util.module_from_spec(_S)
_S.loader.exec_module(zu)

BASE = 'https://sandbox.zenodo.org'


class Resp:
    def __init__(self, js, status=200, text=''):
        self._js, self.status_code, self.text = js, status, text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code} error', response=self)

    def json(self):
        return self._js


class FakeSession:
    def __init__(self, existing=None, bad_checksum=False, fail_put=None):
        self.calls = []          # (method, url)
        self.kwargs = []         # every call's kwargs
        self.fail_put = fail_put
        self.puts = {}
        self.posted = None
        self.existing = existing or {}
        self.bad_checksum = bad_checksum
        self.dep = {'id': 7, 'submitted': False,
                    'links': {'bucket': f'{BASE}/api/files/b', 'html': 'https://x/7'},
                    'metadata': {'prereserve_doi': {'doi': '10.5072/zenodo.7'}}}

    def mount(self, prefix, adapter):
        pass

    def post(self, url, **kw):
        self.calls.append(('POST', url))
        self.kwargs.append(kw)
        self.posted = kw['json']
        return Resp(self.dep)

    def get(self, url, **kw):
        self.calls.append(('GET', url))
        self.kwargs.append(kw)
        if url.endswith('/files'):
            files = dict(self.existing)
            files.update({k: f'md5:{hashlib.md5(v).hexdigest()}' for k, v in self.puts.items()})
            return Resp([{'filename': k, 'checksum': v} for k, v in files.items()])
        return Resp(self.dep)

    def put(self, url, data=None, **kw):
        self.calls.append(('PUT', url))
        self.kwargs.append(kw)
        if self.fail_put:
            return Resp({}, status=self.fail_put[0], text=self.fail_put[1])
        body = data.read()
        self.puts[url.rsplit('/', 1)[1]] = body
        md5 = '0' * 32 if self.bad_checksum else hashlib.md5(body).hexdigest()
        return Resp({'checksum': f'md5:{md5}'})


def make_build(d, extra=None, readme=b'README'):
    """A consistent build dir: two archives, README, MANIFEST."""
    d.mkdir(exist_ok=True)
    arch = {'a.tar': b'AAA', 'b.tar.gz': b'BBBB'}
    for k, v in arch.items():
        (d / k).write_bytes(v)
    (d / 'README_DATA.md').write_bytes(readme)
    man = {'archives': {k: {'size': len(v), 'md5': hashlib.md5(v).hexdigest(),
                            'sdi_git_commit': 'abc', 'sdi_git_dirty': False}
                        for k, v in arch.items()},
           'sdi_git_commit': 'abc', 'sdi_git_dirty': False,
           'readme': {'md5': hashlib.md5(readme).hexdigest()}}
    (d / 'MANIFEST.json').write_text(json.dumps(man))
    return d


@pytest.fixture
def build(tmp_path):
    return make_build(tmp_path / 'build')


def test_creates_draft_and_never_publishes(build, capsys):
    s = FakeSession()
    zu.upload(build, BASE, 'tok', session=s)
    assert s.posted['metadata']['prereserve_doi'] is True
    assert s.posted['metadata']['upload_type'] == 'dataset'
    assert set(s.puts) == {'a.tar', 'b.tar.gz', 'MANIFEST.json', 'README_DATA.md'}
    assert all('publish' not in url for _, url in s.calls)
    assert {m for m, _ in s.calls} <= {'POST', 'GET', 'PUT'}
    out = capsys.readouterr().out
    assert '10.5072/zenodo.7' in out and 'tok' not in out
    assert 'deposition 7' in out and 'https://x/7' in out


def test_no_publish_endpoint_in_source():
    src = (config.REPO_ROOT / 'scripts/zenodo_upload.py').read_text()
    assert 'actions/publish' not in src


def test_resume_skips_matching_md5(build):
    s = FakeSession(existing={'a.tar': hashlib.md5(b'AAA').hexdigest(),
                              'MANIFEST.json': 'stale'})
    zu.upload(build, BASE, 'tok', deposition='7', session=s)
    assert s.posted is None                       # no new deposition
    assert set(s.puts) == {'b.tar.gz', 'MANIFEST.json', 'README_DATA.md'}
    assert all('publish' not in url for _, url in s.calls)


def test_checksum_mismatch_raises(build):
    with pytest.raises(zu.UploadError, match='md5'):
        zu.upload(build, BASE, 'tok', session=FakeSession(bad_checksum=True))


def test_refuses_published_deposition(build):
    s = FakeSession()
    s.dep['submitted'] = True
    with pytest.raises(zu.UploadError, match='published'):
        zu.upload(build, BASE, 'tok', deposition='7', session=s)
    assert not s.puts


def test_missing_token_errors(build, monkeypatch, capsys):
    monkeypatch.delenv('ZENODO_TOKEN', raising=False)
    assert zu.main(['--build-dir', str(build), '--sandbox']) == 1
    assert 'ZENODO_TOKEN' in capsys.readouterr().err


def test_metadata_json_valid():
    m = json.loads((config.REPO_ROOT / 'zenodo/metadata.json').read_text())
    assert m['upload_type'] == 'dataset' and m['license'] == 'cc-by-4.0'
    assert m['access_right'] == 'open' and m['creators']


def no_network(session):
    assert session.calls == []


def test_extra_file_in_build_dir_rejected(build):
    (build / 'stray.bin').write_bytes(b'junk')
    s = FakeSession()
    with pytest.raises(zu.UploadError, match='stray.bin'):
        zu.upload(build, BASE, 'tok', session=s)
    no_network(s)


def test_missing_file_in_build_dir_rejected(build):
    (build / 'b.tar.gz').unlink()
    s = FakeSession()
    with pytest.raises(zu.UploadError, match='b.tar.gz'):
        zu.upload(build, BASE, 'tok', session=s)
    no_network(s)


def test_missing_readme_rejected(build):
    (build / 'README_DATA.md').unlink()
    s = FakeSession()
    with pytest.raises(zu.UploadError, match='README_DATA.md'):
        zu.upload(build, BASE, 'tok', session=s)
    no_network(s)


def test_part_file_rejected(build):
    (build / 'a.tar.part').write_bytes(b'x')
    with pytest.raises(zu.UploadError, match='a.tar.part'):
        zu.upload(build, BASE, 'tok', session=FakeSession())


def test_archive_size_or_md5_mismatch_rejected(build):
    (build / 'a.tar').write_bytes(b'AAAA')            # size differs
    s = FakeSession()
    with pytest.raises(zu.UploadError, match='size'):
        zu.upload(build, BASE, 'tok', session=s)
    (build / 'a.tar').write_bytes(b'AAB')             # same size, md5 differs
    with pytest.raises(zu.UploadError, match='md5'):
        zu.upload(build, BASE, 'tok', session=s)
    no_network(s)


def test_readme_md5_mismatch_rejected(build):
    (build / 'README_DATA.md').write_bytes(b'changed')
    s = FakeSession()
    with pytest.raises(zu.UploadError, match='README'):
        zu.upload(build, BASE, 'tok', session=s)
    no_network(s)


def test_missing_manifest_rejected(build):
    (build / 'MANIFEST.json').unlink()
    s = FakeSession()
    with pytest.raises(zu.UploadError, match='MANIFEST'):
        zu.upload(build, BASE, 'tok', session=s)
    no_network(s)


def test_resume_refuses_stray_remote_file(build):
    s = FakeSession(existing={'a.tar': hashlib.md5(b'AAA').hexdigest(),
                              'old_partial.tar': 'zzz'})
    with pytest.raises(zu.UploadError, match='old_partial.tar'):
        zu.upload(build, BASE, 'tok', deposition='7', session=s)
    assert not s.puts
    assert 'DELETE' not in {m for m, _ in s.calls}


def test_http_error_wrapped_with_truncated_body(build):
    s = FakeSession(fail_put=(403, 'forbidden ' + 'x' * 5000))
    with pytest.raises(zu.UploadError) as ei:
        zu.upload(build, BASE, 'SECRETTOKEN', session=s)
    msg = str(ei.value)
    assert '403' in msg and 'forbidden' in msg and len(msg) < 1000


def test_target_must_be_explicit(build, monkeypatch, capsys):
    monkeypatch.setenv('ZENODO_TOKEN', 'tok')
    with pytest.raises(SystemExit):
        zu.main(['--build-dir', str(build)])
    with pytest.raises(SystemExit):
        zu.main(['--build-dir', str(build), '--sandbox', '--production'])


@pytest.mark.parametrize('flag,host', [('--sandbox', 'sandbox.zenodo.org'),
                                       ('--production', 'https://zenodo.org')])
def test_target_selects_host(build, monkeypatch, flag, host):
    monkeypatch.setenv('ZENODO_TOKEN', 'tok')
    s = FakeSession()
    s.dep['links']['bucket'] = 'https://b/x'
    monkeypatch.setattr(requests, 'Session', lambda: s)
    assert zu.main(['--build-dir', str(build), flag]) == 0
    assert host in s.calls[0][1]
    if flag == '--production':
        assert 'sandbox' not in s.calls[0][1]


TOKEN = 'ZTOK-9f3a7c1d-DISTINCTIVE'


def test_token_only_in_authorization_header_and_never_leaked(build, monkeypatch, capsys):
    monkeypatch.setenv('ZENODO_TOKEN', TOKEN)
    # success path
    s = FakeSession()
    monkeypatch.setattr(requests, 'Session', lambda: s)
    assert zu.main(['--build-dir', str(build), '--sandbox']) == 0
    for (method, url), kw in zip(s.calls, s.kwargs):
        assert TOKEN not in url
        assert not kw.get('params')
        assert kw['headers'] == {'Authorization': f'Bearer {TOKEN}'}
        assert TOKEN not in json.dumps(kw.get('json') or {})
    # HTTP error whose body echoes the token, UploadError paths
    s2 = FakeSession(fail_put=(401, f'bad token {TOKEN}'))
    monkeypatch.setattr(requests, 'Session', lambda: s2)
    assert zu.main(['--build-dir', str(build), '--sandbox']) == 1
    (build / 'extra').write_bytes(b'x')
    assert zu.main(['--build-dir', str(build), '--sandbox']) == 1
    s3 = FakeSession(bad_checksum=True)
    (build / 'extra').unlink()
    monkeypatch.setattr(requests, 'Session', lambda: s3)
    assert zu.main(['--build-dir', str(build), '--production']) == 1
    cap = capsys.readouterr()
    assert TOKEN not in cap.out and TOKEN not in cap.err
    # raw exception text
    s4 = FakeSession(fail_put=(401, f'bad token {TOKEN}'))
    with pytest.raises(zu.UploadError) as ei:
        zu.upload(build, BASE, TOKEN, session=s4)
    assert TOKEN not in str(ei.value) and TOKEN not in repr(ei.value)


@pytest.mark.parametrize('where', ['top', 'archive'])
@pytest.mark.parametrize('key,val', [('sdi_git_dirty', True), ('sdi_git_commit', None)])
def test_dirty_or_unknown_build_refused(build, where, key, val):
    mp = build / 'MANIFEST.json'
    man = json.loads(mp.read_text())
    (man if where == 'top' else man['archives']['a.tar'])[key] = val
    mp.write_text(json.dumps(man))
    with pytest.raises(zu.UploadError, match='allow-dirty-build'):
        zu.verify_build(build)
    assert set(zu.verify_build(build, allow_dirty_build=True)) >= {'a.tar'}


def test_manifest_and_readme_go_last_after_every_archive(build):
    order = []
    s = FakeSession()
    put = s.put
    s.put = lambda url, data=None, **kw: (order.append(url.rsplit('/', 1)[1]),
                                          put(url, data=data, **kw))[1]
    zu.upload(build, BASE, 'tok', session=s, jobs=8)
    assert set(order[:2]) == {'a.tar', 'b.tar.gz'}
    assert order[2:] == ['MANIFEST.json', 'README_DATA.md']


def test_archives_upload_concurrently(build):
    """Both archive PUTs must be in flight at once (a barrier that only a
    second concurrent PUT can release); serial uploads would time out."""
    import threading
    barrier = threading.Barrier(2, timeout=10)
    s = FakeSession()
    put = s.put

    def gated(url, data=None, **kw):
        if not url.endswith(('MANIFEST.json', 'README_DATA.md')):
            barrier.wait()
        return put(url, data=data, **kw)
    s.put = gated
    zu.upload(build, BASE, 'tok', session=s, jobs=2)
    assert set(s.puts) == {'a.tar', 'b.tar.gz', 'MANIFEST.json', 'README_DATA.md'}


def test_failed_archive_blocks_manifest_and_prints_resume(build, capsys):
    s = FakeSession()
    put = s.put

    def failing(url, data=None, **kw):
        if url.endswith('b.tar.gz'):
            return Resp({}, status=500, text='boom')
        return put(url, data=data, **kw)
    s.put = failing
    with pytest.raises(zu.UploadError, match='b.tar.gz'):
        zu.upload(build, BASE, 'tok', session=s)
    assert 'MANIFEST.json' not in s.puts and 'README_DATA.md' not in s.puts
    assert 'resume with --deposition 7' in capsys.readouterr().err


def test_resume_accepts_md5_prefixed_checksum(build):
    s = FakeSession(existing={'a.tar': 'md5:' + hashlib.md5(b'AAA').hexdigest()})
    zu.upload(build, BASE, 'tok', deposition='7', session=s)
    assert 'a.tar' not in s.puts


def _blocking_session(release, started, fail_name=None):
    """PUTs of archives block until `release` is set (after signalling
    `started`); `fail_name` fails at once instead."""
    s = FakeSession()
    put = s.put

    def blocking(url, data=None, **kw):
        name = url.rsplit('/', 1)[1]
        if name == fail_name:
            return Resp({}, status=500, text='boom')
        if name not in ('MANIFEST.json', 'README_DATA.md'):
            started.release()
            release.wait(30)
        return put(url, data=data, **kw)
    s.put = blocking
    return s


def test_interrupt_is_not_held_by_inflight_uploads(build, capsys, monkeypatch):
    """Ctrl-c while archive PUTs are blocked must leave upload() promptly
    (a `with ThreadPoolExecutor` would join the workers first), print the
    draft id, and not start MANIFEST/README."""
    import threading
    import time as _t
    release, started = threading.Event(), threading.Semaphore(0)
    s = _blocking_session(release, started)
    real = zu.as_completed

    def interrupting(fs):
        started.acquire(timeout=10)
        raise KeyboardInterrupt
    monkeypatch.setattr(zu, 'as_completed', interrupting)
    t0 = _t.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            zu.upload(build, BASE, 'tok', session=s)
        assert _t.monotonic() - t0 < 5
    finally:
        release.set()
    assert 'resume with --deposition 7' in capsys.readouterr().err
    assert 'MANIFEST.json' not in s.puts
    monkeypatch.setattr(zu, 'as_completed', real)


def test_archive_failure_reported_while_sibling_in_flight(build, capsys):
    """A failed archive is printed as soon as it fails, not after the other
    uploads end; MANIFEST/README are then withheld. (Guards the early print
    only; the executor-join hazard is guarded by the interrupt test.)"""
    import threading
    release, started = threading.Event(), threading.Semaphore(0)
    s = _blocking_session(release, started, fail_name='b.tar.gz')
    seen = threading.Event()

    def watch():
        started.acquire(timeout=10)
        for _ in range(100):
            if 'FAILED b.tar.gz' in capsys.readouterr().err:
                seen.set()
                break
            threading.Event().wait(0.05)
        release.set()
    w = threading.Thread(target=watch)
    w.start()
    with pytest.raises(zu.UploadError, match='b.tar.gz'):
        zu.upload(build, BASE, 'tok', session=s)
    w.join()
    assert seen.is_set()
    assert 'MANIFEST.json' not in s.puts


def test_jobs_below_one_rejected_before_network(build):
    s = FakeSession()
    with pytest.raises(zu.UploadError, match='jobs'):
        zu.upload(build, BASE, 'tok', session=s, jobs=0)
    assert s.calls == []


def test_post_upload_reconciliation_catches_missing_file(build):
    s = FakeSession()
    get = s.get

    def lossy(url, **kw):
        r = get(url, **kw)
        if url.endswith('/files'):
            r._js = [f for f in r._js if f['filename'] != 'a.tar']
        return r
    s.get = lossy
    with pytest.raises(zu.UploadError, match='a.tar'):
        zu.upload(build, BASE, 'tok', session=s)


def test_failure_cancels_queued_archives(build):
    """With one worker, a failure on the first (largest) archive must stop the
    queued one from starting."""
    s = FakeSession()
    put = s.put

    def failing(url, data=None, **kw):
        if url.endswith('b.tar.gz'):               # largest, so uploaded first
            return Resp({}, status=500, text='boom')
        return put(url, data=data, **kw)
    s.put = failing
    with pytest.raises(zu.UploadError, match='b.tar.gz'):
        zu.upload(build, BASE, 'tok', session=s, jobs=1)
    assert 'a.tar' not in s.puts
