#!/usr/bin/env python
"""Upload a packed build directory to a Zenodo DRAFT deposition. Never publishes.

    ZENODO_TOKEN=... python scripts/zenodo_upload.py --build-dir DIR
                          (--sandbox | --production) [--deposition ID] [--jobs N]

The target is explicit: exactly one of --sandbox / --production.

Before any network call the build dir is checked against its MANIFEST.json:
its file set must be exactly the manifest's archives plus MANIFEST.json and
README_DATA.md, and each archive's size and md5 (and the README's md5) must
match the manifest. The upload enumerates that set, never the directory.

Without --deposition a new draft is created from zenodo/metadata.json (plus
prereserve_doi); with it that draft is reused and files already in its bucket
with a matching md5 are skipped (resume). A draft holding any file outside the
expected set is refused (nothing is deleted remotely; remove it by hand).
If anything fails once a draft exists, its id and URL are printed.

Archives go up --jobs at a time (default 8), largest first, each as its own
PUT: a single stream to Zenodo ran at ~0.85 MB/s from LBNL (round-trip bound),
so parallel streams should multiply throughput. MANIFEST.json and
README_DATA.md follow only after every archive has succeeded.
The token is sent only in the Authorization header. Publishing is deliberately
not implemented: review the draft in the web UI and publish by hand.
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK = 1 << 20
README_NAME = 'README_DATA.md'
MANIFEST_NAME = 'MANIFEST.json'
ERR_BODY_MAX = 500
METADATA_PATH = Path(__file__).resolve().parent.parent / 'zenodo' / 'metadata.json'


class UploadError(Exception):
    pass


def md5sum(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


def verify_build(build_dir, allow_dirty_build=False):
    """Check build_dir against its MANIFEST.json. -> ordered {filename: md5}
    of the files to upload (archives, then MANIFEST.json, README_DATA.md)."""
    build_dir = Path(build_dir)
    mpath = build_dir / MANIFEST_NAME
    if not mpath.is_file():
        raise UploadError(f'{mpath} not found; run zenodo_pack.py first')
    try:
        man = json.loads(mpath.read_text())
        archives = man['archives']
        readme = man['readme']
    except (ValueError, KeyError) as e:
        raise UploadError(f'{mpath} is unusable ({type(e).__name__}: {e})')
    if not allow_dirty_build:
        bad = [k for k, rec in [('(top level)', man), *archives.items()]
               if rec.get('sdi_git_dirty') is not False or not rec.get('sdi_git_commit')]
        if bad:
            raise UploadError(f'build was made from a dirty or unknown SDI checkout '
                              f'(sdi_git_dirty/sdi_git_commit): {bad}; rebuild from a '
                              'clean commit or pass --allow-dirty-build')
    expected = set(archives) | {MANIFEST_NAME, README_NAME}
    present = {p.name for p in build_dir.iterdir()}
    extra, missing = sorted(present - expected), sorted(expected - present)
    if extra or missing:
        raise UploadError(f'build dir does not match MANIFEST.json: '
                          f'unexpected files {extra}, missing files {missing}')
    md5s = {}
    for name in sorted(archives):
        path = build_dir / name
        if not path.is_file():
            raise UploadError(f'{name} is not a regular file')
        if path.stat().st_size != archives[name]['size']:
            raise UploadError(f'{name}: size {path.stat().st_size} != manifest '
                              f'{archives[name]["size"]}')
        md5s[name] = md5sum(path)
        if md5s[name] != archives[name]['md5']:
            raise UploadError(f'{name}: md5 differs from MANIFEST.json')
    md5s[MANIFEST_NAME] = md5sum(mpath)
    md5s[README_NAME] = md5sum(build_dir / README_NAME)
    if md5s[README_NAME] != readme['md5']:
        raise UploadError(f'{README_NAME}: md5 differs from MANIFEST.json')
    return md5s


def _check(r, what, token):
    """raise_for_status, wrapped: UploadError with a truncated body (token
    scrubbed)."""
    import requests
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        resp = getattr(e, 'response', None) or r
        try:
            body = str(resp.text)
        except Exception:
            body = ''
        body = body.replace(token, '***')[:ERR_BODY_MAX]
        status = getattr(resp, 'status_code', '?')
        raise UploadError(f'{what}: HTTP {status}: {body}') from None


def upload(build_dir, base, token, deposition=None, session=None,
           metadata_path=METADATA_PATH, allow_dirty_build=False, jobs=8):
    build_dir = Path(build_dir)
    if jobs < 1:
        raise UploadError(f'--jobs must be >= 1, got {jobs}')
    md5s = verify_build(build_dir, allow_dirty_build)          # before any network call
    if session is None:
        import requests
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=max(10, jobs))
        session.mount('https://', adapter)
    headers = {'Authorization': f'Bearer {token}'}
    api = f'{base}/api/deposit/depositions'

    if deposition is None:
        meta = json.loads(Path(metadata_path).read_text())
        meta['prereserve_doi'] = True
        r = session.post(api, json={'metadata': meta}, headers=headers, timeout=60)
        _check(r, 'create deposition', token)
    else:
        r = session.get(f'{api}/{deposition}', headers=headers, timeout=60)
        _check(r, f'get deposition {deposition}', token)
    dep = r.json()
    if dep.get('submitted'):
        raise UploadError('deposition is already published; refusing to modify it')
    html = (dep.get('links') or {}).get('html')
    print(f'deposition {dep["id"]} (draft, NOT published): {html}', flush=True)
    try:
        bucket = dep['links']['bucket']
        existing = {}
        if deposition is not None:
            rf = session.get(f'{api}/{dep["id"]}/files', headers=headers, timeout=60)
            _check(rf, 'list draft files', token)
            existing = {f['filename']: f.get('checksum') for f in rf.json()}
            stray = sorted(set(existing) - set(md5s))
            if stray:
                raise UploadError(f'draft {dep["id"]} holds files not in the build '
                                  f'manifest: {stray}; remove them in the web UI '
                                  '(nothing is deleted remotely) and rerun')

        stop = threading.Event()        # set by the first failure

        def put_one(name):
            if stop.is_set():
                return                      # a sibling failed; don't start
            try:
                _put_one(name)
            except BaseException:
                stop.set()                  # in the worker, before it takes the next item
                raise

        def _put_one(name):
            md5 = md5s[name]
            if (existing.get(name) or '').split(':')[-1] == md5:
                print(f'skip {name} (already uploaded, md5 matches)', flush=True)
                return
            size = (build_dir / name).stat().st_size
            print(f'uploading {name} ({size / 1e9:.2f} GB) ...', flush=True)
            t0 = time.monotonic()
            with open(build_dir / name, 'rb') as fh:
                put = session.put(f'{bucket}/{name}', data=fh, headers=headers,
                                  timeout=(30, 3600))
            _check(put, f'upload {name}', token)
            got = (put.json().get('checksum') or '').split(':')[-1]
            if got != md5:
                raise UploadError(f'{name}: Zenodo md5 {got!r} != local {md5}')
            dt = time.monotonic() - t0
            print(f'done {name} in {dt / 60:.1f} min ({size / 1e6 / max(dt, 1e-9):.2f} MB/s)',
                  flush=True)

        # Archives in parallel (one TCP stream each; a single stream to Zenodo is
        # latency-bound), then MANIFEST.json and README_DATA.md only once every
        # archive is in, so a draft holding the MANIFEST is complete. No `with`:
        # its __exit__ joins the workers, which would hold an interrupt or a
        # failure until every in-flight PUT (hours) had finished.
        archives = sorted((n for n in md5s if n not in (MANIFEST_NAME, README_NAME)),
                          key=lambda n: -(build_dir / n).stat().st_size)
        pool = ThreadPoolExecutor(max_workers=jobs)
        try:
            futures = {pool.submit(put_one, n): n for n in archives}
            failed = {}
            for f in as_completed(futures):
                e = f.exception()
                if e is not None:
                    failed[futures[f]] = f'{type(e).__name__}: {e}'.replace(token, '***')
                    print(f'FAILED {futures[f]}: {failed[futures[f]]} (in-flight uploads '
                          'continue and are kept; Ctrl-c to stop)', file=sys.stderr,
                          flush=True)
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        if failed:
            first = sorted(failed)[0]           # each one's full error is printed above
            raise UploadError(f'{len(failed)} archive(s) failed: {sorted(failed)} (uploads '
                              f'that finished are kept); {first}: {failed[first]}')
        for name in (MANIFEST_NAME, README_NAME):
            put_one(name)

        # Reconcile: the draft must now hold exactly the build's files, each
        # with the local md5.
        rf = session.get(f'{api}/{dep["id"]}/files', headers=headers, timeout=60)
        _check(rf, 'list draft files after upload', token)
        remote = {f['filename']: (f.get('checksum') or '').split(':')[-1] for f in rf.json()}
        if remote != md5s:
            bad = sorted(n for n in set(remote) | set(md5s) if remote.get(n) != md5s.get(n))
            raise UploadError(f'draft files differ from the build after upload: {bad}')
        print(f'draft {dep["id"]} holds all {len(md5s)} files with matching md5', flush=True)
    except BaseException:
        print(f'FAILED with draft deposition {dep["id"]} still on the server: {html} '
              f'(resume with --deposition {dep["id"]})', file=sys.stderr, flush=True)
        raise

    doi = ((dep.get('metadata') or {}).get('prereserve_doi') or {}).get('doi')
    print(f'reserved DOI: {doi}')
    return dep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--build-dir', required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--sandbox', action='store_true', help='use sandbox.zenodo.org')
    g.add_argument('--production', action='store_true', help='use zenodo.org')
    ap.add_argument('--deposition', help='existing draft deposition id (resume)')
    ap.add_argument('--allow-dirty-build', action='store_true',
                    help='upload a build whose manifest records a dirty/unknown SDI commit')
    ap.add_argument('--jobs', type=int, default=8,
                    help='archives uploaded concurrently (default 8)')
    args = ap.parse_args(argv)
    token = os.environ.get('ZENODO_TOKEN')
    if not token:
        print('ERROR: ZENODO_TOKEN is not set', file=sys.stderr)
        return 1
    base = 'https://sandbox.zenodo.org' if args.sandbox else 'https://zenodo.org'
    try:
        upload(args.build_dir, base, token, args.deposition,
               allow_dirty_build=args.allow_dirty_build, jobs=args.jobs)
    except UploadError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1
    except OSError as e:                 # network errors (requests.RequestException)
        print(f'ERROR: {type(e).__name__}: {str(e).replace(token, "***")}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Worker threads blocked in a PUT would otherwise hold the exit until
        # their uploads finish; an interrupted PUT leaves no file on Zenodo.
        print('interrupted; if a "deposition <ID>" line is printed above, resume with '
              '--deposition <ID>; if the interrupt came while the draft was being '
              'created, check the Zenodo web UI for a stray draft', file=sys.stderr,
              flush=True)
        os._exit(130)
    return 0


if __name__ == '__main__':
    sys.exit(main())
