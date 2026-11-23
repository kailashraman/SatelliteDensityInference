#!/usr/bin/env python
"""Upload a packed build directory to a Zenodo DRAFT deposition. Never publishes.

    ZENODO_TOKEN=... python scripts/zenodo_upload.py --build-dir DIR
                          (--sandbox | --production) [--deposition ID]

The target is explicit: exactly one of --sandbox / --production.

Before any network call the build dir is checked against its MANIFEST.json:
its file set must be exactly the manifest's archives plus MANIFEST.json and
README_DATA.md, and each archive's size and md5 (and the README's md5) must
match the manifest. The upload enumerates that set, never the directory.

Without --deposition a new draft is created from zenodo/metadata.json (plus
prereserve_doi); with it that draft is reused and files already in its bucket
with a matching md5 are skipped (resume). A draft holding any file outside the
expected set is refused (nothing is deleted remotely; remove it by hand).
The token is sent only in the Authorization header. Publishing is deliberately
not implemented: review the draft in the web UI and publish by hand.
"""

import argparse
import hashlib
import json
import os
import sys
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
           metadata_path=METADATA_PATH, allow_dirty_build=False):
    build_dir = Path(build_dir)
    md5s = verify_build(build_dir, allow_dirty_build)          # before any network call
    if session is None:
        import requests
        session = requests.Session()
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
    print(f'deposition {dep["id"]} (draft, NOT published): '
          f'{(dep.get("links") or {}).get("html")}', flush=True)
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

    for name, md5 in md5s.items():
        if existing.get(name) == md5:
            print(f'skip {name} (already uploaded, md5 matches)')
            continue
        print(f'uploading {name} ...', flush=True)
        with open(build_dir / name, 'rb') as fh:
            put = session.put(f'{bucket}/{name}', data=fh, headers=headers,
                              timeout=(30, 3600))
        _check(put, f'upload {name}', token)
        got = (put.json().get('checksum') or '').split(':')[-1]
        if got != md5:
            raise UploadError(f'{name}: Zenodo md5 {got!r} != local {md5}')

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
    args = ap.parse_args(argv)
    token = os.environ.get('ZENODO_TOKEN')
    if not token:
        print('ERROR: ZENODO_TOKEN is not set', file=sys.stderr)
        return 1
    base = 'https://sandbox.zenodo.org' if args.sandbox else 'https://zenodo.org'
    try:
        upload(args.build_dir, base, token, args.deposition,
               allow_dirty_build=args.allow_dirty_build)
    except UploadError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
