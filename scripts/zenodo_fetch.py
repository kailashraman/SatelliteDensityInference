#!/usr/bin/env python
"""Download and unpack the Zenodo data deposit into a repository checkout.

    python scripts/zenodo_fetch.py --record ID [--sandbox] [--tier figures|inputs|all]
                                   [--dest REPO_ROOT] [--tmp DIR] [--files-from DIR] [--force]

Lists the record's files (GET /api/records/ID), downloads MANIFEST.json and the
archives of the requested tier into a temporary directory under --dest (or
--tmp; never $TMPDIR, the archives are large), verifies the Zenodo md5 of
MANIFEST.json and of each archive (a missing record checksum is an error) and
the MANIFEST sha256/md5 of each archive, then extracts under --dest (default:
this repo). Before an archive is extracted its members are validated: each must
be a regular file listed in MANIFEST for that archive, and absolute or '..'
paths are rejected. Before anything is extracted, every selected archive is
checked against existing files: one whose content differs from the MANIFEST
sha256 aborts the run unless --force, and an existing non-file (directory,
symlink, dangling symlink) at a target path always aborts, even with --force.
An existing file already holding the MANIFEST sha256 is left untouched (not
rewritten or re-mtimed); a differing one is unlinked before extraction, so a
hardlinked file is never rewritten through the link (replacing a file needs
write permission on its directory). Each extracted file's sha256 is verified
against the MANIFEST.
Extraction is not transactional: if it fails part-way (corrupt archive, sha256
mismatch, OSError such as a read-only directory or full disk), the files
extracted before the failure stay in --dest (only --tmp is cleaned up); fix the
cause and rerun with --force.
Extracted files get an mtime of the extraction time (the archives carry mtime
0, which would otherwise date every file 1970). A corrupt or unsafe archive
(tarfile errors) is reported as ERROR with a nonzero exit.
Requires Python's tarfile.data_filter (3.12+, or a security backport).
--files-from DIR reads the same files from a local directory instead (offline
testing); no network is touched.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

CHUNK = 1 << 20
DJA_ARC_PREFIX = 'data/dja_results/'


class FetchError(Exception):
    pass


def file_digests(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(CHUNK), b''):
            sha.update(b)
            md5.update(b)
    return sha.hexdigest(), md5.hexdigest()


def list_record(record, base, session):
    r = session.get(f'{base}/api/records/{record}', timeout=60)
    r.raise_for_status()
    files = {}
    for f in r.json().get('files', []):
        name = f.get('key') or f.get('filename')
        md5 = (f.get('checksum') or '').split(':')[-1]
        url = (f.get('links') or {}).get('self') or (f.get('links') or {}).get('download')
        files[name] = {'md5': md5, 'url': url}
    return files


def download(url, dest, session):
    with session.get(url, stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        with open(dest, 'wb') as out:
            for b in r.iter_content(CHUNK):
                out.write(b)


def check_member_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise FetchError(f'unsafe path in archive: {name!r}')


def validate_archive(path, members):
    """Raise FetchError unless every tar member is a regular file with a safe
    name that is listed in members {path: sha256}."""
    with tarfile.open(path, 'r|*') as tar:
        for m in tar:
            check_member_name(m.name)
            if not m.isreg():
                raise FetchError(f'archive member {m.name!r} is not a regular file')
            if m.name not in members:
                raise FetchError(f'archive member {m.name!r} is not listed in '
                                 'MANIFEST.json for this archive')


def extract(archive, dest, members):
    """Validate, then extract, verifying each file's sha256 against members
    {path: sha256}. An identical existing file stays untouched; a differing one
    is unlinked first, so a hardlinked file is never rewritten through the link
    (needs write permission on its directory). Not transactional: files
    extracted before a failure remain."""
    try:
        validate_archive(archive, members)
        now = time.time()
        with tarfile.open(archive, 'r|*') as tar:
            for m in tar:
                target = Path(dest) / m.name
                if target.is_file() and not target.is_symlink():
                    if file_digests(target)[0] == members[m.name]:
                        continue                # identical: leave untouched
                    target.unlink()             # never write through a hardlink
                tar.extract(m, path=dest, filter='data')
                if file_digests(target)[0] != members[m.name]:
                    raise FetchError(f'{m.name}: sha256 differs from MANIFEST.json '
                                     '(extraction stopped)')
                os.utime(target, (now, now))
    except (tarfile.TarError, EOFError, OSError) as e:
        raise FetchError(f'{Path(archive).name}: cannot extract '
                         f'({type(e).__name__}: {e})') from e


def check_conflicts(dest, by_arch, names):
    """Existing files under dest that differ from the MANIFEST sha256, over all
    the selected archives."""
    conflicts = []
    for name in names:
        for rel, sha in by_arch.get(name, {}).items():
            target = Path(dest) / rel
            if target.is_file() and not target.is_symlink() \
                    and file_digests(target)[0] != sha:
                conflicts.append(rel)
    return conflicts


def check_nonfiles(dest, by_arch, names):
    """Target paths occupied by something other than a regular file (directory,
    symlink, dangling symlink), over all the selected archives; --force does not
    override these."""
    return [rel for name in names for rel in by_arch.get(name, {})
            if (t := Path(dest) / rel).is_symlink() or (t.exists() and not t.is_file())]


def fetch(record=None, base='https://zenodo.org', tier='all', dest=None,
          files_from=None, force=False, session=None, tmp=None):
    if not hasattr(tarfile, 'data_filter'):
        raise FetchError('this Python lacks tarfile.data_filter; refusing to '
                         'extract (upgrade to a release that has it)')
    dest = Path(dest)
    tmp_base = Path(tmp) if tmp else dest
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.zenodo_fetch_', dir=tmp_base))
    try:
        if files_from:
            src = Path(files_from)
            remote = {p.name: {'md5': None, 'path': p} for p in src.iterdir() if p.is_file()}
        else:
            if session is None:
                import requests
                session = requests.Session()
            remote = list_record(record, base, session)

        def get(name):
            if name not in remote:
                raise FetchError(f'{name} not in record')
            if files_from:
                return remote[name]['path']
            if not remote[name]['md5']:
                raise FetchError(f'{name}: the Zenodo record has no checksum for it')
            local = tmp / name
            download(remote[name]['url'], local, session)
            if file_digests(local)[1] != remote[name]['md5']:
                raise FetchError(f'{name}: md5 differs from the Zenodo checksum')
            return local

        manifest = json.loads(Path(get('MANIFEST.json')).read_text())
        want = sorted(n for n, a in manifest['archives'].items()
                      if tier == 'all' or tier in a['tiers'])
        if not want:
            raise FetchError(f'no archive in tier {tier!r}')
        by_arch = {}
        for m in manifest['members']:
            by_arch.setdefault(m['archive'], {})[m['path']] = m['sha256']
        for name in want:
            for rel in by_arch.get(name, {}):
                check_member_name(rel)
        nonfiles = check_nonfiles(dest, by_arch, want)
        if nonfiles:
            raise FetchError(f'{len(nonfiles)} target path(s) exist but are not regular '
                             f'files (first: {nonfiles[0]}); remove them and rerun')
        if not force:
            conflicts = check_conflicts(dest, by_arch, want)
            if conflicts:
                raise FetchError(
                    f'{len(conflicts)} existing file(s) differ from the archives '
                    f'(first: {conflicts[0]}); rerun with --force to overwrite')
        for name in want:
            rec = manifest['archives'][name]
            print(f'fetching {name} ({rec["size"] / 1e9:.3f} GB) ...', flush=True)
            path = get(name)
            sha, md5 = file_digests(path)
            if md5 != rec['md5'] or sha != rec['sha256']:
                raise FetchError(f'{name}: checksum differs from MANIFEST.json')
            extract(path, dest, by_arch.get(name, {}))
            if not files_from:
                path.unlink()
            print(f'  extracted {name}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if any(m.startswith(DJA_ARC_PREFIX) for a in want for m in by_arch.get(a, {})):
        print(f'export SDI_DJA_RESULTS_DIR={dest / "data" / "dja_results"}')
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--record', help='Zenodo record id')
    ap.add_argument('--sandbox', action='store_true', help='use sandbox.zenodo.org')
    ap.add_argument('--tier', choices=['figures', 'inputs', 'all'], default='all')
    ap.add_argument('--dest', default=str(Path(__file__).resolve().parent.parent),
                    help='repository root to unpack into')
    ap.add_argument('--tmp', metavar='DIR',
                    help='scratch directory for downloads (default: under --dest)')
    ap.add_argument('--files-from', metavar='DIR', help='local directory instead of Zenodo')
    ap.add_argument('--force', action='store_true',
                    help='overwrite existing files whose content differs')
    args = ap.parse_args(argv)
    if not args.record and not args.files_from:
        ap.error('--record or --files-from is required')
    base = 'https://sandbox.zenodo.org' if args.sandbox else 'https://zenodo.org'
    try:
        fetch(args.record, base, args.tier, args.dest, args.files_from, args.force,
              tmp=args.tmp)
    except (FetchError, OSError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
