#!/usr/bin/env python
"""Pack the Zenodo data deposit described by zenodo/deposit_spec.json.

    python scripts/zenodo_pack.py --out DIR [--dry-run] [--only NAME ...]
                                  [--allow-dirty]

Writes the archives plus DIR/MANIFEST.json (per-member size/sha256, per-archive
size/sha256/md5/tiers and sdi_git_commit/dirty/built_utc stamps, producing SDI
and DwarfJeansAnalysis commits, the path remap table, the README_DATA.md
checksums) and copies zenodo/README_DATA.md into DIR. Archives store
repo-relative paths and unpack at the repo root; the DJA snapshot is stored
under data/dja_results/.

Checks (always, also with --dry-run); any failure is a hard failure:
  - zenodo/README_DATA.md exists;
  - the SDI checkout has no modified/staged tracked files (--allow-dirty
    overrides, recording dirty=true), and the files that define the build
    (deposit_spec.json, README_DATA.md, metadata.json, this script) are tracked
    and equal to HEAD (same override);
  - every include pattern alternative matches at least one file;
  - every input recorded in every paper-figure manifest (inputs, DJA snapshot
    files, top-level h5_fingerprint) is, after the remap, a member of an
    archive whose tiers include "figures", or a git-tracked file (it ships with
    the code);
  - each such non-directory input's on-disk size equals the size recorded in
    the manifest, and each git-tracked input exists and is unmodified vs HEAD.

A build unlinks MANIFEST.json first, writes each archive to <name>.part and
renames it on success, so an interrupted build never leaves a stale manifest
next to new archives. With --only, the other archives' existing records
(including their stamps) are kept.

Builds are deterministic (sorted members, normalized uid/gid/uname/mtime,
gzip mtime=0), so a rebuild is byte-identical within one Python/zlib
environment (the gzip stream depends on the zlib version).
"""

import argparse
import datetime
import glob
import gzip
import hashlib
import io
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python'))

import config

SPEC_PATH = config.REPO_ROOT / 'zenodo' / 'deposit_spec.json'
README_NAME = 'README_DATA.md'
README_PATH = config.REPO_ROOT / 'zenodo' / README_NAME
METADATA_PATH = config.REPO_ROOT / 'zenodo' / 'metadata.json'
EXCLUDED_DIR = '.ipynb_checkpoints'
SIDECAR_SUFFIX = '.prov.json'
DJA_ARC_PREFIX = 'data/dja_results/'
CHUNK = 1 << 20

# Recorded paths predate commit 2e655c4 (data/additional/ split) and the DJA
# checkout move. Applied in order, first match wins, on the repo-relative path
# (or, for DJA, the path relative to the DJA results dir).
REPO_REMAP = [
    (r'^data/additional/weights_gc_published/', 'data/halo_weights_published/'),
    (r'^data/additional/weights_gc/', 'data/halo_weights/'),
    (r'^data/additional/mhalf/', 'data/mhalf/'),
    (r'^data/additional/(?=[^/]+$)', 'data/satgen_catalogs/'),
]
REPO_ROOT_PATTERN = r'^.*?/SatelliteDensityInference/'
DJA_ROOT_PATTERN = r'^.*/DwarfJeansAnalysis/results/production/'


# ---------------------------------------------------------------- spec / files

def load_spec(path=SPEC_PATH):
    return json.loads(Path(path).read_text())['archives']


def expand_braces(pattern):
    """One level of {a,b} alternation (names contain spaces, not braces)."""
    m = re.search(r'\{([^{}]*)\}', pattern)
    if not m:
        return [pattern]
    return [pattern[:m.start()] + alt + pattern[m.end():]
            for alt in m.group(1).split(',')]


def with_sidecars(files):
    """files: {arcname: src Path}. Add <f>.prov.json where it exists."""
    out = dict(files)
    for arc, src in files.items():
        sc = src.with_name(src.name + SIDECAR_SUFFIX)
        if not arc.endswith(SIDECAR_SUFFIX) and sc.is_file():
            out.setdefault(arc + SIDECAR_SUFFIX, sc)
    return out


def resolve_globs(patterns, repo_root):
    """-> ({arcname: src}, [brace-expanded alternatives that matched nothing]).

    Hidden files match (glob include_hidden, Python 3.11+); `.ipynb_checkpoints`
    directories are excluded."""
    files, empty = {}, []
    for pat in patterns:
        for p in expand_braces(pat):
            hit = False
            for m in glob.glob(str(Path(repo_root) / p), recursive=True,
                               include_hidden=True):
                m = Path(m)
                if EXCLUDED_DIR in m.relative_to(repo_root).parts:
                    continue
                if m.is_file():
                    files[m.relative_to(repo_root).as_posix()] = m
                    hit = True
            if not hit:
                empty.append(p)
    return with_sidecars(files), empty


# ------------------------------------------------------------ figure manifests

def paper_manifests(repo_root, names=None):
    """Paper-figure/table manifest paths under plots/ (supplementary skipped).

    names=None uses figure_registry (the drafts' \\includegraphics/\\input
    basenames); a table `t` is recorded as `t.tex`. Every paper artifact must
    have a manifest, otherwise coverage cannot be asserted.
    """
    if names is None:
        import figure_registry as fr
        if not fr.DRAFTS_AVAILABLE:
            raise RuntimeError('no drafts under config.DRAFTS_DIR; cannot '
                               'determine the paper figure list')
        names = set(fr.FIGURES) | {t + '.tex' for t in fr.TABLES}
    found = {}
    for f in Path(repo_root, 'plots').rglob('*' + SIDECAR_SUFFIX):
        base = f.name[:-len(SIDECAR_SUFFIX)]
        s = f.as_posix()
        if 'supplementary' in s or '.ipynb_checkpoints' in s:
            continue
        found.setdefault(base, []).append(f)
    missing = sorted(set(names) - set(found))
    if missing:
        raise RuntimeError(f'paper artifacts with no figure manifest: {missing}')
    dup = {k: v for k, v in found.items() if k in names and len(v) > 1}
    if dup:
        raise RuntimeError(f'ambiguous figure manifests: {dup}')
    return {n: found[n][0] for n in sorted(names)}


def recorded_entries(manifest):
    """[(path, recorded size or None)] for the inputs, DJA snapshot files and
    top-level h5_fingerprint of one figure manifest."""
    d = json.loads(Path(manifest).read_text())
    out = [(i['path'], i.get('size')) for i in d.get('inputs', [])]
    snaps = d.get('dja_snapshot')
    for snap in (snaps if isinstance(snaps, list) else [snaps] if snaps else []):
        for y in snap.get('files', []):
            out.append((y['path'], y.get('size')) if isinstance(y, dict) else (y, None))
    fp = d.get('h5_fingerprint')
    if isinstance(fp, dict) and fp.get('path'):
        out.append((fp['path'], fp.get('size')))
    return out


def recorded_paths(manifest):
    return [p for p, _ in recorded_entries(manifest)]


def remap(path, repo_root, dja_dir):
    """Recorded absolute path -> (kind, rel). kind: 'repo' | 'dja' | 'other'.

    'repo' rel is repo-relative after the data/additional remap; 'dja' rel is
    relative to the DJA results dir.
    """
    path = str(path)
    dja = str(dja_dir).rstrip('/') + '/'
    if path.startswith(dja):
        return 'dja', path[len(dja):]
    m = re.match(DJA_ROOT_PATTERN, path)
    if m:
        return 'dja', path[m.end():]
    root = str(repo_root).rstrip('/') + '/'
    if path.startswith(root):
        rel = path[len(root):]
    else:
        m = re.match(REPO_ROOT_PATTERN, path)
        if not m:
            return 'other', path
        rel = path[m.end():]
    for pat, repl in REPO_REMAP:
        new = re.sub(pat, repl, rel, count=1)
        if new != rel:
            rel = new
            break
    return 'repo', rel


def manifest_inputs(repo_root, dja_dir, names=None):
    """-> ({(kind, rel): sorted figure names}, {(kind, rel): {recorded sizes}}).

    Directory inputs are expanded to their files (no recorded size).
    """
    out, sizes = {}, {}
    for fig, man in paper_manifests(repo_root, names).items():
        for p, size in recorded_entries(man):
            kind, rel = remap(p, repo_root, dja_dir)
            base = Path(repo_root if kind == 'repo' else dja_dir)
            src = base / rel
            if kind != 'other' and src.is_dir():
                for f in sorted(src.rglob('*')):
                    if f.is_file():
                        out.setdefault((kind, f.relative_to(base).as_posix()),
                                       set()).add(fig)
            else:
                out.setdefault((kind, rel), set()).add(fig)
                if size is not None:
                    sizes.setdefault((kind, rel), set()).add(size)
    return {k: sorted(v) for k, v in out.items()}, sizes


def resolve_from_manifests(cfg, inputs, repo_root, dja_dir):
    files = {}
    if cfg['source'] == 'repo':
        for (kind, rel) in inputs:
            if kind == 'repo' and any(rel.startswith(u) for u in cfg['under']):
                files[rel] = Path(repo_root) / rel
    elif cfg['source'] == 'dja':
        for (kind, rel) in inputs:
            if kind == 'dja':
                files[cfg['arc_prefix'] + rel] = Path(dja_dir) / rel
    else:
        raise ValueError(f"unknown from_figure_manifests source {cfg['source']!r}")
    return with_sidecars(files)


def resolve_archives(spec, repo_root, dja_dir, names=None):
    """-> ({archive: {arcname: src}}, inputs, sizes, [problems])."""
    inputs, sizes = manifest_inputs(repo_root, dja_dir, names)
    resolved, problems = {}, []
    for a in spec:
        if 'include' in a:
            files, empty = resolve_globs(a['include'], repo_root)
            problems += [f"{a['name']}: pattern alternative matches no file: {p}"
                         for p in empty]
        else:
            files = resolve_from_manifests(a['from_figure_manifests'], inputs,
                                           repo_root, dja_dir)
        for arc, src in files.items():
            if not src.is_file():
                problems.append(f"{a['name']}: missing source {src}")
            try:
                normalize_arcname(arc)
            except ValueError as e:
                problems.append(f"{a['name']}: {e}")
        resolved[a['name']] = files
    return resolved, inputs, sizes, problems


def tracked_files(repo_root):
    r = subprocess.run(['git', '-C', str(repo_root), 'ls-files', '-z'],
                       capture_output=True, text=True, check=True)
    return set(r.stdout.split('\0'))


def check_coverage(spec, resolved, inputs, tracked):
    """Uncovered manifest inputs: (arcname, figures) not in a figures archive
    and not git-tracked."""
    covered = set()
    for a in spec:
        if 'figures' in a['tiers']:
            covered |= set(resolved[a['name']])
    bad = []
    for (kind, rel), figs in sorted(inputs.items()):
        if kind == 'other':
            bad.append((rel, figs))
            continue
        arc = DJA_ARC_PREFIX + rel if kind == 'dja' else rel
        if arc in covered or (kind == 'repo' and rel in tracked):
            continue
        bad.append((arc, figs))
    return bad


def modified_tracked(repo_root, paths):
    """Subset of repo-relative paths whose working-tree content differs from
    HEAD (batched git diff)."""
    bad = set()
    paths = sorted(paths)
    for i in range(0, len(paths), 500):
        r = subprocess.run(['git', '--literal-pathspecs', '-C', str(repo_root),
                            'diff', '--name-only', '-z', 'HEAD', '--',
                            *paths[i:i + 500]],
                           capture_output=True, text=True, check=True)
        bad |= {p for p in r.stdout.split('\0') if p}
    return bad


def check_identity(spec, resolved, inputs, sizes, tracked, repo_root, dja_dir):
    """Problems for covered/git-tracked inputs whose on-disk identity differs
    from what the figure manifests recorded: size mismatch, git-tracked file
    missing or modified vs HEAD. Directories are skipped."""
    covered = set()
    for a in spec:
        if 'figures' in a['tiers']:
            covered |= set(resolved[a['name']])
    problems, git_reliant = [], []
    for (kind, rel) in sorted(inputs):
        if kind == 'other':
            continue
        arc = DJA_ARC_PREFIX + rel if kind == 'dja' else rel
        in_arc = arc in covered
        is_tracked = kind == 'repo' and rel in tracked
        if not (in_arc or is_tracked):
            continue
        src = Path(repo_root if kind == 'repo' else dja_dir) / rel
        if src.is_dir():
            continue
        if not src.is_file():
            if is_tracked and not in_arc:
                problems.append(f'git-tracked input missing on disk: {rel}')
            continue                  # archive sources: reported as missing elsewhere
        st = src.stat().st_size
        for rec in sorted(sizes.get((kind, rel), ())):
            if rec != st:
                problems.append(f'{arc}: on-disk size {st} != size {rec} recorded '
                                'in a figure manifest')
                break
        if is_tracked and not in_arc:
            git_reliant.append(rel)
    if git_reliant:
        problems += [f'git-tracked input modified vs HEAD: {p}'
                     for p in sorted(modified_tracked(repo_root, git_reliant))]
    return problems


def normalize_arcname(arc):
    """Canonical archive member name; rejects absolute, '..' and '//' names."""
    if arc.startswith('/') or '//' in arc or '..' in arc.split('/'):
        raise ValueError(f'unsafe archive member name: {arc!r}')
    norm = posixpath.normpath(arc)
    if norm in ('', '.'):
        raise ValueError(f'unsafe archive member name: {arc!r}')
    return norm


# -------------------------------------------------------------------- building

class HashingWriter:
    """Write-through file wrapper computing sha256/md5 and byte count."""

    def __init__(self, fh):
        self.fh, self.sha, self.md5, self.n = fh, hashlib.sha256(), hashlib.md5(), 0

    def write(self, b):
        self.sha.update(b)
        self.md5.update(b)
        self.n += len(b)
        return self.fh.write(b)

    def flush(self):
        self.fh.flush()


class HashingReader:
    def __init__(self, fh):
        self.fh, self.sha, self.n = fh, hashlib.sha256(), 0

    def read(self, size=-1):
        b = self.fh.read(size)
        self.sha.update(b)
        self.n += len(b)
        return b


def build_archive(path, compression, files):
    """Stream files {arcname: src} into a deterministic tar[.gz] at path.

    Written to <path>.part and renamed on success. Returns (archive record,
    member records).
    """
    norm = {}
    for arc, src in files.items():
        n = normalize_arcname(arc)
        if n in norm:
            raise ValueError(f'duplicate archive member name after normalisation: {n!r}')
        norm[n] = src
    path = Path(path)
    part = path.with_name(path.name + '.part')
    members = []
    try:
        with open(part, 'wb') as raw:
            out = HashingWriter(raw)
            gz = None
            if compression == 'gzip':
                gz = gzip.GzipFile(filename='', mode='wb', fileobj=out,
                                   compresslevel=6, mtime=0)
                sink = gz
            elif compression == 'none':
                sink = out
            else:
                raise ValueError(f'unknown compression {compression!r}')
            with tarfile.open(fileobj=sink, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for arc in sorted(norm):
                    src = norm[arc]
                    st0 = src.stat()
                    info = tarfile.TarInfo(arc)
                    info.size = st0.st_size
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with open(src, 'rb') as fh:
                        rd = HashingReader(fh)
                        tar.addfile(info, rd)
                    st1 = src.stat()
                    if (st1.st_size, st1.st_mtime_ns) != (st0.st_size, st0.st_mtime_ns):
                        raise RuntimeError(f'{src} changed while packing')
                    members.append({'path': arc, 'size': info.size,
                                    'sha256': rd.sha.hexdigest()})
            if gz is not None:
                gz.close()
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return ({'size': out.n, 'sha256': out.sha.hexdigest(),
             'md5': out.md5.hexdigest()}, members)


def git_state(path):
    """(commit, dirty) of the checkout containing path. dirty = tracked files
    modified or staged (untracked files are ignored)."""
    path = str(path)
    try:
        c = subprocess.run(['git', '-C', path, 'rev-parse', 'HEAD'],
                           capture_output=True, text=True, check=True).stdout.strip()
        d = subprocess.run(['git', '-C', path, 'status', '--porcelain',
                            '--untracked-files=no'],
                           capture_output=True, text=True, check=True).stdout.strip()
        return c, bool(d)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def dirty_build_files(paths):
    """Those of `paths` (the files that define the build) that are untracked,
    modified or staged in their own checkout, or whose git state is unknown."""
    bad = []
    for p in map(Path, paths):
        try:
            d = subprocess.run(['git', '-C', str(p.parent), 'status', '--porcelain',
                                '--', p.name],
                               capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            d = 'unknown'
        if d:
            bad.append(str(p))
    return bad


def fmt_size(n):
    return f'{n / 1e9:.3f} GB' if n >= 1e8 else f'{n / 1e6:.2f} MB'


def file_digests(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(CHUNK), b''):
            sha.update(b)
            md5.update(b)
    return sha.hexdigest(), md5.hexdigest()


def run(out_dir, spec_path=SPEC_PATH, repo_root=None, dja_dir=None, names=None,
        dry_run=False, only=None, allow_dirty=False, readme_path=README_PATH):
    repo_root = Path(repo_root or config.REPO_ROOT)
    dja_dir = Path(dja_dir or config.DJA_RESULTS_DIR)
    readme_path = Path(readme_path)
    spec = load_spec(spec_path)
    if only:
        unknown = set(only) - {a['name'] for a in spec}
        if unknown:
            raise SystemExit(f'--only: unknown archive(s) {sorted(unknown)}')

    early = []
    if not readme_path.is_file():
        early.append(f'{README_NAME} not found at {readme_path} (written by hand; '
                     'it ships in the record)')
    sdi_c, sdi_d = git_state(repo_root)
    if sdi_c is None:
        early.append(f'cannot determine the git state of {repo_root}')
    elif sdi_d:
        if allow_dirty:
            print('WARNING: SDI checkout has modified/staged tracked files; '
                  'recording sdi_git_dirty=true (--allow-dirty)')
        else:
            early.append('SDI checkout has modified/staged tracked files; commit '
                         'them or pass --allow-dirty')
    bad = dirty_build_files([spec_path, readme_path, METADATA_PATH,
                             Path(__file__).resolve()])
    if bad and sdi_c is not None:
        if allow_dirty:
            print(f'WARNING: build-defining file(s) untracked or differ from HEAD: '
                  f'{bad}; recording sdi_git_dirty=true (--allow-dirty)')
            sdi_d = True
        else:
            early.append(f'build-defining file(s) untracked or differ from HEAD: '
                         f'{bad}; commit them or pass --allow-dirty')

    resolved, inputs, sizes, problems = resolve_archives(spec, repo_root, dja_dir, names)
    problems = early + problems
    tracked = tracked_files(repo_root)
    uncovered = check_coverage(spec, resolved, inputs, tracked)
    problems += check_identity(spec, resolved, inputs, sizes, tracked,
                               repo_root, dja_dir)

    print(f'{len(inputs)} manifest inputs from paper-figure manifests')
    asizes = {}
    for a in spec:
        files = resolved[a['name']]
        asizes[a['name']] = sum(s.stat().st_size for s in files.values() if s.is_file())
        print(f"  {a['name']:40s} {len(files):7d} files  {fmt_size(asizes[a['name']]):>12s}"
              f"  tiers={','.join(a['tiers'])}")
    if problems:
        print(f'FAIL: {len(problems)} problem(s):')
        for p in problems[:50]:
            print('  ', p)
    if uncovered:
        print(f'FAIL: {len(uncovered)} figure-manifest input(s) not covered by a '
              '"figures" archive or git:')
        for arc, figs in uncovered[:50]:
            print(f"   {arc}  (used by {', '.join(figs[:3])}{' ...' if len(figs) > 3 else ''})")
    if problems or uncovered:
        return 1
    print('coverage OK: every figure-manifest input is in a "figures" archive '
          'or git-tracked')
    if dry_run:
        return 0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / 'MANIFEST.json'
    prev = None
    if only and mpath.exists():
        prev = json.loads(mpath.read_text())
    mpath.unlink(missing_ok=True)       # no stale manifest next to new archives
    current = {a['name'] for a in spec}
    archives = {k: v for k, v in prev['archives'].items() if k in current} if prev else {}
    members = [m for m in prev['members']
               if m['archive'] not in set(only) and m['archive'] in current] if prev else []
    for a in spec:
        if only and a['name'] not in only:
            continue
        print(f"building {a['name']} ...", flush=True)
        rec, mem = build_archive(out_dir / a['name'], a['compression'],
                                 resolved[a['name']])
        rec.update(tiers=a['tiers'], n_members=len(mem),
                   sdi_git_commit=sdi_c, sdi_git_dirty=sdi_d,
                   built_utc=datetime.datetime.now(datetime.timezone.utc)
                   .isoformat(timespec='seconds'))
        archives[a['name']] = rec
        members += [{'archive': a['name'], **m} for m in mem]
    shutil.copyfile(readme_path, out_dir / README_NAME)
    r_sha, r_md5 = file_digests(out_dir / README_NAME)
    dja_c, dja_d = git_state(dja_dir)
    manifest = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'sdi_git_commit': sdi_c, 'sdi_git_dirty': sdi_d,
        'dja_git_commit': dja_c, 'dja_git_dirty': dja_d,
        'path_remap': {'repo': REPO_REMAP, 'repo_root_pattern': REPO_ROOT_PATTERN,
                       'dja_root_pattern': DJA_ROOT_PATTERN,
                       'dja_arc_prefix': DJA_ARC_PREFIX},
        'readme': {'name': README_NAME, 'size': (out_dir / README_NAME).stat().st_size,
                   'sha256': r_sha, 'md5': r_md5},
        'archives': archives,
        'members': members,
    }
    mpath.write_text(json.dumps(manifest, separators=(',', ':')))
    print(f'wrote {mpath}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--dry-run', action='store_true',
                    help='print counts/sizes and run the coverage check; write nothing')
    ap.add_argument('--only', nargs='+', metavar='NAME', help='build only these archives')
    ap.add_argument('--allow-dirty', action='store_true',
                    help='pack despite modified/staged tracked files (records dirty=true)')
    args = ap.parse_args(argv)
    return run(args.out, dry_run=args.dry_run, only=args.only,
               allow_dirty=args.allow_dirty)


if __name__ == '__main__':
    sys.exit(main())
