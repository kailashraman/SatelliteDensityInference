"""Provenance stamping for derived products.

Silent version mixing is the worst failure mode in this pipeline: a figure
built from `Diemer` quantiles and `Zhao` weights looks entirely normal. Nothing
in the arrays themselves records which SatGen realization, which observational
catalog, or which DwarfJeansAnalysis snapshot produced them.

Every product this repository writes therefore carries a stamp, in three
layers:

1. In-band -- a `_provenance` key inside the .npz, holding JSON in a 0-d string
   array. Survives np.savez/np.load and is invisible to consumers that index by
   name, so it can be added without touching readers.
2. Sidecar -- `<file>.prov.json` next to the product. Greppable without loading
   a multi-GB array, and the only option for .h5 files and for products copied
   in from SatGen_Dwarf that must not be rewritten (those get `migrated_from`).
3. Figure manifest -- written by plot scripts via `figure_manifest()`, embedding
   the full stamp of every input. This is what answers "which snapshot did this
   figure consume" after the fact.

The guard that actually catches mixing is `assert_single_version()`.
"""

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config

PROVENANCE_KEY = '_provenance'
SIDECAR_SUFFIX = '.prov.json'


def _default_file_mode():
    """The mode a plain, umask-respecting `open(..., 'w')` would have produced.

    `tempfile.mkstemp()` always creates its file at mode 0600 -- by design,
    for security, since the whole point of mkstemp is a private scratch file
    -- and `os.replace()` preserves the SOURCE file's mode across the rename,
    so a temp-file-then-replace write silently drops the umask-derived mode a
    direct `open()` would have gotten. On this repo's shared group tree
    (pc_heptheory) every product under results/ must stay group-readable, so
    every mkstemp-based write here re-applies this mode before or after the
    replace.

    Reading the umask requires setting it (there is no read-only query), so
    this sets 0 and immediately restores the previous value. That is a
    process-wide, non-atomic change and therefore not thread-safe -- fine
    here since every caller is a single-threaded script.
    """
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def _git(*args):
    try:
        return subprocess.check_output(
            ['git', '-C', str(config.REPO_ROOT), *args],
            stderr=subprocess.DEVNULL, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _git_state():
    """Commit and dirty flag for this repository, or None outside a checkout.

    The two queries are independent: a failure of the second must not discard
    a commit the first already established.
    """
    commit = _git('rev-parse', 'HEAD')
    status = _git('status', '--porcelain')
    return (commit.strip() if commit else None,
            bool(status.strip()) if status is not None else None)


def fingerprint(path):
    """Size and mtime of a file, or None if it does not exist.

    Not a hash: the inputs here run to gigabytes, and size+mtime is enough to
    detect the swap-underneath-a-result case this is guarding against.
    """
    path = Path(path)
    if not path.exists():
        return None
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime': int(stat.st_mtime)}


def read(path):
    """Return the provenance dict for a product, or None if unstamped.

    Prefers the sidecar (cheap) and falls back to the in-band key, so this
    works on .h5 and on copied-in products as well as on .npz we wrote.
    """
    path = Path(path)
    sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
    if sidecar.exists():
        with open(sidecar) as handle:
            return json.load(handle)
    if path.suffix == '.npz' and path.exists():
        with np.load(path, allow_pickle=False) as data:
            if PROVENANCE_KEY in data.files:
                return json.loads(str(data[PROVENANCE_KEY]))
    return None


def stamp(script, version=None, argv=None, inputs=(), dja_prior=None,
          dja_files=(), migrated_from=None, **extra):
    """Build a provenance dict.

    Parameters
    ----------
    script : str
        Repository-relative path of the producing script, e.g.
        'python/compute_quantiles.py'.
    version : str, optional
        SatGen version this product was built from -- the field
        `assert_single_version` checks. Raises KeyError if unknown, rather than
        recording a stamp that merely looks complete.
    inputs : iterable of path-like
        Files read to produce this product. Each is fingerprinted and its own
        provenance embedded, so the chain back to the h5 is recoverable from
        the leaf product alone.
    dja_files : iterable of path-like
        The DwarfJeansAnalysis files actually read (e.g. a dwarf's
        posterior_samples.npz). Fingerprinted individually -- a directory mtime
        would not change when an existing chain is overwritten in place, which
        is exactly how a DJA re-run updates it.
    migrated_from : path-like, optional
        Set when stamping a product copied in from SatGen_Dwarf rather than
        recomputed here, so a copied intermediate is never mistaken for one
        this repository produced.
    """
    commit, dirty = _git_state()
    resolved = None
    h5_file = None
    h5_fingerprint = None
    if version is not None:
        resolved = config.sim_version(version)
        try:
            h5 = config.h5_path(version)
        except KeyError:
            # Re-raise for a typo; tolerate a version we deliberately did not
            # migrate, whose products may still legitimately be stamped.
            if resolved not in config.NOT_MIGRATED:
                raise
        else:
            h5_file = h5.name
            h5_fingerprint = fingerprint(h5)

    record = {
        'script': script,
        'argv': list(argv) if argv is not None else None,
        'git_commit': commit,
        'git_dirty': dirty,
        'written_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'satgen_version': version,
        'sim_version': resolved,
        'h5_file': h5_file,
        'h5_fingerprint': h5_fingerprint,
        'lvdb_version': config.LVDB_VERSION,
        'dja_snapshot': (_dja_snapshot(dja_prior, dja_files)
                         if (dja_prior or dja_files) else None),
        'inputs': [_input_record(path) for path in inputs],
        'migrated_from': str(migrated_from) if migrated_from else None,
    }
    record.update(extra)
    return record


def _input_record(path):
    """Fingerprint an input and embed its stamp, one hop deep.

    Only one hop: the aggregation points in Part II combine ~39 dwarfs across
    several priors, and every one of those leaves shares the same h5 ancestor.
    Recursing fully would copy that identical subtree once per leaf.

    One hop names each input's producing script and version, which is all
    `assert_single_version` reads. Walking further back is still possible, but
    by opening the input's own sidecar -- so it depends on the intermediate
    still being on disk. That is the arrangement CLAUDE.md's "cache
    intermediates, not just figures" rule assumes; a pruned `results/` tree
    would truncate the audit trail.
    """
    record = fingerprint(path) or {'path': str(path), 'size': None, 'mtime': None}
    upstream = read(path)
    if upstream is not None:
        n_inputs = len(upstream.get('inputs') or ())
        upstream = {k: v for k, v in upstream.items() if k != 'inputs'}
        upstream['n_inputs'] = n_inputs
    record['provenance'] = upstream
    return record


def _dja_snapshot(prior, files=()):
    return {
        'dir': str(config.DJA_RESULTS_DIR),
        'prior': prior,
        'files': [fingerprint(path) or {'path': str(path), 'size': None,
                                        'mtime': None} for path in files],
    }


def _write_sidecar(path, record):
    """Write `<path>.prov.json`, atomically (temp file + os.replace).

    A plain `open(sidecar, 'w')` can leave a truncated, unparseable sidecar
    behind if the process is killed mid-write; writing to a temp name first
    and replacing means the sidecar at its final path is always either the
    previous complete one or the new complete one, never a partial write.
    """
    path = Path(path)
    sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        suffix=SIDECAR_SUFFIX, prefix='.' + path.name + '.tmp', dir=sidecar.parent)
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, 'w') as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
        os.chmod(tmp_path, _default_file_mode())
        os.replace(tmp_path, sidecar)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return sidecar


def savez(path, record, **arrays):
    """np.savez_compressed with the stamp written both in-band and as a sidecar.

    Refuses a `_provenance` array in `arrays` rather than silently overwriting
    the stamp with it.
    """
    if PROVENANCE_KEY in arrays:
        raise ValueError(f'{PROVENANCE_KEY!r} is reserved for the stamp')
    # np.savez_compressed appends .npz when the path lacks it. Normalise first,
    # so the sidecar cannot end up named for a file that was never written.
    path = Path(path)
    if path.suffix != '.npz':
        path = path.with_name(path.name + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload[PROVENANCE_KEY] = np.array(json.dumps(record, sort_keys=True))

    # savez writes two separate artifacts -- the .npz and its sidecar -- and
    # read() prefers the sidecar over the in-band key, but falls back to the
    # in-band key when the sidecar is absent. Every product this function
    # writes carries an in-band key, so that fallback means "no sidecar" does
    # NOT mean "unstamped": a kill between "the .npz is written" and "the
    # sidecar is written" leaves a fresh .npz whose in-band key read() will
    # happily return. The ordering below is not about producing an unstamped
    # file on interruption -- it can't, given the fallback -- it is about
    # which RUN'S record `path` and read() agree on:
    #   1. the existing sidecar is removed FIRST, before the .npz is touched;
    #   2. the .npz itself is written to a temp name and only os.replace()-d
    #      into place once complete.
    # Without (1), a kill in the window between os.replace() below and the
    # final _write_sidecar() call would leave a FRESH .npz sitting next to
    # the PREVIOUS run's still-valid sidecar, which read() would prefer over
    # the fresh in-band key -- the file and the record it stamps would
    # describe different bytes. With this ordering, the file at `path` and
    # the record read() returns always describe the same bytes: either both
    # are the previous run's (interrupted before os.replace) or both are this
    # run's (interrupted after, no sidecar yet, in-band fallback matches the
    # file that's actually there). A killed rerun with an unchanged record is
    # therefore invisible here -- it leaves a stale-but-self-consistent
    # product, not an unstamped one. Detecting "did the rerun actually
    # happen" needs a run-scoped token (e.g. array_job_id), not this
    # ordering.
    sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
    sidecar.unlink(missing_ok=True)

    tmp_fd, tmp_name = tempfile.mkstemp(
        suffix='.npz', prefix='.' + path.stem + '.tmp', dir=path.parent)
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(tmp_path, **payload)
        os.chmod(tmp_path, _default_file_mode())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    _write_sidecar(path, record)
    return path


def stamp_existing(path, record):
    """Attach a sidecar to a product we cannot or should not rewrite.

    Used for .h5 files and for intermediates copied in from SatGen_Dwarf.

    Refuses when the target `.npz` already carries an in-band `_provenance`
    key (only `savez()` writes that) that disagrees with `record`: `savez`
    writes both layers, so an in-band record means the file was produced
    locally by this repository, not copied in. Writing a sidecar over it --
    e.g. a migration sweep's 'copied from SatGen_Dwarf' claim -- would overlay
    a false story on top of a truthful in-band one, with nothing left to
    compare them afterwards.
    """
    path = Path(path)
    if path.suffix == '.npz' and path.exists():
        with np.load(path, allow_pickle=False) as data:
            if PROVENANCE_KEY in data.files:
                inband = json.loads(str(data[PROVENANCE_KEY]))
                if inband != record:
                    raise ValueError(
                        f'{path} already carries an in-band _provenance '
                        'record that disagrees with the sidecar being '
                        'written -- refusing. This product was produced '
                        'locally (savez writes both layers); stamping a '
                        'different record over it would falsify the '
                        'provenance chain.')
    return _write_sidecar(path, record)


# Stages whose products carry an infall -> z=0 stellar-mass convention, i.e.
# those downstream of ResampleMstar.evolved_Mstar. Anything else legitimately
# has no `mstar_evolution` field and is exempt from that comparison.
#
# A script belongs here ONLY once it actually stamps the field. Listing one
# that does not is worse than omitting it: its products stamp as None, and the
# guard then reports a correct product as predating the fix and carrying the
# +1.2 dex offset -- a false accusation that would send someone regenerating a
# whole tree. tests/test_provenance.py asserts the two stay in step.
#
CONVENTION_BEARING_SCRIPTS = frozenset({
    'python/compute_weights.py',
    'python/compute_quantiles.py',
    'python/vcirc_scatter_300pc.py',
})


def assert_single_version(paths, expected=None):
    """Raise unless every product in `paths` came from one SatGen version.

    This is the guard against the failure mode that is invisible in a figure:
    combining products built from different realizations. Unstamped inputs are
    an error too -- an unstamped file is one whose version is unknown, not one
    known to be consistent.

    `sim_version` is the RESOLVED catalog (aliases like `lmc`/`lmc_50`/`lvdb`/
    `mhalf_scatter` all resolve to `Diemer`), so the primary check above groups
    by that key and cannot by itself distinguish those variants from one
    another. A second pass compares the raw `satgen_version` across every
    input that actually carries an alias (`satgen_version != sim_version`) and
    raises on more than one distinct raw variant, regardless of whether
    `expected` was passed -- a bare call with no `expected` still catches
    `lmc` mixed with `lmc_50`.

    This second pass is asymmetric by construction: a plain, non-aliased file
    (e.g. a `Diemer` product with `satgen_version == sim_version`) dropped
    into an `lmc` slot has `raw == sim_version` for that record and is
    skipped by this check, since it never enters `aliased`. Catching that
    case needs a check keyed on the *directory* the file was found in, not on
    its own stamp, and is out of scope here.
    """
    paths = list(paths)
    if not paths:
        if expected is not None:
            # Usually a glob that matched nothing because of a path typo.
            # Returning success here would be the silent failure this guards.
            raise ValueError(
                f'no inputs to check against expected version {expected!r}')
        return None

    records = [read(path) for path in paths]

    seen = {}
    aliased = {}
    conventions = {}
    missing = []
    unstamped = []
    for path, record in zip(paths, records):
        if record is None or record.get('sim_version') is None:
            if not Path(path).exists():
                missing.append(str(path))
            else:
                unstamped.append(str(path))
            continue
        seen.setdefault(record['sim_version'], []).append(str(path))
        raw = record.get('satgen_version')
        if raw is not None and raw != record.get('sim_version'):
            aliased.setdefault(raw, []).append(str(path))
        # The infall -> z=0 stellar-mass convention. Products written before
        # that mapping was fixed carry no field at all, and are numerically
        # indistinguishable from corrected ones -- which is exactly how a tree
        # holding mass_floor_7 under the old +1.2 dex constant alongside eight
        # versions without it reached panel 6 of multipanel_systematics.pdf.
        # `None` is its own class among these: absence means "unknown,
        # predates the fix", never "agrees".
        #
        # Restricted to products of the stage that applies the mapping. The
        # catalog h5, the rho150/M30 globals and the mhalf products have no
        # stellar-mass convention to disagree about, so folding their absent
        # field into the same comparison would make every quantiles run fail
        # against its own inputs.
        if record.get('script') in CONVENTION_BEARING_SCRIPTS:
            conventions.setdefault(
                record.get('mstar_evolution'), []).append(str(path))

    if missing:
        raise FileNotFoundError(
            'cannot verify SatGen version, these inputs do not exist:\n  '
            + '\n  '.join(missing))
    if unstamped:
        raise ValueError(
            'cannot verify SatGen version, these inputs are unstamped '
            '(present on disk but carry no provenance record):\n  '
            + '\n  '.join(unstamped))
    if len(seen) > 1:
        detail = '\n'.join(
            f'  {version}: {len(files)} file(s), e.g. {files[0]}'
            for version, files in sorted(seen.items()))
        raise ValueError(f'inputs mix SatGen versions:\n{detail}')
    if len(aliased) > 1:
        detail = '\n'.join(
            f'  {raw}: {len(files)} file(s), e.g. {files[0]}'
            for raw, files in sorted(aliased.items()))
        raise ValueError(f'inputs mix aliased SatGen variants:\n{detail}')
    if len(conventions) > 1:
        detail = '\n'.join(
            f'  {convention!r}: {len(files)} file(s), e.g. {files[0]}'
            for convention, files in sorted(
                conventions.items(), key=lambda kv: str(kv[0])))
        raise ValueError(
            'inputs mix infall -> z=0 stellar-mass conventions '
            f'(mstar_evolution):\n{detail}\n'
            'Products stamped None predate the per-halo tidal mapping and '
            'carry a constant +1.2 dex offset. Regenerate the whole tree for '
            'this version rather than mixing.')
    if expected is not None and seen:
        found = next(iter(seen))
        resolved = config.sim_version(expected)
        if found != resolved:
            raise ValueError(
                f'inputs are SatGen version {found!r}, expected {resolved!r}')
        # Additional constraint on top of the alias-mixing check above: every
        # aliased input must also match `expected` itself, not merely agree
        # with each other.
        for path, record in zip(paths, records):
            raw = record.get('satgen_version')
            if raw is not None and raw != record.get('sim_version') and raw != expected:
                raise ValueError(
                    f'{path} was built as SatGen version {raw!r}, '
                    f'expected {expected!r}')
    return next(iter(seen)) if seen else None


def figure_manifest(figure_path, script, inputs, version=None, dja_prior=None,
                    dja_files=(), **extra):
    """Write `<figure>.prov.json` recording everything the figure consumed."""
    record = stamp(script, version=version, inputs=inputs, dja_prior=dja_prior,
                   dja_files=dja_files, figure=str(figure_path), **extra)
    return _write_sidecar(figure_path, record)
