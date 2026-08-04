"""Concatenate per-dwarf vcirc-scatter rows into one CSV per SHMR.

Ported from SatGen_Dwarf/scripts/concat_vcirc_scatter_300pc.sh, moved from
bash into Python for the same reason concat_massProfile.py exists here as
Python rather than bash: the aggregation needed a provenance stamp (the bash
version wrote scatter_300pc.txt -- the file every downstream plot reads --
with none), and a strict two-phase validate-then-write split (see below),
neither of which is natural to express in shell.

For one SatGen version it reads
results/vcirc_scatter_300pc/<version>/<SHMR>/per_dwarf/row_<NNN>.csv for
every SHMR in vcirc_scatter_params.SHMR_NAMES, and writes
results/vcirc_scatter_300pc/<version>/<SHMR>/scatter_300pc.txt: a shared
two-line header followed by every row's data line, in row_NNN (numeric)
order. The COLUMN layout is byte-for-byte unchanged from the upstream/bash
version; the second header comment LINE is not -- upstream and the bash
version here printed a single `seed=${SEED}`, which named only the local
logMstar MC draw and said nothing about the (dominant) unseeded SHMR-scatter
stream, so this version names both RNG streams instead.

REFUSES to write anything if: the SHMR subdirectories under
VCIRC_SCATTER_DIR/<version> do not exactly match SHMR_NAMES (missing and
unexpected entries are reported with different messages -- a leftover
renamed directory is the opposite diagnosis from a missing one); any SHMR's
row indices (parsed from the row_<NNN>.csv filenames) are not exactly
{0, ..., n_dwarfs-1} -- there is no separate bare row-count check, since this
single index-based check subsumes one: an incomplete tree, a stray extra
file, and a duplicate all show up as a non-empty missing/stray/duplicated
set, and unlike a bare count comparison it names which case it is (a count
check alone cannot tell a genuine shortfall from a complete set plus a stray
extra file, and reports the latter as "incomplete", backwards); any row is
missing its sidecar; or a row's dwarf identity (both the sidecar's `dwarf`
field and the dwarf name read from the CSV itself) disagrees with
Jdata.dwarf_names[row_index], or the sidecar disagrees with the row's
expected mstar_evolution, satgen_version, or shmr. All SHMRs are validated
before any SHMR is written -- the bash version validated and wrote each SHMR
inside the same loop, and checked SHMR-set completeness only after all
writes, so a failure partway through could leave some aggregates freshly
written and others still holding a previous run's content, indistinguishable
on disk.

Every refusal path additionally reports, per SHMR directory, whether an
existing scatter_300pc.txt is on disk and when it was last written: a
refusal leaves those files exactly as they were (a previous run's content),
and a consumer reading them afterwards has no other way to know they were not
refreshed by this attempt.

Phase 2 (the writes) is not merely gated behind phase 1's validation -- each
aggregate is first built into a temporary file next to its destination, and
only after every SHMR's temporary file has been written successfully are the
destinations replaced (os.replace, atomic) and stamped. Temp files are
created with tempfile.mkstemp() in the destination's own directory, not a
PID-based name -- a PID is only unique within one host, and two concat runs
for the same version launched on different nodes can collide on it. No
destination file is ever truncated before its full replacement content
exists on disk, so an error while a temp file is BEING BUILT (a full disk, a
kill, preemption) cannot leave a destination partially overwritten -- the
failure mode the two-phase split was introduced to eliminate in the first
place, and which the previous implementation of phase 2 (truncating
`open(out_path, 'w')` per SHMR, in a loop) still reintroduced. This
guarantee is about a single destination never being seen half-written; it
does NOT mean a failure during the commit step (the loop that replaces each
destination and stamps it) cannot produce a mixed fresh/stale tree across
SHMRs -- an error partway through committing can still leave earlier SHMRs
freshly written and later ones holding a previous run's content; that loop
therefore reports, on failure, exactly which SHMRs it committed and which it
left stale, rather than a bare traceback, since the mixed-tree failure mode
itself is not eliminated. Concurrent runs against the same destination (e.g.
two submissions for the same version overlapping) are not locked against
each other and are last-writer-wins.

Run tokens: vcirc_scatter_300pc.py stamps each row with array_job_id/
array_task_id, read from SLURM_ARRAY_JOB_ID/SLURM_ARRAY_TASK_ID (None for a
direct shell invocation, and absent entirely -- not merely None -- for any
row written before this field existed, which as of this script's
introduction is every row currently on disk). Those are two distinct states
and are counted separately in the stamp
(n_rows_missing_array_job_id vs n_rows_null_array_job_id), never
merged by a `.get(..., None)` that cannot tell them apart. The set of
distinct array_job_id values seen per SHMR is recorded in that SHMR's
aggregate stamp as `source_array_job_ids`, but is NEVER a refusal condition:
resubmitting a subset of straggler tasks under a new array job id is a
legitimate, intended workflow -- the tree on disk when this script was
written is exactly that, most rows from one array job and a handful from a
straggler rerun -- and requiring a single token per SHMR would reject a
correct tree. A mixed or entirely untokened SHMR is still surfaced: an INFO
line is printed for each SHMR before it is written, and the counts are
recorded in the stamp.

Usage:
    python concat_vcirc_scatter_300pc.py <version>
e.g. python concat_vcirc_scatter_300pc.py Diemer
"""
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import config
import provenance
import Jdata as obs
from vcirc_scatter_params import N_MC, NARROW_SIGMA, R_TARGET_KPC, SEED, SHMR_NAMES

version = sys.argv[1]

BASE = config.VCIRC_SCATTER_DIR / version

# Sentinel distinct from any real value (including None), so record.get()
# below can tell "field absent entirely" apart from "field present and
# explicitly null" -- collapsing those two states is exactly what item 5's
# bug report is about.
_NO_FIELD = object()

ROW_INDEX_RE = re.compile(r'row_(\d+)\.csv$')


def refuse(message):
    """Print a refusal reason and, per SHMR, whether its existing aggregate
    is stale (present on disk, not refreshed by this run) and when it was
    last written -- otherwise a refusal silently leaves a complete-looking,
    outdated tree with nothing on screen to say so."""
    print(message, file=sys.stderr)
    existing = sorted(BASE.glob('*/scatter_300pc.txt')) if BASE.is_dir() else []
    if existing:
        print('The following existing aggregate(s) were NOT refreshed by '
              'this run (stale, from a previous run):', file=sys.stderr)
        for f in existing:
            mtime = datetime.fromtimestamp(
                f.stat().st_mtime, tz=timezone.utc).isoformat(timespec='seconds')
            print(f'  {f}  (last written {mtime})', file=sys.stderr)
    sys.exit(1)


if not BASE.is_dir():
    refuse(f'{BASE}: does not exist')

expected_shmr = set(SHMR_NAMES)
found_shmr = {p.name for p in BASE.iterdir() if p.is_dir()}
missing_shmr = sorted(expected_shmr - found_shmr)
unexpected_shmr = sorted(found_shmr - expected_shmr)
if missing_shmr:
    refuse(f'{version}: MISSING SHMR directories {missing_shmr} under {BASE}; '
           'refusing to write -- resubmit those SHMRs first')
if unexpected_shmr:
    refuse(f'{version}: UNEXPECTED SHMR directories {unexpected_shmr} under '
           f'{BASE} (not in vcirc_scatter_params.SHMR_NAMES) -- a leftover or '
           'renamed directory?; refusing to write')

n_dwarfs = len(obs.dwarf_names)
assert n_dwarfs > 0, 'Jdata.dwarf_names is empty -- nothing to concatenate'

# Phase 1: validate every SHMR -- row index contiguity/identity (which
# subsumes a bare row count, see the module docstring) and every row's
# sidecar content -- before writing any of them.
row_files_by_shmr = {}
run_tokens_by_shmr = {}
for shmr in SHMR_NAMES:
    per_dwarf = BASE / shmr / 'per_dwarf'
    row_files = sorted(per_dwarf.glob('row_*.csv'))

    # Parse the index out of each filename FIRST, and validate against it
    # directly -- there is no separate bare count check (found N, expected
    # M) ahead of this: that check alone cannot tell a genuine shortfall from
    # a complete set plus a stray extra file (a SURPLUS), and for the latter
    # it reports "incomplete?", backwards. The index-based check below
    # strictly subsumes a count check: any count mismatch necessarily shows
    # up as a non-empty missing/stray/duplicated set.
    row_by_index = {}
    unparsed = []
    for row in row_files:
        m = ROW_INDEX_RE.search(row.name)
        if m is None:
            unparsed.append(row.name)
            continue
        idx = int(m.group(1))
        row_by_index.setdefault(idx, []).append(row)
    if unparsed:
        refuse(f'{version}/{shmr}: {len(unparsed)} file(s) under {per_dwarf} '
               f'do not match row_<NNN>.csv: {sorted(unparsed)}; refusing to write')

    expected_indices = set(range(n_dwarfs))
    found_indices = set(row_by_index)
    duplicate_indices = sorted(i for i, paths in row_by_index.items() if len(paths) > 1)
    missing_indices = sorted(expected_indices - found_indices)
    stray_indices = sorted(found_indices - expected_indices)
    if duplicate_indices or missing_indices or stray_indices:
        dup_detail = ''
        if duplicate_indices:
            dup_files = '; '.join(
                f'{i}: {[str(p) for p in row_by_index[i]]}'
                for i in duplicate_indices)
            dup_detail = f' (duplicate files -- {dup_files})'
        refuse(f'{version}/{shmr}: row indices under {per_dwarf} are not '
               f'exactly 0..{n_dwarfs - 1} -- missing {missing_indices}, '
               f'stray {stray_indices}, duplicated {duplicate_indices}'
               f'{dup_detail}; refusing to write')

    job_ids = set()
    n_missing_field = 0
    n_explicit_null = 0
    for idx in range(n_dwarfs):
        row = row_by_index[idx][0]
        expected_dwarf = obs.dwarf_names[idx]

        record = provenance.read(row)
        if record is None:
            refuse(f'{row}: missing {row.name}{provenance.SIDECAR_SUFFIX} '
                   'sidecar (stale survivor of a partial rerun?); refusing '
                   'to write')
        with open(row) as handle:
            dwarf_name = handle.readline().split(',')[0]

        if dwarf_name != expected_dwarf:
            refuse(f'{row}: row {idx} is for dwarf {dwarf_name!r}, but '
                   f'Jdata.dwarf_names[{idx}] is {expected_dwarf!r}; '
                   'refusing to write')
        if record.get('dwarf') != expected_dwarf:
            refuse(f'{row}: row {idx}, sidecar dwarf={record.get("dwarf")!r}, '
                   f'expected {expected_dwarf!r} (Jdata.dwarf_names[{idx}]); '
                   'refusing to write')
        if record.get('mstar_evolution') != 'per_halo_tidal':
            refuse(f'{row}: mstar_evolution={record.get("mstar_evolution")!r}, '
                   "expected 'per_halo_tidal'; refusing to write")
        if record.get('satgen_version') != version:
            refuse(f'{row}: satgen_version={record.get("satgen_version")!r}, '
                   f'expected {version!r}; refusing to write')
        if record.get('shmr') != shmr:
            refuse(f'{row}: sidecar shmr={record.get("shmr")!r}, but it sits '
                   f'under SHMR directory {shmr!r}; refusing to write')

        job_id = record.get('array_job_id', _NO_FIELD)
        if job_id is _NO_FIELD:
            n_missing_field += 1
        elif job_id is None:
            n_explicit_null += 1
        else:
            job_ids.add(job_id)

    ordered_rows = [row_by_index[idx][0] for idx in range(n_dwarfs)]
    row_files_by_shmr[shmr] = ordered_rows
    run_tokens_by_shmr[shmr] = (job_ids, n_missing_field, n_explicit_null)


def _sorted_job_ids(ids):
    """Numeric sort for job-id strings (all-digit); lexicographic otherwise.

    sorted() on strings is lexicographic, so job ids '2' and '10' would sort
    as ['10', '2'] -- misleading in the printed INFO line and the stamp, even
    though it is cosmetic, not a correctness issue (the values are only ever
    compared for set membership).
    """
    ids = list(ids)
    if ids and all(isinstance(i, str) and i.isdigit() for i in ids):
        return sorted(ids, key=int)
    return sorted(ids)


# Phase 2a: every SHMR passed validation above. Build each aggregate into a
# temporary file next to its destination, and construct its provenance
# record here too -- it depends only on row_files and the counters computed
# in phase 1, both already in hand, so hoisting it out of the commit loop
# below shrinks that loop to rename-class operations only. Nothing under
# BASE is touched yet, so an error anywhere in THIS loop (full disk, kill,
# preemption, Ctrl-C) leaves every existing scatter_300pc.txt exactly as it
# was; caught as BaseException (not Exception) so a KeyboardInterrupt does
# not skip the temp-file cleanup below.
tmp_by_shmr = {}
record_by_shmr = {}
created_tmp_paths = []
try:
    for shmr in SHMR_NAMES:
        row_files = row_files_by_shmr[shmr]
        job_ids, n_missing_field, n_explicit_null = run_tokens_by_shmr[shmr]
        out_path = BASE / shmr / 'scatter_300pc.txt'
        sorted_job_ids = _sorted_job_ids(job_ids)

        # mkstemp() rather than a PID-suffixed name: a PID is only unique
        # within one host, and two concat runs for the same version launched
        # on different nodes could otherwise collide on the same temp path
        # and interleave writes into it.
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=out_path.parent, prefix='.' + out_path.name + '.tmp')
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        created_tmp_paths.append(tmp_path)

        header = (
            f'# vcirc scatter at r={R_TARGET_KPC} kpc  '
            '(sigma_dex = (log10(V84)-log10(V16))/2)\n'
            f'# version={version} shmr={shmr} n_mc={N_MC} narrow_sigma={NARROW_SIGMA} '
            f'local_mc_seed={SEED} global_rng=per-task crc32(version:dwarf_idx:shmr)\n'
            'dwarf_name,logMstar,logMstar_err_lo,logMstar_err_hi,V_median_kms,'
            'V_16_kms,V_84_kms,sigma_dex\n'
        )
        with open(tmp_path, 'w') as out:
            out.write(header)
            for row in row_files:
                with open(row) as handle:
                    out.write(handle.read())
        os.chmod(tmp_path, provenance._default_file_mode())
        tmp_by_shmr[shmr] = tmp_path

        print(f'{version}/{shmr}: {len(sorted_job_ids)} distinct run token(s) seen '
              f'{sorted_job_ids}, {n_missing_field}/{len(row_files)} rows missing '
              f'the array_job_id field entirely, {n_explicit_null}/{len(row_files)} '
              'rows carry an explicit null')

        record_by_shmr[shmr] = provenance.stamp(
            'python/concat_vcirc_scatter_300pc.py',
            version=version, argv=sys.argv[1:], inputs=row_files, shmr=shmr,
            n_rows=len(row_files), source_array_job_ids=sorted_job_ids,
            n_rows_missing_array_job_id=n_missing_field,
            n_rows_null_array_job_id=n_explicit_null)
except BaseException:
    for tmp_path in created_tmp_paths:
        tmp_path.unlink(missing_ok=True)
    raise

# Phase 2b (commit): every aggregate built successfully -- now replace the
# destinations in place (os.replace is atomic; no destination is ever seen
# truncated) and stamp. The existing sidecar is removed immediately before
# its data file is replaced, so a kill between the replace and the stamp
# leaves fresh, unstamped data rather than fresh data paired with the
# PREVIOUS run's sidecar, which would pass every content check downstream.
#
# This loop only does rename-class operations (os.replace, sidecar write) --
# the record was already built in phase 2a above -- but it can still fail
# partway through (a stamp write hitting a full disk, a corrupted row
# sidecar surfacing here via a race, preemption): the destinations already
# replaced in earlier iterations stay committed, the one mid-iteration may be
# replaced with its sidecar unlinked but not yet restamped, and later SHMRs
# are untouched (stale). Report exactly which is which rather than a bare
# traceback, and clean up any temp files that were never consumed, before
# re-raising. This is caught as BaseException (not Exception) for the same
# Ctrl-C reason as phase 2a.
committed = []
in_progress = None
try:
    for shmr in SHMR_NAMES:
        in_progress = shmr
        row_files = row_files_by_shmr[shmr]
        out_path = BASE / shmr / 'scatter_300pc.txt'

        sidecar_path = out_path.with_name(out_path.name + provenance.SIDECAR_SUFFIX)
        sidecar_path.unlink(missing_ok=True)
        os.replace(tmp_by_shmr[shmr], out_path)
        provenance.stamp_existing(out_path, record_by_shmr[shmr])
        committed.append(shmr)
        in_progress = None
        print(f'Wrote {out_path} ({len(row_files)} rows)')
except BaseException:
    # `in_progress`, if not None, is the one SHMR whose fate this can't fully
    # pin down: the exception may have fired before os.replace (out_path
    # still holds whatever it held before this run) or after it but before
    # stamp_existing (out_path already holds the FRESH content, sidecar
    # removed -- unstamped, not stale). Either way its temp file is unused
    # (already consumed by os.replace, or never reached) so unlinking it with
    # missing_ok is safe regardless of which.
    untouched = [s for s in SHMR_NAMES if s not in committed and s != in_progress]
    for s in untouched + ([in_progress] if in_progress is not None else []):
        tmp_by_shmr[s].unlink(missing_ok=True)
    print(f'{version}: commit failed while committing {in_progress!r} -- '
          f'committed (fully replaced and stamped): {committed}; '
          f'{in_progress!r} may now hold fresh-but-unstamped content or its '
          f'previous run\'s content, indistinguishable from here; '
          f'never reached (holding any previous run\'s content, if any): '
          f'{untouched}', file=sys.stderr)
    raise
