#!/usr/bin/env python
"""Stamp intermediates copied in from SatGen_Dwarf.

    python scripts/stamp_migrated.py paper_Js [--verified "Diemer/Antlia II"]
    python scripts/stamp_migrated.py weights_gc
    python scripts/stamp_migrated.py mhalf
    python scripts/stamp_migrated.py globals
    python scripts/stamp_migrated.py paper_quantiles
    python scripts/stamp_migrated.py massProfile
    python scripts/stamp_migrated.py paper_contours

Exists as a tracked script rather than an ad-hoc snippet because the stamps it
writes are the provenance record: a sidecar produced by something not in the
repository is a claim nobody can re-derive.

Two rules it enforces, both learned the hard way:

* The producer recorded is the script that actually produces that product, not
  the one that produces its neighbours. `paper_Js/<version>/<dwarf>/halo_Js.npz`
  comes from Jdwarf.py + concat_Js.py; `paper_Js/halo_position/halo_Js.npz`
  comes from Jhalopos.py, which is NOT migrated -- it evaluates J at each
  halo's own position rather than at a dwarf's, and its values differ.
* `verified` names exactly which products were diffed against upstream. A
  blanket "reproduced bit-exactly" across a bulk copy is an assertion about
  files nobody checked. A prior `verified: true` (and its note) is STICKY: a
  bulk re-stamp with no `--verified` for that product carries it forward
  rather than silently downgrading it -- this script never performs the diff
  itself, so it has no way to re-derive a lost verification. That flag is
  tied to the product's own bytes via `verified_fingerprint`: carrying it
  forward over a product whose bytes have since changed would re-create the
  same false-verification failure one layer up, so a fingerprint mismatch (or
  a missing/empty note) downgrades instead of carrying forward. See
  `_verification_status` and docs/migration-notes.md for the incident that
  made this necessary.

It also validates before it writes, rather than minting a stamp that merely
looks complete: a product is skipped (and reported, not silently dropped) if
its `migrated_from` target does not exist upstream, if its existing sidecar
was written by something other than this script (pass its relative path to
`--restamp`, repeatable, to overwrite that one product deliberately -- there
is no tree-wide form), if its existing record has no `migrated_from` at all
(meaning it was produced locally, not copied in -- refused even with
`--restamp`), if the version implied by its directory name is not one
`config` knows about (a stray directory such as `.ipynb_checkpoints`, not an
error condition), or -- for the per-dwarf trees -- if its basename is not a
name `Jdata.dwarf_names` currently carries (an orphan left behind by a sample
change, the same failure class as the removed `Ursa Major III.npz` pair under
`weights_gc/Diemer` and `mhalf/Diemer`).

`weights_gc`, `mhalf`, and `globals` cover the other three trees copied
straight from SatGen_Dwarf/data/additional without a stamp: the per-dwarf
`weights_gc/<version>/<dwarf>.npz` (compute_weights.py), the per-dwarf
`mhalf/<version>/<dwarf>.npz` (compute_mhalf.py per-dwarf mode), and the four
catalog-global `<version>_{rho30,M30,rho150,Mvir200}.npz` files
(compute_mhalf.py --globals mode). All three are the same failure as the
original paper_Js gap: once a consumer calls `assert_single_version` on them,
an unstamped file is rejected outright, so a tree copied in wholesale needs a
stamp before that guard can be turned on for the scripts that read it.

`paper_quantiles` covers `results/paper_quantiles/galactocentric/<version>/
<dwarf>.npz` (compute_quantiles.py), copied in the same way.

`massProfile` covers `results/paper_massProfile/<version>/massProfile/
massProfile-<g>.npz`, the per-group fragments produced upstream by
massProfile.py. It does NOT cover two neighbours under the same
`paper_massProfile` root that this script must leave alone: `<version>/
massProfile.npz`, the concatenation produced locally by
python/concat_massProfile.py (already carries its own in-band provenance,
never a migrated_from), and `<version>/massProfile_reference.npz`, which this
script has no mandate to touch. The glob is scoped to the `massProfile/`
subdirectory rather than every `*.npz` under the version directory so it
cannot walk into either of those.

`paper_contours` covers `results/paper_contours/`, copied in wholesale like
every other tree above and carrying no stamps of its own. Three subtrees,
each stamped by its own loop because their path shapes and producer stories
differ:

* `galactocentric/z0/<version>/<dwarf>_<suffix>.npz` (V_circ-r_max contours;
  `<suffix>` in mhalf/RP17/F18/M18/K24/joint_RP17/joint_F18/joint_M18/
  joint_K24/unweighted) and `rho150_mpeak/<version>/<dwarf>_<suffix>.npz`
  (rho150-Mpeak contours; same suffixes minus the four joint_* -- that tree
  has no joint-weighted output). Both are produced by
  python/save_contours.py, keyed like the other per-dwarf trees by
  '<group>/<version>/<dwarf>_<suffix>' in --verified. Only `z0` exists in
  this repository's tree (upstream also has `infall`, not migrated here); the
  glob is scoped to `galactocentric/z0/` rather than all of `galactocentric/`
  so an `infall/` subtree, if ever added, is not silently swept in without a
  matching --verified convention having been decided.
* `Jeans_<prior>/<dwarf>_jeans.npz`, keyed by 'Jeans_<prior>/<dwarf>_jeans' in
  --verified. `save_contours.py`'s current `jeans_priors` list -- loguniform,
  satgen, satgen_box, and the four satgen_shmr_<SHMR> variants -- names seven
  directories that ARE this script's Jeans_<prior> branch's output, stamped
  with `produced_by='python/save_contours.py (Jeans_<prior> branch)'` and
  `version=None` (that branch reads no SatGen h5/weights, only a
  DwarfJeansAnalysis derived.npz per dwarf/prior; see save_contours.py's
  docstring). Two more directories exist in both this repository's tree AND
  upstream's -- `Jeans_Pace/` and `Jeans_local/` -- naming priors that are in
  neither `jeans_priors` nor `config.DJA_PRIORS`. No script anywhere in this
  repository or in SatGen_Dwarf's `save_contours.py` produces a prior by
  either name; they predate the current jeans_priors list and their producer
  is genuinely unknown, so they are stamped `produced_by=None` with a note
  saying so explicitly, rather than attributed to save_contours.py. Also
  unlike every other prior, `Jeans_Pace/` additionally holds one bare
  `<dwarf>.npz` per dwarf (an undocumented `paths` array, no `_jeans` suffix)
  that no current jeans_priors branch writes at all -- stamped the same way,
  keyed by 'Jeans_Pace/<dwarf>' in --verified.

None of these three groups gets a `dwarf` name check against
`Jdata.dwarf_names` the way the other per-dwarf trees do: a dwarf dropped
from the current sample would leave an orphaned CONTOUR file with the same
"stray file from a sample change" story as `weights_gc`/`mhalf`, but this
script has no mandate here to decide whether that is in fact what happened
for a name it does not recognise -- reported as `dwarf` in every stamp
instead, so a future check can be added deliberately rather than guessed at
now.

None of these trees carry every version that exists upstream: `Symphony`,
`MWest`, `mcdaniel`, and `Geha` were deliberately not copied into this
repository (superseded catalog / orphaned reweighting -- see
docs/migration-notes.md) and are simply absent from the local tree walked
here, not an error condition this script needs to handle specially.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python'))

import numpy as np

import config
import provenance
import Jdata as obs

UPSTREAM = Path('/global/home/users/kraman/pscratch/SatGen_Dwarf')

# Products whose producer is not python/Jdwarf.py + python/concat_Js.py.
# 'Diemer_backup' (the pre-full_Js Jdwarf.py run) is not listed here: its tree
# was dropped from results/paper_Js entirely (see docs/migration-notes.md,
# "Diemer_backup is stale"), so there is nothing left for this key to match.
SPECIAL_PRODUCERS = {
    'halo_position': (
        'python/Jhalopos.py (NOT MIGRATED): J-factors of SatGen halos at their '
        'own heliocentric positions, in 5000-halo chunks. Not reproducible in '
        'this repository; see docs/migration-notes.md'),
}

# Upstream has no per-version directory level for halo_position -- it is
# unversioned there (results/paper_Js/halo_position/halo_Js.npz), unlike every
# other paper_Js product. Without this, migrated_from would point at a
# 'Diemer/' level that does not exist upstream.
MIGRATED_FROM_OVERRIDES = {
    'halo_position': lambda rel: Path(rel.parts[0], rel.parts[-1]),
}

GLOBAL_QUANTITIES = ('rho30', 'M30', 'rho150', 'Mvir200')

# weights_gc/** and paper_quantiles/** hold mstar_weights/joint_weights (or,
# for paper_quantiles, quantile columns built from them): draws through
# halo_weights.py's unseeded global NumPy RNG (SHMR scatter), taken before
# compute_weights.py seeded it. See docs/review-checklist.md, "Stochastic
# stage treated as a deterministic function of its inputs". That fix makes
# *future* runs of compute_weights.py reproducible; it cannot recover a draw
# made before the seed existed, so these migrated bytes are the only route
# back to the published realization. mhalf_weights (and the mhalf-only
# quantile rows) are a pure function of the catalog and reproduce bit-exact.
STOCHASTIC_ARRAYS = ('mstar_weights', 'joint_weights')
STOCHASTIC_CAVEAT = (
    f'{", ".join(STOCHASTIC_ARRAYS)} are one draw from the unseeded global '
    'NumPy RNG in halo_weights.py (SHMR scatter), not a reproducible function '
    'of the catalog. compute_weights.py now seeds that RNG (see its module '
    'docstring), but a seed cannot recover a draw made before it existed, so '
    'rerunning compute_weights.py will not reproduce these bytes. '
    'mhalf_weights is deterministic and reproduces bit-exact.')

# paper_quantiles/** does not hold mstar_weights/joint_weights directly -- it
# holds *_quantiles arrays of shape (3, n_lists) built BY WEIGHTING WITH them
# (compute_quantiles.py's `weights_list`). The stochasticity is per column,
# not per array: column 0 is mhalf_weights (deterministic); columns 1-10 are
# mstar_weights, one per SHMR (stochastic); columns 11-20 are joint_weights,
# one per SHMR (stochastic); column 21, present only for mhalf_scatter, is
# mdyn_errani_weights (deterministic). Every *_quantiles array in a given
# product shares this column layout. Confirmed against compute_quantiles.py's
# `weights_list` construction and against the actual key sets on disk, which
# are NOT identical across products: theta95_quantiles/Mhalf_quantiles/
# dispersion_quantiles are present only where the upstream run had a
# J-factor file to weight, so a fixed array-name tuple (as for weights_gc
# above) would name arrays some files do not contain -- see
# _stamp_per_dwarf_tree's `stochastic_arrays='from_file'` handling.
QUANTILES_STOCHASTIC_CAVEAT = (
    'Every *_quantiles array here has shape (3, n_lists), columns indexed by '
    'weight list: column 0 is mhalf_weights (deterministic); columns 1-10 are '
    'mstar_weights, one per SHMR (stochastic); columns 11-20 are '
    'joint_weights, one per SHMR (stochastic); column 21, present only for '
    'mhalf_scatter, is mdyn_errani_weights (deterministic). Columns 1-20 come '
    'from one draw of the unseeded global NumPy RNG in halo_weights.py (SHMR '
    'scatter), not a reproducible function of the catalog. '
    'compute_weights.py now seeds that RNG (see its module docstring), but a '
    'seed cannot recover a draw made before it existed, so rerunning '
    'compute_quantiles.py will not reproduce these bytes in columns 1-20. '
    'Column 0 (and column 21, where present) reproduce bit-exact.')

# results/paper_Js/halo_position/<version>/halo_Js.npz: the only genuinely
# stochastic paper_Js product. Comes from python/Jhalopos.py's unseeded vegas
# Monte Carlo integration (see that module's docstring and its own per-chunk
# provenance stamp, `integrator='vegas (stochastic, unseeded)'`); Jdwarf.py
# products (every other paper_Js tree) use deterministic Gauss-Legendre
# quadrature and must NOT carry this caveat. The migrated file holds exactly
# one array, `green_Js` -- confirmed against the on-disk product, not assumed
# from the Jdwarf.py shape (which also writes theta95 and full_Js).
HALO_POSITION_STOCHASTIC_ARRAYS = ('green_Js',)
HALO_POSITION_CAVEAT = (
    'green_Js is one draw of an unseeded vegas Monte Carlo integral '
    '(python/Jhalopos.py, NOT migrated); a rerun differs within Monte Carlo '
    'error. Not a reproducible function of the catalog.')

# paper_contours/{galactocentric/z0,rho150_mpeak}/<version>/<dwarf>_<suffix>.npz
# suffixes, longest-first so '_joint_F18' is matched before the '_F18' it
# also ends with. rho150_mpeak has no joint_* output (see save_contours.py).
GALACTOCENTRIC_SUFFIXES = ('_joint_RP17', '_joint_F18', '_joint_M18', '_joint_K24',
                           '_mhalf', '_RP17', '_F18', '_M18', '_K24', '_unweighted')
RHO150_MPEAK_SUFFIXES = ('_mhalf', '_RP17', '_F18', '_M18', '_K24', '_unweighted')

# Jeans_<prior> directories that ARE save_contours.py's current Jeans_<prior>
# branch output -- exactly its `jeans_priors` list (see that module's
# docstring; 'jeffreys', the eighth member of config.DJA_PRIORS, is
# deliberately absent there too).
JEANS_REPRODUCIBLE_PRIORS = ('loguniform', 'satgen', 'satgen_box',
                             'satgen_shmr_danieli23_const', 'satgen_shmr_fattahi18',
                             'satgen_shmr_kim24', 'satgen_shmr_moster18')
JEANS_PRODUCER = 'python/save_contours.py (Jeans_<prior> branch)'
# 'Pace' and 'local' -- present in both this repo's and upstream's tree, but
# absent from JEANS_REPRODUCIBLE_PRIORS and from config.DJA_PRIORS. No script
# in this repository or in SatGen_Dwarf's save_contours.py is known to
# produce either; recorded as such rather than attributed to save_contours.py.
JEANS_UNKNOWN_PRODUCER_NOTE = (
    "prior {prior!r} is not in save_contours.py's jeans_priors list "
    '(loguniform, satgen, satgen_box, and the four satgen_shmr_<SHMR> '
    'variants) nor in config.DJA_PRIORS -- this tree predates the current '
    'script, its producer is not known, and it is copied in wholesale, not '
    'reproducible by anything currently in this repository')


def _target_missing(migrated_from, product, counts):
    """True (and reported) if `migrated_from` does not exist upstream."""
    if not Path(migrated_from).exists():
        print(f'SKIP {product}: migrated_from target does not exist: {migrated_from}')
        counts['skipped'] += 1
        return True
    return False


def _producer_conflict(product, restamp_paths, counts):
    """True (and reported) if an existing sidecar was written by something
    other than this script and `product` is not in `restamp_paths`.

    A record with no `migrated_from` means the file was produced locally (by
    the script named in `script`, not copied in from upstream) -- refuse to
    overwrite that unconditionally, even if `product` is in `restamp_paths`.
    Stamping a 'copied from SatGen_Dwarf' record over a genuine local run
    would overlay a false provenance story with no way to recover the real
    one; `provenance.stamp_existing` also refuses this when the record is
    in-band, but a locally-produced .h5/globals record may only carry a
    sidecar, so that backstop is not load-bearing here.
    """
    existing = provenance.read(product)
    if existing is None or existing.get('script') == 'scripts/stamp_migrated.py':
        return False
    if existing.get('migrated_from') is None:
        print(f'SKIP {product}: existing sidecar records a locally-produced '
              f'run (script={existing.get("script")!r}, no migrated_from) -- '
              'refusing to overwrite even with --restamp')
        counts['skipped'] += 1
        return True
    if product.resolve() not in restamp_paths:
        try:
            hint = product.resolve().relative_to(config.REPO_ROOT)
        except ValueError:
            hint = product.resolve()
        print(f'SKIP {product}: existing sidecar was written by '
              f'{existing.get("script")!r}, not this script -- pass '
              f'--restamp {hint} to overwrite deliberately')
        counts['skipped'] += 1
        return True
    return False


def _dwarf_not_in_sample(name, product, counts):
    """True (and reported) if `name` is not a current dwarf_names entry."""
    if name not in obs.dwarf_names:
        print(f'SKIP {product}: {name!r} is not in Jdata.dwarf_names -- orphan?')
        counts['skipped'] += 1
        return True
    return False


def _unknown_version(version, product, counts):
    """True (and reported) if `version` (taken from a path component) is not
    a version `config` actually knows -- guards against an `rglob` picking up
    a stray directory (e.g. `.ipynb_checkpoints`, which exists under
    paper_quantiles/galactocentric/) whose name reaches `config.h5_path` and
    aborts the whole sweep with an unrelated-looking KeyError."""
    if version not in config.H5_REGISTRY and version not in config.SIM_VERSION_ALIASES:
        print(f'SKIP {product}: {version!r} is not a known SatGen version or '
              'alias -- stray directory?')
        counts['skipped'] += 1
        return True
    return False


def _verification_status(product, is_verified):
    """(is_verified, note, verified_fingerprint) for the record about to be
    written for `product`.

    `is_verified` in: True only if `product`'s '<version>/<key>' was passed to
    this run's `--verified`.

    A prior `verified: true` sidecar is carried forward -- flag AND note --
    unless this run explicitly re-verifies the same product (`is_verified`
    True). Losing a `verified` flag on a bulk re-stamp is worse than losing a
    producer string: it is the only evidence anyone diffed the bytes against
    upstream, and this stamper cannot reconstruct it by rerunning -- it never
    performs the diff itself, only records that one happened. This was
    discovered the hard way: an early bulk re-stamp downgraded the one
    genuinely verified paper_Js product (`Diemer/Antlia II`) to
    'not individually verified' with no diagnostic. See
    docs/migration-notes.md.

    Carrying the flag forward is not enough by itself: nothing tied it to the
    product's own bytes, so a verified flag could survive the file underneath
    it being replaced -- the same false-verification failure one layer up.
    Every `verified: true` record therefore also carries `verified_fingerprint`
    (`provenance.fingerprint(product)` at the moment `verified: true` is
    written). On carry-forward, a stored fingerprint that disagrees with the
    product's current one means the bytes changed since verification and the
    flag does NOT survive. A record with no stored fingerprint at all (written
    before this check existed) has no baseline to compare against -- it is
    carried forward once more and stamped with a fingerprint now, so future
    re-stamps are protected; this is reported, not silent.

    A prior `verified: true` with a missing or empty `note` is treated as
    malformed -- a verified record naming no statement of what was checked is
    not evidence of anything -- and is not carried forward either.
    """
    if is_verified:
        return (True,
                'copied from SatGen_Dwarf; diffed against upstream: all arrays bit-exact',
                provenance.fingerprint(product))
    existing = provenance.read(product)
    if existing is not None and existing.get('verified'):
        note = existing.get('note')
        if not note:
            print(f'{product}: prior verified:true record has no note -- '
                  'treating as malformed, not carrying forward')
            return False, 'copied from SatGen_Dwarf; not individually verified', None
        stored_fp = existing.get('verified_fingerprint')
        current_fp = provenance.fingerprint(product)
        if stored_fp is not None and stored_fp != current_fp:
            print(f'{product}: prior verified:true fingerprint does not match '
                  f'the product\'s current bytes (stored={stored_fp}, '
                  f'current={current_fp}) -- downgrading, not carrying forward')
            return False, 'copied from SatGen_Dwarf; not individually verified', None
        if stored_fp is None:
            print(f'{product}: prior verified:true record has no fingerprint '
                  'on file -- carrying forward and recording one now')
        return True, note, current_fp
    return False, 'copied from SatGen_Dwarf; not individually verified', None


def _stamp_paper_Js(verified, counts, restamp_paths):
    root = config.PAPER_JS_DIR
    for product in sorted(root.rglob('halo_Js.npz')):
        rel = product.relative_to(root)
        top = rel.parts[0]
        dwarf_key = str(Path(*rel.parts[:-1]))

        is_halo_position = top in SPECIAL_PRODUCERS
        if not is_halo_position and _unknown_version(top, product, counts):
            continue
        version = 'Diemer' if is_halo_position else top
        producer = SPECIAL_PRODUCERS.get(
            top, 'python/Jdwarf.py + python/concat_Js.py')

        upstream_rel = MIGRATED_FROM_OVERRIDES.get(top, lambda r: r)(rel)
        migrated_from = str(UPSTREAM / 'results' / 'paper_Js' / upstream_rel)
        if _target_missing(migrated_from, product, counts):
            continue
        if _producer_conflict(product, restamp_paths, counts):
            continue

        is_verified, note, verified_fp = _verification_status(
            product, dwarf_key in verified)

        extra = {}
        if is_halo_position:
            extra['stochastic_arrays'] = HALO_POSITION_STOCHASTIC_ARRAYS
            extra['stochastic_caveat'] = HALO_POSITION_CAVEAT

        record = provenance.stamp(
            'scripts/stamp_migrated.py', version=version,
            migrated_from=migrated_from,
            produced_by=producer, verified=is_verified, note=note,
            verified_fingerprint=verified_fp, **extra)
        provenance.stamp_existing(product, record)
        counts['stamped'] += 1
        counts['verified'] += int(is_verified)


def _stamp_per_dwarf_tree(root, upstream_root, producer, verified, counts,
                          restamp_paths, check_dwarf_names=False,
                          stochastic_caveat=None, stochastic_arrays=None):
    """Stamp `<root>/<version>/<dwarf>.npz`, keyed like paper_Js by
    '<version>/<dwarf>' in --verified.

    `stochastic_caveat`, if given, is attached to every stamp in this tree.
    `stochastic_arrays` is either a fixed tuple of array names (weights_gc,
    where every product carries the same keys), or the sentinel
    'from_file', which reads each product's actual `*_quantiles` keys before
    stamping it -- paper_quantiles products do not all carry the same key set
    (theta95_quantiles/Mhalf_quantiles/dispersion_quantiles are present only
    where the upstream run had a J-factor file), so a fixed tuple there would
    name arrays some files do not contain.
    """
    for product in sorted(root.rglob('*.npz')):
        rel = product.relative_to(root)
        version = rel.parts[0]
        if _unknown_version(version, product, counts):
            continue
        dwarf_key = str(Path(*rel.parts))[:-len('.npz')]

        migrated_from = str(upstream_root / rel)
        if _target_missing(migrated_from, product, counts):
            continue
        if _producer_conflict(product, restamp_paths, counts):
            continue
        if check_dwarf_names and _dwarf_not_in_sample(product.stem, product, counts):
            continue

        is_verified, note, verified_fp = _verification_status(
            product, dwarf_key in verified)

        extra = {}
        if stochastic_caveat is not None:
            if stochastic_arrays == 'from_file':
                with np.load(product, allow_pickle=False) as data:
                    arrays = tuple(k for k in data.files if k.endswith('_quantiles'))
            else:
                arrays = stochastic_arrays
            extra['stochastic_arrays'] = arrays
            extra['stochastic_caveat'] = stochastic_caveat

        record = provenance.stamp(
            'scripts/stamp_migrated.py', version=version,
            migrated_from=migrated_from,
            produced_by=producer, verified=is_verified, note=note,
            verified_fingerprint=verified_fp, **extra)
        provenance.stamp_existing(product, record)
        counts['stamped'] += 1
        counts['verified'] += int(is_verified)


def _stamp_globals(verified, counts, restamp_paths):
    """Stamp `<version>_{rho30,M30,rho150,Mvir200}.npz` directly under
    ADDITIONAL_DIR, keyed like the per-dwarf trees by '<version>/<quantity>'
    in --verified."""
    producer = 'python/compute_mhalf.py (--globals mode)'
    for quantity in GLOBAL_QUANTITIES:
        for product in sorted(config.ADDITIONAL_DIR.glob(f'*_{quantity}.npz')):
            version = product.stem[:-len(f'_{quantity}')]
            if _unknown_version(version, product, counts):
                continue
            key = f'{version}/{quantity}'

            migrated_from = str(UPSTREAM / 'data' / 'additional' / product.name)
            if _target_missing(migrated_from, product, counts):
                continue
            if _producer_conflict(product, restamp_paths, counts):
                continue

            is_verified, note, verified_fp = _verification_status(
                product, key in verified)

            record = provenance.stamp(
                'scripts/stamp_migrated.py', version=version,
                migrated_from=migrated_from,
                produced_by=producer, verified=is_verified, note=note,
                verified_fingerprint=verified_fp)
            provenance.stamp_existing(product, record)
            counts['stamped'] += 1
            counts['verified'] += int(is_verified)


def _stamp_massProfile(verified, counts, restamp_paths):
    """Stamp `<version>/massProfile/massProfile-<g>.npz`, keyed like the
    per-dwarf trees by '<version>/massProfile-<g>' in --verified.

    Globs `*/massProfile/massProfile-*.npz` rather than `root.rglob('*.npz')`
    (as _stamp_per_dwarf_tree does): `paper_massProfile/<version>/` also
    holds `massProfile.npz` (concat_massProfile.py, produced locally) and
    `massProfile_reference.npz`, neither of which this tree covers.
    """
    root = config.PAPER_MASSPROFILE_DIR
    producer = 'python/massProfile.py'
    for product in sorted(root.glob('*/massProfile/massProfile-*.npz')):
        rel = product.relative_to(root)
        version = rel.parts[0]
        if _unknown_version(version, product, counts):
            continue
        key = f'{version}/{product.stem}'

        migrated_from = str(UPSTREAM / 'results' / 'paper_massProfile' / rel)
        if _target_missing(migrated_from, product, counts):
            continue
        if _producer_conflict(product, restamp_paths, counts):
            continue

        is_verified, note, verified_fp = _verification_status(
            product, key in verified)

        record = provenance.stamp(
            'scripts/stamp_migrated.py', version=version,
            migrated_from=migrated_from,
            produced_by=producer, verified=is_verified, note=note,
            verified_fingerprint=verified_fp)
        provenance.stamp_existing(product, record)
        counts['stamped'] += 1
        counts['verified'] += int(is_verified)


def _strip_contour_suffix(stem, suffixes):
    """(dwarf_name, suffix) for a `<dwarf>_<suffix>.npz` contour stem, or
    (None, None) if `stem` does not end in any suffix in `suffixes` (which
    must be ordered longest-first, so e.g. '_joint_F18' is tried before
    '_F18', which it would also match)."""
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[:-len(suffix)], suffix
    return None, None


def _stamp_paper_contours(verified, counts, restamp_paths):
    """Stamp `results/paper_contours/`: galactocentric/z0, rho150_mpeak, and
    Jeans_<prior>. See this script's module docstring for the shape of each
    and which Jeans_<prior> directories have a known producer.
    """
    root = config.PAPER_CONTOURS_DIR

    # -- galactocentric/z0/<version> and rho150_mpeak/<version> -----------
    groups = (
        ('galactocentric', root / 'galactocentric' / 'z0', GALACTOCENTRIC_SUFFIXES,
         lambda version, rel: UPSTREAM / 'results' / 'paper_contours'
         / 'galactocentric' / 'z0' / rel,
         {'redshift': 'z0'}),
        ('rho150_mpeak', root / 'rho150_mpeak', RHO150_MPEAK_SUFFIXES,
         lambda version, rel: UPSTREAM / 'results' / 'paper_contours'
         / 'rho150_mpeak' / rel,
         {}),
    )
    for group_name, group_root, suffixes, migrated_from_fn, extra_fields in groups:
        for product in sorted(group_root.rglob('*.npz')):
            rel = product.relative_to(group_root)
            version = rel.parts[0]
            if _unknown_version(version, product, counts):
                continue
            dwarf, suffix = _strip_contour_suffix(product.stem, suffixes)
            if dwarf is None:
                print(f'SKIP {product}: stem does not end in a known '
                      f'{group_name} suffix -- stray file?')
                counts['skipped'] += 1
                continue
            key = f'{group_name}/{version}/{product.stem}'

            migrated_from = str(migrated_from_fn(version, rel))
            if _target_missing(migrated_from, product, counts):
                continue
            if _producer_conflict(product, restamp_paths, counts):
                continue

            is_verified, note, verified_fp = _verification_status(
                product, key in verified)

            record = provenance.stamp(
                'scripts/stamp_migrated.py', version=version,
                migrated_from=migrated_from,
                produced_by='python/save_contours.py', verified=is_verified,
                note=note, verified_fingerprint=verified_fp,
                dwarf=dwarf, **extra_fields)
            provenance.stamp_existing(product, record)
            counts['stamped'] += 1
            counts['verified'] += int(is_verified)

    # -- Jeans_<prior> ------------------------------------------------------
    for prior_dir in sorted(p for p in root.glob('Jeans_*') if p.is_dir()):
        prior = prior_dir.name[len('Jeans_'):]
        reproducible = prior in JEANS_REPRODUCIBLE_PRIORS
        for product in sorted(prior_dir.glob('*.npz')):
            if product.stem.endswith('_jeans'):
                dwarf = product.stem[:-len('_jeans')]
            else:
                # Only Jeans_Pace holds these: a bare '<dwarf>.npz' with no
                # analog in save_contours.py's Jeans_<prior> branch (which
                # only ever writes '<dwarf>_jeans.npz').
                dwarf = product.stem
            key = f'{prior_dir.name}/{product.stem}'

            migrated_from = str(UPSTREAM / 'results' / 'paper_contours'
                               / prior_dir.name / product.name)
            if _target_missing(migrated_from, product, counts):
                continue
            if _producer_conflict(product, restamp_paths, counts):
                continue

            is_verified, note, verified_fp = _verification_status(
                product, key in verified)

            produced_by = JEANS_PRODUCER if reproducible else None
            if not reproducible:
                note = note + '; ' + JEANS_UNKNOWN_PRODUCER_NOTE.format(prior=prior)

            record = provenance.stamp(
                'scripts/stamp_migrated.py', version=None,
                migrated_from=migrated_from,
                produced_by=produced_by,
                dja_prior=prior if reproducible else None,
                verified=is_verified,
                note=note,
                verified_fingerprint=verified_fp, dwarf=dwarf)
            provenance.stamp_existing(product, record)
            counts['stamped'] += 1
            counts['verified'] += int(is_verified)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('tree', choices=['paper_Js', 'weights_gc', 'mhalf', 'globals',
                                         'paper_quantiles', 'massProfile',
                                         'paper_contours'])
    parser.add_argument('--verified', action='append', default=[],
                        help='relative path of a product diffed against '
                             'upstream, e.g. "Diemer/Antlia II"')
    parser.add_argument('--restamp', action='append', default=[], metavar='PATH',
                        help='relative path (from the repo root) of ONE '
                             'product whose existing sidecar -- written by '
                             'something other than this script -- should be '
                             'overwritten deliberately, e.g. '
                             '"data/additional/weights_gc/Diemer/Sculptor.npz". '
                             'Repeatable. There is no tree-wide form: a '
                             'stamper minting records from directory layout '
                             'must not blanket-overwrite sidecars it did not '
                             'validate individually. Refused regardless for a '
                             'product whose existing record has no '
                             'migrated_from (produced locally, not copied in).')
    args = parser.parse_args(argv)

    verified = set(args.verified)
    restamp_paths = {(config.REPO_ROOT / p).resolve() for p in args.restamp}
    counts = {'stamped': 0, 'verified': 0, 'skipped': 0}

    if args.tree == 'paper_Js':
        _stamp_paper_Js(verified, counts, restamp_paths)
    elif args.tree == 'weights_gc':
        _stamp_per_dwarf_tree(
            config.WEIGHTS_DIR, UPSTREAM / 'data' / 'additional' / 'weights_gc',
            'python/compute_weights.py', verified, counts, restamp_paths,
            check_dwarf_names=True, stochastic_caveat=STOCHASTIC_CAVEAT,
            stochastic_arrays=STOCHASTIC_ARRAYS)
    elif args.tree == 'mhalf':
        _stamp_per_dwarf_tree(
            config.MHALF_DIR, UPSTREAM / 'data' / 'additional' / 'mhalf',
            'python/compute_mhalf.py (per-dwarf mode: logMhalf, logMdyn_errani)',
            verified, counts, restamp_paths, check_dwarf_names=True)
    elif args.tree == 'globals':
        _stamp_globals(verified, counts, restamp_paths)
    elif args.tree == 'paper_quantiles':
        _stamp_per_dwarf_tree(
            config.PAPER_QUANTILES_DIR / 'galactocentric',
            UPSTREAM / 'results' / 'paper_quantiles' / 'galactocentric',
            'python/compute_quantiles.py', verified, counts, restamp_paths,
            check_dwarf_names=True, stochastic_caveat=QUANTILES_STOCHASTIC_CAVEAT,
            stochastic_arrays='from_file')
    elif args.tree == 'massProfile':
        _stamp_massProfile(verified, counts, restamp_paths)
    elif args.tree == 'paper_contours':
        _stamp_paper_contours(verified, counts, restamp_paths)

    print(f"stamped {counts['stamped']} products "
          f"({counts['verified']} verified against upstream, "
          f"{counts['skipped']} skipped)")
    if counts['verified'] == 0:
        print('WARNING: nothing was verified; every stamp says so')
    return 0


if __name__ == '__main__':
    sys.exit(main())
