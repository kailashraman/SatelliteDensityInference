"""Repository paths and the SatGen version registry.

This is the only module in the repository that contains an absolute path. Every
other module derives its paths from here, so a script works regardless of the
directory it is invoked from -- unlike the SatGen_Dwarf tree it was migrated
from, where relative reads assumed cwd == python/.

Two paths point outside the repository. Both are declared upstream dependencies
(see CLAUDE.md, "The reproducibility contract"), and both are overridable by
environment variable so tests can point them at fixtures:

    TREES_DIR        SatGen merger trees and satellite catalogs   $SDI_TREES_DIR
    DJA_RESULTS_DIR  DwarfJeansAnalysis posteriors                $SDI_DJA_RESULTS_DIR

Nothing else may read from outside the repository.
"""

import os
from pathlib import Path

# Resolved from this file, not from cwd, so imports behave the same from
# python/, scripts/, tests/, or a notebook.
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / 'data'

# RESULTS_DIR is overridable so a single-task parity check can be pointed at a
# scratch tree instead of writing where the stage normally writes. Without that,
# verifying a ported stage against the migrated products REPLACES the very
# baseline being verified -- a stage run typically writes several products (one
# per weighting variant, sometimes into more than one tree), so backing up "the
# file I meant to diff" is not enough. This has happened twice during the
# migration. See the review-checklist entry on verification runs overwriting
# their own reference.
#
# SCOPE, and it is narrower than it looks: this redirects products under
# RESULTS_DIR ONLY. compute_weights.py and compute_mhalf.py write under
# ADDITIONAL_DIR (data/additional/{weights_gc,mhalf}), which this does NOT
# move -- and those are the products that cannot be recovered if overwritten,
# since mstar_weights/joint_weights are unseeded single realizations. Before a
# parity rerun of either stage, copy those subtrees by hand; the override will
# not protect them.
#
# Empty and relative values are rejected rather than accepted: `export
# SDI_RESULTS_DIR=` or `sbatch --export=ALL,SDI_RESULTS_DIR=` yields '', and
# Path('') is '.', which would silently root every derived tree at the cwd --
# under a scheduler, whatever directory the task happened to start in.
_results_override = os.environ.get('SDI_RESULTS_DIR')
if _results_override is not None and not _results_override.strip():
    raise ValueError(
        'SDI_RESULTS_DIR is set but empty; unset it to use the default '
        f'{REPO_ROOT / "results"}, or set it to an absolute path')
RESULTS_DIR = (Path(_results_override).expanduser().resolve()
               if _results_override else REPO_ROOT / 'results')
PLOTS_DIR = REPO_ROOT / 'plots'
DOCS_DIR = REPO_ROOT / 'docs'
DRAFTS_DIR = REPO_ROOT / 'drafts_temp'

# Derived-product subdirectories, named as in SatGen_Dwarf so migrated code
# moves with minimal edits.
ADDITIONAL_DIR = DATA_DIR / 'additional'
PAPER_JS_DIR = RESULTS_DIR / 'paper_Js'
PAPER_QUANTILES_DIR = RESULTS_DIR / 'paper_quantiles'
PAPER_CONTOURS_DIR = RESULTS_DIR / 'paper_contours'
PAPER_MASSPROFILE_DIR = RESULTS_DIR / 'paper_massProfile'
FERMI_REWEIGHTING_DIR = RESULTS_DIR / 'fermi_reweighting'
# The 'update' variant reweights against data/fermi_legacy_update/ SEDs
# (Circiello:2026inp, cited throughout the drafts) rather than
# data/fermi_legacy/; see fermi_funcs.py's _LEGACY / _LEGACY_UPDATE split.
# python/fermi_reweighting.py's --legacy-version flag selects between the two
# result trees explicitly -- neither is a silent default.
FERMI_REWEIGHTING_UPDATE_DIR = RESULTS_DIR / 'fermi_reweighting_update'
FERMI_EXPECTED_LIMITS_DIR = RESULTS_DIR / 'fermi_expected_limits'
VCIRC_SCATTER_DIR = RESULTS_DIR / 'vcirc_scatter_300pc'

WEIGHTS_DIR = ADDITIONAL_DIR / 'weights_gc'
MHALF_DIR = ADDITIONAL_DIR / 'mhalf'

# Per-host reductions of the Symphony and Milky-Way-est N-body suites, used by
# multipanel_systematics.pdf and the distance/mass number functions. Upstream
# these were read from ../Symphony/results/, an undeclared third external
# dependency; at ~1.2 MB they belong in the repository instead.
SYMPHONY_DIR = DATA_DIR / 'symphony'

# --------------------------------------------------------------------------
# Upstream dependency 1: SatGen trees and satellite catalogs (dfolsom).
# --------------------------------------------------------------------------
TREES_DIR = Path(os.environ.get(
    'SDI_TREES_DIR', '/global/scratch/projects/pc_heptheory/dfolsom'))

# --------------------------------------------------------------------------
# Upstream dependency 2: DwarfJeansAnalysis production posteriors.
# Layout: <DJA_RESULTS_DIR>/<lvdb_key>/<prior>/{posterior_samples,derived}.npz
# --------------------------------------------------------------------------
DJA_RESULTS_DIR = Path(os.environ.get(
    'SDI_DJA_RESULTS_DIR',
    '/global/scratch/projects/pc_heptheory/kraman/DwarfJeansAnalysis/results/production'))

# Default prior for Dwarf.get_Jeans_results(). The prior ladder in Part II
# selects others explicitly; this is only the fallback.
DJA_PRIOR = 'loguniform'

DJA_PRIORS = (
    'jeffreys', 'loguniform', 'satgen', 'satgen_box',
    'satgen_shmr_danieli23_const', 'satgen_shmr_fattahi18',
    'satgen_shmr_kim24', 'satgen_shmr_moster18',
)

# --------------------------------------------------------------------------
# Observational catalog. Pinned so the dwarf sample cannot change underneath a
# result: SatGen_Dwarf fetched this release over HTTPS at import time, which
# made every script network-dependent and the sample silently mutable.
# --------------------------------------------------------------------------
LVDB_VERSION = 'v1.0.5'
LVDB_DIR = DATA_DIR / 'lvdb'

# LaTeX rendering. The notebook appended a texlive module directory to PATH at
# import; usetex fails without a working latex on PATH.
TEXLIVE_BIN = os.environ.get(
    'SDI_TEXLIVE_BIN',
    '/global/software/sl-7.x86_64/modules/tools/texlive/2016/bin/x86_64-linux')

# --------------------------------------------------------------------------
# SatGen version registry.
#
# `<version>` is an argument, not a tag: it selects the SatGen realization and
# cosmology. Mixing versions corrupts results silently, which is why every
# derived product records the version it was built from (see provenance.py).
#
# Values are filenames under ADDITIONAL_DIR. Entries mapped to None are known
# to the registry but were deliberately not migrated -- h5_path() raises a
# message saying so, rather than failing later with a confusing FileNotFound.
# --------------------------------------------------------------------------
H5_REGISTRY = {
    'Diemer': 'm12res8_10k_Diemer+scatter_sim.h5',
    'Diemer_0p12': 'm12res8_0p12dex_Diemer.h5',
    'Zhao': 'm12res8_z30zhao.h5',
    'm2e12': 'm12res8_2e12_Diemer.h5',
    'mass_floor_7': '26-02-Diemer_mass_floor_7_updated.h5',
    'Symphony': 'SymphonyMilkyWay.h5',
    'MWest': 'MWest.h5',
    # The '_all' catalog, which retains disrupted/splashback halos rather than
    # masking them out. Required by tidal_stripping.pdf: its lower panel is
    # f_disrupted, which is identically ~0 in the Diemer catalog.
    'splashback': 'm12res8_10k_Diemer+scatter_sim_all.h5',
    # Artifacts exist upstream but the code that produced them does not.
    'Geha': None,
}

# Versions that are a reweighting variant of another version's halo catalog:
# the h5 is shared, only the observational inputs or selection differ.
#
# This mirrors the *currently active* branch in compute_weights.py, which reads:
#
#     if version in ('lmc', 'mhalf_scatter', 'lmc_50', 'lvdb', 'mcdaniel'):
#         sim_version = 'Diemer'
#     else:
#         sim_version = version
#
# Note 'Geha' is deliberately absent. A `Geha -> Diemer` branch exists in that
# file and in compute_quantiles.py, but is commented out in both; see
# NOT_MIGRATED below.
SIM_VERSION_ALIASES = {
    'lmc': 'Diemer',
    'lmc_50': 'Diemer',
    'lvdb': 'Diemer',
    'mcdaniel': 'Diemer',
    'mhalf_scatter': 'Diemer',
}

# Known to the registry but not usable here, with the reason reported by
# h5_path(). Naming them beats a bare KeyError: it distinguishes "we chose not
# to migrate this" from "you mistyped a version".
NOT_MIGRATED = {
    'Geha': (
        'unresolved: its Geha->Diemer branch is commented out in both '
        'compute_weights.py and compute_quantiles.py, and the disabled code '
        'also swapped the dwarf list to obs.geha_names, which no longer exists '
        'in Jdata.py. The existing Geha artifacts hold 25 dwarfs, not 43, so '
        'they are keyed to a name list the current code cannot reconstruct. '
        'See docs/migration-notes.md before using them'),
}


def sim_version(version):
    """Halo-catalog version underlying a (possibly reweighting-only) version."""
    return SIM_VERSION_ALIASES.get(version, version)


def h5_path(version, resolve_alias=True):
    """Absolute path to the SatGen halo catalog for `version`.

    Raises KeyError listing the valid keys on an unknown version. The migrated
    `halo_weights.get_h5` had no else-branch, so an unknown version raised
    UnboundLocalError from inside the function instead.

    `resolve_alias=False` refuses a reweighting-only version string instead of
    silently mapping it to the underlying catalog. Callers that pair the
    catalog with a version-keyed directory need that: `ResampleMstar` reads
    `mhalf/<version>/` with the raw string, so resolving here but not there
    would build an object whose halo catalog and mass table disagree.
    """
    key = sim_version(version) if resolve_alias else version
    if key not in H5_REGISTRY:
        raise KeyError(
            f'unknown SatGen version {version!r}; '
            f'known versions: {sorted(H5_REGISTRY)} '
            f'(aliases: {sorted(SIM_VERSION_ALIASES)})')
    filename = H5_REGISTRY[key]
    if filename is None:
        raise KeyError(
            f'SatGen version {key!r} is not usable in this repository: '
            f'{NOT_MIGRATED.get(key, "no reason recorded")}')
    return ADDITIONAL_DIR / filename
