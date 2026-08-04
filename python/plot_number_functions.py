"""Recreates the "number function" plots from
../SatGen_Dwarf/jupyter/PaperPlots.ipynb sections "J-factor Statistics"
(cells 73-79), "Distance number function" (cells 81-91) and "$c_V$,
$\\rho_{150}$, $J$, $M_{\\rm peak}$ functions" (cells 93-100).

Migrated from SatGen_Dwarf/python/plot_number_functions.py. Six figures (the
`plot_lmc_vmax_hist`/cell-91 histogram was dropped from scope -- it did not
share the render() skeleton with the rest and nothing else in scope consumes
its inputs):

    sims_mpeak              -> plots/stats_sims/sims_mpeak.pdf
    sims_rho150              -> plots/stats_sims/sims_rho150.pdf
    inference_rho150          -> plots/stats_inference/rho150.pdf
    jfactors_all              -> plots/number_functions/Jfactors_all.pdf
    distance_statistics       -> plots/number_functions/distance_statistics.pdf
    distance_statistics_sims  -> plots/number_functions/distance_statistics_sims.pdf

The stats_sims/stats_inference subdirectory names are upstream's own
out_dir + 'stats_sims/' / 'stats_inference/' writes; jfactors_all and the two
distance_statistics figures were written flat into upstream's out_dir with no
subfolder, so they are grouped here under a new 'number_functions/' family
(organisational only -- the basename, not the subdirectory, is the contract
tests/test_figure_names.py checks).

Path/import/provenance edits only for five of six figures -- the quantile
computations, masks, binning, seed handling and series definitions are
character-identical to upstream. `jfactors_all` is the exception: upstream's
own `python/plot_number_functions.py` diverges from the notebook cell that
actually produced the published figure (cell 80 -- upstream adds a Kim+24
curve absent from the caption, and collapses the two Jeans-prior curves
cells 79/80 draw (loguniform and satgen_shmr_fattahi18) into one, dropping
the SatGen-informed-prior curve entirely), so upstream is the wrong
reference for that figure's series set; it is built to match notebook cells
79/80 directly instead (see `build_jfactors_all`). Dropped: the
matplotlib/rcParams preamble (now `from plot_style import *`, see
plot_style.py's docstring for cell 1/2 ordering) and the `os.environ["PATH"]
+=` texlive mutation it duplicated (plot_style already does this via
config.TEXLIVE_BIN).

Two upstream absolute paths become config-derived:
    symphony_base_dir / mwest_base_dir -> config.SYMPHONY_DIR / 'SymphonyMilkyWay' / 'MWest'
        (already migrated per-host reductions; config.SYMPHONY_DIR existed)
    out_dir                             -> config.PLOTS_DIR / <family>
config.WEIGHTS_DIR, config.ADDITIONAL_DIR and config.PAPER_JS_DIR already
existed (used by save_contours.py / compute_weights.py / Jhalopos.py
respectively); no new path constant was added for any of the above.

Two upstream reads are corrected here, per docs/migration-notes.md's "Fixed"
section (NOT a numerics change to this script's own logic -- they read a
different, already-established path for the same, unmodified J-factor
values):
  - the SatGen J-factor curve reads paper_Js/halo_position/<version>/halo_Js.npz
    (this repository's Jhalopos.py writes a version subdirectory; upstream's
    on-disk artifact and this script both used the unversioned
    paper_Js/halo_position/halo_Js.npz, which only ever existed for Diemer)
  - the per-dwarf inference curves in build_jfactors_all read
    paper_Js/Diemer/<dwarf>/halo_Js.npz, not the stale paper_Js/Diemer_backup/
    copied from upstream (verified bit-identical on both arrays it holds,
    green_Js and theta95 -- migration-notes.md, "Diemer_backup is stale").

`large_dwarf` and `problem_ultrafaints` are imported from dwarf_categories
rather than re-declared -- they are the same literal lists upstream defined
locally in this module.

Every figure is stamped by provenance.figure_manifest, next to the PDF it
describes, embedding the inputs actually read for that figure (the SatGen h5,
rho150/weights/halo_Js products, and the DJA derived.npz files jfactors_all's
Jeans curve reads).

CRITICAL -- ported verbatim, not fixed (see module docstrings inline at each
site below and docs/migration-notes.md):
  - build_sims applies the Mpeak>1.2e8 floor to the SatGen curve that
    notebook cell 93 left commented out (a deliberate scope decision, not a
    path edit -- see the comment in build_sims).
  - build_inference_rho150's ylim (8e-1, 9e1) is the notebook's commented-out
    alternative to the active (8e-1, 8e2), chosen because it matches this
    figure's counts.
  - build_jfactors_all has a known Tucana V column-fill hazard: the Jeans
    curve fills one fewer dwarf column than the other three curves. Left
    exactly as upstream.
  - build_distance_statistics_sims's explicit d<259 kpc cut is a documented
    numerical no-op against the Diemer catalog's max d_GC (258.92 kpc).
  - The two sims figures (build_sims) and distance_statistics_sims mask
    Symphony/MWest differently from each other (mass-loss cut present in one,
    absent in the other) -- a property of the published figures, reproduced
    as-is; see migration-notes.md, "The two sims panels mask Symphony/MWest
    differently".

Usage:
    conda activate J_calc
    python plot_number_functions.py [figure=all] [version=Diemer] [seed=0]
"""
import sys
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.lines as mlines
import astropy.units as u

from plot_style import *  # noqa: F401,F403 -- brings plt, mpl, colors, config
import halo_weights
import Jdata as obs
import provenance
from dwarf_categories import large_dwarf, problem_ultrafaints

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
FIGURES = ['sims_mpeak', 'sims_rho150', 'inference_rho150', 'jfactors_all',
           'distance_statistics', 'distance_statistics_sims']

# Per-figure RNG keys (cell 75/98's Monte Carlo draws) -- stable small ints,
# NOT python hash(), so np.random.default_rng([seed, key]) reproduces the
# same figure whether it is rendered alone or as part of `all`.
FIGURE_RNG_KEY = {'inference_rho150': 1, 'jfactors_all': 2}

figure = sys.argv[1] if len(sys.argv) > 1 else 'all'
version = sys.argv[2] if len(sys.argv) > 2 else 'Diemer'
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
if figure not in FIGURES + ['all']:
    raise SystemExit(f"figure must be one of {FIGURES + ['all']}, got {figure!r}")

# family -> basename, so each build_* function only needs its own key.
FAMILY = {
    'sims_mpeak': 'stats_sims', 'sims_rho150': 'stats_sims',
    'inference_rho150': 'stats_inference',
    'jfactors_all': 'number_functions', 'distance_statistics': 'number_functions',
    'distance_statistics_sims': 'number_functions',
}
BASENAME = {
    'sims_mpeak': 'sims_mpeak.pdf', 'sims_rho150': 'sims_rho150.pdf',
    'inference_rho150': 'rho150.pdf', 'jfactors_all': 'Jfactors_all.pdf',
    'distance_statistics': 'distance_statistics.pdf',
    'distance_statistics_sims': 'distance_statistics_sims.pdf',
}


def out_path(key):
    path = config.PLOTS_DIR / FAMILY[key] / BASENAME[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


jeans_color = '#4B0082'  # indigo
extra_colors = ["#882255", "#44AA99", "#DDCC77"]

dwarf_names = obs.dwarf_names

# Symphony/MWest per-host simulation output (NOT the SatGen-ingested h5 --
# these directories carry rho150/cV/mpeak/mvir/vmax/x directly per host).
symphony_base_dir = str(config.SYMPHONY_DIR / 'SymphonyMilkyWay') + '/'
mwest_base_dir = str(config.SYMPHONY_DIR / 'MWest') + '/'


# ---------------------------------------------------------------------------
# Core quantile helpers (cells 73/81/93)
# ---------------------------------------------------------------------------
def N_greater(vals, grid):
    return np.array([np.count_nonzero(vals > x) for x in grid])


def N_lesser(vals, grid):
    return np.array([np.count_nonzero(vals < x) for x in grid])


def quantile(values, weights=None, quantiles=0.5):
    # https://stackoverflow.com/a/73905572
    m = ~np.isnan(values)
    if not np.any(m):
        return np.full(np.shape(quantiles), np.nan)
    v = values[m]
    if weights is None:
        weights = np.ones_like(values)
    wgt = weights[m]
    i = np.argsort(v)
    c = np.cumsum(wgt[i])
    q = np.clip(np.searchsorted(c, quantiles * c[-1]), 0, len(c) - 2)
    return np.where(c[q] / c[-1] == quantiles, 0.5 * (v[i[q]] + v[i[q + 1]]), v[i[q]])


def quantile_sorted(values, sorted_idcs, weights=None, quantiles=0.5):
    # Same as quantile(), but takes a precomputed argsort so it can be called
    # once per dwarf per MC sample without re-sorting a 2.4M-row array each
    # time (cell 98).
    m = ~np.isnan(values)
    if not np.any(m):
        return np.full(np.shape(quantiles), np.nan)
    v = values[m]
    if weights is None:
        weights = np.ones_like(values)
    wgt = weights[m]
    i = sorted_idcs
    c = np.cumsum(wgt[i])
    q = np.clip(np.searchsorted(c, quantiles * c[-1]), 0, len(c) - 2)
    return np.where(c[q] / c[-1] == quantiles, 0.5 * (v[i[q]] + v[i[q + 1]]), v[i[q]])


# ---------------------------------------------------------------------------
# Architecture functions
# ---------------------------------------------------------------------------
def number_function_quantiles(values, group_ids, grid, comparator='greater'):
    """
    Per-host cumulative counts over `group_ids` run-length groups (assumes
    group_ids is contiguous per host, as in the SatGen h5 layout -- masking
    preserves this since it never reorders rows), then the 5 quantiles
    [0.025, 0.16, 0.5, 0.84, 0.975] across hosts.
    comparator: 'greater' -> N(>X) [cells 73, 82, 84, 93, 95]
                'less'    -> N(<X) [cells 81, 82, 84, 85, 86]
    """
    cmp_fn = N_greater if comparator == 'greater' else N_lesser
    group_ids = np.asarray(group_ids)
    edges = np.concatenate([[-1]] + list(np.where(np.diff(group_ids))) + [[len(group_ids) - 1]]) + 1
    N_per_MW = np.array([cmp_fn(values[lo:hi], grid) for lo, hi in zip(edges[:-1], edges[1:])])
    qs = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
    return np.array([quantile(N_per_MW[:, i], quantiles=qs) for i in range(len(grid))])


def sim_dir_quantiles(base_dir, key, grid):
    """
    Symphony/MWest: per-npz-file mask (mvir/mpeak within 2 dex, distance <
    259 kpc, mpeak > 1.2e8) + LMC-index (most massive subhalo) removal +
    N(>X), then quantiles across hosts (cells 96, 97).
    """
    N_per_MW = []
    for fname in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, fname)
        if not os.path.isfile(path):
            continue
        arr = np.load(path)
        logobs = np.log10(arr[key])
        distance = np.linalg.norm(arr['x'], axis=1)
        mask = (np.log10(arr['mvir']) - np.log10(arr['mpeak'])) > -2
        mask = mask & (distance < 259) & (arr['mpeak'] > 1.2e8)
        lmc_idx = np.argmax(np.log10(arr['mpeak']))
        mask[lmc_idx] = False
        N_per_MW.append(N_greater(logobs[mask], grid))
    N_per_MW = np.array(N_per_MW)
    qs = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
    return np.array([quantile(N_per_MW[:, i], quantiles=qs) for i in range(len(grid))])


def inference_quantiles(log_obs, sorted_idcs, weights_list, grid, rng, n_samples=1000):
    """
    Weighted-dwarf Monte Carlo: for each of n_samples mock Milky Ways, sample
    U(0,1) per dwarf and invert its SHMR/M-half-weighted quantile of the
    (shared or per-dwarf) SatGen catalog observable, stack the draws into a
    mock-MW array, then take N(>observable) and its 5 quantiles across the MC
    realizations (cells 75, 76, 98).

    log_obs / sorted_idcs may each be a single 1D array shared by all dwarfs
    (cell 98: the same rho150/mpeak catalog, reweighted per-dwarf) or a list
    of per-dwarf 1D arrays (cell 75: J is dwarf-specific, since it depends on
    the dwarf's own angular size/distance).

    rng: a per-figure np.random.Generator (see FIGURE_RNG_KEY) -- NOT the
    global np.random state, so a given figure's curves are identical whether
    it's rendered alone or as part of `all`.
    """
    n_dwarfs = len(weights_list)
    uniform_samples = rng.uniform(0.0, 1.0, size=(n_samples, n_dwarfs))
    joint_obs_MW = np.zeros_like(uniform_samples)
    for i in range(n_dwarfs):
        obs_i = log_obs[i] if isinstance(log_obs, list) else log_obs
        idcs_i = sorted_idcs[i] if isinstance(sorted_idcs, list) else sorted_idcs
        joint_obs_MW[:, i] = quantile_sorted(obs_i, idcs_i, weights=weights_list[i],
                                              quantiles=uniform_samples[:, i])
    N_per_MW = np.array([N_greater(joint_obs_MW[j], grid) for j in range(n_samples)])
    qs = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
    return np.array([quantile(N_per_MW[:, i], quantiles=qs) for i in range(len(grid))])


def render(series, grid, xlabel, ylabel, xlim, ylim, annotation, legend_loc, path,
           annotation_loc=(0.03, 0.03), ha='left', va='bottom', add_artist=True,
           tight_layout=False, lines_then_bands=False):
    """
    series: list of dicts, one per curve, with keys:
        q                 : quantiles array (required)
        color             : line/proxy color (required)
        label             : legend label (required)
        ls                : linestyle (default '-')
        band              : draw the 16/84 (68%) shaded band (default True)
        band_color        : color for that band (default: series color)
        band_alpha        : alpha for that band (default 0.2)
        extra_band        : also draw the 2.5/97.5 (95%) shaded band (default False)
        extra_band_color  : color for that band (default: band_color)
        extra_band_alpha  : alpha for that band (default 0.1)
    lines_then_bands: draw order. The notebook is NOT uniform here, and the
    difference is visible wherever series overlap:
        cell 80  (jfactors_all)        LBLBLBLBLB  interleaved
        cell 103 (sims_*, inference_*) LBLBLB      interleaved
        cell 90  (distance_statistics) LLLLBB      lines first
        cell 89  (distance_stats_sims) LLLBBB      lines first
    Interleaved (default) draws each series' line then its own band, so a
    later series' line sits on top of an earlier series' band. Setting
    lines_then_bands=True draws every line first, so all bands wash over all
    lines -- correct only for cells 90/89.
    add_artist: whether to re-add the legend as a second artist (frameon=False
    styling matches cells 90/89/103; cell 80/jfactors_all leaves this
    commented out in the notebook, so it must be skipped there to avoid a
    doubled/bolder legend).
    tight_layout: cell 103's figures (sims_mpeak/sims_rho150/inference_rho150)
    call fig.tight_layout(); cells 80/90/89 do not.
    NOTE: `grid` is a required argument even though not listed in the task's
    render() signature -- render() cannot plot without knowing the x-grid,
    and grid is shared by all series within a figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    proxy_lines = []

    def draw_line(s):
        color, ls = s['color'], s.get('ls', '-')
        ax.plot(10 ** grid, s['q'][:, 2], color=color, lw=2, ls=ls)
        proxy_lines.append(mlines.Line2D([], [], color=color, linewidth=1.5, ls=ls,
                                          label=s['label']))

    def draw_band(s):
        if not s.get('band', True):
            return
        band_color = s.get('band_color', s['color'])
        ax.fill_between(10 ** grid, s['q'][:, 1], s['q'][:, 3], color=band_color,
                         alpha=s.get('band_alpha', 0.2))
        if s.get('extra_band', False):
            extra_color = s.get('extra_band_color', band_color)
            ax.fill_between(10 ** grid, s['q'][:, 0], s['q'][:, 4], color=extra_color,
                             alpha=s.get('extra_band_alpha', 0.1))

    if lines_then_bands:
        for s in series:
            draw_line(s)
        for s in series:
            draw_band(s)
    else:
        for s in series:
            draw_line(s)
            draw_band(s)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    legend0 = ax.legend(handles=proxy_lines, fontsize=15, loc=legend_loc, frameon=False)
    if add_artist:
        ax.add_artist(legend0)
    if annotation:
        ax.text(annotation_loc[0], annotation_loc[1], annotation, transform=ax.transAxes,
                 fontsize=15, va=va, ha=ha, ma='center')
    if tight_layout:
        fig.tight_layout()

    fig.savefig(path)
    plt.close(fig)
    print('saved:', path)


# ---------------------------------------------------------------------------
# OBSERVABLES -- replaces the toggle-line blocks in notebook cells 93/95/96/97.
# Trimmed to the 3 observables the 6 in-scope figures actually use (mpeak,
# rho150, J); cV/vmax were only needed by the dropped lmc_vmax_hist figure.
# ---------------------------------------------------------------------------
OBSERVABLES = {
    'mpeak': dict(
        grid=np.linspace(6, 12, num=100),
        sim_key='mpeak',
        xlabel=r'$M_{\rm peak}$ [M$_{\odot}$]',
        ylabel=r'$N(>M_{\rm peak})$',
        xlim=(1.2e8, 1e11),
    ),
    'rho150': dict(
        grid=np.linspace(6, 10, num=100),
        sim_key='rho150',
        xlabel=r'$\rho_{150}$ [M$_{\odot}$ kpc$^{-3}$]',
        ylabel=r'$N(>\rho_{150})$',
        xlim=(7e6, 8e8),
    ),
    'J': dict(
        grid=np.linspace(14, 22, num=101),
        sim_key=None,
        xlabel=r'$J(0.5^{\circ})$ [GeV$^2$ cm$^{-5}$]',
        ylabel=r'$N(>J(0.5^{\circ}))$',
        xlim=(1e15, 3e21),
    ),
}


# ---------------------------------------------------------------------------
# Shared SatGen loading: LMC-analog host selection + single-heaviest-satellite
# removal (cells 73/82/93/95). rho150 has no GeV/cm^3 -> Msun/kpc^3 unit
# conversion applied -- see report: the raw npz values already sit in the
# 7e6-8e8 Msun/kpc^3 range that matches the xlim on the on-disk
# stats_sims/sims_rho150.pdf, so the conversion (left commented out in
# notebook cell 93, but applied in the unused/dead cell 90) would be wrong.
# ---------------------------------------------------------------------------
def load_satgen(version):
    with halo_weights.get_h5(version) as sats:
        mwID = sats['mwID'][()]
        logMpeak = np.log10(sats['Green_params'][()][:, 0])
        gc_distance = np.linalg.norm(sats['position'][()], axis=1)

    lmc_dist_cut = gc_distance < 100
    lmc_mass_cut = logMpeak > 11
    lmc_mwIDs = mwID[lmc_dist_cut & lmc_mass_cut]
    mw_hosts_lmc = np.isin(mwID, lmc_mwIDs)

    remove_lmc_mask = np.ones(len(logMpeak), dtype=bool)
    for mw in np.unique(mwID):
        mw_inds = np.where(mwID == mw)[0]
        max_ind = mw_inds[np.argmax(logMpeak[mw_inds])]
        remove_lmc_mask[max_ind] = False

    return mwID, logMpeak, gc_distance, mw_hosts_lmc, remove_lmc_mask


def satgen_logrho150(version, n_rows):
    rho150_file = rho150_path(version)
    rho150 = np.load(rho150_file)['rho150']
    assert len(rho150) == n_rows, (
        f"rho150 array length {len(rho150)} != h5 row count {n_rows} "
        f"(version={version!r})")
    return np.log10(rho150)


def rho150_path(version):
    return config.ADDITIONAL_DIR / f'{version}_rho150.npz'


def weights_path(dwarf_version, name):
    return config.WEIGHTS_DIR / dwarf_version / f'{name}.npz'


# ---------------------------------------------------------------------------
# sims_mpeak / sims_rho150
# ---------------------------------------------------------------------------
def build_sims(observable_key, fig_key, annotation, ylim):
    obs_info = OBSERVABLES[observable_key]
    grid = obs_info['grid']

    mwID, logMpeak, gc_distance, mw_hosts_lmc, remove_lmc_mask = load_satgen(version)
    mask = mw_hosts_lmc & remove_lmc_mask
    # DECIDED BEHAVIOUR: apply the M_peak > 1.2e8 floor to the SatGen curve
    # here so the sims comparison is like-for-like with Symphony/MWest (which
    # already impose it in sim_dir_quantiles). Notebook cell 93 left this
    # commented out for SatGen while cells 96/97 applied it to Symphony/MWest.
    # Measured effect: the floor drops ~51,603 of 332,383 rows and moves the
    # sims_rho150 median by up to 40 counts inside the plotted range (0 inside
    # sims_mpeak's) -- and reproduces the published sims_mpeak/sims_rho150
    # curves pixel-identically, so it is what makes this figure match the
    # paper, not a deviation from it.
    mask = mask & (10 ** logMpeak > 1.2e8)

    inputs = [config.h5_path(version), config.SYMPHONY_DIR / 'SymphonyMilkyWay',
              config.SYMPHONY_DIR / 'MWest']
    if observable_key == 'mpeak':
        values = logMpeak
    else:
        values = satgen_logrho150(version, len(mwID))
        inputs.append(rho150_path(version))

    satgen_q = number_function_quantiles(values[mask], mwID[mask], grid, comparator='greater')
    symphony_q = sim_dir_quantiles(symphony_base_dir, obs_info['sim_key'], grid)
    mwest_q = sim_dir_quantiles(mwest_base_dir, obs_info['sim_key'], grid)

    series = [
        dict(q=satgen_q, color=colors[0], label=r'\texttt{SatGen} with LMC analog'),
        dict(q=symphony_q, color=extra_colors[1], label='Symphony'),
        dict(q=mwest_q, color=extra_colors[2], label='Milky Way-est'),
    ]
    path = out_path(fig_key)
    render(series, grid, obs_info['xlabel'], obs_info['ylabel'], obs_info['xlim'], ylim,
           annotation, 'upper right', path, tight_layout=True)
    provenance.figure_manifest(path, 'python/plot_number_functions.py', inputs,
                               version=version, argv=sys.argv[1:], figure_key=fig_key)


# ---------------------------------------------------------------------------
# inference_rho150
# ---------------------------------------------------------------------------
def load_dwarf_weights(names, keys):
    """Load per-dwarf weight arrays for each key in `keys` from
    data/additional/weights_gc/{version}/<name>.npz. `keys` is a list of
    (array_name, index_or_None) pairs, e.g. ('mhalf_weights', None) or
    ('mstar_weights', 3). Returns a list (one per key) of lists (one per
    dwarf) of weight arrays."""
    out = [[] for _ in keys]
    for name in names:
        with np.load(weights_path(version, name)) as f:
            for j, (arr_name, idx) in enumerate(keys):
                arr = f[arr_name][idx] if idx is not None else f[arr_name]
                out[j].append(np.copy(arr))
    return out


def build_inference_rho150():
    obs_info = OBSERVABLES['rho150']
    grid = obs_info['grid']

    mwID, logMpeak, gc_distance, mw_hosts_lmc, remove_lmc_mask = load_satgen(version)
    logrho150 = satgen_logrho150(version, len(mwID))
    rng = np.random.default_rng([seed, FIGURE_RNG_KEY['inference_rho150']])

    # SatGen curve: LMC-analog hosts, M_peak > 1e8, d_GC < 50 kpc -- this is
    # the intended selection for the stats_inference figures (confirmed by
    # the user), not something derived from the annotation text below.
    mask = mw_hosts_lmc & remove_lmc_mask & (10 ** logMpeak > 1e8) & (gc_distance < 50)
    satgen_q = number_function_quantiles(logrho150[mask], mwID[mask], grid, comparator='greater')

    # Inference curves draw from the FULL (unmasked) catalog, reweighted
    # per-dwarf by the SHMR/M-half weights -- the mask above only restricts
    # the direct SatGen-sample curve, not the inference curves (cell 98).
    sorted_idcs = np.argsort(logrho150)

    dwarfs_50 = []
    for name in dwarf_names:
        if name in large_dwarf:
            continue
        dwarf = obs.Dwarf(name)
        if dwarf.gc_distance.to(u.kpc).value > 50:
            continue
        if name in problem_ultrafaints:
            continue
        dwarfs_50.append(name)

    weight_keys = [('mhalf_weights', None), ('mstar_weights', 3), ('mstar_weights', 9)]
    w_mhalf, w_f18, w_k24 = load_dwarf_weights(dwarfs_50, weight_keys)

    mhalf_q = inference_quantiles(logrho150, sorted_idcs, w_mhalf, grid, rng)
    f18_q = inference_quantiles(logrho150, sorted_idcs, w_f18, grid, rng)
    k24_q = inference_quantiles(logrho150, sorted_idcs, w_k24, grid, rng)

    series = [
        dict(q=satgen_q, color=colors[0], label=r'\texttt{SatGen} with LMC analog'),
        dict(q=mhalf_q, color=colors[1], label=r'$M_{1/2}$ inference'),
        dict(q=f18_q, color=colors[2], label=r'$M_{\star}$ inference - Fattahi+18'),
        dict(q=k24_q, color=colors[3], label=r'$M_{\star}$ inference - Kim+24'),
    ]
    # ylim: cell 103 renders three figures from one cell and carries two ylim
    # options -- the active (8e-1, 8e2) and a commented (8e-1, 9e1). 8e2 suits
    # the sims figures (counts ~250); this one tops out near 30, so 9e1 is the
    # option that belongs to it.
    path = out_path('inference_rho150')
    render(series, grid, obs_info['xlabel'], obs_info['ylabel'], obs_info['xlim'], (8e-1, 9e1),
           r'$M_{\rm peak}>10^8$ M$_{\odot}$, $d<50$ kpc', 'upper right', path, tight_layout=True)
    inputs = [config.h5_path(version), rho150_path(version)] + [
        weights_path(version, name) for name in dwarfs_50]
    provenance.figure_manifest(path, 'python/plot_number_functions.py', inputs,
                               version=version, argv=sys.argv[1:], figure_key='inference_rho150',
                               seed=seed)


# ---------------------------------------------------------------------------
# jfactors_all
# ---------------------------------------------------------------------------
def asymmetric_gaussian_ppf(q, value, lower_err, upper_err):
    from scipy.stats import norm
    q = np.atleast_1d(q)
    result = np.zeros_like(q, dtype=float)
    norm_factor = upper_err / (lower_err + upper_err)
    lower_mask = q <= norm_factor
    if np.any(lower_mask):
        q_scaled = q[lower_mask] / (2 * norm_factor)
        result[lower_mask] = norm.ppf(q_scaled, loc=value, scale=lower_err)
    upper_mask = q > norm_factor
    if np.any(upper_mask):
        q_scaled = 0.5 + (q[upper_mask] - norm_factor) / (2 * (1 - norm_factor))
        result[upper_mask] = norm.ppf(q_scaled, loc=value, scale=upper_err)
    return result[0] if result.size == 1 else result


def build_jfactors_all():
    # This figure's J-factor products (halo_position/<version>/halo_Js.npz
    # and <version>/<dwarf>/halo_Js.npz below) only exist for the Diemer
    # realization on disk. Refuse to silently mix a non-Diemer h5 catalog
    # with Diemer-only J-factors.
    if version != 'Diemer':
        raise SystemExit(
            f"jfactors_all only has J-factor products for version='Diemer'; got {version!r}")

    obs_info = OBSERVABLES['J']
    grid = obs_info['grid']
    n_samples = 1000
    rng = np.random.default_rng([seed, FIGURE_RNG_KEY['jfactors_all']])

    mwID, logMpeak, gc_distance, mw_hosts_lmc, remove_lmc_mask = load_satgen(version)
    # SatGen curve's J-factors: the combined per-halo array lives at
    # results/paper_Js/halo_position/<version>/halo_Js.npz (this repository's
    # Jhalopos.py writes a version subdirectory; see module docstring).
    halo_position_file = config.PAPER_JS_DIR / 'halo_position' / version / 'halo_Js.npz'
    Jfactor_arr = np.load(halo_position_file)['green_Js']
    log_obs_satgen = np.log10(Jfactor_arr)
    mask = mw_hosts_lmc & remove_lmc_mask
    satgen_q = number_function_quantiles(log_obs_satgen[mask], mwID[mask], grid, comparator='greater')

    # M_1/2, Fattahi+18, Kim+24 inference curves (cell 75): all non-large
    # dwarfs, NO d<50 restriction despite the "dwarfs_50" name in the
    # notebook -- that filter is commented out there (stale naming, not a cut).
    dwarfs_all = [name for name in dwarf_names if name not in large_dwarf]
    n_dwarfs_j = len(dwarfs_all)

    log_obs_list = []
    sorted_idcs_list = []
    halo_Js_files = []
    for name in dwarfs_all:
        halo_Js_file = config.PAPER_JS_DIR / version / name / 'halo_Js.npz'
        halo_Js_files.append(halo_Js_file)
        arr = np.log10(np.load(halo_Js_file)['green_Js'])
        log_obs_list.append(arr)
        sorted_idcs_list.append(np.argsort(arr))

    weight_keys = [('mhalf_weights', None), ('mstar_weights', 3)]
    w_mhalf, w_f18 = load_dwarf_weights(dwarfs_all, weight_keys)

    mhalf_q = inference_quantiles(log_obs_list, sorted_idcs_list, w_mhalf, grid, rng, n_samples)
    f18_q = inference_quantiles(log_obs_list, sorted_idcs_list, w_f18, grid, rng, n_samples)

    # Jeans curves (cell 76): point-estimate J with asymmetric Gaussian errors,
    # not a per-halo weighted quantile inversion, so it does not fit the
    # inference_quantiles() (weights_list-against-a-catalog) pattern and is
    # built directly here instead. Cell 79 loops this over both DJA priors
    # (dja_priors = ['loguniform', 'satgen_shmr_fattahi18']), drawing a fresh
    # rng.uniform(...) per prior -- reusing one block for both would change
    # the realization each curve is drawn from.
    # NOTE (KNOWN HAZARD, reproduced faithfully, not "fixed"): cell 76 skips
    # 'Tucana V' while cell 75's dwarf list does not, so only n_dwarfs_j - 1
    # columns of the (n_samples, n_dwarfs_j) array get filled; the last
    # column is left at its np.zeros_like() initial value. This means each
    # Jeans curve uses one fewer dwarf than the other curves. See report.
    dja_priors = ['loguniform', 'satgen_shmr_fattahi18']
    dja_files = []
    jeans_q = {}
    for prior in dja_priors:
        Jeans_J_50 = np.full(len(dwarf_names), np.nan)
        Jeans_J_84 = np.full(len(dwarf_names), np.nan)
        Jeans_J_16 = np.full(len(dwarf_names), np.nan)
        for i, name in enumerate(dwarf_names):
            dwarf = obs.Dwarf(name)
            dwarf.get_Jeans_results(dja_prior=prior)
            if dwarf.jeans_dja_path:
                dja_files.append(dwarf.jeans_dja_path)
            Jeans_J_16[i], Jeans_J_50[i], Jeans_J_84[i] = dwarf.Jeans_J_quantiles
            if np.isnan(Jeans_J_50[i]):
                Jeans_J_50[i] = 10 ** dwarf.logJ_disp
                Jeans_J_84[i] = 10 ** (dwarf.logJ_disp + dwarf.logJ_disp_err[1])
                Jeans_J_16[i] = 10 ** (dwarf.logJ_disp - dwarf.logJ_disp_err[0])

        uniform_samples = rng.uniform(0.0, 1.0, size=(n_samples, n_dwarfs_j))
        joint_obs_MW = np.zeros_like(uniform_samples)
        idx = 0
        for i, name in enumerate(dwarf_names):
            if name in large_dwarf:
                continue
            if name == 'Tucana V':
                continue
            mean = np.log10(Jeans_J_50[i])
            upper_err = np.log10(Jeans_J_84[i]) - mean
            lower_err = mean - np.log10(Jeans_J_16[i])
            joint_obs_MW[:, idx] = asymmetric_gaussian_ppf(uniform_samples[:, idx], mean, lower_err, upper_err)
            idx += 1
        joint_N_per_MW = np.array([N_greater(joint_obs_MW[j], grid) for j in range(n_samples)])
        qs = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
        jeans_q[prior] = np.array([quantile(joint_N_per_MW[:, i], quantiles=qs) for i in range(len(grid))])

    series = [
        dict(q=satgen_q, color=colors[0], label=r'\texttt{SatGen} with LMC analog'),
        dict(q=mhalf_q, color=colors[1], label=r'$M_{1/2}$ inference'),
        dict(q=f18_q, color=colors[2], label=r'$M_{\star}$ inference - Fattahi+18'),
        dict(q=jeans_q['loguniform'], color=jeans_color, label='Jeans analyses - standard prior'),
        dict(q=jeans_q['satgen_shmr_fattahi18'], color=extra_colors[1],
             label=r'Jeans analyses - \texttt{SatGen}-informed prior'),
    ]
    annotation = r'$M_{\rm peak}>10^8$ M$_{\odot}$' + '\n\n' + r'$N_{\rm stars}<10$ dwarfs' + '\n included'
    # cell 80's ax.add_artist(legend0) is commented out in the notebook,
    # unlike cells 87/89/100 -- skip it here so the legend isn't drawn twice.
    path = out_path('jfactors_all')
    render(series, grid, obs_info['xlabel'], obs_info['ylabel'], obs_info['xlim'], (8e-1, 6e3),
           annotation, 'upper right', path,
           annotation_loc=(0.03, 0.03), ha='left', va='bottom', add_artist=False)
    inputs = ([config.h5_path(version), halo_position_file] + halo_Js_files +
              [weights_path(version, name) for name in dwarfs_all])
    provenance.figure_manifest(path, 'python/plot_number_functions.py', inputs,
                               version=version, argv=sys.argv[1:], figure_key='jfactors_all',
                               seed=seed, dja_prior=dja_priors, dja_files=dja_files)


# ---------------------------------------------------------------------------
# distance_statistics / distance_statistics_sims
# ---------------------------------------------------------------------------
def build_distance_statistics():
    distance_grid = np.linspace(0, np.log10(1000), num=100)

    with halo_weights.get_h5(version) as sats:
        mwID = sats['mwID'][()]
        mpeak = sats['Green_params'][()][:, 0]
        distance = np.linalg.norm(sats['position'][()], axis=1)
    log_d = np.log10(distance)

    mask_0 = mpeak > 1.2e8
    lmc_mass_cut = mpeak > 1e11
    lmc_mwIDs_1 = mwID[(distance < 100) & lmc_mass_cut]
    lmc_mwIDs_2 = mwID[(distance < 75) & lmc_mass_cut]
    lmc_mwIDs_3 = mwID[(distance < 50) & lmc_mass_cut]
    mask_1 = np.isin(mwID, lmc_mwIDs_1)
    mask_2 = np.isin(mwID, lmc_mwIDs_2)
    mask_3 = np.isin(mwID, lmc_mwIDs_3)

    q0 = number_function_quantiles(log_d[mask_0], mwID[mask_0], distance_grid, comparator='less')
    q1 = number_function_quantiles(log_d[mask_1], mwID[mask_1], distance_grid, comparator='less')
    q2 = number_function_quantiles(log_d[mask_2], mwID[mask_2], distance_grid, comparator='less')
    q3 = number_function_quantiles(log_d[mask_3], mwID[mask_3], distance_grid, comparator='less')

    # Only q0 (the "all" curve) gets bands, shaded gray to match its dashed/
    # dotted/dash-dot gray siblings even though its median line is colors[0]
    # (black). q1/q2/q3 are unbanded gray curves distinguished by linestyle.
    series = [
        dict(q=q0, color=colors[0], label=r'\texttt{SatGen} - all',
             band_color='gray', band_alpha=0.3, extra_band=True, extra_band_color='gray'),
        dict(q=q1, color='gray', label=r'\texttt{SatGen} - $d_{\rm LMC}<100$ kpc', band=False, ls='--'),
        dict(q=q2, color='gray', label=r'\texttt{SatGen} - $d_{\rm LMC}<75$ kpc', band=False, ls=':'),
        dict(q=q3, color='gray', label=r'\texttt{SatGen} - $d_{\rm LMC}<50$ kpc', band=False, ls='-.'),
    ]
    # xlim upper bound 259 (not 300): the Diemer catalog's max d_GC is
    # 258.92 kpc, so 259-300 kpc is flat/empty.
    path = out_path('distance_statistics')
    render(series, distance_grid, r'$d_{\rm GC}$ [kpc]', r'$N(<d_{\rm GC})$', (5, 259), (0.5, 500),
           r'$M_{\rm peak}>10^8$ M$_{\odot}$', 'upper left', path,
           annotation_loc=(0.97, 0.03), ha='right', va='bottom', lines_then_bands=True)
    provenance.figure_manifest(path, 'python/plot_number_functions.py', [config.h5_path(version)],
                               version=version, argv=sys.argv[1:], figure_key='distance_statistics')


def build_distance_statistics_sims():
    distance_grid = np.linspace(0, np.log10(1000), num=100)

    with halo_weights.get_h5(version) as sats:
        mwID = sats['mwID'][()]
        mpeak = sats['Green_params'][()][:, 0]
        distance = np.linalg.norm(sats['position'][()], axis=1)
    log_d = np.log10(distance)

    lmc_mwIDs = mwID[(distance < 100) & (mpeak > 1e11)]
    mask_lmc = np.isin(mwID, lmc_mwIDs)
    # Explicit d < 259 kpc ceiling to match the Symphony/MWest cut in
    # sim_dist_quantiles() below -- the Diemer catalog's max d_GC (258.92 kpc)
    # already satisfies this, so the cut is a numerical no-op, but this makes
    # the like-for-like comparison auditable in code rather than a
    # coincidence of the catalog's contents.
    mask_4 = mask_lmc & (mpeak > 1.2e8) & (distance < 259)
    q_satgen = number_function_quantiles(log_d[mask_4], mwID[mask_4], distance_grid, comparator='less')

    def sim_dist_quantiles(sim_version):
        with halo_weights.get_h5(sim_version) as s:
            host_id = s['host_id'][()]
            d = np.linalg.norm(s['x'][()], axis=1)
            m = (d < 259) & (s['mpeak'][()] > 1.2e8)
        return number_function_quantiles(np.log10(d)[m], host_id[m], distance_grid, comparator='less')

    q_symphony = sim_dist_quantiles('Symphony')
    q_mwest = sim_dist_quantiles('MWest')

    series = [
        dict(q=q_satgen, color=colors[0], label=r'\texttt{SatGen} with LMC analog',
             band_color='gray', band_alpha=0.3),
        dict(q=q_symphony, color=extra_colors[1], label='Symphony', band_alpha=0.3),
        dict(q=q_mwest, color=extra_colors[2], label='Milky Way-est', band_alpha=0.3),
    ]
    # xlim upper bound 259 (not 300), same reason as distance_statistics.
    path = out_path('distance_statistics_sims')
    render(series, distance_grid, r'$d_{\rm GC}$ [kpc]', r'$N(<d_{\rm GC})$', (3, 259), (8e-1, 8e2),
           r'$M_{\rm peak}>1.2\times 10^8$ M$_{\odot}$', 'upper left', path,
           annotation_loc=(0.97, 0.03), ha='right', va='bottom', lines_then_bands=True)
    inputs = [config.h5_path(version), config.h5_path('Symphony'), config.h5_path('MWest')]
    provenance.figure_manifest(path, 'python/plot_number_functions.py', inputs,
                               version=version, argv=sys.argv[1:], figure_key='distance_statistics_sims')


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
BUILDERS = {
    'sims_mpeak': lambda: build_sims('mpeak', 'sims_mpeak', None, (8e-1, 8e2)),
    'sims_rho150': lambda: build_sims(
        'rho150', 'sims_rho150', r'$M_{\rm peak}>1.2 \times 10^8$ M$_{\odot}$', (8e-1, 8e2)),
    'inference_rho150': build_inference_rho150,
    'jfactors_all': build_jfactors_all,
    'distance_statistics': build_distance_statistics,
    'distance_statistics_sims': build_distance_statistics_sims,
}

if __name__ == '__main__':
    to_run = FIGURES if figure == 'all' else [figure]
    for name in to_run:
        BUILDERS[name]()
