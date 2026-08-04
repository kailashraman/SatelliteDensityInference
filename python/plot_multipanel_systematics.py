"""Six-panel comparison of systematic uncertainties on rho150, one row per
systematic: c-M relation, SHMR, dynamical mass estimator, host mass, LMC
selection, mass floor.

Reproduces ../SatGen_Dwarf/jupyter/PaperPlots.ipynb cells 37-46 (load rho150
quantiles for eight SatGen versions) -> cell 48 (the savefig). Cell 47 is
print-only (a diagnostic diff never plotted) and is dropped, per the step plan.

VERSIONS AS FOUND IN CELL 48 -- the migration brief's version list handed to
this step named ten versions including Symphony and MWest; the cells and the
published PDF (../SatGen_Dwarf/plots/paper_plots_take_3/panels/
multipanel_systematics.pdf, rasterized and read panel-by-panel to confirm)
both show only eight, and Symphony/MWest are not among them even though cells
44/45 load their quantiles: cell 48's fourth panel ("host mass") plots only
`rho150_50_diemer`/`rho150_50_m2e12`, and its `sim_labels`/`sim_proxy_lines`
lists carry four entries (including 'Symphony', 'Milky Way-est') but the
Symphony/MWest `axs[3].errorbar` calls and their proxy-line entries are
commented out in the source cell. This matches drafts_temp/temp_part1.tex's
prose around \\label{fig:systematics} (\\autoref{app:systematics}), which
describes exactly the eight-version, six-panel structure below and never
mentions Symphony/MWest for this figure. Cells 44/45 (Symphony/MWest loads)
are therefore not ported here -- there is nothing in cell 48 that reads them.

Panel / version / weight-column mapping, as found in cell 48 (column 0 is
mhalf_weights; SHMR columns from compute_quantiles.py's SHMR_names order, see
tests/test_weight_column_mapping.py; column 21 is mdyn_errani, present only in
the mhalf_scatter-version quantiles file):

  panel 0 (c-M relation):       Diemer[:,0], Diemer_0p12[:,0], Zhao[:,0]
  panel 1 (SHMR):                Diemer[:,4] (Fattahi18), Diemer[:,5] (Moster18),
                                  Diemer[:,10] (Kim24), Diemer[:,9] (Danieli23_grow
                                  -- column 8 is the other Danieli23 variant,
                                  Danieli23_const, ~1.65x lower; not plotted here)
  panel 2 (mass estimator):      Diemer[:,0] (Wolf), mhalf_scatter[:,0]
                                  (Wolf+scatter), mhalf_scatter[:,21] (Errani)
  panel 3 (host mass):           Diemer[:,0], m2e12[:,0]
  panel 4 (LMC analogue):        Diemer[:,0] (none), lmc[:,0] (100 kpc),
                                  lmc_50[:,0] (50 kpc) -- 'lmc' is the 100 kpc
                                  cut and 'lmc_50' the 50 kpc cut, per cell 48's
                                  own labels, not per either version's name
  panel 5 (mass floor):          Diemer[:,0]/[:,10] (1e8 floor, mhalf/Kim24),
                                  mass_floor_7[:,0]/[:,10] (1e7 floor, mhalf/Kim24)

Dwarf ordering, category masks (problem-ultrafaints / ultrafaints / classicals)
and the group-count/shading logic are taken from dwarf_categories, matching
the caption's claim that this figure shares the main multipanel figure's
x-axis ordering -- not re-derived here.

Path/import/provenance edits only; the quantile selection, weight-column
indices, panel ordering, error-bar definitions, marker/color/offset choices,
and dwarf ordering are otherwise character-identical to cell 48. The
`markers`/Okabe-Ito `colors` lists cell 48 re-pastes locally are the same
values plot_style.py already exposes at module level, so this module imports
`colors` from plot_style rather than repeating the literal (`markers` has no
existing home, so it stays local, same as cell 48).

config constants reused: config.PAPER_QUANTILES_DIR, config.PLOTS_DIR. No new
config constant was needed -- all eight versions' quantiles already live
under config.PAPER_QUANTILES_DIR / 'galactocentric' / <version>.

Provenance: this figure reads eight SatGen versions (344 per-dwarf quantile
files total, 43 dwarfs x 8 versions), so per-version mixing cannot be caught
by a single assert_single_version call the way single-version figures use it.
Instead, `_load_version` calls `provenance.assert_single_version(paths,
expected=version)` separately for each of the eight versions before use, and
raises loudly (FileNotFoundError/ValueError from provenance.py) rather than
skipping a series if any dwarf's quantiles file for that version is missing or
unstamped. The figure manifest embeds every one of the 344 input files
individually (each fingerprinted with its own one-hop-deep provenance via
provenance.figure_manifest), so which version supplied any given point is
recoverable from the sidecar; the manifest's own `version=` kwarg is left
`None` (this figure has no single answer to "which version") with the full
version list recorded under `versions=`.

`assert_single_version(expected=version)` alone does NOT close the per-version
mixing risk for this figure: 'lmc', 'lmc_50' and 'mhalf_scatter' all resolve
(via config.SIM_VERSION_ALIASES) to `sim_version='Diemer'`, and
assert_single_version's alias-mixing check only inspects records whose raw
`satgen_version` differs from that resolved value -- a plain `Diemer` product
misplaced in an `lmc/` directory has `satgen_version == sim_version` and is
invisible to that check. `_load_version` therefore also verifies, for every
path, that the file's own stamped `satgen_version` equals the raw directory
name it was read from -- a hard gate (every currently-stamped product's
`satgen_version` matches its directory), not a heuristic.
"""
import sys

import numpy as np
from matplotlib.lines import Line2D

import config
import provenance
from plot_style import *  # noqa: F401,F403 -- rcParams side effect, plt, mpl, colors
from dwarf_categories import (
    dwarf_names, idcs, n_dwarfs, problem_idcs, ultrafaint_idcs, classical_idcs)

SCRIPT = 'python/plot_multipanel_systematics.py'

VERSIONS = ('Diemer', 'Diemer_0p12', 'Zhao', 'mhalf_scatter', 'm2e12',
            'lmc', 'lmc_50', 'mass_floor_7')


def _quantiles_path(version, dwarf):
    return config.PAPER_QUANTILES_DIR / 'galactocentric' / version / f'{dwarf}.npz'


def _load_version(version):
    """rho150 median/pos_err/neg_err for `version`, one row per dwarf (in
    dwarf_categories.dwarf_names order, i.e. all 43, matching cells 37-46) and
    one column per weight column. Verifies every one of the 43 per-dwarf
    quantile files actually came from `version` before returning.
    """
    paths = [_quantiles_path(version, dwarf) for dwarf in dwarf_names]
    provenance.assert_single_version(paths, expected=version)

    # assert_single_version's alias-mixing check cannot catch a plain,
    # non-aliased product (satgen_version == sim_version) sitting in an
    # aliased version's directory -- see module docstring. Close that gap
    # directly: every file's own stamped satgen_version must equal the raw
    # directory name it was read from.
    for path in paths:
        raw = provenance.read(path).get('satgen_version')
        if raw != version:
            raise ValueError(
                f'{path} carries satgen_version={raw!r}, but was read from '
                f'the {version!r} directory -- misplaced file?')

    rho150quantiles_list = [np.load(p)['rho150_quantiles'] for p in paths]
    rho150quantiles = np.array(rho150quantiles_list)

    rho150_16, rho150_50, rho150_84 = np.array(
        [rho150quantiles[:, 0], rho150quantiles[:, 1], rho150quantiles[:, 2]])

    rho150_pos_err = rho150_84 - rho150_50
    rho150_neg_err = rho150_50 - rho150_16
    return rho150_50, rho150_pos_err, rho150_neg_err, paths


def make_figure():
    inputs = []
    q = {}
    for version in VERSIONS:
        med, pos_err, neg_err, paths = _load_version(version)
        q[version] = (med, pos_err, neg_err)
        inputs.extend(paths)

    markers = ['s', 'D', '^', 'o', 'd', 'p']

    fig, axs = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    plt.subplots_adjust(hspace=0)

    x_positions = np.arange(n_dwarfs) * 40

    def draw_series(ax, version, column, offset, marker, color):
        med, pos_err, neg_err = q[version]
        ax.errorbar(
            x_positions + offset * np.ones(n_dwarfs), med[:, column][idcs],
            yerr=(neg_err[:, column][idcs], pos_err[:, column][idcs]),
            fmt=marker, markersize=5, capsize=2, mec=color, c=color, mfc='white')

    def finish_panel(ax, proxy_lines, ylim):
        ax.set_yscale('log')
        ax.set_ylabel(r'$\rho_{150}$ [M$_{\odot}$ kpc$^{-3}$]', fontsize=15)
        ax.set_ylim(*ylim)
        ax.tick_params(axis='y', labelsize=15)
        ax.legend(loc='lower right', fontsize=13, handles=proxy_lines, ncol=4, frameon=False)

    # -- panel 0: c-M relation --------------------------------------------
    cm_labels = [r'Diemer+19 - 0.16 dex scatter', r'Diemer+19 - 0.12 dex scatter', r'Zhao+09']
    cm_colors = [colors[1], colors[5], colors[6]]
    draw_series(axs[0], 'Diemer', 0, 6 * -1, markers[0], cm_colors[0])
    draw_series(axs[0], 'Diemer_0p12', 0, 6 * 0.5, markers[1], cm_colors[1])
    draw_series(axs[0], 'Zhao', 0, 6 * 2, markers[2], cm_colors[2])
    cm_proxy_lines = [
        Line2D([0], [0], marker=markers[i], color=cm_colors[i], linestyle='None',
               label=cm_labels[i], markersize=10, mfc='white') for i in range(3)]
    finish_panel(axs[0], cm_proxy_lines, (2e6, 1e10))

    # -- panel 1: SHMR -------------------------------------------------------
    shmr_labels = [r'Fattahi+18', r'Moster+18', r'Kim+24', r'Danieli+23']
    shmr_colors = [colors[2], colors[7], colors[3], colors[6]]
    draw_series(axs[1], 'Diemer', 4, 6 * -1, markers[0], shmr_colors[0])
    draw_series(axs[1], 'Diemer', 5, 6 * 0.5, markers[1], shmr_colors[1])
    draw_series(axs[1], 'Diemer', 10, 6 * 2, markers[2], shmr_colors[2])
    draw_series(axs[1], 'Diemer', 9, 6 * 3.5, markers[3], shmr_colors[3])
    shmr_proxy_lines = [
        Line2D([0], [0], marker=markers[i], color=shmr_colors[i], linestyle='None',
               label=shmr_labels[i], markersize=10, mfc='white') for i in range(4)]
    finish_panel(axs[1], shmr_proxy_lines, (2e6, 1e10))

    # -- panel 2: dynamical mass estimator ------------------------------------
    mdyn_labels = [r'Wolf+10', r'Wolf+10 w/ scatter', r'Errani+18 w/ scatter']
    mdyn_colors = [colors[1], colors[5], colors[6]]
    draw_series(axs[2], 'Diemer', 0, 6 * -1, markers[0], mdyn_colors[0])
    draw_series(axs[2], 'mhalf_scatter', 0, 6 * 0.5, markers[1], mdyn_colors[1])
    draw_series(axs[2], 'mhalf_scatter', 21, 6 * 2, markers[2], mdyn_colors[2])
    mdyn_proxy_lines = [
        Line2D([0], [0], marker=markers[i], color=mdyn_colors[i], linestyle='None',
               label=mdyn_labels[i], markersize=10, mfc='white') for i in range(3)]
    finish_panel(axs[2], mdyn_proxy_lines, (2e6, 1e10))

    # -- panel 3: host mass ----------------------------------------------------
    # Cell 48 loads Symphony/MWest quantiles (cells 44/45) and carries
    # sim_labels/sim_proxy_lines entries for them, but their errorbar and
    # proxy-line calls are commented out in the source cell -- only the two
    # host-mass series are actually plotted. See module docstring.
    sim_labels = [r'$M_{\rm host}=1\times 10^{12}$ M$_{\odot}$', r'$M_{\rm host}=2\times 10^{12}$ M$_{\odot}$']
    sim_colors = [colors[1], colors[5]]
    draw_series(axs[3], 'Diemer', 0, 6 * -1, markers[0], sim_colors[0])
    draw_series(axs[3], 'm2e12', 0, 6 * 0.5, markers[1], sim_colors[1])
    sim_proxy_lines = [
        Line2D([0], [0], marker=markers[i], color=sim_colors[i], linestyle='None',
               label=sim_labels[i], markersize=10, mfc='white') for i in range(2)]
    finish_panel(axs[3], sim_proxy_lines, (2e6, 1e10))

    # -- panel 4: LMC analogue -------------------------------------------------
    lmc_labels = [r'No LMC selection', r'LMC within 100 kpc', r'LMC within 50 kpc']
    lmc_colors = [colors[1], colors[5], colors[6]]
    draw_series(axs[4], 'Diemer', 0, 6 * -1, markers[0], lmc_colors[0])
    draw_series(axs[4], 'lmc', 0, 6 * 0.5, markers[1], lmc_colors[1])
    draw_series(axs[4], 'lmc_50', 0, 6 * 2, markers[2], lmc_colors[2])
    lmc_proxy_lines = [
        Line2D([0], [0], marker=markers[i], color=lmc_colors[i], linestyle='None',
               label=lmc_labels[i], markersize=10, mfc='white') for i in range(3)]
    finish_panel(axs[4], lmc_proxy_lines, (2e6, 1e10))

    # -- panel 5: mass floor -----------------------------------------------
    floor_labels = [r'$M_{\rm peak}>10^8\;{\rm M}_{\odot}$ - $M_{1/2}$',
                    r'$M_{\rm peak}>10^7\;{\rm M}_{\odot}$ - $M_{1/2}$',
                    r'$M_{\rm peak}>10^8\;{\rm M}_{\odot}$ - Kim+24',
                    r'$M_{\rm peak}>10^7\;{\rm M}_{\odot}$ - Kim+24']
    floor_colors = [colors[1], colors[5], colors[3], colors[6]]
    draw_series(axs[5], 'Diemer', 0, 6 * -1, markers[0], floor_colors[0])
    draw_series(axs[5], 'mass_floor_7', 0, 6 * 0.5, markers[1], floor_colors[1])
    draw_series(axs[5], 'Diemer', 10, 6 * 2, markers[2], floor_colors[2])
    draw_series(axs[5], 'mass_floor_7', 10, 6 * 3.5, markers[3], floor_colors[3])
    floor_proxy_lines = [
        Line2D([0], [0], marker=markers[i], color=floor_colors[i], linestyle='None',
               label=floor_labels[i], markersize=10, mfc='white') for i in range(4)]
    finish_panel(axs[5], floor_proxy_lines, (7e5, 1e10))

    # -- category shading + labels, dwarf-count separators, x-ticks --------
    group_counts = [len(problem_idcs), len(ultrafaint_idcs), len(classical_idcs)]
    group_colors = [colors[6], colors[5], colors[7]]
    group_labels = [r'$N_{\rm stars}<10$', 'Ultra-faints', r'$M_{\star}>10^5$ M$_{\odot}$']

    for i, ax in enumerate(axs):
        start_idx = 0
        left, right = ax.get_xlim()
        ax.set_xlim(left, right)
        for j, (count, color, label) in enumerate(zip(group_counts, group_colors, group_labels)):
            if count == 0:
                continue
            end_idx = start_idx + count - 1
            left = ax.get_xlim()[0] if j == 0 else x_positions[start_idx] - 12
            right = ax.get_xlim()[1] if j == len(group_counts) - 1 else x_positions[end_idx] + 28
            center = (x_positions[start_idx] + x_positions[end_idx]) / 2
            ax.axvspan(left, right, facecolor=color, alpha=0.1, zorder=0)
            if i == 0:
                ax.text(center, 0.9, label, ha='center', va='top', fontsize=15, color='k',
                        fontweight='bold', transform=ax.get_xaxis_transform())
            start_idx += count

    dwarf_names_latex = np.array([s.replace('Bootes', 'Boötes') for s in dwarf_names])
    axs[5].set_xticks(np.arange(n_dwarfs) * 40 + 5 * np.ones(n_dwarfs),
                      dwarf_names_latex[idcs], rotation=90, fontsize=15)
    axs[5].xaxis.set_minor_locator(plt.NullLocator())

    for ax in axs:
        for k in range(n_dwarfs):
            if k == 0:
                separator_x = x_positions[k] - 12
                ax.axvline(x=separator_x, color='grey', linestyle='--', linewidth=0.5, alpha=0.6, zorder=1)
            separator_x = x_positions[k] + 28
            ax.axvline(x=separator_x, color='grey', linestyle='--', linewidth=0.5, alpha=0.6, zorder=1)

    out_dir = config.PLOTS_DIR / 'panels'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'multipanel_systematics.pdf'
    plt.savefig(out_path)
    plt.close(fig)

    provenance.figure_manifest(out_path, SCRIPT, inputs, version=None,
                               argv=sys.argv[1:], versions=list(VERSIONS))
    return out_path


def main():
    path = make_figure()
    print('saved:', path)


if __name__ == '__main__':
    main()
