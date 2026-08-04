"""Part I's main dwarf multipanel figure: sigma_LOS, M_peak, and rho150,
weighted by three inference variants (M_1/2, Fattahi+18 M_star, Kim+24
M_star), across the 39-dwarf canonical ordering `multipanel_systematics`
references.

Migrated from ../SatGen_Dwarf/jupyter/PaperPlots.ipynb cells 20 (data prep)
and 21 (the savefig). Path/import/provenance edits only -- the seven
per-dwarf *_quantiles arrays, the 16/50/84 -> pos/neg error-bar arithmetic,
the panel layout, marker/colour/index choices (`weights_idcs = [0, 4, 10]`
for rho150/Mpeak, `[4, 10]` for dispersion), and the dwarf ordering (imported
from `dwarf_categories`, which reproduces cell 4 verbatim -- see that
module's docstring) are all character-identical to the cells.

Two blocks of cell 20 are NOT ported, both because cell 21's active plotting
code never reads their output (every reference to it is commented out) --
this is a scope decision, not a cleanup, so it is named here rather than
silently dropped:

* The Jeans-results loop (`Jeans_rho150_quantiles`, `Jeans_J_quantiles`,
  `Jeans_cV_quantiles`, `Jeans_Mvir_quantiles`, `J_scaling_dispersion`,
  `J_scaling_luminosity`, the McDaniel `Jeans_J_*` arrays, `scaling_idcs`):
  it calls `Dwarf.get_Jeans_results` once per dwarf (a DwarfJeansAnalysis
  read), which this figure never plots and would make every run of this
  script pay for regardless.
* `dwarf_mhalf` / `dwarf_mhalf_err` (the observed M_1/2, via
  `Dwarf.logMhalf`/`logMhalf_err`): cell 21's M_1/2 y-label and any M_1/2
  series are commented out; only the measured velocity dispersion
  (`dwarf_disp`/`dwarf_disp_err`) is plotted, and that IS ported.
* The commented-out Geha block in cell 20 (`geha_*` lists) was already
  disabled upstream; nothing here reconstructs it.

The Okabe-Ito `colors` list cell 21 redefines locally is identical to
`plot_style.colors` (both are cell 1's palette) -- reused from `plot_style`
here instead of re-pasted, matching the "reused, not re-derived" convention
(see save_contours.py's docstring for the analogous precedent with
config.WEIGHTS_DIR/config.PAPER_CONTOURS_DIR).

`version` (default 'Diemer', matching cell 20's hardcoded value) selects
`results/paper_quantiles/galactocentric/<version>/` and becomes the figure's
`multipanel_<version>_paper1.pdf` basename, exactly as cell 21's f-string
did. `provenance.assert_single_version` is run over the 43 per-dwarf npz
files actually loaded before rendering -- cell 20 has no equivalent check;
this adds the version-mixing guard used everywhere else in this repo (see
e.g. save_contours.py) rather than trusting the directory name alone.

Usage:
    conda activate J_calc
    python plot_multipanel_paper1.py [--version Diemer]
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_style import *  # noqa: F401,F403 -- rcParams side effects, `colors`
import config
import provenance
import Jdata as obs
from dwarf_categories import (
    dwarf_names, idcs, n_dwarfs, problem_idcs, ultrafaint_idcs, classical_idcs,
)

# The seven per-dwarf *_quantiles series cell 20 loads for every dwarf. Only
# rho150/Mpeak/dispersion are plotted (see module docstring); the rest are
# loaded anyway so this loader stays a faithful, checkable port of cell 20's
# loop -- npz reads are cheap, unlike the dropped Jeans loop.
QUANTILE_KEYS = ['cV_quantiles', 'rho150_quantiles', 'J_quantiles',
                 'Mvir_quantiles', 'Mpeak_quantiles', 'Mhalf_quantiles',
                 'dispersion_quantiles']


def load_quantiles(version):
    """Cell 20's per-dwarf load loop, for all seven *_quantiles series.

    Returns (quantiles, used_files): `quantiles` maps each npz key to a
    (n_dwarfs_all, 3, n_weights) array (dwarf axis in `dwarf_names` order,
    matching cell 20's list-then-np.array construction); `used_files` is the
    list of npz paths read, same order.
    """
    quant_dir = config.PAPER_QUANTILES_DIR / 'galactocentric' / version
    lists = {key: [] for key in QUANTILE_KEYS}
    used_files = []
    for dwarf in dwarf_names:
        path = quant_dir / f'{dwarf}.npz'
        quantiles_file = np.load(path)
        for key in QUANTILE_KEYS:
            lists[key].append(quantiles_file[key])
        used_files.append(path)
    return {key: np.array(vals) for key, vals in lists.items()}, used_files


def load_dwarf_dispersion():
    """Measured <sigma_LOS> and its (lo, hi) error, per dwarf (cell 20)."""
    disp, disp_err = [], []
    for dwarf in dwarf_names:
        d = obs.Dwarf(dwarf)
        disp.append(d.dispersion.value)
        disp_err.append(d.dispersion_err.value)
    return np.array(disp), np.array(disp_err)


# labels, colors, markers (cell 21)
labels = [r'$M_{1/2}$ inference',
          r'$M_{\star}$ inference - Fattahi+18',
          r'$M_{\star}$ inference - Kim+24',
          r'Jeans analyses',
          ]
markers = ['s', 'D', '^', 'o', 'p']

extra_colors = [
    "#882255",  # wine
    "#44AA99",  # teal
    "#DDCC77",  # sand
]

# Unused by any active cell-21 code (only referenced in commented-out Jeans
# lines); kept as upstream's dead constants for a faithful port.
jeans_color = '#4B0082'  # indigo
jeans_fattahi18_prior_color = '#999933'  # olive


def build_figure(version):
    quantiles, quantile_files = load_quantiles(version)
    provenance.assert_single_version(quantile_files, expected=version)

    rho150quantiles = quantiles['rho150_quantiles']
    Mpeakquantiles = quantiles['Mpeak_quantiles']
    dispquantiles = quantiles['dispersion_quantiles']

    dwarf_disp, dwarf_disp_err = load_dwarf_dispersion()

    rho150_16, rho150_50, rho150_84 = rho150quantiles[:, 0], rho150quantiles[:, 1], rho150quantiles[:, 2]
    rho150_pos_err = rho150_84 - rho150_50
    rho150_neg_err = rho150_50 - rho150_16

    Mpeak_16, Mpeak_50, Mpeak_84 = Mpeakquantiles[:, 0], Mpeakquantiles[:, 1], Mpeakquantiles[:, 2]
    Mpeak_pos_err = Mpeak_84 - Mpeak_50
    Mpeak_neg_err = Mpeak_50 - Mpeak_16

    disp_16, disp_50, disp_84 = dispquantiles[:, 0], dispquantiles[:, 1], dispquantiles[:, 2]
    disp_pos_err = disp_84 - disp_50
    disp_neg_err = disp_50 - disp_16

    dwarf_disp_50 = dwarf_disp
    dwarf_disp_84 = dwarf_disp + dwarf_disp_err[:, 1]
    dwarf_disp_16 = dwarf_disp - dwarf_disp_err[:, 0]
    dwarf_disp_pos_err = dwarf_disp_84 - dwarf_disp_50
    dwarf_disp_neg_err = dwarf_disp_50 - dwarf_disp_16

    # three panels sharing x-axis (cell 21's "make four panels" comment is
    # stale w.r.t. its own `plt.subplots(3, 1, ...)` call, kept as upstream)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(14, 11))
    plt.subplots_adjust(hspace=0)

    x_positions = np.arange(n_dwarfs) * 40

    weights_idcs = [0, 4, 10]
    plot_colors = [colors[1], colors[2], colors[3]]

    # plot rho150
    for i, idx in enumerate(weights_idcs):
        axs[2].errorbar(x_positions + 6*i*np.ones(n_dwarfs), rho150_50[:, idx][idcs],
                 yerr=(rho150_neg_err[:, idx][idcs], rho150_pos_err[:, idx][idcs]),
                 fmt=markers[i], markersize=5, capsize=2, mec=plot_colors[i], c=plot_colors[i], mfc='white')

    # plot Mpeak
    for i, idx in enumerate(weights_idcs):
        axs[1].errorbar(x_positions + 6*i*np.ones(n_dwarfs), Mpeak_50[:, idx][idcs],
                 yerr=(Mpeak_neg_err[:, idx][idcs], Mpeak_pos_err[:, idx][idcs]),
                 fmt=markers[i], markersize=5, capsize=2, mec=plot_colors[i], c=plot_colors[i], mfc='white')

    # Add gray cross-hatching below 10^8 on the y-axis
    axs[1].axhspan(1, 1e8, facecolor='none', edgecolor='gray', hatch='//////', alpha=0.5, zorder=0)

    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')

    axs[2].set_ylabel(r'$\rho_{150}$ [M$_{\odot}$ kpc$^{-3}$]', fontsize=15)
    axs[1].set_ylabel(r'$M_{\rm peak}$ [M$_{\odot}$]', fontsize=15)
    axs[0].set_ylabel(r'$\sigma_{\rm LOS}$ [km s$^{-1}$]', fontsize=15)

    # plot dispersion
    weights_idcs = [4, 10]
    for i, idx in enumerate(weights_idcs):
        axs[0].errorbar(x_positions + 6*(i+1)*np.ones(n_dwarfs), disp_50[:, idx][idcs],
                 yerr=(disp_neg_err[:, idx][idcs], disp_pos_err[:, idx][idcs]),
                 fmt=markers[i+1], markersize=5, capsize=2, mec=plot_colors[i+1], c=plot_colors[i+1], mfc='white')

    axs[0].errorbar(x_positions + 6*0*np.ones(n_dwarfs), dwarf_disp_50[idcs],
                 yerr=(dwarf_disp_neg_err[idcs], dwarf_disp_pos_err[idcs]),
                 fmt=markers[0], markersize=5, capsize=2, mec=extra_colors[0], c=extra_colors[0], mfc='white')

    # --- Shading groups ---
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
            center = (x_positions[start_idx] + x_positions[end_idx]) / 2  # center over actual data
            ax.axvspan(left, right, facecolor=color, alpha=0.1, zorder=0)
            if i == 0:
                ax.text(center, 0.9, label, ha='center', va='top', fontsize=15, color='k', fontweight='bold', transform=ax.get_xaxis_transform())
            start_idx += count

    # x-axis dwarf names
    dwarf_names_latex = np.array([s.replace('Bootes', 'Boötes') for s in dwarf_names])
    axs[2].set_xticks(np.arange(n_dwarfs)*40 + 5*np.ones(n_dwarfs), dwarf_names_latex[idcs], rotation=90, fontsize=15)
    axs[2].xaxis.set_minor_locator(plt.NullLocator())

    # Add soft dashed grey vertical lines between each dwarf galaxy
    for ax in axs:
        for k in range(n_dwarfs):
            if k == 0:
                separator_x = x_positions[k] - 12
                ax.axvline(x=separator_x, color='grey', linestyle='--', linewidth=0.5, alpha=0.6, zorder=1)
            separator_x = x_positions[k] + 28  # midpoint between dwarfs (spacing is 40)
            ax.axvline(x=separator_x, color='grey', linestyle='--', linewidth=0.5, alpha=0.6, zorder=1)

    axs[2].set_ylim(6e5, 3e9)
    axs[1].set_ylim(6e7, 5e11)
    axs[0].set_ylim(5e-1, 7e1)

    axs[0].tick_params(axis='y', labelsize=15)
    axs[1].tick_params(axis='y', labelsize=15)
    axs[2].tick_params(axis='y', labelsize=15)

    axs[2].text(
        0.02, 0.05, r'$M_{\rm peak}>10^8$ M$_{\odot}$',
        transform=axs[2].transAxes,
        fontsize=15,
        va='bottom', ha='left',
    )

    # Custom legend entries with markers
    proxy_lines = [
        Line2D([0], [0], marker=markers[0], color=extra_colors[0], linestyle='None', label=r'Measured $\langle\sigma_{\rm LOS}\rangle$', markersize=10, mfc='white'),
        Line2D([0], [0], marker=markers[0], color=colors[1], linestyle='None', label=labels[0], markersize=10, mfc='white'),
        Line2D([0], [0], marker=markers[1], color=colors[2], linestyle='None', label=labels[1], markersize=10, mfc='white'),
        Line2D([0], [0], marker=markers[2], color=colors[3], linestyle='None', label=labels[2], markersize=10, mfc='white'),
    ]

    axs[2].legend(loc='lower right', fontsize=14, handles=proxy_lines, ncol=2, frameon=False)

    return fig, quantile_files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='Diemer')
    args = parser.parse_args(argv)

    fig, quantile_files = build_figure(args.version)

    out_path = config.PLOTS_DIR / 'panels' / f'multipanel_{args.version}_paper1.pdf'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    provenance.figure_manifest(
        out_path, 'python/plot_multipanel_paper1.py',
        inputs=list(quantile_files) + [config.LVDB_DIR / 'dwarf_all.csv'],
        version=args.version, weights_idcs=[0, 4, 10])
    print('wrote', out_path)


if __name__ == '__main__':
    main()
