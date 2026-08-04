"""fermi_limit_comparison.pdf: this work's recast Fermi-LAT sigma-v limits
against Circiello+26's benchmark-sample limit and Abdollahi+26's two
J-factor variants (GS+14 / Bonnivard+15), for Part II's appendix
(\\autoref{app:limit_comparison}).

Migrated from ../SatGen_Dwarf/jupyter/PaperPlots.ipynb cells 190 -> 191.
Unlike the other figure families, this one was never ported to a .py driver
upstream -- the notebook cells are the only prior source, so this is a first
migration rather than a path/import edit of an existing script.

Cell 190 is a standalone diagnostic: the ratio of the Circiello+26 to
McDaniel+23 published limit curves, linearly interpolated onto a common
500-point log-spaced mass grid spanning their overlap, with only
`print(np.min(ratio), np.max(ratio))` as output -- no plot is saved. It feeds
no array into fermi_limit_comparison.pdf; reproduced here verbatim as
`_report_circiello_mcdaniel_ratio`, run for its printed min/max only, to keep
this driver's numerics traceable to both source cells as given.

Cell 191 draws five curves plus the thermal-relic reference line (confirmed
against the rasterized ../SatGen_Dwarf/plots/paper_plots_take_3/
fermi_limit_comparison.pdf, which shows exactly this and no GCE band):
    Circiello2026_limit.csv                                 - solid gray
    abdollahi2025_limit_gs.csv                              - dashed gray
    abdollahi2025_limit_b.csv                                - dotted gray
    This work, mhalf inference       (summed_list[1])        - solid black
    This work, Jeans SatGen-informed (summed_list[8])         - dotted black
The two black curves are our own fermi_reweighting products (WEIGHT_LIST
indices 1 = 'mhalf' and 8 = 'Jeans_satgen_shmr_fattahi18' in
plot_fermi_reweighting.py), stacked over the same SKIP_DWARFS-excluded dwarf
sample and extracted with the same fermi_funcs.get_UL upper-limit routine.
`load_summed`, `draw_thermal_relic` and `find_thermal_crossing_report` are
imported from plot_fermi_reweighting.py rather than re-pasted -- that module
is not one of the shared modules this step is barred from editing, only
imported from.

DROPPED, verbatim residue in the cell that renders nothing in the published
figure: the uncommented `files = ["../data/Abazajian2015.csv",
"../data/Daylan2014.csv", "../data/Calore2014.csv"]` line. Its only consumer,
`shade_bounding_box_to_zero(files, ...)`, is commented out immediately below
it in the cell, along with `box_proxy`, the `first_handles`/`first_legend`
GCE-region legend entry, and the DiMauro+21 GCE errorbar block -- `files`
itself is dead, unused post-run residue, not a live input. Not ported. Also
dropped: the "Dwarfs shared with McDaniel+23" corner annotation, which is
commented out in the cell (only the bb-bar annotation below it is live).

Literature CSVs (`../data/<name>.csv` upstream) are read from
`config.DATA_DIR / 'literature'` here (via plot_fermi_reweighting.LITERATURE_DIR),
all four present: Circiello2026_limit.csv, abdollahi2025_limit_gs.csv,
abdollahi2025_limit_b.csv, sigma_thermal.csv (the thermal-relic line, plus
McDaniel2023_limit.csv for the cell-190 ratio diagnostic only).

`--fermi-version` defaults to 'legacy', matching fermi_reweighting.py's and
plot_fermi_reweighting.py's own CLI-default convention (a behavioural choice
this migration does not silently flip); scripts/plot_fermi_limit_comparison.sh
passes 'update' explicitly, matching cell 173's FERMI_RESULTS_VERSION =
'update' toggle that was active when cell 191 last ran (the drafts cite
Circiello:2026inp).

Output: plots/fermi/fermi_limit_comparison.pdf, with a
provenance.figure_manifest sidecar naming every per-dwarf/per-weight .npz
plus the literature CSVs actually read into the saved figure.

Usage:
    conda activate J_calc
    python plot_fermi_limit_comparison.py [--fermi-version update]
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d

from plot_style import *  # noqa: F401,F403 -- rcParams side effects
import config
import provenance
from fermi_funcs import get_UL
from fermi_plot_helpers import HandlerRectangle, SplitPatchHandler
from plot_fermi_reweighting import (
    load_summed, draw_thermal_relic, find_thermal_crossing_report, LITERATURE_DIR,
)

# Cell 191's own label/linestyle text for the two `summed_list` indices it
# actually plots (WEIGHT_LIST[1] = 'mhalf', WEIGHT_LIST[8] =
# 'Jeans_satgen_shmr_fattahi18'). Cell 191's label text at these indices
# differs from cell 183's LABELS_183 in plot_fermi_reweighting.py ("This
# work - ..." vs the shorter "..." text used there for a different figure),
# so kept local rather than imported.
LABELS = {
    1: r'This work - $M_{1/2}$ inference',
    8: 'This work - Jeans analyses,\n' + r'\texttt{SatGen}-informed prior',
}
LINESTYLES = {1: '-', 8: ':'}


def _report_circiello_mcdaniel_ratio():
    """Cell 190's standalone ratio diagnostic (Circiello+26 / McDaniel+23),
    printed only -- feeds no array into the saved figure (see module
    docstring)."""
    circiello_limit = np.genfromtxt(LITERATURE_DIR / 'Circiello2026_limit.csv', delimiter=',')
    mcdaniel_limit = np.genfromtxt(LITERATURE_DIR / 'McDaniel2023_limit.csv', delimiter=',')

    x_min = max(circiello_limit[:, 0].min(), mcdaniel_limit[:, 0].min())
    x_max = min(circiello_limit[:, 0].max(), mcdaniel_limit[:, 0].max())
    x_common = np.logspace(np.log10(x_min), np.log10(x_max), 500)

    f_circ = interp1d(circiello_limit[:, 0], circiello_limit[:, 1], kind='linear',
                      bounds_error=False, fill_value=np.nan)
    f_mcda = interp1d(mcdaniel_limit[:, 0], mcdaniel_limit[:, 1], kind='linear',
                      bounds_error=False, fill_value=np.nan)

    y_circ = f_circ(x_common)
    y_mcda = f_mcda(x_common)
    ratio = y_circ / y_mcda

    print(np.min(ratio), np.max(ratio))
    return x_common, ratio


def build_figure(fermi_version):
    """Cell 191's five curves plus the thermal-relic line."""
    summed_by_index, mass_vec, sigmav_vec, used_files = load_summed({1, 8}, fermi_version)

    fig, axs = plt.subplots(figsize=(6, 6))

    circiello_limit = np.genfromtxt(LITERATURE_DIR / 'Circiello2026_limit.csv', delimiter=',')
    h1 = axs.plot(circiello_limit[:, 0], circiello_limit[:, 1], color='gray', lw=2, zorder=1000, ls='-')[0]

    gs_limit = np.genfromtxt(LITERATURE_DIR / 'abdollahi2025_limit_gs.csv', delimiter=',')
    h2 = axs.plot(gs_limit[:, 0], gs_limit[:, 1], color='gray', lw=2, zorder=1000, ls='--')[0]

    b_limit = np.genfromtxt(LITERATURE_DIR / 'abdollahi2025_limit_b.csv', delimiter=',')
    h3 = axs.plot(b_limit[:, 0], b_limit[:, 1], color='gray', lw=2, zorder=1000, ls=':')[0]

    sigma_thermal = draw_thermal_relic(axs)

    second_handles = [h1, h2, h3]
    second_labels = [
        'Circiello+26 - benchmark sample',
        'Abdollahi+26 - GS+14 $J$-factors',
        'Abdollahi+26 - Bonnivard+15 $J$-factors',
    ]

    for i in (1, 8):
        xs, ulsamp, _ = get_UL(summed_by_index[i].T, mass_vec, sigmav_vec)
        line = axs.plot(xs, ulsamp, c='k', label=LABELS[i], lw=2, ls=LINESTYLES[i], zorder=1000)[0]
        second_handles.append(line)
        second_labels.append(LABELS[i])
        find_thermal_crossing_report(LABELS[i], mass_vec, ulsamp, sigma_thermal)

    axs.text(
        0.03, 0.03, r'$\chi\chi\to b \bar{b}$',
        transform=axs.transAxes, fontsize=15, va='bottom', ha='left', ma='center')

    axs.set_xlabel(r'$M_{\chi}$ [GeV]', fontsize=15)
    axs.set_ylabel(r'$\left<\sigma v\right>$ [cm$^{3}$ s$^{-1}$]', fontsize=15)

    axs.set_xlim(7, 1e4)
    axs.set_ylim(1e-28, 1e-22)

    axs.tick_params(axis='x', labelsize=15)
    axs.tick_params(axis='y', labelsize=15)

    axs.set_xscale('log')
    axs.set_yscale('log')

    axs.legend(second_handles, second_labels, loc='upper left', fontsize=13, frameon=False,
              handler_map={tuple: SplitPatchHandler(), Rectangle: HandlerRectangle()})

    plt.tight_layout()

    used_literature = [LITERATURE_DIR / f for f in
                       ('Circiello2026_limit.csv', 'abdollahi2025_limit_gs.csv',
                        'abdollahi2025_limit_b.csv', 'sigma_thermal.csv')]
    return fig, used_files + used_literature


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fermi-version', default='legacy', choices=('legacy', 'update'))
    args = parser.parse_args(argv)

    _report_circiello_mcdaniel_ratio()

    fig, used_inputs = build_figure(args.fermi_version)

    out_path = config.PLOTS_DIR / 'fermi' / 'fermi_limit_comparison.pdf'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    provenance.figure_manifest(
        out_path, 'python/plot_fermi_limit_comparison.py', inputs=used_inputs,
        fermi_version=args.fermi_version, weight_indices=[1, 8])
    print('wrote', out_path)


if __name__ == '__main__':
    main()
