"""sigma_los-vs-r_half detectability-contour figure with measured MW satellite
dispersions and the Ursa Major III upper limit overplotted.

Migrated from ../SatGen_Dwarf/jupyter/PaperPlots.ipynb cells 73 (prep) -> 74
(savefig), section "$\\langle \\sigma_{\\rm los}\\rangle - r_{1/2}$" (cell 72).
Path/import/provenance edits only -- the LMC-analog host mask, the per-host
four-most-massive-satellite removal, the sigma_los grid, N_greater/
has_greater counting, and the p(N>=1) contour levels [0.01, 0.05, 0.16] are
all character-identical to upstream.

`quantile` is imported from weighted_stats (see that module's docstring for
why it is not re-pasted) to compute CL_satgen_5/16/50/90, matching cell 73's
own computation -- even though the savefig cell (74) never plots them: every
`contour_CL = axs.contour(...)` line that would consume CL_satgen_* is
commented out there. They are computed anyway here, for parity with cell 73
as the figure's prep step, but the figure itself is built entirely from
`frac_satgen` (has_greater-based, no quantile call in its critical path) plus
the two observational error-bar series. See the module's report for what
`quantile`'s small-N clip regime looks like on the ~45-host mask this cell
uses.

Dropped from upstream: the matplotlib preamble (`from plot_style import *`
supplies it) and the large block of alternative/commented-out plot elements
in cell 74 (the fraction-vs-N_hosts colormesh, the per-selected-dwarf axvline
loop, the colorbar inset, and five alternative `plt.savefig(...)` calls for
other basenames this repository never asked for) -- none of it executes
upstream either. Also dropped: cell 73's `SkyCoord`/`gc`/`distance` block --
computed upstream but never read afterward (only `gc_distance`, from
`np.linalg.norm`, feeds the LMC-analog mask); and the `if not idx % 10:
print(idx)` progress print in the per-radius loop.

config constants reused: config.PAPER_MASSPROFILE_DIR, config.PLOTS_DIR. No
new config constant was needed.
"""
import sys

import numpy as np
from astropy import units as u
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerErrorbar

import config
import provenance
import Jdata as obs
from halo_weights import get_h5
from dwarf_categories import large_dwarf
from weighted_stats import N_greater, has_greater, quantile
from plot_style import *  # noqa: F401,F403 -- rcParams side effect, plt, colors

VERSION = 'Diemer'

# Cell 74's own three-colour list, distinct from plot_style's Okabe-Ito
# `colors` -- used only for the two observational error-bar series.
EXTRA_COLORS = [
    "#882255",  # wine
    "#44AA99",  # teal
    "#DDCC77",  # sand
]

logdispersion_grid = np.linspace(-1, 2, num=101)
obs_grid = logdispersion_grid

massprofile_path = config.PAPER_MASSPROFILE_DIR / VERSION / 'massProfile.npz'
h5_path = config.h5_path(VERSION)

provenance.assert_single_version([h5_path, massprofile_path], expected=VERSION)


def build():
    arr = np.load(massprofile_path)
    vcircProfile = arr['vcircProfile']
    r_grid = arr['r_grid']

    with get_h5(VERSION) as sats:
        logMpeak = np.log10(sats['Green_params'][()][:, 0])
        mwID = sats['mwID'][()]
        position = sats['position'][()]

        gc_distance = np.linalg.norm(position, axis=1)

        # cut on D_gal < 100 (comment inherited from upstream; the value used
        # is 100, matching lmc_dist_cut below -- see cell 73)
        lmc_dist_cut = gc_distance < 100
        lmc_mass_cut = logMpeak > 11
        lmc_mwIDs = mwID[lmc_dist_cut & lmc_mass_cut]
        N_mw = len(np.unique(lmc_mwIDs))
        mw_hosts_lmc = np.isin(mwID, lmc_mwIDs)

        mask = mw_hosts_lmc

        # Prepare mask: start with all True
        remove_lmc_mask = np.ones(len(logMpeak), dtype=bool)

        unique_mwIDs = np.unique(mwID)
        for mw in unique_mwIDs:
            mw_inds = np.where(mwID == mw)[0]
            four_most_massive_indices = mw_inds[np.argsort(logMpeak[mw_inds])[-4:]]
            remove_lmc_mask[four_most_massive_indices] = False

        satellite_mask = np.logical_and(mask, remove_lmc_mask)

        frac_satgen = []
        CL_satgen_5 = []
        CL_satgen_16 = []
        CL_satgen_50 = []
        CL_satgen_90 = []
        for idx in range(vcircProfile.shape[1]):
            logdispersion = np.log10(vcircProfile[:, idx] / np.sqrt(3))
            logobs = logdispersion

            masked_obs = logobs[satellite_mask]

            edges = np.concatenate(
                [[-1]] + list(np.where(np.diff(mwID[satellite_mask]))) +
                [[np.count_nonzero(satellite_mask) - 1]]) + 1
            satgen_N_greater = np.array(
                [N_greater(masked_obs[lo:hi], obs_grid)[1] for lo, hi in zip(edges[:-1], edges[1:])])
            CL_satgen_5.append([quantile(satgen_N_greater[:, i], quantiles=0.95) for i in range(len(obs_grid))])
            CL_satgen_16.append([quantile(satgen_N_greater[:, i], quantiles=0.84) for i in range(len(obs_grid))])
            CL_satgen_50.append([quantile(satgen_N_greater[:, i], quantiles=0.5) for i in range(len(obs_grid))])
            CL_satgen_90.append([quantile(satgen_N_greater[:, i], quantiles=0.1) for i in range(len(obs_grid))])
            satgen_has_greater = np.array(
                [has_greater(masked_obs[lo:hi], obs_grid)[1] for lo, hi in zip(edges[:-1], edges[1:])])
            frac_satgen.append(np.sum(satgen_has_greater, axis=0) / N_mw)

    frac_satgen = np.array(frac_satgen)
    CL_satgen_5 = np.array(CL_satgen_5)
    CL_satgen_16 = np.array(CL_satgen_16)
    CL_satgen_50 = np.array(CL_satgen_50)
    CL_satgen_90 = np.array(CL_satgen_90)

    return r_grid, frac_satgen, CL_satgen_5, CL_satgen_16, CL_satgen_50, CL_satgen_90


def main():
    r_grid, frac_satgen, CL_satgen_5, CL_satgen_16, CL_satgen_50, CL_satgen_90 = build()

    fig, axs = plt.subplots(figsize=(6, 6))

    proxy_lines = []

    xedges = r_grid
    yedges = 10 ** obs_grid
    X, Y = np.meshgrid(xedges, yedges)

    axs.contour(X, Y, frac_satgen.T, levels=[0.01, 0.05, 0.16],
               colors=[colors[0], colors[0], colors[0]], linestyles=['-.', ':', '--'])

    for name in obs.dwarf_names:
        if np.isin(name, large_dwarf):
            continue
        dwarf = obs.Dwarf(name)

        x = dwarf.rhalf.to(u.kpc).value
        xerr = np.transpose([dwarf.rhalf_err.to(u.kpc).value])

        y = dwarf.dispersion.value
        yerr = np.transpose([dwarf.dispersion_err.value])

        eb1 = axs.errorbar(x, y, xerr=xerr, yerr=yerr, c=EXTRA_COLORS[0], fmt='o', zorder=1000,
                           linewidth=1.5, capsize=2, label='MW satellites - measured')

    test_dwarf = obs.Dwarf('Ursa Major III')
    uma3_dispersion_ul = 2.3
    uma3_rhalf = test_dwarf.rhalf.to(u.kpc).value
    uma3_rhalf_err = np.transpose([test_dwarf.rhalf_err.to(u.kpc).value])
    eb2 = axs.errorbar(uma3_rhalf, uma3_dispersion_ul, xerr=uma3_rhalf_err, yerr=0.3, uplims=True,
                       c=EXTRA_COLORS[1], fmt='o', zorder=900, linewidth=1.5, capsize=2, label='UMa3/U1')

    proxy_lines.append(mlines.Line2D([], [], color=colors[0], ls='-.', linewidth=1.5,
                                     label=r'$p(N_> \geq 1) = 0.01$'))
    proxy_lines.append(mlines.Line2D([], [], color=colors[0], ls=':', linewidth=1.5,
                                     label=r'$p(N_> \geq 1) = 0.05$'))
    proxy_lines.append(mlines.Line2D([], [], color=colors[0], ls='--', linewidth=1.5,
                                     label=r'$p(N_> \geq 1) = 0.16$'))
    proxy_lines.append(eb1)
    proxy_lines.append(eb2)

    axs.set_xlabel(r'$r$ [kpc]', fontsize=15)
    axs.set_ylabel(r'$\sigma_{\rm LOS}$ [km s$^{-1}$]', fontsize=15)

    legend0 = axs.legend(handles=proxy_lines, fontsize=15, loc='upper left', frameon=False,
                         handler_map={type(eb1): HandlerErrorbar()})
    axs.add_artist(legend0)

    axs.text(
        0.97, 0.03, r'$M_{\rm peak}>10^8$ M$_{\odot}$',
        transform=axs.transAxes,
        fontsize=15,
        va='bottom', ha='right',
    )

    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.tick_params(axis='x', labelsize=15)
    axs.tick_params(axis='y', labelsize=15)
    axs.set_ylim(5e-1, 2e2)

    out_dir = config.PLOTS_DIR / 'dispersion'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'dispersion_limit_contours.pdf'
    fig.savefig(out_path)
    plt.close(fig)

    inputs = [h5_path, massprofile_path]
    provenance.figure_manifest(out_path, 'python/plot_dispersion_limit.py', inputs,
                               version=VERSION, argv=sys.argv[1:])
    print('wrote', out_path)


if __name__ == '__main__':
    main()
