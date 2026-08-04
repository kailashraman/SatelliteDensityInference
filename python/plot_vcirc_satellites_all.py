"""V_circ(r) satellite comparison figures: measured MW satellite V_circ vs
r_half, overplotted with the SatGen "with LMC analog" median V_circ(r)
profile and, for two variants, the SHMR-inferred (Fattahi+18/Kim+24) V_circ
at r_half.

Migrated from ../SatGen_Dwarf/python/plot_vcirc_satellites_all.py (fattahi18/
kim24) and ../SatGen_Dwarf/jupyter/PaperPlots.ipynb cell 133 (intro), which
produce:

    satellites_vcirc_all_fattahi18.pdf
    satellites_vcirc_all_kim24.pdf
    satellites_vcirc_intro.pdf

The upstream script already carried a `make_plot(which)` toggle for the first
two; `intro` is added as a third branch on that same toggle (CLAUDE.md:
"extend, don't clone") rather than a second module. `intro`'s series
(measured-only errorbars, SatGen median with 68%/95% bands, a reference NFW
halo curve) differ from fattahi18/kim24's (measured + one SHMR-inferred
errorbar series, unbanded median) but the two share the same LMC-analog mask,
r_grid/vcirc_quantiles setup, axis config and save/provenance plumbing, so a
third `which` branch reuses all of that rather than re-deriving it.

Cell 133's *comment state* is not the figure spec here -- most of its body
(the Fattahi18/Kim24 errorbar block, the fill_between calls, one of its two
NFW concentration-model blocks) is commented out in the notebook as it exists
today, but the on-disk rendered
../SatGen_Dwarf/plots/paper_plots_take_3/satellites_vcirc_intro.pdf shows a
single gray 68% band around the SatGen median (quantiles_arr indices 2/4,
i.e. the 16th/84th percentile columns), a single green NFW curve+band, and no
SHMR-inferred points -- i.e. an EARLIER run of this cell, before the
Fattahi18/Kim24 experimentation visible in the current source was added
(per CLAUDE.md/memory: comment state is post-run residue, the rendered PDF is
the spec). Confirmed by flat-fill RGB decomposition of the rendered PDF at
200 dpi: it shows a single gray level (204,204,204), consistent with one
alpha-0.2 fill over white, not two overlapping fills (an earlier version of
this module additionally drew a second, uncontained alpha-0.1 fill at
quantiles_arr indices 1/5 that the published PDF does not have -- fixed,
see build_intro). `make_plot('intro')` reproduces that rendered content: the
SatGen median line + one 68% band (quantiles_arr indices 2/4), the
measured-only errorbar series, and one NFW curve (the second, unlabeled
"Ishiyama" block in the notebook cell -- its legend text has no model
qualifier, matching the rendered legend, whereas the other, commented-out
block's label literally read $10^{8.5}M_\\odot$ NFW halo ..., Ishiyama" for a
c_med_diemer curve, a notebook mislabeling not reproduced here since that
block never rendered). The commented-out 2-sigma/1-sigma NFW bands and the
Diemer-concentration curve are NOT reproduced, matching the rendered PDF.

`which='intro'`'s NFW reference curve is a fixed 10^8.5 Msun/z=2 halo with the
Ishiyama+21 c(M) relation (0.16 dex scatter band), computed directly via
colossus -- not through python/nfw.py, which has no concentration-mass
relation or circularVelocity helper (it only holds the rho150<->NFW-parameter
inversion). Jdata (imported as obs, itself imported for dwarf_names/Dwarf)
registers colossus's 'SatGen' cosmology as current at import time, so no
separate `co.addCosmology`/`co.setCosmology` call is needed here; `obs.col_co`
supplies `.h` for the Mpc/h <-> Mpc-physical conversions upstream used.

Path/import/provenance edits only -- the LMC-analog + top-4-per-host mass
cut, the median/quantile computation, the per-dwarf errorbar series, and the
axis/legend configuration are otherwise character-identical to upstream.
Dropped: the re-pasted rcParams preamble (now `from plot_style import *`) and
the `os.environ["PATH"] +=` texlive mutation it duplicated (plot_style
already does this via config.TEXLIVE_BIN).

Two upstream absolute paths become config-derived:
    root/results/paper_massProfile/<version>/massProfile.npz -> config.PAPER_MASSPROFILE_DIR
    root/results/vcirc_scatter_300pc/<version>/<SHMR>/scatter_300pc.txt -> config.VCIRC_SCATTER_DIR
    root/results/paper_quantiles/galactocentric/<version>/<dwarf>.npz -> config.PAPER_QUANTILES_DIR
    root/plots/paper_plots_take_3/ -> config.PLOTS_DIR / 'vcirc_satellites'
config.PAPER_MASSPROFILE_DIR, config.VCIRC_SCATTER_DIR and
config.PAPER_QUANTILES_DIR already existed; no new config constant was added.

`large_dwarf` is imported from dwarf_categories rather than re-declared -- it
is the same literal array upstream defined locally in this module.

results/vcirc_scatter_300pc/ was REGENERATED in this repository, not
migrated: its V_median values run ~16.5% higher than the migrated-tree values
upstream's rendered fattahi18/kim24 PDFs used, because of a corrected
infall->z=0 stellar-mass mapping (see docs/migration-notes.md, "The
vcirc_scatter_300pc tree was regenerated"). So the fattahi18/kim24 output
here is not expected to match upstream's rendered PDF pixel-for-pixel; it is
verified against upstream's own code run on this repository's regenerated
inputs instead.

Every figure is stamped by provenance.figure_manifest, next to the PDF it
describes.

Usage:
    conda activate J_calc
    python plot_vcirc_satellites_all.py [figure=all] [version=Diemer]
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerErrorbar
import astropy.units as u
from colossus.halo import concentration, profile_nfw

from plot_style import *  # noqa: F401,F403 -- brings plt, mpl, colors, config
import halo_weights
import Jdata as obs
import provenance
from dwarf_categories import large_dwarf
from weighted_stats import quantile

FIGURES = ['fattahi18', 'kim24', 'intro']

extra_colors = ["#882255", "#44AA99", "#DDCC77"]
cm_color = '#6CBE45'  # apple green, cell 133's NFW-reference curve color

BASENAME = {
    'fattahi18': 'satellites_vcirc_all_fattahi18.pdf',
    'kim24': 'satellites_vcirc_all_kim24.pdf',
    'intro': 'satellites_vcirc_intro.pdf',
}


def out_path(which):
    path = config.PLOTS_DIR / 'vcirc_satellites' / BASENAME[which]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load(version):
    """Load the h5, massProfile.npz and vcirc_scatter_300pc.txt inputs
    (~1.4 GB total), apply the LMC-analog + top-4-per-host mass cut, and
    compute the shared vcirc_quantiles grid the three figures plot from.

    Called from main(), not at module import time -- importing this module
    for BASENAME/out_path/FIGURES no longer pays this load cost or requires
    sys.argv to be set.
    """
    global h5_path, massprofile_path, r_grid, vcirc_quantiles
    global scatter_path_f, scatter_path_k, vcirc_sigma_dex_f, vcirc_sigma_dex_k

    # ---- load SatGen samples, apply LMC-analog + top-4-per-host mass cut --
    h5_path = config.h5_path(version)
    provenance.assert_single_version([h5_path], expected=version)
    with halo_weights.get_h5(version) as sats:
        mpeak = sats['Green_params'][()][:, 0]
        mwID = sats['mwID'][()]
        gc_distance = np.linalg.norm(sats['position'], axis=1)

    lmc_dist_cut = gc_distance < 100
    lmc_mass_cut = mpeak > 1e11
    lmc_mwIDs = mwID[lmc_dist_cut & lmc_mass_cut]
    mw_hosts_lmc = np.isin(mwID, lmc_mwIDs)
    mask = mw_hosts_lmc

    remove_lmc_mask = np.ones(len(mpeak), dtype=bool)
    unique_mwIDs = np.unique(mwID)
    for mw in unique_mwIDs:
        mw_inds = np.where(mwID == mw)[0]
        four_most_massive_indices = mw_inds[np.argsort(mpeak[mw_inds])[-4:]]
        remove_lmc_mask[four_most_massive_indices] = False
    mask = np.logical_and(mask, remove_lmc_mask)

    massprofile_path = config.PAPER_MASSPROFILE_DIR / version / 'massProfile.npz'
    provenance.assert_single_version([massprofile_path], expected=version)
    arr = np.load(massprofile_path)
    vcircProfile = arr['vcircProfile'][mask]
    r_grid = arr['r_grid']

    quantiles_arr = np.array([0.0025, 0.025, 0.16, 0.5, 0.84, 0.975, 0.9975])
    vcirc_quantiles = np.zeros((len(r_grid), len(quantiles_arr)))
    for j in range(len(r_grid)):
        vcirc_quantiles[j] = quantile(vcircProfile[:, j], quantiles=quantiles_arr)

    # ---- observational scatter in median vcirc from SHMR M_star scatter ---
    scatter_path_f = config.VCIRC_SCATTER_DIR / version / 'Fattahi18' / 'scatter_300pc.txt'
    scatter_path_k = config.VCIRC_SCATTER_DIR / version / 'Kim24' / 'scatter_300pc.txt'
    provenance.assert_single_version([scatter_path_f, scatter_path_k], expected=version)
    scatter_df_f = pd.read_csv(scatter_path_f, comment='#')
    vcirc_sigma_dex_f = dict(zip(scatter_df_f['dwarf_name'], scatter_df_f['sigma_dex']))
    scatter_df_k = pd.read_csv(scatter_path_k, comment='#')
    vcirc_sigma_dex_k = dict(zip(scatter_df_k['dwarf_name'], scatter_df_k['sigma_dex']))


def build_inference(which):
    """which: 'fattahi18' or 'kim24'"""
    if which == 'fattahi18':
        idx, color, label = 4, colors[2], 'MW satellites - Fattahi+18'
        meas_scatter = vcirc_sigma_dex_f
    else:
        idx, color, label = 10, colors[3], 'MW satellites - Kim+24'
        meas_scatter = vcirc_sigma_dex_k

    fig, axs = plt.subplots(figsize=(6, 6))

    proxy_lines = []
    eb1 = eb2 = None
    dwarf_files = []
    for name in obs.dwarf_names:
        if np.isin(name, large_dwarf):
            continue

        dwarf = obs.Dwarf(name)
        x = dwarf.rhalf.to(u.kpc).value
        xerr = np.transpose([dwarf.rhalf_err.to(u.kpc).value])
        y = dwarf.Vcirc.value
        yerr = np.transpose([dwarf.Vcirc_err.value])

        eb1 = axs.errorbar(x, y, xerr=xerr, yerr=yerr, c=extra_colors[0], zorder=1000,
                            fmt='o', linewidth=1, capsize=2, label='MW satellites - measured')

        quantiles_path = config.PAPER_QUANTILES_DIR / 'galactocentric' / version / f'{name}.npz'
        dwarf_files.append(quantiles_path)
        quantiles_file = np.load(quantiles_path)
        # rows are the 16/50/84th percentiles; only the median is plotted
        dispersion_50 = quantiles_file['dispersion_quantiles'][1]

        # dispersion_* are 1D velocity dispersions; V_circ = sqrt(3) * sigma
        # (isotropic Jeans). Applies to every estimator, cf. notebook cell 123.
        y_est = np.sqrt(3) * dispersion_50[idx]

        # NOTE: this scatter is the uncertainty on the weighted-median V_circ
        # curve evaluated at a FIXED 300 pc, over the full SatGen sample (no
        # LMC-analog/top-4 mask) -- not the uncertainty on y_est at this
        # dwarf's r_half. See vcirc_scatter_300pc.py.
        est_scatter = meas_scatter[name]
        y_est_hi = y_est * 10 ** est_scatter
        y_est_lo = y_est / 10 ** est_scatter
        eb2 = axs.errorbar(x, y_est, xerr=xerr, yerr=[[y_est - y_est_lo], [y_est_hi - y_est]],
                            c=color, zorder=899, fmt='o', linewidth=1, capsize=2, label=label)

    provenance.assert_single_version(dwarf_files, expected=version)

    axs.plot(r_grid, vcirc_quantiles[:, 3], c=colors[0])

    proxy_lines.append(mlines.Line2D([], [], color=colors[0], linewidth=1.5,
                                      label=r'\texttt{SatGen} with LMC analog'))
    proxy_lines.append(eb1)
    proxy_lines.append(eb2)

    axs.text(0.97, 0.03, r'$M_{\rm peak}>10^8$ M$_{\odot}$', transform=axs.transAxes,
              fontsize=15, va='bottom', ha='right')

    legend0 = axs.legend(handles=proxy_lines, fontsize=15, loc='upper left', frameon=False,
                          handler_map={type(eb1): HandlerErrorbar()})
    axs.add_artist(legend0)

    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.tick_params(axis='x', labelsize=15)
    axs.tick_params(axis='y', labelsize=15)
    axs.set_xlim(1e-2, 1e1)
    axs.set_ylim(1e0, 9e1)
    axs.set_xlabel(r'$r$ [kpc]', fontsize=15)
    axs.set_ylabel(r'$V_{\rm circ}(r)$ [km s$^{-1}$]', fontsize=15)

    path = out_path(which)
    fig.savefig(path)
    plt.close(fig)
    print('saved:', path)

    inputs = [h5_path, massprofile_path, scatter_path_f, scatter_path_k] + dwarf_files
    # NOTE: not passing figure=which -- figure_manifest already sets its own
    # `figure` kwarg (str(path)) when it calls provenance.stamp(); stamp() has
    # no named `figure` parameter, so a caller-supplied figure=... lands in
    # its **extra and collides with figure_manifest's own figure=str(path),
    # raising "multiple values for keyword argument 'figure'". Pre-existing
    # in provenance.figure_manifest's contract -- flagged, not fixed here;
    # out of scope for this module.
    provenance.figure_manifest(path, 'python/plot_vcirc_satellites_all.py', inputs,
                               version=version, argv=sys.argv[1:], which=which)


def build_intro():
    """Measured MW satellites vs. the SatGen median+bands vs. a reference
    10^8.5 Msun NFW halo at z=2 (Ishiyama+21 c(M), 0.16 dex scatter band).
    See module docstring for why this reproduces the RENDERED
    satellites_vcirc_intro.pdf rather than the notebook cell's current
    (post-run-edited) comment state.
    """
    fig, axs = plt.subplots(figsize=(6, 6))

    proxy_lines = []
    eb1 = None
    for name in obs.dwarf_names:
        if np.isin(name, large_dwarf):
            continue

        dwarf = obs.Dwarf(name)
        x = dwarf.rhalf.to(u.kpc).value
        xerr = np.transpose([dwarf.rhalf_err.to(u.kpc).value])
        y = dwarf.Vcirc.value
        yerr = np.transpose([dwarf.Vcirc_err.value])

        eb1 = axs.errorbar(x, y, xerr=xerr, yerr=yerr, c=extra_colors[0], zorder=1000,
                            fmt='o', linewidth=1, capsize=2, label='MW satellites - measured')

    axs.plot(r_grid, vcirc_quantiles[:, 3], c=colors[0])
    axs.fill_between(r_grid, vcirc_quantiles[:, 2], vcirc_quantiles[:, 4], color=colors[0], alpha=0.2)

    proxy_lines.append(mlines.Line2D([], [], color=colors[0], linewidth=1.5,
                                      label=r'\texttt{SatGen} with LMC analog'))
    proxy_lines.append(eb1)

    # ---- reference NFW halo: 10^8.5 Msun (physical) at z=2, Ishiyama+21 c(M) ----
    M_phys = 10 ** 8.5  # Msun (physical)
    M = M_phys * obs.col_co.h  # Msun/h (colossus internal units)
    mdef = 'vir'
    sigma_logc = 0.16  # dex scatter in c-M relation
    r_phys_kpc = np.logspace(-2, 1, 200)
    z = 2

    c_med_ishiyama = concentration.concentration(M, mdef, z, model='ishiyama21')
    r_phys_kpch = r_phys_kpc * obs.col_co.h

    vcirc = {}
    for nsig in [0, -1, 1]:
        c = c_med_ishiyama * 10 ** (nsig * sigma_logc)
        p = profile_nfw.NFWProfile(M=M, c=c, z=z, mdef=mdef)
        vcirc[nsig] = p.circularVelocity(r_phys_kpch)

    axs.fill_between(r_phys_kpc, vcirc[-1], vcirc[1], color=cm_color, alpha=0.2, zorder=1)
    line, = axs.plot(r_phys_kpc, vcirc[0], color=cm_color, ls='-', lw=2,
                      label=r'$10^{8.5} \, {\rm M}_\odot$ NFW halo at $z=2$')
    proxy_lines.append(line)

    axs.text(0.97, 0.03, r'$M_{\rm peak}>10^8$ M$_{\odot}$', transform=axs.transAxes,
              fontsize=15, va='bottom', ha='right')

    legend0 = axs.legend(handles=proxy_lines, fontsize=15, loc='upper left', frameon=False,
                          handler_map={type(eb1): HandlerErrorbar()})
    axs.add_artist(legend0)

    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.tick_params(axis='x', labelsize=15)
    axs.tick_params(axis='y', labelsize=15)
    axs.set_xlim(1e-2, 1e1)
    axs.set_ylim(1e0, 9e1)
    axs.set_xlabel(r'$r$ [kpc]', fontsize=15)
    axs.set_ylabel(r'$V_{\rm circ}(r)$ [km s$^{-1}$]', fontsize=15)

    path = out_path('intro')
    fig.savefig(path)
    plt.close(fig)
    print('saved:', path)

    inputs = [h5_path, massprofile_path]
    # See build_inference's NOTE above -- figure=... is not passed for the
    # same reason.
    provenance.figure_manifest(path, 'python/plot_vcirc_satellites_all.py', inputs,
                               version=version, argv=sys.argv[1:], which='intro')


BUILDERS = {
    'fattahi18': lambda: build_inference('fattahi18'),
    'kim24': lambda: build_inference('kim24'),
    'intro': build_intro,
}


def main():
    global figure, version
    figure = sys.argv[1] if len(sys.argv) > 1 else 'all'
    version = sys.argv[2] if len(sys.argv) > 2 else 'Diemer'
    if figure not in FIGURES + ['all']:
        raise SystemExit(f"figure must be one of {FIGURES + ['all']}, got {figure!r}")

    _load(version)

    to_run = FIGURES if figure == 'all' else [figure]
    for name in to_run:
        BUILDERS[name]()


if __name__ == '__main__':
    main()
