"""Four draft scaling-relation figures from
../SatGen_Dwarf/jupyter/PaperPlots.ipynb:

    tidal_stripping.pdf        cells 161, 162 -> 163
    L_vs_Mpeak.pdf              cells 152-155 -> 156
    L_vs_rho150.pdf              cells 152-155 -> 156 (see "L_vs_rho150" below)
    lvdb_kdsa_rho150.pdf          cells 148, 149 -> 150

Grouped into one module (rather than one per family) because all four are
small, read-only consumers of the same already-migrated products
(paper_quantiles galactocentric .npz's, ResampleMstar, Jdata.Dwarf) and share
plot_style/dwarf_categories plumbing -- the same organisation
plot_number_functions.py uses for its six figures. Path/import/provenance
edits only; the binning, mass-loss definitions, f_disrupted computation, and
luminosity conversions below are character-identical to the cells, except
where a dropped dead-code block is called out explicitly.

Catalogs used, as found in the source cells:
  - tidal_stripping.pdf: the Diemer galactocentric quantiles (Mloss_quantiles,
    per-dwarf, upper panel) PLUS the 'splashback' catalog read directly via
    ResampleMstar (lower panel / SatGen band) -- cell 162 sets
    `version = 'splashback'` explicitly. This is a genuine two-catalog
    figure, not a version-mixing bug: 'splashback' is the '_all' h5 that
    retains disrupted/masked-out halos (config.H5_REGISTRY), required
    because f_disrupted is identically ~0 in the (masked) Diemer catalog.
    provenance.assert_single_version is therefore called separately on each
    group below, never across both -- a combined call would raise on this
    figure's own by-design catalog split.
  - L_vs_Mpeak.pdf / L_vs_rho150.pdf: the Diemer galactocentric quantiles for
    the per-dwarf scatter points and the low-res SatGen band, PLUS the
    'mass_floor_7' catalog (a different h5, not an alias of Diemer) for the
    crosshatched high-res band -- cells 152/154 vs. 153/155. Another
    by-design two-catalog figure (the .tex calls the crosshatch "the
    Mpeak>10^7 SatGen runs"); assert_single_version is called per group, not
    across both.
  - lvdb_kdsa_rho150.pdf: the Diemer galactocentric quantiles (KDSA
    dispersion-conditioned weights -- Jdata.Dwarf's default use_kd=True) and
    the 'lvdb' galactocentric quantiles (LVDB dispersion-conditioned
    weights). 'lvdb' is a reweighting-only alias of Diemer
    (config.SIM_VERSION_ALIASES), so both groups resolve to the same
    underlying h5 -- assert_single_version(inputs) (no `expected`) is called
    once across both groups, which passes because they share one resolved
    sim_version and one aliased raw variant.

Dropped, and why:
  - Cell 148 (a print-only diagnostic comparing KDSA vs. LVDB dispersion
    significance per dwarf): no variable it defines is read by cell 149 or
    150 -- it never contributes to the plotted arrays. Dropped per the same
    "diagnostics only, no plotted output" reasoning plot_rmax_vmax.py's
    docstring gives for dropping upstream cells 62/65.
  - `from matplotlib.lines import Line2D` in cell 150: imported, never used
    (the plotted markers are all `ax.errorbar` calls). Dropped.
  - In cells 154/155 (`_shmr_fattahi_kim` below): only `Fattahi18` and
    `Kim24` feed a `logobs_quantiles` array cell 156 actually plots.
    Behroozi13, Moster13, RodriguezPuebla17, Moster18, Behroozi19, Munshi21,
    Danieli23_const and Danieli23_grow are all computed but never read again
    (Danieli23_grow feeds `logobs_quantiles_2`, which cell 156 never
    references either). The `vmax_infall`/`rmax_infall`/`rho150_infall` block
    and the distance-binned `logobs_quantiles_mloss` block at the end of
    cell 154 are similarly computed and never read by cell 156 (the latter
    duplicates cell 162's mass-loss-vs-distance logic, unused here). Traced
    through both cells' full control flow; dropping these changes no array
    cell 156 plots. `quantile()` (locally redefined in cells 154/155/162,
    byte-identical each time) is `weighted_stats.quantile` here, not
    re-pasted.

L_vs_rho150.pdf -- a scope decision, named explicitly rather than made
silently: cell 156 as executed in the notebook only ever produced
L_vs_Mpeak.pdf (`obs_50 = Mpeak_50` is active; `# obs_50 = rho150_50` and
`# plt.savefig(.../L_vs_rho150.pdf)` are commented out, and cells 154/155
likewise carry `logobs = logMpeak` active against a commented
`# logobs = logrho150`). The .tex text (fig:L_vs_Mpeak's caption, "the left
panel ... shows the Mhalf-inferred values of rho150 ... the right panel is
the same as the left, but for the predicted Mpeak") confirms both panels are
real, required figures built from the same pipeline with only the plotted
quantity swapped -- so this module makes that swap a `quantity` parameter
('Mpeak' or 'rho150') to `build_L_panel`, following the commented-out
alternative at each of the three sites it touches:
  - `logobs` in `_shmr_fattahi_kim`'s caller: logMpeak (Mpeak) or logrho150
    (rho150) -- per-halo arrays, both already loaded by the source cells.
  - `obs_50`/`obs_neg_err`/`obs_pos_err`: Mpeak_50/... or rho150_50/... --
    the per-dwarf scatter points, both already loaded by cell 152.
  - the y-axis label and the final `ax.set_ylim` upper bound: the cell 156
    active value (2e12) is paired with Mpeak; the literal commented
    alternative (2e10) sits directly above it and is used for rho150.
  One piece is NOT a straight toggle and is called out here rather than
  guessed at: cell 156's `axs.axhspan(axs.get_ylim()[0], 1e8, ...,
  hatch='//////', ...)`, shading the region below Mpeak = 1e8 Msun (the
  catalog's fiducial mass floor). That threshold is a resolution floor on
  Mpeak specifically, not a generic axis decoration -- shading y < 1e8 on a
  rho150 y-axis (units Msun/kpc^3, not Msun) would not represent the same
  thing, and no rho150-axis analog of this hatch exists anywhere else in the
  notebook (cell 150's rho150-vs-rho150 plot carries the same
  "Mpeak>10^8 Msun" TEXT annotation but no axhspan/hatch at all). This module
  therefore keeps the text annotation for both quantities but omits the
  axhspan/hatch for quantity='rho150'.

config constants reused: config.ADDITIONAL_DIR, config.PAPER_QUANTILES_DIR,
config.PLOTS_DIR, config.h5_path. No new config constant was added.
"""
import sys

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerBase
from astropy import units as u

import config
import provenance
import Jdata as obs
import halo_weights
from plot_style import *  # noqa: F401,F403 -- rcParams side effect, plt, mpl, colors
from dwarf_categories import (
    dwarf_names, classical_idcs, problem_idcs, ultrafaint_idcs)
from weighted_stats import quantile

FIGURES = ['tidal_stripping', 'L_vs_Mpeak', 'L_vs_rho150', 'lvdb_kdsa_rho150']

# Shared marker cycle -- byte-identical `markers = ['s', 'D', '^', 'o', 'd',
# 'p']` list pasted at the top of cells 150, 156 and 163.
MARKERS = ['s', 'D', '^', 'o', 'd', 'p']


def out_path(name):
    out_dir = config.PLOTS_DIR / 'scaling_relations'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f'{name}.pdf'


def _quantiles_path(version, name):
    return config.PAPER_QUANTILES_DIR / 'galactocentric' / version / f'{name}.npz'


# ---------------------------------------------------------------------------
# tidal_stripping.pdf (cells 161, 162 -> 163)
# ---------------------------------------------------------------------------
def build_tidal_stripping():
    quantiles_version = 'Diemer'
    splashback_version = 'splashback'
    mass_floor = 8

    # Cell 161, trimmed to what cell 163 actually plots: Mloss_quantiles
    # (column 0 = mhalf weighting, matching save_contours.py's weights
    # ordering) and each dwarf's galactocentric distance. cell 161 also loads
    # cV/rho150/J/Mvir/Mpeak/rmax quantiles, the Jeans results, and the Geha
    # arrays -- none of these feed tidal_stripping.pdf. Geha is additionally
    # NOT migrated (config.H5_REGISTRY['Geha'] = None); porting its try/except
    # block here would either always hit the except branch or raise.
    Mlossquantiles_list = []
    dgc_median_list = []
    dgc_err_list = []
    quantile_inputs = []
    for name in dwarf_names:
        qpath = _quantiles_path(quantiles_version, name)
        quantile_inputs.append(qpath)
        with np.load(qpath) as qfile:
            Mlossquantiles_list.append(qfile['Mloss_quantiles'])
        this_dwarf = obs.Dwarf(name)
        dgc_median_list.append(this_dwarf.gc_distance.to(u.kpc).value)
        dgc_err_list.append(this_dwarf.gc_distance_err.to(u.kpc).value)
    provenance.assert_single_version(quantile_inputs, expected=quantiles_version)

    Mlossquantiles = np.array(Mlossquantiles_list)
    dgc_median_list = np.array(dgc_median_list)
    dgc_err_list = np.array(dgc_err_list)

    Mloss_16, Mloss_50, Mloss_84 = np.array(
        [Mlossquantiles[:, 0], Mlossquantiles[:, 1], Mlossquantiles[:, 2]])
    Mloss_pos_err = Mloss_84 - Mloss_50
    Mloss_neg_err = Mloss_50 - Mloss_16

    # Cell 162: the splashback catalog directly (retains disrupted halos).
    splashback_h5 = config.h5_path(splashback_version)
    provenance.assert_single_version([splashback_h5], expected=splashback_version)
    with halo_weights.ResampleMstar(version=splashback_version, mass_floor=mass_floor) as weights:
        Mvir = weights.h5['virial_mass'][()]
        Mpeak = 10 ** weights.logMpeak
        gc_distance = weights.gc_distance.value

    logobs_all = np.log10(Mvir / Mpeak)
    mask = logobs_all > -2

    bins_distance = np.linspace(0, 3, 50)
    digitized_all = np.digitize(np.log10(gc_distance), bins_distance)

    logobs = logobs_all[mask]
    digitized = digitized_all[mask]

    logobs_binned = [logobs[digitized == i] for i in range(1, len(bins_distance) + 1)]
    logobs_quantiles_mloss = np.array(
        [quantile(logobs_binned[i], quantiles=np.array([0.16, 0.5, 0.84]))
         for i in range(len(bins_distance))])

    logobs_all_binned = [logobs_all[digitized_all == i] for i in range(1, len(bins_distance) + 1)]
    disrupted_fraction = np.array([
        len(np.where(arr < -2)[0]) / len(arr) if len(arr) > 0 else 0
        for arr in logobs_all_binned
    ])

    # Cell 163: two-panel figure.
    fig = plt.figure(figsize=(6, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    obs_50 = Mloss_50
    obs_neg_err = Mloss_neg_err
    obs_pos_err = Mloss_pos_err

    ax1.errorbar(
        dgc_median_list[problem_idcs], obs_50[:, 0][problem_idcs],
        xerr=np.transpose(dgc_err_list[problem_idcs]),
        yerr=(obs_neg_err[:, 0][problem_idcs], obs_pos_err[:, 0][problem_idcs]),
        fmt=MARKERS[3], markersize=5, capsize=2, mec=colors[6],
        c=colors[6], mfc='white', label=r'$N_{\rm stars}<10$')

    ax1.errorbar(
        dgc_median_list[ultrafaint_idcs], obs_50[:, 0][ultrafaint_idcs],
        xerr=np.transpose(dgc_err_list[ultrafaint_idcs]),
        yerr=(obs_neg_err[:, 0][ultrafaint_idcs], obs_pos_err[:, 0][ultrafaint_idcs]),
        fmt=MARKERS[0], markersize=5, capsize=2, mec=colors[5],
        c=colors[5], mfc='white', label=r'Ultra-faints')

    ax1.errorbar(
        dgc_median_list[classical_idcs], obs_50[:, 0][classical_idcs],
        xerr=np.transpose(dgc_err_list[classical_idcs]),
        yerr=(obs_neg_err[:, 0][classical_idcs], obs_pos_err[:, 0][classical_idcs]),
        fmt=MARKERS[1], markersize=5, capsize=2, mec=colors[7],
        c=colors[7], mfc='white', label=r'$M_{\star}>10^5$ M$_{\odot}$')

    ax1.plot(10 ** bins_distance, 10 ** logobs_quantiles_mloss[:, 1], color=colors[0], ls='--')
    ax1.fill_between(
        10 ** bins_distance, 10 ** logobs_quantiles_mloss[:, 0],
        10 ** logobs_quantiles_mloss[:, 2], color=colors[0], alpha=0.2,
        label=r'\texttt{SatGen}')

    crater2_idx = np.where(dwarf_names == 'Crater II')
    x = dgc_median_list[crater2_idx]
    y = obs_50[:, 0][crater2_idx]
    ax1.annotate(
        'Crater II', xy=(x, y), xytext=(x * 0.65, y / 1.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.",
                        facecolor='black', shrinkA=2, shrinkB=5),
        fontsize=10, zorder=1000)

    hercules_idx = np.where(dwarf_names == 'Hercules')
    x = dgc_median_list[hercules_idx]
    y = obs_50[:, 0][hercules_idx]
    ax1.annotate(
        'Hercules', xy=(x, y), xytext=(x * 1.2, y * 1.3),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.",
                        facecolor='black', shrinkA=2, shrinkB=5),
        fontsize=10, zorder=1000)

    bootes1_idx = np.where(dwarf_names == 'Bootes I')
    x = dgc_median_list[bootes1_idx]
    y = obs_50[:, 0][bootes1_idx]
    ax1.annotate(
        r'Bo\"otes I', xy=(x, y), xytext=(x * 1.2, y / 1.4),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.",
                        facecolor='black', shrinkA=2, shrinkB=5),
        fontsize=10, zorder=1000)

    ax1.text(
        0.03, 0.03, r'$M_{\rm peak}>10^8$ M$_{\odot}$',
        transform=ax1.transAxes, fontsize=15, va='bottom', ha='left')

    ax1.set_ylabel(r'$1 - M_{\rm loss} / M_{\rm peak}$', fontsize=15)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.set_xlim(2e1, 5e2)
    ax1.set_ylim(7e-3, 1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=15, frameon=False, loc='lower right')

    ax2.plot(10 ** bins_distance, disrupted_fraction, c='k')
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.set_xlabel(r'$d_{\rm GC}$ [kpc]', fontsize=15)
    ax2.set_ylabel(r'$f_{\rm disrupted}$', fontsize=15)

    plt.tight_layout()
    path = out_path('tidal_stripping')
    plt.savefig(path)
    plt.close(fig)

    inputs = [config.h5_path(quantiles_version), splashback_h5] + quantile_inputs
    provenance.figure_manifest(
        path, 'python/plot_scaling_relations.py', inputs, version=None,
        argv=sys.argv[1:], quantiles_version=quantiles_version,
        splashback_version=splashback_version,
        note=('deliberately spans two SatGen versions: the upper panel\'s '
              'Mloss inference is from the Diemer galactocentric quantiles, '
              'the lower panel and SatGen band are from the splashback '
              'catalog directly (retains disrupted halos, required for '
              'f_disrupted). assert_single_version is called separately on '
              'each group above, never across both.'))
    return path


# ---------------------------------------------------------------------------
# L_vs_Mpeak.pdf / L_vs_rho150.pdf (cells 152-155 -> 156)
# ---------------------------------------------------------------------------
def _shmr_fattahi_kim(version, mass_floor):
    """Cells 154/155, trimmed to the two SHMRs (Fattahi+18, Kim+24) that
    actually feed a `logobs_quantiles` array cell 156 plots -- see the module
    docstring for what is dropped and why.

    Draws from unseeded `np.random.normal`/`scipy.stats.norm.rvs` inside
    `halo_weights.logMstar_Fattahi18`/`logMstar_Kim24` (the scatter term) --
    NOT reproducible bit-for-bit run-to-run, matching upstream (the notebook
    never seeded these cells either). This is NOT the dominant source of
    L_vs_Mpeak.pdf/L_vs_rho150.pdf's deviation from the published figure,
    though: the unseeded scatter contributes ~0.02 dex to the plotted bands,
    while `evolved_Mstar`'s per-halo tidal correction (the corrected infall
    -> z=0 stellar-mass mapping, see docs/migration-notes.md) shifts the
    Fattahi+18 68% band in logM_peak 0.25-0.40 dex HIGHER than the published
    figure, which was made under the old constant +1.2 dex convention -- an
    order of magnitude larger. That is the measured cause of L_vs_Mpeak.pdf
    differing from published by 12.1% of pixels and L_vs_rho150.pdf by 6.8%.
    """
    logMpeak200 = np.log10(np.load(config.ADDITIONAL_DIR / f'{version}_Mvir200.npz')['Mvir200'])
    logrho150 = np.log10(np.load(config.ADDITIONAL_DIR / f'{version}_rho150.npz')['rho150'])
    with halo_weights.ResampleMstar(version=version, mass_floor=mass_floor) as weights:
        logMpeak = weights.logMpeak
        vpeak = weights.h5['tpeak_v_max'][()]
        Fattahi18 = weights.evolved_Mstar(halo_weights.logMstar_Fattahi18(vpeak))
        Kim24 = weights.evolved_Mstar(halo_weights.logMstar_Kim24(logMpeak200))
    return logMpeak, logrho150, Fattahi18, Kim24


def _logLv_binned_quantiles(logobs, logLv):
    bins = np.linspace(1, 10, 30)
    digitized = np.digitize(logLv, bins)
    logobs_binned = [logobs[digitized == i] for i in range(1, len(bins) + 1)]
    quantiles = np.array([
        quantile(logobs_binned[i], quantiles=np.array([0.16, 0.5, 0.84]))
        for i in range(len(bins))])
    return bins, quantiles


def color_with_alpha_on_white(color, alpha):
    """Blend color with white background to simulate transparency -- PDF-safe."""
    r, g, b, _ = mcolors.to_rgba(color)
    r2 = 1 - alpha * (1 - r)
    g2 = 1 - alpha * (1 - g)
    b2 = 1 - alpha * (1 - b)
    return (r2, g2, b2, 1.0)


class HandlerSplitPatch(HandlerBase):
    """Left half: solid fill (low res), Right half: hatch (high res)"""

    def __init__(self, color_solid, color_hatch, hatch='xx', **kwargs):
        self.color_solid = color_solid
        self.color_hatch = color_hatch
        self.hatch = hatch
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                       fontsize, trans):
        left = plt.Rectangle(
            [xdescent, ydescent], width / 2, height,
            fc=self.color_solid, ec=self.color_solid, alpha=0.3, transform=trans)
        right = plt.Rectangle(
            [xdescent + width / 2, ydescent], width / 2, height,
            fc='none', ec=color_with_alpha_on_white(self.color_hatch, 0.3),
            hatch=self.hatch, transform=trans)
        return [left, right]


def build_L_panel(quantity):
    """quantity: 'Mpeak' -> L_vs_Mpeak.pdf, 'rho150' -> L_vs_rho150.pdf.

    See the module docstring, "L_vs_rho150.pdf", for the toggle this
    generalises and the one piece (the Mpeak-floor axhspan/hatch) that is
    NOT a straight toggle.
    """
    if quantity not in ('Mpeak', 'rho150'):
        raise ValueError(f"quantity must be 'Mpeak' or 'rho150', got {quantity!r}")

    fiducial_version = 'Diemer'
    highres_version = 'mass_floor_7'

    # Cell 152: per-dwarf scatter points (Diemer only) -- the specific
    # quantity key requested, plus luminosity (version-independent).
    key = 'Mpeak_quantiles' if quantity == 'Mpeak' else 'rho150_quantiles'
    quantiles_list = []
    Lv_median_list = []
    Lv_err_list = []
    scatter_inputs = []
    for name in dwarf_names:
        qpath = _quantiles_path(fiducial_version, name)
        scatter_inputs.append(qpath)
        with np.load(qpath) as qfile:
            quantiles_list.append(qfile[key])
        this_dwarf = obs.Dwarf(name)
        Lv_median_list.append(this_dwarf.Lv)
        Lv_err_list.append(this_dwarf.Lv_err)
    provenance.assert_single_version(scatter_inputs, expected=fiducial_version)

    obs_quantiles = np.array(quantiles_list)
    Lv_median_list = np.array(Lv_median_list)
    Lv_err_list = np.array(Lv_err_list)

    obs_16, obs_50, obs_84 = np.array(
        [obs_quantiles[:, 0], obs_quantiles[:, 1], obs_quantiles[:, 2]])
    obs_pos_err = obs_84 - obs_50
    obs_neg_err = obs_50 - obs_16

    # Cells 154/155: the SatGen bands (Fattahi+18, Kim+24), fiducial + high-res.
    fiducial_h5 = config.h5_path(fiducial_version)
    highres_h5 = config.h5_path(highres_version)
    provenance.assert_single_version([fiducial_h5], expected=fiducial_version)
    provenance.assert_single_version([highres_h5], expected=highres_version)

    logMpeak, logrho150, Fattahi18, Kim24 = _shmr_fattahi_kim(fiducial_version, mass_floor=8)
    logobs = logMpeak if quantity == 'Mpeak' else logrho150
    bins, logobs_quantiles = _logLv_binned_quantiles(logobs, Fattahi18 - np.log10(1.2))
    _, logobs_quantiles_1 = _logLv_binned_quantiles(logobs, Kim24 - np.log10(1.2))

    logMpeak_hr, logrho150_hr, Fattahi18_hr, Kim24_hr = _shmr_fattahi_kim(
        highres_version, mass_floor=7)
    logobs_hr = logMpeak_hr if quantity == 'Mpeak' else logrho150_hr
    _, logobs_quantiles_highres = _logLv_binned_quantiles(logobs_hr, Fattahi18_hr - np.log10(1.2))
    _, logobs_quantiles_1_highres = _logLv_binned_quantiles(logobs_hr, Kim24_hr - np.log10(1.2))

    # Cell 156.
    fig, axs = plt.subplots(figsize=(6, 6))

    axs.errorbar(
        Lv_median_list[problem_idcs], obs_50[:, 0][problem_idcs],
        xerr=np.transpose(Lv_err_list[problem_idcs]),
        yerr=(obs_neg_err[:, 0][problem_idcs], obs_pos_err[:, 0][problem_idcs]),
        fmt=MARKERS[3], markersize=5, capsize=2, mec=colors[6], mfc='white',
        c=colors[6], label=r'$M_{1/2}$ inference - $N_{\rm stars}<10$', zorder=1000)

    axs.errorbar(
        Lv_median_list[ultrafaint_idcs], obs_50[:, 0][ultrafaint_idcs],
        xerr=np.transpose(Lv_err_list[ultrafaint_idcs]),
        yerr=(obs_neg_err[:, 0][ultrafaint_idcs], obs_pos_err[:, 0][ultrafaint_idcs]),
        fmt=MARKERS[0], markersize=5, capsize=2, mec=colors[5], mfc='white',
        c=colors[5], label=r'$M_{1/2}$ inference - Ultra-faints', zorder=1000)

    axs.errorbar(
        Lv_median_list[classical_idcs], obs_50[:, 0][classical_idcs],
        xerr=np.transpose(Lv_err_list[classical_idcs]),
        yerr=(obs_neg_err[:, 0][classical_idcs], obs_pos_err[:, 0][classical_idcs]),
        fmt=MARKERS[1], markersize=5, capsize=2, mec=colors[7], mfc='white',
        c=colors[7], label=r'$M_{1/2}$ inference - $M_{\star}>10^5$ M$_{\odot}$', zorder=1000)

    axs.fill_between(
        10 ** bins, 10 ** logobs_quantiles[:, 0], 10 ** logobs_quantiles[:, 2],
        color=colors[2], alpha=0.3, zorder=11)

    axs.fill_between(
        10 ** bins, 10 ** logobs_quantiles_1[:, 0], 10 ** logobs_quantiles_1[:, 2],
        color=colors[3], alpha=0.3, zorder=11)

    axs.fill_between(
        10 ** bins, 10 ** logobs_quantiles_highres[:, 0], 10 ** logobs_quantiles_highres[:, 2],
        fc='none', ec=color_with_alpha_on_white(colors[2], 0.3), hatch='xx', zorder=1)

    axs.fill_between(
        10 ** bins, 10 ** logobs_quantiles_1_highres[:, 0], 10 ** logobs_quantiles_1_highres[:, 2],
        fc='none', ec=color_with_alpha_on_white(colors[3], 0.3), hatch='xx', zorder=1)

    if quantity == 'Mpeak':
        # Mpeak-specific resolution-floor shading -- not ported for
        # quantity='rho150', see the module docstring.
        axs.axhspan(axs.get_ylim()[0], 1e8, facecolor='none', edgecolor='gray',
                   hatch='//////', alpha=0.5, zorder=1000)

    axs.set_xlabel(r'$L_V$ [L$_{\odot}$]', fontsize=15)
    if quantity == 'Mpeak':
        axs.set_ylabel(r'$M_{\rm peak}$ [M$_{\odot}$]', fontsize=15)
    else:
        axs.set_ylabel(r'$\rho_{150}$ [M$_{\odot}$ kpc$^{-3}$]', fontsize=15)

    axs.tick_params(axis='x', labelsize=15)
    axs.tick_params(axis='y', labelsize=15)

    axs.set_xlim(10 ** 1.75, 10 ** 9.5)
    axs.set_xscale('log')
    axs.set_yscale('log')

    fattahi_handle = Patch(
        label=r'\texttt{SatGen} - Fattahi+18' + '\n'
              + r'$\left(M_{\rm peak}>10^8\;(10^7)\;{\rm M}_{\odot}\right)$')
    kim_handle = Patch(
        label=r'\texttt{SatGen} - Kim+24' + '\n'
              + r'$\left(M_{\rm peak}>10^8\;(10^7)\;{\rm M}_{\odot}\right)$')

    handles, labels = axs.get_legend_handles_labels()
    axs.legend(
        handles=[fattahi_handle, kim_handle] + handles,
        labels=[h.get_label() for h in [fattahi_handle, kim_handle]] + labels,
        handler_map={
            fattahi_handle: HandlerSplitPatch(colors[2], colors[2]),
            kim_handle: HandlerSplitPatch(colors[3], colors[3]),
        },
        fontsize=15, frameon=False, loc='upper right')

    # cell 156's active/commented ylim pair: 2e12 (active, Mpeak) sits
    # directly below `# axs.set_ylim(axs.get_ylim()[0], 2e10)` (rho150).
    axs.set_ylim(axs.get_ylim()[0], 2e12 if quantity == 'Mpeak' else 2e10)
    plt.tight_layout()

    name = 'L_vs_Mpeak' if quantity == 'Mpeak' else 'L_vs_rho150'
    path = out_path(name)
    plt.savefig(path)
    plt.close(fig)

    inputs = [fiducial_h5, highres_h5] + scatter_inputs
    provenance.figure_manifest(
        path, 'python/plot_scaling_relations.py', inputs, version=None,
        argv=sys.argv[1:], quantity=quantity, fiducial_version=fiducial_version,
        highres_version=highres_version, mstar_evolution='per_halo_tidal',
        note=('deliberately spans two SatGen versions: the scatter points '
              'and low-res SatGen band are from Diemer, the crosshatched '
              'high-res band from mass_floor_7 (a different h5, not an '
              'alias). assert_single_version is called separately on each '
              'group above, never across both. Fattahi+18/Kim+24 draw from '
              'unseeded scatter (~0.02 dex on the plotted bands, not '
              'bit-reproducible run-to-run) -- NOT the dominant source of '
              'deviation from the published figure: evolved_Mstar\'s '
              'per-halo tidal correction shifts the Fattahi+18 68% band in '
              'logM_peak 0.25-0.40 dex higher than the published figure '
              '(made under the old constant +1.2 dex convention), an order '
              'of magnitude larger than the scatter term.'))
    return path


# ---------------------------------------------------------------------------
# lvdb_kdsa_rho150.pdf (cells 148, 149 -> 150)
# ---------------------------------------------------------------------------
def build_lvdb_kdsa_rho150():
    kd_version = 'Diemer'
    lvdb_version = 'lvdb'

    # Cell 149: rho150 quantiles (column 0 = mhalf weighting) for every
    # kd_names dwarf except Leo T/Leo V (no galactocentric quantile products
    # exist for Leo T; Leo V is excluded upstream too). Cell 148 (a
    # print-only KDSA-vs-LVDB dispersion diagnostic with no variable read
    # here) is dropped -- see the module docstring.
    names = [n for n in obs.kd_names if n not in ('Leo T', 'Leo V')]

    kd_rho150_list = []
    lvdb_rho150_list = []
    kd_inputs = []
    lvdb_inputs = []
    for name in names:
        kd_path = _quantiles_path(kd_version, name)
        lvdb_path = _quantiles_path(lvdb_version, name)
        kd_inputs.append(kd_path)
        lvdb_inputs.append(lvdb_path)
        with np.load(kd_path) as kd_file:
            kd_rho150_list.append(kd_file['rho150_quantiles'][:, 0])
        with np.load(lvdb_path) as lvdb_file:
            lvdb_rho150_list.append(lvdb_file['rho150_quantiles'][:, 0])

    kd_rho150quantiles = np.array(kd_rho150_list)
    lvdb_rho150quantiles = np.array(lvdb_rho150_list)

    kd_rho150_50 = kd_rho150quantiles[:, 1]
    kd_rho150_pos_err = kd_rho150quantiles[:, 2] - kd_rho150quantiles[:, 1]
    kd_rho150_neg_err = kd_rho150quantiles[:, 1] - kd_rho150quantiles[:, 0]

    lvdb_rho150_50 = lvdb_rho150quantiles[:, 1]
    lvdb_rho150_pos_err = lvdb_rho150quantiles[:, 2] - lvdb_rho150quantiles[:, 1]
    lvdb_rho150_neg_err = lvdb_rho150quantiles[:, 1] - lvdb_rho150quantiles[:, 0]

    # Cell 150.
    fig, ax = plt.subplots(figsize=(6, 6))

    h_uf = ax.errorbar(
        [np.nan], [np.nan], xerr=[1], yerr=[1], fmt=MARKERS[0], c=colors[5],
        markersize=5, mec=colors[5], mfc='white', capsize=2, linestyle='',
        label=r'Ultrafaints')
    h_cl = ax.errorbar(
        [np.nan], [np.nan], xerr=[1], yerr=[1], fmt=MARKERS[1], c=colors[7],
        markersize=5, mec=colors[7], mfc='white', capsize=2, linestyle='',
        label=r'$M_{\star}>10^{5}$ M$_{\odot}$')
    legend_handles = [h_uf, h_cl]

    for i, name in enumerate(names):
        if name in dwarf_names[ultrafaint_idcs]:
            color, marker = colors[5], MARKERS[0]
        elif name in dwarf_names[problem_idcs]:
            color, marker = colors[6], MARKERS[4]
        elif name in dwarf_names[classical_idcs]:
            color, marker = colors[7], MARKERS[1]
        else:
            continue
        ax.errorbar(
            x=lvdb_rho150_50[i], y=kd_rho150_50[i],
            xerr=[[lvdb_rho150_neg_err[i]], [lvdb_rho150_pos_err[i]]],
            yerr=[[kd_rho150_neg_err[i]], [kd_rho150_pos_err[i]]],
            linestyle='', c=color, fmt=marker, markersize=5, capsize=2,
            mec=color, mfc='white')

    ax.text(
        0.97, 0.03, r'$M_{\rm peak}>10^8$ M$_{\odot}$', transform=ax.transAxes,
        fontsize=15, va='bottom', ha='right')

    ax.legend(handles=legend_handles, frameon=False, loc='upper left', fontsize=15)
    test_arr = np.logspace(-1, 22)
    ax.plot(test_arr, test_arr, color='k', linestyle='--')

    ax.set_xlabel(r'$\rho_{150,{\rm LVDB}}$ [M$_{\odot}$ kpc$^{-3}$]', fontsize=15)
    ax.set_ylabel(r'$\rho_{150,{\rm KDSA}}$ [M$_{\odot}$ kpc$^{-3}$]', fontsize=15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e7, 1e9)
    ax.set_ylim(1e7, 1e9)

    path = out_path('lvdb_kdsa_rho150')
    plt.savefig(path)
    plt.close(fig)

    inputs = kd_inputs + lvdb_inputs
    # No `expected`: 'lvdb' is a reweighting-only alias of Diemer
    # (config.SIM_VERSION_ALIASES), so both groups resolve to the same
    # sim_version and this call would raise if `expected` pinned it to one
    # raw variant -- see save_contours.py's own note on why it never resolves
    # aliases either.
    provenance.assert_single_version(inputs)
    provenance.figure_manifest(
        path, 'python/plot_scaling_relations.py', inputs, version=None,
        argv=sys.argv[1:], kd_version=kd_version, lvdb_version=lvdb_version)
    return path


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
BUILDERS = {
    'tidal_stripping': build_tidal_stripping,
    'L_vs_Mpeak': lambda: build_L_panel('Mpeak'),
    'L_vs_rho150': lambda: build_L_panel('rho150'),
    'lvdb_kdsa_rho150': build_lvdb_kdsa_rho150,
}

if __name__ == '__main__':
    figure = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if figure not in FIGURES + ['all']:
        raise SystemExit(f"figure must be one of {FIGURES + ['all']}, got {figure!r}")
    to_run = FIGURES if figure == 'all' else [figure]
    for name in to_run:
        print(name)
        BUILDERS[name]()
