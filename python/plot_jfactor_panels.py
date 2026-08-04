"""Recreates the Part II J-factor comparison panels from
../SatGen_Dwarf/jupyter/PaperPlots.ipynb cell 23 (data prep) and cells 24,
26, 27 (savefigs):

    panel_Jfactor_theta95      -> plots/panels/panel_Jfactor_theta95.pdf
    multipanel_jeans_priors    -> plots/panels/multipanel_jeans_priors.pdf
    panel_Jfactor_literature   -> plots/panels/panel_Jfactor_literature.pdf

(Cell 25's FINESST variant is out of scope -- no drafts_temp figure
references `finesst_panel_Jfactor_theta95.pdf`.)

Migrated from SatGen_Dwarf/jupyter/PaperPlots.ipynb. Path/import/provenance
edits only -- the quantile selections, marker/color assignments, x-axis
ordering, shading, legends and axis limits are character-identical to the
notebook cells.

Two data sources feed every panel, both loaded once per figure by
`load_galactocentric` and `load_jeans`, over the full 43-dwarf catalog
(`obs.dwarf_names`), exactly as cell 23's loops do -- the `idcs` (39-dwarf)
subset from dwarf_categories is applied only when plotting, not when
loading:

  - galactocentric: results/paper_quantiles/galactocentric/Diemer/<dwarf>.npz,
    already-migrated compute_quantiles.py output. `J_quantiles`/
    `theta95_quantiles` column 0 is the M_1/2-weighted (mhalf) inference,
    column 4 is the Fattahi+18 M_star-weighted inference -- this ordering
    (weights_list = [mhalf] + mstar_weights[SHMR_names] + joint_weights[...])
    is compute_quantiles.py's, confirmed against fermi_expected_limits.py's
    "F18 J-factor -- index 4" comment. theta95 is stored in radians in that
    file and converted to degrees here, matching the cell.
  - Jeans: obs.Dwarf(name).get_Jeans_results(dja_prior=...), reading
    DwarfJeansAnalysis's <lvdb_key>/<prior>/derived.npz (config.DJA_RESULTS_DIR,
    config.DJA_PRIOR). Jeans_theta95_quantiles is already in degrees (the
    derived.npz's theta95_J_deg field), so it is NOT converted again.

`JEANS_PRIORS` is cell 23's literal 7-prior list (loguniform, satgen,
satgen_box, satgen_shmr_fattahi18, satgen_shmr_kim24,
satgen_shmr_danieli23_const, satgen_shmr_moster18) -- NOT config.DJA_PRIORS,
which additionally lists 'jeffreys', a prior cell 23 never computes.
save_contours.py's `jeans_priors` list already made this same choice for the
same reason; this module does not import it (that script is a CLI entry
point, not a shared module) but reproduces the identical literal. Each
figure only actually plots a subset of these 7 (see FIGURE_PRIORS below);
`THETA95_PRIORS` is the smaller 4-prior subset cell 23 computes theta95 for
(loguniform, satgen, satgen_box, satgen_shmr_fattahi18) -- cell 23 never
computes Jeans theta95 for kim24/danieli23_const/moster18, and neither figure
that uses those three priors (multipanel_jeans_priors' bottom panel) plots
their theta95.

Literature J-factor comparisons (mcdaniel, Geringer-Sameth, Bonnivard) are
static structured arrays already migrated onto Jdata.py
(`mcdaniel_Jeans_J_array`, `gs_J_array`, `bonnivard_J_array`) -- no DJA read,
no per-dwarf loop needed (cell 23 recomputes them on every loop iteration,
but the result does not depend on the iteration; computing once here is not
a numerics change).

Each figure's provenance sidecar fingerprints the SAMPLE files it actually
loaded: the 43 galactocentric quantile npz's (config.PAPER_QUANTILES_DIR)
under `inputs`, and the DJA derived.npz files `Dwarf.get_Jeans_results`
recorded via `jeans_dja_path` for the specific priors that figure plots,
under `dja_files`/`dja_prior` -- not config.DJA_RESULTS_DIR itself and not
its (stale, 2026-05-13) audit.json, since the posteriors were regenerated
2026-07-09 and only per-file fingerprints track that.

Usage:
    conda activate J_calc
    python plot_jfactor_panels.py [figure=all]
"""
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import Line2D

import numpy as np

from plot_style import *  # noqa: F401,F403 -- brings plt, mpl, colors, config
import Jdata as obs
import provenance
from dwarf_categories import (
    dwarf_names, idcs, n_dwarfs, problem_idcs, ultrafaint_idcs, classical_idcs,
)

VERSION = 'Diemer'

# Cell 23's literal 7-prior sweep -- see module docstring for why this is not
# config.DJA_PRIORS.
JEANS_PRIORS = [
    'loguniform', 'satgen', 'satgen_box', 'satgen_shmr_fattahi18',
    'satgen_shmr_kim24', 'satgen_shmr_danieli23_const', 'satgen_shmr_moster18',
]
# Subset cell 23 computes Jeans_theta95_* for.
THETA95_PRIORS = ['loguniform', 'satgen', 'satgen_box', 'satgen_shmr_fattahi18']

# Priors each figure actually plots -- what get_Jeans_results is called for
# and what gets fingerprinted in that figure's provenance sidecar.
FIGURE_PRIORS = {
    'panel_Jfactor_theta95': ['loguniform', 'satgen_shmr_fattahi18'],
    'multipanel_jeans_priors': [
        'loguniform', 'satgen_box', 'satgen', 'satgen_shmr_fattahi18',
        'satgen_shmr_kim24', 'satgen_shmr_danieli23_const', 'satgen_shmr_moster18',
    ],
    'panel_Jfactor_literature': ['satgen_shmr_fattahi18'],
}

FIGURES = list(FIGURE_PRIORS)
BASENAME = {
    'panel_Jfactor_theta95': 'panel_Jfactor_theta95.pdf',
    'multipanel_jeans_priors': 'multipanel_jeans_priors.pdf',
    'panel_Jfactor_literature': 'panel_Jfactor_literature.pdf',
}

figure = sys.argv[1] if len(sys.argv) > 1 else 'all'
if figure not in FIGURES + ['all']:
    raise SystemExit(f"figure must be one of {FIGURES + ['all']}, got {figure!r}")


def out_path(key):
    path = config.PLOTS_DIR / 'panels' / BASENAME[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


# ---------------------------------------------------------------------------
# Data loading (cell 23)
# ---------------------------------------------------------------------------
def load_galactocentric(version=VERSION):
    """43-dwarf J_quantiles/theta95_quantiles from the galactocentric
    weighted-catalog products, plus the files read (for provenance)."""
    Jquantiles_list = []
    theta95quantiles_list = []
    files = []
    for dwarf in dwarf_names:
        path = config.PAPER_QUANTILES_DIR / 'galactocentric' / version / f'{dwarf}.npz'
        files.append(path)
        quantiles_file = np.load(path)
        Jquantiles_list.append(quantiles_file['J_quantiles'])
        theta95quantiles_list.append(quantiles_file['theta95_quantiles'])

    Jquantiles = np.array(Jquantiles_list)
    J_16, J_50, J_84 = Jquantiles[:, 0], Jquantiles[:, 1], Jquantiles[:, 2]
    J_pos_err = J_84 - J_50
    J_neg_err = J_50 - J_16

    theta95quantiles = np.array(theta95quantiles_list)
    theta95_50 = np.degrees(theta95quantiles[:, 1])
    theta95_pos_err = np.degrees(theta95quantiles[:, 2] - theta95quantiles[:, 1])
    theta95_neg_err = np.degrees(theta95quantiles[:, 1] - theta95quantiles[:, 0])

    return dict(J_50=J_50, J_pos_err=J_pos_err, J_neg_err=J_neg_err,
                theta95_50=theta95_50, theta95_pos_err=theta95_pos_err,
                theta95_neg_err=theta95_neg_err, files=files)


def load_jeans(priors):
    """43-dwarf Jeans_J_{16,50,84}_<prior> and (for THETA95_PRIORS members
    of `priors`) Jeans_theta95_{16,50,84}_<prior>, plus the derived.npz files
    actually read across those priors (for provenance)."""
    J = {p: dict(J_16=np.full(len(dwarf_names), np.nan),
                 J_50=np.full(len(dwarf_names), np.nan),
                 J_84=np.full(len(dwarf_names), np.nan))
         for p in priors}
    theta = {p: dict(t_16=np.full(len(dwarf_names), np.nan),
                      t_50=np.full(len(dwarf_names), np.nan),
                      t_84=np.full(len(dwarf_names), np.nan))
             for p in priors if p in THETA95_PRIORS}
    dja_files = []

    for i, name in enumerate(dwarf_names):
        dwarf = obs.Dwarf(name)
        for p in priors:
            dwarf.get_Jeans_results(dja_prior=p)
            if dwarf.jeans_dja_path:
                dja_files.append(dwarf.jeans_dja_path)
            J[p]['J_16'][i], J[p]['J_50'][i], J[p]['J_84'][i] = dwarf.Jeans_J_quantiles
            if p in theta:
                (theta[p]['t_16'][i], theta[p]['t_50'][i],
                 theta[p]['t_84'][i]) = dwarf.Jeans_theta95_quantiles

    for p in priors:
        J[p]['J_pos_err'] = J[p]['J_84'] - J[p]['J_50']
        J[p]['J_neg_err'] = J[p]['J_50'] - J[p]['J_16']
    for p in theta:
        theta[p]['t_pos_err'] = theta[p]['t_84'] - theta[p]['t_50']
        theta[p]['t_neg_err'] = theta[p]['t_50'] - theta[p]['t_16']

    # Deduplicate, preserving order, so a dwarf/prior combination sharing the
    # same derived.npz across priors (it doesn't -- one file per prior -- but
    # a dwarf appearing under multiple priors reads distinct files) is not
    # fingerprinted twice for nothing.
    seen = set()
    dja_files_unique = []
    for f in dja_files:
        if f not in seen:
            seen.add(f)
            dja_files_unique.append(f)

    return J, theta, dja_files_unique


def load_literature():
    """Static McDaniel+23 / Geringer-Sameth+15 / Bonnivard+15 J-factor
    arrays -- no DJA read, computed once (cell 23 recomputes these every
    loop iteration; the result is iteration-independent)."""
    out = {}
    out['mcdaniel_50'] = 10 ** obs.mcdaniel_Jeans_J_array['log10J']
    out['mcdaniel_84'] = 10 ** (obs.mcdaniel_Jeans_J_array['log10J']
                                 + obs.mcdaniel_Jeans_J_array['log10J_err'])
    out['mcdaniel_16'] = 10 ** (obs.mcdaniel_Jeans_J_array['log10J']
                                 - obs.mcdaniel_Jeans_J_array['log10J_err'])
    out['mcdaniel_pos_err'] = out['mcdaniel_84'] - out['mcdaniel_50']
    out['mcdaniel_neg_err'] = out['mcdaniel_50'] - out['mcdaniel_16']

    out['gs_50'] = 10 ** obs.gs_J_array['log10J']
    out['gs_84'] = 10 ** (obs.gs_J_array['log10J'] + obs.gs_J_array['log10J_err_plus'])
    out['gs_16'] = 10 ** (obs.gs_J_array['log10J'] - obs.gs_J_array['log10J_err_minus'])
    out['gs_pos_err'] = out['gs_84'] - out['gs_50']
    out['gs_neg_err'] = out['gs_50'] - out['gs_16']

    out['bonnivard_50'] = 10 ** obs.bonnivard_J_array['log10J']
    out['bonnivard_84'] = 10 ** (obs.bonnivard_J_array['log10J']
                                  + obs.bonnivard_J_array['log10J_err_plus'])
    out['bonnivard_16'] = 10 ** (obs.bonnivard_J_array['log10J']
                                  - obs.bonnivard_J_array['log10J_err_minus'])
    out['bonnivard_pos_err'] = out['bonnivard_84'] - out['bonnivard_50']
    out['bonnivard_neg_err'] = out['bonnivard_50'] - out['bonnivard_16']
    return out


# ---------------------------------------------------------------------------
# Shared shading/xtick block, common to all three figures (cells 24/26/27).
# ---------------------------------------------------------------------------
def _shade_and_ticks(axs):
    group_counts = [len(problem_idcs), len(ultrafaint_idcs), len(classical_idcs)]
    group_colors = [colors[6], colors[5], colors[7]]
    group_labels = [r'$N_{\rm stars}<10$', 'Ultra-faints', r'$M_{\star}>10^5$ M$_{\odot}$']
    x_positions = np.arange(n_dwarfs) * 50

    for ax_i, ax in enumerate(axs):
        start_idx = 0
        left, right = ax.get_xlim()
        ax.set_xlim(left, right)
        for j, (count, color, label) in enumerate(zip(group_counts, group_colors, group_labels)):
            if count == 0:
                continue
            end_idx = start_idx + count - 1
            left = ax.get_xlim()[0] if j == 0 else x_positions[start_idx] - 12
            right = ax.get_xlim()[1] if j == len(group_counts) - 1 else x_positions[end_idx] + 38
            center = (x_positions[start_idx] + x_positions[end_idx]) / 2
            ax.axvspan(left, right, facecolor=color, alpha=0.1, zorder=0)
            if ax_i == 0:
                ax.text(center, 0.9, label, ha='center', va='top', fontsize=15, color='k',
                        fontweight='bold', transform=ax.get_xaxis_transform())
            start_idx += count

        for k in range(n_dwarfs):
            if k == 0:
                ax.axvline(x=x_positions[k] - 12, color='grey', linestyle='--', linewidth=0.5,
                          alpha=0.6, zorder=1)
            ax.axvline(x=x_positions[k] + 38, color='grey', linestyle='--', linewidth=0.5,
                       alpha=0.6, zorder=1)

    dwarf_names_latex = np.array([s.replace('Bootes', 'Boötes') for s in dwarf_names])
    axs[-1].set_xticks(np.arange(n_dwarfs) * 50 + 12 * np.ones(n_dwarfs),
                       dwarf_names_latex[idcs], rotation=90, fontsize=15)
    axs[-1].xaxis.set_minor_locator(plt.NullLocator())


# ---------------------------------------------------------------------------
# panel_Jfactor_theta95 (cell 24)
# ---------------------------------------------------------------------------
def build_panel_Jfactor_theta95():
    gc = load_galactocentric()
    jeans, theta, dja_files = load_jeans(FIGURE_PRIORS['panel_Jfactor_theta95'])
    lg, f18 = jeans['loguniform'], jeans['satgen_shmr_fattahi18']
    lg_t, f18_t = theta['loguniform'], theta['satgen_shmr_fattahi18']

    markers = ['s', 'D', '^', 'o', 'p', 'H']
    extra_colors = ["#882255", "#44AA99", "#DDCC77"]
    jeans_color = '#4B0082'

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(14, 12))
    plt.subplots_adjust(hspace=0)

    x_positions = np.arange(n_dwarfs) * 50
    off = 8  # four markers per dwarf at +4, +12, +20, +28, centered at +16

    # ===== TOP PANEL: J-factor =====
    axs[0].errorbar(x_positions + 0.5 * off, gc['J_50'][:, 0][idcs],
            yerr=(gc['J_neg_err'][:, 0][idcs], gc['J_pos_err'][:, 0][idcs]),
            fmt=markers[0], markersize=5, capsize=2, mec=colors[1], c=colors[1], mfc='white')
    axs[0].errorbar(x_positions + 1.5 * off, gc['J_50'][:, 4][idcs],
            yerr=(gc['J_neg_err'][:, 4][idcs], gc['J_pos_err'][:, 4][idcs]),
            fmt=markers[1], markersize=5, capsize=2, mec=colors[2], c=colors[2], mfc='white')
    axs[0].errorbar(x_positions + 2.5 * off, lg['J_50'][idcs],
            yerr=(lg['J_neg_err'][idcs], lg['J_pos_err'][idcs]),
            fmt=markers[2], markersize=5, capsize=2, mec=jeans_color, c=jeans_color, mfc='white')
    axs[0].errorbar(x_positions + 3.5 * off, f18['J_50'][idcs],
            yerr=(f18['J_neg_err'][idcs], f18['J_pos_err'][idcs]),
            fmt=markers[3], markersize=5, capsize=2, mec=extra_colors[1], c=extra_colors[1], mfc='white')

    axs[0].set_yscale('log')
    axs[0].set_ylabel(r'$J(0.5^{\circ})$ [GeV$^2$ cm$^{-5}$ sr]', fontsize=15)
    axs[0].tick_params(axis='y', labelsize=15)
    axs[0].set_yticks([1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
                       1e20, 1e21, 1e22, 1e23])
    axs[0].set_yticklabels(['', r'$10^{9}$', '', r'$10^{11}$', '', r'$10^{13}$', '', r'$10^{15}$',
                            '', r'$10^{17}$', '', r'$10^{19}$', '', r'$10^{21}$', '', r'$10^{23}$'])
    axs[0].set_ylim(1.5e14, 3e22)

    # ===== BOTTOM PANEL: theta95 =====
    axs[1].errorbar(x_positions + 0.5 * off, gc['theta95_50'][:, 0][idcs],
            yerr=(gc['theta95_neg_err'][:, 0][idcs], gc['theta95_pos_err'][:, 0][idcs]),
            fmt=markers[0], markersize=5, capsize=2, mec=colors[1], c=colors[1], mfc='white')
    axs[1].errorbar(x_positions + 1.5 * off, gc['theta95_50'][:, 4][idcs],
            yerr=(gc['theta95_neg_err'][:, 4][idcs], gc['theta95_pos_err'][:, 4][idcs]),
            fmt=markers[1], markersize=5, capsize=2, mec=colors[2], c=colors[2], mfc='white')
    axs[1].errorbar(x_positions + 2.5 * off, lg_t['t_50'][idcs],
            yerr=(lg_t['t_neg_err'][idcs], lg_t['t_pos_err'][idcs]),
            fmt=markers[2], markersize=5, capsize=2, mec=jeans_color, c=jeans_color, mfc='white')
    axs[1].errorbar(x_positions + 3.5 * off, f18_t['t_50'][idcs],
            yerr=(f18_t['t_neg_err'][idcs], f18_t['t_pos_err'][idcs]),
            fmt=markers[3], markersize=5, capsize=2, mec=extra_colors[1], c=extra_colors[1], mfc='white')

    axs[1].set_yscale('log')
    axs[1].set_ylabel(r'$\theta_{95}$ [deg]', fontsize=15)
    axs[1].tick_params(axis='y', labelsize=15)
    axs[1].set_ylim(6e-3, 2e1)

    _shade_and_ticks(axs)

    proxy_lines = [
        Line2D([0], [0], marker=markers[0], color=colors[1], linestyle='None', markersize=10,
              mfc='white', label=r'$M_{1/2}$ inference'),
        Line2D([0], [0], marker=markers[1], color=colors[2], linestyle='None', markersize=10,
              mfc='white', label=r'$M_{\star}$ inference - Fattahi+18'),
        Line2D([0], [0], marker=markers[2], color=jeans_color, linestyle='None', markersize=10,
              mfc='white', label=r'Jeans analyses - standard prior'),
        Line2D([0], [0], marker=markers[3], color=extra_colors[1], linestyle='None', markersize=10,
              mfc='white', label=r'Jeans analyses - \texttt{SatGen}-informed prior'),
    ]
    axs[1].legend(loc='lower left', fontsize=15, handles=proxy_lines, ncol=2, frameon=False)
    axs[1].text(0.98, 0.04, r'$M_{\rm peak}>10^8$ M$_{\odot}$',
               ha='right', va='bottom', fontsize=15, transform=axs[1].transAxes)

    path = out_path('panel_Jfactor_theta95')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    provenance.figure_manifest(path, 'python/plot_jfactor_panels.py', gc['files'],
                               version=VERSION, argv=sys.argv[1:],
                               figure_key='panel_Jfactor_theta95',
                               dja_prior=FIGURE_PRIORS['panel_Jfactor_theta95'],
                               dja_files=dja_files)
    print('saved:', path)


# ---------------------------------------------------------------------------
# multipanel_jeans_priors (cell 26)
# ---------------------------------------------------------------------------
def build_multipanel_jeans_priors():
    priors = FIGURE_PRIORS['multipanel_jeans_priors']
    jeans, _theta, dja_files = load_jeans(priors)
    j_lg, j_box, j_sg = jeans['loguniform'], jeans['satgen_box'], jeans['satgen']
    j_f18, j_k24 = jeans['satgen_shmr_fattahi18'], jeans['satgen_shmr_kim24']
    j_dan, j_mos = jeans['satgen_shmr_danieli23_const'], jeans['satgen_shmr_moster18']

    markers = ['s', 'D', '^', 'o', 'p', 'H']

    jeans_color = '#4B0082'
    c_loguniform = jeans_color
    c_satgen_box = '#117733'
    c_satgen = '#CC6677'
    c_fattahi_top = colors[2]

    shmr_palette = ["#56B4E9", "#009E73", "#F4A6C8", "#A0522D"]
    c_fattahi_bot = shmr_palette[0]
    c_kim24 = shmr_palette[1]
    c_danieli23 = shmr_palette[2]
    c_moster18 = shmr_palette[3]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(14, 12))
    plt.subplots_adjust(hspace=0)

    x_positions = np.arange(n_dwarfs) * 50
    off = 6  # four markers per dwarf at +3, +9, +15, +21, centered at +12

    # ===== TOP PANEL: Jeans priors =====
    axs[0].errorbar(x_positions + 0.5 * off, j_lg['J_50'][idcs],
            yerr=(j_lg['J_neg_err'][idcs], j_lg['J_pos_err'][idcs]),
            fmt=markers[0], markersize=5, capsize=2, mec=c_loguniform, c=c_loguniform, mfc='white')
    axs[0].errorbar(x_positions + 1.5 * off, j_box['J_50'][idcs],
            yerr=(j_box['J_neg_err'][idcs], j_box['J_pos_err'][idcs]),
            fmt=markers[1], markersize=5, capsize=2, mec=c_satgen_box, c=c_satgen_box, mfc='white')
    axs[0].errorbar(x_positions + 2.5 * off, j_sg['J_50'][idcs],
            yerr=(j_sg['J_neg_err'][idcs], j_sg['J_pos_err'][idcs]),
            fmt=markers[2], markersize=5, capsize=2, mec=c_satgen, c=c_satgen, mfc='white')
    axs[0].errorbar(x_positions + 3.5 * off, j_f18['J_50'][idcs],
            yerr=(j_f18['J_neg_err'][idcs], j_f18['J_pos_err'][idcs]),
            fmt=markers[3], markersize=5, capsize=2, mec=c_fattahi_top, c=c_fattahi_top, mfc='white')

    axs[0].set_yscale('log')
    axs[0].set_ylabel(r'$J(0.5^{\circ})$ [GeV$^2$ cm$^{-5}$ sr]', fontsize=15)
    axs[0].tick_params(axis='y', labelsize=15)
    axs[0].set_yticks([1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
                       1e20, 1e21, 1e22, 1e23])
    axs[0].set_yticklabels(['', r'$10^{9}$', '', r'$10^{11}$', '', r'$10^{13}$', '', r'$10^{15}$',
                            '', r'$10^{17}$', '', r'$10^{19}$', '', r'$10^{21}$', '', r'$10^{23}$'])
    axs[0].set_ylim(6e12, 3e22)

    # ===== BOTTOM PANEL: SHMR-weighted priors =====
    axs[1].errorbar(x_positions + 0.5 * off, j_f18['J_50'][idcs],
            yerr=(j_f18['J_neg_err'][idcs], j_f18['J_pos_err'][idcs]),
            fmt=markers[3], markersize=5, capsize=2, mec=c_fattahi_bot, c=c_fattahi_bot, mfc='white')
    axs[1].errorbar(x_positions + 1.5 * off, j_k24['J_50'][idcs],
            yerr=(j_k24['J_neg_err'][idcs], j_k24['J_pos_err'][idcs]),
            fmt=markers[1], markersize=5, capsize=2, mec=c_kim24, c=c_kim24, mfc='white')
    axs[1].errorbar(x_positions + 2.5 * off, j_dan['J_50'][idcs],
            yerr=(j_dan['J_neg_err'][idcs], j_dan['J_pos_err'][idcs]),
            fmt=markers[2], markersize=5, capsize=2, mec=c_danieli23, c=c_danieli23, mfc='white')
    axs[1].errorbar(x_positions + 3.5 * off, j_mos['J_50'][idcs],
            yerr=(j_mos['J_neg_err'][idcs], j_mos['J_pos_err'][idcs]),
            fmt=markers[0], markersize=5, capsize=2, mec=c_moster18, c=c_moster18, mfc='white')

    axs[1].set_yscale('log')
    axs[1].set_ylabel(r'$J(0.5^{\circ})$ [GeV$^2$ cm$^{-5}$ sr]', fontsize=15)
    axs[1].tick_params(axis='y', labelsize=15)
    axs[1].set_yticks([1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
                       1e20, 1e21, 1e22, 1e23])
    axs[1].set_yticklabels(['', r'$10^{9}$', '', r'$10^{11}$', '', r'$10^{13}$', '', r'$10^{15}$',
                            '', r'$10^{17}$', '', r'$10^{19}$', '', r'$10^{21}$', '', r'$10^{23}$'])
    axs[1].set_ylim(6e12, 3e22)

    _shade_and_ticks(axs)

    top_proxies = [
        Line2D([0], [0], marker=markers[0], color=c_loguniform, linestyle='None', markersize=10,
              mfc='white', label=r'Jeans analyses - Pace+18 log-uniform prior'),
        Line2D([0], [0], marker=markers[1], color=c_satgen_box, linestyle='None', markersize=10,
              mfc='white', label=r'Jeans analyses - \texttt{SatGen} log-uniform prior'),
        Line2D([0], [0], marker=markers[2], color=c_satgen, linestyle='None', markersize=10,
              mfc='white', label=r'Jeans analyses - unweighted \texttt{SatGen} log-normal prior'),
        Line2D([0], [0], marker=markers[3], color=c_fattahi_top, linestyle='None', markersize=10,
              mfc='white', label=r'Jeans analyses - Fattahi+18-weighted \texttt{SatGen} log-normal prior'),
    ]
    axs[0].legend(loc='lower left', fontsize=14, handles=top_proxies, ncol=2, frameon=False)

    bot_proxies = [
        Line2D([0], [0], marker=markers[3], color=c_fattahi_bot, linestyle='None', markersize=10,
              mfc='white', label=r'Jeans analyses - Fattahi+18-weighted \texttt{SatGen} log-normal prior'),
        Line2D([0], [0], marker=markers[1], color=c_kim24, linestyle='None', markersize=10,
              mfc='white', label=r'Jeans analyses - Kim+24-weighted \texttt{SatGen} log-normal prior'),
        Line2D([0], [0], marker=markers[2], color=c_danieli23, linestyle='None', markersize=10,
              mfc='white', label=r'Jeans analyses - Danieli+23-weighted \texttt{SatGen} log-normal prior'),
        Line2D([0], [0], marker=markers[0], color=c_moster18, linestyle='None', markersize=10,
              mfc='white', label=r'Jeans analyses - Moster+18-weighted \texttt{SatGen} log-normal prior'),
    ]
    axs[1].legend(loc='lower left', fontsize=14, handles=bot_proxies, ncol=1, frameon=False)

    path = out_path('multipanel_jeans_priors')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    provenance.figure_manifest(path, 'python/plot_jfactor_panels.py', [],
                               version=None, argv=sys.argv[1:],
                               figure_key='multipanel_jeans_priors',
                               dja_prior=priors, dja_files=dja_files)
    print('saved:', path)


# ---------------------------------------------------------------------------
# panel_Jfactor_literature (cell 27)
# ---------------------------------------------------------------------------
def build_panel_Jfactor_literature():
    gc = load_galactocentric()
    jeans, _theta, dja_files = load_jeans(FIGURE_PRIORS['panel_Jfactor_literature'])
    f18 = jeans['satgen_shmr_fattahi18']
    lit = load_literature()

    markers = ['s', 'D', '^', 'o', 'p', 'H']
    extra_colors = ["#882255", "#44AA99", "#DDCC77"]

    c_mhalf = colors[1]
    c_fattahi = colors[2]
    c_mcdaniel = 'tab:grey'
    c_gs = extra_colors[0]
    c_bonnivard = extra_colors[2]

    fig, ax = plt.subplots(figsize=(14, 6))

    x_positions = np.arange(n_dwarfs) * 50
    off = 6  # four markers per dwarf at +3, +9, +15, +21, centered at +12

    ax.errorbar(x_positions + 0.5 * off, gc['J_50'][:, 0][idcs],
            yerr=(gc['J_neg_err'][:, 0][idcs], gc['J_pos_err'][:, 0][idcs]),
            fmt=markers[0], markersize=5, capsize=2, mec=c_mhalf, c=c_mhalf, mfc='white')
    ax.errorbar(x_positions + 1.5 * off, f18['J_50'][idcs],
            yerr=(f18['J_neg_err'][idcs], f18['J_pos_err'][idcs]),
            fmt=markers[4], markersize=5, capsize=2, mec=c_fattahi, c=c_fattahi, mfc='white')
    ax.errorbar(x_positions + 2.5 * off, lit['mcdaniel_50'][idcs],
            yerr=(lit['mcdaniel_neg_err'][idcs], lit['mcdaniel_pos_err'][idcs]),
            fmt=markers[1], markersize=5, capsize=2, mec=c_mcdaniel, c=c_mcdaniel, mfc='white')
    ax.errorbar(x_positions + 3.5 * off, lit['gs_50'][idcs],
            yerr=(lit['gs_neg_err'][idcs], lit['gs_pos_err'][idcs]),
            fmt=markers[2], markersize=5, capsize=2, mec=c_gs, c=c_gs, mfc='white')
    ax.errorbar(x_positions + 4.5 * off, lit['bonnivard_50'][idcs],
            yerr=(lit['bonnivard_neg_err'][idcs], lit['bonnivard_pos_err'][idcs]),
            fmt=markers[3], markersize=5, capsize=2, mec=c_bonnivard, c=c_bonnivard, mfc='white')

    ax.set_yscale('log')
    ax.set_ylabel(r'$J(0.5^{\circ})$ [GeV$^2$ cm$^{-5}$ sr]', fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_yticks([1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
                  1e20, 1e21, 1e22, 1e23])
    ax.set_yticklabels(['', r'$10^{9}$', '', r'$10^{11}$', '', r'$10^{13}$', '', r'$10^{15}$',
                        '', r'$10^{17}$', '', r'$10^{19}$', '', r'$10^{21}$', '', r'$10^{23}$'])
    ax.set_ylim(2e11, 3e23)

    group_counts = [len(problem_idcs), len(ultrafaint_idcs), len(classical_idcs)]
    group_colors = [colors[6], colors[5], colors[7]]
    group_labels = [r'$N_{\rm stars}<10$', 'Ultra-faints', r'$M_{\star}>10^5$ M$_{\odot}$']

    start_idx = 0
    left, right = ax.get_xlim()
    ax.set_xlim(left, right)
    for j, (count, color, label) in enumerate(zip(group_counts, group_colors, group_labels)):
        if count == 0:
            continue
        end_idx = start_idx + count - 1
        left = ax.get_xlim()[0] if j == 0 else x_positions[start_idx] - 12
        right = ax.get_xlim()[1] if j == len(group_counts) - 1 else x_positions[end_idx] + 38
        center = (x_positions[start_idx] + x_positions[end_idx]) / 2
        ax.axvspan(left, right, facecolor=color, alpha=0.1, zorder=0)
        ax.text(center, 0.9, label, ha='center', va='top', fontsize=15, color='k',
               fontweight='bold', transform=ax.get_xaxis_transform())
        start_idx += count

    for k in range(n_dwarfs):
        if k == 0:
            ax.axvline(x=x_positions[k] - 12, color='grey', linestyle='--', linewidth=0.5,
                      alpha=0.6, zorder=1)
        ax.axvline(x=x_positions[k] + 38, color='grey', linestyle='--', linewidth=0.5,
                  alpha=0.6, zorder=1)

    dwarf_names_latex = np.array([s.replace('Bootes', 'Boötes') for s in dwarf_names])
    ax.set_xticks(np.arange(n_dwarfs) * 50 + 12 * np.ones(n_dwarfs),
                 dwarf_names_latex[idcs], rotation=90, fontsize=15)
    ax.xaxis.set_minor_locator(plt.NullLocator())

    proxy_lines = [
        Line2D([0], [0], marker=markers[0], color=c_mhalf, linestyle='None', markersize=10,
              mfc='white', label=r'$M_{1/2}$ inference'),
        Line2D([0], [0], marker=markers[4], color=c_fattahi, linestyle='None', markersize=10,
              mfc='white', label=r'Jeans analyses - \texttt{SatGen}-informed prior'),
        Line2D([0], [0], marker=markers[1], color=c_mcdaniel, linestyle='None', markersize=10,
              mfc='white', label=r'Circiello+26'),
        Line2D([0], [0], marker=markers[2], color=c_gs, linestyle='None', markersize=10,
              mfc='white', label=r'Geringer-Sameth+15'),
        Line2D([0], [0], marker=markers[3], color=c_bonnivard, linestyle='None', markersize=10,
              mfc='white', label=r'Bonnivard+15'),
    ]
    ax.legend(loc='lower left', fontsize=15, handles=proxy_lines, ncol=3, frameon=False)

    path = out_path('panel_Jfactor_literature')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    provenance.figure_manifest(path, 'python/plot_jfactor_panels.py', gc['files'],
                               version=VERSION, argv=sys.argv[1:],
                               figure_key='panel_Jfactor_literature',
                               dja_prior=FIGURE_PRIORS['panel_Jfactor_literature'],
                               dja_files=dja_files)
    print('saved:', path)


BUILDERS = {
    'panel_Jfactor_theta95': build_panel_Jfactor_theta95,
    'multipanel_jeans_priors': build_multipanel_jeans_priors,
    'panel_Jfactor_literature': build_panel_Jfactor_literature,
}

if __name__ == '__main__':
    to_run = FIGURES if figure == 'all' else [figure]
    for name in to_run:
        BUILDERS[name]()
