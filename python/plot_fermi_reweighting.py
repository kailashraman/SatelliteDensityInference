"""Fermi-LAT sigma-v recasting figures: the prior-ladder comparison, the
SHMR-only slice, the prior envelope against the literature GCE region, and
the Fattahi+18 SHMR Monte-Carlo band.

Migrated from ../SatGen_Dwarf/python/plot_fermi_reweighting.py, which itself
reconstructs PaperPlots.ipynb cells around 169-187 (see that file's docstring
for the cell-by-cell mapping; the real cells for the four figures kept here
are 173 (version toggle), 184 (MC), 186 (all_priors/SHMRs) and 187
(prior_envelope) -- the per-function citations below were previously stale
by ~3-4 cells and have been re-anchored). Four of its five FIGURES entries
are ported here:

    all_priors      -> fermi_reweighting_all_priors.pdf
    shmrs           -> fermi_reweighting_SHMRs.pdf
    prior_envelope  -> fermi_reweighting_prior_envelope.pdf
    mc              -> fermi_reweighting_MC.pdf

`fig_jswap` (fermi_reweighting_jswap.pdf) is deliberately NOT ported: it is
not in figure_registry.FIGURES (drafts_temp/ never \\includegraphics's it),
same as upstream's own `fermi_reweighting_limit.pdf`, which was already
excluded from the source file.

Path/import/provenance edits only -- the TS stacking (`load_summed`,
character-identical to upstream's function of the same name, minus its
`CHANGE_SPECS`/`CHANGE_ORDER` branch, which existed only to serve
`fig_jswap`'s progressive J-factor substitution and has no other caller
among the four figures kept here), the upper-limit extraction
(`fermi_funcs.get_UL`), the containment-band quantiles (`np.percentile` at
[16, 84] / [2.5, 97.5] in `fig_mc`), and the thermal-relic curve
(`draw_thermal_relic`) are all character-identical to upstream.

Plotting furniture already ported to `fermi_plot_helpers.py`
(`HandlerRectangle`, `SplitPatchHandler`, `shade_bounding_box_to_zero`,
`find_thermal_crossing`) is imported from there rather than re-pasted here.
Styling comes from `plot_style` (`from plot_style import *`), not a re-pasted
rcParams preamble -- upstream's module-level rc calls (its own copy of
PaperPlots.ipynb cells 1 and 2) are dropped entirely since `plot_style`
already applies the same net state on import.

Dwarf selection (no dwarf_categories involvement)
--------------------------------------------------
Unlike the panel figures, this figure family does not use
`dwarf_categories`' 39-dwarf ordering at all -- neither does upstream's
source file, which never imports it. Two dwarf lists are ported verbatim
instead:

* `SKIP_DWARFS` (13 names) is subtracted from the full 43-dwarf
  `obs.dwarf_names` catalog to build the TS stack for every figure here:
  43 - 13 = 30 dwarfs. This is the "30 dwarfs shared with Circiello+26"
  sample the in-panel annotation names.
* `BENCHMARK` (30 names, legacy TS-profile file stems) / `BENCHMARK_UPDATE`
  (the same 30, update tree's naming) is the external stacked comparison
  curve `fig_mc` reads (McDaniel+23 legacy / Circiello+26 update) -- a
  hardcoded list, independently the same 30 dwarfs as `SKIP_DWARFS`'
  complement, not derived from it in code.

ROOT (a hardcoded ../SatGen_Dwarf absolute path upstream) becomes
config-derived throughout: `config.FERMI_REWEIGHTING_DIR` /
`config.FERMI_REWEIGHTING_UPDATE_DIR` (both already existed) for the TS
stacks, `config.DATA_DIR / 'literature'` for the GCE-region CSVs and
`sigma_thermal.csv` (already migrated there), and `config.DATA_DIR /
'fermi_legacy(_update)' / 'dSphs' / 'TS_profiles'` for the external
comparison-limit TS profiles -- the legacy half of that reuses
`fermi_funcs.dSphs_TS_prof_path` (already existed); the update half has no
existing constant (`fermi_funcs.py` only defines `dSphs_sed_path_update`,
not a TS-profile counterpart) and is built here the same way that module
builds its own paths, since editing `fermi_funcs.py` is out of scope for
this step.

Every figure lands at `plots/fermi/<basename>.pdf` with a provenance
sidecar (`provenance.figure_manifest`) naming every per-dwarf/per-weight
`.npz` the figure actually stacked, plus the literature CSVs where used.

Usage:
    conda activate J_calc
    python plot_fermi_reweighting.py [figure=all] [--fermi-version legacy]

    figure         : all_priors | shmrs | prior_envelope | mc | all
    --fermi-version: legacy | update
"""
import argparse
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plot_style import *  # noqa: F401,F403 -- rcParams side effects, `colors`
import config
import provenance
import Jdata as obs
from fermi_funcs import get_UL, dSphs_TS_prof_path
from fermi_plot_helpers import (
    HandlerRectangle, SplitPatchHandler, shade_bounding_box_to_zero,
    find_thermal_crossing,
)

FIGURES = ['all_priors', 'shmrs', 'prior_envelope', 'mc']

# ---------------------------------------------------------------------------
# Data loading (cell 175)
# ---------------------------------------------------------------------------
# Full weight list kept verbatim so integer indices match the notebook's
# `summed_list` ordering, even though only indices 0-11 are ever selected by
# a figure kept here -- see the module docstring on why 12-18 are
# unreachable but must still occupy their slots. Only 13-16 (change_tuc2,
# change_seg1, change_ret2, change_car3) are the `fig_jswap`-only `change_*`
# chain; 12 (mcdaniel_jeans), 17 (mhalf_scatter) and 18 (mdyn_errani) are
# live variants with callers elsewhere, just not among the four figures kept
# here.
WEIGHT_LIST = ['Jeans', 'mhalf', 'F18', 'K24', 'Jeans_mhalf_err', 'mhalf_mcdaniel',
               'Jeans_satgen_box', 'Jeans_satgen',
               'Jeans_satgen_shmr_fattahi18', 'Jeans_satgen_shmr_kim24',
               'Jeans_satgen_shmr_danieli23_const', 'Jeans_satgen_shmr_moster18',
               'mcdaniel_jeans', 'change_tuc2', 'change_seg1', 'change_ret2', 'change_car3',
               'mhalf_scatter', 'mdyn_errani']

# Dwarves dropped from the stack (cells 175/178/179 use the same list).
SKIP_DWARFS = ['Sagittarius', 'SMC', 'Fornax', 'LMC', 'Eridanus IV', 'Pegasus IV',
               'Leo VI', 'Sculptor', 'Willman 1', 'Antlia II', 'Crater II',
               'Bootes I', 'Bootes III']


def results_root(fermi_version):
    """Cell 173's FERMI_RESULTS_VERSION toggle."""
    if fermi_version == 'update':
        return config.FERMI_REWEIGHTING_UPDATE_DIR
    return config.FERMI_REWEIGHTING_DIR


def load_summed(indices, fermi_version):
    """Stack TS over dwarves for the requested `summed_list` indices (cell 175).

    Returns (summed_by_index, mass_vec, sigmav_vec, used_files). Only the
    weights named by `indices` are read; the notebook loops over all 19.

    A missing input is fatal: a short stack is a systematically weaker limit
    that would otherwise render and log exactly like a complete one.
    """
    root = results_root(fermi_version)
    summed_by_index = {}
    mass_vec = sigmav_vec = None
    used_files = []
    for idx in sorted(indices):
        weight = WEIGHT_LIST[idx]
        summed = 0
        n_ok = 0
        expected = [n for n in obs.dwarf_names if n not in SKIP_DWARFS]
        missing = []
        for name in expected:
            ts_file = root / name / f'{weight}.npz'
            if not ts_file.exists():
                missing.append(ts_file)
                continue
            tsarray = np.load(ts_file)
            summed += tsarray['TS_array']
            # Grid equality across weights and dwarves is what lets this
            # function subset `indices` at all -- assert rather than assume.
            if mass_vec is None:
                mass_vec, sigmav_vec = tsarray['mass_vec'], tsarray['sigmav_vec']
            else:
                assert np.array_equal(mass_vec, tsarray['mass_vec']) and \
                    np.array_equal(sigmav_vec, tsarray['sigmav_vec']), \
                    f'grid mismatch in {ts_file}'
            n_ok += 1
            used_files.append(ts_file)
        if missing:
            raise FileNotFoundError(
                f'{weight}: {len(missing)} of {len(expected)} inputs missing, '
                f'refusing to plot a partial stack:\n  ' +
                '\n  '.join(str(p) for p in missing))
        print(f'  [{idx}] {weight}: {n_ok}/{len(expected)} dwarves')
        summed_by_index[idx] = summed
    return summed_by_index, mass_vec, sigmav_vec, used_files


# The 30 dwarves McDaniel+23 and this analysis have in common (the
# notebook's `benchmark` list, commented entries dropped -- NOT cell 171,
# which is a kstest, not the source of this list; the actual source cell is
# unverified). Names follow the legacy TS-profile file convention, not
# obs.dwarf_names.
BENCHMARK = [
    # our sample
    'Bootes_2', 'Canes_V1', 'Canes_V2', 'Carina', 'Carina_2', 'Centaurus_1',
    'Berenices', 'Draco', 'Eridanus_2', 'Hercules', 'Hydrus_1', 'Leo_1',
    'Leo_2', 'Leo_4', 'Leo_5', 'Reticulum_2', 'Pegasus_3', 'Segue_1',
    'Sextans', 'Tucana_4', 'Ursa_Major_1', 'Ursa_Major_2', 'Ursa_Minor',
    # less than 10 stars
    'Aquarius_2', 'Carina_3', 'Grus_1', 'Horologium_1', 'Pisces_2',
    'Tucana_2', 'Tucana_5',
]

# The update TS-profile tree names the same dwarves in roman numerals and drops
# the prior/noprior split, so the notebook's filename template does not
# address it (see BENCHMARK's comment on why that source isn't cell 171).
BENCHMARK_UPDATE = {
    'Bootes_2': 'Bootes_II', 'Canes_V1': 'Canes_Venatici_I',
    'Canes_V2': 'Canes_Venatici_II', 'Carina': 'Carina', 'Carina_2': 'Carina_II',
    'Centaurus_1': 'Centaurus_I', 'Berenices': 'Coma_Berenices', 'Draco': 'Draco',
    'Eridanus_2': 'Eridanus_II', 'Hercules': 'Hercules', 'Hydrus_1': 'Hydrus_I',
    'Leo_1': 'Leo_I', 'Leo_2': 'Leo_II', 'Leo_4': 'Leo_IV', 'Leo_5': 'Leo_V',
    'Reticulum_2': 'Reticulum_II', 'Pegasus_3': 'Pegasus_III', 'Segue_1': 'Segue_1',
    'Sextans': 'Sextans', 'Tucana_4': 'Tucana_IV', 'Ursa_Major_1': 'Ursa_Major_I',
    'Ursa_Major_2': 'Ursa_Major_II', 'Ursa_Minor': 'Ursa_Minor',
    'Aquarius_2': 'Aquarius_II', 'Carina_3': 'Carina_III', 'Grus_1': 'Grus_I',
    'Horologium_1': 'Horologium_I', 'Pisces_2': 'Pisces_II', 'Tucana_2': 'Tucana_II',
    'Tucana_5': 'Tucana_V',
}

# BENCHMARK_UPDATE is a hardcoded, independently-typed 30-dwarf list; SKIP_DWARFS
# is subtracted from the full 43-dwarf catalog to get the other 30-dwarf list
# `load_summed`/`load_summed_mc` actually stack. They agree today (see module
# docstring), but nothing in the code enforces it -- a future catalog or
# SKIP_DWARFS edit could silently make fig_mc's comparison curve and TS stack
# cover different samples. obs.dwarf_names uses spaces; BENCHMARK_UPDATE's
# values use underscores, hence the normalization below.
_skip_derived_30 = sorted(n.replace(' ', '_') for n in obs.dwarf_names if n not in SKIP_DWARFS)
_benchmark_update_30 = sorted(BENCHMARK_UPDATE.values())
assert _skip_derived_30 == _benchmark_update_30, (
    "SKIP_DWARFS-derived 30-dwarf set no longer matches BENCHMARK_UPDATE's -- "
    "fig_mc would stack the TS over a different sample than its comparison curve:\n"
    f"  only in SKIP_DWARFS-derived set: {sorted(set(_skip_derived_30) - set(_benchmark_update_30))}\n"
    f"  only in BENCHMARK_UPDATE: {sorted(set(_benchmark_update_30) - set(_skip_derived_30))}")

# Legend label for the external stacked comparison limit, per version.
COMPARISON_LABEL = {
    'legacy': 'McDaniel+23 - observed limit',
    'update': 'Circiello+26 - observed limit',
}

# Update tree's TS-profile directory: no existing fermi_funcs.py constant
# names it (only dSphs_sed_path_update does, for the SEDs); built here the
# same way fermi_funcs.py builds its own 'fermi_legacy_update' paths, since
# editing that module is out of scope for this step.
_TS_PROF_PATH_UPDATE = config.DATA_DIR / 'fermi_legacy_update' / 'dSphs' / 'TS_profiles'


def load_comparison_limit(fermi_version, channelname='bb', prior=True):
    """The external stacked observed limit over the benchmark dwarves (the
    notebook's `benchmark` block; see BENCHMARK's comment above -- not cell
    171, which is a kstest).

    legacy -> data/fermi_legacy/dSphs/TS_profiles/{name}_{Jprior|noprior}_{ch}.npy
    update -> data/fermi_legacy_update/dSphs/TS_profiles/{name}_{ch}.npy

    Both trees store (40, 60) arrays on the mass/sigmav grids fixed by the
    TS-profile generator (jupyter/fermi_reweighting.ipynb cell 8), which differ
    from our own npz grids (40x70) -- hence built here rather than read.

    NOTE: the update tree carries a single file per dwarf with no prior/noprior
    split, so `prior` is inert there; it is taken to be the J-prior product,
    matching the legacy file the notebook selects.
    """
    fermi_mass_vec = np.logspace(0, 4, 40)
    fermi_sigmav_vec = np.logspace(-28, -22, 60)

    if fermi_version == 'update':
        ts_path = _TS_PROF_PATH_UPDATE
        names = [f'{BENCHMARK_UPDATE[s]}_{channelname}' for s in BENCHMARK]
    else:
        ts_path = dSphs_TS_prof_path  # str, trailing slash
        tag = 'Jprior' if prior else 'noprior'
        names = [f'{s}_{tag}_{channelname}' for s in BENCHMARK]

    summed = 0
    counter = 0
    used_files = []
    for stem in names:
        ts_file = (ts_path if isinstance(ts_path, str) else str(ts_path) + '/') + stem + '.npy'
        if os.path.exists(ts_file):
            summed += np.load(ts_file)
            counter += 1
            used_files.append(ts_file)
        else:
            print(ts_file + " does not exist")
    if counter != len(BENCHMARK):
        raise SystemExit(f'comparison limit: stacked {counter}/{len(BENCHMARK)} '
                         f'benchmark dwarves from {ts_path} -- refusing to plot a '
                         f'partial stack')
    print(f'  comparison limit ({fermi_version}): {counter} dwarves')

    xs_fermi, ulsamp_fermi, _ = get_UL(summed.T, fermi_mass_vec, fermi_sigmav_vec)
    return xs_fermi, ulsamp_fermi, used_files


def load_summed_mc(fermi_version, weight='F18_MC'):
    """Per-draw stacked TS for a Monte-Carlo SHMR weight (cell 178).

    Returns (summed, mass_vec, sigmav_vec, used_files); summed has shape
    (N_MC, n_mass, n_sigmav).
    """
    root = results_root(fermi_version)
    summed = None
    mass_vec = sigmav_vec = None
    expected = [n for n in obs.dwarf_names if n not in SKIP_DWARFS]
    missing = []
    used_files = []
    n_ok = 0
    for name in expected:
        ts_file = root / name / f'{weight}.npz'
        if not ts_file.exists():
            missing.append(ts_file)
            continue
        d = np.load(ts_file)
        ts = d['TS_arrays']
        summed = ts if summed is None else summed + ts
        if mass_vec is None:
            mass_vec, sigmav_vec = d['mass_vec'], d['sigmav_vec']
        else:
            assert np.array_equal(mass_vec, d['mass_vec']) and \
                np.array_equal(sigmav_vec, d['sigmav_vec']), \
                f'grid mismatch in {ts_file}'
        n_ok += 1
        used_files.append(ts_file)
    if missing:
        raise FileNotFoundError(
            f'{weight}: {len(missing)} of {len(expected)} inputs missing, '
            f'refusing to plot a partial stack:\n  ' +
            '\n  '.join(str(p) for p in missing))
    print(f'  {weight}: {n_ok}/{len(expected)} dwarves, {summed.shape[0]} draws')
    return summed, mass_vec, sigmav_vec, used_files


# ---------------------------------------------------------------------------
# Shared figure furniture (cell 180)
# ---------------------------------------------------------------------------
def find_thermal_crossing_report(label, mass_vec, ulsamp, sigma_thermal):
    cross_mass, cross_sigmav, excl_factor = find_thermal_crossing(mass_vec, ulsamp, sigma_thermal)
    if cross_mass is not None:
        print(f"{label} crosses thermal relic at m = {cross_mass:.2f} GeV, "
              f"<sv> = {cross_sigmav:.3e} cm^3/s")
    if excl_factor is not None:
        if excl_factor < 1:
            print(f"At m = 40 GeV, the thermal relic cross section is excluded "
                  f"by a factor of {1/excl_factor:.2f} (line/thermal = {excl_factor:.3f}).")
        else:
            print(f"At m = 40 GeV, the line is a factor of {excl_factor:.2f} "
                  f"above the thermal relic (not excluded).")


LITERATURE_DIR = config.DATA_DIR / 'literature'


def draw_thermal_relic(axs):
    """Thermal relic curve, log-log extrapolated past the data to the right
    edge of the frame (cells 184, 186, 187, identical in all three)."""
    sigma_thermal = np.genfromtxt(LITERATURE_DIR / 'sigma_thermal.csv', delimiter=',')
    x_data, y_data = sigma_thermal[:, 0], sigma_thermal[:, 1]
    x_max_plot = 1e4
    N = 5
    slope, intercept = np.polyfit(np.log10(x_data[-N:]), np.log10(y_data[-N:]), 1)
    x_ext = np.logspace(np.log10(x_data[-1]), np.log10(x_max_plot * 1.1), 50)
    y_ext = 10**(slope * np.log10(x_ext) + intercept)
    axs.plot(np.concatenate([x_data, x_ext]), np.concatenate([y_data, y_ext]),
             linestyle='-', c='gray', lw=0.5)
    axs.text(1.5e3, 2.2e-26, 'Thermal relic', fontsize=10, ha='left', va='bottom')
    return sigma_thermal


# The in-panel sample annotation names the external analysis our 30-dwarf
# benchmark sample overlaps with, which is the analysis the version's SEDs come
# from -- so it tracks --fermi-version alongside the comparison curve's label.
SAMPLE_ANNOTATION = {
    'legacy': 'Dwarfs shared with McDaniel+23',
    'update': 'Dwarfs shared with Circiello+26',
}


def finish_axes(axs, handles, labels, handler_map, fermi_version='legacy',
                legend_kwargs=None):
    """Corner annotations, labels, limits and legend -- identical across
    cells 184, 186 and 187."""
    axs.text(0.97, 0.03, SAMPLE_ANNOTATION[fermi_version], transform=axs.transAxes,
             fontsize=15, va='bottom', ha='right', ma='center')
    axs.text(0.03, 0.03, r'$\chi\chi\to b \bar{b}$', transform=axs.transAxes,
             fontsize=15, va='bottom', ha='left', ma='center')

    axs.set_xlabel(r'$M_{\chi}$ [GeV]', fontsize=15)
    axs.set_ylabel(r'$\left<\sigma v\right>$ [cm$^{3}$ s$^{-1}$]', fontsize=15)
    axs.set_xlim(7, 1e4)
    axs.set_ylim(1e-28, 1e-22)
    axs.tick_params(axis='x', labelsize=15)
    axs.tick_params(axis='y', labelsize=15)
    axs.set_xscale('log')
    axs.set_yscale('log')

    kw = dict(loc='upper left', fontsize=14, frameon=False)
    kw.update(legend_kwargs or {})
    axs.legend(handles, labels, handler_map=handler_map, **kw)
    plt.tight_layout()


GCE_FILES_4 = ["Abazajian2015.csv", "Daylan2014.csv", "Calore2014.csv", "DiMauro.csv"]


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
# cell 186 (identifiers below keep their `_182` suffix from the prior,
# stale cell numbering -- not renamed to avoid an unrelated identifier churn)
PALETTE_182 = [
    '#4B0082',  # indigo (jeans)
    '#E69F00',  # orange
    'k', 'k', 'k', 'k',
    '#117733',  # dark green  (Tol)
    '#CC6677',  # rose        (Tol)
    '#88CCEE',  # light cyan  (Tol)
]
SHMR_PALETTE = [
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F4A6C8",  # soft pink
    "#A0522D",  # sienna
]
LABELS_182 = [
    'Jeans analyses - Pace+18 log-uniform prior',
    r'$M_{1/2}$ inference',
    r'$M_{\star}$ inference - Fattahi+18',
    r'$M_{\star}$ inference - Kim+24',
    r'Literature Jeans analyses, $M_{1/2}$ error',
    r'$M_{1/2}$ inference - old $\langle \sigma_{\rm LOS}\rangle$',
    r'Jeans analyses - \texttt{SatGen} log-uniform prior',
    r'Jeans analyses - \texttt{SatGen} log-normal prior',
    'Jeans analyses - Fattahi+18-weighted\nlog-normal prior',
    'Jeans analyses - Kim+24-weighted\nlog-normal prior',
    'Jeans analyses - Danieli+23-weighted\nlog-normal prior',
    'Jeans analyses - Moster+18-weighted\nlog-normal prior',
]

# cell 187 (identifiers below keep their `_183` suffix from the prior,
# stale cell numbering)
LABELS_183 = [
    'Jeans analyses - standard prior',
    r'$M_{1/2}$ inference',
    r'$M_{\star}$ inference - Fattahi+18',
    r'$M_{\star}$ inference - Kim+24',
    r'Literature Jeans analyses, $M_{1/2}$ error',
    r'$M_{1/2}$ inference - old $\langle \sigma_{\rm LOS}\rangle$',
    r'Jeans \texttt{SatGen} box',
    r'Jeans \texttt{SatGen} lognormal',
    r'Jeans analyses - \texttt{SatGen}-informed prior',
]
LINESTYLES_183 = ['--', '-', ':', '-.', '-.', ':', '-', '-', ':', '-.']
BAND_COLOR = '#9467BD'  # muted purple


def save(fig, name, script_inputs, fermi_version, **extra):
    """Write `plots/fermi/<name>` and its `provenance.figure_manifest` sidecar."""
    out_path = config.PLOTS_DIR / 'fermi' / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    provenance.figure_manifest(
        out_path, 'python/plot_fermi_reweighting.py', inputs=script_inputs,
        fermi_version=fermi_version, **extra)
    print('wrote', out_path)


def _cell182(indices, palette_fn, out_name, fermi_version):
    """Shared body of `all_priors` and `shmrs` -- cell 186 differs between the
    two only in the series filter and the colour lookup."""
    summed_by_index, mass_vec, sigmav_vec, used_files = load_summed(indices, fermi_version)

    fig, axs = plt.subplots(figsize=(6, 6))
    sigma_thermal = draw_thermal_relic(axs)

    handles, labels = [], []
    for i in sorted(indices):
        xs, ulsamp, _ = get_UL(summed_by_index[i].T, mass_vec, sigmav_vec)
        line = axs.plot(xs, ulsamp, c=palette_fn(i), label=LABELS_182[i], lw=2, ls='-')[0]
        # M_1/2 (i=1) is hoisted to the top of the legend; it is the reference
        # curve the priors are compared against.
        if i == 1:
            handles.insert(0, line)
            labels.insert(0, LABELS_182[i])
        else:
            handles.append(line)
            labels.append(LABELS_182[i])
        find_thermal_crossing_report(LABELS_182[i], mass_vec, ulsamp, sigma_thermal)

    finish_axes(axs, handles, labels,
                handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=1),
                             Rectangle: HandlerRectangle()},
                fermi_version=fermi_version,
                legend_kwargs=dict(fontsize=13, alignment='center'))
    save(fig, out_name, used_files + [LITERATURE_DIR / 'sigma_thermal.csv'],
         fermi_version, weight_indices=sorted(indices))


def fig_all_priors(fermi_version):
    _cell182({0, 1, 6, 7, 8}, lambda i: PALETTE_182[i],
             'fermi_reweighting_all_priors.pdf', fermi_version)


def fig_shmrs(fermi_version):
    _cell182({8, 9, 10, 11}, lambda i: SHMR_PALETTE[i - 8],
             'fermi_reweighting_SHMRs.pdf', fermi_version)


def fig_prior_envelope(fermi_version):
    line_indices = [0, 1, 8]
    # Cell 187's code (not its comment, which stales-mistranscribes "i = 0, 1,
    # 6, 7, 8, 9, 10, 11") sets band_indices = [0, 6, 7, 8, 9, 10, 11] -- i.e.
    # WITHOUT index 1 (mhalf). The band is scoped to Jeans-analysis prior
    # variants only; index 1 is mhalf, which is not a Jeans analysis, so it is
    # excluded from band_indices.
    band_indices = [0, 6, 7, 8, 9, 10, 11]
    summed_by_index, mass_vec, sigmav_vec, used_files = load_summed(
        set(line_indices) | set(band_indices), fermi_version)

    fig, axs = plt.subplots(figsize=(6, 6))
    sigma_thermal = draw_thermal_relic(axs)

    gce_paths = [LITERATURE_DIR / f for f in GCE_FILES_4]
    shade_bounding_box_to_zero(gce_paths, ax=axs,
                               top_color='steelblue', bottom_color='tab:grey',
                               top_alpha=0.4, bottom_alpha=0.3,
                               label="bounding region", zorder=-100)
    box_proxy = ('steelblue', 0.4, 'tab:grey', 0.3)

    handles, labels = [], []
    for i in line_indices:
        xs, ulsamp, _ = get_UL(summed_by_index[i].T, mass_vec, sigmav_vec)
        line = axs.plot(xs, ulsamp, c='k', label=LABELS_183[i], lw=2,
                        ls=LINESTYLES_183[i], zorder=1000)[0]
        if i == 1:
            handles.insert(0, line)
            labels.insert(0, LABELS_183[i])
        else:
            handles.append(line)
            labels.append(LABELS_183[i])
        find_thermal_crossing_report(LABELS_183[i], mass_vec, ulsamp, sigma_thermal)

    band_ulsamps = []
    for i in band_indices:
        _, ulsamp, _ = get_UL(summed_by_index[i].T, mass_vec, sigmav_vec)
        band_ulsamps.append(ulsamp)
    band_ulsamps = np.array(band_ulsamps)
    band = axs.fill_between(mass_vec, band_ulsamps.min(axis=0), band_ulsamps.max(axis=0),
                            color=BAND_COLOR, alpha=0.3, zorder=0)
    handles.append(band)
    labels.append('Upper limit envelope (all priors)')

    handles.append(box_proxy)
    labels.append('Literature GCE region')

    finish_axes(axs, handles, labels,
                handler_map={tuple: SplitPatchHandler(), Rectangle: HandlerRectangle()},
                fermi_version=fermi_version)
    save(fig, 'fermi_reweighting_prior_envelope.pdf', used_files + gce_paths +
         [LITERATURE_DIR / 'sigma_thermal.csv'], fermi_version,
         line_indices=line_indices, band_indices=band_indices)


# cell 184 (identifiers below keep their `_181` suffix from the prior,
# stale cell numbering)
LABELS_181 = [
    'Jeans analyses - observed limit',
    r'$M_{1/2}$ inference - observed limit',
    r'$M_{\star}$ inference - Fattahi+18',
    r'$M_{\star}$ inference - Kim+24',
    r'Literature Jeans analyses, $M_{1/2}$ error',
    r'$M_{1/2}$ inference - old $\langle \sigma_{\rm LOS}\rangle$',
]
LINESTYLES_181 = ['--', '-', ':', '-.', '-.', ':']


def fig_mc(fermi_version):
    summed_by_index, mass_vec, sigmav_vec, used_files = load_summed({1}, fermi_version)
    summed_mc, mc_mass_vec, mc_sigmav_vec, mc_files = load_summed_mc(fermi_version, 'F18_MC')
    # cell 184 used one global mass_vec for both; here they are loaded
    # independently, so the shared-grid assumption is made explicit.
    assert np.array_equal(mass_vec, mc_mass_vec) and \
        np.array_equal(sigmav_vec, mc_sigmav_vec), 'mhalf and F18_MC grids differ'

    fig, axs = plt.subplots(figsize=(6, 6))

    xs_fermi, ulsamp_fermi, comparison_files = load_comparison_limit(fermi_version)
    h1 = axs.plot(xs_fermi, ulsamp_fermi, color='gray', lw=2)[0]

    sigma_thermal = draw_thermal_relic(axs)

    handles, labels = [h1], [COMPARISON_LABEL[fermi_version]]

    i = 1
    xs, ulsamp, _ = get_UL(summed_by_index[i].T, mass_vec, sigmav_vec)
    line = axs.plot(xs, ulsamp, c='k', label=LABELS_181[i], lw=2, ls=LINESTYLES_181[i])[0]
    handles.append(line)
    labels.append(LABELS_181[i])
    find_thermal_crossing_report(LABELS_181[i], mass_vec, ulsamp, sigma_thermal)

    # 68%/95% bands over the per-draw Fattahi+18 SHMR sampling.
    bcolor = colors[2]  # sky blue
    zorder = -2
    N_MC = summed_mc.shape[0]
    uls_per_draw = np.empty((N_MC, len(mc_mass_vec)))
    for k in range(N_MC):
        _, uls, _ = get_UL(summed_mc[k].T, mc_mass_vec, mc_sigmav_vec)
        uls_per_draw[k] = uls

    lo68, hi68 = np.percentile(uls_per_draw, [16, 84], axis=0)
    axs.fill_between(mc_mass_vec, lo68, hi68, color=bcolor, alpha=0.7,
                     edgecolor='none', zorder=zorder)
    lo95, hi95 = np.percentile(uls_per_draw, [2.5, 97.5], axis=0)
    axs.fill_between(mc_mass_vec, lo95, hi95, color=bcolor, alpha=0.4,
                     edgecolor='none', zorder=zorder - 1)

    # Legend proxy: outer patch = 95% band, inner = 68%. Note the proxy alphas
    # (0.2/0.5) deliberately differ from the drawn bands (0.4/0.7) -- kept as
    # committed in cell 184.
    outer_patch = Rectangle((0, 0.0), 1, 1.0, facecolor=bcolor, alpha=0.2, edgecolor='none')
    inner_patch = Rectangle((0, 0.25), 1, 0.5, facecolor=bcolor, alpha=0.5, edgecolor='none')
    handles.append((outer_patch, inner_patch))
    labels.append(r'Fattahi+18 sampling')

    find_thermal_crossing_report('F18_MC median', mc_mass_vec,
                    np.percentile(uls_per_draw, 50, axis=0), sigma_thermal)

    # cell 184 passes no handler_map -- the composite falls through to
    # matplotlib's default tuple handler.
    finish_axes(axs, handles, labels, handler_map=None, fermi_version=fermi_version)
    save(fig, 'fermi_reweighting_MC.pdf',
         used_files + mc_files + [str(p) for p in comparison_files] +
         [LITERATURE_DIR / 'sigma_thermal.csv'],
         fermi_version, N_MC=N_MC)


BUILDERS = {
    'all_priors': fig_all_priors,
    'shmrs': fig_shmrs,
    'prior_envelope': fig_prior_envelope,
    'mc': fig_mc,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure', nargs='?', default='all', choices=FIGURES + ['all'])
    parser.add_argument('--fermi-version', default='legacy', choices=('legacy', 'update'))
    args = parser.parse_args(argv)

    to_render = FIGURES if args.figure == 'all' else [args.figure]
    for name in to_render:
        print(f'--- {name} [{args.fermi_version}]')
        BUILDERS[name](args.fermi_version)


if __name__ == '__main__':
    main()
