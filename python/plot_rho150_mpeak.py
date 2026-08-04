"""rho150-vs-Mpeak inference figures for Draco, Segue 1, and Hydrus I.

Migrated from ../SatGen_Dwarf/jupyter/PaperPlots.ipynb cell 198 (the
per-dwarf plotting loop), which consumes the NFW helpers defined in cells
196-197 -- already lifted into nfw.py, imported here rather than re-pasted.
Draft target: drafts_temp/temp_part1.tex's fig:rho150_vs_Mpeak
(``rho150_mpeak_Draco.pdf``, ``rho150_mpeak_Segue 1.pdf``,
``rho150_mpeak_Hydrus I.pdf`` -- dwarf names contain spaces and are kept
verbatim, matching the .tex \\includegraphics basenames).

Scope decisions flagged, not made silently:

* Entry point: `nfw_parameters_from_rho150_mpeak(..., z=0)`, not
  `rho150_predicted_direct`. The latter (nfw.py's `z=5`-default function) is
  cell 197's own diagnostic self-check plot and is never called from cell
  198 -- the caption this figure serves ("locus of $z=0$ NFW halo
  parameters") also names z=0 explicitly.
* Version: 'mass_floor_7' for all three dwarfs -- cell 198 sets `version`
  once, outside the per-dwarf loop, and every dwarf in that loop reads from
  `results/paper_contours/rho150_mpeak/mass_floor_7/`. This is the
  Msun=1e7-floor appendix run (fig:rho150_vs_Mpeak / app:mass_floor), not
  the Diemer fiducial catalog that also has a `rho150_mpeak/` tree on disk.
* The grid of NFW solutions (`rho_s_grid_lo/hi`, `r_s_grid_lo/hi`,
  `rho150_max_grid`) is computed by a block that is fully commented out in
  the saved cell 198 text, yet `rho_s_grid_lo` etc. are read, uncommented,
  later in that SAME cell (in `M_half_grid_lo = nfw_mass(rhalf,
  rho_s_grid_lo, r_s_grid_lo)` and the "No NFW solution" contour). No other
  cell in 196-198 defines these names. The saved notebook state therefore
  depended on this exact block having been run, uncommented, in a prior
  kernel execution -- this module reconstructs it uncommented, verbatim
  (same nested loop, same `z=0`, same `len(solutions)` branching), since
  that is the only definition of these arrays anywhere in the source cells,
  not a re-derivation.
* Cell 198 opens `ResampleMstar(version=version, mass_floor=8, cut='distance')`
  -- note `mass_floor=8`, independent of `version='mass_floor_7'`. Verified
  numerically inert for what this script actually reads from that object
  (`weights.logMpeak = log10(h5['mpeak'])`, per halo_weights.py): `mass_floor`
  is consulted only inside `get_mask`/weight-floor logistics, which this
  script never calls, so `logMpeak` is identical regardless of the
  `mass_floor` argument passed at construction. Reproduced as-is.
* Dead loads dropped: `mhalf_weights` (from `data/additional/weights_gc/`)
  and `surviving`/`mw_hosts_lmc`/`vmax_infall`/`rmax_infall`/`vmax`/`rmax`
  (from the h5 catalog) are read in the cell but never referenced again in
  it -- vestigial from an earlier cell version. Not replicated: they have no
  effect on the figure and would only add two unused, non-cheap reads (a
  weights .npz per dwarf and six h5 datasets over the full catalog).
* The rho150/Mpeak grid ("Build grid once (bounds are the same for all
  dwarfs)") is rebuilt with identical bounds inside the cell's per-dwarf
  loop -- same `logrho150`/`logMpeak` arrays, same linspace calls. Built once
  here and reused across all requested dwarfs, per the cell's own comment;
  this is not a numeric change, only skipping an identical recomputation.
* `colors` here is `plot_style.colors` (Okabe-Ito palette), not re-pasted --
  cell 198 defines its own copy of the same eight hex strings.

Runtime note: the shared NFW-solution grid is 100x100 = 10,000 calls to
`nfw_parameters_from_rho150_mpeak`, each of which evaluates a 2000-point
dense grid via a Python-level list comprehension before brentq (that
function lives in nfw.py, which this step does not touch) -- roughly 40-50s
total, paid once per script invocation regardless of how many dwarfs are
requested, not per dwarf.

Every figure's inputs (the h5 catalog, the version's rho150 globals, and the
four contour .npz files it reads) are recorded in a
`<figure>.prov.json` sidecar via `provenance.figure_manifest`.
"""
import argparse
import sys

import numpy as np
from astropy import units as u
from matplotlib.patches import Patch
import matplotlib.lines as mlines

from plot_style import *
import config
import provenance
import Jdata as obs
import dwarf_categories as dc
from halo_weights import ResampleMstar
from nfw import nfw_parameters_from_rho150_mpeak, nfw_mass

SCRIPT = 'python/plot_rho150_mpeak.py'

# The three dwarfs drafts_temp/temp_part1.tex's fig:rho150_vs_Mpeak requires.
DEFAULT_DWARVES = ('Draco', 'Segue 1', 'Hydrus I')
DEFAULT_VERSION = 'mass_floor_7'


def build_solution_grid(logrho150, logMpeak, n_grid=100):
    """The rho150-Mpeak grid of NFW solutions, shared across every dwarf.

    Verbatim reconstruction of the block that sits commented-out in cell 198
    (see module docstring) -- same bounds, same nested loop, same z=0, same
    branching on `len(solutions)`.
    """
    log_rho150_grid = np.linspace(logrho150.min(), logrho150.max(), n_grid)
    log_Mpeak_grid = np.linspace(logMpeak.min(), logMpeak.max(), n_grid)
    LogRho150_grid, LogMpeak_grid = np.meshgrid(log_rho150_grid, log_Mpeak_grid)

    Rho150_grid = 10 ** LogRho150_grid
    Mpeak_grid = 10 ** LogMpeak_grid

    rho_s_grid_lo = np.full_like(Rho150_grid, np.nan)
    rho_s_grid_hi = np.full_like(Rho150_grid, np.nan)
    r_s_grid_lo = np.full_like(Rho150_grid, np.nan)
    r_s_grid_hi = np.full_like(Rho150_grid, np.nan)
    rho150_max_grid = np.full_like(Rho150_grid, np.nan)

    for i in range(Rho150_grid.shape[0]):
        for j in range(Rho150_grid.shape[1]):
            solutions = nfw_parameters_from_rho150_mpeak(
                Rho150_grid[i, j], Mpeak_grid[i, j], z=0)
            if len(solutions) == 0:
                continue

            rho150_max_grid[i, j] = solutions[0]['rho150_max']

            p = solutions[0]
            rho_s_grid_lo[i, j] = p['rho_s']
            r_s_grid_lo[i, j] = p['r_s']

            if len(solutions) == 2:
                p = solutions[1]
                rho_s_grid_hi[i, j] = p['rho_s']
                r_s_grid_hi[i, j] = p['r_s']

    return {
        'Rho150_grid': Rho150_grid, 'Mpeak_grid': Mpeak_grid,
        'rho_s_grid_lo': rho_s_grid_lo, 'rho_s_grid_hi': rho_s_grid_hi,
        'r_s_grid_lo': r_s_grid_lo, 'r_s_grid_hi': r_s_grid_hi,
        'rho150_max_grid': rho150_max_grid,
    }


def make_figure(dwarf, version, grid, h5_inputs, out_dir=None, prefix=True):
    """Build and save the rho150-Mpeak figure for one dwarf.

    `out_dir`/`prefix` default to the draft location and the
    `rho150_mpeak_` basename prefix the .tex needs; the supplementary sweep
    (see main()'s --supplementary) points `out_dir` at
    plots/supplementary/rho150_mpeak/ and drops the prefix, since those
    basenames are not a draft contract.
    """
    contour_dir = config.PAPER_CONTOURS_DIR / 'rho150_mpeak' / version
    Rho150_grid = grid['Rho150_grid']
    Mpeak_grid = grid['Mpeak_grid']
    rho_s_grid_lo = grid['rho_s_grid_lo']
    rho_s_grid_hi = grid['rho_s_grid_hi']
    r_s_grid_lo = grid['r_s_grid_lo']
    r_s_grid_hi = grid['r_s_grid_hi']
    rho150_max_grid = grid['rho150_max_grid']

    test_dwarf = obs.Dwarf(dwarf)
    rhalf = test_dwarf.rhalf.to(u.kpc).value

    M_half_grid_lo = nfw_mass(rhalf, rho_s_grid_lo, r_s_grid_lo)
    M_half_grid_hi = nfw_mass(rhalf, rho_s_grid_hi, r_s_grid_hi)

    fig, ax = plt.subplots(figsize=(6, 6))

    contour_paths = {}

    unweighted_path = contour_dir / f'{dwarf}_unweighted.npz'
    mhalf_path = contour_dir / f'{dwarf}_mhalf.npz'
    F18_path = contour_dir / f'{dwarf}_F18.npz'
    K24_path = contour_dir / f'{dwarf}_K24.npz'
    provenance.assert_single_version(
        [unweighted_path, mhalf_path, F18_path, K24_path], expected=version)

    unweighted = np.load(unweighted_path)
    contour_paths['unweighted'] = unweighted_path
    X_lin = 10 ** unweighted['X_kde']
    Y_lin = 10 ** unweighted['Y_kde']
    hist = unweighted['hist_kde']
    contour_line = unweighted['contour_kde']
    ax.contourf(X_lin, Y_lin, hist.T, levels=[contour_line[0], np.max(hist)],
                colors=['tab:grey'], alpha=0.8)
    ax.contourf(X_lin, Y_lin, hist.T, levels=[contour_line[1], contour_line[0]],
                colors=['tab:grey'], alpha=0.5)
    ax.contourf(X_lin, Y_lin, hist.T, levels=[contour_line[2], contour_line[1]],
                colors=['tab:grey'], alpha=0.2)

    mhalf = np.load(mhalf_path)
    contour_paths['mhalf'] = mhalf_path
    X_lin = 10 ** mhalf['X_kde']
    Y_lin = 10 ** mhalf['Y_kde']
    hist = mhalf['hist_kde']
    contour_line = mhalf['contour_kde']
    ax.contour(X_lin, Y_lin, hist.T, levels=[contour_line], colors=['k'],
               linewidths=2, linestyles='dashed')

    F18 = np.load(F18_path)
    contour_paths['F18'] = F18_path
    X_lin = 10 ** F18['X_kde']
    Y_lin = 10 ** F18['Y_kde']
    hist = F18['hist_kde']
    contour_line = F18['contour_kde']
    ax.contour(X_lin, Y_lin, hist.T, levels=[contour_line], colors=['k'],
               linewidths=2, linestyles='dotted')

    K24 = np.load(K24_path)
    contour_paths['K24'] = K24_path
    X_lin = 10 ** K24['X_kde']
    Y_lin = 10 ** K24['Y_kde']
    hist = K24['hist_kde']
    contour_line = K24['contour_kde']
    ax.contour(X_lin, Y_lin, hist.T, levels=[contour_line], colors=['k'],
               linewidths=2, linestyles=[(0, (3, 5, 1, 5, 1, 5))])

    # -- M_half observational constraint (orange filled band) ------------
    mhalf_obs = test_dwarf.Mhalf
    mhalf_err = test_dwarf.Mhalf_err
    levels = [mhalf_obs - mhalf_err[0], mhalf_obs + mhalf_err[1]]

    ax.contourf(Rho150_grid, Mpeak_grid, M_half_grid_lo, levels=levels,
                colors=[colors[1]], alpha=0.2)
    ax.contourf(Rho150_grid, Mpeak_grid, M_half_grid_hi, levels=levels,
                colors=[colors[1]], alpha=0.2)

    ax.axhspan(ax.get_ylim()[0], 1e8, color='gray', alpha=0.08, zorder=0)
    ax.axhline(1e8, color='gray', ls='--', lw=1.2, zorder=1)
    ax.text(4.2e6, 1e8, r"Fiducial run $M_{\rm peak}$ floor",
            ha='left', va='bottom', fontsize=10, color='gray', style='italic')

    # -- Unphysical region (rho150 > rho150_max for this Mpeak) -----------
    ax.contourf(Rho150_grid, Mpeak_grid, Rho150_grid - rho150_max_grid,
                levels=[0, np.nanmax(Rho150_grid)],
                colors=[colors[6]], alpha=0.1)
    ax.contour(Rho150_grid, Mpeak_grid, Rho150_grid - rho150_max_grid,
               levels=[0], colors=[colors[6]], linewidths=1.5, linestyles='-')

    ax.text(0.97, 0.1, r"No NFW solution at $z=0$",
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=13, color=colors[6])

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(4e6, 3e9)
    ax.set_ylim(1e7, 1e11)

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    ax.set_xlabel(r'$\rho_{150}\;[\mathrm{M}_\odot\;\mathrm{kpc}^{-3}]$', fontsize=15)
    ax.set_ylabel(r'$M_{\rm peak}\;[\mathrm{M}_\odot]$', fontsize=15)

    proxy_styles = []
    proxy_styles.append(Patch(color='gray', label=r'\texttt{SatGen} - all'))
    proxy_styles.append(mlines.Line2D([], [], color='k', linewidth=1.5, linestyle='dashed', label=r'\texttt{SatGen} - $M_{1/2}$ inference'))
    proxy_styles.append(mlines.Line2D([], [], color='k', linewidth=1.5, linestyle='dotted', label=r'\texttt{SatGen} - Fattahi+18 inference'))
    proxy_styles.append(mlines.Line2D([], [], color='k', linewidth=1.5, linestyle=(0, (3, 5, 1, 5, 1, 5)), label=r'\texttt{SatGen} - Kim+24 inference'))
    proxy_styles.append(Patch(color=colors[1], label=r'Wolf+10 $M_{1/2}$ estimate', alpha=0.2))
    ax.legend(
        handles=proxy_styles, title=dwarf,
        fontsize=13, title_fontsize=13,
        loc='upper left', frameon=True, framealpha=0.85,
        borderpad=0.6, handlelength=1.5, labelspacing=0.4,
    )
    plt.tight_layout()

    out_dir = out_dir if out_dir is not None else config.PLOTS_DIR / 'rho150_mpeak'
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f'rho150_mpeak_{dwarf}.pdf' if prefix else f'{dwarf}.pdf'
    out_path = out_dir / fname
    plt.savefig(out_path)
    plt.close(fig)

    inputs = list(h5_inputs) + [
        contour_paths['unweighted'], contour_paths['mhalf'],
        contour_paths['F18'], contour_paths['K24'],
    ]
    provenance.figure_manifest(out_path, SCRIPT, inputs, version=version,
                               dwarf=dwarf)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default=DEFAULT_VERSION,
                        help=f'SatGen version (default: {DEFAULT_VERSION})')
    parser.add_argument('--dwarves', nargs='+', default=list(DEFAULT_DWARVES),
                        help=f'dwarf names (default: {DEFAULT_DWARVES})')
    parser.add_argument('--supplementary', action='store_true',
                        help='Produce the rho150-Mpeak figure for every dwarf in '
                             'dwarf_categories.idcs (the 39-dwarf paper sample), written to '
                             'plots/supplementary/rho150_mpeak/<dwarf>.pdf instead of the '
                             'three draft figures. Ignores --dwarves. A dwarf missing '
                             'required contour files is reported, not silently skipped, '
                             'and the process exits non-zero if any are missing.')
    args = parser.parse_args()

    version = args.version
    rho150_path = config.ADDITIONAL_DIR / f'{version}_rho150.npz'
    logrho150 = np.log10(np.load(rho150_path)['rho150'])

    with ResampleMstar(version=version, mass_floor=8, cut='distance') as weights:
        logMpeak = weights.logMpeak

    h5_inputs = [config.h5_path(version), rho150_path]

    grid = build_solution_grid(logrho150, logMpeak)

    if args.supplementary:
        names = [obs.dwarf_names[i] for i in dc.idcs]
        contour_dir = config.PAPER_CONTOURS_DIR / 'rho150_mpeak' / version
        out_dir = config.PLOTS_DIR / 'supplementary' / 'rho150_mpeak'
        failures = []
        produced = 0
        for dwarf in names:
            paths = [contour_dir / f'{dwarf}_{suf}.npz'
                     for suf in ('unweighted', 'mhalf', 'F18', 'K24')]
            if not all(p.exists() for p in paths):
                failures.append(dwarf)
                continue
            print(dwarf)
            out_path = make_figure(dwarf, version, grid, h5_inputs,
                                   out_dir=out_dir, prefix=False)
            print(f'  wrote {out_path}')
            produced += 1
        print(f'Produced {produced} supplementary rho150_mpeak figures.')
        if failures:
            print(f'FAILED for {len(failures)} dwarf(s) missing required contour files '
                 f'under {contour_dir}: {failures}')
            sys.exit(1)
        return

    for dwarf in args.dwarves:
        print(dwarf)
        out_path = make_figure(dwarf, version, grid, h5_inputs)
        print(f'  wrote {out_path}')


if __name__ == '__main__':
    main()
