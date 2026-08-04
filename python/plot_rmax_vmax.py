"""Per-dwarf r_max-v_max "individual" figure, with the M_vir contour grid and
the Wolf+10 M_1/2 band overlaid on the galactocentric-weighted SatGen
contours, plus (optionally) two Jeans-analysis contours.

Ported from ../SatGen_Dwarf/python/rmax_vmax_all_dwarfs.py rather than
PaperPlots.ipynb cells 56/57->58 (no_Jeans) and 60-65->64 (with_Jeans): that
script is a live producer of exactly the `{name}{_no,with}_Jeans.pdf`
basenames this step targets (already hoisted the per-dwarf notebook cell into
a loop with a `--no-jeans` flag), and its `sci_fmt_mV` / `position_on_contour`
helpers are already confirmed byte-identical to the notebook's cell 60
versions -- see contour_plot.py's docstring, which lifted them from there.
This module imports both rather than repasting them. Cells 62 and 65
(diagnostics) are dropped per the step plan; upstream's `rmax_vmax_all_dwarfs.py`
never had them to begin with.

Path/import/provenance edits only, plus one filename change and one dropped
dead-code block -- the grid construction, contour levels, label placement,
and Jeans overlay are otherwise character-identical to upstream:

  * Output basenames gain an `rmax_vmax_` prefix (upstream saved bare
    `{name}{suffix}.pdf`); the .tex includegraphics calls need
    `rmax_vmax_<dwarf>_{no,with}_Jeans.pdf`.
  * Dropped: `jval_grid`, `cV_grid`, `rho150_grid` (and the `d`, `psimax`,
    and first-pass `rs_grid`/`a_grid` that only fed them) and the unused
    `mean_level`. Traced through upstream's control flow: `rs_grid` and
    `rhos_grid` are each assigned twice in the source cell, and only the
    SECOND assignment (`rs_grid = 0.46241029979236 * rmax_grid`,
    `rhos_grid = 1.7212... * (vmax_grid/rmax_grid)**2 / const.G`, the rho150
    parametrization) feeds the `mV_grid`/`mhalf_grid` loop that is actually
    plotted; the first assignment and everything computed from it
    (`jval_grid`, `cV_grid`, `rho150_grid` itself) is dead -- computed, never
    read again. Dropping it changes no plotted array. Flagged here per
    CLAUDE.md rather than silently trimmed.
  * `mhalf_grid_list`, a `(len(paper_selection), ny, nx)` array upstream
    built to loop over a `paper_selection` list that always held exactly one
    dwarf name, is a plain 2-D `mhalf_grid` here -- this module is already
    one-dwarf-per-call, so the wrapping list added nothing.

config constants reused: config.PAPER_CONTOURS_DIR, config.PLOTS_DIR. No new
config constant was needed.
"""
import argparse
import sys

import numpy as np
from astropy import constants as const
from astropy import units as u
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from colossus.cosmology import cosmology as co
from colossus.halo import profile_nfw

from SatGen import profiles as pf

import config
import provenance
import Jdata as obs
import dwarf_categories as dc
from plot_style import *  # noqa: F401,F403 -- rcParams side effect, plt, colors
from contour_plot import sci_fmt_mV, position_on_contour

# Colossus cosmology matching SatGen, exactly as save_contours.py sets it up
# (needed here for col_co.h, used to convert the NFWProfile's little-h
# convention back to physical units).
co.addCosmology('SatGen', flat=True, H0=pf.cfg.h * 100, Om0=pf.cfg.Om,
                Ob0=pf.cfg.Ob, sigma8=pf.cfg.s8, ns=pf.cfg.ns)
col_co = co.setCosmology('SatGen')

VERSION = 'Diemer'
REDSHIFT = 'z0'

# All four inference contours are drawn in black, distinguished by linestyle
# only. To keep them legible on tight-contour dwarfs: thinner lines +
# absolute (unscaled) dash patterns with short, tightly-repeating periods so
# the pattern still resolves on a small closed contour (default scaled
# patterns at lw=2 render near-solid). Overrides plot_style's dash patterns
# for the duration of the process, matching upstream's script-level mutation.
CONTOUR_LW = 1.2
mpl.rcParams['lines.scale_dashes'] = False
mpl.rcParams['lines.dashed_pattern'] = [4.0, 2.0]
mpl.rcParams['lines.dashdot_pattern'] = [5.0, 2.0, 1.0, 2.0]
mpl.rcParams['lines.dotted_pattern'] = [1.0, 2.0]


def _galacto_dir(version, redshift):
    return config.PAPER_CONTOURS_DIR / 'galactocentric' / redshift / version


def _jeans_paths(name):
    return [
        config.PAPER_CONTOURS_DIR / 'Jeans_loguniform' / f'{name}_jeans.npz',
        config.PAPER_CONTOURS_DIR / 'Jeans_satgen_shmr_fattahi18' / f'{name}_jeans.npz',
    ]


def _has_all_files(name, no_jeans, version, redshift):
    galacto = _galacto_dir(version, redshift)
    paths = [galacto / f'{name}_unweighted.npz', galacto / f'{name}_mhalf.npz',
             galacto / f'{name}_F18.npz']
    if not no_jeans:
        paths += _jeans_paths(name)
    return all(p.exists() for p in paths)


def plot_dwarf(name, no_jeans, version=VERSION, redshift=REDSHIFT, argv=None,
                out_dir=None, prefix=True):
    """Render and save one dwarf's r_max-v_max figure; write its provenance sidecar.

    `out_dir`/`prefix` default to the draft location and the `rmax_vmax_`
    basename prefix the .tex needs; the supplementary sweep (see
    main()'s --supplementary) points `out_dir` at
    plots/supplementary/rmax_vmax/ and drops the prefix, since those
    basenames are not a draft contract.
    """
    this_idx = int(np.where(obs.dwarf_names == name)[0][0])
    abbr = obs.abbreviations[this_idx]

    galacto = _galacto_dir(version, redshift)
    inputs = [galacto / f'{name}_unweighted.npz', galacto / f'{name}_mhalf.npz',
             galacto / f'{name}_F18.npz']
    if not no_jeans:
        inputs += _jeans_paths(name)

    provenance.assert_single_version(
        [galacto / f'{name}_unweighted.npz', galacto / f'{name}_mhalf.npz',
         galacto / f'{name}_F18.npz'], expected=version)

    fig, axs = plt.subplots(figsize=(6, 6))

    x_min = 2
    x_max = 200
    y_min = 2e-2
    y_max = 2e2 if no_jeans else 2e3
    xedges = np.linspace(np.log10(x_min), np.log10(x_max), 101)
    yedges = np.linspace(np.log10(y_min), np.log10(y_max), 101)
    X_log, Y_log = np.meshgrid(xedges, yedges)

    axs.set_xscale('log')
    axs.set_yscale('log')

    unweighted = np.load(galacto / f'{name}_unweighted.npz')
    X_lin = 10 ** unweighted['X_kde']
    Y_lin = 10 ** unweighted['Y_kde']
    hist = unweighted['hist_kde']
    contour_line = unweighted['contour_kde']
    axs.contourf(X_lin, Y_lin, hist.T, levels=[contour_line[0], np.max(hist)],
                 colors=['tab:grey'], alpha=0.8)
    axs.contourf(X_lin, Y_lin, hist.T, levels=[contour_line[1], contour_line[0]],
                 colors=['tab:grey'], alpha=0.5)
    axs.contourf(X_lin, Y_lin, hist.T, levels=[contour_line[2], contour_line[1]],
                 colors=['tab:grey'], alpha=0.2)

    mhalf_infer = np.load(galacto / f'{name}_mhalf.npz')
    X_lin = 10 ** mhalf_infer['X_kde']
    Y_lin = 10 ** mhalf_infer['Y_kde']
    hist = mhalf_infer['hist_kde']
    contour_line = mhalf_infer['contour_kde']
    axs.contour(X_lin, Y_lin, hist.T, levels=[contour_line], colors=['k'],
               linewidths=CONTOUR_LW, linestyles='solid')

    mstar = np.load(galacto / f'{name}_F18.npz')
    X_lin = 10 ** mstar['X_kde']
    Y_lin = 10 ** mstar['Y_kde']
    hist = mstar['hist_kde']
    contour_line = mstar['contour_kde']
    axs.contour(X_lin, Y_lin, hist.T, levels=[contour_line], colors=['k'],
               linewidths=CONTOUR_LW, linestyles='dashdot')

    if not no_jeans:
        jeans_path, jeans_f18_path = _jeans_paths(name)

        jeans = np.load(jeans_path)
        X_lin = 10 ** jeans['X_kde']
        Y_lin = 10 ** jeans['Y_kde']
        hist = jeans['hist_kde']
        contour_line = jeans['contour_kde']
        axs.contour(X_lin, Y_lin, hist.T, levels=[contour_line], colors=['k'],
                   linewidths=CONTOUR_LW, linestyles='dashed')

        jeans_f18 = np.load(jeans_f18_path)
        X_lin = 10 ** jeans_f18['X_kde']
        Y_lin = 10 ** jeans_f18['Y_kde']
        hist = jeans_f18['hist_kde']
        contour_line = jeans_f18['contour_kde']
        axs.contour(X_lin, Y_lin, hist.T, levels=[contour_line], colors=['k'],
                   linewidths=CONTOUR_LW, linestyles='dotted')

    axs.set_ylabel(r'$r_{\rm max}$ [kpc]', fontsize=13)
    axs.set_xlabel(r'$v_{\rm max}$ [km s$^{-1}$]', fontsize=13)
    axs.tick_params(axis='x', labelsize=13)
    axs.tick_params(axis='y', labelsize=13)

    # M_vir(v_max, r_max) grid via the rho150-style NFW parametrization
    # (rs, rhos from vmax/rmax) -- the only grid definition that feeds the
    # plotted contours; see the module docstring for the dropped dead code.
    rmax_grid = 10 ** Y_log * u.kpc
    vmax_grid = 10 ** X_log * u.km / u.s
    rs_grid = 0.46241029979236 * rmax_grid
    rhos_grid = 1.7212585601570 * (vmax_grid / rmax_grid) ** 2 / const.G
    rs_grid_kpc = rs_grid.to(u.kpc).value
    rhos_grid_msun_kpc3 = rhos_grid.to(u.Msun / u.kpc ** 3).value

    dwarf = obs.Dwarf(name)
    rhalf_kpc = dwarf.rhalf.to(u.kpc).value

    mV_grid = np.empty_like(rs_grid_kpc)
    mhalf_grid = np.zeros_like(rs_grid_kpc)
    for i in range(rs_grid_kpc.shape[0]):
        for j in range(rs_grid_kpc.shape[1]):
            rs = rs_grid_kpc[i, j]
            rhos = rhos_grid_msun_kpc3[i, j]
            profile = profile_nfw.NFWProfile(rhos=rhos / col_co.h ** 2, rs=rs * col_co.h)
            try:
                mV_grid[i, j] = profile.MDelta(0, 'vir') / col_co.h
            except Exception:
                # Scale density below the virial overdensity threshold: no
                # R_vir exists for this (low vmax, high rmax) grid corner.
                # Leave as nan; nanmin/nanmax below already expect it.
                mV_grid[i, j] = np.nan
            mhalf_grid[i, j] = profile.enclosedMass(rhalf_kpc * col_co.h) / col_co.h

    contour_color = 'gray'

    min_exp = int(np.floor(np.log10(np.nanmin(mV_grid))))
    max_exp = int(np.ceil(np.log10(np.nanmax(mV_grid))))
    contour_levels = 10. ** np.arange(min_exp, max_exp + 1)  # Exact orders of magnitude

    contours = axs.contour(10 ** X_log, 10 ** Y_log, mV_grid, levels=contour_levels,
                           colors=contour_color, linewidths=1.5, linestyles='dashdot', alpha=0.7)

    levels_to_move = {1e7, 1e8, 1e9, 1e10, 1e11}
    levels_to_skip = {1e6, 1e12, 1e13}
    y_target_fracs = {1e7: 0.30, 1e8: 0.13, 1e9: 0.21, 1e10: 0.30, 1e11: 0.45}  # fraction up from y_min, per level
    if name in ['Crater II', 'Antlia II']:
        y_target_fracs[1e7] = 0.105
    if no_jeans:
        y_target_fracs[1e7] = y_target_fracs[1e8]  # place the 10^7 label at the same height as 10^8
        y_target_fracs[1e11] = 0.55
    y_targets = {L: 10 ** (np.log10(y_min) + f * (np.log10(y_max) - np.log10(y_min)))
                for L, f in y_target_fracs.items()}

    manual_locations = []
    for level, seg_list in zip(contours.levels, contours.allsegs):
        # Kill labels for any Mvir contour below 1e7 (contours themselves still drawn).
        if level < 1e7 * (1 - 1e-3):
            continue
        if any(np.isclose(level, L, rtol=1e-3) for L in levels_to_skip):
            continue
        if any(np.isclose(level, L, rtol=1e-3) for L in levels_to_move):
            matched = next(L for L in levels_to_move if np.isclose(level, L, rtol=1e-3))
            pos = position_on_contour(seg_list, y_targets[matched])
        else:
            # "auto-like" placement: midpoint of longest segment
            longest = max((seg for seg in seg_list if len(seg) > 0), key=len, default=None)
            if longest is not None:
                mid = longest[len(longest) // 2]
                pos = (mid[0], mid[1])
            else:
                pos = None
        if pos is not None:
            manual_locations.append(pos)
    texts = axs.clabel(contours, fmt=sci_fmt_mV, colors=contour_color,
                       fontsize=13, manual=manual_locations)
    for txt in texts:
        txt.set_color("black")

    axs.set_xlim(x_min, x_max)
    axs.set_ylim(y_min, y_max)

    lower_level = 10 ** (dwarf.logMhalf - dwarf.logMhalf_err[0])
    upper_level = 10 ** (dwarf.logMhalf + dwarf.logMhalf_err[1])
    axs.contourf(10 ** X_log, 10 ** Y_log, mhalf_grid,
                levels=[lower_level, upper_level], colors=colors[1], alpha=0.2)

    _fancybox = mpl.rcParams.get('legend.fancybox', True)
    _boxstyle = "round,pad=0.4" if _fancybox else "square,pad=0.4"

    axs.text(
        0.97, 0.03, r'$M_{\rm peak}>10^8$ M$_{\odot}$',
        transform=axs.transAxes,
        fontsize=13,
        va='bottom', ha='right',
        bbox=dict(
            boxstyle=_boxstyle,
            facecolor="white",
            alpha=mpl.rcParams['legend.framealpha'] + 0.15,
            edgecolor=mpl.rcParams['legend.edgecolor'],
            linewidth=mpl.rcParams.get('legend.linewidth') or mpl.rcParams['patch.linewidth']
        )
    )

    proxy_styles = [
        Patch(color='gray', label=rf'\texttt{{SatGen}} - {abbr} distance cut'),
        mlines.Line2D([], [], color='k', linewidth=1.5, linestyle='solid', label=r'\texttt{SatGen} - $M_{1/2}$ inference'),
        mlines.Line2D([], [], color='k', linewidth=1.5, linestyle='dashdot', label=r'\texttt{SatGen} - $M_{\star}$ inference'),
    ]
    if not no_jeans:
        proxy_styles.append(mlines.Line2D([], [], color='k', linewidth=1.5, linestyle='dashed', label=r'Jeans analysis - standard prior'))
        proxy_styles.append(mlines.Line2D([], [], color='k', linewidth=1.5, linestyle='dotted', label=r'Jeans analysis - \texttt{SatGen}-informed prior'))
    proxy_styles.append(Patch(color=colors[1], label=r'Wolf+10 $M_{1/2}$ estimate', alpha=0.2))
    legend1 = axs.legend(title_fontsize=13, handles=proxy_styles, fontsize=13, loc='upper left', ncols=1, frameon=True)
    axs.add_artist(legend1)

    suffix = '_no_Jeans' if no_jeans else '_with_Jeans'
    out_dir = out_dir if out_dir is not None else config.PLOTS_DIR / 'rmax_vmax'
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f'rmax_vmax_{name}{suffix}.pdf' if prefix else f'{name}{suffix}.pdf'
    out_path = out_dir / fname
    plt.savefig(out_path)
    plt.close(fig)

    provenance.figure_manifest(
        out_path, 'python/plot_rmax_vmax.py', inputs, version=version,
        argv=list(argv) if argv is not None else None, dwarf=name,
        redshift=redshift, no_jeans=no_jeans)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dwarf', nargs='*',
                        help='Dwarf name(s) to plot; if omitted, plots every dwarf '
                             'with complete contour data for the chosen --no-jeans setting.')
    parser.add_argument('--no-jeans', action='store_true',
                        help="Skip the two Jeans-analysis contours and don't require their "
                             ".npz files; saves '..._no_Jeans.pdf' instead of '..._with_Jeans.pdf'.")
    parser.add_argument('--version', default=VERSION)
    parser.add_argument('--redshift', default=REDSHIFT, choices=('z0', 'infall'))
    parser.add_argument('--supplementary', action='store_true',
                        help='Produce BOTH the no_Jeans and with_Jeans variants for every '
                             'dwarf in dwarf_categories.idcs (the 39-dwarf paper sample), '
                             "written to plots/supplementary/rmax_vmax/<dwarf>_{no,with}_"
                             'Jeans.pdf instead of the four draft figures. Ignores any '
                             'positional dwarf args and --no-jeans. A dwarf missing required '
                             'contour files is reported, not silently skipped, and the '
                             'process exits non-zero if any are missing.')
    args = parser.parse_args()

    if args.supplementary:
        names = [obs.dwarf_names[i] for i in dc.idcs]
        out_dir = config.PLOTS_DIR / 'supplementary' / 'rmax_vmax'
        failures = []
        produced = 0
        for name in names:
            for no_jeans in (False, True):
                if not _has_all_files(name, no_jeans, args.version, args.redshift):
                    failures.append((name, no_jeans))
                    continue
                print(name, 'no_jeans' if no_jeans else 'with_jeans')
                plot_dwarf(name, no_jeans, version=args.version, redshift=args.redshift,
                          argv=sys.argv[1:], out_dir=out_dir, prefix=False)
                produced += 1
        print(f'Produced {produced} supplementary rmax_vmax figures.')
        if failures:
            print(f'FAILED for {len(failures)} (dwarf, no_jeans) combination(s) missing '
                 f'required contour files: {failures}')
            sys.exit(1)
        return

    if args.dwarf:
        names = args.dwarf
    else:
        names = [name for name in obs.dwarf_names
                 if _has_all_files(name, args.no_jeans, args.version, args.redshift)]
        skipped = [name for name in obs.dwarf_names if name not in names]
        print(f'Plotting {len(names)} dwarfs with complete data.')
        print(f'Skipping {len(skipped)} dwarfs missing one or more contour files: {skipped}')

    for name in names:
        if not _has_all_files(name, args.no_jeans, args.version, args.redshift):
            raise FileNotFoundError(
                f'{name!r} is missing one or more required contour files for '
                f'no_jeans={args.no_jeans}, version={args.version!r}, redshift={args.redshift!r}')
        print(name)
        plot_dwarf(name, args.no_jeans, version=args.version, redshift=args.redshift,
                  argv=sys.argv[1:])


if __name__ == '__main__':
    main()
