"""V_circ-r_1/2 power-law posteriors: `powerlaw_posteriors.pdf` (corner plot
of the rescaled log_alpha_100/beta posteriors) and `powerlaw_fit.pdf` (the
power-law fit overlaid on the ultra-faint data and the SatGen LMC-analog
median curve).

Migrated from ../SatGen_Dwarf/jupyter/PaperPlots.ipynb cells 120-131. Path
edits, config-derived directories, and provenance stamping only -- the model
(priors, likelihood, parameterisation) and the PyMC sampler call (3000 draws,
1000 tune, 4 chains, target_accept=0.95) are character-identical to cell 127.

THIS STEP DIFFERS FROM EVERY OTHER MIGRATED FIGURE: it runs a PyMC NUTS
sampler, so it cannot be verified by exact array comparison against the
cells' output, only by posterior mean/sd/N_eff/r_hat agreement (see the
verification report accompanying this migration).

Seeding: upstream cell 127 ALREADY calls `pm.sample(..., random_seed=42)` for
every fit -- this is not a seed added during migration (contrast
python/vcirc_scatter_300pc.py, which does add one upstream never had).
RANDOM_SEED below is that same literal 42, preserved verbatim. A seed alone
does not make the sampler reproducible across environments, though: NUTS's
compiled step function depends on the installed pymc/pytensor/numpy/scipy
versions, so a rerun under different versions is not guaranteed to reproduce
this figure even with the same seed. See docs/pymc-environment.md, which
pins the versions this migration ran under; a future rerun must match them
to reproduce the published figure, and this module's provenance sidecars
record both the seed and the installed versions for exactly that reason.

Scope narrowed relative to the cells, matching what these two figures
(and only these two) consume:
  * Cell 127 also fits Moster18 and Danieli23_const theory curves
    (flat_m18/trace_m18, flat_d/trace_d). Neither feeds `powerlaw_posteriors`
    or `powerlaw_fit` (cell 131 only ever reads flat_m/flat_f/flat_k) -- so
    those two fits, and the cell-126 machinery that only exists to feed them
    (MOSTER18_IDX/_W_IDX, DANIELI23C_IDX/_W_IDX, vcirc_sigma_dex_m/_d and the
    scatter_300pc.txt reads that build them, ys_m/ys_d and their per-dwarf
    vmed arrays), are dropped here. This is a scope decision, not a
    numerics change: it removes two of five PyMC runs that this step's two
    output figures never read.
  * Cell 131 also saves a third figure, `powerlaw_beta_posterior.pdf`
    (a plain beta histogram). It is not referenced by either draft's .tex
    (only `powerlaw_posteriors.pdf` and `powerlaw_fit.pdf` are, in
    fig:powerlaw of temp_part1.tex) and is out of scope for this step, so it
    is not produced here.
  * Cell 124's `weights = halo_weights.ResampleMstar(version=version)` /
    `Fattahi18 = weights.evolved_Mstar(...)` is dead code even upstream --
    the one line that would consume it,
    `# mask = np.logical_and(mask, Fattahi18 > 2.45)`, is commented out --
    so it is not reproduced here.
  * `markers` (cell 131) is only ever indexed at `markers[0]`; kept here as
    the literal `'s'` rather than a six-entry list with five unused entries.
    `extra_colors` (cell 131) is never read anywhere in cell 131 (including
    the dropped `powerlaw_beta_posterior.pdf` block); not reproduced.
    `colors` (cell 131's locally re-pasted Okabe-Ito palette) is byte-identical
    to `plot_style.colors`, so this module imports it from there instead of
    re-pasting it, per the repository convention.

Cell 124's LMC-analog mask (`_lmc_mask`) is reproduced exactly: for each MW
host, remove its four most-massive halos (`argsort(...)[-4:]`), then keep only
halos hosted by a MW with an LMC-analog subhalo (M_peak > 1e11 Msun within
100 kpc). This is NOT the same selection as
`ResampleMstar.get_mask(lmc_selection=True)`, which removes only the single
most-massive halo per host and additionally folds in the per-dwarf distance
cut and mass floor -- the two do not agree, so this module does not call
`get_mask` and reimplements cell 124's standalone mask instead.

config constants reused: config.PAPER_MASSPROFILE_DIR, config.WEIGHTS_DIR,
config.PAPER_QUANTILES_DIR, config.VCIRC_SCATTER_DIR, config.PLOTS_DIR. No new
config constant was needed.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from astropy import units as u
from scipy.interpolate import interp1d

import arviz
import pymc as pm
import pytensor
import pytensor.tensor as pt
import scipy
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import corner

import config
import provenance
import Jdata as obs
import halo_weights
import dwarf_categories as dc
from plot_style import *  # noqa: F401,F403 -- rcParams side effect, plt, colors
from weighted_stats import quantile

VERSION = 'Diemer'

# PyMC sampler settings -- character-identical to cell 127's `pm.sample(3000,
# tune=1000, chains=4, target_accept=0.95, random_seed=42)`, called unchanged
# for both fit_pymc and fit_pymc_theory.
N_DRAWS = 3000
N_TUNE = 1000
N_CHAINS = 4
TARGET_ACCEPT = 0.95
RANDOM_SEED = 42  # upstream's own literal; not added by this migration -- see module docstring

hist_color = '#BBAA77'  # khaki, cell 131

PACKAGE_VERSIONS = {
    'pymc': pm.__version__,
    'arviz': arviz.__version__,
    'pytensor': pytensor.__version__,
    'numpy': np.__version__,
    'scipy': scipy.__version__,
}


def _lmc_mask(version):
    """Cell 124: halos hosted by an LMC-analog MW, minus each host's four
    most-massive halos."""
    sats = halo_weights.get_h5(version)
    mpeak = sats['Green_params'][()][:, 0]
    mwID = sats['mwID'][()]
    position = sats['position'][()]
    gc_distance = np.linalg.norm(position, axis=1)

    lmc_dist_cut = gc_distance < 100
    lmc_mass_cut = mpeak > 1e11
    lmc_mwIDs = mwID[lmc_dist_cut & lmc_mass_cut]
    mw_hosts_lmc = np.isin(mwID, lmc_mwIDs)

    remove_lmc_mask = np.ones(len(mpeak), dtype=bool)
    for mw in np.unique(mwID):
        mw_inds = np.where(mwID == mw)[0]
        four_most_massive_indices = mw_inds[np.argsort(mpeak[mw_inds])[-4:]]
        remove_lmc_mask[four_most_massive_indices] = False

    return mw_hosts_lmc & remove_lmc_mask


def _load_satgen_curve(version, mask):
    """Cell 124: masked vcircProfile, its unweighted quantiles, and the
    interpolated median curve (`vcirc_median_interp`) used by the fit figure."""
    massprofile_path = config.PAPER_MASSPROFILE_DIR / version / 'massProfile.npz'
    arr = np.load(massprofile_path)
    r_grid = arr['r_grid']
    vcircProfile = arr['vcircProfile'][mask]

    quantiles = np.array([0.0025, 0.025, 0.16, 0.5, 0.84, 0.975, 0.9975])
    vcirc_quantiles = np.zeros((len(r_grid), len(quantiles)))
    for j in range(len(r_grid)):
        vcirc_quantiles[j] = quantile(vcircProfile[:, j], quantiles=quantiles)
    vcirc_median_interp = interp1d(r_grid, vcirc_quantiles[:, 3])  # index 3 == 0.5 quantile

    return r_grid, vcircProfile, vcirc_median_interp, massprofile_path


def _weighted_median_per_r(profile, weights):
    """Cell 126. Weighted median of profile (N_halos, N_r) at each r."""
    N_r = profile.shape[1]
    medians = np.zeros(N_r)
    for j in range(N_r):
        medians[j] = quantile(profile[:, j], weights=weights, quantiles=0.5)
    return medians


def _collect_data(version):
    """Cell 126, narrowed to the Fattahi18/Kim24 theory curves and the
    measured Vcirc-rhalf data -- see the module docstring for what was
    dropped (Moster18/Danieli23_const feed no output of this step)."""
    mask = _lmc_mask(version)
    r_grid, vcircProfile, vcirc_median_interp, massprofile_path = _load_satgen_curve(version, mask)

    FATTAHI_W_IDX, KIM_W_IDX = 3, 9  # mstar_weights index
    FATTAHI_IDX, KIM_IDX = 4, 10  # dispersion_quantiles index (= 1 + mstar_weights index)

    scatter_f_path = config.VCIRC_SCATTER_DIR / version / 'Fattahi18' / 'scatter_300pc.txt'
    scatter_k_path = config.VCIRC_SCATTER_DIR / version / 'Kim24' / 'scatter_300pc.txt'
    scatter_df_f = pd.read_csv(scatter_f_path, comment='#')
    scatter_df_k = pd.read_csv(scatter_k_path, comment='#')
    vcirc_sigma_dex_f = dict(zip(scatter_df_f['dwarf_name'], scatter_df_f['sigma_dex']))
    vcirc_sigma_dex_k = dict(zip(scatter_df_k['dwarf_name'], scatter_df_k['sigma_dex']))

    xs, ys, log_xerr_list, log_yerr_list = [], [], [], []
    ys_f, log_yerr_f = [], []
    ys_k, log_yerr_k = [], []
    log_vmed_f_per_dwarf, log_vmed_k_per_dwarf = [], []
    inputs = [massprofile_path, scatter_f_path, scatter_k_path]

    for name in dc.dwarf_names:
        if name in dc.dwarf_names[dc.problem_idcs]:
            continue
        if name in dc.dwarf_names[dc.classical_idcs]:
            continue

        dwarf = obs.Dwarf(name)
        if dwarf.rhalf.to(u.pc).value > 300:
            continue

        x = dwarf.rhalf.to(u.kpc).value
        xerr = dwarf.rhalf_err.to(u.kpc).value
        y = dwarf.Vcirc.value
        yerr = dwarf.Vcirc_err.value

        log_xerr_i = 0.5 * (np.log10(x + xerr[1]) - np.log10(x - xerr[0]))
        log_yerr_i = 0.5 * (np.log10(y + yerr[1]) - np.log10(y - yerr[0]))

        quantiles_path = config.PAPER_QUANTILES_DIR / 'galactocentric' / version / f'{name}.npz'
        quantiles_file = np.load(quantiles_path)
        dispersion_quantiles = quantiles_file['dispersion_quantiles']
        dispersion_50 = dispersion_quantiles[1]

        f_med = np.sqrt(3) * dispersion_50[FATTAHI_IDX]
        k_med = np.sqrt(3) * dispersion_50[KIM_IDX]
        log_yerr_f_i = vcirc_sigma_dex_f[name]
        log_yerr_k_i = vcirc_sigma_dex_k[name]

        weights_path = config.WEIGHTS_DIR / version / f'{name}.npz'
        weights_file = np.load(weights_path)
        w_f = weights_file['mstar_weights'][FATTAHI_W_IDX][mask]
        w_k = weights_file['mstar_weights'][KIM_W_IDX][mask]

        vmed_f = _weighted_median_per_r(vcircProfile, w_f)
        vmed_k = _weighted_median_per_r(vcircProfile, w_k)
        vmed_f = np.where(vmed_f > 0, vmed_f, np.nan)
        vmed_k = np.where(vmed_k > 0, vmed_k, np.nan)

        log_vmed_f_per_dwarf.append(np.log10(vmed_f))
        log_vmed_k_per_dwarf.append(np.log10(vmed_k))

        xs.append(x)
        ys.append(y)
        log_xerr_list.append(log_xerr_i)
        log_yerr_list.append(log_yerr_i)
        ys_f.append(f_med)
        log_yerr_f.append(log_yerr_f_i)
        ys_k.append(k_med)
        log_yerr_k.append(log_yerr_k_i)

        inputs.append(quantiles_path)
        inputs.append(weights_path)

    xs = np.array(xs)
    ys = np.array(ys)
    log_x = np.log10(xs)
    log_y = np.log10(ys)
    log_xerr = np.array(log_xerr_list)
    log_yerr = np.array(log_yerr_list)
    ys_f = np.array(ys_f)
    ys_k = np.array(ys_k)
    log_yerr_f = np.array(log_yerr_f)
    log_yerr_k = np.array(log_yerr_k)
    log_vmed_f_per_dwarf = np.array(log_vmed_f_per_dwarf)
    log_vmed_k_per_dwarf = np.array(log_vmed_k_per_dwarf)

    return dict(
        xs=xs, ys=ys, log_x=log_x, log_y=log_y, log_xerr=log_xerr, log_yerr=log_yerr,
        log_yerr_f=log_yerr_f, log_yerr_k=log_yerr_k,
        log_vmed_f_per_dwarf=log_vmed_f_per_dwarf, log_vmed_k_per_dwarf=log_vmed_k_per_dwarf,
        r_grid=r_grid, vcirc_median_interp=vcirc_median_interp, inputs=inputs,
    )


def _pt_interp1d(x_query, x_grid, y_grid):
    """Cell 127. Piecewise-linear interpolation along axis 1 of y_grid, per row."""
    x_grid_pt = pt.as_tensor_variable(x_grid)
    M = x_grid.shape[0]
    idx = pt.extra_ops.searchsorted(x_grid_pt, x_query)
    idx = pt.clip(idx, 1, M - 1)
    x0 = x_grid_pt[idx - 1]
    x1 = x_grid_pt[idx]
    rows = pt.arange(y_grid.shape[0])
    y0 = y_grid[rows, idx - 1]
    y1 = y_grid[rows, idx]
    t = (x_query - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _summary_dict(summary):
    cols = ['mean', 'sd', 'ess_bulk', 'r_hat']
    return {param: {c: float(summary.loc[param, c]) for c in cols} for param in summary.index}


def _fit_pymc(log_x, log_y, log_xerr, log_yerr, label=""):
    """Cell 127's fit_pymc: measured Vcirc/rhalf, observed directly."""
    N = len(log_x)
    with pm.Model():
        log_alpha = pm.Uniform('log_alpha', lower=0, upper=2)
        beta = pm.Uniform('beta', lower=-2, upper=2)
        log_sigma_int = pm.Uniform('log_sigma_int', lower=-2, upper=0)
        sigma_int = 10 ** log_sigma_int

        log_x_true = pm.Normal('log_x_true', mu=log_x, sigma=log_xerr, shape=N)

        log_y_pred = log_alpha + beta * log_x_true
        sigma_y_tot = pm.math.sqrt(log_yerr ** 2 + sigma_int ** 2)
        pm.Normal('y_obs', mu=log_y_pred, sigma=sigma_y_tot, observed=log_y)

        trace = pm.sample(N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
                          target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED)

    summary = arviz.summary(trace, var_names=['log_alpha', 'beta', 'log_sigma_int'])
    print(f"\n-- {label} diagnostics --")
    print(summary)

    log_alpha_s = trace.posterior['log_alpha'].values.flatten()
    beta_s = trace.posterior['beta'].values.flatten()
    sigma_int_s = 10 ** trace.posterior['log_sigma_int'].values.flatten()
    flat = np.column_stack([log_alpha_s, beta_s, sigma_int_s])
    return flat, summary


def _fit_pymc_theory(log_x, log_xerr, log_yerr, log_vmed_grid, log_r_grid, label=""):
    """Cell 127's fit_pymc_theory: the weighted-median theory curve stands in
    for the observed log_y, with the same likelihood form."""
    N = len(log_x)
    with pm.Model():
        log_alpha = pm.Uniform('log_alpha', lower=0, upper=2)
        beta = pm.Uniform('beta', lower=-2, upper=2)
        log_sigma_int = pm.Uniform('log_sigma_int', lower=-2, upper=0)
        sigma_int = 10 ** log_sigma_int

        log_x_true = pm.Normal('log_x_true', mu=log_x, sigma=log_xerr, shape=N)

        log_y_theory = _pt_interp1d(log_x_true, log_r_grid, pt.as_tensor_variable(log_vmed_grid))

        log_y_pred = log_alpha + beta * log_x_true
        sigma_y_tot = pm.math.sqrt(log_yerr ** 2 + sigma_int ** 2)

        loglik = -0.5 * pt.sum(((log_y_theory - log_y_pred) / sigma_y_tot) ** 2
                               + pt.log(2 * np.pi * sigma_y_tot ** 2))
        pm.Potential('loglik', loglik)

        trace = pm.sample(N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
                          target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED)

    summary = arviz.summary(trace, var_names=['log_alpha', 'beta', 'log_sigma_int'])
    print(f"\n-- {label} diagnostics --")
    print(summary)

    log_alpha_s = trace.posterior['log_alpha'].values.flatten()
    beta_s = trace.posterior['beta'].values.flatten()
    sigma_int_s = 10 ** trace.posterior['log_sigma_int'].values.flatten()
    flat = np.column_stack([log_alpha_s, beta_s, sigma_int_s])
    return flat, summary


def _rescale_to_100pc(flat):
    """Cells 130/131: log alpha_100 = log alpha + beta * log10(0.1) = log alpha - beta."""
    out = flat.copy()
    out[:, 0] = flat[:, 0] - flat[:, 1]
    return out


def make_posteriors_figure(flat_m, flat_f, flat_k, out_dir):
    """Cell 131, first block: `powerlaw_posteriors.pdf`, the (log alpha_100,
    beta) corner plot."""
    flat_m_r = _rescale_to_100pc(flat_m)
    flat_f_r = _rescale_to_100pc(flat_f)
    flat_k_r = _rescale_to_100pc(flat_k)

    corner_labels = [r"$\log_{10} \frac{\alpha_{100}}{{\rm km\;s}^{-1}}$", r"$\beta$"]
    ndim = 2
    fig = plt.figure(figsize=(6, 6))

    corner_colors = [hist_color, colors[2], colors[3]]
    dataset_labels = ["Ultra-faint\nmeasurements", "Fattahi+18\nprediction", "Kim+24\nprediction"]
    datasets = [(flat_m_r[:, :2], dataset_labels[0]), (flat_f_r[:, :2], dataset_labels[1]),
                (flat_k_r[:, :2], dataset_labels[2])]
    hist_alphas = [0.7, 0.4, 0.4]

    for i, (flat, label) in enumerate(datasets):
        c = corner_colors[i]
        rgb = mcolors.to_rgb(c)
        corner.corner(
            flat,
            labels=corner_labels if i == 0 else None,
            quantiles=None,
            levels=[0.68, 0.95],
            show_titles=False,
            label_kwargs={"fontsize": 15},
            plot_density=False,
            hist2d_kwargs={"bins": 100, "smooth": 2.0},
            color=c,
            fill_contours=True,
            contourf_kwargs={
                "colors": [(*rgb, 0.0), (*rgb, 0.2), (*rgb, 0.5)],
            },
            contour_kwargs={"linewidths": 1.2},
            hist_kwargs={
                "histtype": "stepfilled",
                "color": c,
                "alpha": hist_alphas[i],
                "density": True,
                "edgecolor": c,
            },
            fig=fig,
        )

    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=15)

    axes = np.array(fig.get_axes()).reshape(ndim, ndim)
    handles = [Line2D([0], [0], color=c, lw=3, label=l)
               for c, l in zip(corner_colors, dataset_labels)]
    axes[0, ndim - 1].legend(handles=handles, fontsize=15, frameon=False, loc="center")

    out_path = out_dir / 'powerlaw_posteriors.pdf'
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_fit_figure(data, flat_m, flat_f, flat_k, out_dir):
    """Cell 131, third block: `powerlaw_fit.pdf`, the power-law fit overlaid
    on the ultra-faint data and the SatGen LMC-analog median curve."""
    fig, ax = plt.subplots(figsize=(6, 6))

    x_grid = np.linspace(np.log10(20e-3), np.log10(300e-3), 200)
    x_grid_lin = 10 ** x_grid

    for flat, color, label, zorder in [
        (flat_m, hist_color, "Best fit to Ultra-faint measurements", 10),
        (flat_f, colors[2], "Best fit to Fattahi+18 predictions", 2),
        (flat_k, colors[3], "Best fit to Kim+24 predictions", 2),
    ]:
        log_y_hat = np.array([a + b * x_grid for a, b, _ in flat])
        lo2, lo1, med, hi1, hi2 = np.percentile(log_y_hat, [2.5, 16, 50, 84, 97.5], axis=0)
        ax.fill_between(x_grid_lin, 10 ** lo2, 10 ** hi2, color=color, alpha=0.1, zorder=zorder)
        ax.fill_between(x_grid_lin, 10 ** lo1, 10 ** hi1, color=color, alpha=0.2, zorder=zorder)
        ax.plot(x_grid_lin, 10 ** med, color=color, lw=2, ls="-", label=label, zorder=zorder)

    log_x, log_xerr = data['log_x'], data['log_xerr']
    log_y, log_yerr = data['log_y'], data['log_yerr']
    xerr_asym = np.array([[10 ** lx - 10 ** (lx - le), 10 ** (lx + le) - 10 ** lx]
                          for lx, le in zip(log_x, log_xerr)]).T
    yerr_asym = np.array([[10 ** ly - 10 ** (ly - le), 10 ** (ly + le) - 10 ** ly]
                          for ly, le in zip(log_y, log_yerr)]).T

    ax.errorbar(data['xs'], data['ys'], xerr=xerr_asym, yerr=yerr_asym,
               fmt='s', markersize=5, capsize=2, mec=colors[5], mfc='white',
               c=colors[5], lw=1, zorder=1000, label='Ultra-faints')

    ax.plot(x_grid_lin, data['vcirc_median_interp'](x_grid_lin), color="k", lw=2,
           ls="-", label=r"\texttt{SatGen} with LMC analog")

    ax.text(
        0.97, 0.03, r'$M_{\rm peak}>10^8$ M$_{\odot}$',
        transform=ax.transAxes,
        fontsize=15,
        va='bottom', ha='right',
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r$ [kpc]", fontsize=15)
    ax.set_ylabel(r"$V_{\rm circ}(r)$ [km s$^{-1}$]", fontsize=15)
    ax.set_xlim(20e-3, 300e-3)
    ax.set_ylim(2, 30)
    ax.legend(fontsize=15, frameon=False, loc='upper left')
    ax.tick_params(axis="both", which="both", labelsize=15)
    plt.tight_layout()

    out_path = out_dir / 'powerlaw_fit.pdf'
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default=VERSION)
    args = parser.parse_args()

    data = _collect_data(args.version)
    log_r_grid = np.log10(data['r_grid']).astype('float64')

    flat_m, summary_m = _fit_pymc(data['log_x'], data['log_y'], data['log_xerr'],
                                  data['log_yerr'], label="Measured")
    flat_f, summary_f = _fit_pymc_theory(data['log_x'], data['log_xerr'], data['log_yerr_f'],
                                        data['log_vmed_f_per_dwarf'], log_r_grid, label="Fattahi")
    flat_k, summary_k = _fit_pymc_theory(data['log_x'], data['log_xerr'], data['log_yerr_k'],
                                        data['log_vmed_k_per_dwarf'], log_r_grid, label="Kim")

    diagnostics = {
        'measured': _summary_dict(summary_m),
        'fattahi18': _summary_dict(summary_f),
        'kim24': _summary_dict(summary_k),
    }

    out_dir = config.PLOTS_DIR / 'powerlaw'
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = [config.h5_path(args.version)] + data['inputs']
    provenance.assert_single_version(inputs, expected=args.version)
    manifest_extra = dict(
        argv=sys.argv[1:],
        pymc_random_seed=RANDOM_SEED,
        pymc_sampler_kwargs=dict(draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
                                 target_accept=TARGET_ACCEPT),
        package_versions=PACKAGE_VERSIONS,
        sampler_diagnostics=diagnostics,
    )

    posteriors_path = make_posteriors_figure(flat_m, flat_f, flat_k, out_dir)
    provenance.figure_manifest(posteriors_path, 'python/plot_powerlaw.py', inputs,
                               version=args.version, **manifest_extra)
    print('wrote', posteriors_path)

    fit_path = make_fit_figure(data, flat_m, flat_f, flat_k, out_dir)
    provenance.figure_manifest(fit_path, 'python/plot_powerlaw.py', inputs,
                               version=args.version, **manifest_extra)
    print('wrote', fit_path)


if __name__ == '__main__':
    main()
