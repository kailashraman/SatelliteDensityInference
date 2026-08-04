"""Fermi reweighting plot helpers.

Scope finding (flagged, not silently resolved): the migration-step brief for
this module said all seven functions listed below are "already single-copy
in ../SatGen_Dwarf/python/plot_fermi_reweighting.py -- take them from there,
not from the notebook." That is true for only four of the seven:

    HandlerRectangle, SplitPatchHandler, shade_bounding_box_to_zero,
    find_thermal_crossing

are defined in plot_fermi_reweighting.py (condensed docstrings relative to
the notebook, but the same computation) and are taken from there verbatim.

    plot_ellipse_from_csv, ul_from_stack, positive_preference

do NOT appear anywhere in plot_fermi_reweighting.py -- grepping the whole
../SatGen_Dwarf tree finds them only in the notebook itself (PaperPlots.ipynb
cells 183, 181, 180 respectively). plot_fermi_reweighting.py's own docstring
says it covers the notebook's cells 169-183 but only reconstructs four named
figures from them; these three functions back exploratory/diagnostic prints
in cells 180/181/183 (a "which dwarfs prefer positive sigma*v" summary and a
per-draw stacked-TS 68% band helper) that were apparently not carried into
the migrated figure script. They are lifted here directly from those three
notebook cells instead, verbatim apart from imports.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
from scipy.interpolate import interp1d


# ── from ../SatGen_Dwarf/python/plot_fermi_reweighting.py ──────────────────

class HandlerRectangle(mpl.legend_handler.HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        x = orig_handle.get_x() * width
        y = orig_handle.get_y() * height
        w = orig_handle.get_width() * width
        h = orig_handle.get_height() * height
        patch = Rectangle((x - xdescent, y - ydescent), w, h,
                          facecolor=orig_handle.get_facecolor(),
                          edgecolor='none',
                          alpha=orig_handle.get_alpha(),
                          transform=trans)
        return [patch]


class SplitPatchHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        if isinstance(orig_handle[0], Rectangle):
            h0, h1 = orig_handle  # h0=gold (outer), h1=green (inner)
            outer = plt.Rectangle([xdescent, ydescent], width, height,
                                  facecolor=h0.get_facecolor(),
                                  edgecolor=h0.get_edgecolor(),
                                  transform=trans)
            inner_frac = h1.get_height() / h0.get_height()
            inner_height = height * inner_frac
            inner_y = ydescent + (height - inner_height) / 2
            inner = plt.Rectangle([xdescent, inner_y], width, inner_height,
                                  facecolor=h1.get_facecolor(),
                                  edgecolor=h1.get_edgecolor(),
                                  transform=trans)
            return [outer, inner]
        else:
            # tuple style: (top_color, top_alpha, bot_color, bot_alpha)
            top_fc, top_alpha, bot_fc, bot_alpha = orig_handle
            top = plt.Rectangle([xdescent, ydescent + height / 2], width, height / 2,
                                facecolor=top_fc, alpha=top_alpha, transform=trans)
            bottom = plt.Rectangle([xdescent, ydescent], width, height / 2,
                                   facecolor=bot_fc, alpha=bot_alpha, transform=trans)
            return [top, bottom]


def shade_bounding_box_to_zero(csv_paths, ax=None, top_alpha=0.25, bottom_alpha=0.1,
                               top_color='tab:grey', bottom_color='steelblue', **fill_kwargs):
    """Shade the bounding box of the literature GCE points down to y=0."""
    all_pts = []
    for path in csv_paths:
        df = pd.read_csv(path)
        all_pts.append(df.iloc[:, :2].to_numpy(dtype=float))
    pts = np.vstack(all_pts)

    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()

    if ax is None:
        ax = plt.gca()
    fill_kwargs.setdefault("linewidth", 0)
    ax.fill_between([xmin, xmax], ymin, ymax, alpha=top_alpha, color=top_color, **fill_kwargs)
    ax.fill_between([xmin, xmax], 0, ymin, alpha=bottom_alpha, color=bottom_color, **fill_kwargs)
    return ax, {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}


def find_thermal_crossing(mass_vec, line_values, sigma_thermal, min_mass=10.0, eval_mass=40.0):
    """Where a limit crosses the thermal relic curve, and the exclusion factor
    at `eval_mass`."""
    mass_vec = np.asarray(mass_vec)
    line_values = np.asarray(line_values)

    mask = mass_vec >= min_mass
    mass_vec_cut = mass_vec[mask]
    line_values_cut = line_values[mask]

    cross_mass, cross_sigmav = None, None
    if len(mass_vec_cut) == 0:
        print(f"No mass points found above {min_mass} GeV.")
    else:
        thermal_on_grid = 10**np.interp(np.log10(mass_vec_cut),
                                        np.log10(sigma_thermal[:, 0]),
                                        np.log10(sigma_thermal[:, 1]))
        diff = line_values_cut - thermal_on_grid
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) == 0:
            print(f"No crossing found between the line and thermal relic above {min_mass} GeV.")
        else:
            i = sign_changes[0]
            log_m0, log_m1 = np.log10(mass_vec_cut[i]), np.log10(mass_vec_cut[i + 1])
            log_d0, log_d1 = diff[i], diff[i + 1]
            frac = log_d0 / (log_d0 - log_d1)
            log_cross_mass = log_m0 + frac * (log_m1 - log_m0)
            cross_mass = 10**log_cross_mass
            cross_sigmav = 10**np.interp(log_cross_mass,
                                         np.log10(sigma_thermal[:, 0]),
                                         np.log10(sigma_thermal[:, 1]))

    excl_factor = None
    if eval_mass < mass_vec.min() or eval_mass > mass_vec.max():
        print(f"eval_mass {eval_mass} GeV is outside mass_vec range "
              f"[{mass_vec.min()}, {mass_vec.max()}] GeV.")
    else:
        line_at_eval = 10**np.interp(np.log10(eval_mass), np.log10(mass_vec),
                                     np.log10(line_values))
        thermal_at_eval = 10**np.interp(np.log10(eval_mass),
                                        np.log10(sigma_thermal[:, 0]),
                                        np.log10(sigma_thermal[:, 1]))
        excl_factor = line_at_eval / thermal_at_eval

    return cross_mass, cross_sigmav, excl_factor


# ── notebook-only (cells 180, 181, 183; not in plot_fermi_reweighting.py) ──

def plot_ellipse_from_csv(csv_path, ax=None, close=True, center="bbox",
                          logspace=True, **plot_kwargs):
    """
    Load (possibly unordered) ellipse boundary points and plot them.

    Parameters
    ----------
    csv_path : str or Path
        CSV with two columns (x, y).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Defaults to plt.gca().
    close : bool, default True
        Append first point at end to close the curve.
    center : {"bbox", "centroid"} or (x, y) tuple, default "bbox"
        How to pick the reference point for angle sorting (computed in
        log space if logspace=True, except when given explicitly as a tuple,
        which is taken as-is in the sorting space).
    logspace : bool, default True
        If True, compute the center and sort angles in log10 space.
        Points must be strictly positive. The plot itself is still in
        linear coordinates — set ax.set_xscale('log'), ax.set_yscale('log')
        externally if you want log axes.
    **plot_kwargs
        Forwarded to ax.plot.
    """
    df = pd.read_csv(csv_path)
    pts = df.iloc[:, :2].to_numpy(dtype=float)

    if logspace:
        if np.any(pts <= 0):
            raise ValueError("logspace=True requires strictly positive x and y.")
        sort_pts = np.log10(pts)
    else:
        sort_pts = pts

    if center == "bbox":
        cx = 0.5 * (sort_pts[:, 0].min() + sort_pts[:, 0].max())
        cy = 0.5 * (sort_pts[:, 1].min() + sort_pts[:, 1].max())
    elif center == "centroid":
        cx, cy = sort_pts.mean(axis=0)
    else:
        cx, cy = center  # interpreted in sort_pts space

    angles = np.arctan2(sort_pts[:, 1] - cy, sort_pts[:, 0] - cx)
    order = np.argsort(angles)
    pts_sorted = pts[order]  # keep original (linear) coords for plotting

    if close:
        pts_sorted = np.vstack([pts_sorted, pts_sorted[0]])

    if ax is None:
        ax = plt.gca()
    ax.plot(pts_sorted[:, 0], pts_sorted[:, 1], **plot_kwargs)
    return ax, pts_sorted


def ul_from_stack(TS_row, sigmav_vec, coeff=2.71):
    # TS_row: shape (n_sv,) — stacked TS vs σv at a single mass
    peak = TS_row.max()
    i0 = TS_row.argmax()
    f = interp1d(TS_row[i0:], sigmav_vec[i0:],
                 bounds_error=False, fill_value=np.nan)
    return float(f(peak - coeff))


def positive_preference(ts_row, sigmav_vec, ts_threshold=0.1):
    """
    Returns (sv_best, ts_best) if the dwarf prefers positive sigmav at
    this mass, else None.
    """
    ts_max = np.max(ts_row)
    i_max = int(np.argmax(ts_row))
    if ts_max > ts_threshold and i_max > 0:
        return sigmav_vec[i_max], ts_max
    return None
