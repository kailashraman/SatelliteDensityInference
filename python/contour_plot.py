"""Contour-label placement helpers and sci_fmt_* clabel formatters, lifted
from ../SatGen_Dwarf/jupyter/PaperPlots.ipynb cells 56 and 60.

The plan for this migration step described cells 56 and 60 as byte-identical
201-line duplicates. That is false: `shift_labels_along_contour`,
`get_shifted_label_positions`, `save_contour` and `_compute_contour_level` are
identical between the two cells, but three of the five `sci_fmt_*`
formatters differ in the LaTeX they emit --

    sci_fmt_mV:      cell 56 '$M_{0} = ...'          cell 60 '$M_{\\rm vir} = ...'
    sci_fmt_mhalf:   cell 56 '...\\ M_{\\odot}$'      cell 60 '...\\ {\\rm M}_{\\odot}$'
    sci_fmt_rho150:  cell 56 '...\\ M_{\\odot}$ ...'  cell 60 '...\\ {\\rm M}_{\\odot}$ ...'

(sci_fmt_cV and sci_fmt_J are identical in both cells.) Cell 60's wording
('M_{\\rm vir}', '{\\rm M}_{\\odot}' upright-roman) is kept here, not cell
56's -- see the stronger reasoning below the `position_on_contour` note --
flagged per the migration-step instructions rather than merged in silently.

`save_contour` / `_compute_contour_level` are NOT included here: they are
covered by the separately migrated `python/contour_io.py`.

`position_on_contour` is not defined in either cell 56 or 60 despite being on
this step's function list -- it is not a shared helper anywhere in the
notebook. It exists exactly once there, in cell 64, as a function nested
inside a plotting loop (closing over nothing from that scope; it only takes
`seg_list, y_val`). It is lifted here verbatim from cell 64, hoisted to
module scope with no behavior change. It is, however, byte-identical to a
function of the same name in
../SatGen_Dwarf/python/rmax_vmax_all_dwarfs.py, a live producer of the
rmax_vmax_*.pdf basenames the drafts reference -- so while the notebook
never factored it out as a shared helper, the migrated pipeline already had.

Cell 60's `sci_fmt_*` wording, not cell 56's, is kept here for a stronger
reason than "later and more complete": cell 56's formatters feed a cell
that saves a version-labelled exploratory contour variant not referenced by
either draft, while cell 60's feed the cells that save the basenames the
drafts DO reference. `rmax_vmax_all_dwarfs.py` independently uses cell 60's
wording too, which is corroborating rather than the primary reason.
"""

import numpy as np
import matplotlib.pyplot as plt


def shift_labels_along_contour(contours, shift_fraction=0.02, **label_style):
    """
    Shift contour labels along the contour by a fraction of its length.
    Works even if label positions are not exactly on vertices.
    """
    texts = plt.clabel(contours, **label_style)

    for txt in texts:
        x, y = txt.get_position()

        # Find nearest path and projection
        best_dist = np.inf
        best_path = None
        best_proj_idx = None
        best_proj_point = None
        best_arc_len_to_proj = None

        for col in contours.collections:
            for path in col.get_paths():
                verts = path.vertices
                segs = np.hypot(np.diff(verts[:, 0]), np.diff(verts[:, 1]))
                cumlen = np.concatenate(([0], np.cumsum(segs)))

                for i in range(len(verts) - 1):
                    p1, p2 = verts[i], verts[i + 1]
                    v = p2 - p1
                    w = np.array([x, y]) - p1
                    t = np.clip(np.dot(w, v) / np.dot(v, v), 0, 1)
                    proj = p1 + t * v
                    dist = np.hypot(*(proj - [x, y]))
                    if dist < best_dist:
                        best_dist = dist
                        best_path = verts
                        best_proj_idx = i
                        best_proj_point = proj
                        best_arc_len_to_proj = cumlen[i] + t * segs[i]

        # If we found a projection, shift along the contour
        if best_path is not None:
            segs = np.hypot(np.diff(best_path[:, 0]), np.diff(best_path[:, 1]))
            cumlen = np.concatenate(([0], np.cumsum(segs)))
            total_len = cumlen[-1]

            target_len = best_arc_len_to_proj + shift_fraction * total_len
            target_len = np.mod(target_len, total_len)

            # Find new position along path
            new_idx = np.searchsorted(cumlen, target_len) - 1
            new_idx = np.clip(new_idx, 0, len(best_path) - 2)
            t = (target_len - cumlen[new_idx]) / segs[new_idx]
            new_point = best_path[new_idx] + t * (best_path[new_idx + 1] - best_path[new_idx])

            txt.set_position(new_point)

    return texts

def get_shifted_label_positions(contours, fig, ax, shift_fraction=0.02):
    """
    Compute shifted label positions along contour paths using figure fraction coordinates.
    Returns a list of (x, y) positions in data coordinates for plt.clabel(manual=...).

    shift_fraction : float
        Fraction of the contour length in figure fraction coordinates to shift.
    """
    positions = []

    for col in contours.collections:
        for path in col.get_paths():
            verts_data = path.vertices

            # Convert vertices from data coords -> figure fraction coords
            verts_fig = ax.transData.transform(verts_data)          # data -> display (pixels)
            verts_fig = fig.transFigure.inverted().transform(verts_fig)  # display -> figure fraction

            # Compute arc length in figure fraction space
            segs = np.hypot(np.diff(verts_fig[:, 0]), np.diff(verts_fig[:, 1]))
            cumlen = np.concatenate(([0], np.cumsum(segs)))
            total_len = cumlen[-1]

            # Start from midpoint in figure fraction coords
            start_len = total_len / 2
            target_len = start_len + shift_fraction * total_len
            target_len = np.mod(target_len, total_len)  # wrap around if needed

            # Find new position in figure fraction coords
            new_idx = np.searchsorted(cumlen, target_len) - 1
            new_idx = np.clip(new_idx, 0, len(verts_fig) - 2)
            t = (target_len - cumlen[new_idx]) / segs[new_idx]
            new_point_fig = verts_fig[new_idx] + t * (verts_fig[new_idx + 1] - verts_fig[new_idx])

            # Convert back to data coordinates
            new_point_disp = fig.transFigure.transform(new_point_fig)  # figure fraction -> display
            new_point_data = ax.transData.inverted().transform(new_point_disp)  # display -> data

            positions.append(tuple(new_point_data))

    return positions

def sci_fmt_mV(val):
    exponent = int(np.floor(np.log10(val)))
    mantissa = val / 10**exponent
    return r'$M_{\rm vir} = 10^{%d}\ {\rm M}_{\odot}$' % exponent

def sci_fmt_mhalf(val, rhalf):
    exponent = int(np.floor(np.log10(val)))
    mantissa = val / 10**exponent
    return r'$M({%d}\;pc) = 10^{%d}\ {\rm M}_{\odot}$' % (int(rhalf * 1e3), exponent)

def sci_fmt_cV(val):
    exponent = int(np.floor(np.log10(val)))
    mantissa = val / 10**exponent
    return r'$c_V = 10^{%d}$' % exponent

def sci_fmt_J(val):
    exponent = int(np.floor(np.log10(val)))
    mantissa = val / 10**exponent
    return r'$J_{d=30\;{\rm kpc}}(0.5^{\circ}) = 10^{%d}$ GeV$^2$ cm$^{-5}$' % exponent

def sci_fmt_rho150(val):
    exponent = int(np.floor(np.log10(val)))
    mantissa = val / 10**exponent
    return r'$\rho(150\;{\rm pc}) = 10^{%d}\ {\rm M}_{\odot}$ kpc$^{-3}$' % exponent


def position_on_contour(seg_list, y_val):
    """Find (x, y) where a contour crosses y_val."""
    for seg in seg_list:
        if len(seg) < 2:
            continue
        ys = np.asarray(seg[:, 1], dtype=np.float64)
        xs = np.asarray(seg[:, 0], dtype=np.float64)
        diffs = ys - y_val
        # Include zeros as crossings, but exclude pairs where both are exactly zero
        sign_change = np.sign(diffs[:-1]) * np.sign(diffs[1:]) <= 0
        both_zero = (diffs[:-1] == 0) & (diffs[1:] == 0)
        idx = np.where(sign_change & ~both_zero)[0]
        if len(idx) > 0:
            k = idx[0]
            y0, y1 = ys[k], ys[k+1]
            x0, x1 = xs[k], xs[k+1]
            if y1 == y0:
                return (x0, y_val)
            t = (y_val - y0) / (y1 - y0)
            return (x0 + t * (x1 - x0), y_val)
    return None
