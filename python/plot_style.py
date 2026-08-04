"""Shared matplotlib rcParams for the paper figures.

Reproduces ../SatGen_Dwarf/jupyter/PaperPlots.ipynb cells 1 and 2, IN THAT
ORDER: cell 1's `plt.rc(...)` calls, then cell 2's `mpl.rcParams[...] = ...`
assignments. rcParams are last-write-wins, so where both cells set the same
key, cell 2's value is what the notebook actually plots with even though
cell 1 sets it first. This module applies cell 1's calls and then cell 2's,
reproducing that net state -- it is not a merge, and the conflicts below are
not resolved by picking a "better" value.

Where the two cells set the SAME rcParam to DIFFERENT values, cell 2 wins
(it runs second in the notebook, and does here too):

    rcParam            cell 1                          cell 2 (wins)
    -----------------  ------------------------------  --------------------------------------
    axes.prop_cycle    Okabe-Ito 8-colour cycle         cornflowerblue/forestgreen/maroon/...
    font.family         'serif'                          'DejaVu Sans'
    figure.figsize       [2.7, 2.7]                        [12, 6]
    lines.linewidth        1                                1.5
    legend.frameon           False                            True

`text.usetex` is listed in the migration brief as a conflict to check, but it
is NOT one: cell 1's `plt.rc('text', usetex=False)` line is commented out in
the notebook, so cell 1 never actually sets it. Only cell 2 sets it (to
True) -- recorded here because it looked like a conflict from the key name
alone, and wasn't once the cell was actually read.

rcParams cell 1 sets and cell 2 does not touch keep cell 1's value:
axes.facecolor, font.size, font.serif's mathtext counterpart (mathtext.rm/
it/bf/fontset), figure.dpi, lines.markersize. rcParams only cell 2 sets
(axes.linewidth/labelsize/labelpad, xtick.*, ytick.*, legend.fontsize/
framealpha/fancybox/borderpad/..., savefig.bbox/pad_inches, lines.
antialiased/dashed_pattern/dashdot_pattern/dotted_pattern/scale_dashes,
font.serif, font.sans-serif) apply as-is; there is nothing for them to
conflict with.

`colors` is cell 1's Okabe-Ito colour-blind-safe palette, exposed at module
level -- not just installed into the prop_cycle -- because cell 2 overrides
axes.prop_cycle with a different palette, yet figures reference the
Okabe-Ito list explicitly (by index) regardless of what the cycler is set to.

Cell 2 also appends a texlive bin directory to PATH, required for
text.usetex. The notebook hardcoded that path; here it is
config.TEXLIVE_BIN (overridable via $SDI_TEXLIVE_BIN). Import fails loudly,
rather than silently leaving usetex broken, if no `latex` binary is found
there -- and that check runs FIRST, before any of cell 1's `plt.rc` calls or
cell 2's PATH append, so a failed import leaves matplotlib's global rcParams
state untouched. (The notebook itself runs the check where cell 2 puts it,
after cell 1's mutations; this module moves it earlier because, unlike a
notebook cell re-executed by hand, a failed module import here does not
enter sys.modules, so a later retry re-runs this file from the top and would
otherwise re-apply cell 1's rc calls and re-append to PATH on every failed
attempt.)
"""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

import config

# --------------------------------------------------------------------------
# Cell 2's texlive guard, moved ahead of every rcParams mutation and the PATH
# append below (both cell 1's and cell 2's): raising here before any
# `plt.rc`/`mpl.rcParams` call runs means a failed import leaves matplotlib's
# global state untouched, so a retry after fixing $SDI_TEXLIVE_BIN starts
# from a clean rcParams rather than one already half-styled by a previous,
# failed attempt (the module never enters sys.modules on the raise, so a
# retry re-execs this file from the top).
# --------------------------------------------------------------------------
_texlive_bin = Path(config.TEXLIVE_BIN)
_latex = _texlive_bin / 'latex'
if not _latex.is_file():
    raise FileNotFoundError(
        f'no latex binary at {_latex} (TEXLIVE_BIN={config.TEXLIVE_BIN!r}); '
        'text.usetex requires a working latex on PATH. Point $SDI_TEXLIVE_BIN '
        'at a texlive bin/ directory that has one.')

# --------------------------------------------------------------------------
# Okabe-Ito colour-blind-safe palette (cell 1).
# --------------------------------------------------------------------------
colors = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# --------------------------------------------------------------------------
# Cell 1.
# --------------------------------------------------------------------------
plt.rc('axes', fc='w', prop_cycle=mpl.cycler(color=colors))
plt.rc('font', size=8, family='serif', serif='Nimbus Roman')
plt.rc('mathtext', rm='serif', it='serif:italic', bf='serif:bold', fontset='custom')
plt.rc('figure', figsize=[2.7, 2.7], dpi=500)
plt.rc('lines', lw=1, markersize=2)
plt.rc('legend', frameon=False)

# --------------------------------------------------------------------------
# Cell 2. Applied after cell 1 -- overrides where keys collide, see the table
# above. The texlive existence check itself now runs before cell 1, above.
# --------------------------------------------------------------------------
os.environ['PATH'] += os.pathsep + str(_texlive_bin)

# Line styles
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.antialiased'] = True
mpl.rcParams['lines.dashed_pattern'] = 2.8, 1.5
mpl.rcParams['lines.dashdot_pattern'] = 4.8, 1.5, 0.8, 1.5
mpl.rcParams['lines.dotted_pattern'] = 1.1, 1.1
mpl.rcParams['lines.scale_dashes'] = True

# Default colors
mpl.rcParams['axes.prop_cycle'] = cycler('color', ['cornflowerblue', 'forestgreen', 'maroon', 'goldenrod',
                                                    'firebrick', 'mediumorchid', 'navy', 'brown'])

# Fonts
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.serif'] = 'CMU Serif'
mpl.rcParams['font.sans-serif'] = 'CMU Sans Serif, DejaVu Sans, Bitstream Vera Sans, Lucida Grande,\
Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif'
mpl.rcParams['text.usetex'] = True

# Axes
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['axes.labelpad'] = 9.0

# Tick marks - the essence of life
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 2.5
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['xtick.minor.width'] = 0.75
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 2.5
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 0.75
mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.minor.visible'] = True

# Legend
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.framealpha'] = 0.8
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.borderpad'] = 0.4  # border whitespace
mpl.rcParams['legend.labelspacing'] = 0.5  # the vertical space between the legend entries
mpl.rcParams['legend.handlelength'] = 1.5  # the length of the legend lines
mpl.rcParams['legend.handleheight'] = 0.7  # the height of the legend handle
mpl.rcParams['legend.handletextpad'] = 0.5  # the space between the legend line and legend text
mpl.rcParams['legend.borderaxespad'] = 0.5  # the border between the axes and legend edge
mpl.rcParams['legend.columnspacing'] = 2.0  # column separation

# Figure size
mpl.rcParams['figure.figsize'] = 12, 6

# Save details
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1
mpl.rcParams['text.usetex'] = True
