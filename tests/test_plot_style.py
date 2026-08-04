"""plot_style must reproduce PaperPlots.ipynb cells 1 then 2, net of overrides.

The golden dict is captured by extracting the two cells' actual source from
the notebook and exec'ing them against a freshly-reset mpl.rcParams -- not by
hand-copying values -- so this test would fail if plot_style.py drifted from
the notebook, not just from a remembered snapshot of it.

Only the rcParams-setting text is exec'd from cell 1 (from its `colors = [`
line through its last `plt.rc(...)` call): the rest of the cell is unrelated
notebook setup (imports of SatGen, colossus, Jdata, ...) that plot_style.py
does not and should not reproduce. Cell 2 is exec'd in full -- it is already
self-contained (os, matplotlib, cycler).
"""

import json
import os
from pathlib import Path

import matplotlib as mpl
import pytest

import config

NOTEBOOK = config.REPO_ROOT.parent / 'SatGen_Dwarf' / 'jupyter' / 'PaperPlots.ipynb'

CELL1_START = 'colors = ['
CELL1_END = "plt.rc('legend', frameon=False)"
# Content anchors, checked in addition to the hardcoded cell indices below --
# a notebook edit that inserts/removes a cell would otherwise silently shift
# indices 1 and 2 onto the wrong cells while everything still "runs".
CELL1_MARKER = CELL1_START
CELL2_MARKER = 'texlive/2016/bin/x86_64-linux'

# plot_style.py itself needs a working latex binary at import time (it
# raises FileNotFoundError otherwise); skip the whole module where that is
# not available, mirroring the golden fixture's skip for the missing sibling
# notebook checkout, so the no-flag tier still runs anywhere in seconds.
_TEXLIVE_UNAVAILABLE = not (Path(config.TEXLIVE_BIN) / 'latex').is_file()
pytestmark = pytest.mark.skipif(
    _TEXLIVE_UNAVAILABLE,
    reason=f'no latex binary under TEXLIVE_BIN={config.TEXLIVE_BIN!r}')


def _cell_source(nb, index):
    return ''.join(nb['cells'][index]['source'])


def _extract_cell1_rc_block(source):
    assert CELL1_MARKER in source, (
        f'cell 1 no longer contains {CELL1_MARKER!r} -- notebook cells may '
        'have shifted; update the hardcoded cell index')
    start = source.index(CELL1_START)
    end = source.index(CELL1_END) + len(CELL1_END)
    return source[start:end]


@pytest.fixture(scope='module')
def golden_rcparams():
    if not NOTEBOOK.exists():
        pytest.skip(f'{NOTEBOOK} not available (sibling SatGen_Dwarf checkout missing)')
    nb = json.loads(NOTEBOOK.read_text())
    cell1_block = _extract_cell1_rc_block(_cell_source(nb, 1))
    cell2_source = _cell_source(nb, 2)
    assert CELL2_MARKER in cell2_source, (
        f'cell 2 no longer contains {CELL2_MARKER!r} -- notebook cells may '
        'have shifted; update the hardcoded cell index')

    # Cell 2 appends its hardcoded texlive path to PATH as a side effect;
    # exec'ing it for the golden comparison should not leak that into PATH
    # for the rest of the test session. `pytest.MonkeyPatch` is used
    # directly here (rather than the function-scoped `monkeypatch` fixture,
    # which a module-scoped fixture cannot request) so the mutation is
    # undone regardless of how exec'ing the cells exits.
    mp = pytest.MonkeyPatch()
    saved_rcparams = dict(mpl.rcParams)
    try:
        mp.setattr(os, 'environ', dict(os.environ))
        mpl.rcdefaults()
        namespace = {'mpl': mpl, 'plt': __import__('matplotlib.pyplot', fromlist=['rc'])}
        exec(compile(cell1_block, '<PaperPlots cell 1 rc block>', 'exec'), namespace)
        exec(compile(cell2_source, '<PaperPlots cell 2>', 'exec'), namespace)
        golden = dict(mpl.rcParams)
    finally:
        mpl.rcParams.update(saved_rcparams)
        mp.undo()
    return golden


def test_plot_style_matches_notebook_cells_1_then_2(golden_rcparams):
    mpl.rcdefaults()
    import importlib
    import plot_style
    importlib.reload(plot_style)
    actual = dict(mpl.rcParams)
    assert actual == golden_rcparams


def test_colors_is_the_okabe_ito_palette():
    import plot_style
    assert plot_style.colors == [
        "#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    ]
