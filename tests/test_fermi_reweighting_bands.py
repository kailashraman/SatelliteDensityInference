"""Pins fig_prior_envelope's band_indices in plot_fermi_reweighting.py.

Upstream's python/plot_fermi_reweighting.py:642 took PaperPlots.ipynb cell
187's *comment* ("Compute shaded band across i = 0, 1, 6, 7, 8, 9, 10, 11")
rather than its code (band_indices = [0, 6, 7, 8, 9, 10, 11]), so index 1
(mhalf) ended up in the envelope's input set. The band is scoped to
Jeans-analysis prior variants only; mhalf is not a Jeans analysis, so
including it lowers the envelope's lower edge by up to 28% across most mass
bins.

plot_fermi_reweighting.py imports Jdata/config/provenance/fermi_funcs, which
need the J_calc conda environment, so band_indices is extracted from source
text via ast.literal_eval rather than importing the module -- same approach
as test_weight_column_mapping.py/test_launcher_paths.py.
"""
import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOT_FERMI_REWEIGHTING = REPO_ROOT / 'python' / 'plot_fermi_reweighting.py'

_BAND_INDICES_RE = re.compile(r"band_indices\s*=\s*(\[[^\]]*\])")


def _extract_band_indices():
    text = PLOT_FERMI_REWEIGHTING.read_text()
    match = _BAND_INDICES_RE.search(text)
    assert match, f'{PLOT_FERMI_REWEIGHTING} has no "band_indices = [...]" assignment to extract'
    return ast.literal_eval(match.group(1))


def test_band_indices_excludes_mhalf():
    """The specific bug: index 1 (mhalf, WEIGHT_LIST[1]) is not a Jeans
    analysis and must not be in the envelope's input set. Fails against the
    pre-fix [0, 1, 6, 7, 8, 9, 10, 11]."""
    assert 1 not in _extract_band_indices()


def test_band_indices_matches_notebook_cell_187_code():
    """band_indices must equal cell 187's code, not its stale comment."""
    assert _extract_band_indices() == [0, 6, 7, 8, 9, 10, 11]
