"""Pins the weight-column index mapping that fermi_reweighting.py and
fermi_expected_limits.py hardcode (``idcs=[0,4,10]``, ``mc_idcs=[4,10]``,
``sc_idcs=[0,21]``), and that plot_multipanel_systematics.py hardcodes for
columns 5 (Moster18) and 9 (Danieli23_grow).

Neither file derives those indices from anything -- they are literals that
happen to be correct today because column 0 is ``mhalf_weights``, columns
1-10 are ``mstar_weights`` in SHMR order (so Fattahi18 at list-index 3 lands
in column 4, Kim24 at list-index 9 lands in column 10), columns 11-20 are the
same ten SHMRs' ``joint_weights``, and column 21 (mhalf_scatter only) is
``mdyn_errani``. One inserted SHMR in either producer's ``SHMR_names`` list,
or in only one of the two copies, would silently relabel every F18/K24 curve
in Part II with no error anywhere.

Compute_weights.py and compute_quantiles.py cannot be imported directly (both
read sys.argv at module scope), so the list is extracted from source text via
ast.literal_eval -- the same approach as this repo's other source-inspecting
tests (see test_launcher_paths.py), and it fails loudly if the literal is
gone or malformed rather than silently passing.
"""
import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPUTE_WEIGHTS = REPO_ROOT / 'python' / 'compute_weights.py'
COMPUTE_QUANTILES = REPO_ROOT / 'python' / 'compute_quantiles.py'

_SHMR_NAMES_RE = re.compile(r"SHMR_names\s*=\s*(\[[^\]]*\])")


def _extract_shmr_names(path):
    text = path.read_text()
    match = _SHMR_NAMES_RE.search(text)
    assert match, f'{path} has no "SHMR_names = [...]" assignment to extract'
    return ast.literal_eval(match.group(1))


def test_shmr_name_lists_agree_between_producers():
    """compute_weights.py and compute_quantiles.py duplicate this list; an
    edit to one without the other would misalign weights against quantiles."""
    weights_names = _extract_shmr_names(COMPUTE_WEIGHTS)
    quantiles_names = _extract_shmr_names(COMPUTE_QUANTILES)
    assert weights_names == quantiles_names


def test_fattahi18_lands_in_column_4():
    """Column 0 is mhalf; columns 1..len(SHMR_names) are mstar_weights in
    SHMR_names order, so index i (0-based) lands in column i+1."""
    names = _extract_shmr_names(COMPUTE_QUANTILES)
    assert names.index('Fattahi18') + 1 == 4


def test_kim24_lands_in_column_10():
    names = _extract_shmr_names(COMPUTE_QUANTILES)
    assert names.index('Kim24') + 1 == 10


def test_moster18_lands_in_column_5():
    """plot_multipanel_systematics.py's panel 1 (SHMR) reads Diemer[:,5] as
    Moster18."""
    names = _extract_shmr_names(COMPUTE_QUANTILES)
    assert names.index('Moster18') + 1 == 5


def test_danieli23_grow_lands_in_column_9():
    """plot_multipanel_systematics.py's panel 1 (SHMR) reads Diemer[:,9] as
    Danieli23_grow -- column 8 is the other variant, Danieli23_const."""
    names = _extract_shmr_names(COMPUTE_QUANTILES)
    assert names.index('Danieli23_grow') + 1 == 9


def test_mdyn_errani_lands_in_column_21():
    """Column 0 (mhalf) + len(SHMR_names) mstar_weights columns +
    len(SHMR_names) joint_weights columns = the mdyn_errani column, present
    only in the mhalf_scatter-version quantiles file."""
    names = _extract_shmr_names(COMPUTE_QUANTILES)
    assert 1 + 2 * len(names) == 21
