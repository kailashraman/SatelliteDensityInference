"""Pins build_jfactors_all's series set in plot_number_functions.py.

Notebook cell 80 (the savefig cell that actually produced the published
Jfactors_all.pdf) plots five series with no commented-out lines: SatGen,
mhalf inference, Fattahi+18 inference, and TWO Jeans curves -- cell 79 loops
`dja_priors = ['loguniform', 'satgen_shmr_fattahi18']`, drawing a fresh
`rng.uniform(...)` per prior. drafts_temp/temp_part2.tex enumerates exactly
those five.

The pre-fix port instead read from upstream's own plot_number_functions.py,
which diverges from cell 80: it adds a Kim+24 curve absent from the caption
and collapses the two Jeans-prior curves into one labelled 'Jeans analyses'
(loguniform only), dropping the SatGen-informed-prior curve entirely.

plot_number_functions.py reads sys.argv at import time and imports
halo_weights/Jdata (J_calc-only), so it cannot be imported directly here --
build_jfactors_all's source is extracted from source text instead, the same
approach as test_weight_column_mapping.py.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOT_NUMBER_FUNCTIONS = REPO_ROOT / 'python' / 'plot_number_functions.py'


def _build_jfactors_all_source():
    text = PLOT_NUMBER_FUNCTIONS.read_text()
    match = re.search(
        r"\ndef build_jfactors_all\(\):\n(.*?)\n(?=def [a-zA-Z_]+\()",
        text, re.DOTALL)
    assert match, f'{PLOT_NUMBER_FUNCTIONS} has no build_jfactors_all() to extract'
    return match.group(1)


def test_weight_keys_has_no_kim24_column():
    """weight_keys must be exactly mhalf + Fattahi+18 -- no
    ('mstar_weights', 9) (Kim+24), which is not in the caption. Fails
    against the pre-fix weight_keys that included it."""
    source = _build_jfactors_all_source()
    match = re.search(r"weight_keys\s*=\s*(\[.*?\])", source)
    assert match, 'no weight_keys assignment found in build_jfactors_all'
    weight_keys = eval(match.group(1))
    assert weight_keys == [('mhalf_weights', None), ('mstar_weights', 3)]


def test_dja_priors_loops_both_standard_and_satgen_informed():
    """Fails against the pre-fix single dja_prior='loguniform' call, which
    dropped the SatGen-informed-prior curve entirely."""
    source = _build_jfactors_all_source()
    match = re.search(r"dja_priors\s*=\s*(\[.*?\])", source)
    assert match, 'no dja_priors list found in build_jfactors_all'
    assert eval(match.group(1)) == ['loguniform', 'satgen_shmr_fattahi18']


def test_series_has_five_entries_with_both_jeans_curves():
    """The published figure has 5 series; the pre-fix port had 5 but the
    wrong set (Kim+24 instead of the SatGen-informed Jeans prior)."""
    source = _build_jfactors_all_source()
    match = re.search(r"series\s*=\s*\[(.*?)\n    \]", source, re.DOTALL)
    assert match, 'no series list found in build_jfactors_all'
    series_text = match.group(1)
    entries = re.findall(r"dict\(", series_text)
    assert len(entries) == 5

    assert 'Kim+24' not in series_text, (
        'series still includes a Kim+24 curve, absent from the caption')
    assert "jeans_q['loguniform']" in series_text
    assert "jeans_q['satgen_shmr_fattahi18']" in series_text
    assert 'Jeans analyses - standard prior' in series_text
    assert r'Jeans analyses - \texttt{SatGen}-informed prior' in series_text
