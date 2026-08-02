"""Checks that need to import Jdata itself, which requires the J_calc env.

Kept separate from test_dwarf_sample.py: a module-level importorskip skips the
entire file it appears in, so putting these alongside the pandas-only catalog
checks would silently disable those in the fast tier.
"""

import pytest

import config
from test_dwarf_sample import EXPECTED_NAMES

jdata = pytest.importorskip('Jdata', reason='needs the J_calc environment')

pytestmark = pytest.mark.needs_satgen


def test_jdata_matches_the_pinned_sample():
    assert list(jdata.dwarf_names) == EXPECTED_NAMES


@pytest.mark.parametrize('name', ['abbreviations', 'fermi_names',
                                  'fermi_names_update'])
def test_parallel_arrays_align_with_the_sample(name):
    assert len(getattr(jdata, name)) == len(EXPECTED_NAMES)


@pytest.mark.parametrize('name', ['ps18_names', 'n_stars'])
def test_arrays_with_a_trailing_extra_entry(name):
    """These two carry one element more than there are dwarfs.

    Every one of the 43 valid indices still aligns -- the extra sits past the
    end -- so this is faithful to upstream rather than a live off-by-one. It is
    pinned because appending a dwarf would turn it into one.
    """
    assert len(getattr(jdata, name)) == len(EXPECTED_NAMES) + 1


def test_ps18_names_align_where_they_are_not_none():
    """Spot-check the alignment the trailing extra could have broken."""
    assert jdata.ps18_names[EXPECTED_NAMES.index('Ursa Major II')] == 'ursamajor2'
    assert jdata.ps18_names[EXPECTED_NAMES.index('Segue 1')] == 'segue1'
    assert jdata.ps18_names[EXPECTED_NAMES.index('Leo V')] is None


def test_n_stars_aligns_at_known_entries():
    assert jdata.n_stars[EXPECTED_NAMES.index('LMC')] == r'O($10^{7}$)'
    assert jdata.n_stars[EXPECTED_NAMES.index('Fornax')] == '2483'


def test_dja_dir_comes_from_config():
    assert jdata.DJA_PRODUCTION_DIR == str(config.DJA_RESULTS_DIR)
    assert jdata.DJA_PRIOR == config.DJA_PRIOR


def test_kd_names_stay_keyed_to_the_kinematics_table():
    """`kd_names` is a hardcoded literal; `kd_dispersion` comes from the CSV.

    Dwarf.__init__ indexes the second with a position derived from the first,
    so a catalog swap that changes the row count or order mis-assigns
    dispersions to dwarfs. Asserting len(kd_names) == 25 alone would only
    restate the literal and could not catch that.
    """
    assert len(jdata.kd_names) == len(jdata.kd_dispersion) == len(jdata.kd_table)
    assert len(jdata.kd_dispersion_err) == len(jdata.kd_names)

    # Check the correspondence itself against the table's own Name column,
    # rather than against a remembered dispersion value: this fails if the
    # rows are reordered even when the count is unchanged.
    table_names = list(jdata.kd_table['Name'])
    for full, abbrev in [('Bootes I', 'Boo1'), ('Draco', 'Dra'),
                         ('Segue 1', 'Seg1'), ('Willman 1', 'W1')]:
        index = list(jdata.kd_names).index(full)
        assert table_names[index] == abbrev, f'{full} misaligned'


def test_mc_names_and_dispersions_are_both_csv_derived():
    assert len(jdata.mc_names) == len(jdata.mc_dispersion) == 32


def test_ursa_major_iii_still_constructs():
    """Ursa Major III is not in dwarf_names (excluded from the 39-dwarf sample),
    but `Dwarf.__init__` has a dedicated branch for it, read from
    `gc_ambiguous.csv` rather than `dwarf_all.csv`. Notebook cell 74 calls
    `obs.Dwarf('Ursa Major III')` to place the UMa3/U1 point on
    dispersion_limit_contours.pdf -- a future cleanup must not remove this
    branch as dead code."""
    import numpy as np
    dwarf = jdata.Dwarf('Ursa Major III')
    assert np.isfinite(dwarf.rhalf.value)
    assert np.all(np.isfinite(dwarf.rhalf_err.value))
