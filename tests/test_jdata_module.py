"""Checks that need to import Jdata itself, which requires the J_calc env.

Kept separate from test_dwarf_sample.py: a module-level importorskip skips the
entire file it appears in, so putting these alongside the pandas-only catalog
checks would silently disable those in the fast tier.
"""

from pathlib import Path

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


# The Jeans-results tests patch Jdata.DJA_PRODUCTION_DIR, not config: it is
# captured as a str at import, so a config override after import has no effect.
RUN_DWARF = 'Draco'  # has a DwarfJeansAnalysis run; Fornax is in DJA_SKIP_KEYS
SKIPPED_DWARF = 'Fornax'


def test_jeans_results_raise_when_dja_root_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(jdata, 'DJA_PRODUCTION_DIR', str(tmp_path / 'nope'))
    with pytest.raises(FileNotFoundError, match='SDI_DJA_RESULTS_DIR'):
        jdata.Dwarf(RUN_DWARF).get_Jeans_results()


def test_jeans_results_raise_when_a_priors_derived_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(jdata, 'DJA_PRODUCTION_DIR', str(tmp_path))
    key = jdata.dwarf_all[jdata.dwarf_all['name'] == RUN_DWARF]['key'][0]
    (tmp_path / key / 'loguniform').mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match='derived.npz'):
        jdata.Dwarf(RUN_DWARF).get_Jeans_results(dja_prior='loguniform')


def test_skipped_dwarf_returns_nan_without_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(jdata, 'DJA_PRODUCTION_DIR', str(tmp_path))  # root exists, empty
    dwarf = jdata.Dwarf(SKIPPED_DWARF)
    dwarf.get_Jeans_results()
    assert dwarf.jeans_dja_path is None
    assert all(v != v for v in dwarf.Jeans_J_quantiles)


@pytest.mark.needs_data
def test_default_dja_root_has_every_expected_posterior():
    """What actually catches the DwarfJeansAnalysis checkout moving again.

    Checks whichever root is configured (SDI_DJA_RESULTS_DIR or the sibling
    default): every dwarf outside DJA_SKIP_KEYS has every prior."""
    root = Path(jdata.DJA_PRODUCTION_DIR)
    assert root.is_dir(), f'{root} missing; set SDI_DJA_RESULTS_DIR'
    missing = []
    for name in jdata.dwarf_names:
        match = jdata.dwarf_all[jdata.dwarf_all['name'] == name]
        key = match['key'][0] if len(match) > 0 else None
        if key is None or key in jdata.DJA_SKIP_KEYS:
            continue
        for prior in config.DJA_PRIORS:
            if not (root / key / prior / 'derived.npz').is_file():
                missing.append(f'{key}/{prior}')
    assert not missing, missing



def test_failed_jeans_call_clears_the_previous_path(tmp_path, monkeypatch):
    """A raise must not leave a previous prior's path on a reused Dwarf, where
    a caller collecting jeans_dja_path for provenance would stamp it."""
    monkeypatch.setattr(jdata, 'DJA_PRODUCTION_DIR', str(tmp_path))
    dwarf = jdata.Dwarf(RUN_DWARF)
    dwarf.jeans_dja_path = 'stale/loguniform/derived.npz'
    with pytest.raises(FileNotFoundError):
        dwarf.get_Jeans_results(dja_prior='loguniform')
    assert dwarf.jeans_dja_path is None


def test_name_without_lvdb_key_raises():
    """Unknown names used to fall through to NaN, which the Fermi stages turn
    into the scaling-relation J without a word. Ursa Major III is the one name
    the pipeline constructs a Dwarf for that is absent from dwarf_all."""
    dwarf = jdata.Dwarf('Ursa Major III')
    with pytest.raises(KeyError, match='no LVDB key'):
        dwarf.get_Jeans_results()


def test_skipped_dwarf_ignores_a_run_on_disk(tmp_path, monkeypatch):
    """DJA_SKIP_KEYS means always skip, not "no run exists": a derived.npz
    appearing upstream for a listed dwarf must not be picked up."""
    monkeypatch.setattr(jdata, 'DJA_PRODUCTION_DIR', str(tmp_path))
    key = jdata.dwarf_all[jdata.dwarf_all['name'] == SKIPPED_DWARF]['key'][0]
    assert key in jdata.DJA_SKIP_KEYS
    run = tmp_path / key / jdata.DJA_PRIOR
    run.mkdir(parents=True)
    (run / 'derived.npz').write_bytes(b'not a real npz')  # would fail to load if read
    dwarf = jdata.Dwarf(SKIPPED_DWARF)
    dwarf.get_Jeans_results()
    assert dwarf.jeans_dja_path is None
    assert all(v != v for v in dwarf.Jeans_J_quantiles)
