"""The dwarf sample is the index every `dsph_idx` CLI argument refers to.

`dwarf_names` is built from the vendored LVDB release, and five further arrays
(`abbreviations`, `fermi_names`, `fermi_names_update`, `n_stars`, `ps18_names`)
are positionally keyed to the order it produces. A change in that order silently
re-labels every downstream product, so it is pinned here.

The first test deliberately does not import Jdata: it reconstructs the sample
straight from the CSV with pandas alone, so a catalog swap is caught in the
fast tier without SatGen, astropy, or a network fetch.
"""

import hashlib

import pandas as pd
import pytest

import config

# The 43 dwarfs of LVDB v1.0.5, in the order Jdata builds them: MW hosts with
# vlos_sigma > 0, then LMC hosts with vlos_sigma > 0, then 'Leo V' appended.
# Verified identical to the pre-migration module, which fetched the same
# release over HTTPS.
EXPECTED_NAMES = [
    'Antlia II', 'Aquarius II', 'Bootes I', 'Bootes II', 'Bootes III',
    'Canes Venatici I', 'Canes Venatici II', 'Carina', 'Centaurus I',
    'Coma Berenices', 'Crater II', 'Draco', 'Eridanus II', 'Eridanus IV',
    'Fornax', 'Grus I', 'Hercules', 'Leo I', 'Leo II', 'Leo IV', 'Leo VI',
    'LMC', 'Pegasus III', 'Pegasus IV', 'Pisces II', 'Sagittarius',
    'Sculptor', 'Segue 1', 'Sextans', 'Tucana II', 'Tucana IV', 'Tucana V',
    'Ursa Major I', 'Ursa Major II', 'Ursa Minor', 'Willman 1', 'Carina II',
    'Carina III', 'Horologium I', 'Hydrus I', 'Reticulum II', 'SMC', 'Leo V',
]

# Excluded from every figure by the notebook's category split, which is why the
# papers report 39 while the catalog holds 43.
LARGE_DWARFS = ['Fornax', 'SMC', 'LMC', 'Sagittarius']


def _names_from_csv():
    """Reconstruct dwarf_names from the vendored CSV, as Jdata does."""
    table = pd.read_csv(config.LVDB_DIR / 'dwarf_all.csv')
    names = []
    for host in ('mw', 'lmc'):
        subset = table[table['host'] == host]
        names.extend(subset[subset['vlos_sigma'] > 0]['name'].tolist())
    names.append('Leo V')
    return names


def test_vendored_catalog_reproduces_the_sample():
    """Order matters, not just membership -- five arrays are keyed to it."""
    assert _names_from_csv() == EXPECTED_NAMES


def test_sample_size_is_43():
    assert len(EXPECTED_NAMES) == 43
    assert len(set(EXPECTED_NAMES)) == 43, 'duplicate dwarf name'


def test_drafts_39_equals_catalog_43_minus_the_large_dwarfs():
    """Documents the 39-vs-43 discrepancy as arithmetic rather than drift."""
    assert all(name in EXPECTED_NAMES for name in LARGE_DWARFS)
    assert len(EXPECTED_NAMES) - len(LARGE_DWARFS) == 39


def test_lvdb_version_is_pinned():
    version_file = (config.LVDB_DIR / 'VERSION').read_text()
    assert f'version = {config.LVDB_VERSION}' in version_file
    for name in ('dwarf_all.csv', 'gc_ambiguous.csv'):
        assert (config.LVDB_DIR / name).exists()


def test_vendored_csv_contents_match_the_recorded_hashes():
    """A recorded hash nobody checks is documentation, not a pin.

    Catches a re-fetch that changes bytes without changing the version string --
    the exact swap vendoring is meant to prevent.
    """
    recorded = {}
    for line in (config.LVDB_DIR / 'VERSION').read_text().splitlines():
        if 'sha256 =' in line and not line.strip().startswith('#'):
            name, digest = line.split('sha256 =')
            recorded[name.strip()] = digest.strip()

    assert set(recorded) == {'dwarf_all.csv', 'gc_ambiguous.csv'}
    for name, digest in recorded.items():
        actual = hashlib.sha256((config.LVDB_DIR / name).read_bytes()).hexdigest()
        assert actual == digest, f'{name} does not match its recorded sha256'


def test_no_network_fetch_at_import():
    """The whole point of vendoring: Jdata must not reach the network."""
    source = (config.REPO_ROOT / 'python' / 'Jdata.py').read_text()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        assert 'Table.read(\'http' not in stripped.replace('"', "'")
        assert 'releases/download' not in stripped


# Tests that import Jdata live in test_jdata_module.py: a module-level
# importorskip here would skip this whole file, including the CSV checks above,
# in exactly the environment they are meant to guard.
