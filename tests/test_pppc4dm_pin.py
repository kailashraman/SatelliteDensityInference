"""The PPPC4DM tables are pinned by digest instead of committed.

They are 40 MB of a public literature release, so git carries the recipe
rather than the bytes. That only works if something checks what a re-fetch
produced: the tables carry no version string and the source republishes in
place, so a swap changes every Fermi limit in Part II without changing a
filename.

Kept out of tests/test_fermi_paths.py deliberately -- that module opens with
`importorskip('fermi_funcs')`, so everything in it skips outside the J_calc
environment. These checks read a text file and hash bytes; they should run on
the default tier.
"""

import hashlib

import pytest

import config

PPPC4DM_DIR = config.DATA_DIR / 'fermi_legacy' / 'PPPC4DM'

PARTICLES = ('antideuterons', 'antiprotons', 'gammas', 'neutrinos_e',
             'neutrinos_mu', 'neutrinos_tau', 'positrons')

# Both trees, because _load_cirelli_table selects between them on EWcorr and
# hardcodes a different row count for each (179 with electroweak corrections,
# 180 without) to reshape the flat table. A file swapped between the two
# reshapes without raising and silently mis-maps energies onto masses.
EXPECTED = {f'AtProduction_all/AtProduction_{p}.dat' for p in PARTICLES} | {
    f'AtProductionNoEW_all/AtProductionNoEW_{p}.dat' for p in PARTICLES}


def _recorded():
    """Parse VERSION into {relative path: sha256}."""
    recorded = {}
    for line in (PPPC4DM_DIR / 'VERSION').read_text().splitlines():
        if 'sha256 =' in line and not line.strip().startswith('#'):
            name, digest = line.split('sha256 =')
            recorded[name.strip()] = digest.strip()
    return recorded


def test_manifest_covers_every_table_the_loader_can_request():
    """`particle` is an argument to exctractcirellitable, so pinning only the
    gammas pair the Fermi path currently calls would leave the other twelve
    unchecked one call away."""
    assert _recorded().keys() == EXPECTED


def test_manifest_records_well_formed_digests():
    for name, digest in _recorded().items():
        assert len(digest) == 64, f'{name}: {digest!r} is not a sha256'
        assert all(c in '0123456789abcdef' for c in digest), name


def test_manifest_names_its_source():
    """A digest with no provenance pins bytes to nothing retrievable."""
    header = (PPPC4DM_DIR / 'VERSION').read_text()
    assert 'marcocirelli.net' in header


@pytest.mark.needs_data
@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_local_tables_match_their_recorded_digests(name):
    """A recorded hash nobody checks is documentation, not a pin.

    `needs_data` here means the fetched PPPC4DM tables, not the migrated
    `results/` tree the marker denotes elsewhere. Reusing it keeps one opt-in
    flag rather than adding a marker for a single module; the explicit
    `exists()` skip below is what actually guards the precondition.
    """
    path = PPPC4DM_DIR / name
    if not path.exists():
        pytest.skip(f'{name} not fetched locally')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == _recorded()[name], f'{name} does not match VERSION'
