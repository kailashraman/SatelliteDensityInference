"""Regression tests for the compute_mhalf.py per-dwarf/globals split.

Upstream ran a per-dwarf array task that computed AND wrote both the
per-dwarf product and the four catalog-global products
(<version>_{rho30,M30,rho150,Mvir200}.npz) on every task -- every one of the
43 per-dwarf tasks racing to overwrite the same four files. compute_mhalf.py
now has two modes: `<dsph_idx> <version>` (per-dwarf only) and `--globals
<version>` (globals only, run once). These tests exercise both modes against
a tiny synthetic catalog (built from a handful of literal Green-param rows
taken from the real Diemer catalog, so `Green(*row)` constructs a physically
valid halo without depending on any migrated data file being present).

Requires the J_calc environment (SatGen, colossus, astropy) -- see
conftest.py's `needs_satgen` marker. Dedicated file, all tests need it, so a
module-level skip here does not touch the fast tier elsewhere.
"""
import runpy
import sys

import numpy as np
import pytest

import config
import provenance

h5py = pytest.importorskip('h5py')

pytestmark = pytest.mark.needs_satgen

# Literal rows from data/additional/m12res8_10k_Diemer+scatter_sim.h5
# (Green_params: [Minit, c, Delta, z_infall]; virial_mass in Msun), copied in
# so this fixture does not depend on that (gitignored, multi-GB) file existing.
GREEN_PARAMS = np.array([
    [1.16355232e+08, 2.49739335e+01, 1.01886271e+02, 8.64468153e-03],
    [1.99128509e+09, 2.05027420e+01, 1.03397362e+02, 2.62646870e-02],
    [1.73668265e+08, 1.99771199e+01, 1.03780145e+02, 3.07404080e-02],
])
VIRIAL_MASS = np.array([1.16285353e+08, 1.97552933e+09, 1.72270068e+08])

GLOBAL_NAMES = ('rho30', 'M30', 'rho150', 'Mvir200')

# Expected values for the fixture catalog above, computed once by calling
# Green(*row)/mass_defs.changeMassDefinition directly (the same calls
# compute_mhalf.py makes) and embedded here as literals, so a radius slip
# (0.03 -> 0.3), a rhalf unit slip, or the Errani 1.35 factor changing would
# leave a fast-tier test red instead of green. dwarf index 0 is Antlia II
# (rhalf = 3.1858588764656015 kpc as of this writing).
EXPECTED_LOGMHALF = np.array([7.75700763, 8.61658617, 7.86506142])
EXPECTED_LOGMDYN_ERRANI = np.array([7.83813993, 8.74240223, 7.95749155])
EXPECTED_GLOBALS = {
    'rho30': np.array([4.72111649e+08, 9.80378451e+08, 4.01224082e+08]),
    'M30': np.array([83464.1171855, 169041.26935594, 70206.23888868]),
    'rho150': np.array([6.29335303e+07, 1.69590818e+08, 5.94188525e+07]),
    'Mvir200': np.array([1.04235765e+08, 1.77270758e+09, 1.54520378e+08]),
}


@pytest.fixture
def fixture_catalog(tmp_path, monkeypatch):
    """Register a throwaway SatGen version backed by the tiny catalog above,
    with ADDITIONAL_DIR/MHALF_DIR sandboxed to tmp_path so nothing here can
    write into the real data/ tree.

    config.WEIGHTS_DIR (like config.MHALF_DIR) is derived from
    config.ADDITIONAL_DIR at import time, not re-derived on each access --
    patching ADDITIONAL_DIR alone does NOT sandbox it. compute_mhalf.py never
    reads WEIGHTS_DIR, so it is left unpatched here; any future test in this
    file that exercises a code path touching WEIGHTS_DIR must patch it
    explicitly (`monkeypatch.setattr(config, 'WEIGHTS_DIR', ...)`) or it will
    write into the real data/ tree.
    """
    version = 'sdi_test_mhalf_fixture'
    additional_dir = tmp_path / 'additional'
    additional_dir.mkdir()
    filename = 'fixture.h5'
    with h5py.File(additional_dir / filename, 'w') as f:
        f.create_dataset('Green_params', data=GREEN_PARAMS)
        f.create_dataset('virial_mass', data=VIRIAL_MASS)

    monkeypatch.setattr(config, 'ADDITIONAL_DIR', additional_dir)
    monkeypatch.setattr(config, 'MHALF_DIR', tmp_path / 'mhalf')
    monkeypatch.setitem(config.H5_REGISTRY, version, filename)
    return version


def _run_mhalf(*args):
    argv = sys.argv
    sys.argv = ['compute_mhalf.py', *args]
    try:
        runpy.run_path(str(config.REPO_ROOT / 'python' / 'compute_mhalf.py'),
                       run_name='__main__')
    finally:
        sys.argv = argv


def _dwarf_name(i):
    import Jdata as obs
    return obs.dwarf_names[i]


def test_per_dwarf_mode_writes_exactly_one_file_and_no_globals(fixture_catalog):
    """The race being fixed: a per-dwarf task must never touch the four
    catalog-global files. Fails against the pre-split script, which wrote
    all four on every per-dwarf task."""
    version = fixture_catalog
    dwarf0 = _dwarf_name(0)

    _run_mhalf('0', version)

    mhalf_dir = config.MHALF_DIR / version
    written = sorted(p.name for p in mhalf_dir.glob('*.npz'))
    assert written == [f'{dwarf0}.npz'], \
        f'expected exactly the per-dwarf product, got {written}'

    for name in GLOBAL_NAMES:
        assert not (config.ADDITIONAL_DIR / f'{version}_{name}.npz').exists(), \
            f'per-dwarf mode wrote the global {name} file -- the race is back'


def test_two_concurrent_per_dwarf_tasks_neither_write_globals(fixture_catalog):
    """Simulates two array tasks (different dwarfs) racing on the same
    version: neither may write a catalog-global file."""
    version = fixture_catalog
    dwarf0, dwarf1 = _dwarf_name(0), _dwarf_name(1)

    _run_mhalf('0', version)
    _run_mhalf('1', version)

    mhalf_dir = config.MHALF_DIR / version
    assert (mhalf_dir / f'{dwarf0}.npz').exists()
    assert (mhalf_dir / f'{dwarf1}.npz').exists()

    for name in GLOBAL_NAMES:
        assert not (config.ADDITIONAL_DIR / f'{version}_{name}.npz').exists()


def test_per_dwarf_output_holds_only_the_dwarf_quantities(fixture_catalog):
    version = fixture_catalog
    dwarf0 = _dwarf_name(0)
    _run_mhalf('0', version)

    with np.load(config.MHALF_DIR / version / f'{dwarf0}.npz') as data:
        keys = {k for k in data.files if k != provenance.PROVENANCE_KEY}
    assert keys == {'logMhalf', 'logMdyn_errani'}


def test_per_dwarf_output_matches_expected_values(fixture_catalog):
    """Fast-tier value check: the filename/key-set tests above pass even if a
    radius or the Errani 1.35 factor is wrong. Fails against a sabotaged
    radius (0.03 -> 0.3) or rhalf unit; see the module docstring for the
    sabotage check this was verified against."""
    version = fixture_catalog
    dwarf0 = _dwarf_name(0)
    _run_mhalf('0', version)

    with np.load(config.MHALF_DIR / version / f'{dwarf0}.npz') as data:
        np.testing.assert_allclose(data['logMhalf'], EXPECTED_LOGMHALF, rtol=1e-6)
        np.testing.assert_allclose(data['logMdyn_errani'], EXPECTED_LOGMDYN_ERRANI,
                                    rtol=1e-6)


def test_globals_mode_writes_exactly_the_four_files(fixture_catalog):
    version = fixture_catalog
    _run_mhalf('--globals', version)

    for name in GLOBAL_NAMES:
        path = config.ADDITIONAL_DIR / f'{version}_{name}.npz'
        assert path.exists()
        with np.load(path) as data:
            keys = {k for k in data.files if k != provenance.PROVENANCE_KEY}
            assert keys == {name}
            assert data[name].shape == (len(GREEN_PARAMS),)

    mhalf_dir = config.MHALF_DIR / version
    assert not mhalf_dir.exists() or not list(mhalf_dir.glob('*.npz')), \
        'globals mode wrote a per-dwarf product'


def test_globals_values_match_expected(fixture_catalog):
    """Fast-tier value check: the shape/key-set test above passes even if a
    radius (0.03 -> 0.3) or the mass-definition conversion drifts. See the
    module docstring for the sabotage check this was verified against."""
    version = fixture_catalog
    _run_mhalf('--globals', version)

    for name in GLOBAL_NAMES:
        with np.load(config.ADDITIONAL_DIR / f'{version}_{name}.npz') as data:
            np.testing.assert_allclose(data[name], EXPECTED_GLOBALS[name], rtol=1e-6)


@pytest.mark.needs_data
def test_matches_migrated_products_on_real_catalog_slice(tmp_path, monkeypatch):
    """Runs compute_mhalf.py itself -- both `--globals` and per-dwarf mode --
    against a fixture h5 built from the first N rows of the real Diemer
    catalog, and checks its own output against the migrated
    data/additional/{Diemer_*,mhalf/Diemer/<dwarf>} products.

    Rebuilt to route through the actual entry point rather than
    re-implementing the compute loop inline: a re-implementation that drifts
    from the script (a radius or constant slip) would leave that version of
    this test green, since it would only be checking itself. Registered via
    H5_REGISTRY/ADDITIONAL_DIR/MHALF_DIR monkeypatching (the same pattern as
    `fixture_catalog` above), so the script reads/writes entirely inside
    `tmp_path` and never touches the real data/ tree.

    Confirmed to pass with N=200 as of this writing; if it does not pass here,
    stop and report rather than loosening the tolerance.
    """
    import Jdata as obs

    # Captured before monkeypatching config below, so the comparison reads
    # the real migrated products regardless of where the script under test
    # is redirected to write.
    real_additional_dir = config.ADDITIONAL_DIR
    real_mhalf_dir = config.MHALF_DIR

    n = 200
    dwarf_idx = 0
    dwarf = obs.dwarf_names[dwarf_idx]
    with h5py.File(config.h5_path('Diemer')) as f:
        Greens = f['Green_params'][:n]
        virial_mass = f['virial_mass'][:n]

    version = 'sdi_test_parity_slice'
    additional_dir = tmp_path / 'additional'
    additional_dir.mkdir()
    filename = 'fixture_slice.h5'
    with h5py.File(additional_dir / filename, 'w') as f:
        f.create_dataset('Green_params', data=Greens)
        f.create_dataset('virial_mass', data=virial_mass)

    monkeypatch.setattr(config, 'ADDITIONAL_DIR', additional_dir)
    monkeypatch.setattr(config, 'MHALF_DIR', tmp_path / 'mhalf')
    monkeypatch.setitem(config.H5_REGISTRY, version, filename)

    _run_mhalf('--globals', version)
    _run_mhalf(str(dwarf_idx), version)

    for name in GLOBAL_NAMES:
        with np.load(additional_dir / f'{version}_{name}.npz') as computed, \
             np.load(real_additional_dir / f'Diemer_{name}.npz') as migrated:
            assert np.array_equal(computed[name], migrated[name][:n]), \
                f'Diemer_{name}.npz no longer matches compute_mhalf.py --globals'

    with np.load(tmp_path / 'mhalf' / version / f'{dwarf}.npz') as computed, \
         np.load(real_mhalf_dir / 'Diemer' / f'{dwarf}.npz') as migrated:
        assert np.array_equal(computed['logMhalf'], migrated['logMhalf'][:n])
        assert np.array_equal(computed['logMdyn_errani'], migrated['logMdyn_errani'][:n])
