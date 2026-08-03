"""Regression tests for the infall -> z=0 stellar-mass mapping.

Bug this pins: `ResampleMstar.evolved_Mstar` added a scalar
`self.Mstar_ratio = 1.2` to log10(Mstar) -- a constant +1.2 dex, a factor of
~16 upward -- where the per-halo Errani+18 remaining-fraction belongs. Three
faults compounded: a linear ratio applied in log space (its own comment
described the multiplicative form), a constant standing in for a quantity the
catalog carries per halo (`stellar_mass` / `tpeak_stellar_mass`), and an
upward shift where a tidal track can only remove stellar mass.

It was invisible in the products: eight of nine published weights versions
were built without the offset and `mass_floor_7` with it, and nothing
distinguished them, so panel 6 of multipanel_systematics.pdf compared a
Kim+24 series across the two conventions as though only the mass floor
differed.

Seven of the eight tests below fail on the pre-fix code. The exception is
`test_real_catalog_ratio_is_bounded`, which reads the catalog fields directly
and never calls `evolved_Mstar`, so it passes either way by construction --
it guards the precondition, not the mapping.
"""
import numpy as np
import pytest

h5py = pytest.importorskip('h5py')

pytestmark = pytest.mark.needs_satgen

# Green_params: [Minit, c, Delta, z_infall]; literal rows from the Diemer
# catalog so Green(*row) stays physically valid, as in test_compute_mhalf.py.
GREEN_PARAMS = np.array([
    [1.16355232e+08, 2.49739335e+01, 1.01886271e+02, 8.64468153e-03],
    [1.99128509e+09, 2.05027420e+01, 1.03397362e+02, 2.62646870e-02],
    [1.73668265e+08, 1.99771199e+01, 1.03780145e+02, 3.07404080e-02],
    [5.42310000e+08, 2.21000000e+01, 1.02500000e+02, 1.50000000e-02],
])
VIRIAL_MASS = np.array([1.16285353e+08, 1.97552933e+09, 1.72270068e+08,
                        5.40000000e+08])

# Chosen so the ratios span the real catalogs' range: no loss, heavy loss
# (-0.382 dex is the most-stripped halo in Diemer), mild loss, and a slightly
# POSITIVE ratio -- 1-4% of real halos carry one, max +0.006 dex, which is
# tree-building noise and must be clipped to 0 rather than boosting a halo.
TPEAK_STELLAR_MASS = np.array([1.0e5, 1.0e6, 1.0e4, 1.0e5])
STELLAR_MASS = np.array([1.0e5, 1.0e6 * 10 ** -0.382, 1.0e4 * 10 ** -0.05,
                         1.0e5 * 10 ** 0.004])
EXPECTED_RATIO = np.array([0.0, -0.382, -0.05, 0.0])  # last one clipped


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """A 4-halo catalog carrying every field ResampleMstar.__init__ reads."""
    import halo_weights

    path = tmp_path / 'mini.h5'
    n = len(GREEN_PARAMS)
    with h5py.File(path, 'w') as f:
        f['Green_params'] = GREEN_PARAMS
        f['virial_mass'] = VIRIAL_MASS
        f['tpeak_stellar_mass'] = TPEAK_STELLAR_MASS
        f['stellar_mass'] = STELLAR_MASS
        f['mw_hosts_lmc'] = np.zeros(n, dtype=bool)
        f['mwID'] = np.arange(n)
        f['position'] = np.tile([30.0, 40.0, 0.0], (n, 1))
        f['tpeak_v_max'] = np.full(n, 20.0)

    monkeypatch.setattr(halo_weights, 'get_h5',
                        lambda sim_version: h5py.File(path, 'r'))
    return halo_weights


def test_ratio_matches_catalog_and_is_clipped(catalog):
    with catalog.ResampleMstar(version='Diemer') as w:
        np.testing.assert_allclose(w.logMstar_ratio, EXPECTED_RATIO, atol=1e-12)


def test_no_constant_offset(catalog):
    """Zero input must return the per-halo ratio, not ratio + a constant.

    Fails on the pre-fix code, which returned 1.2 for every halo.
    """
    with catalog.ResampleMstar(version='Diemer') as w:
        out = w.evolved_Mstar(np.zeros(len(GREEN_PARAMS)))
    np.testing.assert_allclose(out, EXPECTED_RATIO, atol=1e-12)
    assert out.max() <= 1e-12, (
        f'evolved_Mstar increased a stellar mass by {out.max():.4f} dex; the '
        'Errani+18 mapping is a remaining fraction and cannot exceed 1.')


def test_mapping_is_per_halo_not_constant(catalog):
    """The shift must vary halo to halo; a constant is the bug's signature."""
    with catalog.ResampleMstar(version='Diemer') as w:
        out = w.evolved_Mstar(np.zeros(len(GREEN_PARAMS)))
    assert np.ptp(out) > 0.1, (
        'evolved_Mstar applied the same shift to every halo, so it is not '
        'reading the catalog per-halo stripping history.')


def test_offset_is_additive_in_dex(catalog):
    """Adding to log10(Mstar) must shift by the ratio, independent of input."""
    with catalog.ResampleMstar(version='Diemer') as w:
        base = np.full(len(GREEN_PARAMS), 6.0)
        np.testing.assert_allclose(
            w.evolved_Mstar(base) - base, EXPECTED_RATIO, atol=1e-12)


def test_rejects_masked_input(catalog):
    """A pre-masked array must raise, not broadcast against the full catalog.

    evolved_Mstar runs before any mask; a shorter array would silently pair
    each halo with a different halo's stripping history.
    """
    with catalog.ResampleMstar(version='Diemer') as w:
        with pytest.raises(ValueError, match='full catalog length'):
            w.evolved_Mstar(np.zeros(len(GREEN_PARAMS) - 1))


@pytest.mark.needs_data
@pytest.mark.parametrize('version', ['Diemer', 'Diemer_0p12', 'Zhao', 'm2e12',
                                     'mass_floor_7', 'splashback'])
def test_real_catalog_ratio_is_bounded(version):
    """The precondition the fix rests on, asserted against real catalogs.

    Everything above is defined by the synthetic fixture; nothing there would
    notice a catalog rebuilt underneath the code with a zero stellar mass or a
    genuinely positive ratio. This is the test that fires in that case.

    splashback is included deliberately: it retains disrupted halos, so it
    reaches -6.2 dex (physical) and a stellar_mass floor of 1.3e-4 Msun, four
    orders below any other catalog -- it is where a zero would appear first.
    """
    import halo_weights

    with halo_weights.get_h5(version) as f:
        ratio = (np.log10(f['stellar_mass'][()])
                 - np.log10(f['tpeak_stellar_mass'][()]))

    assert np.all(np.isfinite(ratio)), (
        f'{version}: {np.sum(~np.isfinite(ratio))} halos have a non-finite '
        'infall -> z=0 ratio (zero stellar_mass or tpeak_stellar_mass); '
        'these poison the whole dwarf via the KDE, not just their own halo.')
    assert ratio.max() < 0.01, (
        f'{version}: raw ratio reaches {ratio.max():+.4f} dex. The mapping is '
        'a tidal remaining-fraction and cannot exceed 1; a positive tail '
        'beyond numerical noise means the catalog fields changed meaning.')


@pytest.mark.needs_data
@pytest.mark.parametrize('version', ['Diemer', 'mass_floor_7', 'splashback'])
def test_round_trips_the_catalogs_own_stellar_mass(version):
    """Feeding the catalog's own infall M* back must return its z=0 M*.

    The strongest available statement that the mapping is the catalog's, not
    a model of our own: evolved_Mstar exists only because Part I swaps in ten
    external SHMRs, each needing its infall mass carried through the same
    tidal stripping. Applied to SatGen's own infall mass it must be the
    identity, since that is where the ratio came from.

    The only permitted deviations are the clipped halos: strictly negative
    (the clip can only remove a positive ratio) and bounded by the noise
    floor. A constant offset, a sign error, or a misaligned ratio all break
    this regardless of which SHMR is in play.
    """
    import halo_weights

    with halo_weights.ResampleMstar(version=version) as w:
        got = w.evolved_Mstar(w.logMstar_infall)
        infall = w.logMstar_infall
        with halo_weights.get_h5(version) as f:
            want = np.log10(f['stellar_mass'][()])

    # Compared at tolerance, not exactly: infall + (want - infall) != want in
    # floating point, so an exact test passes or fails on rounding rather than
    # on the mapping. It happens to hold for Diemer and to fail for 4.3M
    # splashback halos at the 1e-16 level -- which says nothing about the code.
    delta = got - want
    raw = want - infall
    np.testing.assert_allclose(delta, np.minimum(raw, 0.0) - raw, atol=1e-12)
    assert delta.max() <= 1e-12, 'the clip can only lower a stellar mass'
    assert np.abs(delta).max() < 0.01, (
        f'{version}: round trip differs by up to {np.abs(delta).max():.4f} '
        'dex, beyond the clipped positive tail')


def test_symphony_has_no_mapping(tmp_path, monkeypatch):
    """Symphony/MWest carry no stellar-mass fields -- raise, don't invent one."""
    import halo_weights

    path = tmp_path / 'sims.h5'
    n = 3
    with h5py.File(path, 'w') as f:
        f['mpeak'] = np.full(n, 1e9)
        f['mvir'] = np.full(n, 5e8)
        f['host_id'] = np.arange(n)
        f['x'] = np.tile([30.0, 40.0, 0.0], (n, 1))
    monkeypatch.setattr(halo_weights, 'get_h5',
                        lambda sim_version: h5py.File(path, 'r'))

    with halo_weights.ResampleMstar(version='Symphony') as w:
        assert w.logMstar_ratio is None
        with pytest.raises(ValueError, match='unavailable for version'):
            w.evolved_Mstar(np.zeros(n))
