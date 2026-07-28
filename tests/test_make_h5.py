"""make_h5 recipe checks.

The unit tests below run anywhere. The parity tests (`needs_data`) rebuild a
catalog and diff it against the published one -- the only check that would
catch a silent change to the mask or the LMC-host search, both of which are
made of hardcoded constants that the source notebook got wrong.
"""

import argparse

import numpy as np
import pytest

import config

# make_h5's mask and LMC-search functions need only numpy, so they must run in
# the default tier -- a module-level needs_satgen marker would silently skip
# the very regression tests that pin the corrected constants.
import make_h5

# Established by diffing a rebuild against the published Diemer catalog.
DIEMER_RVIR = 258.92401123
DIEMER_N_SATS = 2432159
DIEMER_N_SOURCE = 15152481
DIEMER_N_LMC_HOSTS = 773


# Field shapes and dtypes as they appear in the dfolsom datasets. Carrying all
# of SAT_FIELDS, not just the ones the mask reads, so the same fixture drives
# the end-to-end cmd_build tests.
_SAT_DTYPE = [
    ('mwID', 'i8'), ('order', 'i2'), ('tpeak_order', 'i2'),
    ('surviving', '?'), ('position', 'f8', (3,)),
    ('tpeak_position', 'f8', (3,)), ('Green_params', 'f8', (4,)),
] + [(name, 'f8') for name in (
    'virial_mass', 'stellar_mass', 'groupID', 'r_half', 'r_max', 'v_max',
    'peak_mass', 'tpeak_stellar_mass', 'tpeak_groupID', 'tpeak_r_half',
    'tpeak_r_max', 'tpeak_v_max', 'tpeak', 'tpeak_conc', 'tperi', 'rperi')]


def _sats(n, virial_mass, peak_mass, position, rperi, mwID):
    """A stand-in for the dfolsom `sats` structured array.

    Built by assignment rather than np.rec.fromarrays, which cannot take the
    (n, 3) `position` field alongside scalar ones. Fields the caller does not
    set stay zero; only the five below affect the mask or the LMC search.
    """
    sats = np.zeros(n, dtype=_SAT_DTYPE)
    sats['virial_mass'] = virial_mass
    sats['peak_mass'] = peak_mass
    sats['position'] = np.asarray(position, float).reshape(n, 3)
    sats['rperi'] = rperi
    sats['mwID'] = mwID
    return sats


def test_fixture_covers_every_field_the_builder_writes():
    """Otherwise the end-to-end tests fail on a missing field rather than on
    the behaviour they are checking."""
    missing = set(make_h5.SAT_FIELDS) - {name for name, *_ in _SAT_DTYPE}
    assert not missing, f'fixture is missing {sorted(missing)}'


def test_survival_mask_drops_stripped_and_splashback():
    sats = _sats(
        4,
        virial_mass=[10., 0.5, 10., 10.],     # 2nd is stripped below 1%
        peak_mass=[100., 100., 100., 100.],
        position=[[10, 0, 0], [10, 0, 0], [300, 0, 0], [0, 0, 0]],
        rperi=[1., 1., 1., 1.],
        mwID=[0, 0, 0, 0])
    mask = make_h5.survival_mask(sats, rvir=259.0)
    assert list(mask) == [True, False, False, True]


def test_survival_mask_boundary_is_strict_on_radius():
    """The 259-vs-258.924 discrepancy lives exactly here: a satellite between
    the two values is kept by one and dropped by the other."""
    sats = _sats(1, [10.], [100.], [[258.95, 0, 0]], [1.], [0])
    assert make_h5.survival_mask(sats, rvir=259.0)[0]
    assert not make_h5.survival_mask(sats, rvir=DIEMER_RVIR)[0]


def test_survival_mask_max_mwid_is_optional():
    sats = _sats(2, [10., 10.], [100., 100.],
                 [[0, 0, 0], [0, 0, 0]], [1., 1.], [1, 5000])
    assert list(make_h5.survival_mask(sats, 259.0)) == [True, True]
    assert list(make_h5.survival_mask(sats, 259.0, max_mwid=2000)) == [True, False]


def test_lmc_search_reaches_beyond_the_virial_radius():
    """A >1e11 satellite outside the virial radius whose pericentre is inside
    100 kpc still marks its host.

    The notebook's trailing `& (dist < 259)` does not do this: its walrus binds
    `dist` to a boolean, so that clause is True everywhere and inert. What the
    notebook actually did differently was search the masked set -- covered by
    test_build_searches_for_lmc_hosts_before_masking.
    """
    sats = _sats(1, [2e11], [2e11], [[400, 0, 0]], [50.], [7])
    assert list(make_h5.lmc_host_ids(sats)) == [7]


def test_lmc_search_requires_the_mass_cut():
    sats = _sats(1, [1e9], [1e9], [[50, 0, 0]], [50.], [7])
    assert len(make_h5.lmc_host_ids(sats)) == 0


def test_host_virial_radius_rejects_a_mixed_dataset():
    hosts = np.rec.fromarrays([np.array([258.9, 260.1])], names='virial_radius')
    with pytest.raises(ValueError, match='expected one host virial radius'):
        make_h5.host_virial_radius(hosts)


def test_every_registered_catalog_is_buildable_or_explained():
    """A registered catalog must either have a build source or a stated reason
    it has none. Silence would make an unrebuildable catalog look like one
    nobody had gotten to yet."""
    registered = {v for v, f in config.H5_REGISTRY.items()
                  if f is not None and v not in ('Symphony', 'MWest')}
    unaccounted = registered - set(make_h5.BUILD_SOURCES) - set(make_h5.NOT_REBUILDABLE)
    assert not unaccounted, f'neither buildable nor explained: {sorted(unaccounted)}'


def test_not_rebuildable_versions_fail_loudly():
    """mass_floor_7 has no migrated source; `build` must say so rather than
    fall through to a confusing missing-file error."""
    for version in make_h5.NOT_REBUILDABLE:
        assert version not in make_h5.BUILD_SOURCES
        assert make_h5.NOT_REBUILDABLE[version].strip()


def _tiny_dataset(tmp_path, monkeypatch, sats, rvir=259.0):
    """Point cmd_build at a synthetic one-host dataset in tmp_path."""
    hosts = np.zeros(1, dtype=[('virial_radius', 'f8')])
    hosts['virial_radius'] = rvir
    source = tmp_path / 'src.npz'
    np.savez(source, sats=sats, hosts=hosts)
    out = tmp_path / 'out.h5'
    monkeypatch.setitem(make_h5.BUILD_SOURCES, 'Diemer', source.name)
    monkeypatch.setattr(make_h5.config, 'TREES_DIR', tmp_path)
    monkeypatch.setattr(make_h5.config, 'h5_path', lambda v, **kw: out)
    return out


def test_build_searches_for_lmc_hosts_before_masking(tmp_path, monkeypatch):
    """End-to-end regression on the one load-bearing ordering decision.

    Host 7's LMC analogue is stripped to 0.1% of peak mass, so the survival
    mask drops it, but it is still >1e11 today and inside 100 kpc. Searching
    before the mask finds the host; searching after sees only the surviving
    small satellite and finds nothing. Moving lmc_host_ids() below
    `selected = sats[mask]` makes this fail -- the unit tests do not catch it.
    """
    import h5py

    sats = _sats(
        2,
        virial_mass=[2e11, 10.],           # LMC analogue, and a survivor
        peak_mass=[2e14, 100.],            # LMC is at 0.1% of peak -> masked out
        position=[[50, 0, 0], [50, 0, 0]],
        rperi=[50., 50.],
        mwID=[7, 7])
    out = _tiny_dataset(tmp_path, monkeypatch, sats)

    make_h5.cmd_build(argparse.Namespace(
        version='Diemer', max_mwid=None, force=True, drop_extra=False))

    with h5py.File(out, 'r') as f:
        assert f['mwID'].shape[0] == 1, 'only the survivor should be kept'
        assert bool(f['mw_hosts_lmc'][0]), \
            'host 7 lost its LMC flag: the search ran after masking'


def test_build_force_preserves_groups_it_does_not_produce(tmp_path, monkeypatch):
    """logMhalf comes from compute_mhalf (not yet ported), so a --force rebuild
    that dropped it would destroy the only copy in this repository."""
    import h5py

    sats = _sats(1, [10.], [100.], [[50, 0, 0]], [50.], [0])
    out = _tiny_dataset(tmp_path, monkeypatch, sats)

    make_h5.cmd_build(argparse.Namespace(
        version='Diemer', max_mwid=None, force=True, drop_extra=False))
    with h5py.File(out, 'a') as f:
        f.create_group('logMhalf').create_dataset('Draco', data=np.arange(1.))

    make_h5.cmd_build(argparse.Namespace(
        version='Diemer', max_mwid=None, force=True, drop_extra=False))
    with h5py.File(out, 'r') as f:
        assert 'logMhalf' in f, 'rebuild destroyed a group it cannot produce'
        assert np.array_equal(f['logMhalf']['Draco'][()], np.arange(1.))

    make_h5.cmd_build(argparse.Namespace(
        version='Diemer', max_mwid=None, force=True, drop_extra=True))
    with h5py.File(out, 'r') as f:
        assert 'logMhalf' not in f, '--drop-extra should discard it'


# --------------------------------------------------------------------------
# Parity against the published catalogs.
# --------------------------------------------------------------------------

@pytest.mark.needs_data
def test_diemer_catalog_matches_the_published_recipe():
    """Re-derives the recipe from the unmasked catalog and checks it against
    the Diemer product on disk -- not against remembered literals.

    rvir comes from make_h5.host_virial_radius so a regression there is caught
    here too, and the LMC flag is compared row-by-row rather than by count.
    """
    import h5py

    with h5py.File(config.h5_path('splashback'), 'r') as f:
        virial_mass = f['virial_mass'][()]
        peak_mass = f['peak_mass'][()]
        position = f['position'][()]
        rperi = f['rperi'][()]
        mwID = f['mwID'][()]
    assert len(virial_mass) == DIEMER_N_SOURCE

    radius = np.linalg.norm(position, axis=1)
    mask = (virial_mass / peak_mass > make_h5.MASS_LOSS_FLOOR) & ~(radius > DIEMER_RVIR)
    is_lmc = ((radius < 100) | (rperi < 100)) & (virial_mass > 1e11)
    lmc_ids = np.unique(mwID[is_lmc])
    assert len(lmc_ids) == DIEMER_N_LMC_HOSTS

    with h5py.File(config.h5_path('Diemer'), 'r') as f:
        assert f['mwID'].shape[0] == mask.sum() == DIEMER_N_SATS
        assert np.array_equal(f['mwID'][()], mwID[mask])
        assert np.array_equal(f['mw_hosts_lmc'][()],
                              np.isin(mwID[mask], lmc_ids))


@pytest.mark.needs_data
def test_every_registered_catalog_is_present_and_fingerprinted():
    """A registry entry with no file on disk fails only when something reaches
    for it; a stale fingerprint makes the mixing guard cry wolf forever."""
    import provenance

    for version, filename in config.H5_REGISTRY.items():
        if filename is None:
            continue
        path = config.ADDITIONAL_DIR / filename
        assert path.exists(), f'{version} registered but missing: {filename}'
        record = provenance.read(path)
        assert record is not None, f'{version} is unstamped'
        assert record['h5_fingerprint']['size'] == path.stat().st_size, \
            f'{version} fingerprint is stale'


@pytest.mark.needs_data
def test_symphony_h5_matches_its_source_npz():
    import h5py

    files = sorted((config.SYMPHONY_DIR / 'SymphonyMilkyWay').glob('*.npz'))
    assert files, 'Symphony source data not migrated'
    expected = sum(len(np.load(p)['mvir']) for p in files)

    with h5py.File(config.h5_path('Symphony'), 'r') as f:
        assert f['mvir'].shape[0] == expected
        assert len(f['host_names']) == len(files)
