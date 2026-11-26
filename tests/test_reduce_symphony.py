"""reduce_symphony checks.

The unit tests run anywhere: a fake `symlib` with a tiny synthetic catalog is
injected, so neither symlib nor the raw catalogs are needed (colossus is). The
parity test (`needs_data`) re-reduces the real symlib catalogs and requires the
tracked data/symphony/*.npz to match exactly.
"""

import importlib
import sys
import types
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u

import config
import provenance

pytest.importorskip('colossus')

# kpc (km/s)^2 / Msun, from astropy as the reduction does, so a hardcoded G in
# the code would show up here.
G = const.G.to(u.kpc * (u.km / u.s) ** 2 / u.Msun).value
# Per-host H0 (h = H0 / 100) in the fake, to check that simulation_parameters
# is read on every call (real symlib keys it on the suite, not the host).
HOST_H0 = {'Halo1': 70.0, 'Halo2': 60.0}
H0 = 70.0  # any host not in HOST_H0
# Float32 columns, as in the real symlib catalogs.
_HALO_DTYPE = [('ok', '?'), ('rvmax', 'f4'), ('vmax', 'f4'), ('mvir', 'f4'),
               ('x', 'f4', (3,))]
FAKE_VERSIONS = {'symlib': '9.1', 'colossus': '9.2', 'astropy': '9.3'}


def _fake_symlib():
    """4 halos x 3 snapshots; halo 0 is the host. ok at the last snapshot:
    halo 1 True, halo 2 False (True earlier), halo 3 True. Every column differs
    across snapshots."""
    halos = np.zeros((4, 3), dtype=_HALO_DTYPE)
    halos['ok'] = [[True] * 3, [False, True, True],
                   [True, True, False], [True, False, True]]
    halos['rvmax'] = np.array([[40.], [8.], [5.], [2.]]) * [1, 1.1, 1.2]
    halos['vmax'] = np.array([[200.], [30.], [20.], [10.]]) * [1, 1.1, 1.2]
    halos['mvir'] = np.array([[1e12], [1e10], [1e9], [1e8]]) * [1, 1.1, 1.2]
    halos['x'] = np.arange(36.).reshape(4, 3, 3)
    histories = {'mpeak': np.array([2e12, 3e10, 2e9, 5e8], dtype='f4')}
    return types.SimpleNamespace(
        simulation_parameters=lambda d: dict(
            flat=True, H0=HOST_H0.get(Path(d).name, H0), Om0=0.3, Ob0=0.05,
            sigma8=0.8, ns=0.96),
        read_subhalos=lambda d: (halos, histories),
        util=types.SimpleNamespace(
            DEFAULT_HALO_NAMES={'MWest': ['Halo1', 'Halo2']}))


@pytest.fixture
def rs_mod(monkeypatch, request):
    monkeypatch.setitem(sys.modules, 'symlib', _fake_symlib())
    monkeypatch.delitem(sys.modules, 'reduce_symphony', raising=False)
    mod = importlib.import_module('reduce_symphony')
    # Drop the module bound to the fake symlib so later tests re-import it.
    request.addfinalizer(lambda: sys.modules.pop('reduce_symphony', None))
    monkeypatch.setattr(metadata, 'version', lambda name: FAKE_VERSIONS[name])
    return mod


def test_selection_excludes_host_and_uses_last_snapshot(rs_mod):
    out = rs_mod.reduce_host('x', 'h')
    # last-snapshot ok: halos 1 and 3 (halo 2 fails at the end, halo 0 is the host)
    assert np.allclose(out['rmax'], [8 * 1.2, 2 * 1.2])
    assert np.allclose(out['vmax'], [30 * 1.2, 10 * 1.2])
    assert np.allclose(out['mvir'], [1e10 * 1.2, 1e8 * 1.2], rtol=1e-6)
    assert np.array_equal(out['x'], [[15, 16, 17], [33, 34, 35]])
    assert np.allclose(out['mpeak'], [3e10, 5e8])
    assert out['x'].shape == (2, 3)


def _check_nfw_and_cV(out, host_h0):
    """Float32 outputs, so rtol 1e-6 (~10 float32 ulp); the constants below are
    tested to far better than their 1e-5 perturbations."""
    rmax = out['rmax'].astype('f8')
    vmax = out['vmax'].astype('f8')
    assert np.allclose(out['rs'], 0.46241029979236 * rmax, rtol=1e-6)
    assert np.allclose(out['rhos'],
                       1.7212585601570 * (vmax / rmax) ** 2 / G, rtol=1e-6)
    rs, rhos = out['rs'].astype('f8'), out['rhos'].astype('f8')
    x = 0.15 / rs
    assert np.allclose(out['rho150'], rhos / (x * (1 + x) ** 2), rtol=1e-6)
    assert np.allclose(out['cV'], 2 * (vmax / (host_h0 * rmax / 1000)) ** 2,
                       rtol=1e-6)


def test_nfw_parameters_and_cV(rs_mod):
    _check_nfw_and_cV(rs_mod.reduce_host('x', 'h'), H0)


def test_parameters_are_read_on_every_call(rs_mod):
    out1 = rs_mod.reduce_host('Halo1', 'Halo1')
    out2 = rs_mod.reduce_host('Halo2', 'Halo2')
    _check_nfw_and_cV(out1, HOST_H0['Halo1'])
    _check_nfw_and_cV(out2, HOST_H0['Halo2'])
    assert not np.allclose(out1['cV'], out2['cV'])


def test_output_dtypes_match_tracked_npz(rs_mod):
    out = rs_mod.reduce_host('x', 'h')
    tracked = config.SYMPHONY_DIR / 'MWest' / 'Halo004.npz'
    if not tracked.exists():
        pytest.skip(f'no tracked {tracked}')
    with np.load(tracked) as old:
        keys = set(old.files) - {provenance.PROVENANCE_KEY}
        assert set(out) == keys
        for key in keys:
            assert out[key].dtype == old[key].dtype == np.float32, key


def _raw_tree(tmp_path, suite, hosts):
    for host in hosts:
        halos = tmp_path / 'raw' / suite / host / 'halos'
        halos.mkdir(parents=True)
        (halos / 'halos_0.dat').write_bytes(b'x')
    return tmp_path / 'raw'


def _two_host_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'SYMPHONY_RAW_DIR',
                        _raw_tree(tmp_path, 'MWest', ['Halo1', 'Halo2']))


def test_cmd_reduce_writes_one_stamped_npz_per_host(rs_mod, tmp_path, monkeypatch):
    _two_host_tree(tmp_path, monkeypatch)
    out_dir = tmp_path / 'out'
    argv = ['reduce', 'MWest', '--out-dir', str(out_dir)]
    rs_mod.main(argv)
    assert sorted(p.name for p in out_dir.glob('*.npz')) == ['Halo1.npz', 'Halo2.npz']
    record = provenance.read(out_dir / 'Halo1.npz')
    assert record['n_halos'] == 2
    # Not stamped with a version: that would record the downstream MWest.h5.
    assert record['h5_file'] is None and record['h5_fingerprint'] is None
    assert record['satgen_version'] is None
    assert record['suite'] == 'MWest'
    assert record['argv'] == argv
    assert record['symlib_version'] == FAKE_VERSIONS['symlib']
    assert record['colossus_version'] == FAKE_VERSIONS['colossus']
    assert record['astropy_version'] == FAKE_VERSIONS['astropy']
    with np.load(out_dir / 'Halo1.npz') as data:
        assert len(data['mvir']) == 2
    # simulation_parameters was read per call (fake H0 differs by host).
    with np.load(out_dir / 'Halo1.npz') as d1, np.load(out_dir / 'Halo2.npz') as d2:
        _check_nfw_and_cV(d1, HOST_H0['Halo1'])
        _check_nfw_and_cV(d2, HOST_H0['Halo2'])


def test_cmd_reduce_records_sys_argv_when_called_without_argv(
        rs_mod, tmp_path, monkeypatch):
    _two_host_tree(tmp_path, monkeypatch)
    argv = ['reduce', 'MWest', '--out-dir', str(tmp_path / 'out')]
    monkeypatch.setattr(sys, 'argv', ['reduce_symphony.py'] + argv)
    rs_mod.main()
    assert provenance.read(tmp_path / 'out' / 'Halo1.npz')['argv'] == argv


def test_cmd_reduce_refuses_overwrite_without_force(rs_mod, tmp_path, monkeypatch):
    _two_host_tree(tmp_path, monkeypatch)
    args = ['reduce', 'MWest', '--out-dir', str(tmp_path / 'out')]
    rs_mod.main(args)
    with pytest.raises(SystemExit, match='--force'):
        rs_mod.main(args)
    rs_mod.main(args + ['--force'])


def test_cmd_reduce_overwrite_refusal_leaves_no_partial_output(
        rs_mod, tmp_path, monkeypatch):
    _two_host_tree(tmp_path, monkeypatch)
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    (out_dir / 'Halo2.npz').write_bytes(b'old')  # the later host exists
    with pytest.raises(SystemExit, match='--force'):
        rs_mod.main(['reduce', 'MWest', '--out-dir', str(out_dir)])
    assert [p.name for p in out_dir.iterdir()] == ['Halo2.npz']


def test_cmd_reduce_errors_without_host_dirs(rs_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'SYMPHONY_RAW_DIR', tmp_path / 'empty')
    with pytest.raises(SystemExit, match='no host directories'):
        rs_mod.main(['reduce', 'MWest', '--out-dir', str(tmp_path / 'out')])


def test_cmd_reduce_errors_on_wrong_host_names(rs_mod, tmp_path, monkeypatch):
    # Right count would not do: same number of hosts, one wrong name.
    monkeypatch.setattr(config, 'SYMPHONY_RAW_DIR',
                        _raw_tree(tmp_path, 'MWest', ['Halo1', 'Halo3']))
    out_dir = tmp_path / 'out'
    with pytest.raises(SystemExit, match=r"missing \['Halo2'\].*unexpected \['Halo3'\]"):
        rs_mod.main(['reduce', 'MWest', '--out-dir', str(out_dir)])
    assert not out_dir.exists()


def test_cmd_reduce_skips_dot_directories(rs_mod, tmp_path, monkeypatch):
    _two_host_tree(tmp_path, monkeypatch)
    (config.SYMPHONY_RAW_DIR / 'MWest' / '.cache').mkdir()
    rs_mod.main(['reduce', 'MWest', '--out-dir', str(tmp_path / 'out')])


def test_cmd_reduce_refuses_stray_npz_in_out_dir(rs_mod, tmp_path, monkeypatch):
    _two_host_tree(tmp_path, monkeypatch)
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    (out_dir / 'Halo9.npz').write_bytes(b'old')
    with pytest.raises(SystemExit, match='Halo9.npz'):
        rs_mod.main(['reduce', 'MWest', '--out-dir', str(out_dir), '--force'])
    assert [p.name for p in out_dir.iterdir()] == ['Halo9.npz']


def test_cmd_reduce_failure_on_second_host_writes_nothing(
        rs_mod, tmp_path, monkeypatch):
    _two_host_tree(tmp_path, monkeypatch)
    real = rs_mod.reduce_host

    def flaky(host_dir, name):
        if name == 'Halo2':
            raise RuntimeError('boom')
        return real(host_dir, name)

    monkeypatch.setattr(rs_mod, 'reduce_host', flaky)
    out_dir = tmp_path / 'out'
    with pytest.raises(RuntimeError, match='boom'):
        rs_mod.main(['reduce', 'MWest', '--out-dir', str(out_dir)])
    assert not out_dir.exists()


def test_cmd_reduce_stamps_before_any_write(rs_mod, tmp_path, monkeypatch):
    _two_host_tree(tmp_path, monkeypatch)
    events = []
    real_stamp, real_savez = provenance.stamp, provenance.savez
    monkeypatch.setattr(provenance, 'stamp',
                        lambda *a, **k: (events.append('stamp'),
                                         real_stamp(*a, **k))[1])
    monkeypatch.setattr(provenance, 'savez',
                        lambda *a, **k: (events.append('savez'),
                                         real_savez(*a, **k))[1])
    rs_mod.main(['reduce', 'MWest', '--out-dir', str(tmp_path / 'out')])
    assert events == ['stamp', 'stamp', 'savez', 'savez']


def test_expected_host_names_match_tracked_data():
    for symlib_suite in ['SymphonyMilkyWay', 'MWest']:
        tracked = {p.stem for p in
                   (config.SYMPHONY_DIR / symlib_suite).glob('*.npz')}
        # Imports the real symlib (or fails with ImportError if absent).
        symlib = pytest.importorskip('symlib')
        assert set(symlib.util.DEFAULT_HALO_NAMES[symlib_suite]) == tracked


def test_cmd_download_requires_credentials(rs_mod, monkeypatch):
    monkeypatch.setenv('SYMLIB_USER', 'someone')
    monkeypatch.setenv('SYMLIB_PASSWORD', 'pw')
    # Each variable missing on its own.
    monkeypatch.delenv('SYMLIB_USER')
    with pytest.raises(SystemExit, match='SYMLIB_USER'):
        rs_mod.main(['download', 'MWest'])
    monkeypatch.setenv('SYMLIB_USER', 'someone')
    monkeypatch.delenv('SYMLIB_PASSWORD')
    with pytest.raises(SystemExit, match='SYMLIB_PASSWORD'):
        rs_mod.main(['download', 'MWest'])


def test_cmd_download_calls_symlib(rs_mod, tmp_path, monkeypatch):
    monkeypatch.setenv('SYMLIB_USER', 'someone')
    monkeypatch.setenv('SYMLIB_PASSWORD', 'pw')
    raw_dir = tmp_path / 'raw'
    monkeypatch.setattr(config, 'SYMPHONY_RAW_DIR', raw_dir)
    calls = []
    monkeypatch.setattr(rs_mod.symlib, 'download_files',
                        lambda *a, **k: calls.append((a, k)), raising=False)
    rs_mod.main(['download', 'MWest'])
    assert calls == [(('someone', 'pw', 'MWest', None, str(raw_dir)),
                      {'target': 'halos'})]


def test_cmd_download_error_does_not_leak_credentials(rs_mod, tmp_path, monkeypatch):
    monkeypatch.setenv('SYMLIB_USER', 'someone')
    monkeypatch.setenv('SYMLIB_PASSWORD', 'hunter2-secret')
    monkeypatch.setattr(config, 'SYMPHONY_RAW_DIR', tmp_path / 'raw')

    def fail(user, password, *args, **kwargs):
        raise ConnectionError(f'401 for https://{user}:{password}@host/x')

    monkeypatch.setattr(rs_mod.symlib, 'download_files', fail, raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        rs_mod.main(['download', 'MWest'])
    assert 'hunter2-secret' not in str(excinfo.value)
    assert 'someone' not in str(excinfo.value)
    assert 'ConnectionError' in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__


@pytest.mark.parametrize('user,password,leftovers', [
    # The user is a prefix of the password: shortest-first would leave 'secret'.
    ('bob', 'bobsecret', ['bob', 'secret']),
    # Percent-encoded password, as an HTTP library may render it in a URL.
    ('someone', 'p@ss/w:rd', ['p@ss', 'p%40ss', 'w%3Ard', 'w:rd']),
])
def test_cmd_download_redacts_prefix_and_encoded_secrets(
        rs_mod, tmp_path, monkeypatch, user, password, leftovers):
    from urllib.parse import quote
    monkeypatch.setenv('SYMLIB_USER', user)
    monkeypatch.setenv('SYMLIB_PASSWORD', password)
    monkeypatch.setattr(config, 'SYMPHONY_RAW_DIR', tmp_path / 'raw')

    def fail(user, password, *args, **kwargs):
        raise ConnectionError(f'401 for https://{user}:{password}@host/x '
                              f'(https://{quote(user, safe="")}:'
                              f'{quote(password, safe="")}@host/x)')

    monkeypatch.setattr(rs_mod.symlib, 'download_files', fail, raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        rs_mod.main(['download', 'MWest'])
    for leftover in leftovers:
        assert leftover not in str(excinfo.value)


@pytest.mark.needs_data
@pytest.mark.parametrize('suite,symlib_suite',
                         [('Symphony', 'SymphonyMilkyWay'), ('MWest', 'MWest')])
def test_reduction_matches_tracked_npz(suite, symlib_suite, tmp_path):
    pytest.importorskip('symlib')
    raw = config.SYMPHONY_RAW_DIR / symlib_suite
    if not raw.is_dir():
        pytest.skip(f'no raw symlib catalogs at {raw}')
    import reduce_symphony
    out_dir = tmp_path / 'out'
    reduce_symphony.main(['reduce', suite, '--out-dir', str(out_dir)])
    tracked = config.SYMPHONY_DIR / symlib_suite
    assert ({p.name for p in out_dir.glob('*.npz')}
            == {p.name for p in tracked.glob('*.npz')})
    for path in sorted(out_dir.glob('*.npz')):
        with np.load(path) as new, np.load(tracked / path.name) as old:
            keys = set(new.files) - {provenance.PROVENANCE_KEY}
            assert keys == set(old.files) - {provenance.PROVENANCE_KEY}
            for key in keys:
                assert np.array_equal(new[key], old[key]), (path.name, key)
