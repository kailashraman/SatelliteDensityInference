"""config.py resolves paths from the module, not from cwd."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import config

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize('cwd', ['python', 'scripts', 'tests', '/tmp'])
def test_paths_are_cwd_independent(cwd):
    """SatGen_Dwarf required cwd == python/; nothing here may."""
    directory = cwd if os.path.isabs(cwd) else REPO_ROOT / cwd
    result = subprocess.run(
        [sys.executable, '-c',
         'import sys; sys.path.insert(0, %r); import config; '
         'print(config.REPO_ROOT); print(config.DATA_DIR); '
         'print(config.h5_path("Diemer"))' % str(REPO_ROOT / 'python')],
        cwd=str(directory), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.split()
    assert lines[0] == str(REPO_ROOT)
    assert lines[1] == str(REPO_ROOT / 'data')
    assert lines[2] == str(config.h5_path('Diemer'))


def test_h5_path_known_versions():
    for version in ['Diemer', 'Zhao', 'm2e12', 'Diemer_0p12', 'mass_floor_7',
                    'Symphony', 'MWest']:
        path = config.h5_path(version)
        assert path.parent == config.ADDITIONAL_DIR
        assert path.suffix == '.h5'


def test_h5_path_unknown_version_lists_valid_keys():
    """The migrated get_h5 had no else-branch and raised UnboundLocalError."""
    with pytest.raises(KeyError) as excinfo:
        config.h5_path('Correa')
    message = str(excinfo.value)
    assert 'Correa' in message
    assert 'Diemer' in message


def test_h5_path_not_migrated_version_says_so():
    """The message must give the reason, not just fail -- so a reader can tell
    'we chose not to migrate this' from 'you mistyped a version'."""
    with pytest.raises(KeyError, match='not usable') as excinfo:
        config.h5_path('Geha')
    assert 'unresolved' in str(excinfo.value)


def test_reweighting_aliases_share_a_catalog():
    """lmc/lvdb/mcdaniel etc. reweight the Diemer catalog, not their own.

    This list must match the *active* branch of compute_weights.py, not every
    version string mentioned in that file.
    """
    for alias in ['lmc', 'lmc_50', 'lvdb', 'mcdaniel', 'mhalf_scatter']:
        assert config.sim_version(alias) == 'Diemer'
        assert config.h5_path(alias) == config.h5_path('Diemer')


def test_resolve_alias_false_refuses_reweighting_variants():
    """halo_weights.get_h5 must not resolve aliases.

    ResampleMstar reads mhalf/<version>/ with the raw version string, so
    resolving in get_h5 but not there would make ResampleMstar('lmc')
    constructible with the Diemer catalog paired to mhalf/lmc/ -- a
    mixed-version object the pre-migration code refused to build.
    """
    assert config.h5_path('lmc', resolve_alias=True) == config.h5_path('Diemer')
    with pytest.raises(KeyError):
        config.h5_path('lmc', resolve_alias=False)
    # Non-alias versions behave identically either way.
    assert (config.h5_path('Diemer', resolve_alias=False)
            == config.h5_path('Diemer'))


def test_splashback_is_migrated():
    """It is a live branch upstream: tidal_stripping.pdf's f_disrupted panel
    reads the '_all' catalog, which retains disrupted halos. Diemer has them
    masked out, so substituting it would yield a plausible but wrong panel."""
    path = config.h5_path('splashback')
    assert path.name == 'm12res8_10k_Diemer+scatter_sim_all.h5'
    assert 'splashback' not in config.NOT_MIGRATED


def test_geha_is_not_an_alias():
    """Its Geha->Diemer branch is commented out upstream, and the dwarf list
    it was keyed to (obs.geha_names) no longer exists. Treating it as a plain
    Diemer reweight would silently mis-key 25 dwarfs onto a 43-name index."""
    assert 'Geha' not in config.SIM_VERSION_ALIASES
    assert config.sim_version('Geha') == 'Geha'
    with pytest.raises(KeyError, match='unresolved'):
        config.h5_path('Geha')


def test_external_paths_are_overridable():
    """Tests must be able to point the two upstream deps at fixtures.

    Run in a subprocess rather than reloading the shared module in-process, so
    this cannot leak path state into other tests via collection order.
    """
    env = dict(os.environ,
               SDI_TREES_DIR='/tmp/fake-trees',
               SDI_DJA_RESULTS_DIR='/tmp/fake-dja')
    result = subprocess.run(
        [sys.executable, '-c',
         'import sys; sys.path.insert(0, %r); import config; '
         'print(config.TREES_DIR); print(config.DJA_RESULTS_DIR)'
         % str(REPO_ROOT / 'python')],
        capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.split()
    assert lines[0] == '/tmp/fake-trees'
    assert lines[1] == '/tmp/fake-dja'
