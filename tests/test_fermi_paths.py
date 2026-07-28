"""Regression test for the fermi_funcs path constants.

Upstream, `dSphs_csv_path` was '../fermi_legacy/' while every sibling constant
in the same block carried the 'data/' segment: '../data/fermi_legacy/...'. No
code path read it, so the error never surfaced -- it would have fired the first
time a new figure or variant touched that branch, and looked like the new
caller's fault.

These assertions fail against the pre-fix constant.
"""

from pathlib import Path

import pytest

import config

fermi_funcs = pytest.importorskip('fermi_funcs',
                                  reason='needs the J_calc environment')

PATH_CONSTANTS = [
    'dSphs_csv_path', 'dSphs_sed_path', 'dSphs_sed_path_update',
    'dSphs_TS_prof_path', 'blank_field_csv_path', 'blank_TS_prof_path',
    'cirelli_dir',
]


def _is_under(value, root):
    """Path-component containment.

    String startswith() would accept data/fermi_legacy_update/ as being under
    data/fermi_legacy/, and a sibling data_backup/ as being under data/.
    """
    parts = Path(value).parts
    root_parts = Path(root).parts
    return parts[:len(root_parts)] == root_parts


@pytest.mark.needs_satgen
@pytest.mark.parametrize('name', PATH_CONSTANTS)
def test_paths_are_inside_the_repository_data_dir(name):
    """Catches the missing 'data/' segment, and any path escaping the repo."""
    value = getattr(fermi_funcs, name)
    assert _is_under(value, config.DATA_DIR), (
        f'{name} = {value!r} is not under {config.DATA_DIR}')


@pytest.mark.needs_satgen
def test_dSphs_csv_path_shares_the_legacy_root_with_its_siblings():
    """The specific bug: it sat one directory above the rest."""
    legacy_root = config.DATA_DIR / 'fermi_legacy'
    for name in ['dSphs_csv_path', 'dSphs_sed_path', 'dSphs_TS_prof_path',
                 'blank_field_csv_path', 'blank_TS_prof_path', 'cirelli_dir']:
        assert _is_under(getattr(fermi_funcs, name), legacy_root), name


@pytest.mark.needs_satgen
def test_update_path_is_the_only_one_using_the_update_tree():
    """The drafts recast Circiello:2026inp, so the SED path must point at the
    update tree while TS profiles and blank fields stay on the legacy one."""
    update_root = config.DATA_DIR / 'fermi_legacy_update'
    assert _is_under(fermi_funcs.dSphs_sed_path_update, update_root)
    for name in set(PATH_CONSTANTS) - {'dSphs_sed_path_update'}:
        assert not _is_under(getattr(fermi_funcs, name), update_root), name


@pytest.mark.needs_satgen
@pytest.mark.parametrize('name', PATH_CONSTANTS)
def test_paths_keep_their_trailing_separator(name):
    """Every consumer builds filenames by concatenation, e.g.
    f'{cirelli_dir}AtProduction_all/...'. Losing the slash silently changes
    the directory rather than raising."""
    assert str(getattr(fermi_funcs, name)).endswith('/'), name


@pytest.mark.needs_satgen
@pytest.mark.needs_data
@pytest.mark.parametrize('name', PATH_CONSTANTS)
def test_paths_exist_once_the_data_is_migrated(name):
    """A path constant with no test that its target exists is how the original
    bug survived. Runs only with --rundata, since it needs the copied tree."""
    assert Path(getattr(fermi_funcs, name)).exists()
