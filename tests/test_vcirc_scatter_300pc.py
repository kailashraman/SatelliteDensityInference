"""Regression tests for vcirc_scatter_300pc.py.

Covers the LAUNCHER_N_TASKS equality assert (the len(dwarf_names)*N_SHMR
check), exercised by actually running the script via runpy -- not restated
arithmetic -- with obs.dwarf_names monkeypatched so the two disagree, in both
directions (shrinkage and growth). LAUNCHER_N_TASKS is itself parsed by the
script from scripts/vcirc_scatter_300pc.sh's --array bound, so this exercises
that parsing too rather than a value hand-restated here. `import Jdata as
obs` inside the script returns the SAME cached module object this test
imports, so patching `obs.dwarf_names` here is visible to the script's own
module-level code.

The script executes its whole pipeline at import (it is a script, not a
library, and needs sys.argv plus a version/catalog), so the "assert passes"
case is only exercised as far as the next thing requiring real migrated
data (the MASSPROF input, results/paper_massProfile/<version>/massProfile.npz)
-- reachable in the default tier via a synthetic version + tmp_path, without
touching real results/. Beyond that point (loading real weights, writing a
real row) is not achievable without the migrated tree; that is not attempted
here.
"""

import runpy
import sys

import h5py
import numpy as np
import pytest

import config

pytest.importorskip('Jdata', reason='needs the J_calc environment')
import Jdata as obs  # noqa: E402  (import after importorskip, same as Jdata itself elsewhere)
import vcirc_scatter_params as vsp  # noqa: E402

pytestmark = pytest.mark.needs_satgen

SCRIPT_PATH = config.REPO_ROOT / 'python' / 'vcirc_scatter_300pc.py'


@pytest.fixture
def version_registry(tmp_path, monkeypatch):
    """Register a throwaway SatGen version with a real (empty) h5 on disk.

    Points VCIRC_SCATTER_DIR/PAPER_MASSPROFILE_DIR/ADDITIONAL_DIR at tmp_path
    so the script's os.makedirs and file reads never touch the real repo
    tree.
    """
    version = 'synthetic_task_test'
    h5_path = tmp_path / f'{version}.h5'
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('virial_mass', data=np.ones(1))
    monkeypatch.setattr(config, 'ADDITIONAL_DIR', tmp_path)
    monkeypatch.setitem(config.H5_REGISTRY, version, h5_path.name)
    monkeypatch.setattr(config, 'VCIRC_SCATTER_DIR', tmp_path / 'vcirc_scatter_300pc')
    monkeypatch.setattr(config, 'PAPER_MASSPROFILE_DIR', tmp_path / 'paper_massProfile')
    return version


def _run_script(task_id, version):
    argv = sys.argv
    sys.argv = ['vcirc_scatter_300pc.py', str(task_id), version]
    try:
        runpy.run_path(str(SCRIPT_PATH), run_name='__main__')
    finally:
        sys.argv = argv


# A distinctive phrase from the n_tasks == LAUNCHER_N_TASKS equality assert's
# message ONLY -- not shared with the --array regex-parse-failure assert's
# message, nor with the literal (and never-actually-raised, since it is
# interpolated to a number) string 'LAUNCHER_N_TASKS'. A match on either of
# those would let both tests below pass even if the launcher's --array line
# were unparseable, which is not what they claim to cover.
_EQUALITY_ASSERT_PHRASE = 'update the --array bound to 0-'


def test_task_count_shrinkage_fails_loudly_at_task_0(version_registry, monkeypatch):
    """Truncating dwarf_names so len(dwarf_names)*N_SHMR != LAUNCHER_N_TASKS
    must raise before any file is touched."""
    monkeypatch.setattr(obs, 'dwarf_names', obs.dwarf_names[:-1])
    with pytest.raises(AssertionError, match=_EQUALITY_ASSERT_PHRASE):
        _run_script(0, version_registry)


def test_task_count_growth_fails_loudly_at_task_0(version_registry, monkeypatch):
    """Growing dwarf_names must also raise at task 0: the bug this equality
    assert exists to catch is that a plain `TASK_ID < n_tasks` inequality
    check misses growth entirely -- every task id already in range stays in
    range, and the new tail dwarfs are silently never dispatched."""
    monkeypatch.setattr(obs, 'dwarf_names', list(obs.dwarf_names) + ['synthetic_extra_dwarf'])
    with pytest.raises(AssertionError, match=_EQUALITY_ASSERT_PHRASE):
        _run_script(0, version_registry)


@pytest.fixture
def launcher_copy(tmp_path, monkeypatch):
    """Point config.REPO_ROOT at a scratch tree holding a COPY of the real
    launcher, with its --array line replaced by whatever the caller wants to
    probe. Mutating a copy rather than the real
    scripts/vcirc_scatter_300pc.sh is what lets these tests exercise the
    parse-failure path itself, per docs/review-checklist.md's "cross-file
    constants parsed from another file" class -- asserting against the real
    file only ever exercises the one line it happens to contain today.
    """
    real_source = (config.REPO_ROOT / 'scripts' / 'vcirc_scatter_300pc.sh').read_text()

    def _make(array_line):
        scripts_dir = tmp_path / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            array_line if line.lstrip().startswith('#SBATCH --array=') else line
            for line in real_source.splitlines()
        ]
        (scripts_dir / 'vcirc_scatter_300pc.sh').write_text('\n'.join(lines) + '\n')
        monkeypatch.setattr(config, 'REPO_ROOT', tmp_path)

    return _make


def test_malformed_array_line_fails_loudly(version_registry, launcher_copy):
    """No '#SBATCH --array=0-<N>' line at all in the header -> a specific,
    loud AssertionError, not a silent fall-through to some default."""
    launcher_copy('#SBATCH --array=this-is-not-a-range')
    with pytest.raises(AssertionError, match='could not find a'):
        _run_script(0, version_registry)


def test_array_throttle_suffix_parses_correctly(version_registry, launcher_copy):
    """A '%N' throttle suffix (e.g. '0-429%20') is a routine, non-semantic
    edit to the --array bound -- SLURM parses the bound the same way with or
    without it -- and must not break LAUNCHER_N_TASKS parsing."""
    n_tasks = len(obs.dwarf_names) * len(vsp.SHMR_NAMES)
    launcher_copy(f'#SBATCH --array=0-{n_tasks - 1}%20')
    # Proceeds past the array-bound assert into the next stage (unstamped
    # inputs), proving the %20 suffix was parsed rather than rejected.
    with pytest.raises(ValueError, match='unstamped'):
        _run_script(0, version_registry)


def test_task_count_agreement_proceeds_past_the_assert(version_registry):
    """When len(dwarf_names)*N_SHMR == LAUNCHER_N_TASKS (the real, unpatched
    case), the script must NOT raise AssertionError at that check -- it
    should proceed into the next stage, which for this synthetic version
    fails for an unrelated reason (unstamped inputs), proving execution got
    past the count check rather than merely not being tested."""
    with pytest.raises(ValueError, match='unstamped'):
        _run_script(0, version_registry)


