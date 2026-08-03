"""Regression tests for repository-root resolution in the SLURM launchers.

Bug this pins: every launcher derived REPO from ``${BASH_SOURCE[0]}``. SLURM
copies a batch script to a node-local spool directory (/var/spool/slurmd)
before executing it, so under ``sbatch`` that expands to the spool copy, not
the in-repo script. Every task in a 387-task regeneration died in ~12 s with
``can't open file '/var/spool/slurmd/python/compute_weights.py'``. The failure
is invisible to any test that runs the launcher from a shell, because there
``BASH_SOURCE`` is correct -- it only appears under a real scheduler.

The tests below therefore exercise the resolution snippet with the script
placed *outside* the repository, which is what the scheduler actually does.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHERS = sorted(REPO_ROOT.glob('scripts/*.sh'))


def _repo_resolution_snippet(text, name):
    """Extract the REPO assignment plus its validation guard.

    Fails with a clear message rather than a bare ValueError when a launcher
    is added without the guard, or an existing one's wording drifts.
    """
    marker = 'REPO="${SLURM_SUBMIT_DIR:-'
    if marker not in text:
        pytest.fail(f'{name} has no SLURM_SUBMIT_DIR-based REPO resolution; '
                    'a launcher without it breaks under sbatch (see module '
                    'docstring).')
    start = text.index(marker)
    end = text.index('fi\n', start) + len('fi\n')
    return text[start:end]


def test_launchers_exist():
    assert LAUNCHERS, 'no launcher scripts found under scripts/'


@pytest.mark.parametrize('launcher', LAUNCHERS, ids=lambda p: p.name)
def test_repo_not_derived_from_bash_source_alone(launcher):
    """The pre-fix line is exactly what SLURM breaks; it must not reappear."""
    text = launcher.read_text()
    assert not re.search(r'^REPO="\$\(cd "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/\.\." && pwd\)"$',
                         text, re.MULTILINE), (
        f'{launcher.name} derives REPO from BASH_SOURCE alone; under sbatch '
        'that resolves to the node-local spool copy, not the repository.')
    assert 'SLURM_SUBMIT_DIR' in text, (
        f'{launcher.name} does not consult SLURM_SUBMIT_DIR')


@pytest.mark.parametrize('launcher', LAUNCHERS, ids=lambda p: p.name)
def test_resolves_to_repo_when_run_from_spool(launcher, tmp_path):
    """Script living outside the repo + SLURM_SUBMIT_DIR set -> repo root."""
    spool = tmp_path / 'spool'
    spool.mkdir()
    snippet = _repo_resolution_snippet(launcher.read_text(), launcher.name)
    probe = spool / launcher.name
    probe.write_text(snippet + 'echo "${REPO}"\n')
    out = subprocess.run(['bash', str(probe)], capture_output=True, text=True,
                         env={'PATH': '/usr/bin:/bin',
                              'SLURM_SUBMIT_DIR': str(REPO_ROOT)})
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip()) == REPO_ROOT


@pytest.mark.parametrize('launcher', LAUNCHERS, ids=lambda p: p.name)
def test_fails_loudly_when_repo_root_is_wrong(launcher, tmp_path):
    """Neither signal usable -> exit 2 with a message, not a silent wrong path.

    This is the pre-fix behaviour's actual cost: without the guard the script
    ran on to python and produced a bare ENOENT 43 times per version.
    """
    spool = tmp_path / 'spool'
    spool.mkdir()
    snippet = _repo_resolution_snippet(launcher.read_text(), launcher.name)
    probe = spool / launcher.name
    probe.write_text(snippet + 'echo "${REPO}"\n')
    out = subprocess.run(['bash', str(probe)], capture_output=True, text=True,
                         env={'PATH': '/usr/bin:/bin'})
    assert out.returncode == 2, (
        f'{launcher.name} did not reject a non-repository REPO '
        f'(rc={out.returncode}, stdout={out.stdout!r})')
    assert 'python/config.py' in out.stderr


@pytest.mark.parametrize('launcher', LAUNCHERS, ids=lambda p: p.name)
def test_sbatch_output_dir_is_repo_relative(launcher):
    """--output is resolved by SLURM against the submission directory.

    It cannot use ${REPO}, which does not exist until the body runs, so the
    path must stay relative -- an absolute or BASH_SOURCE-derived one would
    reintroduce the same class of bug at a point no guard can catch.
    """
    text = launcher.read_text()
    for line in text.splitlines():
        if line.startswith('#SBATCH --output='):
            value = line.split('=', 1)[1]
            assert not value.startswith('/'), (
                f'{launcher.name}: --output must be relative to the '
                f'submission directory, got {value}')
            assert '${' not in value, (
                f'{launcher.name}: --output cannot reference shell variables; '
                'SLURM expands it before the body runs')


def test_launcher_out_dirs_are_tracked():
    """--output dirs must survive a fresh clone, not merely exist locally.

    SLURM resolves --output against the submission directory at dispatch,
    before the body's mkdir runs, so a log directory that exists only on this
    machine drops every array task for anyone who clones the repository.
    Asserting is_dir() alone passes here while the fresh-clone case is broken,
    which is exactly how this went unnoticed -- so assert git-tracked.
    """
    tracked = set(subprocess.run(
        ['git', 'ls-files', 'scripts/'], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True).stdout.split())
    for launcher in LAUNCHERS:
        # An untracked launcher is work in progress: it has no fresh-clone
        # behaviour to constrain yet, and its .gitkeep lands in the same commit
        # it does. Checking it here would fail the suite for every launcher
        # between being written and being committed.
        if str(launcher.relative_to(REPO_ROOT)) not in tracked:
            continue
        for line in launcher.read_text().splitlines():
            if line.startswith('#SBATCH --output='):
                out_dir = Path(line.split('=', 1)[1]).parent
                assert (REPO_ROOT / out_dir).is_dir(), (
                    f'{launcher.name} writes stdout to {out_dir}, which does '
                    'not exist; SLURM will drop the job before the body runs')
                keep = str(out_dir / '.gitkeep')
                assert keep in tracked, (
                    f'{launcher.name} writes stdout to {out_dir}, which is not '
                    f'recreated by a fresh clone ({keep} is untracked). Add a '
                    '.gitkeep and a .gitignore negation for it.')
