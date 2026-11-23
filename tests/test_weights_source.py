"""Tests for the --weights-source {seeded,published} option.

data/halo_weights/ is a seeded regeneration of the SHMR weights;
data/halo_weights_published/ is the unseeded draw that built the paper's
SHMR contours and Fermi products. compute_quantiles.py, save_contours.py,
fermi_reweighting.py and fermi_expected_limits.py take --weights-source to
choose between them. compute_quantiles defaults to seeded (its two outputs go
to separate trees); the other three write the same product tree either way,
results/ holds the published-draw products, so they default to published and
refuse seeded unless SDI_RESULTS_DIR is redirected. Before the option existed nothing read
the published tree, so the paper products could not be regenerated.

Tiers:
  default     source/AST checks, launcher text checks, and subprocess runs that
              fail at argument parsing or at the first weights/quantiles read
              (the modules import SatGen first, hence needs_satgen).
  --rundata   Fermi routing messages (need the real data tree for the earlier
              reads), and parity runs, all written under a tmp SDI_RESULTS_DIR:
              save_contours.py (Segue 1) and compute_quantiles.py (Segue 1,
              Diemer) with --weights-source published must reproduce the stored
              arrays bit-exactly, and fermi_reweighting.py (Draco, update tree)
              must reproduce the stored TS arrays. They are marked needs_data
              only, not slow: conftest reserves `slow` for the mock-pipeline
              calibration run.

The routing/parsing tests fail against the pre-change modules (no
--weights-source handling: bad values and unknown flags run on, and
'published' is silently ignored so the seeded path is named).
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import config
import provenance

REPO = Path(config.REPO_ROOT)
PYTHON_DIR = REPO / 'python'
SCRIPTS_DIR = REPO / 'scripts'

MODULES = ['compute_quantiles', 'save_contours',
           'fermi_reweighting', 'fermi_expected_limits']
FERMI_MODULES = ['fermi_reweighting', 'fermi_expected_limits']


def _src(name):
    return (PYTHON_DIR / f'{name}.py').read_text()


def _run(script, args, results_dir, timeout=900):
    """results_dir None runs with SDI_RESULTS_DIR unset (the default results/)."""
    env = dict(os.environ)
    env.pop('SDI_RESULTS_DIR', None)
    if results_dir is not None:
        env['SDI_RESULTS_DIR'] = str(results_dir)
    return subprocess.run([sys.executable, str(PYTHON_DIR / f'{script}.py'), *args],
                          cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=timeout)


# ---------------------------------------------------------------- source checks

@pytest.mark.parametrize('name', MODULES)
def test_module_parses_weights_source(name):
    src = _src(name)
    assert "'--weights-source'" in src
    assert "'seeded'" in src and "'published'" in src


@pytest.mark.parametrize('name', FERMI_MODULES)
def test_fermi_quantiles_reads_use_the_tree_variable(name):
    """No simple statement mentioning PAPER_QUANTILES_DIR may carry a bare
    'galactocentric' literal; each must go through QUANTILES_TREE."""
    tree = ast.parse(_src(name))
    simple = (ast.Assign, ast.AnnAssign, ast.Expr, ast.Raise, ast.Return)
    n_reads = 0
    for node in ast.walk(tree):
        if not isinstance(node, simple):
            continue
        subtree = list(ast.walk(node))
        if not any(isinstance(n, ast.Attribute) and n.attr == 'PAPER_QUANTILES_DIR'
                   for n in subtree):
            continue
        n_reads += 1
        literals = [n.value for n in subtree
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert 'galactocentric' not in literals, ast.get_source_segment(_src(name), node)
        assert any(isinstance(n, ast.Name) and n.id == 'QUANTILES_TREE'
                   for n in subtree), ast.get_source_segment(_src(name), node)
    assert n_reads >= (3 if name == 'fermi_reweighting' else 1)


@pytest.mark.parametrize('name', FERMI_MODULES)
def test_fermi_stamps_record_weights_source_and_tree(name):
    src = _src(name)
    assert 'weights_source=weights_source, quantiles_tree=QUANTILES_TREE' in src


# ------------------------------------------------------------- launcher checks

@pytest.mark.parametrize('sh', ['save_contours', 'fermi_reweighting_update',
                                'fermi_expected_limits'])
def test_paper_launchers_pass_published(sh):
    text = (SCRIPTS_DIR / f'{sh}.sh').read_text()
    exec_lines = [ln for ln in text.splitlines()
                  if '--weights-source published' in ln and not ln.lstrip().startswith('#')]
    assert exec_lines, f'{sh}.sh does not pass --weights-source published'
    assert 'halo_weights_published' in text


def test_compute_quantiles_launcher_passes_env_weights_source():
    text = (SCRIPTS_DIR / 'compute_quantiles.sh').read_text()
    assert '--weights-source "${WEIGHTS_SOURCE-seeded}"' in text
    assert ('sbatch --export=ALL,WEIGHTS_SOURCE=published scripts/compute_quantiles.sh'
            in text)
    assert 'galactocentric_published' in text
    assert ':-seeded' not in text   # an empty WEIGHTS_SOURCE must reach python


# ------------------------------------------------- parsing (subprocess, no data)

BAD_ARGS = {
    'compute_quantiles': ['0', 'Diemer'],
    'save_contours': ['0', 'Diemer', 'z0'],
    'fermi_reweighting': ['0', '--legacy-version', 'update'],
    'fermi_expected_limits': ['0'],
}


@pytest.mark.needs_satgen
@pytest.mark.parametrize('name', MODULES)
def test_invalid_weights_source_value_rejected(name, tmp_path):
    out = _run(name, BAD_ARGS[name] + ['--weights-source', 'bogus'], tmp_path)
    assert out.returncode != 0
    assert 'ValueError' in out.stderr and '--weights-source' in out.stderr


@pytest.mark.needs_satgen
@pytest.mark.parametrize('name', MODULES)
def test_missing_weights_source_value_rejected(name, tmp_path):
    out = _run(name, BAD_ARGS[name] + ['--weights-source'], tmp_path)
    assert out.returncode != 0
    assert 'ValueError' in out.stderr and '--weights-source' in out.stderr


@pytest.mark.needs_satgen
@pytest.mark.parametrize('name', MODULES)
def test_unknown_flag_rejected(name, tmp_path):
    out = _run(name, BAD_ARGS[name] + ['--weights-sources', 'published'], tmp_path)
    assert out.returncode != 0
    assert 'ValueError' in out.stderr and 'unrecognized' in out.stderr


@pytest.mark.needs_satgen
@pytest.mark.parametrize('name', ['compute_quantiles', 'save_contours'])
def test_positional_scripts_reject_any_extra_token(name, tmp_path):
    out = _run(name, BAD_ARGS[name] + ['extra'], tmp_path)
    assert out.returncode != 0
    assert 'unrecognized' in out.stderr


@pytest.mark.needs_satgen
@pytest.mark.parametrize('name', MODULES)
@pytest.mark.parametrize('extra', [['--weights-source', 'published', '--weights-source', 'published'],
                                   ['--weights-source', 'published', '--weights-source', 'seeded']])
def test_repeated_weights_source_rejected(name, extra, tmp_path):
    out = _run(name, BAD_ARGS[name] + extra, tmp_path)
    assert out.returncode != 0
    assert 'ValueError' in out.stderr and ('unrecognized' in out.stderr or 'more than once' in out.stderr)


@pytest.mark.needs_satgen
@pytest.mark.parametrize('name', MODULES)
def test_empty_weights_source_rejected(name, tmp_path):
    out = _run(name, BAD_ARGS[name] + ['--weights-source', ''], tmp_path)
    assert out.returncode != 0
    assert 'ValueError' in out.stderr and '--weights-source' in out.stderr


@pytest.mark.needs_satgen
@pytest.mark.parametrize('name', FERMI_MODULES)
def test_fermi_stray_token_rejected(name, tmp_path):
    # idx 9999: a build that does not reject the token fails later with an
    # IndexError, never reaching data, so no run can write anywhere.
    out = _run(name, ['9999', 'stray', '--weights-source', 'published'], tmp_path)
    assert out.returncode != 0
    assert 'ValueError' in out.stderr and 'unrecognized' in out.stderr, out.stderr[-800:]


@pytest.mark.needs_satgen
@pytest.mark.parametrize('name', FERMI_MODULES)
def test_fermi_stray_token_after_valid_flag_rejected(name, tmp_path):
    out = _run(name, ['9999', '--weights-source', 'published', 'stray'], tmp_path)
    assert out.returncode != 0
    assert 'ValueError' in out.stderr and 'unrecognized' in out.stderr, out.stderr[-800:]


# ----------------------------------------- seeded refused without a redirect

SEEDED_ARGS = {
    'save_contours': ['9999', 'Diemer', 'z0'],
    'fermi_reweighting': ['9999', '--legacy-version', 'update'],
    'fermi_expected_limits': ['9999'],
}


@pytest.mark.needs_satgen
@pytest.mark.parametrize('name', list(SEEDED_ARGS))
def test_seeded_refused_without_redirected_results_dir(name):
    # idx 9999 and no data read: if the refusal were missing the run would die
    # on the dwarf lookup instead, and could not write into the real results/.
    out = _run(name, SEEDED_ARGS[name] + ['--weights-source', 'seeded'], None)
    assert out.returncode != 0
    assert 'redirected SDI_RESULTS_DIR' in out.stderr, out.stderr[-800:]
    assert 'published-draw products' in out.stderr


@pytest.mark.needs_satgen
@pytest.mark.parametrize('name', list(SEEDED_ARGS))
def test_seeded_refused_when_redirect_is_the_default_dir(name):
    out = _run(name, SEEDED_ARGS[name] + ['--weights-source', 'seeded'], REPO / 'results')
    assert out.returncode != 0
    assert 'redirected SDI_RESULTS_DIR' in out.stderr, out.stderr[-800:]


@pytest.mark.needs_satgen
def test_compute_quantiles_seeded_not_refused_without_redirect():
    # Its seeded and published outputs go to separate trees: no refusal. idx
    # 9999 makes it stop at the dwarf lookup, before any write.
    out = _run('compute_quantiles', ['9999', 'Diemer'], None)
    assert out.returncode != 0
    assert 'redirected SDI_RESULTS_DIR' not in out.stderr
    assert 'IndexError' in out.stderr


@pytest.mark.needs_satgen
def test_save_contours_default_reads_published_weights(tmp_path):
    out = _run('save_contours', ['0', 'no_such_version', 'z0'], tmp_path)
    assert out.returncode != 0
    assert 'halo_weights_published/no_such_version' in out.stderr, out.stderr[-1500:]


@pytest.mark.parametrize('name', list(SEEDED_ARGS))
def test_seeded_refusal_present_in_source(name):
    src = _src(name)
    assert "weights_source == 'seeded' and config.RESULTS_DIR_IS_DEFAULT" in src
    assert 'redirected SDI_RESULTS_DIR' in src


def test_results_dir_is_default_flag(tmp_path):
    code = ('import sys; sys.path.insert(0, %r); import config; '
            'print(config.RESULTS_DIR_IS_DEFAULT)' % str(PYTHON_DIR))
    for env_dir, expect in [(None, 'True'), (str(REPO / 'results'), 'True'),
                            (str(tmp_path), 'False')]:
        env = dict(os.environ)
        env.pop('SDI_RESULTS_DIR', None)
        if env_dir:
            env['SDI_RESULTS_DIR'] = env_dir
        out = subprocess.run([sys.executable, '-c', code], env=env, cwd=REPO,
                             capture_output=True, text=True)
        assert out.stdout.strip() == expect, out.stderr


def test_results_dir_is_default_through_a_symlinked_results(tmp_path):
    # Unset SDI_RESULTS_DIR with <repo>/results a symlink must still count as
    # the default tree; comparing an unresolved path to a resolved one did not.
    import shutil
    fake = tmp_path / 'repo'
    (fake / 'python').mkdir(parents=True)
    shutil.copy(PYTHON_DIR / 'config.py', fake / 'python' / 'config.py')
    real = tmp_path / 'real_results'
    real.mkdir()
    (fake / 'results').symlink_to(real)
    code = ('import sys; sys.path.insert(0, %r); import config; '
            'print(config.RESULTS_DIR_IS_DEFAULT)' % str(fake / 'python'))
    env = dict(os.environ)
    env.pop('SDI_RESULTS_DIR', None)
    out = subprocess.run([sys.executable, '-c', code], env=env, cwd=fake,
                         capture_output=True, text=True)
    assert out.stdout.strip() == 'True', out.stderr[-1500:]


# ----------------------- compute_quantiles: migrated references are not ours

@pytest.mark.needs_satgen
def test_compute_quantiles_refuses_to_overwrite_migrated_reference(tmp_path):
    obs = pytest.importorskip('Jdata', reason='needs the J_calc environment')
    dwarf = obs.dwarf_names[0]
    import provenance
    target = tmp_path / 'paper_quantiles' / 'galactocentric_published' / 'Diemer' / f'{dwarf}.npz'
    target.parent.mkdir(parents=True)
    np.savez(target, x=np.arange(3))
    sidecar = target.with_name(target.name + provenance.SIDECAR_SUFFIX)
    sidecar.write_text('{"migrated_from": "SatGen_Dwarf/results/x.npz", "sim_version": "Diemer"}')
    before = target.read_bytes()
    out = _run('compute_quantiles', ['0', 'Diemer', '--weights-source', 'published'], tmp_path)
    assert out.returncode != 0
    assert 'migrated reference' in out.stderr and 'SDI_RESULTS_DIR' in out.stderr, out.stderr[-1500:]
    assert target.read_bytes() == before


@pytest.mark.needs_satgen
def test_compute_quantiles_refuses_to_overwrite_unstamped_output(tmp_path):
    obs = pytest.importorskip('Jdata', reason='needs the J_calc environment')
    dwarf = obs.dwarf_names[0]
    target = tmp_path / 'paper_quantiles' / 'galactocentric_published' / 'Diemer' / f'{dwarf}.npz'
    target.parent.mkdir(parents=True)
    np.savez(target, x=np.arange(3))
    before = target.read_bytes()
    out = _run('compute_quantiles', ['0', 'Diemer', '--weights-source', 'published'], tmp_path)
    assert out.returncode != 0
    assert 'no provenance record' in out.stderr, out.stderr[-1500:]
    assert target.read_bytes() == before


# ------------------------------------------------ mstar_evolution mix guard

def _quantiles_product(tmp_path, name, mstar_evolution):
    import provenance
    record = provenance.stamp('python/compute_quantiles.py', version='Diemer',
                              argv=['0', 'Diemer'], mstar_evolution=mstar_evolution)
    return provenance.savez(tmp_path / name, record, x=np.arange(3))


def test_pure_published_set_passes_the_guard(tmp_path):
    import provenance
    paths = [_quantiles_product(tmp_path, f'p{i}.npz', 'published_unseeded')
             for i in range(2)]
    assert provenance.assert_single_version(paths) == 'Diemer'


def test_published_seeded_mix_raises_naming_the_weights_source(tmp_path):
    import provenance
    paths = [_quantiles_product(tmp_path, 'p.npz', 'published_unseeded'),
             _quantiles_product(tmp_path, 's.npz', 'per_halo_tidal')]
    with pytest.raises(ValueError) as exc:
        provenance.assert_single_version(paths)
    msg = str(exc.value)
    assert 'published_unseeded' in msg and '--weights-source' in msg
    assert 'Regenerate the whole tree' not in msg


def test_published_migrated_none_mix_still_raises(tmp_path):
    import provenance
    paths = [_quantiles_product(tmp_path, 'p.npz', 'published_unseeded'),
             _quantiles_product(tmp_path, 'n.npz', None)]
    with pytest.raises(ValueError, match='mstar_evolution'):
        provenance.assert_single_version(paths)


def test_compute_quantiles_stamps_published_unseeded():
    src = _src('compute_quantiles')
    import provenance
    assert provenance.PUBLISHED_UNSEEDED == 'published_unseeded'
    assert "mstar_evolution = 'published_unseeded'" in src


# ------------------------------- routing: weights path named (nonexistent version)

@pytest.mark.needs_satgen
@pytest.mark.parametrize('name,args', [
    ('compute_quantiles', ['0', 'no_such_version']),
    ('save_contours', ['0', 'no_such_version', 'z0']),
])
@pytest.mark.parametrize('source,tree', [('published', 'halo_weights_published'),
                                         ('seeded', 'halo_weights')])
def test_weights_dir_routing(name, args, source, tree, tmp_path):
    out = _run(name, args + ['--weights-source', source], tmp_path)
    assert out.returncode != 0
    assert 'FileNotFoundError' in out.stderr
    expected = f'{tree}/no_such_version'
    assert expected in out.stderr, out.stderr[-1500:]
    other = 'halo_weights/no_such_version' if source == 'published' \
        else 'halo_weights_published/no_such_version'
    assert other not in out.stderr


# ------------------------------------------------------------ --rundata: Fermi

def _segue1_idx():
    obs = pytest.importorskip('Jdata', reason='needs the J_calc environment')
    return str(list(obs.dwarf_names).index('Segue 1'))


@pytest.mark.needs_data
@pytest.mark.needs_satgen
@pytest.mark.parametrize('source,tree', [('published', 'galactocentric_published'),
                                         ('seeded', 'galactocentric')])
@pytest.mark.parametrize('script,extra', [
    ('fermi_reweighting', ['--legacy-version', 'update', '--variants', 'mhalf']),
    ('fermi_expected_limits', ['--variants', 'mhalf']),
])
def test_fermi_quantiles_routing_error_names_tree(script, extra, source, tree, tmp_path):
    out = _run(script, [_segue1_idx(), *extra, '--weights-source', source], tmp_path)
    assert out.returncode != 0
    assert f'{tree}/Diemer/Segue 1.npz' in out.stderr, out.stderr[-1500:]


@pytest.mark.needs_data
@pytest.mark.needs_satgen
def test_fermi_default_is_published(tmp_path):
    out = _run('fermi_reweighting', [_segue1_idx(), '--legacy-version', 'update',
                                     '--variants', 'mhalf'], tmp_path)
    assert out.returncode != 0
    assert 'galactocentric_published/Diemer/Segue 1.npz' in out.stderr


@pytest.mark.needs_data
@pytest.mark.needs_satgen
@pytest.mark.parametrize('source,tree', [('published', 'galactocentric_published'),
                                         ('seeded', 'galactocentric')])
def test_mcdaniel_has_no_tree_and_no_fallback(source, tree, tmp_path):
    out = _run('fermi_reweighting',
               [_segue1_idx(), '--legacy-version', 'update',
                '--variants', 'mhalf_mcdaniel', '--weights-source', source], tmp_path)
    assert out.returncode != 0
    assert 'FileNotFoundError' in out.stderr
    assert f'{tree}/mcdaniel' in out.stderr, out.stderr[-1500:]


# ---------------------------------------------------- --rundata: contour parity

PARITY_PRODUCTS = [('galactocentric/z0/Diemer', 'Segue 1_mhalf'),
                   ('galactocentric/z0/Diemer', 'Segue 1_F18'),
                   ('galactocentric/z0/Diemer', 'Segue 1_K24'),
                   ('galactocentric/z0/Diemer', 'Segue 1_unweighted'),
                   ('rho150_mpeak/Diemer', 'Segue 1_mhalf'),
                   ('rho150_mpeak/Diemer', 'Segue 1_F18'),
                   ('rho150_mpeak/Diemer', 'Segue 1_K24'),
                   ('rho150_mpeak/Diemer', 'Segue 1_unweighted')]


def _arrays(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files if k != '_provenance'}


@pytest.fixture(scope='module')
def segue1_contour_runs(tmp_path_factory):
    obs = pytest.importorskip('Jdata', reason='needs the J_calc environment')
    idx = str(list(obs.dwarf_names).index('Segue 1'))
    pub = tmp_path_factory.mktemp('contours_published')
    seeded = tmp_path_factory.mktemp('contours_seeded')
    # The baselines are read from the real results/ in this (parent) process;
    # the children write only under the tmp dirs.
    assert config.RESULTS_DIR not in (pub.resolve(), seeded.resolve())
    assert 'SDI_RESULTS_DIR' not in os.environ or \
        Path(os.environ['SDI_RESULTS_DIR']).resolve() not in (pub.resolve(), seeded.resolve())
    runs = {}
    for label, d, extra in [('published', pub, ['--weights-source', 'published']),
                            ('seeded', seeded, ['--weights-source', 'seeded'])]:
        out = _run('save_contours', [idx, 'Diemer', 'z0', *extra], d, timeout=3600)
        assert out.returncode == 0, out.stderr[-3000:]
        runs[label] = d
    return runs


@pytest.mark.needs_data
@pytest.mark.needs_satgen
@pytest.mark.parametrize('subdir,stem', PARITY_PRODUCTS)
def test_published_contours_reproduce_stored_arrays(segue1_contour_runs, subdir, stem):
    base = config.PAPER_CONTOURS_DIR / subdir / f'{stem}.npz'
    new = segue1_contour_runs['published'] / 'paper_contours' / subdir / f'{stem}.npz'
    a, b = _arrays(base), _arrays(new)
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k]), f'{subdir}/{stem}:{k} differs'


@pytest.mark.needs_data
@pytest.mark.needs_satgen
def test_seeded_run_differs_from_stored_F18(segue1_contour_runs):
    """Negative control: the seeded weights are a different SHMR realization,
    so the F18 histogram must differ -- otherwise the parity test above could
    not tell the two sources apart."""
    base = _arrays(config.PAPER_CONTOURS_DIR / 'galactocentric/z0/Diemer/Segue 1_F18.npz')
    new = _arrays(segue1_contour_runs['seeded'] / 'paper_contours'
                  / 'galactocentric/z0/Diemer/Segue 1_F18.npz')
    assert not np.array_equal(base['hist_hist'], new['hist_hist'])


# ------------------------------- --rundata: compute_quantiles published parity

def _link(results_dir, *names):
    for n in names:
        (results_dir / n).symlink_to(config.REPO_ROOT / 'results' / n)


@pytest.mark.needs_data
@pytest.mark.needs_satgen
def test_compute_quantiles_published_reproduces_stored_segue1(tmp_path):
    obs = pytest.importorskip('Jdata', reason='needs the J_calc environment')
    idx = str(list(obs.dwarf_names).index('Segue 1'))
    _link(tmp_path, 'paper_Js')
    out = _run('compute_quantiles', [idx, 'Diemer', '--weights-source', 'published'],
               tmp_path, timeout=3600)
    assert out.returncode == 0, out.stderr[-3000:]
    new = tmp_path / 'paper_quantiles' / 'galactocentric_published' / 'Diemer' / 'Segue 1.npz'
    assert new.exists()
    # output only in the published tree; the seeded tree is never created
    assert not (tmp_path / 'paper_quantiles' / 'galactocentric').exists()
    assert sorted(p.name for p in (tmp_path / 'paper_quantiles').iterdir()) == \
        ['galactocentric_published']
    base = config.PAPER_QUANTILES_DIR / 'galactocentric_published' / 'Diemer' / 'Segue 1.npz'
    a, b = _arrays(base), _arrays(new)
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k]), f'{k} differs'
    import provenance
    rec = provenance.read(new)
    assert rec['weights_source'] == 'published'
    assert rec['mstar_evolution'] == 'published_unseeded'


# ------------------------------------ --rundata: Fermi published real read

@pytest.mark.needs_data
@pytest.mark.needs_satgen
def test_fermi_published_reproduces_stored_update_products(tmp_path):
    obs = pytest.importorskip('Jdata', reason='needs the J_calc environment')
    idx = str(list(obs.dwarf_names).index('Draco'))
    _link(tmp_path, 'paper_quantiles')
    out = _run('fermi_reweighting',
               [idx, '--variants', 'mhalf,F18', '--legacy-version', 'update',
                '--weights-source', 'published'], tmp_path, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    for variant in ('mhalf', 'F18'):
        base = _arrays(config.FERMI_REWEIGHTING_UPDATE_DIR / 'Draco' / f'{variant}.npz')
        new = _arrays(tmp_path / 'fermi_reweighting_update' / 'Draco' / f'{variant}.npz')
        assert base.keys() == new.keys()
        assert 'TS_array' in base
        for k in base:
            assert np.array_equal(base[k], new[k]), f'{variant}:{k} differs'


# ------------------------------- --rundata: mass_floor_7 prior-contour parity

MF7_PRIOR_PRODUCTS = ['galactocentric/z0/mass_floor_7', 'rho150_mpeak/mass_floor_7']


@pytest.fixture(scope='module')
def draco_mf7_contour_run(tmp_path_factory):
    obs = pytest.importorskip('Jdata', reason='needs the J_calc environment')
    idx = str(list(obs.dwarf_names).index('Draco'))
    d = tmp_path_factory.mktemp('contours_mf7')
    assert config.RESULTS_DIR != d.resolve()
    out = _run('save_contours', [idx, 'mass_floor_7', 'z0'], d, timeout=3600)
    assert out.returncode == 0, out.stderr[-3000:]
    return d


@pytest.mark.needs_data
@pytest.mark.needs_satgen
@pytest.mark.parametrize('subdir', MF7_PRIOR_PRODUCTS)
def test_mass_floor_7_prior_contour_uses_lmc_host_selection(draco_mf7_contour_run, subdir):
    """The published mass_floor_7 '_unweighted' prior is drawn from hosts with
    an LMC analog (the analog removed). Without that selection the Draco
    histograms differ, so this fails if the selection is dropped."""
    base = _arrays(config.PAPER_CONTOURS_DIR / subdir / 'Draco_unweighted.npz')
    new_path = draco_mf7_contour_run / 'paper_contours' / subdir / 'Draco_unweighted.npz'
    new = _arrays(new_path)
    assert base.keys() == new.keys()
    for k in base:
        assert np.array_equal(base[k], new[k]), f'{subdir}/Draco_unweighted:{k} differs'
    assert provenance.read(new_path)['prior_lmc_selection'] is True

