"""Regression test: Fermi-stage Monte Carlo seeds must be deterministic.

fermi_reweighting.py and fermi_expected_limits.py used to seed from
abs(hash(name)) % 2**32. Python's str hash is salted per process, so reruns
drew different F18/K24 J-factors. Both now go through
fermi_funcs.mc_seed(name, stage) (zlib.crc32 of '<stage>:<name>'). The modules
run on import (CLI scripts needing data), so the call sites are checked by
reading their source (AST); those tests do not import fermi_funcs and so
cannot skip with it.

These fail against the pre-fix code (no mc_seed; hash( in the seeding lines).
"""

import ast
import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

import config

PYTHON_DIR = Path(config.REPO_ROOT) / 'python'

STAGES = {'fermi_reweighting.py': 'fermi_reweighting',
          'fermi_expected_limits.py': 'fermi_expected_limits'}

_CODE = ('import sys; sys.path.insert(0, %r); '
         'from fermi_funcs import mc_seed; '
         'print(mc_seed("Ursa Major II", "fermi_reweighting"))')


def _seed_under_hashseed(hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run([sys.executable, '-c', _CODE % str(PYTHON_DIR)],
                         env=env, capture_output=True, text=True, check=True)
    return int(out.stdout.strip().splitlines()[-1])


def test_mc_seed_independent_of_pythonhashseed():
    pytest.importorskip('fermi_funcs', reason='needs the J_calc environment')
    assert _seed_under_hashseed(1) == _seed_under_hashseed(2)


def test_mc_seed_equals_explicit_crc32():
    pytest.importorskip('fermi_funcs', reason='needs the J_calc environment')
    from fermi_funcs import mc_seed
    expected = zlib.crc32(b'fermi_reweighting:Segue 1') & 0xffffffff
    assert mc_seed('Segue 1', 'fermi_reweighting') == expected
    assert 0 <= expected < 2**32


def test_mc_seed_differs_between_stages():
    pytest.importorskip('fermi_funcs', reason='needs the J_calc environment')
    from fermi_funcs import mc_seed
    assert (mc_seed('Segue 1', 'fermi_reweighting')
            != mc_seed('Segue 1', 'fermi_expected_limits'))


@pytest.mark.parametrize('module', list(STAGES))
def test_stage_seeds_through_mc_seed_not_hash(module):
    tree = ast.parse((PYTHON_DIR / module).read_text())
    rng_args = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            assert not (isinstance(f, ast.Name) and f.id == 'hash'), \
                f'{module} calls builtin hash()'
            if isinstance(f, ast.Attribute) and f.attr == 'default_rng':
                rng_args.append(ast.unparse(node.args[0]))
    want = f"mc_seed(name, '{STAGES[module]}')"
    assert rng_args and all(a == want for a in rng_args)


@pytest.mark.parametrize('module,output', [
    ('fermi_reweighting.py', None),
    ('fermi_expected_limits.py', 'F18_MC_blank_fields')])
def test_mc_outputs_carry_stochastic_provenance(module, output):
    """Source check (AST), not a data run: every savez of an *_MC product
    passes a record that sets stochastic_arrays and stochastic_caveat."""
    tree = ast.parse((PYTHON_DIR / module).read_text())

    def has_stochastic(node):
        return any(isinstance(k, ast.keyword) and k.arg == 'stochastic_arrays'
                   for n in ast.walk(node) if isinstance(n, ast.Call)
                   for k in n.keywords) and any(
            k.arg == 'stochastic_caveat'
            for n in ast.walk(node) if isinstance(n, ast.Call)
            for k in n.keywords)

    # Names assigned a record carrying the stochastic fields.
    stochastic_names = {t.id for a in ast.walk(tree)
                        if isinstance(a, ast.Assign) and has_stochastic(a.value)
                        for t in a.targets if isinstance(t, ast.Name)}
    checked = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'savez' and len(node.args) >= 2):
            continue
        target = ast.unparse(node.args[0])
        if '_MC' not in target:
            continue
        if output is not None and output not in target:
            continue
        checked += 1
        rec = node.args[1]
        assert (has_stochastic(rec)
                or (isinstance(rec, ast.Name) and rec.id in stochastic_names)), \
            f'{module}: {target} written without stochastic provenance'
    assert checked >= 1
