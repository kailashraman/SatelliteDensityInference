"""Standing anti-drift gate: every figure/table drafts_temp/*.tex references
must exist under plots/, and every file under plots/ must be referenced.

This is the figure-naming contract from python/figure_registry.py. The xfail
ratchet is keyed on the REGISTRY declaration, not on filesystem presence: as
long as `figure_registry.REGISTRY[name]` (or `TABLE_REGISTRY[name]`) is
`None`, the entry is dynamically xfailed -- a *dynamic* xfail, which is
always non-strict regardless of the `xfail_strict` ini setting, so a
basename that starts passing before its producer is registered does not
turn the suite red. The moment a step declares a producer by setting the
registry entry to a module name, the same parametrized case flips to a hard
assertion that the file exists under plots/ (exactly once) -- there is no
separate bookkeeping list to update, and no way for a declared producer to
xfail silently.

The reverse direction has no analog in the registry, so it is checked
directly against the filesystem: every `*.pdf` under plots/ must have its
basename in FIGURES, and every `*.tex` under plots/tables/ must have its
basename (without extension) in TABLES. An orphaned or misnamed output --
one a script wrote but the drafts never asked for, or the wrong basename --
fails this. `.ipynb_checkpoints` directories are excluded from every glob;
plots/rmax_vmax/ has one, and a stale checkpoint sharing a real basename
would otherwise satisfy the forward gate for free.

Do not weaken this by skipping the file or asserting a subset.
"""

import pytest

import config
from figure_registry import FIGURES, REGISTRY, TABLE_REGISTRY, TABLES


def _glob_excluding_checkpoints(root, pattern):
    return [p for p in root.glob(pattern) if '.ipynb_checkpoints' not in p.parts]


@pytest.mark.parametrize('name', FIGURES)
def test_figure_exists_under_plots(name):
    if REGISTRY[name] is None:
        pytest.xfail(f'{name} not produced yet (steps 10-22)')
    matches = _glob_excluding_checkpoints(config.PLOTS_DIR, f'**/{name}')
    assert matches, f'{name} declared in REGISTRY but not found under {config.PLOTS_DIR}'
    assert len(matches) == 1, f'{name} found in more than one place: {matches}'


@pytest.mark.parametrize('name', TABLES)
def test_table_exists_under_plots_tables(name):
    if TABLE_REGISTRY[name] is None:
        pytest.xfail(f'{name}.tex not produced yet (steps 10-22)')
    path = config.PLOTS_DIR / 'tables' / f'{name}.tex'
    assert path.is_file(), f'{name}.tex declared in TABLE_REGISTRY but not found at {path}'


def test_no_orphan_pdfs_under_plots():
    orphans = [p for p in _glob_excluding_checkpoints(config.PLOTS_DIR, '**/*.pdf')
               if p.name not in FIGURES]
    assert not orphans, f'PDFs under plots/ with no matching FIGURES entry: {orphans}'


def test_no_orphan_tables_under_plots_tables():
    tables_dir = config.PLOTS_DIR / 'tables'
    orphans = [p for p in _glob_excluding_checkpoints(tables_dir, '*.tex')
               if p.stem not in TABLES]
    assert not orphans, f'.tex files under plots/tables/ with no matching TABLES entry: {orphans}'
