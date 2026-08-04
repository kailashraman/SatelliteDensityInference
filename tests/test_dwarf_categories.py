"""dwarf_categories reproduces PaperPlots.ipynb cell 4's category split.

Needs the J_calc environment: dwarf_categories imports Jdata, which needs
astropy (see tests/test_jdata_module.py for the same pattern).
"""

import numpy as np
import pytest

dwarf_categories = pytest.importorskip('dwarf_categories', reason='needs the J_calc environment')

pytestmark = pytest.mark.needs_satgen


def test_idcs_has_39_entries():
    assert len(dwarf_categories.idcs) == 39
    assert dwarf_categories.n_dwarfs == 39


def test_idcs_is_a_permutation_of_43_minus_large_dwarfs():
    expected = np.setdiff1d(np.arange(43), dwarf_categories.large_idcs)
    assert sorted(dwarf_categories.idcs) == sorted(expected)
    assert len(set(dwarf_categories.idcs)) == len(dwarf_categories.idcs)


def test_categories_are_disjoint_and_cover_all_43():
    problem = set(dwarf_categories.problem_idcs)
    ultrafaint = set(dwarf_categories.ultrafaint_idcs)
    classical = set(dwarf_categories.classical_idcs)
    large = set(dwarf_categories.large_idcs)
    groups = [problem, ultrafaint, classical, large]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            assert groups[i].isdisjoint(groups[j]), 'category index sets overlap'
    assert problem | ultrafaint | classical | large == set(range(43))


def test_category_membership_matches_the_named_dwarf_lists():
    """Disjointness and cover-43 alone don't pin the split BETWEEN the
    named blocks: if a catalog rename dropped a name from
    `problem_ultrafaints` (so it fell through to `ultrafaint_idcs` instead),
    `problem_idcs` would shrink and `ultrafaint_idcs` would grow, while
    disjointness, cover-43, and len(idcs)==39 all still pass. Pin each named
    block's length against the length of the name list it was built from, so
    that silent reshuffling is caught."""
    assert len(dwarf_categories.classical_idcs) == len(dwarf_categories.classicals)
    assert len(dwarf_categories.problem_idcs) == len(dwarf_categories.problem_ultrafaints)
    assert len(dwarf_categories.large_idcs) == len(dwarf_categories.large_dwarf)
    assert len(dwarf_categories.classical_idcs) == 10
    assert len(dwarf_categories.problem_idcs) == 8
    assert len(dwarf_categories.large_idcs) == 4
    assert len(dwarf_categories.ultrafaint_idcs) == 21
