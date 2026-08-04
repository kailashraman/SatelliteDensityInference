"""Category split and canonical x-axis ordering for the panel figures.

Reproduces ../SatGen_Dwarf/jupyter/PaperPlots.ipynb cell 4: four disjoint
index sets over the 43-dwarf catalog (`classicals`, `problem_ultrafaints`,
the implicit remaining `ultrafaint_idcs`, and `large_dwarf`, the four hosts
every panel figure excludes -- see the 39-vs-43 note in
tests/test_dwarf_sample.py) and `idcs`, the canonical x-axis ordering every
panel figure uses: problem-ultrafaints, then ultrafaints, then classicals,
each block sorted by ascending dwarf `rhalf`. `n_dwarfs` is `len(idcs)` and
comes out to 39.

The notebook cell also carries a commented-out alternative ordering, by
ascending `logMstar` instead of `rhalf`, directly above the active one. It
is kept below as a comment for the same reason it is commented out upstream:
switching a panel figure's x-axis to read by stellar mass instead of size is
a scientific choice for whoever migrates that figure, not a cleanup, and
this module must not make it silently.
"""

import numpy as np

import Jdata as obs

dwarf_names = obs.dwarf_names

logMstar = np.round([obs.Dwarf(dwarf).logMstar for dwarf in dwarf_names], decimals=2)
dwarf_distance = np.round(
    [obs.Dwarf(dwarf).distance.to_value('kpc') for dwarf in dwarf_names], decimals=2)
dwarf_rhalf = np.round(
    [obs.Dwarf(dwarf).rhalf.to_value('pc') for dwarf in dwarf_names], decimals=2)

classicals = ['Antlia II', 'Canes Venatici I', 'Carina', 'Crater II', 'Draco', 'Leo I',
              'Leo II', 'Sculptor', 'Sextans', 'Ursa Minor']
problem_ultrafaints = ['Aquarius II', 'Carina III', 'Grus I', 'Horologium I', 'Leo VI',
                        'Pisces II', 'Tucana II', 'Tucana V']
large_dwarf = np.array(['Fornax', 'SMC', 'LMC', 'Sagittarius'])
large_idcs = np.array([i for i, name in enumerate(dwarf_names) if name in large_dwarf])
classical_idcs = np.array([i for i, name in enumerate(dwarf_names) if name in classicals])
problem_idcs = np.array([i for i, name in enumerate(dwarf_names) if name in problem_ultrafaints])
ultrafaint_idcs = np.setdiff1d(
    np.setdiff1d(np.setdiff1d(np.arange(len(dwarf_names)), large_idcs), classical_idcs),
    problem_idcs)

# Alternative ordering by stellar mass instead of size -- NOT active. See the
# module docstring: switching to this is a scientific choice, not a cleanup.
# idcs = np.concatenate((problem_idcs[np.argsort(logMstar[problem_idcs])],
#                        ultrafaint_idcs[np.argsort(logMstar[ultrafaint_idcs])],
#                        classical_idcs[np.argsort(logMstar[classical_idcs])]))
idcs = np.concatenate((problem_idcs[np.argsort(dwarf_rhalf[problem_idcs])],
                        ultrafaint_idcs[np.argsort(dwarf_rhalf[ultrafaint_idcs])],
                        classical_idcs[np.argsort(dwarf_rhalf[classical_idcs])]))
n_dwarfs = len(idcs)
