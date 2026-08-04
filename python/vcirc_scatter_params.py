"""Constants shared between vcirc_scatter_300pc.py and its concat step.

vcirc_scatter_300pc.py executes its whole pipeline at import (it reads
sys.argv and real catalog data), so concat_vcirc_scatter_300pc.py cannot
import it just to read SHMR_NAMES. Both scripts import this module instead,
so the SHMR list and the two MC parameters exist as one literal rather than
two that can drift apart.
"""

N_MC         = 100
NARROW_SIGMA = 0.16
R_TARGET_KPC = 0.3
SEED         = 0

SHMR_NAMES = [
    'Behroozi13',
    'Moster13',
    'RodriguezPuebla17',
    'Fattahi18',
    'Moster18',
    'Behroozi19',
    'Munshi21',
    'Danieli23_const',
    'Danieli23_grow',
    'Kim24',
]
