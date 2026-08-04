# Per-halo mass/velocity profiles on a fixed radial grid.
#
# Migrated from SatGen_Dwarf/python/massProfile.py. Path/import/provenance
# edits only -- the numerics are unchanged: r_grid, the per-halo loop
# computing mpeakProfile via halo.M(r) BEFORE update_mass(), then massProfile
# and vcircProfile AFTER update_mass(), and the Vcirc unit conversion are all
# character-identical to upstream. The order of those two loops relative to
# update_mass is the whole meaning of mpeak-vs-current profile -- it is not
# reordered here.
#
# CLI argument order is INVERTED relative to compute_weights.py: here
# argv[1]=version, argv[2]=task_idx (upstream's own order). Do not transpose
# them when writing a launcher for this script.
#
# Dropped from upstream: the matplotlib/PdfPages preamble (this script never
# plots), the `sys.path.append` lines (imports are resolved by running from
# python/ or having it on PYTHONPATH, as with every other migrated script),
# and the unused `tqdm` import.
#
# The catalog is read via halo_weights.get_h5, matching the other migrated
# scripts, rather than a hardcoded path.
#
# Also dropped: upstream's `from astropy import constants as const`,
# `from astropy.coordinates import SkyCoord`, and `from astropy.constants
# import G` -- none of const/SkyCoord/G is referenced anywhere in this script
# (upstream carried them unused too); only `astropy.units` is actually needed
# for the Vcirc conversion.
import sys

import numpy as np
from astropy import units as u

from SatGen.profiles import Green
from halo_weights import get_h5
import config
import provenance

# set up colossus cosmology to match SatGen
from SatGen import profiles as pf
from colossus.cosmology import cosmology as co

co.addCosmology('SatGen', flat=True, H0=pf.cfg.h*100, Om0=pf.cfg.Om,
                Ob0=pf.cfg.Ob, sigma8=pf.cfg.s8, ns=pf.cfg.ns)
col_co = co.setCosmology('SatGen')

# ====================================================================================== #

# Halo count per output fragment; also the group-index chunk size used by
# scripts/massProfile.sh to size its --array.
CHUNK_SIZE = 50000

### Get job details
version = sys.argv[1]
task_idx = int(sys.argv[2])
group = task_idx

save_dir = config.PAPER_MASSPROFILE_DIR / version / 'massProfile'
filename = f'massProfile-{group}'

# ====================================================================================== #

### load SatGen run info and make halos

provenance_inputs = [config.h5_path(version)]
provenance.assert_single_version(provenance_inputs, expected=version)

with get_h5(version) as f:
    Greens = f['Green_params'][()]
    Mvir = f['virial_mass'][()]

n_halos = len(Mvir)

####

idcs = np.arange(len(Mvir))[group * CHUNK_SIZE: (group + 1) * CHUNK_SIZE]
Greens = Greens[idcs]
Mvir = Mvir[idcs]

####
r_grid = np.logspace(-3, 1, 101)
massProfile = np.zeros(shape=(len(Greens), len(r_grid)))
mpeakProfile = np.zeros_like(massProfile)
vcircProfile = np.zeros_like(massProfile)
halo_instances = [Green(*Greens[i]) for i in range(len(Greens))]
for i in range(len(halo_instances)):
    for j, r in enumerate(r_grid):
        mpeakProfile[i, j] = halo_instances[i].M(r)
    halo_instances[i].update_mass(Mvir[i])
    for j, r in enumerate(r_grid):
        massProfile[i, j] = halo_instances[i].M(r)
        vcircProfile[i, j] = (halo_instances[i].Vcirc(r) * u.kpc / u.Gyr).to(u.km / u.s).value

# Save results to file.
# start is clamped as well as stop: a group index past the end of the catalog
# produces an empty fragment, and an unclamped start would record a slice with
# start > stop (e.g. [3150000, 2432159] for group 63 of Diemer). Nothing read
# that field until concat_massProfile.py began checking the fragments tile the
# catalog, at which point a nonsense slice becomes a real failure.
start = min(group * CHUNK_SIZE, n_halos)
stop = min((group + 1) * CHUNK_SIZE, n_halos)
record = provenance.stamp('python/massProfile.py', version=version, argv=sys.argv[1:],
                          inputs=provenance_inputs, group=group,
                          halo_slice=[int(start), int(stop)])
provenance.savez(save_dir / filename, record, massProfile=massProfile,
                 mpeakProfile=mpeakProfile, vcircProfile=vcircProfile, r_grid=r_grid)
