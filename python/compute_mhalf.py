"""Mass enclosed within dwarfs' half-light radii, plus four catalog-global
halo properties (rho30, M30, rho150, Mvir200).

Migrated from SatGen_Dwarf/python/compute_mhalf.py, split into two modes:

    python compute_mhalf.py <dsph_idx> <version>   # per-dwarf: logMhalf, logMdyn_errani
    python compute_mhalf.py --globals <version>     # catalog-globals: rho30, M30, rho150, Mvir200

Upstream computed BOTH on every array task (one task per dwarf, 43 total),
writing the same four global files from every task -- a race, since every task
overwrites `<version>_{rho30,M30,rho150,Mvir200}.npz` with its own (statistically
identical, but not synchronized) recomputation. The four global arrays depend
only on `Green_params` and `virial_mass` -- confirmed by reading the source:
rho30/M30/rho150 are evaluated at fixed radii (0.03, 0.15 kpc) that do not
involve any dwarf property, and Mvir200 is a mass-definition change of
`Green_params` alone. Only `logMhalf`/`logMdyn_errani` use the dwarf's
`rhalf`. `--globals` computes them once instead of once per task, and the
per-dwarf mode no longer touches them at all.

This also removes the per-dwarf-task cost of `mass_defs.changeMassDefinition`,
which upstream ran on ~10^6 halos inside every one of the 43 per-dwarf tasks
(~4.3x10^7 calls total, ~43x the real work). It now runs on ~10^6 halos exactly
once, in `--globals` mode only.

Per-dwarf output: data/additional/mhalf/<version>/<dwarf>.npz
Globals output:   data/additional/<version>_{rho30,M30,rho150,Mvir200}.npz
"""
import sys
import numpy as np

from SatGen.profiles import Green, Dekel, NFW
from SatGen import profiles as pf
from colossus.cosmology import cosmology as co
from colossus.halo import profile_nfw, concentration, mass_defs

co.addCosmology('SatGen', flat=True, H0=pf.cfg.h*100, Om0=pf.cfg.Om,
                Ob0=pf.cfg.Ob, sigma8=pf.cfg.s8, ns=pf.cfg.ns)
col_co = co.setCosmology('SatGen')

import Jdata as obs
from halo_weights import *
import config
import provenance

# ====================================================================================== #

### Get job details

if sys.argv[1] == '--globals':
    mode = 'globals'
    version = sys.argv[2]
else:
    mode = 'per_dwarf'
    dsph_idx = int(sys.argv[1])
    version = sys.argv[2]

dwarf_names = obs.dwarf_names

# ====================================================================================== #

if mode == 'per_dwarf':
    dwarf = dwarf_names[dsph_idx]
    print(dwarf)

    with get_h5(version) as f:
        Greens = f['Green_params'][()]
        virial_mass = f['virial_mass'][()]
        rhalf = obs.Dwarf(dwarf).rhalf.to(u.kpc).value

        logMhalf = np.zeros(len(Greens))
        logMdyn_errani = np.zeros(len(Greens))
        for i in range(len(Greens)):
            if not i % 10000:
                print(i)
            halo_instance = Green(*Greens[i])
            halo_instance.update_mass(virial_mass[i])
            logMhalf[i] = np.log10(halo_instance.M(rhalf))
            logMdyn_errani[i] = np.log10(halo_instance.M(rhalf * 1.35))

    record = provenance.stamp('python/compute_mhalf.py', version=version,
                              argv=sys.argv[1:], inputs=[config.h5_path(version)],
                              dwarf=dwarf)
    provenance.savez(config.MHALF_DIR / version / dwarf, record,
                     logMhalf=logMhalf, logMdyn_errani=logMdyn_errani)

else:
    with get_h5(version) as f:
        Greens = f['Green_params'][()]
        virial_mass = f['virial_mass'][()]

        M30 = np.zeros(len(Greens))
        rho30 = np.zeros(len(Greens))
        rho150 = np.zeros(len(Greens))
        Mvir200 = np.zeros(len(Greens))
        for i in range(len(Greens)):
            if not i % 10000:
                print(i)
            halo_instance = Green(*Greens[i])
            halo_instance.update_mass(virial_mass[i])
            M30[i] = halo_instance.M(0.03)
            rho150[i] = halo_instance.rho(0.15)
            rho30[i] = halo_instance.rho(0.03)
            M, c, delta, z = Greens[i]
            M200, r200, c200 = mass_defs.changeMassDefinition(M*col_co.h, c, z, 'vir', '200c')
            Mvir200[i] = M200/col_co.h

    globals_out = {'rho30': rho30, 'M30': M30, 'rho150': rho150, 'Mvir200': Mvir200}
    for name, arr in globals_out.items():
        record = provenance.stamp('python/compute_mhalf.py', version=version,
                                  argv=sys.argv[1:], inputs=[config.h5_path(version)],
                                  quantity=name)
        provenance.savez(config.ADDITIONAL_DIR / f'{version}_{name}', record,
                         **{name: arr})
