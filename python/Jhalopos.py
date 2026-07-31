"""J-factors of SatGen halos viewed at their own positions.

Produces results/paper_Js/<version>/halo_position/, the SatGen curve of
Jfactors_all.pdf (temp_part2.tex, fig:population_jfactors). Unlike Jdwarf.py,
which evaluates every halo at one dwarf's distance, this evaluates each halo at
its own heliocentric distance -- the population a survey would actually see.

Deliberately NOT a variant of Jdwarf.py. Three differences are load-bearing and
ported as-is, because the published product was made this way:

* the integrand uses the small-angle radius sqrt((L-d)^2 + (L*theta)^2); Jdwarf
  uses the exact law of cosines;
* the integral is vegas Monte Carlo; Jdwarf uses Gauss-Legendre quadrature;
* only `green_Js` is written -- no theta95, no full_Js.

Two hazards, both in docs/migration-notes.md:

1. Output is NOT bit-reproducible. vegas is stochastic and unseeded, so a rerun
   differs within Monte Carlo error. This is the only product in the migrated
   tree that cannot be verified by an exact diff.
2. The random rotation is dead code. The module draws a random z-rotation and
   builds `gc_rotated`, but the distance used comes from the unrotated `gc`, so
   the rotation has no effect and the result does not depend on the draw.
   Ported unchanged: removing it would be a behaviour change to a published
   product.

Usage:
    python Jhalopos.py <version> <group>
"""
import sys, os
import numpy as np
from scipy.stats import binned_statistic

import matplotlib.pyplot as plt
import matplotlib as mpl
colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', '#198038']

plt.rc('axes', fc='w', prop_cycle = mpl.cycler(color=colors))
plt.rc('font', size = 8, family='serif', serif='Nimbus Roman')
plt.rc('text', usetex=False)
plt.rc('mathtext', rm = 'serif', it = 'serif:italic', bf = 'serif:bold', fontset = 'custom')
plt.rc('figure', figsize = [2.7, 2.7], dpi = 500)
plt.rc('lines', lw = 1, markersize = 2)
plt.rc('legend', frameon=False)

# Dropped on migration: `from Janalysis import GreenData as Sim, Weights`.
# Janalysis targets the retired J_test tree and neither name is used here.

from SatGen.profiles import Green, Dekel, NFW
import Jdata as obs
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from util import *
import config
import provenance
from halo_weights import *

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.constants import G

# set up colossus cosmology to match SatGen
from SatGen import profiles as pf
from colossus.cosmology import cosmology as co
from colossus.halo import profile_nfw

co.addCosmology('SatGen', flat=True, H0=pf.cfg.h*100, Om0=pf.cfg.Om,
                Ob0=pf.cfg.Ob, sigma8=pf.cfg.s8, ns=pf.cfg.ns)
col_co = co.setCosmology('SatGen')

# ====================================================================================== #

### Get job details
group = int(sys.argv[2])
version = sys.argv[1]

# Layout: paper_Js/halo_position/<version>/, NOT paper_Js/<version>/halo_position/.
# The latter is what upstream writes, but it puts a non-dwarf directory beside
# the per-dwarf ones, where concat_Js would pick it up as a dwarf. Upstream's
# artifact is in a third place again -- paper_Js/halo_position/ with no version
# at all, which a second version's run would silently overwrite.
save_dir = os.path.join(str(config.PAPER_JS_DIR / 'halo_position' / version),
                        'halo_Js') + os.sep
filename = 'halo_Js-' + str(group)

# Check if save directory exists, if not, create it
if not os.path.exists(save_dir):
    print('No pre-existing path, creating directory.')
    os.makedirs(save_dir, exist_ok=True)
else:
    print('Path already exists')

# Check if filename exists, if so, end job
# if os.path.exists(os.path.join(save_dir, filename + '.npz')):
#     raise FileExistsError(os.path.join(save_dir, filename + '.npz') + ' already exists')

# ====================================================================================== #

### load SatGen run info and make halos

with get_h5(version) as f:
    Greens = f['Green_params'][()]
    Mvir = f["virial_mass"][()]
    positions = f['position'][()]

####

CHUNK = 5000                      # halos per array task
idcs = np.arange(len(Mvir))[group * CHUNK: (group + 1) * CHUNK]
Greens = Greens[idcs]
positions = positions[idcs]
Mvir = Mvir[idcs]

####

halo_instances = [Green(*Greens[i]) for i in range(len(Greens))]
for i in range(len(halo_instances)):
    halo_instances[i].update_mass(Mvir[i])
    # halo_instances[i].update_mass(Greens[i][0])

### Compute halo distances
gc = SkyCoord(x=positions[:,0]*u.kpc, y=positions[:,1]*u.kpc, z=positions[:,2]*u.kpc, frame='galactocentric')
# Generate a random angle (e.g., around z-axis)
random_angle = np.random.uniform(0, 360) * u.deg
# Create rotation matrix (rotating around z-axis)
rot_matrix = rotation_matrix(random_angle, axis='z')
# Get the Cartesian representation and apply rotation
gc_rotated = gc.cartesian.transform(rot_matrix)
# Convert back to SkyCoord in galactocentric frame
gc_rotated = SkyCoord(gc_rotated, frame='galactocentric')

d = gc.transform_to('galactic').distance.value

# ====================================================================================== #

#### Green tidal evolution

class tNFW(pf.NFW):
    def __init__(self, Green_profile: pf.Green, z: float = 0) -> None:
        '''
        given a pf.Green profile, get the pf.NFW with the same Rmax and Vmax

        NOTE: it is impossible to read the initialization redshift `z` from
              the Green profile, so this must be provided for the virial
              overdensity parameter Delta to be interpreted correctly
        '''
        # get evolved (rmax, vmax) from the Green profile
        self.Green_p = Green_profile
        unevoNFW = pf.NFW(Green_profile.Minit, Green_profile.ch,
                          Green_profile.Deltah, z=z)
        assert Green_profile.rhoh == unevoNFW.rhoh,\
               'Please provide correct initialization redshift'
        unevoR, unevoV = unevoNFW.rmax, unevoNFW.Vmax
        rmax = unevoR * tNFW.Green_tidalX(Green_profile, 'rmax')
        vmax = unevoV * tNFW.Green_tidalX(Green_profile, 'vmax')
        # move to Colossus parameters (rs, rhos) -- watch factors of h in units
        rs = rmax * u.kpc / 2.16258
        rhos = ((vmax * u.kpc/u.Gyr / (1.64 * rs))**2/const.G)
        rhos = rhos.to(u.Msun/u.kpc**3).value
        rs = rs.to(u.kpc).value
        self.Col_p = profile_nfw.NFWProfile(rhos=rhos/col_co.h**2,
                                            rs=rs*col_co.h)
        # get NFW parameters (Mh, c)
        Rh, Mh = self.Col_p.RMDelta(0, '200c')
        Rh = Rh/col_co.h
        Mh = Mh/col_co.h
        super().__init__(M=Mh, c=Rh/rs, Delta=200, z=0)

    @staticmethod
    def Green_tidalX(Green_profile: pf.Green, kind: str = 'rmax') -> float:
        '''
        given a pf.Green profile, compute the ratio of rmax/rmax(infall) or
        vmax/vmax(infall), as given by `kind`, according to the tidal tracks
        described in Green & Van den Bosch (2019)

        Implements Eq. 11-13 using parameters from Table 2 of that reference
        '''
        fb, log10fb = Green_profile.fb, Green_profile.log10fb
        ch = Green_profile.ch
        if kind == 'rmax':
            p0, p1, p2 = +1.021, +1.463, +0.099
            p3, p4 = -4.643, -0.250
            q0, q1, q2 = -0.525, -0.065, +0.083
        elif kind == 'vmax':
            p0, p1, p2 = +2.980, +0.310, -0.223
            p3, p4 = -3.308, -0.079
            q0, q1, q2 = +0.176, -0.008, +0.452
        else:
            raise ValueError('Tidal track invalid (must be "rmax" or "vmax"')
        mu = p0 + (p1 * ch**p2 * log10fb) + (p3 * ch**p4)
        eta = q0 + (q1 * ch**q2 * log10fb)
        return (2**mu * fb**eta)/(1 + fb)**mu

# ====================================================================================== #

### Calculate J-factor for a given density profile

def radius_sa(L, d, theta):
    # small angle approximation
    R = np.abs(L - d)  # Cylindrical radius in kpc
    z = L * theta  # Vertical distance in kpc
    return(np.sqrt(R**2 + z**2))

# Define profile function

def j_profile(rho, L, theta, d):
    """
    Calculate the density profile for a halo using the cylindrical coordinates R and z.
    
    Parameters:
    - idx : int : Index for the halo
    - L : float : Distance parameter in kpc
    - theta : float : Angular coordinate
    - d : float : Distance to the dwarf galaxy in kpc
    """
    r = radius_sa(L, d, theta)
    return rho(r) ** 2 * 2 * np.pi * np.sin(theta)  # Density squared

def integrator(fnc, interval):
    @vegas.lbatchintegrand
    def integrand(x):
        if len(x.shape) > 1:
            xx = []
            for i in range(x.shape[1]):
                xx.append(x[:,i])
        else:
            xx = x[:,0]
        return fnc(xx)
    integ = vegas.Integrator(interval)
    integ(integrand, nitn = 10, neval = 1e4)
    result = integ(integrand, nitn = 10, neval = 1e4)
    return result.mean

def compute_J_factor(rho, d_kpc, integ_angle):
    integrand = lambda x: j_profile(rho, x[0], x[1], d_kpc)
    result = integrator(integrand, [[0, d_kpc * 2], [0, integ_angle]])
    return (result * u.solMass ** 2 * const.c ** 4 / (u.kpc ** 5)).to(u.GeV ** 2 / u.cm ** 5)

def compute_angle(d_kpc, rs_kpc):
    return np.arctan(rs_kpc / d_kpc)

# ====================================================================================== #
    
# Compute J factors

green_Js = np.zeros(len(idcs))
# nfw_Js = np.zeros(len(idcs))
# rs_green_Js = np.zeros(len(idcs))
# rs_nfw_Js = np.zeros(len(idcs))

for j in range(len(idcs)):
    # idx = idcs[j]
    # nfw_pf = tNFW(halo_instances[idx], z = Greens[idx][-1])
    # rs = nfw_pf.rh
    # rs_angle = compute_angle(d[idx], rs)

    green_Js[j] = compute_J_factor(halo_instances[j].rho, d[j], 0.5 * np.pi / 180).value
    # nfw_Js[j] = compute_J_factor(nfw_pf.rho, d[idx], 0.5 * np.pi / 180).value

    # rs_green_Js[j] = compute_J_factor(halo_instances[idx].rho, d[idx], rs_angle).value
    # rs_nfw_Js[j] = compute_J_factor(nfw_pf.rho, d[idx], rs_angle).value

# ====================================================================================== #
    
# Save results

result_dict = {
    'green_Js': green_Js,
    # 'nfw_Js': nfw_Js,
    # 'rs_green_Js': rs_green_Js,
    # 'rs_nfw_Js': rs_nfw_Js,
}

# Save results to file
record = provenance.stamp('python/Jhalopos.py', version=version, argv=sys.argv[1:],
                          inputs=[config.h5_path(version)],
                          group=group, chunk=CHUNK,
                          halo_slice=([int(idcs[0]), int(idcs[-1]) + 1]
                                      if len(idcs) else [0, 0]),
                          integrator='vegas (stochastic, unseeded)')
provenance.savez(os.path.join(save_dir, filename), record, **result_dict)
    