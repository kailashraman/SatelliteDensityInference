import sys, os
import numpy as np
from numpy.polynomial.legendre import leggauss
import h5py
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

from SatGen.profiles import Green, Dekel, NFW
import Jdata as obs
from halo_weights import *
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from util import *
import config
import provenance

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
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
task_idx = int(sys.argv[2])
version = sys.argv[1]

GROUPS = 20                       # chunks per dwarf (43 dwarfs x 20 = 860 tasks).
dsph_idx = task_idx // GROUPS     # Bigger chunks than before: the quad J-factor
group = task_idx % GROUPS         # is ~95x faster, so per-task SatGen import (~36s)
                                  # dominates -- amortize it over more halos/task.
# dsph_idx = int(task_idx / 103)
# group = task_idx % 103
# dsph_idx = int(task_idx / 51)
# group = task_idx % 51
# dsph_idx = int(sys.argv[1])
# group = int(sys.argv[2])

# Load dwarf galaxy data
# EDIT to load the distances for the UFSC

dwarf_names = obs.dwarf_names
# The array size in scripts/Jdwarf.sh is a literal; the dwarf count comes from
# the LVDB catalog at import. If the catalog ever grows, an array sized for the
# old count silently never computes the tail dwarfs, and concat_Js only lists
# directories that already exist -- so the omission would surface much later as
# a missing file. Fail here instead.
n_tasks = len(dwarf_names) * GROUPS
if not 0 <= task_idx < n_tasks:
    raise SystemExit(
        f'task {task_idx} out of range: {len(dwarf_names)} dwarfs x '
        f'{GROUPS} groups = {n_tasks} tasks (array should be 0-{n_tasks - 1})')
name = dwarf_names[dsph_idx]
dwarf = obs.Dwarf(name)

save_dir = os.path.join(str(config.PAPER_JS_DIR / version / name), 'halo_Js') + os.sep
filename = 'halo_Js-' + str(group)

# Check if save directory exists, if not, create it
if not os.path.exists(save_dir):
    print('No pre-existing path, creating directory.')
    os.makedirs(save_dir, exist_ok=True)
else:
    print('Path already exists')

# Check if filename exists, if so, end job
if os.path.exists(os.path.join(save_dir, filename + '.npz')):
    raise FileExistsError(os.path.join(save_dir, filename + '.npz') + ' already exists')


# ====================================================================================== #

### load SatGen run info and make halos

with get_h5(version) as f:
    # Read ONLY this group's halo slice from disk, not the whole file. Reading
    # the full 155 MB arrays in every task starves the filesystem when thousands
    # of tasks run at once (h5py supports slice reads natively).
    n = f['virial_mass'].shape[0]
    chunk = -(-n // GROUPS)                                # ceil(n / GROUPS) halos per group
    start, stop = group * chunk, min((group + 1) * chunk, n)
    Greens = f['Green_params'][start:stop]
    Mvir = f["virial_mass"][start:stop]
    positions = f['position'][start:stop]

####

idcs = np.arange(start, stop)

####

halo_instances = [Green(*Greens[i]) for i in range(len(Greens))]
for i in range(len(halo_instances)):
    halo_instances[i].update_mass(Mvir[i])
    # halo_instances[i].update_mass(Greens[i][0])

### Compute halo distances
gc = SkyCoord(x=positions[:,0]*u.kpc, y=positions[:,1]*u.kpc, z=positions[:,2]*u.kpc, frame='galactocentric')
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

# Extract the distance to the dwarf galaxy, with error handling for different data formats

# try:
    
    # d_kpc = 10 ** (dwarfs['dm'][dsph_idx] / 5 + 1) / 1e3  # Distance in kpc
d_kpc = dwarf.distance.to(u.kpc).value
print(d_kpc)

    ####### is this a mistake?
d_err_kpc = np.mean(dwarf.distance_err.to(u.kpc).value)
    # dlo_kpc = np.log(10) * dwarfs['edm_plus'][dsph_idx] * d_kpc / 5  # Lower distance error
    # dhi_kpc = np.log(10) * dwarfs['edm_minus'][dsph_idx] * d_kpc / 5  # Upper distance error
    
# derr_kpc = (dlo_kpc + dhi_kpc) / 2  # Average error in distance
# except KeyError:
#     # Use alternative key if 'dm' not found
#     d_kpc = dwarfs['dist'][dsph_idx]
#     derr_kpc = dwarfs['dist_err'][dsph_idx]

# d_kpc = 30

print(f"Found parameters for {name}")
# print(f"Distance: {d_kpc:.1f} kpc  Distance Error: {derr_kpc:.1f} kpc")

# ====================================================================================== #

### Calculate J-factor for a given density profile

def los_radius(L, d, theta):
    # Exact distance from the dwarf centre to a point at line-of-sight distance L
    # and angle theta from the dwarf direction. Algebraically this is the law of
    # cosines L^2 + d^2 - 2*L*d*cos(theta), but that form suffers catastrophic
    # cancellation for L~d, theta~0 and can go slightly negative -> sqrt -> nan.
    # The half-angle form (L-d)^2 + 4*L*d*sin^2(theta/2) is identical and
    # manifestly >= 0 (no cancellation). No small-angle approximation: valid to
    # theta = pi/2. Floor just off zero so the central r=0 (NFW cusp rho ~ 1/r)
    # doesn't divide by zero; the cusp is integrable so the floor doesn't bias J.
    return np.maximum(np.sqrt((L - d)**2 + 4.0 * L * d * np.sin(theta / 2.0)**2), 1e-6)

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
    r = los_radius(L, d, theta)
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

def compute_J_and_containment(rho, d_kpc, aperture, theta_max=np.pi / 2,
                              n_theta=400, n_L=64, frac=0.95):
    # green_Js is the J-factor within `aperture` (0.5 deg); theta95 is the 95%
    # containment angle of the full J-factor (integrated to theta_max); full_Js
    # is that full J-factor itself (integrated to theta_max). All come
    # from ONE deterministic quadrature -- ~95x faster than the previous vegas
    # Monte-Carlo and bias-free to <0.05% vs the production reference (validated
    # against the Diemer_backup green_Js and an independent scipy dblquad). Being
    # deterministic, theta_max=pi/2 (true all-sky) is essentially free, so we no
    # longer truncate (the old 10-deg cut clamped theta95 for the nearest dwarfs).
    d = d_kpc
    # log-spaced theta grid for resolution across scales, with theta=0 and the
    # aperture forced in as exact nodes (so J(<aperture) is read off directly).
    th = np.unique(np.concatenate([
        [0.0],
        np.logspace(np.log10(aperture * 1e-3), np.log10(theta_max), n_theta),
        [aperture]]))
    assert np.any(th == aperture), 'aperture must survive as an exact theta node'

    # Line-of-sight integral I(theta) = int_0^2d rho(r)^2 dL via Gauss-Legendre.
    # r^2 = rmin^2 + (L-L0)^2 with L0 = d cos(theta) (closest approach) and
    # rmin = d sin(theta) (impact parameter). Substitute L-L0 = rmin tan(phi):
    # r = rmin sec(phi), dL = rmin sec^2(phi) dphi -- this clusters GL nodes at
    # closest approach where rho (hence the integrand) peaks. L0 in [0, d] so
    # phi_lo <= 0 <= phi_hi always (closest approach is inside [0, 2d]).
    gx, gw = leggauss(n_L)                       # GL nodes/weights on [-1, 1]
    L0 = d * np.cos(th)
    rmin = np.maximum(d * np.sin(th), 1e-12)
    phi_lo = np.arctan((0.0 - L0) / rmin)
    phi_hi = np.arctan((2 * d - L0) / rmin)
    half = 0.5 * (phi_hi - phi_lo); mid = 0.5 * (phi_hi + phi_lo)
    phi = mid[:, None] + half[:, None] * gx[None, :]         # (T, n_L)
    r = rmin[:, None] / np.cos(phi)                          # = rmin * sec(phi)
    integrand = rho(r)**2 * (rmin[:, None] / np.cos(phi)**2)  # rho^2 * dL/dphi
    I = (integrand * (half[:, None] * gw[None, :])).sum(axis=1)

    w = 2 * np.pi * np.sin(th) * I               # dJ/dtheta
    # At theta=0 the cusp makes I ~ 1/rmin diverge while sin(theta) -> 0; their
    # product (dJ/dtheta) tends to a finite NONZERO constant. The literal
    # w[0] = 2pi*sin(0)*I(0) collapses 0*inf to 0, under-counting the first
    # interval by a theta-grid-INDEPENDENT amount (a ~0.4% low green_Js bias that
    # is invisible to n_theta/n_L convergence). Use the cusp limit instead.
    # NOTE: assumes an NFW-like inner cusp rho~1/r (gamma=1), for which
    # dJ/dtheta ~ theta^(2-2*gamma) -> a finite constant. Production uses Green
    # profiles (gamma=1) so this holds. For a steeper cusp (gamma>1, e.g. Dekel)
    # w(theta) DIVERGES as theta->0 and w[0]=w[1] would bias green_J low -- such
    # profiles need analytic treatment of the divergent first interval.
    w[0] = w[1]
    cdf = np.concatenate([[0.0],
                          np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(th))])

    J_full = cdf[-1]
    J_aperture = np.interp(aperture, th, cdf)    # exact: aperture is a node
    theta95 = np.interp(frac * J_full, cdf, th)  # 95% containment of full J
    green_J = (J_aperture * u.solMass ** 2 * const.c ** 4 / (u.kpc ** 5)).to(u.GeV ** 2 / u.cm ** 5).value
    full_J = (J_full * u.solMass ** 2 * const.c ** 4 / (u.kpc ** 5)).to(u.GeV ** 2 / u.cm ** 5).value
    return green_J, theta95, full_J

# ====================================================================================== #
    
# Compute J factors
green_Js = np.zeros(len(idcs))
theta95 = np.zeros(len(idcs))
full_Js = np.zeros(len(idcs))
progress_file = os.path.join(save_dir, filename + '.progress')
# nfw_Js = np.zeros(len(idcs))
# rs_green_Js = np.zeros(len(idcs))
# rs_nfw_Js = np.zeros(len(idcs))

for j in range(len(idcs)):
    # idx = idcs[j]
    # nfw_pf = tNFW(halo_instances[j], z = Greens[j][-1])
    # rs = nfw_pf.rh
    # rs_angle = compute_angle(d_kpc, rs)

    green_Js[j], theta95[j], full_Js[j] = compute_J_and_containment(halo_instances[j].rho, d_kpc, 0.5 * np.pi / 180)
    if (j + 1) % 1000 == 0:
        with open(progress_file, 'w') as pf:
            pf.write(f'{j + 1}/{len(idcs)}\n')
    # green_Js[j] = compute_J_factor(halo_instances[j].rho, d_kpc, 0.5 * np.pi / 180).value
    # nfw_Js[j] = compute_J_factor(nfw_pf.rho, d_kpc, np.pi / 2).value

    # rs_green_Js[j] = compute_J_factor(halo_instances[j].rho, d_kpc, rs_angle).value
    # rs_nfw_Js[j] = compute_J_factor(nfw_pf.rho, d_kpc, rs_angle).value

# ====================================================================================== #
    
# Save results

result_dict = {
    'green_Js': green_Js,
    # 'nfw_Js': nfw_Js,
    # 'rs_green_Js': rs_green_Js,
    # 'rs_nfw_Js': rs_nfw_Js,
    'theta95': theta95,
    'full_Js': full_Js
}

# Save results to file
record = provenance.stamp('python/Jdwarf.py', version=version, argv=sys.argv[1:],
                          inputs=[config.h5_path(version)],
                          dwarf=name, group=group, n_groups=GROUPS,
                          halo_slice=[int(start), int(stop)])
provenance.savez(os.path.join(save_dir, filename), record, **result_dict)

# remove heartbeat file on success
if os.path.exists(progress_file):
    os.remove(progress_file)
    