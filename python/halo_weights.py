import h5py
import numpy as np
import logging
from scipy import stats
from scipy import interpolate as interp
from scipy.special import erf
from KDEpy import FFTKDE as KDE

import Jdata as obs
import config

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord

import os

# The per-version h5 path constants that used to live here are now
# config.H5_REGISTRY, so the version->file mapping has one home and an unknown
# version fails with a message naming the valid keys. These aliases remain so
# existing references keep working.
# str, not Path, matching their upstream type: consumers may concatenate.
m2e12_h5 = str(config.h5_path('m2e12'))
Diemer_h5 = str(config.h5_path('Diemer'))
Diemer_0p12_h5 = str(config.h5_path('Diemer_0p12'))
Zhao_h5 = str(config.h5_path('Zhao'))
splashback_h5 = str(config.h5_path('splashback'))
Symphony_h5 = str(config.h5_path('Symphony'))
MWest_h5 = str(config.h5_path('MWest'))
mass_floor_7_h5 = str(config.h5_path('mass_floor_7'))

mhalf_folder = str(config.MHALF_DIR) + '/'

class ResampleMstar():
    '''
    This class recalculatates the inference weights of SatGen halos for a given dwarf and set of stellar masses.
    
    Parameters
    ----------
    use_20k : bool, default True
        if True, use the full sample (20k MWs, with disk). If False, use the earlier sample (2k MWs, no disk)

    Attributes
    ----------
    mass_floor : float, default 7
        assign zero weight to halos with logMpeak < mass_floor. By default, restrict to halos with logMpeak > 10^7
    floor_width : float or None, default None
        if None, the mass floor is implemented as a hard-cut step function
        if a float, the mass floor is implemented as a logistic in logMpeak, with width `floor_width` (in dex)
    cut : None or str or np.ndarray(bool)
        Implements a cut on the ``SatGen`` halos. 
        If ``None``, the full sample (above the mass floor) is used to set the weights. 
        If ``"peri"``, the ``SatGen`` pericenter must be within the one-sigma observed value.
        (If no halos fall within this pericenter band, restrict to those halos with pericenter greater than half the maximum observed pericenter.)
        If an ``np.ndarray``, it is assumed that this is a custom mask restricting to ``SatGen`` halos of interest. 
    kde_kwargs : dict, optional
        passed to the KDEs for the logMstar-logMhalf PDF
    '''


    def __init__(self, version='Diemer', mass_floor=8, cut='distance'):
        self.h5 = get_h5(version)
        self.version = version

        if version == 'Symphony' or version == 'MWest':
            self.logMpeak = np.log10(self.h5['mpeak'])
            self.logMvir = np.log10(self.h5['mvir'])
    
            self.mwID = self.h5['host_id'][()]
    
            self.position = self.h5['x'][()]
            gc = SkyCoord(x=self.position[:,0]*u.kpc, y=self.position[:,1]*u.kpc, z=self.position[:,2]*u.kpc, frame='galactocentric')
            self.distance = gc.transform_to('galactic').distance
            self.gc_distance = np.linalg.norm(self.position * u.kpc, axis=1)
        else:
            self.logMpeak = np.log10(self.h5['Green_params'][:,0])
            self.logMvir = np.log10(self.h5['virial_mass'])
            self.logMstar_infall = np.log10(self.h5['tpeak_stellar_mass'][()])
            self.z_in = self.h5['Green_params'][()][:,3]
            self.mw_hosts_lmc = self.h5['mw_hosts_lmc'][()]
    
            self.mwID = self.h5['mwID'][()]
    
            self.position = self.h5['position'][()]
            gc = SkyCoord(x=self.position[:,0]*u.kpc, y=self.position[:,1]*u.kpc, z=self.position[:,2]*u.kpc, frame='galactocentric')
            self.distance = gc.transform_to('galactic').distance
            self.gc_distance = np.linalg.norm(self.position * u.kpc, axis=1)

        self.mass_floor = mass_floor
        self.floor_width = None
        self.cut = cut
        self.kde_kwargs = dict(kernel='epa', bw=0.01, ngrid=2**11)

        # Per-halo infall -> z=0 stellar-mass mapping, consumed by
        # evolved_Mstar. The catalog already carries both endpoints of the
        # Errani+18 tidal track SatGen applied when the trees were built, so
        # this is a lookup, not a model: stellar_mass is the post-stripping
        # mass and tpeak_stellar_mass the mass at peak.
        #
        # Upstream (and this repository until now) used a scalar
        # `self.Mstar_ratio = 1.2` here, added directly to log10(Mstar) -- a
        # constant +1.2 dex, a factor of ~16 upward. Its comment described the
        # multiplicative ratio below, so the value was a linear ratio applied
        # in log space, and a constant standing in for a quantity the data
        # carries per halo. It is upward, while a tidal track can only remove
        # stellar mass. See docs/migration-notes.md.
        #
        # Clipped at 0: 1-4% of halos carry a tiny positive ratio (max
        # +0.006 dex across every catalog), which is numerical noise in the
        # tree-building rather than stellar mass growing after infall. The
        # ratio is the fraction of stellar mass remaining after stripping
        # (sec:model) and is therefore bounded above by 1 (0 in log space);
        # the clip enforces that bound rather than trusting it to hold.
        #
        # Symphony/MWest carry no stellar-mass fields and never reach the SHMR
        # path (compute_weights.py guards on the version), so the ratio is
        # left undefined there and evolved_Mstar raises rather than
        # broadcasting a scalar silently.
        if version == 'Symphony' or version == 'MWest':
            self.logMstar_ratio = None
        else:
            self.logMstar_ratio = np.minimum(
                np.log10(self.h5['stellar_mass'][()]) - self.logMstar_infall,
                0.0)
            if not np.all(np.isfinite(self.logMstar_ratio)):
                # log10(0) is a RuntimeWarning, not an exception, so a zero
                # stellar mass would reach _kde2D as -inf (or nan, if both
                # endpoints are zero) and poison every weight for the dwarf,
                # not just its own halo -- surfacing as NaN weights far from
                # the cause. No catalog currently has one, but splashback
                # reaches 1.3e-4 Msun, four orders below the next lowest.
                # Fail loudly; clamping would bias the conditioning silently.
                n_bad = int(np.sum(~np.isfinite(self.logMstar_ratio)))
                raise ValueError(
                    f'{version}: non-finite infall -> z=0 stellar-mass ratio '
                    f'for {n_bad} halos (zero or missing stellar_mass / '
                    'tpeak_stellar_mass).')

    def get_mask(self, dwarf, distance_factor=2., use_z50=False, use_z90=False, lmc_selection=False, lmc_max_distance=100, use_kd=True, use_mcdaniel=False):
        if isinstance(dwarf, str):
            dwarf = obs.Dwarf(dwarf, use_kd=use_kd, use_mcdaniel=use_mcdaniel)
        name = dwarf.name

        # use all halos for the cut
        if self.cut is None:
            mask = np.ones(position.shape, dtype = bool)
        # restrict to satgen halos with reasonable pericenters
        elif self.cut == 'distance':
            # restrict to halos within factor of 2 of observed distance
            mask = (self.gc_distance > dwarf.gc_distance/distance_factor) & (self.gc_distance < dwarf.gc_distance*distance_factor)
            # if no halos fall within this band, just say that the pericenter has to be big
            if not np.any(mask):
                mask = self.gc_distance > np.max(self.gc_distance)/2
        # elif self.cut == 'distance narrow':
        #     # restrict to halos within factor of 2 of observed distance
        #     mask = (self.gc_distance > dwarf.gc_distance/1.25) & (self.gc_distance < dwarf.gc_distance*1.25)
        #     # if no halos fall within this band, just say that the pericenter has to be big
        #     if not np.any(mask):
        #         mask = self.gc_distance > np.max(self.gc_distance)/2
        # elif self.cut == 'distance extra narrow':
        #     # restrict to halos within factor of 2 of observed distance
        #     mask = (self.gc_distance > dwarf.gc_distance/1.1) & (self.gc_distance < dwarf.gc_distance*1.1)
        #     # if no halos fall within this band, just say that the pericenter has to be big
        #     if not np.any(mask):
        #         mask = self.gc_distance > np.max(self.gc_distance)/2
        # can also just pass a boolean array as the cut 
        elif isinstance(self.cut, np.ndarray) and (np.shape(self.cut) == np.shape(logMstar)):
            mask = cut
        else:
            raise TypeError(f'cut of type {type(self.cut)} unrecognized')

        if self.floor_width is None:
            halo_hosts_galaxy = (self.logMpeak > self.mass_floor) 
        elif isinstance(self.floor_width, (int,float)):
            galaxy_probability = self.galaxy_probability()
            halo_hosts_galaxy = np.random.rand(mask.size) < galaxy_probability
        else:
            raise TypeError(f'floor_width of type {type(self.floor_width)} unrecognized')
            
        mask = mask & halo_hosts_galaxy

        if use_z50:
            z50 = self.h5['z50'][()]
            mask = mask & (z50 > 6)
        elif use_z90:
            z90 = self.h5['z90'][()]
            mask = mask & (z90 > 6)

        if lmc_selection:
            lmc_dist_cut = self.gc_distance.value < lmc_max_distance
            lmc_mass_cut = self.logMpeak > 11
            lmc_mwIDs = self.mwID[lmc_dist_cut & lmc_mass_cut]
            mw_hosts_lmc = np.isin(self.mwID, lmc_mwIDs)
            
            # Prepare mask: start with all True
            remove_lmc_mask = np.ones(len(self.logMpeak), dtype=bool)
                
            # Get unique Milky Way IDs
            unique_mwIDs = np.unique(self.mwID)
            # Loop over each MW and find the index of the max mass
            for mw in unique_mwIDs:
                # indices belonging to this MW
                mw_inds = np.where(self.mwID == mw)[0]
                # index of the heaviest satellite in this MW
                max_ind = mw_inds[np.argmax(self.logMpeak[mw_inds])]
                # mark it for removal
                remove_lmc_mask[max_ind] = False
                    
            mask = mask & remove_lmc_mask
            mask = mask & mw_hosts_lmc

        return mask
        
    def evolved_Mstar(self, logMstar_init, truncate = False):
        '''
        maps from logMstar at tpeak to logMstar at z = 0
        if truncate, use the object's `mass_floor` and `floor_width` to truncate the stellar mass.
        this allows shallow Mstar - Mhalo relations to be effectively truncated at `mass_floor`
        '''
        # Guards first: with truncate=True a wrong-length input would raise an
        # opaque broadcast error from the add below, and a Symphony catalog
        # would evaluate galaxy_probability() before reaching the None check.
        if self.logMstar_ratio is None:
            raise ValueError(
                f'evolved_Mstar is unavailable for version {self.version!r}: '
                'the catalog carries no stellar_mass/tpeak_stellar_mass '
                'fields, so the infall -> z=0 mapping cannot be looked up.')
        if np.shape(logMstar_init) != np.shape(self.logMstar_ratio):
            # The SHMRs are evaluated on the full-length self.logMpeak and
            # masking happens downstream in from_logMstar, so these must align
            # elementwise. A mismatch would broadcast silently and mispair
            # every halo with another halo's stripping history.
            raise ValueError(
                f'logMstar_init has shape {np.shape(logMstar_init)}, expected '
                f'{np.shape(self.logMstar_ratio)} (full catalog length) -- '
                'evolved_Mstar must be applied before any mask.')
        if truncate:
            mstar_suppression = np.log10(self.galaxy_probability())
            logMstar_init = logMstar_init + mstar_suppression
        # new z0 Mstar = (z0 Mstar / infall Mstar) * new infall Mstar
        return logMstar_init + self.logMstar_ratio

    def from_callable(self, callable, dwarf, return_Mstar = False):
        '''
        get weights for a given dwarf input
        callable maps logMpeak to logMstar at tpeak
        since it's at tpeak, before tidal stripping, must be a relation for field dwarfs
        must have no other inputs - use from_logMstar otherwise
        '''
        logMstar_init = callable(self.logMpeak)
        logMstar_z0 = self.evolved_Mstar(logMstar_init)
        joint_mass = self.from_logMstar(logMstar_z0, dwarf)
        return (joint_mass, logMstar_z0) if return_Mstar else joint_mass

    def galaxy_probability(self):
        if self.floor_width is None:
            return (self.logMpeak > self.mass_floor).astype(float)
        elif isinstance(self.floor_width, (int,float)):
            return (1 + erf((self.logMpeak - self.mass_floor)/(np.sqrt(2) * self.floor_width)))/2
        else:
            raise TypeError(f'floor_width of type {type(self.floor_width)} unrecognized')
        
    def from_logMstar(self, logMstar, dwarf, use_z50=False, use_z90=False, lmc_selection=False, lmc_max_distance=100, logMhalf_scatter=0.0, use_kd=True, use_mcdaniel=False):
        '''
        get weights for a given dwarf input
        provide logMstar at z = 0
        '''
        if isinstance(dwarf, str):
            # Upstream drops logMhalf_scatter here, so obs.Dwarf.__init__'s
            # default of 0.1 always applied regardless of the caller's value.
            # This repository forwards it, matching from_logMhalf below.
            dwarf = obs.Dwarf(dwarf, logMhalf_scatter=logMhalf_scatter, use_kd=use_kd, use_mcdaniel=use_mcdaniel)
        name = dwarf.name

        mask = self.get_mask(dwarf, use_z50=use_z50, use_z90=use_z90, lmc_selection=lmc_selection, lmc_max_distance=lmc_max_distance, use_kd=use_kd, use_mcdaniel=use_mcdaniel)
        # logMhalf = self.h5['logMhalf'][name][mask]
        logMhalf = np.load(f'{mhalf_folder}{self.version}/{name}.npz')['logMhalf'][mask]
        logMstar = logMstar[mask]
        
        bayesian_weight = np.zeros(mask.shape)
        obs_weight = dwarf.logMhalf_pdf(logMhalf) * dwarf.logMstar_pdf(logMstar)
        kde = self._kde2D(logMstar, logMhalf, **self.kde_kwargs) # todo: check that our choice of bandwidth is good
        normalization = kde(np.c_[logMstar, logMhalf])
        bayesian_weight[mask] = obs_weight/normalization
        return bayesian_weight

    def from_logMhalf(self, dwarf, use_z50=False, use_z90=False, lmc_selection=False, lmc_max_distance=100, logMhalf_scatter=0.0, use_kd=True, use_mcdaniel=False):
        '''
        get weights for a given dwarf input
        provide logMstar at z = 0
        '''
        if isinstance(dwarf, str):
            dwarf = obs.Dwarf(dwarf, logMhalf_scatter=logMhalf_scatter, use_kd=use_kd, use_mcdaniel=use_mcdaniel)
        name = dwarf.name

        mask = self.get_mask(dwarf, use_z50=use_z50, use_z90=use_z90, lmc_selection=lmc_selection, lmc_max_distance=lmc_max_distance, use_kd=use_kd, use_mcdaniel=use_mcdaniel)
        if self.version == 'Symphony' or self.version == 'MWest':
            logMhalf = self.h5['logMhalf'][name][mask]
        else:
            logMhalf = np.load(f'{mhalf_folder}{self.version}/{name}.npz')['logMhalf'][mask]
        
        bayesian_weight = np.zeros(mask.shape)
        obs_weight = dwarf.logMhalf_pdf(logMhalf)
        kde = self._kde1D(logMhalf, **self.kde_kwargs) # todo: check that our choice of bandwidth is good
        normalization = kde(logMhalf)
        bayesian_weight[mask] = obs_weight/normalization
        return bayesian_weight

    def from_logMdyn_errani(self, dwarf, use_z50=False, use_z90=False, lmc_selection=False, lmc_max_distance=100, use_kd=True, use_mcdaniel=False):
        '''
        get weights for a given dwarf input
        provide logMstar at z = 0
        '''
        if isinstance(dwarf, str):
            dwarf = obs.Dwarf(dwarf, use_kd=use_kd, use_mcdaniel=use_mcdaniel)
        name = dwarf.name

        mask = self.get_mask(dwarf, use_z50=use_z50, use_z90=use_z90, lmc_selection=lmc_selection, lmc_max_distance=lmc_max_distance, use_kd=use_kd, use_mcdaniel=use_mcdaniel)
        # logMhalf = self.h5['logMhalf'][name][mask]
        logMdyn_errani = np.load(f'{mhalf_folder}{self.version}/{name}.npz')['logMdyn_errani'][mask]
        
        bayesian_weight = np.zeros(mask.shape)
        obs_weight = dwarf.logMdyn_errani_pdf(logMdyn_errani)
        kde = self._kde1D(logMdyn_errani, **self.kde_kwargs) # todo: check that our choice of bandwidth is good
        normalization = kde(logMdyn_errani)
        bayesian_weight[mask] = obs_weight/normalization
        return bayesian_weight

    def from_logMstar_only(self, logMstar, dwarf, use_z50=False, use_z90=False, lmc_selection=False, lmc_max_distance=100, use_kd=True, use_mcdaniel=False):
        '''
        get weights for a given dwarf input
        provide logMstar at z = 0
        '''
        if isinstance(dwarf, str):
            dwarf = obs.Dwarf(dwarf, use_kd=use_kd, use_mcdaniel=use_mcdaniel)
        name = dwarf.name

        mask = self.get_mask(dwarf, use_z50=use_z50, use_z90=use_z90, lmc_selection=lmc_selection, lmc_max_distance=lmc_max_distance, use_kd=use_kd, use_mcdaniel=use_mcdaniel)
        logMstar = logMstar[mask]
        
        bayesian_weight = np.zeros(mask.shape)
        obs_weight = dwarf.logMstar_pdf(logMstar)
        kde = self._kde1D(logMstar, **self.kde_kwargs) # todo: check that our choice of bandwidth is good
        normalization = kde(logMstar)
        bayesian_weight[mask] = obs_weight/normalization
        return bayesian_weight
        
    # for use as a context manager: clean up the h5 file stream
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.h5.close()
    
    @staticmethod
    def _kde1D(valuesx, weights=None, kernel='epa', bw=0.01, ngrid=2**11):
        '''Create a 1D PDF for the distribution of ``valuesx`` using KDE.
        
        Parameters
        ----------
        valuesx : list 
            The x values from which the PDF is generated.
        weights : list, optional
            The weights to apply to the points. This must have the same length as ``valuesx``. Defaults to None, indicating equal weight.
        kernel : optional
            The kernel to use in the KDE, used as input to ``KDEpy.FFTKDE``. Defaults to 'epa' (Epanechnikov kernel).
        bw : optional
            The bandwidth to use in the KDE, used as input to ``KDEpy.FFTKDE``. Defaults to '0.01' (this MUST be set appropriately based on the number of values and their distribution).
        ngrid : optional
            The number of grid points on which to evaluate the KDE. 
        
        Returns
        -------
        scipy.interpolate.RegularGridInterpolator
            Interpolates the 1D histogram for the PDF of ``valuesx``. This uses the ``'nearest'`` method to preserve PDF volume.
        '''
        grid, density = KDE(kernel=kernel, bw=bw).fit(valuesx, weights=weights).evaluate(ngrid)
        
        # Extend x-axis for interpolation
        xax_spacing = np.diff(grid)[0]
        xax_interp = np.concatenate(([grid[0] - xax_spacing], grid, [grid[-1] + xax_spacing]))
        
        # Zero-pad to ensure probability is zero outside the domain
        interp_density = np.zeros(len(grid) + 2)
        interp_density[1:-1] = density
        
        pdf = interp.RegularGridInterpolator((xax_interp,), interp_density, 
                                             bounds_error=False, fill_value=0, method='nearest')
        return pdf
        
    @staticmethod
    def _kde2D(valuesx, valuesy, weights = None, kernel='epa', bw=0.01, ngrid=2**11):
        '''Create a 2D PDF for the joint distribution of ``valuesx`` and ``valuesy`` by KDEing on a fine grid and piecewise-constantly interpolating the KDE values.
        
        Parameters
        ----------
        valuesx, valuesy : list 
            The x- and y values from which the PDF is generated.
        weights : list, optional
            The weights to apply to the points. This must have the same length as ``valuesx`` and ``valuesy``. Defaults to None, indicating equal weight.
        kernel : optional
            The kernel to use in the KDE, used as input to ``KDEpy.FFTKDE``. Defaults to 'epa' (Epanechnikov kernel).
        bw : optional
            The bandwidth to use in the KDE, used as input to ``KDEpy.FFTKDE``. Defaults to '0.01' (this MUST be set appropriately based on the number of values and their distribution).
        ngrid : optional
            The number of gridpoints on which to evaluate the KDE. The final PDF is piecewise-constantly interpolated between these gridpoints.
        
        Returns
        -------
        scipy.interpolate.RegularGridInterpolator
            Interpolates the 2D histogram for the joint PDF of (``valuesx``, ``valuesy``). This uses the ``'nearest'`` method to preserve PDF volume.
        
        '''
        grid, height = KDE(kernel='epa', bw=0.01).fit(np.c_[valuesx, valuesy], weights=weights).evaluate(ngrid)
        # extend xaxis
        xax = np.linspace(grid[0,0], grid[-1,0], num = ngrid)
        xax_spacing = np.diff(xax)[0]
        xax_interp = np.concatenate(([xax[0] - xax_spacing], xax, [xax[-1] + xax_spacing]))
        # extend yaxis
        yax = np.linspace(grid[0,1], grid[-1,1], num = ngrid)
        yax_spacing = np.diff(yax)[0]
        yax_interp = np.concatenate(([yax[0] - yax_spacing], yax, [yax[-1] + yax_spacing]))
        # zero-pad the interpolation array to ensure the probability is zero outside of the domain of interest
        interpheight = np.zeros((ngrid +2, ngrid+2))
        interpheight[1:-1, 1:-1] = height.reshape((ngrid,ngrid))
        pdf = interp.RegularGridInterpolator((xax_interp, yax_interp), interpheight, 
                                             bounds_error = False, fill_value = 0, method = 'nearest')
        return pdf

def get_h5(sim_version):
    """Open the SatGen halo catalog for `sim_version`.

    The version->file mapping lives in config.H5_REGISTRY. The if/elif chain
    this replaced had no else-branch, so an unknown version raised
    UnboundLocalError from inside this function rather than saying which
    versions exist.

    Aliases are deliberately not resolved, matching the replaced chain: callers
    compute their own sim_version and pass it in. Resolving here would make
    ResampleMstar('lmc') constructible, pairing the Diemer catalog with
    mhalf/lmc/ -- a mixed-version object the old code refused to build.
    """
    return h5py.File(config.h5_path(sim_version, resolve_alias=False))

def logMstar_Kim24(logM200, return_params=False, low_scatter=False):
    scatter = -0.227 * logM200 + 2.59
    scatter = np.clip(scatter, a_min = 0.2, a_max=None)
    mean = 1.76 * logM200 - 10.7
    if return_params:
        return mean, scatter
    else:
        if low_scatter:
            scatter=0.01
        return stats.norm.rvs(loc=mean, scale=scatter)

def lgMs_B13(lgMv,z=0.):
    ''' taken from satgen '''
    a = 1./(1.+z)
    v = np.exp(-4.*a**2)
    e0, ea, ez, ea2 = -1.777, -0.006, 0.000, -0.119
    M0, Ma, Mz = 11.514, -1.793, -0.251
    lge = e0 + (ea*(a-1.)+ez*z)*v + ea2*(a-1.)
    lgM = M0 + (Ma*(a-1.)+Mz*z)*v
    
    def f_B13(x,a):
        r"""
        Auxiliary function for lgMs_B13.
        """
        a0, aa, az = -1.412, 0.731, 0.0
        d0, da, dz = 3.508, 2.608, -0.043
        g0, ga, gz = 0.316, 1.319, 0.279
        v = np.exp(-4.*a**2)
        z = 1./a-1.
        alpha = a0 + (aa*(a-1.)+az*z)*v
        delta = d0 + (da*(a-1.)+dz*z)*v
        gamma = g0 + (ga*(a-1.)+gz*z)*v
        return delta*(np.log10(1.+np.exp(x)))**gamma/(1.+np.exp(10**(-x)))-\
            np.log10(1.+10**(alpha*x))
    return lge+lgM + f_B13(lgMv-lgM,a) - f_B13(0.,a)


def lgMs_RP17(lgMv,z=0.):
    """ taken from satgen """
    a = 1./(1.+z)
    v = np.exp(-4.*a**2)
    e0, ea, ez, ea2 = -1.758, 0.110, -0.061, -0.023
    M0, Ma, Mz = 11.548, -1.297, -0.026
    lge = e0 + (ea*(a-1.)+ez*z)*v + ea2*(a-1.)
    lgM = M0 + (Ma*(a-1.)+Mz*z)*v
    def f_RP17(x,a):
        r"""
        Auxiliary function for lgMs_RP17.
        Note that RP+17 use 10**( - alpha*x) while B+13 used 10**( +alpha*x).
        """
        a0, aa, az = 1.975, 0.714, 0.042
        d0, da, dz = 3.390, -0.472, -0.931
        g0, ga, gz = 0.498, -0.157, 0.0
        v = np.exp(-4.*a**2)
        z = 1./a-1.
        alpha = a0 + (aa*(a-1.)+az*z)*v
        delta = d0 + (da*(a-1.)+dz*z)*v
        gamma = g0 + (ga*(a-1.)+gz*z)*v
        return delta*(np.log10(1.+np.exp(x)))**gamma/(1.+np.exp(10**(-x)))-\
            np.log10(1.+10**( - alpha*x))
    return lge+lgM + f_RP17(lgMv-lgM,a) - f_RP17(0.,a)


def logMstar_B13(logMvir, z, return_params=False, low_scatter=False):
    mean = lgMs_B13(logMvir, z)
    # scatter = 0.2
    a = 1/(1+z)
    scatter = 0.218 - 0.023 * (a - 1)
    if return_params:
        return mean, scatter
    else:
        if low_scatter:
            scatter=0.01
        return np.random.normal(mean,scatter)
    
def logMstar_RP17(logMvir, z, return_params=False, low_scatter=False):
    mean = lgMs_RP17(logMvir, z)
    scatter = 0.15 # diff from satgen value
    if return_params:
        return mean, scatter
    else:
        if low_scatter:
            scatter=0.01
        return np.random.normal(mean, scatter)

def logMstar_Moster13(logMvir, z, return_params=False, low_scatter=False):
    # logmvir_over_m1 = logMvir-11.59
    a = 1/(1 + z)
    logmvir_over_m1 = logMvir - (11.59 + 1.195*(1-a))
    mvir_over_m1 = 10**logmvir_over_m1
    # beta, gamma = 1.376, 0.608
    beta, gamma = 1.376 - 0.826*(1-a), 0.608 + 0.329 * (1-a)
    denom = np.log10(mvir_over_m1**(-beta) + mvir_over_m1**gamma)
    # norm_factor = 0.0702
    norm_factor = 2 * (0.0351 - 0.0247 * (1 - a))
    mean = np.log10(norm_factor) - denom + logMvir
    if return_params:
        return mean, 0.15
    else:
        if low_scatter:
            scatter=0.01
        return np.random.normal(mean, 0.15)

def logMstar_Moster18(logMvir, z, return_params=False, low_scatter=False):
    a = 1/(1 + z)
    logmvir_over_m1 = logMvir - (11.339 + 0.692*(1-a))
    mvir_over_m1 = 10**logmvir_over_m1
    beta, gamma = 3.344 - 2.079*(1-a), 0.966
    denom = np.log10(mvir_over_m1**(-beta) + mvir_over_m1**gamma)
    norm_factor = 2 * (0.005 + 0.689 * (1 - a))
    mean = np.log10(norm_factor) - denom + logMvir
    
    sigma0, a, logMsigma = 0.16, 1, 10.85
    scatter = sigma0 #+ np.log10(10**(-a * (logMvir - logMsigma)) + 1)
    if return_params:
        return mean, scatter
    else:
        if low_scatter:
            scatter=0.01
        return np.random.normal(mean, scatter)

def logMstar_Fattahi18(Vmax, return_params=False, low_scatter=False):
    eta = Vmax/50
    logM0, alpha, mu = np.log10(3)+8, 3.36, -2.4
    mean = alpha * np.log10(eta) - (eta**mu)/np.log(10) + logM0      # fattahi provide no scatter
    scatter = np.where(Vmax < 57, -1.26 * np.log10(Vmax/88.6), 0.24) # from santos-santos 22, eq 5
    if return_params:
        return mean, scatter
    else:
        if low_scatter:
            scatter=0.01
        return np.random.normal(mean, scatter)

def logMstar_Behroozi19(lgMvir, z, return_params=False, low_scatter=False):
    '''
    Behroozi+19 SMHMRL Appendix J
    https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3143B

    with mass-dependent scatter from Munshi+21
    '''
    a = 1/(1 + z)
    lna = np.log(a)
    # table J1, "SM = Obs., Q/SF = All, Cen/Sat = All, IHL = Excl."
    epsilon_0, epsilon_a, epsilon_lna, epsilon_z = -1.435, 1.831, 1.368, -0.217
    M_0, M_a, M_lna, M_z = 12.035, 4.556, 4.417, -0.731,
    alpha_0, alpha_a, alpha_lna, alpha_z = 1.963, -2.316, -1.732, 0.178
    beta_0, beta_a, beta_z =  0.482, -0.841, -0.471
    delta = 0.411
    gamma_0, gamma_a, gamma_z = -1.034, -3.100, -1.055
    # J3 - J8
    lgM1 = M_0 + M_a * (a - 1) - M_lna * lna + M_z * z
    epsilon = epsilon_0 + epsilon_a * (a - 1) - epsilon_lna * lna + epsilon_z * z
    alpha = alpha_0 + alpha_a * (a - 1) - alpha_lna * lna + alpha_z * z
    beta = beta_0 + beta_a * (a - 1) + beta_z * z
    gamma = 10**(gamma_0 + gamma_a * (a - 1) + gamma_z * z)
    # J2 and J1
    x = lgMvir - lgM1
    lgMstar = epsilon - np.log10(10**(-alpha * x) + 10**(-beta * x)) + gamma * np.exp(-0.5*(x/delta)**2) + lgM1
    # Mass-dependent scatter from Munshi21
    # scatter = 0.3 + np.where(lgMvir > 10, 0, -0.39 * (lgMvir - 10))
    scatter = 0.2
    if return_params:
        return lgMstar, scatter
    else:
        if low_scatter:
            scatter=0.01
        return np.random.normal(lgMstar, scatter)

def logMstar_Munshi21(logMvir, return_params=False, low_scatter=False):
    low_mass_slope = 2.81
    high_mass_slope = 1.9
    mean = np.where(logMvir > 10, high_mass_slope, low_mass_slope) * (logMvir - 10) + 6.9
    scatter = 0.3 + np.where(logMvir > 10, 0, -0.39 * (logMvir - 10))
    if return_params:
        return mean, scatter
    else:
        if low_scatter:
            scatter=0.01
        return np.random.normal(mean, scatter)

def logMstar_Danieli23(logMvir, const = True, return_params=False, low_scatter=False):
    if const:
        logeps, alpha = -1.5, 1.82
        sigma0, gamma = 0.28, 0
    else:
        logeps, alpha = -2.17, 2.02
        sigma0, gamma = 0.29, -0.45
    logM1 = 12.5
    mean = logeps + logM1 + alpha*logMvir - alpha*logM1
    scatter = sigma0 + np.where(logMvir > logM1, 0, gamma * (logMvir - logM1))
    if return_params:
        return mean, scatter
    else:
        if low_scatter:
            scatter=0.01
        return np.random.normal(mean,scatter)