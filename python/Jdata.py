'''The :mod:`data` module contains the observational data used in the analysis.
The module contains a :mod:`numpy` structured array, :data:`data.raw`, which is
intended to be accessed through the :class:`Dwarf` class. The array contains the 
following information for each dwarf:

 - The distance modulus [1]_,  
 - The half-light radius (in arcminutes) [1]_,  
 - The ellipticity [1]_,  
 - The heliocentric line-of-sight velocity dispersion (in km/s) [1]_,  
 - The base-ten logarithm of the V-band luminosity (in :math:`L_\odot`) [2]_,  
 - The pericentric distance (in kpc) [3]_,  
 - The infall time (in Gyr lookback) [4]_, and  
 - The present-day galactocentric distance (in kpc) [2]_.  

Each of these (save the galactocentric distance) has an associated upper and lower uncertainty, 
corresponding to the 16th and 85th percentiles for the value. 

References
----------
.. [1] Battaglia+22, `2022A&A...657A..54B <https://ui.adsabs.harvard.edu/abs/2022A%26A...657A..54B/abstract>`_
.. [2] Muñoz+18, `2018ApJ...860...66M <https://ui.adsabs.harvard.edu/abs/2018ApJ...860...66M/abstract>`_
.. [3] Pace+22, `2022ApJ...940..136P <https://ui.adsabs.harvard.edu/abs/2022ApJ...940..136P/abstract>`_
.. [4] Fillingham+19, `2019arXiv190604180F <https://ui.adsabs.harvard.edu/abs/2019arXiv190604180F/abstract>`_

'''

import os
import glob
import numpy as np
from astropy import constants as const, units as u, coordinates as crd
from scipy import stats
import scipy.interpolate as interp
from util import quantile

import config

# DwarfJeansAnalysis posterior_samples.npz tree (one timestamped dir per
# dwarf/prior). Replaces the legacy Pace J-Factor-Scaling chains.
DJA_PRODUCTION_DIR = str(config.DJA_RESULTS_DIR)
# DJA_PRIOR = 'jeffreys'
DJA_PRIOR = config.DJA_PRIOR
# Dwarfs whose DJA posteriors should be ignored (e.g. unresolved dispersion).
# DJA_SKIP_KEYS = {'tucana_5'}
DJA_SKIP_KEYS = {}

# set up colossus cosmology to match SatGen
from SatGen import profiles as pf
from colossus.cosmology import cosmology as co
from colossus.halo import mass_so

import pandas as pd

co.addCosmology('SatGen', flat=True, H0=pf.cfg.h*100, Om0=pf.cfg.Om,
                Ob0=pf.cfg.Ob, sigma8=pf.cfg.s8, ns=pf.cfg.ns)
col_co = co.setCosmology('SatGen')

u.kms = u.km/u.s
h = 0.7

import astropy.table as table
# Dwarf galaxy catalog, read from the pinned copy vendored under data/lvdb/.
# Upstream reads these from the release page over HTTPS at import; see
# data/lvdb/VERSION for why that is not acceptable here. To re-pin, bump
# config.LVDB_VERSION and re-fetch -- do not point this back at a URL.
version_number_string = config.LVDB_VERSION
dwarf_all = table.Table.read(config.LVDB_DIR / 'dwarf_all.csv')
gc_ambiguous = table.Table.read(config.LVDB_DIR / 'gc_ambiguous.csv')
dwarf_mw = dwarf_all[dwarf_all['host'] == 'mw']
dwarf_lmc = dwarf_all[dwarf_all['host'] == 'lmc']

dwarf_names = []
for dwarf in dwarf_mw[dwarf_mw['vlos_sigma'] > 0]:
    dwarf_names.append(dwarf['name'])
for dwarf in dwarf_lmc[dwarf_lmc['vlos_sigma'] > 0]:
    dwarf_names.append(dwarf['name'])
dwarf_names.append('Leo V')

dwarf_names = np.array(dwarf_names)

dwarf_names_all = []
dwarf_distances_all = []
for dwarf in dwarf_mw:
    dwarf_names_all.append(dwarf['name'])
    dwarf_distances_all.append(dwarf['distance'])
for dwarf in dwarf_lmc:
    dwarf_names_all.append(dwarf['name'])
    dwarf_distances_all.append(dwarf['distance'])

temp = []
for i in range(len(dwarf_names)):
    table_row = dwarf_all[dwarf_all['name'] == dwarf_names[i]][0]
    temp.append((dwarf_names[i], table_row['ra'], table_row['dec'], 
                table_row['distance_modulus'], table_row['distance_modulus_em'], table_row['distance_modulus_ep'],
                table_row['rhalf'], table_row['rhalf_em'], table_row['rhalf_ep'],
                table_row['ellipticity'], table_row['ellipticity_em'], table_row['ellipticity_ep'],
                table_row['position_angle'], table_row['position_angle_em'], table_row['position_angle_ep'],
                table_row['vlos_systemic'], table_row['vlos_systemic_em'], table_row['vlos_systemic_ep'],
                table_row['vlos_sigma'], table_row['vlos_sigma_em'], table_row['vlos_sigma_ep'],
                table_row['apparent_magnitude_v'], table_row['apparent_magnitude_v_em'], table_row['apparent_magnitude_v_ep']))


temp = np.array(temp, dtype=[('name', '<U32'), ('ra', '<f8'), ('decl', '<f8'), 
               ('dm', '<f8'), ('edm_minus', '<f8'), ('edm_plus', '<f8'),
               ('rh', '<f8'), ('erh_minus', '<f8'), ('erh_plus', '<f8'), 
               ('ellip', '<f8'), ('e_ellip_minus', '<f8'), ('e_ellip_plus', '<f8'), 
               ('PA', '<f8'), ('ePA_minus', '<f8'), ('ePA_plus', '<f8'), 
               ('av_vlos', '<f8'), ('e_av_vlos_minus', '<f8'), ('e_av_vlos_plus', '<f8'), 
               ('sig_v_vlos', '<f8'), ('e_sig_v_vlos_minus', '<f8'), ('e_sig_v_vlos_plus', '<f8'), 
               ('Vmag', '<f8'), ('e_Vmag_minus', '<f8'), ('e_Vmag_plus', '<f8'),])

uma3_row = gc_ambiguous[gc_ambiguous['name'] == 'Ursa Major III'][0]
uma3_info = [('Ursa Major III', uma3_row['ra'], uma3_row['dec'], 
                uma3_row['distance_modulus'], uma3_row['distance_modulus_em'], uma3_row['distance_modulus_ep'],
                uma3_row['rhalf'], uma3_row['rhalf_em'], uma3_row['rhalf_ep'],
                uma3_row['ellipticity'], uma3_row['ellipticity_em'], uma3_row['ellipticity_ep'],
                uma3_row['position_angle'], uma3_row['position_angle_em'], uma3_row['position_angle_ep'],
                uma3_row['vlos_systemic'], uma3_row['vlos_systemic_em'], uma3_row['vlos_systemic_ep'],
                uma3_row['vlos_sigma'], uma3_row['vlos_sigma_em'], uma3_row['vlos_sigma_ep'],
                uma3_row['apparent_magnitude_v'], uma3_row['apparent_magnitude_v_em'], uma3_row['apparent_magnitude_v_ep']),]
uma3_info = np.array(uma3_info, dtype=[('name', '<U32'), ('ra', '<f8'), ('decl', '<f8'), 
               ('dm', '<f8'), ('edm_minus', '<f8'), ('edm_plus', '<f8'),
               ('rh', '<f8'), ('erh_minus', '<f8'), ('erh_plus', '<f8'), 
               ('ellip', '<f8'), ('e_ellip_minus', '<f8'), ('e_ellip_plus', '<f8'), 
               ('PA', '<f8'), ('ePA_minus', '<f8'), ('ePA_plus', '<f8'), 
               ('av_vlos', '<f8'), ('e_av_vlos_minus', '<f8'), ('e_av_vlos_plus', '<f8'), 
               ('sig_v_vlos', '<f8'), ('e_sig_v_vlos_minus', '<f8'), ('e_sig_v_vlos_plus', '<f8'), 
               ('Vmag', '<f8'), ('e_Vmag_minus', '<f8'), ('e_Vmag_plus', '<f8'),])

crater2_idx = np.where(dwarf_names == 'Crater II')[0]
peg4_idx = np.where(dwarf_names == 'Pegasus IV')[0]
smc_idx = np.where(dwarf_names == 'SMC')[0]
lmc_idx = np.where(dwarf_names == 'LMC')[0]

temp['ellip'][crater2_idx] = 0.18
temp['e_ellip_minus'][crater2_idx] = 0.07
temp['e_ellip_plus'][crater2_idx] = 0.08
# temp['Crater II']['PA'] = 135.
# temp['Crater II']['ePA_minus'] = 4.
# temp['Crater II']['ePA_plus'] = 4.

temp['ellip'][peg4_idx] = 0.
temp['e_ellip_minus'][peg4_idx] = 0.
# used 95% upper limit, is this the right thing to do? No!
temp['e_ellip_plus'][peg4_idx] = 0.205

# temp['sig_v_vlos'][uma3_idx] = 1.9
# temp['e_sig_v_vlos_minus'][uma3_idx] = 1.1
# temp['e_sig_v_vlos_plus'][uma3_idx] = 1.4

# temp['sig_v_vlos'][uma3ex_idx] = 3.7
# temp['e_sig_v_vlos_minus'][uma3ex_idx] = 1.0
# temp['e_sig_v_vlos_plus'][uma3ex_idx] = 1.4

lmc_rhalf = 3.392
vrot_median = 68.412 + (71.108 - 68.412)/0.2 * (lmc_rhalf - 3.2)
vrot_low = 68.251 + (70.871 - 68.251)/0.2 * (lmc_rhalf - 3.2)
vrot_high = 68.646 + (71.311 - 68.646)/0.2 * (lmc_rhalf - 3.2)

sigma_vlos_median = np.sqrt(vrot_median**2 / 3)
sigma_vlos_low = np.sqrt(vrot_low**2 / 3)
sigma_vlos_high = np.sqrt(vrot_high**2 / 3)

temp['sig_v_vlos'][lmc_idx] = sigma_vlos_median
temp['e_sig_v_vlos_minus'][lmc_idx] = sigma_vlos_median - sigma_vlos_low
temp['e_sig_v_vlos_plus'][lmc_idx] = sigma_vlos_high - sigma_vlos_median

vrot_median = 37.93
vrot_low = 35.07
vrot_high = 40.72

sigma_vlos_median = np.sqrt(vrot_median**2 / 3)
sigma_vlos_low = np.sqrt(vrot_low**2 / 3)
sigma_vlos_high = np.sqrt(vrot_high**2 / 3)

temp['ellip'][smc_idx] = 0.155
temp['e_ellip_minus'][smc_idx] = 0.006
temp['e_ellip_plus'][smc_idx] = 0.006

temp['sig_v_vlos'][smc_idx] = sigma_vlos_median
temp['e_sig_v_vlos_minus'][smc_idx] = sigma_vlos_median - sigma_vlos_low
temp['e_sig_v_vlos_plus'][smc_idx] = sigma_vlos_high - sigma_vlos_median



# names of dwarfs in Pace & Strigari 2018
ps18_names = [
    None, 'aquarius2', 'bootes1', None, None,
    'canesvenatici1', 'canesvenatici2', 'carina', None,
    'comaberenices', None, 'draco1', 'eridanus2', None,
    'fornax', 'grus1', 'hercules', 'leo1', 'leo2', 'leo4', None,
    None, 'pegasus3', None, 'pisces2', None,
    'sculptor', 'segue1', 'sextans', 'tucana2_des', None, None,
    'ursamajor1', 'ursamajor2', 'ursaminor', 'willman1', 'carina2',
    None, 'horologium1_des', None, 'reticulum2_des', None, 
    None, None
]

# abbreviations for table
abbreviations = np.array([
    'AntII', 'AqrII', r'Bo\"oI', r'Bo\"oII', r'Bo\"oIII',
    'CVnI', 'CVnII', 'Car', 'CenI',
    'CB', 'CraII', 'Dra', 'EriII', 'EriIV', 'For',
    'GruI', 'Her', 'LeoI', 'LeoII', 'LeoIV', 'LeoVI', 'LMC',
    'PegIII', 'PegIV', 'PscII', 'Sgr', 'Scl', 'Seg1',
    'Sex', 'TucII', 'TucIV', 'TucV', 'UMaI',
    'UMaII', 'UMi', 'Wil1', 'CarII', 'CarIII',
    'HorI', 'HyiI', 'RetII', 'SMC', 'LeoV'
])

# names of dwarfs in McDaniel et al. 2023 (Fermi legacy analysis)
fermi_names = np.array([
    'Antlia_2', 'Aquarius_2', 'Bootes_1', 'Bootes_2', 'Bootes_3',
    'Canes_V1', 'Canes_V2', 'Carina', 'Centaurus_1',
    'Berenices', 'Crater_2', 'Draco', 'Eridanus_2', None, 'Fornax',
    'Grus_1', 'Hercules', 'Leo_1', 'Leo_2', 'Leo_4', 'Leo_6', None,
    'Pegasus_3', None, 'Pisces_2', 'Sagittarius', 'Sculptor', 'Segue_1',
    'Sextans', 'Tucana_2', 'Tucana_4', 'Tucana_5', 'Ursa_Major_1',
    'Ursa_Major_2', 'Ursa_Minor', 'Willman_1', 'Carina_2', 'Carina_3',
    'Horologium_1', 'Hydrus_1', 'Reticulum_2', None, 'Leo_5'
])

# same 43 dwarfs, renamed to match data/fermi_legacy_update/dSphs/SEDs/ filenames.
# None where the update SED set has no file at all (Leo_6, Sculptor).
fermi_names_update = np.array([
    'Antlia_II', 'Aquarius_II', 'Bootes_I', 'Bootes_II', 'Bootes_III',
    'Canes_Venatici_I', 'Canes_Venatici_II', 'Carina', 'Centaurus_I',
    'Coma_Berenices', 'Crater_II', 'Draco', 'Eridanus_II', None, 'Fornax',
    'Grus_I', 'Hercules', 'Leo_I', 'Leo_II', 'Leo_IV', None, None,
    'Pegasus_III', None, 'Pisces_II', 'Sagittarius', None, 'Segue_1',
    'Sextans', 'Tucana_II', 'Tucana_IV', 'Tucana_V', 'Ursa_Major_I',
    'Ursa_Major_II', 'Ursa_Minor', 'Willman_1', 'Carina_II', 'Carina_III',
    'Horologium_I', 'Hydrus_I', 'Reticulum_II', None, 'Leo_V'
])

n_stars = np.array([
    '283', '8', '37', '12', '20',
    '214', '25', '774', '34',
    '59', '141', '468', '28', '28', 
    '2483', '8', '30', '328', '196', '20', '9', 
    r'O($10^{7}$)', '7', '7', '7', '323', 
    '1365', '69', '441', '5', '11', '5', 
    '39', '20', '680', '40', '14',
    '4', '5', '30', '25', 'n/a (HI)', 
    '11', '10'
])

kd_table = pd.read_csv(config.DATA_DIR / 'literature' / 'table1_systems_20251205.csv')
kd_mask = (kd_table['Type'] == 'G') & (kd_table['sigma'] > 0)
kd_table = kd_table[kd_mask]

kd_names = np.array(['Bootes I', 'Bootes II', 'Bootes III', 'Coma Berenices', 'Canes Venatici I', 'Canes Venatici II', 
            'Draco', 'Eridanus IV', 'Fornax', 'Hercules', 'Leo I', 'Leo II', 'Leo IV', 
            'Leo V', 'Leo T', 'Pegasus III', 'Pegasus IV', 'Sculptor', 'Segue 1', 
            'Sextans', 'Sagittarius', 'Ursa Major I', 'Ursa Major II', 'Ursa Minor', 'Willman 1'])
kd_dispersion = np.array(kd_table['sigma'])
kd_dispersion_err_lo = np.array(kd_table['serr_low'])
kd_dispersion_err_hi = np.array(kd_table['serr_up'])
kd_dispersion_err = np.transpose([kd_dispersion_err_lo, kd_dispersion_err_hi])

# McDaniel et al. 2023 velocity dispersion table (older compilation)
mc_table = pd.read_csv(config.DATA_DIR / 'literature' / 'mcdaniel2023_dwarf_velocity_dispersions.csv')
mc_table = mc_table[mc_table['sigma_los_km_s'].notna()]
mc_names = np.array(mc_table['lvdb_name'])
mc_dispersion = np.array(mc_table['sigma_los_km_s'])
mc_dispersion_err_lo = np.array(mc_table['sigma_lower_km_s'])
mc_dispersion_err_hi = np.array(mc_table['sigma_upper_km_s'])
mc_dispersion_err = np.transpose([mc_dispersion_err_lo, mc_dispersion_err_hi])


class Dwarf:
    r'''Interprets the raw observational data to allow easy access to probability 
    distributions of known parameters.

    PDFs are constructed internally as two-sided Gaussians to account for asymmetric
    observational uncertainties. For an observational value :math:`\mu_{-\sigma_{\downarrow}}^{+\sigma_{\uparrow}}`,
    the PDF is

    .. math:: math:: f_i(\theta) = \frac{2}{\sigma_\uparrow + \sigma_\downarrow} \times \begin{cases}\operatorname{Norm}(\theta;\;\mu,\,\sigma_\downarrow) & \theta < \mu \\ \operatorname{Norm}(\theta;\;\mu,\,\sigma_\uparrow) & \theta \geq \mu \end{cases}

    with 

    .. math:: \operatorname{Norm}(x;\;\mu,\,\sigma) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left[-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2\right]

    Attributes
    ----------
        name : str
            The full name of the dwarf
        rhalf : float
            The 3D sphericalized, deprojected half-light radius in pc, computed as per [1]_.
        logMhalf : float
            The base-ten logarithm of the mass in :math:`M_\odot` within :attr:`rhalf`, as determined by the Wolf+16 estimator [2]_.
        logMstar : float
            The base-ten logarithm of the stellar mass in :math:`M_\odot`, computed via a mass-to-light ratio.

    Notes
    -----
    For each physical quantity (:attr:`rhalf`, :attr:`logMhalf`, etc.), there is also an associated ``<quantity>_err`` 
    attribute containing the lower and upper error bars, both as non-negative floats. These are used to compute the two-sided Gaussian PDFs.
    
    References
    ----------
    .. [1] Sanders-Evans16 `2016ApJ...830L..26S <https://ui.adsabs.harvard.edu/abs/2016ApJ...830L..26S>`_
    .. [2] Wolf+16 `2010MNRAS.406.1220W <https://ui.adsabs.harvard.edu/abs/2010MNRAS.406.1220W>`_
    '''
    def __init__(self, name, ML_ratio = 1.2, ML_error = 0, logMstar_scatter = 0.16, logMhalf_scatter = 0.1, logMdyn_errani_scatter = 0.043, use_kd = True, use_mcdaniel = False):
        '''Constructs the :class:`Dwarf` object from the :data:`Dwarf.raw` data. 

        In particular, it converts the angular on-sky information into a physical size :math:`r_{1/2}`, which is used in the Wolf+16 estimator 
        (along with the velocity dispersion) to determine the :math:`M_{1/2}` of the dwarf. Further, it converts the V-band luminosity into
        a stellar mass through the provided mass-to-light ratio. Uncertainties are propagated throughout. 

        Parameters
        ----------
        name : str
            The name of the dwarf
        ML_ratio : float, optional
            The mass-to-light ratio to assume for this dwarf. The default is 1.2.
        ML_error : float, optional
            An uncertainty on the mass-to-light ratio, propagated through to the stellar mass. The default is zero.
        logMstar_scatter : float, optional
            An additional uncertainty (in dex) to add to the stellar mass. Similar to :attr:`ML_error`, but added directly instead of being propagated. The default is 0.16.
        '''
        if name == 'Ursa Major III':
            info = uma3_info[0]
            self.idx = 0
        else:
            info = temp[temp['name'] == name][0]
            self.idx = np.where(temp['name'] == name)[0][0]
        # old_info = raw[dwarf_names == name][0]
        
        self.info = info
        
        if name in kd_names and use_kd:
            use_kd = True
        else:
            use_kd = False
        if use_kd:
            self.kd_idx = np.where(np.array(kd_names) == name)[0]
        if name in mc_names and use_mcdaniel:
            use_mcdaniel = True
            use_kd = False
            self.mc_idx = np.where(mc_names == name)[0]
        else:
            use_mcdaniel = False
        self.name = name
        
        # load / compute on-sky information
        #----------------------------------
        dist = 10**(info['dm']/5 + 1) * u.pc
        d_lo, d_hi = dist * info['edm_minus']/5* np.log(10), dist * info['edm_plus']/5* np.log(10)
        self.distance = dist
        self.distance_err = np.array([d_lo.to(dist.unit).value, d_hi.to(dist.unit).value]) * dist.unit 
        self.coords = crd.SkyCoord(ra = info['ra'] * u.deg, dec = info['decl'] * u.deg, distance=dist)
        self.gc_coords = self.coords.transform_to(crd.Galactocentric())
        self.gal_coords = self.coords.transform_to(crd.Galactic())
        self.Omega = np.sqrt(self.gal_coords.b.to(u.rad)**2 + self.gal_coords.l.to(u.rad)**2)
        self.gc_distance = np.sqrt(self.gc_coords.x**2 + self.gc_coords.y**2 + self.gc_coords.z**2)
        d_sun = crd.Galactocentric().galcen_distance
        self.gc_distance_err = self.distance_err * np.sqrt(self.distance**2 + 4 * d_sun**2 * np.cos(self.Omega)**2) / self.gc_distance
        self.size = (info['rh'] * u.arcmin).to(u.deg)
        self.size_err = np.array([(info['erh_minus'] * u.arcmin).to(self.size.unit).value, (info['erh_plus'] * u.arcmin).to(self.size.unit).value]) * self.size.unit
        self.ellipticity = info['ellip']
        self.ellipticity_err = np.array([info['e_ellip_minus'], info['e_ellip_plus']])
        # self.position_angle = info['PA'] * u.deg
        # self.position_angle_err = np.array([info['ePA_minus'], info['ePA_plus']]) * u.deg
        # get pericenter cut: LMC pericenter values where present, else galactocentric distance
        # note to self: LMC_minus, _plus are actual quantile values, other models are the error bars
        # self.peri_lim = (old_info['peri_LMC_minus'], old_info['peri_LMC_plus']) if old_info['peri_LMC'] < 999 \
        #                     else tuple((self.distance + [-1, 1] * self.distance_err).to(u.kpc).value)
        # velocity info
        #--------------
        self.v_los = info['av_vlos'] * u.kms
        self.v_los_err = np.array([info['e_av_vlos_minus'], info['e_av_vlos_plus']]) * u.kms
        self.dispersion = info['sig_v_vlos'] * u.kms
        self.dispersion_err = np.array([info['e_sig_v_vlos_minus'], info['e_sig_v_vlos_plus']]) * u.kms
        if use_kd:
            self.dispersion = kd_dispersion[self.kd_idx][0] * u.kms
            self.dispersion_err = kd_dispersion_err[self.kd_idx][0] * u.kms
        if use_mcdaniel:
            self.dispersion = mc_dispersion[self.mc_idx][0] * u.kms
            self.dispersion_err = mc_dispersion_err[self.mc_idx][0] * u.kms
        self.Vcirc = np.sqrt(3) * self.dispersion
        self.Vcirc_err = np.sqrt(3) * self.dispersion_err

        # half-light radius and mass within
        #----------------------------------
        self.major_axis = (self.size * dist).to(u.pc, u.dimensionless_angles())
        self.majax_err = self.major_axis * np.array([np.hypot(d_lo/dist, self.size_err[0]/self.size), 
                                           np.hypot(d_hi/dist, self.size_err[1]/self.size)])
        # THIS IS THE 2D RHALF -- the 3D deprojected r1/2 is 4/3 Re
        self.Re = self.major_axis * np.sqrt(1 - self.ellipticity) # sphericalize as per Sanders & Evans (2016)
        self.Re_err = self.Re * np.array([np.hypot(self.majax_err[0]/self.major_axis, info['e_ellip_minus']/(2 - 2 * self.ellipticity)), 
                                np.hypot(self.majax_err[1]/self.major_axis, info['e_ellip_plus'] /(2 - 2 * self.ellipticity))])
        self.rhalf = self.Re * 4/3
        self.rhalf_err = self.Re_err * 4/3
        # use the Wolf et al. (2010) estimator
        self.Mhalf = (4 * self.dispersion**2 * self.Re / const.G).to(u.Msun).value
        self.Mhalf_err = self.Mhalf * np.sqrt((2 * self.dispersion_err / self.dispersion)**2 + (self.Re_err / self.Re)**2 + (logMhalf_scatter * np.log(10))**2).to(u.dimensionless_unscaled).value
        self.logMhalf = np.log10(self.Mhalf)
        self.logMhalf_err = self.Mhalf_err / (self.Mhalf * np.log(10))

        self.Mdyn_errani = (3.5 * 1.8 * self.dispersion**2 * self.Re / const.G).to(u.Msun).value
        self.Mdyn_errani_err = self.Mdyn_errani * np.sqrt((2 * self.dispersion_err / self.dispersion)**2 + (self.Re_err / self.Re)**2 + (logMdyn_errani_scatter * np.log(10))**2).to(u.dimensionless_unscaled).value
        self.logMdyn_errani = np.log10(self.Mdyn_errani)
        self.logMdyn_errani_err = self.Mdyn_errani_err / (self.Mdyn_errani * np.log(10))

        self.rho_rhalf = self.Mhalf / (4 / 3 * np.pi * self.rhalf.to(u.kpc).value**3)
        self.rho_rhalf_err = self.rho_rhalf * np.sqrt((self.Mhalf_err / self.Mhalf)**2 + (3 * self.rhalf_err.value / self.rhalf.value)**2)
        self.logrho_rhalf = np.log10(self.rho_rhalf)
        self.logrho_rhalf_err = self.rho_rhalf_err / (self.rho_rhalf * np.log(10))
        
        # stellar mass
        #-------------
        self.logLv = (4.83 - (info['Vmag'] - info['dm']))/2.5
        # upper bound on Mv -> lower bound on Lv
        self.logLv_err = (np.array([info['e_Vmag_plus'] + info['edm_minus'], 
                               info['e_Vmag_minus'] + info['edm_plus']]))/2.5

        self.Lv = 10**self.logLv
        # self.Lv_err = self.Lv * np.log(10) * self.logLv_err
        self.Lv_err = np.array([self.Lv - 10**(self.logLv - self.logLv_err[0]), 10**(self.logLv + self.logLv_err[1]) - self.Lv])

        self.logMstar = self.logLv + np.log10(ML_ratio)
        self.logMstar_err = self.logLv_err + ML_error / (ML_ratio * np.log(10)) + logMstar_scatter

        # self.get_Jeans_results()
        self.J_disp = J_scaling_dispersion(self.dispersion, self.distance.to(u.kpc), self.Re)
        self.J_disp_err = self.J_disp * np.sqrt((4 * self.dispersion_err / self.dispersion)**2 + (-2 * self.distance_err / self.distance)**2 + (-1 * self.Re_err / self.Re)**2).to(u.dimensionless_unscaled).value
        self.logJ_disp = np.log10(self.J_disp)
        self.logJ_disp_err = self.J_disp_err / (self.J_disp * np.log(10))
        
        self.J_lum = J_scaling_luminosity(self.Lv, self.distance.to(u.kpc), self.Re)
        self.J_lum_err = self.J_lum * np.sqrt((0.23 * self.Lv_err / self.Lv)**2 + (-2 * self.distance_err / self.distance)**2 + (-0.5 * self.Re_err / self.Re)**2).to(u.dimensionless_unscaled).value
        self.logJ_lum = np.log10(self.J_lum)
        self.logJ_lum_err = self.J_lum_err / (self.J_lum * np.log(10))
    
    @staticmethod
    def _twosided(x, mean, lowerr, higherr):
        scale = 2/(lowerr + higherr)
        lowgauss = scale * stats.norm.pdf((x-mean)/lowerr)
        highgauss = scale * stats.norm.pdf((x-mean)/higherr)
        return np.where(x < mean, lowgauss, highgauss)
  
    def logMstar_pdf(self, logMstar):
        '''Evaluates the PDF for :math:`f_i(\log_{10}M_\star)` at the value :attr:`logMstar`

        Parameters
        ----------
        logMstar : float
            the value at which to evaluate the PDF.
        '''
        return self._twosided(logMstar, self.logMstar, *self.logMstar_err)

    def logMhalf_pdf(self, logMhalf):
        '''Evaluates the PDF for :math:`f_i(\log_{10}M_{1/2})` at the value :attr:`logMhalf`

        Parameters
        ----------
        logMhalf : float
            the value at which to evaluate the PDF.
        '''
        return self._twosided(logMhalf, self.logMhalf, *self.logMhalf_err)

    def logMdyn_errani_pdf(self, logMdyn_errani):
        '''Evaluates the PDF for :math:`f_i(\log_{10}M_{dyn,Errani})` at the value :attr:`logMdyn_errani`

        Parameters
        ----------
        logMdyn_errani : float
            the value at which to evaluate the PDF.
        '''
        return self._twosided(logMdyn_errani, self.logMdyn_errani, *self.logMdyn_errani_err)

    def get_Jeans_results(self, dja_prior = DJA_PRIOR, vmax_min = 1.0):
        # Look up the LVDB key for this dwarf and load its DwarfJeansAnalysis
        # derived.npz under the chosen prior (flat layout, no timestamps). The
        # derived export (postprocess.save_derived) carries the full per-draw
        # NFW + J/D + theta95 arrays; the chain-only posterior_samples.npz does not.
        match = dwarf_all[dwarf_all['name'] == self.name]
        lvdb_key = match['key'][0] if len(match) > 0 else None
        ps_path = None
        if lvdb_key is not None and lvdb_key not in DJA_SKIP_KEYS:
            candidate = os.path.join(DJA_PRODUCTION_DIR, lvdb_key, dja_prior,
                                     'derived.npz')
            if os.path.isfile(candidate):
                ps_path = candidate
        # The DwarfJeansAnalysis file actually read for this prior, so a
        # caller stamping the result can fingerprint the real dependency
        # (derived.npz under DJA_RESULTS_DIR) rather than the unrelated
        # SatGen catalog/weights this method never touches.
        self.jeans_dja_path = ps_path
        if ps_path is not None:
            dwarf_file = np.load(ps_path)
            # NFW chain: equal-weighted samples, uniform weights.
            self.jeans_rs = 10 ** dwarf_file['log10_rs']  # kpc
            self.jeans_rhos = 10 ** dwarf_file['log10_rhos']  # Msun/kpc^3

            to_betatilde = lambda b: b / (2 - b)

            self.jeans_beta_tilde = to_betatilde(np.asarray(dwarf_file['beta']))
            self.jeans_weights = np.ones_like(self.jeans_rs)
            # log10_J samples are independently thinned (different length than
            # the NFW chain) -- store with their own uniform weights.
            self.jeans_J = 10 ** np.asarray(dwarf_file['log10_J_0p5deg'])  # GeV^2/cm^5
            self.jeans_J_weights = np.ones_like(self.jeans_J)

            # theta95(J) shares the thinned J/D grid (same length as jeans_J),
            # so it carries jeans_J_weights, not the full-chain jeans_weights.
            self.jeans_theta95 = np.asarray(dwarf_file['theta95_J_deg'])  # deg

            # Vectorized closed-form NFW (replaces per-sample colossus calls).
            # Agrees with the prior per-sample colossus loop to <=1e-10 rel on
            # rmax/vmax/rho150/Mhalf/cV and <=2e-7 rel on Mvir (bisection
            # solver tolerance); both far below chain MC noise.
            G_KPC = 4.30091727e-6  # (km/s)^2 kpc / Msun
            XMAX = 2.16258         # argmax of mu(x)/x
            rs = self.jeans_rs
            rhos = self.jeans_rhos
            def _mu(x):
                return np.log1p(x) - x / (1.0 + x)
            self.jeans_rmax = XMAX * rs
            self.jeans_vmax = np.sqrt(4*np.pi * G_KPC * rhos * rs**2 * _mu(XMAX) / XMAX)
            x150 = 0.15 / rs
            self.jeans_rho150 = rhos / (x150 * (1.0 + x150)**2)
            xh = self.rhalf.to(u.kpc).value / rs
            self.jeans_Mhalf = 4*np.pi * rhos * rs**3 * _mu(xh)
            # Mvir: solve mu(xv)/xv^3 = Delta_vir * rho_c / (3 rhos)
            Dvir = mass_so.deltaVir(0.0)
            rho_c_phys = col_co.rho_c(0.0) * col_co.h**2  # Msun/kpc^3
            rhs = Dvir * rho_c_phys / (3.0 * rhos)
            lo = np.full_like(rhs, 1e-4)
            hi = np.full_like(rhs, 1e5)
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                f = _mu(mid) / mid**3 - rhs
                hi = np.where(f < 0, mid, hi)
                lo = np.where(f < 0, lo, mid)
            xv = 0.5 * (lo + hi)
            self.jeans_Mvir = 4*np.pi * rhos * rs**3 * _mu(xv)
            self.jeans_cV = (2 * (self.jeans_vmax * u.km / u.s / (self.jeans_rmax * u.kpc * (col_co.H0 * u.km / u.s / u.Mpc).to(1 / u.s))) ** 2).to(u.dimensionless_unscaled).value

            # Restrict the posterior to draws with vmax > vmax_min (km/s). The
            # J/D grid is a thinned subset of the chain; idx_jd maps each J/D
            # draw to its parent chain draw, so the cut is propagated onto the J
            # arrays via the full (unfiltered) chain vmax indexed by idx_jd.
            idx_jd = np.asarray(dwarf_file['idx_jd'])
            vmax_full = self.jeans_vmax.copy()  # snapshot before filtering
            chain_mask = vmax_full > vmax_min
            jd_mask = vmax_full[idx_jd] > vmax_min
            self.jeans_rs = self.jeans_rs[chain_mask]
            self.jeans_rhos = self.jeans_rhos[chain_mask]
            self.jeans_beta_tilde = self.jeans_beta_tilde[chain_mask]
            self.jeans_weights = self.jeans_weights[chain_mask]
            self.jeans_rmax = self.jeans_rmax[chain_mask]
            self.jeans_vmax = self.jeans_vmax[chain_mask]
            self.jeans_rho150 = self.jeans_rho150[chain_mask]
            self.jeans_Mhalf = self.jeans_Mhalf[chain_mask]
            self.jeans_Mvir = self.jeans_Mvir[chain_mask]
            self.jeans_cV = self.jeans_cV[chain_mask]
            self.jeans_J = self.jeans_J[jd_mask]
            self.jeans_theta95 = self.jeans_theta95[jd_mask]
            self.jeans_J_weights = self.jeans_J_weights[jd_mask]

            self.Jeans_J_quantiles = quantile(self.jeans_J, self.jeans_J_weights, quantiles=np.array([0.16, 0.5, 0.84]))
            self.Jeans_rho150_quantiles = quantile(self.jeans_rho150, self.jeans_weights, quantiles=np.array([0.16, 0.5, 0.84]))
            self.Jeans_cV_quantiles = quantile(self.jeans_cV, self.jeans_weights, quantiles=np.array([0.16, 0.5, 0.84]))
            self.Jeans_Mvir_quantiles = quantile(self.jeans_Mvir, self.jeans_weights, quantiles=np.array([0.16, 0.5, 0.84]))
            self.Jeans_beta_tilde_quantiles = quantile(self.jeans_beta_tilde, self.jeans_weights, quantiles=np.array([0.16, 0.5, 0.84]))
            self.Jeans_theta95_quantiles = quantile(self.jeans_theta95, self.jeans_J_weights, quantiles=np.array([0.16, 0.5, 0.84]))
        else:
            self.Jeans_J_quantiles = np.full(3, np.nan)
            self.Jeans_rho150_quantiles = np.full(3, np.nan)
            self.Jeans_cV_quantiles = np.full(3, np.nan)
            self.Jeans_Mvir_quantiles = np.full(3, np.nan)
            self.Jeans_beta_tilde_quantiles = np.full(3, np.nan)
            self.Jeans_theta95_quantiles = np.full(3, np.nan)
            self.jeans_weights = np.array([])
            self.jeans_J_weights = np.array([])
            self.jeans_rs = np.array([])
            self.jeans_rhos = np.array([])
            self.jeans_J = np.array([])
            self.jeans_logJ = np.array([])
            self.jeans_rho150 = np.array([])
            self.jeans_Mhalf = np.array([])
            self.jeans_Mvir = np.array([])
            self.jeans_vmax = np.array([])
            self.jeans_rmax = np.array([])
            self.jeans_beta_tilde = np.array([])
            self.jeans_theta95 = np.array([])
            self.jeans_cV = np.array([])
        return

    def __repr__(self):
        return f'{self.name}:'+ (16 - len(self.name)) * ' '+f'logMstar = {self.logMstar:0.2f} ± {self.logMstar_err.mean():0.2f} dex, logMhalf = {self.logMhalf:0.2f} ± {self.logMhalf_err.mean():0.2f} dex @ r1/2 = {self.rhalf.to(u.pc).value:0.0f} pc'





class CharacteristicDensity:
    # r grid is linearly spaced in log(r) between these limits
    interpolator_limits = (-3, 1.6)
    interpolator_npoints = 50
    
    def __init__(self, semianalytic_data, weights, mask, dwarf_name, interpolate = True, slope = None):
        self.dw = dwarf_name
        self.wgt = weights
        self.mask = mask
        self.sim = semianalytic_data
        self.slope = slope
        if interpolate:
            self.has_interpolation = True
            self.setup_interpolator()
        else:
            self.has_interpolation = False
            
    @staticmethod
    def quantile1D(values, weights, quantiles=0.5):
        # https://stackoverflow.com/a/73905572
        m = ~np.isnan(values)
        if not np.any(m):
            return np.full(np.shape(quantiles), np.nan)
        v = values[m]
        wgt = weights[m]
        i = np.argsort(v)
        c = np.cumsum(wgt[i])
        q = np.clip(np.searchsorted(c, quantiles * c[-1]), 0, len(c) - 2)
        return np.where(c[q]/c[-1] == quantiles, 0.5 * (v[i[q]] + v[i[q+1]]), v[i[q]])

    @staticmethod
    def quantile(values, weights, quantiles=0.5):
        if len(np.shape(values)) == 1:
            return CharacteristicDensity.quantile1D(values, weights, quantiles)
        else:
            return np.array([CharacteristicDensity.quantile1D(v, weights, quantiles) for v in values])
    
    def setup_interpolator(self):
        interp_xaxis = np.linspace(*self.interpolator_limits, self.interpolator_npoints)
        interp_yaxis = np.log10(self.at(10**interp_xaxis, nsigma_from_mean = 0))
        self.interp_logrhologr = interp.Akima1DInterpolator(interp_xaxis, interp_yaxis)
        # interp.RGI expects R^n domain; doesn't like the 1D shape of interp_xaxis 
        self.extrapolate_interp = interp.RegularGridInterpolator(interp_xaxis.reshape((1,) + interp_xaxis.shape), interp_yaxis, 
                                                                 method = 'linear', bounds_error = False, fill_value = None)
    
    def interp(self, r):
        if not self.has_interpolation:
            raise NotImplementedError('interpolated density profile not defined for this object')
        logr = np.log10(r)
        within_interp_bounds = (self.interpolator_limits[0] < logr) & (logr < self.interpolator_limits[1])
        logrho = np.where(within_interp_bounds, self.interp_logrhologr(logr), self.extrapolate_interp(logr))
        return 10**logrho
    
    def at(self, r, nsigma_from_mean = 0):
        # take advantage of SatGen vectorization by doing rho(r) even when r has len > 1
        satgen_densities = np.array([p.rho(r) for p in np.asarray(self.sim.profiles)[self.mask]]).T 
        quantile = stats.norm.cdf(nsigma_from_mean)
        return self.quantile(satgen_densities, self.wgt, quantile)
    
    def __call__(self, r):
        if self.has_interpolation:
            return self.interp(r)
        else:
            return self.at(r, nsigma_from_mean = 0)


# Corresponding log10J and error
mcdaniel_Jeans_log10J_values = np.array([
    [16.50, 0.60],
    [17.80, 0.55],
    [18.17, 0.30],
    [18.30, 0.95],
    [18.65, 0.60],
    [17.42, 0.16],
    [17.82, 0.47],
    [17.83, 0.10],
    [18.14, 0.60],
    [19.00, 0.35],
    [15.35, 0.26],
    [18.83, 0.12],
    [16.60, 0.90],
    [np.nan, np.nan],
    [18.09, 0.10],
    [16.50, 0.80],
    [17.37, 0.53],
    [17.64, 0.13],
    [17.76, 0.20],
    [16.40, 1.08],
    [np.nan, np.nan],
    [np.nan, np.nan],
    [18.30, 0.93],
    [np.nan, np.nan],
    [17.30, 1.04],
    [19.60, 0.20],
    [18.58, 0.05],
    [19.12, 0.53],
    [17.73, 0.12],
    [18.97, 0.54],
    [18.40, 0.55],
    [18.90, 0.60],
    [18.26, 0.28],
    [19.44, 0.40],
    [18.75, 0.12],
    [19.53, 0.50],
    [18.25, 0.55],
    [19.70, 0.60],
    [19.00, 0.81],
    [18.33, 0.36],
    [18.90, 0.38],
    [np.nan, np.nan], 
    [17.65, 0.97],
])

gs_log10J_values = np.array([
    [np.nan, np.nan, np.nan],  # Antlia II
    [np.nan, np.nan, np.nan],  # Aquarius II
    [18.24, 0.40, 0.37],       # Bootes I
    [np.nan, np.nan, np.nan],  # Bootes II
    [np.nan, np.nan, np.nan],  # Bootes III
    [17.44, 0.37, 0.28],       # Canes Venatici I
    [17.65, 0.45, 0.43],       # Canes Venatici II
    [17.92, 0.19, 0.11],       # Carina
    [np.nan, np.nan, np.nan],  # Centaurus I
    [19.02, 0.37, 0.41],       # Coma Berenices
    [np.nan, np.nan, np.nan],  # Crater II
    [19.05, 0.22, 0.21],       # Draco
    [np.nan, np.nan, np.nan],  # Eridanus II
    [np.nan, np.nan, np.nan],  # Eridanus IV
    [17.84, 0.11, 0.06],       # Fornax
    [np.nan, np.nan, np.nan],  # Grus I
    [16.86, 0.74, 0.68],       # Hercules
    [17.84, 0.20, 0.16],       # Leo I
    [17.97, 0.20, 0.18],       # Leo II
    [16.32, 1.06, 1.70],       # Leo IV
    [np.nan, np.nan, np.nan],  # Leo VI
    [np.nan, np.nan, np.nan],  # LMC
    [np.nan, np.nan, np.nan],  # Pegasus III
    [np.nan, np.nan, np.nan],  # Pegasus IV
    [np.nan, np.nan, np.nan],  # Pisces II
    [np.nan, np.nan, np.nan],  # Sagittarius
    [18.57, 0.07, 0.05],       # Sculptor
    [19.36, 0.32, 0.35],       # Segue 1
    [17.92, 0.35, 0.29],       # Sextans
    [np.nan, np.nan, np.nan],  # Tucana II
    [np.nan, np.nan, np.nan],  # Tucana IV
    [np.nan, np.nan, np.nan],  # Tucana V
    [17.87, 0.56, 0.33],       # Ursa Major I
    [19.42, 0.44, 0.42],       # Ursa Major II
    [18.95, 0.26, 0.18],       # Ursa Minor
    [np.nan, np.nan, np.nan],  # Willman 1
    [np.nan, np.nan, np.nan],  # Carina II
    [np.nan, np.nan, np.nan],  # Carina III
    [np.nan, np.nan, np.nan],  # Horologium I
    [np.nan, np.nan, np.nan],  # Hydrus I
    [np.nan, np.nan, np.nan],  # Reticulum II
    [np.nan, np.nan, np.nan],  # SMC
    [16.37, 0.94, 0.87],       # Leo V
])

bonnivard_log10J_values = np.array([
    [np.nan, np.nan, np.nan],  # Antlia II
    [np.nan, np.nan, np.nan],  # Aquarius II
    [18.85, 1.10, 0.61],       # Bootes I
    [np.nan, np.nan, np.nan],  # Bootes II
    [np.nan, np.nan, np.nan],  # Bootes III
    [17.63, 0.50, 0.20],       # Canes Venatici I
    [18.67, 1.54, 0.97],       # Canes Venatici II
    [18.02, 0.36, 0.15],       # Carina
    [np.nan, np.nan, np.nan],  # Centaurus I
    [20.13, 1.56, 1.08],       # Coma Berenices
    [np.nan, np.nan, np.nan],  # Crater II
    [19.42, 0.92, 0.47],       # Draco
    [np.nan, np.nan, np.nan],  # Eridanus II
    [np.nan, np.nan, np.nan],  # Eridanus IV
    [17.85, 0.11, 0.08],       # Fornax
    [np.nan, np.nan, np.nan],  # Grus I
    [17.70, 1.08, 0.73],       # Hercules
    [17.93, 0.65, 0.25],       # Leo I
    [18.11, 0.71, 0.25],       # Leo II
    [16.36, 1.44, 1.65],       # Leo IV
    [np.nan, np.nan, np.nan],  # Leo VI
    [np.nan, np.nan, np.nan],  # LMC
    [np.nan, np.nan, np.nan],  # Pegasus III
    [np.nan, np.nan, np.nan],  # Pegasus IV
    [np.nan, np.nan, np.nan],  # Pisces II
    [np.nan, np.nan, np.nan],  # Sagittarius
    [18.63, 0.14, 0.08],       # Sculptor
    [17.52, 2.54, 2.65],       # Segue 1
    [18.04, 0.50, 0.28],       # Sextans
    [np.nan, np.nan, np.nan],  # Tucana II
    [np.nan, np.nan, np.nan],  # Tucana IV
    [np.nan, np.nan, np.nan],  # Tucana V
    [18.84, 0.97, 0.43],       # Ursa Major I
    [20.60, 1.46, 0.95],       # Ursa Major II
    [19.08, 0.21, 0.13],       # Ursa Minor
    [np.nan, np.nan, np.nan],  # Willman 1
    [np.nan, np.nan, np.nan],  # Carina II
    [np.nan, np.nan, np.nan],  # Carina III
    [np.nan, np.nan, np.nan],  # Horologium I
    [np.nan, np.nan, np.nan],  # Hydrus I
    [np.nan, np.nan, np.nan],  # Reticulum II
    [np.nan, np.nan, np.nan],  # SMC
    [16.30, 1.33, 1.16],       # Leo V
])

# Create structured numpy array
mcdaniel_Jeans_J_array = np.zeros(len(dwarf_names), dtype=[('name', 'U30'), ('log10J', 'f8'), ('log10J_err', 'f8')])

for i in range(len(mcdaniel_Jeans_J_array)):
    mcdaniel_Jeans_J_array['name'][i] = dwarf_names[i]
    mcdaniel_Jeans_J_array['log10J'][i] = mcdaniel_Jeans_log10J_values[i,0]
    mcdaniel_Jeans_J_array['log10J_err'][i] = mcdaniel_Jeans_log10J_values[i,1]

# GS set structured array
gs_J_array = np.zeros(len(dwarf_names), dtype=[('name', 'U30'),
                                               ('log10J', 'f8'),
                                               ('log10J_err_plus', 'f8'),
                                               ('log10J_err_minus', 'f8')])
for i in range(len(gs_J_array)):
    gs_J_array['name'][i] = dwarf_names[i]
    gs_J_array['log10J'][i] = gs_log10J_values[i, 0]
    gs_J_array['log10J_err_plus'][i] = gs_log10J_values[i, 1]
    gs_J_array['log10J_err_minus'][i] = gs_log10J_values[i, 2]

# Bonnivard set structured array
bonnivard_J_array = np.zeros(len(dwarf_names), dtype=[('name', 'U30'),
                                                      ('log10J', 'f8'),
                                                      ('log10J_err_plus', 'f8'),
                                                      ('log10J_err_minus', 'f8')])
for i in range(len(bonnivard_J_array)):
    bonnivard_J_array['name'][i] = dwarf_names[i]
    bonnivard_J_array['log10J'][i] = bonnivard_log10J_values[i, 0]
    bonnivard_J_array['log10J_err_plus'][i] = bonnivard_log10J_values[i, 1]
    bonnivard_J_array['log10J_err_minus'][i] = bonnivard_log10J_values[i, 2]

def J_scaling_dispersion(dispersion, distance, Re):
    ''' Parameters
        ----------
        dispersion : float (km/s)
            Velocity dispersion of dwarf galaxy
        distance : float (kpc)
            Distance to dwarf galaxy
        Re : float (pc)
            Azimuthally averaged half-light radius of dwarf galaxy (3/4*r_{1/2})
    '''
    return (10**17.87 * (dispersion / (5 * u.kms))**4 * (distance / (100 * u.kpc))**(-2) * (Re / (100 * u.pc))**(-1)).value

def J_scaling_luminosity(luminosity, distance, Re):
    ''' Parameters
        ----------
        luminosity : float (L_sun)
            V-band luminosity of dwarf galaxy
        distance : float (kpc)
            Distance to dwarf galaxy
        Re : float (pc)
            Azimuthally averaged half-light radius of dwarf galaxy (3/4*r_{1/2})
    '''
    return (10**18.17 * (luminosity / 1e4)**0.23 * (distance / (100 * u.kpc))**(-2) * (Re / (100 * u.pc))**(-0.5)).value



