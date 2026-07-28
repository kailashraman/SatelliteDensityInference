import os,sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
import copy
from functools import lru_cache
from scipy import interpolate
from iminuit import Minuit
from scipy.optimize import minimize, minimize_scalar, fmin

from scipy.interpolate import RegularGridInterpolator

import config

# Paths into the extracted Fermi-LAT legacy data release, resolved from
# config so they no longer depend on the working directory.
#
# dSphs_csv_path was '../fermi_legacy/' upstream -- one level short of its
# siblings, which all carry the 'data/' segment. No code path read it, so the
# error was invisible; see tests/test_fermi_paths.py.
_LEGACY = config.DATA_DIR / 'fermi_legacy'
_LEGACY_UPDATE = config.DATA_DIR / 'fermi_legacy_update'

dSphs_csv_path = str(_LEGACY) + '/'
dSphs_sed_path = str(_LEGACY / 'dSphs' / 'SEDs') + '/'
dSphs_sed_path_update = str(_LEGACY_UPDATE / 'dSphs' / 'SEDs') + '/'
dSphs_TS_prof_path = str(_LEGACY / 'dSphs' / 'TS_profiles') + '/'

blank_field_csv_path = str(_LEGACY) + '/'
blank_TS_prof_path = str(_LEGACY / 'blank_fields' / 'TS_profiles') + '/'

cirelli_dir = str(_LEGACY / 'PPPC4DM') + '/'

def setplot():
    '''Defines standard plot parameters'''
    
    fig = plt.figure(figsize=(3.5, 3), dpi=150)
    ax = plt.gca()
    ax.tick_params(direction='in', which='both', top=True, right=True)
    return ax, fig

def get_ylims(sed):
    ''' Automatically sets flux range for the sed likelihood plots'''
    
    
    fmin = np.log10(np.nanmin(sed['e2dnde_ul'])) - 0.5
    fmax = np.log10(np.nanmax(sed['e2dnde_ul'])) + 0.5
    fdelta = fmax - fmin
    if fdelta < 2.0:
        fmin -= 0.5 * (2.0 - fdelta)
        fmax += 0.5 * (2.0 - fdelta)

    return fmin, fmax

def plot_lnlscan(sed, **kwargs):
    '''Make plot of the SED likelihood profile (very similar to fermipy SEDPlotter method)'''
    
    ax = kwargs.pop('ax', plt.gca())
    llhcut = kwargs.pop('llhcut', -2.70)
    cmap = kwargs.pop('cmap', 'BuGn')
    cmap_trunc_lo = kwargs.pop('cmap_trunc_lo', None)
    cmap_trunc_hi = kwargs.pop('cmap_trunc_hi', None)

    ylim = kwargs.pop('ylim', None)

    if ylim is None:
        fmin, fmax = get_ylims(sed)
    else:
        fmin, fmax = np.log10(ylim)

    fluxEdges = np.arange(fmin, fmax, 0.01)
    fluxCenters = 0.5*(fluxEdges[1:] + fluxEdges[:-1])
    fbins = len(fluxCenters)
    llhMatrix = np.zeros((len(sed['e_ref']), fbins))

    # loop over energy bins
    for i in range(len(sed['e_ref'])):
        m = sed['norm_scan'][i] > 0
        e2dnde_scan = sed['norm_scan'][i][m] * sed['ref_dnde'][i]*pow(sed['e_ref'][i],2)
        flux = np.log10(e2dnde_scan)
        logl = sed['dloglike_scan'][i][m]
        logl -= np.max(logl)
        try:
            fn = interpolate.interp1d(flux, logl, fill_value='extrapolate')
            logli = fn(fluxCenters)
        except:
            logli = np.interp(fluxCenters, flux, logl)
        llhMatrix[i, :] = logli

    cmap = copy.deepcopy(plt.cm.get_cmap(cmap))
    # cmap.set_under('w')

    if cmap_trunc_lo is not None or cmap_trunc_hi is not None:
        cmap = truncate_colormap(cmap, cmap_trunc_lo, cmap_trunc_hi, 1024)

    xedge = np.insert(sed['e_max'], 0, sed['e_min'][0])
    yedge = 10**fluxEdges
    xedge, yedge = np.meshgrid(xedge, yedge)
    im = ax.pcolormesh(xedge, yedge, llhMatrix.T,
                       vmin=llhcut, vmax=0, cmap=cmap,
                       linewidth=0, shading='auto') 
    cb = plt.colorbar(im)
    cb.set_label('Delta LogLikelihood')

    plt.gca().set_ylim(10 ** fmin, 10 ** fmax)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_xlim(sed['e_min'][0], sed['e_max'][-1])
    plt.gca().set_ylabel(r'$E^2dN/dE$ [MeV cm$^{-2}$ s$^{-1}$]')
    
def plot_flux_points(sed, **kwargs):
    '''Add flux data points to SED plot'''
    ax = kwargs.pop('ax', plt.gca())

    ul_ts_threshold = kwargs.pop('ul_ts_threshold', 4)

    kw = {}
    kw['marker'] = kwargs.get('marker', 'None')
    kw['markersize'] = kwargs.get('size', 3)
    kw['linestyle'] = kwargs.get('linestyle', 'None')
    kw['color'] = kwargs.get('color', 'k')

    fmin, fmax = get_ylims(sed)

    m = sed['ts'] < ul_ts_threshold
    x = sed['e_ref']
    y = sed['e2dnde']
    yerr = sed['e2dnde_err']
    yerr_lo = sed['e2dnde_errn']
    yerr_hi = sed['e2dnde_errp']
    yul = sed['e2dnde_ul']

    delo = sed['e_ref'] - sed['e_min']
    dehi = sed['e_max'] - sed['e_ref']
    xerr0 = np.vstack((delo[m], dehi[m]))
    xerr1 = np.vstack((delo[~m], dehi[~m]))

    plt.errorbar(x[~m], y[~m], xerr=xerr1,
                 yerr=(yerr_lo[~m], yerr_hi[~m]), **kw)
    plt.errorbar(x[m], yul[m], xerr=xerr0,
                 yerr=yul[m] * 0.2, uplims=True,**kw)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(sed['e_min'][0], sed['e_max'][-1])
    ax.set_ylim(10 ** fmin, 10 ** fmax)
    ax.set_ylabel(r'$E^2dN/dE$ [MeV cm$^{-2}$ s$^{-1}$]')


@lru_cache(maxsize=8)
def _load_cirelli_table(particle, EWcorr):
    '''
    Cached loader for the PPPC4DM spectra table. The raw file depends only on
    (particle, EWcorr), not on DMmass or DMchannel, so we parse it once per
    process and reuse for every (mass, channel) query.

    Returns (massvec, energy_vec, table_2d) where table_2d has shape
    (len(massvec), listenergies, n_columns) so callers can slice any channel.
    '''
    if EWcorr == 'Yes':
        listenergies = 179
        energy_vec = np.arange(-8.9, 0.05, 0.05)
        table = np.loadtxt(
            f'{cirelli_dir}AtProduction_all/AtProduction_%s.dat' % (particle,),
            skiprows=1,
        )
    elif EWcorr == 'No':
        listenergies = 180
        energy_vec = np.arange(-8.95, 0.05, 0.05)
        table = np.loadtxt(
            f'{cirelli_dir}AtProductionNoEW_all/AtProduction%sEW_%s.dat'
            % (EWcorr, particle),
            skiprows=1,
        )
    else:
        raise ValueError("EWcorr must be 'Yes' or 'No'")

    massvec = table[::listenergies, 0]
    table_3d = table.reshape(len(massvec), listenergies, table.shape[1])
    return massvec, energy_vec, table_3d


def exctractcirellitable(DMmass, DMchannel, particle, EWcorr):
    '''
    This function returns the energy spectrum in 1/GeV for the particle production from DM annihilation.
    These results come from the PPPC4DM http://www.marcocirelli.net/PPPC4DMID.html
    Relevant tables should be downloaded and stored locally

    DMmass: dark matter mass in GeV
    DMchannel: dark matter annihilation channel with EW correction (no EW correction)
    e 4 (2), mu 7 (3), tau 10 (4), bb 13 (7), tt 14 (8), WW 17 (9), ZZ 20 (10), gamma 22 (12), h 23 (13)
    particle: particle produced from the DM annihilation ('gammas' or 'positrons')
    EWcorr: electroweak corrections ('Yes' or 'No')
    '''

    massvec, energy_vec, table_3d = _load_cirelli_table(particle, EWcorr)
    flux = table_3d[:, :, DMchannel]

    # Create interpolator - note: RegularGridInterpolator expects (x, y) grid points
    # and values with shape (len(x), len(y))
    f = RegularGridInterpolator((massvec, energy_vec), flux, method='linear', bounds_error=False, fill_value=None)

    # Evaluate at all energy points for the given DMmass
    points = np.column_stack([np.full(len(energy_vec), DMmass), energy_vec])
    fluxDM = f(points)

    return np.power(10., energy_vec) * DMmass, fluxDM / (np.log(10.) * np.power(10., energy_vec) * DMmass)

def func_interpolate(varval,variablevec,funcvec):
    
    result = 0.
    if varval<variablevec[0]:
        result = 0.
    elif varval>variablevec[len(variablevec)-1]:
        result = 0.
    else:
        Log10E_bin = np.log10(variablevec[1])-np.log10(variablevec[0])
        nbin = ( np.log10(varval)-np.log10(variablevec[0]) )/Log10E_bin
        binval = int(nbin)
        result = pow(10.,  np.log10(funcvec[binval]) + (np.log10(funcvec[binval+1])-np.log10(funcvec[binval]))*(np.log10(varval)-np.log10(variablevec[binval]))/(np.log10(variablevec[binval+1])-np.log10(variablevec[binval]))  )

    return result

# NOTE: For alterante DM flux models, replace this function
def flux_DM_prompt(Energy,DMprop,Jfact):
    '''
        This function returns the DM gamma-ray flux in (MeV^-1 cm^-2 s-1)
    '''
    #Precompute dNdE DM
    DMmass = DMprop[0]
    sigmav = DMprop[1]
    DMchannel = DMprop[2]
    EWcorr = DMprop[3]    
    EnergyDM,fluxDM = exctractcirellitable(DMmass,DMchannel,'gammas',EWcorr)
    for t in range(len(EnergyDM)):
        if fluxDM[t]>0.:
            fluxDM[t]=fluxDM[t]
        else:
            fluxDM[t]=1e-30
    
    dNdE = func_interpolate(Energy,EnergyDM,fluxDM)

    return 0.5*(1./(4.*np.pi))*pow(1./DMmass,2.)*Jfact*sigmav*dNdE *(1e-3) #1e-3 converts 1/GeV->1/MeV

def interpolate_Loglike(fluxval,table_norm,table_loglike):
    
    #print(table_norm,table_loglike)
    
    if fluxval>table_norm[len(table_norm)-1]:
        return table_loglikescan[len(table_norm)-1]
    else:
        f = interpolate.interp1d(table_norm,table_loglike, kind='linear')
        return f(fluxval)
    

def funcLogLike(fluxvec, loglike_interps):
    '''Sum of SED loglikes given a precomputed per-bin flux vector and
    prebuilt per-bin interp1d objects. Returns -LogLike (consistent with the
    original sign convention).'''
    LogLike = 0.
    for t in range(len(loglike_interps)):
        LogLike += loglike_interps[t](fluxvec[t])
    return -LogLike


def funcLogLike_profJ(fluxvec_unit, loglike_interps, norm_tops, loglike_tops,
                      Jfactor, sigmaJ):
    '''Profile over J-factor nuisance parameter par0.
    fluxvec_unit: flux at each SED energy bin at par0=1.
    loglike_interps: prebuilt interp1d per energy bin.
    norm_tops, loglike_tops: per-bin high-end clamp values
    (i.e. table_norm[t][-1] and table_loglike[t][-1]).'''
    N_E = len(loglike_interps)

    def f(par0):
        LogLikeJ = -pow((np.log10(Jfactor*par0)-np.log10(Jfactor)),2.)/(2.*sigmaJ*sigmaJ)
        Loglike = 0.
        for t in range(N_E):
            fluxval = fluxvec_unit[t]*par0
            if fluxval < norm_tops[t]:
                Loglike += loglike_interps[t](fluxval)
            else:
                Loglike += loglike_tops[t]
        return -Loglike - LogLikeJ

    m = Minuit(f, par0=0.01)
    m.limits = (1e-3, 1e3)
    m.tol = 1e-4
    m.errordef = 1
    m.migrad()

    return m.fval, m.values["par0"], m.errors["par0"]

def convert_sed(sed, Jfactor, sigmaJ=None, EWcorr='Yes', indexDMchannel = 10,
               mass_vec=np.logspace(0,4,40), sigmav_vec=np.logspace(-28,-22,60)):

    # With a J-factor prior, delegate to the tabulated profile in
    # convert_sed_batch so single-J results share the same global-minimum
    # machinery as the MC path — Minuit.migrad starting at par0=0.01 otherwise
    # gets stuck in local minima on ~3% of strong-signal cells.
    if sigmaJ is not None:
        TS = convert_sed_batch(
            sed, Jfactors=np.asarray([Jfactor]), sigmaJ=sigmaJ,
            EWcorr=EWcorr, indexDMchannel=indexDMchannel,
            mass_vec=mass_vec, sigmav_vec=sigmav_vec,
        )
        return TS[0]

    # No-prior path: kept for API compatibility; not exercised by production
    # drivers (every call in fermi_reweighting.py passes sigmaJ).
    table_refdnde = copy.deepcopy(sed['ref_dnde'])
    table_normscan = copy.deepcopy(sed['norm_scan'])
    for t in range(len(table_refdnde)):
        table_normscan[t] = sed['norm_scan'][t,:]*table_refdnde[t]
    table_loglikescan = copy.deepcopy(sed['dloglike_scan'])
    eref_vec = copy.deepcopy(sed['e_ref'])/1.e3
    N_E = len(eref_vec)

    loglike_interps = [
        interpolate.interp1d(table_normscan[t], table_loglikescan[t],
                             bounds_error=False, fill_value='extrapolate')
        for t in range(N_E)
    ]

    LogLike_vec = np.zeros(shape=(len(mass_vec),len(sigmav_vec)))
    mass_factor = 0.5 * (1./(4.*np.pi)) * Jfactor * 1e-3

    for u in range(len(mass_vec)):
        DMmass = mass_vec[u]
        EnergyDM, fluxDM = exctractcirellitable(DMmass, indexDMchannel, 'gammas', EWcorr)
        fluxDM = np.where(fluxDM > 0., fluxDM, 1e-30)
        dNdE_at_eref = np.array([
            func_interpolate(eref_vec[t], EnergyDM, fluxDM) for t in range(N_E)
        ])
        mass_prefactor = mass_factor * (1.0/DMmass)**2 * dNdE_at_eref
        for t in range(len(sigmav_vec)):
            fluxvec = mass_prefactor * sigmav_vec[t]
            LogLike_vec[u,t] = -funcLogLike(fluxvec, loglike_interps)

    return 2*(LogLike_vec - LogLike_vec[-1,0])


def convert_sed_batch(sed, Jfactors, sigmaJ, EWcorr='Yes', indexDMchannel=10,
                      mass_vec=np.logspace(0, 4, 40),
                      sigmav_vec=np.logspace(-28, -22, 60),
                      n_x_grid=2000, x_pad_dex=0.05):
    '''
    Batch version of convert_sed for many J-factor draws sharing a common
    sigmaJ. Tabulates the J-independent part of the profile likelihood once
    per (mass, sigmav) cell and derives each draw's TS by argmin over that
    tabulation — avoiding a per-draw Minuit.migrad().

    Mathematical basis. After §2 we have
        fluxvec_unit[t] = (0.5/(4π)) · (1/M²) · dNdE_at_eref[t] · J · 1e-3
    so flux in the Minuit objective depends on J and par0 only through the
    product x ≡ J · par0. Define
        g(x) = -Σ_t L_t(scale[t] · x)   with the original high-side clamp
    where scale[t] = (σv · common[t]) and common[t] is the J-independent
    prefactor above. Then
        f_k(x) = g(x) + (log10(x/J_k))² / (2 σJ²)
    equals the Minuit objective evaluated at par0 = x / J_k. We scan x on a
    log-spaced grid, argmin f_k, and parabolically refine.

    Returns array of shape (len(Jfactors), len(mass_vec), len(sigmav_vec))
    with the same convention as convert_sed(sigmaJ=...) output: the per-draw
    TS relative to its (mass_vec[-1], sigmav_vec[0]) reference cell.
    '''
    Jfactors = np.asarray(Jfactors, dtype=float)

    # --- SED setup (matches convert_sed) ---
    table_refdnde = copy.deepcopy(sed['ref_dnde'])
    table_normscan = copy.deepcopy(sed['norm_scan'])
    for t in range(len(table_refdnde)):
        table_normscan[t] = sed['norm_scan'][t, :] * table_refdnde[t]
    table_loglikescan = copy.deepcopy(sed['dloglike_scan'])
    eref_vec = copy.deepcopy(sed['e_ref']) / 1.e3
    N_E = len(eref_vec)

    loglike_interps = [
        interpolate.interp1d(table_normscan[t], table_loglikescan[t],
                             bounds_error=False, fill_value='extrapolate')
        for t in range(N_E)
    ]
    norm_tops = np.array([table_normscan[t][-1] for t in range(N_E)])
    loglike_tops = np.array([table_loglikescan[t][-1] for t in range(N_E)])
    # Symmetric low-side clamp: SciPy's interp1d(fill_value='extrapolate')
    # would otherwise linearly extrapolate the slope at the lowest tabulated
    # norm into the flat-tail regime, which fakes per-bin loglike preference
    # (and doesn't cancel in TS = LL - LL_ref). Hold flat at the boundary,
    # mirroring the existing high-side clamp.
    norm_bots = np.array([table_normscan[t][0] for t in range(N_E)])
    loglike_bots = np.array([table_loglikescan[t][0] for t in range(N_E)])

    # --- x = J·par0 grid, union over all draws ---
    par0_min, par0_max = 1e-3, 1e3
    log10_Jfactors = np.log10(Jfactors)
    log10_Jmin = log10_Jfactors.min()
    log10_Jmax = log10_Jfactors.max()
    log10_x_min = log10_Jmin + np.log10(par0_min) - x_pad_dex
    log10_x_max = log10_Jmax + np.log10(par0_max) + x_pad_dex
    log10_x_grid = np.linspace(log10_x_min, log10_x_max, n_x_grid)
    x_grid = 10.0 ** log10_x_grid

    # Per-draw par0 bounds mask, shape (n_draws, n_x_grid).
    log10_par0_mat = log10_x_grid[None, :] - log10_Jfactors[:, None]
    bound_mask = (log10_par0_mat >= np.log10(par0_min)) & (log10_par0_mat <= np.log10(par0_max))

    n_draws = len(Jfactors)
    n_mass = len(mass_vec)
    n_sv = len(sigmav_vec)
    LL_profJ = np.zeros((n_draws, n_mass, n_sv))
    inv_M2 = (1.0 / mass_vec) ** 2
    inv_2sigJ2 = 1.0 / (2.0 * sigmaJ * sigmaJ)

    for u in range(n_mass):
        DMmass = mass_vec[u]
        EnergyDM, fluxDM = exctractcirellitable(DMmass, indexDMchannel, 'gammas', EWcorr)
        fluxDM = np.where(fluxDM > 0., fluxDM, 1e-30)
        dNdE_at_eref = np.array([
            func_interpolate(eref_vec[t], EnergyDM, fluxDM) for t in range(N_E)
        ])
        # common[t] strips Jfactor and sigmav from fluxvec_unit·par0.
        common = 0.5 * (1.0 / (4.0 * np.pi)) * inv_M2[u] * dNdE_at_eref * 1e-3

        for sv_i in range(n_sv):
            scale = sigmav_vec[sv_i] * common  # (N_E,)
            # Compute g(x_grid) with the original high-side clamp.
            # flux_mat[t, i] = scale[t] * x_grid[i]
            flux_mat = np.outer(scale, x_grid)  # (N_E, n_x_grid)
            L_mat = np.empty_like(flux_mat)
            for t in range(N_E):
                below_top = flux_mat[t] < norm_tops[t]
                above_bot = flux_mat[t] > norm_bots[t]
                in_range = below_top & above_bot
                if in_range.any():
                    L_mat[t, in_range] = loglike_interps[t](flux_mat[t, in_range])
                L_mat[t, ~below_top] = loglike_tops[t]
                L_mat[t, ~above_bot] = loglike_bots[t]
            g = -L_mat.sum(axis=0)  # (n_x_grid,)

            # Per-draw: f_k(x) = g(x) + (log10(x) - log10(J_k))² / (2 σJ²)
            prior = log10_par0_mat * log10_par0_mat * inv_2sigJ2
            f_k = g[None, :] + prior                                  # (n_draws, n_x_grid)
            f_k_masked = np.where(bound_mask, f_k, np.inf)

            i_min = f_k_masked.argmin(axis=1)                         # (n_draws,)
            fval = f_k_masked[np.arange(n_draws), i_min]

            # 3-point parabolic refinement on the interior.
            interior = (i_min > 0) & (i_min < n_x_grid - 1)
            if interior.any():
                draw_idx = np.where(interior)[0]
                iv = i_min[draw_idx]
                y0 = f_k_masked[draw_idx, iv - 1]
                y1 = f_k_masked[draw_idx, iv]
                y2 = f_k_masked[draw_idx, iv + 1]
                denom = y0 - 2.0 * y1 + y2
                # Accept only well-formed upward-curving triples
                # (all finite and denom > 0).
                good = np.isfinite(y0) & np.isfinite(y2) & (denom > 0.0)
                if good.any():
                    g_idx = draw_idx[good]
                    y0g = y0[good]; y1g = y1[good]; y2g = y2[good]
                    dg = denom[good]
                    # y_min = y1 - (y0 - y2)^2 / (8 · (y0 - 2 y1 + y2))
                    # for three equally spaced samples at k = -1, 0, +1.
                    fval[g_idx] = y1g - (y0g - y2g) ** 2 / (8.0 * dg)

            # Store -fval to match convert_sed's sign convention
            # (LogLikeJprof_vec = -m.fval).
            LL_profJ[:, u, sv_i] = -fval

    # TS per draw, relative to (mass_vec[-1], sigmav_vec[0]) reference.
    TS_arrays = 2.0 * (LL_profJ - LL_profJ[:, -1:, 0:1])
    return TS_arrays


def get_UL(z,  xs, ys,  coeff=2.71):
    ''' calculates upper limit (to the right)'''
    x, y, ULs = [],[],[]

    
    for i,x_val in enumerate(xs):
        min_idx = z[:,i].argmax()
        #print('{:.2f} {} {:2e}'.format(x_val, min_idx, ys[min_idx]))
        zcut = z[:,i].max()-coeff
        f = interpolate.interp1d(z[:,i][min_idx:], ys[min_idx:], bounds_error=False, fill_value='extrapolate')

        idx = (np.abs(z[:,i]-(z[:,i].max()-coeff))[min_idx:].argmin()+min_idx)
        if z[:,i].min()> z[:,i].max()-coeff:
            ULs.append(min(ys))
            x.append(x_val)
            y.append(min(ys))

        else:

            ul = f(z[:,i].max()-coeff)
            ULs.append(ul)
            x.append(x_val)
            y.append(ys[np.argmax(z[:,i])])
        
    return x, ULs, y

