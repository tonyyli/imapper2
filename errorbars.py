import logging
import numpy as np
import astropy.constants as const
import astropy.units as au

def powersph_error(power,
        tsys, nfeeds, tobs, xarclen, yarclen, fwhm, nu_min, nu_max, dnu, nu0, k, cosmo):
    """
    Get instrument sensitivity in power spectrum, provided a set of instrument parameters
    Required parameters for:

    Parameters
    ----------
    power : ndarray
        must have same length as `k`
    tsys : float
        System temperature [K]
    nfeeds
        Number of feeds
    tobs
        Total observing time [s]
    xarclen
        Length of FOV [arcmin]
    fwhm
        FWHM of beam [arcmin]
    nu_min
        minimum observing frequency [GHz]
    nu_max
        maximum observing frequency [GHz]
    dnu
        frequency resolution [GHz]
    nu0
        rest-frame line frequency [GHz]
    k : array
        a vector of k at which to calculate power noise
    cosmo : astropy.cosmology object
        For cosmology calculations

    Returns
    -------
    sigma_power : array
        Expected error bars in power spectrum as a function of `k`.  Units of [uK^2 * Mpc^3].  Same dimensions as `k`.
    noise_power : array
    nmodes : array
    res_factor : array
    """

    nu_obs  = 0.5*(nu_min + nu_max) # Mean observed frequency

    # Get redshifts (from frequency band and rest-frame line frequency)
    zmax    = nu0/nu_min - 1.       # Max redshift <==> Min observed frequency
    zmin    = nu0/nu_max - 1.       # Min redshift <==> Max observed frequency
    zobs    = nu0/nu_obs - 1.       # Redshift of mean observed frequency (approximate redshift for the entire volume)

    # Get cosmological quantities
    c       = 2.99792458e5                                      # Speed of light            [km/s]
    hubH0   = cosmo.H0 .to(au.km/au.s/au.Mpc) .value            # Hubble constant at z=0    [km/s/Mpc]
    hubHz   = hubH0 * cosmo.efunc(zobs)                         # Hubble constant at zobs   [km/s/Mpc]
    dco     = cosmo.comoving_distance(zobs) .to(au.Mpc) .value  # Comoving distance to zobs [Mpc]
    
    # Survey dimensions
    # -- Important results:
    #   -- survey area  (for: sigma_vox)
    #   -- survey vol   (for: nmodes)
    surv_xanlen     = xarclen/60.                               # Angular x length of survey area   [deg]
    surv_yanlen     = yarclen/60.                               # Angular y length of survey area   [deg]
    surv_area       = surv_xanlen * surv_yanlen                 # Angular area of square survey     [deg^2]
    surv_xcolen     = dco * surv_xanlen * (np.pi/180.)          # Comoving x length of survey area  [Mpc]
    surv_ycolen     = dco * surv_yanlen * (np.pi/180.)          # Comoving y length of survey area  [Mpc]
    surv_zcolen     = dco_between(zmin,zmax,cosmo)              # Comoving survey depth             [Mpc]
    surv_vco        = surv_xcolen * surv_ycolen * surv_zcolen   # Comoving survey volume            [Mpc^3]

    # Voxel dimensions
    # -- Important results:
    #   -- sig_beam      (for: voxel volume, kprp_res)
    #   -- Voxel vol    (for: sigma_power)
    sig_beam        = fwhm_to_sigma(fwhm/60. * np.pi/180.)                  # Beam size (Gaussian sigma)           [rad]
    vox_area        = (sig_beam * 180./np.pi * 60.)**2                      # Angular area of voxel [arcmin^2]
    vox_vco         = voxel_volume(dco, sig_beam, nu0, nu_obs, dnu, cosmo)  # Comoving voxel volume [Mpc^3]

    # Calculate the highest-k (smallest-scale) modes in the parallel (sky) and perpendicular (line-of-sight) directions
    kmax_par        = kmax_parallel(zobs, nu_obs, dnu, cosmo)   # (Max) k at parallel (line-of-sight) resolution     [1/Mpc]
    kmax_prp        = kmax_perpendicular(dco, sig_beam)         # (Max) k at perpendicular (across sky) resolution   [1/Mpc]
    
    sigma_vox       = sigma_radiometer(surv_area, vox_area, tobs, tsys, dnu, nfeeds)    # Calculate sensitivity per sky pixel (using radiometer equation)
    res_factor      = res_damping(k, kmax_par, kmax_prp)                            # Calculate exponential damping, as a function of k -- TODO: explain?
    nmodes          = num_modes(k, surv_vco)                                        # Calculate number of modes, as a function of k -- TODO: explain?

    # Calculate variance in power spectrum, as a function of k
    noise_power   = sigma_vox**2 * vox_vco * np.ones(power.size)   # Noise power spectrum
    sigma_noise   = noise_power / np.sqrt(nmodes) # Sample variance error in noise power spectrum

    # Add sample variance (~cosmic variance) if requested
    sigma_samplevar = power / np.sqrt(nmodes)

    sigma_power = sigma_samplevar + sigma_noise

    sigma_power /= res_factor # Include resolution-limited scaling
    
    logging.info('sigma_vox (uK)        = {:f}'.format(sigma_vox))
    logging.info('dco (Mpc)             = {:f}'.format(dco))    
    logging.info('H(zobs)               = {:.4f}'.format(hubHz) + "\n")
    
    logging.info('Survey Parameters (comoving coords)')
    logging.info('z max                 = %f' % zmax)
    logging.info('z min                 = %f' % zmin)
    logging.info('nu obs (GHz)          = %f' % nu_obs)
    logging.info('LOS Delta D (Volume)  = %f' % surv_zcolen)
    logging.info('Transverse X (Volume) = %f' % surv_xcolen)
    logging.info('Transverse Y (Volume) = %f' % surv_ycolen)
    logging.info('Survey Volume         = %f' % surv_vco + "\n")
    
    return sigma_power, noise_power, nmodes, res_factor


def powercyl_error():
    pass # TODO


####################
# Helper functions #
####################
def sigma_radiometer(area_surv, area_res, tobs, tsys, dnu, nfeeds):
    """Calculate sensitivity per voxel by evaluating the radiometer equation.

    Parameters
    ----------
    area_surv    [deg^2]     : Area of full survey
    area_res  [arcmin^2]    : Area of angular resolution (beam size)
    tobs    [s]         : Total observing time
    tsys    [K]         : System temperature
    dnu     [GHz]       : Voxel width in frequency
    nfeeds              : Number of collecting feeds

    Returns
    -------
    sigma : expected temperature fluctuation per pixel [uK]
    """
    tpixel  = tobs * area_res / (area_surv * 3600.)                 # Time per sky pixel [s]
    sigma   = (tsys*1e6) / np.sqrt( (dnu*1e9) * nfeeds * tpixel )   # Fluctuation in each pixel [uK]
    return sigma


def voxel_volume(dco, sig_beam, nu_rest, nu_obs, dnu, cosmo):
    """
    Get comoving voxel volume

    Parameters
    ----------
    dco :
        comoving distance
    sig_beam :
        beam width
    nu_rest :
        rest frame line frequency
    nu_obs :
        observed frequency
    dnu :
        frequency channel width
    cosmo :
        astropy.cosmology object (contains cosmological info)
    """
    vox_xycolen     = dco * sig_beam                 # Comoving transverse length of voxel   [Mpc] (Note: calculated from sig_beam, not FWHM)
    zobs            = nu_rest/nu_obs - 1.
    zvox            = nu_rest/(nu_obs-dnu) - 1.         # Redshift of deep side of the voxel
    vox_zcolen      = dco_between(zobs,zvox,cosmo)  # Comoving pixel depth                  [Mpc]
    vox_vco         = vox_xycolen**2 * vox_zcolen  # Comoving pixel volume                 [Mpc^3]
    return vox_vco


def fwhm_to_sigma(fwhm):
    """Calculate Gaussian sigma (standard deviation) from FWHM value.
    
    The relation between them is: fwhm = sqrt(8ln2) * sigma
    """
    return 0.4246609 * fwhm


def dco_between(zlo, zhi, cosmo):
    """Calculate the comoving distance [Mpc] between two redshifts in a given cosmology.
    """
    return ( cosmo.comoving_distance(zhi) - cosmo.comoving_distance(zlo) ).to(au.Mpc).value


def kmax_parallel(zobs, nu_obs, dnu, cosmo):
    """Calculate maximum k (along line of sight) from frequency resolution
    """
    c       = 2.99792458e5                                        # Speed of light    [km/s]
    hubHz   = cosmo.H(zobs) .to(au.km / au.s / au.Mpc)  .value  # H(z)              [km/s/Mpc]
    return (hubHz / c) / (1+zobs) * (nu_obs / dnu)   #Note: should *NOT* have 2pi as prefactor


def kmax_perpendicular(d_comoving, sigma_beam):
    """Calculate maximum k (across sky) from angular resolution

    Parameters
    ----------
    d_comoving : 
        comoving radial distance (pre-calculated from redshift) [Mpc]
    sigma_beam : 
        arc length of beam width [radians]
    """
    kmax_perp = 1. / (d_comoving * sigma_beam)

    return kmax_perp


def num_modes(k, vol):
    """Calculate the number of modes in a given volume as a function of k.

    The result is premultiplied by a factor 1/2 because the 3D intensity map is real-valued, so only half the modes are independent.
    """
    #dk = 0.5 * k # Assume uniform log spacing, dlnk = 0.5
    dk = (k[1]-k[0])*np.ones(k.shape) # Assuming uniform linear spacing
    return 0.5 * (k**2 * dk * vol) / (2. * np.pi**2)


def res_damping(k, kmax_par, kmax_prp, nmu=100):
    """
    Resolution-limit "damping" factor on the power spectrum, as a function of k

    Parameters
    ----------
    k : ndarray
    kmax_par : float
    kmax_prp : float
    nmu : int
    """
    mu          = np.linspace(0, 1, nmu, endpoint=False) # Cos(theta), where theta is the angle between k and k_parallel (LOS projection)
    k_par       = np.outer( k, mu )
    k_prp       = np.outer( k, np.sqrt(1.-mu**2) )
    #integrand   = np.exp( -2.*( (k_par/kmax_par)**2 + (k_prp/kmax_prp)**2 ) )
    integrand   = np.exp( -1.*( (k_par/kmax_par)**2 + (k_prp/kmax_prp)**2 ) )
    
    exp_sigma   = np.trapz( integrand, mu, axis=1)
    return exp_sigma

