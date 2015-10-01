"""
    Implements the model of Li+2015, which itself combines Behroozi+13, Kennicutt98, and Carilli+Walter13.
"""

import os
import numpy as np
import scipy.interpolate

def line_luminosity(halos, sigma_sfr=0.3, delta_mf=1.0, alpha=1.37, beta=-1.74, sigma_lco=0.3, min_mass=1e10):
    """
    Parameters
    halos : HaloList object
        A HaloList object containing all halos and their properties
    line_freq : 
        rest-frame line frequency [GHz]

    Returns
    lco : float array
        halo CO luminosities [Lsun]
    """
    line_freq   = 115.27
    
    hm          = halos.m
    hz          = halos.zcos

    ### HALOS to SFR ###
    dat_path = "%s/bwc13data/sfr_release.dat" % (os.path.dirname(os.path.realpath(__file__)))
    dat_zp1, dat_logm, dat_logsfr, _ = np.loadtxt(dat_path, unpack=True) # Columns are: z+1, logmass, logsfr, logstellarmass

    # Intermediate processing of tabulated data
    dat_logzp1 = np.log10(dat_zp1)
    dat_sfr = 10.**dat_logsfr

    # Reshape arrays
    dat_logzp1  = np.unique(dat_logzp1)  # log(z+1), 1D
    dat_logm    = np.unique(dat_logm)  # log(Mhalo), 1D
    dat_sfr     = np.reshape(dat_sfr, (dat_logm.size, dat_logzp1.size))

    # Get interpolated SFR value(s)
    rbv         = scipy.interpolate.RectBivariateSpline(dat_logm, dat_logzp1, dat_sfr, kx=1, ky=1)
    sfr  = rbv.ev(np.log10(hm), np.log10(hz+1.))

    halos.sfr = sfr
    
    ### SFR to LCO ###
    lir  = sfr * 1e10 / delta_mf

    alphainv = 1./alpha
    lcop = lir**alphainv * 10**(-beta * alphainv)
    
    lco = np.where(
            hm >= min_mass, 
            3e-11 * line_freq**3 * lcop, 
            0. ) # Set all halos below minimum halo mass to have 0 luminosity

    return lco
