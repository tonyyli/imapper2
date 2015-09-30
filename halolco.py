"""
Module for calculating line luminosities from halos.

Broadly speaking, halos come in, line luminosities go out.
"""

import os.path
import numpy as np
import scipy.interpolate

##############
# SFR to LCO #
##############

def sfr_to_lir(sfr, deltamf=1.5):
    """Convert SFR to LIR using Kennicutt98 relation.
    """
    return sfr * 1e10 / deltamf

def lir_to_lcoprime(lir, alpha=1.37, beta=-1.74):
    """Compute LCO' from LIR assuming a power-law relation between them.

    Assumed LIR-LCO' relation of the form:
        log(LIR) = alpha * log(LCO') + beta
    This function inverts that relation.  alpha=1.37, beta=-1.74 are values from Carilli+Walter13.
    """
    alphainv = 1./alpha
    return lir**alphainv * 10**(-beta * alphainv)

def lprime_to_l(lprime, nurest):
    """Convert L' (integrated brightness temperature) to L (luminosity)

    Parameters
    ----------
    lprime : float or array
        Area- and velocity-integrated brightness temperature [K km/s pc^2]
    nurest : float
        Rest-frame line frequency [GHz]
    """
    return 3e-11 * nurest**3 * lprime

###############
# Halo to SFR #
###############
def sfr_bwc13(mass, redshift, mode=None):
    """
    SFR(M,z) relation of Behroozi, Wechsler, and Conroy 2013

    Takes in Mhalo and redshift, returns an interpolated SFR

    Parameters
    ----------
    redshift : float
        redshift at which to interpolate
    mass : float array
        halo masses
    mode : None or str
        None : default mode, uses 'sfr_release' values from paper
        'lo' : uses sfr - error
        'hi' : uses sfr + error
    """
    #TODO: raise warning if mass is outside of well-constrained values

    #TODO: preload data and interpolation function
    #TODO: calculate BWC13 SFR once and store as a 'base' SFR
    
    if mode==None:
        # Load in BWC13 halo-SFR data
        dat_path = "%s/bwc13data/sfr_release.dat" % (os.path.dirname(os.path.realpath(__file__)))
        zp1, logm, logsfr, _ = np.loadtxt(dat_path, unpack=True) # Columns are: z+1, logmass, logsfr, logstellarmass

        # Intermediate processing of tabulated data
        logzp1 = np.log10(zp1)
        sfr = 10.**logsfr
    
        # Reshape arrays
        logzp1  = np.unique(logzp1)  # log(z+1), 1D
        logm    = np.unique(logm)  # log(Mhalo), 1D
        sfr     = np.reshape(sfr, (logm.size, logzp1.size)) # SFR, 2D
    elif mode=='hi':
        fp = "%s/bwc13data/sfr_adderr.npy" % (os.path.dirname(os.path.realpath(__file__)))
        logm, z, sfr = np.load(fp)
        sfr = np.maximum(sfr, 0.) # Ensure non-negative values
        #logsfr = np.log10(sfr)
        logzp1 = np.log10(z+1)
    elif mode=='lo':
        fp = "%s/bwc13data/sfr_suberr.npy" % (os.path.dirname(os.path.realpath(__file__)))
        logm, z, sfr = np.load(fp)
        sfr = np.maximum(sfr, 0.) # Ensure non-negative values
        #logsfr = np.log10(sfr)
        logzp1 = np.log10(z+1)
    
    if isinstance(redshift, np.ndarray):
        logzp1arr = np.log10(redshift+1.)
    else:
        logzp1arr = np.log10(redshift+1.)*np.ones(mass.shape) # Ensure input redshift is same length as mass array
    
    # Get interpolated SFR value(s)
    rbv         = scipy.interpolate.RectBivariateSpline(logm, logzp1, sfr, kx=1, ky=1)
    sfr_interp  = rbv.ev(np.log10(mass), logzp1arr)

    return sfr_interp

def sfr_bwc13_scatter(mass, redshift, scatter=0.25, mode='dex'):
    if mode == 'posterior':
        # Normal distribution with same 68% confidence intervals as BWC13 posteriors -- ignore BWC13 mean
        sfr_addsigma = sfr_bwc13(mass, redshift, mode='hi')
        sfr_subsigma = sfr_bwc13(mass, redshift, mode='lo')
        
        # Normal distribution approximation to posterior
        mu      = 0.5*(sfr_addsigma + sfr_subsigma)
        sigma   = 0.5*(sfr_addsigma - sfr_subsigma)
        
        sfr_scattered = np.random.normal(mu, sigma)
    elif mode == 'dex':
        sfr_mean    = sfr_bwc13(mass, redshift)
        # Scatter halo-to-halo SFR by X dex from the mean, where X is the scatter we want
        sfr_scattered = logscatter(sfr_mean, scatter)
    else:
        raise Exception("SFR scatter mode not recognized!")
    
    return sfr_scattered

##########################################
# Linear (literature) Halo-LCO relations #
##########################################
def righi08(mass):
    """LCO(M) of Righi+2008, as linearly approximated by Breysse+2014"""
    fduty   = 0.0364 # Only for z=2.4
    norm    = 4e-6     # TODO: Verify this (copied from Breysse+2014)
    return fduty*norm*mass

def visbal10(mass):
    """LCO(M) of Visbal+10 at z=2.4"""
    fduty   = 0.0364 # Only for z=2.4
    norm    = 6.24e-7  # TODO: Verify this (copied from Breysse+2014)
    return fduty*norm*mass

def visbal10_z6(mass):
    fduty   = 0.1
    norm    = 6.6e-7
    return fduty*norm*mass

def lidz11(mass):
    """LCO(M) of Lidz+11 at z~7"""
    fduty   = 0.1 # Only for z=7, taken from Eq. 13 of Lidz+2011
    norm    = 2.8e-5
    return fduty*norm*mass

def pullen13A(mass):
    """LCO(M) of Pullen+13A at z=2.4"""
    fduty   = 0.0364 # Only for z=2.4
    norm    = 2e-6     # TODO: Verify this (copied from Breysse+2014)
    return fduty*norm*mass

def pullen13B(mass):
    """LCO(M) of Pullen+13B at z=2.4, as linearly approximated by Breysse+2014"""
    fduty   = 0.0364 # Only for z=2.4
    norm    = 9.6e-6   # TODO: Verify this (copied from Breysse+2014)
    return fduty*norm*mass


############################
# General functional forms #
############################
def powerlaw(x, pow, prefac):
    return prefac * x**pow

def logscatter(x, dexscatter):
    """Return array x, randomly scattered by a log-normal distribution with sigma=dexscatter.
    
    Note: scatter maintains mean in linear space (not log space).
    """
    # Calculate random scalings
    sigma       = dexscatter * 2.302585         # Stdev in log space (DIFFERENT from stdev in linear space), note: ln(10)=2.302585
    mu          = -0.5*sigma**2
    randscaling = np.random.lognormal(mu, sigma, x.shape)
    xscattered  = np.where(x > 0, x*randscaling, x)
    return xscattered

###########
# CO SLED #
###########
def co_sled(j, mode='thermal'):
    """Returns luminosity normalization factor L(j -> j-1)/L(1 -> 0)
    """
    if mode == 'thermal':
        sled = j**2    # Thermal SLED
    # TODO: implement other SLED functional forms
    return sled


########################
# DEPRECATED FUNCTIONS #
########################
def mhalo_to_sfr(*args, **kw):
    """DEPRECATED"""
    raise Exception("The function `halolco.mhalo_to_sfr()` is deprecated, use `mapit.get_lco()` [2015-05-08]")
    pass

def sfr_to_lco(*args, **kw):
    """DEPRECATED"""
    raise Exception("The function `halolco.sfr_to_lco()` is deprecated, use `mapit.get_lco()` [2015-05-08]")
    pass

def sfr_to_lco_shortcut1(*args, **kw):
    """DEPRECATED"""
    raise Exception("Model 'shortcut1' is deprecated")
