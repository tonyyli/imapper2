"""
    Helper functions for intensity mapping code.
"""
import logging
import glob
import os
import re
import shutil
import errno
import numpy as np
import h5py

# For timing
import time

# AstroPy imports
import astropy.cosmology as ac
import astropy.units as au
from astropy import constants as const

# Other imports from this folder
import halolist as hlist

#############
# COSMOLOGY #
#############
def arcmin_to_cmpc(arcmin, z, cosmo):
    """Convert arcmin to transverse comoving Mpc, assuming a given redshift and cosmology.

    Parameters
    ----------
    arcmin : float or array-like
        length in arcminutes
    z : float
        redshift
    cosmo : astropy.cosmology object
        assumed cosmology
    
    Returns
    -------
    dco : float or array-like
        Comoving distance in Mpc
    """
    mpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).to(au.Mpc/au.arcmin).value
    dco = arcmin * mpc_per_arcmin
    return dco


def z_to_cmpc(redshift, cosmo):
    """Convert redshift to comoving depth.
    
    Parameters
    ----------

    Returns
    -------
    """
    dco = cosmo.comoving_distance(redshift).to(au.Mpc).value
    return dco


####################
# UNIT CONVERSIONS 

def lsun_to_uk(lum, nurest, vol, redshift, cosmo):
    """
    Convert line luminosity, at a given frequency, within a given comoving volume, to R-J brightness temperature.

    To reconfirm the value of the `prefac`:
      import astropy.constants as const
      import astropy.units as u
      prefac = const.c**3 * u.Lsun / ( 8 * np.pi * const.k_B * u.GHz**3 * u.km / u.s / u.Mpc * u.Mpc**3 )
      prefac.to(u.uK)

    Parameters
    ----------
    lum : float or array of floats
        total luminosity within spectral bin [Lsun]

    nurest : float
        rest-frame line frequency (GHz)

    vol : float or array
        comoving volume [Mpc^3]

    redshift : float or array
        redshift

    cosmo : astropy.cosmology object

    Returns
    -------
    temprj : float or array
        Rayleigh-Jeans brightness temperature
    """
    prefac = 3.136514e4        # Prefactor for temperature conversion [uK]
    
    # Calculate H(z) on a 1D grid, and interpolate to get H(z) across entire `redshift` array
    if isinstance(redshift, np.ndarray):
        ngrid = max(redshift.shape)*3
    else: 
        ngrid = 20
    zgrid = np.linspace(np.min(redshift)*.9, np.max(redshift)*1.1, ngrid)
    Hgrid = cosmo.H(zgrid).to(au.km/au.s/au.Mpc).value
    Hofz  = np.interp(redshift, zgrid, Hgrid)   # Interpolate

    temprj = (prefac/(nurest**3 * Hofz)) * (redshift + 1.)**2 * lum / vol
    
    return temprj


def uk_to_mjy(t_uK, nu_GHz, th_arcmin):
    """Convert brightness temperature [uK] to flux density [mJy].

    See equation at https://science.nrao.edu/facilities/vla/proposing/TBconv:
        T = 1.36 * ( lambda[cm]^2 / theta[arcsec]^2 ) * S[mJy/beam]

    Parameters
    ----------
        t_uK: brightness temperature in uK
        nu_GHz: frequency in GHz
        th_arcmin: FWHM in arcmin

    Returns
    -------
        s_mJy (float): flux density in mJy
    """
    l_cm = 3e1 / nu_GHz # wavelength [cm]
    t_K = t_uK / 1e6
    th_arcsec = th_arcmin * 60.
    s_mJy = t_K / 1.36 / (l_cm/th_arcsec)**2

    return s_mJy


######################
# ARRAY MANIPULATION #
######################
def bin_midpoints(bin_edges):
    """Given an ordered sequence of bin edges, compute the midpoints
    """
    midpoints = 0.5*(bin_edges[:-1] + bin_edges[1:])
    return midpoints


def bin_lengths(bin_edges):
    """
    Parameters
    ----------
    bin_edges : 1d array
        n-element 1D array of monotonically increasing bin edges
        
    Returns
    -------
    bin_lengths : 1d array
        (n-1)-element 1D array of bin lengths
    """
    return bin_edges[1:] - bin_edges[:-1]



################
# INPUT/OUTPUT #
################
def load_halos(lc_path, cosmo, with_rsd):
    """
    Load halo catalog, and store halo properties in a HaloList object (see 'halolist' module).

    The lightcone file itself should in the *.npz format, and the halo properties included must be:
        ...[unfinished]

    Parameters
    ----------

    Returns
    -------
    halos : HaloList object
        object containing halo properties
    """

    logging.info("Loading in halo data...")
    logging.info("  From: %s" % lc_path)

    # Load halo data into a new HaloList object
    halos = hlist.HaloList(lc_path)

    # Remove factors of h (from x, y, z, mass)
    hubble_h = cosmo.h
    halos.x /= hubble_h
    halos.y /= hubble_h
    halos.z /= hubble_h
    halos.m /= hubble_h

    # Convert RA and Dec from degrees to arcmin
    halos.ra  *= 60.
    halos.dec *= 60.

    # Re-center angular coordinates so that `ra`, `dec` >= 0
    halos.ra  -= np.min(halos.ra)
    halos.dec -= np.min(halos.dec)

    # Check if we're including distorted line-of-sight redshifts (pre-calculated in halo catalog)
    if not with_rsd:
        halos.zlos = halos.zcos

    logging.info("---------- HALO PROPERTIES ----------")
    logging.info("=====================================")
    logging.info("  Number of halos: %d" % halos.x.size)
    logging.info("")
    logging.info("  Comoving space occupied by halos:")
    logging.info("    x : %.2f to %.2f Mpc" % (np.min(halos.x), np.max(halos.x)))
    logging.info("    y : %.2f to %.2f Mpc" % (np.min(halos.y), np.max(halos.y)))
    logging.info("    z : %.2f to %.2f Mpc" % (np.min(halos.z), np.max(halos.z)))
    logging.info("")
    logging.info("  RA and Dec range occupied by halos:")
    logging.info("    RA  : %.2f to %.2f arcmin" % (np.min(halos.ra), np.max(halos.ra)))
    logging.info("    Dec : %.2f to %.2f arcmin" % (np.min(halos.dec), np.max(halos.dec)))
    logging.info("")
    logging.info("  Mass range of halos:")
    logging.info("    m : %.2e to %.2e Msun" % (np.min(halos.m), np.max(halos.m)))
    logging.info("")
    logging.info("  Redshift range of halos:")
    logging.info("    redshift : %.2f to %.2f" % (np.min(halos.zcos), np.max(halos.zcos)))
    logging.info("")

    return halos

def save_cube(fpath, x, y, z, t):
    """Save temperature cube to file.
    """
    ensure_dir(fpath)

    logging.info("Saving numpy data cube...")
    logging.info("  TO : {}.npz".format(fpath) + "\n")
    np.savez(fpath, x=x, y=y, z=z, t=t)


def load_cube(fpath):
    """Load data cube from file.
    """
    keys = ['x','y','z','t']
    data = np.load(tcube_path)
    x,y,z,t = [data[k] for k in keys]
    return x, y, z, t


def save_powersph(fpath, k, p, perr, pnoise, nmodes, fres):
    header = "k[Mpc^-1]  P(k)[uK^2 Mpc^3]  sigmaP(k)[uK^2 Mpc^3]  Pnoise(k)[uK^2 Mpc^3]  Nmodes(k)  fres(k)"

    logging.info("Saving power spectrum...")
    logging.info("  TO : {}".format(fpath))
    np.savetxt(fpath, np.c_[k, p, perr, pnoise, nmodes, fres], header=header)


def save_powercyl(fpath, kprp, kpar, pcyl):
    np.savez(fpath, kprp=kprp, kpar=kpar, pcyl=pcyl)


def ensure_dir(path):
    """
    Ensure that directory exists (create if it doesn't)
    """
    dir, base = os.path.split(path)
    try:
        os.makedirs(dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def save_data(tcube_data, psph_data, pcyl_data, params):
    """Save data to 

    Parameters
    ----------
    tcube_data : dictionary
        contains the keys 'x', 'y', 'z', 't'.  xyz are 1D arrays, t is a 3D array.
    psph_data : dictionary
        contains the keys 'ksph', 'psph', 'perr', 'pnoise', 'nmodes', 'wres'
    pcyl_data : dictionary
        contains the keys 'kprp', 'kpar', 'pcyl'
    """
    outdir = params['io']['output_folder']
    fname = None # TODO: string
    fp_out = os.path.join(outdir, fname)

    f = h5py.File(fp_out, 'w')

    # Write tcube data
    tcube_group = f.create_group('tcube')
    tcube_group.create_dataset('t', data=tcube_data['t'])
    tcube_group.create_dataset('x', data=tcube_data['x'])
    tcube_group.create_dataset('y', data=tcube_data['y'])
    tcube_group.create_dataset('z', data=tcube_data['z'])

    # Write psph data
    psph_group = f.create_group('psph')
    psph_group.create_dataset('ksph', data=psph_data['ksph'])
    psph_group.create_dataset('psph', data=psph_data['psph'])
    psph_group.create_dataset('perr', data=psph_data['perr'])
    psph_group.create_dataset('pnoise', data=psph_data['pnoise'])
    psph_group.create_dataset('nmodes', data=psph_data['nmodes'])
    psph_group.create_dataset('wres', data=psph_data['wres'])

    # write pcyl data
    pcyl_group = f.create_group('pcyl')
    pcyl_group.create_dataset('kprp', data=pcyl_data['kprp'])
    pcyl_group.create_dataset('kpar', data=pcyl_data['kpar'])
    pcyl_group.create_dataset('pcyl', data=pcyl_data['pcyl'])

    # TODO write parameters and timestamp

    # TODO test this method


################################################################################
# Obsolete?

def cmpc_to_arcmin(dco, z, cosmo):
    """Convert transverse comoving Mpc to arcmin, assuming a given redshift and cosmology.

    Parameters
    ----------
    dco : float or array-like
        Comoving distance in Mpc
    z : float or array-like
        redshift
    cosmo : astropy.cosmology object
        assumed cosmology
    
    Returns
    -------
    arcmin : float or array-like
        length in arcminutes
    """
    # Interpolate to speed up calculation
    zgrid                   = np.logspace(-4,3,1000)    # Redshift on preset grid
    arcmin_per_cmpc_grid    = 1./cosmo.kpc_comoving_per_arcmin(zgrid).to(au.Mpc/au.arcmin).value    # Arcmin per comoving Mpc on preset grid

    arcmin_per_cmpc = np.interp(z, zgrid, arcmin_per_cmpc_grid)
    arcmin          = dco * arcmin_per_cmpc
    return arcmin

def cmpc_to_nu(dco, nu0, cosmo):
    """Convert comoving radial distance to observed frequency, assuming a given rest-frame frequency.

    Parameters
    ----------
    dco : float or array-like
        Comoving distance in Mpc
    nu0 : float
        Rest frame frequency
    cosmo : astropy.cosmology object
        assumed cosmology

    Returns
    -------
    nu : float or array-like
        Observed frequency, same units as nu0
    """
    redshift = cmpc_to_z(dco, cosmo)
    nu = nu0 / (1. + redshift)
    return nu

def cmpc_to_z(dco, cosmo):
    """Convert comoving depth to redshift.

    Parameters
    ----------

    Returns
    -------
    """
    # Interpolate to get redshift from comoving_distance
    zgrid = np.logspace(-4,3,1000)
    dgrid = cosmo.comoving_distance(zgrid).to(au.Mpc).value
    redshift = np.interp(dco, dgrid, zgrid)
    return redshift

def dnu_to_cmpc(dnu, nu0, redshift, cosmo):
    """Convert frequency interval to comoving depth interval, given rest-frame frequency.

    Parameters
    ----------

    Returns
    -------
    """
    dldz = (const.c / cosmo.H(redshift)).to(au.Mpc).value
    dco = dldz * (1.+redshift)**2 * (dnu/nu0)
    return dco
