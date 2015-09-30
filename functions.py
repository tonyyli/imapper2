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
def arcmin_to_comovingMpc(arcmin, z, cosmo):
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

def comovingMpc_to_arcmin(dco, z, cosmo):
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

def comovingMpc_to_nu(dco, nu0, cosmo):
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
    redshift = comovingMpc_to_z(dco, cosmo)
    nu = nu0 / (1. + redshift)
    return nu

def comovingMpc_to_z(dco, cosmo):
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

def z_to_comovingMpc(redshift, cosmo):
    """Convert redshift to comoving depth.
    
    Parameters
    ----------

    Returns
    -------
    """
    dco = cosmo.comoving_distance(redshift).to(au.Mpc).value
    return dco

def dnu_to_comovingMpc(dnu, nu0, redshift, cosmo):
    """Convert frequency interval to comoving depth interval, given rest-frame frequency.

    Parameters
    ----------

    Returns
    -------
    """
    dldz = (const.c / cosmo.H(redshift)).to(au.Mpc).value
    dco = dldz * (1.+redshift)**2 * (dnu/nu0)
    return dco


####################
# UNIT CONVERSIONS #
####################
def lsun_to_uk(lum, nurest, vol, redshift, cosmo):
    """
    Convert line luminosity at a given frequency to Raleigh-Jeans temperature.

    Parameters
    ----------
    lum : float or array
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
    # To reconfirm the value:
    #   import astropy.constants as const
    #   import astropy.units as u
    #   prefac = const.c**3 * u.Lsun / ( 8 * np.pi * const.k_B * u.GHz**3 * u.km / u.s / u.Mpc * u.Mpc**3 )
    #   prefac.to(u.uK)
    
    # Calculate H(z) on a 1D grid, and interpolate to get H(z) across entire `redshift` array
    if isinstance(redshift, np.ndarray):
        ngrid = max(redshift.shape)*3
    else: 
        ngrid = 20
    zgrid = np.linspace(np.min(redshift)*.9, np.max(redshift)*1.1, ngrid)
    Hgrid = cosmo.H(zgrid).to(au.km/au.s/au.Mpc).value
    Hofz  = np.interp(redshift, zgrid, Hgrid)   # Interpolate

    temprj = (prefac/(nurest**3 * Hofz)) * (redshift + 1.)**2 * lum / vol
    # (1+z)^2 is correct, (1+z)^3 is not
    
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


def comov_to_angfreq(comov, nurest, cosmo):
    """Convert comoving grid to angxangxfreq grid.

    Parameters
    ----------
    comov : 3-tuple
        a 3-tuple with 1D x, y, z arrays, respectively

    nurest : float
        rest frame frequency of spectral line of interest

    cosmo : 

    Returns
    -------
    xa : float array
        x in units of arcmin

    ya : float array
        y in units of arcmin

    zf : float array
        z in units of frequency, same units as nurest
    """
    x, y, z = comov

    # Re-zero transverse coordinates: ensure all x,y > 0
    x -= np.min(x)
    y -= np.min(y)

    redshift = comovingMpc_to_z(z, cosmo) # Assume z is depth dimension, convert from cMpc to redshift

    logging.info("  Converting x from comoving to angular units...")
    xa = comovingMpc_to_arcmin(x, redshift, cosmo)
    logging.info("  Converting y from comoving to angular units...")
    ya = comovingMpc_to_arcmin(y, redshift, cosmo)
    logging.info("  Converting z from comoving to frequency units...")
    zf = comovingMpc_to_nu(z, nurest, cosmo)

    return xa, ya, zf


def angfreqbins_to_vol(angfreqbins, nurest, cosmo):
    """Convert angxangxfreq bins to grid of comoving cell volumes.

    Parameters
    ----------
    angfreqbins : 3-tuple of 1d arrays
        arrays of bin edges in (arcmin, arcmin, and [freq]) respectively

    nurest : float
        rest frame frequency of spectral line of interest

    cosmo : 

    Returns
    -------
    vol : 3d array
        3d array of volume elements, in Mpc^3
    """

    xabins, yabins, zfbins = angfreqbins
    zred_bins    = nurest/zfbins - 1.            # 1d array of redshift bin edges, in depth direction
    zred_mid    = bin_centers(zred_bins)        # 1d array of redshift bin midpoints, in depth direction

    zcobins    = z_to_comovingMpc(zred_bins, cosmo)    # Bin edges in z-direction [comoving Mpc]
    dzco    = bin_lengths(zcobins)                # Bin lengths in z-direction [comoving Mpc]

    # Check that binning is equal in x and y directions
    assert np.allclose( (xabins[1]-xabins[0]), bin_lengths(xabins) )
    assert np.allclose( (yabins[1]-yabins[0]), bin_lengths(yabins) )

    dxa = xabins[1]-xabins[0] # single bin length, in x[arcmin]
    dya = yabins[1]-yabins[0] # single bin length, in y[arcmin]

    dxco = arcmin_to_comovingMpc(dxa, zred_mid, cosmo)    # Bin lengths in x-direction, sampled along the z-direction [co-moving Mpc]
    dyco = arcmin_to_comovingMpc(dya, zred_mid, cosmo)    # Bin lengths in y-direction, sampled along the z-direction [co-moving Mpc]

    dvco_1d = np.abs(dxco*dyco*dzco)
    vol = np.tile(dvco_1d, (xabins.size-1, yabins.size-1, 1)) # Create 3D array of comoving element volumes [Mpc^3]

    return vol

def comovbins_to_vol(comovbins, nurest):
    xbins, ybins, zbins = comovbins

    # Check that binning is even in x, y, z directions
    assert np.allclose( (xbins[1]-xbins[0]), bin_lengths(xbins) )
    assert np.allclose( (ybins[1]-ybins[0]), bin_lengths(ybins) )
    assert np.allclose( (zbins[1]-zbins[0]), bin_lengths(zbins) )

    dxco = xbins[1]-xbins[0]
    dyco = ybins[1]-ybins[0]
    dzco = zbins[1]-zbins[0]

    dvco_1d = dxco*dyco*dzco
    vol = np.tile(dvco_1d, (xbins.size-1, ybins.size-1, 1)) # Create 3D array of comoving element volumes [Mpc^3]

    return vol

def angfreqbins_to_redshiftcube(angfreq, nurest):
    """
    Parameters
    ----------
    angfreq : 3-tuple of 1d arrays
        arrays of bin edges in (arcmin, arcmin, and [freq]) respectively

    nurest : float
        rest frame frequency of spectral line of interest

    Returns
    -------
    redshiftcube : 3d array
        3d array of redshifts (will only vary in depth, aka z, direction)
    """

    xabins, yabins, zfbins = angfreq
    zred_bins = nurest/zfbins - 1.            # 1d array of redshift bin edges, in depth direction
    zred_mid = bin_centers(zred_bins)    # 1d array of redshift bin midpoints, in depth direction

    redshiftcube = np.tile(zred_mid, (xabins.size-1, yabins.size-1, 1)) # Create 3D array of redshifts

    return redshiftcube


######################
# ARRAY MANIPULATION #
######################
def bin_centers(bin_edges, log=False):
    """Given an ordered sequence of bin edges, compute the midpoints
    """
    if log:
        midpoints = np.sqrt(bin_edges[:-1]*bin_edges[1:])
    else:
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

def is_evenly_spaced(arr):
    """Check if an array has evenly spaced elements (linear)"""
    cond = np.allclose((arr[1:]-arr[:-1]), (arr[1]-arr[0]))
    return cond

def cut_halos(halos, params):
    """
    halos : HaloList object
        halo properties (x position, parent id, etc.)

    params : 
        dictionary of parameter values
    """

    cut_mode = params['cut_mode']
    condition = np.ones(halos.m.size, dtype=bool) # By default, select all halos

    if cut_mode == "masscut":
        try:
            condition &= halos.m >= params['minmass']
        except KeyError:
            pass
        try:
            condition &= halos.m <= params['maxmass']
        except KeyError:
            pass
    elif cut_mode == "random":
        pass # TODO: Finish enforcing halo cut

    if 'fduty' in params:
        condition &= np.random.random(condition.size) <= params['fduty']

    new_halos = _cut_halos(halos, condition)


    if 'shuffle_mass' in params:
        if params['shuffle_mass']:
            np.random.shuffle(new_halos.m)

    return new_halos

def _cut_halos(halos, condition):
    """Cut halo property arrays based on given boolean condition

    Parameters
    ----------
    halos : HaloList object

    condition : boolean array, or array of ints
        If boolean, must be same length as all arrays in arrs.
        If array of indices, maximum value cannot be greater than length of any array in arrs

    Returns
    -------
    halos_cut : HaloList object, with condition applied
    """
    halos_cut = halos
    for v in vars(halos_cut):
        attr = getattr(halos_cut, v)

        # Only apply cut to attribute if it is an array
        if isinstance(attr, np.ndarray):
            setattr(halos_cut, v, attr[condition])

    return halos_cut


################
# INPUT/OUTPUT #
################
def load_halos(params, filenum=None, verbose=True):
    """
    Load halo catalog, and store halo properties in a HaloList object (see 'halolist' module).

    The location of the halo data should be stored in 'params' as a string.  The file itself should in the *.npz format, and the halo properties included must be:
        x
        y
        z
        m
        ra
        dec
        ...[unfinished]

    Parameters
    ----------
    params
        dictionary containing all parameter values

    Returns
    -------
    halos : HaloList object
        object containing halo properties
    """

    hl_path = params['lightcone_path']
    cosmo = params['cosmo']

    logging.info("Loading in halo data...")
    logging.info("  From: %s" % hl_path)

    # Load halo data into a new HaloList object
    halos = hlist.HaloList(hl_path)

    # Remove factors of h (from x, y, z, mass)
    hubble_h = cosmo.h
    halos.x /= hubble_h
    halos.y /= hubble_h
    halos.z /= hubble_h
    halos.m /= hubble_h

    # Convert RA and Dec to arcmin (from degrees)
    halos.ra  *= 60.
    halos.dec *= 60.

    # Re-center angular coordinates so that `ra`, `dec` >= 0
    halos.ra  -= np.min(halos.ra)
    halos.dec -= np.min(halos.dec)

    # Are we using line-of sight redshifts (with RSDs) or cosmological redshifts (without RSDs)?
    if not params['enable_rsd']:
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

def save_cube(x, y, z, t, params):
    """Save temperature cube to file.
    """
    if params['save_tcube'] == False:
        logging.info("Note: Not saving data cube")
        return

    outdir  = params['output_folder']
    fname   = params['fname_tcube']
    fpath   = os.path.join(outdir, fname)
    ensure_dir(fpath)

    logging.info("Saving numpy data cube...")
    logging.info("  TO : {}.npz".format(fpath) + "\n")
    #np.save(fpath, (x,y,z,f)) # OLD: save to *.npy file
    np.savez(fpath, x=x, y=y, z=z, t=t)

def load_cube(params):
    fpath = "%s/%s.npz" % (params['output_folder'], params['fname_tcube'])
    x, y, z, f = np.load(fpath)

def save_powersph(k, p, perr, pnoise, nmodes, fres, params):
    try:
        fpath = "%s/%s.dat" % (params['output_folder'], params['fname_powerspectrum'])
    except KeyError:
        fpath = "%s/pspec.dat" % (params['output_folder'])

    header = "k[Mpc^-1]  P(k)[uK^2 Mpc^3]  sigmaP(k)[uK^2 Mpc^3]  Pnoise(k)[uK^2 Mpc^3]  Nmodes(k)  fres(k)"

    logging.info("Saving power spectrum...")
    logging.info("  TO : {}".format(fpath))
    np.savetxt(fpath, np.c_[k, p, perr, pnoise, nmodes, fres], header=header)

def save_powercyl(kprp, kpar, pcyl, params):
    fpath = "%s/%s_cyl.npz" % (params['output_folder'], params['fname_powerspectrum'])
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

def save_paramfile(params):
    fp_in    = params['param_file_path']

    outdir    = params['output_folder']
    fname    = os.path.basename(fp_in)
    fp_out    = os.path.join(outdir, fname)

    logging.info("Copying parameter file...")
    logging.info("  FROM : {}".format(fp_in))
    logging.info("    TO : {}".format(fp_out) + "\n")
    
    shutil.copyfile(fp_in, fp_out)

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
    outdir = params['output_folder']
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


###################
# DEPRECATED CODE #
###################
def save_powerspectrum(k, p, perr, pnoise, nmodes, fres, params):
    logging.warning("The method `save_powerspectrum` is deprecated!  Using `save_powersph` instead!")
    save_powersph(k, p, perr, pnoise, nmodes, fres, params)
