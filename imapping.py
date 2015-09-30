import logging
import numpy as np

import functions

"""
Module for turning halo luminosities into a gridded intensity map.

1) Calculated binned halo luminosities on a gridded 3D "cube" (not necessarily equal sides)
2) Convert that luminosity cube to a temperature cube
"""

def lhalo_to_lcube(hxa, hya, hzf, hlco, angfreqbins, nurest, cosmo, hbinidx=None, verbose=True):
    """
    [placeholder]
    """
    logging.info("")
    logging.info("Cube dimensions:")
    logging.info("  x : %.2f to %.2f arcmin" % (np.min(angfreqbins[0]), np.max(angfreqbins[0])))
    logging.info("  y : %.2f to %.2f arcmin" % (np.min(angfreqbins[1]), np.max(angfreqbins[1])))
    logging.info("  z : %.2f to %.2f GHz" % (np.min(angfreqbins[2]), np.max(angfreqbins[2])))
    logging.info("")
    logging.info("In these coordinates, the space occupied by halos is:")
    logging.info("  x : %.2f to %.2f arcmin" % (np.min(hxa), np.max(hxa)))
    logging.info("  y : %.2f to %.2f arcmin" % (np.min(hya), np.max(hya)))
    logging.info("  z : %.2f to %.2f GHz" % (np.min(hzf), np.max(hzf)))
    logging.info("")
    logging.info("Binning halos and generating a luminosity cube...")
    logging.info("")

    # Bin up the halos (and reuse bin indices, in case this step needs to be repeated multiple times)
    xe, ye, ze = angfreqbins
    nxbins, nybins, nzbins = xe.size-1, ye.size-1, ze.size-1
    nbins = nxbins*nybins*nzbins
    try:
        # Try to calculate binned luminosity cube, assuming bin indices have been precalculated
        lcube = np.bincount(hbinidx, weights=hlco, minlength=nbins)[:nbins].reshape(nxbins, nybins, nzbins)
        # Note: `hbinidx` is a 1D index array of length (xe.size-1)*(ye.size-1)*(ze.size-1). `hbinidx[i]` is the index of the i-th halo in the data cube
    except ValueError:
        # If bin indices don't exist yet, calculate them, bin halos for the first time
        xidx    = np.digitize(hxa, xe)-1 # <0 or >nxbins-1 out of range; [0, nxbins-1] in range
        yidx    = np.digitize(hya, ye)-1
        zidx    = np.digitize(hzf, ze)-1
        hbinidx = np.ravel_multi_index((xidx, yidx, zidx), (nxbins, nybins, nzbins), mode='clip')

        # Handle halos outside of grid bounds (their bin index is nbins, i.e. one greater than the maximum)
        oobounds = (xidx < 0) | (xidx > nxbins-1) | (yidx < 0) | (yidx > nybins-1) | (zidx < 0) | (zidx > nzbins-1) 
        hbinidx[oobounds] = nbins

        # Calculate binned luminosity cube
        lcube = np.bincount(hbinidx, weights=hlco, minlength=nbins)[:nbins].reshape(nxbins, nybins, nzbins)

    return lcube


def lcube_to_tcube(lcube, angfreqbins, nurest, cosmo):

    # Now that we have the luminosity cube, convert it to a temperature cube
    logging.info("Converting luminosity cube to temperature (Rayleigh-Jeans) cube...")
    logging.info("")

    # Get volume and redshifts of cells (needed to calculate brightness temperature)
    dv          = functions.angfreqbins_to_vol(angfreqbins, nurest, cosmo) # cell volumes
    redshift    = functions.angfreqbins_to_redshiftcube(angfreqbins, nurest)

    return functions.lsun_to_uk(lcube, nurest, dv, redshift, cosmo)


def get_halo_cellidx(hxa, hya, hzf, angfreqbins):
    """
    Get the cell indices of each halo

    Parameters
    ----------
    hxa : 1D float array
        halo x-coordinates, angular units (RA)
    hya : 1D float array
        halo y-coordinates, angular units (Dec)
    hzf : 1D float array
        halo z-coordinates, frequency units (observed frequency)
    angfreqbins : list or tuple
        3 arrays of the bin edges in the x,y,z directions. RA, Dec, and
        frequency bin edges are given by ``angfreqbins[0,1,2]`` respectively.

    Returns
    -------
    cellidx :
        [UNFINISHED]
    """
    xe, ye, ze = angfreqbins
    #nxbins, nybins, nzbins = xe.size-1, ye.size-1, ze.size-1
    nxbins, nybins, nzbins = [e.size-1 for e in (xe, ye, ze)] # Number of cells along each direction
    nbins = nxbins*nybins*nzbins # Total number of cells in the 3D grid

    # Calculate halo bin indices
    xidx    = np.digitize(hxa, xe)-1 # <0 or >nxbins-1 out of range; [0, nxbins-1] in range
    yidx    = np.digitize(hya, ye)-1
    zidx    = np.digitize(hzf, ze)-1
    cellidx  = np.ravel_multi_index((xidx, yidx, zidx), (nxbins, nybins, nzbins), mode='clip')

    # Handle halos outside of grid bounds (their bin index is nbins, i.e. one greater than the maximum)
    oobounds = (xidx < 0) | (xidx > nxbins-1) | (yidx < 0) | (yidx > nybins-1) | (zidx < 0) | (zidx > nzbins-1) 
    cellidx[oobounds] = nbins

    return cellidx

