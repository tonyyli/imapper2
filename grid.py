"""
    Module containing IMGrid, a class for grid objects, as well as helper functions
"""

import numpy as np
import logging
import functions as fn

class IMGrid(object):
    """Object storing intensity mapping grids.
    
    This is useful for conserving memory in case repeated runs are necessary within the same volume (e.g. when sampling parameter space during an MCMC run).

    Contains the following attributes
    - angfreqbins:  ang x ang x freq bins
    - lumcube:      luminosities in each grid cell (not necessarily defined at creation)
    The binning in each should be equivalent (i.e. objects in the ijk-th bin of
    the ang-ang-freq grid should also be in the same ijk-th bin of the comoving
    grid).
    """
    def __setattr__(self, name, value):
        # Prevent assigning properties that have not been explicitly defined in __init__
        if not hasattr(self, name):
            raise NameError("%s is not (yet) an explicitly defined attribute.  Edit the 'grid' module directly." % name)
        object.__setattr__(self, name, value)

    def __init__(self, abounds, da, fbounds, df, nurest, cosmo):
        """
        Parameters
        ----------
        abounds
            (lo, hi) angular bounds of survey volume [arcmin]
        da
            voxel angular width [arcmin]
        fbounds
            (lo, hi) frequency bounds of survey volume [GHz]
        df
            voxel frequency width [GHz]
        nurest
            rest-frame line frequency [GHz]
        cosmo
            astropy.cosmology object
        """

        # Create grids
        angfreqbins = getangfreqbins(abounds, da, fbounds, df)         # 3 arrays of length M+1, N+1, and P+1
        comovbins   = angfreq_to_comov_grid(angfreqbins, nurest, cosmo)   # 3 arrays of length M+1, N+1, and P+1
        tcube       = np.zeros( [a.size-1 for a in angfreqbins] )   # 3D array of shape MxNxP; initialize temperature in all cells to 0

        # Store grid arrays as object attributes
        # 
        # Note: in order to prevent additional attributes from being created
        # dynamically, we explicitly use object.__setattr__ for this task.
        object.__setattr__(self, 'angfreqbins', angfreqbins)
        object.__setattr__(self, 'comovbins',   comovbins)
        object.__setattr__(self, 'tcube',       tcube)


def getangfreqbins(abounds, da, fbounds, df):
    """
    Get data cube bins in angular and frequency coordinates.

    The 3D data cube is assumed to have 2 angular (x,y) and 1 frequency (z)
    dimensions.  Units are arcmin for angle, GHz for frequency.

    Parameters
    ----------
    abounds
    da
    fbounds
    df


    Returns
    -------
    xbins : 1D array
        Angular bins edges in x direction [arcmin]

    ybins : 1D array
        Angular bins edges in y direction [arcmin]

    zbins : 1D array
        Frequency bins edges in z direction [GHz]
    """

    alo, ahi = abounds
    flo, fhi = fbounds

    # Calculate angular bins (assume square FOV so x and y bins are same)
    nabins = int((ahi-alo)/da + 0.5)
    abins  = np.linspace(alo, ahi, nabins+1)
    xbins  = abins.copy()
    ybins  = abins.copy()

    # Calculate frequency bins
    nfbins = int((fhi-flo)/df + 0.5)
    zbins  = np.linspace(flo, fhi, nfbins+1)

    return xbins, ybins, zbins 


def angfreq_to_comov_grid(angfreqbins, nurest, cosmo):
    """
    Convert a field on an (ang x ang x freq) grid to a 3D comoving grid

    Parameters
    ----------
    angfreqbins : tuple or list of arrays, length (M+1, N+1, P+1)
        bin edges of angular and frequency directions

    nurest : float
        observed (redshifted) frequency of mapped line at z=0

    cosmo : 
    """

    # APPROXIMATION: Assume a single (mean) redshift for the entire cube, convert angle to cMpc based on that
    
    logging.info("Converting (ang x ang x frq) grid to comoving grid...")

    xa, ya, zf  = angfreqbins

    redshift        = nurest/zf - 1.
    redshift_avg    = np.mean(redshift)
    depth_coMpc     = fn.z_to_comovingMpc(redshift, cosmo)

    xc = fn.arcmin_to_comovingMpc(xa, redshift_avg, cosmo)
    yc = fn.arcmin_to_comovingMpc(ya, redshift_avg, cosmo)
    zc = np.linspace(np.min(depth_coMpc), np.max(depth_coMpc), depth_coMpc.size) # TODO: interpolate f to make this more accurate!

    logging.info("  x : (%8.2f to %8.2f arcmin) --> (%8.2f to %8.2f Mpc)" % (np.min(xa), np.max(xa), np.min(xc), np.max(xc)))
    logging.info("  y : (%8.2f to %8.2f arcmin) --> (%8.2f to %8.2f Mpc)" % (np.min(ya), np.max(ya), np.min(yc), np.max(yc)))
    logging.info("  z : (%8.2f to %8.2f GHz   ) --> (%8.2f to %8.2f Mpc)" % (np.min(zf), np.max(zf), np.min(zc), np.max(zc)) + "\n" )

    return xc, yc, zc

