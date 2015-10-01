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
    - obins:  ang x ang x freq bins
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
        obins = observed_bins(abounds, da, fbounds, df)         # 3 arrays of length M+1, N+1, and P+1
        cbins   = observed_to_comoving_grid(obins, nurest, cosmo)   # 3 arrays of length M+1, N+1, and P+1
        tcube       = np.zeros( [a.size-1 for a in obins] )   # 3D array of shape MxNxP; initialize temperature in all cells to 0

        # Store grid arrays as object attributes
        # 
        # Note: in order to prevent additional attributes from being created
        # dynamically, we explicitly use object.__setattr__ for this task.
        object.__setattr__(self, 'obins', obins)
        object.__setattr__(self, 'cbins', cbins)
        object.__setattr__(self, 'tcube', tcube)


    def observed_cell_centers(self):
        xo, yo, zo = ( _bin_midpoints(bins) for bins in self.obins )
        return xo, yo, zo


    def comoving_cell_centers(self):
        xc, yc, zc = ( _bin_midpoints(bins) for bins in self.cbins )
        return xc, yc, zc


def observed_bins(abounds, da, fbounds, df):
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


def observed_to_comoving_grid(obins, nurest, cosmo):
    """
    Convert an observed (ang x ang x freq) grid to a comoving grid

    APPROXIMATION: Assume a single (mean) redshift for the entire cube, for converting angle to transverse comoving distance.

    Parameters
    ----------
    obins : tuple or list of arrays, length (M+1, N+1, P+1)
        bin edges of angular and frequency directions

    nurest : float
        observed (redshifted) frequency of mapped line at z=0

    cosmo : 
    """
    
    logging.info("Converting (ang x ang x frq) grid to comoving grid...")

    xa, ya, zf  = obins

    redshift        = nurest/zf - 1.
    redshift_avg    = np.mean(redshift)
    depth_coMpc     = fn.z_to_cmpc(redshift, cosmo)

    xc = fn.arcmin_to_cmpc(xa, redshift_avg, cosmo)
    yc = fn.arcmin_to_cmpc(ya, redshift_avg, cosmo)
    zc = np.linspace(np.min(depth_coMpc), np.max(depth_coMpc), depth_coMpc.size)

    logging.info("  x : (%8.2f to %8.2f arcmin) --> (%8.2f to %8.2f Mpc)" % (np.min(xa), np.max(xa), np.min(xc), np.max(xc)))
    logging.info("  y : (%8.2f to %8.2f arcmin) --> (%8.2f to %8.2f Mpc)" % (np.min(ya), np.max(ya), np.min(yc), np.max(yc)))
    logging.info("  z : (%8.2f to %8.2f GHz   ) --> (%8.2f to %8.2f Mpc)" % (np.min(zf), np.max(zf), np.min(zc), np.max(zc)))
    logging.info("")

    return xc, yc, zc


def observed_to_covolume_grid(obins, nurest, cosmo):
    """Convert angxangxfreq bins to grid of comoving cell volumes.

    Parameters
    ----------
    obins : 3-tuple of 1d arrays
        arrays of bin edges in (arcmin, arcmin, and [freq]) respectively

    nurest : float
        rest frame frequency of spectral line of interest

    cosmo : 

    Returns
    -------
    vol : 3d array
        3d array of volume elements, in Mpc^3
    """

    xo, yo, zo  = obins
    zred_bins   = nurest/zo - 1.            # 1d array of redshift bin edges, in depth direction
    zred_mid    = _bin_midpoints(zred_bins)        # 1d array of redshift bin midpoints, in depth direction

    zcobins = fn.z_to_cmpc(zred_bins, cosmo)    # Bin edges in z-direction [comoving Mpc]
    dzco    = _bin_lengths(zcobins)                # Bin lengths in z-direction [comoving Mpc]

    # Check that binning is equal in x and y directions
    assert np.allclose( (xo[1]-xo[0]), _bin_lengths(xo) )
    assert np.allclose( (yo[1]-yo[0]), _bin_lengths(yo) )

    dxa = xo[1]-xo[0] # single bin length, in x[arcmin]
    dya = yo[1]-yo[0] # single bin length, in y[arcmin]

    dxco = fn.arcmin_to_cmpc(dxa, zred_mid, cosmo)    # Bin lengths in x-direction, sampled along the z-direction [co-moving Mpc]
    dyco = fn.arcmin_to_cmpc(dya, zred_mid, cosmo)    # Bin lengths in y-direction, sampled along the z-direction [co-moving Mpc]

    dvco_1d = np.abs(dxco*dyco*dzco)
    vol = np.tile(dvco_1d, (xo.size-1, yo.size-1, 1)) # Create 3D array of comoving element volumes [Mpc^3]

    return vol


def observed_to_redshift_grid(obins, nurest):
    """
    Parameters
    ----------
    obins : 3-tuple of 1d arrays
        arrays of bin edges in (arcmin, arcmin, and [freq]) respectively

    nurest : float
        rest frame frequency of spectral line of interest

    Returns
    -------
    redshiftcube : 3d array
        3d array of redshifts (will only vary in depth, aka z, direction)
    """

    xo, yo, zo = obins
    zred_bins = nurest/zo - 1.            # 1d array of redshift bin edges, in depth direction
    zred_mid = _bin_midpoints(zred_bins)    # 1d array of redshift bin midpoints, in depth direction

    redshiftcube = np.tile(zred_mid, (xo.size-1, yo.size-1, 1)) # Create 3D array of redshifts

    return redshiftcube


################################################################################

def _bin_midpoints(bin_edges):
    return 0.5*(bin_edges[:-1] + bin_edges[1:])

def _bin_lengths(bin_edges):
    return bin_edges[1:] - bin_edges[:-1]
