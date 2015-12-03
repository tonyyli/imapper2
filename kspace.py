import numpy as np
import logging
from numpy import fft as ft
import functions as fn
from scipy import integrate
from scipy import interpolate

"""
    Functions for k-space (power spectrum) calculations
"""

def real_to_powsph(f, xyz):
    """
    Convert real-valued 3D scalar field to a spherically averaged power spectrum, P(k).

    Parameters
    ----------
    f : array of float
        Values of function on 3D grid
    xyz : tuple of arrays
        1D Cartesian arrays of the x, y, and z points of the 3D grid 
        
    Returns
    -------
    k : array of float
        Radial points in k-space
    psph : array of float
        Spherically averaged power spectrum, P(k)
    """
    df       = _subtract_mean(f)
    p, kx,ky,kz = real_to_pow3d(df, xyz)
    k, psph     = f3d_to_fsphavg(p, kx, ky, kz)

    return k, psph


def real_to_powcyl(f, xyz):
    """Convert a real 3d function to cylindrically averaged power spectrum, P(ks, kpar).

    Parameters
    ----------
    f :
    x :
    y :
    z :

    Returns
    -------
    ks :
        perpendicular component of k (i.e. x and y)
    kpar :
        parallel component of k (i.e. z)
    pcyl :
    """
    df       = _subtract_mean(f)
    p, kx,ky,kz = real_to_pow3d(df, xyz)
    ks, kpar, pcyl = f3d_to_fcylavg(p, kx, ky, kz)

    return ks, kpar, pcyl


def real_to_xpowsph(f1, f2, xyz):
    """
    Convert 2 intensity maps to a spherically averaged cross power spectrum, P(k)
    """
    df1 = _subtract_mean(f1)
    df2 = _subtract_mean(f2)

    power, kx, ky, kz   = real_to_xpow3d(df1, df2, xyz)
    power = np.real(power) # Throw away imaginary components, because they'll spherically average out to 0 (assuming f1 and f2 are real-valued)

    k, powersph = f3d_to_fsphavg(power, kx, ky, kz)

    return k, powersph


def real_to_xpowcyl(f1, f2, xyz):
    df1 = _subtract_mean(f1)
    df2 = _subtract_mean(f2)

    power, kx, ky, kz   = real_to_xpow3d(df1, df2, xyz)
    power = np.real(power) # Throw away imaginary components, because they'll spherically average out to 0 (assuming f1 and f2 are real-valued)

    ks, kpar, pcyl = f3d_to_fcylavg(power, kx, ky, kz)

    return ks, kpar, pcyl


def real_to_pow3d(f, xyz):
    """Calculate a 3D power spectrum, given:
    -- a function defined on a 3D grid of evenly-spaced (x,y,z) points
    -- a 1D array of the sampled points in each dimension (x,y,z)
    """
    x, y, z = xyz

    ftrans, kx, ky, kz  = real_to_fourier(f, x, y, z)
    
    # "Power spectrum" = power spectral density = |FT|^2 / volume
    vol = np.abs( (x[-1]-x[0])*(y[-1]-y[0])*(z[-1]-z[0]) ) # Total volume
    power = np.abs(ftrans)**2./vol 
    
    return power, kx, ky, kz


def real_to_xpow3d(f1, f2, xyz):
    """Calculate a 3D cross power spectrum, given:
    -- two functions defined on a 3D grid of evenly-spaced (x,y,z) points
    -- a 1D array of the sampled points in each dimension (x,y,z)
    """
    x, y, z = xyz

    ftrans1, kx, ky, kz    = real_to_fourier(f1, x, y, z)
    ftrans2, kx, ky, kz    = real_to_fourier(f2, x, y, z)
    
    vol             = np.abs( (x[-1]-x[0])*(y[-1]-y[0])*(z[-1]-z[0]) ) # Volume
    xpower          = ftrans1 * np.conj(ftrans2) / vol # "Power spectrum" is power spectral density: |FT|^2 / volume

    return xpower, kx, ky, kz


def real_to_fourier(f, x, y, z):
    # Check that real-space grid spacing is all equal
    if not (_is_evenly_spaced(x) and _is_evenly_spaced(y) and _is_evenly_spaced(z)):
        raise ValueError('Sample points in real space are not evenly spaced.')
    dx = x[1]-x[0]  # Grid spacing
    dy = y[1]-y[0]
    dz = z[1]-z[0]

    ftrans  = ft.rfftn(f)

    # Wavenumber arrays
    kx        = 2*np.pi * ft.fftfreq(x.size, d=dx)
    ky        = 2*np.pi * ft.fftfreq(y.size, d=dy)
    kz        = 2*np.pi * ft.rfftfreq(z.size, d=dz) # Only last axis is halved in length when using numpy.fft.rfftn()

    # Normalize (convert DFT to continuous FT)
    ftrans *= dx*dy*dz

    return ftrans, kx, ky, kz


def f3d_to_fsphavg(f, x, y, z, bins=None, log=False):
    """
    Spherically average a function initially defined on a 3d grid
    """

    rsphbins   = xyz_to_rsphbins(x,y,z, bins=bins, log=log)

    rr      = np.sqrt( sum(xx**2 for xx in np.meshgrid(x,y,z, indexing='ij')) )        # 3d grid of r (distance from origin)
    gt0     = rr > 0 # selection for r>0 (do not include origin)
    fofr    = np.histogram(rr[gt0], bins=rsphbins, weights=f[gt0])[0] / np.histogram(rr[gt0], bins=rsphbins)[0]
    rmid    = _bin_midpoints(rsphbins)

    return rmid, fofr


def f3d_to_fcylavg(f, x, y, z, bins=None, log=False):
    """Cylindrically average a function defined on a 3D grid

    Parameters
    ----------
    f :
    x :
    y :
    z :
    bins : 
        Number of bins, in the form (ks, kpar)

    Returns
    -------
    
    """
    if bins == None:
        # Count the maximum number of gridpts outward from origin in x, y, or z, then subtract 1
        bins_prp = max([ max( np.count_nonzero(a>0), np.count_nonzero(a<0) ) for a in (x,y) ]) - 1
        bins_par = max( np.count_nonzero(z>0), np.count_nonzero(z<0) ) - 1.
        bins = (bins_prp, bins_par)

    # Get (r,z) cylindrical bin edges from (x,y,z) cartesian grid
    rcylbins    = xyz_to_rcylbins(x,y,z, bins=bins, log=log)

    xx,yy,zz = np.meshgrid(x, y, z, indexing='ij')
    rr  = np.sqrt( xx**2 + yy**2 )                                  # 3D grid of r (cylindrical radius from origin)
    nn  = np.histogram2d(rr.ravel(), zz.ravel(), bins=rcylbins)[0]  # Number of cells used to compute average
    favg= np.histogram2d(rr.ravel(), zz.ravel(), bins=rcylbins, weights=f.ravel())[0] / nn

    rmid_prp    = _bin_midpoints(rcylbins[0])
    rmid_par    = _bin_midpoints(rcylbins[1])

    return rmid_prp, rmid_par, favg


def xyz_to_rsphbins(x, y, z, bins=None, log=False):
    """Transform Cartesian grid (x,y,z) to spherically radial bins

    Parameters
    ----------
    x, y, z : 1D arrays
        cartesian grid arrays
    bins : int or None
        Number or radial bins.  If None, defaults to bin count such that dr = max(dx, dy, dz)

    Returns
    -------
    rsphbins : 1D array
        radial bin edges
    """
    # The lowest bin edge is r=0 and the highest bin edge is the largest x, y, or z coordinate value
    rmin    = 0
    rmax    = max(np.amax(np.abs(x)), np.amax(np.abs(y)), np.amax(np.abs(z)))
    
    # If the number of radial bins has not been specified, the bin width dr is the largest of (dx,dy,dz).
    if bins == None:
        dr = max( [ np.min(np.abs(q[ q!=0 ])) for q in (x,y,z) ] ) # Smallest nonzero, absolute x,y,z coordinate value
        bins = int( np.ceil((rmax-rmin)/dr) )
        rsphbins = np.linspace(0, bins*dr, bins+1)
    else:
        rsphbins    = np.linspace(rmin, rmax, bins+1)

    return rsphbins


def xyz_to_rcylbins(x, y, z, bins=None, log=False):
    """Transform Cartesian grid (x,y,z) to cylindrical bins (radial and z)
    
    Parameters
    ----------
    x :
    y :
    z : 
    bins :
    log :

    Returns
    -------
    rcylbins :
    """
    rmin = 0
    rmax = max(np.amax(np.abs(x)), np.amax(np.abs(y)))

    zmin = 0
    zmax = np.amax(np.abs(z))

    if bins == None:
        # Count the maximum number of gridpts outward from origin in x, y, or z, then subtract 1
        bins_prp = max([ max( np.count_nonzero(a>0), np.count_nonzero(a<0) ) for a in (x,y) ]) - 1
        bins_par = max( np.count_nonzero(z>0), np.count_nonzero(z<0) ) - 1.
        bins = (bins_prp, bins_par)

    rbins_prp = np.linspace(rmin, rmax, bins[0]+1)
    rbins_par = np.linspace(zmin, zmax, bins[1]+1)
    rcylbins = (rbins_prp, rbins_par)

    return rcylbins


################################################################################

def _is_evenly_spaced(arr):
    """Check if an array has evenly spaced elements (linear)"""
    cond = np.allclose((arr[1:]-arr[:-1]), (arr[1]-arr[0]))
    return cond


def _bin_midpoints(bin_edges):
    return 0.5*(bin_edges[:-1] + bin_edges[1:])

def _subtract_mean(f):
    return f - np.mean(f)
