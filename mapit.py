#!/usr/bin/env python

# Standard modules
import os.path
import sys
import shutil
import time
import logging
import importlib

# Modules within this package
import functions as fn
import parameters # Changed from "import parameters"
import imapping
import kspace 
import errorbars
import grid

def main():
    # Set up logging
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s  %(message)s', 
            datefmt='%Y-%d-%m %I:%M:%S %p')
    
    # Get parameters from the provided parameter file
    param_file_path = sys.argv[1]
    params = parameters.get_params(param_file_path)

    # From parameter file, calculate intensity map + other outputs
    t0 = time.time()

    imgrid = get_grid(params) # Create the grid object that will contain the 3D brightness cube
    halos = get_halos(params) # Load and pre-process halo data

    halos.lum = get_lum(halos, params) # Calculate halo line luminosities
    imgrid.tcube = get_tcube(halos, imgrid, params) # From halos, make brightness temperature cube

    ##################################################
    
    # Calculate power spectra from temperature cube
    ksph, psph        = get_powersph(imgrid)
    kprp, kpar, pcyl  = get_powercyl(imgrid)
    
    # Calculate error bars
    errsph, noise_power, nmodes, fres = get_powersph_errorbars(ksph, psph, params)
    # TODO: get cylindrical power spectrum error bars

    ##################################################
    
    # Write temperature cube and other stuff to file
    save_tcube(imgrid, params) 
    save_powersph(ksph, psph, errsph, noise_power, nmodes, fres, params)
    save_powercyl(kprp, kpar, pcyl, params)
    save_paramfile(params) # Copy parameter file to output folder
    
    ##################################################
    
    # Summarize timing
    t1 = time.time()
    tpass = t1-t0

    logging.info("Done!")
    logging.info("")
    logging.info("Total time                        : {:.4f}\n".format(tpass))


def get_grid(params):
    """Returns IMGrid object based on instrument and survey parameters"""

    # Get angular dimensions
    fovlen = params['obs']['fovlen']
    ares = params['obs']['angres'] # Units: arcmin
    angrefine = params['angrefine']
        # Factor by which to refine angular resolution for final temperature map.
        # This a fudge parameter, introduced so that the final 2D intensity maps
        # can look more realistically smoothed after binning.  (NOTE: Not needed
        # if we can find a fast way to do kernel smoothing over a large number of
        # points -- i.e., halos.)
    dang   = ares / angrefine

    # Get frequency dimensions
    nulo = params['obs']['nulo']    # Units: GHz
    nuhi = params['obs']['nuhi']    # Units: GHz
    dnu  = params['obs']['dnu']    # Units: GHz
    
    cosmo   = params['cosmo']
    nurest  = params['line_nu0']

    angrange = [0., fovlen]
    nurange = [nulo, nuhi]

    imgrid = grid.IMGrid(angrange, dang, nurange, dnu, nurest, cosmo)

    return imgrid


def get_halos(params):
    """Returns HaloList object
    """
    lc_path = params['io']['lightcone_path']
    cosmo   = params['cosmo']
    with_rsd = params['enable_rsd']

    halos = fn.load_halos(lc_path, cosmo, with_rsd)
    return halos


def get_lum(halos, params):
    """Get line luminosities for all halos"""

    model_name = params['model']['name']
    model_parameters = params['model']['parameters']

    model = importlib.import_module('models.{:s}'.format(model_name)) # `model` is a custom module that defines a function "line_luminosity"

    lum = model.line_luminosity(halos, **model_parameters)

    return lum

def get_tcube(halos, imgrid, params):
    logging.info("---------- GENERATING TEMPERATURE CUBE ----------")
    logging.info("=================================================")

    ### Get all needed quantities from objects that were passed into this method

    # Get parameters
    cosmo   = params['cosmo']
    nurest  = params['line_nu0'] # Rest-frame line frequency
    logging.info("Mapping line with rest frame frequency: %.1f GHz" % (nurest))

    # Get grid
    obins = imgrid.obins    # Angular and frequency bins for the grid 

    # Get halo properties
    hxa = halos.ra                  # halo x-coordinates, angular [arcmin]
    hya = halos.dec                 # halo y-coordinates, angular [arcmin]
    hzf = nurest/(halos.zlos+1.)    # halo z-coordinates, frequency [GHz]
    hlum = halos.lum                # halo CO luminosities [Lsun]
    if halos.binidx is None:
        halos.binidx = imapping.get_halo_cellidx(hxa, hya, hzf, obins) # cell indices (on final 3D intensity map) for each halo
    hbinidx = halos.binidx      # Halo bin indices

    ### We have everything we need. Now bin the halos and get the luminosity cube, then temperature cube...
    lcube = imapping.lhalo_to_lcube(hxa, hya, hzf, hlum, obins, nurest, cosmo, hbinidx=hbinidx)
    tcube = imapping.lcube_to_tcube(lcube, obins, nurest, cosmo)
    return tcube


def get_powersph(imgrid):
    """Return k, P(k) for spherically averaged power spectrum"""
    xc, yc, zc  = imgrid.comoving_cell_centers()
    tcube       = imgrid.tcube
    ksph, psph  = kspace.real_to_powsph(tcube, xc, yc, zc)
    return ksph, psph


def get_powercyl(imgrid):
    """Return kprp, kpar, P(kprp, kpar) for cylindrically averaged power spectrum"""
    xc, yc, zc  = imgrid.comoving_cell_centers()
    tcube       = imgrid.tcube
    kprp, kpar, pcyl = kspace.real_to_powcyl(tcube, xc, yc, zc)
    return kprp, kpar, pcyl


def get_powersph_errorbars(k, psph, params):
    """
    Calculate the error bars on spherically-averaged P(k) (1-sigma uncertainty) as a function of k.

    This is a convenience method, which calls the internal method.

    Parameters
    ----------
    k : 1D array
        Values of k at which to calculate error bars on the power spectrum. [1/Mpc]
    psph : 1D array
        Autopower spectrum at the corresponding values of k.  Should be the same length as `k`. [uK^2 Mpc^3]
    """
    # Necessary parameters
    tsys            = params['obs']['tsys']
    dnu             = params['obs']['dnu']
    nfeeds          = params['obs']['nfeeds']
    dualpol         = params['obs']['dualpol']
    tobs            = params['obs']['nhours']*3600.
    fovlen          = params['obs']['fovlen']
    fwhm            = params['obs']['angres']
    nu_min          = params['obs']['nulo']
    nu_max          = params['obs']['nuhi']
    nu0             = params['line_nu0']
    cosmo           = params['cosmo']
    
    # Dual polarization doubles the number of effective feeds
    if dualpol: nfeeds *= 2

    return errorbars.powersph_error(psph, tsys, nfeeds, tobs, fovlen, fovlen, fwhm, nu_min, nu_max, dnu, nu0, k, cosmo)


def save_tcube(imgrid, params):
    if params['io']['save_tcube'] == False:
        logging.info("Note: Not saving data cube")
        return

    outdir  = params['io']['output_folder']
    fname   = params['io']['fname_tcube']
    fpath   = os.path.join(outdir, fname)
    
    xo, yo, zo  = imgrid.observed_cell_centers()
    tcube       = imgrid.tcube

    fn.save_cube(fpath, xo, yo, zo, tcube)


def save_powersph(ksph, psph, errsph, noise_power, nmodes, fres, params):
    try:
        fpath = "%s/%s.dat" % (params['io']['output_folder'], params['io']['fname_powerspectrum'])
    except KeyError:
        fpath = "%s/pspec.dat" % (params['io']['output_folder'])

    fn.save_powersph(fpath, ksph, psph, errsph, noise_power, nmodes, fres)


def save_powercyl(kprp, kpar, pcyl, params):
    fpath = "%s/%s_cyl.npz" % (params['io']['output_folder'], params['io']['fname_powerspectrum'])

    fn.save_powercyl(fpath, kprp, kpar, pcyl)


def save_paramfile(params):
    fp_in    = params['io']['param_file_path']

    outdir    = params['io']['output_folder']
    fname    = os.path.basename(fp_in)
    fp_out    = os.path.join(outdir, fname)

    logging.info("Copying parameter file...")
    logging.info("  FROM : {}".format(fp_in))
    logging.info("    TO : {}".format(fp_out))
    logging.info("")
    
    shutil.copyfile(fp_in, fp_out)


if __name__=="__main__":
    main()
else:
    logging.info("Note: `mapit` module not being run as main executable.")
