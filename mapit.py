#!/usr/bin/env python

"""
    TESTING for using configparser module to load parameters (instead of own parameter file)
"""

# Standard modules
import sys
import argparse
import time
import logging

# Modules within this package
import functions as fn
import parameters # Changed from "import parameters"
import halolco
import imapping
import powerspectrum as pspec
import errorbars
import grid

def main():
    # Set up logging
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s  %(message)s', 
            datefmt='%Y-%d-%m %I:%M:%S %p')
    
    # Parse command line arguments
    args = get_args() 

    # From each parameter file, calculate intensity maps + other outputs
    for param_file_path in args.param_file:
        t0 = time.time()

        # Get parameters from parameter file
        params = parameters.get_params(param_file_path)

        # Create the grid on which brightness temperatures will be calculated
        imgrid = get_imgrid(params) # NEW
        
        # Load and pre-process halo data
        halos = get_halos(params)

        # Get halo SFRs, LCOs

        halos.lco = get_lco(halos, params)
        
        # Convert halo L_COs to a temperature cube
        imgrid.tcube = get_tcube(halos, imgrid, params)

        ##################################################
        # Write temperature cube to file
        xarr    = fn.bin_centers(imgrid.angfreqbins[0])
        yarr    = fn.bin_centers(imgrid.angfreqbins[1])
        zarr    = fn.bin_centers(imgrid.angfreqbins[2])
        tcube   = imgrid.tcube
        fn.save_cube(xarr, yarr, zarr, tcube, params)
        
        ##################################################
        # Calculate power spectra from temperature cube
        ksph, powsph        = get_powersph(imgrid)
        kprp, kpar, powcyl  = get_powercyl(imgrid)
        
        # Calculate error bars
        errsph, noise_power, nmodes, fres = get_powersph_errorbars(ksph, powsph, params)
        # TODO: get cylindrical power spectrum error bars

        # Save power spectrum (spherical 1D and cylindrical 2D)
        fn.save_powersph(ksph, powsph, errsph, noise_power, nmodes, fres, params)
        fn.save_powercyl(kprp, kpar, powcyl, params)
        
        ##################################################
        # Save copy of parameter file to output folder
        fn.save_paramfile(params)
        
        # Summarize timing
        t1 = time.time()
        
        tpass = t1-t0
        logging.info("Done!\n")

        logging.info("Total time                        : {:.4f}\n".format(tpass))


def get_imgrid(params):
    """Returns IMGrid object based on instrument and survey parameters"""

    # Get angular dimensions
    fovlen = params['survey']['fovlen']
    ares = params['instr']['angres'] # Units: arcmin
    angrefine = params['angrefine']
        # Factor by which to refine angular resolution for final temperature map.
        # This a fudge parameter, introduced so that the final 2D intensity maps
        # can look more realistically smoothed after binning.  (NOTE: Not needed
        # if we can find a fast way to do kernel smoothing over a large number of
        # points -- i.e., halos.)
    dang   = ares / angrefine

    # Get frequency dimensions
    nulo = params['instr']['nulo']    # Units: GHz
    nuhi = params['instr']['nuhi']    # Units: GHz
    dnu  = params['instr']['dnu']    # Units: GHz
    
    cosmo   = params['cosmo']
    nurest  = params['line_nu0']

    angrange = [0., fovlen]
    nurange = [nulo, nuhi]

    imgrid = grid.IMGrid(angrange, dang, nurange, dnu, nurest, cosmo)

    return imgrid

def get_halos(params):
    """Returns HaloList object (with selection cuts applied, etc)"""
    halos = fn.load_halos(params)
    halos = fn.cut_halos(halos, params)
    return halos


def get_lco(halos, params, reuse_sfr=False):
    """Get Lco for all halos"""

    ### HALOS TO SFR ###
    logging.info("Assigning SFR to halos...")
    model = params['halo_to_sfr']['model']

    hm      = halos.m
    hzcos   = halos.zcos

    # Model: mean SFR(M,z) of Behroozi+13
    if model == 'BWC13':
        logging.info("  Using SFR(M) relation of Behroozi, Wechsler, Conroy 2013")
        sfr = halolco.sfr_bwc13(hm, hzcos) # Behroozi, Wechsler, Conroy 2013

    # Model: -1sigma mean SFRs of Behroozi+13
    elif model == 'BWC13_lo':
        logging.info("  Using SFR(M) relation of Behroozi, Wechsler, Conroy 2013 using lower error bound")
        sfr = halolco.sfr_bwc13(hm, hzcos, mode='lo')

    # Model: +1sigma mean SFRs of Behroozi+13
    elif model == 'BWC13_hi':
        logging.info("  Using SFR(M) relation of Behroozi, Wechsler, Conroy 2013 using upper error bound")
        sfr = halolco.sfr_bwc13(hm, hzcos, mode='hi')

    # Model: Mean SFRs of Behroozi+13 with log scatter
    elif model == 'BWC13_scatter':
        logging.info("  Using SFR(M) relation of Behroozi, Wechsler, Conroy 2013 using halo-to-halo scatter within X dex")
        dexscatter = params['halo_to_sfr']['dexscatter']
        
        if reuse_sfr: # If requested, reuse calculated 'base' SFR
            if halos.sfr_base is not None:
                sfr = halolco.logscatter(halos.sfr_base, dexscatter)
            else:
                halos.sfr_base = halolco.sfr_bwc13(hm, hzcos)
                sfr = halolco.logscatter(halos.sfr_base, dexscatter)
        
        else: # Calculate scattered SFR 'from scratch'
            sfr = halolco.sfr_bwc13_scatter(hm, hzcos, scatter=dexscatter, mode='dex')

    # Model: Power law SFR(M)
    elif model == 'powerlaw':
        alpha = params['halo_to_sfr']['alpha']
        norm = params['halo_to_sfr']['norm']
        logging.info("  Using SFR(M) powerlaw relation: SFR = %.1e * M ^ %.2f" % (norm, alpha))
        sfr = halolco.powerlaw(hm, alpha, norm)

    elif model in ['righi08', 'visbal10', 'pullen13A', 'pullen13B', 'lidz11']:
        # Skip the halo-to-sfr step (use SFR=M). The LCO-M normalization is absorbed into the sfr-lco step.
        sfr = hm

    elif model == 'constant':
        sfr = np.ones(hm.size)
    
    halos.sfr = sfr
    

    ### SFR TO LCO ###
    logging.info("Converting SFR to CO Luminosity...")

    model = params['sfr_to_lco']['model']

    sfr = halos.sfr

    if model == 'powerlaw': # Simple power law, user-provided normalization and exponent
        alpha   = params['sfr_to_lco']['alpha']
        norm    = params['sfr_to_lco']['norm']
        logging.info("  Using LCO(SFR) powerlaw relation: LCO = %.1e * SFR ^ %.2f" % (norm, alpha))
        lco     = halolco.powerlaw(sfr, alpha, norm)

    elif model == 'kennicutt_to_lirlco': # Kennicutt98 SFR-LIR relation, followed by LIR-LCO power law (e.g. Carilli+Walter13, S4.5)
        logging.info("  Using LCO(SFR) relation derived from:")
        logging.info("  -- Linear LIR-SFR scaling (Kennicutt 1998)")
        logging.info("  -- LIR-LCO power law")

        # Convert SFR to L_IR (Kennicutt 1998)
        deltamf = params['sfr_to_lco']['deltamf']
        lir     = halolco.sfr_to_lir(sfr, deltamf=deltamf)    # SFR in Msun/yr, LIR in Lsun
        
        # Convert L_IR to LCO' (S4.5, Carilli & Walter 2013)
        alpha   = params['sfr_to_lco']['alpha']
        beta    = params['sfr_to_lco']['beta']
        lcop    = halolco.lir_to_lcoprime(lir, alpha=alpha, beta=beta) # LIR in Lsun, LCOprime in K km/s pc^2

        # Covert LCO' to LCO (S2.4, Carilli & Walter 2013)
        nurest  = params['line_nu0']
        lco     = halolco.lprime_to_l(lcop, nurest)    # LCOprime in K km/s pc^2, LCO in Lsun

    elif model == 'kennicutt_to_lirlco_scatter': # Power law between LCO and LIR with log scatter
        deltamf = params['sfr_to_lco']['deltamf']
        lir     = halolco.sfr_to_lir(sfr, deltamf=deltamf)    # SFR in Msun/yr, LIR in Lsun
        
        alpha   = params['sfr_to_lco']['alpha']
        beta    = params['sfr_to_lco']['beta']
        lcop    = halolco.lir_to_lcoprime(lir, alpha=alpha, beta=beta) # LIR in Lsun, LCOprime in K km/s pc^2

        # Covert LCO' to LCO (S2.4, Carilli & Walter 2013)
        nurest  = params['line_nu0']
        lco     = halolco.lprime_to_l(lcop, nurest)    # LCO' in K km/s pc^2, LCO in Lsun

        sigmalco = params['sfr_to_lco']['dexscatter']
        lco     = halolco.logscatter(lco, sigmalco)


    elif model in ['righi08', 'visbal10', 'pullen13A', 'pullen13B', 'lidz11']:
        lco = getattr(halolco, model)(hm)

    elif model == "constant":
        lco = np.ones(sfr.size)

    else:
        raise Exception("  LCO(SFR) relation not recognized!")

    return lco

def get_tcube(halos, imgrid, params):
    logging.info("---------- GENERATING TEMPERATURE CUBE ----------")
    logging.info("=================================================")

    # Get all needed quantities from objects that were passed into this method

    # Get parameters
    cosmo   = params['cosmo']
    nurest  = params['line_nu0'] # Rest-frame line frequency
    logging.info("Mapping line with rest frame frequency: %.1f GHz" % (nurest))

    # Get grid
    angfreqbins = imgrid.angfreqbins    # Angular and frequency bins for the grid 

    # Get halo properties
    hxa = halos.ra                  # halo x-coordinates, angular [arcmin]
    hya = halos.dec                 # halo y-coordinates, angular [arcmin]
    hzf = nurest/(halos.zlos+1.)    # halo z-coordinates, frequency [GHz]
    hlco = halos.lco                # halo CO luminosities [Lsun]
    if halos.binidx is None:
        halos.binidx = imapping.get_halo_cellidx(hxa, hya, hzf, angfreqbins) # cell indices (on final 3D intensity map) for each halo
    hbinidx = halos.binidx      # Halo bin indices

    # We have everything we need. Now bin the halos and get the luminosity cube, then temperature cube...
    lcube = imapping.lhalo_to_lcube(hxa, hya, hzf, hlco, angfreqbins, nurest, cosmo, hbinidx=hbinidx)
    tcube = imapping.lcube_to_tcube(lcube, angfreqbins, nurest, cosmo)
    return tcube

def get_powersph(imgrid):
    """Return k, P(k) for spherically averaged power spectrum"""
    xc      = fn.bin_centers(imgrid.comovbins[0])
    yc      = fn.bin_centers(imgrid.comovbins[1])
    zc      = fn.bin_centers(imgrid.comovbins[2])
    tcube   = imgrid.tcube
    ksph, powsph = pspec.real_to_powsph(tcube, xc, yc, zc)
    return ksph, powsph

def get_powercyl(imgrid):
    """Return kprp, kpar, P(kprp, kpar) for cylindrically averaged power spectrum"""
    xc      = fn.bin_centers(imgrid.comovbins[0])
    yc      = fn.bin_centers(imgrid.comovbins[1])
    zc      = fn.bin_centers(imgrid.comovbins[2])
    tcube   = imgrid.tcube
    kprp, kpar, powcyl = pspec.real_to_powcyl(tcube, xc, yc, zc)
    return kprp, kpar, powcyl

def get_powersph_errorbars(k, power, params):
    """
    Calculate the error bars on spherically-averaged P(k) (1-sigma uncertainty) as a function of k.

    This is a convenience method, which calls the internal method.

    Parameters
    ----------
    k : 1D array
        Values of k at which to calculate error bars on the power spectrum. [1/Mpc]
    power : 1D array
        Autopower spectrum at the corresponding values of k.  Should be the same length as `k`. [uK^2 Mpc^3]
    """
    # Necessary parameters
    tsys            = params['instr']['tsys']
    dnu             = params['instr']['dnu']
    nfeeds          = params['instr']['nfeeds']
    dualpol         = params['instr']['dualpol']
    tobs            = params['survey']['nhours']*3600.
    fovlen          = params['survey']['fovlen']
    fwhm            = params['instr']['angres']
    nu_min          = params['instr']['nulo']
    nu_max          = params['instr']['nuhi']
    nu0             = params['line_nu0']
    cosmo           = params['cosmo']
    
    # Dual polarization doubles the number of effective feeds
    if dualpol: nfeeds *= 2

    return errorbars.powersph_error(power, tsys, nfeeds, tobs, fovlen, fovlen, fwhm, nu_min, nu_max, dnu, nu0, k, cosmo)


def get_args():
    parser = argparse.ArgumentParser(description="Command line script to get intensity maps")

    parser.add_argument("-i", action="store_true",
                        help="Interactive: prompt to confirm all parameter values")
    parser.add_argument("param_file", nargs="+",
                        help="Parameter file")
    return parser.parse_args()

if __name__=="__main__":
    main()
else:
    print "Note: `mapit` module not being run as main executable."
