import ConfigParser
import os
import logging
import pprint
import astropy.cosmology as ac

# OLD: without configparser
# def get_params(filepath, verbose=True):
#     """
#     Get parameter values and return them in the form of a dictionary.
# 
#     Parameters
#     ----------
#     filepath : str
#         path to parameter file
#     """
#     params      = None
#     raw_params  = _parse_paramfile(filepath)
#     check_deprecated(raw_params)
#     params      = _reorganize_params(raw_params)
# 
#     logging.info("---------- PARAMETER VALUES ----------")
#     logging.info("======================================")
#     logging.info("\n" + pprint.pformat(params, indent=4) + "\n")
#     
#     return params

def get_params(filepath):
    """
    Get parameter values and return them in the form of a dictionary.

    Parameters
    ----------
    filepath : str
        path to parameter file

    Returns
    -------
    params : dict
    """

    config = ConfigParser.SafeConfigParser()
    config.read(filepath)

    # Get "raw" dictionaries from `config` object
    raw_params = dict(config.items('test_header')) # Currently assuming 'test_header' is the only section header
    raw_params['param_file_path'] = os.path.abspath(filepath) # Store parameter file path
    raw_halotosfr_params = dict(config.items('halo_to_sfr'))
    raw_sfrtolco_params = dict(config.items('sfr_to_lco'))
    raw_instr_params = dict(config.items('instr'))
    raw_survey_params = dict(config.items('survey'))

    params = _reorganize_general_params(raw_params) # Convert "raw" config dictionary to "organized" dictionary `params`
    params['halo_to_sfr'] = _reorganize_halotosfr_params(raw_halotosfr_params)
    params['sfr_to_lco'] = _reorganize_sfrtolco_params(raw_sfrtolco_params)
    params['instr'] = get_instrument_parameters(raw_instr_params)
    params['survey'] = get_survey_parameters(raw_survey_params)
    
    logging.info("---------- PARAMETER VALUES ----------")
    logging.info("======================================")
    logging.info("\n" + pprint.pformat(params, indent=4) + "\n")

    return params


# def _parse_paramfile(fp):
#     """
#     Parse any parameter file, and return a dictionary.
# 
#     Any line in the file of the form
#         "key : value"
#     results in a dictionary entry of the form
#         raw_params['key'] = 'value'
#     """
#     raw_params = {}                                        # Initialize parameter dictionary
#     raw_params['param_file_path'] = os.path.abspath(fp) # Store parameter file path
# 
#     with open(fp) as f:
#         for line in f:
#             # Skip line if it's (1) empty or (2) a comment
#             if len(line.strip())==0:
#                 continue
#             elif line.lstrip()[0]=='#':
#                 continue
# 
#             line = line.split('#')[0] # Remove comments from end of line (if applicable)
#             tokens = line.split(':', 1)
# 
#             # If line is not properly formatted as a token, skip it
#             if len(tokens) < 2:
#                 continue
# 
#             tokens = [t.strip() for t in tokens]
#             key, val = tokens
#             raw_params[key] = val
#             
#     return raw_params

def _reorganize_halotosfr_params(raw_params):
    ### HALO TO SFR
    halo_to_sfr = {}
    halo_to_sfr['model'] = raw_params['model']
    if raw_params['model'] == 'BWC13_scatter':
        if 'dexscatter' in raw_params:
            halo_to_sfr['dexscatter'] = float(raw_params['dexscatter'])
    elif raw_params['model'] == 'powerlaw':
        halo_to_sfr['alpha']     = float(raw_params['powerlaw_alpha'])
        halo_to_sfr['norm']      = float(raw_params['powerlaw_norm'])
    return halo_to_sfr


def _reorganize_sfrtolco_params(raw_params):
    ### sfr TO LCO
    sfr_to_lco = {}
    sfr_to_lco['model'] = raw_params['model']
    if raw_params['model'] == 'powerlaw':
        sfr_to_lco['alpha']     = float(raw_params['powerlaw_alpha'])
        sfr_to_lco['norm']      = float(raw_params['powerlaw_norm'])
    elif raw_params['model'] == 'kennicutt_to_lirlco':
        sfr_to_lco['alpha']     = float(raw_params['alpha'])
        sfr_to_lco['beta']      = float(raw_params['beta'])
        sfr_to_lco['deltamf']   = float(raw_params['deltamf'])
    elif raw_params['model'] == 'kennicutt_to_lirlco_scatter':
        sfr_to_lco['dexscatter']= float(raw_params['dexscatter'])
        sfr_to_lco['alpha']     = float(raw_params['alpha'])
        sfr_to_lco['beta']      = float(raw_params['beta'])
        sfr_to_lco['deltamf']   = float(raw_params['deltamf'])
    elif raw_params['model'] == 'shortcut1':
        sfr_to_lco['dexscatter']= float(raw_params['dexscatter'])
        sfr_to_lco['alpha']     = float(raw_params['alpha'])
        sfr_to_lco['beta']      = float(raw_params['beta'])
        sfr_to_lco['deltamf']   = float(raw_params['deltamf'])
    return sfr_to_lco


def _reorganize_general_params(raw_params):
    """
    Reorganize parameters into a structured, nested format for easy use with intensity mapping code
    """
    params = {} # Initialize parameter dictionary

    ### LIGHTCONE HALO DATA
    params['lightcone_path'] = raw_params['lightcone_path']

    ### OUTPUT
    params['output_folder']         = raw_params['output_folder']
    params['fname_tcube']           = raw_params['fname_tcube']
    params['fname_powerspectrum']   = raw_params['fname_powerspectrum']
    params['param_file_path']       = raw_params['param_file_path']
    try:
        params['save_tcube']        = is_true(raw_params, 'save_tcube')
    except KeyError:
        params['save_tcube']        = True

    ### COSMOLOGY
    cosmo       = get_cosmology_parameters(raw_params)
    params['cosmo']         = cosmo

    ### MISC
    params['line_nu0']      = float(raw_params['line_nu0'])

    # Angular grid refinement
    try:
        params['angrefine'] = float(raw_params['angrefine'])
    except KeyError:
        params['angrefine'] = 10.
    
    # Redshift-space distortions
    try:
        params['enable_rsd'] = is_true(raw_params, 'enable_rsd')
    except KeyError:
        params['enable_rsd'] = False

    ### Halo cuts
    params['cut_mode']      = raw_params['cut_mode']
    try:
        params['minmass']   = float(eval(raw_params['minmass']))
    except KeyError:
        pass
    try:
        params['maxmass']   = float(eval(raw_params['maxmass']))
    except KeyError:
        pass

    ### Duty cycle
    try:
        params['fduty'] = float(eval(raw_params['fduty']))
    except KeyError:
        pass

    ### Shuffle masses
    try:
        params['shuffle_mass'] = is_true(raw_params, 'shuffle_mass')
    except KeyError:
        pass


    return params

def validate_params(params):
    """
    Validate parameters and correct if necessary.
    """

    # TODO: Check that all required parameters are present
    # TODO: Check that all present parameters are valid
    # TODO: If interactive:
    # - prompt to fill/replace missing/invalid values
    # - prompt to overwrite original paramfile or save separate version

    pass


### TODO: Getter methods (raw parameter dictionary as input) ###

def get_cosmology_parameters(raw_params):
    """
    Returns
    -------
    cosmo : astropy.cosmology object
        object containing cosmological parameters
    """
    omega_m0    = float(raw_params['cosmo_omega_m'])    # Present-day matter density
    omega_l0    = float(raw_params['cosmo_omega_l'])    # Present-day dark energy density
    omega_k0    = float(raw_params['cosmo_omega_k'])    # Present-day spatial curvature density
    hubble_h0   = float(raw_params['cosmo_h'])          # Present-day reduced Hubble constant: h0 = H0 / (100 km/s/Mpc)

    H0          = hubble_h0*100.
    cosmo       = ac.LambdaCDM(H0=H0, Om0=omega_m0, Ode0=omega_l0)
    
    return cosmo

def get_instrument_parameters(raw_params):
    """
    Returns
    -------
    instr : dict
        instrument parameters
    """
    instr = {}

    # If `preset` name is specified, ignore all other parameters and load preset values
    try:
        preset_name = raw_params['preset']
        preset_fp = "{}/parameters/preset/{}.param".format(os.path.dirname(os.path.realpath(__file__)), preset_name)
        preset_params = _parse_paramfile(preset_fp)
        instr['tsys']       = float(preset_params['tsys'])
        instr['angres']     = float(preset_params['angres'])
        instr['dnu']        = float(preset_params['dnu'])
        instr['nulo']       = float(preset_params['nulo'])
        instr['nuhi']       = float(preset_params['nuhi'])
        instr['nfeeds']     = int(preset_params['nfeeds'])
        instr['dualpol']    = is_true(preset_params, 'dualpol')
    except KeyError:
        instr['tsys']       = float(raw_params['tsys'])
        instr['angres']     = float(raw_params['angres'])
        instr['dnu']        = float(raw_params['dnu'])
        instr['nulo']       = float(raw_params['nulo'])
        instr['nuhi']       = float(raw_params['nuhi'])
        instr['nfeeds']     = int(raw_params['nfeeds'])
        instr['dualpol']    = is_true(raw_params, 'dualpol')

    return instr


def get_survey_parameters(raw_params):
    """
    Returns
    -------
    survey : dict
        survey parameters
    """
    survey = {}
    
    # If `preset` name is specified, ignore all other parameters and load preset values
    try:
        preset_name = raw_params['preset']
        preset_fp = "{}/parameters/preset/{}.param".format(os.path.dirname(os.path.realpath(__file__)), preset_name)
        preset_params = _parse_paramfile(preset_fp)
        survey['fovlen'] = float(preset_params['fovlen'])
        survey['nhours'] = float(preset_params['nhours'])
    except KeyError:
        survey['fovlen'] = float(raw_params['fovlen'])
        survey['nhours'] = float(raw_params['nhours'])

    return survey

def get_tsys(raw_params):
    # TODO: Unfinished
    pass

def is_true(raw_params, key):
    """Is raw_params[key] true? Returns boolean value.
    """
    sraw    = raw_params[key]
    s       = sraw.lower() # Make case-insensitive

    # Lists of acceptable 'True' and 'False' strings
    true_strings    = ['true', 't', 'yes', 'y', '1']
    false_strings    = ['false', 'f', 'no', 'n', '0']
    if s in true_strings:
        return True
    elif s in false_strings:
        return False
    else:
        logging.warning("Input not recognized for parameter: %s" % (key))
        logging.warning("You provided: %s" % (sraw))
        raise

def check_deprecated(raw_params):
    if 'line_j' in raw_params:
        logging.warning("`line_j` for CO lines is deprecated. Specify the exact rest-frame frequency in `line_nu0`.")
        if int(raw_params['line_j']) != 1:
            logging.warning("`line_j` is not 1, intensity map may not be properly calculated")


### FOR TESTING ###
if __name__=='__main__':
    import os, sys
    import pprint

    param_fp = sys.argv[1]
    print("")
    print("Testing %s on %s..." % (os.path.basename(__file__), param_fp))
    print("")
    pprint.pprint(get_params(param_fp))
