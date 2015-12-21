import os
import logging
import pprint
import astropy.cosmology as ac

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

def get_params(param_file_path):
    """
    Get parameter values and return them in the form of a dictionary.

    Parameters
    ----------
    param_file_path : str
        path to parameter file

    Returns
    -------
    params : dict
    """

    try:
        config = configparser.ConfigParser(inline_comment_prefixes=(';',))
    except TypeError: # In case "inline_comment_prefixes" isn't a valid argument, i.e. using Python 2
        config = configparser.ConfigParser()

    config.read(param_file_path)

    # Get "raw" dictionaries from `config` object
    raw_params = dict(config.items('general'))
    raw_io_params = dict(config.items('io'))
    raw_cosmo_params = dict(config.items('cosmology'))
    raw_model_params = dict(config.items('model'))
    raw_obs_params = dict(config.items('observing'))

    raw_io_params['param_file_path'] = os.path.abspath(param_file_path) # Store parameter file path
    
    params = get_general_params(raw_params) # Convert "raw" config dictionary to "organized" dictionary `params`
    params['io'] = get_io_parameters(raw_io_params)
    params['cosmo'] = get_cosmology_parameters(raw_cosmo_params)
    params['model'] = get_model_parameters(raw_model_params)
    params['obs'] = get_observing_parameters(raw_obs_params)
    
    logging.info("---------- PARAMETER VALUES ----------")
    logging.info("======================================")
    logging.info("\n" + pprint.pformat(params, indent=4) + "\n")

    return params

def get_io_parameters(raw_params):
    io = {}
    
    io['lightcone_path'] = raw_params['lightcone_path']

    io['output_folder']         = raw_params['output_folder']
    io['fname_tcube']           = raw_params['fname_tcube']
    io['fname_powerspectrum']   = raw_params['fname_powerspectrum']
    try:
        io['save_tcube']        = is_true(raw_params, 'save_tcube')
    except KeyError:
        io['save_tcube']        = True

    io['param_file_path']       = raw_params['param_file_path']
    
    return io


def get_model_parameters(raw_params):
    model = {}
    model_parameters = {}

    for k in raw_params.keys():
        if k == 'model':
            model['name'] = raw_params[k]
        else:
            model_parameters[k] = float(raw_params[k])

    model['parameters'] = model_parameters
    return model


def get_general_params(raw_params):
    params = {} # Initialize parameter dictionary
    
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


def get_cosmology_parameters(raw_params):
    """
    Returns
    -------
    cosmo : astropy.cosmology object
        object containing cosmological parameters
    """
    omega_m0    = float(raw_params['omega_m'])    # Present-day matter density
    omega_l0    = float(raw_params['omega_l'])    # Present-day dark energy density
    omega_k0    = float(raw_params['omega_k'])    # Present-day spatial curvature density
    hubble_h0   = float(raw_params['h'])          # Present-day reduced Hubble constant: h0 = H0 / (100 km/s/Mpc)

    H0          = hubble_h0*100.
    cosmo       = ac.LambdaCDM(H0=H0, Om0=omega_m0, Ode0=omega_l0)
    
    return cosmo


def get_observing_parameters(raw_params):
    """
    Returns
    -------
    obs : dict
        observing parameters
    """
    obs = {}

    obs['tsys']       = float(raw_params['tsys'])
    obs['angres']     = float(raw_params['angres'])
    obs['dnu']        = float(raw_params['dnu'])
    obs['nulo']       = float(raw_params['nulo'])
    obs['nuhi']       = float(raw_params['nuhi'])
    obs['nfeeds']     = int(raw_params['nfeeds'])
    obs['dualpol']    = is_true(raw_params, 'dualpol')

    obs['fovlen'] = float(raw_params['fovlen'])
    obs['nhours'] = float(raw_params['nhours'])
    
    return obs


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


### FOR TESTING ###
if __name__=='__main__':
    import os, sys
    import pprint

    param_fp = sys.argv[1]
    print("")
    print("Testing %s on %s..." % (os.path.basename(__file__), param_fp))
    print("")
    pprint.pprint(get_params(param_fp))
