import numpy as np
import warnings

"""
    Class containing all halo properties
"""

class HaloList(object):
    def __setattr__(self, name, value):
        # Prevent assigning properties that have not been explicitly defined in __init__
        if not hasattr(self, name):
            warnings.warn("{:s} is not a default halo property.".format(name))
        object.__setattr__(self, name, value)


    def __init__(self, filename):
        # Get .npz file containing halo properties
        with np.load(filename) as data:
            x   = data['x']
            y   = data['y']
            z   = data['z']
            ra  = data['ra']
            dec = data['dec']
            m   = data['m']
            pid = data['pid']
            zcos= data['zcos']
            zlos= data['zlos']

        # For each property, assign values to appropriate field. Note: in order
        # to prevent additional attributes from being created dynamically, we
        # explicitly use object.__setattr__ for this task.
        object.__setattr__(self, 'x',    x)
        object.__setattr__(self, 'y',    y)
        object.__setattr__(self, 'z',    z)
        object.__setattr__(self, 'ra',   ra)
        object.__setattr__(self, 'dec',  dec)
        object.__setattr__(self, 'm',    m)
        object.__setattr__(self, 'pid',  pid)
        object.__setattr__(self, 'zcos', zcos)
        object.__setattr__(self, 'zlos', zlos)

        # Derived properties, to be calculated later
        object.__setattr__(self, 'sfr',  None)  # Star formation rate
        object.__setattr__(self, 'lum',  None)  # Line luminosity
        object.__setattr__(self, 'binidx',  None) # 1D (flattened) array of gridded bin indices
        
        object.__setattr__(self, 'sfr_base',  None)  # Star formation rate



