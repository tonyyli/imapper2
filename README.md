Python package for generating 3D intensity maps and power spectra.

This is code for research, so it may break and change often.  Nevertheless,
questions, suggestions, and feedback are always welcome
([tony.y.li@gmail.com](mailto:tony.y.li@gmail.com)).


# Overview

Line intensity mapping code.  Produces:

- A 3D data cube, with brightness temperature defined (in uK) for each cell.
  See below for more info on data cube format.
- Spherically averaged spatial power spectrum

Required Python modules:

- [NumPy](http://www.numpy.org), tested on v1.7.1.  Used for handling and
  manipulating arrays.
- [AstroPy](http://www.astropy.org), tested on v0.3.  Used for calculations that
  depend on cosmology (submodules used: `astropy.cosmology`,
  `astropy.constants`, `astropy.units`).


# Usage

All parameters should be specified in a single parameter file.  The executable
`mapit.py` takes the parameter file as an argument, generates mock intensity
mapping data, and writes the data cube and other outputs to the location
specified.  Command-line usage:

```shell
$ mapit.py /path/to/parameter_file.param
```


## Input

- **Parameter file**: 
Text file containing all parameter values.  More details on format and
parameter descriptions to come.

- **Halo lightcone** :
This is a NumPy `*.npz` file containing named arrays of all dark matter halo
properties in a "lightcone" volume.  The file location should be provided in
the parameter file, in the `lightcone_path` field.

Each array should have the same length (the total number of halos).  Currently,
the following arrays are expected:

- 'x'   : x-component of halo position [Mpc/h]
- 'y'   : y-component of halo position [Mpc/h]
- 'z'   : z-component of halo position [Mpc/h]
- 'ra'  : right ascension [deg]
- 'dec' : declination [deg]
- 'm'   : halo mass [Msun/h]
- 'pid' : parent halo ID (-1 for central halos)
- 'zcos' : cosmological redshift (i.e. without redshift space distortions)
- 'zlos' : line-of-sight redshift (i.e. *with* redshift space distortions)

Note that this lightcone file needs to be generated separately.


## Output

### Brightness temperature cube

The data cube is saved as a NumPy NPZ file and is loaded like so:

```numpy
import numpy as np
data = np.load("/path/to/data/cube.npz")
x, y, z, t = [ data[k] for k in ('x', 'y', 'z', 't') ]
```

`x`, `y`, and `z` are 1D numpy arrays.  `x` and `y` are in arcmin, while `z` is in GHz.

`t` is a 3D numpy array with units of uK.


### Power spectra

- Spherically averaged (1D)
- Cylindrically averaged (2D)


# Additional details

## Parameters

Descriptions for input parameters:

[needs updating]
