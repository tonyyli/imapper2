import numpy as np

def line_luminosity(halos, log_m1=12.64, log_pnorm=6.51, p1=1.42, p2=-0.98, scatter=0.3, log_mmin=10.):
    """Lco(M) parameterized as double power law + Gaussian (sharing the same central mass).

    Parameters
    ----------

    Returns
    -------
    lco
    """
    m = halos.m

    lco = double_power(m, log_m1, log_pnorm, p1, p2)

    lco *= 10.**np.random.normal(size=lco.size, scale=scatter) # Add scatter

    lco[m < 10**log_mmin] = 0. # Cut off halo luminosities below mass threshold

    return lco

def double_power(m, log_m1, log_pnorm, p1, p2):
    m1 = 10**log_m1
    norm_p = 10**log_pnorm
    lco = norm_p * (m/m1)**p1 * (1. + m/m1)**(p2-p1)

    return lco

