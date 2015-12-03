import numpy as np

def line_luminosity(halos, log_m1=, log_pnorm=, p1=, p2=, log_gnorm=, gsigma=, scatter=):
    """Lco(M) parameterized as double power law + Gaussian (sharing the same central mass).

    Parameters
    ----------
    halos
    log_m1
    log_pnorm
    p1
    p2
    log_gnorm
    gsigma
    scatter

    Returns
    -------
    lco
    """
    m = halos.get_mass()
    log_m = np.log10(m)

    m1 = 10**log_m1
    norm_p, norm_g = 10**log_pnorm, 10**log_gnorm

    fpower = norm_p * (m/m1)**p1 * (1. + m/m1)**(p2-p1)
    fgauss = norm_g * np.exp( -0.5* ((log_m - log_m1)/gsigma)**2 )

    lco = fpower + fgauss

    return lco
