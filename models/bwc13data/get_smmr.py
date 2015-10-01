import os
import glob
import numpy as np

"""
Repackage stellarmass-halomass data from BWC13 to numpy binary

Info on stellarmass-halomass data:
    Stellar mass to halo mass relations for central halos.  Filenames contain
    redshift information.  For files labelled "smmr_then," the halo mass column
    is the present-day halo mass; for files labelled just "smmr", the halo mass
    column is the halo mass at the specified redshift.  Columns are: Log10(Halo
    Mass), Log10(Stellar Mass / Halo Mass), Err_Up (dex), and Err_Down (dex).
"""


def main():
    folderpath = "./release-sfh_z0_z8_052913/smmr"
    fbasename = "c_smmr_z*"
    fullpath = os.path.join(folderpath, fbasename)
    flist = sorted(glob.glob(fullpath))

    z_all = []
    loghm_all = []
    logsm_all = []

    for fn in flist:
        loghm, logsm_o_hm, errup, errdn = np.loadtxt(fn, unpack=True)

        z       = z_from_fname(fn) * np.ones(loghm.size) # array of a single redshift
        logsm   = logsm_o_hm + loghm

        z_all.append(z)
        loghm_all.append(loghm)
        logsm_all.append(logsm)

    z_all = np.hstack(z_all)
    loghm_all = np.hstack(loghm_all)
    logsm_all = np.hstack(logsm_all)

    np.save("smmr", (z_all, loghm_all, logsm_all))


def z_from_fname(fname):
    s = os.path.basename(fname) # Ensure we're only using the file name
    scut = s.split('_z')[1]
    snum = scut[:scut.find('_')]
    return float(snum)

if __name__=='__main__':
    main()
