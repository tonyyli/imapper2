import numpy as np

def file_lines(fp):
    nlines = 0
    with open(fp, 'r') as f:
        for line in f:
            nlines += 1
    return nlines


sfrlo_fname = "sfr_suberr.npy"
sfrmid_fname = "sfr_middle.npy"
sfrhi_fname = "sfr_adderr.npy"

logm = np.linspace(7, 16, 46)

# Set up output arrays
nmass = logm.size
nsnap = file_lines("sfr_7.0.dat")
scale = np.loadtxt("sfr_7.0.dat", unpack=True)[0]
redshift = 1./scale - 1.

out_logm        = logm.copy()
out_z           = redshift.copy()
out_sfr_merr    = np.zeros((nmass, nsnap))
out_sfr_perr    = np.zeros((nmass, nsnap))
out_sfr_mean    = np.zeros((nmass, nsnap))

for i, lm in enumerate(logm):
    fname = "sfr_%.1f.dat" % (lm)

    try:
        scale, sfr, errup, errdown = np.loadtxt(fname, unpack=True)
        startsnap = nsnap - scale.size

        out_sfr_merr[i, startsnap:] = sfr - errdown
        out_sfr_perr[i, startsnap:] = sfr + errup
        out_sfr_mean[i, startsnap:] = sfr
    except:
        pass

# z needs to be monotonically increasing for later interpolation
out_z = out_z[::-1]
out_sfr_merr = out_sfr_merr[:, ::-1]
out_sfr_perr = out_sfr_perr[:, ::-1]
out_sfr_mean = out_sfr_mean[:, ::-1]

np.save(sfrlo_fname, (out_logm, out_z, out_sfr_merr))
np.save(sfrhi_fname, (out_logm, out_z, out_sfr_perr))
np.save(sfrmid_fname, (out_logm, out_z, out_sfr_mean))
