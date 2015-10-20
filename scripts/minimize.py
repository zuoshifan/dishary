import sys
import optparse
import aipy as a
import numpy as np
from scipy.optimize import minimize


# fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2


# cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
#         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
#         {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})


# # bnds = ((0, None), (0, None))
# bnds = ((0, 2), (0, 2))


# res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds, constraints=cons)
# # res = minimize(fun, (2, 0), method='COBYLA', bounds=bnds, constraints=cons)
# print res


o = optparse.OptionParser()
o.set_usage('%s [options] *.uv' % sys.argv[0].split('/')[-1].strip())
o.set_description(__doc__)
a.scripting.add_standard_options(o, ant=True, pol=True, chan=True, cal=True, dec=True)
o.add_option('-t', '--time', dest='time', default='all',
    help='Select which time sample to plot. Options are: "all" (default), "<time1 #>_<time2 #>" (a range of times to plot), or "<time1 #>,<time2 #>" (a list of times to plot).')
o.add_option('--time_axis', dest='time_axis', default='physical',
    help='Choose time axis to be integration/fringe index (index), or physical coordinates (physical).  Default is physical.')
o.add_option('-N','--Niter', dest='Niter', type='int', default=10,
    help='Maximum number of iterations. Default is 10')
o.add_option('--precision', dest='precision', type='float', default=1.0e-12,
    help='Numerical computation precision. Default is 1.0e-12.')
# o.add_option('-o', '--output_file', dest='output_file', default='gain.hdf5',
#     help='Output file name. Default is gain.hdf5.')
o.add_option('-v', '--verbose', dest='verbose', action='store_true',
    help='Print some info.')
opts, args = o.parse_args(sys.argv[1:])


def distance(pos1, pos2=np.zeros(3)):
    """Distance between two positions `pos1` and `pos2`."""
    r = pos1 - pos2
    return np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)


# r = 0.005
# r = 0.0001
r = 1e-8
# r = 0.0
# ph_r = 0.1
ph_r = 2 * np.pi

freq = 750.0 # MHz
k0 = 2 * np.pi * (1.0e6 * freq) / (0.01 * a.const.c)

fix_nf = True # fix the position of near-field sources
num_nf = 2 # number of near-field sources

ndish = 16
# the reference dish
center_dish = 15 # number start from 0
# baselines
bls = [(i, j) for i in range(ndish) for j in range(i+1, ndish)]
nbl = len(bls)

npol = 2 # use only xx and yy
nprm = 3 + npol # number of parameters of each antenna
# phase of the measured visibilities
# Phi = np.zeros((2, nbl))
Phi = np.load('Phi.npy')

# initial values
# beta_k[nprm*i:nprm*(i+1)] is for [xi, yi, zi, phixi, [phiyi if npol=2]] of antena i, beta_k[nprm*ndish:] are for near-field sources [x, y, z]
# beta_k = np.zeros(nprm*ndish + 3, dtype=np.float64)
beta0 = np.load('beta0.npy')
if fix_nf:
    x0 = np.zeros(nprm*(ndish-1), dtype=beta0.dtype)
    x0[:nprm*center_dish] = beta0[:nprm*center_dish]
    x0[nprm*center_dish:] = beta0[nprm*(center_dish+1):nprm*ndish]
else:
    x0 = np.zeros(nprm*(ndish-1)+3*num_nf, dtype=beta0.dtype)
    x0[:nprm*center_dish] = beta0[:nprm*center_dish]
    x0[nprm*center_dish:] = beta0[nprm*(center_dish+1):]


def chi2(x):
    tmp_Phi = np.zeros_like(Phi)
    for n in range(num_nf):
        for ind, (i, j) in enumerate(bls):
            # position of the near-field source
            # NOTE: this only applies to center_dish = 15 case
            if n == 0:
                slc = slice(-3, None)
            else:
                slc = slice(-3*(n+1), -3*n)
            if fix_nf:
                nf = beta0[slc]
            else:
                nf = x[slc]
            if i == center_dish:
                ai = np.zeros(3)
                phii = np.zeros(npol)
            else:
                ai = x[nprm*i:nprm*i+3]
                phii = x[nprm*i+3:nprm*i+3+npol]
            if j == center_dish:
                aj = np.zeros(3)
                phij = np.zeros(npol)
            else:
                aj = x[nprm*j:nprm*j+3]
                phij = x[nprm*j+3:nprm*j+3+npol]
            # distance between two points
            # r = distance(nf)
            ri = distance(nf, ai)
            rj = distance(nf, aj)
            for p in range(npol):
                tmp_Phi[n, p, ind] = k0 * (ri - rj) + phii[p] - phij[p]
    tmp_Phi = np.mod(tmp_Phi, 2*np.pi)
    return np.sum((Phi - tmp_Phi)**2)

bnds = []
for i in range(ndish-1): #NOTE: this only applies to center_dish = 15 case
    for j in range(3):
        bnds += [(x0[5*i+j] - r, x0[5*i+j] + r)]
    for p in range(npol):
        bnds += [(x0[5*i+3+p] - ph_r, x0[5*i+3+p] + ph_r)]
if not fix_nf:
    for n in range(num_nf):
        for j in range(3):
            bnds += [(x0[-3*(n+1)+j] - r, x0[-3*(n+1)+j] + r)]

res1 = minimize(chi2, x0, method='SLSQP', bounds=bnds)
print res1
res2 = minimize(chi2, x0, method='L-BFGS-B', bounds=bnds)
print res2
# res3 = minimize(chi2, x0, method='TNC', bounds=bnds)
# print res3
