import sys
import optparse
import aipy as a
import numpy as np
from scipy.optimize import minimize


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


# common
def rand0(size):
    """Return random numbers in [-1, 1)."""
    return 2.0 * (np.random.random(size) - 0.5)

def distance(pos1, pos2=np.zeros(3)):
    """Distance between two positions `pos1` and `pos2`."""
    r = pos1 - pos2
    return np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)

freq = 750.0 # MHz
k0 = 2 * np.pi * (1.0e6 * freq) / (0.01 * a.const.c)



# options
add_noise = False # add some noise to the simulated phase
sigma = 0.001 # sigma of the noise
fix_nf = True # fix the position of near-field sources
num_nf = 2 # number of near-field sources
npol = 2 # use only xx and yy
# r = 2 * np.pi
# r = 5
# r = 0.5
# r = 0.05
# r = 0.01
# r = 0.005
# r = 0.0001
r = 1e-8
# r = 0.0
# ph_r = 0.1
ph_r = 0.1
# ph_r = 2 * np.pi


########################################################################
# generate simulation data
ndish = 16
# the reference dish
center_dish = 15 # number start from 0
# baselines
bls = [(i, j) for i in range(ndish) for j in range(i+1, ndish)]
nbl = len(bls)

nprm = 3 + npol # number of parameters of each antenna
# phase of the measured visibilities
Phi = np.zeros((num_nf, npol, nbl))


# pos of the near field source, unit: m
if num_nf == 1:
    nf = np.array([1000.0, 1200.0, 100.0])
elif num_nf == 2:
    nf = np.array([1000.0, 1200.0, 100.0, -1200.0, 1000.0, 120.0])
elif num_nf == 3:
    nf = np.array([1000.0, 1200.0, 100.0, -1200.0, 1000.0, 120.0, 400.0, -500.0, 80.0])
elif num_nf == 4:
    nf = np.array([1000.0, 1200.0, 100.0, -1200.0, 1000.0, 120.0, 400.0, -500.0, 80.0, -450.0, -700.0, 90.0])
else:
    nf = 500.0 + 1000.0 * np.random.rand(3*num_nf)
# pos of dishes
antpos = np.zeros((ndish, 3))
antpos[:, :2] = np.loadtxt('/home/zuoshifan/programming/python/21cmcosmology/dishary/example_cal/tldishes/16dishes_coord.txt')[:ndish]
antpos -= antpos[center_dish]

# initial phase of each dish of the two pol
# phi = 2 * np.pi * rand0((ndish, npol)) # phase relative to the center dish, between (-2pi, 2pi)
phi = np.zeros((ndish, npol)) # initial value 0
phi[center_dish] = 0

# perturbed values
# pos perturbation in [-r, r] m
if fix_nf:
    nf1 = nf
else:
    nf1 = nf + r * rand0(3*num_nf)
dantpos = r * rand0((ndish, 3))
dantpos[center_dish] = 0
antpos1 = antpos + dantpos
# phase perturbation in [-r, r] rad
dphi = ph_r * rand0((ndish, npol))
# dphi = 2 * np.pi * rand0((ndish, npol))
dphi[center_dish] = 0
phi1 = np.fmod(phi + dphi, 2 * np.pi)

# compute phase of measurements
for n in range(num_nf):
    for ind, (i, j) in enumerate(bls):
        if n == 0:
            slc = slice(-3, None)
        else:
            slc = slice(-3*(n+1), -3*n)
        ri = distance(nf1[slc], antpos1[i])
        rj = distance(nf1[slc], antpos1[j])
        for p in range(npol):
            phii = phi1[i, p]
            phij = phi1[j, p]
            Phi[n, p, ind] = k0 * (ri - rj) + phii - phij
if add_noise:
    Phi += np.random.normal(0.0, sigma, (num_nf, npol, nbl))
Phi = np.mod(Phi, 2 * np.pi)

beta0 = np.zeros(nprm*ndish + 3*num_nf)
beta1 = np.zeros(nprm*ndish + 3*num_nf)
for i in range(ndish):
    beta0[nprm*i:nprm*i+3] = antpos[i]
    beta0[nprm*i+3:nprm*i+3+npol] = phi[i]
    beta1[nprm*i:nprm*i+3] = antpos1[i]
    beta1[nprm*i+3:nprm*i+3+npol] = phi1[i]
beta0[-3*num_nf:] = nf
beta1[-3*num_nf:] = nf1
#######################################################################



########################################################################
# solve parameters by minimization chi^2
# initial values
# beta_k[nprm*i:nprm*(i+1)] is for [xi, yi, zi, phixi, [phiyi if npol=2]] of antena i, beta_k[nprm*ndish:] are for near-field sources [x, y, z]
# beta_k = np.zeros(nprm*ndish + 3, dtype=np.float64)
if fix_nf:
    x0 = np.zeros(nprm*(ndish-1), dtype=beta0.dtype)
    x0[:nprm*center_dish] = beta0[:nprm*center_dish]
    x0[nprm*center_dish:] = beta0[nprm*(center_dish+1):nprm*ndish]
    x1 = np.zeros(nprm*(ndish-1), dtype=beta1.dtype)
    x1[:nprm*center_dish] = beta1[:nprm*center_dish]
    x1[nprm*center_dish:] = beta1[nprm*(center_dish+1):nprm*ndish]
else:
    x0 = np.zeros(nprm*(ndish-1)+3*num_nf, dtype=beta0.dtype)
    x0[:nprm*center_dish] = beta0[:nprm*center_dish]
    x0[nprm*center_dish:] = beta0[nprm*(center_dish+1):]
    x1 = np.zeros(nprm*(ndish-1)+3*num_nf, dtype=beta1.dtype)
    x1[:nprm*center_dish] = beta1[:nprm*center_dish]
    x1[nprm*center_dish:] = beta1[nprm*(center_dish+1):]


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

Niter = 100
precision = 1.0e-6
print 'True res:', chi2(x1)
print 'Inital res:', chi2(x0)
# print 'Nelder-Mead:', minimize(chi2, x0, method='Nelder-Mead', options={'disp': False, 'maxiter': Niter, 'return_all': False, 'maxfev': None, 'xtol': precision, 'ftol': precision})
# print 'Powell:', minimize(chi2, x0, method='Powell', options={'disp': False, 'maxiter': Niter, 'return_all': False, 'direc': None, 'maxfev': None, 'xtol': precision, 'ftol': precision})
# print 'CG:', minimize(chi2, x0, method='CG', options={'disp': False, 'gtol': precision, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': Niter, 'norm': np.Inf})
# print 'BFGS:', minimize(chi2, x0, method='BFGS', options={'disp': False, 'gtol': precision, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': Niter, 'norm': np.Inf})
# print 'Newton-CG:', minimize(chi2, x0, method='Newton-CG', options={'disp': False, 'xtol': precision, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': Niter})
# print 'TNC:', minimize(chi2, x0, method='TNC', options={'disp': False, 'minfev': 0, 'scale': None, 'rescale': -1, 'offset': None, 'gtol': -1, 'eps': 1e-08, 'eta': -1, 'maxiter': Niter, 'maxCGit': -1, 'mesg_num': None, 'ftol': -1, 'xtol': -1, 'stepmx': 0, 'accuracy': 0})
# print 'COBYLA:', minimize(chi2, x0, method='COBYLA', options={'iprint': 1, 'disp': False, 'maxiter': Niter, 'catol': 0.0002, 'rhobeg': 1.0})
# print 'dogleg:', minimize(chi2, x0, method='dogleg', options={})
# print 'trust-ncg:', minimize(chi2, x0, method='trust-ncg', options={})

res1 = minimize(chi2, x0, method='SLSQP', bounds=bnds)
print 'SLSQP:', res1
res2 = minimize(chi2, x0, method='L-BFGS-B', bounds=bnds)
print 'L-BFGS-B', res2
# res3 = minimize(chi2, x0, method='TNC', bounds=bnds)
# print res3
