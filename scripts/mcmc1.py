#!/usr/bin/env python

# try:
#     import cPickle as pickle
# except:
#     import pickle
# import pickle
import sys
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import emcee
from emcee.utils import MPIPool
# import triangle
import aipy as a


# Initialize the MPI-based pool used for parallelization.
pool = MPIPool(loadbalance=True)
# if not pool.is_master():
#     # Wait for instructions from the master process.
#     pool.wait()
#     sys.exit(0)

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
add_noise = True # add some noise to the simulated phase
sigma = 0.1 # sigma of the noise
fix_nf = False # fix the position of near-field sources
num_nf = 2 # number of near-field sources
npol = 2 # use only xx and yy
# r = 2 * np.pi
# r = 5
# r = 0.5
r = 0.03
# r = 0.01
# r = 0.005
# r = 0.0001
# r = 1e-8
# r = 0.0
nf_r = r
# ph_r = 0.1
# ph_r = 0.0001
ph_r = 2 * np.pi


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


x0, x1, bnds = None, None, None
beta0, beta1 = None, None
if pool.is_master():
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

    beta0 = np.zeros(3*(ndish + num_nf))
    beta1 = np.zeros(3*(ndish + num_nf))
    for i in range(ndish):
        beta0[3*i:3*(i+1)] = antpos[i]
        beta1[3*i:3*(i+1)] = antpos1[i]
    beta0[-3*num_nf:] = nf
    beta1[-3*num_nf:] = nf1
    #######################################################################


    ########################################################################
    # solve parameters by minimization chi^2
    # initial values
    # beta_k[nprm*i:nprm*(i+1)] is for [xi, yi, zi, phixi, [phiyi if npol=2]] of antena i, beta_k[nprm*ndish:] are for near-field sources [x, y, z]
    # beta_k = np.zeros(nprm*ndish + 3, dtype=np.float64)
    bnd = np.zeros(3, dtype=beta0.dtype) + r
    if fix_nf:
        x0 = np.zeros(3*(ndish-1), dtype=beta0.dtype) # as inital values
        x0[:3*center_dish] = beta0[:3*center_dish]
        x0[3*center_dish:] = beta0[3*(center_dish+1):3*ndish]
        x1 = np.zeros(3*(ndish-1), dtype=beta1.dtype) # the true values
        x1[:3*center_dish] = beta1[:3*center_dish]
        x1[3*center_dish:] = beta1[3*(center_dish+1):3*ndish]
        bnds = np.zeros(3*(ndish-1), dtype=beta0.dtype) # as bounds of values
        for i in range(ndish-1): # for center_dish == 15 only
            bnds[3*i:3*(i+1)] = bnd
    else:
        x0 = np.zeros(3*(ndish-1)+3*num_nf, dtype=beta0.dtype)
        x0[:3*center_dish] = beta0[:3*center_dish]
        x0[3*center_dish:] = beta0[3*(center_dish+1):]
        x1 = np.zeros(3*(ndish-1)+3*num_nf, dtype=beta1.dtype)
        x1[:3*center_dish] = beta1[:3*center_dish]
        x1[3*center_dish:] = beta1[3*(center_dish+1):]
        bnds = np.zeros(3*(ndish-1)+3*num_nf, dtype=beta0.dtype)
        for i in range(ndish-1): # for center_dish == 15 only
            bnds[3*i:3*(i+1)] = bnd
        bnds[-3*num_nf:] = r

Phi = pool.bcast(Phi, root=0)
x0 = pool.bcast(x0, root=0)
x1 = pool.bcast(x1, root=0)
beta0 = pool.bcast(beta0, root=0)
beta1 = pool.bcast(beta1, root=0)
bnds = pool.bcast(bnds, root=0)

ndiff = num_nf * (num_nf - 1) / 2
diff_Phi = np.zeros((ndiff, npol, nbl), dtype=Phi.dtype)
diff_inds = [(i, j) for i in range(num_nf) for j in range(i+1, num_nf)]
for ind, (i, j) in enumerate(diff_inds):
    diff_Phi[ind] = Phi[j] - Phi[i]
diff_Phi = np.mod(diff_Phi, 2*np.pi)

def chi2(x):

    def slc(n):
        if n == 0:
            return slice(-3, None)
        else:
            return slice(-3*(n+1), -3*n)

    def pos(i):
        if i == center_dish:
            return np.zeros(3)
        else:
            return x[3*i:3*(i+1)]

    tmp_Phi = np.zeros_like(diff_Phi)
    # for n in range(num_nf):
    for ni, (n1, n2) in enumerate(diff_inds):
        for ind, (i, j) in enumerate(bls):
            # position of the near-field source
            # NOTE: this only applies to center_dish = 15 case
            if fix_nf:
                nf1 = beta0[slc(n1)]
                nf2 = beta0[slc(n2)]
            else:
                nf1 = x[slc(n1)]
                nf2 = x[slc(n2)]
            ai = pos(i)
            aj = pos(j)
            # distance between two points
            # r = distance(nf)
            ri1 = distance(nf1, ai)
            rj1 = distance(nf1, aj)
            ri2 = distance(nf2, ai)
            rj2 = distance(nf2, aj)
            for p in range(npol):
                tmp_Phi[ni, p, ind] = k0 * ( (ri2 - rj2) - (ri1 - rj1) )
    tmp_Phi = np.mod(tmp_Phi, 2*np.pi)
    return np.sum((diff_Phi - tmp_Phi)**2)


# Define the probability function as likelihood * prior.
def lnprior(x):
    if (np.abs(x - x0) < bnds).all():
        return 0.0
    else:
        return -np.inf

def lnlike(x):
    return -0.5 * chi2(x)

def lnprob(x):
    lp = lnprior(x)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(x)

# pool = MPIPool(loadbalance=True)
if not pool.is_master():
    pool.wait()
    sys.exit(0)

# Set up the sampler.
ndim, nwalkers = len(x0), 1000
# ndim, nwalkers = len(x0), 150
pos = [x0 + bnds*rand0(ndim) for i in range(nwalkers)]
# Initialize the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# Clear and run the production chain.
print("Running MCMC...")
# Run 100 steps as a burn-in.
# burnin = 1
# pos, prob, state = sampler.run_mcmc(pos, burnin, rstate0=np.random.get_state())
pos, prob, state = sampler.run_mcmc(pos, 2000, rstate0=np.random.get_state())
# Reset the chain to remove the burn-in samples.
# sampler.reset()
# Starting from the final position in the burn-in chain, sample for 1000 steps.
# sampler.run_mcmc(pos, 20, rstate0=state)
print("Done.")

# Make the triangle plot.
# samples = sampler.chain
# lnprobs = sampler.lnprobability
# print lnprobs[1, 1]
# print sampler.get_lnprob(samples[1, 1])
# err

samples = sampler.chain.reshape((-1, ndim))
lnprobs = sampler.lnprobability.reshape(-1)
print 'Optimal res2 and x:'
print -2 * np.max(lnprobs)
print samples[np.argmax(lnprobs)]
print 'Real res2 and x:'
print chi2(x1)
print x1
print 'difference:'
print x1 - samples[np.argmax(lnprobs)]
# print chi2(samples[np.argmax(lnprobs)])
# print sampler.get_lnprob(samples[np.argmax(lnprobs)]) * (-2)

# import corner
# fig = corner.corner(samples[:, :5])
# fig.savefig("triangle.png")

# for ind, s in enumerate(samples):
#     if ind == 0:
#         r2 = chi2(s)
#         x = s
#     else:
#         tmp = chi2(s)
#         if tmp < r2:
#             r2 = tmp
#             x = s
# print r2
# print x
# print sampler.get_lnprob(x) * (-2)

# # Compute the quantiles.
# samples[:, 2] = np.exp(samples[:, 2])
# x_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                              zip(*np.percentile(samples, [16, 50, 84],
#                                                 axis=0)))
# print x_mcmc


# Close the processes.
pool.close()