#!/usr/bin/env python

import numpy as np
import emcee
import triangle
import aipy as a



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
bnd = np.zeros(nprm, dtype=beta0.dtype) + r
bnd[-npol:] = ph_r
if fix_nf:
    x0 = np.zeros(nprm*(ndish-1), dtype=beta0.dtype) # as inital values
    x0[:nprm*center_dish] = beta0[:nprm*center_dish]
    x0[nprm*center_dish:] = beta0[nprm*(center_dish+1):nprm*ndish]
    x1 = np.zeros(nprm*(ndish-1), dtype=beta1.dtype) # the true values
    x1[:nprm*center_dish] = beta1[:nprm*center_dish]
    x1[nprm*center_dish:] = beta1[nprm*(center_dish+1):nprm*ndish]
    bnds = np.zeros(nprm*(ndish-1), dtype=beta0.dtype) # as bounds of values
    for i in range(ndish-1): # for center_dish == 15 only
        bnds[nprm*i:nprm*(i+1)] = bnd
else:
    x0 = np.zeros(nprm*(ndish-1)+3*num_nf, dtype=beta0.dtype)
    x0[:nprm*center_dish] = beta0[:nprm*center_dish]
    x0[nprm*center_dish:] = beta0[nprm*(center_dish+1):]
    x1 = np.zeros(nprm*(ndish-1)+3*num_nf, dtype=beta1.dtype)
    x1[:nprm*center_dish] = beta1[:nprm*center_dish]
    x1[nprm*center_dish:] = beta1[nprm*(center_dish+1):]
    bnds = np.zeros(nprm*(ndish-1)+3*num_nf, dtype=beta0.dtype)
    for i in range(ndish-1): # for center_dish == 15 only
        bnds[nprm*i:nprm*(i+1)] = bnd
    bnds[-3*num_nf] = r


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

# bounds of the values
# bnds = []
# for i in range(ndish-1): #NOTE: this only applies to center_dish = 15 case
#     for j in range(3):
#         bnds += [(x0[5*i+j] - r, x0[5*i+j] + r)]
#     for p in range(npol):
#         bnds += [(x0[5*i+3+p] - ph_r, x0[5*i+3+p] + ph_r)]
# if not fix_nf:
#     for n in range(num_nf):
#         for j in range(3):
#             bnds += [(x0[-3*(n+1)+j] - r, x0[-3*(n+1)+j] + r)]


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

# Set up the sampler.
ndim, nwalkers = len(x0), 500
pos = [x0 + bnds*rand0(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos, 1000, rstate0=np.random.get_state())
print("Done.")

# Make the triangle plot.
burnin = 50
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

# Compute the quantiles.
samples[:, 2] = np.exp(samples[:, 2])
x_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
print x_mcmc