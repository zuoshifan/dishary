#!/usr/bin/env python

import os
# import time
# import ephem
import aipy as a
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.linalg import eigh
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# data_dir = '/home/zuoshifan/programming/python/21cmcosmology/dishary/example_cal/tldishes/Cas_20151110/'
# data_file = data_dir + '20151110225250_20151111030000_cal.npy'
# freq_file = data_dir + '20151110225250_20151111030000_xconj_freq.npy'
# time_file = data_dir + '20151110225250_20151111030000_xconj_time.npy'
# ants_file = data_dir + '20151110225250_20151111030000_xconj_ants.npy'


data_dir = '/home/zuoshifan/programming/python/21cmcosmology/dishary/example_cal/tldishes/Cas_20150921/'
data_file = data_dir + '20150922011000_20150922012800.npy'
freq_file = data_dir + '20150922011000_20150922012800_freq.npy'
time_file = data_dir + '20150922011000_20150922012800_time.npy'
ants_file = data_dir + '20150922011000_20150922012800_ants.npy'

output_dir = data_dir + 'output/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# bl_ind = 3
# pol_ind = 0

data = np.load(data_file) # pol: xx, yy, xy, yx
ants = np.load(ants_file)
freq = np.load(freq_file) # MHz
ts = np.load(time_file) # Julian date

data_phs2zen = np.zeros_like(data) # save data that phased to zenith
data_int_time = np.zeros(data.shape[1:], dtype=data.dtype) # save data integrage over time

pol_dict = {0: 'xx', 1: 'yy', 2: 'xy', 3: 'yx'}
npol = data.shape[2]
nt = len(ts)
nfreq = len(freq)
# cut central 80% of the data
# data[np.int(0.1*nt):np.int(0.9*nt)] = 0
# only early 10%
# data[np.int(0.1*nt):] = 0
# only late 10%
# data[:np.int(0.9*nt)] = 0

nants = len(ants)
bls = [(ants[i], ants[j]) for i in range(nants) for j in range(i, nants)]
nbls = len(bls)

def gauss(x, a, x0, sigma, b):
    return a * np.exp(-(x-x0)**2 / (2 * sigma**2)) + b

######------------------
for pol_ind in range(npol):
    for bl_ind in range(nbls):
        # # ignore auto-correlation
        # bl = bls[bl_ind]
        # if bl[0] == bl[1]:
        #     continue

        data_slice = data[:, bl_ind, pol_ind, :]

        # subtract the mean
        # data -= np.mean(data, axis=1)

        # freq fft
        data_slice_fft_freq = np.fft.fft(data_slice, axis=1)
        data_slice_fft_freq = np.fft.fftshift(data_slice_fft_freq, axes=1)
        # time fft
        data_slice_fft_time = np.fft.fft(data_slice, axis=0)
        data_slice_fft_time = np.fft.fftshift(data_slice_fft_time, axes=0)
        # freq and time fft
        data_slice_fft2 = np.fft.fft2(data_slice)
        data_slice_fft2 = np.fft.fftshift(data_slice_fft2)

        ########################
        # find max in time fft
        max_row_ind = np.argmax(np.abs(data_slice_fft_time), axis=0)
        data_slice_fft_time_max = np.zeros_like(data_slice_fft_time)
        for ci in range(len(freq)):
            data_slice_fft_time_max[max_row_ind[ci], ci] = data_slice_fft_time[max_row_ind[ci], ci]

        # ifft for time
        data_slice_new = np.fft.ifft(np.fft.ifftshift(data_slice_fft_time_max, axes=0), axis=0)

        # freq fft
        data_slice_new_fft_freq = np.fft.fft(data_slice_new, axis=1)
        data_slice_new_fft_freq = np.fft.fftshift(data_slice_new_fft_freq, axes=1)
        # time fft
        data_slice_new_fft_time = np.fft.fft(data_slice_new, axis=0)
        data_slice_new_fft_time = np.fft.fftshift(data_slice_new_fft_time, axes=0)
        # freq and time fft
        data_slice_new_fft2 = np.fft.fft2(data_slice_new)
        data_slice_new_fft2 = np.fft.fftshift(data_slice_new_fft2)


        # divide phase
        data_slice_dphs = data_slice / (data_slice_new / np.abs(data_slice_new))
        # save data after phas2 to zenith
        data_phs2zen[:, bl_ind, pol_ind, :] = data_slice_dphs

        # freq fft
        data_slice_dphs_fft_freq = np.fft.fft(data_slice_dphs, axis=1)
        data_slice_dphs_fft_freq = np.fft.fftshift(data_slice_dphs_fft_freq, axes=1)
        # time fft
        data_slice_dphs_fft_time = np.fft.fft(data_slice_dphs, axis=0)
        data_slice_dphs_fft_time = np.fft.fftshift(data_slice_dphs_fft_time, axes=0)
        # freq and time fft
        data_slice_dphs_fft2 = np.fft.fft2(data_slice_dphs)
        data_slice_dphs_fft2 = np.fft.fftshift(data_slice_dphs_fft2)

        # # Fit a Gaussian function to data_slice_dphs
        # data_slice_dphs_gauss_fit = np.zeros_like(data_slice_dphs)
        # data_slice_dphs_xgauss = np.zeros_like(data_slice_dphs)
        # for fi in range(nfreq):
        #     try:
        #         # data_slice_dphs_smooth = UnivariateSpline(ts, data_slice_dphs[:, fi].real, s=1)(ts)
        #         data_slice_dphs_smooth = data_slice_dphs[:, fi].real # maybe should try some smooth
        #         max_val = np.max(data_slice_dphs_smooth)
        #         max_ind = np.argmax(data_slice_dphs_smooth)
        #         # sigma = np.sum(np.sqrt((ts-ts[max_ind])**2 / (2 * np.log(np.abs(max_val / (data_slice_dphs[:, fi].real - 0.0)))))) / nt
        #         sigma = 1.0
        #         popt,pcov = curve_fit(gauss, ts, data_slice_dphs_smooth, p0=[max_val, ts[max_ind], sigma, 0.0]) # now only fit real part
        #         data_slice_dphs_gauss_fit[:, fi] = gauss(ts, *popt)
        #     except RuntimeError:
        #         print 'Error occured while fitting pol: %s, bl: (%d, %d), fi: %d' % (pol_dict[pol_ind], bls[bl_ind][0], bls[bl_ind][1], fi)
        #         # print data_slice_dphs[:, fi].real
        #         plt.figure()
        #         plt.plot(ts, data_slice_dphs[:, fi].real)
        #         plt.plot(ts, data_slice_dphs_smooth)
        #         plt.xlabel('t')
        #         figname = output_dir + 'data_slice_%d_%d_%s_%d.png' % (bls[bl_ind][0], bls[bl_ind][1], pol_dict[pol_ind], fi)
        #         plt.savefig(figname)
        #         # data_slice_dphs_gauss_fit[:, fi] = data_slice_dphs[:, fi].real
        #         data_slice_dphs_gauss_fit[:, fi] = data_slice_dphs_smooth
        # data_slice_dphs_xgauss = data_slice_dphs * data_slice_dphs_gauss_fit # assume int_time = 1 here

        # # integrate over time
        # data_slice_int_time = np.sum(data_slice_dphs_xgauss, axis=0)
        # data_int_time[bl_ind, pol_ind, :] = data_slice_int_time

        # integrate over time
        data_slice_int_time = np.sum(data_slice_dphs, axis=0)
        data_int_time[bl_ind, pol_ind, :] = data_slice_int_time

        # save data to file
        filename = output_dir + 'data_slice_%d_%d_%s.hdf5' % (bls[bl_ind][0], bls[bl_ind][1], pol_dict[pol_ind])
        with h5py.File(filename, 'w') as f:
            f.create_dataset('data_slice', data=data_slice)
            f.create_dataset('data_slice_fft_freq', data=data_slice_fft_freq)
            f.create_dataset('data_slice_fft_time', data=data_slice_fft_time)
            f.create_dataset('data_slice_fft2', data=data_slice_fft2)
            f.create_dataset('data_slice_fft_time_max', data=data_slice_fft_time_max)
            f.create_dataset('data_slice_new', data=data_slice_new)
            f.create_dataset('data_slice_new_fft_freq', data=data_slice_new_fft_freq)
            f.create_dataset('data_slice_new_fft_time', data=data_slice_new_fft_time)
            f.create_dataset('data_slice_new_fft2', data=data_slice_new_fft2)
            f.create_dataset('data_slice_dphs', data=data_slice_dphs)
            f.create_dataset('data_slice_dphs_fft_freq', data=data_slice_dphs_fft_freq)
            f.create_dataset('data_slice_dphs_fft_time', data=data_slice_dphs_fft_time)
            f.create_dataset('data_slice_dphs_fft2', data=data_slice_dphs_fft2)
            # f.create_dataset('data_slice_dphs_gauss_fit', data=data_slice_dphs_gauss_fit)
            # f.create_dataset('data_slice_dphs_xgauss', data=data_slice_dphs_xgauss)
            f.create_dataset('data_slice_int_time', data=data_slice_int_time)


# save data phased to zenith
with h5py.File(output_dir + 'data_phs2zen.hdf5', 'w') as f:
    f.create_dataset('data_phs2zen', data=data_phs2zen)
# save data integrate over time
with h5py.File(output_dir + 'data_int_time.hdf5', 'w') as f:
    f.create_dataset('data_int_time ', data=data_int_time)



#######################################################
# construct visiblity matrix
Vmat = np.zeros((2*nants, 2*nants), dtype=data.dtype)
for fi in range(nfreq):
    for i, ai in enumerate(ants):
        for j, aj in enumerate(ants):
            try:
                ind = bls.index((ai, aj))
                Vmat[2*i, 2*j] = data_int_time[ind, 0, fi] # xx
                Vmat[2*i+1, 2*j+1] = data_int_time[ind, 1, fi] # yy
                Vmat[2*i, 2*j+1] = data_int_time[ind, 2, fi] # xy
                Vmat[2*i+1, 2*j] = data_int_time[ind, 3, fi] # yx
            except ValueError:
                ind = bls.index((aj, ai))
                Vmat[2*i, 2*j] = data_int_time[ind, 0, fi].conj() # xx
                Vmat[2*i+1, 2*j+1] = data_int_time[ind, 1, fi].conj() # yy
                Vmat[2*i, 2*j+1] = data_int_time[ind, 2, fi].conj() # xy
                Vmat[2*i+1, 2*j] = data_int_time[ind, 3, fi].conj() # yx
    # Eigen decomposition
    s, U = eigh(Vmat)
    print 'Eig val:', s
    # G = U * np.sqrt(s)

err

#######################################################
# create the uv and uv-coverage matrix
res = 1 # resolution, unit: wavelength
max_wl = 200 # max wavelength
max_lm = 1.0 / res
size = (2 * max_wl / res) + 1
center = max_wl / res # the central pixel
uv = np.zeros((size, size), dtype=np.complex128)
uv_cov = np.zeros((size, size), dtype=np.float64)


src = 'cas'
cat = 'misc'
cal = 'tldishes'
# calibrator
srclist, cutoff, catalogs = a.scripting.parse_srcs(src, cat)
cat = a.cal.get_catalog(cal, srclist, cutoff, catalogs)
assert(len(cat) == 1), 'Allow only one calibrator'
s = cat.values()[0]
print 'Calibrating for source with',
print 'strength', s._jys,
print 'measured at', s.mfreq, 'GHz',
print 'with index', s.index

# array
aa = a.cal.get_aa(cal, 1.0e-3 * freq) # use GHz
for ti, t in enumerate(ts):
    aa.set_jultime(t)
    s.compute(aa)
    for bl_ind in range(len(bls)):
        i, j = bls[bl_ind]
        if i == j:
            continue
        us, vs, ws = aa.gen_uvw(i-1, j-1, src=s) # NOTE start from 0
        for fi, (u, v) in enumerate(zip(us.flat, vs.flat)):
            val = data[ti, bl_ind, pol_ind, fi]
            if np.isfinite(val):
                up = np.int(u / res)
                vp = np.int(v / res)
                uv_cov[center+vp, center+up] += 1.0
                uv_cov[center-vp, center-up] += 1.0 # append conjugate
                uv[center+vp, center+up] += val
                uv[center-vp, center-up] += np.conj(val)# append conjugate

uv_cov_fft = np.fft.ifft2(np.fft.ifftshift(uv_cov))
uv_cov_fft = np.fft.ifftshift(uv_cov_fft)
uv_fft = np.fft.ifft2(np.fft.ifftshift(uv))
uv_fft = np.fft.ifftshift(uv_fft)

# save data

with h5py.File('uv_img_late.hdf5', 'w') as f:
    f.create_dataset('uv_cov', data=uv_cov)
    f.create_dataset('uv', data=uv)
    f.create_dataset('uv_cov_fft', data=uv_cov_fft)
    f.create_dataset('uv_fft', data=uv_fft)
    f.attrs['max_wl'] = max_wl
    f.attrs['max_lm'] = max_lm


plt.figure(figsize=(13, 8))
plt.subplot(231)
extent = [-max_wl, max_wl, -max_wl, max_wl]
plt.imshow(uv_cov, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
plt.xlabel(r'$u$ / $\lambda$')
plt.ylabel(r'$v$ / $\lambda$')
plt.colorbar()
plt.subplot(232)
plt.imshow(uv.real, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
plt.xlabel(r'$u$ / $\lambda$')
plt.ylabel(r'$v$ / $\lambda$')
plt.colorbar()
plt.subplot(233)
plt.imshow(uv.imag, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
plt.xlabel(r'$u$ / $\lambda$')
plt.ylabel(r'$v$ / $\lambda$')
plt.colorbar()

plt.subplot(234)
extent = [-max_lm, max_lm, -max_lm, max_lm]
plt.imshow(uv_cov_fft.real, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
plt.xlabel(r'$l$')
plt.ylabel(r'$m$')
plt.colorbar()
plt.subplot(235)
plt.imshow(uv_fft.real, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
plt.xlabel(r'$l$')
plt.ylabel(r'$m$')
plt.colorbar()
plt.subplot(236)
plt.imshow(uv_fft.imag, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
plt.xlabel(r'$l$')
plt.ylabel(r'$m$')
plt.colorbar()



plt.savefig('uv_xx_all_late.png')
