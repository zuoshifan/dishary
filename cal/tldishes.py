import numpy as np
import aipy as ap
import utils as ut

# location of the antenna array
lat = '44:9:11.00'
lon = '91:48:23.00'
elev = 1504.3 # m

# (+E, +N) coordinates of each antenna in unit m
dishes_coord = np.loadtxt('16dishes_coord.txt')
center_coord = dishes_coord[7]
dishes_coord -= center_coord
# antenna positions
ant_pos_m = np.zeros((dishes_coord.shape[0], 3), dtype=dishes_coord.dtype)
ant_pos_m[:, :2] = dishes_coord
nants = ant_pos_m.shape[0]
m2ns = 100.0 / ap.const.c * 1.0e9 # c in unit cm
ant_pos_ns = m2ns * ant_pos_m
ant_pos_ns = np.dot(ut.xyz2XYZ_m(ut.latlong_conv(lat)), ant_pos_ns.T).T


prms = {
    'loc': (lat, lon, elev),
    'antpos': ant_pos_ns,
    'delays': [0.] * nants, # zero delays for all antennas
    'offsets': [0.] * nants, # zero offsets for all antennas
    'amps': [1.] * nants,
    'bp_r': [np.array([1.])] * nants,
    'bp_i': [np.array([0.])] * nants,
    'beam': ap.fit.Beam2DGaussian,
    'bm_xwidth': np.radians(4.0),
    'bm_ywidth': np.radians(4.0),
    'pointing': (0.0, ut.latlong_conv(lat), 0.0), # pointing to the North Pole, az (clockwise around z = up, 0 at x axis = north), alt (from horizon), also see coord.py
}

def get_aa(freqs):
    '''Return the AntennaArray to be used for simulation.'''
    beam = prms['beam'](freqs)
    try: beam.set_params(prms)
    except(AttributeError): pass
    # location = prms['loc']
    antennas = []
    pointing = prms['pointing']
    # nants = len(prms['antpos'])
    assert(len(prms['delays']) == nants and len(prms['offsets']) == nants and len(prms['bp_r']) == nants and len(prms['bp_i']) == nants and len(prms['amps']) == nants)
    for pos, dly, off, bp_r, bp_i, amp in zip(prms['antpos'], prms['delays'], prms['offsets'], prms['bp_r'], prms['bp_i'], prms['amps']):
        antennas.append(ap.fit.Antenna(pos[0],pos[1],pos[2], beam, phsoff=[dly, off], bp_r=bp_r, bp_i=bp_i, amp=amp, pointing=pointing))
    aa = ap.fit.AntennaArray(prms['loc'], antennas)
    return aa
