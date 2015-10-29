"""Polarization utilities."""

import numpy as np


xy2s_m = 0.5 * np.array([[1., 0., 0., 1.],
                   [1., 0., 0., -1.],
                   [0., 1., 1., 0.],
                   [0., -1.j, 1.j, 0.]])

s2xy_m = np.linalg.inv(xy2s_m)

def stokes2xy(V_s):
    """Rotate a Stokes visibility to an XY visibility."""
    if type(V_s) == dict:
        try:
            V_s = np.array([V_s['I'],V_s['Q'],V_s['U'],V_s['V']])
            V_xy_arr = np.dot(s2xy_m,V_s)
            V_xy = {}
            for i,prm in enumerate(('xx','xy','yx','yy')):
                V_xy[prm] = V_xy_arr[i]
            return V_xy
        except(KeyError):
            print 'Label your data array differently!',V_s.keys()
            return None
    else: return np.dot(s2xy_m,V_xy)

def xy2stokes(V_xy):
    """Rotate an XY visibility into a Stokes' visibility."""
    if type(V_xy) == dict:
        try:
            V_xy = np.array([V_xy['xx'],V_xy['xy'],V_xy['yx'],V_xy['yy']])
            V_s_arr = np.dot(xy2s_m,V_xy)
            V_s = {}
            for i,prm in enumerate(('I','Q','U','V')):
                V_s[prm] = V_s_arr[i]
            return V_s
        except(KeyError):
            print 'Label your data array differently!',V_xy.keys()
            return None
    else: return np.dot(xy2s_m,V_xy)

def QU2p(V):
    """If you can't get an absolute polarization calibration, p = \sqrt{Q^2+U^2}/I may be useful. Do that transformation. Make sure input visibility is stored as a dictionary!!!"""
    V = normalizeI(V)
    try: V['p'] = np.sqrt(np.abs(V['Q'])**2 + np.abs(V['U'])**2)
    except(KeyError):
        V = xy2stokes(V)
        V['p'] = np.sqrt(np.abs(V['Q'])**2 + np.abs(V['U'])**2)
    return V

def normalizeI(V):
    """ Divide each visibility by Stokes' I."""
    try: I = V['I']
    except(KeyError):
        V_s = xy2stokes(V)
        I = V_s['I']
    for prm in V:
        V[prm] /= I
    return V


