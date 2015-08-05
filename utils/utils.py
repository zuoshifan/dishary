"""
Some useful functions used by other modules.
"""
import numpy as np
import constant as cst

def deg2str(deg):
    """
    Convert the deg in unit degree to string represent, e.g. '40:25:34.75'.
    Arguments:
    - `deg`: in unit degree.
    """
    deg = float(deg)
    int_deg = int(deg)
    minite = 60*(deg - int_deg)
    int_min = int(minite)
    second = 60*(minite - int_min)
    return str(int_deg) + ':' + str(int_min) + ':' + str(second)

def deg2hstr(deg):
    """
    Convert the deg in unit degree to string represent, but in format 'hh:mm:ss' e.g. '12:25:34.75'.
    Arguments:
    - `deg`: in unit degree.
    """
    hour = float(deg) * (24.0 / 360)
    return deg2str(hour)

def gen_name(i,j,max_i,max_j):
    """
    Generate a unique name for each Radio Baddies in a sky map.
    Arguments:
    - `i`: First axes index;
    - `j`: Second axes index;
    - `max_i`: max(i);
    - `max_j`: max(j).
    """
    assert i <= max_i and j <= max_j, "i must less equal than max_i, and j must less equal than max_j."
    name = ''
    for n in range(len(str(max_i)) - len(str(i))):
        name += '0'
    name += str(i)
    name += '_'
    for n in range(len(str(max_j)) - len(str(j))):
        name += '0'
    name += str(j)
    return name

def xyz2XYZ_m(lat):
    """
    Matrix of coordinates conversion through xyz to XYZ.
    xyz coord: z toward zenith, x toward East, y toward North, xy in the horizon plane;
    XYZ coord: Z toward north pole, X in the local meridian plane, Y toward East, XY plane parallel to equatorial plane.
    Arguments:
    - `lat`: latitude of the observing position, in unit radian.
    """
    sin_a, cos_a = np.sin(lat), np.cos(lat)
    zero = np.zeros_like(lat)
    one = np.ones_like(lat)
    mat =  np.array([[  zero,   -sin_a,   cos_a  ],
                     [   one,     zero,    zero  ],
                     [  zero,    cos_a,   sin_a  ]])
    if len(mat.shape) == 3: mat = mat.transpose([2, 0, 1])
    return mat

def latlong_conv(lat):
    """
    Covert the string represent latitude/longitude to radian.
    Arguments:
    - `lat`: string represent latitude
    """
    str_lat = lat.split(":")
    lat = 0.0
    for n in range(len(str_lat)):
        lat += float(str_lat[n])/(60.0**n)
    return lat*cst.deg2rad