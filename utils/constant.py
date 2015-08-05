import numpy as np

# Physical constants. All constants in SI units.
k_B = 1.380658e-23          # Boltzmann constant
c = 2.99792458e8            # Speed of light
Omega_m = 0.27              # Matter content
Hubble_h = 0.68             # Hubble constant h
arcmin = np.pi / (60*180)   # Arcminute in unit rad
deg2rad = np.pi / 180.0     # Degree to rad
rad2deg = 180.0 / np.pi     # Radian to degree
m2ns = 1.0e9 / c            # Light travel 1 meter in ns
ns2m = c / 1.0e9            # Light travel 1 ns in meter