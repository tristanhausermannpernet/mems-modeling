# Import additional libraries
import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt
import time

import newmark_solver as nk
from numba import njit

# Computes the response curve (amplitude vs pulsation) for a MEMS with added mass
@njit(fastmath=True)
def compute_response_curve_mass(OMEGA_start, OMEGA_end, dOMEGA_step, delta_m=0, sweep_up=False, data = False):
    T, Vdc, Vac, omega0, M, C, K, d = nk.init_params(delta_m)  # Parameter initialization with added mass
    nb_pts_per, nb_per = 50, 500  # Numerical parameters
    if sweep_up:
        OME, AMPL = nk.compute_response_curve(T, Vdc, Vac, omega0, M, C, K, OMEGA_start, OMEGA_end, dOMEGA_step, nb_pts_per, nb_per)  # Upward curve

    # Frequency sweep down
    OME2, AMPL2 = nk.compute_response_curve(T, Vdc, Vac, omega0, M, C, K, OMEGA_end, OMEGA_start, -dOMEGA_step, nb_pts_per, nb_per)  # Downward curve

    OMEGA_MAX = OME2[np.argmax(AMPL2)]  # Extracting maximum amplitude
    return OMEGA_MAX, OME, AMPL, OME2, AMPL2
