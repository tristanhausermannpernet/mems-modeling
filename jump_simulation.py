from newmark_jump import *
import numpy as np
import newmark_solver as nk
from numba import prange, njit

# Function to simulate amplitude jump during a pulsation sweep
@njit
def simulate_amplitude_jump(T, Vdc, Vac, omega0, M, C, K, OMEGA_sweep, OMEGA_min, OMEGA_max, nb_pts_per, dt, nb_per, t_init, NT, nb_newmark, delta_m):
    Y0 = np.array([0.0])  # Initial displacement
    dY0 = np.array([0.0]) # Initial velocity
    dtheta_list = np.zeros((nb_per*nb_pts_per*nb_newmark, 1))  # Stores pulsation evolution
    Yt_list = np.zeros((nb_per*nb_pts_per*nb_newmark, 1))      # Stores displacement evolution

    # First phase: without added mass
    for i in range(nb_newmark):
        tt, Yt, dYt, dtheta = Newmark(Y0[0], dY0[0], t_init[0], dt, NT, omega0, T, Vdc, Vac, OMEGA_min, OMEGA_max, M, C, K, OMEGA_sweep)
        t_init = tt[-1]
        Y0 = Yt[-1]
        dY0 = dYt[-1]
        for j in prange(len(Yt)):
            dtheta_list[j+i*(len(Yt)),0] = dtheta[j]
            Yt_list[j+i*(len(Yt)),0] = Yt[j,0]


    # Update initial conditions and parameters for the phase with added mass
    t_init = tt[-1]
    Y0 = Yt[-1]
    dY0 = dYt[-1]
    T, Vdc, Vac, omega0, M, C, K, d = nk.init_params(delta_m)

    dtheta_list_new = np.zeros((nb_per*nb_pts_per*nb_newmark*2,1))
    Yt_list_new = np.zeros((nb_per*nb_pts_per*nb_newmark*2,1))

    # Second phase: with added mass
    for i in range(nb_newmark*2):
        tt_new, Yt_new, dYt_new, dtheta_new = Newmark(Y0[0], dY0[0], t_init[0], dt, NT, omega0, T, Vdc, Vac, OMEGA_min, OMEGA_max, M, C, K, OMEGA_sweep)
        t_init = tt_new[-1]
        Y0 = Yt_new[-1]
        dY0 = dYt_new[-1]
        for j in prange(len(Yt)):
            dtheta_list_new[j+i*(len(Yt)),0] = dtheta_new[j]
            Yt_list_new[j+i*(len(Yt)),0] = Yt_new[j,0]

    return dtheta_list,Yt_list,dtheta_list_new,Yt_list_new

# Utility function to load response curves from text files
# and return them for display/comparison

def load_data_for_jump(delta_m):
    OME, AMPL = nk.load_response_curve_data('response_curves/response_curve_mass_0e+00_up.txt')
    OME2, AMPL2 = nk.load_response_curve_data('response_curves/response_curve_mass_0e+00_down.txt')
    OME_data, AMPL_data = nk.load_response_curve_data('response_curves/reference_curve.txt')
    OME3, AMPL3 = nk.load_response_curve_data(f'response_curves/response_curve_mass_{delta_m}_up.txt')
    OME4, AMPL4 = nk.load_response_curve_data(f'response_curves/response_curve_mass_{delta_m}_down.txt')
    # Warning: Ensure the file paths and names match exactly what is in the folder
    return OME, AMPL, OME2, AMPL2, OME_data, AMPL_data, OME3, AMPL3, OME4, AMPL4
