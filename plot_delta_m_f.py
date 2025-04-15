import newmark_added_mass as nkm
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

# Calculates frequency shift (delta_f) for a list of added masses
@njit
def compute_delta_f(delta_m_list, OMEGA_start, OMEGA_end, dOMEGA_step, f0):
    OMEGA_ref = nkm.compute_response_curve_mass(OMEGA_start, OMEGA_end, dOMEGA_step)  # Reference without added mass
    delta_f_list = np.zeros(len(delta_m_list))
    for i in prange(len(delta_m_list)):
        mass = delta_m_list[i]
        OMEGA = nkm.compute_response_curve_mass(OMEGA_start, OMEGA_end, dOMEGA_step, mass)  # With added mass
        A = OMEGA[0]
        B = OMEGA_ref[0]
        delta_f_list[i] = f0 * np.abs((A - B)[0])  # Calculation of frequency shift
    return delta_f_list

# Plots the delta_m vs delta_f curve and linear regression
def plot_frequency_shift_vs_mass(delta_m_list, OMEGA_start,OMEGA_end,dOMEGA_step, f0):
    delta_m_arr = np.array(delta_m_list)
    delta_f_list = compute_delta_f(delta_m_arr, OMEGA_start, OMEGA_end, dOMEGA_step, f0)
    plt.plot(delta_m_list, delta_f_list, label='Real data')
    
    # Linear regression on calculated points
    m, b = np.polyfit(delta_m_list, delta_f_list, 1)
    y_line = m * delta_m_arr + b
    plt.plot(delta_m_list, y_line, '--r', label=f"Linear regression: y = {m:.2e}x + {b:.2e}")

# Plots the delta_m vs delta_f curve (log version)
def plot_frequency_shift_vs_mass_log(delta_m_list, OMEGA_start,OMEGA_end,dOMEGA_step, f0):
    delta_f_list = compute_delta_f(np.array(delta_m_list), OMEGA_start, OMEGA_end, dOMEGA_step, f0)
    plt.plot(delta_m_list, delta_f_list)
