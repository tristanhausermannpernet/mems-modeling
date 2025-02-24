from numba import njit
import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

# Calculates the external excitation p(t) for the MEMS system
@njit(fastmath=True)
def calc_P(T, Vdc, Vac, OMEGA, t):
    return T * Vdc**2 + 2 * T * Vdc * Vac * np.cos(OMEGA * t)

# Calculates the internal nonlinear force
@njit(fastmath=True)
def calc_Fnl(T, Vdc, y):
    return -(3 * T * Vdc**2) * y**2 - (4 * T * Vdc**2) * y**3

# Calculates the derivative of the nonlinear force for the Newton-Raphson scheme
@njit(fastmath=True)
def calc_dFnl(T, Vdc, y):
    dFY = -6 * T * (Vdc**2) * (y + 2 * y**2)
    dFdY = 0
    return dFY, dFdY

# Newmark integration scheme to solve the MEMS equation of motion
@njit(fastmath=True)
def Newmark(Y0, dY0, t_init, dt, NT, omega0, T, Vdc, Vac, OMEGA, M, C, K):
    precNR = 1.e-12
    t = t_init
    Y, dY = Y0, dY0
    tt, Yt, dYt = zeros((NT, 1)), zeros((NT, 1)), zeros((NT, 1))
    
    P = calc_P(T, Vdc, Vac, OMEGA, t)
    Fnl = calc_Fnl(T, Vdc, Y)
    ddY = (P - C * dY - K * Y - Fnl) / M
    tt[0], Yt[0], dYt[0] = t, Y, dY

    for n in range(1, NT):
        # Time integration with Newton-Raphson correction
        t += dt
        iter = 0
        Y += dt * dY + (dt**2 / 2) * ddY
        dY += dt * ddY
        res = calc_P(T, Vdc, Vac, OMEGA, t) - M * ddY - C * dY - K * Y - calc_Fnl(T, Vdc, Y)
        normres = np.abs(res / P)

        while normres > precNR:
            iter += 1
            dFY, dFdY = calc_dFnl(T, Vdc, Y)
            J = (4 / dt**2) * M + (2 / dt) * (C + dFdY) + K + dFY
            deltaY = res / J
            Y += deltaY
            dY += (2 / dt) * deltaY
            ddY += (4 / dt**2) * deltaY
            res = calc_P(T, Vdc, Vac, OMEGA, t) - M * ddY - C * dY - K * Y - calc_Fnl(T, Vdc, Y)
            normres = np.abs(deltaY / Y)

        tt[n], Yt[n], dYt[n] = t, Y, dY

    return tt, Yt, dYt

# Calculates the response curve (steady-state amplitude vs excitation pulsation)
@njit(fastmath=True)
def compute_response_curve(T, Vdc, Vac, omega0, M, C, K, OMEGA_start, OMEGA_end, dOMEGA_step, nb_pts_per, nb_per, tolerance=0.00001):
    n_steps = int(abs((OMEGA_end - OMEGA_start) / dOMEGA_step) + 1)
    OME, AMPL = zeros((n_steps, 1)), zeros((n_steps, 1))
    Y0, dY0 = 0.25, 0.25
    k, OMEGA = 0, OMEGA_start

    while (dOMEGA_step > 0 and OMEGA <= OMEGA_end) or (dOMEGA_step < 0 and OMEGA >= OMEGA_end):
        OME[k] = OMEGA
        period = 2 * np.pi / OMEGA
        dt = period / nb_pts_per
        NT = nb_per * nb_pts_per
        tt, Yt, dYt = Newmark(Y0, dY0, 0, dt, NT, omega0, T, Vdc, Vac, OMEGA, M, C, K)
        AMPL[k] = max(Yt[-3 * nb_pts_per:])
        Y0, dY0 = Yt[-1, 0], dYt[-1, 0]
        OMEGA += dOMEGA_step
        k += 1
    
    val = -1000

    return OME[:k], AMPL[:k]

# Plots the response curve on a matplotlib axis
# Can display reference curve and zoom on region of interest
def plot_response_curve(OME, AMPL, OME2, AMPL2, OMEGA_data, AMPL_data, ax, delta_m=0, plot_reference=False, zoom = False):

    if plot_reference:
        ax.plot(OMEGA_data, AMPL_data, color='green', marker='o', label='Reference curve')

    ax.plot(OME, AMPL, marker='>', label=f'Frequency sweep up for {delta_m}')
    ax.plot(OME2, AMPL2, marker='<', label=f'Frequency sweep down for {delta_m}')
    plt.xlabel(r"$\Omega$ dimensionless excitation pulsation")
    plt.ylabel("Dimensionless response amplitude = $max(y(t))$")
    plt.title("Resonator Response Curve")
    plt.legend()
    plt.xlim(0.988,0.996)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    

# Initialize physical and dimensionless parameters of the system
@njit(fastmath=True)
def init_params(delta_m=0):
    rho, l, b, h, d = 2500, 250e-6, 40e-6, 1e-6, 0.03e-6
    Vdc, Vac = 5, 5 / 10
    epsilon0 = 8.5e-12
    f0 = 100e6
    omega0 = 2 * np.pi * f0
    Q = 1000
    xi = 1 / Q
    A = l * b
    m = rho * A * h
    T = (epsilon0 * A) / (2 * m * omega0**2 * d**3)
    M = 1 + (delta_m / m)
    C = xi
    K = 1 - 2 * T * Vdc**2
    return T, Vdc, Vac, omega0, M, C, K, d

# Load a response curve from a text file
def load_response_curve_data(path):
    # Loading response curve data
    OMEGA_data, AMPL_data = 0, 0
    
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    OMEGA_data, AMPL_data = data[:, 0], data[:, 1]
    return OMEGA_data, AMPL_data
