from numba import njit
import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt

# Calculates external excitation p(t) and phase derivative for the sweep
@njit(fastmath=True)
def calc_P(T, Vdc, Vac, OMEGA_min, OMEGA_max, t, OMEGA_sweep):
    OMEGA_c = (OMEGA_max + OMEGA_min)/2
    delta_OMEGA = (OMEGA_max - OMEGA_min)/2
    theta = OMEGA_c*t + (delta_OMEGA/OMEGA_sweep)*np.cos(OMEGA_sweep*t)
    dtheta = OMEGA_c - (delta_OMEGA)*(np.sin(OMEGA_sweep*t))
    return T * Vdc**2 + 2 * T * Vdc * Vac * np.cos(theta), dtheta

# Calculates internal nonlinear force
@njit(fastmath=True)
def calc_Fnl(T, Vdc, y):
    return -(3 * T * Vdc**2) * y**2 - (4 * T * Vdc**2) * y**3

# Calculates derivative of nonlinear force for Newton-Raphson
@njit(fastmath=True)
def calc_dFnl(T, Vdc, y):
    dFY = -6 * T * (Vdc**2) * (y + 2 * y**2)
    dFdY = 0
    return dFY, dFdY

# Newmark scheme adapted for frequency sweeping (variable OMEGA)
@njit(fastmath=True)
def Newmark(Y0, dY0, t_init, dt, NT, omega0, T, Vdc, Vac, OMEGA_min, OMEGA_max, M, C, K, OMEGA_sweep):
    precNR = 1.e-12
    t = t_init
    Y, dY = Y0, dY0
    tt, Yt, dYt = zeros((NT, 1)), zeros((NT, 1)), zeros((NT, 1))
    P, dtheta = calc_P(T, Vdc, Vac, OMEGA_min, OMEGA_max, t, OMEGA_sweep)
    Fnl = calc_Fnl(T, Vdc, Y)
    ddY = (P - C * dY - K * Y - Fnl) / M
    dtheta_list = []
    dtheta_list.append(dtheta)
    tt[0], Yt[0], dYt[0] = t, Y, dY
    
    for n in range(1, NT):
        # Time integration with Newton-Raphson correction
        t += dt
        Y += dt * dY + (dt**2 / 2) * ddY
        dY += dt * ddY
        P, dtheta = calc_P(T, Vdc, Vac, OMEGA_min, OMEGA_max, t, OMEGA_sweep)
        dtheta_list.append(dtheta)
        res = calc_P(T, Vdc, Vac, OMEGA_min, OMEGA_max, t, OMEGA_sweep)[0] - M * ddY - C * dY - K * Y - calc_Fnl(T, Vdc, Y)
        normres = np.abs(res / P)

        while normres > precNR:
            dFY, dFdY = calc_dFnl(T, Vdc, Y)
            J = (4 / dt**2) * M + (2 / dt) * (C + dFdY) + K + dFY
            deltaY = res / J
            Y += deltaY
            dY += (2 / dt) * deltaY
            ddY += (4 / dt**2) * deltaY
            res = calc_P(T, Vdc, Vac, OMEGA_min, OMEGA_max, t, OMEGA_sweep)[0] - M * ddY - C * dY - K * Y - calc_Fnl(T, Vdc, Y)
            normres = np.abs(deltaY / Y)

        tt[n], Yt[n], dYt[n] = t, Y, dY

    return tt, Yt, dYt, dtheta_list

# Calculates response curve for an excitation pulsation sweep
@njit(fastmath=True)
def compute_response_curve(T, Vdc, Vac, omega0, M, C, K, OMEGA_start, OMEGA_end, dOMEGA_step, nb_pts_per, nb_per,OMEGA_sweep, tolerance=0.00001):
    n_steps = int(abs((OMEGA_end - OMEGA_start) / dOMEGA_step) + 1)
    OME, AMPL = zeros((n_steps, 1)), zeros((n_steps, 1))
    Y0, dY0 = 0.25, 0.25
    k, OMEGA = 0, OMEGA_start

    while (dOMEGA_step > 0 and OMEGA <= OMEGA_end) or (dOMEGA_step < 0 and OMEGA >= OMEGA_end):
        OME[k] = OMEGA
        period = 2 * np.pi / OMEGA
        dt = period / nb_pts_per
        NT = nb_per * nb_pts_per
        tt, Yt, dYt, l_theta = Newmark(Y0, dY0, 0, dt, NT, omega0, T, Vdc, Vac, OMEGA, OMEGA, M, C, K,OMEGA_sweep)
        AMPL[k] = max(Yt[-3 * nb_pts_per:])
        Y0, dY0 = Yt[-1, 0], dYt[-1, 0]
        OMEGA += dOMEGA_step
        k += 1
    
    val = -1000
    
    return OME[:k], AMPL[:k]

# Plots the response curve on a matplotlib axis
# Can display reference curve

def plot_response_curve(OME, AMPL, OME2, AMPL2, OMEGA_data, AMPL_data, ax, delta_m=0, plot_reference=False):

    if plot_reference:
        ax.plot(OMEGA_data, AMPL_data, color='green', marker='o', label='Reference Curve')

    ax.plot(OME, AMPL, marker='>', label=f'frequency sweep up for {delta_m}')
    ax.plot(OME2, AMPL2, marker='<', label=f'frequency sweep down for {delta_m}')
    plt.xlabel(r"$\Omega$ excitation pulsation")
    plt.ylabel("Dimensionless amplitude response = $max(y(t))$")
    plt.title("Resonator Response Curve")
