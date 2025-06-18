import numpy as np
import matplotlib.pyplot as plt
import newmark_solver as nk
import newmark_added_mass as nk_ma
import plot_delta_m_f as dmf
# import jump_simulation as js # Uncomment if you want to implement the jump simulation demo

def demo_time_evolution():
    print("--- Demo 1: Time Evolution of the Beam ---")
    
    # 1. Initialize parameters
    T, Vdc, Vac, omega0, M, C, K, d = nk.init_params()
    
    # Simulation parameters
    OMEGA = 0.99                 # Dimensionless excitation frequency
    Y0 = 0.5                     # Initial dimensionless position
    dY0 = 0.5                    # Initial dimensionless velocity
    
    period = 2 * np.pi / OMEGA
    nb_pts_per = 50
    dt = period / nb_pts_per
    nb_per = 1000 # Reduced from 2500 for faster execution in demo
    NT = nb_per * nb_pts_per
    
    print(f"Running Newmark simulation for OMEGA={OMEGA}...")
    tt, Yt, dYt = nk.Newmark(Y0, dY0, 0, dt, NT, omega0, T, Vdc, Vac, OMEGA, M, C, K)
    
    # Detect steady state (simplified logic)
    # We just plot the whole thing to show transient -> steady state
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tt/omega0, Yt*d)
    ax.set_title(f"Time Evolution (OMEGA={OMEGA})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (m)")
    ax.grid(True)
    
    print("Close the plot window to proceed to the next demo.")
    plt.show()

def demo_response_curve():
    print("\n--- Demo 2: Resonator Response Curve ---")
    
    T, Vdc, Vac, omega0, M, C, K, d = nk.init_params()
    
    OMEGA_start = 0.98
    OMEGA_end = 1.0
    dOMEGA_step = 0.0001 # Coarser step for faster demo (notebook used 0.00005)
    nb_pts_per = 50
    nb_per = 200 # Reduced for speed
    
    print("Computing frequency sweep UP...")
    OME, AMPL = nk.compute_response_curve(T, Vdc, Vac, omega0, M, C, K, OMEGA_start, OMEGA_end, dOMEGA_step, nb_pts_per, nb_per)
    
    print("Computing frequency sweep DOWN...")
    OME2, AMPL2 = nk.compute_response_curve(T, Vdc, Vac, omega0, M, C, K, OMEGA_end, OMEGA_start, -dOMEGA_step, nb_pts_per, nb_per)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    nk.plot_response_curve(OME, AMPL, OME2, AMPL2, None, None, ax, delta_m=0, plot_reference=False)
    ax.set_title("Response Curve (Sweep Up and Down)")
    
    print("Close the plot window to proceed to the next demo.")
    plt.show()

def demo_frequency_shift():
    print("\n--- Demo 3: Frequency Shift Measurement (Mass Detection) ---")
    
    # Define a range of masses to test (dimensionless or small values)
    # Based on the file names in response_curves, we have very small masses
    # e.g., 1e-18 kg to 1e-25 kg. 
    # Let's pick a few points.
    
    masses = [0, 1e-20, 5e-20, 1e-19]
    print(f"Calculating frequency shift for masses: {masses}")
    
    T, Vdc, Vac, omega0, M, C, K, d = nk.init_params()
    f0 = omega0 / (2*np.pi)
    
    OMEGA_start = 0.98
    OMEGA_end = 1.0
    dOMEGA_step = 0.0001
    
    # We use the function from plot_delta_m_f
    # Note: plot_frequency_shift_vs_mass plots directly.
    
    fig = plt.figure(figsize=(10, 6))
    dmf.plot_frequency_shift_vs_mass(masses, OMEGA_start, OMEGA_end, dOMEGA_step, f0)
    plt.title("Frequency Shift vs Added Mass")
    plt.xlabel("Added Mass (kg)")
    plt.ylabel("Frequency Shift (Hz)")
    plt.grid(True)
    
    print("Close the plot window to finish.")
    plt.show()

if __name__ == "__main__":
    print("========================================================")
    print("       MEMS Sensor Simulation - Main Demo")
    print("========================================================")
    
    try:
        demo_time_evolution()
        demo_response_curve()
        demo_frequency_shift()
        print("\nAll demos completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
