# mems-sensor-modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3572A5.svg)](https://www.python.org/)

Numerical simulation of a resonant MEMS nano-beam mass sensor. Models the
nonlinear electrostatic dynamics of a nano-beam and demonstrates two
independent detection strategies — **frequency-shift tracking** and
**amplitude-jump bifurcation** — for measuring deposited masses down to the
sub-femtogram scale.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Physics](#physics)
  - [Sensor Principle](#sensor-principle)
  - [Equation of Motion](#equation-of-motion)
  - [Non-dimensionalisation](#non-dimensionalisation)
- [Numerical Method](#numerical-method)
- [Detection Methods](#detection-methods)
  - [Method 1 — Frequency Shift](#method-1--frequency-shift)
  - [Method 2 — Amplitude Jump](#method-2--amplitude-jump)
- [Project Structure](#project-structure)
- [Pre-calculated Data](#pre-calculated-data)
- [Authors](#authors)
- [License](#license)

---

## Quick Start

**Prerequisites:** Python 3.8+

```bash
git clone <repository-url>
cd mems-sensor-modeling
pip install -r requirements.txt
python main.py
```

`main.py` runs three sequential demos (close each plot window to continue):

| # | Demo | What you see |
|---|------|--------------|
| 1 | Time Evolution | Beam tip displacement: transient settling into steady-state oscillation |
| 2 | Response Curve | Hysteretic amplitude vs. frequency (sweep-up and sweep-down overlay) |
| 3 | Frequency Shift | Δf vs. added mass with a linear-regression calibration fit |

Each demo is also callable individually:

```python
import main

main.demo_time_evolution()   # time-series plot
main.demo_response_curve()   # hysteresis loop
main.demo_frequency_shift()  # calibration curve
```

> **First-run note:** Numba JIT-compiles the inner loops on first call.
> Expect a few seconds of compilation overhead before the first plot appears.

---

## Physics

### Sensor Principle

A thin conductive nano-beam is suspended between two electrodes.  A DC voltage
$V_{dc}$ sets a static deflection; an AC voltage $V_{ac}$ drives transverse
vibration at a tunable frequency.  When a small mass $\delta m$ lands on the
beam its natural frequency drops.

That shift is detectable in two independent ways:

1. **Directly** — the resonance peak moves to a lower frequency.
2. **Via nonlinear bistability** — the frequency at which the amplitude "jumps"
   between two stable branches also shifts, and this bifurcation point is
   *more sensitive* than the peak position itself.

Reference parameters used in the simulation (all in `newmark_solver.py ›
init_params`):

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Beam density | ρ | 2500 kg/m³ |
| Beam length | l | 250 µm |
| Beam width | b | 40 µm |
| Beam height | h | 1 µm |
| Electrode gap | d | 30 nm |
| DC voltage | $V_{dc}$ | 5 V |
| AC voltage | $V_{ac}$ | 0.5 V |
| Natural frequency | $f_0$ | 100 MHz |
| Quality factor | Q | 1000 |

### Equation of Motion

Single-degree-of-freedom mass-spring model driven by the nonlinear
electrostatic force:

$$m\,\ddot{x} + c\,\dot{x} + k\,x \;=\; F_e(x,\,t)$$

$$F_e(x,t) \;=\; \frac{\varepsilon_0\, A}{2}
\cdot \frac{\bigl[V_{dc} + V_{ac}\cos(\omega t)\bigr]^2}{(d - x)^2}$$

where $A = l \times b$ is the beam cross-section area.

### Non-dimensionalisation

Physical quantities span many orders of magnitude; rescaling keeps the
numerics well-conditioned and the solver parameter-free.

| Symbol | Definition | Meaning |
|--------|-----------|---------|
| $\tau$ | $\omega_0\, t$ | Dimensionless time |
| $\Omega$ | $\omega / \omega_0$ | Dimensionless excitation frequency |
| $y$ | $x / d$ | Dimensionless displacement ($y < 1$ is the physical range) |
| $M$ | $1 + \delta m\,/\,m$ | Dimensionless total mass |

Taylor-expanding the electrostatic force around $y=0$ yields the ODE the
solver actually integrates:

$$M\,\ddot{y} + C\,\dot{y} + K\,y + f_{nl}(y) \;=\; p(\tau)$$

| Term | Expression | Origin |
|------|-----------|--------|
| $f_{nl}(y)$ | $-3T\,V_{dc}^2\;y^2 \;-\; 4T\,V_{dc}^2\;y^3$ | Cubic nonlinearity from electrostatic force |
| $p(\tau)$ | $T\,V_{dc}^2 + 2T\,V_{dc}\,V_{ac}\cos(\Omega\tau)$ | Dimensionless excitation |
| $T$ | $\varepsilon_0 A\,/\,(2\,m\,\omega_0^2\,d^3)$ | Electrostatic coupling constant |
| $K$ | $1 - 2T\,V_{dc}^2$ | Effective dimensionless stiffness |
| $C$ | $1/Q$ | Dimensionless damping |

---

## Numerical Method

The nonlinear ODE is time-stepped with the **Newmark constant-acceleration
scheme**, corrected at every step with **Newton-Raphson** iteration
(convergence tolerance $10^{-12}$).  All inner loops — force evaluations, the
NR loop, and the full frequency sweep — are compiled to native code on first
call via **Numba** `@njit(fastmath=True)`, giving near-C throughput from pure
Python.

**Stepped sweep** (`newmark_solver.py`): the frequency $\Omega$ advances in
discrete increments.  The final state at each $\Omega$ is used as the initial
condition for the next step.  This naturally follows the hysteretic branch and
faithfully reproduces physical sweep behaviour without explicit
branch-tracking logic.

**Continuous sweep** (`newmark_jump.py`): for the jump-detection method a
single long integration is run with a sinusoidally varying excitation phase

$$\theta(t) = \Omega_c\,t + \frac{\Delta\Omega}{\Omega_{\text{sweep}}}
\cos\!\bigl(\Omega_{\text{sweep}}\,t\bigr)$$

so the instantaneous frequency sweeps smoothly between $\Omega_{\min}$ and
$\Omega_{\max}$ without restarting the integrator.

---

## Detection Methods

### Method 1 — Frequency Shift

The resonance-peak frequency $\Omega_{\max}$ decreases linearly with added
mass.  The simulation pipeline:

1. Runs a sweep-down for each $\delta m$ in the target range
   (`newmark_added_mass.py`).
2. Extracts $\Omega_{\max}$ from each curve (location of peak amplitude).
3. Computes $\Delta f = f_0\,\bigl|\Omega_{\max}(\delta m) -
   \Omega_{\max}(0)\bigr|$ and fits a linear calibration
   (`plot_delta_m_f.py`).

The linear $\Delta f$–$\delta m$ relationship makes this method directly
invertible: measuring the frequency shift gives the deposited mass in one
step.

### Method 2 — Amplitude Jump

For masses too small to produce a measurable peak shift, the nonlinear
jump bifurcation becomes the readout:

1. A slow continuous frequency sweep (see *Numerical Method* above) drives
   the system across the bistable region in a single integration.
2. At a critical frequency the amplitude jumps discontinuously between the
   low- and high-amplitude stable branches.
3. `jump_simulation.py` runs a **two-phase** simulation: first the bare beam
   sweeps through the jump, then $\delta m$ is deposited and the sweep
   continues.  The shift in the bifurcation frequency between the two phases
   is the mass signal.

This method is sensitive to smaller masses than Method 1 because the
bifurcation point moves more per unit mass than the resonance peak does.

---

## Project Structure

```
mems-sensor-modeling/
│
├── main.py                   # Entry point — interactive three-demo suite
├── newmark_solver.py         # Core Newmark + Newton-Raphson integrator,
│                             #   physical/dimensionless parameters,
│                             #   stepped-sweep response curve, CSV loader
├── newmark_added_mass.py     # Stepped sweep parameterised by added mass δm
├── newmark_jump.py           # Newmark variant with continuous sinusoidal sweep
├── jump_simulation.py        # Two-phase jump simulation (before / after mass)
├── plot_delta_m_f.py         # Δf vs δm calibration curve + linear regression
│
├── response_curves/          # Pre-calculated high-resolution CSV data
│
├── requirements.txt          # Python dependencies (numpy, matplotlib, numba)
├── .gitignore
├── LICENSE.txt
└── README.md                 # This file
```

---

## Pre-calculated Data

`response_curves/` ships with high-resolution sweeps generated at
$\Delta\Omega = 5 \times 10^{-5}$ and 500 periods per frequency step —
significantly finer than the interactive demos, which use coarser settings for
speed.  These files serve as ground-truth reference data for validation and
comparison.

Each file is a two-column CSV (frequency, amplitude) with a one-row header:

| File pattern | Content |
|---|---|
| `reference_curve.txt` | Single coarse reference sweep |
| `response_curve_mass_<M>_up.txt` | High-res sweep-up at added mass $M$ |
| `response_curve_mass_<M>_down.txt` | High-res sweep-down at added mass $M$ |

Available masses: `0e+00`, `1e-13`, `1e-14`, … , `1e-25` (full decade
ladder from 100 pg to 0.1 yg).

Load any file directly:

```python
from newmark_solver import load_response_curve_data

omega, amplitude = load_response_curve_data(
    "response_curves/response_curve_mass_1e-18_down.txt"
)
```

---

## Authors

- Thomas Bertrand
- Marc Doumit
- Taihani Chan
- Tristan Hausermann

**Supervisor:** Sebastien Baguet

---

## License

[MIT](LICENSE.txt)
