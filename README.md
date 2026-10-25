# Final-project-Introduction-to-Numerical-Algorithms
# The H1N1 Virus
# Andrea Moragon Dominguez (DSDIWT) and Almudena Menendez Jimenez (VR691J)
# Modelling an H1N1 Outbreak on a College Campus

**Course:** Introduction to Numerical Algorithms — Budapest University of Technology and Economics (BME)  
**Authors:**  
- Almudena Menendez Jimenez (Neptun: `VR691J`)  
- Andrea Moragon Dominguez (Neptun: `DSDIWT`)  

---

## 1. Overview & Objectives

This repository contains the numerical implementation, verification scripts, and technical documentation for modelling an Influenza A (H1N1) outbreak in a closed residential college campus of $N = 1500$ students over a 60-day horizon.

Unlike a classical SIR model with an exponentially distributed infectious period, we apply the **linear chain trick** by splitting the 9-day infectious period into nine sequential one-day compartments ($I_1, \dots, I_9$). This yields an $\text{Erlang}(9, 1)$ infectious-period distribution and an 11-dimensional nonlinear system of ordinary differential equations (ODEs).

The primary objectives of the project are:
1. **Model Formulation:** Construct an 11-compartment ODE model tracking each day of infection separately.
2. **Numerical Integration:** Solve the nonlinear ODE system using an adaptive Runge-Kutta method.
3. **Parameter Calibration:** Estimate the unknown transmission rate $\beta$ from the early outbreak observation that 10 students are infected on day 7.
4. **Numerical Verification:** Validate mass conservation, analytical final-size convergence, fifth-order accuracy, and step-size stability bounds.
5. **Control Strategies:** Evaluate and compare **quarantine**, **vaccination**, and a **combined intervention** strategy.

---

## 2. Numerical Methods Implemented

In accordance with the course requirements, the project integrates multiple numerical algorithms implemented in **Python 3.11** using `NumPy`, `SciPy`, and `Matplotlib`:

1. **Time Integration — Dormand-Prince RK45 (`scipy.integrate.solve_ivp`):**  
   An explicit embedded Runge-Kutta pair $\text{RK5(4)}$ with FSAL (First Same As Last) property and adaptive step-size control (`rtol = 1e-3`, `atol = 1e-6`). Dense output interpolation is used to evaluate states on the daily grid $t = 0, 1, \dots, 60$.
2. **Parameter Calibration — Bounded Brent Minimisation (`scipy.optimize.minimize_scalar`):**  
   Combines golden-section search and successive parabolic interpolation to find $\beta^* \in [10^{-7}, 10^{-3}]$ minimising the squared residual $J(\beta) = \left(\sum_{j=1}^9 I_j(7; \beta) - 10\right)^2$ without requiring analytical derivatives.
3. **Root-Finding & Stiff Reference Verification (`brentq` & `Radau`):**  
   - **Brent's Root Finder (`scipy.optimize.brentq`):** Used to solve the characteristic equation for the initial exponential growth rate $r$, the analytical final-size relation $S_\infty$, the critical quarantine eradication threshold $\alpha^*$, and refined calibration checks.
   - **Implicit Radau IIA (`method='Radau'`):** Used at tight tolerances (`rtol = atol = 1e-12`) to generate a high-precision reference solution for the work-precision convergence study.

---

## 3. Repository Structure

```text
├── README.md                          # Project overview, mathematical summary, and usage guide
├── docs/
│   ├── H1N1_report.pdf                # Full technical report (30 pages)
│   └── presentation.pdf               # Presentation slides
├── PROJECT/
│   ├── Z1_data_model.py               # Campus parameters (N=1500), time grid, initial conditions
│   ├── Z2_model.py                    # RHS ODE definitions & solve_ivp wrappers (base & extensions)
│   ├── main.py                        # Base model calibration, simulation, and main figures
│   ├── H1N1_Main_Dynamics.png         # S, Total Infected, and R trajectories
│   └── H1N1_Infection_Stages.png      # Progression across individual stages I_1 to I_9
├── PROJECT_QUARANTINE/
│   ├── Z1_data_model.py               # Parameters for quarantine simulation
│   ├── Z2_model.py                    # Quarantine ODE model (12 compartments)
│   ├── main_extension.py              # Baseline vs. 10% daily isolation comparison
│   └── H1N1_Quarantine_Comparison.png # Quarantine impact plot
├── PROJECT_VACCINATION/
│   ├── Z1_data_model.py               # Parameters for vaccination simulation
│   ├── Z2_model.py                    # Vaccination ODE model (12 compartments)
│   ├── main_vaccination.py            # Baseline vs. 1.5% daily vaccination comparison
│   └── H1N1_Vaccination_Comparison.png# Vaccination impact plot
└── analysis_verification.py           # Work-precision study, SIR comparison, sensitivity & combined strategy