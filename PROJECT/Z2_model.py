
import numpy as np
from scipy.integrate import solve_ivp

# -------------------------------------------------------------------------
# H1N1 model with 11 compartments (S, I_1, ..., I_9, R)
def modelo_h1n1(t, y, params):
    # Unpack states: S (index 0), I_1 to I_9 (indices 1 to 9), R (index 10)
    S = y[0]
    I_compartments = y[1:10]
    R = y[10]

    # Force non-negative states
    S = max(S, 0.0)
    I_compartments = np.maximum(I_compartments, 0.0)
    R = max(R, 0.0)

    # Unpack parameters (beta)
    beta = params[0]
    N = params[1]  # Fixed total population

    # Total sum of current infected individuals (from I_1 to I_9)
    total_infected = np.sum(I_compartments)

    # Force of infection
    # Rate proportional to the product of susceptibles and the sum of all infected
    force_infection = beta * S * total_infected

    # Derivatives
    dS = -force_infection

    # Transitions between infection compartments (1 day in each)
    # dI_1/dt = new infections - outflow from I_1
    dI_1 = force_infection - I_compartments[0]
    
    # dI_j/dt = inflow from previous compartment - outflow from current compartment (for j=2..9)
    dI_remaining = I_compartments[:-1] - I_compartments[1:]
    
    dI_vector = np.concatenate(([dI_1], dI_remaining))

    # dR/dt = individuals leaving the last day of infection (I_9)
    dR = I_compartments[8]

    # Return the list with the 11 derivatives
    return [dS, *dI_vector, dR]

# -----------------------------
# MODEL OUTPUT
# -----------------------------
def OUTPUT(params, lista_out):
    """
    params: [beta, N, S0, I1_0, ..., I9_0, R0] or similar, 
            where beta is the parameter to be optimized.
    lista_out: array or list containing the time points (days) to evaluate the solution.
    """
    beta, N = params[0], params[1]

    # Initial conditions according to the project description:
    # S(0) = 1499, I_1(0) = 1, rest of I and R are 0.
    S0 = N - 1.0
    I1_0 = 1.0
    I_remaining_0 = np.zeros(8)
    R0 = 0.0

    y0 = np.array([S0, I1_0, *I_remaining_0, R0])

    # Final simulation time
    t_span = [min(lista_out), max(lista_out)]

    # Solve the system of ODEs using Runge-Kutta (solve_ivp defaults to RK45)
    sol = solve_ivp(
        modelo_h1n1, 
        t_span, 
        y0, 
        args=([beta, N],),
        t_eval=lista_out,
        method='RK45'
    )
    
    return sol