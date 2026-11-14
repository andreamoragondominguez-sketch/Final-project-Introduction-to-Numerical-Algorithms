
import numpy as np
from scipy.integrate import solve_ivp

def modelo_h1n1_vacunacion(t, y, params):
    # Unpack states: S, I1..I9, V (Vaccinated), R (Recovered)
    S = y[0]
    I1, I2, I3, I4, I5, I6, I7, I8, I9 = y[1:10]
    V = y[10]
    R = y[11]

    beta = params[0]
    nu = params[2]  # Daily vaccination rate (e.g., 0.02 for 2% of susceptibles vaccinated per day)
    
    total_infected = I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9
    force_of_infection = beta * S * total_infected

    # Derivatives
    # Susceptibles decrease due to infections AND daily vaccination
    dS = -force_of_infection - nu * S
    
    # Infection compartments transitions
    dI1 = force_of_infection - I1
    dI2 = I1 - I2
    dI3 = I2 - I3
    dI4 = I3 - I4
    dI5 = I4 - I5
    dI6 = I5 - I6
    dI7 = I6 - I7
    dI8 = I7 - I8
    dI9 = I8 - I9

    # Vaccinated: accumulates individuals moved directly from S
    dV = nu * S
    
    # Recovered: leaves from the last infection stage (I9)
    dR = I9

    return [dS, dI1, dI2, dI3, dI4, dI5, dI6, dI7, dI8, dI9, dV, dR]

def OUTPUT_vacunacion(params, lista_out):
    beta, N, nu = params[0], params[1], params[2]
    S0 = N - 1.0
    # Initial conditions: 11 infection compartments + V and R initialized to 0
    y0 = np.array([S0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    sol = solve_ivp(
        modelo_h1n1_vacunacion, 
        [min(lista_out), max(lista_out)], 
        y0, 
        args=([beta, N, nu],),
        t_eval=lista_out
    )
    return sol