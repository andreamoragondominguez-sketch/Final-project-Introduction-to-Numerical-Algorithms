
import numpy as np
from scipy.integrate import solve_ivp

def modelo_h1n1_quarantine(t, y, params):
    # Unpack states
    S = y[0]
    I1, I2, I3, I4, I5, I6, I7, I8, I9 = y[1:10]
    Q = y[10]
    R = y[11]

    beta = params[0]
    alpha = params[2]  # Quarantine rate (e.g., 0.1 or 0.2)
    
    # Total active infected individuals contributing to transmission
    total_infected = I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9
    force_of_infection = beta * S * total_infected

    # Derivatives
    dS = -force_of_infection
    
    # Infection compartments transitions (subtracting those moving to quarantine)
    dI1 = force_of_infection - I1 - alpha * I1
    dI2 = I1 - I2 - alpha * I2
    dI3 = I2 - I3 - alpha * I3
    dI4 = I3 - I4 - alpha * I4
    dI5 = I4 - I5 - alpha * I5
    dI6 = I5 - I6 - alpha * I6
    dI7 = I6 - I7 - alpha * I7
    dI8 = I7 - I8 - alpha * I8
    dI9 = I8 - I9 - alpha * I9

    # Quarantine: receives isolated individuals from all stages and recovers
    dQ = alpha * total_infected - Q
    
    # Recovered: those finishing all 9 days (I9) plus those leaving quarantine (Q)
    dR = I9 + Q

    return [dS, dI1, dI2, dI3, dI4, dI5, dI6, dI7, dI8, dI9, dQ, dR]

def OUTPUT_quarantine(params, lista_out):
    beta, N, alpha = params[0], params[1], params[2]
    S0 = N - 1.0
    # Initial conditions: 11 compartments + Q and R initialized to 0
    y0 = np.array([S0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    sol = solve_ivp(
        modelo_h1n1_quarantine, 
        [min(lista_out), max(lista_out)], 
        y0, 
        args=([beta, N, alpha],),
        t_eval=lista_out
    )
    return sol