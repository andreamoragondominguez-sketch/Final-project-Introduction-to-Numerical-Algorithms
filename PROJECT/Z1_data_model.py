
import numpy as np

# -------------------------------------------------------------------------
# H1N1 Model Data and Parameters
# -------------------------------------------------------------------------

# Total population size (small college campus)
N = 1500

# Initial simulation time and final time (e.g., simulating for 60 days)
t0 = 0.0
t_fin = 60.0

# Time evaluation points (every day)
lista_out = np.linspace(t0, t_fin, int(t_fin) + 1)

# Initial conditions according to project specifications:
# S(0) = 1499, I_1(0) = 1, I_2 to I_9 = 0, R(0) = 0
S0 = 1499.0
I1_0 = 1.0
I_restantes_0 = np.zeros(8)  # I_2 to I_9 initialized to 0
R0 = 0.0

# Estimated or initial guess for the infection rate (beta)
# Note: This beta should ideally be optimized so that total infections equal 10 after 7 days.
beta_initial = 0.0001 

# Parameter array to pass into the model functions
# Format: [beta, N]
params = [beta_initial, N]