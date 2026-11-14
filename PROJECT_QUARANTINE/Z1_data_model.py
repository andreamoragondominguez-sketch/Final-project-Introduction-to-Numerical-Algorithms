
import numpy as np

N = 1500
t0 = 0.0
t_fin = 60.0
lista_out = np.linspace(t0, t_fin, int(t_fin) + 1)

# Initial conditions
S0 = 1499.0
I1_0 = 1.0
R0 = 0.0
Q0 = 0.0  # Initial quarantined population

# Extension parameter: Quarantined rate (fraction of infected sent to quarantine per day)
# Try 0.0 (no intervention) vs 0.3 (30% isolation rate)
quarantine_rate = 0.2