
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from Z1_data_model import *
from Z2_model import OUTPUT_quarantine

# 1. Find the optimal beta without intervention (alpha = 0)
def objective(beta):
    sol = OUTPUT_quarantine([beta, N, 0.0], lista_out)
    if not sol.success:
        return 1e9
    return (np.sum(sol.y[1:10, 7]) - 10.0)**2

result = minimize_scalar(objective, bounds=(1e-7, 1e-3), method='bounded')
optimal_beta = result.x

# 2. Run simulation without intervention (alpha = 0.0)
sol_baseline = OUTPUT_quarantine([optimal_beta, N, 0.0], lista_out)
total_inf_base = np.sum(sol_baseline.y[1:10, :], axis=0)

# 3. Run simulation with active quarantine (e.g., alpha = 0.1, meaning 10% daily isolation)
sol_quarantine = OUTPUT_quarantine([optimal_beta, N, 0.1], lista_out)
total_inf_q = np.sum(sol_quarantine.y[1:10, :], axis=0)

# 4. Comparative Plot
plt.figure(figsize=(10, 6))
plt.plot(sol_baseline.t, total_inf_base, label='No Intervention (Baseline)', color='red', linestyle='--', linewidth=2)
plt.plot(sol_quarantine.t, total_inf_q, label='With Quarantine (10% Daily Isolation)', color='purple', linewidth=2)

plt.title('Impact of Quarantine on H1N1 Outbreak ($N = 1500$)')
plt.xlabel('Time (days)')
plt.ylabel('Active Infected Individuals')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('H1N1_Quarantine_Comparison.png', dpi=300)
plt.show()