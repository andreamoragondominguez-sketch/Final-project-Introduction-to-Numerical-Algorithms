
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from Z1_data_model import *
from Z2_model import OUTPUT_vacunacion

# 1. Find optimal beta without intervention (nu = 0.0) based on baseline condition
def objective(beta):
    sol = OUTPUT_vacunacion([beta, N, 0.0], lista_out)
    if not sol.success:
        return 1e9
    return (np.sum(sol.y[1:10, 7]) - 10.0)**2

result = minimize_scalar(objective, bounds=(1e-7, 1e-3), method='bounded')
optimal_beta = result.x

# 2. Run simulation without vaccination (nu = 0.0)
sol_baseline = OUTPUT_vacunacion([optimal_beta, N, 0.0], lista_out)
total_inf_base = np.sum(sol_baseline.y[1:10, :], axis=0)

# 3. Run simulation with an active vaccination campaign ( nu = 0.015)
sol_vaccine = OUTPUT_vacunacion([optimal_beta, N, 0.015], lista_out)
total_inf_v = np.sum(sol_vaccine.y[1:10, :], axis=0)

# 4. Comparative Plot
plt.figure(figsize=(10, 6))
plt.plot(sol_baseline.t, total_inf_base, label='No Vaccination (Baseline)', color='red', linestyle='--', linewidth=2)
plt.plot(sol_vaccine.t, total_inf_v, label='With Active Vaccination (1.5% Daily)', color='green', linewidth=2)

plt.title('Impact of Vaccination Campaign on H1N1 Outbreak ($N = 1500$)')
plt.xlabel('Time (days)')
plt.ylabel('Active Infected Individuals')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('H1N1_Vaccination_Comparison.png', dpi=300)
plt.show()