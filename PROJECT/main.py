
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from Z1_data_model import *
from Z2_model import OUTPUT

# -------------------------------------------------------------
# Find beta so that at day 7 there are 10 total infected people
# -------------------------------------------------------------
def objective(beta):
    sol = OUTPUT([beta, N], lista_out)
    if not sol.success:
        return 1e9
    # Sum of infected individuals (compartments I_1 to I_9) at day 7 (index 7)
    total_inf_day_7 = np.sum(sol.y[1:10, 7])
    # We want it to be exactly 10
    return (total_inf_day_7 - 10.0)**2

# Simple and fast scalar optimizer
result = minimize_scalar(objective, bounds=(1e-7, 1e-3), method='bounded')
optimal_beta = result.x
print(f"Optimal beta value found: {optimal_beta}")

# -------------------------------------------------------------
# Final simulation with optimal beta and plots
# -------------------------------------------------------------
sol_final = OUTPUT([optimal_beta, N], lista_out)
t = sol_final.t
S = sol_final.y[0, :]
total_infected = np.sum(sol_final.y[1:10, :], axis=0)
R = sol_final.y[10, :]

# Main plot (Susceptible, Total Infected, Recovered)
plt.figure(figsize=(9, 5))
plt.plot(t, S, label='Susceptible (S)', color='blue')
plt.plot(t, total_infected, label='Total Infected ($\sum I_j$)', color='red')
plt.plot(t, R, label='Recovered (R)', color='green')
plt.title('H1N1 Outbreak Dynamics ($N = 1500$)')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('H1N1_Main_Dynamics.png', dpi=300)
plt.show()

# -------------------------------------------------------------
# Plot Individual Infection Stages (I_1 to I_9)
# -------------------------------------------------------------
plt.figure(figsize=(11, 6))

# Use a colormap to clearly distinguish the 9 daily infection stages
colors = plt.cm.plasma(np.linspace(0, 1, 9))

for j in range(9):
    # sol_final.y rows 1 to 9 correspond to compartments I_1 through I_9
    plt.plot(t, sol_final.y[j+1, :], label=f'Stage $I_{j+1}$ (Day {j+1})', color=colors[j], linewidth=1.8)

plt.title('H1N1 Infection Progression Across the 9 Individual Stages ($I_1$ to $I_9$)')
plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save the second plot
plt.savefig('H1N1_Infection_Stages.png', dpi=300)
plt.show()