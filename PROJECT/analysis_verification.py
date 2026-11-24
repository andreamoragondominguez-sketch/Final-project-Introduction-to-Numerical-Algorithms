"""
Additional analysis and numerical verification for the H1N1 project.

Generates:
  H1N1_Calibration.png     day-7 infections and objective J(beta)
  H1N1_WorkPrecision.png   work-precision diagram + adaptive step sizes
  H1N1_SIR_Comparison.png  9-stage model vs classical SIR, period distributions
  H1N1_Sensitivity.png     peak vs quarantine / vaccination rate
  H1N1_Combined.png        cumulative infections in all scenarios
and prints every number quoted in the report.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import factorial
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, brentq
from Z1_data_model import N, lista_out
from Z2_model import (OUTPUT, OUTPUT_quarantine, OUTPUT_vacunacion,
                      modelo_h1n1, modelo_h1n1_quarantine, modelo_h1n1_vacunacion)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.4})
y0_base = np.r_[N - 1.0, 1.0, np.zeros(9)]   # S, I1..I9, R
y0_ext  = np.r_[N - 1.0, 1.0, np.zeros(10)]  # S, I1..I9, Q/V, R
TIGHT   = dict(rtol=1e-11, atol=1e-11)

# ------------------------------------------------------------------
# 1. Calibration (same procedure as main.py) and refined calibration
# ------------------------------------------------------------------
def objective(beta):
    sol = OUTPUT([beta, N], lista_out)
    return 1e9 if not sol.success else (np.sum(sol.y[1:10, 7]) - 10.0)**2

res  = minimize_scalar(objective, bounds=(1e-7, 1e-3), method="bounded")
beta = res.x
print(f"[calib] beta = {beta:.6e}, nfev = {res.nfev}, I(7) = {np.sum(OUTPUT([beta, N], lista_out).y[1:10, 7]):.4f}")

def infected_day7(b, target=10.0):
    s = solve_ivp(modelo_h1n1, [0, 7], y0_base, args=([b, N],), **TIGHT)
    return s.y[1:10, -1].sum() - target

beta_ref = brentq(infected_day7, 1e-6, 1e-3, xtol=1e-16)
print(f"[calib] reference beta (brentq, tight) = {beta_ref:.6e}, rel. diff = {abs(beta - beta_ref)/beta_ref:.2%}")

# Sensitivity of beta to the observation
print("[calib] target   beta         R0      peak    peak_day")
for target in [8, 9, 10, 11, 12]:
    b = brentq(lambda x: infected_day7(x, target), 1e-6, 1e-3, xtol=1e-16)
    s = solve_ivp(modelo_h1n1, [0, 60], y0_base, args=([b, N],), t_eval=lista_out, **TIGHT)
    I = s.y[1:10].sum(0)
    print(f"        {target:3d}      {b:.4e}   {9*b*N:.3f}   {I.max():6.1f}  {I.argmax():3d}")

bs   = np.linspace(1.5e-4, 3.0e-4, 121)
vals = np.array([np.sum(OUTPUT([b, N], lista_out).y[1:10, 7]) for b in bs])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(bs * 1e4, vals, "k-")
ax[0].axhline(10, color="gray", ls="--"); ax[0].axvline(beta * 1e4, color="gray", ls=":")
ax[0].set(xlabel=r"$\beta\ (\times 10^{-4})$", ylabel=r"$\sum_j I_j(7)$", title=r"Infected at day 7 as a function of $\beta$")
ax[1].semilogy(bs * 1e4, (vals - 10)**2 + 1e-12, "k-"); ax[1].axvline(beta * 1e4, color="gray", ls=":")
ax[1].set(xlabel=r"$\beta\ (\times 10^{-4})$", ylabel=r"$J(\beta)$", title=r"Objective $J(\beta)=(\sum_j I_j(7)-10)^2$")
plt.tight_layout(); plt.savefig("H1N1_Calibration.png", dpi=300); plt.close()

# ------------------------------------------------------------------
# 2. Base model: key quantities and checks
# ------------------------------------------------------------------
sol  = OUTPUT([beta, N], lista_out)
I    = sol.y[1:10].sum(0)
R0   = 9 * beta * N
r    = brentq(lambda x: beta * (N - 1) * sum((1 + x)**(-k) for k in range(1, 10)) - 1, 1e-6, 2)
Sinf = brentq(lambda x: x - (N - 1) * np.exp(-9 * beta * (N - x)), 1e-6, N - 1.5)
long = solve_ivp(modelo_h1n1, [0, 200], y0_base, args=([beta, N],), **TIGHT)
print(f"[base] R0 = {R0:.3f}, r = {r:.4f}/day, doubling = {np.log(2)/r:.2f} d, threshold N/R0 = {N/R0:.0f}")
print(f"[base] peak = {I.max():.1f} on day {I.argmax()}, S(60) = {sol.y[0, -1]:.2f}, R(60) = {sol.y[10, -1]:.1f}")
print(f"[base] mass error = {np.max(np.abs(sol.y.sum(0) - N)):.1e}")
print(f"[base] S_inf: final-size eq. = {Sinf:.6f}, ODE t=200 = {long.y[0, -1]:.6f}")
t_cross = lista_out[np.argmax(sol.y[0] < N / R0)]
print(f"[base] S drops below N/R0 on day {t_cross:.0f}")
over = lista_out[I > 0.10 * N]
print(f"[base] days with >10% of campus sick: {over.min():.0f}-{over.max():.0f} ({len(over)} days)")

# Stiffness: eigenvalues of the Jacobian at t=0 and at the peak
def jacobian(y, b):
    J = np.zeros((11, 11)); S, Itot = y[0], y[1:10].sum()
    J[0, 0] = -b * Itot; J[0, 1:10] = -b * S
    J[1, 0] =  b * Itot; J[1, 1:10] =  b * S; J[1, 1] -= 1
    for j in range(2, 10):
        J[j, j - 1] = 1; J[j, j] = -1
    J[10, 9] = 1
    return J

for name, k in [("t=0", 0), ("peak", I.argmax())]:
    ev = np.linalg.eigvals(jacobian(sol.y[:, k], beta))
    print(f"[stiff] {name}: max Re = {ev.real.max():.3f}, spectral radius = {np.abs(ev).max():.3f}")

# ------------------------------------------------------------------
# 3. Work-precision diagram and adaptive step sizes
# ------------------------------------------------------------------
ref  = solve_ivp(modelo_h1n1, [0, 60], y0_base, args=([beta, N],), t_eval=lista_out,
                 method="Radau", rtol=1e-12, atol=1e-12)
tols = 10.0 ** -np.arange(2, 11); err, work = [], []
for tol in tols:
    s = solve_ivp(modelo_h1n1, [0, 60], y0_base, args=([beta, N],), t_eval=lista_out, rtol=tol, atol=tol)
    err.append(np.max(np.abs(s.y - ref.y))); work.append(s.nfev)
d = solve_ivp(modelo_h1n1, [0, 60], y0_base, args=([beta, N],), t_eval=lista_out)
slope = np.polyfit(np.log(work[4:]), np.log(err[4:]), 1)[0]
print(f"[wp] default error = {np.max(np.abs(d.y - ref.y)):.2e} ({d.nfev} evals), fitted slope = {slope:.2f}")
for tol, e, w in zip(tols, err, work):
    print(f"     tol={tol:.0e}  err={e:.2e}  nfev={w}")

steps = solve_ivp(modelo_h1n1, [0, 60], y0_base, args=([beta, N],))
h     = np.diff(steps.t)
print(f"[steps] default RK45: {len(h)} steps, h in [{h.min():.3f}, {h.max():.3f}]")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].loglog(work, err, "ko-", label="RK45, rtol = atol = tol")
ax[0].loglog([d.nfev], [np.max(np.abs(d.y - ref.y))], "s", color="gray", ms=9, label="RK45 default tolerances")
W = np.array(work, float); ax[0].loglog(W, err[-1] * (W / W[-1])**-5, "k:", label=r"slope $-5$")
ax[0].set(xlabel="Number of RHS evaluations", ylabel="Max. absolute error on [0, 60]", title="Work-precision diagram")
ax[0].legend(fontsize=8); ax[0].grid(which="both", alpha=0.4, ls="--")
ax2 = ax[1]; ax2.step(steps.t[:-1], h, "k-", where="post", label="step size $h_n$")
ax2.set(xlabel="Time (days)", ylabel="Step size $h$ (days)", title="Adaptive step sizes (default RK45)")
ax3 = ax2.twinx(); ax3.plot(sol.t, I, color="gray", ls="--", label=r"$\sum_j I_j$"); ax3.set_ylabel("Total infected"); ax3.grid(False)
l1, n1 = ax2.get_legend_handles_labels(); l2, n2 = ax3.get_legend_handles_labels(); ax2.legend(l1 + l2, n1 + n2, fontsize=8, loc="upper right")
plt.tight_layout(); plt.savefig("H1N1_WorkPrecision.png", dpi=300); plt.close()

# ------------------------------------------------------------------
# 4. Comparison with the classical SIR model (same R0, same mean period)
# ------------------------------------------------------------------
def sir(t, y, b):
    S, I1, R = y
    return [-b * S * I1, b * S * I1 - I1 / 9, I1 / 9]

s_sir    = solve_ivp(sir, [0, 60], [N - 1, 1, 0], args=(beta,), t_eval=lista_out, **TIGHT)
beta_sir = brentq(lambda b: solve_ivp(sir, [0, 7], [N - 1, 1, 0], args=(b,), **TIGHT).y[1, -1] - 10, 1e-6, 1e-3)
s_sir2   = solve_ivp(sir, [0, 60], [N - 1, 1, 0], args=(beta_sir,), t_eval=lista_out, **TIGHT)
print(f"[sir] same beta: peak = {s_sir.y[1].max():.1f} day {s_sir.y[1].argmax()}, S(60) = {s_sir.y[0, -1]:.1f}")
print(f"[sir] recalibrated beta = {beta_sir:.4e}, R0 = {9*beta_sir*N:.2f}, peak = {s_sir2.y[1].max():.1f} day {s_sir2.y[1].argmax()}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(lista_out, I, "k-", lw=2, label="9-stage model")
ax[0].plot(lista_out, s_sir.y[1], color="gray", ls="--", lw=2, label=r"SIR, same $\beta$ ($\mathcal{R}_0$ = 3.06)")
ax[0].plot(lista_out, s_sir2.y[1], color="gray", ls=":", lw=2, label="SIR, recalibrated to 10 on day 7")
ax[0].set(xlabel="Time (days)", ylabel="Total infected", title="9-stage model vs. classical SIR"); ax[0].legend(fontsize=8)
tt = np.linspace(0, 30, 400)
ax[1].plot(tt, tt**8 * np.exp(-tt) / factorial(8), "k-", lw=2, label="Erlang(9, 1): 9 stages")
ax[1].plot(tt, np.exp(-tt / 9) / 9, color="gray", ls="--", lw=2, label="Exponential, mean 9: SIR")
ax[1].set(xlabel="Infectious period (days)", ylabel="Probability density", title="Distribution of the infectious period"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("H1N1_SIR_Comparison.png", dpi=300); plt.close()
print(f"[sir] P(T<3): exp = {1 - np.exp(-3/9):.3f}; P(T>18): exp = {np.exp(-2):.3f}")

# ------------------------------------------------------------------
# 5. Interventions: sensitivity, eradication threshold, combined strategy
# ------------------------------------------------------------------
TQ     = lambda a: (1 - (1 + a)**-9) / a if a > 0 else 9.0
a_star = brentq(lambda a: beta * N * TQ(a) - 1, 1e-6, 2)
SinfQ  = brentq(lambda x: x - (N - 1) * np.exp(-TQ(0.1) * beta * (N - x)), 1e-6, N - 1.5)
print(f"[quar] T_Q(0.1) = {TQ(0.1):.3f}, R0_Q = {beta*N*TQ(0.1):.3f}, alpha* = {a_star:.3f}, S_inf = {SinfQ:.1f}")

al = np.linspace(0, 0.5, 26); nu = np.linspace(0, 0.05, 26)
pq = [OUTPUT_quarantine([beta, N, a], lista_out).y[1:10].sum(0).max() for a in al]
pv = [OUTPUT_vacunacion([beta, N, n], lista_out).y[1:10].sum(0).max() for n in nu]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(al, pq, "ko-", ms=3); ax[0].axvline(a_star, color="gray", ls=":", label=rf"$\mathcal{{R}}_0^Q=1$ at $\alpha\approx{a_star:.2f}$")
ax[0].set(xlabel=r"Quarantine rate $\alpha$ (day$^{-1}$)", ylabel="Peak active infected", title="Peak vs. quarantine rate"); ax[0].legend()
ax[1].plot(nu * 100, pv, "ko-", ms=3)
ax[1].set(xlabel=r"Vaccination rate $\nu$ (% of $S$ per day)", ylabel="Peak active infected", title="Peak vs. vaccination rate")
plt.tight_layout(); plt.savefig("H1N1_Sensitivity.png", dpi=300); plt.close()

for a in [0.05, 0.08, 0.2]:
    print(f"[sens] alpha={a}: peak = {OUTPUT_quarantine([beta, N, a], lista_out).y[1:10].sum(0).max():.1f}")
for n in [0.01, 0.03, 0.05]:
    print(f"[sens] nu={n}: peak = {OUTPUT_vacunacion([beta, N, n], lista_out).y[1:10].sum(0).max():.1f}")

def combined(t, y, b, a, n):
    S = y[0]; Ij = y[1:10]; Itot = Ij.sum(); Q, V = y[10], y[11]
    F = b * S * Itot
    dI = np.empty(9); dI[0] = F - (1 + a) * Ij[0]; dI[1:] = Ij[:-1] - (1 + a) * Ij[1:]
    return [-F - n * S, *dI, a * Itot - Q, n * S, Ij[8] + Q]

y0c = np.r_[N - 1.0, 1.0, np.zeros(11)]
sc  = solve_ivp(combined, [0, 60], y0c, args=(beta, 0.1, 0.015), t_eval=lista_out, **TIGHT)
Ic  = sc.y[1:10].sum(0)
print(f"[comb] peak = {Ic.max():.1f} day {Ic.argmax()}, infected by 60 = {N - sc.y[0, -1] - sc.y[11, -1]:.0f}, "
      f"V(60) = {sc.y[11, -1]:.0f}, S(60) = {sc.y[0, -1]:.0f}, I(60) = {Ic[-1]:.1f}, mass = {abs(sc.y[:, -1].sum() - N):.1e}")

sq  = OUTPUT_quarantine([beta, N, 0.1], lista_out); sv = OUTPUT_vacunacion([beta, N, 0.015], lista_out)
cum = {
    "No intervention": N - sol.y[0],
    r"Quarantine ($\alpha=0.1$)": N - sq.y[0],
    r"Vaccination ($\nu=0.015$)": N - sv.y[0] - sv.y[10],
    "Quarantine + vaccination": N - sc.y[0] - sc.y[11]
}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
styles  = [("red", "--"), ("purple", "-"), ("green", "-"), ("black", "-.")]
actives = [I, sq.y[1:10].sum(0), sv.y[1:10].sum(0), Ic]
for (lab, c), (col, ls), act in zip(cum.items(), styles, actives):
    ax[0].plot(lista_out, act, color=col, ls=ls, lw=2, label=lab)
    ax[1].plot(lista_out, c, color=col, ls=ls, lw=2, label=lab)
ax[0].set(xlabel="Time (days)", ylabel="Active infected", title="Active infected individuals")
ax[1].set(xlabel="Time (days)", ylabel="Cumulative infections", title="Cumulative number of infections")
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("H1N1_Combined.png", dpi=300); plt.close()
print(f"[quar] quarantined peak Q = {sq.y[10].max():.1f} on day {sq.y[10].argmax()}")
