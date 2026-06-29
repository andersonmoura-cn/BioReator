import numpy as np
import control as ct
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ====================================================================
# 1. PARÂMETROS FÍSICOS E NOMINAIS DO BIORREATOR
# ====================================================================
# Constantes Cinéticas (Monod)
mu_max = 0.5
Ks = 0.1
Y = 0.4

# Ponto de Operação Nominal
x_bar, S_bar = 0.3066, 0.2333
D_bar, Sf_bar = 0.35, 1.0

# ====================================================================
# 2. PROJETO DO CONTROLADOR LQR (Feito na aproximação linear)
# ====================================================================
A = np.array([[0, 0.138], [-0.875, -0.695]])
B = np.array([[-0.3066, 0], [0.7667, 0.35]])
C = np.eye(2)

Q = np.diag([1063.79, 459.31])
R = np.diag([90.70, 8.16])
K, _, _ = ct.lqr(A, B, Q, R)

A_cl = A - B @ K
DC_gain = C @ np.linalg.inv(A_cl) @ B
Nr = -np.linalg.inv(DC_gain)

# ====================================================================
# 3. FUNÇÃO PADRONIZADA DE SIMULAÇÃO NÃO-LINEAR
# ====================================================================
def simular_cenario_nao_linear(nome_cenario, start_x, start_S, setpoint_x, setpoint_S, pert_Sf_time, pert_Sf_mag):

    def bioreactor_nl(t, state):
        x, S = state

        delta_x = np.array([x - x_bar, S - S_bar])
        r_signal = np.array([setpoint_x - x_bar, setpoint_S - S_bar])

        delta_u = -K @ delta_x + Nr @ r_signal

        dist_Sf = pert_Sf_mag if t >= pert_Sf_time else 0.0

        D_real = D_bar + delta_u[0]
        Sf_real = Sf_bar + delta_u[1] + dist_Sf

        D_real = max(0.0, D_real)
        Sf_real = max(0.0, Sf_real)

        mu = mu_max * S / (Ks + S)

        dxdt = (mu - D_real) * x
        dSdt = D_real * (Sf_real - S) - (mu / Y) * x

        return [dxdt, dSdt]

    T_end = 70.0
    estado_inicial = [start_x, start_S]

    sol = solve_ivp(
        bioreactor_nl,
        [0, T_end],
        estado_inicial,
        method="RK45",
        t_eval=np.linspace(0, T_end, 1500),
        dense_output=True
    )

    T_out = sol.t
    x_abs = sol.y[0]
    S_abs = sol.y[1]

    D_abs = np.zeros_like(T_out)
    Sf_abs = np.zeros_like(T_out)

    for i, t in enumerate(T_out):
        delta_x = np.array([x_abs[i] - x_bar, S_abs[i] - S_bar])
        r_signal = np.array([setpoint_x - x_bar, setpoint_S - S_bar])

        delta_u = -K @ delta_x + Nr @ r_signal

        dist_Sf = pert_Sf_mag if t >= pert_Sf_time else 0.0

        D_real = D_bar + delta_u[0]
        Sf_real = Sf_bar + delta_u[1] + dist_Sf

        D_abs[i] = max(0.0, D_real)
        Sf_abs[i] = max(0.0, Sf_real)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(T_out, x_abs, label="Biomassa (X)", linewidth=2)
    ax1.plot(T_out, S_abs, label="Substrato (S)", linewidth=2)
    ax1.axhline(setpoint_x, linestyle="--", alpha=0.5, label="Setpoint X")
    ax1.axhline(setpoint_S, linestyle="--", alpha=0.5, label="Setpoint S")
    ax1.set_ylabel("Concentração (g/L)")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2.plot(T_out, D_abs, label="Válvula Diluição (D)", linewidth=2)
    ax2.plot(T_out, Sf_abs, label="Válvula Alimentação (Sf)", linewidth=2)

    if pert_Sf_mag != 0:
        ax2.axvline(
            pert_Sf_time,
            linestyle="dotted",
            alpha=0.7,
            label=f"Perturbação Sf em t={pert_Sf_time}"
        )

    ax2.set_ylabel("Sinal do Atuador")
    ax2.set_xlabel("Tempo (h)")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# ====================================================================
# 5. EXECUÇÃO DAS 3 SIMULAÇÕES REQUISITADAS
# ====================================================================

# --- SIMULAÇÃO I ---
simular_cenario_nao_linear(
    "Simulação I - Rejeição a Distúrbio",
    start_x=0.281,
    start_S=0.21,
    setpoint_x=0.3066,
    setpoint_S=0.2333,
    pert_Sf_time=0,
    pert_Sf_mag=0.0
)