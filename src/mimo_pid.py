import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# =====================================================================
# 1. PARÂMETROS E PONTOS DE EQUILÍBRIO (BASE DO MODELO)
# =====================================================================
mu_max, Ks, Y = 0.5, 0.1, 0.4

# Ponto de equilíbrio de linearização (usado como referência)
x_eq, S_eq = 0.3066, 0.2333
D0, Sf0 = 0.35, 1.0
S_sp_fixo = 0.2333 

# Limites de restrição para análise
limite_u_D_perc = 30.0
limite_u_Sf_perc = 35.0

# Ganhos dos Controladores
Kp1, Ki1, Kd1 = 4.80, 0.833, 6.91   # PID para Biomassa (Sf)
Kp2, Ki2, Kd2 = 0.435, 0.1449, 0.0   # PI para Substrato (D)

# Matrizes de Espaço de Estados Lineares
A = np.array([[0, 0.138], [-0.875, -0.695]])
B = np.array([[-0.3066, 0], [0.7667, 0.35]])

tempo = np.linspace(0, 80, 4000)

# =====================================================================
# 2. FUNÇÃO DE SIMULAÇÃO (LINEAR E NÃO LINEAR SIMULTÂNEOS)
# =====================================================================
def simular_cenario_comparativo(X_init, S_init, X_sp, dist_Sf_flag):
    def biorreator_ambos(Z, t):
        x_nl, S_nl, int_ex_nl, int_eS_nl, dx_lin, dS_lin, int_ex_lin, int_eS_lin = Z
        
        # Perturbação: Sf sofre degrau de +0.2 após t=20
        d_Sf = 0.2 if (dist_Sf_flag and t > 20.0) else 0.0
        
        # --- DINÂMICA NÃO LINEAR ---
        e_x_nl, e_S_nl = X_sp - x_nl, S_sp_fixo - S_nl
        mu = mu_max * S_nl / (Ks + S_nl)

        u_D_c_nl = Kp2 * e_S_nl + Ki2 * int_eS_nl
        D_real_nl = max(0.0, D0 + u_D_c_nl)

        dxdt_nl = mu * x_nl - D_real_nl * x_nl

        u_Sf_c_nl = Kp1 * e_x_nl + Ki1 * int_ex_nl - Kd1 * dxdt_nl
        Sf_real_nl = max(0.0, Sf0 + u_Sf_c_nl + d_Sf)

        dSdt_nl = D_real_nl * (Sf_real_nl - S_nl) - (mu / Y) * x_nl

        # --- DINÂMICA LINEAR ---
        dX_sp, dS_sp = X_sp - x_eq, S_sp_fixo - S_eq 
        e_x_lin, e_S_lin = dX_sp - dx_lin, dS_sp - dS_lin

        u_D_c_lin = Kp2 * e_S_lin + Ki2 * int_eS_lin
        dxdt_lin = A[0,0]*dx_lin + A[0,1]*dS_lin + B[0,0]*u_D_c_lin
        
        u_Sf_c_lin = Kp1 * e_x_lin + Ki1 * int_ex_lin - Kd1 * dxdt_lin
        dSdt_lin = A[1,0]*dx_lin + A[1,1]*dS_lin + B[1,0]*u_D_c_lin + B[1,1]*(u_Sf_c_lin + d_Sf)

        return [dxdt_nl, dSdt_nl, e_x_nl, e_S_nl, dxdt_lin, dSdt_lin, e_x_lin, e_S_lin]

    Z0 = [X_init, S_init, 0.0, 0.0, X_init - x_eq, S_init - S_eq, 0.0, 0.0]
    sol = odeint(biorreator_ambos, Z0, tempo)

    x_nl, S_nl = sol[:, 0], sol[:, 1]
    x_lin, S_lin = sol[:, 4] + x_eq, sol[:, 5] + S_eq 

    u_D_nl_out, u_Sf_nl_out = np.zeros_like(tempo), np.zeros_like(tempo)
    u_D_lin_out, u_Sf_lin_out = np.zeros_like(tempo), np.zeros_like(tempo)

    for i in range(len(tempo)):
        # Recálculo NL
        e_x_nl, e_S_nl = X_sp - x_nl[i], S_sp_fixo - S_nl[i]
        mu = mu_max * S_nl[i] / (Ks + S_nl[i])
        u_D_c_nl = Kp2 * e_S_nl + Ki2 * sol[i, 3]
        D_real_nl = max(0.0, D0 + u_D_c_nl)
        dxdt_nl = mu * x_nl[i] - D_real_nl * x_nl[i]
        u_Sf_c_nl = Kp1 * e_x_nl + Ki1 * sol[i, 2] - Kd1 * dxdt_nl
        u_D_nl_out[i], u_Sf_nl_out[i] = D0 + u_D_c_nl, Sf0 + u_Sf_c_nl
        
        # Recálculo LIN
        e_x_lin, e_S_lin = (X_sp - x_eq) - sol[i, 4], (S_sp_fixo - S_eq) - sol[i, 5]
        u_D_c_lin = Kp2 * e_S_lin + Ki2 * sol[i, 7]
        dxdt_lin = A[0,0]*sol[i,4] + A[0,1]*sol[i,5] + B[0,0]*u_D_c_lin
        u_Sf_c_lin = Kp1 * e_x_lin + Ki1 * sol[i, 6] - Kd1 * dxdt_lin
        u_D_lin_out[i], u_Sf_lin_out[i] = D0 + u_D_c_lin, Sf0 + u_Sf_c_lin

    return x_nl, S_nl, u_D_nl_out, u_Sf_nl_out, x_lin, S_lin, u_D_lin_out, u_Sf_lin_out

def calcular_imprimir_variacao(nome, x_arr, S_arr, uD_arr, uSf_arr, tipo):
    max_delta_x = np.max(np.abs(x_arr - x_eq))
    max_delta_S = np.max(np.abs(S_arr - S_eq))
    max_delta_uD = np.max(np.abs(uD_arr - D0))
    max_delta_uSf = np.max(np.abs(uSf_arr - Sf0))

    perc_x = (max_delta_x / x_eq) * 100
    perc_S = (max_delta_S / S_eq) * 100
    perc_uD = (max_delta_uD / D0) * 100
    perc_uSf = (max_delta_uSf / Sf0) * 100

    print(f"Modelo {tipo}:")
    print(f"  Biomassa (X): Pico de {perc_x:6.2f}% | Ultrapassou 10%? {'SIM' if perc_x > 10 else 'NÃO'}")
    print(f"  Substrato(S): Pico de {perc_S:6.2f}% | Ultrapassou 20%? {'SIM' if perc_S > 20 else 'NÃO'}")
    print(f"  Controle D  : Pico de {perc_uD:6.2f}% | Ultrapassou {limite_u_D_perc}%? {'SIM' if perc_uD > limite_u_D_perc else 'NÃO'}")
    print(f"  Controle Sf : Pico de {perc_uSf:6.2f}% | Ultrapassou {limite_u_Sf_perc}%? {'SIM' if perc_uSf > limite_u_Sf_perc else 'NÃO'}")
    print("-" * 65)

# =====================================================================
# 3. DEFINIÇÃO DOS CENÁRIOS E CÁLCULOS
# =====================================================================
cenarios = [
    {"nome": "I",   "X0": 0.3066, "S0": 0.2333, "Xsp": 0.3066, "dist_Sf": True},
    {"nome": "II",  "X0": 0.3066, "S0": 0.2333, "Xsp": 0.3400, "dist_Sf": False},
    {"nome": "III", "X0": 0.2800, "S0": 0.2333, "Xsp": 0.3066, "dist_Sf": True}
]

print("\n" + "="*65)
print("RELATÓRIO DE DESEMPENHO E VARIAÇÃO DE SINAIS".center(65))
print("="*65)

for c in cenarios:
    resultados = simular_cenario_comparativo(c["X0"], c["S0"], c["Xsp"], c["dist_Sf"])
    c["x_nl"], c["S_nl"], c["u_D_nl"], c["u_Sf_nl"] = resultados[0:4]
    c["x_lin"], c["S_lin"], c["u_D_lin"], c["u_Sf_lin"] = resultados[4:8]
    
    print(f"\n>>> CENÁRIO {c['nome']} <<<")
    calcular_imprimir_variacao(c["nome"], c["x_lin"], c["S_lin"], c["u_D_lin"], c["u_Sf_lin"], "LINEAR")
    calcular_imprimir_variacao(c["nome"], c["x_nl"], c["S_nl"], c["u_D_nl"], c["u_Sf_nl"], "NÃO LINEAR")

# =====================================================================
# 4. PLOTAGEM (UMA FIGURA PARA CADA CENÁRIO)
# =====================================================================
for c in cenarios:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Cenário {c["nome"]}: Comparação Linear vs Não Linear', fontsize=16)

    # --- Gráfico de Variáveis de Estado ---
    ax1.plot(tempo, c["x_nl"], label='Biomassa Real (NL)', color='blue', lw=2)
    ax1.plot(tempo, c["S_nl"], label='Substrato Real (NL)', color='red', lw=2)
    ax1.plot(tempo, c["x_lin"], label='Biomassa Linear', color='cyan', lw=2, linestyle='--')
    ax1.plot(tempo, c["S_lin"], label='Substrato Linear', color='orange', lw=2, linestyle='--')
    
    ax1.axhline(c["Xsp"], color='blue', linestyle=':', alpha=0.5, lw=1.5)
    ax1.axhline(S_sp_fixo, color='red', linestyle=':', alpha=0.5, lw=1.5)
    
    ax1.set_title('Concentrações (Estado)')
    ax1.set_xlabel('Tempo (h)'); ax1.set_ylabel('Concentração (g/L)')
    ax1.grid(True, alpha=0.4); ax1.legend(loc='best', fontsize='small')

    # --- Gráfico de Esforços de Controle ---
    ax2.plot(tempo, c["u_D_nl"], label='Comando D Real (NL)', color='purple', lw=2)
    ax2.plot(tempo, c["u_Sf_nl"], label='Comando Sf Real (NL)', color='green', lw=2)
    ax2.plot(tempo, c["u_D_lin"], label='Comando D Linear', color='magenta', lw=2, linestyle='--')
    ax2.plot(tempo, c["u_Sf_lin"], label='Comando Sf Linear', color='lime', lw=2, linestyle='--')
    
    if c["dist_Sf"]:
        ax2.axvline(20, color='black', linestyle=':', alpha=0.7, label='Início Distúrbio (t=20)')
        
    ax2.set_title('Esforços de Controle')
    ax2.set_xlabel('Tempo (h)'); ax2.set_ylabel('Sinal de Controle Absoluto')
    ax2.grid(True, alpha=0.4); ax2.legend(loc='best', fontsize='small')

    plt.tight_layout()

# O comando show no final abrirá as três janelas simultaneamente
plt.show()