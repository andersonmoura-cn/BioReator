import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, step, TransferFunction, lsim

from config.tf_config import G_X_D, G_S_D, G_S_Sf, G_X_Sf
from utils.saveFig import img_save

# Definição da função de transferência G(s)
# K = -83333.333333333
# numerador = [K, -K * 180e3]
# denominador = [1, 3125, 2.25e8]

def gerar_entrada(tipo="degrau", A=1.0, t_final=50, n=1000, t_troca=20):
    t = np.linspace(0, t_final, n)

    if tipo == "degrau":
        u = np.ones_like(t) * A

    elif tipo == "degrau_pos_neg":
        u = np.array([A if tempo <= t_troca else -A for tempo in t])

    elif tipo == "degrau_neg_pos":
        u = np.array([-A if tempo <= t_troca else A for tempo in t])

    else:
        raise ValueError(f"Tipo de entrada inválido: {tipo}")

    return t, u

def params(G: TransferFunction, label, t, u):

    # Simulação  com lsim
    t, y_delta, _ = lsim(G, U=u, T=t)

    # Pontos de operação
    x_op = 0.3066   # g/L
    s_op = 0.2333   # g/L

    # Converte da variação para valor físico
    if label.startswith("G_X"):
        y = y_delta + x_op
    elif label.startswith("G_S"):
        y = y_delta + s_op
    else:
        y = y_delta

    # --- Cálculo das métricas da resposta ---

    # 1. Valor de estado estacionário (c_final)
    c_final = y[-1]

    # 2. Valor de pico (c_max) e % Overshoot
    c_max = np.max(y)
    if c_final > 0:
        overshoot_percent = ((c_max - c_final) / c_final) * 100
    else:
        overshoot_percent = 0

    # 3. Tempo de subida (Tr) - tempo para ir de 10% a 90% do valor final
    try:
        idx_10_percent = np.where(y >= 0.1 * c_final)[0][0]
        idx_90_percent = np.where(y >= 0.9 * c_final)[0][0]
        t_10 = t[idx_10_percent]
        t_90 = t[idx_90_percent]
        t_rise = t_90 - t_10
    except IndexError:
        t_rise = "Não foi possível calcular"
        t_10 = None
        t_90 = None

    # 4. Tempo de estabilização (Ts) - critério de 2%
    try:
        limite_superior = c_final * 1.02
        limite_inferior = c_final * 0.98
        idx_fora_faixa = np.where((y > limite_superior) | (y < limite_inferior))[0]
        if len(idx_fora_faixa) > 0:
            t_settling = t[idx_fora_faixa[-1]]
        else:
            t_settling = t[0]
    except IndexError:
        t_settling = "Não foi possível calcular"

    # --- Impressão dos Resultados Precisos ---
    print("Análise da Resposta ao Degrau (Valores Precisos):")
    print("-" * 50)
    print(f"Valor de Estado Estacionário (c_final): {c_final}")
    print(f"Valor de Pico (c_max): {c_max}")
    print(f"Percentual de Overshoot (%OS): {overshoot_percent}")
    if isinstance(t_rise, float):
        print(f"Tempo de Subida (10%-90%) em segundos: {t_rise}")
    else:
        print(f"Tempo de Subida (10%-90%): {t_rise}")

    if isinstance(t_settling, float):
        print(f"Tempo de Estabilização (critério 2%) em segundos: {t_settling}")
    else:
        print(f"Tempo de Estabilização (critério 2%): {t_settling}")
    print("-" * 50)


    # --- Geração do Gráfico ---
    plt.figure(figsize=(12, 7))
    plt.plot(t * 1000, y, label='Resposta ao Degrau', linewidth=2) # Eixo x em ms

    # Linhas de referência e anotações
    plt.axhline(c_final, color='red', linestyle='--', label=f'Valor Estacionário ({c_final:.3f})')
    plt.axhline(c_max, color='green', linestyle='--', label=f'Valor de Pico ({c_max:.3f})')
    plt.axhline(c_final * 1.02, color='gray', linestyle=':', label='Faixa de Estabilização (2%)')
    plt.axhline(c_final * 0.98, color='gray', linestyle=':')

    # Anotações para tempo de estabilização
    if isinstance(t_settling, float):
        plt.axvline(t_settling * 1000, color='purple', linestyle='--', label=f'Tempo de Estabilização ({t_settling*1000:.2f} ms)')

    # Anotações para tempo de subida (MODIFICADO FINAL)
    if t_10 is not None and t_90 is not None:
        # Adiciona o label a uma das linhas para aparecer na legenda
        label_tr = f'Tempo de Subida (Tr = {t_rise*1000:.2f} ms)'
        plt.axvline(t_10 * 1000, color='orange', linestyle='--', label=label_tr)
        plt.axvline(t_90 * 1000, color='orange', linestyle='--')


    # Configurações do gráfico
    # plt.title(r'Resposta com uma pequena pertubação', fontsize=16)
    plt.xlabel('Tempo (ms)', fontsize=12)
    if label.startswith("G_X"):
        plt.ylabel('Concentração de Biomassa (g/L)', fontsize=12)
    elif label.startswith("G_S"):
        plt.ylabel('Concentração de Substrato (g/L)', fontsize=12)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    y_min = np.min(y)
    y_max = np.max(y)
    margem = 0.05 * (y_max - y_min if y_max != y_min else 1)
    plt.ylim(y_min - margem, y_max + margem)

    # Mostra o gráfico na tela
    img_save(f"{label}",dir_name="RespostaLinear")



numerador = G_X_D["num"]
denominador = G_X_D["den"]
G = TransferFunction(numerador, denominador)

t, u = gerar_entrada(tipo="degrau_pos_neg", A=0.1 * 0.35, t_troca=20)
params(TransferFunction(G_X_D["num"], G_X_D["den"]), label="G_X_D", t=t, u=u)

t, u = gerar_entrada(tipo="degrau", A=0.1 * 0.35)
params(TransferFunction(G_X_Sf["num"], G_X_Sf["den"]), label="G_X_Sf", t=t, u=u)

t, u = gerar_entrada(tipo="degrau", A=0.1 * 1)
params(TransferFunction(G_S_D["num"], G_S_D["den"]), label="G_S_D", t=t, u=u)
params(TransferFunction(G_S_Sf["num"], G_S_Sf["den"]), label="G_S_Sf", t=t, u=u)
