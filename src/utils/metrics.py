import pandas as pd
import math
import numpy as np

def trunc(x, casas=5):
    fator = 10**casas
    return math.trunc(x * fator) / fator

# simulação linearização
def metrics(noLinear, absol, tempo, label:str):
    # erro abs:
    erro_abs = np.abs(noLinear - absol)

    # erro rel. ponto a ponto
    eps = 1e-12
    erro_rel = erro_abs / np.maximum(np.abs(noLinear), eps)

    # norma do erro normalizada -> p/ resumir desvio total
    norma_erro = np.linalg.norm(noLinear - absol)
    norma_rel = norma_erro / np.linalg.norm(noLinear)

    # erro maximo
    erro_max = np.max(erro_abs)

    # rmse
    rmse = np.sqrt(np.mean((noLinear - absol)**2))

    # validade da linearizacao
    limite = .05
    idx_validos = np.where(erro_rel <= limite)[0]

    # ultimo instante que ainda são válidos:
    t_valido = tempo[idx_validos[-1]] if len(idx_validos) > 0 else None


    print("=== Validação da linearização ===")
    print(f"Norma relativa do erro em {label}: {norma_rel*100:.2f}%")
    print(f"Erro máximo em {label}: {trunc(erro_max, 5)}")
    print(f"Validade em {label} até: {math.ceil(t_valido)} h")
    print(f"RMSE {label}: {rmse:.3f}") 

# simulação controle
def metricsControl(Xsp, X_history, tempo):
    X = np.array(X_history)
    tempo = np.array(tempo)

    overshoot = np.max(X) - Xsp
    undershoot = Xsp - np.min(X)

    # metricas
    e = np.array(Xsp) - np.array(X)

    IAE = np.trapezoid(np.abs(e), tempo)
    ISE = np.trapezoid(e**2, tempo)

    return {
        "IAE": IAE, 
        "ISE": ISE,
        "overshoot":overshoot,
        "undershoot":undershoot}