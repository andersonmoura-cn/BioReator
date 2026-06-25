import numpy as np
from scipy.signal import cont2discrete
from config.matrizes_config import matrizes

def bloco(m: int , r: int , P: int , N: int , Ts: int = 0.1):
    # print(m, r)
    S_blocks = []
    Ad, Bd, Cd, Dd = discretizando(matrizes, Ts)

    for i in range(1, N + 1):
        Si = np.zeros((m, r))

        for entrada in range(r):
            x = np.zeros((m, 1))
            u = np.zeros((r, 1))
            u[entrada, 0] = 1.0  # degrau unitário nessa entrada

            for k in range(i):
                y = Cd @ x + Dd @ u
                x = Ad @ x + Bd @ u

            Si[:, entrada] = y.flatten()

        S_blocks.append(Si)

    for j in range(N+1, P+1):
        S_blocks.append(S_blocks[-1])

    return S_blocks


def discretizando(matrizes: np.ndarray, Ts: int ):
    A = matrizes["A"]

    B = matrizes["B"]

    C = matrizes["C"]
    
    D = matrizes["D"]

    if D == 0: D = np.zeros((2, 2))   # KKKKK SIM, EU SEI QUE É KKKKK

    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), Ts)

    return Ad, Bd, Cd, Dd



def matriz_S(m: int , r: int , M: int , P: int , N: int , Ts: int ):
    Sdyn = np.zeros((m * P, r * M))
    S_blocks = bloco(m, r, P, N, Ts)

    for linha in range(P):
        for coluna in range(M):
            idx = linha - coluna 

            if idx >= 0:
                Sdyn[
                    linha*m:(linha+1)*m,
                    coluna*r:(coluna+1)*r
                ] = S_blocks[idx]

    return Sdyn

