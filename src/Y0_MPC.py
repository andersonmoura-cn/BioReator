import numpy as np
from numpy.typing import NDArray

def calculo_y0_j(S: NDArray, hist_du: NDArray, hist_u: NDArray, N: int, j: int):     
    y1 = y2 = soma_11 = soma_21 = soma_12 = soma_22 = 0

    for i in range(j*2, (N-1)*2, 2):
        # somatorios entrada e saída
        soma_11 += S[i, 0] * hist_du[i - 2*j + 2] # historico de delta u. Ex: u(k - i)
        soma_21 += S[i + 1, 0] * hist_du[i - 2*j + 2]
        
        soma_12 += S[i, 1] * hist_du[(i + 1) - 2*j + 2]
        soma_22 += S[i + 1, 1] * hist_du[(i + 1) - 2*j + 2]
        
    y1 = soma_11 + soma_12 + (S[2*N - 2, 0] * hist_u[2*(N - j - 1)]) + (S[2*N - 2, 1] * hist_u[2*(N - j) - 1])
    y2 = soma_21 + soma_22 + (S[2*N - 1, 0] * hist_u[2*(N - j - 1)]) + (S[2*N - 1, 1] * hist_u[2*(N - j) - 1])
        
    return y1, y2


def y_0(P: int, N: int, S: NDArray, hist_du: NDArray, hist_u: NDArray):
    y_0 = []
    for j in range(1, N):
        y1, y2 = calculo_y0_j(S, hist_du, hist_u, N, j)
        y_0.append(y1)
        y_0.append(y2)

    for j in range(N, P+1):
        y_0.append(y1)
        y_0.append(y2) 

    return np.array(y_0)


