import numpy as np
from numpy.typing import NDArray

def calculo_y0_j(S, delta_u, u_final, N, j):     
    y1 = y2 = soma_11 = soma_21 = soma_12 = soma_22 = 0

    for i in range(j*2, N*2, 2):
        # somatorios entrada e saída
        soma_11 += S[i, 0] * delta_u[i - 2*j]
        soma_21 += S[i + 1, 0] * delta_u[i - 2*j]
        
        soma_12 += S[i, 1] * delta_u[(i + 1) - 2*j]
        soma_22 += S[i + 1, 1] * delta_u[(i + 1) - 2*j]
        
    y1 = soma_11 + soma_12 + (S[2*N - 2, 0] * u_final[0]) + (S[2*N - 2, 1] * u_final[1])
    y2 = soma_21 + soma_22 + (S[2*N - 1, 0] * u_final[0]) + (S[2*N - 1, 1] * u_final[1])
        
    return y1, y2


def y_0(m: int, P: int, N: int, S: NDArray, delta_u: NDArray, u_final: NDArray):
    y_0 = []
    for j in range(1, N+1):
        y1, y2 = calculo_y0_j(S, delta_u, u_final, N, j)
        y_0.append(y1)
        y_0.append(y2)

    for j in range(N+1, P+1):
        y_0.append(y1)
        y_0.append(y2) 

    return np.array(y_0)


