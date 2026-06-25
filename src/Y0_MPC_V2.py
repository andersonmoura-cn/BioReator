import numpy as np
from numpy.typing import NDArray

def calculo_y0_j(S, hist_du, hist_u, N, j):     
    soma_11 = soma_21 = soma_12 = soma_22 = 0

    for i in range(j*2, (N-1)*2, 2):
        du1 = hist_du[i - 2*j]
        du2 = hist_du[i - 2*j + 1]

        soma_11 += S[i, 0]     * du1
        soma_21 += S[i + 1, 0] * du1
        
        soma_12 += S[i, 1]     * du2
        soma_22 += S[i + 1, 1] * du2
        
    u1_antigo = hist_u[2*(N - j - 1)]
    u2_antigo = hist_u[2*(N - j - 1) + 1]

    y1 = soma_11 + soma_12 + S[2*N - 2, 0]*u1_antigo + S[2*N - 2, 1]*u2_antigo
    y2 = soma_21 + soma_22 + S[2*N - 1, 0]*u1_antigo + S[2*N - 1, 1]*u2_antigo
        
    return y1, y2


def y_0(P, N, S, hist_du, hist_u):
    y0 = []

    for j in range(1, N+1):
        y1, y2 = calculo_y0_j(S, hist_du, hist_u, N, j)
        y0.append(y1)
        y0.append(y2)

    for j in range(N+1, P+1):
        y0.append(y1)
        y0.append(y2) 

    return np.array(y0)


