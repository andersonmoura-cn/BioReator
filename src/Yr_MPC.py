import numpy as np
from numpy.typing import NDArray

def alvo(alfa: NDArray, y_k: NDArray, y_sp: NDArray, m: int, P: int):
    y_0 = np.zeros((m*P))

    for j in range(0, 2*P, 2):
        passinho = j//2 + 1

        y_0[j] = (alfa[0]**passinho) * y_k[0] + (1 - (alfa[0]**passinho)) * y_sp[0]
        y_0[j+1] = (alfa[1]**passinho) * y_k[1] + (1 - (alfa[1]**passinho)) * y_sp[1]

    return y_0