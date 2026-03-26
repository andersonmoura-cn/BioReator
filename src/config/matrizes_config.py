import numpy as np
from config.config import values

matrizes = {
    "A": np.array([[0.0,     0.138],
              [-0.875, -0.695]]),
    "B":   np.array([[-values["X_bar"], 0.0],
              [(values["Sf"] - values["S_bar"]), values["D"]]]),
    "C":    np.array([[1.0, 0.0],
                      [0.0, 1.0]]),
    "D": 0.0
}