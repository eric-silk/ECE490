from typing import Tuple
import numpy as np


def _get_Q_A_b(m: int, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = np.random.rand(n, n) - 0.5
    Q = 10 * Q @ Q.T + 0.1 * np.eye(n)
    A = np.random.normal(size=(m, n))
    b = 2 * (np.random.rand(m) - 0.5)

    return Q, A, b
