"""
Problem types should go here. For now, we'll assume they require are differentiable functions w/o constraints.
This interface may change as we get more problems.
"""
from typing import Optional
from abc import ABC, abstractclassmethod
import numpy as np

from ..Exceptions import DimensionMismatch


class Problem(ABC):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    @abstractclassmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractclassmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        pass


class QuadraticForm(Problem):
    def __init__(
        self,
        Q: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        n: Optional[int] = None,
    ) -> None:
        if all(v is None for v in [Q, b, c, n]):
            raise ValueError(
                "You must provide either a dimension N or at least one of Q, b, or c"
            )
        if Q is None:
            Q = np.random.rand(n, n) - 0.5
            self.Q = 10 * Q @ Q.T
        else:
            self.Q = Q
        if b is None:
            self.b = 5 * (np.random.rand(n) - 0.5)
        else:
            self.b = b
        if c is None:
            self.c = 2 * (np.random.rand(1) - 0.5)
        else:
            self.c = c

        # Check the dims
        q_n0, q_n1 = self.Q.shape
        b_n = self.b.shape
        c_n = self.c.shape

        if len(set((q_n0, q_n1, b_n))) != 1 and c_n != (1,):
            raise DimensionMismatch("Matrix dimensions for the problem are invalid!")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return x.T @ self.Q @ x + self.b @ x + self.c

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.Q.T @ x + self.Q @ x + self.b
