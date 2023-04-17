"""
Problem types should go here. For now, we'll assume they require are differentiable functions w/o constraints.
This interface may change as we get more problems.
"""
from typing import Optional, Tuple
from abc import ABC, abstractclassmethod
import numpy as np

from ..Constraints import EqualityConstraint
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


class SimpleQuadraticForm(Problem):
    """
    A "simple" Quadratic problem x^TQx (no linear b^Tx or bias +c term)
    """

    def __init__(self, Q: np.ndarray) -> None:
        super().__init__()
        self.Q = Q

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return x.T @ self.Q @ x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.Q.T @ x + self.Q @ x


class AugmentedLagrangian(Problem):
    """
    A problem encapsulating the augmented lagrangian/method of multipliers for an equality constrained problem
    """

    def __init__(
        self,
        f: Problem,
        h: EqualityConstraint,
        lambda0: Optional[np.ndarray] = None,
        c0: float = 0.0,
    ) -> None:
        super().__init__()
        self.f = f
        self.h = h
        # TODO looks like we need to defer the initialization of lambda0?
        if lambda0 is None:
            self.lambda_ = None
        else:
            self.lambda_ = lambda0
        self.c = c0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        h = self.h(x)
        return self.f(x) + self.lambda_.T @ h + self.c / 2 * np.norm(h, ord=2) ** 2


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
        return x.T @ self.Q @ x + np.dot(self.b.flatten(), x.flatten()) + self.c

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.Q.T @ x + self.Q @ x + self.b

    def get_Q(self) -> np.ndarray:
        """Return the matrix Q from x^TQx+bx+c"""
        return self.Q

    def get_b(self) -> np.ndarray:
        """Return the vector b from x^TQx+bx+c"""
        return self.b

    def get_c(self) -> float:
        """Return the constant from x^TQx+bx+c"""
        return self.c

    def get_values(self) -> Tuple[np.ndarray, np.ndarray, float]:
        return (self.Q, self.b, self.c)
