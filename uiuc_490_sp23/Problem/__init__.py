"""
Problem types should go here. For now, we'll assume they require are differentiable functions w/o constraints.
This interface may change as we get more problems.
"""
from typing import Optional, Tuple, Union
from abc import ABC, abstractclassmethod
import numpy as np

from ..Constraints import LinearEqualityConstraint
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

    @abstractclassmethod
    def input_dimension(self) -> int:
        """Gives the expected size of inputs"""
        pass


class SimpleQuadraticForm(Problem):
    """
    A "simple" Quadratic problem x^TQx (no linear b^Tx or bias +c term)
    """

    def __init__(self, Q: np.ndarray) -> None:
        self.Q = Q

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return x.T @ self.Q @ x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.Q.T @ x + self.Q @ x

    def input_dimension(self) -> int:
        return self.Q.shape[1]


class AugmentedLagrangian(Problem):
    """
    A problem encapsulating the augmented lagrangian/method of multipliers for an equality constrained problem
    This DOES assume a linear equality constraint, simplifying the evaluation/gradient calculation
        f: The problem of interest, without constraints
        h: A linear equality constraint object
        lambda0: Optional. If none, zero initialized to the same size as h. If a float, initialized to that value at the size of h. If an array, use that directly.
        c0: The initial value for the scalar c for the augmented lagrangian term. 0 by default.
    """

    def __init__(
        self,
        f: Problem,
        h: LinearEqualityConstraint,
        lambda0: Optional[float] = None,
        c0: float = 0.0,
    ) -> None:
        super().__init__()
        self.f = f
        self.h = h
        if lambda0 is None:
            self.lambda_ = np.zeros(self.h.dimension()) + 0.1
        else:
            self.lambda_ = np.ones(self.h.dimension()) * lambda0

        self.c = c0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        h = self.h(x)
        return (
            self.f(x) + self.lambda_.T @ h + self.c / 2 * np.linalg.norm(h, ord=2) ** 2
        )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.f.gradient(x) + self.h.gradient(x) @ (
            self.lambda_ + self.c * self.h(x)
        )

    def input_dimension(self) -> int:
        return self.f.input_dimension()

    def update_lambda(self, x: np.ndarray) -> None:
        self.lambda_ += self.c * self.h(x)


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
