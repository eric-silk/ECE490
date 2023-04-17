"""
A module to implement constraints for optimization problems
"""
from abc import ABC, abstractclassmethod
import numpy as np


class Constraint(ABC):
    def __call__(self, x: np.ndarray, norm: bool = False) -> float:
        return self.evaluate(x, norm=norm)

    @abstractclassmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractclassmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        pass


class EqualityConstraint(Constraint):
    """
    An Equality Constraint to be enforced.
    """


class InequalityConstraint(Constraint):
    """
    An Inequality Constraint to be enforced.
    """


class LinearEqualityConstraint(EqualityConstraint):
    """
    A simple linear equality constraint Ax=b, or h(x) = Ax-b = 0
    """

    def __init__(self, A: np.ndarray, b: np.ndarray) -> None:
        self.A = A
        self.b = b

    def evaluate(self, x: np.ndarray, norm: bool = False) -> float:
        """
        The evaluate method, pass in x to check it
        By default, returns Ax-b directly. Set to True to return the 2-norm.
        """
        h = self.A @ x - self.b
        if norm:
            return np.linalg.norm(h, ord=2)
        else:
            return h

    def gradient(self, x: np.ndarray) -> np.ndarray:
        _ = x  # Linear, no dependence on x
        return self.A.T

    def dimension(self) -> int:
        return self.A.shape[0]
