"""
A module to implement constraints for optimization problems
"""
from abc import ABC, abstractclassmethod
import numpy as np


class Constraint(ABC):
    def __call__(self, x: np.ndarray, norm: bool = True) -> float:
        return self.evaluate(x, norm=norm)

    @abstractclassmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        pass


class EqualityConstraint(Constraint):
    """
    An Equality Constraint to be enforced.
    """


class InequalityConstraint(Constraint):
    """
    An Inequality Constraint to be enforced.
    """


class LinearEquality(EqualityConstraint):
    """
    A simple linear equality constraint Ax=b, or h(x) = Ax-b = 0
    """

    def __init__(self, A: np.ndarray, b: np.ndarray) -> None:
        self.A = A
        self.b = b

    def evaluate(self, x: np.ndarray, norm: bool = True) -> float:
        """
        The evaluate method, pass in x to check it
        By default, returns the 2-norm of Ax-b; set norm=False to return Ax-b directly
        """
        h = self.A @ x - self.b
        if norm:
            return np.linalg.norm(h, ord=2)
        else:
            return h

    def __call__(self, x: np.ndarray, norm: bool = True) -> float:
        return self.evaluate(x, norm=norm)
