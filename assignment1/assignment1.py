from typing import Optional
import numpy as np


class BadLineSearchSpec(ValueError):
    """Something is dorked in your Line search Specs"""


class LineSearch:
    def get_alpha(self, *args, **kwargs):
        raise NotImplementedError("This is just a base class!")


class FixedStep(LineSearch):
    def __init__(self, step_size: float) -> None:
        if step_size <= 0.0:
            raise BadLineSearchSpec("Stepsize must be >0")
        self.alpha = step_size

    def get_alpha(self, *args, **kwargs):
        _ = args
        _ = kwargs

        return self.alpha


class DimensionMismatch(ValueError):
    """Something in your problem formulation is borked"""


class Problem:
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

        if len(set(q_n0, q_n1, b_n)) != 1 and c_n != (1,):
            raise DimensionMismatch("Matrix dimensions for the problem are invalid!")

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return x.T @ self.Q @ x + self.b @ x + self.c

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.A.T @ x + self.A @ x + self.b


class Optimizer:
    pass


class GradientDescent(Optimizer):
    def __init__(self, problem: Problem, line_search: LineSearch) -> None:
        self.problem = problem
        self.line_search = line_search

    def step(self, x: np.ndarray) -> np.ndarray:
        alpha = self.line_search.get_alpha(x)

        return x - alpha * self.problem.gradient(x)
