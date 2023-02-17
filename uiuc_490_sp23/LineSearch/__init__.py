from abc import ABC, abstractmethod

import numpy as np

from ..Problem import Problem
from ..Exceptions import BadLineSearchSpec


class LineSearch(ABC):
    def __init__(self, f: Problem):
        self.f = f

    @abstractmethod
    def get_alpha(self, x: np.ndarray) -> float:
        pass

    def __call__(self, x: np.ndarray) -> float:
        return self.get_alpha(x)


class FixedStep(LineSearch):
    def __init__(self, f: Problem, step_size: float) -> None:
        super().__init__(f)
        if step_size <= 0.0:
            raise BadLineSearchSpec("Stepsize must be >0")
        self.alpha = step_size

    def get_alpha(self, x: np.ndarray) -> float:
        _ = x

        return self.alpha


class Backtracking(LineSearch):
    """
    Using the notation in Algorithm 9.2 of Boyd & Vandenberghe:
        t0: initial step size
        alpha: Armijo constant
        beta: backtracking factor
    """

    def __init__(self, f: Problem, t0: float, alpha: float, beta: float) -> None:
        super().__init__(f)
        if t0 <= 0:
            raise BadLineSearchSpec("t0 (initial step-size) must be >=0")
        self.t0 = t0

        if not (alpha > 0.0 and alpha < 0.5):
            raise BadLineSearchSpec(
                "alpha (Armijo factor) must be in the range (0, 0.5)"
            )
        self.alpha = alpha

        if not (beta > 0.0 and beta < 1.0):
            raise BadLineSearchSpec("beta (scale factor) must be in range (0, 1)")
        self.beta = beta

    def get_alpha(self, x: np.ndarray):
        t = self.t0
        grad = self.f.gradient(x)
        delta_x = -grad
        f_x = self.f(x)
        inner_product = np.dot(grad, delta_x)

        converged = False
        while not converged:
            x_new = x + t * delta_x
            converged = self.f(x_new) <= f_x + self.alpha * t * inner_product
            t *= self.beta

        return t
