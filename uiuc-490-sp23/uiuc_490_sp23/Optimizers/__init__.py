from abc import ABC, abstractmethod

import numpy as np

from ..LineSearch import LineSearch
from ..Problem import Problem


class Optimizer(ABC):
    @abstractmethod
    def step(self, x: np.ndarray, debug: bool = False) -> np.ndarray:
        pass


class GradientDescent(Optimizer):
    def __init__(self, f: Problem, line_search: LineSearch) -> None:
        self.f = f
        self.line_search = line_search

    def step(self, x: np.ndarray, debug: bool = False) -> np.ndarray:
        alpha = self.line_search(x)
        new_x = x - alpha * self.f.gradient(x)
        if debug:
            if self.f(new_x) >= self.f(x):
                print("Optimization increased value, check your value of alpha...")
                print(f"alpha = {alpha}, ||x||: {np.linalg.norm(x)}")
            if not np.all(np.isfinite(new_x)):
                raise ValueError("x became nonfinite!")

        return x - alpha * self.f.gradient(x)
