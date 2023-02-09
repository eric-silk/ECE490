#!/usr/bin/env python3
import numpy as np

from ..Problem import Problem, QuadraticForm
from ..Optimizers import Optimizer, GradientDescent
from ..LineSearch import FixedStep, Backtracking


def _do_optimization(
    f: Problem, opt: Optimizer, x0: np.ndarray, epsilon: float, max_iter: int = 1000
) -> None:
    xk = np.copy(x0)
    print(f"f(x0): {f(x0)}")
    norm = None

    for i in range(max_iter):
        xk = opt.step(xk, debug=True)
        norm = np.linalg.norm(f.gradient(xk))
        if norm <= epsilon:
            print(f"Converged at iteration {i}")
            break
    else:
        print("Failed to converge!")

    print(f"x*:\n{xk}\nf(x*): {f(xk)[0]}, ||∇f(x*)||: {norm}")


def assignment1(seed: int, epsilon: float, n: int, max_iter: int) -> None:
    np.random.seed(seed)
    f = QuadraticForm(n=n)

    # TODO choose alpha fixed better
    gradient_descent_fixed_alpha = GradientDescent(f, FixedStep(f, 0.05))
    gradient_descent_armijo = GradientDescent(
        f,
        Backtracking(f, 1.0, 0.1, 0.5),
    )

    x0 = np.random.random(n)

    print(f"ε: {epsilon}")
    _do_optimization(f, gradient_descent_fixed_alpha, x0, epsilon, max_iter=max_iter)
    _do_optimization(f, gradient_descent_armijo, x0, epsilon, max_iter=max_iter)
