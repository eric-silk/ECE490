#!/usr/bin/env python3
import time

import numpy as np

from ..Problem import Problem, QuadraticForm
from ..Optimizers import Optimizer, GradientDescent
from ..LineSearch import FixedStep, Backtracking


def _do_optimization(
    f: Problem, opt: Optimizer, x0: np.ndarray, epsilon: float, max_iter: int = 1000
) -> None:
    """
    Do the optimization.
    args:
        f: Problem object with a gradient() method to check convergence
        opt: An optimizer object with a step() method
        x0: Initial value for x
        epsilon: Absolute value of the 2-norm of the gradient for convergence
        max_iter: Maximum number of iterations allowed before failure (default=1000)
    """
    xk = np.copy(x0)
    norm = None

    t0 = time.time()
    for i in range(max_iter):
        xk = opt.step(xk, debug=True)
        norm = np.linalg.norm(f.gradient(xk))
        if norm <= epsilon:
            t1 = time.time()
            print(f"Converged at iteration {i} in {t1-t0} seconds.")
            break
    else:
        t1 = time.time()
        print(f"Failed to converge in {max_iter} iterations! Ran for {t1-t0} seconds.")

    print(f"x*:\n{xk}\nf(x*): {f(xk)[0]}, ||∇f(x*)||: {norm}")


def _directly_calculate_minimizer(f: QuadraticForm) -> None:
    """
    Given a Quadratic Form problem, directly solve for the minimum.
    args:
        f: A QuadraticForm Problem object.
    """
    Q = f.get_Q()
    b = f.get_b()
    # The quadratic form is normally listed as (1/2)(x^T Q x) - bx + c with a minimizer at Qx=b
    # However, our problem is of the form x^T Q x + bx + c, so we need to solve 2Qx = -b
    x_star = np.linalg.lstsq(2 * Q, -b, rcond=None)[0]
    grad_norm = np.linalg.norm(f.gradient(x_star))

    print(f"x*:\n{x_star}\nf(x*): {f(x_star)[0]}, ||∇f(x*)||: {grad_norm}")


def _calculate_optimal_alpha(f: QuadraticForm, scaling: float = 0.99) -> float:
    """
    Given a QuadraticForm problem, determine the optimal stepsize alpha.
    args:
        f: a QuadraticForm problem
        scaling: We need alpha < 2/L, so calculate 2/L and scale it by this value (default=0.99)
    """
    Q = 2 * f.get_Q()
    s = np.linalg.svd(Q, compute_uv=False)
    L = np.max(s).item()
    # Make it just slightly less than 2/L
    alpha = (2 / L) * 0.99
    print("alpha:", alpha)
    return alpha


def assignment1(seed: int, epsilon: float, n: int, max_iter: int) -> None:
    np.set_printoptions(linewidth=1000)
    np.random.seed(seed)
    f = QuadraticForm(n=n)

    step_size = _calculate_optimal_alpha(f)
    gradient_descent_fixed_alpha = GradientDescent(f, FixedStep(f, step_size))
    gradient_descent_armijo = GradientDescent(
        f,
        Backtracking(f, 1.0, 0.1, 0.5),
    )

    x0 = np.random.random(n)

    print(f"ε: {epsilon}")
    print("=" * 80)
    _do_optimization(f, gradient_descent_fixed_alpha, x0, epsilon, max_iter=max_iter)
    print("=" * 80)
    _do_optimization(f, gradient_descent_armijo, x0, epsilon, max_iter=max_iter)
    print("=" * 80)
    _directly_calculate_minimizer(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="assignment1",
        description=(
            "A basic Gradient Descent implementation to "
            "solve a randomly generated quadratic problem"
        ),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1234,
        help="The seed to use for randomization (default=1234)",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=1e-6,
        help="Optimization convergence tolerance (float, >=0, default=1e-6)",
    )
    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        default=10,
        help="The problem dimension (integer, >0, default=10)",
    )
    parser.add_argument(
        "-k",
        "--max_iter",
        type=int,
        default=1000,
        help="The maximum number of iterations allowed (integer, >0, default=1000)",
    )
    args = parser.parse_args()
    assignment1(args.seed, args.epsilon, args.dimension, args.max_iter)
