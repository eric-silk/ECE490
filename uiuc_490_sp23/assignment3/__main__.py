import argparse
from typing import Tuple, Callable, List
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, LinearConstraint

from ..Problem import SimpleQuadraticForm, AugmentedLagrangian
from ..Constraints import LinearEqualityConstraint
from ..Optimizers import GradientDescent
from ..LineSearch import Backtracking


def _get_Q_A_b(m: int, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    As provided by the lab assignment, generate a a PSD Q matrix along with some randomized linear equality constraints
    """
    Q = np.random.rand(n, n) - 0.5
    Q = 10 * Q @ Q.T + 0.1 * np.eye(n)
    A = np.random.normal(size=(m, n))
    b = 2 * (np.random.rand(m) - 0.5)

    return Q, A, b


def _get_solution_via_scipy(Q: np.ndarray, A: np.ndarray, b: np.ndarray):
    """
    This function exists to give us ground truth to compare our convergence results to
    """
    x0 = np.random.random(Q.shape[1])
    quadratic = SimpleQuadraticForm(Q)
    h = LinearEqualityConstraint(A, b)
    lagrangian = AugmentedLagrangian(quadratic, h)
    linear_constraints = LinearConstraint(A, b, b)
    res = minimize(
        lagrangian, x0, method="trust-constr", constraints=[linear_constraints]
    )
    return res["x"]


def _do_optimization(
    Q: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    inner_iter_count: int,
    c_update: Callable[[float], float],
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
    """
    Do the optimization loop for a given set of params.
    """
    # Set up the problem
    quadratic = SimpleQuadraticForm(Q)
    h = LinearEqualityConstraint(A, b)
    lagrangian = AugmentedLagrangian(quadratic, h)

    # Set up the Optimizer
    line_search = Backtracking(lagrangian, 1.0, 0.1, 0.5)
    optimizer = GradientDescent(lagrangian, line_search)

    n = Q.shape[1]
    x = np.ones(n)
    xk = [x]
    ck = [lagrangian.c]
    lambda_k = [lagrangian.lambda_]
    i = 0
    while np.linalg.norm(h(x), ord=2) >= epsilon:
        for _ in range(inner_iter_count):
            x = optimizer(x)
            xk.append(x)
            if np.linalg.norm(lagrangian.gradient(x), ord=2) <= epsilon:
                # GD converged, early termination of inner loop
                break

        lagrangian.update_lambda(x)
        lagrangian.c = c_update(lagrangian.c)
        ck.append(lagrangian.c)
        lambda_k.append(lagrangian.lambda_)
        i += 1

    print(f"Converged after {i} outer iterations and {len(xk)} total iterations!")
    print("final x:", x)
    print("final L(x):", lagrangian.f(x))

    return xk, ck, lambda_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, required=True)
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="The convergence tolerance (default: 1e-4)",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=100,
        help="The inner (gradient descent) iteration count (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="The seed to the RNG (default: 1234)"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    Q, A, b = _get_Q_A_b(args.m, args.n)

    x_star = _get_solution_via_scipy(Q, A, b)
    print("Ground truth value:", x_star)

    c_updates = [
        ("c=1000", lambda c: 1000),
        ("1.1*c", lambda c: 1.1 * c),
        ("2*c", lambda c: 2 * c),
        ("c+1", lambda c: c + 1),
        ("c+10", lambda c: c + 10),
    ]

    results = []

    for _, c_update in c_updates:
        results.append(_do_optimization(Q, A, b, args.epsilon, args.iter, c_update))

    plt.figure(figsize=(16, 9))
    plt.title(f"x*: {x_star}")
    for (name, _), result in zip(c_updates, results):
        xk, ck, lambda_k = result
        # There's certainly a vectorizable way of doing this but I just want it to work
        xdist = np.array([np.linalg.norm(x_star - x, ord=2) for x in xk])
        plt.plot(xdist, label=name)
        plt.xlabel("Iteration count (inner AND outer)")
        plt.ylabel("Distance metric from x*")
    plt.legend()

    plt.figure(figsize=(16, 9))
    plt.title(f"x*: {x_star}")
    for (name, _), result in zip(c_updates, results):
        xk, ck, lambda_k = result
        # There's certainly a vectorizable way of doing this but I just want it to work
        xdist = np.array([np.linalg.norm(x_star - x, ord=2) for x in xk])
        plt.semilogy(xdist, label=name)
        plt.xlabel("Iteration count (inner AND outer)")
        plt.ylabel("Distance metric from x*")

    plt.legend()
    plt.show()
