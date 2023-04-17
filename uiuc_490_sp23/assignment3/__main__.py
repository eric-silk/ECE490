import argparse
from typing import Tuple, Callable, List
import numpy as np
import matplotlib.pyplot as plt

from ..Problem import SimpleQuadraticForm, AugmentedLagrangian
from ..Constraints import LinearEqualityConstraint
from ..Optimizers import GradientDescent
from ..LineSearch import Backtracking


def _get_Q_A_b(m: int, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = np.random.rand(n, n) - 0.5
    Q = 10 * Q @ Q.T + 0.1 * np.eye(n)
    A = np.random.normal(size=(m, n))
    b = 2 * (np.random.rand(m) - 0.5)

    return Q, A, b


def _do_optimization(
    m: int,
    n: int,
    epsilon: float,
    inner_iter_count: int,
    c_update: Callable[[float], float],
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
    # Set up the problem
    Q, A, b = _get_Q_A_b(m, n)
    quadratic = SimpleQuadraticForm(Q)
    h = LinearEqualityConstraint(A, b)
    lagrangian = AugmentedLagrangian(quadratic, h)

    # Set up the Optimizer
    line_search = Backtracking(lagrangian, 1.0, 0.1, 0.5)
    optimizer = GradientDescent(lagrangian, line_search)

    x = np.zeros(n)
    xk = [x]
    ck = [lagrangian.c]
    lambda_k = [lagrangian.lambda_]
    i = 0
    while np.linalg.norm(h(x), ord=2) >= epsilon:
        for _ in inner_iter_count:
            x = optimizer(x)
            xk.append(x)
        lagrangian.update_lambda(x)
        lagrangian.c = c_update(lagrangian.c)
        ck.append(lagrangian.c)
        lambda_k.append(lagrangian.lambda_)
        i += 1

    print(f"Converged after {i} outer iterations!")

    return xk, ck, lambda_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, required=True)
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="The convergence tolerance (default: 1e-3)",
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

    c_update1 = lambda c: 1.1 * c
    c_update2 = lambda c: 2 * c
    c_update3 = lambda c: c + 1
    c_update4 = lambda c: c + 10

    c_updates = [c_update1, c_update2, c_update3, c_update4]
    c_updates = [c_update1]

    results = []
    for c_update in c_updates:
        results.append(
            _do_optimization(args.m, args.n, args.epsilon, args.iter, c_update)
        )

    for result in results:
        xk, ck, lambda_k = result
        x_star = xk[-1]
        # There's certainly a vectorizable way of doing this but I just want it to work
        xdist = np.array([np.linalg.norm(x_star - x, ord=2) for x in xk])
        plt.figure(figsize=(16, 9))
        plt.plot(xdist)

    plt.show()
