#!/usr/bin/env python3
from typing import Tuple
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt


class F:
    def __init__(self) -> None:
        pass

    def __call__(self, z: np.ndarray) -> np.ndarray:
        assert z.ndim == 1
        assert z.shape[0] == 2

        x, y = z[0], z[1]

        return x**3 * y - x * y**3 - y

    def grad(self, z: np.ndarray) -> np.ndarray:
        assert z.ndim == 1
        assert z.shape[0] == 2
        x, y = z[0], z[1]

        grad = np.array([3 * x**2 * y - y**3, x**3 - 3 * x * y**2 - 1])
        return grad

    def hessian(self, z: np.ndarray) -> np.ndarray:
        assert z.ndim == 1
        assert z.shape[0] == 2
        x, y = z[0], z[1]
        a = 6 * x * y
        b = 3 * x**2 - 3 * y**2

        hess = np.array([[a, b], [b, -a]])
        return hess

    def hessian_inv(self, z: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(self.hessian(z))


def newton_method(z0: np.array, N: int = 50) -> np.ndarray:
    # Write your code here.
    # If needed, you can define other functions as well to be used here.
    # Input z0 and output zN should be numpy.ndarray objects with 2 elements:
    # e.g. np.array([ x , y ]).
    f = F()
    zN = z0
    # TODO add convergence check?
    for i in range(N):
        if not np.any(zN):
            # Per Piazza, stop iterating if you land at the origin
            return zN
        zN -= f.hessian_inv(zN) @ f.grad(zN)

    return zN


def plot_image(
    s_points: np.array,
    n: int = 500,
    domain: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
):
    m = np.zeros((n, n))
    xmin, xmax, ymin, ymax = domain
    for ix, x in enumerate(np.linspace(xmin, xmax, n)):
        for iy, y in enumerate(np.linspace(ymin, ymax, n)):
            z0 = np.array([x, y])
            zN = newton_method(z0)
            code = np.argmin(np.linalg.norm(s_points - zN, ord=2, axis=1))
            m[iy, ix] = code

    plt.imshow(m, cmap="brg")
    plt.axis("off ")
    plt.savefig("q2_hw3.png ")
    plt.show()


def job(job_spec: Tuple[np.ndarray, int, float, int, float]) -> Tuple[int, int, int]:
    s_points, ix, x, iy, y = job_spec
    z0 = np.array([x, y])
    zN = newton_method(z0)
    code = np.argmin(np.linalg.norm(s_points - zN, ord=2, axis=1))
    rslt = (iy, ix, code)

    return rslt


def plot_image_parallel(
    s_points: np.ndarray,
    n: int = 500,
    domain: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
):
    m = np.zeros((n, n))
    xmin, xmax, ymin, ymax = domain
    job_specs = []
    for ix, x in enumerate(np.linspace(xmin, xmax, n)):
        for iy, y in enumerate(np.linspace(ymin, ymax, n)):
            job_specs.append((s_points, ix, x, iy, y))

    with mp.Pool() as pool:
        rslts = pool.map(job, job_specs)

    for rslt in rslts:
        iy, ix, code = rslt
        m[iy, ix] = code

    plt.imshow(m, cmap="brg")
    for p in s_points:
        plt.scatter(p[0], p[1])
    plt.axis("off")
    # plt.savefig("q2_hw3.png")
    plt.show()


def main() -> None:
    # Example of usage .
    # In this example , the stationary points are (0 , 0 ) , (1 , 1 ) , (2 , 2 ) and (3 , 3 ) .
    # Replace these with the ones obatined in part a).

    # Calculated using WolframAlpha
    stationary_points = np.array(
        [
            [-1.0 / 2.0, np.sqrt(3.0) / 2.0],
            [-1.0 / 2.0, -np.sqrt(3.0) / 2.0],
            [1.0, 0.0],
        ]
    )

    # Old serial implementation, de-comment to enable
    # plot_image(stationary_points)
    plot_image_parallel(stationary_points, n=500)
