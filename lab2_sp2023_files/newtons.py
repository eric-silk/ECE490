import numpy as np
import matplotlib.pyplot as plt


class F:
    def __init__(self) -> None:
        pass

    def __call__(self, z) -> np.ndarray:
        assert z.ndim == 1
        assert z.shape[0] == 2

        x, y = z[0], z[1]

        return x**3 * y - x * y**3 - y

    def grad(self, z) -> np.ndarray:
        assert z.ndim == 1
        assert z.shape[0] == 2
        x, y = z[0], z[1]

        grad = np.array([3 * x**2 * y - y**3, x**3 - 3 * x * y**2 - 1])
        return grad

    def hessian(self, z) -> np.ndarray:
        assert z.ndim == 1
        assert z.shape[0] == 2
        x, y = z[0], z[1]
        a = 6 * x * y
        b = 3 * x**2 - 3 * y**2

        hess = np.array([[a, b], [b, -a]])
        return hess

    def hessian_inv(self, z) -> np.ndarray:
        return np.linalg.pinv(self.hessian(z))


def newton_method(z0: np.array, N=50) -> np.array:
    # Write your code here.
    # If needed, you can define other functions as well to be used here.
    # Input z0 and output zN should be numpy.ndarray objects with 2 elements:
    # e.g. np.array([ x , y ]).
    f = F()
    zN = z0
    for i in range(N):
        zN -= f.hessian_inv(zN) @ f.grad(zN)
    return zN


def plot_image(s_points: np.array, n=500, domain=(-1, 1, -1, 1)):
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


if __name__ == "__main__":
    # Example of usage .
    # In this example , the stationary points are (0 , 0 ) , (1 , 1 ) , (2 , 2 ) and (3 , 3 ) .
    # Replace these with the ones obatined in part a).
    stationary_points = np.array(
        [[-1.0 / 2.0, np.sqrt(3) / 2.0], [-1.0 / 2.0, -np.sqrt(3) / 2.0], [1.0, 0.0]]
    )
    plot_image(stationary_points)
