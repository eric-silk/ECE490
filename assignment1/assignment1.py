from typing import Optional
import numpy as np


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

        if len(set((q_n0, q_n1, b_n))) != 1 and c_n != (1,):
            raise DimensionMismatch("Matrix dimensions for the problem are invalid!")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return x.T @ self.Q @ x + self.b @ x + self.c

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.Q.T @ x + self.Q @ x + self.b


class BadLineSearchSpec(ValueError):
    """Something is dorked in your Line search Specs"""


class LineSearch:
    def __init__(self, f: Problem):
        self.f = f

    def get_alpha(self, x: np.ndarray) -> float:
        _ = x
        raise NotImplementedError("This is just a base class!")

    def __call__(self, x: np.ndarray) -> float:
        return self.get_alpha(x)


class FixedStep(LineSearch):
    def __init__(self, f: Problem, step_size: float) -> None:
        super().__init__(f)
        if step_size <= 0.0:
            raise BadLineSearchSpec("Stepsize must be >0")
        self.alpha = step_size

    def get_alpha(self, x: np.ndarray) -> float:
        _ = x, f

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

        if not (alpha > 0.0 and alpha < 1.0):
            raise BadLineSearchSpec("alpha (Armijo factor) must be in the range (0, 1)")
        self.alpha = alpha

        if not (beta > 0.0 and beta < 1.0):
            raise BadLineSearchSpec("beta (scale factor) must be in range (0, 1)")
        self.beta = beta

    def get_alpha(self, x: np.ndarray):
        t = self.t0
        grad = f.gradient(x)
        delta_x = -grad
        f_x = f(x)
        inner_product = np.dot(grad, delta_x)

        converged = False
        while not converged:
            x_new = x + t * delta_x
            converged = f(x_new) <= f_x + self.alpha * t * inner_product
            t *= self.beta

        return t


class DimensionMismatch(ValueError):
    """Something in your problem formulation is borked"""


class Optimizer:
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


def do_optimization(f: Problem, opt: Optimizer, x0: np.ndarray, k: int = 1000) -> None:
    xk = np.copy(x0)
    print(f"f(x0): {f(x0)}")
    norm = None

    for i in range(1000):
        xk = opt.step(xk, debug=True)
        norm = np.linalg.norm(f.gradient(xk))
        if norm <= epsilon:
            print(f"Converged at iteration {i}")
            break
    else:
        print("Failed to converge!")

    print(f"x*:\n{xk}\nf(x*): {f(xk)[0]}, ||∇f(x*)||: {norm}")


if __name__ == "__main__":
    np.random.seed(1234)
    epsilon = 1e-6
    n = 10
    f = Problem(n=n)
    # TODO choose alpha fixed better
    gradient_descent_fixed_alpha = GradientDescent(f, FixedStep(f, 0.05))
    gradient_descent_armijo = GradientDescent(
        f,
        Backtracking(f, 1.0, 0.1, 0.5),
    )

    x0 = np.random.random(n)

    do_optimization(f, gradient_descent_fixed_alpha, x0)
    do_optimization(f, gradient_descent_armijo, x0)
