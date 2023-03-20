import asgn_source as asou
import numpy as np
import scipy.optimize as so


MAX_ITER = 1000
EPSILON = 0.01
SEED = 1234


class F:
    def __init__(self, Q, b, c):
        self.Q = Q
        self.b = b
        self.c = c

    def __call__(self, x):
        return asou.opfun(x, self.Q, self.b, self.c)

    def grad(self, x):
        return x.T @ self.Q + self.b


def armijo(x, f, t0=1.0, alpha=0.1, beta=0.5):
    t = t0
    grad_x = f.grad(x)
    delta_x = -grad_x
    f_x = f(x)
    inner_product = np.dot(grad_x, delta_x)
    converged = False
    while not converged:
        x_new = x + t * delta_x
        converged = f(x_new) <= f_x + alpha * t * inner_product
        t *= beta

    return t


def grad_descent(x, f):
    alpha = armijo(x, f)
    return x - alpha * f.grad(x)


def project_coordinate(x, constraint_min, constraint_max, n):
    ### write your code here
    assert x.ndim == 1
    assert x.shape[0] == n
    assert np.all(constraint_min <= constraint_max)

    x_projected = np.maximum(np.minimum(x, constraint_max), constraint_min)

    return x_projected


def run_projected_GD(constraint_min, constraint_max, Q, b, c, n):
    ### write your code here
    result = None
    f = F(Q, b, c)

    x0 = np.random.rand(n)
    x_i = x0
    for i in range(MAX_ITER):
        x_next = grad_descent(x_i, f)
        x_next = project_coordinate(x_next, constraint_min, constraint_max, n)
        diff = np.linalg.norm(x_next - x_i, ord=2)
        if diff <= EPSILON:
            return x_next
        x_i = x_next

    else:
        raise RuntimeError(f"Failed to converge in {MAX_ITER} iterations!")


def check_solution(constraint_min, constraint_max, f, n):
    bounds = so.LinearConstraint(np.eye(n), lb=constraint_min, ub=constraint_max)
    result = so.minimize(f, np.random.rand(n), constraints=(bounds,))
    return result


if __name__ == "__main__":
    n = 25
    np.random.seed(SEED)
    constraint_min, constraint_max, Q_val, b_val, c_val = asou.get_parameters(n)
    armijo_sol = run_projected_GD(
        constraint_min, constraint_max, Q_val, b_val, c_val, n
    )

    f = F(Q_val, b_val, c_val)

    so_rslt = check_solution(constraint_min, constraint_max, f, n)

    print("Mine:", armijo_sol)
    print("Scipy:", so_rslt.x)
    print("Normed Difference:", np.linalg.norm(so_rslt.x - armijo_sol))
