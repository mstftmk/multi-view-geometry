"""Non-linear least squares: Gauss-Newton and Levenberg-Marquardt.

Many geometry problems end as "find parameters p that make the residuals r(p)
small", for example reprojection errors. Linear methods (DLT) give a good start
but minimise an algebraic quantity with no physical meaning; refining with
non-linear least squares minimises the actual error in pixels.

Gauss-Newton linearises r around p: r(p + d) ~ r + J d, and solves
(J^T J) d = -J^T r. It converges fast near the minimum but can diverge far from
it. Levenberg-Marquardt adds a damping term, (J^T J + lambda D) d = -J^T r: a
large lambda takes small gradient-descent-like steps, a small lambda becomes
Gauss-Newton. lambda shrinks after a successful step and grows after a failed one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

Residual = Callable[[np.ndarray], np.ndarray]


def numerical_jacobian(fun: Residual, p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central differences, one column per parameter."""
    r0 = fun(p)
    J = np.empty((r0.size, p.size))
    for j in range(p.size):
        step = eps * max(1.0, abs(p[j]))
        dp = np.zeros_like(p)
        dp[j] = step
        J[:, j] = (fun(p + dp) - fun(p - dp)) / (2 * step)
    return J


@dataclass
class LMResult:
    x: np.ndarray
    cost: float  # 0.5 * sum of squared residuals
    iterations: int
    converged: bool
    history: list[float] = field(default_factory=list)


def levenberg_marquardt(
    fun: Residual,
    x0: np.ndarray,
    jac: Callable[[np.ndarray], np.ndarray] | None = None,
    max_iterations: int = 100,
    lam: float = 1e-3,
    tol: float = 1e-10,
    gauss_newton: bool = False,
) -> LMResult:
    """Minimise 0.5 * ||fun(x)||^2.

    With ``gauss_newton=True`` the damping is switched off (lambda = 0) and
    every step is taken, which is useful to show where plain Gauss-Newton fails.
    """
    x = np.asarray(x0, dtype=float).copy()
    jac = jac or (lambda p: numerical_jacobian(fun, p))
    r = fun(x)
    cost = 0.5 * r @ r
    history = [cost]
    for it in range(1, max_iterations + 1):
        J = jac(x)
        A = J.T @ J
        g = J.T @ r
        if np.linalg.norm(g, np.inf) < tol:
            return LMResult(x, cost, it - 1, True, history)
        while True:
            # Marquardt's scaling: damp each parameter by its own curvature
            damping = 0.0 if gauss_newton else lam
            D = np.diag(np.maximum(np.diag(A), 1e-12))
            try:
                step = np.linalg.solve(A + damping * D, -g)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(A + damping * D, g, rcond=None)[0]
            x_new = x + step
            r_new = fun(x_new)
            cost_new = 0.5 * r_new @ r_new
            if gauss_newton or cost_new < cost:
                break
            lam *= 10
            if lam > 1e12:
                return LMResult(x, cost, it, False, history)
        improvement = cost - cost_new
        x, r, cost = x_new, r_new, cost_new
        history.append(cost)
        if not gauss_newton:
            lam = max(lam / 10, 1e-12)
        if np.linalg.norm(step) < tol * (np.linalg.norm(x) + tol) or abs(improvement) < tol * max(cost, 1e-30):
            return LMResult(x, cost, it, True, history)
    return LMResult(x, cost, max_iterations, False, history)
