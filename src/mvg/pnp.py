"""Camera pose from known 3D points (Perspective-n-Point).

With K known, normalised image points x_n = K^-1 x satisfy x_n ~ [R | t] X.
That is a DLT for the 12 entries of [R | t] (2 equations per point, so at least
6 points), followed by projecting the left 3x3 block onto the nearest rotation,
and then Levenberg-Marquardt on the reprojection error in pixels. RANSAC
around the DLT handles wrong 2D-3D matches.

Derivations in docs/math/pnp_and_bundle_adjustment.md.
"""

from __future__ import annotations

import numpy as np

from mvg.camera import project, projection_matrix, rodrigues, rotation_to_axis_angle
from mvg.homogeneous import apply_transform, normalizing_transform, to_homogeneous
from mvg.optimize import levenberg_marquardt
from mvg.ransac import ransac


def pnp_dlt(X: np.ndarray, x: np.ndarray, K: np.ndarray):
    """Linear pose from >= 6 2D-3D matches; returns (R, t) or None if degenerate."""
    if len(X) < 6:
        raise ValueError("the DLT needs at least 6 points")
    xn = np.linalg.solve(K, to_homogeneous(x).T).T[:, :2]
    T3 = normalizing_transform(X)  # condition the 3D points as well
    Xn = apply_transform(T3, X)
    Xh = to_homogeneous(Xn)
    zeros = np.zeros_like(Xh)
    A = np.vstack([np.hstack([Xh, zeros, -xn[:, :1] * Xh]), np.hstack([zeros, Xh, -xn[:, 1:2] * Xh])])
    _, s, Vt = np.linalg.svd(A)
    if s[-2] < 1e-10 * s[0]:
        return None
    M = Vt[-1].reshape(3, 4) @ T3  # undo the 3D normalisation
    U, S, Vt3 = np.linalg.svd(M[:, :3])
    R = U @ Vt3
    scale = S.mean()
    if np.linalg.det(R) < 0:
        R, scale = -R, -scale
    t = M[:, 3] / scale
    # the solution is defined up to sign: put the points in front of the camera
    if np.median((X @ R.T + t)[:, 2]) < 0:
        return None
    return R, t


def reprojection_errors(R, t, X, x, K) -> np.ndarray:
    return np.linalg.norm(project(projection_matrix(K, R, t), X) - x, axis=1)


def refine_pnp(R, t, X, x, K, max_iterations: int = 50):
    def residuals(p):
        return (project(projection_matrix(K, rodrigues(p[:3]), p[3:]), X) - x).ravel()

    res = levenberg_marquardt(residuals, np.r_[rotation_to_axis_angle(R), t], max_iterations=max_iterations)
    return rodrigues(res.x[:3]), res.x[3:], res


def solve_pnp(X, x, K, threshold: float = 3.0, refine: bool = True, rng=None, init=None):
    """RANSAC (6-point samples) + LM refinement; returns (R, t, inlier mask).

    Each sample is solved with the linear DLT, or, when ``init = (R, t)`` is
    given, by Levenberg-Marquardt started from that pose. The DLT becomes badly
    conditioned when the field of view is narrow and the points span little
    depth (see docs/math/pnp_and_bundle_adjustment.md); in incremental
    reconstruction the previous camera is a good starting point instead.
    """

    def fit(idx):
        if init is None:
            return pnp_dlt(X[idx], x[idx], K)
        R_, t_, res = refine_pnp(init[0], init[1], X[idx], x[idx], K, max_iterations=30)
        return (R_, t_) if np.isfinite(res.cost) else None

    result = ransac(
        len(X),
        fit=fit,
        errors=lambda model, idx: reprojection_errors(*model, X[idx], x[idx], K),
        sample_size=6,
        threshold=threshold,
        rng=rng,
    )
    if result.model is None:
        return None, None, result.inliers
    R, t = result.model
    mask = result.inliers
    if refine and mask.sum() >= 6:
        R, t, _ = refine_pnp(R, t, X[mask], x[mask], K)
        mask = reprojection_errors(R, t, X, x, K) < threshold
    return R, t, mask
