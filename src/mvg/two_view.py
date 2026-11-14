"""Two-view geometry: fundamental and essential matrices, relative pose, triangulation.

* The **fundamental matrix** F (3x3, rank 2, 7 degrees of freedom) encodes the
  epipolar constraint x2^T F x1 = 0 for every match: the match of x1 must lie
  on the epipolar line F x1 in image 2.
* With calibrated cameras, the **essential matrix** E = K2^T F K1 = [t]x R
  holds the relative rotation R and the direction of the translation t.
* E has four (R, t) decompositions; only one puts the scene **in front of both
  cameras** (cheirality).
* **Triangulation** recovers each 3D point from its two rays.

Derivations in docs/math/two_view.md.
"""

from __future__ import annotations

import numpy as np

from mvg.homogeneous import apply_transform, normalizing_transform, skew, to_homogeneous
from mvg.ransac import RansacResult, ransac


def _enforce_rank2(F: np.ndarray) -> np.ndarray:
    U, s, Vt = np.linalg.svd(F)
    return U @ np.diag([s[0], s[1], 0.0]) @ Vt


def eight_point(x1: np.ndarray, x2: np.ndarray, normalize: bool = True) -> np.ndarray | None:
    """Fundamental matrix from >= 8 matches (Hartley's normalised 8-point algorithm)."""
    if len(x1) < 8:
        raise ValueError("the 8-point algorithm needs at least 8 matches")
    if normalize:
        T1, T2 = normalizing_transform(x1), normalizing_transform(x2)
        a, b = apply_transform(T1, x1), apply_transform(T2, x2)
    else:
        T1 = T2 = np.eye(3)
        a, b = x1, x2
    # each match gives one row: x2^T F x1 = 0 is linear in the entries of F
    A = np.column_stack([b[:, 0] * a[:, 0], b[:, 0] * a[:, 1], b[:, 0],
                         b[:, 1] * a[:, 0], b[:, 1] * a[:, 1], b[:, 1],
                         a[:, 0], a[:, 1], np.ones(len(a))])
    _, s, Vt = np.linalg.svd(A)
    if len(a) == 8 and s[-2] < 1e-10 * s[0]:
        return None
    F = _enforce_rank2(Vt[-1].reshape(3, 3))
    F = T2.T @ F @ T1
    return F / np.linalg.norm(F)


def sampson_error(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """First-order approximation of the geometric (reprojection) distance to the epipolar constraint, in pixels."""
    a, b = to_homogeneous(x1), to_homogeneous(x2)
    Fa = a @ F.T  # epipolar lines in image 2
    Ftb = b @ F  # epipolar lines in image 1
    num = np.einsum("ij,ij->i", b, Fa)
    den = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2
    return np.abs(num) / np.sqrt(den)


def estimate_fundamental(x1, x2, threshold: float = 1.0, confidence: float = 0.999,
                         rng=None) -> tuple[np.ndarray | None, np.ndarray, RansacResult]:
    result = ransac(
        len(x1),
        fit=lambda idx: eight_point(x1[idx], x2[idx]),
        errors=lambda F, idx: sampson_error(F, x1[idx], x2[idx]),
        sample_size=8,
        threshold=threshold,
        confidence=confidence,
        rng=rng,
    )
    return result.model, result.inliers, result


def essential_from_fundamental(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """E = K2^T F K1, projected onto essential matrices (two equal singular values, one zero)."""
    E = K2.T @ F @ K1
    U, s, Vt = np.linalg.svd(E)
    m = (s[0] + s[1]) / 2
    return U @ np.diag([m, m, 0.0]) @ Vt


def essential_from_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return skew(t) @ R


def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The four (R, t) with E ~ [t]x R; t has unit length."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    R1, R2 = U @ W @ Vt, U @ W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def triangulate(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation: (N, 3) points from matches in two views."""
    X = np.empty((len(x1), 3))
    for i, ((u1, v1), (u2, v2)) in enumerate(zip(x1, x2)):
        # x ~ P X  <=>  x cross (P X) = 0: two independent rows per view
        A = np.array([u1 * P1[2] - P1[0], v1 * P1[2] - P1[1], u2 * P2[2] - P2[0], v2 * P2[2] - P2[1]])
        _, _, Vt = np.linalg.svd(A)
        X[i] = Vt[-1, :3] / Vt[-1, 3]
    return X


def recover_pose(E: np.ndarray, x1: np.ndarray, x2: np.ndarray, K1: np.ndarray, K2: np.ndarray):
    """Pick the decomposition with the most triangulated points in front of both cameras.

    Returns (R, t, mask of points in front, triangulated points) with camera 1
    at the origin: P1 = K1 [I | 0], P2 = K2 [R | t].
    """
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    best = None
    for R, t in decompose_essential(E):
        P2 = K2 @ np.hstack([R, t[:, None]])
        X = triangulate(P1, P2, x1, x2)
        depth1 = X[:, 2]
        depth2 = (X @ R.T + t)[:, 2]
        front = (depth1 > 0) & (depth2 > 0)
        if best is None or front.sum() > best[2].sum():
            best = (R, t, front, X)
    return best


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle of the rotation Ra^T Rb, in degrees.

    Uses atan2(sin, cos) rather than arccos((trace - 1) / 2): arccos loses
    precision near 0 degrees (an error of about 1e-6 degrees for identical
    rotations), while sin comes accurately from the skew-symmetric part R - R^T.
    """
    R = Ra.T @ Rb
    cos = (np.trace(R) - 1) / 2
    sin = np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return float(np.degrees(np.arctan2(sin, cos)))


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def refine_pose(R: np.ndarray, t: np.ndarray, x1, x2, K1, K2, max_iterations: int = 50):
    """Levenberg-Marquardt on the Sampson error over the 5 degrees of freedom of (R, t / |t|).

    R is updated by a small rotation (axis-angle, 3 parameters); t stays on the
    unit sphere by updating it in the 2D tangent plane at the current t.
    """
    from mvg.camera import rodrigues
    from mvg.optimize import levenberg_marquardt

    t = t / np.linalg.norm(t)
    # orthonormal basis of the tangent plane at t
    a = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 1.0, 0])
    b1 = np.cross(t, a)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(t, b1)
    K1i, K2i = np.linalg.inv(K1), np.linalg.inv(K2)

    def unpack(p):
        Rp = rodrigues(p[:3]) @ R
        tp = t + p[3] * b1 + p[4] * b2
        return Rp, tp / np.linalg.norm(tp)

    def residuals(p):
        Rp, tp = unpack(p)
        F = K2i.T @ skew(tp) @ Rp @ K1i
        return sampson_error(F, x1, x2)

    res = levenberg_marquardt(residuals, np.zeros(5), max_iterations=max_iterations)
    Rr, tr = unpack(res.x)
    return Rr, tr, res
