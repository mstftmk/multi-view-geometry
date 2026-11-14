"""The pinhole camera, lens distortion and rotations.

A 3D point X (world coordinates) reaches the image through

    x ~ K [R | t] X

R and t move the point into the camera frame (extrinsics), and K maps the
camera frame to pixels (intrinsics: focal lengths, principal point, skew).
The 3x4 matrix P = K [R | t] is the camera matrix.
"""

from __future__ import annotations

import numpy as np

from mvg.homogeneous import from_homogeneous, skew, to_homogeneous


def intrinsics(fx: float, fy: float | None = None, cx: float = 0.0, cy: float = 0.0,
               skew_: float = 0.0) -> np.ndarray:
    fy = fx if fy is None else fy
    return np.array([[fx, skew_, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle vector (axis * angle) -> rotation matrix (Rodrigues' formula)."""
    rvec = np.asarray(rvec, dtype=float)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3) + skew(rvec)  # first-order term keeps derivatives right near 0
    k = skew(rvec / theta)
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def rotation_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Inverse of rodrigues (angles in [0, pi])."""
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-12:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # near 180 degrees: axis from the largest column of R + I
        M = (R + np.eye(3)) / 2
        axis = M[:, np.argmax(np.diag(M))]
        return theta * axis / np.linalg.norm(axis)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta * w / (2 * np.sin(theta))


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([R, np.asarray(t, dtype=float).reshape(3, 1)])


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(N, 3) world points -> (N, 2) pixels."""
    return from_homogeneous(to_homogeneous(X) @ P.T)


def camera_center(P: np.ndarray) -> np.ndarray:
    """The centre C satisfies P C = 0 (the null space of P)."""
    _, _, Vt = np.linalg.svd(P)
    C = Vt[-1]
    return C[:3] / C[3]


def rq(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RQ decomposition M = R Q, R upper triangular with a positive diagonal, Q orthogonal."""
    flip = np.flipud(np.eye(3))
    Q_, R_ = np.linalg.qr((flip @ M).T)
    R = flip @ R_.T @ flip
    Q = flip @ Q_.T
    signs = np.diag(np.sign(np.diag(R)))
    return R @ signs, signs @ Q


def decompose_projection(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split P into K, R, t (P = K [R | t] up to scale) by RQ-decomposing its left 3x3 block."""
    M = P[:, :3]
    K, R = rq(M)
    if np.linalg.det(R) < 0:  # P is only known up to sign
        P = -P
        K, R = rq(P[:, :3])
    # K still carries the unknown scale of P, so solving with it gives t directly
    t = np.linalg.solve(K, P[:, 3])
    return K / K[2, 2], R, t


# ---------------------------------------------------------------- distortion
def distort(xn: np.ndarray, k: tuple[float, ...]) -> np.ndarray:
    """Radial distortion of normalised image points: x_d = x (1 + k1 r^2 + k2 r^4 + ...)."""
    r2 = (xn**2).sum(axis=1, keepdims=True)
    factor = 1 + sum(ki * r2 ** (i + 1) for i, ki in enumerate(k))
    return xn * factor


def undistort(xd: np.ndarray, k: tuple[float, ...], iterations: int = 20) -> np.ndarray:
    """Invert radial distortion by fixed-point iteration (no closed form exists)."""
    x = xd.copy()
    for _ in range(iterations):
        r2 = (x**2).sum(axis=1, keepdims=True)
        x = xd / (1 + sum(ki * r2 ** (i + 1) for i, ki in enumerate(k)))
    return x
