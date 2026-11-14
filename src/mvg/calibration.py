"""Camera calibration from several views of a plane (Zhang, 2000).

1. For each view, estimate the homography H = K [r1 r2 t] (up to scale) from the
   known board points (X, Y) to their image positions.
2. r1 and r2 are orthonormal, so with B = K^-T K^-1 each view gives two linear
   equations in the 6 distinct entries of the symmetric B:
       h1^T B h2 = 0,   h1^T B h1 - h2^T B h2 = 0.
   Three or more views determine B; K follows from B in closed form.
3. Recover each view's pose from K^-1 H.
4. Refine everything (K, radial distortion, all poses) by minimising the
   reprojection error with Levenberg-Marquardt.

Derivations in docs/math/calibration.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvg.camera import distort, rodrigues, rotation_to_axis_angle
from mvg.homography import dlt_homography
from mvg.optimize import levenberg_marquardt


def _v(H: np.ndarray, i: int, j: int) -> np.ndarray:
    """Row such that v . b = h_i^T B h_j, with b = (B11, B12, B22, B13, B23, B33)."""
    hi, hj = H[:, i], H[:, j]
    return np.array([hi[0] * hj[0], hi[0] * hj[1] + hi[1] * hj[0], hi[1] * hj[1],
                     hi[2] * hj[0] + hi[0] * hj[2], hi[2] * hj[1] + hi[1] * hj[2], hi[2] * hj[2]])


def intrinsics_from_homographies(Hs: list[np.ndarray]) -> np.ndarray:
    """Zhang's closed-form K (with skew) from >= 3 plane homographies."""
    V = np.vstack([np.vstack([_v(H, 0, 1), _v(H, 0, 0) - _v(H, 1, 1)]) for H in Hs])
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]
    B11, B12, B22, B13, B23, B33 = b
    if B11 < 0:  # B is only known up to scale (and sign); K^-T K^-1 is positive definite
        B11, B12, B22, B13, B23, B33 = -b
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / (B11 * B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha**2 / lam
    return np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])


def pose_from_homography(K: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t of the board in the camera frame from H ~ K [r1 r2 t]."""
    A = np.linalg.solve(K, H)
    lam = 1.0 / np.linalg.norm(A[:, 0])
    if A[2, 2] * lam < 0:  # the board must be in front of the camera (t_z > 0)
        lam = -lam
    r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)  # nearest rotation in the Frobenius norm
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
    return R, t


@dataclass
class Calibration:
    K: np.ndarray
    dist: np.ndarray  # (k1, k2)
    rvecs: np.ndarray  # (n, 3)
    tvecs: np.ndarray  # (n, 3)
    rms: float  # root-mean-square reprojection error in pixels
    rms_closed_form: float


def project_board(K, dist, rvec, tvec, board_xy):
    X = np.column_stack([board_xy, np.zeros(len(board_xy))])
    Xc = X @ rodrigues(rvec).T + tvec
    xn = Xc[:, :2] / Xc[:, 2:3]
    xd = distort(xn, tuple(dist))
    return xd @ K[:2, :2].T + K[:2, 2]


def calibrate(board_xy: np.ndarray, image_points: list[np.ndarray], skew: bool = False,
              max_iterations: int = 100) -> Calibration:
    """``board_xy``: (N, 2) board coordinates; ``image_points``: one (N, 2) array per view."""
    Hs = [dlt_homography(board_xy, x) for x in image_points]
    K0 = intrinsics_from_homographies(Hs)
    if not skew:
        K0[0, 1] = 0.0
    poses = [pose_from_homography(K0, H) for H in Hs]
    n = len(image_points)
    obs = np.concatenate(image_points)

    def unpack(p):
        fx, fy, cx, cy, s, k1, k2 = p[:7]
        K = np.array([[fx, s if skew else 0.0, cx], [0, fy, cy], [0, 0, 1]])
        rest = p[7:].reshape(n, 6)
        return K, np.array([k1, k2]), rest[:, :3], rest[:, 3:]

    def residuals(p):
        K, dist, rv, tv = unpack(p)
        pred = np.concatenate([project_board(K, dist, rv[i], tv[i], board_xy) for i in range(n)])
        return (pred - obs).ravel()

    p0 = np.concatenate([[K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2], K0[0, 1], 0.0, 0.0],
                         np.concatenate([np.r_[rotation_to_axis_angle(R), t] for R, t in poses])])
    rms0 = np.sqrt(np.mean(residuals(p0).reshape(-1, 2) ** 2) * 2)
    res = levenberg_marquardt(residuals, p0, max_iterations=max_iterations)
    K, dist, rv, tv = unpack(res.x)
    rms = np.sqrt(np.mean(residuals(res.x).reshape(-1, 2) ** 2) * 2)
    return Calibration(K, dist, rv, tv, float(rms), float(rms0))
