"""Homography estimation: the projective map between two images of a plane.

A homography H (3x3, 8 degrees of freedom) relates two views of a planar
scene, or any two views taken by a camera that only rotates (which is why
panoramas work). Each point match x -> x' gives x' ~ H x, which is two linear
equations in the 9 entries of H, so 4 matches determine it.

Pipeline used here:

1. **Normalised DLT** (direct linear transform): stack the equations, take the
   right singular vector with the smallest singular value. Hartley
   normalisation of both point sets first keeps the system well conditioned.
2. **RANSAC** around the DLT to ignore wrong matches.
3. **Levenberg-Marquardt** on the inliers, minimising the symmetric transfer
   error in pixels instead of the algebraic error that DLT minimises.
"""

from __future__ import annotations

import numpy as np

from mvg.homogeneous import apply_transform, from_homogeneous, normalizing_transform, to_homogeneous
from mvg.optimize import levenberg_marquardt
from mvg.ransac import RansacResult, ransac


def apply_homography(H: np.ndarray, x: np.ndarray) -> np.ndarray:
    return from_homogeneous(to_homogeneous(x) @ H.T)


def _dlt_rows(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    x, y = src[:, 0], src[:, 1]
    u, v = dst[:, 0], dst[:, 1]
    zeros, ones = np.zeros_like(x), np.ones_like(x)
    row1 = np.stack([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u], axis=1)
    row2 = np.stack([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v], axis=1)
    return np.concatenate([row1, row2])


def _scale(H: np.ndarray) -> np.ndarray:
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H / np.linalg.norm(H)


def has_collinear_triple(x: np.ndarray, rel_tol: float = 1e-6) -> bool:
    """True if any three of the points lie (almost) on one line.

    The area of the triangle (a, b, c) is half the determinant of their
    homogeneous coordinates; it is zero exactly when they are collinear.
    """
    xn = apply_transform(normalizing_transform(x), x)  # scale-free tolerance
    xh = to_homogeneous(xn)
    n = len(xh)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if abs(np.linalg.det(xh[[i, j, k]])) < rel_tol:
                    return True
    return False


def dlt_homography(src: np.ndarray, dst: np.ndarray, normalize: bool = True) -> np.ndarray | None:
    """Least-squares homography from >= 4 matches (src[i] -> dst[i]).

    Returns None for a degenerate minimal sample: with three collinear points
    among four, the equations do not pin down a non-singular H.
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 4:
        raise ValueError("a homography needs at least 4 matches")
    if len(src) == 4 and (has_collinear_triple(src) or has_collinear_triple(dst)):
        return None
    if normalize:
        T1, T2 = normalizing_transform(src), normalizing_transform(dst)
        src_n, dst_n = apply_transform(T1, src), apply_transform(T2, dst)
    else:
        T1 = T2 = np.eye(3)
        src_n, dst_n = src, dst
    A = _dlt_rows(src_n, dst_n)
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    return _scale(np.linalg.inv(T2) @ Hn @ T1)


def transfer_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Pixel distance between H(src) and dst."""
    return np.linalg.norm(apply_homography(H, src) - dst, axis=1)


def symmetric_transfer_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Root of the sum of squared forward and backward transfer errors, per match."""
    fwd = transfer_error(H, src, dst)
    bwd = transfer_error(np.linalg.inv(H), dst, src)
    return np.sqrt(fwd**2 + bwd**2)


def refine_homography(H: np.ndarray, src: np.ndarray, dst: np.ndarray, max_iterations: int = 50):
    """Levenberg-Marquardt on the symmetric transfer error (H[2,2] fixed to 1)."""
    H = _scale(H)

    def residuals(h8: np.ndarray) -> np.ndarray:
        Hc = np.append(h8, 1.0).reshape(3, 3)
        fwd = apply_homography(Hc, src) - dst
        bwd = apply_homography(np.linalg.inv(Hc), dst) - src
        return np.concatenate([fwd.ravel(), bwd.ravel()])

    result = levenberg_marquardt(residuals, H.ravel()[:8], max_iterations=max_iterations)
    return np.append(result.x, 1.0).reshape(3, 3), result


def estimate_homography(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float = 3.0,
    confidence: float = 0.999,
    refine: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray, RansacResult]:
    """Robust homography: RANSAC + normalised DLT, then LM on the inliers.

    Returns (H, inlier mask, RANSAC result). ``threshold`` is in pixels of the
    destination image.
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    result = ransac(
        len(src),
        fit=lambda idx: dlt_homography(src[idx], dst[idx]),
        errors=lambda H, idx: transfer_error(H, src[idx], dst[idx]),
        sample_size=4,
        threshold=threshold,
        confidence=confidence,
        rng=rng,
    )
    H, mask = result.model, result.inliers
    if H is not None and refine and mask.sum() >= 4:
        H, _ = refine_homography(H, src[mask], dst[mask])
        mask = transfer_error(H, src, dst) < threshold
    return H, mask, result
