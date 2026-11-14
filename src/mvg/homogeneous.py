"""Homogeneous coordinates and the 2D transformation hierarchy.

A point (x, y) in the image is written as (x, y, 1), or any multiple (kx, ky, k).
This lets one 3x3 matrix express translation, rotation, scaling and perspective,
and lets points at infinity (k = 0), such as vanishing points, be handled like
any other point.

The hierarchy, from most to least constrained (Hartley and Zisserman, ch. 2):

=============  ===========  ====================================
Transform      Degrees of   Preserves
               freedom
=============  ===========  ====================================
Euclidean      3            lengths, angles, areas
Similarity     4            angles, ratios of lengths
Affine         6            parallel lines, ratios of areas
Projective     8            straight lines (collinearity) only
=============  ===========  ====================================
"""

from __future__ import annotations

import numpy as np


def to_homogeneous(x: np.ndarray) -> np.ndarray:
    """(N, d) points -> (N, d + 1) with a trailing column of ones."""
    x = np.asarray(x, dtype=float)
    return np.hstack([x, np.ones((x.shape[0], 1))])


def from_homogeneous(xh: np.ndarray) -> np.ndarray:
    """(N, d + 1) -> (N, d) by dividing by the last coordinate."""
    xh = np.asarray(xh, dtype=float)
    return xh[:, :-1] / xh[:, -1:]


def normalizing_transform(x: np.ndarray) -> np.ndarray:
    """Hartley normalisation: move the centroid to the origin and scale so the
    mean distance from it is sqrt(d) (sqrt(2) for image points).

    Without this, DLT matrices mix entries of order 1 with entries of order
    (pixel coordinate)^2, and the least-squares solution becomes badly
    conditioned (Hartley, "In defense of the eight-point algorithm", 1997).
    """
    x = np.asarray(x, dtype=float)
    d = x.shape[1]
    centroid = x.mean(axis=0)
    mean_dist = np.linalg.norm(x - centroid, axis=1).mean()
    s = np.sqrt(d) / mean_dist if mean_dist > 0 else 1.0
    T = np.eye(d + 1)
    T[:d, :d] *= s
    T[:d, d] = -s * centroid
    return T


def apply_transform(T: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply a (d+1)x(d+1) homogeneous transform to (N, d) points."""
    return from_homogeneous(to_homogeneous(x) @ T.T)


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix: skew(a) @ b == np.cross(a, b)."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def line_through(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The homogeneous line through two homogeneous points is their cross product."""
    return np.cross(p, q)


def intersect(l: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Two homogeneous lines meet at their cross product (at infinity if parallel)."""
    return np.cross(l, m)


# ---------------------------------------------------------------- hierarchy
def euclidean(theta: float, tx: float, ty: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1]])


def similarity(scale: float, theta: float, tx: float, ty: float) -> np.ndarray:
    T = euclidean(theta, tx, ty)
    T[:2, :2] *= scale
    return T


def affine(A: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(3)
    T[:2, :2] = A
    T[:2, 2] = t
    return T
