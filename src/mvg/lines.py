"""Edges, straight lines and vanishing points.

* **Canny** (1986): gradient magnitude, thinned by non-maximum suppression
  along the gradient direction, then hysteresis: weak edge pixels are kept only
  if they connect to a strong one.
* **Hough transform**: a line is x cos(theta) + y sin(theta) = rho. Each edge
  pixel votes for every (rho, theta) it could lie on (a sinusoid in the
  accumulator); peaks are lines supported by many pixels.
* **Vanishing points**: parallel 3D lines meet in the image at their vanishing
  point v = l1 x l2. RANSAC on pairs of lines finds the dominant ones.
* **Focal length**: vanishing points of two orthogonal 3D directions satisfy
  (v1 - c) . (v2 - c) + f^2 = 0, for square pixels and principal point c.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from mvg.features.filters import gradients


def canny(img: np.ndarray, sigma: float = 1.4, low: float = 0.05, high: float = 0.15) -> np.ndarray:
    """Boolean edge map. ``low`` and ``high`` are fractions of the largest gradient."""
    gx, gy = gradients(img, sigma)
    mag = np.hypot(gx, gy)
    # quantise the gradient direction to 0, 45, 90 or 135 degrees
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    q = (np.round(angle / 45).astype(int) % 4)
    offsets = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}  # (dy, dx) along the gradient
    keep = np.zeros_like(mag, bool)
    pad = np.pad(mag, 1)
    h, w = mag.shape
    for d, (dy, dx) in offsets.items():
        ahead = pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        behind = pad[1 - dy : 1 - dy + h, 1 - dx : 1 - dx + w]
        keep |= (q == d) & (mag >= ahead) & (mag >= behind)
    thin = np.where(keep, mag, 0)
    top = thin.max() if thin.max() > 0 else 1.0
    strong = thin >= high * top
    weak = thin >= low * top
    # hysteresis: keep weak components (8-connected) that contain a strong pixel
    labels, n = ndimage.label(weak, structure=np.ones((3, 3)))
    good = np.zeros(n + 1, bool)
    good[np.unique(labels[strong])] = True
    good[0] = False
    return good[labels]


@dataclass
class HoughLines:
    rho: np.ndarray
    theta: np.ndarray
    votes: np.ndarray

    def homogeneous(self) -> np.ndarray:
        """(N, 3) lines (cos t, sin t, -rho): a point x is on line l when l . (x, y, 1) = 0."""
        return np.column_stack([np.cos(self.theta), np.sin(self.theta), -self.rho])


def hough_lines(edges: np.ndarray, n_theta: int = 180, rho_step: float = 1.0, max_lines: int = 20,
                min_votes: int = 30, nms: tuple[int, int] = (9, 9)) -> HoughLines:
    """Lines from an edge map; theta in [0, pi), rho may be negative."""
    ys, xs = np.nonzero(edges)
    thetas = np.arange(n_theta) * np.pi / n_theta
    diag = np.hypot(*edges.shape)
    n_rho = int(np.ceil(2 * diag / rho_step)) + 1
    rho = xs[:, None] * np.cos(thetas) + ys[:, None] * np.sin(thetas)  # (pixels, thetas)
    r_idx = np.round((rho + diag) / rho_step).astype(int)
    acc = np.zeros((n_rho, n_theta), np.int32)
    np.add.at(acc, (r_idx, np.broadcast_to(np.arange(n_theta), r_idx.shape)), 1)
    # peaks: local maxima (theta wraps around) with enough votes
    wrapped = np.concatenate([acc[::-1, -nms[1]:], acc, acc[::-1, : nms[1]]], axis=1)
    local = ndimage.maximum_filter(wrapped, size=nms)[:, nms[1] : -nms[1]]
    peak = (acc == local) & (acc >= min_votes)
    r, t = np.nonzero(peak)
    order = np.argsort(-acc[r, t])[:max_lines]
    r, t = r[order], t[order]
    return HoughLines(r * rho_step - diag, thetas[t], acc[r, t])


# ---------------------------------------------------------------- vanishing points
def _normalized_lines(lines: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Express lines in coordinates centred at ``center`` and divided by ``scale`` (conditioning)."""
    T = np.array([[1 / scale, 0, -center[0] / scale], [0, 1 / scale, -center[1] / scale], [0, 0, 1]])
    out = lines @ np.linalg.inv(T)  # lines transform as l' = T^-T l
    return out / np.linalg.norm(out[:, :2], axis=1, keepdims=True)


def vanishing_point_error(lines_n: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle-like consistency |l . v| for unit-normal lines and a unit vanishing direction v."""
    v = v / np.linalg.norm(v)
    return np.abs(lines_n @ v)


def vanishing_points(lines: np.ndarray, image_shape, n_points: int = 3, threshold: float = 0.01,
                     min_lines: int = 3, iterations: int = 500, rng=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sequential RANSAC: returns [(v (homogeneous, pixels), inlier mask over all lines), ...]."""
    rng = rng or np.random.default_rng(0)
    h, w = image_shape
    center, scale = np.array([w / 2, h / 2]), max(h, w) / 2
    ln = _normalized_lines(lines, center, scale)
    T = np.array([[1 / scale, 0, -center[0] / scale], [0, 1 / scale, -center[1] / scale], [0, 0, 1]])
    remaining = np.ones(len(lines), bool)
    found = []
    for _ in range(n_points):
        idx = np.flatnonzero(remaining)
        if len(idx) < min_lines:
            break
        best_v, best_in = None, np.zeros(len(lines), bool)
        for _ in range(iterations):
            i, j = rng.choice(idx, 2, replace=False)
            v = np.cross(ln[i], ln[j])
            if np.linalg.norm(v) < 1e-12:
                continue
            inl = remaining & (vanishing_point_error(ln, v) < threshold)
            if inl.sum() > best_in.sum():
                best_v, best_in = v, inl
        if best_v is None or best_in.sum() < min_lines:
            break
        # least-squares refinement: v minimising sum (l_i . v)^2 is the smallest singular vector
        _, _, Vt = np.linalg.svd(ln[best_in])
        v = Vt[-1]
        inl = remaining & (vanishing_point_error(ln, v) < threshold)
        found.append((np.linalg.inv(T) @ v, inl))
        remaining &= ~inl
    return found


def focal_from_orthogonal_vps(v1: np.ndarray, v2: np.ndarray, principal_point: np.ndarray) -> float:
    """f from two finite vanishing points of orthogonal directions (square pixels, zero skew)."""
    a = v1[:2] / v1[2] - principal_point
    b = v2[:2] / v2[2] - principal_point
    f2 = -(a @ b)
    if f2 <= 0:
        raise ValueError("these vanishing points cannot come from orthogonal directions")
    return float(np.sqrt(f2))
