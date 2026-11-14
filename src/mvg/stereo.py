"""Stereo: rectification, disparity and depth.

* **Rectification** (Fusiello, Trucco and Verri, 2000). Rotate both cameras
  about their centres so that they share one orientation, with the x axis
  along the baseline. Then epipolar lines are image rows, and a match of a
  pixel is searched along the same row only. Each image is warped by the
  homography T = K_new R_new (K_old R_old)^-1.
* **Disparity.** For rectified images, a point at depth Z appears shifted by
  d = f B / Z pixels between the views (B is the baseline). For each pixel,
  try every shift d and compare small windows around the pixel (block
  matching), then keep the best (winner-take-all).
* **Costs.** SAD compares raw intensities. The census transform replaces each
  pixel by a bit string, "is this neighbour darker than the centre?", and
  compares bit strings with the Hamming distance; this depends only on the
  order of intensities, so it tolerates exposure and gain differences.
* **Semi-global matching** (Hirschmuller, 2008). Block matching decides each
  pixel alone. SGM adds a smoothness penalty, P1 for disparity changes of one
  and P2 for larger jumps, along several straight paths through the image,
  and sums the paths: a cheap approximation to a 2D smoothness prior.

Derivations in docs/math/stereo.md.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from mvg.features.matching import _POPCOUNT

# ---------------------------------------------------------------- rectification


def rectify(K1, R1, t1, K2, R2, t2):
    """Rectifying homographies and new cameras for P_i = K_i [R_i | t_i].

    Returns (T1, T2, K_new, R_new, C1, C2): warping image i by T_i gives the
    view of camera K_new [R_new | -R_new C_i].
    """
    C1, C2 = -R1.T @ t1, -R2.T @ t2
    x = C2 - C1
    x /= np.linalg.norm(x)  # new x axis: along the baseline
    y = np.cross(R1[2], x)  # new y axis: orthogonal to x and to the old optical axis
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    R_new = np.vstack([x, y, z])
    K_new = (K1 + K2) / 2
    K_new[0, 1] = 0.0
    T1 = K_new @ R_new @ np.linalg.inv(K1 @ R1)
    T2 = K_new @ R_new @ np.linalg.inv(K2 @ R2)
    return T1, T2, K_new, R_new, C1, C2


# ---------------------------------------------------------------- costs


def census(img: np.ndarray, radius: tuple[int, int] = (2, 3)) -> np.ndarray:
    """Census transform: one bit per neighbour in a (2ry+1) x (2rx+1) window, packed into uint64."""
    ry, rx = radius
    h, w = img.shape
    pad = np.pad(img, ((ry, ry), (rx, rx)), mode="edge")
    out = np.zeros((h, w), np.uint64)
    bit = 0
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if dy == 0 and dx == 0:
                continue
            nb = pad[ry + dy : ry + dy + h, rx + dx : rx + dx + w]
            out |= (nb < img).astype(np.uint64) << np.uint64(bit)
            bit += 1
    return out


def _hamming64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(a, b)
    return _POPCOUNT[x.view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1)


def cost_volume(left: np.ndarray, right: np.ndarray, max_disp: int, cost: str = "census") -> np.ndarray:
    """Pixel-wise matching cost C[d, y, x] of left pixel x against right pixel x - d (float32)."""
    h, w = left.shape
    C = np.empty((max_disp, h, w), np.float32)
    if cost == "census":
        cl, cr = census(left), census(right)
        worst = 64.0
    else:
        cl, cr = left, right
        worst = 1.0
    for d in range(max_disp):
        C[d, :, :d] = worst  # no right pixel to compare with
        if cost == "census":
            C[d, :, d:] = _hamming64(cl[:, d:], cr[:, : w - d])
        else:
            C[d, :, d:] = np.abs(cl[:, d:] - cr[:, : w - d])
    return C


def aggregate_box(C: np.ndarray, window: int) -> np.ndarray:
    """Block matching: sum the cost over a window around each pixel (a box filter per disparity)."""
    return ndimage.uniform_filter(C, size=(1, window, window), mode="nearest")


# ---------------------------------------------------------------- semi-global matching


def _aggregate_path(C: np.ndarray, axis: int, reverse: bool, P1: float, P2: float) -> np.ndarray:
    """L_r(p, d) = C(p, d) + min(L_r(p-r, d), L_r(p-r, d+-1) + P1, min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k).

    ``axis`` 2 walks along image rows (r = +-x), ``axis`` 1 along columns (r = +-y).
    Subtracting min_k keeps the values bounded without changing the argmin.
    """
    L = np.empty_like(C)
    n = C.shape[axis]
    prev = None
    for i in (range(n - 1, -1, -1) if reverse else range(n)):
        sl = [slice(None)] * 3
        sl[axis] = i
        sl = tuple(sl)
        c = C[sl]  # (D, pixels across the path)
        if prev is None:
            cur = c.copy()
        else:
            best = prev.min(axis=0, keepdims=True)
            up = np.full_like(prev, np.inf)
            up[1:] = prev[:-1] + P1  # neighbour had disparity d - 1
            down = np.full_like(prev, np.inf)
            down[:-1] = prev[1:] + P1  # neighbour had disparity d + 1
            cur = c + np.minimum(np.minimum(prev, up), np.minimum(down, best + P2)) - best
        L[sl] = cur
        prev = cur
    return L


def sgm(C: np.ndarray, P1: float = 8.0, P2: float = 64.0) -> np.ndarray:
    """Sum of the path costs along 4 directions: left to right, right to left, top down, bottom up."""
    S = np.zeros_like(C)
    for axis in (2, 1):
        for reverse in (False, True):
            S += _aggregate_path(C, axis, reverse, P1, P2)
    return S


# ---------------------------------------------------------------- disparity


def winner_take_all(S: np.ndarray, subpixel: bool = True) -> np.ndarray:
    D = S.shape[0]
    d = np.argmin(S, axis=0)
    if not subpixel:
        return d.astype(np.float32)
    # parabola through the costs at d - 1, d, d + 1
    dm, dp = np.clip(d - 1, 0, D - 1), np.clip(d + 1, 0, D - 1)
    c0 = np.take_along_axis(S, d[None], 0)[0]
    cm = np.take_along_axis(S, dm[None], 0)[0]
    cp = np.take_along_axis(S, dp[None], 0)[0]
    denom = cm - 2 * c0 + cp
    offset = np.where(denom > 0, 0.5 * (cm - cp) / np.where(denom > 0, denom, 1), 0.0)
    offset[(d == 0) | (d == D - 1)] = 0
    return (d + np.clip(offset, -0.5, 0.5)).astype(np.float32)


def right_disparity(S: np.ndarray) -> np.ndarray:
    """Disparity of the right image from the same cost volume: right pixel x matches left pixel x + d."""
    D, h, w = S.shape
    Sr = np.full_like(S, np.inf)
    for d in range(D):
        Sr[d, :, : w - d] = S[d, :, d:]
    return np.argmin(Sr, axis=0).astype(np.float32)


def left_right_check(d_left: np.ndarray, d_right: np.ndarray, tol: float = 1.0) -> np.ndarray:
    """Invalidate (NaN) left disparities that the right image does not confirm (occlusions, mismatches)."""
    h, w = d_left.shape
    xs = np.arange(w)[None, :] - np.round(d_left).astype(int)
    ok = xs >= 0
    back = np.take_along_axis(d_right, np.clip(xs, 0, w - 1), axis=1)
    good = ok & (np.abs(d_left - back) <= tol)
    out = d_left.copy()
    out[~good] = np.nan
    return out


def disparity(left: np.ndarray, right: np.ndarray, max_disp: int = 64, method: str = "sgm",
              cost: str = "census", window: int = 7, P1: float = 8.0, P2: float = 64.0,
              lr_check: bool = True, median: int = 3) -> np.ndarray:
    """Disparity of the left image (NaN where invalid). Images are rectified grayscale in [0, 1]."""
    C = cost_volume(left, right, max_disp, cost)
    if method == "bm":
        S = aggregate_box(C, window)
    elif method == "sgm":
        S = sgm(aggregate_box(C, 3) if cost == "census" else C, P1, P2)
    else:
        raise ValueError(method)
    d = winner_take_all(S)
    if median:
        d = ndimage.median_filter(d, size=median)
    if lr_check:
        d = left_right_check(d, right_disparity(S))
    return d


def depth_from_disparity(d: np.ndarray, f: float, baseline: float, doffs: float = 0.0) -> np.ndarray:
    """Z = f B / (d + doffs); doffs is the difference of the principal points (x) of the two cameras."""
    return f * baseline / (d + doffs)
