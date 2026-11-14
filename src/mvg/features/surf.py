"""SURF: Speeded-Up Robust Features (Bay, Tuytelaars and Van Gool, 2006).

OpenCV's pip packages do not include SURF (it lives in the non-free contrib
module), so it is written here from the paper.

The idea is to make every filter a sum of boxes, because any box sum costs 4
look-ups in an **integral image** II(y, x) = sum of img[:y, :x], whatever the
box size. So the filters are scaled up instead of the image being scaled down.

* **Detector:** the determinant of the Hessian, with the second derivatives of
  a Gaussian approximated by box filters D_xx, D_yy, D_xy of size L (9, 15,
  21, ... pixels; L = 9 matches sigma = 1.2). det(H) ~ D_xx D_yy - (0.9 D_xy)^2,
  where 0.9 corrects for the box approximation. Keypoint scales use the
  paper's rule sigma = 1.2 L / 9; on Gaussian blobs the box response actually
  peaks at about 0.7 times the blob's sigma (measured in docs/math/features.md).
* **Orientation:** Haar wavelet responses (box differences) around the point,
  summed in a sliding 60-degree window; the longest summed vector wins.
* **Descriptor:** a 20s x 20s window in the keypoint frame, split into 4 x 4
  sub-regions; each stores (sum dx, sum dy, sum |dx|, sum |dy|) of the Haar
  responses, giving 64 numbers.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from mvg.features.filters import local_maxima
from mvg.features.sift import Keypoints


def integral_image(img: np.ndarray) -> np.ndarray:
    """II with a leading row and column of zeros: II[y, x] = img[:y, :x].sum()."""
    ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
    ii[1:, 1:] = img.cumsum(0).cumsum(1)
    return ii


def box_sum(ii: np.ndarray, y0, x0, y1, x1):
    """Sum of img[y0:y1, x0:x1] for arrays of corners (clamped to the image)."""
    h, w = ii.shape[0] - 1, ii.shape[1] - 1
    y0, y1 = np.clip(y0, 0, h), np.clip(y1, 0, h)
    x0, x1 = np.clip(x0, 0, w), np.clip(x1, 0, w)
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def hessian_determinant(ii: np.ndarray, L: int, step: int = 1):
    """det(H) approximation for filter size L, sampled every ``step`` pixels."""
    h, w = ii.shape[0] - 1, ii.shape[1] - 1
    l = L // 3  # lobe size
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    b = (L - 1) // 2
    # Dyy: three stacked boxes (white, black x2, white), each l tall and 2l - 1 wide
    half_w = l - 1
    dyy = (box_sum(ii, ys - b, xs - half_w, ys + b + 1, xs + half_w + 1)
           - 3 * box_sum(ii, ys - l // 2, xs - half_w, ys + l // 2 + 1, xs + half_w + 1))
    dxx = (box_sum(ii, ys - half_w, xs - b, ys + half_w + 1, xs + b + 1)
           - 3 * box_sum(ii, ys - half_w, xs - l // 2, ys + half_w + 1, xs + l // 2 + 1))
    # Dxy: four l x l boxes in the quadrants, one pixel away from the centre lines
    dxy = (box_sum(ii, ys - l, xs - l, ys, xs) + box_sum(ii, ys + 1, xs + 1, ys + l + 1, xs + l + 1)
           - box_sum(ii, ys - l, xs + 1, ys, xs + l + 1) - box_sum(ii, ys + 1, xs - l, ys + l + 1, xs))
    area = float(L * L)
    dxx, dyy, dxy = dxx / area, dyy / area, dxy / area
    return dxx * dyy - (0.9 * dxy) ** 2


OCTAVES = [(9, 15, 21, 27), (15, 27, 39, 51), (27, 51, 75, 99), (51, 99, 147, 195)]


def detect(img: np.ndarray, max_keypoints: int = 2000, threshold: float = 1e-4, n_octaves: int = 3):
    ii = integral_image(img)
    rows = []
    for o, sizes in enumerate(OCTAVES[:n_octaves]):
        step = 2**o
        stack = np.stack([hessian_determinant(ii, L, step) for L in sizes])
        border = sizes[-1] // (2 * step) + 1
        for j in (1, 2):  # the middle layers have neighbours above and below
            layer = stack[j]
            # maximum of its 3 x 3 x 3 neighbourhood in (scale, y, x)
            is_max = local_maxima(layer, 1) & (layer > threshold)
            is_max &= layer > ndimage.maximum_filter(stack[j - 1], size=3, mode="nearest")
            is_max &= layer > ndimage.maximum_filter(stack[j + 1], size=3, mode="nearest")
            is_max[:border], is_max[-border:], is_max[:, :border], is_max[:, -border:] = False, False, False, False
            for y, x in zip(*np.nonzero(is_max)):
                # sub-pixel position from a parabola through each axis
                dx = 0.5 * (layer[y, x - 1] - layer[y, x + 1]) / (layer[y, x - 1] - 2 * layer[y, x] + layer[y, x + 1])
                dy = 0.5 * (layer[y - 1, x] - layer[y + 1, x]) / (layer[y - 1, x] - 2 * layer[y, x] + layer[y + 1, x])
                dx, dy = np.clip(dx, -0.5, 0.5), np.clip(dy, -0.5, 0.5)
                rows.append(((x + dx) * step, (y + dy) * step, 1.2 * sizes[j] / 9, layer[y, x], o, j))
    if not rows:
        e = np.zeros(0)
        return Keypoints(np.zeros((0, 2)), e, e, e, e.astype(int), e)
    a = np.array(rows)
    a = a[np.argsort(-a[:, 3])][:max_keypoints]
    return Keypoints(a[:, :2], a[:, 2], np.zeros(len(a)), a[:, 3], a[:, 4].astype(int), a[:, 5])


def haar(ii, x, y, size):
    """Haar wavelet responses (dx, dy) of side ``size`` centred at integer (x, y)."""
    h = size // 2
    dx = box_sum(ii, y - h, x, y + h, x + h) - box_sum(ii, y - h, x - h, y + h, x)
    dy = box_sum(ii, y, x - h, y + h, x + h) - box_sum(ii, y - h, x - h, y, x + h)
    return dx, dy


def orientations(ii: np.ndarray, kp: Keypoints) -> np.ndarray:
    gy, gx = np.mgrid[-6:7, -6:7]
    inside = gx**2 + gy**2 <= 36
    gx, gy = gx[inside], gy[inside]
    weight = np.exp(-(gx**2 + gy**2) / (2 * 2.5**2))
    windows = np.linspace(-np.pi, np.pi, 42, endpoint=False)
    angles = np.zeros(len(kp))
    for i, ((x, y), s) in enumerate(zip(kp.xy, kp.scale)):
        size = max(2, int(round(2 * s)) * 2)  # wavelet side 4s, kept even
        px = np.round(x + gx * s).astype(int)
        py = np.round(y + gy * s).astype(int)
        dx, dy = haar(ii, px, py, size)
        dx, dy = dx * weight, dy * weight
        phi = np.arctan2(dy, dx)
        best, best_len = 0.0, -1.0
        for start in windows:  # sliding pi/3 window
            sel = ((phi - start) % (2 * np.pi)) < np.pi / 3
            vx, vy = dx[sel].sum(), dy[sel].sum()
            length = vx * vx + vy * vy
            if length > best_len:
                best, best_len = np.arctan2(vy, vx), length
        angles[i] = best
    return angles


def describe(ii: np.ndarray, kp: Keypoints, upright: bool = False) -> np.ndarray:
    """(N, 64) float32 descriptors."""
    # 20 x 20 sample grid (in units of s), centred on the keypoint
    u = np.arange(-10, 10) + 0.5
    gu, gv = np.meshgrid(u, u)  # gu: x in keypoint frame, gv: y
    sub = (np.floor((gv + 10) / 5).astype(int) * 4 + np.floor((gu + 10) / 5).astype(int)).ravel()
    gauss = np.exp(-(gu**2 + gv**2) / (2 * 3.3**2)).ravel()
    out = np.zeros((len(kp), 64), np.float32)
    for i, ((x, y), s) in enumerate(zip(kp.xy, kp.scale)):
        theta = 0.0 if upright else kp.angle[i]
        c, sn = np.cos(theta), np.sin(theta)
        px = np.round(x + s * (c * gu - sn * gv)).astype(int).ravel()
        py = np.round(y + s * (sn * gu + c * gv)).astype(int).ravel()
        size = max(2, int(round(s)) * 2)
        dx, dy = haar(ii, px, py, size)
        # rotate the responses into the keypoint frame
        rx = (c * dx + sn * dy) * gauss
        ry = (-sn * dx + c * dy) * gauss
        v = np.zeros((16, 4))
        np.add.at(v[:, 0], sub, rx)
        np.add.at(v[:, 1], sub, ry)
        np.add.at(v[:, 2], sub, np.abs(rx))
        np.add.at(v[:, 3], sub, np.abs(ry))
        v = v.ravel()
        out[i] = v / max(np.linalg.norm(v), 1e-12)
    return out


def detect_and_describe(img: np.ndarray, max_keypoints: int = 2000, upright: bool = False):
    ii = integral_image(img)
    kp = detect(img, max_keypoints)
    if not upright:
        kp.angle = orientations(ii, kp)
    return kp, describe(ii, kp, upright)
