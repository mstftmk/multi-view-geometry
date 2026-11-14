"""Small image helpers shared by the detectors.

SciPy is used only for separable Gaussian filtering and running maxima; the
detectors and descriptors themselves are written out in full.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def gradients(img: np.ndarray, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Derivative-of-Gaussian gradients (Ix, Iy); x is the column direction."""
    ix = ndimage.gaussian_filter(img, sigma, order=(0, 1))
    iy = ndimage.gaussian_filter(img, sigma, order=(1, 0))
    return ix, iy


def local_maxima(response: np.ndarray, radius: int) -> np.ndarray:
    """Pixels equal to the maximum of their (2r+1)^2 neighbourhood."""
    return response == ndimage.maximum_filter(response, size=2 * radius + 1, mode="nearest")


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample img at real-valued (x, y), clamping at the border."""
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1]
            + (1 - fx) * fy * img[y0 + 1, x0] + fx * fy * img[y0 + 1, x0 + 1])
