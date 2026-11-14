"""Corner detectors: Harris, Shi-Tomasi and FAST.

**Harris** (Harris and Stephens, 1988). Shifting a window by (u, v) changes it by
E(u, v) ~ [u v] M [u v]^T, where the structure tensor

    M = G_sigma * [[Ix^2, Ix Iy], [Ix Iy, Iy^2]]

sums gradient products over the window. Both eigenvalues of M are large only
at a corner (one is large on an edge, both are small on flat regions). Harris
avoids the eigen-decomposition with R = det(M) - k trace(M)^2.

**Shi-Tomasi** (1994) scores the corner by the smaller eigenvalue instead.

**FAST** (Rosten and Drummond, 2006) is a segment test: a pixel is a corner if
at least 9 contiguous pixels of the 16 on a radius-3 circle are all brighter
than it by more than t, or all darker.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from mvg.features.filters import gradients, local_maxima


def structure_tensor(img: np.ndarray, sigma_d: float = 1.0, sigma_i: float = 2.0):
    """Entries (Sxx, Sxy, Syy) of the structure tensor at every pixel."""
    ix, iy = gradients(img, sigma_d)
    sxx = ndimage.gaussian_filter(ix * ix, sigma_i)
    sxy = ndimage.gaussian_filter(ix * iy, sigma_i)
    syy = ndimage.gaussian_filter(iy * iy, sigma_i)
    return sxx, sxy, syy


def harris_response(img, sigma_d=1.0, sigma_i=2.0, k=0.04) -> np.ndarray:
    sxx, sxy, syy = structure_tensor(img, sigma_d, sigma_i)
    return sxx * syy - sxy**2 - k * (sxx + syy) ** 2


def shi_tomasi_response(img, sigma_d=1.0, sigma_i=2.0) -> np.ndarray:
    sxx, sxy, syy = structure_tensor(img, sigma_d, sigma_i)
    # smaller eigenvalue of [[a, b], [b, c]]
    return 0.5 * (sxx + syy - np.sqrt((sxx - syy) ** 2 + 4 * sxy**2))


def detect_corners(response: np.ndarray, max_corners: int = 1000, min_distance: int = 5,
                   rel_threshold: float = 0.01, border: int = 8) -> np.ndarray:
    """Strongest local maxima of a corner response, as (N, 2) (x, y) points."""
    mask = local_maxima(response, min_distance) & (response > rel_threshold * response.max())
    mask[:border], mask[-border:], mask[:, :border], mask[:, -border:] = False, False, False, False
    ys, xs = np.nonzero(mask)
    order = np.argsort(-response[ys, xs])[:max_corners]
    return np.column_stack([xs[order], ys[order]]).astype(float)


# radius-3 Bresenham circle used by FAST, in order around the circle
CIRCLE = np.array([(0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
                   (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)])


def fast_score(img: np.ndarray, threshold: float = 0.05, arc: int = 9) -> np.ndarray:
    """FAST-9 segment test at every pixel, vectorised over the whole image.

    Returns a score map (0 where the test fails). The score is the sum of
    absolute differences over the circle beyond the threshold, a common choice
    for non-maximum suppression.
    """
    h, w = img.shape
    pad = np.pad(img, 3, mode="edge")
    ring = np.stack([pad[3 + dy : 3 + dy + h, 3 + dx : 3 + dx + w] for dx, dy in CIRCLE])  # (16, h, w)
    brighter = ring > img + threshold
    darker = ring < img - threshold

    def has_arc(flags: np.ndarray) -> np.ndarray:
        # run lengths around a circle: duplicate the ring and look for `arc` Trues in a row
        doubled = np.concatenate([flags, flags[: arc - 1]]).astype(np.int8)
        run = np.zeros(flags.shape[1:], np.int8)
        best = np.zeros(flags.shape[1:], np.int8)
        for f in doubled:
            run = (run + 1) * f
            np.maximum(best, run, out=best)
        return best >= arc

    corner = has_arc(brighter) | has_arc(darker)
    excess = np.maximum(np.abs(ring - img) - threshold, 0).sum(axis=0)
    return np.where(corner, excess, 0.0)
