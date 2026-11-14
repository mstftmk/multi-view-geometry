"""An ORB-style binary feature (Rublee et al., "ORB: an efficient alternative to SIFT or SURF", 2011).

* **Detector:** FAST corners on an image pyramid (scale factor 1.2), ranked
  by the Harris response, because FAST alone has no notion of corner strength.
* **Orientation:** the intensity centroid. With patch moments
  m_pq = sum x^p y^q I(x, y) over a disc of radius 15, the vector from the
  patch centre to the centroid (m10 / m00, m01 / m00) has angle atan2(m01, m10).
* **Descriptor:** steered BRIEF (Calonder et al., 2010). 256 pairs of points
  (p, q) are drawn once from a Gaussian around the centre; bit i is 1 if
  I(p_i) < I(q_i) in the smoothed patch. The pairs are rotated by the
  keypoint orientation first. Descriptors are compared with the Hamming distance.

The original ORB learns its 256 pairs to be uncorrelated; here they are
random (the "BRIEF G II" pattern), which is simpler and a little weaker.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from mvg.features.corners import fast_score, harris_response
from mvg.features.filters import bilinear, local_maxima
from mvg.features.sift import Keypoints

PATCH = 31
RADIUS = 15


def brief_pairs(n_bits: int = 256, seed: int = 0) -> np.ndarray:
    """(n_bits, 4) test pairs (x1, y1, x2, y2), isotropic Gaussian with variance S^2 / 25."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(scale=PATCH / 5, size=(n_bits, 4))
    return np.clip(pts, -RADIUS + 1, RADIUS - 1)


PAIRS = brief_pairs()


def _pyramid(img: np.ndarray, levels: int, factor: float):
    out = [img]
    for _ in range(1, levels):
        prev = ndimage.gaussian_filter(out[-1], 0.6)
        out.append(ndimage.zoom(prev, 1 / factor, order=1))
    return out


def intensity_centroid_angle(img: np.ndarray, x: float, y: float) -> float:
    ys, xs = np.mgrid[-RADIUS : RADIUS + 1, -RADIUS : RADIUS + 1]
    disc = xs**2 + ys**2 <= RADIUS**2
    vals = bilinear(img, x + xs[disc], y + ys[disc])
    return float(np.arctan2((ys[disc] * vals).sum(), (xs[disc] * vals).sum()))


def detect_and_describe(img: np.ndarray, max_keypoints: int = 2000, levels: int = 8,
                        factor: float = 1.2, fast_threshold: float = 0.08):
    pyr = _pyramid(img, levels, factor)
    # keypoints per level in proportion to the level's area, as in ORB
    weights = factor ** (-2.0 * np.arange(levels))
    quota = np.round(max_keypoints * weights / weights.sum()).astype(int)
    rows, descs = [], []
    for lvl, (im, q) in enumerate(zip(pyr, quota)):
        if q == 0 or min(im.shape) < 2 * RADIUS + 4:
            continue
        fast = fast_score(im, fast_threshold)
        harris = harris_response(im, 1.0, 1.5)
        cand = (fast > 0) & local_maxima(np.where(fast > 0, harris, -np.inf), 3)
        b = RADIUS + 3
        cand[:b], cand[-b:], cand[:, :b], cand[:, -b:] = False, False, False, False
        ys, xs = np.nonzero(cand)
        order = np.argsort(-harris[ys, xs])[:q]
        smooth = ndimage.gaussian_filter(im, 2.0)  # BRIEF compares smoothed intensities
        s = factor**lvl
        for y, x in zip(ys[order], xs[order]):
            theta = intensity_centroid_angle(im, x, y)
            c, sn = np.cos(theta), np.sin(theta)
            x1 = x + c * PAIRS[:, 0] - sn * PAIRS[:, 1]
            y1 = y + sn * PAIRS[:, 0] + c * PAIRS[:, 1]
            x2 = x + c * PAIRS[:, 2] - sn * PAIRS[:, 3]
            y2 = y + sn * PAIRS[:, 2] + c * PAIRS[:, 3]
            bits = bilinear(smooth, x1, y1) < bilinear(smooth, x2, y2)
            descs.append(np.packbits(bits))
            rows.append((x * s, y * s, s * 7.0, harris[y, x], lvl, theta))
    if not rows:
        e = np.zeros(0)
        return Keypoints(np.zeros((0, 2)), e, e, e, e.astype(int), e), np.zeros((0, 32), np.uint8)
    a = np.array(rows)
    kp = Keypoints(a[:, :2], a[:, 2], a[:, 5], a[:, 3], a[:, 4].astype(int), a[:, 4])
    return kp, np.array(descs, dtype=np.uint8)
