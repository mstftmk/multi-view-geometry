"""Measuring detectors and descriptors against a ground-truth homography.

* **Repeatability** (simplified from Mikolajczyk et al., 2005): of the
  keypoints that fall in the part of the scene both images show, the fraction
  in image 1 that land within ``eps`` pixels of a keypoint in image 2 after
  mapping by the true H. The original also requires the detected regions to
  overlap; this version checks positions only.
* **Matching precision:** the fraction of ratio-test matches whose transfer
  error under the true H is below ``eps`` pixels.
* **Correct matches:** the number of such matches, which is what a later
  RANSAC step can actually use.
* **Homography error:** estimate H from the matches with RANSAC, map the four
  image corners with the estimate and with the truth, and average the
  distances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvg.homography import apply_homography, estimate_homography, transfer_error


def _inside(xy: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    return (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)


def repeatability(xy1, xy2, H, shape1, shape2, eps: float = 3.0) -> float:
    a = xy1[_inside(apply_homography(H, xy1), shape2)]
    b = xy2[_inside(apply_homography(np.linalg.inv(H), xy2), shape1)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    pa = apply_homography(H, a)
    d = np.sqrt(((pa[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    # count from both sides and take the smaller, so several points of one
    # image near a single point of the other cannot push the score above 1
    found = min((d.min(axis=1) < eps).sum(), (d.min(axis=0) < eps).sum())
    return float(found / min(len(a), len(b)))


def corner_error(H_est, H_true, shape) -> float:
    h, w = shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    if H_est is None:
        return float("inf")
    return float(np.linalg.norm(apply_homography(H_est, corners) - apply_homography(H_true, corners), axis=1).mean())


@dataclass
class PairResult:
    matches: int
    correct: int
    precision: float
    repeatability: float
    homography_error: float


def evaluate_pair(xy1, xy2, matches, H, shape1, shape2, eps: float = 3.0, seed: int = 0) -> PairResult:
    rep = repeatability(xy1, xy2, H, shape1, shape2, eps)
    if len(matches) == 0:
        return PairResult(0, 0, 0.0, rep, float("inf"))
    src, dst = xy1[matches[:, 0]], xy2[matches[:, 1]]
    ok = transfer_error(H, src, dst) < eps
    H_est = None
    if len(matches) >= 4:
        H_est, _, _ = estimate_homography(src, dst, threshold=eps, rng=np.random.default_rng(seed))
    return PairResult(len(matches), int(ok.sum()), float(ok.mean()), rep, corner_error(H_est, H, shape2))
