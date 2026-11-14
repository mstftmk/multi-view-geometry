"""RANSAC: fitting a model when many of the data points are wrong.

Feature matching always produces some wrong matches (outliers). Least squares
lets a single bad match drag the whole fit, so RANSAC (Fischler and Bolles,
1981) instead repeats: pick a minimal random sample (4 matches for a
homography), fit a model to it, and count how many other points agree within a
threshold. The model with the most support wins, and is then refitted on all
of its inliers.

Two improvements over the original are used here:

* **Adaptive number of iterations.** If a fraction w of the data are inliers,
  a sample of size s is all-inlier with probability w^s. To find one such
  sample with confidence p we need N = log(1 - p) / log(1 - w^s) samples, and
  w is re-estimated from the best model so far.
* **MSAC scoring** (Torr and Zisserman, 2000). Plain RANSAC scores a model by
  its inlier count, so all models with the same count tie. MSAC adds up the
  errors of inliers and a fixed penalty for outliers, which prefers the model
  that fits its inliers more tightly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class RansacResult:
    model: object
    inliers: np.ndarray  # boolean mask
    iterations: int


def required_iterations(inlier_ratio: float, sample_size: int, confidence: float) -> float:
    if inlier_ratio <= 0:
        return float("inf")
    p_good = inlier_ratio**sample_size
    if p_good >= 1:
        return 1
    return np.log(1 - confidence) / np.log(1 - p_good)


def ransac(
    n: int,
    fit: Callable[[np.ndarray], object | None],
    errors: Callable[[object, np.ndarray], np.ndarray],
    sample_size: int,
    threshold: float,
    confidence: float = 0.999,
    max_iterations: int = 10_000,
    scoring: str = "msac",
    rng: np.random.Generator | None = None,
) -> RansacResult:
    """Generic RANSAC over ``n`` data items addressed by index.

    ``fit(indices)`` returns a model or None (degenerate sample).
    ``errors(model, indices)`` returns one non-negative error per item, in the
    same units as ``threshold``.
    """
    rng = rng or np.random.default_rng()
    all_idx = np.arange(n)
    best_score, best_model, best_mask = np.inf, None, np.zeros(n, bool)
    needed, it = max_iterations, 0
    t2 = threshold**2
    while it < min(needed, max_iterations):
        it += 1
        sample = rng.choice(n, sample_size, replace=False)
        model = fit(sample)
        if model is None:
            continue
        e2 = errors(model, all_idx) ** 2
        mask = e2 < t2
        score = np.minimum(e2, t2).sum() if scoring == "msac" else -mask.sum()
        if score < best_score:
            best_score, best_model, best_mask = score, model, mask
            needed = required_iterations(mask.mean(), sample_size, confidence)
    if best_model is None:
        return RansacResult(None, best_mask, it)
    # refit on all inliers, then recompute the inlier set once
    if best_mask.sum() < sample_size:
        return RansacResult(best_model, best_mask, it)
    refit = fit(np.flatnonzero(best_mask))
    if refit is not None:
        mask = errors(refit, all_idx) < threshold
        if mask.sum() >= best_mask.sum():
            best_model, best_mask = refit, mask
    return RansacResult(best_model, best_mask, it)
