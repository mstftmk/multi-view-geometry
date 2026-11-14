"""Descriptor matching: nearest neighbours, Lowe's ratio test and the mutual check.

**Ratio test** (Lowe, 2004): accept the nearest neighbour only if it is clearly
closer than the second nearest, d1 < ratio * d2. A correct match usually has
one clear winner; a wrong one (repetitive texture, or a point with no true
partner) tends to have several similar candidates.

**Mutual check:** keep (i, j) only if j is i's nearest neighbour and i is j's.
"""

from __future__ import annotations

import numpy as np

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def l2_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(d2, 0))


def hamming_distances(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Hamming distance between packed bit strings (uint8 rows)."""
    out = np.empty((len(a), len(b)), np.uint16)
    for i in range(0, len(a), chunk):
        x = np.bitwise_xor(a[i : i + chunk, None, :], b[None, :, :])
        out[i : i + chunk] = _POPCOUNT[x].sum(axis=2)
    return out


def match(a: np.ndarray, b: np.ndarray, ratio: float | None = 0.8, mutual: bool = False,
          metric: str = "l2") -> np.ndarray:
    """(M, 2) index pairs (i into a, j into b)."""
    if len(a) == 0 or len(b) < 2:
        return np.zeros((0, 2), int)
    D = (hamming_distances(a, b) if metric == "hamming" else l2_distances(a, b)).astype(np.float64)
    nn = np.argpartition(D, 1, axis=1)[:, :2]
    d_nn = np.take_along_axis(D, nn, axis=1)
    order = np.argsort(d_nn, axis=1)
    nn = np.take_along_axis(nn, order, axis=1)
    d_nn = np.take_along_axis(d_nn, order, axis=1)
    keep = np.ones(len(a), bool)
    if ratio is not None:
        keep &= d_nn[:, 0] < ratio * d_nn[:, 1]
    if mutual:
        keep &= D.argmin(axis=0)[nn[:, 0]] == np.arange(len(a))
    i = np.flatnonzero(keep)
    return np.column_stack([i, nn[i, 0]])
