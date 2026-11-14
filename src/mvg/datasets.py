"""Loaders for the evaluation datasets fetched by scripts/download_data.py.

The data directory is ``$MVG_DATA`` if set, otherwise ``data/`` in the
current working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def data_dir() -> Path:
    return Path(os.environ.get("MVG_DATA", "data"))


def read_image(path: Path) -> np.ndarray:
    """Read PNG / PPM / PGM as float grayscale in [0, 1] (OpenCV only for file decoding)."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float64) / 255.0


@dataclass
class HomographySequence:
    """Oxford affine benchmark: image 1 and images 2..6 with H mapping image 1 to image k."""

    name: str
    images: list[np.ndarray]
    homographies: list[np.ndarray]  # homographies[k - 2] maps image 1 -> image k


def load_oxford(name: str, root: Path | None = None) -> HomographySequence:
    folder = (root or data_dir()) / name
    images = [read_image(next(folder.glob(f"img{i}.p?m"))) for i in range(1, 7)]
    homographies = [np.loadtxt(folder / f"H1to{i}p") for i in range(2, 7)]
    return HomographySequence(name, images, homographies)


@dataclass
class CalibratedViews:
    """Middlebury multi-view data: per-image K, R, t with x ~ K [R | t] X."""

    names: list[str]
    images: list[np.ndarray]
    K: np.ndarray  # (n, 3, 3)
    R: np.ndarray  # (n, 3, 3)
    t: np.ndarray  # (n, 3)
    bbox: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def P(self) -> np.ndarray:
        return np.einsum("nij,njk->nik", self.K, np.concatenate([self.R, self.t[:, :, None]], axis=2))

    @property
    def centers(self) -> np.ndarray:
        return -np.einsum("nji,nj->ni", self.R, self.t)  # C = -R^T t


def load_middlebury(name: str = "templeSparseRing", root: Path | None = None) -> CalibratedViews:
    folder = (root or data_dir()) / name / name
    lines = (folder / f"{name.replace('SparseRing', 'SR')}_par.txt").read_text().split("\n")
    count = int(lines[0])
    names, Ks, Rs, ts = [], [], [], []
    for line in lines[1 : count + 1]:
        parts = line.split()
        values = np.array(parts[1:], dtype=float)
        names.append(parts[0])
        Ks.append(values[0:9].reshape(3, 3))
        Rs.append(values[9:18].reshape(3, 3))
        ts.append(values[18:21])
    images = [read_image(folder / n) for n in names]
    bbox = None
    readme = folder / "README.txt"
    if readme.exists():
        rows = [ln.strip() for ln in readme.read_text().splitlines() if ln.strip().startswith("(")]
        if len(rows) >= 2:
            bbox = tuple(np.array(r.strip("()").split(), dtype=float) for r in rows[:2])
    return CalibratedViews(names, images, np.array(Ks), np.array(Rs), np.array(ts), bbox)


def upright(img: np.ndarray) -> np.ndarray:
    """Turn a Middlebury temple image upright for display.

    The temple cameras are rolled by about 90 degrees, so in the raw images the
    model lies on its side, base on the left. Rotating 90 degrees
    counter-clockwise stands it up. This is for figures only: every
    computation uses the images and calibration as supplied.
    """
    return np.rot90(img, 1)


def upright_points(xy: np.ndarray, width: int) -> np.ndarray:
    """Pixel coordinates (x, y) in a raw image of ``width`` columns, moved to match ``upright``."""
    xy = np.asarray(xy, float)
    return np.column_stack([xy[:, 1], width - 1 - xy[:, 0]])
