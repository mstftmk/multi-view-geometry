"""Sanity checks of the downloaded data (skipped when the data is not present, as in CI)."""

import itertools

import numpy as np
import pytest

from mvg.camera import project
from mvg.datasets import data_dir, load_middlebury, load_oxford
from mvg.homography import transfer_error

needs_data = pytest.mark.skipif(not (data_dir() / "graf").exists(), reason="run scripts/download_data.py first")


@needs_data
@pytest.mark.parametrize("name", ["graf", "boat"])
def test_oxford_homographies_agree_with_sift_matches(name):
    cv2 = pytest.importorskip("cv2")
    seq = load_oxford(name)
    sift = cv2.SIFT_create()
    img1 = (seq.images[0] * 255).astype(np.uint8)
    img2 = (seq.images[1] * 255).astype(np.uint8)
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in pairs if m.distance < 0.7 * n.distance]
    src = np.array([k1[m.queryIdx].pt for m in good])
    dst = np.array([k2[m.trainIdx].pt for m in good])
    err = transfer_error(seq.homographies[0], src, dst)
    # most confident matches are consistent with the given H to within a few pixels
    assert np.median(err) < 2.0


@needs_data
def test_middlebury_cameras_form_a_ring_and_see_the_object():
    views = load_middlebury()
    assert len(views.images) == 16
    for R in views.R:
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-4)
    radius = np.linalg.norm(views.centers - views.centers.mean(axis=0), axis=1)
    assert radius.std() / radius.mean() < 0.05  # cameras at nearly the same distance
    lo, hi = views.bbox
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    for P, img in zip(views.P, views.images):
        x = project(P, corners)
        h, w = img.shape
        assert (x[:, 0] > 0).all() and (x[:, 0] < w).all() and (x[:, 1] > 0).all() and (x[:, 1] < h).all()


def test_upright_moves_points_with_the_image():
    from mvg.datasets import upright, upright_points

    rng = np.random.default_rng(0)
    img = rng.random((6, 9))
    ys, xs = np.mgrid[0:6, 0:9]
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    new = upright_points(xy, img.shape[1]).astype(int)
    rot = upright(img)
    assert rot.shape == (9, 6)
    assert np.array_equal(rot[new[:, 1], new[:, 0]], img[ys.ravel(), xs.ravel()])
