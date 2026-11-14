import numpy as np
import pytest
from scipy import ndimage

from mvg.camera import intrinsics, project, projection_matrix, rodrigues
from mvg.homography import apply_homography
from mvg.stereo import census, cost_volume, depth_from_disparity, disparity, left_right_check, rectify


def texture(h=96, w=160, seed=0):
    rng = np.random.default_rng(seed)
    img = sum(ndimage.gaussian_filter(rng.normal(size=(h, w)), s) * s for s in (1, 2, 4))
    return (img - img.min()) / (img.max() - img.min())


def shift_right_image(left, d):
    """Right view of a fronto-parallel plane at disparity d: right(x) = left(x + d)."""
    h, w = left.shape
    xs = np.arange(w) + d
    return np.stack([np.interp(xs, np.arange(w), row) for row in left])


# ------------------------------------------------------------ rectification
def test_rectification_puts_matches_on_the_same_row():
    rng = np.random.default_rng(0)
    K1 = intrinsics(700, cx=320, cy=240)
    K2 = intrinsics(720, cx=330, cy=235)
    R1 = rodrigues(np.array([0.02, -0.15, 0.03]))
    R2 = rodrigues(np.array([-0.04, 0.12, -0.02]))
    t1, t2 = np.array([0.3, 0.0, 4.0]), np.array([-0.4, 0.1, 4.2])
    X = rng.uniform([-1, -1, -1], [1, 1, 1], size=(50, 3))
    x1 = project(projection_matrix(K1, R1, t1), X)
    x2 = project(projection_matrix(K2, R2, t2), X)
    T1, T2, K, R, C1, C2 = rectify(K1, R1, t1, K2, R2, t2)
    y1, y2 = apply_homography(T1, x1), apply_homography(T2, x2)
    assert np.abs(y1[:, 1] - y2[:, 1]).max() < 1e-6
    # the warped points are what the new cameras would see
    assert np.allclose(y1, project(projection_matrix(K, R, -R @ C1), X), atol=1e-6)
    # disparity x1 - x2 equals f * B / Z in the rectified frame
    Z = (X - C1) @ R[2]
    B = np.linalg.norm(C2 - C1)
    assert np.allclose(y1[:, 0] - y2[:, 0], K[0, 0] * B / Z, rtol=1e-6)


# ------------------------------------------------------------ costs and disparity
def test_census_ignores_monotonic_intensity_changes():
    img = texture()
    assert np.array_equal(census(img), census(0.5 * img**0.7 + 0.2))  # gain, gamma and offset


@pytest.mark.parametrize("cost", ["sad", "census"])
def test_block_matching_recovers_a_constant_shift(cost):
    left = texture()
    right = shift_right_image(left, 6.0)  # a left pixel x appears at x - 6 in the right image
    d = disparity(left, right, max_disp=16, method="bm", cost=cost, lr_check=False, median=0)
    inner = d[10:-10, 20:-10]
    assert np.median(np.abs(inner - 6)) < 0.2


def test_subpixel_disparity():
    left = texture(seed=1)
    right = shift_right_image(left, 3.4)
    d = disparity(left, right, max_disp=12, method="bm", cost="sad", window=9, lr_check=False, median=0)
    assert abs(np.median(d[10:-10, 20:-10]) - 3.4) < 0.15


def test_sgm_beats_block_matching_on_a_noisy_two_plane_scene():
    rng = np.random.default_rng(2)
    left = texture(seed=2)
    true = np.where(np.arange(left.shape[1])[None, :] < 80, 4.0, 9.0) * np.ones_like(left)
    right = np.stack([np.interp(np.arange(left.shape[1]) - 0, np.arange(left.shape[1]) - t, row)
                      for row, t in zip(left, true)])
    right = right + rng.normal(scale=0.03, size=right.shape)
    errors = {}
    for method in ("bm", "sgm"):
        d = disparity(left, right, max_disp=16, method=method, cost="census", window=5, lr_check=False, median=0)
        errors[method] = (np.abs(d - true)[8:-8, 20:-8] > 1).mean()
    assert errors["sgm"] <= errors["bm"]
    assert errors["sgm"] < 0.1


def test_left_right_check_flags_mismatched_pixels():
    d_left = np.full((4, 20), 3.0)
    d_right = np.full((4, 20), 3.0)
    d_right[:, 5:8] = 9.0  # the right image disagrees for left pixels 8..10
    out = left_right_check(d_left, d_right)
    assert np.isnan(out[:, 8:11]).all()
    assert np.isfinite(out[:, 12:]).all()


def test_cost_volume_shape_and_border():
    left = texture()
    C = cost_volume(left, left, 8, "sad")
    assert C.shape == (8,) + left.shape
    assert np.all(C[0] == 0)  # identical images agree at zero shift
    assert np.all(C[5, :, :5] == 1.0)  # no partner pixel: worst cost


def test_depth_from_disparity():
    assert np.isclose(depth_from_disparity(np.array(20.0), f=1000.0, baseline=0.2), 10.0)
    assert np.isclose(depth_from_disparity(np.array(20.0), f=1000.0, baseline=0.2, doffs=5.0), 8.0)


def test_motorcycle_pair_against_ground_truth():
    data = pytest.importorskip("skimage.data")
    color = pytest.importorskip("skimage.color")
    left_rgb, right_rgb, gt = data.stereo_motorcycle()
    left, right = color.rgb2gray(left_rgb), color.rgb2gray(right_rgb)
    d = disparity(left, right, max_disp=64)
    valid = np.isfinite(gt)
    ok = valid & np.isfinite(d)
    assert ok.sum() / valid.sum() > 0.85  # density after the left-right check
    assert (np.abs(d - gt)[ok] > 2).mean() < 0.08  # bad pixels among those kept
