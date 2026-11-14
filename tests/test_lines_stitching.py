import itertools

import numpy as np
import pytest
from scipy import ndimage

from mvg.camera import intrinsics, project, projection_matrix, rodrigues
from mvg.homogeneous import to_homogeneous
from mvg.lines import canny, focal_from_orthogonal_vps, hough_lines, vanishing_points
from mvg.stitching import (
    canvas_for,
    cylindrical_coords,
    mosaic,
    multiband_blend,
    warp_perspective,
)


def square(size=100, lo=30, hi=70):
    img = np.zeros((size, size))
    img[lo:hi, lo:hi] = 1.0
    return img


# ------------------------------------------------------------ edges and lines
def test_canny_edges_are_thin_and_on_the_boundary():
    e = canny(square())
    ys, xs = np.nonzero(e)
    on_boundary = (np.minimum.reduce([abs(xs - 29.5), abs(xs - 69.5), abs(ys - 29.5), abs(ys - 69.5)]) <= 1.5)
    assert on_boundary.mean() > 0.95
    assert e[50, 25:35].sum() <= 2  # one edge crossing the left side, about one pixel wide


def test_hough_recovers_the_sides_of_a_square():
    lines = hough_lines(canny(square(120, 30, 90)), max_lines=4, min_votes=30)
    found = sorted(zip(np.round(np.rad2deg(lines.theta)), np.round(lines.rho)))
    # vertical sides: theta = 0, rho = x; horizontal sides: theta = 90, rho = y
    assert [t for t, _ in found] == [0, 0, 90, 90]
    for _, r in found:
        assert min(abs(r - 29.5), abs(r - 89.5)) <= 1.5


def room_lines(rng, K, R, t, noise=0.3, per_axis=12):
    """Image lines of 3D segments parallel to the x, y and z axes, and their labels."""
    lines, labels = [], []
    P = projection_matrix(K, R, t)
    for axis in range(3):
        for _ in range(per_axis):
            a = rng.uniform(-2, 2, size=3)
            b = a.copy()
            b[axis] += rng.uniform(1.0, 2.0)
            x = project(P, np.array([a, b])) + rng.normal(scale=noise, size=(2, 2))
            xh = to_homogeneous(x)
            lines.append(np.cross(xh[0], xh[1]))
            labels.append(axis)
    return np.array(lines), np.array(labels)


def test_vanishing_points_and_focal_length_from_a_synthetic_room():
    rng = np.random.default_rng(0)
    K = intrinsics(800, cx=320, cy=240)
    R = rodrigues(np.array([0.35, 0.55, 0.1]))
    t = np.array([0.0, 0.0, 8.0])
    lines, labels = room_lines(rng, K, R, t)
    found = vanishing_points(lines, (480, 640), n_points=3, threshold=0.004, rng=np.random.default_rng(1))
    assert len(found) == 3
    true_dirs = [R[:, i] for i in range(3)]  # direction d of axis i in the camera frame
    vps = []
    for v, inliers in found:
        d = np.linalg.solve(K, v)
        d /= np.linalg.norm(d)
        angles = [np.degrees(np.arccos(min(1.0, abs(d @ td)))) for td in true_dirs]
        axis = int(np.argmin(angles))
        assert angles[axis] < 1.0
        assert (labels[inliers] == axis).mean() > 0.9
        vps.append(v)
    focals = [focal_from_orthogonal_vps(a, b, np.array([320.0, 240.0])) for a, b in itertools.combinations(vps, 2)]
    assert np.allclose(focals, 800, rtol=0.03)


# ------------------------------------------------------------ warping and blending
def texture(size=128, seed=0):
    rng = np.random.default_rng(seed)
    img = sum(ndimage.gaussian_filter(rng.normal(size=(size, size)), s) * s for s in (2, 4))
    return (img - img.min()) / (img.max() - img.min())


def test_warp_by_translation_shifts_the_image():
    img = texture()
    H = np.array([[1, 0, 5.0], [0, 1, -3.0], [0, 0, 1]])
    out, valid = warp_perspective(img, H, img.shape)
    assert np.allclose(out[10:100, 10:100], img[13:103, 5:95])
    assert not valid[:, :5].any()


def test_warp_and_inverse_warp_recover_the_image():
    img = texture()
    H = np.array([[0.97, 0.05, 4.0], [-0.04, 1.02, 2.0], [1e-4, 5e-5, 1.0]])
    once, _ = warp_perspective(img, H, img.shape)
    back, valid = warp_perspective(once, np.linalg.inv(H), img.shape)
    inner = np.zeros_like(valid)
    inner[15:-15, 15:-15] = True
    assert np.abs(back - img)[inner & valid].mean() < 0.01  # two bilinear resamplings


def test_canvas_of_identity_is_the_image():
    assert canvas_for([(50, 80)], [np.eye(3)]) == ((50, 80), (0.0, 0.0))


@pytest.mark.parametrize("blend", ["feather", "multiband"])
def test_blending_identical_aligned_images_changes_nothing(blend):
    img = texture()
    H = np.array([[1, 0, 30.0], [0, 1, 0.0], [0, 0, 1]])  # image 2 sits 30 px to the right
    shifted, _ = warp_perspective(img, np.linalg.inv(H), img.shape)  # what a camera 30 px right would see
    out, offset = mosaic([img, shifted], [np.eye(3), H], blend=blend)
    overlap = out[10:-10, 40:100]
    assert np.abs(overlap - img[10:-10, 40:100]).mean() < 0.02


def test_multiband_blend_hides_an_exposure_step():
    a, b = np.full((64, 128), 0.3), np.full((64, 128), 0.7)
    left = np.zeros((64, 128))
    left[:, :64] = 1
    out = multiband_blend([a, b], [left, 1 - left])
    assert np.abs(np.diff(out[32])).max() < 0.1  # no hard 0.4 jump at the seam


def test_panning_becomes_a_shift_on_the_cylinder():
    f, alpha = 600.0, np.deg2rad(12)
    rng = np.random.default_rng(3)
    X = rng.uniform([-3, -1, 6], [3, 1, 12], size=(20, 3))
    Ry = np.array([[np.cos(alpha), 0, np.sin(alpha)], [0, 1, 0], [-np.sin(alpha), 0, np.cos(alpha)]])
    K = intrinsics(f)  # principal point at 0: coordinates are already centred
    x1 = project(projection_matrix(K, np.eye(3), np.zeros(3)), X)
    x2 = project(projection_matrix(K, Ry, np.zeros(3)), X)
    u1, h1 = cylindrical_coords(x1[:, 0], x1[:, 1], f)
    u2, h2 = cylindrical_coords(x2[:, 0], x2[:, 1], f)
    assert np.allclose(u2 - u1, f * alpha)  # same shift for every point, whatever its depth
    assert np.allclose(h1, h2)
