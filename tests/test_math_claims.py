"""Checks for the statements made in docs/math/*.md, so the notes cannot drift from the code."""

import numpy as np
import pytest

from mvg.camera import intrinsics, project, projection_matrix, rodrigues
from mvg.homography import _dlt_rows, apply_homography, dlt_homography
from mvg.optimize import numerical_jacobian
from mvg.ransac import required_iterations

K = intrinsics(700, cx=320, cy=240)


def test_plane_homography_is_k_r1_r2_t():
    rng = np.random.default_rng(0)
    R = rodrigues(np.array([0.3, -0.2, 0.1]))
    t = np.array([0.2, -0.1, 5.0])
    XY = rng.uniform(-1, 1, size=(20, 2))
    X = np.hstack([XY, np.zeros((20, 1))])  # points on the plane Z = 0
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    assert np.allclose(apply_homography(H, XY), project(projection_matrix(K, R, t), X))


def test_rotating_camera_gives_homography_k_r_k_inv_at_any_depth():
    rng = np.random.default_rng(1)
    R = rodrigues(np.array([0.05, 0.2, -0.03]))
    X = rng.uniform([-3, -3, 2], [3, 3, 50], size=(30, 3))  # depths from 2 to 50
    x1 = project(projection_matrix(K, np.eye(3), np.zeros(3)), X)
    x2 = project(projection_matrix(K, R, np.zeros(3)), X)
    assert np.allclose(apply_homography(K @ R @ np.linalg.inv(K), x1), x2)


def test_third_cross_product_row_is_dependent():
    x = np.array([2.0, -1.0, 1.0])
    u, v = 3.0, 4.0
    z = np.zeros(3)
    row1 = np.concatenate([z, -x, v * x])
    row2 = np.concatenate([x, z, -u * x])
    row3 = np.concatenate([-v * x, u * x, z])
    assert np.allclose(u * row1 + v * row2 + row3, 0)


def test_dlt_solution_is_smallest_eigenvector_of_ata():
    rng = np.random.default_rng(2)
    src = rng.uniform(-1, 1, size=(12, 2))
    dst = src @ np.array([[1.1, 0.2], [-0.1, 0.9]]) + rng.normal(scale=0.01, size=(12, 2))
    A = _dlt_rows(src, dst)
    w, V = np.linalg.eigh(A.T @ A)
    h_eig = V[:, 0] / V[-1, 0]
    h_dlt = dlt_homography(src, dst, normalize=False).ravel()
    assert np.allclose(h_eig, h_dlt, atol=1e-8)


def test_analytic_jacobian_matches_numerical():
    rng = np.random.default_rng(3)
    H = np.eye(3) + rng.normal(scale=0.05, size=(3, 3))
    x = np.array([120.0, 80.0, 1.0])

    def projected(h9):
        a, b, c = h9.reshape(3, 3) @ x
        return np.array([a / c, b / c])

    a, b, c = H @ x
    J = np.zeros((2, 9))
    J[0, 0:3] = x / c
    J[0, 6:9] = -a / c**2 * x
    J[1, 3:6] = x / c
    J[1, 6:9] = -b / c**2 * x
    assert np.allclose(J, numerical_jacobian(projected, H.ravel()), rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "w,s,p,expected",
    [(0.9, 4, 0.99, 5), (0.7, 4, 0.99, 17), (0.5, 4, 0.99, 72), (0.3, 4, 0.99, 567),
     (0.5, 4, 0.999, 108), (0.5, 8, 0.99, 1177), (0.3, 8, 0.99, 70188)],
)
def test_ransac_iteration_table(w, s, p, expected):
    assert int(np.ceil(required_iterations(w, s, p))) == expected


def test_threshold_2_45_sigma_keeps_95_percent_of_inliers():
    rng = np.random.default_rng(4)
    sigma = 1.7
    err = np.linalg.norm(rng.normal(scale=sigma, size=(200_000, 2)), axis=1)
    kept = (err < np.sqrt(5.991) * sigma).mean()
    assert abs(kept - 0.95) < 0.003


# ------------------------------------------------------------ features.md
from scipy import ndimage  # noqa: E402

from mvg.features import sift as _sift  # noqa: E402
from mvg.features.surf import hessian_determinant, integral_image  # noqa: E402


def _blob(s, size=241):
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    return np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * s**2)), c


@pytest.mark.parametrize("s", [3.0, 5.0])
def test_normalised_laplacian_peaks_at_blob_sigma(s):
    img, c = _blob(s)
    sigmas = np.linspace(0.5 * s, 2 * s, 61)
    resp = [abs(sg**2 * ndimage.gaussian_laplace(img, sg)[c, c]) for sg in sigmas]
    assert abs(sigmas[int(np.argmax(resp))] / s - 1) < 0.03


def test_dog_closed_form_peaks_at_s_over_sqrt_k():
    s, k = 4.0, 2 ** (1 / 3)
    sig = np.linspace(1, 8, 70001)
    g = sig**2 / ((s**2 + sig**2) * (s**2 + k**2 * sig**2))
    assert abs(sig[np.argmax(g)] - s / np.sqrt(k)) < 1e-3


def test_edge_ratio_identity():
    for r in (1.0, 3.0, 10.0):
        H = np.diag([r * 2.5, 2.5])
        assert np.isclose(np.trace(H) ** 2 / np.linalg.det(H), (r + 1) ** 2 / r)


def test_sift_descriptor_ignores_affine_intensity_change():
    rng = np.random.default_rng(7)
    img = sum(ndimage.gaussian_filter(rng.normal(size=(128, 128)), s) * s for s in (2, 4))
    img = (img - img.min()) / (img.max() - img.min())
    kp, space = _sift.detect(img, 50)
    d1 = _sift.describe(kp, _sift.build_scale_space(img))
    d2 = _sift.describe(kp, _sift.build_scale_space(0.6 * img + 0.25))
    assert np.allclose(d1, d2, atol=1e-5)


@pytest.mark.parametrize("s", [4.0, 6.0])
def test_box_hessian_peaks_below_blob_sigma(s):
    img, c = _blob(s)
    ii = integral_image(img)
    sizes = np.arange(9, 99, 6)
    resp = [hessian_determinant(ii, int(L))[c, c] for L in sizes]
    ratio = 1.2 * sizes[int(np.argmax(resp))] / 9 / s
    assert 0.6 <= ratio <= 0.8


# ------------------------------------------------------------ lines_and_stitching.md
def test_points_far_along_a_3d_line_approach_its_vanishing_point():
    R = rodrigues(np.array([0.2, -0.3, 0.1]))
    t = np.array([0.1, 0.2, 5.0])
    P = projection_matrix(K, R, t)
    d = np.array([1.0, 0.5, 2.0])
    v = K @ R @ d
    v = v[:2] / v[2]
    for A in (np.array([0.0, 0, 0]), np.array([1.0, -2, 3])):  # two different parallel lines
        far = project(P, (A + 1e7 * d)[None])[0]
        assert np.allclose(far, v, atol=1e-3)


def test_orthogonal_directions_satisfy_the_absolute_conic_constraint():
    R = rodrigues(np.array([0.4, 0.3, -0.2]))
    v1, v2 = K @ R[:, 0], K @ R[:, 1]
    omega = np.linalg.inv(K).T @ np.linalg.inv(K)
    assert abs(v1 @ omega @ v2) < 1e-12


# ------------------------------------------------------------ pnp_and_bundle_adjustment.md
from mvg.pnp import pnp_dlt, refine_pnp  # noqa: E402
from mvg.two_view import rotation_angle_deg  # noqa: E402


def _pnp_errors(f, spread, depth, trials=50):
    Kf = intrinsics(f, cx=320, cy=240)
    dlt, lm = [], []
    for s in range(trials):
        rng = np.random.default_rng(s)
        R = rodrigues(rng.normal(scale=0.2, size=3))
        t = np.array([0.0, 0.0, depth])
        X = rng.uniform(-spread / 2, spread / 2, (6, 3))
        X -= X.mean(0)
        x = project(projection_matrix(Kf, R, t), X) + rng.normal(scale=0.5, size=(6, 2))
        out = pnp_dlt(X, x, Kf)
        dlt.append(rotation_angle_deg(out[0], R) if out else 180.0)
        R1, _, _ = refine_pnp(rodrigues(np.array([0.05, 0.05, 0.0])) @ R, t + 0.05, X, x, Kf)
        lm.append(rotation_angle_deg(R1, R))
    return np.median(dlt), np.median(lm)


def test_pnp_dlt_breaks_down_for_narrow_field_of_view():
    wide_dlt, wide_lm = _pnp_errors(500, 2.0, 6.0)
    narrow_dlt, narrow_lm = _pnp_errors(1520, 0.1, 2.2)
    assert narrow_dlt > 5 * wide_dlt  # about 62 vs 6 degrees
    assert narrow_lm < 2.0 and wide_lm < 1.0  # starting near the answer fixes it


# ------------------------------------------------------------ stereo.md
from mvg.stereo import depth_from_disparity  # noqa: E402


def test_depth_error_grows_with_the_square_of_depth():
    f, B = 1000.0, 0.2
    for Z in (2.0, 4.0, 8.0):
        d = f * B / Z
        dZ = depth_from_disparity(np.array(d - 0.01), f, B) - depth_from_disparity(np.array(d), f, B)
        assert np.isclose(dZ, Z**2 / (f * B) * 0.01, rtol=0.01)
