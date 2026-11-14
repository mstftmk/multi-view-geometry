import numpy as np
import pytest

from mvg.calibration import _v, calibrate, intrinsics_from_homographies, pose_from_homography, project_board
from mvg.camera import intrinsics, project, projection_matrix, rodrigues
from mvg.homography import dlt_homography
from mvg.two_view import (
    angle_between_deg,
    decompose_essential,
    eight_point,
    essential_from_fundamental,
    essential_from_pose,
    estimate_fundamental,
    recover_pose,
    refine_pose,
    rotation_angle_deg,
    sampson_error,
    triangulate,
)

K_TRUE = np.array([[820.0, 0, 330], [0, 810, 245], [0, 0, 1]])
DIST_TRUE = np.array([-0.2, 0.05])


def board():
    gx, gy = np.meshgrid(np.arange(9), np.arange(6))
    return np.column_stack([gx.ravel(), gy.ravel()]) * 0.03


def board_views(n=10, noise=0.3, dist=DIST_TRUE, seed=0):
    rng = np.random.default_rng(seed)
    b = board()
    views, poses = [], []
    for _ in range(n):
        rv = rng.normal(scale=0.35, size=3)
        tv = np.array([-0.12, -0.08, 0.5]) + rng.normal(scale=0.03, size=3)
        views.append(project_board(K_TRUE, dist, rv, tv, b) + rng.normal(scale=noise, size=(len(b), 2)))
        poses.append((rv, tv))
    return b, views, poses


# ------------------------------------------------------------ calibration
def test_zhang_closed_form_is_exact_without_noise_or_distortion():
    b, views, _ = board_views(noise=0.0, dist=np.zeros(2))
    Hs = [dlt_homography(b, x) for x in views]
    K = intrinsics_from_homographies(Hs)
    assert np.allclose(K, K_TRUE, atol=1e-4)


def test_board_homography_satisfies_the_two_zhang_constraints():
    b, views, _ = board_views(noise=0.0, dist=np.zeros(2))
    H = dlt_homography(b, views[0])
    Kinv = np.linalg.inv(K_TRUE)
    B = Kinv.T @ Kinv
    bvec = np.array([B[0, 0], B[0, 1], B[1, 1], B[0, 2], B[1, 2], B[2, 2]])
    scale = abs(_v(H, 0, 0) @ bvec)
    assert abs(_v(H, 0, 1) @ bvec) < 1e-8 * scale  # h1^T B h2 = 0
    assert abs((_v(H, 0, 0) - _v(H, 1, 1)) @ bvec) < 1e-8 * scale  # h1^T B h1 = h2^T B h2


def test_pose_from_homography_recovers_board_pose():
    b, views, poses = board_views(noise=0.0, dist=np.zeros(2))
    R, t = pose_from_homography(K_TRUE, dlt_homography(b, views[0]))
    assert rotation_angle_deg(R, rodrigues(poses[0][0])) < 1e-4
    assert np.allclose(t, poses[0][1], atol=1e-6)


def test_full_calibration_with_noise_and_distortion():
    b, views, _ = board_views()
    cal = calibrate(b, views)
    assert np.allclose(cal.K, K_TRUE, rtol=0.01, atol=2.0)
    assert np.allclose(cal.dist, DIST_TRUE, atol=0.03)
    assert cal.rms < 0.3 * np.sqrt(2) * 1.05  # at the noise level (0.3 px per coordinate)
    assert cal.rms < cal.rms_closed_form  # refinement helps


def test_calibration_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    b, views, _ = board_views()
    ours = calibrate(b, views)
    obj = [np.column_stack([b, np.zeros(len(b))]).astype(np.float32)] * len(views)
    rms, K, d, _, _ = cv2.calibrateCamera(obj, [v.astype(np.float32) for v in views], (640, 480), None, None,
                                          flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    assert np.allclose(ours.K, K, atol=0.05)
    assert np.allclose(ours.dist, d.ravel()[:2], atol=1e-4)
    assert abs(ours.rms - rms) < 1e-3


# ------------------------------------------------------------ two views
def scene(seed=0, n=100, noise=0.0):
    rng = np.random.default_rng(seed)
    K = intrinsics(700, cx=320, cy=240)
    R = rodrigues(np.array([0.05, -0.25, 0.03]))
    t = np.array([1.0, 0.1, 0.2])
    X = rng.uniform([-2, -2, 5], [2, 2, 9], size=(n, 3))
    x1 = project(projection_matrix(K, np.eye(3), np.zeros(3)), X) + rng.normal(scale=noise, size=(n, 2))
    x2 = project(projection_matrix(K, R, t), X) + rng.normal(scale=noise, size=(n, 2))
    return K, R, t, X, x1, x2


def test_eight_point_satisfies_epipolar_constraint_and_has_rank_two():
    K, R, t, X, x1, x2 = scene()
    F = eight_point(x1, x2)
    assert np.linalg.matrix_rank(F, tol=1e-8) == 2
    assert sampson_error(F, x1, x2).max() < 1e-6


def test_fundamental_matches_the_closed_form_from_the_cameras():
    K, R, t, X, x1, x2 = scene()
    F_true = np.linalg.inv(K).T @ essential_from_pose(R, t) @ np.linalg.inv(K)
    F = eight_point(x1, x2)
    F_true /= np.linalg.norm(F_true)
    assert min(np.abs(F - F_true).max(), np.abs(F + F_true).max()) < 1e-6


def test_essential_has_two_equal_singular_values_and_one_zero():
    s = np.linalg.svd(essential_from_pose(rodrigues(np.array([0.3, 0.1, -0.2])), np.array([0.2, 1.0, 0.5])),
                      compute_uv=False)
    assert np.isclose(s[0], s[1]) and abs(s[2]) < 1e-12


def test_decomposition_contains_the_true_pose_and_cheirality_picks_it():
    K, R, t, X, x1, x2 = scene()
    E = essential_from_fundamental(eight_point(x1, x2), K, K)
    candidates = decompose_essential(E)
    assert min(rotation_angle_deg(Rc, R) + angle_between_deg(tc, t) for Rc, tc in candidates) < 1e-4
    R_est, t_est, front, Xs = recover_pose(E, x1, x2, K, K)
    assert rotation_angle_deg(R_est, R) < 1e-4
    assert np.allclose(t_est, t / np.linalg.norm(t), atol=1e-6)  # direction and sign
    assert front.all()
    # triangulated points equal the truth up to the unknown scale |t|
    assert np.allclose(Xs * np.linalg.norm(t), X, atol=1e-5)


def test_triangulation_is_exact_without_noise():
    K, R, t, X, x1, x2 = scene(n=20)
    P1 = projection_matrix(K, np.eye(3), np.zeros(3))
    P2 = projection_matrix(K, R, t)
    assert np.allclose(triangulate(P1, P2, x1, x2), X, atol=1e-8)


def test_sampson_error_approximates_distance_to_epipolar_line():
    K, R, t, X, x1, x2 = scene(noise=0.0)
    F = np.linalg.inv(K).T @ essential_from_pose(R, t) @ np.linalg.inv(K)
    x2n = x2.copy()
    x2n[:, 1] += 0.5  # move points 0.5 px off their epipolar lines (roughly)
    l = np.column_stack([x1, np.ones(len(x1))]) @ F.T
    d2 = np.abs((np.column_stack([x2n, np.ones(len(x2n))]) * l).sum(1)) / np.hypot(l[:, 0], l[:, 1])
    s = sampson_error(F, x1, x2n)
    # Sampson spreads the error over both images, so it is the distance divided by about sqrt(2)
    assert np.allclose(s, d2 / np.sqrt(2), rtol=0.15)


def test_ransac_and_refinement_with_noise_and_outliers():
    K, R, t, X, x1, x2 = scene(n=200, noise=0.5, seed=4)
    rng = np.random.default_rng(1)
    bad = rng.choice(200, 60, replace=False)
    x2[bad] = rng.uniform([0, 0], [640, 480], size=(60, 2))
    F, inl, _ = estimate_fundamental(x1, x2, threshold=1.5, rng=np.random.default_rng(0))
    assert inl[bad].mean() < 0.1
    E = essential_from_fundamental(F, K, K)
    R0, t0, _, _ = recover_pose(E, x1[inl], x2[inl], K, K)
    R1, t1, _ = refine_pose(R0, t0, x1[inl], x2[inl], K, K)
    assert rotation_angle_deg(R1, R) < 0.5
    assert angle_between_deg(t1, t) < 2.0
    Ki = np.linalg.inv(K)
    before = (sampson_error(Ki.T @ essential_from_pose(R0, t0) @ Ki, x1[inl], x2[inl]) ** 2).sum()
    after = (sampson_error(Ki.T @ essential_from_pose(R1, t1) @ Ki, x1[inl], x2[inl]) ** 2).sum()
    assert after <= before


def test_rotation_angle_is_accurate_near_zero_and_at_large_angles():
    for deg in (0.0, 1e-6, 0.5, 90.0, 179.0):
        R = rodrigues(np.array([0.0, 0.0, np.deg2rad(deg)]))
        assert abs(rotation_angle_deg(np.eye(3), R) - deg) < 1e-9
