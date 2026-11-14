import numpy as np
import pytest

from mvg.bundle_adjustment import _residuals_and_jacobians, bundle_adjust, umeyama
from mvg.camera import intrinsics, project, projection_matrix, rodrigues
from mvg.optimize import numerical_jacobian
from mvg.pnp import pnp_dlt, refine_pnp, reprojection_errors, solve_pnp
from mvg.two_view import rotation_angle_deg

K = intrinsics(700, cx=320, cy=240)


def pnp_scene(n=60, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    R = rodrigues(np.array([0.2, -0.4, 0.1]))
    t = np.array([0.3, -0.2, 6.0])
    X = rng.uniform(-2, 2, size=(n, 3))
    x = project(projection_matrix(K, R, t), X) + rng.normal(scale=noise, size=(n, 2))
    return R, t, X, x


# ------------------------------------------------------------ PnP
def test_pnp_dlt_is_exact_without_noise():
    R, t, X, x = pnp_scene()
    R_est, t_est = pnp_dlt(X, x, K)
    assert rotation_angle_deg(R_est, R) < 1e-6
    assert np.allclose(t_est, t, atol=1e-6)


def test_pnp_refinement_lowers_reprojection_error():
    R, t, X, x = pnp_scene(noise=1.0)
    R0, t0 = pnp_dlt(X, x, K)
    R1, t1, _ = refine_pnp(R0, t0, X, x, K)
    assert (reprojection_errors(R1, t1, X, x, K) ** 2).sum() <= (reprojection_errors(R0, t0, X, x, K) ** 2).sum()
    assert rotation_angle_deg(R1, R) < 0.5


def test_pnp_ransac_with_outliers_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    R, t, X, x = pnp_scene(n=150, noise=0.5, seed=1)
    rng = np.random.default_rng(2)
    bad = rng.choice(150, 45, replace=False)
    x[bad] = rng.uniform([0, 0], [640, 480], size=(45, 2))
    R_est, t_est, mask = solve_pnp(X, x, K, threshold=2.0, rng=np.random.default_rng(0))
    assert mask[bad].mean() < 0.05
    ok, rvec, tvec, inl = cv2.solvePnPRansac(X, x, K, None, reprojectionError=2.0)
    Rc = cv2.Rodrigues(rvec)[0]
    assert rotation_angle_deg(R_est, R) < 0.3
    assert rotation_angle_deg(R_est, Rc) < 0.3
    assert np.allclose(t_est, tvec.ravel(), atol=0.05)


# ------------------------------------------------------------ bundle adjustment
def ring_scene(m=6, n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 3))
    Rs, ts = [], []
    for i in range(m):
        a = np.deg2rad(20 * i)
        C = np.array([6 * np.sin(a), 0.5, -6 * np.cos(a)])  # cameras on a ring, looking at the origin
        z = -C / np.linalg.norm(C)
        x_ax = np.cross([0, 1, 0], z)
        x_ax /= np.linalg.norm(x_ax)
        R = np.vstack([x_ax, np.cross(z, x_ax), z])
        Rs.append(R)
        ts.append(-R @ C)
    cam_idx, pt_idx, obs = [], [], []
    for i in range(m):
        x = project(projection_matrix(K, Rs[i], ts[i]), X)
        cam_idx += [i] * n
        pt_idx += list(range(n))
        obs.append(x)
    return np.array(Rs), np.array(ts), X, np.array(cam_idx), np.array(pt_idx), np.vstack(obs)


def test_ba_jacobian_matches_numerical():
    R, t, X, cam_idx, pt_idx, obs = ring_scene(m=2, n=3)
    _, A, B = _residuals_and_jacobians(K, R, t, X, cam_idx, pt_idx, obs)
    k = 4  # camera 1, point 1
    ci, pi = cam_idx[k], pt_idx[k]

    def f_cam(p):
        Rp, tp = R.copy(), t.copy()
        Rp[ci] = rodrigues(p[:3]) @ R[ci]
        tp[ci] = t[ci] + p[3:]
        return _residuals_and_jacobians(K, Rp, tp, X, cam_idx[[k]], pt_idx[[k]], obs[[k]])[0].ravel()

    def f_pt(p):
        Xp = X.copy()
        Xp[pi] = X[pi] + p
        return _residuals_and_jacobians(K, R, t, Xp, cam_idx[[k]], pt_idx[[k]], obs[[k]])[0].ravel()

    assert np.allclose(A[k], numerical_jacobian(f_cam, np.zeros(6)), atol=1e-4)
    assert np.allclose(B[k], numerical_jacobian(f_pt, np.zeros(3)), atol=1e-4)


def test_schur_step_equals_dense_solve():
    """The reduced camera system gives the same step as solving the full normal equations."""
    R, t, X, cam_idx, pt_idx, obs = ring_scene(m=3, n=15)
    rng = np.random.default_rng(1)
    Xn = X + rng.normal(scale=0.05, size=X.shape)
    r, A, B = _residuals_and_jacobians(K, R, t, Xn, cam_idx, pt_idx, obs)
    n = len(X)
    # cameras 0 and 1 fixed: with only camera 0 fixed the scale is still free and J^T J is singular
    J = np.zeros((2 * len(obs), 6 + 3 * n))
    for k, (c, p) in enumerate(zip(cam_idx, pt_idx)):
        if c == 2:
            J[2 * k : 2 * k + 2, 0:6] = A[k]
        J[2 * k : 2 * k + 2, 6 + 3 * p : 6 + 3 * p + 3] = B[k]
    step_dense = np.linalg.solve(J.T @ J, -J.T @ r.ravel())
    res = bundle_adjust(K, R, t, Xn, cam_idx, pt_idx, obs, fixed_cameras=(0, 1), max_iterations=1, lam=1e-12)
    dX = res.X - Xn
    assert np.allclose(dX.ravel(), step_dense[6:], atol=1e-6)
    assert np.allclose(res.t[2] - t[2], step_dense[3:6], atol=1e-6)


def test_bundle_adjustment_brings_error_to_noise_level():
    R, t, X, cam_idx, pt_idx, obs = ring_scene()
    rng = np.random.default_rng(3)
    obs_noisy = obs + rng.normal(scale=0.5, size=obs.shape)
    # perturb every camera but the first, and every point
    R0, t0 = R.copy(), t.copy()
    for i in range(1, len(R)):
        R0[i] = rodrigues(rng.normal(scale=0.02, size=3)) @ R[i]
        t0[i] = t[i] + rng.normal(scale=0.1, size=3)
    X0 = X + rng.normal(scale=0.05, size=X.shape)
    res = bundle_adjust(K, R0, t0, X0, cam_idx, pt_idx, obs_noisy)
    assert res.rms_before > 5
    assert res.rms_after < 0.5 * np.sqrt(2) * 1.05  # at the noise level
    assert all(b <= a for a, b in zip(res.history, res.history[1:]))
    # camera centres after a similarity alignment (scale is free)
    C_est = np.array([-r.T @ tt for r, tt in zip(res.R, res.t)])
    C_true = np.array([-r.T @ tt for r, tt in zip(R, t)])
    s, Ra, ta = umeyama(C_est, C_true)
    aligned = s * C_est @ Ra.T + ta
    assert np.abs(aligned - C_true).max() < 0.05


def test_umeyama_recovers_a_similarity():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(30, 3))
    R = rodrigues(np.array([0.3, -0.7, 0.2]))
    b = 2.5 * a @ R.T + np.array([1.0, -2.0, 0.5])
    s, R_est, t_est = umeyama(a, b)
    assert np.isclose(s, 2.5) and rotation_angle_deg(R_est, R) < 1e-8
    assert np.allclose(s * a @ R_est.T + t_est, b)
