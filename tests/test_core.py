import numpy as np

from mvg.camera import (
    camera_center,
    decompose_projection,
    distort,
    intrinsics,
    project,
    projection_matrix,
    rodrigues,
    rotation_to_axis_angle,
    undistort,
)
from mvg.homogeneous import (
    affine,
    apply_transform,
    euclidean,
    from_homogeneous,
    intersect,
    line_through,
    normalizing_transform,
    similarity,
    skew,
    to_homogeneous,
)
from mvg.optimize import levenberg_marquardt


# ------------------------------------------------------------ homogeneous
def test_homogeneous_round_trip():
    x = np.random.default_rng(0).normal(size=(10, 2))
    assert np.allclose(from_homogeneous(to_homogeneous(x)), x)
    assert np.allclose(from_homogeneous(3.5 * to_homogeneous(x)), x)  # scale does not matter


def test_normalizing_transform_centres_and_scales():
    x = np.random.default_rng(1).uniform(0, 4000, size=(50, 2))
    xn = apply_transform(normalizing_transform(x), x)
    assert np.allclose(xn.mean(axis=0), 0, atol=1e-9)
    assert np.isclose(np.linalg.norm(xn, axis=1).mean(), np.sqrt(2))


def test_skew_is_cross_product():
    a, b = np.array([1.0, 2, 3]), np.array([-4.0, 0.5, 2])
    assert np.allclose(skew(a) @ b, np.cross(a, b))


def test_parallel_lines_meet_at_infinity():
    l1 = line_through(np.array([0, 0, 1.0]), np.array([1, 1, 1.0]))
    l2 = line_through(np.array([0, 1, 1.0]), np.array([1, 2, 1.0]))
    p = intersect(l1, l2)
    assert abs(p[2]) < 1e-12  # the point at infinity in direction (1, 1)
    assert np.isclose(p[0], p[1])


def test_transform_hierarchy_invariants():
    pts = np.random.default_rng(2).normal(size=(3, 2))
    d = lambda p: np.linalg.norm(p[0] - p[1])  # noqa: E731
    E = euclidean(0.7, 3, -2)
    S = similarity(2.5, 0.7, 3, -2)
    A = affine(np.array([[2.0, 0.3], [0.1, 0.8]]), np.array([1.0, 1.0]))
    assert np.isclose(d(apply_transform(E, pts)), d(pts))  # Euclidean keeps lengths
    assert np.isclose(d(apply_transform(S, pts)), 2.5 * d(pts))  # similarity scales them
    # affine keeps parallelism: two parallel segments stay parallel
    seg = np.array([[0, 0], [1, 2], [3, 0], [4, 2]], dtype=float)
    a = apply_transform(A, seg)
    v1, v2 = a[1] - a[0], a[3] - a[2]
    assert np.isclose(v1[0] * v2[1] - v1[1] * v2[0], 0)


# ------------------------------------------------------------ camera
def test_rodrigues_gives_rotations_and_inverts():
    rng = np.random.default_rng(3)
    for _ in range(20):
        r = rng.normal(size=3)
        r *= rng.uniform(0.01, 3.0) / np.linalg.norm(r)
        R = rodrigues(r)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1)
        assert np.allclose(rotation_to_axis_angle(R), r, atol=1e-8)


def test_decompose_projection_recovers_k_r_t():
    K = intrinsics(800, 820, 320, 240, skew_=1.5)
    R = rodrigues(np.array([0.1, -0.4, 0.2]))
    t = np.array([0.5, -0.2, 4.0])
    P = 7.3 * projection_matrix(K, R, t)  # arbitrary scale
    K2, R2, t2 = decompose_projection(P)
    assert np.allclose(K2, K, atol=1e-6)
    assert np.allclose(R2, R, atol=1e-9)
    assert np.allclose(t2, t, atol=1e-9)
    assert np.allclose(camera_center(P), -R.T @ t)


def test_projection_of_point_in_front_of_camera():
    K = intrinsics(500, cx=320, cy=240)
    P = projection_matrix(K, np.eye(3), np.zeros(3))
    assert np.allclose(project(P, np.array([[0.0, 0, 5]])), [[320, 240]])  # on the optical axis
    assert np.allclose(project(P, np.array([[1.0, 0, 5]])), [[420, 240]])  # f * X / Z


def test_undistort_inverts_distort():
    x = np.random.default_rng(4).uniform(-0.6, 0.6, size=(100, 2))
    k = (-0.25, 0.08)
    assert np.allclose(undistort(distort(x, k), k), x, atol=1e-8)


# ------------------------------------------------------------ optimisation
def test_levenberg_marquardt_solves_rosenbrock():
    # Rosenbrock as least squares: r = (10 (y - x^2), 1 - x); minimum at (1, 1)
    fun = lambda p: np.array([10 * (p[1] - p[0] ** 2), 1 - p[0]])  # noqa: E731
    res = levenberg_marquardt(fun, np.array([-1.2, 1.0]))
    assert res.converged
    assert np.allclose(res.x, [1, 1], atol=1e-6)
    assert all(b <= a for a, b in zip(res.history, res.history[1:]))  # LM never increases the cost


def test_levenberg_marquardt_fits_exponential_from_bad_start():
    t = np.linspace(0, 4, 40)
    y = 2.0 * np.exp(-1.3 * t) + 0.5
    fun = lambda p: p[0] * np.exp(-p[1] * t) + p[2] - y  # noqa: E731
    res = levenberg_marquardt(fun, np.array([10.0, 0.01, -3.0]))
    assert np.allclose(res.x, [2.0, 1.3, 0.5], atol=1e-6)
