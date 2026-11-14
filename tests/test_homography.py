import numpy as np
import pytest

from mvg.homography import (
    apply_homography,
    dlt_homography,
    estimate_homography,
    refine_homography,
    symmetric_transfer_error,
    transfer_error,
)


def random_homography(rng: np.random.Generator) -> np.ndarray:
    """A realistic homography: a plane seen from two nearby camera poses (640 x 480 image)."""
    H = np.eye(3)
    H[:2, :2] += rng.normal(scale=0.1, size=(2, 2))
    H[:2, 2] = rng.normal(scale=30, size=2)
    H[2, :2] = rng.normal(scale=2e-4, size=2)
    return H


def matches(rng, n, H, noise=0.0, outlier_ratio=0.0):
    src = rng.uniform([0, 0], [640, 480], size=(n, 2))
    dst = apply_homography(H, src) + rng.normal(scale=noise, size=(n, 2))
    n_out = int(outlier_ratio * n)
    outliers = rng.choice(n, n_out, replace=False)
    dst[outliers] = rng.uniform([0, 0], [640, 480], size=(n_out, 2))
    is_inlier = np.ones(n, bool)
    is_inlier[outliers] = False
    return src, dst, is_inlier


def test_dlt_is_exact_on_four_noise_free_points():
    rng = np.random.default_rng(0)
    H = random_homography(rng)
    src, dst, _ = matches(rng, 4, H)
    H_est = dlt_homography(src, dst)
    assert np.allclose(H_est, H / H[2, 2], atol=1e-8)


def test_dlt_rejects_collinear_minimal_sample():
    src = np.array([[0, 0], [1, 1], [2, 2], [5, 1]], dtype=float)
    dst = src + 3
    assert dlt_homography(src, dst) is None


@pytest.mark.parametrize("normalize", [True, False])
def test_dlt_least_squares_with_noise(normalize):
    rng = np.random.default_rng(1)
    H = random_homography(rng)
    src, dst, _ = matches(rng, 200, H, noise=0.5)
    H_est = dlt_homography(src, dst, normalize=normalize)
    clean = rng.uniform([0, 0], [640, 480], size=(500, 2))
    err = transfer_error(H_est, clean, apply_homography(H, clean))
    assert err.mean() < (0.5 if normalize else 5.0)


def test_normalization_helps_on_average():
    """Hartley's point: without normalisation, DLT is less accurate on pixel coordinates."""
    rng = np.random.default_rng(2)
    errors = {True: [], False: []}
    for _ in range(30):
        H = random_homography(rng)
        src, dst, _ = matches(rng, 20, H, noise=1.0)
        clean = rng.uniform([0, 0], [640, 480], size=(200, 2))
        for norm in (True, False):
            H_est = dlt_homography(src, dst, normalize=norm)
            errors[norm].append(transfer_error(H_est, clean, apply_homography(H, clean)).mean())
    assert np.mean(errors[True]) < np.mean(errors[False])


def test_lm_refinement_lowers_geometric_error():
    rng = np.random.default_rng(3)
    H = random_homography(rng)
    src, dst, _ = matches(rng, 30, H, noise=2.0)
    H_dlt = dlt_homography(src, dst)
    H_lm, result = refine_homography(H_dlt, src, dst)
    before = (symmetric_transfer_error(H_dlt, src, dst) ** 2).sum()
    after = (symmetric_transfer_error(H_lm, src, dst) ** 2).sum()
    assert after <= before
    assert result.converged


def test_ransac_finds_homography_despite_outliers():
    rng = np.random.default_rng(4)
    H = random_homography(rng)
    src, dst, is_inlier = matches(rng, 300, H, noise=0.5, outlier_ratio=0.5)
    H_est, mask, result = estimate_homography(src, dst, threshold=3.0, rng=np.random.default_rng(0))
    clean = rng.uniform([0, 0], [640, 480], size=(200, 2))
    assert transfer_error(H_est, clean, apply_homography(H, clean)).mean() < 0.5
    # almost every true inlier kept, almost every outlier rejected
    assert (mask & is_inlier).sum() / is_inlier.sum() > 0.97
    assert (mask & ~is_inlier).sum() / (~is_inlier).sum() < 0.03
    # adaptive stopping: far fewer than the 10,000 cap at 50% inliers
    assert result.iterations < 400


def test_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(5)
    H = random_homography(rng)
    src, dst, _ = matches(rng, 200, H, noise=0.7, outlier_ratio=0.3)
    H_ours, _, _ = estimate_homography(src, dst, threshold=3.0, rng=np.random.default_rng(1))
    H_cv, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    clean = rng.uniform([0, 0], [640, 480], size=(200, 2))
    ours = transfer_error(H_ours, clean, apply_homography(H, clean)).mean()
    theirs = transfer_error(H_cv, clean, apply_homography(H, clean)).mean()
    assert ours < 0.5 and theirs < 0.5
    assert ours <= theirs * 1.5  # comparable accuracy
