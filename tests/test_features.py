import numpy as np
import pytest
from scipy import ndimage

from mvg.features import orb, sift, surf
from mvg.features.corners import detect_corners, fast_score, harris_response, shi_tomasi_response, structure_tensor
from mvg.features.evaluation import corner_error, repeatability
from mvg.features.matching import hamming_distances, match
from mvg.homography import estimate_homography


def square(size=100, lo=30, hi=70):
    img = np.zeros((size, size))
    img[lo:hi, lo:hi] = 1.0
    return img


def blob(sigma, size=160):
    yy, xx = np.mgrid[0:size, 0:size]
    c = size / 2
    return np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma**2))


def texture(size=192, seed=0):
    """Smooth random texture: many corners and blobs at several scales."""
    rng = np.random.default_rng(seed)
    img = sum(ndimage.gaussian_filter(rng.normal(size=(size, size)), s) * s for s in (1.5, 3, 6))
    img = (img - img.min()) / (img.max() - img.min())
    return img


# ------------------------------------------------------------ corners
def test_harris_finds_the_four_corners_of_a_square():
    pts = detect_corners(harris_response(square()), max_corners=4)
    expected = np.array([[30, 30], [69, 30], [30, 69], [69, 69]])
    for e in expected:
        assert np.min(np.linalg.norm(pts - e, axis=1)) <= 1.5


def test_shi_tomasi_is_the_smaller_eigenvalue():
    img = texture(64)
    sxx, sxy, syy = structure_tensor(img)
    r = shi_tomasi_response(img)
    y, x = 30, 33
    eig = np.linalg.eigvalsh(np.array([[sxx[y, x], sxy[y, x]], [sxy[y, x], syy[y, x]]]))
    assert np.isclose(r[y, x], eig[0])


def test_fast_fires_on_corners_not_on_edges_or_flat_areas():
    score = fast_score(square(), threshold=0.2)
    assert score[30, 30] > 0 or score[31, 31] > 0  # corner
    assert score[30, 50] == 0  # middle of an edge: only 7 circle pixels differ
    assert score[50, 50] == 0  # flat inside
    assert score[10, 10] == 0  # flat outside


# ------------------------------------------------------------ SIFT
@pytest.mark.parametrize("sigma_blob", [3.0, 6.0])
def test_dog_detects_blob_at_sigma_over_sqrt_k(sigma_blob):
    """The DoG between sigma and k*sigma peaks at sigma = sigma_blob / sqrt(k) (see docs/math/features.md)."""
    kp, _ = sift.detect(blob(sigma_blob), max_keypoints=5)
    i = np.argmax(kp.response)
    assert np.allclose(kp.xy[i], [80, 80], atol=0.5)
    expected = sigma_blob / np.sqrt(2 ** (1 / 3))
    assert abs(kp.scale[i] / expected - 1) < 0.05


def _corresponding(xy_a, xy_b, transform, tol=1.0):
    pairs = []
    mapped = transform(xy_a)
    for i, p in enumerate(mapped):
        d = np.linalg.norm(xy_b - p, axis=1)
        j = int(np.argmin(d))
        if d[j] < tol:
            pairs.append((i, j))
    return np.array(pairs)


@pytest.mark.parametrize("module", [sift, surf])
def test_descriptors_survive_a_90_degree_rotation(module):
    img = texture()
    rot = np.rot90(img)  # (x, y) -> (y, W - 1 - x)
    w = img.shape[1]
    kp_a, d_a = module.detect_and_describe(img, 300)
    kp_b, d_b = module.detect_and_describe(rot, 300)
    pairs = _corresponding(kp_a.xy, kp_b.xy, lambda p: np.column_stack([p[:, 1], w - 1 - p[:, 0]]))
    assert len(pairs) > 20
    same = np.linalg.norm(d_a[pairs[:, 0]] - d_b[pairs[:, 1]], axis=1)
    rng = np.random.default_rng(0)
    other = np.linalg.norm(d_a[pairs[:, 0]] - d_b[rng.permutation(pairs[:, 1])], axis=1)
    assert np.median(same) < 0.5 * np.median(other)


# ------------------------------------------------------------ SURF
def test_integral_image_box_sum():
    img = np.random.default_rng(1).random((30, 40))
    ii = surf.integral_image(img)
    assert np.isclose(surf.box_sum(ii, 3, 5, 17, 22), img[3:17, 5:22].sum())


def test_surf_finds_blob_centre():
    kp = surf.detect(blob(6.0, 240), max_keypoints=5, threshold=1e-5)
    assert np.allclose(kp.xy[np.argmax(kp.response)], [120, 120], atol=1.0)


# ------------------------------------------------------------ ORB-style
def test_intensity_centroid_points_towards_brighter_side():
    yy, xx = np.mgrid[0:64, 0:64].astype(float)
    ramp = xx / 63  # brighter to the right
    assert abs(orb.intensity_centroid_angle(ramp, 32, 32)) < 0.05
    assert abs(orb.intensity_centroid_angle(yy / 63, 32, 32) - np.pi / 2) < 0.05  # brighter downwards


def test_hamming_distance_matches_bit_count():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, size=(5, 32), dtype=np.uint8)
    b = rng.integers(0, 256, size=(7, 32), dtype=np.uint8)
    naive = np.array([[np.unpackbits(x ^ y).sum() for y in b] for x in a])
    assert np.array_equal(hamming_distances(a, b), naive)


# ------------------------------------------------------------ matching and evaluation
def test_ratio_test_drops_ambiguous_matches_and_mutual_check_drops_one_sided():
    a = np.array([[0.0, 0], [10, 10], [5, 5.1]])
    b = np.array([[0.1, 0], [10, 10.1], [5, 5], [5, 5.2]])  # a[2] has two equally close candidates
    m = match(a, b, ratio=0.8)
    assert {tuple(r) for r in m} == {(0, 0), (1, 1)}
    m = match(a[:2], b[:2], ratio=None, mutual=True)
    assert {tuple(r) for r in m} == {(0, 0), (1, 1)}


def test_repeatability_is_one_for_identical_points_and_never_above_one():
    xy = np.random.default_rng(3).uniform(0, 100, size=(50, 2))
    assert repeatability(xy, xy, np.eye(3), (100, 100), (100, 100)) == 1.0
    crowded = np.vstack([xy, xy + 0.5])
    assert repeatability(crowded, xy, np.eye(3), (100, 100), (100, 100)) <= 1.0


def test_end_to_end_sift_recovers_a_known_homography():
    cv2 = pytest.importorskip("cv2")
    img = texture(256, seed=5)
    H = np.array([[0.95, 0.12, 8.0], [-0.08, 1.02, -5.0], [1e-4, -2e-4, 1.0]])
    warped = cv2.warpPerspective(img, H, (256, 256), flags=cv2.INTER_LINEAR)
    kp1, d1 = sift.detect_and_describe(img, 500)
    kp2, d2 = sift.detect_and_describe(warped, 500)
    m = match(d1, d2, ratio=0.8)
    H_est, mask, _ = estimate_homography(kp1.xy[m[:, 0]], kp2.xy[m[:, 1]], threshold=2.0,
                                         rng=np.random.default_rng(0))
    assert corner_error(H_est, H, (256, 256)) < 1.0
