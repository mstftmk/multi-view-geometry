"""SIFT: scale-space extrema of the difference of Gaussians, and the 128-D descriptor.

Follows D. Lowe, "Distinctive image features from scale-invariant keypoints"
(IJCV 2004), with the usual simplifications noted inline. See
docs/math/features.md for the derivations.

Steps:

1. **Gaussian scale space.** Blur the image with sigma0 * k^i, k = 2^(1/s), s
   layers per octave; halve the image after each octave.
2. **Difference of Gaussians** between neighbouring layers approximates the
   scale-normalised Laplacian, (k - 1) sigma^2 laplacian(G).
3. **Extrema** in a 3x3x3 neighbourhood (space and scale).
4. **Sub-pixel refinement** by a quadratic (Taylor) fit, then rejection of
   low-contrast points and of points on edges (ratio of principal curvatures).
5. **Orientation** from a 36-bin histogram of gradient directions.
6. **Descriptor**: 4x4 cells x 8 orientation bins, in the keypoint's own frame,
   with trilinear interpolation, then normalise, clip at 0.2, normalise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Keypoints:
    xy: np.ndarray  # (N, 2) in input-image pixels
    scale: np.ndarray  # (N,) sigma in input-image pixels
    angle: np.ndarray  # (N,) radians
    response: np.ndarray  # (N,)
    octave: np.ndarray  # (N,) int
    layer: np.ndarray  # (N,) float, layer index within the octave

    def __len__(self) -> int:
        return len(self.xy)

    def take(self, idx) -> "Keypoints":
        return Keypoints(*(a[idx] for a in (self.xy, self.scale, self.angle, self.response,
                                             self.octave, self.layer)))


@dataclass
class ScaleSpace:
    gaussians: list[np.ndarray]  # per octave: (s + 3, h, w)
    dogs: list[np.ndarray]  # per octave: (s + 2, h, w)
    sigma0: float
    s: int


def build_scale_space(img: np.ndarray, s: int = 3, sigma0: float = 1.6, assumed_blur: float = 0.5,
                      n_octaves: int | None = None) -> ScaleSpace:
    if n_octaves is None:
        n_octaves = max(1, int(np.log2(min(img.shape))) - 3)
    k = 2 ** (1 / s)
    sigmas = sigma0 * k ** np.arange(s + 3)
    base = ndimage.gaussian_filter(img, np.sqrt(max(sigma0**2 - assumed_blur**2, 0.01)))
    gaussians, dogs = [], []
    for _ in range(n_octaves):
        layers = [base]
        for i in range(1, s + 3):
            # blur the previous layer by the extra amount needed to reach sigmas[i]
            layers.append(ndimage.gaussian_filter(layers[-1], np.sqrt(sigmas[i] ** 2 - sigmas[i - 1] ** 2)))
        g = np.stack(layers)
        gaussians.append(g)
        dogs.append(g[1:] - g[:-1])
        base = g[s][::2, ::2]  # layer s has twice the base blur: it starts the next octave
    return ScaleSpace(gaussians, dogs, sigma0, s)


def _refine(D: np.ndarray, l: int, y: int, x: int, s: int, max_steps: int = 5):
    """Quadratic fit of D around (l, y, x); returns the offset, refined position and value."""
    for _ in range(max_steps):
        g = 0.5 * np.array([D[l, y, x + 1] - D[l, y, x - 1], D[l, y + 1, x] - D[l, y - 1, x],
                            D[l + 1, y, x] - D[l - 1, y, x]])
        c = D[l, y, x]
        dxx = D[l, y, x + 1] + D[l, y, x - 1] - 2 * c
        dyy = D[l, y + 1, x] + D[l, y - 1, x] - 2 * c
        dss = D[l + 1, y, x] + D[l - 1, y, x] - 2 * c
        dxy = 0.25 * (D[l, y + 1, x + 1] - D[l, y + 1, x - 1] - D[l, y - 1, x + 1] + D[l, y - 1, x - 1])
        dxs = 0.25 * (D[l + 1, y, x + 1] - D[l + 1, y, x - 1] - D[l - 1, y, x + 1] + D[l - 1, y, x - 1])
        dys = 0.25 * (D[l + 1, y + 1, x] - D[l + 1, y - 1, x] - D[l - 1, y + 1, x] + D[l - 1, y - 1, x])
        H = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
        try:
            off = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        if np.all(np.abs(off) < 0.5):
            value = c + 0.5 * g @ off
            return off, (l, y, x), value, (dxx, dyy, dxy)
        x, y, l = x + int(round(off[0])), y + int(round(off[1])), l + int(round(off[2]))
        h, w = D.shape[1:]
        if not (1 <= l <= s and 1 <= y < h - 1 and 1 <= x < w - 1):
            return None
    return None


def detect(img: np.ndarray, max_keypoints: int = 2000, s: int = 3, sigma0: float = 1.6,
           contrast_threshold: float = 0.04, edge_ratio: float = 10.0,
           space: ScaleSpace | None = None) -> tuple[Keypoints, ScaleSpace]:
    space = space or build_scale_space(img, s, sigma0)
    rows = []
    for o, D in enumerate(space.dogs):
        # prefilter at half the final threshold, as in Lowe's implementation
        mx = ndimage.maximum_filter(D, size=3, mode="nearest")
        mn = ndimage.minimum_filter(D, size=3, mode="nearest")
        cand = ((D == mx) | (D == mn)) & (np.abs(D) > 0.5 * contrast_threshold / s)
        cand[[0, -1]] = False
        cand[:, :5], cand[:, -5:], cand[:, :, :5], cand[:, :, -5:] = False, False, False, False
        for l, y, x in zip(*np.nonzero(cand)):
            res = _refine(D, l, y, x, s)
            if res is None:
                continue
            off, (l2, y2, x2), value, (dxx, dyy, dxy) = res
            if abs(value) < contrast_threshold / s:
                continue
            tr, det = dxx + dyy, dxx * dyy - dxy**2
            if det <= 0 or tr**2 * edge_ratio >= (edge_ratio + 1) ** 2 * det:
                continue  # on an edge: one principal curvature much larger than the other
            layer = l2 + off[2]
            rows.append((((x2 + off[0]) * 2**o), ((y2 + off[1]) * 2**o),
                         sigma0 * 2 ** (layer / s) * 2**o, abs(value), o, layer))
    if not rows:
        empty = np.zeros(0)
        return Keypoints(np.zeros((0, 2)), empty, empty, empty, empty.astype(int), empty), space
    a = np.array(rows)
    order = np.argsort(-a[:, 3])
    a = a[order]
    kp = Keypoints(a[:, :2], a[:, 2], np.zeros(len(a)), a[:, 3], a[:, 4].astype(int), a[:, 5])
    kp = assign_orientations(kp, space)
    return kp.take(np.arange(min(max_keypoints, len(kp)))), space


def _grad_polar(g: np.ndarray):
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def _patch(img_shape, cx, cy, radius):
    h, w = img_shape
    ys, xs = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    px, py = np.round(cx).astype(int) + xs, np.round(cy).astype(int) + ys
    inside = (px >= 1) & (px < w - 1) & (py >= 1) & (py < h - 1)
    return xs[inside], ys[inside], px[inside], py[inside]


def assign_orientations(kp: Keypoints, space: ScaleSpace, bins: int = 36, peak_ratio: float = 0.8) -> Keypoints:
    """Dominant gradient directions; a keypoint with several strong peaks is duplicated."""
    cache: dict = {}
    out = []
    for i in range(len(kp)):
        o, l = kp.octave[i], int(round(kp.layer[i]))
        if (o, l) not in cache:
            cache[(o, l)] = _grad_polar(space.gaussians[o][l])
        mag, ang = cache[(o, l)]
        sigma_oct = kp.scale[i] / 2**o
        cx, cy = kp.xy[i] / 2**o
        xs, ys, px, py = _patch(mag.shape, cx, cy, int(round(3 * 1.5 * sigma_oct)))
        weight = np.exp(-(xs**2 + ys**2) / (2 * (1.5 * sigma_oct) ** 2)) * mag[py, px]
        b = np.floor((ang[py, px] + np.pi) / (2 * np.pi) * bins).astype(int) % bins
        hist = np.bincount(b, weight, minlength=bins)
        hist = np.convolve(np.r_[hist[-2:], hist, hist[:2]], [1, 4, 6, 4, 1], "valid") / 16  # circular smoothing
        top = hist.max()
        for j in range(bins):
            left, right = hist[j - 1], hist[(j + 1) % bins]
            if hist[j] >= peak_ratio * top and hist[j] > left and hist[j] > right:
                shift = 0.5 * (left - right) / (left - 2 * hist[j] + right)  # parabola through 3 bins
                theta = (j + 0.5 + shift) / bins * 2 * np.pi - np.pi
                out.append((i, theta))
    idx = np.array([i for i, _ in out], dtype=int)
    res = kp.take(idx)
    res.angle = np.array([t for _, t in out])
    return res


def describe(kp: Keypoints, space: ScaleSpace, d: int = 4, n: int = 8, scale_factor: float = 3.0) -> np.ndarray:
    """(N, 128) float32 SIFT descriptors."""
    cache: dict = {}
    desc = np.zeros((len(kp), d * d * n), np.float32)
    for i in range(len(kp)):
        o, l = kp.octave[i], int(round(kp.layer[i]))
        if (o, l) not in cache:
            cache[(o, l)] = _grad_polar(space.gaussians[o][l])
        mag, ang = cache[(o, l)]
        sigma_oct = kp.scale[i] / 2**o
        cx, cy = kp.xy[i] / 2**o
        cell = scale_factor * sigma_oct  # width of one of the d x d cells, in pixels
        radius = int(round(cell * np.sqrt(2) * (d + 1) / 2))
        xs, ys, px, py = _patch(mag.shape, cx, cy, radius)
        c, s_ = np.cos(kp.angle[i]), np.sin(kp.angle[i])
        # offsets in the keypoint frame, in cell units, shifted so cell centres sit at 0..d-1
        xr = (c * xs + s_ * ys) / cell
        yr = (-s_ * xs + c * ys) / cell
        cbin, rbin = xr + d / 2 - 0.5, yr + d / 2 - 0.5
        obin = ((ang[py, px] - kp.angle[i]) % (2 * np.pi)) / (2 * np.pi) * n
        keep = (rbin > -1) & (rbin < d) & (cbin > -1) & (cbin < d)
        w = mag[py, px] * np.exp(-(xr**2 + yr**2) / (2 * (0.5 * d) ** 2))
        rbin, cbin, obin, w = rbin[keep], cbin[keep], obin[keep], w[keep]
        r0, c0, o0 = np.floor(rbin).astype(int), np.floor(cbin).astype(int), np.floor(obin).astype(int)
        fr, fc, fo = rbin - r0, cbin - c0, obin - o0
        hist = np.zeros((d + 2, d + 2, n))
        # trilinear interpolation: each sample spreads over 2 x 2 x 2 bins
        for dr in (0, 1):
            wr = fr if dr else 1 - fr
            for dc in (0, 1):
                wc = fc if dc else 1 - fc
                for do in (0, 1):
                    wo = fo if do else 1 - fo
                    np.add.at(hist, (r0 + 1 + dr, c0 + 1 + dc, (o0 + do) % n), w * wr * wc * wo)
        v = hist[1:-1, 1:-1].ravel()
        v /= max(np.linalg.norm(v), 1e-12)
        v = np.minimum(v, 0.2)  # limit the influence of large gradients (non-linear lighting)
        desc[i] = v / max(np.linalg.norm(v), 1e-12)
    return desc


def detect_and_describe(img: np.ndarray, max_keypoints: int = 2000):
    kp, space = detect(img, max_keypoints)
    return kp, describe(kp, space)
