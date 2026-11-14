"""Warping, blending and panoramas.

* **Inverse warping.** To warp an image by H, loop over *output* pixels and
  sample the input at H^-1 x. Forward mapping (pushing each input pixel to H x)
  leaves holes and overlaps; inverse mapping fills every output pixel exactly
  once, with bilinear interpolation between the four nearest input pixels.
* **Feather blending.** Where images overlap, weight each by its distance to its
  own border, so the seam fades across the overlap.
* **Multi-band blending** (Burt and Adelson, 1983). Split each image into a
  Laplacian pyramid (band-pass layers), blend each layer with a mask blurred to
  match that band, and add the layers back up. Low frequencies (exposure
  differences) blend over a wide region, high frequencies (detail) over a narrow
  one, so there is neither a visible seam nor ghosting.
* **Cylindrical projection.** For a camera that only pans, map each pixel onto a
  cylinder of radius f: x' = f atan(x / f), y' = f y / sqrt(x^2 + f^2)
  (coordinates relative to the principal point). Panning by an angle then
  becomes a horizontal shift by f * angle, so a 360-degree panorama is a strip.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from mvg.features.filters import bilinear
from mvg.homography import apply_homography


def warp_perspective(img: np.ndarray, H: np.ndarray, out_shape, offset=(0.0, 0.0)):
    """Warp ``img`` by H onto a canvas of ``out_shape`` whose pixel (0, 0) is at ``offset``.

    Returns (warped, valid) where ``valid`` marks output pixels that came from
    inside the input image. Works for grayscale (h, w) or colour (h, w, c).
    """
    h, w = out_shape
    ys, xs = np.mgrid[0:h, 0:w]
    canvas = np.column_stack([xs.ravel() + offset[0], ys.ravel() + offset[1]])
    src = apply_homography(np.linalg.inv(H), canvas)
    sx, sy = src[:, 0], src[:, 1]
    ih, iw = img.shape[:2]
    valid = (sx >= 0) & (sx <= iw - 1) & (sy >= 0) & (sy <= ih - 1)
    if img.ndim == 2:
        out = bilinear(img, sx, sy).reshape(h, w)
    else:
        out = np.stack([bilinear(img[..., c], sx, sy) for c in range(img.shape[2])], axis=-1).reshape(h, w, -1)
    valid = valid.reshape(h, w)
    out[~valid] = 0
    return out, valid


def canvas_for(shapes, homographies):
    """Bounding box of all images after warping; returns (out_shape, offset)."""
    pts = []
    for (h, w), H in zip(shapes, homographies):
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
        pts.append(apply_homography(H, corners))
    pts = np.vstack(pts)
    lo, hi = np.floor(pts.min(axis=0)), np.ceil(pts.max(axis=0))
    return (int(hi[1] - lo[1] + 1), int(hi[0] - lo[0] + 1)), (lo[0], lo[1])


def feather_weights(valid: np.ndarray) -> np.ndarray:
    """Distance to the nearest invalid pixel (0 outside the image)."""
    return ndimage.distance_transform_edt(valid)


def _gaussian_pyramid(img, levels):
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(ndimage.gaussian_filter(pyr[-1], 1.0)[::2, ::2])
    return pyr


def _upsample(img, shape):
    factors = (shape[0] / img.shape[0], shape[1] / img.shape[1])
    return ndimage.zoom(img, factors, order=1)[: shape[0], : shape[1]]


def _laplacian_pyramid(img, levels):
    g = _gaussian_pyramid(img, levels)
    return [g[i] - _upsample(g[i + 1], g[i].shape) for i in range(levels - 1)] + [g[-1]]


def multiband_blend(images, masks, levels: int = 5) -> np.ndarray:
    """Blend grayscale images with soft masks (weights) using Laplacian pyramids."""
    total = np.sum(masks, axis=0)
    total[total == 0] = 1
    weights = [m / total for m in masks]
    blended = None
    for img, w in zip(images, weights):
        lp = _laplacian_pyramid(img, levels)
        gw = _gaussian_pyramid(w, levels)
        layer = [l * g for l, g in zip(lp, gw)]
        blended = layer if blended is None else [a + b for a, b in zip(blended, layer)]
    out = blended[-1]
    for lap in reversed(blended[:-1]):
        out = _upsample(out, lap.shape) + lap
    return out


def mosaic(images, homographies, blend: str = "feather"):
    """Warp every image into the frame of the first (homographies map image i -> image 0)."""
    shape, offset = canvas_for([im.shape[:2] for im in images], homographies)
    warped, valids = zip(*(warp_perspective(im, H, shape, offset) for im, H in zip(images, homographies)))
    if blend == "none":  # later images overwrite earlier ones
        out = np.zeros(shape)
        for wimg, v in zip(warped, valids):
            out[v] = wimg[v]
        return out, offset
    weights = [feather_weights(v) for v in valids]
    if blend == "feather":
        total = np.sum(weights, axis=0)
        total[total == 0] = 1
        return sum(wimg * wt for wimg, wt in zip(warped, weights)) / total, offset
    if blend == "multiband":
        # hard masks (nearest image centre wins) blended across scales
        hard = np.argmax(np.stack(weights), axis=0)
        masks = [((hard == i) & v).astype(float) for i, v in enumerate(valids)]
        out = multiband_blend(list(warped), masks)
        out[~np.any(valids, axis=0)] = 0
        return out, offset
    raise ValueError(blend)


def cylindrical_coords(x: np.ndarray, y: np.ndarray, f: float):
    """Map image coordinates (relative to the principal point) onto a cylinder of radius f."""
    return f * np.arctan2(x, f), f * y / np.sqrt(x**2 + f**2)


def to_cylinder(img: np.ndarray, f: float) -> np.ndarray:
    """Inverse-warp an image onto the cylinder (same size, principal point at the centre)."""
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    theta = (xs - cx) / f  # cylinder angle for each output column
    hcyl = (ys - cy) / f  # height on the unit cylinder
    x = f * np.tan(theta) + cx
    y = hcyl * f / np.cos(theta) + cy
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1) & (np.abs(theta) < np.pi / 2)
    out = bilinear(img, x.ravel(), y.ravel()).reshape(h, w)
    out[~valid] = 0
    return out
