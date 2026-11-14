"""Stereo results: disparity on the Middlebury motorcycle pair, rectification of a temple pair.

* Motorcycle (bundled with scikit-image, already rectified, with ground-truth
  disparity): our block matching and SGM against OpenCV's StereoBM and
  StereoSGBM. A pixel is "bad" if its disparity is off by more than 2 px.
* Temple views 7 and 8 (calibrated, not rectified): rectify with the known
  cameras, and measure how far our SIFT matches are from lying on the same row
  before and after.

    MVG_DATA=data python scripts/make_stereo_figures.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mvg.homography import apply_homography  # noqa: E402
from mvg.stereo import depth_from_disparity, disparity, rectify  # noqa: E402
from mvg.stitching import warp_perspective  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX_DISP = 64
# calibration of the down-sampled motorcycle pair, from scikit-image's documentation
F, BASELINE_MM, DOFFS = 994.978, 193.001, 31.086


def scores(d, gt):
    valid = np.isfinite(gt)
    ok = valid & np.isfinite(d)
    err = np.abs(d - gt)
    return {"density": ok.sum() / valid.sum(), "bad2_of_kept": (err[ok] > 2).mean(),
            "bad2_counting_missing": ((err > 2) | ~np.isfinite(d))[valid].mean(), "mean_abs_error_px": err[ok].mean()}


def motorcycle():
    import cv2
    from skimage import color, data

    left_rgb, right_rgb, gt = data.stereo_motorcycle()
    left, right = color.rgb2gray(left_rgb), color.rgb2gray(right_rgb)
    lu, ru = (left * 255).astype(np.uint8), (right * 255).astype(np.uint8)
    runs = {}
    for name, fn in {
        "Block matching, SAD (ours)": lambda: disparity(left, right, MAX_DISP, method="bm", cost="sad", window=9),
        "Block matching, census (ours)": lambda: disparity(left, right, MAX_DISP, method="bm", cost="census", window=9),
        "SGM, census (ours)": lambda: disparity(left, right, MAX_DISP, method="sgm", cost="census"),
        "StereoBM (OpenCV)": lambda: cv2.StereoBM_create(MAX_DISP, 15).compute(lu, ru).astype(np.float32) / 16,
        "StereoSGBM (OpenCV)": lambda: cv2.StereoSGBM_create(
            0, MAX_DISP, 5, P1=8 * 25, P2=32 * 25, uniquenessRatio=10, disp12MaxDiff=1,
            speckleWindowSize=100, speckleRange=2).compute(lu, ru).astype(np.float32) / 16,
    }.items():
        t = time.perf_counter()
        d = fn()
        seconds = time.perf_counter() - t
        d[d <= 0] = np.nan  # OpenCV marks invalid pixels with negative values
        runs[name] = (d, seconds)
    rows = []
    for name, (d, seconds) in runs.items():
        s = scores(d, gt)
        rows.append({"data": "motorcycle", "method": name, **{k: round(float(v), 3) for k, v in s.items()},
                     "seconds": round(seconds, 2)})
        print(rows[-1])

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    show = [("Left image", left, "gray"), ("Ground truth", gt, "magma"),
            ("SGM, census (ours)", runs["SGM, census (ours)"][0], "magma"),
            ("StereoSGBM (OpenCV)", runs["StereoSGBM (OpenCV)"][0], "magma")]
    for ax, (title, img, cmap) in zip(axes, show):
        if cmap == "gray":
            ax.imshow(img, cmap="gray")
        else:
            m = np.ma.masked_invalid(img)
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("#d9d7d0")
            im = ax.imshow(m, cmap=cm, vmin=0, vmax=60)
        ax.set_title(title, loc="left", fontsize=10)
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes[1:], fraction=0.015, pad=0.01)
    ticks = [10, 20, 30, 40, 50, 60]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{d} px = {depth_from_disparity(d, F, BASELINE_MM, DOFFS) / 1000:.1f} m" for d in ticks],
                        fontsize=8)
    fig.savefig(ROOT / "docs/stereo_motorcycle.png", dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return rows


def temple_rectification():
    from mvg.datasets import load_middlebury
    from mvg.features import sift
    from mvg.features.matching import match
    from mvg.two_view import estimate_fundamental

    v = load_middlebury()
    i, j = 6, 7  # views 7 and 8
    T1, T2, K, R, C1, C2 = rectify(v.K[i], v.R[i], v.t[i], v.K[j], v.R[j], v.t[j])
    (k1, d1), (k2, d2) = sift.detect_and_describe(v.images[i]), sift.detect_and_describe(v.images[j])
    m = match(d1, d2, ratio=0.8)
    x1, x2 = k1.xy[m[:, 0]], k2.xy[m[:, 1]]
    _, inl, _ = estimate_fundamental(x1, x2, 1.0, rng=np.random.default_rng(0))
    x1, x2 = x1[inl], x2[inl]
    r1, r2 = apply_homography(T1, x1), apply_homography(T2, x2)
    dy_before = np.abs(x1[:, 1] - x2[:, 1])
    dy_after = np.abs(r1[:, 1] - r2[:, 1])
    print(f"temple 7-8: {len(x1)} matches; median |dy| {np.median(dy_before):.1f} px before, "
          f"{np.median(dy_after):.2f} px after")

    # common vertical range, separate horizontal ranges
    h, w = v.images[i].shape
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    c1, c2 = apply_homography(T1, corners), apply_homography(T2, corners)
    y0, y1 = min(c1[:, 1].min(), c2[:, 1].min()), max(c1[:, 1].max(), c2[:, 1].max())
    out = []
    for T, c in ((T1, c1), (T2, c2)):
        x0 = c[:, 0].min()
        shape = (int(y1 - y0) + 1, int(c[:, 0].max() - x0) + 1)
        out.append((warp_perspective(v.images[i] if T is T1 else v.images[j], T, shape, (x0, y0))[0], x0))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    pick = np.random.default_rng(1).choice(len(x1), min(12, len(x1)), replace=False)
    colors = plt.cm.tab20(np.linspace(0, 1, len(pick)))
    for ax, img, pts, title in ((axes[0, 0], v.images[i], x1, "View 7, as recorded (the camera is rolled 90°)"),
                                (axes[0, 1], v.images[j], x2, "View 8, as recorded")):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        for c, k in zip(colors, pick):
            ax.axhline(x1[k, 1], color=c, lw=0.7)
            ax.scatter(*pts[k], color=c, s=22, edgecolor="white", lw=0.6, zorder=3)
        suffix = " (lines at the rows of the view 7 points)" if pts is x2 else ""
        ax.set_title(title + suffix, loc="left", fontsize=9.5)
        ax.axis("off")
    for ax, (img, x0), pts, title in ((axes[1, 0], out[0], r1, "View 7, rectified"),
                                      (axes[1, 1], out[1], r2, "View 8, rectified: each match is on the same row")):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        for c, k in zip(colors, pick):
            ax.axhline(r1[k, 1] - y0, color=c, lw=0.7)
            ax.scatter(pts[k, 0] - x0, pts[k, 1] - y0, color=c, s=22, edgecolor="white", lw=0.6, zorder=3)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "docs/rectified_temple.png", dpi=105, facecolor="white")
    plt.close(fig)
    return [{"data": "temple 7-8", "method": "rectification with the calibrated cameras",
             "density": "", "bad2_of_kept": "", "bad2_counting_missing": "",
             "mean_abs_error_px": f"median |row difference| {np.median(dy_before):.1f} px before, "
                                  f"{np.median(dy_after):.2f} px after ({len(x1)} matches)", "seconds": ""}]


def main():
    rows = motorcycle() + temple_rectification()
    with (ROOT / "results/stereo.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/stereo.csv, docs/stereo_motorcycle.png, docs/rectified_temple.png")


if __name__ == "__main__":
    main()
