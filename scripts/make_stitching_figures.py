"""Mosaics of real Oxford image pairs, and the vanishing-point figure.

Each mosaic uses only this repository: our SIFT features, ratio-test matching,
RANSAC + DLT + Levenberg-Marquardt for the homography, and our warping and
blending. The ground-truth homography is used only to measure the error.

    MVG_DATA=data python scripts/make_stitching_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mvg.camera import intrinsics, rodrigues  # noqa: E402
from mvg.datasets import load_oxford  # noqa: E402
from mvg.features import sift  # noqa: E402
from mvg.features.evaluation import corner_error  # noqa: E402
from mvg.features.matching import match  # noqa: E402
from mvg.homography import estimate_homography  # noqa: E402
from mvg.lines import focal_from_orthogonal_vps, vanishing_points  # noqa: E402
from mvg.stitching import mosaic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK = "#0b0b0b"
VP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def stitch(seq_name: str, k: int):
    seq = load_oxford(seq_name)
    img1, imgk = seq.images[0], seq.images[k - 1]
    kp1, d1 = sift.detect_and_describe(img1)
    kpk, dk = sift.detect_and_describe(imgk)
    m = match(dk, d1, ratio=0.8)
    # H maps image k -> image 1, so the mosaic is in image 1's frame
    H, inliers, _ = estimate_homography(kpk.xy[m[:, 0]], kp1.xy[m[:, 1]], threshold=3.0,
                                        rng=np.random.default_rng(0))
    H_true = np.linalg.inv(seq.homographies[k - 2])
    err = corner_error(H, H_true / H_true[2, 2], imgk.shape)
    images = [img1, imgk]
    results = {b: mosaic(images, [np.eye(3), H], blend=b)[0] for b in ("none", "multiband")}
    return img1, imgk, results, len(m), int(inliers.sum()), err


def mosaic_figure(rows):
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 3.6 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 1, 1.6]})
    for ax_row, (name, k, img1, imgk, res, n_matches, n_in, err) in zip(np.atleast_2d(axes), rows):
        ax_row[0].imshow(img1, cmap="gray", vmin=0, vmax=1)
        ax_row[0].set_title(f"{name}, image 1", loc="left", fontsize=10, color=INK)
        ax_row[1].imshow(imgk, cmap="gray", vmin=0, vmax=1)
        ax_row[1].set_title(f"{name}, image {k}", loc="left", fontsize=10, color=INK)
        ax_row[2].imshow(res["multiband"], cmap="gray", vmin=0, vmax=1)
        ax_row[2].set_title(f"Mosaic: {n_in} of {n_matches} matches kept, {err:.1f} px corner error",
                            loc="left", fontsize=10, color=INK)
        for ax in ax_row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "docs/mosaic_oxford.png", dpi=110, facecolor="white")
    plt.close(fig)


def vanishing_point_figure():
    """Synthetic room seen by a known camera: line clusters and the recovered focal length."""
    rng = np.random.default_rng(0)
    K = intrinsics(800, cx=320, cy=240)
    R = rodrigues(np.array([0.35, 0.55, 0.1]))
    t = np.array([0.0, 0.0, 8.0])
    from synthetic_scenes import room_segments  # noqa: E402

    segs, labels = room_segments(rng, K, R, t)
    xh = np.concatenate([segs, np.ones((*segs.shape[:2], 1))], axis=2)
    lines = np.cross(xh[:, 0], xh[:, 1])
    found = vanishing_points(lines, (480, 640), n_points=3, threshold=0.004, rng=np.random.default_rng(1))
    vps = [v[:2] / v[2] for v, _ in found]
    f_est = np.mean([focal_from_orthogonal_vps(np.r_[a, 1], np.r_[b, 1], np.array([320.0, 240.0]))
                     for i, a in enumerate(vps) for b in vps[i + 1:]])
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (_, inl) in enumerate(found):
        c = VP_COLORS[i]
        for s in segs[inl]:
            ax.plot(s[:, 0], s[:, 1], color=c, lw=2)
            # extend each segment towards its vanishing point, faintly
            ax.plot([s[0, 0], vps[i][0]], [s[0, 1], vps[i][1]], color=c, lw=0.6, alpha=0.35)
        ax.scatter(*vps[i], color=c, s=60, edgecolor="white", zorder=3)
    ax.add_patch(plt.Rectangle((0, 0), 640, 480, fill=False, ec="#52514e", lw=1))
    allpts = np.vstack([np.array(vps), [[0, 0], [640, 480]]])
    pad = 80
    ax.set_xlim(allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
    ax.set_ylim(allpts[:, 1].max() + pad, allpts[:, 1].min() - pad)
    ax.set_aspect("equal")
    ax.set_title(f"Three vanishing points from 36 noisy segments; focal length {f_est:.0f} px (true 800)",
                 loc="left", fontsize=10, color=INK)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "docs/vanishing_points.png", dpi=120, facecolor="white")
    plt.close(fig)
    return f_est


def main():
    rows, table = [], []
    for name, k in (("boat", 3), ("graf", 2)):
        img1, imgk, res, n_matches, n_in, err = stitch(name, k)
        rows.append((name, k, img1, imgk, res, n_matches, n_in, err))
        table.append({"pair": f"{name} 1-{k}", "matches": n_matches, "inliers": n_in, "corner_error_px": round(err, 2)})
        print(table[-1])
    mosaic_figure(rows)
    f_est = vanishing_point_figure()
    table.append({"pair": "synthetic room", "matches": "", "inliers": "",
                  "corner_error_px": f"focal {f_est:.1f} (true 800)"})
    with (ROOT / "results/stitching.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print("wrote docs/mosaic_oxford.png, docs/vanishing_points.png, results/stitching.csv")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
