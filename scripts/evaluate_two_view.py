"""Relative pose on all 16 neighbouring pairs of the Middlebury templeSparseRing, against calibrated ground truth.

For each pair (i, i+1): our SIFT features and ratio-test matches, then
  (a) RANSAC + normalised 8-point F -> E -> 4 decompositions -> cheirality,
  (b) the same followed by Levenberg-Marquardt on the Sampson error (5 DoF),
  (c) OpenCV's findEssentialMat (5-point RANSAC) + recoverPose on the same matches.
Errors: rotation angle of R_est^T R_true, and angle between translation directions.

    MVG_DATA=data python scripts/evaluate_two_view.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mvg.datasets import load_middlebury, upright, upright_points  # noqa: E402
from mvg.features import sift  # noqa: E402
from mvg.features.matching import match  # noqa: E402
from mvg.two_view import (  # noqa: E402
    angle_between_deg,
    essential_from_fundamental,
    estimate_fundamental,
    recover_pose,
    refine_pose,
    rotation_angle_deg,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    import cv2

    v = load_middlebury()
    feats = [sift.detect_and_describe(img) for img in v.images]
    n = len(v.images)
    rows, best_pair = [], None
    for i in range(n):
        j = (i + 1) % n
        R_true = v.R[j] @ v.R[i].T
        t_true = v.t[j] - R_true @ v.t[i]
        (k1, d1), (k2, d2) = feats[i], feats[j]
        m = match(d1, d2, ratio=0.8)
        x1, x2 = k1.xy[m[:, 0]], k2.xy[m[:, 1]]
        row = {"pair": f"{i + 1}-{j + 1}", "true_rotation_deg": round(rotation_angle_deg(np.eye(3), R_true), 1),
               "matches": len(m)}
        F, inl, _ = estimate_fundamental(x1, x2, threshold=1.0, rng=np.random.default_rng(0))
        row["inliers"] = int(inl.sum())
        E = essential_from_fundamental(F, v.K[i], v.K[j])
        R0, t0, _, _ = recover_pose(E, x1[inl], x2[inl], v.K[i], v.K[j])
        R1, t1, _ = refine_pose(R0, t0, x1[inl], x2[inl], v.K[i], v.K[j])
        Ec, mask = cv2.findEssentialMat(x1, x2, v.K[i], cv2.RANSAC, 0.999, 1.0)
        _, Rc, tc, _ = cv2.recoverPose(Ec[:3], x1, x2, v.K[i], mask=mask)
        for name, (R, t) in {"8pt": (R0, t0), "8pt_lm": (R1, t1), "opencv_5pt": (Rc, tc.ravel())}.items():
            row[f"{name}_rot_err"] = round(rotation_angle_deg(R, R_true), 2)
            row[f"{name}_t_err"] = round(angle_between_deg(t, t_true), 2)
        rows.append(row)
        print(row)
        if best_pair is None or row["8pt_lm_rot_err"] + row["8pt_lm_t_err"] < best_pair[0]:
            best_pair = (row["8pt_lm_rot_err"] + row["8pt_lm_t_err"], i, j, x1[inl], x2[inl], F)

    with (ROOT / "results/two_view.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nmedian errors over the 16 pairs (degrees), and pairs with rotation error < 5 degrees:")
    for name in ("8pt", "8pt_lm", "opencv_5pt"):
        r = np.array([row[f"{name}_rot_err"] for row in rows])
        t = np.array([row[f"{name}_t_err"] for row in rows])
        print(f"  {name:11s} rotation {np.median(r):6.2f}  translation {np.median(t):6.2f}  ok {(r < 5).sum()}/16")

    # epipolar lines for the best pair
    _, i, j, x1, x2, F = best_pair
    fig, axes = plt.subplots(1, 2, figsize=(8, 5.6))
    colors = plt.cm.tab10(np.arange(10))
    pick = np.random.default_rng(0).choice(len(x1), min(10, len(x1)), replace=False)
    # The temple cameras are rolled by about 90 degrees: show the images upright, and move the points
    # and lines with them. F is estimated and applied in the raw image coordinates.
    h, w = v.images[j].shape
    for ax, img in zip(axes, (v.images[i], v.images[j])):
        ax.imshow(upright(img), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    for c, k in zip(colors, pick):
        axes[0].scatter(*upright_points(x1[k:k + 1], w)[0], color=c, s=25, edgecolor="white", lw=0.8)
        l = F @ np.r_[x1[k], 1]
        if abs(l[0]) > abs(l[1]):  # steep in the raw image: step along y
            ys = np.linspace(0, h - 1, 50)
            pts = np.column_stack([-(l[1] * ys + l[2]) / l[0], ys])
        else:
            xs = np.linspace(0, w - 1, 50)
            pts = np.column_stack([xs, -(l[0] * xs + l[2]) / l[1]])
        axes[1].plot(*upright_points(pts, w).T, color=c, lw=1.2)
        axes[1].scatter(*upright_points(x2[k:k + 1], w)[0], color=c, s=25, edgecolor="white", lw=0.8)
    axes[1].set_xlim(0, h - 1)
    axes[1].set_ylim(w - 1, 0)
    axes[0].set_title(f"View {i + 1}: points", loc="left", fontsize=10)
    axes[1].set_title(f"View {j + 1}: each match lies on its epipolar line F x", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/epipolar_temple.png", dpi=110, facecolor="white")
    print("wrote results/two_view.csv and docs/epipolar_temple.png")


if __name__ == "__main__":
    main()
