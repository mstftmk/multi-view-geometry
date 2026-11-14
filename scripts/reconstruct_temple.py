"""Incremental reconstruction of the Middlebury temple from views 6 to 11, checked against the calibrated cameras.

These six views are consecutive and 23 degrees apart (see results/two_view.csv).
Our SIFT features, two-view start, PnP, triangulation and bundle adjustment
produce camera poses in an arbitrary frame and scale; they are compared with
the ground truth after a similarity alignment of the camera centres (Umeyama).

    MVG_DATA=data python scripts/reconstruct_temple.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mvg.bundle_adjustment import umeyama  # noqa: E402
from mvg.datasets import load_middlebury, upright  # noqa: E402
from mvg.features import sift  # noqa: E402
from mvg.sfm import reconstruct_chain  # noqa: E402
from mvg.two_view import rotation_angle_deg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VIEWS = list(range(5, 11))  # views 6..11 (0-based indices)
VIEW = (70, -60)  # elevation and azimuth of the 3D plot


def evaluate(R, t, R_true, C_true):
    C = np.array([-r.T @ tt for r, tt in zip(R, t)])
    s, Ra, ta = umeyama(C, C_true)
    C_al = s * C @ Ra.T + ta
    rot_err = [rotation_angle_deg(r @ Ra.T, rt) for r, rt in zip(R, R_true)]
    spacing = np.linalg.norm(np.diff(C_true, axis=0), axis=1).mean()
    center_err = np.linalg.norm(C_al - C_true, axis=1) / spacing * 100  # % of camera spacing
    return C_al, np.array(rot_err), center_err


def main():
    v = load_middlebury()
    feats = [sift.detect_and_describe(v.images[i], 3000) for i in VIEWS]
    kps = [k.xy for k, _ in feats]
    descs = [d for _, d in feats]
    Ks = v.K[VIEWS]
    rec = reconstruct_chain(kps, descs, Ks)
    R_true, C_true = v.R[VIEWS], v.centers[VIEWS]
    rows = []
    results = {}
    for name, (R, t) in {"before BA": (rec.R, rec.t), "after BA": (rec.ba.R, rec.ba.t)}.items():
        C_al, rot_err, center_err = evaluate(R, t, R_true, C_true)
        results[name] = C_al
        rows.append({"stage": name, "median_rotation_error_deg": round(float(np.median(rot_err)), 2),
                     "max_rotation_error_deg": round(float(rot_err.max()), 2),
                     "median_center_error_pct_of_spacing": round(float(np.median(center_err)), 2),
                     "max_center_error_pct_of_spacing": round(float(center_err.max()), 2)})
    rows[0]["rms_reprojection_px"] = round(rec.ba.rms_before, 3)
    rows[1]["rms_reprojection_px"] = round(rec.ba.rms_after, 3)
    for r in rows:
        r.update(cameras=len(VIEWS), points=len(rec.X), observations=len(rec.obs))
        print(r)
    with (ROOT / "results/reconstruction.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # top view (x-z plane of the ground-truth frame)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    s, Ra, ta = umeyama(np.array([-r.T @ tt for r, tt in zip(rec.ba.R, rec.ba.t)]), C_true)
    Xw = s * rec.ba.X @ Ra.T + ta
    ax.scatter(Xw[:, 0], Xw[:, 2], s=3, color="#9a9890", label=f"{len(Xw)} triangulated points")
    ax.plot(C_true[:, 0], C_true[:, 2], "o", ms=10, mfc="none", mec="#0b0b0b", mew=1.5, label="Calibrated cameras")
    ax.plot(results["before BA"][:, 0], results["before BA"][:, 2], "x", ms=8, color="#eb6834", mew=2,
            label="Ours, before bundle adjustment")
    ax.plot(results["after BA"][:, 0], results["after BA"][:, 2], "+", ms=11, color="#2a78d6", mew=2,
            label="Ours, after bundle adjustment")
    for k, i in enumerate(VIEWS):
        ax.annotate(str(i + 1), C_true[k, [0, 2]], xytext=(8, 4), textcoords="offset points", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    ax.set_title("Temple views 6-11, seen from above", loc="left", fontsize=10.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/reconstruction_temple.png", dpi=130, facecolor="white")
    print("wrote results/reconstruction.csv and docs/reconstruction_temple.png")
    overview_figure(v, rec)



def frustum(R, t, K, w, h, depth):
    """Camera centre and the four image-corner points at ``depth`` along the viewing rays (world frame)."""
    C = -R.T @ t
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float)
    # K^-1 (u, v, 1) has camera-frame depth 1; scale it, then rotate into the world frame
    rays = (R.T @ (np.linalg.solve(K, corners.T) * depth)).T
    return C, C + rays


def overview_figure(v, rec):
    """Top: the six input photos. Bottom: where the photos were taken, ours vs calibrated, in 3D."""
    from matplotlib.gridspec import GridSpec

    C_true = v.centers[VIEWS]
    C_est = np.array([-r.T @ tt for r, tt in zip(rec.ba.R, rec.ba.t)])
    s, Ra, ta = umeyama(C_est, C_true)
    fig = plt.figure(figsize=(12, 9.6))
    gs = GridSpec(2, 6, height_ratios=[1.75, 2.6], hspace=0.06, wspace=0.04, figure=fig)
    for k, i in enumerate(VIEWS):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(upright(v.images[i]), cmap="gray", vmin=0, vmax=1)  # turned upright for display
        ax.set_title(f"Photo {i + 1}", fontsize=9.5)
        ax.axis("off")
    ax = fig.add_subplot(gs[1, :], projection="3d")
    h, w = v.images[0].shape
    Xw = s * rec.ba.X @ Ra.T + ta
    ax.scatter(Xw[:, 0], Xw[:, 2], -Xw[:, 1], s=4, color="#7a786f", depthshade=False)
    depth = 0.12
    for k, i in enumerate(VIEWS):
        for (R, t), color, lw in (((v.R[i], v.t[i]), "#0b0b0b", 1.6),
                                  ((rec.ba.R[k] @ Ra.T, None), "#2a78d6", 1.6)):
            if t is None:  # our camera, moved into the calibrated frame by the similarity
                C = s * Ra @ C_est[k] + ta
                t = -R @ C
            C, pts = frustum(R, t, v.K[i], w, h, depth)
            P = np.vstack([C, pts])
            P = np.column_stack([P[:, 0], P[:, 2], -P[:, 1]])  # plot with the vertical axis up
            for c in range(1, 5):
                ax.plot(*zip(P[0], P[c]), color=color, lw=lw * 0.6)
            ring = P[[1, 2, 3, 4, 1]]
            ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, lw=lw)
        Ct = C_true[k]
        centre = Xw.mean(axis=0)
        # faint line of sight from each true camera to the temple
        ax.plot([Ct[0], centre[0]], [Ct[2], centre[2]], [-Ct[1], -centre[1]], color="#c9c7c0", lw=0.8, ls=":")
        ax.text(Ct[0], Ct[2], -Ct[1] + 0.05, str(i + 1), fontsize=11, ha="center", weight="bold")
    ax.plot([], [], color="#0b0b0b", lw=1.6, label="Where each photo was taken (calibration supplied with the dataset)")
    ax.plot([], [], color="#2a78d6", lw=1.6, label="Where this code says it was taken, from the photos alone")
    ax.scatter([], [], s=8, color="#7a786f", label=f"{len(Xw)} points on the temple, triangulated by this code")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    allp = np.vstack([np.column_stack([C_true[:, 0], C_true[:, 2], -C_true[:, 1]]),
                      np.column_stack([Xw[:, 0], Xw[:, 2], -Xw[:, 1]])])
    lo, hi = allp.min(0) - 0.08, allp.max(0) + 0.08
    span = (hi - lo).max()
    mid = (hi + lo) / 2
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_zlim(mid[2] - span / 4, mid[2] + span / 4)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=VIEW[0], azim=VIEW[1])
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_edgecolor("#e6e4df")
    fig.savefig(ROOT / "docs/overview_temple.png", dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote docs/overview_temple.png")


if __name__ == "__main__":
    main()
