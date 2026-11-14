"""A small incremental reconstruction along a chain of views.

1. Two-view start: essential matrix from the first pair, camera 1 at the origin,
   triangulate the matches.
2. For each next view: its descriptors are matched against the descriptors
   stored with every 3D point so far (and tracks continued from the previous
   view), which gives 2D-3D correspondences, so the new camera comes from PnP
   (RANSAC + LM). Matches with the previous view that have no 3D point yet are
   triangulated.
3. Bundle adjustment over all cameras and points.

Intrinsics are known. New points are only triangulated between consecutive
views, which keeps the bookkeeping simple and is enough for a chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvg.bundle_adjustment import BAResult, bundle_adjust
from mvg.camera import project, projection_matrix
from mvg.features.matching import match
from mvg.pnp import solve_pnp
from mvg.two_view import essential_from_fundamental, estimate_fundamental, recover_pose, refine_pose, triangulate


@dataclass
class Reconstruction:
    R: np.ndarray
    t: np.ndarray
    X: np.ndarray
    cam_idx: np.ndarray
    pt_idx: np.ndarray
    obs: np.ndarray
    ba: BAResult | None = None


def _good_points(X, P1, P2, x1, x2, max_err):
    e1 = np.linalg.norm(project(P1, X) - x1, axis=1)
    e2 = np.linalg.norm(project(P2, X) - x2, axis=1)
    d1 = (np.c_[X, np.ones(len(X))] @ P1[2])  # depth up to a positive factor
    d2 = (np.c_[X, np.ones(len(X))] @ P2[2])
    return (e1 < max_err) & (e2 < max_err) & (d1 > 0) & (d2 > 0)


def reconstruct_chain(keypoints, descriptors, Ks, ratio=0.8, f_threshold=1.0, max_err=2.0, seed=0,
                      run_ba: bool = True) -> Reconstruction:
    """``keypoints[i]``: (n_i, 2) pixels of view i; views are consecutive along the chain."""
    rng = np.random.default_rng(seed)
    m = len(keypoints)
    R = np.zeros((m, 3, 3))
    t = np.zeros((m, 3))
    points: list[np.ndarray] = []
    point_desc: list[np.ndarray] = []  # descriptor of each point's latest observation
    point_of = [dict() for _ in range(m)]  # keypoint index -> point id, per view
    observations: list[tuple[int, int, np.ndarray]] = []

    def matches(i, j):
        mm = match(descriptors[i], descriptors[j], ratio=ratio)
        x1, x2 = keypoints[i][mm[:, 0]], keypoints[j][mm[:, 1]]
        F, inl, _ = estimate_fundamental(x1, x2, threshold=f_threshold, rng=rng)
        return mm[inl]

    def add_points(i, j, mm):
        P1, P2 = projection_matrix(Ks[i], R[i], t[i]), projection_matrix(Ks[j], R[j], t[j])
        new = np.array([a not in point_of[i] and b not in point_of[j] for a, b in mm], bool)
        if not new.any():
            return
        sel = mm[new]
        x1, x2 = keypoints[i][sel[:, 0]], keypoints[j][sel[:, 1]]
        X = triangulate(P1, P2, x1, x2)
        ok = _good_points(X, P1, P2, x1, x2, max_err)
        for (a, b), Xk, xa, xb in zip(sel[ok], X[ok], x1[ok], x2[ok]):
            pid = len(points)
            points.append(Xk)
            point_desc.append(descriptors[j][b])
            point_of[i][a], point_of[j][b] = pid, pid
            observations.extend([(i, pid, xa), (j, pid, xb)])

    # 1. start from views 0 and 1
    mm = matches(0, 1)
    x1, x2 = keypoints[0][mm[:, 0]], keypoints[1][mm[:, 1]]
    F, _, _ = estimate_fundamental(x1, x2, threshold=f_threshold, rng=rng)
    E = essential_from_fundamental(F, Ks[0], Ks[1])
    R1, t1, _, _ = recover_pose(E, x1, x2, Ks[0], Ks[1])
    R1, t1, _ = refine_pose(R1, t1, x1, x2, Ks[0], Ks[1])
    R[0], t[0] = np.eye(3), np.zeros(3)
    R[1], t[1] = R1, t1
    add_points(0, 1, mm)

    # 2. add the remaining views with PnP
    for j in range(2, m):
        i = j - 1
        mm = matches(i, j)
        corr = {}  # keypoint of view j -> point id
        for a, b in mm:  # tracks continued from the previous view
            if a in point_of[i]:
                corr[b] = point_of[i][a]
        to_points = match(descriptors[j], np.array(point_desc), ratio=ratio)  # direct 2D-3D matches
        for b, pid in to_points:
            corr.setdefault(int(b), int(pid))
        kb = np.array(list(corr.keys()))
        pids = np.array([corr[b] for b in kb])
        X3 = np.array(points)[pids]
        x2d = keypoints[j][kb]
        Rj, tj, inl = solve_pnp(X3, x2d, Ks[j], threshold=max_err, rng=rng, init=(R[i], t[i]))
        if Rj is None or inl.sum() < 6:
            raise RuntimeError(f"PnP failed for view {j}")
        R[j], t[j] = Rj, tj
        for b, pid, ok in zip(kb, pids, inl):
            if ok:
                point_of[j][int(b)] = int(pid)
                point_desc[pid] = descriptors[j][b]
                observations.append((j, int(pid), keypoints[j][b]))
        add_points(i, j, mm)

    cam_idx = np.array([o[0] for o in observations])
    pt_idx = np.array([o[1] for o in observations])
    obs = np.array([o[2] for o in observations])
    rec = Reconstruction(R, t, np.array(points), cam_idx, pt_idx, obs)
    if run_ba:
        rec.ba = bundle_adjust(Ks[0], R, t, rec.X, cam_idx, pt_idx, obs, fixed_cameras=(0,))
    return rec
